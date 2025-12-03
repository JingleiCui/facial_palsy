#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2: GQCA (Geometry-guided Query Cross-Attention)
几何引导的查询交叉注意力模块

功能：
1. 从video_features表读取 geo_refined_features + visual_features + wrinkle_features
2. 使用 GQCA 模块进行几何引导的视觉特征增强
3. 融合皱纹特征（10维）到视觉特征中
4. 输出 visual_guided_features (256维) 并写回数据库

直接在PyCharm中点击运行即可！可重复运行，已处理的样本会跳过。
"""

import sqlite3
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import pickle

# ==================== 配置参数 ====================
DB_PATH = "facialPalsy.db"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 模型参数
GEO_DIM = 256
VISUAL_DIM = 1280
WRINKLE_DIM = 10  # 皱纹特征维度
HIDDEN_DIM = 256
NUM_HEADS = 4
DROPOUT = 0.1
BATCH_SIZE = 128

# 是否强制重新处理
FORCE_REPROCESS = True

# ====== 统一的特征编码函数 ======
def encode_feature(arr: np.ndarray) -> bytes:
    """
    将任意 numpy 数组统一编码为 float32 BLOB
    """
    arr = np.asarray(arr, dtype=np.float32).ravel()
    return arr.tobytes()

# ==================== GQCA模块 ====================
class GQCA(nn.Module):
    """Geometry-guided Query Cross-Attention Module with Wrinkle Features"""

    def __init__(self, geo_dim=256, visual_dim=1280, wrinkle_dim=10,
                 hidden_dim=256, num_heads=4, dropout=0.1):
        super().__init__()

        self.geo_dim = geo_dim
        self.visual_dim = visual_dim
        self.wrinkle_dim = wrinkle_dim
        self.hidden_dim = hidden_dim

        # 扩展视觉特征维度以包含皱纹特征
        self.extended_visual_dim = visual_dim + wrinkle_dim  # 1280 + 10 = 1290

        # FiLM modulation: 使用几何特征调制视觉特征
        self.film_gamma = nn.Linear(geo_dim, self.extended_visual_dim)
        self.film_beta = nn.Linear(geo_dim, self.extended_visual_dim)

        # Visual feature projection (降维)
        self.visual_proj = nn.Sequential(
            nn.Linear(self.extended_visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Geometry query projection
        self.geo_query_proj = nn.Linear(geo_dim, hidden_dim)

        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, geo_features, visual_features, wrinkle_features):
        """
        Args:
            geo_features: (batch, geo_dim=256)
            visual_features: (batch, visual_dim=1280)
            wrinkle_features: (batch, wrinkle_dim=10)
        Returns:
            visual_guided_features: (batch, hidden_dim=256)
        """
        batch_size = geo_features.size(0)

        # 1. 融合视觉特征和皱纹特征
        extended_visual = torch.cat([visual_features, wrinkle_features], dim=-1)  # (batch, 1290)

        # 2. FiLM modulation
        gamma = self.film_gamma(geo_features)  # (batch, 1290)
        beta = self.film_beta(geo_features)  # (batch, 1290)

        modulated_visual = extended_visual * (1 + gamma) + beta  # (batch, 1290)

        # 3. Project visual features
        visual_proj = self.visual_proj(modulated_visual)  # (batch, hidden_dim)

        # 4. 创建伪空间tokens (7x7 = 49 tokens)
        visual_tokens = visual_proj.unsqueeze(1).repeat(1, 49, 1)  # (batch, 49, hidden_dim)

        # 5. Geometry as query
        geo_query = self.geo_query_proj(geo_features)  # (batch, hidden_dim)
        geo_query = geo_query.unsqueeze(1)  # (batch, 1, hidden_dim)

        # 6. Cross-attention
        attn_output, _ = self.cross_attention(
            query=geo_query,
            key=visual_tokens,
            value=visual_tokens
        )  # (batch, 1, hidden_dim)

        # 7. Residual connection
        attn_output = attn_output.squeeze(1)  # (batch, hidden_dim)
        x = self.norm1(geo_query.squeeze(1) + attn_output)

        # 8. Feed-forward network
        ffn_output = self.ffn(x)
        output = self.norm2(x + ffn_output)

        return output


def load_features_from_db(db_path, force_reprocess=False):
    """从数据库加载特征"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if force_reprocess:
        query = """
            SELECT feature_id, geo_refined_features, visual_features, wrinkle_features
            FROM video_features
            WHERE geo_refined_features IS NOT NULL
              AND visual_features IS NOT NULL
            ORDER BY feature_id
        """
    else:
        query = """
            SELECT feature_id, geo_refined_features, visual_features, wrinkle_features
            FROM video_features
            WHERE geo_refined_features IS NOT NULL
              AND visual_features IS NOT NULL
              AND visual_guided_features IS NULL
            ORDER BY feature_id
        """

    cursor.execute(query)

    data = []
    for row in cursor.fetchall():
        feature_id, geo_blob, visual_blob, wrinkle_blob = row

        # geo_refined / visual 必须存在
        if geo_blob is None or visual_blob is None:
            continue

        # 按固定维度解码 float32 BLOB
        geo_features = np.frombuffer(geo_blob, dtype=np.float32, count=GEO_DIM)
        visual_features = np.frombuffer(visual_blob, dtype=np.float32, count=VISUAL_DIM)

        # 皱纹特征可能为 NULL，用 0 向量替代
        if wrinkle_blob is not None:
            wrinkle_features = np.frombuffer(wrinkle_blob, dtype=np.float32, count=WRINKLE_DIM)
        else:
            wrinkle_features = np.zeros(WRINKLE_DIM, dtype=np.float32)

        data.append({
            'feature_id': feature_id,
            'geo_refined_features': geo_features,
            'visual_features': visual_features,
            'wrinkle_features': wrinkle_features
        })

    conn.close()
    return data


def save_visual_guided_features(db_path, feature_id, features):
    """保存visual_guided_features到数据库"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    features_blob = encode_feature(features)

    cursor.execute("""
        UPDATE video_features
        SET visual_guided_features = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE feature_id = ?
    """, (features_blob, feature_id))

    conn.commit()
    conn.close()


def process_stage2():
    """执行Stage 2特征融合"""
    print("=" * 60)
    print("Stage 2: GQCA特征融合（包含皱纹特征）")
    print("=" * 60)
    print(f"数据库路径: {DB_PATH}")
    print(f"设备: {DEVICE}")
    print(f"强制重新处理: {FORCE_REPROCESS}")
    print()

    # 1. 加载数据
    print("正在加载特征数据...")
    data = load_features_from_db(DB_PATH, FORCE_REPROCESS)

    if len(data) == 0:
        if FORCE_REPROCESS:
            print("❌ 错误: 数据库中没有找到必要的特征！")
            print("请确保已运行:")
            print("  1. stage1_cdcaf.py (生成geo_refined_features)")
            print("  2. visual_feature_extractor.py (生成visual_features)")
            print("  3. wrinkle_feature.py (生成wrinkle_features，可选)")
        else:
            print("✓ 所有样本已经处理过了！")
            print("如需重新处理，请设置 FORCE_REPROCESS = True")
        return

    print(f"✓ 需要处理 {len(data)} 个样本")
    print()

    # 2. 验证特征维度
    sample = data[0]
    geo_dim = len(sample['geo_refined_features'])
    visual_dim = len(sample['visual_features'])
    wrinkle_dim = len(sample['wrinkle_features'])

    print(f"特征维度验证:")
    print(f"  geo_refined_features: {geo_dim} (期望: 256)")
    print(f"  visual_features: {visual_dim} (期望: 1280)")
    print(f"  wrinkle_features: {wrinkle_dim} (期望: 10)")

    if geo_dim != 256:
        print(f"⚠ 警告: geo_refined_features维度不是256！")
    if visual_dim != 1280:
        print(f"⚠ 警告: visual_features维度不是1280！")
    if wrinkle_dim != 10:
        print(f"⚠ 提示: wrinkle_features维度不是10，可能未提取皱纹特征")
    print()

    # 3. 创建GQCA模型
    print("创建GQCA模型...")
    model = GQCA(
        geo_dim=geo_dim,
        visual_dim=visual_dim,
        wrinkle_dim=wrinkle_dim,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT
    ).to(DEVICE)

    model.eval()
    print(f"✓ 模型已创建并移至 {DEVICE}")
    print()

    # 4. 批量处理
    num_batches = (len(data) + BATCH_SIZE - 1) // BATCH_SIZE
    total_processed = 0

    print(f"开始处理 {len(data)} 个样本 (batch_size={BATCH_SIZE})...")

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="GQCA融合"):
            batch_items = data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

            # 准备批量数据
            geo_batch = []
            visual_batch = []
            wrinkle_batch = []

            for item in batch_items:
                geo_batch.append(item['geo_refined_features'])
                visual_batch.append(item['visual_features'])
                wrinkle_batch.append(item['wrinkle_features'])

            # 转换为tensor
            geo_tensor = torch.FloatTensor(np.array(geo_batch)).to(DEVICE)
            visual_tensor = torch.FloatTensor(np.array(visual_batch)).to(DEVICE)
            wrinkle_tensor = torch.FloatTensor(np.array(wrinkle_batch)).to(DEVICE)

            # 前向传播
            visual_guided = model(geo_tensor, visual_tensor, wrinkle_tensor)

            # 保存结果
            visual_guided_np = visual_guided.cpu().numpy()
            for j, item in enumerate(batch_items):
                save_visual_guided_features(
                    DB_PATH,
                    item['feature_id'],
                    visual_guided_np[j]
                )
                total_processed += 1

    print()
    print("=" * 60)
    print(f"✓ Stage 2完成！共处理 {total_processed} 个样本")
    print(f"✓ visual_guided_features (256维) 已保存到数据库")
    print(f"✓ 已融合皱纹特征 ({wrinkle_dim}维)")
    print("=" * 60)
    print()
    print("下一步: 运行 stage3_mfa.py")


if __name__ == "__main__":
    process_stage2()