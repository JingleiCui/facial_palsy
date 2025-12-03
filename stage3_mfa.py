#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3: MFA (Multi-modal Fusion Attention)
多模态融合注意力模块

功能：
1. 从video_features表读取 geo_refined_features, visual_guided_features, visual_features
2. 使用 MFA 模块进行最终的多模态融合
3. 输出 fused_action_features (512维) 并写回数据库

直接在PyCharm中点击运行即可！可重复运行，已处理的样本会跳过。
"""

import sqlite3
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

# ==================== 配置参数 ====================
DB_PATH = "facialPalsy.db"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 模型参数
GEO_DIM = 256
VISUAL_GUIDED_DIM = 256
VISUAL_GLOBAL_DIM = 1280
OUTPUT_DIM = 512
NUM_HEADS = 8
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

# ==================== MFA模块 ====================
class MFA(nn.Module):
    """Multi-modal Fusion Attention Module"""

    def __init__(self, geo_dim=256, visual_guided_dim=256, visual_global_dim=1280,
                 output_dim=512, num_heads=8, dropout=0.1):
        super().__init__()

        self.geo_dim = geo_dim
        self.visual_guided_dim = visual_guided_dim
        self.visual_global_dim = visual_global_dim
        self.output_dim = output_dim

        # 特征投影到统一维度
        self.geo_proj = nn.Sequential(
            nn.Linear(geo_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )

        self.visual_guided_proj = nn.Sequential(
            nn.Linear(visual_guided_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )

        self.visual_global_proj = nn.Sequential(
            nn.Linear(visual_global_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )

        # Token gating: 动态调整各模态的权重
        self.token_gate = nn.Sequential(
            nn.Linear(output_dim * 3, 3),
            nn.Softmax(dim=-1)
        )

        # Transformer Decoder for multi-modal fusion
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=output_dim,
            nhead=num_heads,
            dim_feedforward=output_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        # Final projection
        self.final_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim)
        )

    def forward(self, geo_features, visual_guided_features, visual_global_features):
        """
        Args:
            geo_features: (batch, geo_dim=256)
            visual_guided_features: (batch, visual_guided_dim=256)
            visual_global_features: (batch, visual_global_dim=1280)
        Returns:
            fused_features: (batch, output_dim=512)
        """
        batch_size = geo_features.size(0)

        # 1. 投影到统一维度
        geo_proj = self.geo_proj(geo_features)
        visual_guided_proj = self.visual_guided_proj(visual_guided_features)
        visual_global_proj = self.visual_global_proj(visual_global_features)

        # 2. Token gating: 计算各模态的权重
        concat_features = torch.cat([geo_proj, visual_guided_proj, visual_global_proj], dim=-1)
        gate_weights = self.token_gate(concat_features)  # (batch, 3)

        # 应用门控权重
        geo_weighted = geo_proj * gate_weights[:, 0:1]
        visual_guided_weighted = visual_guided_proj * gate_weights[:, 1:2]
        visual_global_weighted = visual_global_proj * gate_weights[:, 2:3]

        # 3. Stack as sequence tokens
        memory = torch.stack([geo_weighted, visual_guided_weighted, visual_global_weighted], dim=1)

        # 4. Initial target
        initial_target = (geo_weighted + visual_guided_weighted + visual_global_weighted) / 3.0
        target = initial_target.unsqueeze(1)

        # 5. Transformer Decoder fusion
        fused = self.transformer_decoder(tgt=target, memory=memory)
        fused = fused.squeeze(1)

        # 6. Final projection with residual
        output = self.final_proj(fused) + initial_target

        return output


def load_features_from_db(db_path, force_reprocess=False):
    """从数据库加载特征"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if force_reprocess:
        query = """
            SELECT feature_id, geo_refined_features, visual_guided_features, visual_features
            FROM video_features
            WHERE geo_refined_features IS NOT NULL
              AND visual_guided_features IS NOT NULL
              AND visual_features IS NOT NULL
            ORDER BY feature_id
        """
    else:
        query = """
            SELECT feature_id, geo_refined_features, visual_guided_features, visual_features
            FROM video_features
            WHERE geo_refined_features IS NOT NULL
              AND visual_guided_features IS NOT NULL
              AND visual_features IS NOT NULL
              AND fused_action_features IS NULL
            ORDER BY feature_id
        """

    cursor.execute(query)

    data = []
    for row in cursor.fetchall():
        feature_id, geo_blob, visual_guided_blob, visual_blob = row

        # 三个模态都必须存在
        if geo_blob is None or visual_guided_blob is None or visual_blob is None:
            continue

        # 按固定维度解码 float32 BLOB
        geo_features = np.frombuffer(geo_blob, dtype=np.float32, count=GEO_DIM)
        visual_guided_features = np.frombuffer(visual_guided_blob, dtype=np.float32, count=VISUAL_GUIDED_DIM)
        visual_features = np.frombuffer(visual_blob, dtype=np.float32, count=VISUAL_GLOBAL_DIM)

        data.append({
            'feature_id': feature_id,
            'geo_refined_features': geo_features,
            'visual_guided_features': visual_guided_features,
            'visual_features': visual_features
        })

    conn.close()
    return data


def save_fused_features(db_path, feature_id, features):
    """保存fused_action_features到数据库"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    features_blob = encode_feature(features)

    cursor.execute("""
        UPDATE video_features
        SET fused_action_features = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE feature_id = ?
    """, (features_blob, feature_id))

    conn.commit()
    conn.close()


def process_stage3():
    """执行Stage 3特征融合"""
    print("=" * 60)
    print("Stage 3: MFA多模态融合")
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
            print("❌ 错误: 数据库中缺少必要的特征！")
            print("请确保已运行:")
            print("  1. stage1_cdcaf.py (生成geo_refined_features)")
            print("  2. stage2_gqca.py (生成visual_guided_features)")
            print("  3. visual_feature_extractor.py (生成visual_features)")
        else:
            print("✓ 所有样本已经处理过了！")
            print("如需重新处理，请设置 FORCE_REPROCESS = True")
        return

    print(f"✓ 需要处理 {len(data)} 个样本")
    print()

    # 2. 验证特征维度
    sample = data[0]
    geo_dim = len(sample['geo_refined_features'])
    visual_guided_dim = len(sample['visual_guided_features'])
    visual_global_dim = len(sample['visual_features'])

    print(f"特征维度验证:")
    print(f"  geo_refined_features: {geo_dim} (期望: 256)")
    print(f"  visual_guided_features: {visual_guided_dim} (期望: 256)")
    print(f"  visual_features: {visual_global_dim} (期望: 1280)")

    if geo_dim != 256:
        print(f"⚠ 警告: geo_refined_features维度不是256！")
    if visual_guided_dim != 256:
        print(f"⚠ 警告: visual_guided_features维度不是256！")
    if visual_global_dim != 1280:
        print(f"⚠ 警告: visual_features维度不是1280！")
    print()

    # 3. 创建MFA模型
    print("创建MFA模型...")
    model = MFA(
        geo_dim=geo_dim,
        visual_guided_dim=visual_guided_dim,
        visual_global_dim=visual_global_dim,
        output_dim=OUTPUT_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT
    ).to(DEVICE)

    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ 模型已创建并移至 {DEVICE}")
    print(f"✓ 模型参数量: {total_params:,}")
    print()

    # 4. 批量处理
    num_batches = (len(data) + BATCH_SIZE - 1) // BATCH_SIZE
    total_processed = 0

    print(f"开始处理 {len(data)} 个样本 (batch_size={BATCH_SIZE})...")

    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="MFA融合"):
            batch_items = data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

            # 准备批量数据
            geo_batch = []
            visual_guided_batch = []
            visual_global_batch = []

            for item in batch_items:
                geo_batch.append(item['geo_refined_features'])
                visual_guided_batch.append(item['visual_guided_features'])
                visual_global_batch.append(item['visual_features'])

            # 转换为tensor
            geo_tensor = torch.FloatTensor(np.array(geo_batch)).to(DEVICE)
            visual_guided_tensor = torch.FloatTensor(np.array(visual_guided_batch)).to(DEVICE)
            visual_global_tensor = torch.FloatTensor(np.array(visual_global_batch)).to(DEVICE)

            # 前向传播
            fused_features = model(geo_tensor, visual_guided_tensor, visual_global_tensor)

            # 保存结果
            fused_features_np = fused_features.cpu().numpy()
            for j, item in enumerate(batch_items):
                save_fused_features(
                    DB_PATH,
                    item['feature_id'],
                    fused_features_np[j]
                )
                total_processed += 1

    print()
    print("=" * 60)
    print(f"✓ Stage 3完成！共处理 {total_processed} 个样本")
    print(f"✓ fused_action_features (512维) 已保存到数据库")
    print("=" * 60)
    print()
    print("下一步: 运行 dataset_splitter.py 进行数据集划分")


if __name__ == "__main__":
    process_stage3()