#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1: CDCAF (Clinical-Driven Cross-Attention Fusion)
临床驱动的交叉注意力融合模块

功能：
1. 从video_features表读取 static_features 和 dynamic_features
2. 使用 CDCAF 模块进行几何特征融合
3. 输出 geo_refined_features (256维) 并写回数据库

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
HIDDEN_DIM = 256
NUM_HEADS = 4
DROPOUT = 0.1
BATCH_SIZE = 128

# 是否强制重新处理（设为True会覆盖已有结果）
FORCE_REPROCESS = True

# ====== 统一的特征编码函数 ======
def encode_feature(arr: np.ndarray) -> bytes:
    """
    将任意 numpy 数组统一编码为 float32 BLOB
    """
    arr = np.asarray(arr, dtype=np.float32).ravel()
    return arr.tobytes()

# ==================== CDCAF模块 ====================
class CDCAF(nn.Module):
    """Clinical-Driven Cross-Attention Fusion Module"""

    def __init__(self, static_dim, dynamic_dim, hidden_dim=256, num_heads=4, dropout=0.1):
        super().__init__()
        self.static_dim = static_dim
        self.dynamic_dim = dynamic_dim
        self.hidden_dim = hidden_dim

        # Static features encoder
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Dynamic features encoder
        if dynamic_dim > 0:
            self.dynamic_encoder = nn.Sequential(
                nn.Linear(dynamic_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            # Modality gating
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
        else:
            self.dynamic_encoder = None
            self.gate = None

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, static_features, dynamic_features=None):
        static_encoded = self.static_encoder(static_features)

        if self.dynamic_encoder is not None and dynamic_features is not None and self.dynamic_dim > 0:
            dynamic_encoded = self.dynamic_encoder(dynamic_features)
            concat_features = torch.cat([static_encoded, dynamic_encoded], dim=-1)
            gate_weight = self.gate(concat_features)
            fused = gate_weight * static_encoded + (1 - gate_weight) * dynamic_encoded
        else:
            fused = static_encoded

        fused = fused.unsqueeze(1)
        refined = self.transformer(fused)
        refined = refined.squeeze(1)
        output = self.output_proj(refined)

        return output


def load_features_from_db(db_path, force_reprocess=False):
    """从数据库加载特征"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 查询条件：如果不强制重新处理，则只处理未处理过的
    if force_reprocess:
        query = """
            SELECT vf.feature_id, vf.static_features, vf.dynamic_features,
                   vf.static_dim, vf.dynamic_dim, at.action_name_en
            FROM video_features vf
            JOIN video_files v ON vf.video_id = v.video_id
            JOIN action_types at ON v.action_id = at.action_id
            WHERE vf.static_features IS NOT NULL
            ORDER BY vf.feature_id
        """
    else:
        query = """
            SELECT vf.feature_id, vf.static_features, vf.dynamic_features,
                   vf.static_dim, vf.dynamic_dim, at.action_name_en
            FROM video_features vf
            JOIN video_files v ON vf.video_id = v.video_id
            JOIN action_types at ON v.action_id = at.action_id
            WHERE vf.static_features IS NOT NULL
              AND vf.geo_refined_features IS NULL
            ORDER BY vf.feature_id
        """

    cursor.execute(query)

    data = []
    for row in cursor.fetchall():
        feature_id, static_blob, dynamic_blob, static_dim, dynamic_dim, action_name = row

        static_features = np.frombuffer(static_blob, dtype=np.float32, count=static_dim) if static_blob else None
        dynamic_features = np.frombuffer(dynamic_blob, dtype=np.float32, count=dynamic_dim) if dynamic_blob and dynamic_dim > 0 else None

        if static_features is not None:
            data.append({
                'feature_id': feature_id,
                'action_name': action_name,
                'static_features': static_features,
                'dynamic_features': dynamic_features,
                'static_dim': static_dim or len(static_features),
                'dynamic_dim': dynamic_dim or (len(dynamic_features) if dynamic_features is not None else 0)
            })

    conn.close()
    return data


def save_geo_refined_features(db_path, feature_id, features):
    """保存geo_refined_features到数据库"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    features_blob = encode_feature(features)

    cursor.execute("""
        UPDATE video_features
        SET geo_refined_features = ?,
            fusion_processed_at = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        WHERE feature_id = ?
    """, (features_blob, feature_id))

    conn.commit()
    conn.close()


def process_stage1():
    """执行Stage 1特征融合"""
    print("=" * 60)
    print("Stage 1: CDCAF特征融合")
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
            print("❌ 错误: 数据库中没有找到static_features！")
            print("请先运行 video_pipeline.py 提取几何特征")
        else:
            print("✓ 所有样本已经处理过了！")
            print("如需重新处理，请设置 FORCE_REPROCESS = True")
        return

    print(f"✓ 需要处理 {len(data)} 个样本")
    print()

    # 2. 按动作类型分组
    action_types = {}
    for item in data:
        action = item['action_name']
        if action not in action_types:
            action_types[action] = []
        action_types[action].append(item)

    print(f"发现 {len(action_types)} 种动作类型:")
    for action, items in action_types.items():
        print(f"  - {action}: {len(items)} 个样本")
    print()

    # 3. 为每种动作类型处理
    total_processed = 0

    for action_name, items in action_types.items():
        print(f"处理动作类型: {action_name}")

        # 获取实际特征维度
        sample = items[0]
        static_dim = sample['static_dim']
        dynamic_dim = sample['dynamic_dim']

        print(f"  特征维度: static={static_dim}, dynamic={dynamic_dim}")

        # 创建模型
        model = CDCAF(
            static_dim=static_dim,
            dynamic_dim=dynamic_dim,
            hidden_dim=HIDDEN_DIM,
            num_heads=NUM_HEADS,
            dropout=DROPOUT
        ).to(DEVICE)

        model.eval()

        # 批量处理
        num_batches = (len(items) + BATCH_SIZE - 1) // BATCH_SIZE

        with torch.no_grad():
            for i in tqdm(range(num_batches), desc=f"  {action_name}"):
                batch_items = items[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

                # 准备批量数据
                static_batch = []
                dynamic_batch = []

                for item in batch_items:
                    static_batch.append(item['static_features'])
                    if item['dynamic_features'] is not None and dynamic_dim > 0:
                        dynamic_batch.append(item['dynamic_features'])
                    else:
                        dynamic_batch.append(np.zeros(dynamic_dim) if dynamic_dim > 0 else None)

                # 转换为tensor
                static_tensor = torch.FloatTensor(np.array(static_batch)).to(DEVICE)
                dynamic_tensor = None
                if dynamic_dim > 0 and dynamic_batch[0] is not None:
                    dynamic_tensor = torch.FloatTensor(np.array(dynamic_batch)).to(DEVICE)

                # 前向传播
                geo_refined = model(static_tensor, dynamic_tensor)

                # 保存结果
                geo_refined_np = geo_refined.cpu().numpy()
                for j, item in enumerate(batch_items):
                    save_geo_refined_features(
                        DB_PATH,
                        item['feature_id'],
                        geo_refined_np[j]
                    )
                    total_processed += 1

        print(f"  ✓ 完成 {len(items)} 个样本")
        print()

    print("=" * 60)
    print(f"✓ Stage 1完成！共处理 {total_processed} 个样本")
    print(f"✓ geo_refined_features (256维) 已保存到数据库")
    print("=" * 60)
    print()
    print("下一步: 运行 stage2_gqca.py")


if __name__ == "__main__":
    process_stage1()