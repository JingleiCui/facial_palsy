#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 1 配置文件
================
集中管理所有超参数
"""

from dataclasses import dataclass, field
from typing import List, Dict
from pathlib import Path

# =============================================================================
# 动作关键点映射
# =============================================================================
# 每个动作只使用相关区域的关键点，减少噪声

# 眼部关键点 (左+右)
EYE_LANDMARKS = [
    # 左眼
    33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246,
    130, 25, 110, 24, 23, 22, 26, 112, 243,
    # 右眼
    263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466,
    359, 255, 339, 254, 253, 252, 256, 341, 463,
]

# 眉毛关键点 (左+右)
BROW_LANDMARKS = [
    # 左眉
    70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
    # 右眉
    300, 293, 334, 296, 336, 285, 295, 282, 283, 276,
]

# 嘴部关键点
MOUTH_LANDMARKS = [
    # 外唇轮廓
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    375, 321, 405, 314, 17, 84, 181, 91, 146,
    # 内唇轮廓
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    415, 310, 311, 312, 13, 82, 81, 80, 191,
]

# 鼻部关键点
NOSE_LANDMARKS = [
    1, 2, 98, 327, 168, 6,  # 鼻梁
    195, 5, 4, 45, 275,  # 鼻尖
    60, 75, 59, 166, 79, 238, 239,  # 左鼻翼
    290, 305, 289, 392, 309, 458, 459,  # 右鼻翼
]

# 脸颊关键点
CHEEK_LANDMARKS = [
    # 左脸颊
    36, 205, 206, 207, 187, 123, 116, 117, 118, 119, 100, 47, 114, 188,
    # 右脸颊
    266, 425, 426, 427, 411, 352, 345, 346, 347, 348, 329, 277, 343, 412,
]

# 各动作使用的关键点
ACTION_LANDMARKS = {
    "NeutralFace": None,  # 使用全部478个关键点
    "CloseEyeSoftly": EYE_LANDMARKS,
    "CloseEyeHardly": EYE_LANDMARKS,
    "VoluntaryEyeBlink": EYE_LANDMARKS,
    "SpontaneousEyeBlink": EYE_LANDMARKS,
    "RaiseEyebrow": BROW_LANDMARKS + EYE_LANDMARKS,
    "Smile": MOUTH_LANDMARKS + CHEEK_LANDMARKS,
    "ShowTeeth": MOUTH_LANDMARKS,
    "ShrugNose": NOSE_LANDMARKS + MOUTH_LANDMARKS[:10],  # 鼻部+上唇
    "BlowCheek": CHEEK_LANDMARKS + MOUTH_LANDMARKS,
    "LipPucker": MOUTH_LANDMARKS,
}


@dataclass
class Stage1Config:
    """Stage 1 完整配置"""

    # ===== 动作配置 =====
    action_names: List[str] = field(default_factory=lambda: [
        "NeutralFace", "CloseEyeSoftly", "CloseEyeHardly", "RaiseEyebrow",
        "Smile", "ShowTeeth", "ShrugNose", "BlowCheek", "LipPucker",
        "VoluntaryEyeBlink", "SpontaneousEyeBlink",
    ])

    # ===== 数据配置 =====
    data_root: str = "./data/processed"
    max_seq_len: int = 150
    total_landmarks: int = 478
    landmark_dim: int = 3  # x, y, z

    # ===== TCN编码器配置 =====
    tcn_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    tcn_kernel_size: int = 3
    tcn_dropout: float = 0.2

    # ===== 动作头配置 =====
    shared_dim: int = 256  # 共享编码器输出维度
    head_hidden_dim: int = 128  # 动作头隐藏层维度
    embedding_dim: int = 64  # 每个动作的Embedding维度
    num_classes: int = 5  # 伪标签分类数 (1-5)
    head_dropout: float = 0.3

    # ===== 几何特征配置 (Stage 0) =====
    geometric_dim: int = 66  # Stage 0几何特征向量维度

    # ===== 训练配置 =====
    batch_size: int = 16
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 5

    # 损失权重
    cls_weight: float = 1.0  # 分类损失
    aux_weight: float = 0.1  # 辅助损失 (几何特征回归)

    # 早停
    patience: int = 10

    # ===== 其他 =====
    device: str = "cuda"
    num_workers: int = 4
    seed: int = 42
    save_dir: str = "./checkpoints/stage1"

    @property
    def num_actions(self) -> int:
        return len(self.action_names)

    def get_landmark_indices(self, action_name: str) -> List[int]:
        """获取动作使用的关键点索引"""
        indices = ACTION_LANDMARKS.get(action_name)
        if indices is None:
            return list(range(self.total_landmarks))
        return indices

    def get_input_dim(self, action_name: str) -> int:
        """获取动作的输入维度 (关键点数 × 3)"""
        indices = self.get_landmark_indices(action_name)
        return len(indices) * self.landmark_dim


CONFIG = Stage1Config()