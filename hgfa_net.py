#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H-GFA Net: 灵活可配置版本
=============================================

核心改进:
1. **灵活特征选择**: 可配置使用哪些特征模态 (geometric, visual, wrinkle, motion)
2. **序数分类支持**: CORAL, 累积链路, 标准分类可选
3. **简化融合策略**: 根据启用的特征数量自动选择融合方式

使用示例:
    # 只使用wrinkle特征 + 序数分类
    config = HGFANet.get_default_config()
    config['enabled_features'] = ['wrinkle']
    config['ordinal_method'] = 'coral'  # 'standard', 'coral', 'cumulative'
    model = HGFANet(config)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataset_palsy import HierarchicalPalsyDataset


# ============================================================================
# Feature Encoders (特征编码器)
# ============================================================================

class GeometricEncoder(nn.Module):
    """几何特征编码器 (static + dynamic)"""

    def __init__(self, max_static_dim=11, max_dynamic_dim=8, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.static_encoder = nn.Sequential(
            nn.Linear(max_static_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.dynamic_encoder = nn.Sequential(
            nn.Linear(max_dynamic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, static_feat, dynamic_feat):
        static_enc = self.static_encoder(static_feat)
        dynamic_enc = self.dynamic_encoder(dynamic_feat)
        concat = torch.cat([static_enc, dynamic_enc], dim=-1)
        gate_weight = self.gate(concat)
        fused = gate_weight * static_enc + (1 - gate_weight) * dynamic_enc
        return self.output_proj(fused)


class WrinkleEncoder(nn.Module):
    """皱纹特征编码器"""

    def __init__(self, wrinkle_dim=10, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(wrinkle_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, wrinkle_feat):
        return self.encoder(wrinkle_feat)


class VisualEncoder(nn.Module):
    """视觉特征编码器"""

    def __init__(self, visual_dim=1280, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(visual_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, visual_feat):
        return self.encoder(visual_feat)


class MotionEncoder(nn.Module):
    """运动特征编码器"""

    def __init__(self, motion_dim=12, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(motion_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, motion_feat):
        return self.encoder(motion_feat)


# ============================================================================
# Flexible Feature Fusion (灵活特征融合)
# ============================================================================

class FlexibleFusion(nn.Module):
    """
    灵活特征融合模块

    根据启用的特征数量自动选择融合策略:
    - 1个特征: 直接投影
    - 2-4个特征: 拼接 + MLP 或 注意力融合
    """

    def __init__(self, num_features: int, feature_dim: int = 128,
                 output_dim: int = 256, dropout: float = 0.3,
                 fusion_type: str = 'concat'):
        """
        Args:
            num_features: 启用的特征模态数量
            feature_dim: 每个特征的维度 (编码后)
            output_dim: 输出维度
            dropout: dropout率
            fusion_type: 'concat' (拼接+MLP) 或 'attention' (注意力融合)
        """
        super().__init__()
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.fusion_type = fusion_type

        if num_features == 1:
            # 单特征: 直接投影
            self.fusion = nn.Sequential(
                nn.Linear(feature_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        elif fusion_type == 'concat':
            # 拼接融合
            self.fusion = nn.Sequential(
                nn.Linear(feature_dim * num_features, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        else:
            # 注意力融合
            self.attention = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            self.fusion = nn.Sequential(
                nn.Linear(feature_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of (batch, feature_dim) tensors
        Returns:
            fused: (batch, output_dim)
        """
        if self.num_features == 1:
            return self.fusion(features[0])

        if self.fusion_type == 'concat':
            concat = torch.cat(features, dim=-1)
            return self.fusion(concat)
        else:
            # 堆叠特征: (B, num_features, feature_dim)
            stacked = torch.stack(features, dim=1)
            attended, _ = self.attention(stacked, stacked, stacked)
            pooled = attended.mean(dim=1)  # (B, feature_dim)
            return self.fusion(pooled)


# ============================================================================
# Ordinal Classification Heads (序数分类头)
# ============================================================================

class StandardClassificationHead(nn.Module):
    """标准分类头 (CrossEntropy)"""

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回 logits (B, num_classes)"""
        return self.head(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """返回预测类别"""
        return self.head(x).argmax(dim=-1)


class CORALHead(nn.Module):
    """
    CORAL (Consistent Rank Logits) 序数回归头

    参考: Cao et al., "Rank Consistent Ordinal Regression for Neural Networks"

    关键思想: 为K个类别学习K-1个二分类阈值，确保排序一致性
    """

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.num_classes = num_classes

        # 共享特征提取
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 单个权重向量 (rank-consistent)
        self.fc = nn.Linear(128, 1, bias=False)

        # K-1个可学习的阈值 (biases)
        # 初始化为有序的值，确保 b_0 < b_1 < ... < b_{K-2}
        self.biases = nn.Parameter(
            torch.arange(num_classes - 1, dtype=torch.float32) - (num_classes - 2) / 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        返回累积概率 P(Y > k) 对应的 logits

        Returns:
            logits: (B, num_classes - 1) 每个位置是 P(Y > k) 的 logit
        """
        feat = self.feature(x)  # (B, 128)
        score = self.fc(feat)  # (B, 1)

        # logit_k = score - bias_k
        # P(Y > k) = sigmoid(logit_k)
        logits = score - self.biases  # (B, K-1), broadcast
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        预测类别

        Y = sum_{k=0}^{K-2} I(P(Y > k) > 0.5)
        """
        logits = self.forward(x)  # (B, K-1)
        probs = torch.sigmoid(logits)  # P(Y > k)

        # 预测 = 累积概率超过0.5的数量
        predictions = (probs > 0.5).sum(dim=-1).long()
        return predictions

    def to_class_probs(self, x: torch.Tensor) -> torch.Tensor:
        """
        将累积概率转换为类别概率

        P(Y = k) = P(Y > k-1) - P(Y > k)
        特殊情况: P(Y = 0) = 1 - P(Y > 0)
                  P(Y = K-1) = P(Y > K-2)

        Returns:
            class_probs: (B, num_classes)
        """
        logits = self.forward(x)  # (B, K-1)
        cumprobs = torch.sigmoid(logits)  # P(Y > k), shape (B, K-1)

        # 构建类别概率
        # P(Y = 0) = 1 - P(Y > 0)
        # P(Y = k) = P(Y > k-1) - P(Y > k) for k in [1, K-2]
        # P(Y = K-1) = P(Y > K-2)

        batch_size = x.size(0)
        class_probs = torch.zeros(batch_size, self.num_classes, device=x.device)

        class_probs[:, 0] = 1 - cumprobs[:, 0]
        for k in range(1, self.num_classes - 1):
            class_probs[:, k] = cumprobs[:, k - 1] - cumprobs[:, k]
        class_probs[:, -1] = cumprobs[:, -1]

        # 确保概率非负 (数值稳定性)
        class_probs = F.relu(class_probs) + 1e-7
        class_probs = class_probs / class_probs.sum(dim=-1, keepdim=True)

        return class_probs


class CumulativeLinkHead(nn.Module):
    """
    累积链路模型 (Cumulative Link Model / Proportional Odds Model)

    使用累积softmax: P(Y <= k) = sigmoid(threshold_k - latent_score)
    """

    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.num_classes = num_classes

        self.feature = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)  # 输出单个潜在分数
        )

        # 有序阈值: threshold_0 < threshold_1 < ... < threshold_{K-2}
        # 使用softplus确保正增量
        self.threshold_base = nn.Parameter(torch.tensor(0.0))
        self.threshold_deltas = nn.Parameter(torch.ones(num_classes - 2))

    @property
    def thresholds(self) -> torch.Tensor:
        """获取有序阈值"""
        deltas = F.softplus(self.threshold_deltas)
        cumsum = torch.cumsum(deltas, dim=0)
        return torch.cat([
            self.threshold_base.unsqueeze(0),
            self.threshold_base + cumsum
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            cumulative_logits: (B, K-1) P(Y <= k) 的 logits
        """
        latent = self.feature(x)  # (B, 1)
        thresholds = self.thresholds  # (K-1,)

        # P(Y <= k) = sigmoid(threshold_k - latent)
        cumulative_logits = thresholds.unsqueeze(0) - latent  # (B, K-1)
        return cumulative_logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """预测类别"""
        cum_logits = self.forward(x)
        cum_probs = torch.sigmoid(cum_logits)  # P(Y <= k)

        # 预测 = 第一个 P(Y <= k) > 0.5 的 k
        predictions = (cum_probs < 0.5).sum(dim=-1).long()
        return predictions

    def to_class_probs(self, x: torch.Tensor) -> torch.Tensor:
        """转换为类别概率"""
        cum_logits = self.forward(x)
        cum_probs = torch.sigmoid(cum_logits)  # P(Y <= k), (B, K-1)

        batch_size = x.size(0)
        class_probs = torch.zeros(batch_size, self.num_classes, device=x.device)

        # P(Y = 0) = P(Y <= 0)
        class_probs[:, 0] = cum_probs[:, 0]
        # P(Y = k) = P(Y <= k) - P(Y <= k-1)
        for k in range(1, self.num_classes - 1):
            class_probs[:, k] = cum_probs[:, k] - cum_probs[:, k - 1]
        # P(Y = K-1) = 1 - P(Y <= K-2)
        class_probs[:, -1] = 1 - cum_probs[:, -1]

        class_probs = F.relu(class_probs) + 1e-7
        class_probs = class_probs / class_probs.sum(dim=-1, keepdim=True)

        return class_probs


def create_classification_head(method: str, input_dim: int, num_classes: int,
                               dropout: float = 0.3) -> nn.Module:
    """工厂函数: 创建分类头"""
    if method == 'standard':
        return StandardClassificationHead(input_dim, num_classes, dropout)
    elif method == 'coral':
        return CORALHead(input_dim, num_classes, dropout)
    elif method == 'cumulative':
        return CumulativeLinkHead(input_dim, num_classes, dropout)
    else:
        raise ValueError(f"Unknown ordinal method: {method}")


# ============================================================================
# Action Encoder (动作编码器)
# ============================================================================

class FlexibleActionEncoder(nn.Module):
    """
    灵活动作编码器

    支持可配置的特征选择和序数分类
    """

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.enabled_features = config.get('enabled_features', ['wrinkle'])
        self.ordinal_method = config.get('ordinal_method', 'standard')

        # 创建启用的编码器
        self.encoders = nn.ModuleDict()

        if 'geometric' in self.enabled_features:
            self.encoders['geometric'] = GeometricEncoder(
                max_static_dim=config['max_static_dim'],
                max_dynamic_dim=config['max_dynamic_dim'],
                hidden_dim=128,
                dropout=config['dropout']
            )

        if 'visual' in self.enabled_features:
            self.encoders['visual'] = VisualEncoder(
                visual_dim=config['visual_dim'],
                hidden_dim=128,
                dropout=config['dropout']
            )

        if 'wrinkle' in self.enabled_features:
            self.encoders['wrinkle'] = WrinkleEncoder(
                wrinkle_dim=config['wrinkle_dim'],
                hidden_dim=128,
                dropout=config['dropout']
            )

        if 'motion' in self.enabled_features:
            self.encoders['motion'] = MotionEncoder(
                motion_dim=config['motion_dim'],
                hidden_dim=128,
                dropout=config['dropout']
            )

        # 特征融合
        num_features = len(self.enabled_features)
        fusion_type = config.get('fusion_type', 'concat')
        self.fusion = FlexibleFusion(
            num_features=num_features,
            feature_dim=128,
            output_dim=config['action_embed_dim'],
            dropout=config['dropout'],
            fusion_type=fusion_type
        )

        # Severity分类头 (支持序数分类)
        self.severity_head = create_classification_head(
            method=self.ordinal_method,
            input_dim=config['action_embed_dim'],
            num_classes=config['num_severity_classes'],
            dropout=config['dropout']
        )

    def forward(self, static, dynamic, visual, wrinkle, motion) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            action_embed: (batch, action_embed_dim)
            severity_output:
                - standard: logits (batch, num_classes)
                - coral/cumulative: cumulative logits (batch, num_classes-1)
        """
        # 编码启用的特征
        encoded_features = []

        if 'geometric' in self.enabled_features:
            encoded_features.append(self.encoders['geometric'](static, dynamic))

        if 'visual' in self.enabled_features:
            encoded_features.append(self.encoders['visual'](visual))

        if 'wrinkle' in self.enabled_features:
            encoded_features.append(self.encoders['wrinkle'](wrinkle))

        if 'motion' in self.enabled_features:
            encoded_features.append(self.encoders['motion'](motion))

        # 融合
        action_embed = self.fusion(encoded_features)

        # Severity分类
        severity_output = self.severity_head(action_embed)

        return action_embed, severity_output

    def get_severity_probs(self, severity_output: torch.Tensor) -> torch.Tensor:
        """获取类别概率 (用于聚合器)"""
        if self.ordinal_method == 'standard':
            return F.softmax(severity_output, dim=-1)
        else:
            # CORAL 或 cumulative 需要转换
            # 这里需要访问 severity_head 的 to_class_probs
            # 但 forward 只返回 output，我们需要重新计算
            # 为简化，直接用 sigmoid 近似
            probs = torch.sigmoid(severity_output)
            # 转换为近似类别概率
            batch_size = severity_output.size(0)
            num_classes = severity_output.size(1) + 1

            class_probs = torch.zeros(batch_size, num_classes, device=severity_output.device)

            if self.ordinal_method == 'coral':
                # P(Y > k) -> P(Y = k)
                class_probs[:, 0] = 1 - probs[:, 0]
                for k in range(1, num_classes - 1):
                    class_probs[:, k] = probs[:, k - 1] - probs[:, k]
                class_probs[:, -1] = probs[:, -1]
            else:
                # cumulative: P(Y <= k) -> P(Y = k)
                class_probs[:, 0] = probs[:, 0]
                for k in range(1, num_classes - 1):
                    class_probs[:, k] = probs[:, k] - probs[:, k - 1]
                class_probs[:, -1] = 1 - probs[:, -1]

            class_probs = F.relu(class_probs) + 1e-7
            class_probs = class_probs / class_probs.sum(dim=-1, keepdim=True)

            return class_probs


# ============================================================================
# Session Aggregator (检查级聚合器)
# ============================================================================

class SessionAggregator(nn.Module):
    """检查级聚合器"""

    def __init__(self, action_embed_dim=256, num_actions=11,
                 num_severity_classes=5, dropout=0.3,
                 ordinal_method='standard'):
        super().__init__()
        self.num_actions = num_actions
        self.ordinal_method = ordinal_method

        # 拼接 action_embed + severity_prob
        aggregator_input_dim = action_embed_dim + num_severity_classes

        # 动作位置编码
        self.action_pos_embed = nn.Parameter(
            torch.randn(1, num_actions, aggregator_input_dim) * 0.02
        )

        # Transformer聚合
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=aggregator_input_dim,
                nhead=3,
                dim_feedforward=aggregator_input_dim * 2,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=2
        )

        # Session表示
        session_dim = 512
        self.session_proj = nn.Sequential(
            nn.Linear(aggregator_input_dim * num_actions, session_dim),
            nn.LayerNorm(session_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 多任务输出头 (session级使用标准分类，因为类别少且无序数关系)
        self.has_palsy_head = self._make_head(session_dim, 2, dropout)
        self.palsy_side_head = self._make_head(session_dim, 3, dropout)

        # HB Grade 使用序数分类
        self.hb_grade_head = create_classification_head(
            method=ordinal_method,
            input_dim=session_dim,
            num_classes=6,
            dropout=dropout
        )

        # Sunnybrook 回归
        self.sunnybrook_head = nn.Sequential(
            nn.Linear(session_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def _make_head(self, in_dim, out_dim, dropout):
        return nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim)
        )

    def forward(self, action_embeds, severity_probs, action_mask):
        """
        Args:
            action_embeds: (batch, num_actions, action_embed_dim)
            severity_probs: (batch, num_actions, num_severity_classes)
            action_mask: (batch, num_actions)
        """
        # 拼接
        augmented_embeds = torch.cat([action_embeds, severity_probs], dim=-1)

        # 位置编码
        augmented_embeds = augmented_embeds + self.action_pos_embed

        # Padding mask
        if action_mask is not None:
            padding_mask = ~action_mask.bool()
        else:
            padding_mask = None

        # Transformer聚合
        aggregated = self.transformer(augmented_embeds, src_key_padding_mask=padding_mask)

        # Session表示
        session_repr = self.session_proj(aggregated.flatten(1))

        # 多任务输出
        return {
            'has_palsy': self.has_palsy_head(session_repr),
            'palsy_side': self.palsy_side_head(session_repr),
            'hb_grade': self.hb_grade_head(session_repr),
            'sunnybrook': self.sunnybrook_head(session_repr).squeeze(-1)
        }


# ============================================================================
# Complete H-GFA Net (完整模型)
# ============================================================================

class HGFANet(nn.Module):
    """
    H-GFA Net: 灵活可配置版本

    配置示例:
        # 只用wrinkle + 标准分类
        config['enabled_features'] = ['wrinkle']
        config['ordinal_method'] = 'standard'

        # 用wrinkle + geometric + CORAL序数分类
        config['enabled_features'] = ['wrinkle', 'geometric']
        config['ordinal_method'] = 'coral'
    """

    ACTION_NAMES = HierarchicalPalsyDataset.ACTION_NAMES

    def __init__(self, config=None):
        super().__init__()

        if config is None:
            config = self.get_default_config()
        self.config = config

        self.num_actions = config['num_actions']
        self.ordinal_method = config.get('ordinal_method', 'standard')

        # 动作编码器
        self.action_encoder = FlexibleActionEncoder(config)

        # 检查级聚合器
        self.session_aggregator = SessionAggregator(
            action_embed_dim=config['action_embed_dim'],
            num_actions=config['num_actions'],
            num_severity_classes=config['num_severity_classes'],
            dropout=config['dropout'],
            ordinal_method=self.ordinal_method
        )

        print(f"✓ HGFANet 初始化完成")
        print(f"  - 启用特征: {config.get('enabled_features', ['wrinkle'])}")
        print(f"  - 序数方法: {self.ordinal_method}")
        print(f"  - 融合类型: {config.get('fusion_type', 'concat')}")

    @staticmethod
    def get_default_config():
        return {
            # 特征维度
            'max_static_dim': 11,
            'max_dynamic_dim': 8,
            'visual_dim': 1280,
            'wrinkle_dim': 10,
            'motion_dim': 12,

            # 模型维度
            'action_embed_dim': 256,
            'num_actions': 11,

            # 任务
            'num_severity_classes': 5,
            'num_has_palsy_classes': 2,
            'num_palsy_side_classes': 3,
            'num_hb_grade_classes': 6,

            # 训练
            'dropout': 0.3,

            # === 新增配置 ===
            # 启用的特征: ['geometric', 'visual', 'wrinkle', 'motion']
            'enabled_features': ['wrinkle'],  # 默认只用wrinkle

            # 序数分类方法: 'standard', 'coral', 'cumulative'
            'ordinal_method': 'standard',

            # 融合类型: 'concat' 或 'attention'
            'fusion_type': 'concat',
        }

    def forward(self, batch):
        action_mask = batch['action_mask']
        device = action_mask.device
        batch_size = action_mask.size(0)
        num_actions = self.num_actions

        action_embed_dim = self.config['action_embed_dim']
        num_severity_classes = self.config['num_severity_classes']

        # 预分配
        all_action_embeddings = torch.zeros(
            batch_size, num_actions, action_embed_dim, device=device
        )
        all_severity_probs = torch.zeros(
            batch_size, num_actions, num_severity_classes, device=device
        )

        # 存储原始输出 (用于损失计算)
        if self.ordinal_method == 'standard':
            all_severity_logits = torch.zeros(
                batch_size, num_actions, num_severity_classes, device=device
            )
        else:
            all_severity_logits = torch.zeros(
                batch_size, num_actions, num_severity_classes - 1, device=device
            )

        # 遍历动作
        for action_idx, action_name in enumerate(self.ACTION_NAMES):
            if action_name not in batch['actions']:
                continue

            action_data = batch['actions'][action_name]
            exam_indices = batch['action_indices'].get(action_name, [])
            if len(exam_indices) == 0:
                continue

            static = action_data['static'].to(device)
            dynamic = action_data['dynamic'].to(device)
            visual = action_data['visual'].to(device)
            wrinkle = action_data['wrinkle'].to(device)
            motion = action_data['motion'].to(device)

            action_embed, severity_output = self.action_encoder(
                static, dynamic, visual, wrinkle, motion
            )

            # 获取类别概率
            severity_probs = self.action_encoder.get_severity_probs(severity_output)

            exam_indices_tensor = torch.tensor(exam_indices, dtype=torch.long, device=device)
            all_action_embeddings[exam_indices_tensor, action_idx, :] = action_embed
            all_severity_probs[exam_indices_tensor, action_idx, :] = severity_probs
            all_severity_logits[exam_indices_tensor, action_idx, :] = severity_output

        # Session聚合
        session_outputs = self.session_aggregator(
            action_embeds=all_action_embeddings,
            severity_probs=all_severity_probs,
            action_mask=action_mask
        )

        return {
            'action_severity': all_severity_logits,
            'action_severity_probs': all_severity_probs,
            'session_outputs': session_outputs
        }


# ============================================================================
# Loss Functions (损失函数)
# ============================================================================

class CORALLoss(nn.Module):
    """CORAL损失函数"""

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, K-1) 累积logits
            targets: (B,) 类别标签 0 to K-1
        """
        # 创建累积标签: label_k = 1 if y > k else 0
        batch_size = logits.size(0)
        num_thresholds = self.num_classes - 1

        # targets: (B,) -> (B, K-1)
        targets_expanded = targets.unsqueeze(1).expand(-1, num_thresholds)
        thresholds = torch.arange(num_thresholds, device=logits.device).unsqueeze(0)

        # cumulative_labels[i, k] = 1 if targets[i] > k else 0
        cumulative_labels = (targets_expanded > thresholds).float()

        # Binary cross-entropy for each threshold
        loss = F.binary_cross_entropy_with_logits(logits, cumulative_labels, reduction='mean')

        return loss


class CumulativeLinkLoss(nn.Module):
    """累积链路损失函数"""

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, K-1) P(Y <= k) 的 logits
            targets: (B,) 类别标签 0 to K-1
        """
        batch_size = logits.size(0)
        num_thresholds = self.num_classes - 1

        # 累积标签: label_k = 1 if y <= k else 0
        targets_expanded = targets.unsqueeze(1).expand(-1, num_thresholds)
        thresholds = torch.arange(num_thresholds, device=logits.device).unsqueeze(0)

        cumulative_labels = (targets_expanded <= thresholds).float()

        loss = F.binary_cross_entropy_with_logits(logits, cumulative_labels, reduction='mean')

        return loss


class FlexibleMultiTaskLoss(nn.Module):
    """
    灵活多任务损失

    支持:
    - 标准分类: CrossEntropyLoss
    - CORAL序数分类: CORALLoss
    - 累积链路序数分类: CumulativeLinkLoss
    """

    def __init__(self, ordinal_method='standard', use_uncertainty_weighting=True):
        super().__init__()

        self.ordinal_method = ordinal_method
        self.use_uncertainty_weighting = use_uncertainty_weighting

        # 标准损失
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.mse_loss = nn.MSELoss()

        # 序数损失
        if ordinal_method == 'coral':
            self.ordinal_severity_loss = CORALLoss(num_classes=5)
            self.ordinal_hb_loss = CORALLoss(num_classes=6)
        elif ordinal_method == 'cumulative':
            self.ordinal_severity_loss = CumulativeLinkLoss(num_classes=5)
            self.ordinal_hb_loss = CumulativeLinkLoss(num_classes=6)

        if use_uncertainty_weighting:
            self.log_vars = nn.Parameter(torch.zeros(5))
        else:
            self.register_buffer('weights', torch.tensor([
                1.0,  # action_severity
                1.0,  # has_palsy
                1.0,  # palsy_side
                1.5,  # hb_grade
                0.5  # sunnybrook
            ]))

    def forward(self, outputs, targets):
        losses = {}

        # 1. Action-level severity loss
        if 'action_severity' in outputs and 'action_severity' in targets:
            severity_logits = outputs['action_severity']
            severity_targets = targets['action_severity']

            if self.ordinal_method == 'standard':
                severity_logits_flat = severity_logits.view(-1, 5)
                severity_targets_flat = severity_targets.view(-1)
                losses['action_severity'] = self.ce_loss(severity_logits_flat, severity_targets_flat)
            else:
                # 序数损失需要过滤掉 -1 (ignore_index)
                severity_logits_flat = severity_logits.view(-1, severity_logits.size(-1))
                severity_targets_flat = severity_targets.view(-1)

                valid_mask = severity_targets_flat != -1
                if valid_mask.sum() > 0:
                    valid_logits = severity_logits_flat[valid_mask]
                    valid_targets = severity_targets_flat[valid_mask]
                    losses['action_severity'] = self.ordinal_severity_loss(valid_logits, valid_targets)
                else:
                    losses['action_severity'] = torch.tensor(0.0, device=severity_logits.device)

        # 2. Session-level tasks
        session_outputs = outputs['session_outputs']

        losses['has_palsy'] = self.ce_loss(
            session_outputs['has_palsy'],
            targets['has_palsy']
        )

        losses['palsy_side'] = self.ce_loss(
            session_outputs['palsy_side'],
            targets['palsy_side']
        )

        # HB Grade
        hb_output = session_outputs['hb_grade']
        if self.ordinal_method == 'standard':
            losses['hb_grade'] = self.ce_loss(hb_output, targets['hb_grade'])
        else:
            losses['hb_grade'] = self.ordinal_hb_loss(hb_output, targets['hb_grade'])

        # Sunnybrook (回归)
        losses['sunnybrook'] = self.mse_loss(
            session_outputs['sunnybrook'],
            targets['sunnybrook'] / 100.0  # 归一化
        )

        # 计算总损失
        if self.use_uncertainty_weighting:
            total_loss = 0
            task_names = ['action_severity', 'has_palsy', 'palsy_side', 'hb_grade', 'sunnybrook']
            for i, task_name in enumerate(task_names):
                if task_name in losses:
                    precision = torch.exp(-self.log_vars[i])
                    total_loss += precision * losses[task_name] + self.log_vars[i]
        else:
            total_loss = 0
            task_names = ['action_severity', 'has_palsy', 'palsy_side', 'hb_grade', 'sunnybrook']
            for i, task_name in enumerate(task_names):
                if task_name in losses:
                    total_loss += self.weights[i] * losses[task_name]

        loss_dict = {k: v.item() for k, v in losses.items()}
        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("测试灵活配置的 HGFANet")
    print("=" * 60)

    # 配置1: 只用wrinkle + 标准分类
    print("\n--- 配置1: 只用wrinkle特征 + 标准分类 ---")
    config1 = HGFANet.get_default_config()
    config1['enabled_features'] = ['wrinkle']
    config1['ordinal_method'] = 'standard'

    model1 = HGFANet(config1)
    params1 = sum(p.numel() for p in model1.parameters())
    print(f"参数量: {params1:,}")

    # 配置2: wrinkle + CORAL序数分类
    print("\n--- 配置2: 只用wrinkle特征 + CORAL序数分类 ---")
    config2 = HGFANet.get_default_config()
    config2['enabled_features'] = ['wrinkle']
    config2['ordinal_method'] = 'coral'

    model2 = HGFANet(config2)
    params2 = sum(p.numel() for p in model2.parameters())
    print(f"参数量: {params2:,}")

    # 配置3: wrinkle + geometric + 累积链路
    print("\n--- 配置3: wrinkle + geometric + 累积链路序数分类 ---")
    config3 = HGFANet.get_default_config()
    config3['enabled_features'] = ['wrinkle', 'geometric']
    config3['ordinal_method'] = 'cumulative'

    model3 = HGFANet(config3)
    params3 = sum(p.numel() for p in model3.parameters())
    print(f"参数量: {params3:,}")

    # 配置4: 全部特征 + 注意力融合 + CORAL
    print("\n--- 配置4: 全部特征 + 注意力融合 + CORAL ---")
    config4 = HGFANet.get_default_config()
    config4['enabled_features'] = ['geometric', 'visual', 'wrinkle', 'motion']
    config4['ordinal_method'] = 'coral'
    config4['fusion_type'] = 'attention'

    model4 = HGFANet(config4)
    params4 = sum(p.numel() for p in model4.parameters())
    print(f"参数量: {params4:,}")

    # 模拟前向传播
    print("\n--- 测试前向传播 ---")
    batch_size = 4
    ACTION_NAMES = HierarchicalPalsyDataset.ACTION_NAMES

    batch = {
        'actions': {},
        'action_indices': {},
        'action_mask': torch.ones(batch_size, 11)
    }

    for idx, action_name in enumerate(ACTION_NAMES):
        batch['actions'][action_name] = {
            'static': torch.randn(batch_size, 11),
            'dynamic': torch.randn(batch_size, 8),
            'visual': torch.randn(batch_size, 1280),
            'wrinkle': torch.randn(batch_size, 10),
            'motion': torch.randn(batch_size, 12)
        }
        batch['action_indices'][action_name] = list(range(batch_size))

    batch['targets'] = {
        'action_severity': torch.randint(0, 5, (batch_size, 11)),
        'has_palsy': torch.randint(0, 2, (batch_size,)),
        'palsy_side': torch.randint(0, 3, (batch_size,)),
        'hb_grade': torch.randint(0, 6, (batch_size,)),
        'sunnybrook': torch.rand(batch_size) * 100
    }

    # 测试模型1
    outputs1 = model1(batch)
    print(f"Model1 action_severity shape: {outputs1['action_severity'].shape}")
    print(f"Model1 hb_grade shape: {outputs1['session_outputs']['hb_grade'].shape}")

    # 测试模型2 (CORAL)
    outputs2 = model2(batch)
    print(f"Model2 (CORAL) action_severity shape: {outputs2['action_severity'].shape}")
    print(f"Model2 (CORAL) hb_grade shape: {outputs2['session_outputs']['hb_grade'].shape}")

    # 测试损失函数
    print("\n--- 测试损失函数 ---")
    loss_fn1 = FlexibleMultiTaskLoss(ordinal_method='standard')
    loss_fn2 = FlexibleMultiTaskLoss(ordinal_method='coral')

    loss1, dict1 = loss_fn1(outputs1, batch['targets'])
    print(f"Standard loss: {loss1.item():.4f}")

    loss2, dict2 = loss_fn2(outputs2, batch['targets'])
    print(f"CORAL loss: {loss2.item():.4f}")

    print("\n✓ 所有测试通过!")