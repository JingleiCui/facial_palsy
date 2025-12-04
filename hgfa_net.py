#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H-GFA Net: 改进版层级多任务网络
=============================================

核心改进 (基于MLST-Net思想):
1. **Proxy Task Fusion**: 动作级任务作为辅助任务
2. **Task Cascade**: 先预测severity,再用它辅助检查级诊断
3. **Wrinkle特征独立编码**: 不直接concat,而是通过乘法融合
4. **任务层级建模**: 显式利用任务间的依赖关系

架构流程:
Stage 1: Feature Extraction
  ├─ CDCAF: static + dynamic → geo_feat (128)
  ├─ Visual: MobileNetV3 → visual_feat (1280)
  ├─ Wrinkle: wrinkle_scalar (10) → wrinkle_feat (128)
  └─ Motion: motion_feat (12) → motion_feat (128)

Stage 2: Proxy Task (Action-Level Severity)
  ├─ Multi-modal Fusion → action_embed (256)
  └─ Severity Head → severity_logits (5类)
       [这个任务帮助网络学习面部特征的判别性表示]

Stage 3: Main Task (Session-Level Diagnosis)
  ├─ Input: action_embed + severity_prob
  ├─ Aggregator: 聚合11个动作
  └─ 4 Heads: has_palsy, palsy_side, hb_grade, sunnybrook
       [利用severity信息辅助更复杂的诊断]
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
from dataset_palsy import HierarchicalPalsyDataset

# ============================================================================
# Stage 1: Feature Encoders
# ============================================================================

class CDCAF(nn.Module):
    """几何特征融合 (保持不变)"""

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
    """
    皱纹特征独立编码器
    将10维标量编码到128维,与其他模态维度对齐
    """

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
        """
        Args:
            wrinkle_feat: (batch, 10)
        Returns:
            encoded: (batch, 128)
        """
        return self.encoder(wrinkle_feat)


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


# ============================================================================
# Stage 2: Multi-Modal Fusion (改进版)
# ============================================================================

class ComplementaryFusion(nn.Module):
    """
    互补特征融合 (参考MLST-Net)

    核心思想: 不同模态通过乘法融合,保留各自的判别性信息
    visual ⊗ wrinkle → 视觉-纹理互补
    (visual ⊗ wrinkle) + geo + motion → 多模态融合
    """

    def __init__(self, feature_dim=128, output_dim=256, dropout=0.3):
        super().__init__()

        # 互补融合权重
        self.visual_wrinkle_gate = nn.Parameter(torch.ones(1, feature_dim))

        # 多模态注意力融合
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, visual_feat, wrinkle_feat, geo_feat, motion_feat):
        """
        Args:
            visual_feat: (batch, 128)
            wrinkle_feat: (batch, 128)
            geo_feat: (batch, 128)
            motion_feat: (batch, 128)
        Returns:
            fused: (batch, 256)
        """
        # 1. Visual-Wrinkle互补融合 (element-wise multiply)
        visual_wrinkle = visual_feat * (1 + self.visual_wrinkle_gate * wrinkle_feat)

        # 2. 堆叠所有模态
        modalities = torch.stack([visual_wrinkle, geo_feat, motion_feat], dim=1)  # (B, 3, 128)

        # 3. 自注意力融合
        fused, _ = self.attention(modalities, modalities, modalities)
        fused = fused.mean(dim=1)  # (B, 128)

        # 4. 输出投影
        return self.output_proj(fused)


# ============================================================================
# Stage 3: Action Encoder with Proxy Task
# ============================================================================

class ActionEncoder(nn.Module):
    """
    动作编码器 + 辅助任务

    包含两个输出:
    1. action_embed: 用于检查级聚合
    2. severity_logits: 辅助任务,帮助学习判别性特征
    """

    def __init__(self, config):
        super().__init__()

        # Feature encoders
        self.cdcaf = CDCAF(
            max_static_dim=config['max_static_dim'],
            max_dynamic_dim=config['max_dynamic_dim'],
            hidden_dim=128,
            dropout=config['dropout']
        )

        self.visual_encoder = VisualEncoder(
            visual_dim=config['visual_dim'],
            hidden_dim=128,
            dropout=config['dropout']
        )

        self.wrinkle_encoder = WrinkleEncoder(
            wrinkle_dim=config['wrinkle_dim'],
            hidden_dim=128,
            dropout=config['dropout']
        )

        self.motion_encoder = MotionEncoder(
            motion_dim=config['motion_dim'],
            hidden_dim=128,
            dropout=config['dropout']
        )

        # Complementary fusion
        self.fusion = ComplementaryFusion(
            feature_dim=128,
            output_dim=config['action_embed_dim'],
            dropout=config['dropout']
        )

        # Proxy task: Action-level severity classification
        self.severity_head = nn.Sequential(
            nn.Linear(config['action_embed_dim'], 128),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(128, config['num_severity_classes'])
        )

    def forward(self, static, dynamic, visual, wrinkle, motion):
        """
        Returns:
            action_embed: (batch, 256) - 用于session聚合
            severity_logits: (batch, 5) - 辅助任务输出
        """
        # 1. Encode each modality
        geo_feat = self.cdcaf(static, dynamic)
        visual_feat = self.visual_encoder(visual)
        wrinkle_feat = self.wrinkle_encoder(wrinkle)
        motion_feat = self.motion_encoder(motion)

        # 2. Multi-modal fusion
        action_embed = self.fusion(visual_feat, wrinkle_feat, geo_feat, motion_feat)

        # 3. Proxy task: severity classification
        severity_logits = self.severity_head(action_embed)

        return action_embed, severity_logits


# ============================================================================
# Stage 4: Session Aggregator with Task Cascade
# ============================================================================

class SessionAggregator(nn.Module):
    """
    检查级聚合器 (改进版)

    核心改进: 利用动作级severity概率辅助检查级诊断
    """

    def __init__(self, action_embed_dim=256, num_actions=11,
                 num_severity_classes=5, dropout=0.3):
        super().__init__()
        self.num_actions = num_actions

        # 1. 拼接action_embed + severity_prob
        aggregator_input_dim = action_embed_dim + num_severity_classes

        # 2. 动作位置编码
        self.action_pos_embed = nn.Parameter(
            torch.randn(1, num_actions, aggregator_input_dim) * 0.02
        )

        # 3. Transformer聚合
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

        # 4. 检查级表示
        session_dim = 512
        self.session_proj = nn.Sequential(
            nn.Linear(aggregator_input_dim * num_actions, session_dim),
            nn.LayerNorm(session_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 5. 多任务输出头
        self.has_palsy_head = self._make_head(session_dim, 2)
        self.palsy_side_head = self._make_head(session_dim, 3)
        self.hb_grade_head = self._make_head(session_dim, 6)
        self.sunnybrook_head = nn.Sequential(
            nn.Linear(session_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def _make_head(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, out_dim)
        )

    def forward(self, action_embeds, severity_logits, action_mask):
        """
        Args:
            action_embeds: (batch, num_actions, action_embed_dim)
            severity_logits: (batch, num_actions, 5)
            action_mask: (batch, num_actions) - 1表示有效动作, 0表示该动作缺失
        Returns:
            session_outputs: dict with 4 tasks
        """
        # 1. severity logits → 概率
        severity_probs = torch.softmax(severity_logits, dim=-1)

        # 2. 拼接 action_embed 和 severity_prob
        augmented_embeds = torch.cat([action_embeds, severity_probs], dim=-1)  # (B, A, 261)

        # 3. 添加位置编码
        augmented_embeds = augmented_embeds + self.action_pos_embed  # broadcast 到 (B, A, 261)

        # 4. 构造 padding mask: True 表示要忽略
        if action_mask is not None:
            # action_mask: 1=有效 → padding_mask False; 0=无效 → padding_mask True
            padding_mask = ~action_mask.bool()  # (B, A)
        else:
            padding_mask = None

        # 5. Transformer 聚合
        aggregated = self.transformer(
            augmented_embeds,
            src_key_padding_mask=padding_mask
        )  # (B, A, 261)

        # 6. Flatten 并映射到 session 表示
        session_repr = self.session_proj(aggregated.flatten(1))  # (B, session_dim)

        # 7. 多任务输出
        return {
            'has_palsy': self.has_palsy_head(session_repr),
            'palsy_side': self.palsy_side_head(session_repr),
            'hb_grade': self.hb_grade_head(session_repr),
            'sunnybrook': self.sunnybrook_head(session_repr).squeeze(-1)
        }

# ============================================================================
# Complete H-GFA Net
# ============================================================================

class HGFANet(nn.Module):
    """
    H-GFA Net: 层级多任务网络

    关键点:
    1. 每个样本是一次检查(examination)
    2. 先对每个动作做多模态编码 + severity proxy task
    3. 再把 11 个动作的 embedding 聚合成检查级表征
    4. 用检查级表征做 has_palsy / palsy_side / HB / Sunnybrook 多任务预测
    """
    ACTION_NAMES = ['NeutralFace', 'Smile', 'RaiseEyebrow', 'CloseEyeHardly',
                    'CloseEyeSoftly', 'BlowCheek', 'LipPucker', 'ShowTeeth',
                    'ShrugNose', 'SpontaneousEyeBlink', 'VoluntaryEyeBlink']

    def __init__(self, config=None):
        super().__init__()

        if config is None:
            config = self.get_default_config()
        self.config = config

        # 与数据集保持同一套动作顺序
        self.ACTION_NAMES = HierarchicalPalsyDataset.ACTION_NAMES
        self.num_actions = config['num_actions']
        assert self.num_actions == len(self.ACTION_NAMES), \
            f"num_actions({self.num_actions}) 与 ACTION_NAMES({len(self.ACTION_NAMES)}) 不一致"

        # 动作级编码器（11 个动作共享参数）
        self.action_encoder = ActionEncoder(config)

        # 检查级聚合器
        self.session_aggregator = SessionAggregator(
            action_embed_dim=config['action_embed_dim'],
            num_actions=config['num_actions'],
            num_severity_classes=config['num_severity_classes'],
            dropout=config['dropout']
        )

    @staticmethod
    def get_default_config():
        return {
            # Feature dimensions
            'max_static_dim': 11,
            'max_dynamic_dim': 8,
            'visual_dim': 1280,
            'wrinkle_dim': 10,
            'motion_dim': 12,

            # Model dimensions
            'action_embed_dim': 256,
            'num_actions': 11,

            # Tasks
            'num_severity_classes': 5,
            'num_has_palsy_classes': 2,
            'num_palsy_side_classes': 3,
            'num_hb_grade_classes': 6,

            # Training
            'dropout': 0.3
        }

    def forward(self, batch):
        """
        Args:
            batch: collate_hierarchical 输出的字典, 包含:
                - actions: {action_name: {static, dynamic, visual, wrinkle, motion}}
                  每个动作的 batch 大小不一定等于全局 batch_size
                - action_indices: {action_name: [exam_idx0, exam_idx1, ...]}
                  记录该动作对应的是哪些检查
                - action_mask: (B, 11)  1=该检查有这个动作, 0=缺失
                - targets: {action_severity, has_palsy, ...}
        Returns:
            outputs: dict
                - action_severity: (B, 11, 5)
                - session_outputs: {has_palsy, palsy_side, hb_grade, sunnybrook}
        """
        # 全局 batch_size = 检查数
        action_mask = batch['action_mask']  # (B, 11)
        device = action_mask.device
        batch_size = action_mask.size(0)
        num_actions = self.num_actions

        # 1. 预先分配全量的动作 embedding & severity logits
        action_embed_dim = self.config['action_embed_dim']
        num_severity_classes = self.config['num_severity_classes']

        # (B, 11, 256)
        all_action_embeddings = torch.zeros(
            batch_size, num_actions, action_embed_dim, device=device
        )
        # (B, 11, 5)
        all_severity_logits = torch.zeros(
            batch_size, num_actions, num_severity_classes, device=device
        )

        # 2. 按固定顺序遍历 11 个动作, 用 action_indices 把结果写回全局矩阵
        for action_idx, action_name in enumerate(self.ACTION_NAMES):
            if action_name not in batch['actions']:
                # 这个 batch 里所有检查都缺这个动作 → 保持全 0, 由 action_mask 控制
                continue

            action_data = batch['actions'][action_name]
            exam_indices = batch['action_indices'].get(action_name, [])
            if len(exam_indices) == 0:
                continue

            # 该动作在当前 batch 中的有效样本数 = len(exam_indices)
            # 形状: (n_i, feat_dim)
            static = action_data['static'].to(device)
            dynamic = action_data['dynamic'].to(device)
            visual = action_data['visual'].to(device)
            wrinkle = action_data['wrinkle'].to(device)
            motion = action_data['motion'].to(device)

            # 编码该动作
            action_embed, severity_logits = self.action_encoder(
                static=static,
                dynamic=dynamic,
                visual=visual,
                wrinkle=wrinkle,
                motion=motion
            )  # (n_i, 256), (n_i, 5)

            # 将局部 n_i 样本写回到全局 B 样本的对应检查位置
            exam_indices_tensor = torch.tensor(exam_indices, dtype=torch.long, device=device)
            all_action_embeddings[exam_indices_tensor, action_idx, :] = action_embed
            all_severity_logits[exam_indices_tensor, action_idx, :] = severity_logits

        # 3. 检查级聚合 (使用 action_mask 屏蔽缺失动作)
        session_outputs = self.session_aggregator(
            action_embeds=all_action_embeddings,      # (B, 11, 256)
            severity_logits=all_severity_logits,      # (B, 11, 5)
            action_mask=action_mask                   # (B, 11)
        )

        return {
            'action_severity': all_severity_logits,   # (B, 11, 5)
            'session_outputs': session_outputs
        }

# ============================================================================
# Multi-Task Loss with Task Weighting
# ============================================================================

class HierarchicalMultiTaskLoss(nn.Module):
    """
    层级多任务损失 (改进版)

    关键改进:
    1. 增加action-level severity loss (proxy task)
    2. 使用固定权重或可学习权重平衡任务
    3. 支持任务层级: action级 → session级
    """

    def __init__(self, use_uncertainty_weighting=True):
        super().__init__()

        self.use_uncertainty_weighting = use_uncertainty_weighting

        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.mse_loss = nn.MSELoss()

        if use_uncertainty_weighting:
            # 可学习的任务权重 (log(σ²))
            self.log_vars = nn.Parameter(torch.zeros(5))  # 5个任务
        else:
            # 固定权重
            self.register_buffer('weights', torch.tensor([
                1.0,  # action_severity
                1.0,  # has_palsy
                1.0,  # palsy_side
                1.5,  # hb_grade (更重要)
                0.5  # sunnybrook (回归任务,降低权重)
            ]))

    def forward(self, outputs, targets):
        """
        Args:
            outputs: model outputs dict
            targets: ground truth dict

        Returns:
            total_loss, loss_dict
        """
        losses = {}

        # 1. Action-level severity loss (proxy task)
        if 'action_severity' in outputs and 'action_severity' in targets:
            severity_logits = outputs['action_severity']  # (B, 11, 5)
            severity_targets = targets['action_severity']  # (B, 11)

            # Flatten for CE loss
            severity_logits_flat = severity_logits.view(-1, 5)
            severity_targets_flat = severity_targets.view(-1)

            losses['action_severity'] = self.ce_loss(severity_logits_flat, severity_targets_flat)

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

        losses['hb_grade'] = self.ce_loss(
            session_outputs['hb_grade'],
            targets['hb_grade']
        )

        losses['sunnybrook'] = self.mse_loss(
            session_outputs['sunnybrook'],
            targets['sunnybrook']
        )

        # 3. 计算总损失
        if self.use_uncertainty_weighting:
            # Uncertainty weighting: L = Σ (1/2σ²) * L_i + log(σ)
            total_loss = 0
            task_names = ['action_severity', 'has_palsy', 'palsy_side', 'hb_grade', 'sunnybrook']

            for i, task_name in enumerate(task_names):
                if task_name in losses:
                    precision = torch.exp(-self.log_vars[i])
                    total_loss += precision * losses[task_name] + self.log_vars[i]
        else:
            # Fixed weighting
            total_loss = 0
            task_names = ['action_severity', 'has_palsy', 'palsy_side', 'hb_grade', 'sunnybrook']

            for i, task_name in enumerate(task_names):
                if task_name in losses:
                    total_loss += self.weights[i] * losses[task_name]

        # Add individual losses to dict for monitoring
        loss_dict = {k: v.item() for k, v in losses.items()}
        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


# ============================================================================
# Example Usage
# ============================================================================
ACTION_NAMES = ['NeutralFace', 'Smile', 'RaiseEyebrow', 'CloseEyeHardly',
                    'CloseEyeSoftly', 'BlowCheek', 'LipPucker', 'ShowTeeth',
                    'ShrugNose', 'SpontaneousEyeBlink', 'VoluntaryEyeBlink']

if __name__ == "__main__":
    # 创建模型
    model = HGFANet()

    # 模拟输入
    batch_size = 4
    batch = {
        'actions': {},
        'action_mask': torch.ones(batch_size, 11)
    }

    for action_name in ACTION_NAMES:
        batch['actions'][action_name] = {
            'static': torch.randn(batch_size, 11),
            'dynamic': torch.randn(batch_size, 8),
            'visual': torch.randn(batch_size, 1280),
            'wrinkle': torch.randn(batch_size, 10),
            'motion': torch.randn(batch_size, 12)
        }

    # 前向传播
    outputs = model(batch)

    print("✓ 模型创建成功!")
    print(f"Action severity shape: {outputs['action_severity'].shape}")
    print(f"Has palsy shape: {outputs['session_outputs']['has_palsy'].shape}")
    print(f"HB grade shape: {outputs['session_outputs']['hb_grade'].shape}")

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n总参数量: {total_params:,}")