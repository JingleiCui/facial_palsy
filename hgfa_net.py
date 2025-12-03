#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H-GFA Net: Hierarchical Geometric-Feature Attention Network
端到端层级多任务学习模型

架构:
1. ActionEncoder: 对每个动作编码 (CDCAF -> GQCA -> MFA)
2. SessionAggregator: 聚合11个动作到检查级表示
3. 多任务输出头:
   - 动作级: 严重程度分类 (5类)
   - 检查级: 是否面瘫(2类), 面瘫侧别(3类), HB分级(6类), Sunnybrook(回归)

关键设计:
- 所有特征融合模块都是可训练的 (端到端)
- 支持缺失动作 (通过action_mask处理)
- 不确定性加权的多任务损失

版本: V3.0 (修复batch维度不一致问题)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


class CDCAF(nn.Module):
    """
    Clinical-Driven Cross-Attention Fusion
    几何特征融合: 静态特征 + 动态特征 -> 融合几何特征
    """

    def __init__(self, max_static_dim=11, max_dynamic_dim=8, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 静态特征编码器
        self.static_encoder = nn.Sequential(
            nn.Linear(max_static_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 动态特征编码器
        self.dynamic_encoder = nn.Sequential(
            nn.Linear(max_dynamic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 门控融合机制
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, static_feat, dynamic_feat):
        """
        Args:
            static_feat: (batch, max_static_dim)
            dynamic_feat: (batch, max_dynamic_dim)
        Returns:
            geo_feat: (batch, hidden_dim)
        """
        static_enc = self.static_encoder(static_feat)
        dynamic_enc = self.dynamic_encoder(dynamic_feat)

        # 门控融合
        concat = torch.cat([static_enc, dynamic_enc], dim=-1)
        gate_weight = self.gate(concat)
        fused = gate_weight * static_enc + (1 - gate_weight) * dynamic_enc

        return self.output_proj(fused)


class GQCA(nn.Module):
    """
    Geometry-guided Query Cross-Attention
    几何引导的视觉特征增强: 几何特征 + 视觉特征 + 皱纹特征 -> 增强视觉特征
    """

    def __init__(self, geo_dim=128, visual_dim=1280, wrinkle_dim=10,
                 hidden_dim=256, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 扩展视觉维度 = visual + wrinkle
        extended_visual_dim = visual_dim + wrinkle_dim

        # FiLM调制: 用几何特征调制视觉特征
        self.film_gamma = nn.Linear(geo_dim, extended_visual_dim)
        self.film_beta = nn.Linear(geo_dim, extended_visual_dim)

        # 视觉特征降维
        self.visual_proj = nn.Sequential(
            nn.Linear(extended_visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 几何查询投影
        self.geo_query_proj = nn.Linear(geo_dim, hidden_dim)

        # 交叉注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, geo_feat, visual_feat, wrinkle_scalar):
        """
        Args:
            geo_feat: (batch, geo_dim) - 来自CDCAF
            visual_feat: (batch, 1280) - MobileNetV3特征
            wrinkle_scalar: (batch, 10) - 皱纹标量特征
        Returns:
            visual_guided: (batch, hidden_dim)
        """
        # 拼接视觉特征和皱纹特征
        extended_visual = torch.cat([visual_feat, wrinkle_scalar], dim=-1)

        # FiLM调制
        gamma = self.film_gamma(geo_feat)
        beta = self.film_beta(geo_feat)
        modulated = extended_visual * (1 + gamma) + beta

        # 投影
        visual_proj = self.visual_proj(modulated)  # (batch, hidden_dim)
        geo_query = self.geo_query_proj(geo_feat).unsqueeze(1)  # (batch, 1, hidden_dim)
        visual_kv = visual_proj.unsqueeze(1)  # (batch, 1, hidden_dim)

        # 交叉注意力
        attn_out, _ = self.cross_attention(geo_query, visual_kv, visual_kv)

        # FFN + 残差
        out = self.norm(geo_query.squeeze(1) + attn_out.squeeze(1))
        out = out + self.ffn(out)

        return out


class MFA(nn.Module):
    """
    Multi-modal Fusion Attention
    最终多模态融合: 几何特征 + 视觉引导特征 + 运动特征 -> 动作嵌入
    """

    def __init__(self, geo_dim=128, visual_guided_dim=256, motion_dim=12,
                 output_dim=256, dropout=0.3):
        super().__init__()

        # 各模态投影
        self.geo_proj = nn.Linear(geo_dim, output_dim)
        self.visual_proj = nn.Linear(visual_guided_dim, output_dim)
        self.motion_proj = nn.Linear(motion_dim, output_dim)

        # 动态权重门
        self.modal_gate = nn.Sequential(
            nn.Linear(output_dim * 3, 3),
            nn.Softmax(dim=-1)
        )

        # Transformer融合
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=output_dim,
                nhead=4,
                dim_feedforward=output_dim * 2,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=1
        )

        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, geo_feat, visual_guided_feat, motion_feat):
        """
        Args:
            geo_feat: (batch, geo_dim)
            visual_guided_feat: (batch, visual_guided_dim)
            motion_feat: (batch, motion_dim)
        Returns:
            action_embed: (batch, output_dim)
        """
        geo = self.geo_proj(geo_feat)
        visual = self.visual_proj(visual_guided_feat)
        motion = self.motion_proj(motion_feat)

        # 计算动态权重
        concat = torch.cat([geo, visual, motion], dim=-1)
        weights = self.modal_gate(concat)  # (batch, 3)

        # 加权融合
        weighted = (weights[:, 0:1] * geo +
                    weights[:, 1:2] * visual +
                    weights[:, 2:3] * motion)

        # Transformer refinement
        x = weighted.unsqueeze(1)  # (batch, 1, dim)
        x = self.transformer(x)

        return self.output_norm(x.squeeze(1))


class ActionEncoder(nn.Module):
    """
    单个动作的完整编码器
    原始特征 -> CDCAF -> GQCA -> MFA -> 动作嵌入 + 严重程度分类
    """

    def __init__(self, config):
        super().__init__()

        self.cdcaf = CDCAF(
            max_static_dim=config['max_static_dim'],
            max_dynamic_dim=config['max_dynamic_dim'],
            hidden_dim=config['geo_hidden_dim'],
            dropout=config['dropout']
        )

        self.gqca = GQCA(
            geo_dim=config['geo_hidden_dim'],
            visual_dim=config['visual_dim'],
            wrinkle_dim=config['wrinkle_dim'],
            hidden_dim=config['gqca_hidden_dim'],
            dropout=config['dropout']
        )

        self.mfa = MFA(
            geo_dim=config['geo_hidden_dim'],
            visual_guided_dim=config['gqca_hidden_dim'],
            motion_dim=config['motion_dim'],
            output_dim=config['action_embed_dim'],
            dropout=config['dropout']
        )

        # 动作级严重程度分类头
        self.severity_head = nn.Sequential(
            nn.Linear(config['action_embed_dim'], 64),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(64, config['num_severity_classes'])
        )

    def forward(self, static, dynamic, visual, wrinkle_scalar, motion):
        """
        Returns:
            action_embed: (batch, action_embed_dim)
            severity_logits: (batch, num_severity_classes)
        """
        geo = self.cdcaf(static, dynamic)
        visual_guided = self.gqca(geo, visual, wrinkle_scalar)
        action_embed = self.mfa(geo, visual_guided, motion)
        severity_logits = self.severity_head(action_embed)

        return action_embed, severity_logits


class SessionAggregator(nn.Module):
    """
    检查级聚合器
    聚合11个动作的嵌入 -> 检查级表示 -> 多任务输出
    """

    def __init__(self, action_embed_dim=256, num_actions=11, dropout=0.3):
        super().__init__()
        self.num_actions = num_actions

        # 动作位置编码
        self.action_pos_embed = nn.Parameter(
            torch.randn(1, num_actions, action_embed_dim) * 0.02
        )

        # 注意力聚合
        self.attention_aggregator = nn.MultiheadAttention(
            embed_dim=action_embed_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        # 可学习的查询token
        self.session_query = nn.Parameter(torch.randn(1, 1, action_embed_dim) * 0.02)

        # 最终投影
        self.output_proj = nn.Sequential(
            nn.Linear(action_embed_dim, action_embed_dim),
            nn.LayerNorm(action_embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 检查级任务头
        self.has_palsy_head = nn.Linear(action_embed_dim, 2)       # 是否面瘫
        self.palsy_side_head = nn.Linear(action_embed_dim, 3)      # 无/左/右
        self.hb_grade_head = nn.Linear(action_embed_dim, 6)        # HB 1-6级
        self.sunnybrook_head = nn.Linear(action_embed_dim, 1)      # 回归分数

    def forward(self, action_embeddings: torch.Tensor, action_mask: torch.Tensor = None):
        """
        Args:
            action_embeddings: (batch, num_actions, action_embed_dim)
            action_mask: (batch, num_actions) - 1=有效, 0=缺失
        Returns:
            session_outputs: dict of task outputs
        """
        batch_size = action_embeddings.size(0)

        # 添加位置编码
        x = action_embeddings + self.action_pos_embed

        # 扩展session query
        query = self.session_query.expand(batch_size, -1, -1)

        # 创建key_padding_mask (True表示忽略)
        key_padding_mask = None
        if action_mask is not None:
            key_padding_mask = (action_mask == 0)

        # 注意力聚合
        session_embed, attn_weights = self.attention_aggregator(
            query, x, x,
            key_padding_mask=key_padding_mask
        )
        session_embed = session_embed.squeeze(1)

        # 投影
        session_embed = self.output_proj(session_embed)

        # 任务输出
        outputs = {
            'has_palsy': self.has_palsy_head(session_embed),
            'palsy_side': self.palsy_side_head(session_embed),
            'hb_grade': self.hb_grade_head(session_embed),
            'sunnybrook': self.sunnybrook_head(session_embed).squeeze(-1),
            'attention_weights': attn_weights,
            'session_embed': session_embed
        }

        return outputs


class HGFANet(nn.Module):
    """
    H-GFA Net: 完整的层级多任务学习网络

    层级结构:
    1. 动作级 (Action-Level): 对每个动作预测严重程度
    2. 检查级 (Session-Level): 聚合所有动作，预测整体诊断
    """

    ACTION_NAMES = [
        'NeutralFace', 'Smile', 'RaiseEyebrow', 'CloseEyeHardly',
        'CloseEyeSoftly', 'BlowCheek', 'LipPucker', 'ShowTeeth',
        'ShrugNose', 'SpontaneousEyeBlink', 'VoluntaryEyeBlink'
    ]

    def __init__(self, config=None):
        super().__init__()

        if config is None:
            config = self.default_config()

        self.config = config
        self.num_actions = config['num_actions']

        # 共享的动作编码器 (所有动作共用一个编码器)
        self.action_encoder = ActionEncoder(config)

        # 检查级聚合器
        self.session_aggregator = SessionAggregator(
            action_embed_dim=config['action_embed_dim'],
            num_actions=config['num_actions'],
            dropout=config['dropout']
        )

    @staticmethod
    def default_config():
        return {
            'num_actions': 11,
            'max_static_dim': 11,
            'max_dynamic_dim': 8,
            'visual_dim': 1280,
            'wrinkle_dim': 10,
            'motion_dim': 12,
            'geo_hidden_dim': 128,
            'gqca_hidden_dim': 256,
            'action_embed_dim': 256,
            'num_severity_classes': 5,
            'dropout': 0.3,
        }

    def forward(self, batch: Dict) -> Dict:
        """
        Args:
            batch: {
                'actions': {action_name: {static, dynamic, visual, wrinkle_scalar, motion}},
                'action_indices': {action_name: [exam_indices]},
                'action_mask': (batch, 11),
            }
        Returns:
            {
                'action_severity': {action_name: (N, 5)},
                'action_severity_masks': {action_name: [indices]},
                'action_embeddings': (batch, 11, embed_dim),
                'session_outputs': {...}
            }
        """
        device = batch['action_mask'].device
        batch_size = batch['action_mask'].size(0)

        # 初始化全零的动作嵌入矩阵
        all_action_embeddings = torch.zeros(
            batch_size, self.num_actions, self.config['action_embed_dim'],
            device=device
        )

        action_severities = {}
        action_severity_masks = {}

        # 对每个动作进行编码
        for action_idx, action_name in enumerate(self.ACTION_NAMES):
            if action_name not in batch['actions']:
                continue

            action_data = batch['actions'][action_name]
            exam_indices = batch['action_indices'][action_name]

            if len(exam_indices) == 0:
                continue

            # 编码动作
            embed, severity = self.action_encoder(
                static=action_data['static'],
                dynamic=action_data['dynamic'],
                visual=action_data['visual'],
                wrinkle_scalar=action_data['wrinkle_scalar'],
                motion=action_data['motion']
            )

            # 将embedding放到正确的位置
            for i, exam_idx in enumerate(exam_indices):
                all_action_embeddings[exam_idx, action_idx] = embed[i]

            # 记录severity预测
            action_severities[action_name] = severity
            action_severity_masks[action_name] = exam_indices

        # 检查级聚合
        session_outputs = self.session_aggregator(
            all_action_embeddings,
            batch['action_mask']
        )

        return {
            'action_severity': action_severities,
            'action_severity_masks': action_severity_masks,
            'action_embeddings': all_action_embeddings,
            'session_outputs': session_outputs
        }

    def count_parameters(self):
        """统计模型参数"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}