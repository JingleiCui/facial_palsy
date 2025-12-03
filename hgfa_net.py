"""
H-GFA Net: Hierarchical Geometric-Feature Attention Network
端到端多任务学习模型
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


class CDCAF(nn.Module):
    """Clinical-Driven Cross-Attention Fusion - 几何特征融合"""

    def __init__(self, max_static_dim=11, max_dynamic_dim=8, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 静态特征编码器 (处理变长输入)
        self.static_encoder = nn.Sequential(
            nn.Linear(max_static_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        # 动态特征编码器
        self.dynamic_encoder = nn.Sequential(
            nn.Linear(max_dynamic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, static_feat, dynamic_feat, static_mask=None, dynamic_mask=None):
        """
        Args:
            static_feat: (batch, max_static_dim) - 零填充的静态特征
            dynamic_feat: (batch, max_dynamic_dim) - 零填充的动态特征
        Returns:
            geo_refined: (batch, hidden_dim)
        """
        static_enc = self.static_encoder(static_feat)
        dynamic_enc = self.dynamic_encoder(dynamic_feat)

        # 门控融合
        concat = torch.cat([static_enc, dynamic_enc], dim=-1)
        gate_weight = self.gate(concat)
        fused = gate_weight * static_enc + (1 - gate_weight) * dynamic_enc

        return self.output_proj(fused)


class GQCA(nn.Module):
    """Geometry-guided Query Cross-Attention - 几何引导视觉特征"""

    def __init__(self, geo_dim=128, visual_dim=1280, wrinkle_dim=10,
                 wrinkle_heatmap_dim=64, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 皱纹热力图编码器 (如果使用热力图)
        self.wrinkle_heatmap_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),  # 假设输入 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, wrinkle_heatmap_dim)
        )

        # 扩展视觉维度 = visual + wrinkle_scalar + wrinkle_heatmap
        extended_visual_dim = visual_dim + wrinkle_dim + wrinkle_heatmap_dim

        # FiLM调制: 用几何特征调制视觉特征
        self.film_gamma = nn.Linear(geo_dim, extended_visual_dim)
        self.film_beta = nn.Linear(geo_dim, extended_visual_dim)

        # 视觉特征降维
        self.visual_proj = nn.Sequential(
            nn.Linear(extended_visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        # 几何查询投影
        self.geo_query_proj = nn.Linear(geo_dim, hidden_dim)

        # 交叉注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.2,
            batch_first=True
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

    def forward(self, geo_feat, visual_feat, wrinkle_scalar, wrinkle_heatmap=None):
        """
        Args:
            geo_feat: (batch, geo_dim) - 来自CDCAF
            visual_feat: (batch, 1280) - MobileNetV3特征
            wrinkle_scalar: (batch, 10) - 皱纹标量特征
            wrinkle_heatmap: (batch, 1, H, W) - 皱纹热力图 (可选)
        """
        batch_size = geo_feat.size(0)

        # 编码皱纹热力图
        if wrinkle_heatmap is not None:
            wrinkle_hm_feat = self.wrinkle_heatmap_encoder(wrinkle_heatmap)
        else:
            wrinkle_hm_feat = torch.zeros(batch_size, 64, device=geo_feat.device)

        # 拼接视觉特征
        extended_visual = torch.cat([visual_feat, wrinkle_scalar, wrinkle_hm_feat], dim=-1)

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
    """Multi-modal Fusion Attention - 最终多模态融合"""

    def __init__(self, geo_dim=128, visual_guided_dim=256, motion_dim=12,
                 output_dim=256):
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
                dropout=0.3,
                batch_first=True
            ),
            num_layers=1
        )

        self.output_norm = nn.LayerNorm(output_dim)

    def forward(self, geo_feat, visual_guided_feat, motion_feat):
        """
        Returns:
            action_embedding: (batch, output_dim) - 单个动作的最终表示
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
    """单个动作的完整编码器: 原始特征 → 动作嵌入"""

    def __init__(self, config):
        super().__init__()
        self.cdcaf = CDCAF(
            max_static_dim=config['max_static_dim'],
            max_dynamic_dim=config['max_dynamic_dim'],
            hidden_dim=config['geo_hidden_dim']
        )
        self.gqca = GQCA(
            geo_dim=config['geo_hidden_dim'],
            visual_dim=config['visual_dim'],
            wrinkle_dim=config['wrinkle_dim'],
            hidden_dim=config['gqca_hidden_dim']
        )
        self.mfa = MFA(
            geo_dim=config['geo_hidden_dim'],
            visual_guided_dim=config['gqca_hidden_dim'],
            motion_dim=config['motion_dim'],
            output_dim=config['action_embed_dim']
        )

        # 动作级严重程度分类头
        self.severity_head = nn.Sequential(
            nn.Linear(config['action_embed_dim'], 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 5)  # 5个严重程度等级
        )

    def forward(self, static, dynamic, visual, wrinkle_scalar,
                wrinkle_heatmap, motion):
        """
        Returns:
            action_embed: (batch, action_embed_dim)
            severity_logits: (batch, 5)
        """
        geo = self.cdcaf(static, dynamic)
        visual_guided = self.gqca(geo, visual, wrinkle_scalar, wrinkle_heatmap)
        action_embed = self.mfa(geo, visual_guided, motion)
        severity_logits = self.severity_head(action_embed)

        return action_embed, severity_logits


class SessionAggregator(nn.Module):
    """检查级聚合器: 聚合11个动作 → 整体诊断"""

    def __init__(self, action_embed_dim=256, num_actions=11):
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
            dropout=0.2,
            batch_first=True
        )

        # 可学习的查询token (用于聚合)
        self.session_query = nn.Parameter(torch.randn(1, 1, action_embed_dim) * 0.02)

        # 最终投影
        self.output_proj = nn.Sequential(
            nn.Linear(action_embed_dim, action_embed_dim),
            nn.LayerNorm(action_embed_dim),
            nn.GELU()
        )

        # 整体级任务头
        self.has_palsy_head = nn.Linear(action_embed_dim, 2)  # 是否面瘫
        self.palsy_side_head = nn.Linear(action_embed_dim, 3)  # 左/右/无
        self.hb_grade_head = nn.Linear(action_embed_dim, 6)  # HB 1-6级
        self.sunnybrook_head = nn.Linear(action_embed_dim, 1)  # 回归分数

    def forward(self, action_embeddings: torch.Tensor, action_mask: torch.Tensor = None):
        """
        Args:
            action_embeddings: (batch, num_actions, action_embed_dim)
            action_mask: (batch, num_actions) - 1=有效, 0=缺失
        Returns:
            session_embed: (batch, action_embed_dim)
            task_outputs: dict of logits/scores
        """
        batch_size = action_embeddings.size(0)

        # 添加位置编码
        x = action_embeddings + self.action_pos_embed

        # 扩展session query
        query = self.session_query.expand(batch_size, -1, -1)

        # 注意力聚合
        session_embed, attn_weights = self.attention_aggregator(
            query, x, x,
            key_padding_mask=(~action_mask.bool()) if action_mask is not None else None
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
            'attention_weights': attn_weights
        }

        return session_embed, outputs


class HGFANet(nn.Module):
    """
    H-GFA Net: 完整的层级多任务学习网络

    层级结构:
    1. 动作级 (Action-Level): 对每个动作预测严重程度
    2. 检查级 (Session-Level): 聚合所有动作，预测整体诊断
    """

    def __init__(self, config=None):
        super().__init__()

        if config is None:
            config = self.default_config()

        self.config = config
        self.num_actions = config['num_actions']

        # 共享的动作编码器 (所有动作共用)
        self.action_encoder = ActionEncoder(config)

        # 检查级聚合器
        self.session_aggregator = SessionAggregator(
            action_embed_dim=config['action_embed_dim'],
            num_actions=config['num_actions']
        )

    @staticmethod
    def default_config():
        return {
            'num_actions': 11,
            'max_static_dim': 11,  # 最大静态特征维度
            'max_dynamic_dim': 8,  # 最大动态特征维度
            'visual_dim': 1280,  # MobileNetV3输出
            'wrinkle_dim': 10,  # 皱纹标量特征
            'motion_dim': 12,  # 运动特征
            'geo_hidden_dim': 128,  # CDCAF输出维度
            'gqca_hidden_dim': 256,  # GQCA输出维度
            'action_embed_dim': 256,  # 动作嵌入维度
        }

    def forward(self, batch: Dict) -> Dict:
        """
        Args:
            batch: {
                'actions': {  # 11个动作的特征
                    'Smile': {
                        'static': (B, static_dim),
                        'dynamic': (B, dynamic_dim),
                        'visual': (B, 1280),
                        'wrinkle_scalar': (B, 10),
                        'wrinkle_heatmap': (B, 1, H, W) or None,
                        'motion': (B, 12),
                    },
                    'RaiseEyebrow': {...},
                    ...
                },
                'action_mask': (B, 11),  # 哪些动作有效
            }
        Returns:
            {
                'action_severity': {action_name: (B, 5) for each action},
                'action_embeddings': (B, 11, embed_dim),
                'session_outputs': {
                    'has_palsy': (B, 2),
                    'palsy_side': (B, 3),
                    'hb_grade': (B, 6),
                    'sunnybrook': (B,),
                }
            }
        """
        batch_size = batch['action_mask'].size(0)
        device = batch['action_mask'].device

        action_embeddings = []
        action_severities = {}

        action_names = ['NeutralFace', 'Smile', 'RaiseEyebrow', 'CloseEyeHardly',
                        'CloseEyeSoftly', 'BlowCheek', 'LipPucker', 'ShowTeeth',
                        'ShrugNose', 'SpontaneousEyeBlink', 'VoluntaryEyeBlink']

        for i, action_name in enumerate(action_names):
            action_data = batch['actions'].get(action_name, None)

            if action_data is not None:
                embed, severity = self.action_encoder(
                    static=action_data['static'],
                    dynamic=action_data['dynamic'],
                    visual=action_data['visual'],
                    wrinkle_scalar=action_data['wrinkle_scalar'],
                    wrinkle_heatmap=action_data.get('wrinkle_heatmap', None),
                    motion=action_data['motion']
                )
                action_embeddings.append(embed)
                action_severities[action_name] = severity
            else:
                # 缺失的动作用零向量
                action_embeddings.append(
                    torch.zeros(batch_size, self.config['action_embed_dim'], device=device)
                )

        # Stack为 (B, 11, embed_dim)
        action_embeddings = torch.stack(action_embeddings, dim=1)

        # 检查级聚合
        _, session_outputs = self.session_aggregator(
            action_embeddings,
            batch['action_mask']
        )

        return {
            'action_severity': action_severities,
            'action_embeddings': action_embeddings,
            'session_outputs': session_outputs
        }