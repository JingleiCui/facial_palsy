"""
H-GFA Net - 网络架构
===================================

主要更新:
1. 新增皱纹特征编码器 (Wrinkle Encoder)
2. 新增运动特征编码器 (Motion Encoder)
3. 扩展的多模态融合 (5个模态)
4. 多任务学习头 (动作级 + 检查级)
5. 检查级特征聚合

架构概览:

    输入特征:
    ├── 静态几何特征 (5-11维, 动作相关)
    ├── 动态几何特征 (0-8维, 动作相关)
    ├── 视觉特征 (1280维, MobileNetV3)
    ├── 皱纹特征 (10维)
    └── 运动特征 (12维)

    处理阶段:
    ├── Stage 1: 几何特征融合 (CDCAF) → 256维
    ├── Stage 2: 视觉引导注意力 (GQCA) → 256维
    ├── Stage 2b: 皱纹/运动编码 → 各64维
    ├── Stage 3: 多模态融合 (MFA) → 512维
    └── Stage 4: 多任务预测头

    输出:
    ├── 动作级: severity (1-5)
    └── 检查级: palsy, side, hb_grade, sunnybrook
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any

# 导入Stage模块 (保持兼容)
try:
    from stage1_cdcaf import CDCAF
    from stage2_gqca import GQCA
    from stage3_mfa import MFA
except ImportError:
    print("[WARN] 无法导入Stage模块，将使用内置简化版本")
    CDCAF = None
    GQCA = None
    MFA = None

from multi_task_loss import (
    MultiTaskLoss,
    MultiTaskHead,
    ExaminationAggregator,
    TASK_CONFIGS
)


# =============================================================================
# 特征编码器
# =============================================================================

class WrinkleEncoder(nn.Module):
    """
    皱纹特征编码器

    将10维皱纹统计特征编码为高维表示
    """

    def __init__(
            self,
            input_dim: int = 10,
            hidden_dim: int = 32,
            output_dim: int = 64,
            dropout: float = 0.1
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) 皱纹特征

        Returns:
            (B, output_dim) 编码后的特征
        """
        return self.encoder(x)


class MotionEncoder(nn.Module):
    """
    运动特征编码器

    将12维运动统计特征编码为高维表示
    """

    def __init__(
            self,
            input_dim: int = 12,
            hidden_dim: int = 32,
            output_dim: int = 64,
            dropout: float = 0.1
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) 运动特征

        Returns:
            (B, output_dim) 编码后的特征
        """
        return self.encoder(x)


# =============================================================================
# 简化版Stage模块 (备用)
# =============================================================================

class SimpleCDCAF(nn.Module):
    """简化版CDCAF (当无法导入原版时使用)"""

    def __init__(
            self,
            static_dim: int,
            dynamic_dim: int,
            output_dim: int = 256,
            d_model: int = 128
    ):
        super().__init__()

        self.static_dim = static_dim
        self.dynamic_dim = dynamic_dim

        total_dim = static_dim + max(dynamic_dim, 0)

        self.encoder = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Linear(total_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_dim),
        )

    def forward(
            self,
            static: torch.Tensor,
            dynamic: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if dynamic is not None and dynamic.numel() > 0:
            x = torch.cat([static, dynamic], dim=-1)
        else:
            # 补零
            if self.dynamic_dim > 0:
                zeros = torch.zeros(
                    static.size(0), self.dynamic_dim,
                    device=static.device, dtype=static.dtype
                )
                x = torch.cat([static, zeros], dim=-1)
            else:
                x = static

        return self.encoder(x)


class SimpleGQCA(nn.Module):
    """简化版GQCA"""

    def __init__(
            self,
            geo_dim: int = 256,
            visual_dim: int = 1280,
            output_dim: int = 256
    ):
        super().__init__()

        self.geo_proj = nn.Linear(geo_dim, output_dim)
        self.visual_proj = nn.Linear(visual_dim, output_dim)

        self.fusion = nn.Sequential(
            nn.LayerNorm(output_dim * 2),
            nn.Linear(output_dim * 2, output_dim),
            nn.GELU(),
        )

    def forward(
            self,
            geo: torch.Tensor,
            visual: torch.Tensor
    ) -> torch.Tensor:
        geo_feat = self.geo_proj(geo)
        visual_feat = self.visual_proj(visual)

        combined = torch.cat([geo_feat, visual_feat], dim=-1)
        return self.fusion(combined)


class SimpleMFA(nn.Module):
    """简化版MFA"""

    def __init__(
            self,
            geo_dim: int = 256,
            visual_guided_dim: int = 256,
            visual_global_dim: int = 1280,
            output_dim: int = 512
    ):
        super().__init__()

        total_dim = geo_dim + visual_guided_dim + visual_global_dim

        self.fusion = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Linear(total_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(output_dim * 2, output_dim),
        )

    def forward(
            self,
            geo: torch.Tensor,
            visual_guided: torch.Tensor,
            visual_global: torch.Tensor
    ) -> torch.Tensor:
        combined = torch.cat([geo, visual_guided, visual_global], dim=-1)
        return self.fusion(combined)


# =============================================================================
# H-GFA Net 主模型
# =============================================================================

class HGFANet(nn.Module):
    """
    H-GFA Net - 层次化几何引导特征注意力网络

    支持多模态特征融合和多任务学习
    """

    # 动作维度配置
    ACTION_DIMS = {
        'BlowCheek': (5, 2),
        'CloseEyeHardly': (10, 8),
        'CloseEyeSoftly': (7, 4),
        'LipPucker': (5, 2),
        'NeutralFace': (8, 0),
        'RaiseEyebrow': (10, 3),
        'ShowTeeth': (7, 3),
        'ShrugNose': (5, 2),
        'Smile': (11, 4),
        'SpontaneousEyeBlink': (5, 7),
        'VoluntaryEyeBlink': (5, 4),
    }

    def __init__(
            self,
            action_dims: Optional[Dict[str, Tuple[int, int]]] = None,
            visual_dim: int = 1280,
            wrinkle_dim: int = 10,
            motion_dim: int = 12,
            geo_refined_dim: int = 256,
            visual_guided_dim: int = 256,
            wrinkle_encoded_dim: int = 64,
            motion_encoded_dim: int = 64,
            fused_dim: int = 512,
            num_actions: int = 11,
            task_names: Optional[List[str]] = None,
            device: str = 'cpu'
    ):
        """
        初始化

        Args:
            action_dims: {action_name: (static_dim, dynamic_dim)}
            visual_dim: 视觉特征维度
            wrinkle_dim: 皱纹特征维度
            motion_dim: 运动特征维度
            geo_refined_dim: 几何融合输出维度
            visual_guided_dim: 视觉引导输出维度
            wrinkle_encoded_dim: 皱纹编码输出维度
            motion_encoded_dim: 运动编码输出维度
            fused_dim: 最终融合维度
            num_actions: 动作数量
            task_names: 多任务名称列表
            device: 设备
        """
        super().__init__()

        self.action_dims = action_dims or self.ACTION_DIMS
        self.visual_dim = visual_dim
        self.wrinkle_dim = wrinkle_dim
        self.motion_dim = motion_dim
        self.device = device
        self.num_actions = num_actions

        # 默认任务
        if task_names is None:
            task_names = ['severity', 'hb_grading', 'sunnybrook']
        self.task_names = task_names

        # ============ Stage 1: 动作特定的几何编码器 ============
        self.stage1_modules = nn.ModuleDict()

        # 使用CDCAF或简化版
        CDCAFClass = CDCAF if CDCAF is not None else SimpleCDCAF

        for action, (s_dim, d_dim) in self.action_dims.items():
            if CDCAFClass == CDCAF:
                self.stage1_modules[action] = CDCAFClass(
                    static_dim=s_dim,
                    dynamic_dim=d_dim,
                    clinical_dim=0,
                    d_model=128,
                    num_layers=2,
                    num_heads=4,
                    output_dim=geo_refined_dim
                )
            else:
                self.stage1_modules[action] = CDCAFClass(
                    static_dim=s_dim,
                    dynamic_dim=d_dim,
                    output_dim=geo_refined_dim
                )

        # ============ Stage 2: 视觉引导注意力 ============
        GQCAClass = GQCA if GQCA is not None else SimpleGQCA

        if GQCAClass == GQCA:
            self.stage2 = GQCAClass(
                geo_dim=geo_refined_dim,
                visual_dim=visual_dim,
                d_model=256,
                num_heads=8,
                num_layers=2,
                num_tokens=49,
                out_dim=visual_guided_dim
            )
        else:
            self.stage2 = GQCAClass(
                geo_dim=geo_refined_dim,
                visual_dim=visual_dim,
                output_dim=visual_guided_dim
            )

        # ============ Stage 2b: 辅助特征编码 ============
        self.wrinkle_encoder = WrinkleEncoder(
            input_dim=wrinkle_dim,
            hidden_dim=32,
            output_dim=wrinkle_encoded_dim
        )

        self.motion_encoder = MotionEncoder(
            input_dim=motion_dim,
            hidden_dim=32,
            output_dim=motion_encoded_dim
        )

        # ============ Stage 3: 多模态融合 ============
        # 输入: geo_refined + visual_guided + visual_global + wrinkle + motion
        self.multimodal_fusion = nn.Sequential(
            nn.LayerNorm(
                geo_refined_dim + visual_guided_dim + visual_dim +
                wrinkle_encoded_dim + motion_encoded_dim
            ),
            nn.Linear(
                geo_refined_dim + visual_guided_dim + visual_dim +
                wrinkle_encoded_dim + motion_encoded_dim,
                fused_dim * 2
            ),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fused_dim * 2, fused_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # 模态门控 (自适应权重)
        self.modal_gate = nn.Sequential(
            nn.Linear(
                geo_refined_dim + visual_guided_dim + visual_dim +
                wrinkle_encoded_dim + motion_encoded_dim,
                128
            ),
            nn.GELU(),
            nn.Linear(128, 5),  # 5个模态的权重
            nn.Softmax(dim=-1)
        )

        # ============ Stage 4: 多任务预测头 ============
        # 动作级任务头
        action_task_configs = {
            name: TASK_CONFIGS[name]
            for name in task_names
            if name in TASK_CONFIGS and TASK_CONFIGS[name].level == 'action'
        }

        self.action_head = MultiTaskHead(
            input_dim=fused_dim,
            task_configs=action_task_configs,
            hidden_dim=256,
            dropout=0.3
        ) if action_task_configs else None

        # 检查级任务需要聚合
        exam_task_configs = {
            name: TASK_CONFIGS[name]
            for name in task_names
            if name in TASK_CONFIGS and TASK_CONFIGS[name].level == 'examination'
        }

        self.examination_aggregator = ExaminationAggregator(
            input_dim=fused_dim,
            num_actions=num_actions,
            aggregation='attention',
            hidden_dim=256
        ) if exam_task_configs else None

        self.examination_head = MultiTaskHead(
            input_dim=fused_dim,
            task_configs=exam_task_configs,
            hidden_dim=256,
            dropout=0.3
        ) if exam_task_configs else None

        # 移动到设备
        self.to(device)

        # 打印模型信息
        self._print_model_info()

    def _print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print("\n" + "=" * 60)
        print("H-GFA Net")
        print("=" * 60)
        print(f"动作数量: {len(self.action_dims)}")
        print(f"任务列表: {self.task_names}")
        print(f"总参数量: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        print(f"模型大小: ~{total_params * 4 / 1024 / 1024:.1f} MB")
        print("=" * 60)

    def encode_geometry(
            self,
            static: torch.Tensor,
            dynamic: Optional[torch.Tensor],
            action_name: str
    ) -> torch.Tensor:
        """
        编码几何特征 (Stage 1)

        Args:
            static: (B, static_dim) 静态几何特征
            dynamic: (B, dynamic_dim) 动态几何特征 或 None
            action_name: 动作名称

        Returns:
            geo_refined: (B, 256) 融合后的几何特征
        """
        if action_name not in self.stage1_modules:
            raise ValueError(f"Unknown action: {action_name}")

        stage1 = self.stage1_modules[action_name]
        return stage1(static, dynamic)

    def encode_visual(
            self,
            geo_refined: torch.Tensor,
            visual: torch.Tensor
    ) -> torch.Tensor:
        """
        视觉引导编码 (Stage 2)

        Args:
            geo_refined: (B, 256) 几何特征
            visual: (B, 1280) 视觉特征

        Returns:
            visual_guided: (B, 256) 视觉引导特征
        """
        return self.stage2(geo_refined, visual)

    def fuse_all_modalities(
            self,
            geo_refined: torch.Tensor,
            visual_guided: torch.Tensor,
            visual_global: torch.Tensor,
            wrinkle: Optional[torch.Tensor],
            motion: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        多模态融合 (Stage 3)

        Args:
            geo_refined: (B, 256) 几何特征
            visual_guided: (B, 256) 视觉引导特征
            visual_global: (B, 1280) 全局视觉特征
            wrinkle: (B, 10) 皱纹特征 或 None
            motion: (B, 12) 运动特征 或 None

        Returns:
            fused: (B, 512) 融合特征
            gate_weights: (B, 5) 模态权重
        """
        B = geo_refined.size(0)
        device = geo_refined.device
        dtype = geo_refined.dtype

        # 编码辅助特征
        if wrinkle is not None and wrinkle.numel() > 0:
            wrinkle_feat = self.wrinkle_encoder(wrinkle)
        else:
            wrinkle_feat = torch.zeros(B, 64, device=device, dtype=dtype)

        if motion is not None and motion.numel() > 0:
            motion_feat = self.motion_encoder(motion)
        else:
            motion_feat = torch.zeros(B, 64, device=device, dtype=dtype)

        # 拼接所有模态
        all_features = torch.cat([
            geo_refined,  # 256
            visual_guided,  # 256
            visual_global,  # 1280
            wrinkle_feat,  # 64
            motion_feat  # 64
        ], dim=-1)  # Total: 1920

        # 计算模态门控权重
        gate_weights = self.modal_gate(all_features)

        # 融合
        fused = self.multimodal_fusion(all_features)

        return fused, gate_weights

    def forward_single_action(
            self,
            static: torch.Tensor,
            dynamic: Optional[torch.Tensor],
            visual: torch.Tensor,
            wrinkle: Optional[torch.Tensor],
            motion: Optional[torch.Tensor],
            action_name: str
    ) -> Dict[str, Any]:
        """
        处理单个动作

        Returns:
            {
                'fused_features': (B, 512),
                'action_predictions': {task: logits},
                'gate_weights': (B, 5)
            }
        """
        # Stage 1: 几何编码
        geo_refined = self.encode_geometry(static, dynamic, action_name)

        # Stage 2: 视觉引导
        visual_guided = self.encode_visual(geo_refined, visual)

        # Stage 3: 多模态融合
        fused, gate_weights = self.fuse_all_modalities(
            geo_refined, visual_guided, visual, wrinkle, motion
        )

        # Stage 4: 动作级预测
        action_preds = {}
        if self.action_head is not None:
            action_preds = self.action_head(fused)

        return {
            'fused_features': fused,
            'action_predictions': action_preds,
            'gate_weights': gate_weights
        }

    def forward(
            self,
            batch_dict: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, Any]:
        """
        前向传播

        Args:
            batch_dict: 按动作分组的批次数据
                {
                    'Smile': {
                        'static': (B, s_dim),
                        'dynamic': (B, d_dim) or None,
                        'visual': (B, 1280),
                        'wrinkle': (B, 10) or None,
                        'motion': (B, 12) or None,
                        'labels': (B,),
                        ...
                    },
                    ...
                }

        Returns:
            {
                'action_logits': (total_B, num_classes),
                'action_labels': (total_B,),
                'action_features': (total_B, 512),
                'gate_weights': (total_B, 5),
                'examination_predictions': {...} (如果有检查级任务)
            }
        """
        all_features = []
        all_logits = []
        all_labels = []
        all_gate_weights = []

        for action_name, data in batch_dict.items():
            # 移动到设备
            static = data['static'].to(self.device, non_blocking=True)
            dynamic = data.get('dynamic')
            if dynamic is not None:
                dynamic = dynamic.to(self.device, non_blocking=True)

            visual = data['visual'].to(self.device, non_blocking=True)

            wrinkle = data.get('wrinkle')
            if wrinkle is not None:
                wrinkle = wrinkle.to(self.device, non_blocking=True)

            motion = data.get('motion')
            if motion is not None:
                motion = motion.to(self.device, non_blocking=True)

            labels = data['labels'].to(self.device, non_blocking=True)

            # 处理
            result = self.forward_single_action(
                static, dynamic, visual, wrinkle, motion, action_name
            )

            all_features.append(result['fused_features'])
            all_gate_weights.append(result['gate_weights'])
            all_labels.append(labels)

            if 'severity' in result['action_predictions']:
                all_logits.append(result['action_predictions']['severity'])

        # 合并结果
        output = {
            'action_features': torch.cat(all_features, dim=0) if all_features else None,
            'gate_weights': torch.cat(all_gate_weights, dim=0) if all_gate_weights else None,
            'action_labels': torch.cat(all_labels, dim=0) if all_labels else None,
        }

        if all_logits:
            output['action_logits'] = torch.cat(all_logits, dim=0)

        return output


# =============================================================================
# 工厂函数
# =============================================================================

def create_hgfa_net(
        device: str = 'cpu',
        task_names: Optional[List[str]] = None
) -> HGFANet:
    """
    创建 H-GFA Net 模型

    Args:
        device: 设备
        task_names: 任务列表

    Returns:
        HGFANet模型实例
    """
    if task_names is None:
        task_names = ['severity', 'hb_grading', 'sunnybrook']

    model = HGFANet(
        task_names=task_names,
        device=device
    )

    return model


# =============================================================================
# 测试
# =============================================================================

if __name__ == '__main__':
    print("测试 H-GFA Net")

    # 检测设备
    if torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f"使用设备: {device}")

    # 创建模型
    model = create_hgfa_net(device=device)

    # 模拟输入
    batch_size = 4

    batch_dict = {
        'Smile': {
            'static': torch.randn(batch_size, 11),
            'dynamic': torch.randn(batch_size, 4),
            'visual': torch.randn(batch_size, 1280),
            'wrinkle': torch.randn(batch_size, 10),
            'motion': torch.randn(batch_size, 12),
            'labels': torch.randint(0, 5, (batch_size,)),
        },
        'NeutralFace': {
            'static': torch.randn(batch_size, 8),
            'dynamic': None,
            'visual': torch.randn(batch_size, 1280),
            'wrinkle': torch.randn(batch_size, 10),
            'motion': torch.randn(batch_size, 12),
            'labels': torch.randint(0, 5, (batch_size,)),
        },
    }

    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(batch_dict)

    print("\n输出:")
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")

    print("\n✓ 测试通过!")