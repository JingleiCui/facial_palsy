"""
多任务学习模块 (Multi-Task Learning)
====================================

实现策略: Uncertainty Weighting (不确定性加权)

原理:
- 每个任务有一个可学习的log方差参数 (log_sigma)
- 任务损失 = loss / (2 * sigma^2) + log(sigma)
- 自动学习每个任务的权重

参考文献:
- Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses
  for Scene Geometry and Semantics", CVPR 2018

任务列表:
1. 动作级任务:
   - severity_classification: 严重程度分类 (1-5)

2. 检查级任务:
   - palsy_detection: 面瘫检测 (二分类)
   - side_classification: 患侧识别 (三分类: 无/左/右)
   - hb_grading: HB分级 (1-6)
   - sunnybrook_regression: Sunnybrook评分 (回归)

用法:
    from multi_task_loss import MultiTaskLoss

    loss_fn = MultiTaskLoss(tasks=['severity', 'hb', 'sunnybrook'])
    total_loss, task_losses = loss_fn(predictions, targets)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


# =============================================================================
# 任务配置
# =============================================================================

@dataclass
class TaskConfig:
    """任务配置"""
    name: str
    task_type: str  # 'classification', 'regression', 'ordinal'
    num_classes: int = 0  # 分类任务的类别数
    loss_fn: str = 'ce'  # 'ce', 'bce', 'mse', 'l1', 'ordinal'
    weight: float = 1.0  # 初始权重
    level: str = 'action'  # 'action' or 'examination'


# 预定义任务配置
TASK_CONFIGS = {
    # 动作级任务
    'severity': TaskConfig(
        name='severity',
        task_type='classification',
        num_classes=5,
        loss_fn='ce',
        level='action'
    ),

    # 检查级任务
    'palsy_detection': TaskConfig(
        name='palsy_detection',
        task_type='classification',
        num_classes=2,
        loss_fn='bce',
        level='examination'
    ),

    'side_classification': TaskConfig(
        name='side_classification',
        task_type='classification',
        num_classes=3,  # 无/左/右
        loss_fn='ce',
        level='examination'
    ),

    'hb_grading': TaskConfig(
        name='hb_grading',
        task_type='ordinal',  # 有序分类
        num_classes=6,
        loss_fn='ordinal',
        level='examination'
    ),

    'sunnybrook': TaskConfig(
        name='sunnybrook',
        task_type='regression',
        loss_fn='mse',
        level='examination'
    ),
}


# =============================================================================
# Ordinal Loss (CORN Loss for ordinal regression)
# =============================================================================

class OrdinalLoss(nn.Module):
    """
    有序分类损失 (用于HB分级等有序标签)

    使用累积概率方法: P(Y > k) for k = 0, 1, ..., K-1
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(
            self,
            logits: torch.Tensor,  # (B, num_classes)
            targets: torch.Tensor  # (B,) 0-indexed
    ) -> torch.Tensor:
        """
        Args:
            logits: 模型输出
            targets: 目标类别 (0 to K-1)

        Returns:
            loss: 标量损失
        """
        batch_size = logits.size(0)
        num_classes = logits.size(1)

        # 计算累积概率
        probs = F.softmax(logits, dim=1)
        cum_probs = torch.cumsum(probs, dim=1)  # P(Y <= k)

        # 构造目标: 对于类别c，y_k = 1 if k < c, else 0
        targets_expanded = targets.unsqueeze(1).expand(-1, num_classes - 1)
        threshold = torch.arange(1, num_classes, device=logits.device).unsqueeze(0)
        binary_targets = (targets_expanded >= threshold).float()  # (B, K-1)

        # 二元交叉熵
        cum_probs_clipped = torch.clamp(cum_probs[:, :-1], 1e-7, 1 - 1e-7)

        loss = F.binary_cross_entropy(
            cum_probs_clipped,
            binary_targets,
            reduction='mean'
        )

        return loss


# =============================================================================
# Label Smoothing Loss
# =============================================================================

class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑交叉熵损失"""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, C)
            targets: (B,) 类别索引
        """
        num_classes = logits.size(1)

        # 构造平滑标签
        with torch.no_grad():
            smooth_targets = torch.full_like(
                logits,
                self.smoothing / (num_classes - 1)
            )
            smooth_targets.scatter_(
                1,
                targets.unsqueeze(1),
                1.0 - self.smoothing
            )

        # KL散度
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(smooth_targets * log_probs).sum(dim=1).mean()

        return loss


# =============================================================================
# Focal Loss (处理类别不平衡)
# =============================================================================

class FocalLoss(nn.Module):
    """Focal Loss用于处理类别不平衡"""

    def __init__(
            self,
            gamma: float = 2.0,
            alpha: Optional[torch.Tensor] = None
    ):
        """
        Args:
            gamma: focusing parameter
            alpha: 类别权重 (可选)
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(
            self,
            logits: torch.Tensor,
            targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, C)
            targets: (B,)
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')

        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_weight = self.alpha.to(logits.device).gather(0, targets)
            focal_weight = focal_weight * alpha_weight

        loss = (focal_weight * ce_loss).mean()

        return loss


# =============================================================================
# Uncertainty Weighting Multi-Task Loss
# =============================================================================

class MultiTaskLoss(nn.Module):
    """
    多任务学习损失 - Uncertainty Weighting

    自动学习每个任务的权重
    """

    def __init__(
            self,
            task_names: List[str],
            use_focal: bool = True,
            label_smoothing: float = 0.1,
            device: str = 'cpu'
    ):
        """
        初始化

        Args:
            task_names: 要使用的任务名称列表
            use_focal: 是否使用Focal Loss
            label_smoothing: 标签平滑系数
            device: 设备
        """
        super().__init__()

        self.task_names = task_names
        self.device = device

        # 获取任务配置
        self.task_configs = {
            name: TASK_CONFIGS[name]
            for name in task_names
            if name in TASK_CONFIGS
        }

        # 可学习的log方差参数 (Uncertainty Weighting核心)
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1, device=device))
            for name in self.task_configs.keys()
        })

        # 初始化各任务的损失函数
        self.loss_fns = nn.ModuleDict()

        for name, config in self.task_configs.items():
            if config.loss_fn == 'ce':
                if use_focal:
                    self.loss_fns[name] = FocalLoss(gamma=2.0)
                elif label_smoothing > 0:
                    self.loss_fns[name] = LabelSmoothingCrossEntropy(label_smoothing)
                else:
                    self.loss_fns[name] = nn.CrossEntropyLoss()

            elif config.loss_fn == 'bce':
                self.loss_fns[name] = nn.BCEWithLogitsLoss()

            elif config.loss_fn == 'mse':
                self.loss_fns[name] = nn.MSELoss()

            elif config.loss_fn == 'l1':
                self.loss_fns[name] = nn.L1Loss()

            elif config.loss_fn == 'ordinal':
                self.loss_fns[name] = OrdinalLoss(config.num_classes)

        print(f"[MultiTaskLoss] 初始化 {len(self.task_configs)} 个任务:")
        for name in self.task_configs:
            print(f"  - {name}")

    def forward(
            self,
            predictions: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            valid_mask: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算多任务加权损失

        Args:
            predictions: {task_name: prediction_tensor}
            targets: {task_name: target_tensor}
            valid_mask: {task_name: bool_mask} 标记有效样本

        Returns:
            total_loss: 加权总损失
            task_losses: {task_name: task_loss}
        """
        task_losses = {}
        weighted_losses = []

        for name in self.task_configs:
            if name not in predictions or name not in targets:
                continue

            pred = predictions[name]
            target = targets[name]

            # 应用有效掩码
            if valid_mask is not None and name in valid_mask:
                mask = valid_mask[name]
                if mask.sum() == 0:
                    continue
                pred = pred[mask]
                target = target[mask]

            # 计算原始损失
            loss_fn = self.loss_fns[name]
            raw_loss = loss_fn(pred, target)

            # Uncertainty Weighting
            # loss = loss / (2 * sigma^2) + log(sigma)
            # = loss * exp(-2*log_var) + log_var
            log_var = self.log_vars[name]

            # 🔧 限制 log_var 范围,防止 exp 爆炸
            log_var = torch.clamp(log_var, min=-10, max=10)

            # 计算加权损失
            weighted_loss = raw_loss * torch.exp(-2 * log_var) + log_var

            # 🔧 检查加权损失是否有效
            if torch.isnan(weighted_loss) or torch.isinf(weighted_loss):
                # 如果无效,直接使用原始损失
                weighted_loss = raw_loss

            task_losses[name] = raw_loss.detach()
            weighted_losses.append(weighted_loss)

        # 总损失
        if weighted_losses:
            total_loss = torch.stack(weighted_losses).sum()
        else:
            total_loss = torch.tensor(0.0, device=self.device)

        return total_loss, task_losses

    def get_task_weights(self) -> Dict[str, float]:
        """
        获取当前任务权重 (用于监控和可视化)

        权重 = 1 / (2 * sigma^2) = exp(-2 * log_var)
        """
        weights = {}
        for name in self.task_configs:
            log_var = self.log_vars[name].item()
            weight = float(torch.exp(-2 * torch.tensor(log_var)))
            weights[name] = weight

        # 归一化
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def get_log_vars(self) -> Dict[str, float]:
        """获取log方差参数"""
        return {
            name: self.log_vars[name].item()
            for name in self.task_configs
        }


# =============================================================================
# 多任务分类头
# =============================================================================

class MultiTaskHead(nn.Module):
    """
    多任务分类头

    为每个任务创建独立的分类/回归头
    """

    def __init__(
            self,
            input_dim: int,
            task_configs: Dict[str, TaskConfig],
            hidden_dim: int = 256,
            dropout: float = 0.3
    ):
        """
        初始化

        Args:
            input_dim: 输入特征维度
            task_configs: 任务配置字典
            hidden_dim: 隐藏层维度
            dropout: Dropout率
        """
        super().__init__()

        self.task_configs = task_configs

        # 共享的特征变换层
        self.shared_mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 每个任务的专属头
        self.task_heads = nn.ModuleDict()

        for name, config in task_configs.items():
            if config.task_type == 'regression':
                # 回归任务
                self.task_heads[name] = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, 1)
                )
            else:
                # 分类任务
                self.task_heads[name] = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, config.num_classes)
                )

    def forward(
            self,
            features: torch.Tensor,
            task_names: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            features: (B, input_dim) 输入特征
            task_names: 要计算的任务列表 (None表示全部)

        Returns:
            outputs: {task_name: prediction}
        """
        # 共享变换
        shared_features = self.shared_mlp(features)

        # 各任务预测
        outputs = {}

        for name, head in self.task_heads.items():
            if task_names is not None and name not in task_names:
                continue

            outputs[name] = head(shared_features)

            # 回归任务输出squeeze
            if self.task_configs[name].task_type == 'regression':
                outputs[name] = outputs[name].squeeze(-1)

        return outputs


# =============================================================================
# 检查级特征聚合
# =============================================================================

class ExaminationAggregator(nn.Module):
    """
    检查级特征聚合器

    将同一检查的多个动作特征聚合为一个检查级特征
    """

    def __init__(
            self,
            input_dim: int,
            num_actions: int = 11,
            aggregation: str = 'attention',
            hidden_dim: int = 256
    ):
        """
        Args:
            input_dim: 每个动作的特征维度
            num_actions: 动作数量
            aggregation: 聚合方式 ('mean', 'max', 'attention')
            hidden_dim: 注意力隐藏维度
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_actions = num_actions
        self.aggregation = aggregation

        if aggregation == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )

    def forward(
            self,
            action_features: torch.Tensor,
            action_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            action_features: (B, num_actions, input_dim)
            action_mask: (B, num_actions) 有效动作掩码

        Returns:
            exam_features: (B, input_dim)
        """
        if self.aggregation == 'mean':
            if action_mask is not None:
                # 掩码平均
                mask = action_mask.unsqueeze(-1).float()
                summed = (action_features * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1)
                return summed / counts
            else:
                return action_features.mean(dim=1)

        elif self.aggregation == 'max':
            if action_mask is not None:
                action_features = action_features.masked_fill(
                    ~action_mask.unsqueeze(-1),
                    float('-inf')
                )
            return action_features.max(dim=1)[0]

        elif self.aggregation == 'attention':
            # 注意力加权
            scores = self.attention(action_features).squeeze(-1)  # (B, num_actions)

            if action_mask is not None:
                scores = scores.masked_fill(~action_mask, float('-inf'))

            weights = F.softmax(scores, dim=1).unsqueeze(-1)  # (B, num_actions, 1)
            return (action_features * weights).sum(dim=1)  # (B, input_dim)

        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")


# =============================================================================
# 工具函数
# =============================================================================

def create_multi_task_loss(
        task_names: List[str],
        device: str = 'cpu'
) -> MultiTaskLoss:
    """创建多任务损失函数"""
    return MultiTaskLoss(
        task_names=task_names,
        use_focal=True,
        label_smoothing=0.1,
        device=device
    )


def print_task_weights(loss_fn: MultiTaskLoss):
    """打印当前任务权重"""
    weights = loss_fn.get_task_weights()
    log_vars = loss_fn.get_log_vars()

    print("\n任务权重:")
    for name in sorted(weights.keys()):
        print(f"  {name:20} weight={weights[name]:.4f}  log_var={log_vars[name]:.4f}")


# =============================================================================
# 测试
# =============================================================================

if __name__ == '__main__':
    # 测试多任务损失
    print("测试 MultiTaskLoss...")

    loss_fn = MultiTaskLoss(
        task_names=['severity', 'hb_grading', 'sunnybrook'],
        use_focal=True,
        device='mps'
    )

    # 模拟数据
    batch_size = 8

    predictions = {
        'severity': torch.randn(batch_size, 5),
        'hb_grading': torch.randn(batch_size, 6),
        'sunnybrook': torch.randn(batch_size),
    }

    targets = {
        'severity': torch.randint(0, 5, (batch_size,)),
        'hb_grading': torch.randint(0, 6, (batch_size,)),
        'sunnybrook': torch.rand(batch_size) * 100,
    }

    # 计算损失
    total_loss, task_losses = loss_fn(predictions, targets)

    print(f"\n总损失: {total_loss.item():.4f}")
    print("各任务损失:")
    for name, loss in task_losses.items():
        print(f"  {name}: {loss.item():.4f}")

    print_task_weights(loss_fn)

    # 测试多任务头
    print("\n测试 MultiTaskHead...")

    task_configs = {
        'severity': TASK_CONFIGS['severity'],
        'hb_grading': TASK_CONFIGS['hb_grading'],
    }

    head = MultiTaskHead(
        input_dim=512,
        task_configs=task_configs
    )

    features = torch.randn(batch_size, 512)
    outputs = head(features)

    for name, output in outputs.items():
        print(f"  {name}: {output.shape}")

    print("\n✓ 测试通过!")