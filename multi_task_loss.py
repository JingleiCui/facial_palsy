"""
多任务不确定性加权损失
参考: Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyWeightedLoss(nn.Module):
    """
    不确定性加权的多任务损失

    L_total = Σ (1/(2*σ_i^2)) * L_i + log(σ_i)

    其中 σ_i 是每个任务的可学习不确定性参数
    """

    def __init__(self, num_tasks: int, task_names: list = None):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_names = task_names or [f'task_{i}' for i in range(num_tasks)]

        # 可学习的 log(σ^2) 参数 (初始化为0表示σ=1)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses: dict) -> tuple:
        """
        Args:
            losses: {task_name: loss_value}
        Returns:
            total_loss, loss_dict (包含权重信息)
        """
        total_loss = 0
        weighted_losses = {}
        weights = {}

        for i, task_name in enumerate(self.task_names):
            if task_name not in losses:
                continue

            loss = losses[task_name]
            log_var = self.log_vars[i]

            # precision = 1/σ^2 = exp(-log_var)
            precision = torch.exp(-log_var)

            # weighted_loss = 0.5 * precision * loss + 0.5 * log_var
            weighted_loss = 0.5 * precision * loss + 0.5 * log_var

            total_loss += weighted_loss
            weighted_losses[task_name] = weighted_loss.item()
            weights[task_name] = precision.item()

        return total_loss, {
            'total': total_loss.item(),
            'weighted_losses': weighted_losses,
            'weights': weights,
            'uncertainties': {
                name: torch.exp(self.log_vars[i]).item()
                for i, name in enumerate(self.task_names)
            }
        }


class HierarchicalMultiTaskLoss(nn.Module):
    """
    层级多任务损失

    包含:
    - 动作级任务: severity分类 (11个动作)
    - 检查级任务: has_palsy, palsy_side, hb_grade, sunnybrook
    """

    def __init__(self, config=None):
        super().__init__()

        if config is None:
            config = {
                'action_severity_weight': 1.0,
                'has_palsy_weight': 1.0,
                'palsy_side_weight': 1.0,
                'hb_grade_weight': 1.0,
                'sunnybrook_weight': 0.5,
            }

        # 定义任务
        self.task_names = [
            'action_severity',  # 动作级
            'has_palsy',  # 检查级
            'palsy_side',
            'hb_grade',
            'sunnybrook'
        ]

        # 不确定性加权
        self.uncertainty_loss = UncertaintyWeightedLoss(
            num_tasks=len(self.task_names),
            task_names=self.task_names
        )

        # 各任务的损失函数
        self.severity_ce = nn.CrossEntropyLoss(
            label_smoothing=0.1,
            reduction='none'
        )
        self.has_palsy_ce = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.palsy_side_ce = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.hb_grade_ce = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.sunnybrook_smooth_l1 = nn.SmoothL1Loss()

    def forward(self, outputs: dict, targets: dict) -> tuple:
        """
        Args:
            outputs: 模型输出
            targets: {
                'action_severity': {action_name: (B,) labels},
                'has_palsy': (B,),
                'palsy_side': (B,),
                'hb_grade': (B,),
                'sunnybrook': (B,),
                'action_mask': (B, 11)
            }
        """
        losses = {}

        # 1. 动作级严重程度损失 (对所有有效动作平均)
        action_severity_losses = []
        for action_name, severity_logits in outputs['action_severity'].items():
            if action_name in targets['action_severity']:
                target = targets['action_severity'][action_name]
                loss = self.severity_ce(severity_logits, target)
                action_severity_losses.append(loss.mean())

        if action_severity_losses:
            losses['action_severity'] = torch.stack(action_severity_losses).mean()

        # 2. 检查级任务损失
        session_out = outputs['session_outputs']

        losses['has_palsy'] = self.has_palsy_ce(
            session_out['has_palsy'],
            targets['has_palsy']
        )

        losses['palsy_side'] = self.palsy_side_ce(
            session_out['palsy_side'],
            targets['palsy_side']
        )

        losses['hb_grade'] = self.hb_grade_ce(
            session_out['hb_grade'],
            targets['hb_grade']
        )

        # Sunnybrook是回归任务，归一化到[0,1]
        losses['sunnybrook'] = self.sunnybrook_smooth_l1(
            session_out['sunnybrook'],
            targets['sunnybrook'] / 100.0  # 归一化
        )

        # 3. 不确定性加权合并
        total_loss, loss_info = self.uncertainty_loss(losses)

        return total_loss, loss_info