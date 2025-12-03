#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多任务不确定性加权损失

参考论文:
- Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh Losses"

损失函数:
L_total = Σ (1/(2*σ_i^2)) * L_i + log(σ_i)

其中 σ_i 是每个任务的可学习不确定性参数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple


class UncertaintyWeightedLoss(nn.Module):
    """
    不确定性加权的多任务损失

    自动学习每个任务的权重，基于任务的不确定性
    """

    def __init__(self, num_tasks: int, task_names: List[str] = None):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_names = task_names or [f'task_{i}' for i in range(num_tasks)]

        # 可学习的 log(σ^2) 参数 (初始化为0表示σ=1)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            losses: {task_name: loss_value}
        Returns:
            total_loss, loss_info
        """
        total_loss = torch.tensor(0.0, device=self.log_vars.device)
        weighted_losses = {}
        weights = {}

        for i, task_name in enumerate(self.task_names):
            if task_name not in losses or losses[task_name] is None:
                continue

            loss = losses[task_name]
            if not isinstance(loss, torch.Tensor) or loss.numel() == 0:
                continue

            log_var = self.log_vars[i]

            # precision = 1/σ^2 = exp(-log_var)
            precision = torch.exp(-log_var)

            # weighted_loss = 0.5 * precision * loss + 0.5 * log_var
            weighted_loss = 0.5 * precision * loss + 0.5 * log_var

            total_loss = total_loss + weighted_loss
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
                'severity_label_smoothing': 0.1,
                'classification_label_smoothing': 0.05,
            }

        # 定义任务
        self.task_names = [
            'action_severity',  # 动作级
            'has_palsy',        # 检查级
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
            label_smoothing=config.get('severity_label_smoothing', 0.1)
        )
        self.has_palsy_ce = nn.CrossEntropyLoss(
            label_smoothing=config.get('classification_label_smoothing', 0.05)
        )
        self.palsy_side_ce = nn.CrossEntropyLoss(
            label_smoothing=config.get('classification_label_smoothing', 0.05)
        )
        self.hb_grade_ce = nn.CrossEntropyLoss(
            label_smoothing=config.get('severity_label_smoothing', 0.1)
        )
        self.sunnybrook_loss = nn.SmoothL1Loss()

    def forward(self, outputs: Dict, targets: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            outputs: 模型输出
            targets: 标签
        Returns:
            total_loss, loss_info
        """
        losses = {}
        device = targets['has_palsy'].device

        # 1. 动作级严重程度损失
        action_severity_losses = []
        for action_name, severity_logits in outputs['action_severity'].items():
            if action_name in targets['action_severity']:
                target = targets['action_severity'][action_name]
                if len(target) == len(severity_logits) and len(target) > 0:
                    # 确保target在正确的设备上
                    target = target.to(device)
                    loss = self.severity_ce(severity_logits, target)
                    if not torch.isnan(loss):
                        action_severity_losses.append(loss)

        if action_severity_losses:
            losses['action_severity'] = torch.stack(action_severity_losses).mean()
        else:
            losses['action_severity'] = torch.tensor(0.0, device=device, requires_grad=True)

        # 2. 检查级任务损失
        session_out = outputs['session_outputs']

        # 是否面瘫
        losses['has_palsy'] = self.has_palsy_ce(
            session_out['has_palsy'],
            targets['has_palsy'].to(device)
        )

        # 面瘫侧别
        losses['palsy_side'] = self.palsy_side_ce(
            session_out['palsy_side'],
            targets['palsy_side'].to(device)
        )

        # HB分级
        losses['hb_grade'] = self.hb_grade_ce(
            session_out['hb_grade'],
            targets['hb_grade'].to(device)
        )

        # Sunnybrook (归一化到[0,1])
        sunnybrook_pred = session_out['sunnybrook']
        sunnybrook_target = targets['sunnybrook'].to(device) / 100.0
        losses['sunnybrook'] = self.sunnybrook_loss(sunnybrook_pred, sunnybrook_target)

        # 3. 检查NaN
        for name, loss in losses.items():
            if torch.isnan(loss):
                print(f"Warning: NaN loss in {name}")
                losses[name] = torch.tensor(0.0, device=device, requires_grad=True)

        # 4. 不确定性加权合并
        total_loss, loss_info = self.uncertainty_loss(losses)

        # 添加原始损失信息
        loss_info['raw_losses'] = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}

        return total_loss, loss_info