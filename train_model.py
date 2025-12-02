"""
H-GFA Net 训练脚本 (修复版)
==============================

修复内容:
1. 添加特征 NaN/Inf 检查和清理
2. 修复 classification_report 的 labels 参数
3. 添加梯度异常检测
4. 改进损失计算的数值稳定性

用法:
    python train_model.py
"""

import os
import sys
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt

# 本地模块
from hgfa_net import HGFANet, create_hgfa_net
from multi_task_loss import MultiTaskLoss, print_task_weights
from data_split import (
    DatasetSplitter,
    load_samples_from_db,
    print_dataset_stats
)


# =============================================================================
# 配置
# =============================================================================

@dataclass
class TrainConfig:
    """训练配置"""
    db_path: str = 'facialPalsy.db'
    batch_size: int = 16
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    lr_scheduler: str = 'plateau'
    lr_patience: int = 10
    lr_factor: float = 0.5
    min_lr: float = 1e-6
    early_stopping_patience: int = 20
    task_names: List[str] = None
    augment_prob: float = 0.5
    save_dir: str = 'checkpoints'
    experiment_name: str = None
    device: str = 'auto'

    def __post_init__(self):
        if self.task_names is None:
            self.task_names = ['severity']

        if self.experiment_name is None:
            self.experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')

        if self.device == 'auto':
            if torch.backends.mps.is_available():
                self.device = 'mps'
            elif torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'


# =============================================================================
# 特征清理工具
# =============================================================================

def clean_features(features: np.ndarray, name: str = "features") -> np.ndarray:
    """
    清理特征中的 NaN 和 Inf 值

    Args:
        features: numpy数组
        name: 特征名称（用于日志）

    Returns:
        清理后的特征
    """
    if features is None:
        return None

    # 检查并替换 NaN
    nan_count = np.isnan(features).sum()
    if nan_count > 0:
        print(f"  [WARN] {name} 包含 {nan_count} 个 NaN，替换为0")
        features = np.nan_to_num(features, nan=0.0)

    # 检查并替换 Inf
    inf_count = np.isinf(features).sum()
    if inf_count > 0:
        print(f"  [WARN] {name} 包含 {inf_count} 个 Inf，替换为0")
        features = np.nan_to_num(features, posinf=0.0, neginf=0.0)

    return features


def validate_sample(sample: Dict[str, Any]) -> bool:
    """
    验证样本是否有效

    Returns:
        True 如果样本有效
    """
    # 检查必要字段
    if sample.get('static_features') is None:
        return False

    if sample.get('visual_features') is None:
        return False

    # 检查标签
    severity = sample.get('severity')
    if severity is None or severity < 1 or severity > 5:
        return False

    return True


# =============================================================================
# 数据集
# =============================================================================

class FacialPalsyDataset(Dataset):
    """面瘫评估数据集（带特征清理）"""

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        is_training: bool = True
    ):
        # 过滤无效样本
        valid_samples = []
        invalid_count = 0

        for sample in samples:
            if validate_sample(sample):
                # 清理特征
                sample['static_features'] = clean_features(
                    sample.get('static_features'),
                    f"static_{sample.get('video_id')}"
                )
                sample['dynamic_features'] = clean_features(
                    sample.get('dynamic_features'),
                    f"dynamic_{sample.get('video_id')}"
                )
                sample['visual_features'] = clean_features(
                    sample.get('visual_features'),
                    f"visual_{sample.get('video_id')}"
                )
                sample['wrinkle_features'] = clean_features(
                    sample.get('wrinkle_features'),
                    f"wrinkle_{sample.get('video_id')}"
                )
                sample['motion_features'] = clean_features(
                    sample.get('motion_features'),
                    f"motion_{sample.get('video_id')}"
                )
                valid_samples.append(sample)
            else:
                invalid_count += 1

        if invalid_count > 0:
            print(f"[Dataset] 过滤了 {invalid_count} 个无效样本")

        self.samples = valid_samples
        self.is_training = is_training
        self.action_groups = self._group_by_action()

        print(f"[Dataset] 有效样本数: {len(self.samples)}")
        print(f"[Dataset] 动作分布: {dict(sorted((k, len(v)) for k, v in self.action_groups.items()))}")

        # 打印标签分布
        self._print_label_distribution()

    def _print_label_distribution(self):
        """打印标签分布"""
        severity_counts = defaultdict(int)
        for sample in self.samples:
            sev = sample.get('severity', -1)
            severity_counts[sev] += 1

        print(f"[Dataset] Severity分布: {dict(sorted(severity_counts.items()))}")

    def _group_by_action(self) -> Dict[str, List[int]]:
        groups = defaultdict(list)
        for idx, sample in enumerate(self.samples):
            action = sample.get('action_name', 'Unknown')
            groups[action].append(idx)
        return groups

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx].copy()

        result = {
            'video_id': sample['video_id'],
            'action_name': sample['action_name'],
        }

        # 几何特征
        if sample.get('static_features') is not None:
            result['static'] = torch.from_numpy(
                sample['static_features'].astype(np.float32)
            ).contiguous()
        else:
            result['static'] = None

        if sample.get('dynamic_features') is not None and len(sample['dynamic_features']) > 0:
            result['dynamic'] = torch.from_numpy(
                sample['dynamic_features'].astype(np.float32)
            ).contiguous()
        else:
            result['dynamic'] = None

        # 视觉特征
        if sample.get('visual_features') is not None:
            result['visual'] = torch.from_numpy(
                sample['visual_features'].astype(np.float32)
            ).contiguous()
        else:
            result['visual'] = torch.zeros(1280)

        # 皱纹特征
        if sample.get('wrinkle_features') is not None:
            result['wrinkle'] = torch.from_numpy(
                sample['wrinkle_features'].astype(np.float32)
            ).contiguous()
        else:
            result['wrinkle'] = torch.zeros(10)

        # 运动特征
        if sample.get('motion_features') is not None:
            result['motion'] = torch.from_numpy(
                sample['motion_features'].astype(np.float32)
            ).contiguous()
        else:
            result['motion'] = torch.zeros(12)

        # 标签 (severity 1-5 转为 0-4)
        severity = sample.get('severity')
        if severity is not None and 1 <= severity <= 5:
            result['severity_label'] = int(severity) - 1
        else:
            result['severity_label'] = -1

        hb = sample.get('hb_grade')
        if hb is not None:
            result['hb_label'] = int(hb) - 1
        else:
            result['hb_label'] = -1

        sunnybrook = sample.get('sunnybrook')
        if sunnybrook is not None:
            result['sunnybrook_label'] = float(sunnybrook)
        else:
            result['sunnybrook_label'] = -1.0

        return result


def collate_fn(batch: List[Dict]) -> Dict[str, Dict[str, torch.Tensor]]:
    """按动作分组的collate函数"""
    action_groups = defaultdict(list)
    for item in batch:
        action = item['action_name']
        action_groups[action].append(item)

    result = {}

    for action, items in action_groups.items():
        if not items:
            continue

        # 找最大维度
        static_dims = [item['static'].shape[0] for item in items if item['static'] is not None]
        max_static_dim = max(static_dims) if static_dims else 0

        dynamic_dims = [
            item['dynamic'].shape[0]
            for item in items
            if item['dynamic'] is not None
        ]
        max_dynamic_dim = max(dynamic_dims) if dynamic_dims else 0

        # Padding
        padded_static = []
        padded_dynamic = []
        visual_list = []
        wrinkle_list = []
        motion_list = []
        severity_labels = []
        hb_labels = []
        sunnybrook_labels = []

        for item in items:
            # 静态特征
            static = item['static']
            if static is not None:
                if static.shape[0] < max_static_dim:
                    pad = torch.zeros(max_static_dim - static.shape[0])
                    static = torch.cat([static, pad])
                padded_static.append(static)

            # 动态特征
            if max_dynamic_dim > 0:
                dyn = item['dynamic']
                if dyn is None or dyn.numel() == 0:
                    dyn = torch.zeros(max_dynamic_dim)
                elif dyn.shape[0] < max_dynamic_dim:
                    pad = torch.zeros(max_dynamic_dim - dyn.shape[0])
                    dyn = torch.cat([dyn, pad])
                padded_dynamic.append(dyn)

            visual_list.append(item['visual'])
            wrinkle_list.append(item['wrinkle'])
            motion_list.append(item['motion'])

            severity_labels.append(item['severity_label'])
            hb_labels.append(item['hb_label'])
            sunnybrook_labels.append(item['sunnybrook_label'])

        result[action] = {
            'static': torch.stack(padded_static) if padded_static else None,
            'dynamic': torch.stack(padded_dynamic) if padded_dynamic else None,
            'visual': torch.stack(visual_list),
            'wrinkle': torch.stack(wrinkle_list),
            'motion': torch.stack(motion_list),
            'labels': torch.tensor(severity_labels, dtype=torch.long),
            'hb_labels': torch.tensor(hb_labels, dtype=torch.long),
            'sunnybrook_labels': torch.tensor(sunnybrook_labels, dtype=torch.float32),
            'static_dim': max_static_dim,
            'dynamic_dim': max_dynamic_dim,
        }

    return result


# =============================================================================
# 训练器
# =============================================================================

class Trainer:
    """H-GFA Net 训练器（修复版）"""

    def __init__(
        self,
        model: HGFANet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainConfig
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(config.device)

        # 多任务损失
        self.loss_fn = MultiTaskLoss(
            task_names=config.task_names,
            use_focal=True,
            device=config.device
        )

        # 优化器
        self.optimizer = optim.AdamW(
            list(model.parameters()) + list(self.loss_fn.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # 学习率调度器
        if config.lr_scheduler == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=config.lr_factor,
                patience=config.lr_patience,
                min_lr=config.min_lr
            )
        elif config.lr_scheduler == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.num_epochs,
                eta_min=config.min_lr
            )
        else:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.5
            )

        # 保存目录
        self.save_dir = Path(config.save_dir) / config.experiment_name
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 日志
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.patience_counter = 0

        # 保存配置
        with open(self.save_dir / 'config.json', 'w') as f:
            json.dump(asdict(config), f, indent=2)

    def _check_tensor(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        """检查并清理张量中的 NaN/Inf"""
        if tensor is None:
            return None

        if torch.isnan(tensor).any():
            print(f"  [WARN] {name} 包含 NaN，替换为0")
            tensor = torch.nan_to_num(tensor, nan=0.0)

        if torch.isinf(tensor).any():
            print(f"  [WARN] {name} 包含 Inf，替换为0")
            tensor = torch.nan_to_num(tensor, posinf=0.0, neginf=0.0)

        return tensor

    def train_epoch(self, epoch: int) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        nan_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            # 前向传播
            output = self.model(batch)

            # 准备预测和目标
            predictions = {}
            targets = {}
            valid_masks = {}

            if 'severity' in self.config.task_names and output.get('action_logits') is not None:
                logits = output['action_logits']
                labels = output['action_labels']

                # 检查 NaN
                logits = self._check_tensor(logits, f"logits_batch{batch_idx}")

                predictions['severity'] = logits
                targets['severity'] = labels
                valid_masks['severity'] = labels >= 0

            # 计算损失
            loss, task_losses = self.loss_fn(predictions, targets, valid_masks)

            # 检查损失是否为 NaN
            if torch.isnan(loss) or torch.isinf(loss):
                nan_batches += 1
                print(f"  [WARN] Batch {batch_idx}: Loss is NaN/Inf, skipping")
                continue

            # 反向传播
            loss.backward()

            # 检查梯度
            total_grad_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        print(f"  [WARN] Batch {batch_idx}: Gradient contains NaN/Inf")
                        p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                    total_grad_norm += p.grad.data.norm(2).item() ** 2
            total_grad_norm = total_grad_norm ** 0.5

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}/{len(self.train_loader)}, "
                      f"Loss: {loss.item():.4f}, GradNorm: {total_grad_norm:.2f}")

        if nan_batches > 0:
            print(f"  [WARN] 本Epoch有 {nan_batches} 个batch的loss为NaN")

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    @torch.no_grad()
    def validate(self, epoch: int) -> Tuple[float, float, List, List]:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_preds = []
        all_labels = []

        for batch in self.val_loader:
            output = self.model(batch)

            predictions = {}
            targets = {}
            valid_masks = {}

            if 'severity' in self.config.task_names and output.get('action_logits') is not None:
                predictions['severity'] = output['action_logits']
                targets['severity'] = output['action_labels']
                valid_masks['severity'] = targets['severity'] >= 0

                mask = valid_masks['severity']
                if mask.sum() > 0:
                    preds = output['action_logits'][mask].argmax(dim=1).cpu().numpy()
                    labels = output['action_labels'][mask].cpu().numpy()
                    all_preds.extend(preds.tolist())
                    all_labels.extend(labels.tolist())

            loss, _ = self.loss_fn(predictions, targets, valid_masks)

            # 跳过 NaN loss
            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        if all_preds:
            accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_preds)
        else:
            accuracy = 0.0

        self.val_losses.append(avg_loss)
        self.val_accs.append(accuracy)

        return avg_loss, accuracy, all_preds, all_labels

    def save_checkpoint(self, epoch: int, filename: str):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss_fn_state_dict': self.loss_fn.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'config': asdict(self.config),
        }

        torch.save(checkpoint, self.save_dir / filename)
        print(f"[Checkpoint] 保存: {self.save_dir / filename}")

    def plot_curves(self):
        """绘制训练曲线"""
        if not self.train_losses:
            return

        # 过滤掉 NaN
        valid_epochs = []
        valid_train_losses = []
        valid_val_losses = []
        valid_val_accs = []

        for i, (tl, vl, va) in enumerate(zip(self.train_losses, self.val_losses, self.val_accs)):
            if not (np.isnan(tl) or np.isnan(vl)):
                valid_epochs.append(i + 1)
                valid_train_losses.append(tl)
                valid_val_losses.append(vl)
                valid_val_accs.append(va)

        if not valid_epochs:
            print("[Plot] 没有有效数据可绘制")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        axes[0].plot(valid_epochs, valid_train_losses, 'b-', label='Train')
        axes[0].plot(valid_epochs, valid_val_losses, 'r-', label='Val')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss Curve')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(valid_epochs, valid_val_accs, 'g-', label='Val Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=150)
        plt.close()

        print(f"[Plot] 保存: {self.save_dir / 'training_curves.png'}")

    def train(self):
        """完整训练流程"""
        print("\n" + "=" * 60)
        print("开始训练 H-GFA Net")
        print(f"设备: {self.device}")
        print(f"实验: {self.config.experiment_name}")
        print("=" * 60)

        for epoch in range(1, self.config.num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{self.config.num_epochs}")
            print(f"{'='*60}")

            # 训练
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)

            # 验证
            val_loss, val_acc, preds, labels = self.validate(epoch)

            # 学习率调度
            if self.config.lr_scheduler == 'plateau':
                if not np.isnan(val_loss):
                    self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            # 打印结果
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"\n[Epoch {epoch}] 结果:")
            print(f"  训练损失: {train_loss:.4f}")
            print(f"  验证损失: {val_loss:.4f}")
            print(f"  验证准确率: {val_acc:.4f}")
            print(f"  学习率: {current_lr:.2e}")

            # 任务权重
            if epoch % 5 == 0:
                print_task_weights(self.loss_fn)

            # 分类报告 - 修复：动态确定类别数
            if epoch % 10 == 0 and labels:
                print("\n分类报告:")

                # 获取实际出现的类别
                unique_labels = sorted(set(labels) | set(preds))
                target_names = [f"Grade {i+1}" for i in unique_labels]

                try:
                    print(classification_report(
                        labels, preds,
                        labels=unique_labels,
                        target_names=target_names,
                        zero_division=0
                    ))
                except Exception as e:
                    print(f"  [WARN] 无法生成分类报告: {e}")
                    print(f"  实际标签: {set(labels)}, 预测标签: {set(preds)}")

            # 保存最佳模型 (跳过 NaN)
            if not np.isnan(val_loss) and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.patience_counter = 0
                self.save_checkpoint(epoch, 'best_model.pth')
            else:
                self.patience_counter += 1

            # 定期保存
            if epoch % 20 == 0:
                self.save_checkpoint(epoch, f'checkpoint_epoch_{epoch}.pth')

            # 早停
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"\n[早停] {self.config.early_stopping_patience} epochs无改善")
                break

            # MPS内存清理
            if self.device.type == 'mps':
                torch.mps.empty_cache()

        print("\n" + "=" * 60)
        print("✓ 训练完成!")
        print(f"最佳验证损失: {self.best_val_loss:.4f}")
        print(f"最佳验证准确率: {self.best_val_acc:.4f}")
        print(f"模型保存: {self.save_dir}")
        print("=" * 60)

        self.plot_curves()
        self.save_checkpoint(epoch, 'final_model.pth')


# =============================================================================
# 诊断工具
# =============================================================================

def diagnose_features(db_path: str):
    """诊断特征数据质量"""
    print("\n" + "=" * 60)
    print("特征数据诊断")
    print("=" * 60)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 检查各类特征的统计
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN static_features IS NOT NULL THEN 1 ELSE 0 END) as has_static,
            SUM(CASE WHEN visual_features IS NOT NULL THEN 1 ELSE 0 END) as has_visual,
            SUM(CASE WHEN wrinkle_features IS NOT NULL THEN 1 ELSE 0 END) as has_wrinkle,
            SUM(CASE WHEN motion_features IS NOT NULL THEN 1 ELSE 0 END) as has_motion
        FROM video_features
    """)

    row = cursor.fetchone()
    print(f"\n特征完整性:")
    print(f"  总记录数: {row[0]}")
    print(f"  static_features: {row[1]}/{row[0]}")
    print(f"  visual_features: {row[2]}/{row[0]}")
    print(f"  wrinkle_features: {row[3]}/{row[0]}")
    print(f"  motion_features: {row[4]}/{row[0]}")

    # 检查标签分布
    cursor.execute("""
        SELECT severity_score, COUNT(*) 
        FROM action_labels 
        GROUP BY severity_score 
        ORDER BY severity_score
    """)

    print(f"\nSeverity标签分布:")
    for row in cursor.fetchall():
        print(f"  Grade {row[0]}: {row[1]} 条")

    conn.close()


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    config = TrainConfig(
        db_path='facialPalsy.db',
        batch_size=16,
        num_epochs=100,
        learning_rate=1e-4,
        task_names=['severity'],
        augment_prob=0.5,
        early_stopping_patience=25,
    )

    print("=" * 60)
    print("H-GFA Net 训练")
    print("=" * 60)
    print(f"数据库: {config.db_path}")
    print(f"设备: {config.device}")
    print(f"批大小: {config.batch_size}")
    print(f"学习率: {config.learning_rate}")
    print(f"任务: {config.task_names}")
    print("=" * 60)

    # 诊断特征
    diagnose_features(config.db_path)

    # 检查数据划分
    print("\n[1/5] 检查数据划分...")
    print_dataset_stats(config.db_path)

    # 加载数据
    print("\n[2/5] 加载数据...")
    train_samples = load_samples_from_db(config.db_path, 'train')
    val_samples = load_samples_from_db(config.db_path, 'val')

    if not train_samples:
        print("[!] 训练集为空，请先执行数据划分:")
        print("    python data_split.py split")
        return

    print(f"训练样本: {len(train_samples)}")
    print(f"验证样本: {len(val_samples)}")

    # 创建数据集
    print("\n[3/5] 创建数据加载器...")

    train_dataset = FacialPalsyDataset(
        train_samples,
        is_training=True
    )

    val_dataset = FacialPalsyDataset(
        val_samples,
        is_training=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # 创建模型
    print("\n[4/5] 创建模型...")
    model = create_hgfa_net(
        device=config.device,
        task_names=config.task_names
    )

    # 训练
    print("\n[5/5] 开始训练...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config
    )

    trainer.train()

    print("\n✓ 完成!")


if __name__ == '__main__':
    main()