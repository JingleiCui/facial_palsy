#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H-GFA Net 层级多任务训练脚本
============================================

功能：
1. 端到端训练整个网络 (新架构: Proxy Task + Task Cascade)
2. 多任务学习: 动作级(severity) + 检查级(has_palsy, palsy_side, hb_grade, sunnybrook)
3. 不确定性加权损失自动平衡各任务
4. 支持3折交叉验证
5. Early stopping + 模型保存

修改说明:
- 适配新的HGFANet架构 (包含Wrinkle独立编码器)
- 支持action_severity作为proxy task
- Session aggregator使用severity_probs辅助
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import json

# 导入自定义模块
from hgfa_net import HGFANet, HierarchicalMultiTaskLoss
from dataset_palsy import HierarchicalPalsyDataset, collate_hierarchical

# ==================== 配置参数 ====================
# 训练哪一个fold:
#   >=0 : 只训练对应的 fold
#   <0  : 自动循环训练所有 fold (0,1,2)
FOLD = -1
N_FOLDS = 3

# 数据库路径
DB_PATH = "facialPalsy.db"

# 输出路径
CHECKPOINT_DIR = Path("checkpoints")

# 训练超参数
NUM_EPOCHS = 100
BATCH_SIZE = 4  # 降低batch size适应新架构
LEARNING_RATE = 5e-5  # 降低学习率,新架构更敏感
WEIGHT_DECAY = 0.01  # 降低weight decay
EARLY_STOPPING_PATIENCE = 30
GRAD_CLIP_NORM = 1.0

# 数据集版本
SPLIT_VERSION = "v1.0"


def get_device():
    """获取最佳可用设备"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def move_to_device(batch, device):
    """递归移动batch到设备"""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    return batch


def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch, grad_clip=1.0):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    # 记录各任务损失
    task_losses = {
        'action_severity': [],
        'has_palsy': [],
        'palsy_side': [],
        'hb_grade': [],
        'sunnybrook': []
    }

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(progress_bar):
        batch = move_to_device(batch, device)

        try:
            # 前向传播
            outputs = model(batch)

            # 计算损失
            loss, loss_dict = loss_fn(outputs, batch['targets'])

            # 检查NaN
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at batch {batch_idx}, skipping")
                continue

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            # 记录损失
            total_loss += loss.item()
            num_batches += 1

            for task in task_losses.keys():
                if task in loss_dict:
                    task_losses[task].append(loss_dict[task])

            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'severity': f'{loss_dict.get("action_severity", 0):.3f}',
                'hb': f'{loss_dict.get("hb_grade", 0):.3f}'
            })

        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 计算平均损失
    avg_losses = {
        'total': total_loss / max(num_batches, 1)
    }
    for task, losses in task_losses.items():
        if losses:
            avg_losses[task] = np.mean(losses)

    return avg_losses


def validate(model, dataloader, loss_fn, device):
    """验证"""
    model.eval()
    total_loss = 0
    num_batches = 0

    # 收集预测和标签
    all_preds = {
        'action_severity': [],  # 新增
        'has_palsy': [],
        'palsy_side': [],
        'hb_grade': [],
        'sunnybrook': []
    }
    all_labels = {
        'action_severity': [],  # 新增
        'has_palsy': [],
        'palsy_side': [],
        'hb_grade': [],
        'sunnybrook': []
    }

    # 记录各任务损失
    task_losses = {
        'action_severity': [],
        'has_palsy': [],
        'palsy_side': [],
        'hb_grade': [],
        'sunnybrook': []
    }

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            batch = move_to_device(batch, device)

            try:
                outputs = model(batch)
                loss, loss_dict = loss_fn(outputs, batch['targets'])

                if not torch.isnan(loss):
                    total_loss += loss.item()
                    num_batches += 1

                    for task in task_losses.keys():
                        if task in loss_dict:
                            task_losses[task].append(loss_dict[task])

                # 收集action-level severity预测
                severity_logits = outputs['action_severity']  # (B, 11, 5)
                severity_preds = severity_logits.argmax(dim=-1)  # (B, 11)
                severity_targets = batch['targets']['action_severity']  # (B, 11)

                all_preds['action_severity'].extend(severity_preds.cpu().numpy().flatten())
                all_labels['action_severity'].extend(severity_targets.cpu().numpy().flatten())

                # 收集session-level预测
                session = outputs['session_outputs']
                for task in ['has_palsy', 'palsy_side', 'hb_grade']:
                    preds = session[task].argmax(dim=1).cpu().numpy()
                    labels = batch['targets'][task].cpu().numpy()
                    all_preds[task].extend(preds)
                    all_labels[task].extend(labels)

                # Sunnybrook (回归任务)
                all_preds['sunnybrook'].extend(session['sunnybrook'].cpu().numpy())
                all_labels['sunnybrook'].extend(batch['targets']['sunnybrook'].cpu().numpy())

            except Exception as e:
                print(f"Error in validation batch: {e}")
                continue

    # 计算指标
    metrics = compute_metrics(all_preds, all_labels)
    metrics['loss'] = total_loss / max(num_batches, 1)

    # 添加各任务平均损失
    for task, losses in task_losses.items():
        if losses:
            metrics[f'{task}_loss'] = np.mean(losses)

    return metrics


def compute_metrics(preds, labels):
    """计算评估指标"""
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

    metrics = {}

    # Action-level severity (新增)
    if len(preds['action_severity']) > 0 and len(labels['action_severity']) > 0:
        # 过滤掉-1 (padding)
        valid_mask = np.array(labels['action_severity']) != -1
        if valid_mask.sum() > 0:
            valid_preds = np.array(preds['action_severity'])[valid_mask]
            valid_labels = np.array(labels['action_severity'])[valid_mask]

            metrics['action_severity_acc'] = accuracy_score(valid_labels, valid_preds)
            metrics['action_severity_f1'] = f1_score(
                valid_labels, valid_preds,
                average='weighted', zero_division=0
            )
        else:
            metrics['action_severity_acc'] = 0.0
            metrics['action_severity_f1'] = 0.0
    else:
        metrics['action_severity_acc'] = 0.0
        metrics['action_severity_f1'] = 0.0

    # Session-level分类任务
    for task in ['has_palsy', 'palsy_side', 'hb_grade']:
        if len(preds[task]) > 0 and len(labels[task]) > 0:
            metrics[f'{task}_acc'] = accuracy_score(labels[task], preds[task])
            metrics[f'{task}_f1'] = f1_score(
                labels[task], preds[task],
                average='weighted', zero_division=0
            )
        else:
            metrics[f'{task}_acc'] = 0.0
            metrics[f'{task}_f1'] = 0.0

    # Session-level回归任务
    if len(preds['sunnybrook']) > 0:
        metrics['sunnybrook_mae'] = mean_absolute_error(labels['sunnybrook'], preds['sunnybrook'])
    else:
        metrics['sunnybrook_mae'] = 100.0

    return metrics


def plot_training_curves(history, output_dir):
    """绘制训练曲线"""
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
    if history['val_loss']:
        axes[0, 0].plot(history['val_loss'], label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Action Severity Accuracy (新增)
    if history['val_action_severity_acc']:
        axes[0, 1].plot(history['val_action_severity_acc'], linewidth=2, color='purple')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Action Severity Accuracy (Proxy Task)')
        axes[0, 1].grid(True, alpha=0.3)

    # Has Palsy Accuracy
    if history['val_has_palsy_acc']:
        axes[0, 2].plot(history['val_has_palsy_acc'], linewidth=2, color='green')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Accuracy')
        axes[0, 2].set_title('Has Palsy Accuracy')
        axes[0, 2].grid(True, alpha=0.3)

    # Palsy Side Accuracy
    if history['val_palsy_side_acc']:
        axes[1, 0].plot(history['val_palsy_side_acc'], linewidth=2, color='blue')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('Palsy Side Accuracy')
        axes[1, 0].grid(True, alpha=0.3)

    # HB Grade Accuracy
    if history['val_hb_grade_acc']:
        axes[1, 1].plot(history['val_hb_grade_acc'], linewidth=2, color='red')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_title('HB Grade Accuracy')
        axes[1, 1].grid(True, alpha=0.3)

    # Sunnybrook MAE
    if history['val_sunnybrook_mae']:
        axes[1, 2].plot(history['val_sunnybrook_mae'], linewidth=2, color='orange')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('MAE')
        axes[1, 2].set_title('Sunnybrook MAE')
        axes[1, 2].grid(True, alpha=0.3)

    # Combined Metric
    if history['val_combined_metric']:
        axes[2, 0].plot(history['val_combined_metric'], linewidth=2, color='darkgreen')
        axes[2, 0].set_xlabel('Epoch')
        axes[2, 0].set_ylabel('Combined Metric')
        axes[2, 0].set_title('Combined Metric (Higher is Better)')
        best_idx = np.argmax(history['val_combined_metric'])
        best_val = history['val_combined_metric'][best_idx]
        axes[2, 0].axhline(y=best_val, color='r', linestyle='--',
                           label=f'Best: {best_val:.4f} @ Epoch {best_idx + 1}')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)

    # Task Loss Breakdown (新增)
    if history['val_action_severity_loss']:
        axes[2, 1].plot(history['val_action_severity_loss'], label='Severity', linewidth=2)
        axes[2, 1].plot(history['val_hb_grade_loss'], label='HB Grade', linewidth=2)
        axes[2, 1].plot(history['val_sunnybrook_loss'], label='Sunnybrook', linewidth=2)
        axes[2, 1].set_xlabel('Epoch')
        axes[2, 1].set_ylabel('Loss')
        axes[2, 1].set_title('Task Loss Breakdown')
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)

    # Hide unused subplot
    axes[2, 2].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ 训练曲线已保存: {output_dir / 'training_curves.png'}")


def run_single_fold(args):
    """真正执行一个 fold 的训练逻辑"""
    device = get_device()
    print(f"{'=' * 60}")
    print(f"H-GFA Net 层级多任务训练")
    print(f"{'=' * 60}")
    print(f"Device: {device}")
    print(f"Fold: {args.fold}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Weight Decay: {args.weight_decay}")

    # 创建数据集
    print(f"\n加载数据集...")
    train_dataset = HierarchicalPalsyDataset(
        args.db_path, args.fold, 'train', args.split_version,
        use_augmentation=True
    )
    val_dataset = HierarchicalPalsyDataset(
        args.db_path, args.fold, 'val', args.split_version,
        use_augmentation=False
    )

    print(f"Train: {len(train_dataset)} examinations")
    print(f"Val: {len(val_dataset)} examinations")

    if len(train_dataset) == 0:
        print("❌ 错误: 训练集为空！请检查数据集划分。")
        return

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_hierarchical, num_workers=0, drop_last=False
    )

    val_loader = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_hierarchical, num_workers=0
        )

    # 创建模型
    print(f"\n创建模型...")
    model = HGFANet().to(device)

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 损失函数
    loss_fn = HierarchicalMultiTaskLoss(use_uncertainty_weighting=True).to(device)
    print(f"✓ 使用不确定性加权多任务损失")

    # 优化器 (包含模型参数和损失函数的学习权重)
    optimizer = optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 创建输出目录
    checkpoint_dir = Path(args.checkpoint_dir) / f"fold_{args.fold}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_action_severity_acc': [],
        'val_action_severity_f1': [],
        'val_has_palsy_acc': [],
        'val_palsy_side_acc': [],
        'val_hb_grade_acc': [],
        'val_sunnybrook_mae': [],
        'val_combined_metric': [],
        # 任务损失
        'val_action_severity_loss': [],
        'val_hb_grade_loss': [],
        'val_sunnybrook_loss': [],
    }

    # 训练循环
    best_metric = 0
    patience = 0
    max_patience = EARLY_STOPPING_PATIENCE

    print(f"\n开始训练 Fold {args.fold} ...")
    print(f"{'=' * 60}")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # 训练
        train_losses = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch + 1)
        history['train_loss'].append(train_losses['total'])
        print(f"Train Loss: {train_losses['total']:.4f}")

        # 打印各任务训练损失
        if 'action_severity' in train_losses:
            print(f"  - Action Severity: {train_losses['action_severity']:.4f}")
        if 'hb_grade' in train_losses:
            print(f"  - HB Grade: {train_losses['hb_grade']:.4f}")

        # 验证
        if val_loader is not None:
            val_metrics = validate(model, val_loader, loss_fn, device)
            scheduler.step()

            # 记录历史
            history['val_loss'].append(val_metrics['loss'])
            history['val_action_severity_acc'].append(val_metrics['action_severity_acc'])  # 新增
            history['val_action_severity_f1'].append(val_metrics['action_severity_f1'])  # 新增
            history['val_has_palsy_acc'].append(val_metrics['has_palsy_acc'])
            history['val_palsy_side_acc'].append(val_metrics['palsy_side_acc'])
            history['val_hb_grade_acc'].append(val_metrics['hb_grade_acc'])
            history['val_sunnybrook_mae'].append(val_metrics['sunnybrook_mae'])

            # 记录任务损失
            history['val_action_severity_loss'].append(val_metrics.get('action_severity_loss', 0))
            history['val_hb_grade_loss'].append(val_metrics.get('hb_grade_loss', 0))
            history['val_sunnybrook_loss'].append(val_metrics.get('sunnybrook_loss', 0))

            # 综合指标 (包含action severity)
            combined_metric = (
                    val_metrics['action_severity_acc'] * 0.15 +  # 新增proxy task
                    val_metrics['has_palsy_acc'] * 0.15 +
                    val_metrics['palsy_side_acc'] * 0.15 +
                    val_metrics['hb_grade_acc'] * 0.40 +  # 最重要
                    max(0, 1 - val_metrics['sunnybrook_mae'] / 100) * 0.15
            )
            history['val_combined_metric'].append(combined_metric)

            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(
                f"  action_severity: acc={val_metrics['action_severity_acc']:.4f}, F1={val_metrics['action_severity_f1']:.4f}")
            print(f"  has_palsy:       acc={val_metrics['has_palsy_acc']:.4f}, F1={val_metrics['has_palsy_f1']:.4f}")
            print(f"  palsy_side:      acc={val_metrics['palsy_side_acc']:.4f}, F1={val_metrics['palsy_side_f1']:.4f}")
            print(f"  hb_grade:        acc={val_metrics['hb_grade_acc']:.4f}, F1={val_metrics['hb_grade_f1']:.4f}")
            print(f"  sunnybrook:      MAE={val_metrics['sunnybrook_mae']:.2f}")
            print(f"  Combined:        {combined_metric:.4f}")

            # 打印学到的任务权重
            if hasattr(loss_fn, 'log_vars'):
                uncertainties = torch.exp(loss_fn.log_vars).detach().cpu().numpy()
                weights = 1.0 / uncertainties
                print(f"  Task Weights: severity={weights[0]:.3f}, has_palsy={weights[1]:.3f}, "
                      f"side={weights[2]:.3f}, hb={weights[3]:.3f}, sunnybrook={weights[4]:.3f}")

            # 保存最佳模型
            if combined_metric > best_metric:
                best_metric = combined_metric
                patience = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss_fn_state_dict': loss_fn.state_dict(),
                    'metrics': val_metrics,
                    'combined_metric': combined_metric,
                    'config': model.config,
                }, checkpoint_dir / 'best.pth')
                print(f"  ✓ Saved best model (metric: {best_metric:.4f})")
            else:
                patience += 1
                if patience >= max_patience:
                    print(f"\nEarly stopping at epoch {epoch + 1} (fold {args.fold})")
                    break
        else:
            # 没有验证集，每10个epoch保存一次
            scheduler.step()
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'train_loss': train_losses['total'],
                }, checkpoint_dir / f'epoch_{epoch + 1}.pth')

    # 保存最终模型
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_fn_state_dict': loss_fn.state_dict(),
        'history': history,
    }, checkpoint_dir / 'final.pth')

    # 绘制训练曲线
    if val_loader is not None:
        plot_training_curves(history, checkpoint_dir)

    # 保存训练配置
    config = {
        'fold': args.fold,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'epochs': epoch + 1,
        'best_metric': best_metric,
        'model_params': {
            'total': total_params,
            'trainable': trainable_params
        },
        'architecture': 'HGFANet',
        'improvements': [
            'Wrinkle独立编码',
            'Action-level severity作为proxy task',
            'Task cascade: severity辅助session-level诊断',
            '不确定性加权多任务学习'
        ],
        'timestamp': datetime.now().isoformat(),
    }
    with open(checkpoint_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Fold {args.fold} 训练完成!")
    if val_loader is not None:
        print(f"Best Combined Metric: {best_metric:.4f}")
    print(f"模型保存到: {checkpoint_dir}")
    print(f"{'=' * 60}")


def main():
    """支持单 fold / 多 fold 训练的入口"""

    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', type=str, default=DB_PATH)
    parser.add_argument('--fold', type=int, default=FOLD)
    parser.add_argument('--split_version', type=str, default=SPLIT_VERSION)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY)
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--checkpoint_dir', type=str, default=str(CHECKPOINT_DIR))
    args = parser.parse_args()

    if args.fold < 0:
        # 自动跑完所有 fold
        for fold in range(N_FOLDS):
            print(f"\n================== 开始训练 Fold {fold} ==================\n")
            args_single = argparse.Namespace(**vars(args))
            args_single.fold = fold
            run_single_fold(args_single)
    else:
        # 只训练指定 fold
        run_single_fold(args)


if __name__ == '__main__':
    main()