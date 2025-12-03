#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H-GFA Net 层级多任务训练脚本

功能：
1. 端到端训练整个网络 (CDCAF -> GQCA -> MFA -> Aggregator)
2. 多任务学习: 动作级(severity) + 检查级(has_palsy, palsy_side, hb_grade, sunnybrook)
3. 不确定性加权损失自动平衡各任务
4. 3折交叉验证
5. Early stopping + 模型保存

直接在PyCharm中点击运行即可！
修改下面的FOLD变量来训练不同的fold (0, 1, 2)
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
from hgfa_net import HGFANet
from dataset_palsy import HierarchicalPalsyDataset, collate_hierarchical
from multi_task_loss import HierarchicalMultiTaskLoss

# ==================== 配置参数 ====================
# 训练哪一个fold (0, 1, 或 2)
FOLD = 1  # <--- 修改这里来训练不同的fold

# 数据库路径
DB_PATH = "facialPalsy.db"

# 输出路径
CHECKPOINT_DIR = Path("checkpoints")

# 训练超参数
NUM_EPOCHS = 100
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.05
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


def train_epoch(model, dataloader, optimizer, loss_fn, device, grad_clip=1.0):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="Training")
    for batch in progress_bar:
        batch = move_to_device(batch, device)

        try:
            # 前向传播
            outputs = model(batch)

            # 计算损失
            loss, loss_info = loss_fn(outputs, batch['targets'])

            # 检查NaN
            if torch.isnan(loss):
                print("Warning: NaN loss detected, skipping batch")
                continue

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        except Exception as e:
            print(f"Error in batch: {e}")
            continue

    return total_loss / max(num_batches, 1)


def validate(model, dataloader, loss_fn, device):
    """验证"""
    model.eval()
    total_loss = 0
    num_batches = 0

    # 收集预测和标签
    all_preds = {'has_palsy': [], 'palsy_side': [], 'hb_grade': [], 'sunnybrook': []}
    all_labels = {'has_palsy': [], 'palsy_side': [], 'hb_grade': [], 'sunnybrook': []}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            batch = move_to_device(batch, device)

            try:
                outputs = model(batch)
                loss, _ = loss_fn(outputs, batch['targets'])

                if not torch.isnan(loss):
                    total_loss += loss.item()
                    num_batches += 1

                # 收集预测
                session = outputs['session_outputs']
                for task in ['has_palsy', 'palsy_side', 'hb_grade']:
                    preds = session[task].argmax(dim=1).cpu().numpy()
                    labels = batch['targets'][task].cpu().numpy()
                    all_preds[task].extend(preds)
                    all_labels[task].extend(labels)

                # Sunnybrook (回归任务，需要反归一化)
                all_preds['sunnybrook'].extend(session['sunnybrook'].cpu().numpy() * 100)
                all_labels['sunnybrook'].extend(batch['targets']['sunnybrook'].cpu().numpy())

            except Exception as e:
                print(f"Error in validation batch: {e}")
                continue

    # 计算指标
    metrics = compute_metrics(all_preds, all_labels)
    metrics['loss'] = total_loss / max(num_batches, 1)

    return metrics


def compute_metrics(preds, labels):
    """计算评估指标"""
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

    metrics = {}

    # 分类任务
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

    # 回归任务
    if len(preds['sunnybrook']) > 0:
        metrics['sunnybrook_mae'] = mean_absolute_error(labels['sunnybrook'], preds['sunnybrook'])
    else:
        metrics['sunnybrook_mae'] = 100.0  # 最大误差

    return metrics


def plot_training_curves(history, output_dir):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    if history['val_loss']:
        axes[0, 0].plot(history['val_loss'], label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()

    # Has Palsy Accuracy
    if history['val_has_palsy_acc']:
        axes[0, 1].plot(history['val_has_palsy_acc'])
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Has Palsy Accuracy')

    # Palsy Side Accuracy
    if history['val_palsy_side_acc']:
        axes[0, 2].plot(history['val_palsy_side_acc'])
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Accuracy')
        axes[0, 2].set_title('Palsy Side Accuracy')

    # HB Grade Accuracy
    if history['val_hb_grade_acc']:
        axes[1, 0].plot(history['val_hb_grade_acc'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_title('HB Grade Accuracy')

    # Sunnybrook MAE
    if history['val_sunnybrook_mae']:
        axes[1, 1].plot(history['val_sunnybrook_mae'])
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].set_title('Sunnybrook MAE')

    # Combined Metric
    if history['val_combined_metric']:
        axes[1, 2].plot(history['val_combined_metric'])
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Combined Metric')
        axes[1, 2].set_title('Combined Metric')
        axes[1, 2].axhline(y=max(history['val_combined_metric']), color='r', linestyle='--', label='Best')
        axes[1, 2].legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    plt.close()


def main(args=None):
    """主函数"""
    # 解析参数
    if args is None:
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

    # 设备
    device = get_device()
    print(f"{'='*60}")
    print(f"H-GFA Net 层级多任务训练")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Fold: {args.fold}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")

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

    print(f"训练集: {len(train_dataset)} examinations")
    print(f"验证集: {len(val_dataset)} examinations")

    if len(train_dataset) == 0:
        print("错误: 训练集为空！请检查数据集划分。")
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
    model = HGFANet().to(device)
    param_info = model.count_parameters()
    print(f"\n模型参数: {param_info['total']:,} (可训练: {param_info['trainable']:,})")

    # 损失函数和优化器
    loss_fn = HierarchicalMultiTaskLoss().to(device)

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
        'val_has_palsy_acc': [],
        'val_palsy_side_acc': [],
        'val_hb_grade_acc': [],
        'val_sunnybrook_mae': [],
        'val_combined_metric': [],
    }

    # 训练循环
    best_metric = 0
    patience = 0
    max_patience = EARLY_STOPPING_PATIENCE

    print(f"\n开始训练...")
    print(f"{'='*60}")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        history['train_loss'].append(train_loss)
        print(f"Train Loss: {train_loss:.4f}")

        # 验证
        if val_loader is not None:
            val_metrics = validate(model, val_loader, loss_fn, device)
            scheduler.step()

            # 记录历史
            history['val_loss'].append(val_metrics['loss'])
            history['val_has_palsy_acc'].append(val_metrics['has_palsy_acc'])
            history['val_palsy_side_acc'].append(val_metrics['palsy_side_acc'])
            history['val_hb_grade_acc'].append(val_metrics['hb_grade_acc'])
            history['val_sunnybrook_mae'].append(val_metrics['sunnybrook_mae'])

            # 综合指标
            combined_metric = (
                val_metrics['has_palsy_acc'] * 0.2 +
                val_metrics['palsy_side_acc'] * 0.2 +
                val_metrics['hb_grade_acc'] * 0.4 +
                max(0, 1 - val_metrics['sunnybrook_mae'] / 100) * 0.2
            )
            history['val_combined_metric'].append(combined_metric)

            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"  has_palsy:   acc={val_metrics['has_palsy_acc']:.4f}, F1={val_metrics['has_palsy_f1']:.4f}")
            print(f"  palsy_side:  acc={val_metrics['palsy_side_acc']:.4f}, F1={val_metrics['palsy_side_f1']:.4f}")
            print(f"  hb_grade:    acc={val_metrics['hb_grade_acc']:.4f}, F1={val_metrics['hb_grade_f1']:.4f}")
            print(f"  sunnybrook:  MAE={val_metrics['sunnybrook_mae']:.2f}")
            print(f"  Combined:    {combined_metric:.4f}")

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
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break
        else:
            # 没有验证集，每10个epoch保存一次
            scheduler.step()
            if (epoch + 1) % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'train_loss': train_loss,
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
        'model_params': param_info,
        'timestamp': datetime.now().isoformat(),
    }
    with open(checkpoint_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"训练完成!")
    if val_loader is not None:
        print(f"Best Combined Metric: {best_metric:.4f}")
    print(f"模型保存到: {checkpoint_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()