#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H-GFA Net 灵活配置训练脚本
============================================

支持:
1. 灵活特征选择 (可以只用wrinkle)
2. 序数分类方法选择 (standard, coral, cumulative)
3. 融合策略选择 (concat, attention)

配置方式:
- 修改下方的 FEATURE_CONFIG 字典

使用示例:
    # 只用wrinkle + CORAL序数分类
    FEATURE_CONFIG = {
        'enabled_features': ['wrinkle'],
        'ordinal_method': 'coral',
        'fusion_type': 'concat'
    }
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
from hgfa_net import HGFANet, FlexibleMultiTaskLoss
from dataset_palsy import HierarchicalPalsyDataset, collate_hierarchical

# ==================== 特征配置 (主要修改这里!) ====================
FEATURE_CONFIG = {
    # 启用的特征模态: ['geometric', 'visual', 'wrinkle', 'motion']
    # 可以只选一个，也可以组合多个
    'enabled_features': ['wrinkle'],  # 只用皱纹特征

    # 序数分类方法: 'standard', 'coral', 'cumulative'
    # - standard: 标准CrossEntropy分类
    # - coral: CORAL序数回归 (推荐用于HB分级)
    # - cumulative: 累积链路模型
    'ordinal_method': 'coral',

    # 融合类型 (多特征时有效): 'concat' 或 'attention'
    'fusion_type': 'concat',
}

# ==================== 训练配置 ====================
# 训练哪一个fold:
#   >=0 : 只训练对应的 fold
#   <0  : 自动循环训练所有 fold (0,1,2)
FOLD = -1
N_FOLDS = 3

# 数据库路径
DB_PATH = "facialPalsy.db"

# 输出路径
CHECKPOINT_DIR = Path("checkpoints_flexible")

# 训练超参数
NUM_EPOCHS = 100
BATCH_SIZE = 8  # 简化模型可以用更大的batch
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
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


def get_predictions(outputs, batch, ordinal_method):
    """
    从模型输出获取预测类别

    序数分类方法需要特殊处理
    """
    preds = {}

    # Action-level severity
    if ordinal_method == 'standard':
        severity_logits = outputs['action_severity']  # (B, 11, 5)
        preds['action_severity'] = severity_logits.argmax(dim=-1)  # (B, 11)
    else:
        # CORAL/Cumulative: 使用类别概率
        severity_probs = outputs['action_severity_probs']  # (B, 11, 5)
        preds['action_severity'] = severity_probs.argmax(dim=-1)  # (B, 11)

    # Session-level
    session = outputs['session_outputs']

    preds['has_palsy'] = session['has_palsy'].argmax(dim=1)
    preds['palsy_side'] = session['palsy_side'].argmax(dim=1)

    # HB Grade
    if ordinal_method == 'standard':
        preds['hb_grade'] = session['hb_grade'].argmax(dim=1)
    else:
        # CORAL/Cumulative
        hb_logits = session['hb_grade']  # (B, 5) for CORAL/cumulative
        if ordinal_method == 'coral':
            # P(Y > k) > 0.5
            probs = torch.sigmoid(hb_logits)
            preds['hb_grade'] = (probs > 0.5).sum(dim=-1).long()
        else:
            # cumulative: P(Y <= k) > 0.5
            probs = torch.sigmoid(hb_logits)
            preds['hb_grade'] = (probs < 0.5).sum(dim=-1).long()

    preds['sunnybrook'] = session['sunnybrook']

    return preds


def train_epoch(model, dataloader, optimizer, loss_fn, device, epoch,
                grad_clip=1.0, ordinal_method='standard'):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

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
            outputs = model(batch)
            loss, loss_dict = loss_fn(outputs, batch['targets'])

            if torch.isnan(loss):
                print(f"Warning: NaN loss at batch {batch_idx}, skipping")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            for task in task_losses.keys():
                if task in loss_dict:
                    task_losses[task].append(loss_dict[task])

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

    avg_losses = {'total': total_loss / max(num_batches, 1)}
    for task, losses in task_losses.items():
        if losses:
            avg_losses[task] = np.mean(losses)

    return avg_losses


def validate(model, dataloader, loss_fn, device, ordinal_method='standard'):
    """验证"""
    model.eval()
    total_loss = 0
    num_batches = 0

    all_preds = {
        'action_severity': [],
        'has_palsy': [],
        'palsy_side': [],
        'hb_grade': [],
        'sunnybrook': []
    }
    all_labels = {
        'action_severity': [],
        'has_palsy': [],
        'palsy_side': [],
        'hb_grade': [],
        'sunnybrook': []
    }

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

                # 获取预测
                preds = get_predictions(outputs, batch, ordinal_method)

                # Action severity
                all_preds['action_severity'].extend(preds['action_severity'].cpu().numpy().flatten())
                all_labels['action_severity'].extend(batch['targets']['action_severity'].cpu().numpy().flatten())

                # Session-level
                for task in ['has_palsy', 'palsy_side', 'hb_grade']:
                    all_preds[task].extend(preds[task].cpu().numpy())
                    all_labels[task].extend(batch['targets'][task].cpu().numpy())

                # Sunnybrook
                all_preds['sunnybrook'].extend(preds['sunnybrook'].cpu().numpy())
                all_labels['sunnybrook'].extend(batch['targets']['sunnybrook'].cpu().numpy())

            except Exception as e:
                print(f"Error in validation batch: {e}")
                continue

    metrics = compute_metrics(all_preds, all_labels)
    metrics['loss'] = total_loss / max(num_batches, 1)

    for task, losses in task_losses.items():
        if losses:
            metrics[f'{task}_loss'] = np.mean(losses)

    return metrics


def compute_metrics(preds, labels):
    """计算评估指标"""
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

    metrics = {}

    # Action-level severity
    if len(preds['action_severity']) > 0:
        valid_mask = np.array(labels['action_severity']) != -1
        if valid_mask.sum() > 0:
            valid_preds = np.array(preds['action_severity'])[valid_mask]
            valid_labels = np.array(labels['action_severity'])[valid_mask]

            metrics['action_severity_acc'] = accuracy_score(valid_labels, valid_preds)
            metrics['action_severity_f1'] = f1_score(
                valid_labels, valid_preds, average='weighted', zero_division=0
            )

            # MAE (对序数任务更有意义)
            metrics['action_severity_mae'] = mean_absolute_error(valid_labels, valid_preds)
        else:
            metrics['action_severity_acc'] = 0.0
            metrics['action_severity_f1'] = 0.0
            metrics['action_severity_mae'] = 5.0
    else:
        metrics['action_severity_acc'] = 0.0
        metrics['action_severity_f1'] = 0.0
        metrics['action_severity_mae'] = 5.0

    # Session-level分类任务
    for task in ['has_palsy', 'palsy_side', 'hb_grade']:
        if len(preds[task]) > 0 and len(labels[task]) > 0:
            metrics[f'{task}_acc'] = accuracy_score(labels[task], preds[task])
            metrics[f'{task}_f1'] = f1_score(
                labels[task], preds[task], average='weighted', zero_division=0
            )

            # 对HB grade计算MAE
            if task == 'hb_grade':
                metrics[f'{task}_mae'] = mean_absolute_error(labels[task], preds[task])
        else:
            metrics[f'{task}_acc'] = 0.0
            metrics[f'{task}_f1'] = 0.0
            if task == 'hb_grade':
                metrics[f'{task}_mae'] = 6.0

    # Sunnybrook MAE
    if len(preds['sunnybrook']) > 0:
        # 模型输出是归一化的 (0-1)，需要乘100
        pred_scores = np.array(preds['sunnybrook']) * 100
        true_scores = np.array(labels['sunnybrook'])
        metrics['sunnybrook_mae'] = mean_absolute_error(true_scores, pred_scores)
    else:
        metrics['sunnybrook_mae'] = 100.0

    return metrics


def plot_training_curves(history, output_dir):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Loss
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train')
    ax.plot(history['val_loss'], label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()

    # Action Severity Acc
    ax = axes[0, 1]
    ax.plot(history['val_action_severity_acc'], label='Accuracy')
    if 'val_action_severity_mae' in history:
        ax2 = ax.twinx()
        ax2.plot(history['val_action_severity_mae'], 'r--', label='MAE')
        ax2.set_ylabel('MAE', color='r')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Action Severity')
    ax.legend()

    # HB Grade
    ax = axes[0, 2]
    ax.plot(history['val_hb_grade_acc'], label='Accuracy')
    if 'val_hb_grade_mae' in history:
        ax2 = ax.twinx()
        ax2.plot(history['val_hb_grade_mae'], 'r--', label='MAE')
        ax2.set_ylabel('MAE', color='r')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('HB Grade')
    ax.legend()

    # Has Palsy
    ax = axes[1, 0]
    ax.plot(history['val_has_palsy_acc'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Has Palsy Accuracy')

    # Palsy Side
    ax = axes[1, 1]
    ax.plot(history['val_palsy_side_acc'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Palsy Side Accuracy')

    # Sunnybrook MAE
    ax = axes[1, 2]
    ax.plot(history['val_sunnybrook_mae'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE')
    ax.set_title('Sunnybrook MAE')

    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150)
    plt.close()


def run_single_fold(args):
    """训练单个fold"""
    device = get_device()

    print("=" * 60)
    print("H-GFA Net 灵活配置训练")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Fold: {args.fold}")
    print(f"\n特征配置:")
    print(f"  - 启用特征: {args.enabled_features}")
    print(f"  - 序数方法: {args.ordinal_method}")
    print(f"  - 融合类型: {args.fusion_type}")

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
        print("❌ 错误: 训练集为空！")
        return

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
    config = HGFANet.get_default_config()
    config['enabled_features'] = args.enabled_features
    config['ordinal_method'] = args.ordinal_method
    config['fusion_type'] = args.fusion_type

    model = HGFANet(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 损失函数
    loss_fn = FlexibleMultiTaskLoss(
        ordinal_method=args.ordinal_method,
        use_uncertainty_weighting=True
    ).to(device)
    print(f"✓ 使用 {args.ordinal_method} 序数方法")

    # 优化器
    optimizer = optim.AdamW(
        list(model.parameters()) + list(loss_fn.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 输出目录
    checkpoint_dir = Path(args.checkpoint_dir) / f"fold_{args.fold}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_action_severity_acc': [],
        'val_action_severity_f1': [],
        'val_action_severity_mae': [],
        'val_has_palsy_acc': [],
        'val_palsy_side_acc': [],
        'val_hb_grade_acc': [],
        'val_hb_grade_mae': [],
        'val_sunnybrook_mae': [],
        'val_combined_metric': [],
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
        train_losses = train_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch + 1,
            ordinal_method=args.ordinal_method
        )
        history['train_loss'].append(train_losses['total'])
        print(f"Train Loss: {train_losses['total']:.4f}")

        # 验证
        if val_loader is not None:
            val_metrics = validate(model, val_loader, loss_fn, device, args.ordinal_method)
            scheduler.step()

            # 记录历史
            history['val_loss'].append(val_metrics['loss'])
            history['val_action_severity_acc'].append(val_metrics['action_severity_acc'])
            history['val_action_severity_f1'].append(val_metrics['action_severity_f1'])
            history['val_action_severity_mae'].append(val_metrics.get('action_severity_mae', 0))
            history['val_has_palsy_acc'].append(val_metrics['has_palsy_acc'])
            history['val_palsy_side_acc'].append(val_metrics['palsy_side_acc'])
            history['val_hb_grade_acc'].append(val_metrics['hb_grade_acc'])
            history['val_hb_grade_mae'].append(val_metrics.get('hb_grade_mae', 0))
            history['val_sunnybrook_mae'].append(val_metrics['sunnybrook_mae'])

            # 综合指标 (对序数任务加入MAE考量)
            # 对于序数任务，MAE比准确率更重要
            hb_mae_score = max(0, 1 - val_metrics.get('hb_grade_mae', 0) / 5)  # 5是最大可能MAE

            combined_metric = (
                    val_metrics['action_severity_acc'] * 0.10 +
                    val_metrics['has_palsy_acc'] * 0.15 +
                    val_metrics['palsy_side_acc'] * 0.15 +
                    val_metrics['hb_grade_acc'] * 0.25 +
                    hb_mae_score * 0.20 +  # HB MAE权重
                    max(0, 1 - val_metrics['sunnybrook_mae'] / 100) * 0.15
            )
            history['val_combined_metric'].append(combined_metric)

            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"  action_severity: acc={val_metrics['action_severity_acc']:.4f}, "
                  f"MAE={val_metrics.get('action_severity_mae', 0):.2f}")
            print(f"  has_palsy:       acc={val_metrics['has_palsy_acc']:.4f}")
            print(f"  palsy_side:      acc={val_metrics['palsy_side_acc']:.4f}")
            print(f"  hb_grade:        acc={val_metrics['hb_grade_acc']:.4f}, "
                  f"MAE={val_metrics.get('hb_grade_mae', 0):.2f}")
            print(f"  sunnybrook:      MAE={val_metrics['sunnybrook_mae']:.2f}")
            print(f"  Combined:        {combined_metric:.4f}")

            # 打印任务权重
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
                    'config': config,
                }, checkpoint_dir / 'best.pth')
                print(f"  ✓ Saved best model (metric: {best_metric:.4f})")
            else:
                patience += 1
                if patience >= max_patience:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break
        else:
            scheduler.step()

    # 保存最终模型
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_fn_state_dict': loss_fn.state_dict(),
        'history': history,
        'config': config,
    }, checkpoint_dir / 'final.pth')

    # 绘制训练曲线
    if val_loader is not None:
        plot_training_curves(history, checkpoint_dir)

    # 保存配置
    train_config = {
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
        'feature_config': {
            'enabled_features': args.enabled_features,
            'ordinal_method': args.ordinal_method,
            'fusion_type': args.fusion_type,
        },
        'timestamp': datetime.now().isoformat(),
    }
    with open(checkpoint_dir / 'config.json', 'w', encoding='utf-8') as f:
        json.dump(train_config, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Fold {args.fold} 训练完成!")
    if val_loader is not None:
        print(f"Best Combined Metric: {best_metric:.4f}")
    print(f"模型保存到: {checkpoint_dir}")
    print(f"{'=' * 60}")


def main():
    """主入口"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_path', type=str, default=DB_PATH)
    parser.add_argument('--fold', type=int, default=FOLD)
    parser.add_argument('--split_version', type=str, default=SPLIT_VERSION)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LEARNING_RATE)
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY)
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--checkpoint_dir', type=str, default=str(CHECKPOINT_DIR))

    # 特征配置参数
    parser.add_argument('--enabled_features', type=str, nargs='+',
                        default=FEATURE_CONFIG['enabled_features'],
                        help='启用的特征: geometric, visual, wrinkle, motion')
    parser.add_argument('--ordinal_method', type=str,
                        default=FEATURE_CONFIG['ordinal_method'],
                        choices=['standard', 'coral', 'cumulative'],
                        help='序数分类方法')
    parser.add_argument('--fusion_type', type=str,
                        default=FEATURE_CONFIG['fusion_type'],
                        choices=['concat', 'attention'],
                        help='特征融合类型')

    args = parser.parse_args()

    if args.fold < 0:
        for fold in range(N_FOLDS):
            print(f"\n================== 开始训练 Fold {fold} ==================\n")
            args_single = argparse.Namespace(**vars(args))
            args_single.fold = fold
            run_single_fold(args_single)
    else:
        run_single_fold(args)


if __name__ == '__main__':
    main()