#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H-GFA Net 灵活配置评估脚本
============================================

支持:
1. 序数分类评估 (MAE, 混淆矩阵)
2. 相邻类别准确率 (Adjacent Accuracy)
3. 可视化分析

直接在PyCharm中点击运行即可！
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
import json

# 导入自定义模块
from hgfa_net import HGFANet
from dataset_palsy import HierarchicalPalsyDataset, collate_hierarchical

# ==================== 配置参数 ====================
FOLD = -1  # <0 评估所有fold
N_FOLDS = 3
CHECKPOINT_TYPE = "best"
DB_PATH = "facialPalsy.db"
CHECKPOINT_DIR = Path("checkpoints_flexible")
OUTPUT_DIR = Path("evaluation_flexible")
BATCH_SIZE = 8
SPLIT_VERSION = "v1.0"


def get_device():
    """获取最佳可用设备"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
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


def get_predictions_from_outputs(outputs, ordinal_method):
    """从模型输出获取预测"""
    preds = {}

    # Action severity
    if ordinal_method == 'standard':
        severity_logits = outputs['action_severity']
        preds['action_severity'] = severity_logits.argmax(dim=-1)
    else:
        severity_probs = outputs['action_severity_probs']
        preds['action_severity'] = severity_probs.argmax(dim=-1)

    # Session level
    session = outputs['session_outputs']
    preds['has_palsy'] = session['has_palsy'].argmax(dim=1)
    preds['palsy_side'] = session['palsy_side'].argmax(dim=1)

    # HB Grade
    if ordinal_method == 'standard':
        preds['hb_grade'] = session['hb_grade'].argmax(dim=1)
    else:
        hb_logits = session['hb_grade']
        if ordinal_method == 'coral':
            probs = torch.sigmoid(hb_logits)
            preds['hb_grade'] = (probs > 0.5).sum(dim=-1).long()
        else:  # cumulative
            probs = torch.sigmoid(hb_logits)
            preds['hb_grade'] = (probs < 0.5).sum(dim=-1).long()

    preds['sunnybrook'] = session['sunnybrook']

    return preds


def evaluate_model(model, dataloader, device, ordinal_method='standard'):
    """评估模型"""
    model.eval()

    all_preds = {
        'has_palsy': [], 'palsy_side': [], 'hb_grade': [],
        'sunnybrook_pred': [], 'sunnybrook_true': []
    }
    all_labels = {'has_palsy': [], 'palsy_side': [], 'hb_grade': []}

    # 动作级
    action_preds = {name: [] for name in HGFANet.ACTION_NAMES}
    action_labels = {name: [] for name in HGFANet.ACTION_NAMES}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = move_to_device(batch, device)

            try:
                outputs = model(batch)
                preds = get_predictions_from_outputs(outputs, ordinal_method)

                # 检查级预测
                for task in ['has_palsy', 'palsy_side', 'hb_grade']:
                    all_preds[task].extend(preds[task].cpu().numpy())
                    all_labels[task].extend(batch['targets'][task].cpu().numpy())

                # Sunnybrook
                all_preds['sunnybrook_pred'].extend(preds['sunnybrook'].cpu().numpy() * 100)
                all_preds['sunnybrook_true'].extend(batch['targets']['sunnybrook'].cpu().numpy())

                # 动作级
                severity_preds = preds['action_severity']  # (B, 11)
                severity_targets = batch['targets']['action_severity']  # (B, 11)

                for action_idx, action_name in enumerate(HGFANet.ACTION_NAMES):
                    for sample_idx in range(severity_preds.size(0)):
                        target = severity_targets[sample_idx, action_idx].item()
                        if target != -1:  # 有效标签
                            pred = severity_preds[sample_idx, action_idx].item()
                            action_preds[action_name].append(pred)
                            action_labels[action_name].append(target)

            except Exception as e:
                print(f"Error: {e}")
                continue

    return all_preds, all_labels, action_preds, action_labels


def adjacent_accuracy(y_true, y_pred, tolerance=1):
    """
    计算相邻类别准确率

    对于序数分类，预测在正确答案±tolerance范围内都算正确
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    correct = np.abs(y_true - y_pred) <= tolerance
    return correct.mean()


def plot_confusion_matrix(y_true, y_pred, labels, title, output_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_ordinal_analysis(y_true, y_pred, num_classes, title, output_path):
    """
    序数分类专用分析图

    包含:
    1. 混淆矩阵 (带权重显示距离)
    2. 预测偏差分布
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    ax = axes[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=[f'{i}' for i in range(num_classes)],
                yticklabels=[f'{i}' for i in range(num_classes)])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')

    # 2. 偏差分布
    ax = axes[1]
    errors = y_pred - y_true
    error_counts = {}
    for e in range(-num_classes + 1, num_classes):
        error_counts[e] = np.sum(errors == e)

    bars = ax.bar(list(error_counts.keys()), list(error_counts.values()),
                  color=['red' if k != 0 else 'green' for k in error_counts.keys()])
    ax.set_xlabel('Prediction Error (Pred - True)')
    ax.set_ylabel('Count')
    ax.set_title('Prediction Error Distribution')
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)

    # 3. 每个类别的准确率
    ax = axes[2]
    class_acc = []
    class_adj_acc = []  # 相邻准确率
    for c in range(num_classes):
        mask = y_true == c
        if mask.sum() > 0:
            class_acc.append(np.mean(y_pred[mask] == c))
            class_adj_acc.append(np.mean(np.abs(y_pred[mask] - c) <= 1))
        else:
            class_acc.append(0)
            class_adj_acc.append(0)

    x = np.arange(num_classes)
    width = 0.35
    ax.bar(x - width / 2, class_acc, width, label='Exact Acc')
    ax.bar(x + width / 2, class_adj_acc, width, label='Adjacent Acc (±1)')
    ax.set_xlabel('Class')
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Class Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{i}' for i in range(num_classes)])
    ax.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_regression_scatter(y_true, y_pred, title, output_path):
    """绘制回归散点图"""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([0, 100], [0, 100], 'r--', label='Ideal')
    plt.xlabel('True Sunnybrook Score')
    plt.ylabel('Predicted Sunnybrook Score')
    plt.title(title)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_report(all_preds, all_labels, action_preds, action_labels,
                    output_dir, ordinal_method='standard'):
    """生成评估报告"""
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("H-GFA Net 灵活配置评估报告")
    report_lines.append(f"序数分类方法: {ordinal_method}")
    report_lines.append("=" * 60)

    metrics = {}

    # 1. 检查级任务
    report_lines.append("\n" + "=" * 60)
    report_lines.append("检查级任务 (Session-Level)")
    report_lines.append("=" * 60)

    # Has Palsy
    report_lines.append("\n--- 是否面瘫 (Has Palsy) ---")
    acc = accuracy_score(all_labels['has_palsy'], all_preds['has_palsy'])
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels['has_palsy'], all_preds['has_palsy'], average='weighted', zero_division=0
    )
    report_lines.append(f"Accuracy: {acc:.4f}")
    report_lines.append(f"Precision: {prec:.4f}")
    report_lines.append(f"Recall: {rec:.4f}")
    report_lines.append(f"F1-Score: {f1:.4f}")
    metrics['has_palsy_acc'] = acc

    # Palsy Side
    report_lines.append("\n--- 面瘫侧别 (Palsy Side) ---")
    acc = accuracy_score(all_labels['palsy_side'], all_preds['palsy_side'])
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels['palsy_side'], all_preds['palsy_side'], average='weighted', zero_division=0
    )
    report_lines.append(f"Accuracy: {acc:.4f}")
    report_lines.append(f"Precision: {prec:.4f}")
    report_lines.append(f"Recall: {rec:.4f}")
    report_lines.append(f"F1-Score: {f1:.4f}")
    metrics['palsy_side_acc'] = acc

    # HB Grade (序数任务重点)
    report_lines.append("\n--- HB分级 (HB Grade) [序数任务] ---")
    acc = accuracy_score(all_labels['hb_grade'], all_preds['hb_grade'])
    mae = mean_absolute_error(all_labels['hb_grade'], all_preds['hb_grade'])
    adj_acc = adjacent_accuracy(all_labels['hb_grade'], all_preds['hb_grade'], tolerance=1)

    report_lines.append(f"Exact Accuracy: {acc:.4f}")
    report_lines.append(f"Adjacent Accuracy (±1): {adj_acc:.4f}")
    report_lines.append(f"MAE: {mae:.4f}")
    report_lines.append(f"\n分类报告:")
    report_lines.append(classification_report(
        all_labels['hb_grade'], all_preds['hb_grade'],
        labels=list(range(6)),
        target_names=[f'Grade {i + 1}' for i in range(6)],
        zero_division=0
    ))
    metrics['hb_grade_acc'] = acc
    metrics['hb_grade_adj_acc'] = adj_acc
    metrics['hb_grade_mae'] = mae

    # Sunnybrook
    report_lines.append("\n--- Sunnybrook分数 (回归) ---")
    mae = mean_absolute_error(all_preds['sunnybrook_true'], all_preds['sunnybrook_pred'])
    rmse = np.sqrt(mean_squared_error(all_preds['sunnybrook_true'], all_preds['sunnybrook_pred']))
    r2 = r2_score(all_preds['sunnybrook_true'], all_preds['sunnybrook_pred'])
    report_lines.append(f"MAE: {mae:.2f}")
    report_lines.append(f"RMSE: {rmse:.2f}")
    report_lines.append(f"R²: {r2:.4f}")
    metrics['sunnybrook_mae'] = mae
    metrics['sunnybrook_rmse'] = rmse
    metrics['sunnybrook_r2'] = r2

    # 2. 动作级任务
    report_lines.append("\n" + "=" * 60)
    report_lines.append("动作级任务 (Action-Level) [序数任务]")
    report_lines.append("=" * 60)

    all_action_preds = []
    all_action_labels = []

    for action_name in HGFANet.ACTION_NAMES:
        if len(action_labels[action_name]) > 0:
            report_lines.append(f"\n--- {action_name} ---")
            acc = accuracy_score(action_labels[action_name], action_preds[action_name])
            mae = mean_absolute_error(action_labels[action_name], action_preds[action_name])
            adj_acc = adjacent_accuracy(action_labels[action_name], action_preds[action_name])

            report_lines.append(f"Exact Accuracy: {acc:.4f}")
            report_lines.append(f"Adjacent Accuracy (±1): {adj_acc:.4f}")
            report_lines.append(f"MAE: {mae:.4f}")
            report_lines.append(f"Samples: {len(action_labels[action_name])}")

            all_action_preds.extend(action_preds[action_name])
            all_action_labels.extend(action_labels[action_name])

    # 汇总动作级指标
    if all_action_labels:
        report_lines.append("\n--- 总体动作级指标 ---")
        overall_acc = accuracy_score(all_action_labels, all_action_preds)
        overall_mae = mean_absolute_error(all_action_labels, all_action_preds)
        overall_adj_acc = adjacent_accuracy(all_action_labels, all_action_preds)

        report_lines.append(f"Overall Exact Accuracy: {overall_acc:.4f}")
        report_lines.append(f"Overall Adjacent Accuracy (±1): {overall_adj_acc:.4f}")
        report_lines.append(f"Overall MAE: {overall_mae:.4f}")

        metrics['action_severity_acc'] = overall_acc
        metrics['action_severity_adj_acc'] = overall_adj_acc
        metrics['action_severity_mae'] = overall_mae

    # 保存报告
    report_text = '\n'.join(report_lines)
    with open(output_dir / 'evaluation_report.txt', 'w') as f:
        f.write(report_text)

    print(report_text)

    # 确保 metrics 里的数值都是 Python 原生 float，避免 json.dump 报错
    clean_metrics = {}
    for k, v in metrics.items():
        # numpy 标量 / torch 标量都转成 float
        if isinstance(v, (np.generic, np.floating)):
            clean_metrics[k] = float(v)
        else:
            clean_metrics[k] = v

    return clean_metrics


def run_single_fold(args):
    """评估单个fold"""
    device = get_device()

    print("=" * 60)
    print("H-GFA Net 灵活配置评估")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Fold: {args.fold}")
    print(f"Checkpoint: {args.checkpoint_type}")

    # 检查点路径
    checkpoint_path = Path(args.checkpoint_dir) / f"fold_{args.fold}" / f"{args.checkpoint_type}.pth"
    if not checkpoint_path.exists():
        print(f"错误: 检查点不存在 - {checkpoint_path}")
        return None

    # 输出目录
    output_dir = Path(args.output_dir) / f"fold_{args.fold}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    print(f"\n加载模型...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config')

    if config is None:
        print("警告: 检查点中没有config，使用默认配置")
        config = HGFANet.get_default_config()

    ordinal_method = config.get('ordinal_method', 'standard')
    print(f"序数分类方法: {ordinal_method}")
    print(f"启用特征: {config.get('enabled_features', ['wrinkle'])}")

    model = HGFANet(config=config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'combined_metric' in checkpoint:
        print(f"Combined metric: {checkpoint['combined_metric']:.4f}")

    # 加载数据集
    print(f"\n加载验证集...")
    val_dataset = HierarchicalPalsyDataset(
        args.db_path, args.fold, 'val', args.split_version,
        use_augmentation=False
    )
    print(f"验证集: {len(val_dataset)} examinations")

    if len(val_dataset) == 0:
        print("错误: 验证集为空！")
        return None

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_hierarchical, num_workers=0
    )

    # 评估
    print(f"\n开始评估 Fold {args.fold} ...")
    all_preds, all_labels, action_preds, action_labels = evaluate_model(
        model, val_loader, device, ordinal_method
    )

    # 生成报告
    metrics = generate_report(
        all_preds, all_labels, action_preds, action_labels,
        output_dir, ordinal_method
    )

    # 绘制可视化
    print(f"\n生成可视化...")

    # Has Palsy混淆矩阵
    plot_confusion_matrix(
        all_labels['has_palsy'], all_preds['has_palsy'],
        ['No Palsy', 'Palsy'],
        'Has Palsy Confusion Matrix',
        output_dir / 'cm_has_palsy.png'
    )

    # Palsy Side混淆矩阵
    plot_confusion_matrix(
        all_labels['palsy_side'], all_preds['palsy_side'],
        ['None', 'Left', 'Right'],
        'Palsy Side Confusion Matrix',
        output_dir / 'cm_palsy_side.png'
    )

    # HB Grade序数分析
    plot_ordinal_analysis(
        all_labels['hb_grade'], all_preds['hb_grade'],
        num_classes=6,
        title='HB Grade Ordinal Analysis',
        output_path=output_dir / 'ordinal_hb_grade.png'
    )

    # Sunnybrook散点图
    plot_regression_scatter(
        all_preds['sunnybrook_true'], all_preds['sunnybrook_pred'],
        'Sunnybrook Score Prediction',
        output_dir / 'scatter_sunnybrook.png'
    )

    # 汇总动作级序数分析
    all_action_preds = []
    all_action_labels = []
    for action_name in HGFANet.ACTION_NAMES:
        all_action_preds.extend(action_preds[action_name])
        all_action_labels.extend(action_labels[action_name])

    if all_action_labels:
        plot_ordinal_analysis(
            all_action_labels, all_action_preds,
            num_classes=5,
            title='Action Severity Ordinal Analysis',
            output_path=output_dir / 'ordinal_action_severity.png'
        )

    # 保存指标
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Fold {args.fold} 评估完成！")
    print(f"结果保存到: {output_dir}")
    print(f"{'=' * 60}")

    return metrics


def main(args=None):
    """主入口"""
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument('--db_path', type=str, default=DB_PATH)
        parser.add_argument('--fold', type=int, default=FOLD)
        parser.add_argument('--checkpoint_type', type=str, default=CHECKPOINT_TYPE)
        parser.add_argument('--checkpoint_dir', type=str, default=str(CHECKPOINT_DIR))
        parser.add_argument('--output_dir', type=str, default=str(OUTPUT_DIR))
        parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
        parser.add_argument('--split_version', type=str, default=SPLIT_VERSION)
        args = parser.parse_args()

    if args.fold < 0:
        # 评估所有fold并汇总
        all_metrics = []
        for fold in range(N_FOLDS):
            print(f"\n================== 开始评估 Fold {fold} ==================\n")
            args_single = argparse.Namespace(**vars(args))
            args_single.fold = fold
            metrics = run_single_fold(args_single)
            if metrics:
                all_metrics.append(metrics)

        # 汇总结果
        if all_metrics:
            print("\n" + "=" * 60)
            print("跨Fold汇总结果")
            print("=" * 60)

            summary = {}
            for key in all_metrics[0].keys():
                values = [m[key] for m in all_metrics if key in m]

                # 先把 values 里也都转成普通 float，保证干净
                values_clean = [float(v) for v in values]

                mean_v = float(np.mean(values_clean))
                std_v = float(np.std(values_clean))

                summary[key] = {
                    'mean': mean_v,
                    'std': std_v,
                    'values': values_clean,
                }
                print(f"{key}: {mean_v:.4f} ± {std_v:.4f}")

            # 保存汇总
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            with open(output_dir / 'summary.json', 'w') as f:
                json.dump(summary, f, indent=2)

    else:
        run_single_fold(args)


if __name__ == '__main__':
    main()