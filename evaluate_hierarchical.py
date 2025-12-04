#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H-GFA Net 层级多任务评估脚本

功能：
1. 加载训练好的模型
2. 在验证集上评估所有任务
3. 生成详细的评估报告:
   - 动作级: severity分类准确率、混淆矩阵
   - 检查级: has_palsy, palsy_side, hb_grade混淆矩阵
   - 回归任务: sunnybrook MAE、散点图
4. 可视化注意力权重

直接在PyCharm中点击运行即可！
修改下面的FOLD变量来评估不同的fold
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
#   >=0 : 只评估对应 fold
#   <0  : 自动评估所有 fold (0,1,2)
FOLD = -1
N_FOLDS = 3

CHECKPOINT_TYPE = "best"  # "best" 或 "final"

DB_PATH = "facialPalsy.db"
CHECKPOINT_DIR = Path("checkpoints")
OUTPUT_DIR = Path("evaluation")

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


def evaluate_model(model, dataloader, device):
    """评估模型"""
    model.eval()

    # 收集预测和标签
    all_preds = {
        'has_palsy': [], 'palsy_side': [], 'hb_grade': [],
        'sunnybrook_pred': [], 'sunnybrook_true': []
    }
    all_labels = {'has_palsy': [], 'palsy_side': [], 'hb_grade': []}

    # 动作级
    action_preds = {name: [] for name in HGFANet.ACTION_NAMES}
    action_labels = {name: [] for name in HGFANet.ACTION_NAMES}

    # 注意力权重
    attention_weights_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = move_to_device(batch, device)

            try:
                outputs = model(batch)
                session = outputs['session_outputs']

                # 检查级预测
                for task in ['has_palsy', 'palsy_side', 'hb_grade']:
                    preds = session[task].argmax(dim=1).cpu().numpy()
                    labels = batch['targets'][task].cpu().numpy()
                    all_preds[task].extend(preds)
                    all_labels[task].extend(labels)

                # Sunnybrook
                all_preds['sunnybrook_pred'].extend(session['sunnybrook'].cpu().numpy() * 100)
                all_preds['sunnybrook_true'].extend(batch['targets']['sunnybrook'].cpu().numpy())

                # 动作级预测 (适配新的输出格式: action_severity 为 (B, num_actions, num_classes) 张量)
                action_severity_logits = outputs['action_severity']  # (B, 11, 5)

                for action_idx, action_name in enumerate(HierarchicalPalsyDataset.ACTION_NAMES):
                    # 只有当该动作在本 batch 存在标签时才评估
                    if action_name not in batch['targets']['action_severity']:
                        continue

                    exam_indices = batch['action_indices'].get(action_name, [])
                    if not exam_indices:
                        continue

                    # 这些 exam_indices 对应的是 batch 内的检查下标
                    exam_indices_tensor = torch.tensor(
                        exam_indices, dtype=torch.long, device=device
                    )

                    # 取出对应检查、对应动作的 logits: 形状 (n_i, 5)
                    logits = action_severity_logits[exam_indices_tensor, action_idx, :]

                    preds = logits.argmax(dim=1).cpu().numpy()
                    labels = batch['targets']['action_severity'][action_name].cpu().numpy()

                    action_preds[action_name].extend(preds)
                    action_labels[action_name].extend(labels)

                # 注意力权重
                if 'attention_weights' in session:
                    attention_weights_list.append(session['attention_weights'].cpu().numpy())

            except Exception as e:
                print(f"Error: {e}")
                continue

    return all_preds, all_labels, action_preds, action_labels, attention_weights_list


def plot_confusion_matrix(y_true, y_pred, labels, title, output_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
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


def plot_attention_heatmap(attention_weights, output_path):
    """绘制注意力权重热力图"""
    if not attention_weights:
        return

    # 平均注意力权重
    avg_attn = np.mean(np.concatenate(attention_weights, axis=0), axis=0)

    plt.figure(figsize=(12, 4))
    sns.heatmap(avg_attn, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=HGFANet.ACTION_NAMES,
                yticklabels=['Session Query'])
    plt.xlabel('Actions')
    plt.title('Average Attention Weights')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_report(all_preds, all_labels, action_preds, action_labels, output_dir):
    """生成评估报告"""
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("H-GFA Net 层级多任务评估报告")
    report_lines.append("=" * 60)

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
    report_lines.append("\n" + classification_report(
        all_labels['has_palsy'], all_preds['has_palsy'],
        target_names=['No Palsy', 'Palsy'], zero_division=0
    ))

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
    report_lines.append("\n" + classification_report(
        all_labels['palsy_side'], all_preds['palsy_side'],
        target_names=['None', 'Left', 'Right'], zero_division=0
    ))

    # HB Grade
    report_lines.append("\n--- HB分级 (HB Grade) ---")
    acc = accuracy_score(all_labels['hb_grade'], all_preds['hb_grade'])
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels['hb_grade'], all_preds['hb_grade'], average='weighted', zero_division=0
    )
    report_lines.append(f"Accuracy: {acc:.4f}")
    report_lines.append(f"Precision: {prec:.4f}")
    report_lines.append(f"Recall: {rec:.4f}")
    report_lines.append(f"F1-Score: {f1:.4f}")
    report_lines.append("\n" + classification_report(
        all_labels['hb_grade'],
        all_preds['hb_grade'],
        labels=list(range(6)),  # <<< 明确告诉它有 6 个类：0~5
        target_names=[f'Grade {i + 1}' for i in range(6)],
        zero_division=0  # support=0 的类，precision/recall 等直接写 0
    ))

    # Sunnybrook
    report_lines.append("\n--- Sunnybrook分数 (回归) ---")
    mae = mean_absolute_error(all_preds['sunnybrook_true'], all_preds['sunnybrook_pred'])
    rmse = np.sqrt(mean_squared_error(all_preds['sunnybrook_true'], all_preds['sunnybrook_pred']))
    r2 = r2_score(all_preds['sunnybrook_true'], all_preds['sunnybrook_pred'])
    report_lines.append(f"MAE: {mae:.2f}")
    report_lines.append(f"RMSE: {rmse:.2f}")
    report_lines.append(f"R²: {r2:.4f}")

    # 2. 动作级任务
    report_lines.append("\n" + "=" * 60)
    report_lines.append("动作级任务 (Action-Level)")
    report_lines.append("=" * 60)

    for action_name in HGFANet.ACTION_NAMES:
        if len(action_labels[action_name]) > 0:
            report_lines.append(f"\n--- {action_name} ---")
            acc = accuracy_score(action_labels[action_name], action_preds[action_name])
            report_lines.append(f"Accuracy: {acc:.4f}")
            report_lines.append(f"Samples: {len(action_labels[action_name])}")

    # 保存报告
    report_text = '\n'.join(report_lines)
    with open(output_dir / 'evaluation_report.txt', 'w') as f:
        f.write(report_text)

    print(report_text)

    has_palsy_acc = accuracy_score(all_labels['has_palsy'], all_preds['has_palsy'])
    palsy_side_acc = accuracy_score(all_labels['palsy_side'], all_preds['palsy_side'])
    hb_grade_acc = accuracy_score(all_labels['hb_grade'], all_preds['hb_grade'])

    return {
        'has_palsy_acc': float(has_palsy_acc),
        'palsy_side_acc': float(palsy_side_acc),
        'hb_grade_acc': float(hb_grade_acc),
        'sunnybrook_mae': float(mae),
        # 既然上面算了，可以顺便把 rmse、r2 也一起存进去
        'sunnybrook_rmse': float(rmse),
        'sunnybrook_r2': float(r2),
    }

def run_single_fold(args):
    """评估单个 fold 的主流程"""
    device = get_device()

    print("=" * 60)
    print("H-GFA Net 层级多任务评估")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Fold: {args.fold}")
    print(f"Checkpoint: {args.checkpoint_type}")

    # 检查点路径
    checkpoint_path = Path(args.checkpoint_dir) / f"fold_{args.fold}" / f"{args.checkpoint_type}.pth"
    if not checkpoint_path.exists():
        print(f"错误: 检查点不存在 - {checkpoint_path}")
        return

    # 输出目录
    output_dir = Path(args.output_dir) / f"fold_{args.fold}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    print(f"\n加载模型...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = HGFANet(config=checkpoint.get('config')).to(device)
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
        return

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_hierarchical, num_workers=0
    )

    # 评估
    print(f"\n开始评估 Fold {args.fold} ...")
    all_preds, all_labels, action_preds, action_labels, attn_weights = evaluate_model(
        model, val_loader, device
    )

    # 生成报告
    metrics = generate_report(all_preds, all_labels, action_preds, action_labels, output_dir)

    # 绘制混淆矩阵
    print(f"\n生成可视化...")

    # Has Palsy
    plot_confusion_matrix(
        all_labels['has_palsy'], all_preds['has_palsy'],
        ['No Palsy', 'Palsy'],
        'Has Palsy Confusion Matrix',
        output_dir / 'cm_has_palsy.png'
    )

    # Palsy Side
    plot_confusion_matrix(
        all_labels['palsy_side'], all_preds['palsy_side'],
        ['None', 'Left', 'Right'],
        'Palsy Side Confusion Matrix',
        output_dir / 'cm_palsy_side.png'
    )

    # HB Grade
    plot_confusion_matrix(
        all_labels['hb_grade'], all_preds['hb_grade'],
        [f'G{i+1}' for i in range(6)],
        'HB Grade Confusion Matrix',
        output_dir / 'cm_hb_grade.png'
    )

    # Sunnybrook散点图
    plot_regression_scatter(
        all_preds['sunnybrook_true'], all_preds['sunnybrook_pred'],
        'Sunnybrook Score Prediction',
        output_dir / 'scatter_sunnybrook.png'
    )

    # 注意力热力图
    plot_attention_heatmap(attn_weights, output_dir / 'attention_heatmap.png')

    # 保存指标
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Fold {args.fold} 评估完成！")
    print(f"结果保存到: {output_dir}")
    print(f"{'='*60}")


def main(args=None):
    """支持单 fold / 多 fold 评估的入口"""
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
        # 自动评估所有 fold
        for fold in range(N_FOLDS):
            print(f"\n================== 开始评估 Fold {fold} ==================\n")
            args_single = argparse.Namespace(**vars(args))
            args_single.fold = fold
            run_single_fold(args_single)
    else:
        # 只评估指定 fold
        run_single_fold(args)


if __name__ == '__main__':
    main()
