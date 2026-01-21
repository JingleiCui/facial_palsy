#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阈值分析工具
============

分析阈值在真实数据中的表现：
1. 真实指标值的分布
2. 阈值划分的效果
3. 准确率统计

用法:
    python threshold_analyzer.py --data_dir ./output --action ShrugNose
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import argparse


def load_session_results(data_dir: Path, action_name: str) -> List[Dict]:
    """
    加载所有session的某个动作的结果

    Returns:
        [{"session_id": ..., "indicators": ..., "gt_palsy_side": ...}, ...]
    """
    results = []

    for session_dir in sorted(data_dir.iterdir()):
        if not session_dir.is_dir():
            continue

        action_dir = session_dir / action_name
        indicators_file = action_dir / "indicators.json"

        if not indicators_file.exists():
            continue

        with open(indicators_file, 'r', encoding='utf-8') as f:
            indicators = json.load(f)

        # 尝试加载ground truth（如果有）
        gt_file = session_dir / "ground_truth.json"
        gt_palsy_side = None
        gt_hb_grade = None

        if gt_file.exists():
            with open(gt_file, 'r', encoding='utf-8') as f:
                gt = json.load(f)
            gt_palsy_side = gt.get("palsy_side")
            gt_hb_grade = gt.get("hb_grade")

        results.append({
            "session_id": session_dir.name,
            "indicators": indicators,
            "gt_palsy_side": gt_palsy_side,
            "gt_hb_grade": gt_hb_grade,
        })

    return results


def extract_shrug_nose_metrics(results: List[Dict]) -> Dict[str, List]:
    """
    从结果中提取皱鼻动作的关键指标
    """
    data = {
        "session_ids": [],
        "left_dist": [],
        "right_dist": [],
        "asymmetry": [],
        "predicted_side": [],
        "gt_side": [],
        "has_gt": [],
    }

    for r in results:
        indicators = r["indicators"]
        action_specific = indicators.get("action_specific", {})

        # 尝试从不同版本的数据结构中提取
        perp_dist = action_specific.get("perpendicular_distance", {})
        if not perp_dist:
            # 旧版本的数据结构
            vert_dist = action_specific.get("vertical_distance", {})
            left = vert_dist.get("left", 0)
            right = vert_dist.get("right", 0)
            asym = vert_dist.get("asymmetry", 0)
        else:
            left = perp_dist.get("left_px", 0)
            right = perp_dist.get("right_px", 0)
            asym = perp_dist.get("asymmetry", 0)

        # 预测的面瘫侧别
        palsy_det = action_specific.get("palsy_detection", {})
        pred_side = palsy_det.get("palsy_side", 0)

        data["session_ids"].append(r["session_id"])
        data["left_dist"].append(left)
        data["right_dist"].append(right)
        data["asymmetry"].append(asym)
        data["predicted_side"].append(pred_side)
        data["gt_side"].append(r["gt_palsy_side"])
        data["has_gt"].append(r["gt_palsy_side"] is not None)

    return data


def analyze_threshold_performance(
        data: Dict[str, List],
        threshold: float,
        metric_name: str = "asymmetry"
) -> Dict[str, Any]:
    """
    分析给定阈值的性能

    Returns:
        {
            "threshold": float,
            "total_samples": int,
            "samples_with_gt": int,
            "predictions": {...},
            "accuracy": {...},
            "distribution": {...},
        }
    """
    metric_values = np.array(data[metric_name])
    predicted_sides = np.array(data["predicted_side"])
    gt_sides = np.array(data["gt_side"])
    has_gt = np.array(data["has_gt"])

    n_total = len(metric_values)
    n_with_gt = sum(has_gt)

    # 根据阈值划分
    above_threshold = metric_values >= threshold
    below_threshold = metric_values < threshold

    # 统计预测分布
    pred_normal = sum(predicted_sides == 0)
    pred_left = sum(predicted_sides == 1)
    pred_right = sum(predicted_sides == 2)

    # 计算准确率（如果有GT）
    accuracy = None
    confusion = None

    if n_with_gt > 0:
        gt_valid = gt_sides[has_gt].astype(int)
        pred_valid = predicted_sides[has_gt].astype(int)

        correct = sum(gt_valid == pred_valid)
        accuracy = correct / n_with_gt

        # 混淆矩阵
        confusion = np.zeros((3, 3), dtype=int)
        for gt, pred in zip(gt_valid, pred_valid):
            confusion[gt, pred] += 1

    # 指标分布统计
    distribution = {
        "mean": float(np.mean(metric_values)),
        "std": float(np.std(metric_values)),
        "min": float(np.min(metric_values)),
        "max": float(np.max(metric_values)),
        "median": float(np.median(metric_values)),
        "q25": float(np.percentile(metric_values, 25)),
        "q75": float(np.percentile(metric_values, 75)),
    }

    # 阈值划分效果
    threshold_effect = {
        "n_above": int(sum(above_threshold)),
        "n_below": int(sum(below_threshold)),
        "pct_above": float(sum(above_threshold) / n_total) if n_total > 0 else 0,
        "pct_below": float(sum(below_threshold) / n_total) if n_total > 0 else 0,
    }

    return {
        "threshold": threshold,
        "metric_name": metric_name,
        "total_samples": n_total,
        "samples_with_gt": n_with_gt,
        "predictions": {
            "normal": pred_normal,
            "left_palsy": pred_left,
            "right_palsy": pred_right,
        },
        "accuracy": accuracy,
        "confusion_matrix": confusion.tolist() if confusion is not None else None,
        "distribution": distribution,
        "threshold_effect": threshold_effect,
    }


def plot_threshold_analysis(
        data: Dict[str, List],
        threshold: float,
        output_path: Path,
        metric_name: str = "asymmetry"
) -> None:
    """
    绘制阈值分析图
    """
    metric_values = np.array(data[metric_name])
    predicted_sides = np.array(data["predicted_side"])
    gt_sides = np.array(data["gt_side"])
    has_gt = np.array(data["has_gt"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ===== 图1: 指标分布直方图 =====
    ax1 = axes[0, 0]

    # 按预测侧别分组
    normal_mask = predicted_sides == 0
    left_mask = predicted_sides == 1
    right_mask = predicted_sides == 2

    bins = np.linspace(0, max(metric_values) * 1.1, 30)

    ax1.hist(metric_values[normal_mask], bins=bins, alpha=0.6, label='Predicted Normal', color='green')
    ax1.hist(metric_values[left_mask], bins=bins, alpha=0.6, label='Predicted Left Palsy', color='blue')
    ax1.hist(metric_values[right_mask], bins=bins, alpha=0.6, label='Predicted Right Palsy', color='red')

    # 阈值线
    ax1.axvline(x=threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.2%})')

    ax1.set_xlabel(f'{metric_name.capitalize()}', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title(f'{metric_name.capitalize()} Distribution by Predicted Side', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ===== 图2: 左右距离散点图 =====
    ax2 = axes[0, 1]

    left_dist = np.array(data["left_dist"])
    right_dist = np.array(data["right_dist"])

    colors = ['green' if p == 0 else 'blue' if p == 1 else 'red' for p in predicted_sides]

    ax2.scatter(left_dist, right_dist, c=colors, alpha=0.6, s=50)

    # 对角线 (左=右)
    max_val = max(max(left_dist), max(right_dist)) * 1.1
    ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='L=R')

    ax2.set_xlabel('Left Ala Distance (px)', fontsize=11)
    ax2.set_ylabel('Right Ala Distance (px)', fontsize=11)
    ax2.set_title('Left vs Right Ala-to-EyeLine Distance', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # ===== 图3: 按GT侧别分组的指标分布 (如果有GT) =====
    ax3 = axes[1, 0]

    if any(has_gt):
        # 只取有GT的样本
        gt_valid = gt_sides[has_gt].astype(int)
        metric_valid = metric_values[has_gt]

        gt_0 = metric_valid[gt_valid == 0]
        gt_1 = metric_valid[gt_valid == 1]
        gt_2 = metric_valid[gt_valid == 2]

        positions = []
        data_to_plot = []
        labels = []

        if len(gt_0) > 0:
            positions.append(1)
            data_to_plot.append(gt_0)
            labels.append(f'GT Normal\n(n={len(gt_0)})')
        if len(gt_1) > 0:
            positions.append(2)
            data_to_plot.append(gt_1)
            labels.append(f'GT Left Palsy\n(n={len(gt_1)})')
        if len(gt_2) > 0:
            positions.append(3)
            data_to_plot.append(gt_2)
            labels.append(f'GT Right Palsy\n(n={len(gt_2)})')

        if data_to_plot:
            bp = ax3.boxplot(data_to_plot, positions=positions, widths=0.6)
            ax3.axhline(y=threshold, color='red', linestyle='--', linewidth=2,
                        label=f'Threshold ({threshold:.2%})')
            ax3.set_xticks(positions)
            ax3.set_xticklabels(labels)
            ax3.set_ylabel(f'{metric_name.capitalize()}', fontsize=11)
            ax3.set_title(f'{metric_name.capitalize()} by Ground Truth Side', fontsize=12, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No Ground Truth Available', ha='center', va='center', fontsize=14)
        ax3.set_title('Ground Truth Analysis', fontsize=12, fontweight='bold')

    # ===== 图4: 阈值敏感性分析 =====
    ax4 = axes[1, 1]

    if any(has_gt):
        # 测试不同阈值的准确率
        thresholds_to_test = np.linspace(0, 0.3, 50)
        accuracies = []

        gt_valid = gt_sides[has_gt].astype(int)
        metric_valid = metric_values[has_gt]
        left_valid = left_dist[has_gt]
        right_valid = right_dist[has_gt]

        for thr in thresholds_to_test:
            # 根据阈值预测
            pred = np.zeros(len(metric_valid), dtype=int)
            for i, (m, l, r) in enumerate(zip(metric_valid, left_valid, right_valid)):
                if m >= thr:
                    pred[i] = 1 if l > r else 2
                else:
                    pred[i] = 0

            acc = sum(pred == gt_valid) / len(gt_valid)
            accuracies.append(acc)

        ax4.plot(thresholds_to_test * 100, np.array(accuracies) * 100, 'b-', linewidth=2)
        ax4.axvline(x=threshold * 100, color='red', linestyle='--', linewidth=2,
                    label=f'Current ({threshold:.1%})')

        # 找最优阈值
        best_idx = np.argmax(accuracies)
        best_thr = thresholds_to_test[best_idx]
        best_acc = accuracies[best_idx]
        ax4.scatter([best_thr * 100], [best_acc * 100], color='green', s=100, zorder=5,
                    label=f'Best ({best_thr:.1%}, acc={best_acc:.1%})')

        ax4.set_xlabel('Threshold (%)', fontsize=11)
        ax4.set_ylabel('Accuracy (%)', fontsize=11)
        ax4.set_title('Threshold Sensitivity Analysis', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, 30)
        ax4.set_ylim(0, 105)
    else:
        ax4.text(0.5, 0.5, 'No Ground Truth Available', ha='center', va='center', fontsize=14)
        ax4.set_title('Threshold Sensitivity', fontsize=12, fontweight='bold')

    plt.suptitle(f'Threshold Analysis for ShrugNose (Current Threshold: {threshold:.2%})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()

    print(f"[OK] Threshold analysis plot saved to: {output_path}")


def print_analysis_report(analysis: Dict[str, Any]) -> None:
    """打印分析报告"""
    print("\n" + "=" * 60)
    print("阈值分析报告")
    print("=" * 60)

    print(f"\n指标: {analysis['metric_name']}")
    print(f"阈值: {analysis['threshold']:.2%}")
    print(f"总样本数: {analysis['total_samples']}")
    print(f"有GT标注的样本数: {analysis['samples_with_gt']}")

    print(f"\n--- 指标分布 ---")
    dist = analysis['distribution']
    print(f"  均值: {dist['mean']:.4f}")
    print(f"  标准差: {dist['std']:.4f}")
    print(f"  最小值: {dist['min']:.4f}")
    print(f"  最大值: {dist['max']:.4f}")
    print(f"  中位数: {dist['median']:.4f}")
    print(f"  25%分位: {dist['q25']:.4f}")
    print(f"  75%分位: {dist['q75']:.4f}")

    print(f"\n--- 阈值划分效果 ---")
    te = analysis['threshold_effect']
    print(f"  高于阈值: {te['n_above']} ({te['pct_above']:.1%})")
    print(f"  低于阈值: {te['n_below']} ({te['pct_below']:.1%})")

    print(f"\n--- 预测分布 ---")
    pred = analysis['predictions']
    total = pred['normal'] + pred['left_palsy'] + pred['right_palsy']
    print(f"  正常: {pred['normal']} ({pred['normal'] / total * 100:.1f}%)")
    print(f"  左侧面瘫: {pred['left_palsy']} ({pred['left_palsy'] / total * 100:.1f}%)")
    print(f"  右侧面瘫: {pred['right_palsy']} ({pred['right_palsy'] / total * 100:.1f}%)")

    if analysis['accuracy'] is not None:
        print(f"\n--- 准确率 (基于GT) ---")
        print(f"  整体准确率: {analysis['accuracy']:.2%}")

        if analysis['confusion_matrix']:
            print(f"\n  混淆矩阵 (行=GT, 列=Pred):")
            print(f"           Pred_N  Pred_L  Pred_R")
            labels = ['GT_N', 'GT_L', 'GT_R']
            cm = analysis['confusion_matrix']
            for i, label in enumerate(labels):
                print(f"  {label}:   {cm[i][0]:5d}   {cm[i][1]:5d}   {cm[i][2]:5d}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="阈值分析工具")
    parser.add_argument("--data_dir", type=str, required=True, help="数据输出目录")
    parser.add_argument("--action", type=str, default="ShrugNose", help="动作名称")
    parser.add_argument("--threshold", type=float, default=None, help="阈值 (默认使用配置文件)")
    parser.add_argument("--output", type=str, default="./threshold_analysis.png", help="输出图片路径")

    args = parser.parse_args()

    # 加载阈值
    if args.threshold is None:
        from thresholds import THR
        threshold = THR.SHRUG_NOSE_STATIC_ASYMMETRY
    else:
        threshold = args.threshold

    # 加载数据
    data_dir = Path(args.data_dir)
    print(f"加载数据: {data_dir}")

    results = load_session_results(data_dir, args.action)
    print(f"加载了 {len(results)} 个session的结果")

    if len(results) == 0:
        print("错误: 没有找到数据")
        return

    # 提取指标
    data = extract_shrug_nose_metrics(results)

    # 分析
    analysis = analyze_threshold_performance(data, threshold, "asymmetry")

    # 打印报告
    print_analysis_report(analysis)

    # 绘图
    output_path = Path(args.output)
    plot_threshold_analysis(data, threshold, output_path, "asymmetry")

    # 保存分析结果JSON
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"[OK] Analysis JSON saved to: {json_path}")


if __name__ == "__main__":
    main()