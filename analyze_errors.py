#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 error_cases.json 中的错误模式和阈值范围
"""

import json
from collections import defaultdict, Counter
from typing import Dict, List, Any
import numpy as np


ERROR_PATH = "/Users/cuijinglei/Documents/facial_palsy/HGFA/clinical_grading_debug/error_cases/error_cases.json"
OUT_PATH = "/Users/cuijinglei/Documents/facial_palsy/HGFA/clinical_grading_debug/error_cases/error_analysis_summary.json"

def load_error_cases(filepath: str) -> Dict[str, List[Dict]]:
    """加载错误案例"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def analyze_by_action(cases: List[Dict]) -> Dict[str, List[Dict]]:
    """按动作分类错误案例"""
    by_action = defaultdict(list)
    for case in cases:
        action = case.get('action', 'Unknown')
        by_action[action].append(case)
    return dict(by_action)


def analyze_error_types(cases: List[Dict]) -> Dict[str, int]:
    """统计错误类型"""
    error_types = Counter()
    for case in cases:
        result = case.get('result', 'Unknown')
        error_types[result] += 1
    return dict(error_types)


def extract_metrics_stats(cases: List[Dict], action: str) -> Dict[str, Dict]:
    """提取各动作的关键指标统计"""
    stats = defaultdict(list)

    for case in cases:
        dm = case.get('detailed_metrics', {})
        if not dm:
            continue

        for key, value in dm.items():
            # 跳过布尔值和非数值类型
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and value is not None:
                try:
                    if not np.isnan(float(value)) and not np.isinf(float(value)):
                        stats[key].append(float(value))
                except (ValueError, TypeError):
                    continue

    result = {}
    for key, values in stats.items():
        if values:
            result[key] = {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values),
                'p10': np.percentile(values, 10),
                'p90': np.percentile(values, 90),
            }
    return result


def analyze_palsy_prediction_errors(cases: List[Dict]) -> Dict:
    """分析面瘫侧别预测错误"""
    wrong_cases = [c for c in cases if c.get('result') == 'WRONG']
    fp_cases = [c for c in cases if c.get('result') == 'FP']
    fn_cases = [c for c in cases if c.get('result') == 'FN']

    # 按HB等级分组
    hb_distribution = Counter()
    for case in cases:
        hb = case.get('hb', 0)
        hb_distribution[hb] += 1

    return {
        'total_errors': len(cases),
        'wrong': len(wrong_cases),
        'false_positive': len(fp_cases),
        'false_negative': len(fn_cases),
        'hb_distribution': dict(hb_distribution),
    }


def main():
    print("=" * 80)
    print("Error Cases Analysis Report")
    print("=" * 80)

    data = load_error_cases(ERROR_PATH)

    all_cases = data.get('WRONG', [])
    print(f"\n总错误案例数: {len(all_cases)}")

    # 按动作分类
    by_action = analyze_by_action(all_cases)

    print("\n" + "=" * 80)
    print("按动作分类的错误统计")
    print("=" * 80)

    action_summary = []
    for action in sorted(by_action.keys()):
        cases = by_action[action]
        error_analysis = analyze_palsy_prediction_errors(cases)
        action_summary.append({
            'action': action,
            'total': len(cases),
            'wrong': error_analysis['wrong'],
            'fp': error_analysis['false_positive'],
            'fn': error_analysis['false_negative'],
            'hb_dist': error_analysis['hb_distribution']
        })

    # 按错误数排序
    action_summary.sort(key=lambda x: x['total'], reverse=True)

    for item in action_summary:
        print(f"\n{'=' * 60}")
        print(f"动作: {item['action']}")
        print(f"  总错误数: {item['total']}")
        print(f"  WRONG (侧别预测错误): {item['wrong']}")
        print(f"  FP (假阳性): {item['fp']}")
        print(f"  FN (假阴性): {item['fn']}")
        print(f"  HB分布: {item['hb_dist']}")

    # 详细分析每个动作的指标
    print("\n" + "=" * 80)
    print("各动作关键指标统计 (用于阈值调整参考)")
    print("=" * 80)

    for action in sorted(by_action.keys()):
        cases = by_action[action]
        if not cases:
            continue

        print(f"\n{'=' * 60}")
        print(f"动作: {action} (共{len(cases)}个错误案例)")
        print(f"{'=' * 60}")

        stats = extract_metrics_stats(cases, action)

        # 按指标名排序显示
        for metric in sorted(stats.keys()):
            s = stats[metric]
            if s['count'] >= 3:  # 只显示有足够样本的指标
                print(f"\n  {metric}:")
                print(f"    样本数: {s['count']}")
                print(f"    范围: [{s['min']:.4f}, {s['max']:.4f}]")
                print(f"    均值: {s['mean']:.4f} ± {s['std']:.4f}")
                print(f"    中位数: {s['median']:.4f}")
                print(f"    P10-P90: [{s['p10']:.4f}, {s['p90']:.4f}]")

    # 分析各类错误的模式
    print("\n" + "=" * 80)
    print("错误模式详细分析")
    print("=" * 80)

    # 1. 侧别预测错误 (WRONG)
    wrong_cases = [c for c in all_cases if c.get('result') == 'WRONG']
    print(f"\n侧别预测错误 (WRONG): {len(wrong_cases)}例")

    # 分析WRONG案例的预测vs真值
    pred_vs_gt = Counter()
    for case in wrong_cases:
        gt = case.get('gt_text', '?')
        pred = case.get('pred_text', '?')
        pred_vs_gt[f"{gt}->{pred}"] += 1
    print("  预测错误模式:")
    for pattern, count in sorted(pred_vs_gt.items(), key=lambda x: -x[1]):
        print(f"    {pattern}: {count}例")

    # 2. 假阳性 (FP) - 正常人被预测为面瘫
    fp_cases = [c for c in all_cases if c.get('result') == 'FP']
    print(f"\n假阳性 (FP): {len(fp_cases)}例")
    if fp_cases:
        fp_actions = Counter(c.get('action') for c in fp_cases)
        print("  按动作分布:")
        for action, count in sorted(fp_actions.items(), key=lambda x: -x[1]):
            print(f"    {action}: {count}例")

    # 3. 假阴性 (FN) - 面瘫被漏诊
    fn_cases = [c for c in all_cases if c.get('result') == 'FN']
    print(f"\n假阴性 (FN): {len(fn_cases)}例")
    if fn_cases:
        fn_actions = Counter(c.get('action') for c in fn_cases)
        print("  按动作分布:")
        for action, count in sorted(fn_actions.items(), key=lambda x: -x[1]):
            print(f"    {action}: {count}例")

    # 输出JSON格式的统计摘要
    summary = {
        'total_errors': len(all_cases),
        'by_result': {
            'WRONG': len(wrong_cases),
            'FP': len(fp_cases),
            'FN': len(fn_cases),
        },
        'by_action': {action: len(cases) for action, cases in by_action.items()},
        'metrics_stats': {}
    }

    for action in by_action:
        stats = extract_metrics_stats(by_action[action], action)
        summary['metrics_stats'][action] = stats

    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n\n统计摘要已保存到: {OUT_PATH}")


if __name__ == '__main__':
    main()