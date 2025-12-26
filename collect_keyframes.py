# -*- coding: utf-8 -*-
"""
collect_keyframes.py (增强版 v2)

批量收集 clinical_grading 下所有动作的 peak_raw / peak_indicators 图片，
并统计每个动作的面瘫侧别预测准确性

功能：
1. 按动作归档关键帧图片到 clinical_grading_debug
2. 收集每个动作的面瘫侧别预测结果
3. 与 ground truth 对比计算准确率
4. 生成汇总 CSV 报告（按动作分开，真值预测挨着）

运行方式：PyCharm 直接点运行即可
"""

import shutil
import json
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple


# =============================================================================
# 配置
# =============================================================================
SRC_ROOT = Path("/Users/cuijinglei/Documents/facialPalsy/HGFA/clinical_grading")
DST_ROOT = Path("/Users/cuijinglei/Documents/facialPalsy/HGFA/clinical_grading_debug")

# 11个标准动作
ACTIONS = [
    "NeutralFace",
    "SpontaneousEyeBlink",
    "VoluntaryEyeBlink",
    "CloseEyeSoftly",
    "CloseEyeHardly",
    "RaiseEyebrow",
    "Smile",
    "ShrugNose",
    "ShowTeeth",
    "BlowCheek",
    "LipPucker",
]

# 面瘫侧别映射
PALSY_SIDE_MAP = {
    0: "Sym",
    1: "L",
    2: "R",
}


# =============================================================================
# 辅助函数
# =============================================================================
def ensure_dirs():
    """创建输出目录"""
    DST_ROOT.mkdir(parents=True, exist_ok=True)
    for a in ACTIONS:
        (DST_ROOT / a).mkdir(parents=True, exist_ok=True)


def find_exam_dirs(src_root: Path):
    """以"包含 report.html 的目录"为一次检查(exam)的根目录"""
    for report in src_root.rglob("report.html"):
        yield report.parent


def load_ground_truth(exam_dir: Path) -> Dict[str, Any]:
    """从 summary.json 加载 ground truth"""
    summary_path = exam_dir / "summary.json"
    if not summary_path.exists():
        return {}

    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        gt = summary.get("ground_truth", {})
        return {
            "has_palsy": gt.get("has_palsy", None),
            "palsy_side": gt.get("palsy_side", None),
            "hb_grade": gt.get("hb_grade", None),
        }
    except Exception as e:
        print(f"[WARN] Failed to load summary.json: {exam_dir} - {e}")
        return {}


def load_action_palsy_prediction(action_dir: Path) -> Dict[str, Any]:
    """从动作的 indicators.json 加载面瘫侧别预测"""
    json_path = action_dir / "indicators.json"
    if not json_path.exists():
        return {}

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        action_specific = data.get("action_specific", {})
        palsy_detection = action_specific.get("palsy_detection", {})

        return {
            "palsy_side": palsy_detection.get("palsy_side", None),
            "confidence": palsy_detection.get("confidence", None),
            "interpretation": palsy_detection.get("interpretation", None),
            "method": palsy_detection.get("method", None),
        }
    except Exception as e:
        print(f"[WARN] Failed to load indicators.json: {action_dir} - {e}")
        return {}


def evaluate_prediction(gt_side: Optional[int], pred_side: Optional[int]) -> str:
    """评估预测是否正确"""
    if gt_side is None:
        return "miss_gt"
    if pred_side is None:
        return "miss_pred"

    if gt_side == 0 and pred_side == 0:
        return "both_sym"
    if gt_side == 0 and pred_side != 0:
        return "FP"  # False Positive
    if gt_side != 0 and pred_side == 0:
        return "FN"  # False Negative
    if gt_side == pred_side:
        return "OK"  # Correct
    else:
        return "WRONG"  # Wrong side


# =============================================================================
# 核心收集函数
# =============================================================================
def collect_one_exam(exam_dir: Path) -> Tuple[int, int, List[Dict[str, Any]]]:
    """处理单个检查目录"""
    exam_id = exam_dir.name

    ground_truth = load_ground_truth(exam_dir)
    gt_side = ground_truth.get("palsy_side")
    gt_has_palsy = ground_truth.get("has_palsy")
    gt_hb = ground_truth.get("hb_grade")

    copied = 0
    skipped = 0
    palsy_records = []

    exts = ("jpg", "jpeg", "png", "webp")
    targets = ("peak_raw", "peak_indicators")

    for action_dir in exam_dir.iterdir():
        if not action_dir.is_dir():
            continue

        action_name = action_dir.name
        dst_action_dir = DST_ROOT / action_name
        dst_action_dir.mkdir(parents=True, exist_ok=True)

        # 复制图片
        for base in targets:
            src_file = None
            for ext in exts:
                f = action_dir / f"{base}.{ext}"
                if f.exists():
                    src_file = f
                    break

            if src_file is None:
                continue

            dst_name = f"{exam_id}_{action_name}_{base}{src_file.suffix.lower()}"
            dst_path = dst_action_dir / dst_name

            try:
                shutil.copy2(src_file, dst_path)
                copied += 1
            except Exception as e:
                print(f"[ERROR] copy failed: {src_file} -> {dst_path} | {e}")
                skipped += 1

        # 加载面瘫预测并记录
        prediction = load_action_palsy_prediction(action_dir)
        pred_side = prediction.get("palsy_side")
        eval_result = evaluate_prediction(gt_side, pred_side)

        record = {
            "exam_id": exam_id,
            "action": action_name,
            "hb": gt_hb,
            "gt": gt_side,
            "gt_text": PALSY_SIDE_MAP.get(gt_side, "?") if gt_side is not None else "?",
            "pred": pred_side,
            "pred_text": PALSY_SIDE_MAP.get(pred_side, "?") if pred_side is not None else "?",
            "conf": prediction.get("confidence"),
            "result": eval_result,
            "method": prediction.get("method"),
            "interpretation": prediction.get("interpretation"),
        }
        palsy_records.append(record)

    return copied, skipped, palsy_records


def compute_statistics(all_records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """计算每个动作的统计信息"""
    stats = defaultdict(lambda: {
        "total": 0, "OK": 0, "WRONG": 0, "FP": 0, "FN": 0,
        "both_sym": 0, "miss_gt": 0, "miss_pred": 0,
    })

    for record in all_records:
        action = record["action"]
        eval_result = record["result"]
        stats[action]["total"] += 1
        if eval_result in stats[action]:
            stats[action][eval_result] += 1

    # 计算准确率
    for action, s in stats.items():
        valid = s["total"] - s["miss_gt"] - s["miss_pred"]
        palsy_cases = s["OK"] + s["WRONG"] + s["FN"]

        if palsy_cases > 0:
            s["side_accuracy"] = s["OK"] / palsy_cases
        else:
            s["side_accuracy"] = None

        if valid > 0:
            s["overall_accuracy"] = (s["OK"] + s["both_sym"]) / valid
        else:
            s["overall_accuracy"] = None

        s["valid_samples"] = valid
        s["palsy_cases"] = palsy_cases

    return dict(stats)


def save_by_action_csv(records: List[Dict[str, Any]], output_dir: Path):
    """按动作分别保存CSV（真值和预测挨着，方便检查）"""
    by_action = defaultdict(list)
    for r in records:
        by_action[r["action"]].append(r)

    # 字段顺序：exam_id, hb, gt, pred, result, conf, method
    fieldnames = ["exam_id", "hb", "gt", "gt_text", "pred", "pred_text", "result", "conf", "method"]

    for action in ACTIONS:
        if action not in by_action:
            continue

        action_records = by_action[action]
        # 按 exam_id 排序
        action_records.sort(key=lambda x: x["exam_id"])

        output_path = output_dir / f"palsy_{action}.csv"
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(action_records)

    print(f"[OK] Per-action CSV files saved to: {output_dir}")


def save_all_records_csv(records: List[Dict[str, Any]], output_path: Path):
    """保存所有记录到单个CSV（按动作和exam_id排序）"""
    # 按动作优先，exam_id其次排序
    action_order = {a: i for i, a in enumerate(ACTIONS)}
    sorted_records = sorted(records, key=lambda x: (action_order.get(x["action"], 99), x["exam_id"]))

    fieldnames = ["action", "exam_id", "hb", "gt", "gt_text", "pred", "pred_text", "result", "conf", "method"]

    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(sorted_records)

    print(f"[OK] All records saved: {output_path}")


def save_summary_csv(stats: Dict[str, Dict[str, Any]], output_path: Path):
    """保存动作级别的汇总统计 CSV"""
    fieldnames = [
        "action", "total", "valid", "palsy_n",
        "OK", "WRONG", "FP", "FN", "both_sym", "miss_gt", "miss_pred",
        "side_acc", "overall_acc"
    ]

    rows = []
    for action in ACTIONS:
        if action in stats:
            s = stats[action]
            row = {
                "action": action,
                "total": s["total"],
                "valid": s["valid_samples"],
                "palsy_n": s["palsy_cases"],
                "OK": s["OK"],
                "WRONG": s["WRONG"],
                "FP": s["FP"],
                "FN": s["FN"],
                "both_sym": s["both_sym"],
                "miss_gt": s["miss_gt"],
                "miss_pred": s["miss_pred"],
                "side_acc": f"{s['side_accuracy']:.1%}" if s['side_accuracy'] is not None else "N/A",
                "overall_acc": f"{s['overall_accuracy']:.1%}" if s['overall_accuracy'] is not None else "N/A",
            }
            rows.append(row)

    # 计算总计
    total_ok = sum(stats[a]["OK"] for a in stats)
    total_wrong = sum(stats[a]["WRONG"] for a in stats)
    total_fn = sum(stats[a]["FN"] for a in stats)
    total_fp = sum(stats[a]["FP"] for a in stats)
    total_both = sum(stats[a]["both_sym"] for a in stats)
    total_valid = sum(stats[a]["valid_samples"] for a in stats)
    total_palsy = total_ok + total_wrong + total_fn

    total_stats = {
        "action": "TOTAL",
        "total": sum(stats[a]["total"] for a in stats),
        "valid": total_valid,
        "palsy_n": total_palsy,
        "OK": total_ok,
        "WRONG": total_wrong,
        "FP": total_fp,
        "FN": total_fn,
        "both_sym": total_both,
        "miss_gt": sum(stats[a]["miss_gt"] for a in stats),
        "miss_pred": sum(stats[a]["miss_pred"] for a in stats),
        "side_acc": f"{total_ok / total_palsy:.1%}" if total_palsy > 0 else "N/A",
        "overall_acc": f"{(total_ok + total_both) / total_valid:.1%}" if total_valid > 0 else "N/A",
    }
    rows.append(total_stats)

    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Summary saved: {output_path}")


def save_confusion_matrix(records: List[Dict[str, Any]], output_path: Path):
    """保存混淆矩阵统计"""
    by_action = defaultdict(list)
    for r in records:
        by_action[r["action"]].append(r)

    lines = []
    lines.append("=" * 70)
    lines.append("面瘫侧别预测 - 混淆矩阵统计")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    for action in ACTIONS:
        if action not in by_action:
            continue

        action_records = by_action[action]
        matrix = defaultdict(lambda: defaultdict(int))

        for r in action_records:
            gt = r["gt"] if r["gt"] is not None else -1
            pred = r["pred"] if r["pred"] is not None else -1
            matrix[gt][pred] += 1

        # 计算准确率
        ok = sum(1 for r in action_records if r["result"] == "OK")
        wrong = sum(1 for r in action_records if r["result"] == "WRONG")
        palsy_n = ok + wrong + sum(1 for r in action_records if r["result"] == "FN")
        acc = f"{ok/palsy_n:.1%}" if palsy_n > 0 else "N/A"

        lines.append(f"\n{'─' * 50}")
        lines.append(f"【{action}】 样本={len(action_records)}, 面瘫样本={palsy_n}, 侧别准确率={acc}")
        lines.append("─" * 50)
        lines.append(f"{'GT \\ Pred':>10} | {'Sym':>5} | {'Left':>5} | {'Right':>5} | {'N/A':>5}")
        lines.append("-" * 45)

        for gt_val, gt_name in [(-1, "N/A"), (0, "Sym"), (1, "Left"), (2, "Right")]:
            row = matrix.get(gt_val, {})
            vals = [row.get(0, 0), row.get(1, 0), row.get(2, 0), row.get(-1, 0)]
            lines.append(f"{gt_name:>10} | {vals[0]:>5} | {vals[1]:>5} | {vals[2]:>5} | {vals[3]:>5}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"[OK] Confusion matrix saved: {output_path}")


# =============================================================================
# 主函数
# =============================================================================
def main():
    if not SRC_ROOT.exists():
        raise FileNotFoundError(f"SRC_ROOT 不存在: {SRC_ROOT}")

    ensure_dirs()

    total_exam = 0
    total_copied = 0
    total_skipped = 0
    all_palsy_records = []

    print("=" * 60)
    print("开始收集关键帧和面瘫预测统计")
    print("=" * 60)

    for exam_dir in find_exam_dirs(SRC_ROOT):
        total_exam += 1
        c, s, records = collect_one_exam(exam_dir)
        total_copied += c
        total_skipped += s
        all_palsy_records.extend(records)

        if total_exam % 20 == 0:
            print(f"[INFO] exams={total_exam}, records={len(all_palsy_records)}")

    print(f"\n图片收集: exams={total_exam}, copied={total_copied}, skipped={total_skipped}")

    # 计算统计
    print("\n" + "=" * 60)
    print("计算面瘫侧别预测统计...")
    print("=" * 60)

    stats = compute_statistics(all_palsy_records)

    # 打印统计摘要
    print(f"\n{'Action':<20} {'Total':>5} {'OK':>5} {'WRONG':>6} {'FN':>4} {'SideAcc':>9}")
    print("-" * 55)

    for action in ACTIONS:
        if action in stats:
            s = stats[action]
            side_acc = f"{s['side_accuracy']:.1%}" if s['side_accuracy'] is not None else "N/A"
            print(f"{action:<20} {s['total']:>5} {s['OK']:>5} {s['WRONG']:>6} {s['FN']:>4} {side_acc:>9}")

    # 保存文件（固定名称，覆盖模式）
    save_by_action_csv(all_palsy_records, DST_ROOT)
    save_all_records_csv(all_palsy_records, DST_ROOT / "palsy_all_records.csv")
    save_summary_csv(stats, DST_ROOT / "palsy_summary.csv")
    save_confusion_matrix(all_palsy_records, DST_ROOT / "palsy_confusion.txt")

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)
    print(f"输出目录: {DST_ROOT}")
    print(f"  - palsy_<Action>.csv  : 每个动作的详细预测")
    print(f"  - palsy_all_records.csv : 所有记录汇总")
    print(f"  - palsy_summary.csv   : 统计摘要")
    print(f"  - palsy_confusion.txt : 混淆矩阵")


if __name__ == "__main__":
    main()