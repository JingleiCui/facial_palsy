# -*- coding: utf-8 -*-
"""
collect_keyframes.py

批量收集 clinical_grading 下所有动作的 peak_raw / peak_indicators 图片，
并统计每个动作的面瘫侧别预测准确性

增强功能：
1. 提取每个动作的详细判断指标
2. 按动作类型输出不同的metrics列
3. 生成错误案例分析报告
4. 支持阈值边界案例识别

"""

import json
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional, Tuple
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

# =============================================================================
# 配置
# =============================================================================
SRC_ROOT = Path("/Users/cuijinglei/Documents/facial_palsy/HGFA/clinical_grading")
DST_ROOT = Path("/Users/cuijinglei/Documents/facial_palsy/HGFA/clinical_grading_debug")

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
    # 创建错误案例目录
    (DST_ROOT / "error_cases").mkdir(parents=True, exist_ok=True)


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
            "sunnybrook_score": gt.get("sunnybrook_score", None),  # 新增
        }
    except Exception as e:
        print(f"[WARN] Failed to load summary.json: {exam_dir} - {e}")
        return {}


def load_session_diagnosis(exam_dir: Path) -> Dict[str, Any]:
    """从 summary.json 加载 Session 级诊断预测结果"""
    summary_path = exam_dir / "summary.json"
    if not summary_path.exists():
        return {}

    try:
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)

        diagnosis = summary.get("diagnosis", {})
        sunnybrook_detail = summary.get("sunnybrook", None)
        if not diagnosis:
            return {}

        return {
            "has_palsy": diagnosis.get("has_palsy", None),
            "palsy_side": diagnosis.get("palsy_side", None),
            "hb_grade": diagnosis.get("hb_grade", None),
            "sunnybrook_score": diagnosis.get("sunnybrook_score", None),
            "confidence": diagnosis.get("confidence", None),
            "palsy_side_confidence": diagnosis.get("palsy_side_confidence", None),
            "resting_score": diagnosis.get("resting_score", None),
            "voluntary_score": diagnosis.get("voluntary_score", None),
            "synkinesis_score": diagnosis.get("synkinesis_score", None),
            "hb_evidence": diagnosis.get("hb_evidence", None),
            "votes": diagnosis.get("votes", None),
            "top_evidence": diagnosis.get("top_evidence", None),
            "consistency_checks": diagnosis.get("consistency_checks", None),
            "adjustments_made": diagnosis.get("adjustments_made", None),
            "sunnybrook_detail": sunnybrook_detail,
        }
    except Exception as e:
        print(f"[WARN] Failed to load diagnosis from summary.json: {exam_dir} - {e}")
        return {}


def flatten_evidence(evidence: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """将嵌套的evidence字典展平为单层"""
    result = {}
    for key, value in evidence.items():
        full_key = f"{prefix}{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten_evidence(value, f"{full_key}_"))
        elif isinstance(value, (int, float, str, bool, type(None))):
            result[full_key] = value
        else:
            result[full_key] = str(value)
    return result


def load_action_palsy_prediction(action_dir: Path) -> Dict[str, Any]:
    """从动作的 indicators.json 加载面瘫侧别预测和详细指标"""
    json_path = action_dir / "indicators.json"
    if not json_path.exists():
        return {}

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        action_specific = data.get("action_specific", {})
        palsy_detection = action_specific.get("palsy_detection", {})

        # 提取基本信息
        result = {
            "palsy_side": palsy_detection.get("palsy_side", None),
            "confidence": palsy_detection.get("confidence", None),
            "interpretation": palsy_detection.get("interpretation", None),
            "method": palsy_detection.get("method", None),
        }

        # 提取严重度分数 (医生标注标准: 1=正常, 5=面瘫)
        severity_score = action_specific.get("severity_score", None)
        severity_desc = action_specific.get("severity_desc", "")
        result["severity_score"] = severity_score
        result["severity_desc"] = severity_desc

        # 优先使用 evidence_used（真正用于判侧别的关键指标）
        evidence_used = palsy_detection.get("evidence_used", None)
        evidence_dump = palsy_detection.get("evidence_dump", None)

        if evidence_used is not None:
            detailed_metrics = flatten_evidence(evidence_used)
            result["detailed_metrics"] = detailed_metrics
            result["metrics_source"] = "evidence_used"
        else:
            # 兼容旧版：没有 evidence_used 时退回到 evidence
            evidence = palsy_detection.get("evidence", {})
            detailed_metrics = flatten_evidence(evidence)
            result["detailed_metrics"] = detailed_metrics
            result["metrics_source"] = "evidence"

        # 可选：保留 dump（用于你想深挖时），但后续 error_analysis 不会用它
        if evidence_dump is not None:
            result["debug_dump"] = evidence_dump

        return result
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


def format_metrics_summary(action: str, metrics: Dict[str, Any]) -> str:
    """生成人类可读的关键指标摘要"""
    method = metrics.get("method", "unknown")

    if action in ["SpontaneousEyeBlink", "VoluntaryEyeBlink"]:
        if method == "persistent_asymmetry":
            lw = metrics.get("left_worse_frame_ratio", 0)
            rw = metrics.get("right_worse_frame_ratio", 0)
            return f"持续不对称: L差帧={lw:.0%} R差帧={rw:.0%}"
        else:
            lc = metrics.get("left_closure_pct", 0)
            rc = metrics.get("right_closure_pct", 0)
            asym = metrics.get("asymmetry_ratio", 0)
            return f"闭合: L={lc:.1f}% R={rc:.1f}% asym={asym:.2f}"

    elif action in ["CloseEyeSoftly", "CloseEyeHardly"]:
        le = metrics.get("left_ear", 0)
        re = metrics.get("right_ear", 0)
        lc = metrics.get("left_closure_pct", 0)
        rc = metrics.get("right_closure_pct", 0)
        asym = metrics.get("asymmetry_ratio", 0)
        return f"EAR: L={le:.4f} R={re:.4f} | 闭合: L={lc:.1f}% R={rc:.1f}% asym={asym:.2f}"

    elif action == "RaiseEyebrow":
        lc = metrics.get("left_change", 0)
        rc = metrics.get("right_change", 0)
        asym = metrics.get("asymmetry_ratio", 0)
        bed_ratio = metrics.get("brow_eye_distance_ratio", 1.0)
        return f"变化: L={lc:+.1f}px R={rc:+.1f}px asym={asym:.2f} ratio={bed_ratio:.3f}"

    elif action in ["Smile", "ShowTeeth", "BlowCheek", "LipPucker"]:
        if action in ["ShowTeeth", "LipPucker"]:
            offset = metrics.get("current_offset", 0)
            offset_norm = metrics.get("offset_norm", 0)
            return f"嘴唇偏移: {offset:+.1f}px ({offset_norm:.1%})"
        elif method == "lip_midline_offset_change":
            current = metrics.get("current_offset", 0)
            baseline = metrics.get("baseline_offset", 0)
            change = metrics.get("offset_change", 0)
            change_norm = metrics.get("offset_change_norm", 0)
            return f"中线偏移: 当前={current:+.1f}px 基线={baseline:+.1f}px 变化={change:+.1f}px ({change_norm:.1%})"
        elif method == "lip_midline_symmetry":
            lm = metrics.get("left_to_midline", 0)
            rm = metrics.get("right_to_midline", 0)
            offset = metrics.get("lip_offset", 0)
            asym = metrics.get("asymmetry_ratio", 0)
            return f"中线: L={lm:.1f}px R={rm:.1f}px 偏移={offset:+.1f}px asym={asym:.2f}"
        elif method == "eye_line_excursion":
            left_red = metrics.get("left_reduction", 0)
            right_red = metrics.get("right_reduction", 0)
            red_asym = metrics.get("reduction_asymmetry", 0)
            return f"嘴角上提: L减小{left_red:.1f}px R减小{right_red:.1f}px asym={red_asym:.2f}"
        else:
            aoe = metrics.get("AOE_right", 0)
            bof = metrics.get("BOF_left", 0)
            diff = metrics.get("angle_diff", 0)
            return f"口角: AOE={aoe:+.1f}° BOF={bof:+.1f}° diff={diff:.1f}°"

    elif action == "ShrugNose":
        ld = metrics.get("left_ala_to_eye_line", 0)
        rd = metrics.get("right_ala_to_eye_line", 0)
        tilt = metrics.get("tilt_angle", 0)
        asym = metrics.get("asymmetry_ratio", 0)
        return f"鼻翼距: L={ld:.1f}px R={rd:.1f}px 倾斜={tilt:+.1f}° asym={asym:.2f}"

    return f"method={method}"


# =============================================================================
# 核心收集函数
# =============================================================================
def collect_one_exam(exam_dir: Path) -> Tuple[int, int, List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """处理单个检查目录

    Returns:
        (copied, skipped, palsy_records, session_record)
        - session_record: Session级诊断记录，用于统计整体级别预测准确率
    """
    exam_id = exam_dir.name

    ground_truth = load_ground_truth(exam_dir)
    gt_side = ground_truth.get("palsy_side")
    gt_has_palsy = ground_truth.get("has_palsy")
    gt_hb = ground_truth.get("hb_grade")
    gt_sunnybrook = ground_truth.get("sunnybrook_score")

    # 加载Session级诊断预测
    session_pred = load_session_diagnosis(exam_dir)

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

        # 复制关键帧图片（peak_raw / peak_indicators）
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

        # 复制曲线图
        curve_files = sorted(action_dir.glob("*curve*.png"))
        for curve_file in curve_files:
            dst_name = f"{exam_id}_{action_name}_{curve_file.name}"
            dst_path = dst_action_dir / dst_name
            try:
                shutil.copy2(curve_file, dst_path)
                copied += 1
            except Exception as e:
                print(f"[ERROR] copy curve failed: {curve_file} -> {dst_path} | {e}")
                skipped += 1

        # 加载面瘫预测并记录
        prediction = load_action_palsy_prediction(action_dir)
        pred_side = prediction.get("palsy_side")
        detailed_metrics = prediction.get("detailed_metrics", {})
        eval_result = evaluate_prediction(gt_side, pred_side)

        # 生成指标摘要
        metrics_summary = format_metrics_summary(action_name, detailed_metrics)

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
            "metrics_summary": metrics_summary,
            "detailed_metrics": detailed_metrics,
            "severity_score": prediction.get("severity_score"),
            "severity_desc": prediction.get("severity_desc", ""),
        }
        palsy_records.append(record)

    # 构建Session级诊断记录
    session_record = None
    if session_pred:
        session_record = {
            "exam_id": exam_id,
            # Ground Truth
            "gt_has_palsy": gt_has_palsy,
            "gt_palsy_side": gt_side,
            "gt_hb_grade": gt_hb,
            "gt_sunnybrook": gt_sunnybrook,
            # Prediction
            "pred_has_palsy": session_pred.get("has_palsy"),
            "pred_palsy_side": session_pred.get("palsy_side"),
            "pred_hb_grade": session_pred.get("hb_grade"),
            "pred_sunnybrook": session_pred.get("sunnybrook_score"),
            "pred_confidence": session_pred.get("confidence"),
            "pred_palsy_side_confidence": session_pred.get("palsy_side_confidence"),
            # Sunnybrook详细分数
            "pred_resting_score": session_pred.get("resting_score"),
            "pred_voluntary_score": session_pred.get("voluntary_score"),
            "pred_synkinesis_score": session_pred.get("synkinesis_score"),
            # add evidence
            "pred_hb_evidence": session_pred.get("hb_evidence"),
            "pred_votes": session_pred.get("votes"),
            "pred_top_evidence": session_pred.get("top_evidence"),
            "pred_consistency_checks": session_pred.get("consistency_checks"),
            "pred_adjustments_made": session_pred.get("adjustments_made"),
            "pred_sunnybrook_detail": session_pred.get("sunnybrook_detail"),
        }

    # === 按 GT HB 分组打包复制 11 动作图片（便于肉眼检查）===
    pred_hb = session_pred.get("hb_grade") if session_pred else None
    copy_exam_pack_by_gt_hb(
        exam_dir=exam_dir,
        gt_hb=gt_hb,
        pred_hb=pred_hb,
        dst_root=DST_ROOT,
        copy_indicators=True,
        copy_selection_curve=True,  # 如果你只想要11张图，把这里改 False
        copy_peak_raw=False
    )

    return copied, skipped, palsy_records, session_record


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

        # 严格准确率
        if palsy_cases > 0:
            s["side_accuracy"] = s["OK"] / palsy_cases
        else:
            s["side_accuracy"] = None

        # 宽松准确率
        if palsy_cases > 0:
            s["relaxed_accuracy"] = (s["OK"] + s["FN"]) / palsy_cases
        else:
            s["relaxed_accuracy"] = None

        s["valid_samples"] = valid
        s["palsy_cases"] = palsy_cases

    return stats


def compute_session_diagnosis_statistics(session_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    计算Session级诊断的准确率统计

    统计指标:
    - has_palsy: 是否面瘫的准确率
    - palsy_side: 面瘫侧别的准确率（严格/宽松）
    - hb_grade: HB分级的准确率（精确匹配/±1级）
    - sunnybrook: Sunnybrook评分的MAE和相关性
    """
    if not session_records:
        return {}

    stats = {
        "total": len(session_records),
        "has_palsy": {"correct": 0, "total": 0, "accuracy": None},
        "palsy_side": {
            "correct": 0, "wrong_side": 0, "fn": 0, "fp": 0,
            "total_palsy": 0, "strict_acc": None, "relaxed_acc": None
        },
        "hb_grade": {
            "exact_match": 0, "within_1": 0, "total": 0,
            "exact_acc": None, "within_1_acc": None, "mae": None
        },
        "sunnybrook": {
            "total": 0, "mae": None, "rmse": None, "correlation": None,
            "errors": []
        }
    }

    # 用于计算Sunnybrook统计的列表
    sb_gt_list = []
    sb_pred_list = []
    hb_errors = []

    for r in session_records:
        gt_has_palsy = r.get("gt_has_palsy")
        pred_has_palsy = r.get("pred_has_palsy")
        gt_side = r.get("gt_palsy_side")
        pred_side = r.get("pred_palsy_side")
        gt_hb = r.get("gt_hb_grade")
        pred_hb = r.get("pred_hb_grade")
        gt_sb = r.get("gt_sunnybrook")
        pred_sb = r.get("pred_sunnybrook")

        # === 1. has_palsy 准确率 ===
        if gt_has_palsy is not None and pred_has_palsy is not None:
            stats["has_palsy"]["total"] += 1
            # 转换为bool比较
            gt_bool = bool(gt_has_palsy)
            pred_bool = bool(pred_has_palsy)
            if gt_bool == pred_bool:
                stats["has_palsy"]["correct"] += 1

        # === 2. palsy_side 准确率 ===
        if gt_side is not None and pred_side is not None:
            if gt_side == 0 and pred_side == 0:
                # 都是对称，不计入面瘫侧别统计
                pass
            elif gt_side == 0 and pred_side != 0:
                stats["palsy_side"]["fp"] += 1  # False Positive
                stats["palsy_side"]["total_palsy"] += 1
            elif gt_side != 0 and pred_side == 0:
                stats["palsy_side"]["fn"] += 1  # False Negative
                stats["palsy_side"]["total_palsy"] += 1
            elif gt_side == pred_side:
                stats["palsy_side"]["correct"] += 1  # 正确
                stats["palsy_side"]["total_palsy"] += 1
            else:
                stats["palsy_side"]["wrong_side"] += 1  # 错误侧别
                stats["palsy_side"]["total_palsy"] += 1

        # === 3. HB Grade 准确率 ===
        if gt_hb is not None and pred_hb is not None:
            stats["hb_grade"]["total"] += 1
            hb_diff = abs(gt_hb - pred_hb)
            hb_errors.append(hb_diff)
            if hb_diff == 0:
                stats["hb_grade"]["exact_match"] += 1
            if hb_diff <= 1:
                stats["hb_grade"]["within_1"] += 1

        # === 4. Sunnybrook 评分误差 ===
        if gt_sb is not None and pred_sb is not None:
            stats["sunnybrook"]["total"] += 1
            sb_gt_list.append(gt_sb)
            sb_pred_list.append(pred_sb)
            stats["sunnybrook"]["errors"].append(pred_sb - gt_sb)

    # === 计算汇总指标 ===

    # has_palsy accuracy
    if stats["has_palsy"]["total"] > 0:
        stats["has_palsy"]["accuracy"] = stats["has_palsy"]["correct"] / stats["has_palsy"]["total"]

    # palsy_side accuracy
    total_palsy = stats["palsy_side"]["total_palsy"]
    if total_palsy > 0:
        stats["palsy_side"]["strict_acc"] = stats["palsy_side"]["correct"] / total_palsy
        # 宽松准确率：FN也算对（预测对称但实际有面瘫）
        stats["palsy_side"]["relaxed_acc"] = (stats["palsy_side"]["correct"] + stats["palsy_side"]["fn"]) / total_palsy

    # HB Grade accuracy
    if stats["hb_grade"]["total"] > 0:
        stats["hb_grade"]["exact_acc"] = stats["hb_grade"]["exact_match"] / stats["hb_grade"]["total"]
        stats["hb_grade"]["within_1_acc"] = stats["hb_grade"]["within_1"] / stats["hb_grade"]["total"]
        stats["hb_grade"]["mae"] = sum(hb_errors) / len(hb_errors)

    # Sunnybrook statistics
    if stats["sunnybrook"]["total"] > 0:
        import numpy as np
        sb_gt = np.array(sb_gt_list)
        sb_pred = np.array(sb_pred_list)
        errors = sb_pred - sb_gt

        stats["sunnybrook"]["mae"] = float(np.mean(np.abs(errors)))
        stats["sunnybrook"]["rmse"] = float(np.sqrt(np.mean(errors ** 2)))

        # Pearson相关系数
        if len(sb_gt) > 1 and np.std(sb_gt) > 0 and np.std(sb_pred) > 0:
            stats["sunnybrook"]["correlation"] = float(np.corrcoef(sb_gt, sb_pred)[0, 1])

        # 清理errors列表（只保留统计，不保存原始数据）
        stats["sunnybrook"]["errors"] = None

    return stats


# =============================================================================
# HB分级详细统计
# =============================================================================

def compute_hb_confusion_matrix(session_records: List[Dict[str, Any]]) -> np.ndarray:
    """计算HB分级的6x6混淆矩阵"""
    confusion = np.zeros((6, 6), dtype=int)

    for r in session_records:
        gt = r.get("gt_hb_grade")
        pred = r.get("pred_hb_grade")

        if gt is not None and pred is not None:
            gt, pred = int(gt), int(pred)
            if 1 <= gt <= 6 and 1 <= pred <= 6:
                confusion[gt - 1, pred - 1] += 1

    return confusion


def print_hb_detailed_statistics(session_records: List[Dict[str, Any]]):
    """打印HB分级详细统计"""
    confusion = compute_hb_confusion_matrix(session_records)
    total = confusion.sum()

    if total == 0:
        print("  无有效HB分级数据")
        return

    print(f"\n{'=' * 65}")
    print("HB分级详细统计")
    print(f"{'=' * 65}")

    # 混淆矩阵
    print("\n混淆矩阵 (行=GT, 列=Pred):")
    header = "GT\\Pred"
    for i in range(1, 7):
        header += f"  HB{i}"
    header += "  Total Recall"
    print(header)
    print("-" * 60)

    for i in range(6):
        row = f"HB{i + 1}   "
        for j in range(6):
            val = confusion[i, j]
            row += f"  {val:3d}" if val > 0 else "    -"

        row_total = confusion[i].sum()
        recall = confusion[i, i] / row_total if row_total > 0 else 0
        row += f"   {row_total:3d} {recall:5.1%}"
        print(row)

    print("-" * 60)

    # 总计行和精确率
    total_row = "Total "
    prec_row = "Prec  "
    for j in range(6):
        col_sum = confusion[:, j].sum()
        total_row += f"  {col_sum:3d}"
        prec = confusion[j, j] / col_sum if col_sum > 0 else 0
        prec_row += f" {prec:4.0%}"
    total_row += f"   {total:3d}"
    print(total_row)
    print(prec_row)

    # 每级详情
    print(f"\n{'Grade':<6}{'GT#':>5}{'Pred#':>6}{'Correct':>8}{'Precision':>10}{'Recall':>8}{'F1':>6}")
    print("-" * 50)

    for i in range(6):
        grade = i + 1
        gt_count = confusion[i, :].sum()
        pred_count = confusion[:, i].sum()
        correct = confusion[i, i]

        precision = correct / pred_count if pred_count > 0 else 0
        recall = correct / gt_count if gt_count > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"HB{grade:<4}{gt_count:>5}{pred_count:>6}{correct:>8}"
              f"{precision:>9.1%}{recall:>7.1%}{f1:>6.2f}")

    # 总体指标
    exact = int(np.trace(confusion))
    within_1 = sum(confusion[i, j] for i in range(6) for j in range(6) if abs(i - j) <= 1)

    # MAE
    mae_sum = 0
    count = 0
    for i in range(6):
        for j in range(6):
            mae_sum += confusion[i, j] * abs(i - j)
            count += confusion[i, j]
    mae = mae_sum / count if count > 0 else 0

    print(f"\n总体指标:")
    print(f"  精确匹配: {exact}/{total} = {exact / total:.1%}")
    print(f"  ±1级匹配: {within_1}/{total} = {within_1 / total:.1%}")
    print(f"  MAE: {mae:.2f} 级")

    # 主要误差来源
    errors = []
    for i in range(6):
        for j in range(6):
            if i != j and confusion[i, j] > 0:
                errors.append(((i + 1, j + 1), confusion[i, j]))
    errors.sort(key=lambda x: -x[1])

    if errors:
        print(f"\n主要误差来源 (GT→Pred: 次数):")
        for (gt, pred), count in errors[:8]:
            diff = pred - gt
            direction = "高估" if diff > 0 else "低估"
            print(f"  HB{gt}→HB{pred}: {count}次 ({direction}{abs(diff)}级)")


# =============================================================================
# Sunnybrook评分详细统计
# =============================================================================

def compute_sunnybrook_confusion_matrix(session_records: List[Dict[str, Any]]) -> np.ndarray:
    """计算Sunnybrook评分的5x5分段混淆矩阵"""
    bins = [(0, 20), (21, 40), (41, 60), (61, 80), (81, 100)]

    def get_bin(score):
        if score is None:
            return -1
        score = float(score)
        for i, (low, high) in enumerate(bins):
            if low <= score <= high:
                return i
        return 4 if score > 100 else 0

    confusion = np.zeros((5, 5), dtype=int)

    for r in session_records:
        gt = r.get("gt_sunnybrook")
        pred = r.get("pred_sunnybrook")

        if gt is not None and pred is not None:
            gt_bin = get_bin(gt)
            pred_bin = get_bin(pred)
            if gt_bin >= 0 and pred_bin >= 0:
                confusion[gt_bin, pred_bin] += 1

    return confusion


def print_sunnybrook_detailed_statistics(session_records: List[Dict[str, Any]]):
    """打印Sunnybrook评分详细统计"""
    labels = ["0-20", "21-40", "41-60", "61-80", "81-100"]
    names = ["重度", "中重", "中度", "轻度", "正常"]

    confusion = compute_sunnybrook_confusion_matrix(session_records)
    total = confusion.sum()

    if total == 0:
        print("  无有效Sunnybrook数据")
        return

    print(f"\n{'=' * 65}")
    print("Sunnybrook评分详细统计")
    print(f"{'=' * 65}")

    # 混淆矩阵
    print("\n分段混淆矩阵 (行=GT, 列=Pred):")
    header = f"{'GT\\Pred':<10}"
    for label in labels:
        header += f"{label:>7}"
    header += f"{'Total':>7}{'Acc':>6}"
    print(header)
    print("-" * 58)

    for i, (label, name) in enumerate(zip(labels, names)):
        row = f"{label:<6}{name:<4}"
        for j in range(5):
            val = confusion[i, j]
            row += f"{val:>7}" if val > 0 else f"{'':>6}-"

        row_total = confusion[i].sum()
        acc = confusion[i, i] / row_total if row_total > 0 else 0
        row += f"{row_total:>7}{acc:>5.0%}"
        print(row)

    print("-" * 58)
    total_row = f"{'Total':<10}"
    for j in range(5):
        total_row += f"{confusion[:, j].sum():>7}"
    total_row += f"{total:>7}"
    print(total_row)

    # 计算MAE、RMSE、相关性
    gt_list = []
    pred_list = []
    for r in session_records:
        gt = r.get("gt_sunnybrook")
        pred = r.get("pred_sunnybrook")
        if gt is not None and pred is not None:
            gt_list.append(float(gt))
            pred_list.append(float(pred))

    if gt_list:
        gt_arr = np.array(gt_list)
        pred_arr = np.array(pred_list)
        errors = pred_arr - gt_arr

        mae = float(np.mean(np.abs(errors)))
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        corr = float(np.corrcoef(gt_arr, pred_arr)[0, 1]) if len(gt_list) > 1 else 0
        mean_err = float(np.mean(errors))

        # 每段MAE
        print("\n每段MAE:")
        bins = [(0, 20), (21, 40), (41, 60), (61, 80), (81, 100)]
        for i, ((low, high), name) in enumerate(zip(bins, names)):
            mask = (gt_arr >= low) & (gt_arr <= high)
            if mask.sum() > 0:
                seg_mae = float(np.mean(np.abs(errors[mask])))
                print(f"  {labels[i]} ({name}): MAE={seg_mae:.1f}分 (n={mask.sum()})")

        print(f"\n总体指标:")
        print(f"  样本数: {len(gt_list)}")
        print(f"  MAE: {mae:.1f} 分")
        print(f"  RMSE: {rmse:.1f} 分")
        print(f"  Pearson相关: {corr:.3f}")
        print(f"  平均误差: {mean_err:+.1f} 分 (正=高估, 负=低估)")


def print_session_diagnosis_statistics(session_records: List[Dict[str, Any]]):
    """
    增强版Session诊断统计输出
    """
    total = len(session_records)
    if total == 0:
        print("无Session记录")
        return

    print("\n" + "=" * 70)
    print("Session级诊断预测准确率统计 (增强版)")
    print("=" * 70)
    print(f"总检查数: {total}")

    # 1. 是否面瘫
    has_palsy_correct = sum(1 for r in session_records
                            if r.get("gt_has_palsy") == r.get("pred_has_palsy"))
    print(f"\n1. 是否面瘫 (has_palsy):")
    print(f"   准确率: {has_palsy_correct / total:.1%} ({has_palsy_correct}/{total})")

    # 2. 面瘫侧别
    side_ok = side_wrong = side_fn = side_fp = 0
    for r in session_records:
        gt = r.get("gt_palsy_side", 0)
        pred = r.get("pred_palsy_side", 0)
        if gt != 0:
            if pred == gt:
                side_ok += 1
            elif pred == 0:
                side_fn += 1
            else:
                side_wrong += 1
        else:
            if pred != 0:
                side_fp += 1

    side_total = side_ok + side_wrong + side_fn
    if side_total > 0:
        print(f"\n2. 面瘫侧别 (palsy_side):")
        print(f"   严格准确率: {side_ok / side_total:.1%} ({side_ok}/{side_total})")
        print(f"   宽松准确率: {(side_ok + side_fn) / side_total:.1%} (含FN)")
        print(f"   详情: 正确={side_ok}, 错侧={side_wrong}, FN={side_fn}, FP={side_fp}")

    # 3. HB详细统计
    print_hb_detailed_statistics(session_records)

    # 4. Sunnybrook详细统计
    print_sunnybrook_detailed_statistics(session_records)

def save_session_diagnosis_csv(session_records: List[Dict[str, Any]], output_path: Path):
    """保存Session级诊断记录到CSV"""
    if not session_records:
        return

    fieldnames = [
        "exam_id",
        "gt_has_palsy", "pred_has_palsy", "has_palsy_match",
        "gt_palsy_side", "pred_palsy_side", "palsy_side_match",
        "gt_hb_grade", "pred_hb_grade", "hb_diff",
        "gt_sunnybrook", "pred_sunnybrook", "sunnybrook_error",
        "pred_confidence", "pred_palsy_side_confidence",
        "pred_resting_score", "pred_voluntary_score", "pred_synkinesis_score"
    ]

    rows = []
    for r in session_records:
        gt_hp = r.get("gt_has_palsy")
        pred_hp = r.get("pred_has_palsy")
        gt_side = r.get("gt_palsy_side")
        pred_side = r.get("pred_palsy_side")
        gt_hb = r.get("gt_hb_grade")
        pred_hb = r.get("pred_hb_grade")
        gt_sb = r.get("gt_sunnybrook")
        pred_sb = r.get("pred_sunnybrook")

        row = {
            "exam_id": r.get("exam_id"),
            "gt_has_palsy": gt_hp,
            "pred_has_palsy": pred_hp,
            "has_palsy_match": "✓" if gt_hp is not None and pred_hp is not None and bool(gt_hp) == bool(
                pred_hp) else "✗",
            "gt_palsy_side": gt_side,
            "pred_palsy_side": pred_side,
            "palsy_side_match": "✓" if gt_side == pred_side else (
                "FN" if pred_side == 0 and gt_side != 0 else ("FP" if pred_side != 0 and gt_side == 0 else "✗")),
            "gt_hb_grade": gt_hb,
            "pred_hb_grade": pred_hb,
            "hb_diff": abs(gt_hb - pred_hb) if gt_hb is not None and pred_hb is not None else None,
            "gt_sunnybrook": gt_sb,
            "pred_sunnybrook": pred_sb,
            "sunnybrook_error": pred_sb - gt_sb if gt_sb is not None and pred_sb is not None else None,
            "pred_confidence": r.get("pred_confidence"),
            "pred_palsy_side_confidence": r.get("pred_palsy_side_confidence"),
            "pred_resting_score": r.get("pred_resting_score"),
            "pred_voluntary_score": r.get("pred_voluntary_score"),
            "pred_synkinesis_score": r.get("pred_synkinesis_score"),
        }
        rows.append(row)

    # 按exam_id排序
    rows.sort(key=lambda x: x["exam_id"])

    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Session diagnosis CSV saved: {output_path}")


def save_hb_trace_csv(session_records: List[Dict[str, Any]], output_path: Path):
    """
    输出 HB 决策链路审计 CSV：一行一个 session
    重点字段：输入指标 / 阈值 / 命中分支 / 与真值差异
    """
    if not session_records:
        return

    def _get(d, *keys):
        cur = d
        for k in keys:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(k)
        return cur

    fieldnames = [
        "exam_id",
        "gt_hb_grade", "pred_hb_grade", "hb_diff",
        "gt_has_palsy", "pred_has_palsy",
        "gt_palsy_side", "pred_palsy_side",

        # HB 输入指标
        "eye_closure",
        "forehead_aff", "forehead_healthy", "forehead_ratio",
        "mouth_aff", "mouth_healthy", "mouth_ratio",
        "avg_asymmetry",
        "mouth_source_action",

        # HB 阈值
        "thr_eye_complete", "thr_forehead_min", "thr_mouth_min",
        "thr_forehead_ratio_grade3", "thr_asym_grade1",

        # 决策路径
        "decision_path",

        # Sunnybrook（便于你对照）
        "gt_sunnybrook", "pred_sunnybrook", "sunnybrook_error",
        "pred_resting_score", "pred_voluntary_score", "pred_synkinesis_score",
    ]

    rows = []
    for r in session_records:
        ev = r.get("pred_hb_evidence") or {}
        thr = ev.get("thresholds") or {}
        dp = ev.get("decision_path") or []
        if isinstance(dp, list):
            dp_str = " ; ".join([str(x) for x in dp])
        else:
            dp_str = str(dp)

        gt_hb = r.get("gt_hb_grade")
        pred_hb = r.get("pred_hb_grade")
        gt_sb = r.get("gt_sunnybrook")
        pred_sb = r.get("pred_sunnybrook")

        rows.append({
            "exam_id": r.get("exam_id"),
            "gt_hb_grade": gt_hb,
            "pred_hb_grade": pred_hb,
            "hb_diff": abs(gt_hb - pred_hb) if gt_hb is not None and pred_hb is not None else None,
            "gt_has_palsy": r.get("gt_has_palsy"),
            "pred_has_palsy": r.get("pred_has_palsy"),
            "gt_palsy_side": r.get("gt_palsy_side"),
            "pred_palsy_side": r.get("pred_palsy_side"),

            "eye_closure": ev.get("eye_closure"),

            "forehead_aff": _get(ev, "forehead_movement", "affected"),
            "forehead_healthy": _get(ev, "forehead_movement", "healthy"),
            "forehead_ratio": ev.get("forehead_ratio"),

            "mouth_aff": _get(ev, "mouth_movement", "affected"),
            "mouth_healthy": _get(ev, "mouth_movement", "healthy"),
            "mouth_ratio": ev.get("mouth_ratio"),

            "avg_asymmetry": ev.get("avg_asymmetry", ev.get("overall_asymmetry")),
            "mouth_source_action": ev.get("mouth_source_action"),

            "thr_eye_complete": thr.get("EYE_CLOSURE_COMPLETE"),
            "thr_forehead_min": thr.get("FOREHEAD_MINIMAL"),
            "thr_mouth_min": thr.get("MOUTH_MINIMAL"),
            "thr_forehead_ratio_grade3": thr.get("FOREHEAD_RATIO_GRADE_III_MAX"),
            "thr_asym_grade1": thr.get("ASYM_GRADE_I_MAX"),

            "decision_path": dp_str,

            "gt_sunnybrook": gt_sb,
            "pred_sunnybrook": pred_sb,
            "sunnybrook_error": (pred_sb - gt_sb) if gt_sb is not None and pred_sb is not None else None,
            "pred_resting_score": r.get("pred_resting_score"),
            "pred_voluntary_score": r.get("pred_voluntary_score"),
            "pred_synkinesis_score": r.get("pred_synkinesis_score"),
        })

    rows.sort(key=lambda x: x["exam_id"])

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] HB trace CSV saved: {output_path}")


def save_hb_trace_error_json(session_records: List[Dict[str, Any]], output_path: Path):
    """
    输出“错误样本”JSON：便于逐条打开看完整链路
    规则：HB不对 or has_palsy错 or palsy_side错 or Sunnybrook误差很大
    """
    if not session_records:
        return

    error_cases = []

    for r in session_records:
        gt_hb = r.get("gt_hb_grade")
        pred_hb = r.get("pred_hb_grade")
        gt_hp = r.get("gt_has_palsy")
        pred_hp = r.get("pred_has_palsy")
        gt_side = r.get("gt_palsy_side")
        pred_side = r.get("pred_palsy_side")
        gt_sb = r.get("gt_sunnybrook")
        pred_sb = r.get("pred_sunnybrook")

        hb_bad = (gt_hb is not None and pred_hb is not None and gt_hb != pred_hb)
        hp_bad = (gt_hp is not None and pred_hp is not None and bool(gt_hp) != bool(pred_hp))
        side_bad = (gt_side is not None and pred_side is not None and gt_side != pred_side and not (gt_side == 0 and pred_side == 0))

        sb_bad = False
        sb_err = None
        if gt_sb is not None and pred_sb is not None:
            sb_err = float(pred_sb - gt_sb)
            sb_bad = abs(sb_err) >= 20.0  # 你可改 10/15/20

        if hb_bad or hp_bad or side_bad or sb_bad:
            error_cases.append({
                "exam_id": r.get("exam_id"),
                "gt": {
                    "has_palsy": gt_hp,
                    "palsy_side": gt_side,
                    "hb_grade": gt_hb,
                    "sunnybrook": gt_sb,
                },
                "pred": {
                    "has_palsy": pred_hp,
                    "palsy_side": pred_side,
                    "hb_grade": pred_hb,
                    "sunnybrook": pred_sb,
                    "confidence": r.get("pred_confidence"),
                    "palsy_side_confidence": r.get("pred_palsy_side_confidence"),
                },
                "diff": {
                    "hb_diff": abs(gt_hb - pred_hb) if gt_hb is not None and pred_hb is not None else None,
                    "sunnybrook_error": sb_err,
                },
                "hb_evidence": r.get("pred_hb_evidence"),
                "votes": r.get("pred_votes"),
                "top_evidence": r.get("pred_top_evidence"),
                "consistency_checks": r.get("pred_consistency_checks"),
                "adjustments_made": r.get("pred_adjustments_made"),
                "sunnybrook_detail": r.get("pred_sunnybrook_detail"),
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(error_cases, f, indent=2, ensure_ascii=False)

    print(f"[OK] HB trace error JSON saved: {output_path} (n={len(error_cases)})")


def save_session_diagnosis_summary(stats: Dict[str, Any], output_path: Path):
    """保存Session级诊断统计摘要到文件"""
    if not stats:
        return

    lines = [
        "=" * 70,
        "Session级诊断预测准确率统计",
        "=" * 70,
        f"总检查数: {stats['total']}",
        "",
    ]

    # 1. 是否面瘫
    hp = stats["has_palsy"]
    if hp["total"] > 0:
        lines.append("1. 是否面瘫 (has_palsy):")
        lines.append(f"   准确率: {hp['accuracy']:.1%} ({hp['correct']}/{hp['total']})")
        lines.append("")

    # 2. 面瘫侧别
    ps = stats["palsy_side"]
    if ps["total_palsy"] > 0:
        lines.append("2. 面瘫侧别 (palsy_side):")
        lines.append(f"   严格准确率: {ps['strict_acc']:.1%} ({ps['correct']}/{ps['total_palsy']})")
        lines.append(f"   宽松准确率: {ps['relaxed_acc']:.1%} (含FN)")
        lines.append(f"   详情: 正确={ps['correct']}, 错侧={ps['wrong_side']}, FN={ps['fn']}, FP={ps['fp']}")
        lines.append("")

    # 3. HB Grade
    hb = stats["hb_grade"]
    if hb["total"] > 0:
        lines.append("3. HB分级 (hb_grade):")
        lines.append(f"   精确匹配: {hb['exact_acc']:.1%} ({hb['exact_match']}/{hb['total']})")
        lines.append(f"   ±1级准确: {hb['within_1_acc']:.1%} ({hb['within_1']}/{hb['total']})")
        lines.append(f"   MAE: {hb['mae']:.2f} 级")
        lines.append("")

    # 4. Sunnybrook
    sb = stats["sunnybrook"]
    if sb["total"] > 0:
        lines.append("4. Sunnybrook评分:")
        lines.append(f"   样本数: {sb['total']}")
        lines.append(f"   MAE: {sb['mae']:.1f} 分")
        lines.append(f"   RMSE: {sb['rmse']:.1f} 分")
        if sb["correlation"] is not None:
            lines.append(f"   Pearson相关系数: {sb['correlation']:.3f}")
        lines.append("")

    lines.append("=" * 70)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    print(f"[OK] Session diagnosis summary saved: {output_path}")


# =============================================================================
# 保存函数
# =============================================================================
def save_by_action_csv(records: List[Dict[str, Any]], output_dir: Path):
    """按动作分别保存CSV"""
    by_action = defaultdict(list)
    for r in records:
        by_action[r["action"]].append(r)

    base_fieldnames = ["exam_id", "hb", "gt", "gt_text", "pred", "pred_text",
                       "result", "conf", "method", "metrics_summary", "boundary_note"]

    for action in ACTIONS:
        if action not in by_action:
            continue

        action_records = by_action[action]
        action_records.sort(key=lambda x: x["exam_id"])

        output_path = output_dir / f"palsy_{action}.csv"
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=base_fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(action_records)

    print(f"[OK] Per-action CSV files saved to: {output_dir}")


def save_all_records_csv(records: List[Dict[str, Any]], output_path: Path):
    """保存所有记录到单个CSV"""
    action_order = {a: i for i, a in enumerate(ACTIONS)}
    sorted_records = sorted(records, key=lambda x: (action_order.get(x["action"], 99), x["exam_id"]))

    fieldnames = ["action", "exam_id", "hb", "gt", "gt_text", "pred", "pred_text",
                  "result", "conf", "method", "metrics_summary", "boundary_note"]

    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(sorted_records)

    print(f"[OK] All records saved: {output_path}")


def save_summary_csv(stats: Dict[str, Dict[str, Any]], output_path: Path):
    """保存统计摘要"""
    header = ["action", "total", "valid_samples", "palsy_cases",
              "OK", "WRONG", "FP", "FN", "both_sym", "miss_gt", "miss_pred",
              "side_accuracy", "relaxed_accuracy"]

    rows = []
    for action in ACTIONS:
        if action in stats:
            s = stats[action]
            rows.append({
                "action": action,
                "total": s["total"],
                "valid_samples": s.get("valid_samples", 0),
                "palsy_cases": s.get("palsy_cases", 0),
                "OK": s["OK"],
                "WRONG": s["WRONG"],
                "FP": s["FP"],
                "FN": s["FN"],
                "both_sym": s["both_sym"],
                "miss_gt": s["miss_gt"],
                "miss_pred": s["miss_pred"],
                "side_accuracy": f"{s['side_accuracy']:.4f}" if s['side_accuracy'] is not None else "",
                "relaxed_accuracy": f"{s['relaxed_accuracy']:.4f}" if s['relaxed_accuracy'] is not None else "",
            })

    # 汇总行
    total_ok = sum(stats[a]["OK"] for a in stats)
    total_wrong = sum(stats[a]["WRONG"] for a in stats)
    total_fn = sum(stats[a]["FN"] for a in stats)
    total_fp = sum(stats[a]["FP"] for a in stats)
    total_palsy = total_ok + total_wrong + total_fn

    rows.append({
        "action": "TOTAL",
        "total": sum(stats[a]["total"] for a in stats),
        "valid_samples": sum(stats[a].get("valid_samples", 0) for a in stats),
        "palsy_cases": total_palsy,
        "OK": total_ok,
        "WRONG": total_wrong,
        "FP": total_fp,
        "FN": total_fn,
        "both_sym": sum(stats[a]["both_sym"] for a in stats),
        "miss_gt": sum(stats[a]["miss_gt"] for a in stats),
        "miss_pred": sum(stats[a]["miss_pred"] for a in stats),
        "side_accuracy": f"{total_ok / total_palsy:.4f}" if total_palsy > 0 else "",
        "relaxed_accuracy": f"{(total_ok + total_fn) / total_palsy:.4f}" if total_palsy > 0 else "",
    })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Summary saved: {output_path}")


def save_confusion_matrix(records: List[Dict[str, Any]], output_path: Path):
    """保存混淆矩阵统计"""
    by_action = defaultdict(list)
    for r in records:
        by_action[r["action"]].append(r)

    lines = ["=" * 70, "面瘫侧别预测 - 混淆矩阵统计",
             f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "=" * 70]

    for action in ACTIONS:
        if action not in by_action:
            continue

        action_records = by_action[action]
        matrix = defaultdict(lambda: defaultdict(int))

        for r in action_records:
            gt = r["gt"] if r["gt"] is not None else -1
            pred = r["pred"] if r["pred"] is not None else -1
            matrix[gt][pred] += 1

        ok = sum(1 for r in action_records if r["result"] == "OK")
        wrong = sum(1 for r in action_records if r["result"] == "WRONG")
        palsy_n = ok + wrong + sum(1 for r in action_records if r["result"] == "FN")
        acc = f"{ok / palsy_n:.1%}" if palsy_n > 0 else "N/A"

        lines.extend([f"\n{'─' * 50}",
                      f"【{action}】 样本={len(action_records)}, 面瘫样本={palsy_n}, 侧别准确率={acc}",
                      "─" * 50,
                      f"{'GT \\ Pred':>10} | {'Sym':>5} | {'Left':>5} | {'Right':>5} | {'N/A':>5}",
                      "-" * 45])

        for gt_val, gt_name in [(-1, "N/A"), (0, "Sym"), (1, "Left"), (2, "Right")]:
            row = matrix.get(gt_val, {})
            vals = [row.get(0, 0), row.get(1, 0), row.get(2, 0), row.get(-1, 0)]
            lines.append(f"{gt_name:>10} | {vals[0]:>5} | {vals[1]:>5} | {vals[2]:>5} | {vals[3]:>5}")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"[OK] Confusion matrix saved: {output_path}")


def analyze_thresholds_and_plot(records: List[Dict[str, Any]], output_dir: Path) -> None:
    """
    从 collect_keyframes 的 records 里提取“指标值 vs 阈值”，并按 GT(真实侧别) / 结果类型画分布图。
    输出:
      - output_dir/threshold_analysis/threshold_metrics.csv
      - output_dir/threshold_analysis/*.png
    """
    out_dir = output_dir / "threshold_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 你现在 error_cases.json 里能稳定看到的阈值字段组合（先做这几个最确定的）
    SPECS = [
        # action, value_key, thr_key, plot_name
        ("LipPucker", "offset_norm", "thr_offset_norm", "LipPucker_offset_norm"),
        ("LipPucker", "midline_angle_deg", "thr_angle_deg", "LipPucker_midline_angle_deg"),
        ("ShowTeeth", "offset_norm", "thr_offset_norm", "ShowTeeth_offset_norm"),
        ("BlowCheek", "lip_offset_normalized_value", "lip_offset_threshold", "BlowCheek_lip_offset_norm"),
        ("BlowCheek", "lip_shape_asymmetry_ratio", "lip_shape_threshold", "BlowCheek_lip_shape_asym"),
        # CloseEye 的 frame_area_diff_threshold 是“逐帧差异阈值”，不是这里这种 value-vs-thr 结构，
        # 所以先不画“value对thr”，后面你补充帧级统计再做。
    ]

    rows = []
    for r in records:
        act = r.get("action")
        dm = r.get("detailed_metrics") or {}
        for (a, vkey, tkey, plot_name) in SPECS:
            if act != a:
                continue
            if vkey not in dm or tkey not in dm:
                continue
            try:
                v = float(dm[vkey])
                t = float(dm[tkey])
            except Exception:
                continue
            rows.append({
                "exam_id": r.get("exam_id"),
                "action": act,
                "metric": vkey,
                "value": v,
                "threshold": t,
                "gt_text": r.get("gt_text"),
                "pred_text": r.get("pred_text"),
                "result": r.get("result"),
                "conf": r.get("conf"),
                "hb": r.get("hb"),
                "near_20pct_of_thr": (abs(v - t) <= max(1e-12, 0.2 * abs(t))),
                "plot_name": plot_name,
            })

    # 没数据就直接返回
    if not rows:
        print("[WARN] threshold_analysis: no rows found (metrics/thr keys not present).")
        return

    # 1) CSV 落盘，便于你用 Excel / pandas 再分析
    csv_path = out_dir / "threshold_metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] threshold_analysis csv saved: {csv_path}")

    # 2) 画图：每个 plot_name 一张图
    # 图1：按 GT(L/R/Sym) 分布 + 画阈值线
    grouped = {}
    for x in rows:
        grouped.setdefault((x["plot_name"], x["metric"]), []).append(x)

    for (plot_name, metric), items in grouped.items():
        # 收集 GT 分类
        gts = ["Sym", "L", "R"]
        data_by_gt = {g: [] for g in gts}
        thr_vals = []
        for it in items:
            gt = it.get("gt_text")
            if gt in data_by_gt:
                data_by_gt[gt].append(it["value"])
            thr_vals.append(it["threshold"])

        # 阈值通常是固定值；如果不同样本有微小差异，用中位数画线
        thr = float(np.median(thr_vals)) if thr_vals else None

        plt.figure(figsize=(10, 6))
        # 用 step hist 叠加，方便对比
        for gt in gts:
            vals = [v for v in data_by_gt[gt] if np.isfinite(v)]
            if not vals:
                continue
            plt.hist(vals, bins=30, histtype="step", linewidth=2, label=f"GT={gt} (n={len(vals)})")

        if thr is not None and np.isfinite(thr):
            plt.axvline(thr, linestyle="--", linewidth=2, label=f"threshold={thr:.4g}")

        plt.title(f"{plot_name} | {metric} distribution by GT")
        plt.xlabel(metric)
        plt.ylabel("count")
        plt.legend()
        plt.tight_layout()
        fig_path = out_dir / f"{plot_name}__{metric}__byGT.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()

        # 图2：按 result(OK/WRONG/FN/FP) 分布（帮助你看阈值对错误类型的影响）
        res_types = ["OK", "WRONG", "FN", "FP"]
        data_by_res = {k: [] for k in res_types}
        for it in items:
            rt = it.get("result")
            if rt in data_by_res:
                data_by_res[rt].append(it["value"])

        plt.figure(figsize=(10, 6))
        for rt in res_types:
            vals = [v for v in data_by_res[rt] if np.isfinite(v)]
            if not vals:
                continue
            plt.hist(vals, bins=30, histtype="step", linewidth=2, label=f"{rt} (n={len(vals)})")

        if thr is not None and np.isfinite(thr):
            plt.axvline(thr, linestyle="--", linewidth=2, label=f"threshold={thr:.4g}")

        plt.title(f"{plot_name} | {metric} distribution by result")
        plt.xlabel(metric)
        plt.ylabel("count")
        plt.legend()
        plt.tight_layout()
        fig_path = out_dir / f"{plot_name}__{metric}__byResult.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()

    print(f"[OK] threshold_analysis plots saved under: {out_dir}")


def plot_all_actions_threshold_analysis(records: List[Dict[str, Any]], output_dir: Path) -> None:
    """
    为所有11个动作绘制阈值统计分析图

    每个动作区分：
    - 决定性Evidence: 实际用于判断的指标
    - 冗余Evidence: 仅供参考的指标

    输出:
    - output_dir/threshold_analysis/{action}/*.png
    - output_dir/threshold_analysis/summary.csv
    """
    out_dir = output_dir / "threshold_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 每个动作的决定性Evidence配置
    # 格式: (action, evidence_key, threshold_value, plot_title, is_decisive)
    ACTION_EVIDENCE_CONFIG = {
        # ========== 眼部动作 ==========
        "SpontaneousEyeBlink": [
            ("left_worse_ratio", 0.30, "左眼表现差帧比例", True),
            ("right_worse_ratio", 0.30, "右眼表现差帧比例", True),
            ("left_closure_at_peak", None, "左眼峰值闭合度", False),
            ("right_closure_at_peak", None, "右眼峰值闭合度", False),
        ],
        "VoluntaryEyeBlink": [
            ("left_worse_ratio", 0.30, "左眼表现差帧比例", True),
            ("right_worse_ratio", 0.30, "右眼表现差帧比例", True),
        ],
        "CloseEyeSoftly": [
            ("left_worse_ratio", 0.30, "左眼表现差帧比例", True),
            ("right_worse_ratio", 0.30, "右眼表现差帧比例", True),
        ],
        "CloseEyeHardly": [
            ("left_worse_ratio", 0.30, "左眼表现差帧比例", True),
            ("right_worse_ratio", 0.30, "右眼表现差帧比例", True),
        ],

        # ========== RaiseEyebrow ==========
        "RaiseEyebrow": [
            ("asymmetry_ratio", 0.10, "眉眼距变化不对称比", True),
            ("left_change", None, "左侧眉眼距变化(px)", True),
            ("right_change", None, "右侧眉眼距变化(px)", True),
        ],

        # ========== Smile ==========
        "Smile": [
            ("reduction_asymmetry", 0.08, "嘴角上提不对称比", True),
            ("left_reduction", None, "左嘴角上提量(px)", True),
            ("right_reduction", None, "右嘴角上提量(px)", True),
        ],

        # ========== ShrugNose ==========
        "ShrugNose": [
            ("reduction_asymmetry", 0.08, "鼻翼收缩不对称比", True),
            ("left_reduction", None, "左鼻翼收缩(px)", True),
            ("right_reduction", None, "右鼻翼收缩(px)", True),
        ],

        # ========== ShowTeeth ==========
        "ShowTeeth": [
            ("offset_norm", 0.025, "嘴唇中心偏移(ICD归一化)", True),
            ("current_offset_px", None, "嘴唇中心偏移(px)", True),
            ("midline_angle_deg", 1.5, "嘴唇中线角度(°)", False),
        ],

        # ========== BlowCheek ==========
        "BlowCheek": [
            ("lip_offset_normalized_value", 0.020, "嘴唇偏移(ICD归一化)", True),
            ("cheek_bulge_asymmetry_ratio", 0.10, "脸颊膨胀不对称比", True),
            ("lip_shape_asymmetry_ratio", 0.10, "嘴唇形状不对称比", True),
        ],

        # ========== LipPucker ==========
        "LipPucker": [
            ("offset_norm", 0.01, "嘴唇中心偏移(ICD归一化)", True),
            ("midline_angle_deg", 0.3, "嘴唇中线角度(°)", True),
            ("corner_contraction_asymmetry", 0.06, "嘴角收缩不对称比", False),
        ],
    }

    all_rows = []

    for action, evidence_list in ACTION_EVIDENCE_CONFIG.items():
        action_dir = out_dir / action
        action_dir.mkdir(parents=True, exist_ok=True)

        # 筛选该动作的记录
        action_records = [r for r in records if r.get("action") == action]
        if not action_records:
            continue

        for evidence_key, threshold, title, is_decisive in evidence_list:
            # 提取数据
            data_by_gt = {"Sym": [], "L": [], "R": []}
            data_by_result = {"OK": [], "WRONG": [], "FN": [], "FP": []}

            for r in action_records:
                dm = r.get("detailed_metrics") or {}
                value = dm.get(evidence_key)
                if value is None:
                    continue
                try:
                    value = float(value)
                except:
                    continue

                gt = r.get("gt_text", "")
                result = r.get("result", "")

                if gt in data_by_gt:
                    data_by_gt[gt].append(value)
                if result in data_by_result:
                    data_by_result[result].append(value)

                # 记录到汇总
                all_rows.append({
                    "action": action,
                    "evidence_key": evidence_key,
                    "is_decisive": is_decisive,
                    "threshold": threshold,
                    "value": value,
                    "gt": gt,
                    "result": result,
                    "exam_id": r.get("exam_id"),
                })

            # ========== 图1: 按GT分布 ==========
            plt.figure(figsize=(12, 6))
            colors = {"Sym": "green", "L": "blue", "R": "red"}

            for gt in ["Sym", "L", "R"]:
                vals = [v for v in data_by_gt[gt] if np.isfinite(v)]
                if vals:
                    plt.hist(vals, bins=30, histtype="step", linewidth=2,
                             color=colors[gt], label=f"GT={gt} (n={len(vals)})")

            if threshold is not None:
                plt.axvline(threshold, linestyle="--", color="black", linewidth=2,
                            label=f"阈值={threshold}")

            decisive_mark = "★决定性" if is_decisive else "○辅助"
            plt.title(f"{action} | {title} ({decisive_mark})\n按真实侧别分布", fontsize=12)
            plt.xlabel(evidence_key)
            plt.ylabel("样本数")
            plt.legend()
            plt.tight_layout()
            plt.savefig(action_dir / f"{evidence_key}__byGT.png", dpi=150)
            plt.close()

            # ========== 图2: 按结果分布 ==========
            plt.figure(figsize=(12, 6))
            colors = {"OK": "green", "WRONG": "red", "FN": "orange", "FP": "purple"}

            for res in ["OK", "WRONG", "FN", "FP"]:
                vals = [v for v in data_by_result[res] if np.isfinite(v)]
                if vals:
                    plt.hist(vals, bins=30, histtype="step", linewidth=2,
                             color=colors[res], label=f"{res} (n={len(vals)})")

            if threshold is not None:
                plt.axvline(threshold, linestyle="--", color="black", linewidth=2,
                            label=f"阈值={threshold}")

            plt.title(f"{action} | {title} ({decisive_mark})\n按判断结果分布", fontsize=12)
            plt.xlabel(evidence_key)
            plt.ylabel("样本数")
            plt.legend()
            plt.tight_layout()
            plt.savefig(action_dir / f"{evidence_key}__byResult.png", dpi=150)
            plt.close()

            # ========== 图3: 左右对比散点图(如果是paired数据) ==========
            if evidence_key.startswith("left_") or evidence_key.startswith("right_"):
                # 找到配对的key
                if evidence_key.startswith("left_"):
                    pair_key = "right_" + evidence_key[5:]
                else:
                    pair_key = "left_" + evidence_key[6:]

                left_vals = []
                right_vals = []
                gt_labels = []

                for r in action_records:
                    dm = r.get("detailed_metrics") or {}
                    l = dm.get("left_" + evidence_key[5:] if evidence_key.startswith("left_") else evidence_key)
                    r_val = dm.get("right_" + evidence_key[5:] if evidence_key.startswith("left_") else pair_key)

                    if l is not None and r_val is not None:
                        try:
                            left_vals.append(float(l) if evidence_key.startswith("left_") else float(r_val))
                            right_vals.append(float(r_val) if evidence_key.startswith("left_") else float(l))
                            gt_labels.append(r.get("gt_text", ""))
                        except:
                            continue

                if left_vals and right_vals:
                    plt.figure(figsize=(10, 10))
                    colors_scatter = {"Sym": "green", "L": "blue", "R": "red", "": "gray"}

                    for gt in ["Sym", "L", "R"]:
                        indices = [i for i, g in enumerate(gt_labels) if g == gt]
                        if indices:
                            x = [left_vals[i] for i in indices]
                            y = [right_vals[i] for i in indices]
                            plt.scatter(x, y, c=colors_scatter.get(gt, "gray"),
                                        label=f"GT={gt} (n={len(indices)})", alpha=0.6)

                    # 对角线
                    all_vals = left_vals + right_vals
                    min_v, max_v = min(all_vals), max(all_vals)
                    plt.plot([min_v, max_v], [min_v, max_v], 'k--', alpha=0.5, label="y=x")

                    plt.xlabel("Left")
                    plt.ylabel("Right")
                    plt.title(f"{action} | 左右对比散点图")
                    plt.legend()
                    plt.axis('equal')
                    plt.tight_layout()
                    plt.savefig(action_dir / f"LR_scatter.png", dpi=150)
                    plt.close()

    # 保存汇总CSV
    if all_rows:
        csv_path = out_dir / "all_evidence_summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"[OK] 阈值分析汇总已保存: {csv_path}")

    print(f"[OK] 所有动作阈值分析图已保存: {out_dir}")

def save_error_analysis(records: List[Dict[str, Any]], output_dir: Path):
    """保存错误案例分析"""
    error_cases = {"WRONG": [], "FN": [], "FP": []}

    for r in records:
        result = r["result"]
        if result in error_cases:
            error_cases[result].append(r)

    # 保存JSON
    json_path = output_dir / "error_cases.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(error_cases, f, indent=2, ensure_ascii=False)

    # 生成人类可读报告
    report_path = output_dir / "error_analysis.txt"
    lines = ["=" * 80, "面瘫侧别预测 - 错误案例分析报告",
             f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "=" * 80]

    for error_type, cases in error_cases.items():
        lines.extend([f"\n\n{'=' * 40}", f"【{error_type}】 共 {len(cases)} 例", "=" * 40])

        by_action = defaultdict(list)
        for c in cases:
            by_action[c["action"]].append(c)

        for action in ACTIONS:
            if action not in by_action:
                continue

            action_cases = by_action[action]
            lines.append(f"\n--- {action} ({len(action_cases)}例) ---")

            for c in action_cases:
                lines.extend([f"\n  {c['exam_id']}:",
                              f"    GT={c['gt_text']} → Pred={c['pred_text']} (HB={c['hb']})",
                              f"    Method: {c['method']}",
                              f"    Metrics: {c['metrics_summary']}"])
                if c.get('boundary_note'):
                    lines.append(f"    ⚠ 边界: {c['boundary_note']}")
                lines.append(f"    解释: {c.get('interpretation', 'N/A')}")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    analyze_thresholds_and_plot(records, output_dir)

    print(f"[OK] Error analysis saved to: {output_dir}")


def save_severity_stats(records: List[Dict[str, Any]], output_path: Path, labels_db_path: str = None):
    """
    保存严重度分数统计 - 详细版

    每个动作的每个分数都单独一行展示
    """
    from collections import Counter
    import sqlite3

    # 加载GT标签
    gt_severity = {}
    gt_hb_grade = {}

    if labels_db_path:
        try:
            conn = sqlite3.connect(labels_db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT al.examination_id, at.action_name_en, al.severity_score 
                FROM action_labels al
                JOIN action_types at ON al.action_id = at.action_id
                WHERE al.severity_score IS NOT NULL
            """)
            for row in cursor.fetchall():
                exam_id, action_name, severity = row
                gt_severity[(exam_id, action_name)] = severity

            cursor.execute("""
                SELECT examination_id, hb_grade
                FROM examination_labels
                WHERE hb_grade IS NOT NULL
            """)
            for row in cursor.fetchall():
                exam_id, hb = row
                gt_hb_grade[exam_id] = hb

            conn.close()
            print(f"[INFO] Loaded {len(gt_severity)} GT severity labels")
            print(f"[INFO] Loaded {len(gt_hb_grade)} HB grades")

        except Exception as e:
            print(f"[WARN] Failed to load GT labels: {e}")
            import traceback
            traceback.print_exc()

    # 按动作分组
    by_action = defaultdict(list)
    for r in records:
        exam_id = r["exam_id"]
        action = r["action"]
        by_action[action].append({
            "exam_id": exam_id,
            "hb": r.get("hb") or gt_hb_grade.get(exam_id),
            "pred": r.get("severity_score"),
            "gt": gt_severity.get((exam_id, action)),
        })

    # ===== 详细CSV输出 =====
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Action", "Score", "GT_N", "Pred_N", "Correct", "Recall", "Precision"
        ])

        for action in ACTIONS:

            data = by_action.get(action, [])
            if not data:
                continue

            paired = [(d["pred"], d["gt"]) for d in data if d["pred"] is not None and d["gt"] is not None]
            gt_dist = Counter(d["gt"] for d in data if d["gt"] is not None)
            pred_dist = Counter(d["pred"] for d in data if d["pred"] is not None)

            # 构建混淆矩阵
            conf = defaultdict(lambda: defaultdict(int))
            for pred, gt in paired:
                conf[gt][pred] += 1

            # 每个分数一行
            for score in range(1, 6):
                gt_n = gt_dist.get(score, 0)
                pred_n = pred_dist.get(score, 0)
                correct = conf[score][score]

                recall = f"{correct / gt_n:.1%}" if gt_n > 0 else "-"
                precision = f"{correct / pred_n:.1%}" if pred_n > 0 else "-"

                writer.writerow([
                    action, score, gt_n, pred_n, correct, recall, precision
                ])

            # 动作汇总行
            total_correct = sum(conf[s][s] for s in range(1, 6))
            n_paired = len(paired)
            writer.writerow([
                f"{action}_TOTAL", "-",
                sum(gt_dist.values()),
                sum(pred_dist.values()),
                total_correct,
                f"{total_correct / n_paired:.1%}" if n_paired > 0 else "-",
                "-"
            ])
            writer.writerow([])  # 空行分隔

    print(f"[OK] Severity stats saved: {output_path}")

    # ===== 详细TXT报告 =====
    report_path = output_path.parent / "severity_detail.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("Severity Score 详细分析报告\n")
        f.write("=" * 70 + "\n")

        grand_total = {"paired": 0, "exact": 0, "within1": 0}

        for action in ACTIONS:

            data = by_action.get(action, [])
            if not data:
                continue

            paired = [(d["pred"], d["gt"]) for d in data if d["pred"] is not None and d["gt"] is not None]
            gt_dist = Counter(d["gt"] for d in data if d["gt"] is not None)
            pred_dist = Counter(d["pred"] for d in data if d["pred"] is not None)

            if not paired:
                f.write(f"\n{action}: 无配对数据\n")
                continue

            # 构建混淆矩阵
            conf = defaultdict(lambda: defaultdict(int))
            for pred, gt in paired:
                conf[gt][pred] += 1

            n_paired = len(paired)
            exact = sum(conf[s][s] for s in range(1, 6))
            within1 = sum(1 for p, g in paired if abs(p - g) <= 1)

            grand_total["paired"] += n_paired
            grand_total["exact"] += exact
            grand_total["within1"] += within1

            f.write(f"\n{'=' * 70}\n")
            f.write(f"{action} (N={n_paired})\n")
            f.write(f"{'=' * 70}\n\n")

            # 每个分数的详细统计
            f.write(f"{'Score':<8} {'GT_N':>6} {'Pred_N':>8} {'Correct':>9} {'Recall':>10} {'Precision':>11}\n")
            f.write("-" * 55 + "\n")

            for score in range(1, 6):
                gt_n = gt_dist.get(score, 0)
                pred_n = pred_dist.get(score, 0)
                correct = conf[score][score]

                if gt_n > 0 or pred_n > 0:
                    recall = f"{correct / gt_n:.1%}" if gt_n > 0 else "-"
                    precision = f"{correct / pred_n:.1%}" if pred_n > 0 else "-"

                    # 标记问题
                    flag = ""
                    if gt_n > 0 and correct == 0:
                        flag = " ← 全部漏检!"
                    elif pred_n > 0 and correct == 0:
                        flag = " ← 全部误报!"
                    elif gt_n > 0 and correct / gt_n < 0.5:
                        flag = " ← 召回率低"

                    f.write(f"{score:<8} {gt_n:>6} {pred_n:>8} {correct:>9} {recall:>10} {precision:>11}{flag}\n")

            f.write("-" * 55 + "\n")
            f.write(f"{'Total':<8} {sum(gt_dist.values()):>6} {sum(pred_dist.values()):>8} {exact:>9} "
                    f"{exact / n_paired:>9.1%} {'-':>11}\n")

            # 汇总指标
            f.write(f"\n准确率指标:\n")
            f.write(f"  Exact Match (完全正确): {exact}/{n_paired} = {exact / n_paired:.1%}\n")
            f.write(f"  Within ±1 (误差≤1):     {within1}/{n_paired} = {within1 / n_paired:.1%}\n")

            # 混淆矩阵
            f.write(f"\n混淆矩阵 (行=GT, 列=Pred):\n")
            f.write(f"{'GT\\Pred':>8}")
            for p in range(1, 6):
                f.write(f"{p:>6}")
            f.write("\n")

            for gt in range(1, 6):
                if gt_dist.get(gt, 0) == 0:
                    continue
                f.write(f"{gt:>8}")
                for pred in range(1, 6):
                    cnt = conf[gt][pred]
                    if cnt > 0:
                        f.write(f"{cnt:>6}")
                    else:
                        f.write(f"{'·':>6}")
                f.write(f"  (GT={gt}: {gt_dist.get(gt, 0)}例)\n")

            # 错误分析
            errors = [(p, g) for p, g in paired if p != g]
            if errors:
                f.write(f"\n错误分析 ({len(errors)}例):\n")
                error_types = Counter((g, p) for p, g in errors)
                for (gt, pred), count in error_types.most_common(10):
                    direction = "↑高估" if pred > gt else "↓低估"
                    f.write(f"  GT={gt} → Pred={pred}: {count}例 ({direction}{abs(pred - gt)}分)\n")

        # 总汇总
        f.write(f"\n{'=' * 70}\n")
        f.write(f"总体汇总\n")
        f.write(f"{'=' * 70}\n")
        if grand_total["paired"] > 0:
            f.write(f"总配对数: {grand_total['paired']}\n")
            f.write(
                f"Exact Match: {grand_total['exact']}/{grand_total['paired']} = {grand_total['exact'] / grand_total['paired']:.1%}\n")
            f.write(
                f"Within ±1:   {grand_total['within1']}/{grand_total['paired']} = {grand_total['within1'] / grand_total['paired']:.1%}\n")

    print(f"[OK] Detail report saved: {report_path}")

    # ===== 控制台输出 =====
    print(f"\n" + "=" * 80)
    print("Severity Score 详细统计")
    print("=" * 80)

    for action in ACTIONS:

        data = by_action.get(action, [])
        paired = [(d["pred"], d["gt"]) for d in data if d["pred"] is not None and d["gt"] is not None]

        if not paired:
            continue

        gt_dist = Counter(d["gt"] for d in data if d["gt"] is not None)
        pred_dist = Counter(d["pred"] for d in data if d["pred"] is not None)

        conf = defaultdict(lambda: defaultdict(int))
        for pred, gt in paired:
            conf[gt][pred] += 1

        n_paired = len(paired)
        exact = sum(conf[s][s] for s in range(1, 6))
        within1 = sum(1 for p, g in paired if abs(p - g) <= 1)

        print(f"\n{action} (N={n_paired}, Exact={exact / n_paired:.1%}, ±1={within1 / n_paired:.1%})")
        print("-" * 60)
        print(f"{'Score':<6} {'GT':>5} {'Pred':>6} {'OK':>5} {'Recall':>8} {'Prec':>8} {'Status'}")
        print("-" * 60)

        for score in range(1, 6):
            gt_n = gt_dist.get(score, 0)
            pred_n = pred_dist.get(score, 0)
            correct = conf[score][score]

            if gt_n == 0 and pred_n == 0:
                continue

            recall = f"{correct / gt_n:.0%}" if gt_n > 0 else "-"
            precision = f"{correct / pred_n:.0%}" if pred_n > 0 else "-"

            # 状态标记
            if gt_n > 0 and correct == 0:
                status = "❌ 全漏"
            elif pred_n > 0 and correct == 0:
                status = "⚠️ 全误"
            elif gt_n > 0 and correct / gt_n >= 0.8:
                status = "✓ 良好"
            elif gt_n > 0 and correct / gt_n >= 0.5:
                status = "△ 一般"
            elif gt_n > 0:
                status = "✗ 较差"
            else:
                status = "-"

            print(f"{score:<6} {gt_n:>5} {pred_n:>6} {correct:>5} {recall:>8} {precision:>8} {status}")

    # 总汇总
    total_paired = sum(len([(d["pred"], d["gt"]) for d in by_action.get(a, [])
                            if d["pred"] is not None and d["gt"] is not None])
                       for a in ACTIONS if a != "NeutralFace")

    if grand_total["paired"] > 0:
        print("\n" + "=" * 80)
        print(f"总体: N={grand_total['paired']}, "
              f"Exact={grand_total['exact'] / grand_total['paired']:.1%}, "
              f"±1={grand_total['within1'] / grand_total['paired']:.1%}")
        print("=" * 80)


def copy_classified_images(records: List[Dict[str, Any]], output_dir: Path):
    """
    按预测结果分类复制关键帧图片

    目录结构:
    output_dir/
      Action/
        OK/
          exam_id_peak_raw.jpg
          exam_id_peak_indicators.jpg
        WRONG/
        FN/
        FP/
        both_sym/
    """
    categories = ["OK", "WRONG", "FN", "FP", "both_sym"]

    # 创建目录结构
    for action in ACTIONS:

        for cat in categories:
            cat_dir = output_dir / action / cat
            cat_dir.mkdir(parents=True, exist_ok=True)

    # 统计复制数量
    copy_stats = defaultdict(lambda: defaultdict(int))

    # 复制图片
    for r in records:
        action = r["action"]

        result = r["result"]
        if result not in categories:
            continue

        exam_id = r["exam_id"]

        # 源目录
        src_action_dir = SRC_ROOT / exam_id / action
        if not src_action_dir.exists():
            continue

        # 目标目录
        dst_cat_dir = output_dir / action / result

        # 复制关键帧图片（peak_raw / peak_indicators）
        copied_files = 0
        for img_base in ["peak_raw", "peak_indicators"]:
            src_path = None
            for ext in ["jpg", "jpeg", "png", "webp"]:
                candidate = src_action_dir / f"{img_base}.{ext}"
                if candidate.exists():
                    src_path = candidate
                    break

            if src_path and src_path.exists():
                dst_name = f"{exam_id}_{img_base}{src_path.suffix}"
                dst_path = dst_cat_dir / dst_name
                try:
                    shutil.copy2(src_path, dst_path)
                    if img_base == "peak_raw":  # 只统计一次
                        copy_stats[action][result] += 1
                except Exception as e:
                    print(f"[WARN] Copy failed: {src_path} -> {dst_path}: {e}")

        # 复制证据曲线图, 规则：复制 src_action_dir 下所有 *_curve.png
        curve_candidates = []
        curve_candidates.extend(sorted(src_action_dir.glob("*_curve.png")))

        # 去重（按文件名）
        seen = set()
        for curve_path in curve_candidates:
            if not curve_path.is_file():
                continue
            if curve_path.name in seen:
                continue
            seen.add(curve_path.name)

            dst_name = f"{exam_id}_{curve_path.name}"  # 放在动作目录下，文件名带 exam_id
            dst_path = dst_cat_dir / dst_name
            try:
                shutil.copy2(curve_path, dst_path)  # 不压缩，原样复制
                copied_files += 1
            except Exception as e:
                print(f"[WARN] Copy curve failed: {curve_path} -> {dst_path}: {e}")

    # 打印统计
    total_pairs = sum(sum(v.values()) for v in copy_stats.values())
    print(f"\n[OK] Classified exam-action copied: {total_pairs} pairs (peak_raw+peak_indicators) to {output_dir}")
    print("[OK] Curve images (*_curve.png) are also copied when present.")

    print(f"\n分类图片统计:")
    print(f"{'Action':<20} {'OK':>5} {'WRONG':>6} {'FN':>4} {'FP':>4} {'both_sym':>8}")
    print("-" * 52)
    for action in ACTIONS:
        stats = copy_stats.get(action, {})
        if sum(stats.values()) > 0:
            print(f"{action:<20} {stats.get('OK', 0):>5} {stats.get('WRONG', 0):>6} "
                  f"{stats.get('FN', 0):>4} {stats.get('FP', 0):>4} {stats.get('both_sym', 0):>8}")


def copy_exam_pack_by_gt_hb(exam_dir: Path,
                            gt_hb: Optional[int],
                            pred_hb: Optional[int],
                            dst_root: Path,
                            copy_indicators: bool = True,
                            copy_selection_curve: bool = True,
                            copy_peak_raw: bool = False):
    """
    按 GT HB 分组，把一个 exam 的 11 个动作图片打包复制到一个目录，方便肉眼检查。

    目录建议：
      dst_root/gt_hb_groups/HB{gt}/PredHB{pred}_Diff{diff}/{exam_id}/
        NeutralFace_peak_indicators.jpg
        NeutralFace_peak_selection_curve.png
        ...
    """
    if gt_hb is None:
        return

    exam_id = exam_dir.name
    diff = None
    if gt_hb is not None and pred_hb is not None:
        diff = abs(gt_hb - pred_hb)

    group_dir = dst_root / "gt_hb_groups" / f"HB{gt_hb}"
    if pred_hb is None:
        sub_dir = group_dir / "PredHB_None"
    else:
        sub_dir = group_dir / f"PredHB{pred_hb}_Diff{diff}"

    dst_exam_dir = sub_dir / exam_id
    dst_exam_dir.mkdir(parents=True, exist_ok=True)

    exts = ["jpg", "jpeg", "png", "webp"]

    for action in ACTIONS:
        action_dir = exam_dir / action
        if not action_dir.exists():
            continue

        # 1) peak_indicators
        if copy_indicators:
            src = None
            for ext in exts:
                p = action_dir / f"peak_indicators.{ext}"
                if p.exists():
                    src = p
                    break
            if src:
                dst = dst_exam_dir / f"{action}_peak_indicators{src.suffix.lower()}"
                shutil.copy2(src, dst)

        # 2) peak_selection_curve（如果你命名不固定，就找 *_curve.png）
        if copy_selection_curve:
            src_curve = action_dir / "peak_selection_curve.png"
            if not src_curve.exists():
                curves = sorted(action_dir.glob("*_curve.png"))
                src_curve = curves[0] if curves else None
            if src_curve and src_curve.exists():
                dst = dst_exam_dir / f"{action}_peak_selection_curve.png"
                shutil.copy2(src_curve, dst)

        # 3) peak_raw（可选）
        if copy_peak_raw:
            src = None
            for ext in exts:
                p = action_dir / f"peak_raw.{ext}"
                if p.exists():
                    src = p
                    break
            if src:
                dst = dst_exam_dir / f"{action}_peak_raw{src.suffix.lower()}"
                shutil.copy2(src, dst)


def load_indicators_json(exam_dir: Path, action: str) -> Optional[Dict[str, Any]]:
    """
    加载指定动作的indicators.json

    Args:
        exam_dir: 检查目录 (如 /path/to/P001/E001)
        action: 动作名称 (如 "BlowCheek")

    Returns:
        indicators字典或None
    """
    indicators_path = exam_dir / action / "indicators.json"
    if indicators_path.exists():
        try:
            with open(indicators_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] 加载 {indicators_path} 失败: {e}")
    return None


def extract_palsy_evidence(indicators: Dict[str, Any]) -> Dict[str, Any]:
    """
    从indicators中提取面瘫检测的证据信息

    Args:
        indicators: indicators.json的内容

    Returns:
        {
            "palsy_side": int,
            "confidence": float,
            "method": str,
            "evidence": {
                "indicator_name": {
                    "raw_value": ...,
                    "threshold": ...,
                    "is_abnormal": ...,
                    "contribution": ...,
                },
                ...
            },
            "votes": [...],
            "interpretation": str,
        }
    """
    result = {
        "palsy_side": 0,
        "confidence": 0.0,
        "method": "unknown",
        "evidence": {},
        "votes": [],
        "interpretation": "",
    }

    # 尝试从不同位置提取palsy_detection
    palsy_detection = None

    if "palsy_detection" in indicators:
        palsy_detection = indicators["palsy_detection"]
    elif "action_specific" in indicators:
        action_specific = indicators["action_specific"]
        if "palsy_detection" in action_specific:
            palsy_detection = action_specific["palsy_detection"]

    if palsy_detection is None:
        return result

    result["palsy_side"] = palsy_detection.get("palsy_side", 0)
    result["confidence"] = palsy_detection.get("confidence", 0.0)
    result["method"] = palsy_detection.get("method", "unknown")
    result["interpretation"] = palsy_detection.get("interpretation", "")

    # 提取evidence
    if "evidence" in palsy_detection:
        result["evidence"] = palsy_detection["evidence"]

    # 提取votes
    if "votes" in palsy_detection:
        result["votes"] = palsy_detection["votes"]

    return result


def generate_error_report(error_cases: List[Dict[str, Any]],
                          output_path: Path,
                          action: str) -> None:
    """
    生成错误案例详细报告

    Args:
        error_cases: 错误案例列表
        output_path: 输出路径
        action: 动作名称
    """
    report_lines = []
    report_lines.append(f"# {action} 错误案例分析报告")
    report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"错误案例数: {len(error_cases)}")
    report_lines.append("")
    report_lines.append("=" * 80)

    for i, case in enumerate(error_cases, 1):
        report_lines.append("")
        report_lines.append(f"## 案例 {i}: {case['exam_id']}")
        report_lines.append("-" * 40)
        report_lines.append(
            f"GT侧别: {case['gt_side']} | 预测侧别: {case['pred_side']} | 错误类型: {case['error_type']}")
        report_lines.append("")

        # 提取证据
        evidence = case.get("evidence", {})

        report_lines.append("### 判决依据分析:")
        report_lines.append("")

        if not evidence:
            report_lines.append("  (无详细证据)")
        else:
            for indicator_name, indicator_data in evidence.items():
                report_lines.append(f"  **{indicator_name}:**")

                if isinstance(indicator_data, dict):
                    raw_value = indicator_data.get("raw_value", indicator_data.get("raw_value_px", "N/A"))
                    normalized = indicator_data.get("normalized_value", indicator_data.get("offset_norm", "N/A"))
                    threshold = indicator_data.get("threshold", "N/A")
                    is_abnormal = indicator_data.get("is_abnormal", "N/A")
                    contribution = indicator_data.get("contribution", "N/A")
                    description = indicator_data.get("description", "")

                    report_lines.append(f"    - 原始值: {raw_value}")
                    if normalized != "N/A":
                        report_lines.append(f"    - 归一化值: {normalized}")
                    report_lines.append(f"    - 阈值: {threshold}")
                    report_lines.append(f"    - 是否异常: {is_abnormal}")
                    report_lines.append(f"    - 贡献方向: {contribution}")
                    if description:
                        report_lines.append(f"    - 说明: {description}")
                else:
                    report_lines.append(f"    - 值: {indicator_data}")

                report_lines.append("")

        # 投票信息
        votes = case.get("votes", [])
        if votes:
            report_lines.append("### 投票汇总:")
            for vote in votes:
                indicator = vote.get("indicator", "?")
                side = vote.get("side", 0)
                side_str = {0: "对称", 1: "左瘫", 2: "右瘫"}.get(side, "?")
                conf = vote.get("confidence", 0)
                weight = vote.get("weight", 0)
                report_lines.append(f"    - {indicator}: {side_str} (置信度={conf:.2f}, 权重={weight:.2f})")
            report_lines.append("")

        # 解释
        interpretation = case.get("interpretation", "")
        if interpretation:
            report_lines.append(f"### 系统解释: {interpretation}")

        report_lines.append("")
        report_lines.append("=" * 80)

    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"[INFO] 错误报告已保存: {output_path}")


def analyze_threshold_sensitivity(cases: List[Dict[str, Any]],
                                  indicator_name: str,
                                  threshold_field: str = "threshold") -> Dict[str, Any]:
    """
    分析阈值敏感性 - 找出边界案例

    Args:
        cases: 所有案例
        indicator_name: 指标名称
        threshold_field: 阈值字段名

    Returns:
        {
            "threshold": float,
            "correct_below_threshold": int,
            "wrong_below_threshold": int,
            "correct_above_threshold": int,
            "wrong_above_threshold": int,
            "boundary_cases": [...]  # 值接近阈值的案例
        }
    """
    result = {
        "indicator": indicator_name,
        "threshold": None,
        "correct_below": 0,
        "wrong_below": 0,
        "correct_above": 0,
        "wrong_above": 0,
        "boundary_cases": [],
    }

    threshold = None
    values_and_results = []

    for case in cases:
        evidence = case.get("evidence", {})
        indicator_data = evidence.get(indicator_name, {})

        if not isinstance(indicator_data, dict):
            continue

        value = indicator_data.get("normalized_value", indicator_data.get("offset_norm"))
        thr = indicator_data.get(threshold_field)

        if value is None or thr is None:
            continue

        if threshold is None:
            threshold = thr

        is_correct = case.get("is_correct", case.get("error_type") == "OK")

        values_and_results.append({
            "exam_id": case.get("exam_id", "?"),
            "value": value,
            "is_correct": is_correct,
        })

    result["threshold"] = threshold

    if threshold is None:
        return result

    # 统计
    for item in values_and_results:
        if item["value"] < threshold:
            if item["is_correct"]:
                result["correct_below"] += 1
            else:
                result["wrong_below"] += 1
        else:
            if item["is_correct"]:
                result["correct_above"] += 1
            else:
                result["wrong_above"] += 1

        # 边界案例：值在阈值的±20%范围内
        if abs(item["value"] - threshold) / threshold < 0.20:
            result["boundary_cases"].append(item)

    return result


def save_action_error_analysis(src_root: Path,
                               dst_root: Path,
                               action: str,
                               exam_results: List[Dict[str, Any]]) -> None:
    """
    保存单个动作的错误分析报告

    Args:
        src_root: 源数据根目录
        dst_root: 输出根目录
        action: 动作名称
        exam_results: 该动作的所有检查结果
    """
    # 筛选错误案例
    error_cases = []
    all_cases = []

    for result in exam_results:
        exam_id = result.get("exam_id", "")
        gt_side = result.get("gt_palsy_side", 0)
        pred_side = result.get("pred_palsy_side", 0)

        # 确定错误类型
        if gt_side == pred_side:
            error_type = "OK"
            is_correct = True
        elif gt_side == 0 and pred_side != 0:
            error_type = "FP"  # 假阳性
            is_correct = False
        elif gt_side != 0 and pred_side == 0:
            error_type = "FN"  # 假阴性
            is_correct = False
        else:
            error_type = "WRONG"  # 错侧
            is_correct = False

        # 解析exam_id获取路径
        parts = exam_id.split('_')
        if len(parts) >= 2:
            patient_id, exam_code = parts[0], parts[1]
            exam_dir = src_root / patient_id / exam_code
        else:
            continue

        # 加载indicators.json
        indicators = load_indicators_json(exam_dir, action)

        # 提取证据
        evidence_data = {}
        votes = []
        interpretation = ""

        if indicators:
            palsy_info = extract_palsy_evidence(indicators)
            evidence_data = palsy_info.get("evidence", {})
            votes = palsy_info.get("votes", [])
            interpretation = palsy_info.get("interpretation", "")

        case_data = {
            "exam_id": exam_id,
            "gt_side": gt_side,
            "pred_side": pred_side,
            "error_type": error_type,
            "is_correct": is_correct,
            "evidence": evidence_data,
            "votes": votes,
            "interpretation": interpretation,
        }

        all_cases.append(case_data)

        if not is_correct:
            error_cases.append(case_data)

    # 生成错误报告
    if error_cases:
        error_report_path = dst_root / f"{action}_error_analysis.md"
        generate_error_report(error_cases, error_report_path, action)

    # 生成阈值敏感性分析（如果有嘴唇偏移指标）
    if all_cases:
        sensitivity = analyze_threshold_sensitivity(all_cases, "lip_offset")
        if sensitivity.get("threshold") is not None:
            sensitivity_path = dst_root / f"{action}_threshold_sensitivity.json"
            with open(sensitivity_path, 'w', encoding='utf-8') as f:
                json.dump(sensitivity, f, indent=2, ensure_ascii=False)
            print(f"[INFO] 阈值敏感性分析已保存: {sensitivity_path}")


def enhanced_error_analysis(src_root: Path, dst_root: Path, action_results: Dict[str, List]) -> None:
    """
    增强版错误分析 - 添加到主流程末尾

    Args:
        src_root: 源数据根目录
        dst_root: 输出根目录
        action_results: {action_name: [exam_results...]}
    """
    print("\n" + "=" * 70)
    print("开始生成错误分析报告...")
    print("=" * 70)

    for action, results in action_results.items():
        if not results:
            continue

        # 统计错误数
        error_count = sum(1 for r in results if r.get("gt_palsy_side", 0) != r.get("pred_palsy_side", 0))

        if error_count > 0:
            print(f"\n[{action}] 错误案例: {error_count}/{len(results)}")
            save_action_error_analysis(src_root, dst_root, action, results)

        # 为嘴部动作生成带evidence的CSV
        if action in ["BlowCheek", "LipPucker", "ShowTeeth", "Smile"]:
            csv_path = dst_root / f"{action}_evidence.csv"
            # 构建案例数据（需要从indicators.json读取）
            cases_with_evidence = []
            for r in results:
                exam_id = r.get("exam_id", "")
                parts = exam_id.split('_')
                if len(parts) >= 2:
                    exam_dir = src_root / parts[0] / parts[1]
                    indicators = load_indicators_json(exam_dir, action)
                    if indicators:
                        palsy_info = extract_palsy_evidence(indicators)
                        cases_with_evidence.append({
                            "exam_id": exam_id,
                            "gt_side": r.get("gt_palsy_side", 0),
                            "pred_side": r.get("pred_palsy_side", 0),
                            "error_type": "OK" if r.get("gt_palsy_side", 0) == r.get("pred_palsy_side",
                                                                                     0) else "ERROR",
                            "confidence": palsy_info.get("confidence", 0),
                            "interpretation": palsy_info.get("interpretation", ""),
                            "evidence": palsy_info.get("evidence", {}),
                        })
            if cases_with_evidence:
                save_debug_csv_with_evidence(cases_with_evidence, csv_path, action)

    print("\n" + "=" * 70)
    print("错误分析报告生成完成")
    print("=" * 70)


# =============================================================================
# 新增：生成综合调试CSV（包含evidence）
# =============================================================================

def save_debug_csv_with_evidence(results: List[Dict[str, Any]],
                                 output_path: Path,
                                 action: str) -> None:
    """
    保存包含详细evidence的调试CSV

    每行一个案例，列包括：
    - 基本信息
    - 各指标的value
    - 各指标的threshold
    - 各指标的is_abnormal
    - 最终判断
    """
    if not results:
        return

    # 收集所有可能的evidence字段
    all_evidence_keys = set()
    for r in results:
        evidence = r.get("evidence", {})
        for key in evidence.keys():
            all_evidence_keys.add(key)

    # 构建列头
    base_columns = ["exam_id", "gt_side", "pred_side", "error_type", "confidence", "interpretation"]

    evidence_columns = []
    for key in sorted(all_evidence_keys):
        evidence_columns.extend([
            f"{key}_value",
            f"{key}_threshold",
            f"{key}_is_abnormal",
            f"{key}_contribution",
        ])

    all_columns = base_columns + evidence_columns

    # 写入CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_columns)
        writer.writeheader()

        for r in results:
            row = {
                "exam_id": r.get("exam_id", ""),
                "gt_side": r.get("gt_side", ""),
                "pred_side": r.get("pred_side", ""),
                "error_type": r.get("error_type", ""),
                "confidence": r.get("confidence", ""),
                "interpretation": r.get("interpretation", ""),
            }

            # 添加evidence字段
            evidence = r.get("evidence", {})
            for key in all_evidence_keys:
                data = evidence.get(key, {})
                if isinstance(data, dict):
                    row[f"{key}_value"] = data.get("normalized_value", data.get("raw_value", ""))
                    row[f"{key}_threshold"] = data.get("threshold", "")
                    row[f"{key}_is_abnormal"] = data.get("is_abnormal", "")
                    row[f"{key}_contribution"] = data.get("contribution", "")

            writer.writerow(row)

    print(f"[INFO] 调试CSV已保存: {output_path}")


def setup_matplotlib_chinese_font():
    """
    让 matplotlib 支持中文，避免：
    Glyph xxxx missing from font(s) DejaVu Sans
    """
    import matplotlib
    # 重要：先设 backend，再 import pyplot（如果你在其他地方已经 import pyplot，就把这一行去掉）
    try:
        matplotlib.use("Agg")
    except Exception:
        pass

    # Mac / Windows / Linux 常见中文字体候选
    preferred = [
        "PingFang SC",        # macOS
        "Heiti SC",           # macOS
        "STHeiti",            # macOS
        "Microsoft YaHei",    # Windows
        "SimHei",             # Windows
        "Noto Sans CJK SC",   # Linux 常用
        "Source Han Sans SC", # 思源黑体
        "Arial Unicode MS",   # 兼容性字体(不一定有)
    ]

    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name

    # 找不到就退回默认（仍会有警告，但不中断）
    matplotlib.rcParams["axes.unicode_minus"] = False
    return None

# =============================================================================
# 主函数
# =============================================================================
def main():
    font_used = setup_matplotlib_chinese_font()
    print(f"[INFO] Matplotlib font: {font_used}")
    if not SRC_ROOT.exists():
        raise FileNotFoundError(f"SRC_ROOT 不存在: {SRC_ROOT}")

    ensure_dirs()
    total_exam = 0
    total_copied = 0
    total_skipped = 0
    all_palsy_records = []
    all_session_records = []
    action_results = {action: [] for action in ACTIONS}

    print("=" * 60)
    print("开始收集关键帧和面瘫预测统计")
    print("=" * 60)

    for exam_dir in find_exam_dirs(SRC_ROOT):
        total_exam += 1
        c, s, records, session_record = collect_one_exam(exam_dir)
        total_copied += c
        total_skipped += s
        all_palsy_records.extend(records)

        # 按动作分类收集结果（用于错误分析）
        for record in records:
            action_name = record.get("action", "")
            if action_name in action_results:
                action_results[action_name].append(record)

        # 收集Session诊断记录
        if session_record:
            all_session_records.append(session_record)

        if total_exam % 20 == 0:
            print(f"[INFO] exams={total_exam}, records={len(all_palsy_records)}, sessions={len(all_session_records)}")

    print(f"\n图片收集: exams={total_exam}, copied={total_copied}, skipped={total_skipped}")
    print(f"Session诊断记录: {len(all_session_records)}")

    # ========== 动作级统计 ==========
    print("\n" + "=" * 60)
    print("计算动作级面瘫侧别预测统计...")
    print("=" * 60)

    stats = compute_statistics(all_palsy_records)

    # 打印统计摘要
    print(f"\n{'Action':<20} {'Total':>5} {'OK':>5} {'WRONG':>6} {'FN':>4} {'FP':>4} {'Strict':>8} {'Relaxed':>8}")
    print("-" * 65)

    for action in ACTIONS:
        if action in stats:
            s = stats[action]
            strict_acc = f"{s['side_accuracy']:.1%}" if s['side_accuracy'] is not None else "N/A"
            relaxed_acc = f"{s['relaxed_accuracy']:.1%}" if s['relaxed_accuracy'] is not None else "N/A"
            print(
                f"{action:<20} {s['total']:>5} {s['OK']:>5} {s['WRONG']:>6} {s['FN']:>4} {s['FP']:>4} {strict_acc:>8} {relaxed_acc:>8}")

    # 打印汇总
    print("-" * 70)
    total_ok = sum(stats[a]["OK"] for a in stats if a != "NeutralFace")
    total_wrong = sum(stats[a]["WRONG"] for a in stats if a != "NeutralFace")
    total_fn = sum(stats[a]["FN"] for a in stats if a != "NeutralFace")
    total_palsy = total_ok + total_wrong + total_fn
    total_fp = sum(stats[a]["FP"] for a in stats if a != "NeutralFace")

    if total_palsy > 0:
        overall_strict = total_ok / total_palsy
        overall_relaxed = (total_ok + total_fn) / total_palsy
        print(
            f"{'TOTAL (excl.Neutral)':<20} {'-':>5} {total_ok:>5} {total_wrong:>6} {total_fn:>4} {overall_strict:>7.1%} {overall_relaxed:>8.1%}")

    print("\n说明:")
    print("  Strict  = OK / (OK + WRONG + FN)  -- 严格匹配")
    print("  Relaxed = (OK + FN) / (OK + WRONG + FN)  -- 宽松匹配 (预测对称也算正确)")

    # ========== Session级诊断统计（新增）==========
    if all_session_records:
        session_stats = compute_session_diagnosis_statistics(all_session_records)
        print_session_diagnosis_statistics(all_session_records)

        # 保存Session诊断结果
        save_session_diagnosis_csv(all_session_records, DST_ROOT / "session_diagnosis.csv")
        save_session_diagnosis_summary(session_stats, DST_ROOT / "session_diagnosis_summary.txt")
        save_hb_trace_csv(all_session_records, DST_ROOT / "hb_trace.csv")
        save_hb_trace_error_json(all_session_records, DST_ROOT / "hb_trace_error_cases.json")

    # 保存动作级文件
    save_by_action_csv(all_palsy_records, DST_ROOT)
    save_all_records_csv(all_palsy_records, DST_ROOT / "palsy_all_records.csv")
    save_summary_csv(stats, DST_ROOT / "palsy_summary.csv")
    save_confusion_matrix(all_palsy_records, DST_ROOT / "palsy_confusion.txt")
    save_error_analysis(all_palsy_records, DST_ROOT / "error_cases")

    # 尝试从数据库加载GT severity标签
    labels_db = Path("/Users/cuijinglei/PycharmProjects/medicalProject/facial_palsy/facialPalsy.db")  # 修改为实际路径
    if labels_db.exists():
        save_severity_stats(all_palsy_records, DST_ROOT / "severity_summary.csv", str(labels_db))
    else:
        save_severity_stats(all_palsy_records, DST_ROOT / "severity_summary.csv")

    classified_dir = DST_ROOT / "classified_images"
    copy_classified_images(all_palsy_records, classified_dir)

    enhanced_error_analysis(SRC_ROOT, DST_ROOT, action_results)

    # ========== 阈值分析图 ==========
    print("\n生成阈值分析图...")
    plot_all_actions_threshold_analysis(all_palsy_records, DST_ROOT)

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)
    print(f"输出目录: {DST_ROOT}")


if __name__ == "__main__":
    main()