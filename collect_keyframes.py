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

运行方式：PyCharm 直接点运行即可
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple
import shutil
from collections import Counter

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
# 各动作的关键指标Schema (用于CSV列排序)
# =============================================================================
ACTION_METRICS_SCHEMA = {
    "NeutralFace": [
        # NeutralFace不做面瘫检测,主要是基线指标
        "left_ear", "right_ear", "ear_ratio",
        "left_palpebral_height", "right_palpebral_height", "palpebral_ratio",
        "left_nlf_length", "right_nlf_length", "nlf_ratio",
        "oral_angle_diff"
    ],

    "SpontaneousEyeBlink": [
        "method", "palsy_side", "confidence",
        "left_closure_pct", "right_closure_pct",
        "persistent_asymmetry_side", "left_worse_frame_ratio", "right_worse_frame_ratio",
        "asymmetry_ratio", "status"
    ],

    "VoluntaryEyeBlink": [
        "method", "palsy_side", "confidence",
        "left_closure_pct", "right_closure_pct",
        "persistent_asymmetry_side", "left_worse_frame_ratio", "right_worse_frame_ratio",
        "asymmetry_ratio", "status"
    ],

    "CloseEyeSoftly": [
        "method", "palsy_side", "confidence",
        "left_ear", "right_ear",
        "baseline_left_area", "baseline_right_area",
        "left_closure_pct", "right_closure_pct",
        "asymmetry_ratio", "status"
    ],

    "CloseEyeHardly": [
        "method", "palsy_side", "confidence",
        "left_ear", "right_ear",
        "baseline_left_area", "baseline_right_area",
        "left_closure_pct", "right_closure_pct",
        "asymmetry_ratio", "status"
    ],

    "RaiseEyebrow": [
        "method", "palsy_side", "confidence",
        "left_change", "right_change",
        "asymmetry_ratio", "brow_eye_distance_ratio",
        "status"
    ],

    "Smile": [
        "method", "palsy_side", "confidence",
        # 嘴唇中线对称性(来自lip_symmetry)
        "lip_left_to_midline", "lip_right_to_midline",
        "lip_offset", "lip_asymmetry_ratio", "lip_symmetry_ratio",
        # 口角角度
        "AOE_right", "BOF_left", "angle_diff",
        # 嘴角上提幅度(eye_line_excursion,如果有基线)
        "left_reduction", "right_reduction", "reduction_asymmetry",
        "status"
    ],

    "ShowTeeth": [
        "method", "palsy_side", "confidence",
        # 嘴唇中线偏移（主要指标）
        "face_midline_x", "lip_midline_x",
        "current_offset", "baseline_offset",
        "offset_change", "offset_change_norm",
        # 口角角度（备用指标）
        "AOE_right", "BOF_left", "angle_diff",
        # 嘴角位移（参考指标）
        "left_excursion", "right_excursion", "excursion_ratio",
        "status"
    ],

    "ShrugNose": [
        "method", "palsy_side", "confidence",
        # 鼻翼到眼部水平线的距离
        "left_ala_to_eye_line", "right_ala_to_eye_line",
        "distance_diff", "tilt_angle",
        # 鼻翼-内眦距离变化(如果有基线)
        "left_reduction", "right_reduction", "reduction_asymmetry",
        "asymmetry_ratio", "status"
    ],

    "BlowCheek": [
        "method", "palsy_side", "confidence",
        # 脸颊深度变化
        "cheek_left_delta", "cheek_right_delta", "cheek_asymmetry",
        # 嘴唇中线对称性(来自lip_symmetry)
        "lip_left_to_midline", "lip_right_to_midline",
        "lip_offset", "lip_asymmetry_ratio",
        # 口角角度
        "AOE_right", "BOF_left", "angle_diff",
        # 门控状态
        "seal_valid", "mouth_valid", "inner_area_valid",
        "status"
    ],

    "LipPucker": [
        "method", "palsy_side", "confidence",
        # 嘴唇中线偏移变化(来自lip_midline_offset,如果有基线)
        "current_offset", "baseline_offset", "offset_change", "offset_change_norm",
        # 嘴唇中线对称性(来自lip_symmetry)
        "lip_left_to_midline", "lip_right_to_midline",
        "lip_offset", "lip_asymmetry_ratio",
        # 口角角度
        "AOE_right", "BOF_left", "angle_diff",
        # 嘴宽变化
        "width_ratio", "width_change_percent",
        "status"
    ],
}

# 各动作的对称阈值 (用于边界案例识别)
SYMMETRY_THRESHOLDS = {
    "NeutralFace": {"ear_ratio_dev": 0.15, "palpebral_ratio_dev": 0.15, "nlf_ratio_dev": 0.15},
    "SpontaneousEyeBlink": {"asymmetry_ratio": 0.15, "persistent_ratio": 0.60},
    "VoluntaryEyeBlink": {"asymmetry_ratio": 0.15, "persistent_ratio": 0.60},
    "CloseEyeSoftly": {"asymmetry_ratio": 0.15, "min_closure": 0.25},
    "CloseEyeHardly": {"asymmetry_ratio": 0.15, "min_closure": 0.25},
    "RaiseEyebrow": {"asymmetry_ratio": 0.15, "min_change": 2.0, "bed_ratio_dev": 0.10},
    "Smile": {"asymmetry_ratio": 0.15, "angle_diff": 3.0, "min_reduction": 5.0},
    "ShowTeeth": {"offset_change_norm": 0.02, "angle_diff": 3.0, "min_offset_change": 5.0},
    "ShrugNose": {"asymmetry_ratio": 0.15, "tilt_angle": 2.0, "min_reduction": 3.0},
    "BlowCheek": {"cheek_asymmetry": 0.15, "angle_diff": 3.0, "min_delta": 0.02},
    "LipPucker": {"asymmetry_ratio": 0.08, "angle_diff": 3.0, "min_offset_change": 0.02},
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
        }
    except Exception as e:
        print(f"[WARN] Failed to load summary.json: {exam_dir} - {e}")
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

        # 展平evidence字段
        evidence = palsy_detection.get("evidence", {})
        detailed_metrics = flatten_evidence(evidence)
        result["detailed_metrics"] = detailed_metrics

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
        # ShowTeeth使用嘴唇中线偏移
        if action == "ShowTeeth" and method == "lip_midline_offset_change":
            current = metrics.get("current_offset", 0)
            baseline = metrics.get("baseline_offset", 0)
            change = metrics.get("offset_change", 0)
            change_norm = metrics.get("offset_change_norm", 0)
            return f"中线偏移: 当前={current:+.1f}px 基线={baseline:+.1f}px 变化={change:+.1f}px ({change_norm:.1%})"
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


def identify_boundary_case(action: str, metrics: Dict[str, Any]) -> Optional[str]:
    """识别阈值边界案例"""
    thresholds = SYMMETRY_THRESHOLDS.get(action, {})
    boundary_notes = []

    # 检查asymmetry_ratio
    asym = metrics.get("asymmetry_ratio")
    asym_thr = thresholds.get("asymmetry_ratio")
    if asym is not None and asym_thr is not None:
        margin = abs(asym - asym_thr) / asym_thr
        if margin < 0.3:  # 30%以内
            boundary_notes.append(f"asym={asym:.3f}≈thr={asym_thr}")

    # 检查angle_diff
    angle_diff = metrics.get("angle_diff")
    angle_thr = thresholds.get("angle_diff")
    if angle_diff is not None and angle_thr is not None:
        margin = abs(angle_diff - angle_thr) / angle_thr
        if margin < 0.3:
            boundary_notes.append(f"angle_diff={angle_diff:.1f}≈thr={angle_thr}")

    # 检查tilt_angle
    tilt = metrics.get("tilt_angle")
    tilt_thr = thresholds.get("tilt_angle")
    if tilt is not None and tilt_thr is not None:
        margin = abs(abs(tilt) - tilt_thr) / tilt_thr
        if margin < 0.3:
            boundary_notes.append(f"|tilt|={abs(tilt):.1f}≈thr={tilt_thr}")

    # 检查offset_change_norm (ShowTeeth/BlowCheek/LipPucker专用)
    if action in ["ShowTeeth", "BlowCheek", "LipPucker"]:
        offset_norm = metrics.get("offset_change_norm")
        offset_thr = thresholds.get("offset_change_norm", 0.02)
        if offset_norm is not None and offset_thr > 0:
            margin = abs(offset_norm - offset_thr) / offset_thr
            if margin < 0.5:  # 50%以内认为边界
                boundary_notes.append(f"offset_norm={offset_norm:.3f}≈thr={offset_thr}")

    return "; ".join(boundary_notes) if boundary_notes else None


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
        detailed_metrics = prediction.get("detailed_metrics", {})
        eval_result = evaluate_prediction(gt_side, pred_side)

        # 生成指标摘要
        metrics_summary = format_metrics_summary(action_name, detailed_metrics)

        # 识别边界案例
        boundary_note = identify_boundary_case(action_name, detailed_metrics)

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
            "boundary_note": boundary_note,
            "detailed_metrics": detailed_metrics,
            "severity_score": prediction.get("severity_score"),
            "severity_desc": prediction.get("severity_desc", ""),
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
        lines.extend([f"\n\n{'='*40}", f"【{error_type}】 共 {len(cases)} 例", "=" * 40])

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
            if action == "NeutralFace":
                continue

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
            if action == "NeutralFace":
                continue

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
        if action == "NeutralFace":
            continue

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
        if action == "NeutralFace":
            continue
        for cat in categories:
            cat_dir = output_dir / action / cat
            cat_dir.mkdir(parents=True, exist_ok=True)

    # 统计复制数量
    copy_stats = defaultdict(lambda: defaultdict(int))

    # 复制图片
    for r in records:
        action = r["action"]
        if action == "NeutralFace":
            continue

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

        # 复制图片
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

    # 打印统计
    total_copied = sum(sum(v.values()) for v in copy_stats.values())
    print(f"\n[OK] Classified images copied: {total_copied * 2} files to {output_dir}")

    print(f"\n分类图片统计:")
    print(f"{'Action':<20} {'OK':>5} {'WRONG':>6} {'FN':>4} {'FP':>4} {'both_sym':>8}")
    print("-" * 52)
    for action in ACTIONS:
        if action == "NeutralFace":
            continue
        stats = copy_stats.get(action, {})
        if sum(stats.values()) > 0:
            print(f"{action:<20} {stats.get('OK', 0):>5} {stats.get('WRONG', 0):>6} "
                  f"{stats.get('FN', 0):>4} {stats.get('FP', 0):>4} {stats.get('both_sym', 0):>8}")


def print_threshold_reference():
    """打印阈值参考"""
    print("\n" + "=" * 60)
    print("阈值参考 (各动作对称性判断阈值)")
    print("=" * 60)
    for action, thresholds in SYMMETRY_THRESHOLDS.items():
        thr_str = ", ".join([f"{k}={v}" for k, v in thresholds.items()])
        print(f"  {action}: {thr_str}")
    print()


# =============================================================================
# 主函数
# =============================================================================
def main():
    if not SRC_ROOT.exists():
        raise FileNotFoundError(f"SRC_ROOT 不存在: {SRC_ROOT}")

    ensure_dirs()
    print_threshold_reference()

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
    print(f"\n{'Action':<20} {'Total':>5} {'OK':>5} {'WRONG':>6} {'FN':>4} {'Strict':>8} {'Relaxed':>8}")
    print("-" * 65)

    for action in ACTIONS:
        if action in stats:
            s = stats[action]
            strict_acc = f"{s['side_accuracy']:.1%}" if s['side_accuracy'] is not None else "N/A"
            relaxed_acc = f"{s['relaxed_accuracy']:.1%}" if s['relaxed_accuracy'] is not None else "N/A"
            print(f"{action:<20} {s['total']:>5} {s['OK']:>5} {s['WRONG']:>6} {s['FN']:>4} {strict_acc:>8} {relaxed_acc:>8}")

    # 打印汇总
    print("-" * 65)
    total_ok = sum(stats[a]["OK"] for a in stats if a != "NeutralFace")
    total_wrong = sum(stats[a]["WRONG"] for a in stats if a != "NeutralFace")
    total_fn = sum(stats[a]["FN"] for a in stats if a != "NeutralFace")
    total_palsy = total_ok + total_wrong + total_fn

    if total_palsy > 0:
        overall_strict = total_ok / total_palsy
        overall_relaxed = (total_ok + total_fn) / total_palsy
        print(f"{'TOTAL (excl.Neutral)':<20} {'-':>5} {total_ok:>5} {total_wrong:>6} {total_fn:>4} {overall_strict:>7.1%} {overall_relaxed:>8.1%}")

    print("\n说明:")
    print("  Strict  = OK / (OK + WRONG + FN)  -- 严格匹配")
    print("  Relaxed = (OK + FN) / (OK + WRONG + FN)  -- 宽松匹配 (预测对称也算正确)")

    # 保存文件
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

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)
    print(f"输出目录: {DST_ROOT}")
    print(f"  - palsy_<Action>.csv      : 每个动作的基本预测")
    print(f"  - palsy_all_records.csv   : 所有记录汇总")
    print(f"  - palsy_summary.csv       : 统计摘要")
    print(f"  - palsy_confusion.txt     : 混淆矩阵")
    print(f"  - error_cases/            : 错误案例分析")
    print(f"  - severity_summary.csv    : 严重度分数统计")
    print(f"  - classified_images/      : 按结果分类的关键帧图片")


if __name__ == "__main__":
    main()