#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RaiseEyebrow 动作处理模块
=========================

核心改进:
=========
1. 面瘫侧别判断：不需要基线
   - 直接比较当前帧的左右眉眼距
   - 眉眼距小的那侧 = 患侧（抬不起来）
   - 使用归一化不对称度阈值

2. 严重程度判断：需要基线
   - 比较患侧眉毛抬起的幅度
   - 幅度越小 = 越严重

分析抬眉/皱额动作:
1. 眉眼距
2. 眉眼距变化度
3. 双侧眉眼距比和变化度比
4. 联动运动检测

对应Sunnybrook: Brow (Forehead wrinkle)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from clinical_base import (
    LM, pt2d, pts2d, dist, compute_ear, compute_eye_area,
    compute_brow_height, compute_brow_position, compute_mouth_metrics,
    compute_icd, extract_common_indicators,
    compute_brow_eye_distance, compute_brow_eye_distance_ratio,
    compute_brow_eye_distance_change, compute_brow_eye_distance_change_ratio,
    compute_brow_centroid, compute_scale_to_baseline,
    ActionResult, draw_polygon, draw_landmarks,
    add_valid_region_shading, get_palsy_side_text,
    draw_palsy_side_label, draw_palsy_annotation_header
)

from thresholds import THR

ACTION_NAME = "RaiseEyebrow"
ACTION_NAME_CN = "抬眉/皱额"

# OpenCV字体
FONT = cv2.FONT_HERSHEY_SIMPLEX

# 字体大小
FONT_SCALE_TITLE = 1.4      # 标题
FONT_SCALE_LARGE = 1.2      # 大号文字
FONT_SCALE_NORMAL = 0.9     # 正常文字
FONT_SCALE_SMALL = 0.7      # 小号文字

# 线条粗细
THICKNESS_TITLE = 3
THICKNESS_NORMAL = 2
THICKNESS_THIN = 1

# 行高
LINE_HEIGHT = 45
LINE_HEIGHT_SMALL = 30

def find_peak_frame(
    landmarks_seq: List,
    frames_seq: List,
    w: int,
    h: int,
    baseline_landmarks=None,  # 保留参数但不使用
    smooth_win: int = 5,
) -> Tuple[int, Dict[str, Any]]:
    """
    找抬眉峰值帧 - 眉眼距最大

    选择逻辑：
    - 抬眉时眉毛上抬，眉眼距增大
    - 选择眉眼距（左右平均）最大的帧
    - 不和静息帧基准做比较
    """
    n = len(landmarks_seq)
    if n == 0:
        return 0, {}

    left_series = [None] * n
    right_series = [None] * n
    mean_series = [None] * n

    left_raw = [None] * n
    right_raw = [None] * n
    mean_raw = [None] * n

    metric_name = "BED_abs_px"  # 绝对眉眼距

    best_val = -1e18
    best_idx = 0

    for i, lm in enumerate(landmarks_seq):
        if lm is None:
            continue

        try:
            # 始终使用绝对值（不和基线比较）
            bed = compute_brow_eye_distance_ratio(lm, w, h)
            lv = float(bed["left_distance"])
            rv = float(bed["right_distance"])

            mv = 0.5 * (lv + rv)

            left_raw[i] = lv
            right_raw[i] = rv
            mean_raw[i] = mv

            left_series[i] = lv
            right_series[i] = rv
            mean_series[i] = mv

            # 选择眉眼距最大的帧
            if mv > best_val:
                best_val = mv
                best_idx = i
        except Exception:
            continue

    # 平滑处理
    def _smooth(arr: List[Optional[float]], win: int) -> List[Optional[float]]:
        if win <= 1:
            return arr
        half = win // 2
        out = [None] * len(arr)
        for k in range(len(arr)):
            a = max(0, k - half)
            b = min(len(arr), k + half + 1)
            seg = [v for v in arr[a:b] if v is not None]
            out[k] = float(sum(seg) / len(seg)) if seg else None
        return out

    left_s = _smooth(left_series, smooth_win)
    right_s = _smooth(right_series, smooth_win)
    mean_s = _smooth(mean_series, smooth_win)

    # 平滑后重新选峰值
    best_val2 = -1e18
    best_idx2 = best_idx
    for i, v in enumerate(mean_s):
        if v is None:
            continue
        if v > best_val2:
            best_val2 = v
            best_idx2 = i

    peak_idx = int(best_idx2)

    peak_debug = {
        "metric": metric_name,
        "smooth_win": int(smooth_win),
        "left_curve": left_s,
        "right_curve": right_s,
        "mean_curve": mean_s,
        "left_raw": left_raw,
        "right_raw": right_raw,
        "mean_raw": mean_raw,
        "peak_idx": peak_idx,
        "peak_value": float(mean_s[peak_idx]) if mean_s[peak_idx] is not None else None,
        "selection_criterion": "max_brow_eye_distance",
    }

    return peak_idx, peak_debug


def extract_raise_eyebrow_sequences(
        landmarks_seq: List,
        w: int, h: int,
        baseline_landmarks=None
) -> Dict[str, List[float]]:
    """
    提取抬眉关键指标的时序序列

    Returns:
        包含眉眼距和变化度的时序数据
    """
    left_bed_seq = []
    right_bed_seq = []
    avg_bed_seq = []
    left_change_seq = []
    right_change_seq = []

    for lm in landmarks_seq:
        if lm is None:
            left_bed_seq.append(np.nan)
            right_bed_seq.append(np.nan)
            avg_bed_seq.append(np.nan)
            left_change_seq.append(np.nan)
            right_change_seq.append(np.nan)
        else:
            bed_result = compute_brow_eye_distance_ratio(lm, w, h)
            left_bed_seq.append(bed_result["left_distance"])
            right_bed_seq.append(bed_result["right_distance"])
            avg_bed_seq.append((bed_result["left_distance"] + bed_result["right_distance"]) / 2)

            if baseline_landmarks is not None:
                left_change = compute_brow_eye_distance_change(lm, baseline_landmarks, w, h, left=True)
                right_change = compute_brow_eye_distance_change(lm, baseline_landmarks, w, h, left=False)
                left_change_seq.append(left_change["change"])
                right_change_seq.append(right_change["change"])
            else:
                left_change_seq.append(np.nan)
                right_change_seq.append(np.nan)

    return {
        "Left BED": left_bed_seq,
        "Right BED": right_bed_seq,
        "Average BED": avg_bed_seq,
        "Left Change": left_change_seq,
        "Right Change": right_change_seq,
    }


def plot_raise_eyebrow_peak_selection(
        sequences: Dict[str, List[float]],
        fps: float,
        peak_idx: int,
        output_path: Path,
        has_baseline: bool = False,
        valid_mask: List[bool] = None,
        palsy_detection: Dict[str, Any] = None
) -> None:
    """
    绘制抬眉关键帧选择的可解释性曲线

    抬眉选择标准:
    - 有基线时：眉眼距变化度最大
    - 无基线时：眉眼距绝对值最大
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_frames = len(sequences["Left BED"])
    frames = np.arange(n_frames)
    time_sec = frames / fps if fps > 0 else frames
    x_label = 'Time (seconds)' if fps > 0 else 'Frame'
    peak_time = peak_idx / fps if fps > 0 else peak_idx

    if has_baseline:
        fig, axes = plt.subplots(2, 1, figsize=(16, 9))

        # 上图: 眉眼距变化曲线 (关键帧选择依据)
        ax1 = axes[0]
        if valid_mask is not None:
            add_valid_region_shading(ax1, valid_mask, time_sec)

        left_change = np.array(sequences["Left Change"])
        right_change = np.array(sequences["Right Change"])

        ax1.plot(time_sec, left_change, 'b-', label='Left BED Change', linewidth=2)
        ax1.plot(time_sec, right_change, 'r-', label='Right BED Change', linewidth=2)
        ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax1.axvline(x=peak_time, color='green', linestyle='--', linewidth=2, alpha=0.7)

        if 0 <= peak_idx < n_frames:
            ax1.scatter([peak_time], [left_change[peak_idx]], color='blue', s=150, zorder=5,
                        edgecolors='black', linewidths=1.5, marker='*')
            ax1.scatter([peak_time], [right_change[peak_idx]], color='red', s=150, zorder=5,
                        edgecolors='black', linewidths=1.5, marker='*')

        title = "RaiseEyebrow: Brow-Eye Distance Change (Selection Criterion)"
        if palsy_detection:
            palsy_text = get_palsy_side_text(palsy_detection.get("palsy_side", 0))
            title += f' | Detected: {palsy_text}'

        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_xlabel(x_label, fontsize=11)
        ax1.set_ylabel('BED Change (pixels)', fontsize=11)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.4)

        # 下图: 眉眼距绝对值
        ax2 = axes[1]
        left_bed = np.array(sequences["Left BED"])
        right_bed = np.array(sequences["Right BED"])

        ax2.plot(time_sec, left_bed, 'b--', label='Left BED (absolute)', linewidth=1.5, alpha=0.7)
        ax2.plot(time_sec, right_bed, 'r--', label='Right BED (absolute)', linewidth=1.5, alpha=0.7)
        ax2.axvline(x=peak_time, color='green', linestyle='--', linewidth=2, alpha=0.7)

        ax2.set_title('Brow-Eye Distance (Reference)', fontsize=12)
        ax2.set_xlabel(x_label, fontsize=11)
        ax2.set_ylabel('BED (pixels)', fontsize=11)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.4)

    else:
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))

        if valid_mask is not None:
            add_valid_region_shading(ax, valid_mask, time_sec)

        left_bed = np.array(sequences["Left BED"])
        right_bed = np.array(sequences["Right BED"])
        avg_bed = np.array(sequences["Average BED"])

        ax.plot(time_sec, left_bed, 'b-', label='Left BED', linewidth=2)
        ax.plot(time_sec, right_bed, 'r-', label='Right BED', linewidth=2)
        ax.plot(time_sec, avg_bed, 'g--', label='Average BED (Selection)', linewidth=2.5)
        ax.axvline(x=peak_time, color='black', linestyle='--', linewidth=2, alpha=0.7)

        if 0 <= peak_idx < n_frames and np.isfinite(avg_bed[peak_idx]):
            ax.scatter([peak_time], [avg_bed[peak_idx]], color='red', s=200, zorder=5,
                       edgecolors='black', linewidths=2, marker='*', label=f'Peak Frame {peak_idx}')

        title = "RaiseEyebrow: Brow-Eye Distance (Selection: Max Average)"
        if palsy_detection:
            palsy_text = get_palsy_side_text(palsy_detection.get("palsy_side", 0))
            title += f' | Detected: {palsy_text}'

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel('BED (pixels)', fontsize=11)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()


def compute_raise_eyebrow_metrics(landmarks, w: int, h: int, baseline_landmarks=None) -> Dict[str, Any]:
    """
    计算抬眉动作指标

    包含:
    - 眉眼距（像素和归一化）
    - 眉眼距比（左/右）
    - 眉眼距变化（需要基线）
    """
    # 眉眼距比
    bed_result = compute_brow_eye_distance_ratio(landmarks, w, h)

    # 获取眉毛和眼部关键点
    left_eye_inner = pt2d(landmarks[LM.EYE_INNER_L], w, h)
    right_eye_inner = pt2d(landmarks[LM.EYE_INNER_R], w, h)
    left_brow_centroid = compute_brow_centroid(landmarks, w, h, left=True)
    right_brow_centroid = compute_brow_centroid(landmarks, w, h, left=False)

    # 计算ICD用于归一化
    icd = compute_icd(landmarks, w, h)
    icd = max(icd, 1e-6)

    # 归一化眉眼距
    left_bed_norm = bed_result["left_distance"] / icd
    right_bed_norm = bed_result["right_distance"] / icd

    metrics = {
        # 原始眉眼距（像素）
        "left_brow_eye_distance": bed_result["left_distance"],
        "right_brow_eye_distance": bed_result["right_distance"],
        "brow_eye_distance_ratio": bed_result["ratio"],
        # 归一化眉眼距（相对于ICD）
        "left_bed_norm": float(left_bed_norm),
        "right_bed_norm": float(right_bed_norm),
        "icd": float(icd),
        # 关键点位置
        "left_eye_inner": left_eye_inner,
        "right_eye_inner": right_eye_inner,
        "left_brow_centroid": left_brow_centroid,
        "right_brow_centroid": right_brow_centroid,
    }

    # 如果有基线，计算变化
    if baseline_landmarks is not None:
        baseline_bed = compute_brow_eye_distance_ratio(baseline_landmarks, w, h)
        baseline_icd = compute_icd(baseline_landmarks, w, h)

        # 尺度校正
        scale = compute_scale_to_baseline(landmarks, baseline_landmarks, w, h)
        metrics["scale"] = scale

        # 使用尺度校正后的值计算变化
        left_scaled = bed_result["left_distance"] * scale
        right_scaled = bed_result["right_distance"] * scale

        left_change = left_scaled - baseline_bed["left_distance"]
        right_change = right_scaled - baseline_bed["right_distance"]

        metrics["left_change"] = left_change
        metrics["right_change"] = right_change

        # 变化比值
        if abs(right_change) > 1e-9:
            metrics["change_ratio"] = left_change / right_change
        else:
            metrics["change_ratio"] = 1.0 if abs(left_change) < 1e-9 else float('inf')

        metrics["left_baseline_distance"] = baseline_bed["left_distance"]
        metrics["right_baseline_distance"] = baseline_bed["right_distance"]

        # 变化百分比
        if baseline_bed["left_distance"] > 1e-9:
            metrics["left_change_percent"] = left_change / baseline_bed["left_distance"] * 100
        else:
            metrics["left_change_percent"] = 0

        if baseline_bed["right_distance"] > 1e-9:
            metrics["right_change_percent"] = right_change / baseline_bed["right_distance"] * 100
        else:
            metrics["right_change_percent"] = 0

    return metrics


def detect_palsy_side(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    抬眉动作面瘫侧别检测 - 不需要基线

    核心逻辑:
    =========
    抬眉时，患侧眉毛无法正常上抬，导致眉眼距较小。
    因此：眉眼距小的那侧 = 患侧

    判断方式:
    - 直接比较当前帧的左右眉眼距（归一化到ICD）
    - 不需要与静息基线比较
    - 使用不对称度阈值判断是否异常

    阈值:
    - RAISE_EYEBROW_ASYM_NORMAL: 不对称度 < % 视为正常
    """
    result = {
        "palsy_side": 0,          # 0=对称, 1=左患侧, 2=右患侧
        "confidence": 0.0,
        "interpretation": "",
        "method": "bed_asymmetry",
        "evidence": {},
    }

    # 获取归一化眉眼距
    left_bed_norm = metrics.get("left_bed_norm")
    right_bed_norm = metrics.get("right_bed_norm")

    # 如果没有归一化值，使用原始值计算
    if left_bed_norm is None or right_bed_norm is None:
        left_bed = metrics.get("left_brow_eye_distance", 0)
        right_bed = metrics.get("right_brow_eye_distance", 0)
        icd = metrics.get("icd", 1)
        if icd > 1e-6:
            left_bed_norm = left_bed / icd
            right_bed_norm = right_bed / icd
        else:
            result["interpretation"] = "无法计算眉眼距"
            return result

    # 计算不对称度
    max_bed = max(left_bed_norm, right_bed_norm)
    min_bed = min(left_bed_norm, right_bed_norm)

    if max_bed < 0.004:
        result["interpretation"] = "眉眼距过小，无法判断"
        return result

    # 不对称度 = |左-右| / max
    asymmetry = abs(left_bed_norm - right_bed_norm) / max_bed

    # 阈值
    threshold = THR.RAISE_EYEBROW_ASYM_NORMAL

    # 记录证据
    result["evidence"] = {
        "left_bed_norm": float(left_bed_norm),
        "right_bed_norm": float(right_bed_norm),
        "asymmetry": float(asymmetry),
        "threshold": float(threshold),
        "logic": "眉眼距小的一侧 = 患侧（抬不起来）",
    }

    # 判断
    is_abnormal = asymmetry > threshold

    if not is_abnormal:
        result["palsy_side"] = 0
        result["confidence"] = 0.0
        result["interpretation"] = (
            f"双侧眉眼距对称 (L={left_bed_norm:.3f}, R={right_bed_norm:.3f}, "
            f"不对称{asymmetry:.4%} ≤ {threshold:.4%})"
        )
        return result

    # 计算置信度
    conf = min(1.0, (asymmetry - threshold) / max(threshold, 1e-6))
    result["confidence"] = float(conf)

    # 眉眼距小的那侧是患侧
    if left_bed_norm < right_bed_norm:
        result["palsy_side"] = 1
        result["interpretation"] = (
            f"左侧面瘫：左眉眼距较小 (L={left_bed_norm:.3f} < R={right_bed_norm:.3f}, "
            f"不对称{asymmetry:.4%} > {threshold:.4%})"
        )
    else:
        result["palsy_side"] = 2
        result["interpretation"] = (
            f"右侧面瘫：右眉眼距较小 (R={right_bed_norm:.3f} < L={left_bed_norm:.3f}, "
            f"不对称{asymmetry:.4%} > {threshold:.4%})"
        )

    return result


def compute_severity_score(metrics: Dict[str, Any]) -> Tuple[int, str]:
    """
    计算严重度分数 - 需要基线

    核心逻辑:
    =========
    基于患侧眉毛抬起的幅度（变化量）
    - 幅度越小 = 越严重

    需要有基线才能计算变化量。
    如果没有基线，则使用静态不对称度作为备选。
    """
    # 检查是否有变化量数据（需要基线）
    left_change = metrics.get("left_change")
    right_change = metrics.get("right_change")

    if left_change is not None and right_change is not None:
        # 有基线：使用变化量评估严重度
        abs_left = abs(left_change)
        abs_right = abs(right_change)
        max_change = max(abs_left, abs_right)
        min_change = min(abs_left, abs_right)

        # 检查是否有足够的运动
        if max_change < 3.0:
            return 5, f"几乎无抬眉动作 (L变化={left_change:.1f}px, R变化={right_change:.1f}px)"

        # 计算对称性比值（较小/较大）
        symmetry_ratio = min_change / max_change if max_change > 0 else 1.0

        # 根据对称性评估严重度
        if symmetry_ratio >= 0.90:
            return 1, f"正常 (对称性{symmetry_ratio:.1%}, L={left_change:+.1f}px, R={right_change:+.1f}px)"
        elif symmetry_ratio >= 0.72:
            return 2, f"轻度 (对称性{symmetry_ratio:.1%})"
        elif symmetry_ratio >= 0.50:
            return 3, f"中度 (对称性{symmetry_ratio:.1%})"
        elif symmetry_ratio >= 0.30:
            return 4, f"重度 (对称性{symmetry_ratio:.1%})"
        else:
            return 5, f"完全面瘫 (对称性{symmetry_ratio:.1%})"

    else:
        # 无基线：使用静态眉眼距比值作为备选
        ratio = metrics.get("brow_eye_distance_ratio", 1.0)
        deviation = abs(ratio - 1.0)

        return (
            1 if deviation <= 0.05 else
            2 if deviation <= 0.10 else
            3 if deviation <= 0.20 else
            4 if deviation <= 0.35 else 5,
            f"无基线数据，基于静态比值 (ratio={ratio:.3f}, 偏差={deviation:.1%})"
        )


def compute_voluntary_score(metrics: Dict[str, Any], baseline_landmarks=None) -> Tuple[int, str]:
    """
    计算Voluntary Movement评分

    基于眉眼距变化度的对称性
    """
    if baseline_landmarks is not None and "change_ratio" in metrics:
        ratio = metrics["change_ratio"]
        left_change = metrics.get("left_change", 0)
        right_change = metrics.get("right_change", 0)

        # 首先检查是否有明显的运动
        min_change = min(abs(left_change), abs(right_change))
        max_change = max(abs(left_change), abs(right_change))

        if max_change < 3:  # 几乎没有运动
            return 1, "无法启动运动 (变化度过小)"

        # 基于比值的对称性评分
        if min_change < 1e-9:
            # 一侧完全没动
            return 1, "无法启动运动 (单侧无运动)"

        # 计算对称比 (较小/较大)
        symmetry_ratio = min_change / max_change

        if symmetry_ratio >= 0.90:
            return 5, "运动完整 (对称性>90%)"
        elif symmetry_ratio >= 0.75:
            return 4, "几乎完整 (对称性75-90%)"
        elif symmetry_ratio >= 0.50:
            return 3, "启动但不对称 (对称性50-75%)"
        elif symmetry_ratio >= 0.25:
            return 2, "轻微启动 (对称性25-50%)"
        else:
            return 1, "无法启动 (对称性<25%)"
    else:
        # 没有基线，使用静态比值
        ratio = metrics.get("brow_eye_distance_ratio", 1.0)
        deviation = abs(ratio - 1.0)

        if deviation <= 0.05:
            return 5, "运动完整"
        elif deviation <= 0.10:
            return 4, "几乎完整"
        elif deviation <= 0.20:
            return 3, "启动但不对称"
        elif deviation <= 0.35:
            return 2, "轻微启动"
        else:
            return 1, "无法启动"


def detect_synkinesis(baseline_result: Optional[ActionResult],
                      current_landmarks, w: int, h: int) -> Dict[str, int]:
    """检测抬眉时的联动运动"""
    synkinesis = {
        "eye_synkinesis": 0,
        "mouth_synkinesis": 0,
    }

    if baseline_result is None:
        return synkinesis

    # 检测眼部联动 (抬眉时眼睛变大)
    l_ear = compute_ear(current_landmarks, w, h, True)
    r_ear = compute_ear(current_landmarks, w, h, False)

    baseline_l_ear = baseline_result.left_ear
    baseline_r_ear = baseline_result.right_ear

    if baseline_l_ear > 1e-9 and baseline_r_ear > 1e-9:
        l_change = (l_ear - baseline_l_ear) / baseline_l_ear
        r_change = (r_ear - baseline_r_ear) / baseline_r_ear
        avg_change = (l_change + r_change) / 2

        if abs(avg_change) > 0.20:
            synkinesis["eye_synkinesis"] = 3
        elif abs(avg_change) > 0.12:
            synkinesis["eye_synkinesis"] = 2
        elif abs(avg_change) > 0.06:
            synkinesis["eye_synkinesis"] = 1

    # 检测嘴部联动
    mouth = compute_mouth_metrics(current_landmarks, w, h)
    baseline_mouth_w = baseline_result.mouth_width

    if baseline_mouth_w > 1e-9:
        mouth_change = abs(mouth["width"] - baseline_mouth_w) / baseline_mouth_w
        if mouth_change > 0.15:
            synkinesis["mouth_synkinesis"] = 3
        elif mouth_change > 0.08:
            synkinesis["mouth_synkinesis"] = 2
        elif mouth_change > 0.04:
            synkinesis["mouth_synkinesis"] = 1

    return synkinesis


def visualize_raise_eyebrow(frame: np.ndarray, landmarks, w: int, h: int,
                            result: ActionResult,
                            metrics: Dict[str, Any],
                            baseline_metrics: Optional[Dict[str, Any]] = None,
                            palsy_detection: Dict[str, Any] = None) -> np.ndarray:
    """可视化眉眼距"""
    img = frame.copy()

    # 绘制眉毛轮廓
    draw_polygon(img, landmarks, w, h, LM.BROW_L, (255, 100, 100), 2, False)
    draw_polygon(img, landmarks, w, h, LM.BROW_R, (100, 165, 255), 2, False)

    # 绘制眼部轮廓
    draw_polygon(img, landmarks, w, h, LM.EYE_CONTOUR_L, (255, 0, 0), 1)
    draw_polygon(img, landmarks, w, h, LM.EYE_CONTOUR_R, (0, 165, 255), 1)

    # 获取关键点
    left_eye_inner = metrics["left_eye_inner"]
    right_eye_inner = metrics["right_eye_inner"]
    left_brow_centroid = metrics["left_brow_centroid"]
    right_brow_centroid = metrics["right_brow_centroid"]

    # 绘制眉毛质心 (红色)
    cv2.circle(img, (int(left_brow_centroid[0]), int(left_brow_centroid[1])), 6, (0, 0, 255), -1)
    cv2.circle(img, (int(right_brow_centroid[0]), int(right_brow_centroid[1])), 6, (0, 0, 255), -1)

    # 绘制眼内眦点 (蓝色)
    cv2.circle(img, (int(left_eye_inner[0]), int(left_eye_inner[1])), 5, (255, 0, 0), -1)
    cv2.circle(img, (int(right_eye_inner[0]), int(right_eye_inner[1])), 5, (255, 0, 0), -1)

    # 画"眼部水平线(内眦连线延长到眉毛极值范围)" + "眉毛质心到该线的垂线"
    bedL = compute_brow_eye_distance(landmarks, w, h, left=True)
    bedR = compute_brow_eye_distance(landmarks, w, h, left=False)

    # 眼部水平线（端点对齐到 300/70）
    p0 = bedL.get("eye_line_p0", left_eye_inner)
    p1 = bedL.get("eye_line_p1", right_eye_inner)
    cv2.line(img, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (0, 255, 0), 2)

    # 画眉毛极值点（可选：你想看"对齐范围"就保留）
    bl = bedL.get("brow_extreme_left", None)  # index 300
    br = bedL.get("brow_extreme_right", None)  # index 70
    if bl is not None:
        cv2.circle(img, (int(bl[0]), int(bl[1])), 4, (0, 255, 0), -1)
    if br is not None:
        cv2.circle(img, (int(br[0]), int(br[1])), 4, (0, 255, 0), -1)

    # 左侧垂线
    footL = bedL.get("foot", None)
    if footL is not None:
        cv2.line(img, (int(left_brow_centroid[0]), int(left_brow_centroid[1])),
                 (int(footL[0]), int(footL[1])), (0, 255, 255), 2)
        cv2.circle(img, (int(footL[0]), int(footL[1])), 4, (0, 255, 255), -1)

    # 右侧垂线
    footR = bedR.get("foot", None)
    if footR is not None:
        cv2.line(img, (int(right_brow_centroid[0]), int(right_brow_centroid[1])),
                 (int(footR[0]), int(footR[1])), (0, 255, 255), 2)
        cv2.circle(img, (int(footR[0]), int(footR[1])), 4, (0, 255, 255), -1)

    # 信息面板
    panel_h = 380  # 增加高度
    panel_top = 80  # 面板顶部留出空间
    cv2.rectangle(img, (5, panel_top), (480, panel_h), (0, 0, 0), -1)
    cv2.rectangle(img, (5, panel_top), (480, panel_h), (255, 255, 255), 1)

    y = panel_top + 40
    cv2.putText(img, f"{ACTION_NAME}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 2)
    y += 50

    # 眉眼距信息（主要指标）
    cv2.putText(img, "=== Brow-Eye Distance (BED) ===", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1)
    y += 40

    left_bed = metrics['left_brow_eye_distance']
    right_bed = metrics['right_brow_eye_distance']
    left_bed_norm = metrics.get('left_bed_norm', 0)
    right_bed_norm = metrics.get('right_bed_norm', 0)

    # 颜色编码：较小的眉眼距用红色标记（可能是患侧）
    left_color = (0, 0, 255) if left_bed < right_bed else (255, 255, 255)
    right_color = (0, 0, 255) if right_bed < left_bed else (255, 255, 255)

    cv2.putText(img, f"Left: {left_bed:.1f}px ({left_bed_norm:.3f} ICD)", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, left_color, 1)
    y += 30

    cv2.putText(img, f"Right: {right_bed:.1f}px ({right_bed_norm:.3f} ICD)", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, right_color, 1)
    y += 30

    # 不对称度
    evidence = palsy_detection.get("evidence", {}) if palsy_detection else {}
    asymmetry = evidence.get("asymmetry", 0)
    threshold = evidence.get("threshold", 0.08)
    asym_color = (0, 0, 255) if asymmetry > threshold else (0, 255, 0)
    cv2.putText(img, f"Asymmetry: {asymmetry:.1%} (thr: {threshold:.1%})", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, asym_color, 1)
    y += 40

    # 如果有基线，显示变化量
    if "left_change" in metrics:
        cv2.putText(img, "=== Change from Baseline ===", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 1)
        y += 35

        left_change = metrics["left_change"]
        right_change = metrics["right_change"]

        cv2.putText(img, f"Left: {left_change:+.1f}px ({metrics.get('left_change_percent', 0):+.1f}%)", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        y += 28

        cv2.putText(img, f"Right: {right_change:+.1f}px ({metrics.get('right_change_percent', 0):+.1f}%)",
                    (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
        y += 35

    # Voluntary Score
    cv2.putText(img, f"Voluntary Score: {result.voluntary_movement_score}/5", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # 图例
    legend_y = panel_h + 30
    cv2.circle(img, (20, legend_y), 5, (0, 0, 255), -1)
    cv2.putText(img, "Brow Centroid", (35, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.circle(img, (180, legend_y), 5, (255, 0, 0), -1)
    cv2.putText(img, "Eye Inner", (195, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.line(img, (300, legend_y), (330, legend_y), (0, 255, 255), 2)
    cv2.putText(img, "BED", (335, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    # ========== 最后绘制患侧标注（放在最上方，不被覆盖）==========
    img, _ = draw_palsy_annotation_header(img, palsy_detection, ACTION_NAME)

    return img


def process(landmarks_seq: List, frames_seq: List, w: int, h: int,
            video_info: Dict[str, Any], output_dir: Path,
            baseline_result: Optional[ActionResult] = None,
            baseline_landmarks=None) -> Optional[ActionResult]:
    """
    处理RaiseEyebrow动作

    Args:
        landmarks_seq: landmarks序列
        frames_seq: 帧序列
        w, h: 图像尺寸
        video_info: 视频信息
        output_dir: 输出目录
        baseline_result: NeutralFace的结果 (用于联动检测)
        baseline_landmarks: NeutralFace的landmarks (用于变化度计算)

    Returns:
        ActionResult 或 None
    """
    if not landmarks_seq or not frames_seq:
        return None

    # 找峰值帧
    peak_idx, peak_debug = find_peak_frame(landmarks_seq, frames_seq, w, h, baseline_landmarks)
    peak_landmarks = landmarks_seq[peak_idx]
    peak_frame = frames_seq[peak_idx]

    # 提取时序序列用于可视化
    sequences = extract_raise_eyebrow_sequences(landmarks_seq, w, h, baseline_landmarks)

    if peak_landmarks is None:
        return None

    # 创建结果对象
    result = ActionResult(
        action_name=ACTION_NAME,
        action_name_cn=ACTION_NAME_CN,
        video_path=video_info.get("file_path", ""),
        total_frames=len(frames_seq),
        peak_frame_idx=peak_idx,
        image_size=(w, h),
        fps=video_info.get("fps", 30.0)
    )

    # 提取通用指标
    extract_common_indicators(peak_landmarks, w, h, result, baseline_landmarks)

    # 计算抬眉特有指标
    metrics = compute_raise_eyebrow_metrics(peak_landmarks, w, h, baseline_landmarks)

    # 检测面瘫侧别（不需要基线）
    palsy_detection = detect_palsy_side(metrics)

    # 计算Voluntary Movement评分
    score, interpretation = compute_voluntary_score(metrics, baseline_landmarks)
    result.voluntary_movement_score = score

    # 检测联动
    synkinesis = detect_synkinesis(baseline_result, peak_landmarks, w, h)
    result.synkinesis_scores = synkinesis

    # 计算严重度分数（需要基线）
    severity_score, severity_desc = compute_severity_score(metrics)

    # 存储动作特有指标
    result.action_specific = {
        "brow_eye_metrics": {
            "left_brow_eye_distance": metrics["left_brow_eye_distance"],
            "right_brow_eye_distance": metrics["right_brow_eye_distance"],
            "brow_eye_distance_ratio": metrics["brow_eye_distance_ratio"],
            "left_bed_norm": metrics.get("left_bed_norm", 0),
            "right_bed_norm": metrics.get("right_bed_norm", 0),
        },
        "voluntary_interpretation": interpretation,
        "synkinesis": synkinesis,
        "palsy_detection": palsy_detection,
        "severity_score": severity_score,
        "severity_desc": severity_desc,
    }

    if "left_change" in metrics:
        result.action_specific["brow_eye_metrics"].update({
            "left_change": metrics["left_change"],
            "right_change": metrics["right_change"],
            "change_ratio": metrics.get("change_ratio", 1.0),
            "left_change_percent": metrics.get("left_change_percent", 0),
            "right_change_percent": metrics.get("right_change_percent", 0),
        })
    result.action_specific["peak_debug"] = peak_debug

    # 创建输出目录
    action_dir = output_dir / ACTION_NAME
    action_dir.mkdir(parents=True, exist_ok=True)

    # 保存原始帧
    cv2.imwrite(str(action_dir / "peak_raw.jpg"), peak_frame)

    # 绘制关键帧选择曲线
    plot_raise_eyebrow_peak_selection(
        sequences,
        video_info.get("fps", 30.0),
        peak_idx,
        action_dir / "peak_selection_curve.png",
        has_baseline=(baseline_landmarks is not None),
        palsy_detection=palsy_detection
    )

    # 保存可视化
    baseline_metrics = None
    if baseline_landmarks is not None:
        baseline_metrics = compute_raise_eyebrow_metrics(baseline_landmarks, w, h, None)

    vis = visualize_raise_eyebrow(peak_frame, peak_landmarks, w, h, result, metrics,
                                  baseline_metrics, palsy_detection)
    cv2.imwrite(str(action_dir / "peak_indicators.jpg"), vis)

    # 保存JSON
    with open(action_dir / "indicators.json", 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    # 打印信息
    print(f"    [OK] {ACTION_NAME}: BED L={metrics['left_brow_eye_distance']:.1f}px "
          f"({metrics.get('left_bed_norm', 0):.3f} ICD) "
          f"R={metrics['right_brow_eye_distance']:.1f}px "
          f"({metrics.get('right_bed_norm', 0):.3f} ICD)")
    print(f"         Palsy: {palsy_detection.get('interpretation', 'N/A')}")
    if "left_change" in metrics:
        print(f"         Change L={metrics['left_change']:+.1f}px R={metrics['right_change']:+.1f}px")
    print(f"         Severity: {severity_score}/5 ({severity_desc})")
    print(f"         Voluntary Score: {result.voluntary_movement_score}/5 ({interpretation})")

    return result