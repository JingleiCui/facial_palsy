#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RaiseEyebrow 动作处理模块
=========================

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
    draw_palsy_side_label,
)

from thresholds import THR

ACTION_NAME = "RaiseEyebrow"
ACTION_NAME_CN = "抬眉/皱额"


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

        ax1.plot(time_sec, sequences["Left Change"], 'b-', label='Left BED Change', linewidth=2)
        ax1.plot(time_sec, sequences["Right Change"], 'r-', label='Right BED Change', linewidth=2)

        # 计算平均变化
        avg_change = [(l + r) / 2 if not (np.isnan(l) or np.isnan(r)) else np.nan
                      for l, r in zip(sequences["Left Change"], sequences["Right Change"])]
        ax1.plot(time_sec, avg_change, 'g--', label='Average Change', linewidth=2, alpha=0.7)

        ax1.axvline(x=peak_time, color='black', linestyle='--', linewidth=2, alpha=0.7)
        change_at_peak = avg_change[peak_idx] if peak_idx < len(avg_change) else 0
        if not np.isnan(change_at_peak):
            ax1.scatter([peak_time], [change_at_peak], color='red', s=150, zorder=5,
                        edgecolors='black', linewidths=2, marker='*', label=f'Peak Frame {peak_idx}')

        ax1.set_xlabel(x_label, fontsize=11)
        ax1.set_ylabel('BED Change (pixels)', fontsize=11)

        title = 'RaiseEyebrow Peak Selection: Maximum BED Change'
        if palsy_detection:
            palsy_text = get_palsy_side_text(palsy_detection.get("palsy_side", 0))
            title += f' | Detected: {palsy_text}'
        ax1.set_title(title, fontsize=13, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # 下图: 眉眼距绝对值
        ax2 = axes[1]

        if valid_mask is not None:
            add_valid_region_shading(ax2, valid_mask, time_sec)

        ax2.plot(time_sec, sequences["Left BED"], 'b-', label='Left BED', linewidth=2)
        ax2.plot(time_sec, sequences["Right BED"], 'r-', label='Right BED', linewidth=2)
        ax2.plot(time_sec, sequences["Average BED"], 'g--', label='Average BED', linewidth=1.5, alpha=0.7)
        ax2.axvline(x=peak_time, color='black', linestyle='--', linewidth=2, alpha=0.7)

        ax2.set_xlabel(x_label, fontsize=11)
        ax2.set_ylabel('BED (pixels)', fontsize=11)
        ax2.set_title('Brow-Eye Distance Over Time', fontsize=12)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

    else:
        # 没有基线时只画一张图
        fig, ax = plt.subplots(figsize=(16, 5))  # 增加宽度

        if valid_mask is not None:
            add_valid_region_shading(ax, valid_mask, time_sec)

        ax.plot(time_sec, sequences["Left BED"], 'b-', label='Left BED', linewidth=2)
        ax.plot(time_sec, sequences["Right BED"], 'r-', label='Right BED', linewidth=2)
        ax.plot(time_sec, sequences["Average BED"], 'g--', label='Average BED', linewidth=2, alpha=0.7)

        ax.axvline(x=peak_time, color='black', linestyle='--', linewidth=2, alpha=0.7)
        bed_at_peak = sequences["Average BED"][peak_idx] if peak_idx < len(sequences["Average BED"]) else 0
        if not np.isnan(bed_at_peak):
            ax.scatter([peak_time], [bed_at_peak], color='red', s=150, zorder=5,
                       edgecolors='black', linewidths=2, marker='*', label=f'Peak Frame {peak_idx}')

        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel('Brow-Eye Distance (pixels)', fontsize=11)

        title = 'RaiseEyebrow Peak Selection: Maximum BED'
        if palsy_detection:
            palsy_text = get_palsy_side_text(palsy_detection.get("palsy_side", 0))
            title += f' | Detected: {palsy_text}'
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()


def save_brow_curve_plot(peak_debug: Dict[str, Any], save_path: Path):
    """
    画“眉眼距/眉眼距变化量”的时序曲线，并标注关键帧点（peak_idx）
    输出：brow_curve.png
    """
    if not peak_debug:
        return

    x = list(range(len(peak_debug.get("mean_curve", []))))
    left = peak_debug.get("left_curve", [])
    right = peak_debug.get("right_curve", [])
    mean = peak_debug.get("mean_curve", [])
    peak_idx = int(peak_debug.get("peak_idx", 0))
    metric = peak_debug.get("metric", "BED")

    def _to_xy(arr):
        xs, ys = [], []
        for i, v in enumerate(arr):
            if v is None:
                continue
            xs.append(i)
            ys.append(v)
        return xs, ys

    xl, yl = _to_xy(left)
    xr, yr = _to_xy(right)
    xm, ym = _to_xy(mean)

    plt.figure(figsize=(10, 6))
    if xl:
        plt.plot(xl, yl, label="Left (BED)")
    if xr:
        plt.plot(xr, yr, label="Right (BED)")
    if xm:
        plt.plot(xm, ym, label="Mean (BED)")

    # 标注关键帧：竖线 + 点
    if 0 <= peak_idx < len(mean) and mean[peak_idx] is not None:
        plt.axvline(peak_idx, linestyle="--")
        plt.scatter([peak_idx], [mean[peak_idx]], zorder=5)
        plt.text(peak_idx, mean[peak_idx], f" peak={peak_idx}", fontsize=10)

    plt.title(f"RaiseEyebrow curve ({metric})")
    plt.xlabel("Frame")
    plt.ylabel("Value (px)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(save_path))
    plt.close()


def compute_raise_eyebrow_metrics(landmarks, w: int, h: int,
                                  baseline_landmarks=None) -> Dict[str, Any]:
    """计算抬眉特有指标"""
    # 当前眉眼距
    bed_result = compute_brow_eye_distance_ratio(landmarks, w, h)

    metrics = {
        "left_brow_eye_distance": bed_result["left_distance"],
        "right_brow_eye_distance": bed_result["right_distance"],
        "brow_eye_distance_ratio": bed_result["ratio"],
        "left_eye_inner": bed_result["left_eye_inner"],
        "right_eye_inner": bed_result["right_eye_inner"],
        "left_brow_centroid": bed_result["left_brow_centroid"],
        "right_brow_centroid": bed_result["right_brow_centroid"],
    }

    # 如果有基线，计算变化度
    if baseline_landmarks is not None:
        # ========== 计算统一 scale ==========
        scale = compute_scale_to_baseline(landmarks, baseline_landmarks, w, h)
        metrics["scale"] = scale
        # ====================================

        baseline_bed = compute_brow_eye_distance_ratio(baseline_landmarks, w, h)

        # ========== 缩放到 baseline 尺度后计算变化 ==========
        left_scaled = bed_result["left_distance"] * scale
        right_scaled = bed_result["right_distance"] * scale

        left_change = left_scaled - baseline_bed["left_distance"]
        right_change = right_scaled - baseline_bed["right_distance"]
        # ===================================================

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
    从抬眉动作检测面瘫侧别

    原理: 面瘫侧眉毛无法上抬，眉眼距变化小
    """
    result = {"palsy_side": 0, "confidence": 0.0, "interpretation": ""}

    # 优先使用眉眼距变化
    left_change = metrics.get("left_change", 0)
    right_change = metrics.get("right_change", 0)

    # 也检查静态眉眼距比
    bed_ratio = metrics.get("brow_eye_distance_ratio", 1.0)

    max_change = max(abs(left_change), abs(right_change))

    if max_change < 2:  # 运动幅度过小
        # 检查静态比例
        asymmetry = abs(bed_ratio - 1.0)
        if asymmetry < 0.10:
            result["interpretation"] = "眉眼距对称，运动幅度小"
        else:
            if bed_ratio > 1.0:
                result["palsy_side"] = 2
                result["interpretation"] = f"右侧眉眼距较小 (比值={bed_ratio:.3f})"
            else:
                result["palsy_side"] = 1
                result["interpretation"] = f"左侧眉眼距较小 (比值={bed_ratio:.3f})"
            result["confidence"] = min(1.0, asymmetry * 2)
        return result

    # 使用变化量比较
    asymmetry = abs(left_change - right_change) / max_change
    result["confidence"] = min(1.0, asymmetry * 3)

    # 详细输出用于调试
    result["evidence"] = {
        "asymmetry_ratio": asymmetry,
        "left_change": left_change,
        "right_change": right_change,
        "max_change": max_change,
        "change_ratio": min(abs(left_change), abs(right_change)) / max_change if max_change > 0 else 1.0
    }

    if asymmetry < 0.10:
        result["palsy_side"] = 0
        result[
            "interpretation"] = f"双侧抬眉对称 (L变化={left_change:.1f}px, R变化={right_change:.1f}px, 不对称{asymmetry:.1%})"
    elif left_change < right_change:
        result["palsy_side"] = 1
        result["interpretation"] = f"左侧抬眉弱 (L={left_change:.1f}px < R={right_change:.1f}px, 不对称{asymmetry:.1%})"
    else:
        result["palsy_side"] = 2
        result["interpretation"] = f"右侧抬眉弱 (R={right_change:.1f}px < L={left_change:.1f}px, 不对称{asymmetry:.1%})"

    return result


def compute_severity_score(metrics: Dict[str, Any]) -> Tuple[int, str]:
    """
    计算动作严重度分数(医生标注标准)

    计算依据: 眉眼距变化的对称性

    修改: 提高Score = 1
    的阈值
    """
    left_change = metrics.get("left_change", 0)
    right_change = metrics.get("right_change", 0)

    abs_left = abs(left_change)
    abs_right = abs(right_change)
    max_change = max(abs_left, abs_right)
    min_change = min(abs_left, abs_right)

    # 检查是否有足够的运动
    if max_change < 3.0:
        return 1, f"运动幅度过小 (L={left_change:.1f}px, R={right_change:.1f}px)"

    # 计算对称性比值
    symmetry_ratio = min_change / max_change if max_change > 0 else 1.0

    # 阈值调整
    if symmetry_ratio >= 0.90:
        return 1, f"正常 (对称性{symmetry_ratio:.2%})"
    elif symmetry_ratio >= 0.72:
        return 2, f"轻度异常 (对称性{symmetry_ratio:.2%})"
    elif symmetry_ratio >= 0.50:
        return 3, f"中度异常 (对称性{symmetry_ratio:.2%})"
    elif symmetry_ratio >= 0.30:
        return 4, f"重度异常 (对称性{symmetry_ratio:.2%})"
    else:
        return 5, f"完全面瘫 (对称性{symmetry_ratio:.2%})"


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

    # 添加患侧标签
    img = draw_palsy_side_label(img, palsy_detection, x=20, y=70, font_scale=1.4)

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

    # 画“眼部水平线(内眦连线延长到眉毛极值范围)” + “眉毛质心到该线的垂线”
    bedL = compute_brow_eye_distance(landmarks, w, h, left=True)
    bedR = compute_brow_eye_distance(landmarks, w, h, left=False)

    # 眼部水平线（端点对齐到 300/70）
    p0 = bedL.get("eye_line_p0", left_eye_inner)
    p1 = bedL.get("eye_line_p1", right_eye_inner)
    cv2.line(img, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (0, 255, 0), 2)

    # 画眉毛极值点（可选：你想看“对齐范围”就保留）
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
    panel_h = 280
    cv2.rectangle(img, (5, 5), (380, panel_h), (0, 0, 0), -1)
    cv2.rectangle(img, (5, 5), (380, panel_h), (255, 255, 255), 1)

    y = 50
    cv2.putText(img, f"{ACTION_NAME}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 2)
    y += 50

    cv2.putText(img, "=== Brow-Eye Distance ===", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 1)
    y += 50

    cv2.putText(img, f"Left: {metrics['left_brow_eye_distance']:.1f}px", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1)
    y += 30

    cv2.putText(img, f"Right: {metrics['right_brow_eye_distance']:.1f}px", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1)
    y += 30

    ratio = metrics['brow_eye_distance_ratio']
    ratio_color = (0, 255, 0) if 0.9 <= ratio <= 1.1 else (0, 165, 255) if 0.8 <= ratio <= 1.2 else (0, 0, 255)
    cv2.putText(img, f"Ratio: {ratio:.3f}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, ratio_color, 1)
    y += 50

    if "left_change" in metrics:
        cv2.putText(img, "=== Distance Change ===", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 1)
        y += 50

        left_change = metrics["left_change"]
        right_change = metrics["right_change"]

        cv2.putText(img, f"Left: {left_change:+.1f}px ({metrics.get('left_change_percent', 0):+.1f}%)", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1)
        y += 50

        cv2.putText(img, f"Right: {right_change:+.1f}px ({metrics.get('right_change_percent', 0):+.1f}%)",
                    (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1)
        y += 50

        change_ratio = metrics.get("change_ratio", 1.0)
        if not np.isinf(change_ratio):
            cv2.putText(img, f"Change Ratio: {change_ratio:.3f}", (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1)
        y += 80

    # Voluntary Score
    cv2.putText(img, f"Voluntary Score: {result.voluntary_movement_score}/5", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # 图例
    legend_y = panel_h + 50
    cv2.circle(img, (20, legend_y), 5, (0, 0, 255), -1)
    cv2.putText(img, "Brow Centroid", (35, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1)
    cv2.circle(img, (150, legend_y), 5, (255, 0, 0), -1)
    cv2.putText(img, "Eye Inner", (165, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1)
    cv2.line(img, (260, legend_y), (290, legend_y), (0, 255, 255), 2)
    cv2.putText(img, "BED", (295, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1)

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

    # 检测面瘫侧别
    palsy_detection = detect_palsy_side(metrics)

    # 计算Voluntary Movement评分
    score, interpretation = compute_voluntary_score(metrics, baseline_landmarks)
    result.voluntary_movement_score = score

    # 检测联动
    synkinesis = detect_synkinesis(baseline_result, peak_landmarks, w, h)
    result.synkinesis_scores = synkinesis

    # 计算严重度分数 (医生标注标准: 1=正常, 5=面瘫)
    severity_score, severity_desc = compute_severity_score(metrics)

    # 存储动作特有指标
    result.action_specific = {
        "brow_eye_metrics": {
            "left_brow_eye_distance": metrics["left_brow_eye_distance"],
            "right_brow_eye_distance": metrics["right_brow_eye_distance"],
            "brow_eye_distance_ratio": metrics["brow_eye_distance_ratio"],
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

    save_brow_curve_plot(peak_debug, action_dir / "brow_curve.png")
    # 保存原始帧
    cv2.imwrite(str(action_dir / "peak_raw.jpg"), peak_frame)

    # 绘制关键帧选择曲线
    plot_raise_eyebrow_peak_selection(
        sequences,
        video_info.get("fps", 30.0),
        peak_idx,
        action_dir / "peak_selection_curve.png",
        has_baseline=(baseline_landmarks is not None)
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

    print(
        f"    [OK] {ACTION_NAME}: BED L={metrics['left_brow_eye_distance']:.1f} R={metrics['right_brow_eye_distance']:.1f}")
    if "left_change" in metrics:
        print(f"         Change L={metrics['left_change']:+.1f}px R={metrics['right_change']:+.1f}px")
    print(f"         Voluntary Score: {result.voluntary_movement_score}/5 ({interpretation})")

    return result