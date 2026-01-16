#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CloseEye 动作处理模块
==================================

核心修改：
1. 逐帧对比两只眼睛的闭合度（面积），统计哪侧更多帧表现差
2. 曲线图改为：上边画眼睛面积，下边画眼睛闭合度
3. 移除EAR相关的判断逻辑

面瘫侧别编码: 0=无/对称, 1=左, 2=右
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from clinical_base import (
    LM, pt2d, pts2d, dist, compute_ear, compute_eye_area,
    compute_palpebral_height, compute_mouth_metrics,
    compute_icd, extract_common_indicators,
    ActionResult, draw_polygon, compute_scale_to_baseline,
    draw_palsy_side_label, compute_eye_closure_by_area,
    compute_eye_closure_sequence, compute_eye_synchrony,
    compute_eye_symmetry_at_peak, draw_palsy_annotation_header,
)

from thresholds import THR


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


def find_peak_frame_close_eye(landmarks_seq: List, frames_seq: List, w: int, h: int,
                              baseline_landmarks=None) -> int:
    """
    找闭眼峰值帧 - 使用眼睛面积
    """
    if baseline_landmarks is not None:
        baseline_left_area, _ = compute_eye_area(baseline_landmarks, w, h, True)
        baseline_right_area, _ = compute_eye_area(baseline_landmarks, w, h, False)

        max_closure = -float('inf')
        max_idx = 0

        for i, lm in enumerate(landmarks_seq):
            if lm is None:
                continue

            scale = compute_scale_to_baseline(lm, baseline_landmarks, w, h)

            l_area, _ = compute_eye_area(lm, w, h, True)
            r_area, _ = compute_eye_area(lm, w, h, False)
            scaled_l_area = l_area * (scale ** 2)
            scaled_r_area = r_area * (scale ** 2)

            l_closure = 1.0 - (scaled_l_area / baseline_left_area) if baseline_left_area > 1e-6 else 0
            r_closure = 1.0 - (scaled_r_area / baseline_right_area) if baseline_right_area > 1e-6 else 0

            avg_closure = (max(0, min(1, l_closure)) + max(0, min(1, r_closure))) / 2

            if avg_closure > max_closure:
                max_closure = avg_closure
                max_idx = i

        return max_idx
    else:
        min_area = float('inf')
        min_idx = 0

        for i, lm in enumerate(landmarks_seq):
            if lm is None:
                continue
            l_area, _ = compute_eye_area(lm, w, h, True)
            r_area, _ = compute_eye_area(lm, w, h, False)
            avg_area = (l_area + r_area) / 2
            if avg_area < min_area:
                min_area = avg_area
                min_idx = i

        return min_idx


def extract_eye_sequence(landmarks_seq: List, w: int, h: int) -> Dict[str, Dict[str, List[float]]]:
    """提取整个序列的眼睛指标"""
    left_ear_seq = []
    right_ear_seq = []
    left_area_seq = []
    right_area_seq = []

    for lm in landmarks_seq:
        if lm is None:
            left_ear_seq.append(np.nan)
            right_ear_seq.append(np.nan)
            left_area_seq.append(np.nan)
            right_area_seq.append(np.nan)
        else:
            l_ear = compute_ear(lm, w, h, True)
            r_ear = compute_ear(lm, w, h, False)
            l_area, _ = compute_eye_area(lm, w, h, True)
            r_area, _ = compute_eye_area(lm, w, h, False)
            left_ear_seq.append(l_ear)
            right_ear_seq.append(r_ear)
            left_area_seq.append(l_area)
            right_area_seq.append(r_area)

    return {
        "ear": {
            "left": left_ear_seq,
            "right": right_ear_seq,
            "average": [(l + r) / 2 if not (np.isnan(l) or np.isnan(r)) else np.nan
                        for l, r in zip(left_ear_seq, right_ear_seq)]
        },
        "area": {
            "left": left_area_seq,
            "right": right_area_seq
        }
    }


def detect_palsy_side(
        closure_data: Dict[str, Any],
        peak_idx: int,
) -> Dict[str, Any]:
    """
    逐帧对比两只眼睛的闭合度检测面瘫侧别

    核心逻辑：
    - 逐帧比较左右眼的面积
    - 闭眼期间，面积大的一侧 = 闭不紧 = 患侧
    - 统计哪侧更多帧表现差（面积更大）
    - 这样可以捕捉到闭眼过程中任意时刻的差异，不会遗漏早期差异

    Args:
        closure_data: 闭合度序列数据
        peak_idx: 峰值帧索引

    Returns:
        palsy_detection dict
    """
    result = {
        "palsy_side": 0,
        "confidence": 0.0,
        "method": "frame_by_frame_area_comparison",
        "interpretation": "",
        "evidence": {},
    }

    # ========== 提取数据 ==========
    left_closure_seq = closure_data.get("left_closure", [])
    right_closure_seq = closure_data.get("right_closure", [])
    left_area_seq = closure_data.get("left_area", [])
    right_area_seq = closure_data.get("right_area", [])

    # 峰值帧数据
    left_closure_peak = left_closure_seq[peak_idx] if peak_idx < len(left_closure_seq) else 0
    right_closure_peak = right_closure_seq[peak_idx] if peak_idx < len(right_closure_seq) else 0

    # ========== 逐帧对比 ==========
    left_worse_count = 0  # 左眼面积更大（闭合差）的帧数
    right_worse_count = 0  # 右眼面积更大（闭合差）的帧数
    total_closing_frames = 0  # 闭眼期间的总帧数

    frame_details = []  # 记录每帧的对比结果

    for i, (lc, rc, la, ra) in enumerate(zip(
            left_closure_seq, right_closure_seq, left_area_seq, right_area_seq)):

        if not (np.isfinite(lc) and np.isfinite(rc) and np.isfinite(la) and np.isfinite(ra)):
            continue

        avg_closure = (lc + rc) / 2

        # 只统计闭眼期间的帧
        if avg_closure >= THR.MIN_CLOSURE_FOR_COUNT:
            total_closing_frames += 1

            # 计算面积差异比例
            max_area = max(la, ra)
            if max_area > 1:
                area_diff_ratio = abs(la - ra) / max_area

                # 面积差异超过阈值才计入
                if area_diff_ratio >= THR.EYE_AREA_DIFF_THRESHOLD:
                    if la > ra:  # 左眼面积大 = 左眼闭不紧 = 左侧表现差
                        left_worse_count += 1
                        frame_details.append((i, "left_worse", area_diff_ratio))
                    else:  # 右眼面积大 = 右眼闭不紧 = 右侧表现差
                        right_worse_count += 1
                        frame_details.append((i, "right_worse", area_diff_ratio))

    # ========== 计算比例 ==========
    if total_closing_frames > 0:
        left_worse_ratio = left_worse_count / total_closing_frames
        right_worse_ratio = right_worse_count / total_closing_frames
    else:
        left_worse_ratio = 0
        right_worse_ratio = 0

    # 收集证据
    evidence = {
        "left_closure_at_peak": float(left_closure_peak),
        "right_closure_at_peak": float(right_closure_peak),
        "total_closing_frames": total_closing_frames,
        "left_worse_count": left_worse_count,
        "right_worse_count": right_worse_count,
        "left_worse_ratio": float(left_worse_ratio),
        "right_worse_ratio": float(right_worse_ratio),
        "frame_area_diff_threshold": THR.EYE_AREA_DIFF_THRESHOLD,
    }

    result["evidence"] = evidence

    # ========== 判断逻辑 ==========

    # 检查闭眼帧数是否足够
    if total_closing_frames < 5:
        result["interpretation"] = f"闭眼帧数不足 ({total_closing_frames}帧)，无法判断"
        return result

    # 判断哪侧更多帧表现差
    if left_worse_ratio >= THR.FRAME_RATIO_THRESHOLD and left_worse_ratio > right_worse_ratio:
        result["palsy_side"] = 1  # 左侧面瘫
        result["confidence"] = min(1.0, left_worse_ratio / 0.60)
        result["interpretation"] = (
            f"左侧面瘫: {left_worse_count}/{total_closing_frames}帧 "
            f"({left_worse_ratio:.0%}) 左眼闭合差"
        )
    elif right_worse_ratio >= THR.FRAME_RATIO_THRESHOLD and right_worse_ratio > left_worse_ratio:
        result["palsy_side"] = 2  # 右侧面瘫
        result["confidence"] = min(1.0, right_worse_ratio / 0.60)
        result["interpretation"] = (
            f"右侧面瘫: {right_worse_count}/{total_closing_frames}帧 "
            f"({right_worse_ratio:.0%}) 右眼闭合差"
        )
    else:
        # 对称
        result["palsy_side"] = 0
        result["confidence"] = 1.0 - max(left_worse_ratio, right_worse_ratio)
        result["interpretation"] = (
            f"双眼闭合对称 (左差{left_worse_ratio:.0%}, 右差{right_worse_ratio:.0%})"
        )

    return result


def compute_close_eye_metrics(landmarks, w: int, h: int,
                              baseline_landmarks=None) -> Dict[str, Any]:
    """计算闭眼特有指标"""
    l_ear = compute_ear(landmarks, w, h, True)
    r_ear = compute_ear(landmarks, w, h, False)
    l_area, _ = compute_eye_area(landmarks, w, h, True)
    r_area, _ = compute_eye_area(landmarks, w, h, False)
    l_height = compute_palpebral_height(landmarks, w, h, True)
    r_height = compute_palpebral_height(landmarks, w, h, False)

    metrics = {
        "left_ear": l_ear,
        "right_ear": r_ear,
        "ear_ratio": l_ear / r_ear if r_ear > 1e-9 else 1.0,
        "left_area": l_area,
        "right_area": r_area,
        "area_ratio": l_area / r_area if r_area > 1e-9 else 1.0,
        "left_palpebral_height": l_height,
        "right_palpebral_height": r_height,
        "height_ratio": l_height / r_height if r_height > 1e-9 else 1.0,
    }

    if baseline_landmarks is not None:
        scale = compute_scale_to_baseline(landmarks, baseline_landmarks, w, h)
        metrics["scale"] = scale

        baseline_l_area, _ = compute_eye_area(baseline_landmarks, w, h, True)
        baseline_r_area, _ = compute_eye_area(baseline_landmarks, w, h, False)

        scaled_l_area = l_area * (scale ** 2)
        scaled_r_area = r_area * (scale ** 2)

        left_closure_ratio = 1.0 - (scaled_l_area / baseline_l_area) if baseline_l_area > 1e-6 else 0
        right_closure_ratio = 1.0 - (scaled_r_area / baseline_r_area) if baseline_r_area > 1e-6 else 0

        left_closure_ratio = max(0, min(1, left_closure_ratio))
        right_closure_ratio = max(0, min(1, right_closure_ratio))

        metrics["left_closure_percent"] = left_closure_ratio * 100
        metrics["right_closure_percent"] = right_closure_ratio * 100
        metrics["left_closure_ratio"] = left_closure_ratio
        metrics["right_closure_ratio"] = right_closure_ratio
        metrics["baseline_left_area"] = baseline_l_area
        metrics["baseline_right_area"] = baseline_r_area

    return metrics


def compute_voluntary_score(metrics: Dict[str, Any], baseline_landmarks=None) -> Tuple[int, str]:
    """计算Voluntary Movement评分"""
    if baseline_landmarks is not None and "left_closure_percent" in metrics:
        left_closure = metrics["left_closure_percent"]
        right_closure = metrics["right_closure_percent"]

        max_closure = max(left_closure, right_closure)
        min_closure = min(left_closure, right_closure)

        if max_closure < 30:
            return 1, "无法启动运动 (闭眼不足30%)"

        if max_closure < 1e-9:
            symmetry_ratio = 1.0
        else:
            symmetry_ratio = min_closure / max_closure

        if symmetry_ratio >= 0.85:
            if max_closure >= 90:
                return 5, "运动完整 (双眼完全闭合且对称)"
            elif max_closure >= 70:
                return 4, "几乎完整 (闭合良好)"
            else:
                return 3, "启动但幅度不足"
        elif symmetry_ratio >= 0.60:
            return 3, "启动但不对称"
        elif symmetry_ratio >= 0.30:
            return 2, "轻微启动"
        else:
            return 1, "无法启动"
    else:
        ratio = metrics.get("ear_ratio", 1.0)
        deviation = abs(ratio - 1.0)

        if deviation <= 0.10:
            return 5, "运动完整"
        elif deviation <= 0.20:
            return 4, "几乎完整"
        elif deviation <= 0.35:
            return 3, "启动但不对称"
        elif deviation <= 0.50:
            return 2, "轻微启动"
        else:
            return 1, "无法启动"


def compute_severity_score(metrics: Dict[str, Any]) -> Tuple[int, str]:
    """计算严重度分数"""
    left_closure = metrics.get("left_closure_percent")
    right_closure = metrics.get("right_closure_percent")

    if left_closure is None or right_closure is None:
        area_ratio = metrics.get("area_ratio", 1.0)
        deviation = abs(area_ratio - 1.0)

        if deviation <= 0.08:
            return 1, f"对称性良好 (area_ratio={area_ratio:.3f})"
        elif deviation <= 0.15:
            return 2, f"轻度不对称 (area_ratio={area_ratio:.3f})"
        elif deviation <= 0.25:
            return 3, f"中度不对称 (area_ratio={area_ratio:.3f})"
        elif deviation <= 0.40:
            return 4, f"重度不对称 (area_ratio={area_ratio:.3f})"
        else:
            return 5, f"严重不对称 (area_ratio={area_ratio:.3f})"

    max_closure = max(left_closure, right_closure)
    min_closure = min(left_closure, right_closure)

    if max_closure < 20:
        return 1, f"闭眼幅度过小 (L={left_closure:.1f}%, R={right_closure:.1f}%)"

    symmetry_ratio = min_closure / max_closure if max_closure > 0 else 1.0

    if symmetry_ratio >= 0.95:
        return 1, f"正常 (对称性{symmetry_ratio:.2%})"
    elif symmetry_ratio >= 0.85:
        return 2, f"轻度异常 (对称性{symmetry_ratio:.2%})"
    elif symmetry_ratio >= 0.70:
        return 3, f"中度异常 (对称性{symmetry_ratio:.2%})"
    elif symmetry_ratio >= 0.50:
        return 4, f"重度异常 (对称性{symmetry_ratio:.2%})"
    else:
        return 5, f"完全面瘫 (对称性{symmetry_ratio:.2%})"


def detect_synkinesis(baseline_result: Optional[ActionResult],
                      current_landmarks, w: int, h: int) -> Dict[str, int]:
    """检测联动运动"""
    synkinesis = {"mouth_synkinesis": 0, "cheek_synkinesis": 0}

    if baseline_result is None:
        return synkinesis

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


def plot_eye_curve(eye_seq: Dict, fps: float, peak_idx: int,
                   output_path: Path, action_name: str,
                   closure_data: Dict = None,
                   valid_mask: List[bool] = None,
                   palsy_detection: Dict[str, Any] = None) -> None:
    """
    绘制眼睛曲线 - 修改版
    上图：眼睛面积
    下图：眼睛闭合度
    """
    from clinical_base import add_valid_region_shading, get_palsy_side_text

    fig, axes = plt.subplots(2, 1, figsize=(16, 9))

    frames = np.arange(len(eye_seq["area"]["left"]))
    time_sec = frames / fps if fps > 0 else frames
    x_label = 'Time (seconds)' if fps > 0 else 'Frame'
    peak_time = time_sec[peak_idx] if peak_idx < len(time_sec) else 0

    # ========== 上图: 眼睛面积 ==========
    ax1 = axes[0]

    if valid_mask is not None:
        add_valid_region_shading(ax1, valid_mask, time_sec)

    ax1.plot(time_sec, eye_seq["area"]["left"], 'b-', label='Left Eye Area', linewidth=2)
    ax1.plot(time_sec, eye_seq["area"]["right"], 'r-', label='Right Eye Area', linewidth=2)
    ax1.axvline(x=peak_time, color='k', linestyle='--', alpha=0.5, label=f'Peak Frame ({peak_idx})')

    title = f'{action_name} - Eye Area Over Time'
    if palsy_detection:
        palsy_text = get_palsy_side_text(palsy_detection.get("palsy_side", 0))
        confidence = palsy_detection.get("confidence", 0)
        title += f' | Detected: {palsy_text} (conf={confidence:.0%})'
    ax1.set_title(title, fontsize=13, fontweight='bold')

    ax1.set_xlabel(x_label, fontsize=11)
    ax1.set_ylabel('Eye Area (pixels²)', fontsize=11)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 添加面积差异信息
    if palsy_detection and "evidence" in palsy_detection:
        ev = palsy_detection["evidence"]
        left_worse = ev.get("left_worse_ratio", 0)
        right_worse = ev.get("right_worse_ratio", 0)
        info_text = f"Left worse: {left_worse:.0%} frames | Right worse: {right_worse:.0%} frames"
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # ========== 下图: 眼睛闭合度 ==========
    ax2 = axes[1]

    if closure_data is not None:
        left_closure = closure_data.get("left_closure", [])
        right_closure = closure_data.get("right_closure", [])

        if len(left_closure) > 0 and len(right_closure) > 0:
            if valid_mask is not None:
                add_valid_region_shading(ax2, valid_mask, time_sec)

            ax2.plot(time_sec, left_closure, 'b-', label='Left Eye Closure', linewidth=2)
            ax2.plot(time_sec, right_closure, 'r-', label='Right Eye Closure', linewidth=2)

            # 平均闭合度
            avg_closure = [(l + r) / 2 if (np.isfinite(l) and np.isfinite(r)) else np.nan
                           for l, r in zip(left_closure, right_closure)]
            ax2.plot(time_sec, avg_closure, 'g--', label='Average Closure', linewidth=1.5, alpha=0.7)

            ax2.axvline(x=peak_time, color='k', linestyle='--', alpha=0.5, label=f'Peak Frame ({peak_idx})')

            # 阈值线
            ax2.axhline(y=0.85, color='green', linestyle=':', alpha=0.5, label='Complete (85%)')
            ax2.axhline(y=0.20, color='orange', linestyle=':', alpha=0.5, label='Threshold (20%)')

            ax2.set_ylim(0, 1.1)
            ax2.set_ylabel('Closure Ratio (1=fully closed)', fontsize=11)

            # 峰值帧闭合度信息
            if 0 <= peak_idx < len(left_closure):
                left_peak = left_closure[peak_idx] if np.isfinite(left_closure[peak_idx]) else 0
                right_peak = right_closure[peak_idx] if np.isfinite(right_closure[peak_idx]) else 0
                info_text = f"Peak: L={left_peak:.1%}, R={right_peak:.1%}"
                ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes, fontsize=10,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        # 没有闭合度数据，显示EAR（参考）
        ax2.plot(time_sec, eye_seq["ear"]["left"], 'b-', label='Left EAR (ref)', linewidth=1.5, alpha=0.7)
        ax2.plot(time_sec, eye_seq["ear"]["right"], 'r-', label='Right EAR (ref)', linewidth=1.5, alpha=0.7)
        ax2.axvline(x=peak_time, color='k', linestyle='--', alpha=0.5)
        ax2.set_ylabel('EAR (reference)', fontsize=11)

    ax2.set_title(f'{action_name} - Eye Closure Over Time', fontsize=13)
    ax2.set_xlabel(x_label, fontsize=11)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()


def visualize_close_eye(frame: np.ndarray, landmarks, w: int, h: int,
                        result: ActionResult,
                        metrics: Dict[str, Any],
                        palsy_detection: Dict[str, Any] = None) -> np.ndarray:
    """可视化闭眼指标"""
    img = frame.copy()

    # ========== 第一行绘制患侧标注 ==========
    img, header_end_y = draw_palsy_annotation_header(img, palsy_detection, result.action_name)

    # 绘制眼部轮廓
    draw_polygon(img, landmarks, w, h, LM.EYE_CONTOUR_L, (255, 0, 0), 3)
    draw_polygon(img, landmarks, w, h, LM.EYE_CONTOUR_R, (0, 165, 255), 3)

    # 信息面板 - 顶部下移
    panel_top = header_end_y + 20
    panel_w, panel_h = 700, panel_top + 520
    cv2.rectangle(img, (10, panel_top), (panel_w, panel_h), (0, 0, 0), -1)

    y = panel_top + 60
    cv2.putText(img, f"{result.action_name}", (25, y), FONT, FONT_SCALE_TITLE, (0, 255, 0), THICKNESS_TITLE)
    y += LINE_HEIGHT + 15

    # 眼睛面积
    cv2.putText(img, "=== Eye Area ===", (25, y), FONT, FONT_SCALE_NORMAL, (0, 255, 255), THICKNESS_NORMAL)
    y += LINE_HEIGHT

    left_area = metrics["left_area"]
    right_area = metrics["right_area"]
    left_color = (0, 0, 255) if left_area > right_area * 1.15 else (255, 255, 255)
    right_color = (0, 0, 255) if right_area > left_area * 1.15 else (255, 255, 255)

    cv2.putText(img, f"Left: {left_area:.0f} px²", (25, y), FONT, FONT_SCALE_NORMAL, left_color, THICKNESS_NORMAL)
    y += LINE_HEIGHT
    cv2.putText(img, f"Right: {right_area:.0f} px²", (25, y), FONT, FONT_SCALE_NORMAL, right_color, THICKNESS_NORMAL)
    y += LINE_HEIGHT + 10

    # 闭合度
    if "left_closure_percent" in metrics:
        cv2.putText(img, "=== Closure (from baseline) ===", (25, y), FONT, FONT_SCALE_NORMAL, (0, 255, 255),
                    THICKNESS_NORMAL)
        y += LINE_HEIGHT
        cv2.putText(img,
                    f"Left: {metrics['left_closure_percent']:.1f}%  Right: {metrics['right_closure_percent']:.1f}%",
                    (25, y), FONT, FONT_SCALE_NORMAL, (255, 255, 255), THICKNESS_NORMAL)
        y += LINE_HEIGHT + 10

    # 逐帧对比结果
    if palsy_detection and "evidence" in palsy_detection:
        ev = palsy_detection["evidence"]
        cv2.putText(img, "=== Frame-by-Frame Analysis ===", (25, y), FONT, 0.8, (0, 255, 255), 2)
        y += 35
        left_worse = ev.get("left_worse_ratio", 0)
        right_worse = ev.get("right_worse_ratio", 0)
        total_frames = ev.get("total_closing_frames", 0)
        cv2.putText(img, f"Closing frames: {total_frames}", (25, y), FONT, 0.7, (200, 200, 200), 2)
        y += 30
        cv2.putText(img, f"Left worse: {left_worse:.0%}  Right worse: {right_worse:.0%}",
                    (25, y), FONT, 0.7, (200, 200, 200), 2)
        y += LINE_HEIGHT

    # 面瘫侧别
    if palsy_detection:
        palsy_side = palsy_detection.get("palsy_side", 0)
        palsy_text = {0: "Symmetric", 1: "LEFT PALSY", 2: "RIGHT PALSY"}.get(palsy_side, "Unknown")
        palsy_color = (0, 255, 0) if palsy_side == 0 else (0, 0, 255)
        confidence = palsy_detection.get("confidence", 0)
        cv2.putText(img, f"Result: {palsy_text} ({confidence:.0%})", (25, y), FONT, FONT_SCALE_TITLE, palsy_color,
                    THICKNESS_TITLE)
        y += LINE_HEIGHT

    cv2.putText(img, f"Voluntary Score: {result.voluntary_movement_score}/5", (25, y),
                FONT, FONT_SCALE_TITLE, (0, 255, 255), THICKNESS_TITLE)

    return img


def process_close_eye_softly(landmarks_seq: List, frames_seq: List, w: int, h: int,
                             video_info: Dict[str, Any], output_dir: Path,
                             baseline_result: Optional[ActionResult] = None,
                             baseline_landmarks=None) -> Optional[ActionResult]:
    """处理CloseEyeSoftly动作"""
    return _process_close_eye(
        landmarks_seq, frames_seq, w, h, video_info, output_dir,
        action_name="CloseEyeSoftly",
        action_name_cn="轻闭眼",
        baseline_result=baseline_result,
        baseline_landmarks=baseline_landmarks
    )


def process_close_eye_hardly(landmarks_seq: List, frames_seq: List, w: int, h: int,
                             video_info: Dict[str, Any], output_dir: Path,
                             baseline_result: Optional[ActionResult] = None,
                             baseline_landmarks=None) -> Optional[ActionResult]:
    """处理CloseEyeHardly动作"""
    return _process_close_eye(
        landmarks_seq, frames_seq, w, h, video_info, output_dir,
        action_name="CloseEyeHardly",
        action_name_cn="用力闭眼",
        baseline_result=baseline_result,
        baseline_landmarks=baseline_landmarks
    )


def _process_close_eye(landmarks_seq: List, frames_seq: List, w: int, h: int,
                       video_info: Dict[str, Any], output_dir: Path,
                       action_name: str, action_name_cn: str,
                       baseline_result: Optional[ActionResult] = None,
                       baseline_landmarks=None) -> Optional[ActionResult]:
    """处理闭眼动作的通用函数"""
    if not landmarks_seq or not frames_seq:
        return None

    # 计算整个视频的闭合度序列
    closure_data = compute_eye_closure_sequence(landmarks_seq, w, h, baseline_landmarks)

    # 找峰值帧
    peak_idx = find_peak_frame_close_eye(landmarks_seq, frames_seq, w, h, baseline_landmarks)
    peak_landmarks = landmarks_seq[peak_idx]
    peak_frame = frames_seq[peak_idx]

    if peak_landmarks is None:
        return None

    # 计算同步度（用于记录）
    synchrony = compute_eye_synchrony(
        closure_data["left_closure"],
        closure_data["right_closure"]
    )

    # 峰值帧闭合度
    left_closure_at_peak = closure_data["left_closure"][peak_idx]
    right_closure_at_peak = closure_data["right_closure"][peak_idx]
    symmetry_at_peak = compute_eye_symmetry_at_peak(left_closure_at_peak, right_closure_at_peak)

    fps = video_info.get("fps", 30.0)

    result = ActionResult(
        action_name=action_name,
        action_name_cn=action_name_cn,
        video_path=video_info.get("file_path", ""),
        total_frames=len(frames_seq),
        peak_frame_idx=peak_idx,
        image_size=(w, h),
        fps=fps
    )

    extract_common_indicators(peak_landmarks, w, h, result, baseline_landmarks)

    metrics = compute_close_eye_metrics(peak_landmarks, w, h, baseline_landmarks)

    eye_seq = extract_eye_sequence(landmarks_seq, w, h)

    score, interpretation = compute_voluntary_score(metrics, baseline_landmarks)
    result.voluntary_movement_score = score

    synkinesis = detect_synkinesis(baseline_result, peak_landmarks, w, h)
    result.synkinesis_scores = synkinesis

    # ========== 面瘫侧别检测 - 逐帧对比 ==========
    palsy_detection = detect_palsy_side(
        closure_data=closure_data,
        peak_idx=peak_idx,
    )

    severity_score, severity_desc = compute_severity_score(metrics)

    result.action_specific = {
        "close_eye_metrics": {
            "left_ear": metrics["left_ear"],
            "right_ear": metrics["right_ear"],
            "ear_ratio": metrics["ear_ratio"],
            "left_area": metrics["left_area"],
            "right_area": metrics["right_area"],
            "area_ratio": metrics["area_ratio"],
        },
        "eye_sequence": {
            "ear_left": [float(x) if not np.isnan(x) else None for x in eye_seq["ear"]["left"]],
            "ear_right": [float(x) if not np.isnan(x) else None for x in eye_seq["ear"]["right"]],
            "area_left": [float(x) if not np.isnan(x) else None for x in eye_seq["area"]["left"]],
            "area_right": [float(x) if not np.isnan(x) else None for x in eye_seq["area"]["right"]],
        },
        "closure_sequence": {
            "left": [float(v) if np.isfinite(v) else None for v in closure_data["left_closure"]],
            "right": [float(v) if np.isfinite(v) else None for v in closure_data["right_closure"]],
        },
        "baseline": {
            "left_area": float(closure_data["baseline_left_area"]),
            "right_area": float(closure_data["baseline_right_area"]),
        },
        "peak_frame": {
            "idx": int(peak_idx),
            "left_closure": float(left_closure_at_peak),
            "right_closure": float(right_closure_at_peak),
            "left_area": float(closure_data["left_area"][peak_idx]),
            "right_area": float(closure_data["right_area"][peak_idx]),
        },
        "symmetry_analysis": symmetry_at_peak,
        "synchrony_analysis": synchrony,
        "palsy_detection": palsy_detection,
        "diagnosis_summary": {
            "palsy_side": palsy_detection["palsy_side"],
            "confidence": palsy_detection["confidence"],
            "method": palsy_detection["method"],
        },
        "severity_score": severity_score,
        "severity_desc": severity_desc,
    }

    if "left_closure_percent" in metrics:
        result.action_specific["closure_metrics"] = {
            "left_closure_percent": metrics["left_closure_percent"],
            "right_closure_percent": metrics["right_closure_percent"],
            "left_closure_ratio": metrics["left_closure_ratio"],
            "right_closure_ratio": metrics["right_closure_ratio"],
        }

    action_dir = output_dir / action_name
    action_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(action_dir / "peak_raw.jpg"), peak_frame)

    vis = visualize_close_eye(peak_frame, peak_landmarks, w, h, result, metrics, palsy_detection)
    cv2.imwrite(str(action_dir / "peak_indicators.jpg"), vis)

    # 绘制曲线 - 上边面积，下边闭合度
    plot_eye_curve(eye_seq, fps, peak_idx, action_dir / "peak_selection_curve.png", action_name,
                   closure_data=closure_data,
                   valid_mask=None,
                   palsy_detection=palsy_detection)

    with open(action_dir / "indicators.json", 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"    [OK] {action_name}: Area L={metrics['left_area']:.0f} R={metrics['right_area']:.0f}")
    if "left_closure_percent" in metrics:
        print(f"         Closure L={metrics['left_closure_percent']:.1f}% R={metrics['right_closure_percent']:.1f}%")
    print(f"         Palsy: {palsy_detection.get('interpretation', 'N/A')}")
    print(f"         Voluntary Score: {result.voluntary_movement_score}/5 ({interpretation})")

    return result