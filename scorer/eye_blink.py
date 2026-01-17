#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
眨眼动作处理模块
==========================================================

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
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from clinical_base import (
    LM, compute_ear, compute_eye_area, extract_common_indicators,
    ActionResult, draw_polygon, compute_scale_to_baseline,
    add_valid_region_shading, get_palsy_side_text,
    compute_eye_closure_by_area, draw_palsy_annotation_header,
)

from thresholds import THR

# OpenCV字体
FONT = cv2.FONT_HERSHEY_SIMPLEX

# 字体大小
FONT_SCALE_TITLE = 1.4  # 标题
FONT_SCALE_LARGE = 1.2  # 大号文字
FONT_SCALE_NORMAL = 0.9  # 正常文字
FONT_SCALE_SMALL = 0.7  # 小号文字

# 线条粗细
THICKNESS_TITLE = 3
THICKNESS_NORMAL = 2
THICKNESS_THIN = 1

# 行高
LINE_HEIGHT = 45
LINE_HEIGHT_SMALL = 30


def find_peak_frame_blink(landmarks_seq: List, frames_seq: List, w: int, h: int,
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


def extract_eye_area_sequence(landmarks_seq: List, w: int, h: int) -> Dict[str, List[float]]:
    """提取整个序列的眼睛面积"""
    left_area_seq = []
    right_area_seq = []

    for lm in landmarks_seq:
        if lm is None:
            left_area_seq.append(np.nan)
            right_area_seq.append(np.nan)
        else:
            l_area, _ = compute_eye_area(lm, w, h, True)
            r_area, _ = compute_eye_area(lm, w, h, False)
            left_area_seq.append(l_area)
            right_area_seq.append(r_area)

    return {
        "left": left_area_seq,
        "right": right_area_seq
    }


def compute_closure_sequence(landmarks_seq: List, w: int, h: int,
                             baseline_landmarks=None) -> Dict[str, Any]:
    """
    计算整个视频序列的眼睛闭合度和睁眼度（基于面积）

    openness = scaled_area / baseline_area  (1=完全睁开, 0=完全闭合)
    closure = 1 - openness
    """
    n = len(landmarks_seq)

    left_area_seq = []
    right_area_seq = []
    left_closure_seq = []
    right_closure_seq = []
    left_openness_seq = []
    right_openness_seq = []

    if baseline_landmarks is not None:
        baseline_left_area, _ = compute_eye_area(baseline_landmarks, w, h, True)
        baseline_right_area, _ = compute_eye_area(baseline_landmarks, w, h, False)
        baseline_source = "neutral"
    else:
        baseline_source = "video_init"
        init_left = []
        init_right = []
        for lm in landmarks_seq[:min(10, n)]:
            if lm is not None:
                la, _ = compute_eye_area(lm, w, h, True)
                ra, _ = compute_eye_area(lm, w, h, False)
                if la > 0:
                    init_left.append(la)
                if ra > 0:
                    init_right.append(ra)

        baseline_left_area = float(np.percentile(init_left, 90)) if init_left else 1.0
        baseline_right_area = float(np.percentile(init_right, 90)) if init_right else 1.0

    baseline_left_area = max(baseline_left_area, 1e-6)
    baseline_right_area = max(baseline_right_area, 1e-6)

    for lm in landmarks_seq:
        if lm is None:
            left_area_seq.append(np.nan)
            right_area_seq.append(np.nan)
            left_closure_seq.append(np.nan)
            right_closure_seq.append(np.nan)
            left_openness_seq.append(np.nan)
            right_openness_seq.append(np.nan)
        else:
            scale = compute_scale_to_baseline(lm, baseline_landmarks, w, h) if baseline_landmarks else 1.0

            l_area, _ = compute_eye_area(lm, w, h, True)
            r_area, _ = compute_eye_area(lm, w, h, False)

            scaled_l = l_area * (scale ** 2)
            scaled_r = r_area * (scale ** 2)

            left_area_seq.append(float(scaled_l))
            right_area_seq.append(float(scaled_r))

            # 直接计算睁眼度
            l_openness = scaled_l / baseline_left_area
            r_openness = scaled_r / baseline_right_area

            left_openness_seq.append(float(max(0, min(2, l_openness))))
            right_openness_seq.append(float(max(0, min(2, r_openness))))
            left_closure_seq.append(float(max(0, min(1, 1.0 - l_openness))))
            right_closure_seq.append(float(max(0, min(1, 1.0 - r_openness))))

    return {
        "left_closure": left_closure_seq,
        "right_closure": right_closure_seq,
        "left_openness": left_openness_seq,
        "right_openness": right_openness_seq,
        "left_area": left_area_seq,
        "right_area": right_area_seq,
        "baseline_left_area": float(baseline_left_area),
        "baseline_right_area": float(baseline_right_area),
        "baseline_source": baseline_source,
    }


def analyze_blink_dynamics(area_seq: Dict[str, List[float]],
                           fps: float,
                           baseline_left_area: float = None,
                           baseline_right_area: float = None,
                           scale_seq: List[float] = None) -> Dict[str, Any]:
    """分析眨眼动态特征"""
    left_area = np.array(area_seq["left"])
    right_area = np.array(area_seq["right"])

    valid_mask = ~np.isnan(left_area) & ~np.isnan(right_area)
    if not np.any(valid_mask):
        return {}

    valid_left_area = left_area[valid_mask]
    valid_right_area = right_area[valid_mask]

    left_max_area = float(np.max(valid_left_area))
    left_min_area = float(np.min(valid_left_area))
    right_max_area = float(np.max(valid_right_area))
    right_min_area = float(np.min(valid_right_area))

    total_frames = len(left_area)
    duration_sec = total_frames / fps if fps > 0 else 0

    result = {
        "left_area_max": left_max_area,
        "left_area_min": left_min_area,
        "right_area_max": right_max_area,
        "right_area_min": right_min_area,
        "total_frames": total_frames,
        "duration_sec": duration_sec,
        "fps": fps,
    }

    if baseline_left_area is not None and baseline_right_area is not None:
        if scale_seq is not None and len(scale_seq) == len(left_area):
            scales = np.array(scale_seq)
            scaled_left = left_area * (scales ** 2)
            scaled_right = right_area * (scales ** 2)
        else:
            scaled_left = left_area
            scaled_right = right_area

        left_closure_seq = 1.0 - (scaled_left / baseline_left_area)
        right_closure_seq = 1.0 - (scaled_right / baseline_right_area)

        left_closure_seq = np.clip(left_closure_seq, 0, 1)
        right_closure_seq = np.clip(right_closure_seq, 0, 1)

        valid_left_closure = left_closure_seq[valid_mask]
        valid_right_closure = right_closure_seq[valid_mask]

        left_max_closure = float(np.max(valid_left_closure))
        right_max_closure = float(np.max(valid_right_closure))

        result["left_closure_ratio"] = left_max_closure
        result["right_closure_ratio"] = right_max_closure
        result["baseline_left_area"] = baseline_left_area
        result["baseline_right_area"] = baseline_right_area

    return result


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
    """
    result = {
        "palsy_side": 0,
        "confidence": 0.0,
        "method": "frame_by_frame_area_comparison",
        "interpretation": "",
        "evidence": {}
    }

    # 提取数据
    left_closure_seq = closure_data.get("left_closure", [])
    right_closure_seq = closure_data.get("right_closure", [])
    left_area_seq = closure_data.get("left_area", [])
    right_area_seq = closure_data.get("right_area", [])

    left_closure_peak = left_closure_seq[peak_idx] if peak_idx < len(left_closure_seq) else 0
    right_closure_peak = right_closure_seq[peak_idx] if peak_idx < len(right_closure_seq) else 0

    # 逐帧对比
    left_worse_count = 0
    right_worse_count = 0
    total_closing_frames = 0

    for i, (lc, rc, la, ra) in enumerate(zip(
            left_closure_seq, right_closure_seq, left_area_seq, right_area_seq)):

        if not (np.isfinite(lc) and np.isfinite(rc) and np.isfinite(la) and np.isfinite(ra)):
            continue

        avg_closure = (lc + rc) / 2

        if avg_closure >= THR.MIN_CLOSURE_FOR_COUNT:
            total_closing_frames += 1

            max_area = max(la, ra)
            if max_area > 1:
                area_diff_ratio = abs(la - ra) / max_area

                if area_diff_ratio >= THR.EYE_AREA_DIFF_THRESHOLD:
                    if la > ra:
                        left_worse_count += 1
                    else:
                        right_worse_count += 1

    # 计算比例
    if total_closing_frames > 0:
        left_worse_ratio = left_worse_count / total_closing_frames
        right_worse_ratio = right_worse_count / total_closing_frames
    else:
        left_worse_ratio = 0
        right_worse_ratio = 0

    evidence = {
        "left_closure_at_peak": float(left_closure_peak),
        "right_closure_at_peak": float(right_closure_peak),
        "total_closing_frames": total_closing_frames,
        "left_worse_count": left_worse_count,
        "right_worse_count": right_worse_count,
        "left_worse_ratio": float(left_worse_ratio),
        "right_worse_ratio": float(right_worse_ratio),
    }

    result["evidence"] = evidence

    if total_closing_frames < 5:
        result["interpretation"] = f"闭眼帧数不足 ({total_closing_frames}帧)，无法判断"
        return result

    if left_worse_ratio >= THR.FRAME_RATIO_THRESHOLD and left_worse_ratio > right_worse_ratio:
        result["palsy_side"] = 1
        result["confidence"] = min(1.0, left_worse_ratio / 0.60)
        result["interpretation"] = (
            f"左侧面瘫: {left_worse_count}/{total_closing_frames}帧 "
            f"({left_worse_ratio:.0%}) 左眼闭合差"
        )
    elif right_worse_ratio >= THR.FRAME_RATIO_THRESHOLD and right_worse_ratio > left_worse_ratio:
        result["palsy_side"] = 2
        result["confidence"] = min(1.0, right_worse_ratio / 0.60)
        result["interpretation"] = (
            f"右侧面瘫: {right_worse_count}/{total_closing_frames}帧 "
            f"({right_worse_ratio:.0%}) 右眼闭合差"
        )
    else:
        result["palsy_side"] = 0
        result["confidence"] = 1.0 - max(left_worse_ratio, right_worse_ratio)
        result["interpretation"] = (
            f"双眼闭合对称 (左差{left_worse_ratio:.0%}, 右差{right_worse_ratio:.0%})"
        )

    return result


def compute_severity_score(dynamics: Dict[str, Any],
                           peak_closure_data: Dict[str, Any] = None) -> Tuple[int, str]:
    """计算严重度分数"""
    if peak_closure_data is not None:
        left_closure = peak_closure_data.get("left_closure_ratio", 0) or 0
        right_closure = peak_closure_data.get("right_closure_ratio", 0) or 0
    else:
        left_closure = dynamics.get("left_closure_ratio", 0) or 0
        right_closure = dynamics.get("right_closure_ratio", 0) or 0

    max_closure = max(left_closure, right_closure)
    min_closure = min(left_closure, right_closure)

    if max_closure < 0.20:
        return 1, f"眨眼幅度过小 (L={left_closure * 100:.1f}%, R={right_closure * 100:.1f}%)"

    symmetry_ratio = min_closure / max_closure if max_closure > 0.01 else 1.0

    if symmetry_ratio >= 0.92:
        return 1, f"正常 (对称性{symmetry_ratio:.2%})"
    elif symmetry_ratio >= 0.78:
        return 2, f"轻度异常 (对称性{symmetry_ratio:.2%})"
    elif symmetry_ratio >= 0.58:
        return 3, f"中度异常 (对称性{symmetry_ratio:.2%})"
    elif symmetry_ratio >= 0.35:
        return 4, f"重度异常 (对称性{symmetry_ratio:.2%})"
    else:
        return 5, f"完全面瘫 (对称性{symmetry_ratio:.2%})"


def detect_synkinesis_from_blink(baseline_result: Optional[ActionResult],
                                 current_result: ActionResult) -> Dict[str, int]:
    """检测眨眼时的联动运动"""
    synkinesis = {"mouth_synkinesis": 0, "cheek_synkinesis": 0}

    if baseline_result is None:
        return synkinesis

    baseline_mouth_w = baseline_result.mouth_width
    current_mouth_w = current_result.mouth_width

    if baseline_mouth_w > 1e-9:
        mouth_change = abs(current_mouth_w - baseline_mouth_w) / baseline_mouth_w
        if mouth_change > 0.20:
            synkinesis["mouth_synkinesis"] = 3
        elif mouth_change > 0.10:
            synkinesis["mouth_synkinesis"] = 2
        elif mouth_change > 0.05:
            synkinesis["mouth_synkinesis"] = 1

    return synkinesis


def plot_eye_curve(area_seq: Dict[str, List[float]],
                   fps: float,
                   peak_idx: int,
                   output_path: Path,
                   action_name: str,
                   closure_data: Dict = None,
                   valid_mask: List[bool] = None,
                   palsy_detection: Dict[str, Any] = None) -> None:
    """
    绘制眼睛曲线 - 修改版
    上图：眼睛面积
    下图：眼睛闭合度
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 9))

    frames = np.arange(len(area_seq["left"]))
    time_sec = frames / fps if fps > 0 else frames
    x_label = 'Time (seconds)' if fps > 0 else 'Frame'
    peak_time = time_sec[peak_idx] if peak_idx < len(time_sec) else 0

    # ========== 上图: 眼睛面积 ==========
    ax1 = axes[0]
    if valid_mask is not None:
        add_valid_region_shading(ax1, valid_mask, time_sec)

    ax1.plot(time_sec, area_seq["left"], 'b-', label='Left Eye Area', linewidth=2)
    ax1.plot(time_sec, area_seq["right"], 'r-', label='Right Eye Area', linewidth=2)
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

    # ========== 下图: 眼睛睁眼度 ==========
    ax2 = axes[1]

    if closure_data is not None:
        left_openness = closure_data.get("left_openness", [])
        right_openness = closure_data.get("right_openness", [])

        if len(left_openness) > 0 and len(right_openness) > 0:
            if valid_mask is not None:
                add_valid_region_shading(ax2, valid_mask, time_sec)

            ax2.plot(time_sec, left_openness, 'b-', label='Left Eye Openness', linewidth=2)
            ax2.plot(time_sec, right_openness, 'r-', label='Right Eye Openness', linewidth=2)

            avg_openness = [(l + r) / 2 if (np.isfinite(l) and np.isfinite(r)) else np.nan
                            for l, r in zip(left_openness, right_openness)]
            ax2.plot(time_sec, avg_openness, 'g--', label='Average', linewidth=1.5, alpha=0.7)

            ax2.axvline(x=peak_time, color='k', linestyle='--', alpha=0.5, label=f'Peak ({peak_idx})')
            ax2.axhline(y=0.15, color='red', linestyle=':', alpha=0.5, label='Closed (<15%)')
            ax2.axhline(y=0.80, color='green', linestyle=':', alpha=0.5, label='Open (>80%)')

            ax2.set_ylim(-0.05, 1.2)
            ax2.set_ylabel('Openness (0=closed, 1=open)', fontsize=11)

            if 0 <= peak_idx < len(left_openness):
                lp = left_openness[peak_idx] if np.isfinite(left_openness[peak_idx]) else 0
                rp = right_openness[peak_idx] if np.isfinite(right_openness[peak_idx]) else 0
                ax2.text(0.02, 0.98, f"Peak: L={lp:.1%}, R={rp:.1%}", transform=ax2.transAxes,
                         fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax2.set_title(f'{action_name} - Eye Openness Over Time', fontsize=13)
    ax2.set_xlabel(x_label, fontsize=11)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()


def visualize_blink_indicators(frame: np.ndarray, landmarks, w: int, h: int,
                               result: ActionResult,
                               dynamics: Dict[str, Any],
                               palsy_detection: Dict[str, Any] = None) -> np.ndarray:
    """可视化眨眼指标"""
    img = frame.copy()

    # ========== 第一行绘制患侧标注 ==========
    img, header_end_y = draw_palsy_annotation_header(img, palsy_detection, result.action_name)

    draw_polygon(img, landmarks, w, h, LM.EYE_CONTOUR_L, (255, 0, 0), 2)
    draw_polygon(img, landmarks, w, h, LM.EYE_CONTOUR_R, (0, 165, 255), 2)

    y = header_end_y + 10  # 从标注下方开始
    cv2.putText(img, f"{result.action_name}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y += 30

    cv2.putText(img, f"Peak Frame: {result.peak_frame_idx}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y += 22

    cv2.putText(img, f"Eye Area L: {result.left_eye_area:.1f}  R: {result.right_eye_area:.1f}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y += 25

    if dynamics:
        cv2.putText(img, "=== Blink Dynamics ===", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y += 22

        cv2.putText(img, f"Left Closure: {dynamics.get('left_closure_ratio', 0) * 100:.1f}%", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 18

        cv2.putText(img, f"Right Closure: {dynamics.get('right_closure_ratio', 0) * 100:.1f}%", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 25

    # 逐帧对比结果
    if palsy_detection and "evidence" in palsy_detection:
        ev = palsy_detection["evidence"]
        cv2.putText(img, "=== Frame Analysis ===", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y += 20
        left_worse = ev.get("left_worse_ratio", 0)
        right_worse = ev.get("right_worse_ratio", 0)
        cv2.putText(img, f"Left worse: {left_worse:.0%}  Right worse: {right_worse:.0%}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 25

    # 面瘫判定
    if palsy_detection:
        palsy_side = palsy_detection.get("palsy_side", 0)
        palsy_text = {0: "Symmetric", 1: "LEFT PALSY", 2: "RIGHT PALSY"}.get(palsy_side, "Unknown")
        palsy_color = (0, 255, 0) if palsy_side == 0 else (0, 0, 255)
        cv2.putText(img, f"Result: {palsy_text}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, palsy_color, 2)

    return img


def process_voluntary_blink(landmarks_seq: List, frames_seq: List, w: int, h: int,
                            video_info: Dict[str, Any], output_dir: Path,
                            baseline_result: Optional[ActionResult] = None,
                            baseline_landmarks=None) -> Optional[ActionResult]:
    """处理自主眨眼"""
    return _process_blink(
        landmarks_seq, frames_seq, w, h, video_info, output_dir,
        action_name="VoluntaryEyeBlink",
        action_name_cn="自主眨眼",
        baseline_result=baseline_result,
        baseline_landmarks=baseline_landmarks
    )


def process_spontaneous_blink(landmarks_seq: List, frames_seq: List, w: int, h: int,
                              video_info: Dict[str, Any], output_dir: Path,
                              baseline_result: Optional[ActionResult] = None,
                              baseline_landmarks=None) -> Optional[ActionResult]:
    """处理自然眨眼"""
    return _process_blink(
        landmarks_seq, frames_seq, w, h, video_info, output_dir,
        action_name="SpontaneousEyeBlink",
        action_name_cn="自然眨眼",
        baseline_result=baseline_result,
        baseline_landmarks=baseline_landmarks
    )


def _process_blink(landmarks_seq: List, frames_seq: List, w: int, h: int,
                   video_info: Dict[str, Any], output_dir: Path,
                   action_name: str, action_name_cn: str,
                   baseline_result: Optional[ActionResult] = None,
                   baseline_landmarks=None) -> Optional[ActionResult]:
    """处理眨眼动作的通用函数"""
    if not landmarks_seq or not frames_seq:
        return None

    baseline_left_area = None
    baseline_right_area = None
    if baseline_landmarks is not None:
        baseline_left_area, _ = compute_eye_area(baseline_landmarks, w, h, True)
        baseline_right_area, _ = compute_eye_area(baseline_landmarks, w, h, False)

    peak_idx = find_peak_frame_blink(landmarks_seq, frames_seq, w, h, baseline_landmarks)
    peak_landmarks = landmarks_seq[peak_idx]
    peak_frame = frames_seq[peak_idx]

    if peak_landmarks is None:
        return None

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

    extract_common_indicators(peak_landmarks, w, h, result)
    area_seq = extract_eye_area_sequence(landmarks_seq, w, h)

    # 计算闭合度序列
    closure_data = compute_closure_sequence(landmarks_seq, w, h, baseline_landmarks)

    scale_seq = None
    if baseline_landmarks is not None:
        scale_seq = []
        for lm in landmarks_seq:
            if lm is not None:
                s = compute_scale_to_baseline(lm, baseline_landmarks, w, h)
                scale_seq.append(s)
            else:
                scale_seq.append(1.0)

    dynamics = analyze_blink_dynamics(
        area_seq, fps,
        baseline_left_area, baseline_right_area,
        scale_seq
    )

    peak_closure_data = compute_eye_closure_by_area(
        peak_landmarks, w, h, baseline_landmarks
    )

    # ========== 面瘫侧别检测 - 逐帧对比 ==========
    palsy_detection = detect_palsy_side(
        closure_data=closure_data,
        peak_idx=peak_idx,
    )

    synkinesis = detect_synkinesis_from_blink(baseline_result, result)
    result.synkinesis_scores = synkinesis

    severity_score, severity_desc = compute_severity_score(dynamics, peak_closure_data)

    result.action_specific = {
        "blink_dynamics": dynamics,
        "peak_closure_data": peak_closure_data,
        "eye_area_sequence": {
            "left": [float(x) if not np.isnan(x) else None for x in area_seq["left"]],
            "right": [float(x) if not np.isnan(x) else None for x in area_seq["right"]],
        },
        "closure_sequence": {
            "left": [float(v) if np.isfinite(v) else None for v in closure_data["left_closure"]],
            "right": [float(v) if np.isfinite(v) else None for v in closure_data["right_closure"]],
        },
        "openness_sequence": {
            "left": [float(v) if np.isfinite(v) else None for v in closure_data["left_openness"]],
            "right": [float(v) if np.isfinite(v) else None for v in closure_data["right_openness"]],
        },
        "baseline_areas": {
            "left": closure_data["baseline_left_area"],
            "right": closure_data["baseline_right_area"],
        },
        "synkinesis": synkinesis,
        "palsy_detection": palsy_detection,
        "severity_score": severity_score,
        "severity_desc": severity_desc,
    }

    action_dir = output_dir / action_name
    action_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(action_dir / "peak_raw.jpg"), peak_frame)

    vis_indicators = visualize_blink_indicators(peak_frame, peak_landmarks, w, h, result,
                                                dynamics, palsy_detection)
    cv2.imwrite(str(action_dir / "peak_indicators.jpg"), vis_indicators)

    # 绘制曲线 - 上边面积，下边闭合度
    plot_eye_curve(area_seq, fps, peak_idx,
                   action_dir / "peak_selection_curve.png", action_name,
                   closure_data=closure_data,
                   palsy_detection=palsy_detection)

    import json
    with open(action_dir / "indicators.json", 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    print(
        f"    [OK] {action_name}: Area L={peak_closure_data.get('left_area', 0):.0f} R={peak_closure_data.get('right_area', 0):.0f}")
    print(
        f"         Closure: L={dynamics.get('left_closure_ratio', 0) * 100:.1f}% R={dynamics.get('right_closure_ratio', 0) * 100:.1f}%")
    print(f"         Palsy: {palsy_detection.get('interpretation', 'N/A')}")

    return result