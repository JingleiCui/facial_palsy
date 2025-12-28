#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
眨眼动作处理模块 (VoluntaryEyeBlink / SpontaneousEyeBlink)
==========================================================

分析眨眼过程中:
1. 眼睛睁开度变化曲线 (EAR曲线)
2. 左右眼对称性
3. 闭眼完整度
4. 联动运动检测
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import matplotlib

matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

from clinical_base import (
    LM, pt2d, pts2d, dist, compute_ear, compute_eye_area,
    compute_palpebral_height, compute_mouth_metrics,
    compute_oral_angle, compute_icd, extract_common_indicators,
    ActionResult, draw_polygon, compute_scale_to_baseline,
    add_valid_region_shading, get_palsy_side_text,
    compute_eye_closure_by_area,
)

from thresholds import THR


def find_peak_frame_blink(landmarks_seq: List, frames_seq: List, w: int, h: int,
                          baseline_landmarks=None) -> int:
    """
    找闭眼峰值帧 - 使用眼睛面积而不是EAR

    改进: 使用眼睛面积相对于基线的闭合程度来找峰值帧
    这样可以解决天生大小眼导致的误判问题
    """
    if baseline_landmarks is not None:
        # 有基线：使用面积闭合度
        baseline_left_area, _ = compute_eye_area(baseline_landmarks, w, h, True)
        baseline_right_area, _ = compute_eye_area(baseline_landmarks, w, h, False)

        max_closure = -float('inf')
        max_idx = 0

        for i, lm in enumerate(landmarks_seq):
            if lm is None:
                continue

            # 计算尺度因子
            scale = compute_scale_to_baseline(lm, baseline_landmarks, w, h)

            # 当前眼睛面积（换算到基线尺度）
            l_area, _ = compute_eye_area(lm, w, h, True)
            r_area, _ = compute_eye_area(lm, w, h, False)
            scaled_l_area = l_area * (scale ** 2)
            scaled_r_area = r_area * (scale ** 2)

            # 闭合度 = 1 - (当前面积/基线面积)
            l_closure = 1.0 - (scaled_l_area / baseline_left_area) if baseline_left_area > 1e-6 else 0
            r_closure = 1.0 - (scaled_r_area / baseline_right_area) if baseline_right_area > 1e-6 else 0

            # 平均闭合度（限制在0-1范围）
            avg_closure = (max(0, min(1, l_closure)) + max(0, min(1, r_closure))) / 2

            if avg_closure > max_closure:
                max_closure = avg_closure
                max_idx = i

        return max_idx
    else:
        # 没有基线：使用面积最小的帧
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


def extract_ear_sequence(landmarks_seq: List, w: int, h: int) -> Dict[str, List[float]]:
    """提取整个序列的EAR值"""
    left_ear_seq = []
    right_ear_seq = []
    avg_ear_seq = []

    for lm in landmarks_seq:
        if lm is None:
            left_ear_seq.append(np.nan)
            right_ear_seq.append(np.nan)
            avg_ear_seq.append(np.nan)
        else:
            l_ear = compute_ear(lm, w, h, True)
            r_ear = compute_ear(lm, w, h, False)
            left_ear_seq.append(l_ear)
            right_ear_seq.append(r_ear)
            avg_ear_seq.append((l_ear + r_ear) / 2)

    return {
        "left": left_ear_seq,
        "right": right_ear_seq,
        "average": avg_ear_seq
    }


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


def analyze_blink_dynamics(ear_seq: Dict[str, List[float]],
                           area_seq: Dict[str, List[float]],
                           fps: float,
                           baseline_left_area: float = None,
                           baseline_right_area: float = None,
                           scale_seq: List[float] = None) -> Dict[str, Any]:
    """
    分析眨眼动态特征 - 增强版，使用眼睛面积

    Args:
        ear_seq: EAR序列（保留用于参考）
        area_seq: 眼睛面积序列
        fps: 帧率
        baseline_left_area: 基线左眼面积
        baseline_right_area: 基线右眼面积
        scale_seq: 尺度因子序列（用于换算到基线尺度）
    """
    left_area = np.array(area_seq["left"])
    right_area = np.array(area_seq["right"])
    avg_ear = np.array(ear_seq["average"])

    # 处理nan
    valid_mask = ~np.isnan(left_area) & ~np.isnan(right_area)
    if not np.any(valid_mask):
        return {}

    valid_left_area = left_area[valid_mask]
    valid_right_area = right_area[valid_mask]
    valid_indices = np.where(valid_mask)[0]

    # 基本统计
    left_max_area = float(np.max(valid_left_area))
    left_min_area = float(np.min(valid_left_area))
    right_max_area = float(np.max(valid_right_area))
    right_min_area = float(np.min(valid_right_area))

    # 时间分析
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

    # 如果有基线，计算基于面积的闭合度
    if baseline_left_area is not None and baseline_right_area is not None:
        # 计算每帧的闭合度
        if scale_seq is not None and len(scale_seq) == len(left_area):
            scales = np.array(scale_seq)
            scaled_left = left_area * (scales ** 2)
            scaled_right = right_area * (scales ** 2)
        else:
            scaled_left = left_area
            scaled_right = right_area

        # 闭合度 = 1 - (当前面积/基线面积)
        left_closure_seq = 1.0 - (scaled_left / baseline_left_area)
        right_closure_seq = 1.0 - (scaled_right / baseline_right_area)

        # 限制在0-1范围
        left_closure_seq = np.clip(left_closure_seq, 0, 1)
        right_closure_seq = np.clip(right_closure_seq, 0, 1)

        # 有效值
        valid_left_closure = left_closure_seq[valid_mask]
        valid_right_closure = right_closure_seq[valid_mask]

        # 最大闭合度（峰值）
        left_max_closure = float(np.max(valid_left_closure))
        right_max_closure = float(np.max(valid_right_closure))

        result["left_closure_ratio"] = left_max_closure
        result["right_closure_ratio"] = right_max_closure
        result["baseline_left_area"] = baseline_left_area
        result["baseline_right_area"] = baseline_right_area

        # 持续性不对称检测（使用面积闭合度）
        closure_diff = valid_left_closure - valid_right_closure
        left_worse_ratio = np.mean(closure_diff < -0.05)  # 左眼闭合度低（闭合差）
        right_worse_ratio = np.mean(closure_diff > 0.05)  # 右眼闭合度低

        result["left_worse_frame_ratio"] = float(left_worse_ratio)
        result["right_worse_frame_ratio"] = float(right_worse_ratio)

        # 持续性不对称侧
        if left_worse_ratio > 0.6:
            result["persistent_asymmetry_side"] = 1
            result["persistent_asymmetry_ratio"] = left_worse_ratio
        elif right_worse_ratio > 0.6:
            result["persistent_asymmetry_side"] = 2
            result["persistent_asymmetry_ratio"] = right_worse_ratio
        else:
            result["persistent_asymmetry_side"] = 0
            result["persistent_asymmetry_ratio"] = 0
    else:
        # 没有基线，使用面积变化比例
        left_closure = (left_max_area - left_min_area) / left_max_area if left_max_area > 1e-6 else 0
        right_closure = (right_max_area - right_min_area) / right_max_area if right_max_area > 1e-6 else 0

        result["left_closure_ratio"] = float(left_closure)
        result["right_closure_ratio"] = float(right_closure)

        # 基于面积的持续性不对称
        area_diff = valid_left_area - valid_right_area
        # 面积大的眼睛 = 闭合差的眼睛
        avg_area = (valid_left_area + valid_right_area) / 2
        rel_diff = area_diff / (avg_area + 1e-6)

        left_worse_ratio = np.mean(rel_diff > 0.10)  # 左眼面积大（闭合差）
        right_worse_ratio = np.mean(rel_diff < -0.10)

        result["left_worse_frame_ratio"] = float(left_worse_ratio)
        result["right_worse_frame_ratio"] = float(right_worse_ratio)

        if left_worse_ratio > 0.6:
            result["persistent_asymmetry_side"] = 1
            result["persistent_asymmetry_ratio"] = left_worse_ratio
        elif right_worse_ratio > 0.6:
            result["persistent_asymmetry_side"] = 2
            result["persistent_asymmetry_ratio"] = right_worse_ratio
        else:
            result["persistent_asymmetry_side"] = 0
            result["persistent_asymmetry_ratio"] = 0

    return result


def plot_ear_curve(ear_seq: Dict[str, List[float]],
                   area_seq: Dict[str, List[float]],
                   fps: float,
                   peak_idx: int,
                   output_path: Path,
                   action_name: str,
                   valid_mask: List[bool] = None,
                   palsy_detection: Dict[str, Any] = None) -> None:
    """绘制EAR变化曲线"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 9))

    frames = np.arange(len(ear_seq["left"]))
    time_sec = frames / fps if fps > 0 else frames

    # 上图: EAR曲线
    ax1 = axes[0]
    if valid_mask is not None:
        add_valid_region_shading(ax1, valid_mask, time_sec)

    ax1.plot(time_sec, ear_seq["left"], 'b-', label='Left Eye EAR', linewidth=2)
    ax1.plot(time_sec, ear_seq["right"], 'r-', label='Right Eye EAR', linewidth=2)
    ax1.plot(time_sec, ear_seq["average"], 'g--', label='Average EAR', linewidth=1.5, alpha=0.7)
    ax1.axvline(x=time_sec[peak_idx], color='k', linestyle='--', alpha=0.5, label=f'Peak Frame ({peak_idx})')
    ax1.set_xlabel('Time (seconds)' if fps > 0 else 'Frame', fontsize=11)
    ax1.set_ylabel('Eye Aspect Ratio (EAR)', fontsize=11)
    title = f'{action_name} - Eye Aspect Ratio Over Time'
    if palsy_detection:
        palsy_text = get_palsy_side_text(palsy_detection.get("palsy_side", 0))
        title += f' | Detected: {palsy_text}'
    ax1.set_title(title, fontsize=13)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(max(ear_seq["left"]), max(ear_seq["right"])) * 1.1)

    # 下图: 眼睛面积曲线
    ax2 = axes[1]
    if valid_mask is not None:
        add_valid_region_shading(ax2, valid_mask, time_sec)

    ax2.plot(time_sec, area_seq["left"], 'b-', label='Left Eye Area', linewidth=2)
    ax2.plot(time_sec, area_seq["right"], 'r-', label='Right Eye Area', linewidth=2)
    ax2.axvline(x=time_sec[peak_idx], color='k', linestyle='--', alpha=0.5, label=f'Peak Frame ({peak_idx})')
    ax2.set_xlabel('Time (seconds)' if fps > 0 else 'Frame', fontsize=11)
    ax2.set_ylabel('Eye Area (pixels²)', fontsize=11)
    ax2.set_title(f'{action_name} - Eye Area Over Time', fontsize=13)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()


def visualize_blink_indicators(frame: np.ndarray, landmarks, w: int, h: int,
                               result: ActionResult,
                               dynamics: Dict[str, Any]) -> np.ndarray:
    """可视化眨眼指标"""
    img = frame.copy()

    # 绘制眼部轮廓
    draw_polygon(img, landmarks, w, h, LM.EYE_CONTOUR_L, (255, 0, 0), 2)
    draw_polygon(img, landmarks, w, h, LM.EYE_CONTOUR_R, (0, 165, 255), 2)

    # 信息面板
    y = 25
    cv2.putText(img, f"{result.action_name}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y += 30

    cv2.putText(img, f"Peak Frame: {result.peak_frame_idx}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y += 22

    cv2.putText(img, f"EAR L: {result.left_ear:.4f}  R: {result.right_ear:.4f}", (10, y),
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
        y += 18

        cv2.putText(img, f"Symmetry Ratio: {dynamics.get('symmetry_ratio_mean', 1):.3f}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 18

        cv2.putText(img, f"Duration: {dynamics.get('duration_sec', 0):.2f}s ({dynamics.get('total_frames', 0)} frames)",
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return img


def detect_palsy_side(dynamics: Dict[str, Any],
                      peak_closure_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    从眨眼动作检测面瘫侧别 - 基于眼睛面积

    原理:
    1. 面瘫侧眼睛闭合比例低（相对于基线面积的减小比例）
    2. 面瘫侧在整段视频中持续表现出闭合差（面积相对更大）

    Args:
        dynamics: 眨眼动态分析结果
        peak_closure_data: 峰值帧的眼睛闭合数据（来自compute_eye_closure_by_area）
    """
    result = {
        "palsy_side": 0,
        "confidence": 0.0,
        "interpretation": "",
        "method": "",
        "evidence": {}
    }

    left_closure = dynamics.get("left_closure_ratio", 0)
    right_closure = dynamics.get("right_closure_ratio", 0)

    result["evidence"] = {
        "left_closure_pct": left_closure * 100,
        "right_closure_pct": right_closure * 100,
    }

    max_closure = max(left_closure, right_closure)

    # 检查是否有足够的闭眼动作
    if max_closure < 0.20:  # 闭合不足20%
        result["method"] = "insufficient"
        result["interpretation"] = f"眨眼幅度过小 (L={left_closure * 100:.1f}%, R={right_closure * 100:.1f}%)"
        result["evidence"]["status"] = "insufficient_movement"
        return result

    # 方法1：使用峰值帧的面积闭合数据（优先）
    if peak_closure_data is not None:
        result["method"] = "area_closure"

        peak_left_closure = peak_closure_data.get("left_closure_ratio", left_closure)
        peak_right_closure = peak_closure_data.get("right_closure_ratio", right_closure)

        result["evidence"]["peak_left_closure_pct"] = peak_left_closure * 100
        result["evidence"]["peak_right_closure_pct"] = peak_right_closure * 100

        max_peak_closure = max(peak_left_closure, peak_right_closure)
        if max_peak_closure > 0.1:
            asymmetry = abs(peak_left_closure - peak_right_closure) / max_peak_closure
            result["evidence"]["closure_asymmetry"] = asymmetry
            result["confidence"] = min(1.0, asymmetry * 3)

            if asymmetry < 0.15:
                result["palsy_side"] = 0
                result["interpretation"] = (
                    f"双眼闭合对称 (L={peak_left_closure * 100:.1f}%, R={peak_right_closure * 100:.1f}%)"
                )
            elif peak_left_closure < peak_right_closure:
                result["palsy_side"] = 1
                result["interpretation"] = (
                    f"左眼闭合差 (L={peak_left_closure * 100:.1f}% < R={peak_right_closure * 100:.1f}%) → 左侧面瘫"
                )
            else:
                result["palsy_side"] = 2
                result["interpretation"] = (
                    f"右眼闭合差 (R={peak_right_closure * 100:.1f}% < L={peak_left_closure * 100:.1f}%) → 右侧面瘫"
                )
            return result

    # 方法2：持续性不对称（面积方法）
    persistent_side = dynamics.get("persistent_asymmetry_side", 0)
    persistent_ratio = dynamics.get("persistent_asymmetry_ratio", 0)
    left_worse_ratio = dynamics.get("left_worse_frame_ratio", 0)
    right_worse_ratio = dynamics.get("right_worse_frame_ratio", 0)

    result["evidence"]["persistent_asymmetry_side"] = persistent_side
    result["evidence"]["left_worse_frame_ratio"] = left_worse_ratio
    result["evidence"]["right_worse_frame_ratio"] = right_worse_ratio

    if persistent_side != 0 and persistent_ratio > 0.6:
        result["method"] = "persistent_asymmetry"
        result["confidence"] = min(1.0, persistent_ratio)

        if persistent_side == 1:
            result["palsy_side"] = 1
            result["interpretation"] = (
                f"左眼持续闭合差 ({left_worse_ratio * 100:.0f}%帧) → 左侧面瘫"
            )
        else:
            result["palsy_side"] = 2
            result["interpretation"] = (
                f"右眼持续闭合差 ({right_worse_ratio * 100:.0f}%帧) → 右侧面瘫"
            )
        return result

    # 方法3：闭合比例比较
    result["method"] = "closure_ratio"

    asymmetry = abs(left_closure - right_closure) / max_closure if max_closure > 0 else 0
    result["confidence"] = min(1.0, asymmetry * 2.5)
    result["evidence"]["asymmetry_ratio"] = asymmetry

    if asymmetry < 0.15:
        result["palsy_side"] = 0
        result["interpretation"] = (
            f"双眼闭合对称 (L={left_closure * 100:.1f}%, R={right_closure * 100:.1f}%)"
        )
    elif left_closure < right_closure:
        result["palsy_side"] = 1
        result["interpretation"] = (
            f"左眼闭合弱 (L={left_closure * 100:.1f}% < R={right_closure * 100:.1f}%) → 左侧面瘫"
        )
    else:
        result["palsy_side"] = 2
        result["interpretation"] = (
            f"右眼闭合弱 (R={right_closure * 100:.1f}% < L={left_closure * 100:.1f}%) → 右侧面瘫"
        )

    return result


def detect_synkinesis_from_blink(baseline_result: Optional[ActionResult],
                                 current_result: ActionResult) -> Dict[str, int]:
    """检测眨眼时的联动运动"""
    synkinesis = {
        "mouth_synkinesis": 0,
        "cheek_synkinesis": 0,
    }

    if baseline_result is None:
        return synkinesis

    # 检测嘴部联动 (闭眼时嘴部变化)
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
    """处理眨眼动作的通用函数 - 增强版，使用眼睛面积"""
    if not landmarks_seq or not frames_seq:
        return None

    # 获取基线眼睛面积
    baseline_left_area = None
    baseline_right_area = None
    if baseline_landmarks is not None:
        baseline_left_area, _ = compute_eye_area(baseline_landmarks, w, h, True)
        baseline_right_area, _ = compute_eye_area(baseline_landmarks, w, h, False)

    # 找峰值帧 (使用眼睛面积)
    peak_idx = find_peak_frame_blink(landmarks_seq, frames_seq, w, h, baseline_landmarks)
    peak_landmarks = landmarks_seq[peak_idx]
    peak_frame = frames_seq[peak_idx]

    if peak_landmarks is None:
        return None

    fps = video_info.get("fps", 30.0)

    # 创建结果对象
    result = ActionResult(
        action_name=action_name,
        action_name_cn=action_name_cn,
        video_path=video_info.get("file_path", ""),
        total_frames=len(frames_seq),
        peak_frame_idx=peak_idx,
        image_size=(w, h),
        fps=fps
    )

    # 提取通用指标
    extract_common_indicators(peak_landmarks, w, h, result)

    # 提取EAR序列（保留用于曲线绘制）
    ear_seq = extract_ear_sequence(landmarks_seq, w, h)
    area_seq = extract_eye_area_sequence(landmarks_seq, w, h)

    # 计算尺度序列（如果有基线）
    scale_seq = None
    if baseline_landmarks is not None:
        scale_seq = []
        for lm in landmarks_seq:
            if lm is not None:
                s = compute_scale_to_baseline(lm, baseline_landmarks, w, h)
                scale_seq.append(s)
            else:
                scale_seq.append(1.0)

    # 分析眨眼动态
    dynamics = analyze_blink_dynamics(
        ear_seq, area_seq, fps,
        baseline_left_area, baseline_right_area,
        scale_seq
    )

    # 计算峰值帧的眼睛闭合数据
    peak_closure_data = compute_eye_closure_by_area(
        peak_landmarks, w, h, baseline_landmarks
    )

    # 获取面瘫侧别检测结果
    palsy_detection = detect_palsy_side(dynamics, peak_closure_data)

    # 检测联动
    synkinesis = detect_synkinesis_from_blink(baseline_result, result)
    result.synkinesis_scores = synkinesis

    # 存储动作特有指标
    result.action_specific = {
        "blink_dynamics": dynamics,
        "peak_closure_data": peak_closure_data,
        "ear_sequence": {
            "left": [float(x) if not np.isnan(x) else None for x in ear_seq["left"]],
            "right": [float(x) if not np.isnan(x) else None for x in ear_seq["right"]],
            "average": [float(x) if not np.isnan(x) else None for x in ear_seq["average"]],
        },
        "eye_area_sequence": {
            "left": [float(x) if not np.isnan(x) else None for x in area_seq["left"]],
            "right": [float(x) if not np.isnan(x) else None for x in area_seq["right"]],
        },
        "synkinesis": synkinesis,
        "palsy_detection": palsy_detection,
    }

    # 创建输出目录
    action_dir = output_dir / action_name
    action_dir.mkdir(parents=True, exist_ok=True)

    # 保存原始帧
    cv2.imwrite(str(action_dir / "peak_raw.jpg"), peak_frame)

    # 保存可视化
    vis_indicators = visualize_blink_indicators(peak_frame, peak_landmarks, w, h, result, dynamics)
    cv2.imwrite(str(action_dir / "peak_indicators.jpg"), vis_indicators)

    # 绘制EAR和面积曲线
    plot_ear_curve(ear_seq, area_seq, fps, peak_idx,
                   action_dir / "ear_curve.png", action_name,
                   palsy_detection=palsy_detection)

    # 保存JSON
    import json
    with open(action_dir / "indicators.json", 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    print(
        f"    [OK] {action_name}: Area L={peak_closure_data.get('left_area', 0):.0f} R={peak_closure_data.get('right_area', 0):.0f}")
    print(
        f"         Closure: L={dynamics.get('left_closure_ratio', 0) * 100:.1f}% R={dynamics.get('right_closure_ratio', 0) * 100:.1f}%")
    print(f"         Palsy: {palsy_detection.get('interpretation', 'N/A')}")

    return result