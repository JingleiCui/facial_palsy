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
)

from thresholds import THR


def find_peak_frame_blink(landmarks_seq: List, frames_seq: List, w: int, h: int) -> int:
    """找闭眼峰值帧 (EAR最小)"""
    min_ear = float('inf')
    min_idx = 0

    for i, lm in enumerate(landmarks_seq):
        if lm is None:
            continue
        l_ear = compute_ear(lm, w, h, True)
        r_ear = compute_ear(lm, w, h, False)
        avg_ear = (l_ear + r_ear) / 2
        if avg_ear < min_ear:
            min_ear = avg_ear
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
                           fps: float, scale: float = 1.0) -> Dict[str, Any]:
    """分析眨眼动态特征"""
    left = np.array(ear_seq["left"])
    right = np.array(ear_seq["right"])
    avg = np.array(ear_seq["average"])

    # 处理nan
    valid_mask = ~np.isnan(avg)
    if not np.any(valid_mask):
        return {}

    valid_left = left[valid_mask]
    valid_right = right[valid_mask]
    valid_avg = avg[valid_mask]
    valid_indices = np.where(valid_mask)[0]

    # 基本统计
    left_max = float(np.max(valid_left))
    left_min = float(np.min(valid_left))
    right_max = float(np.max(valid_right))
    right_min = float(np.min(valid_right))

    # 峰值和谷值位置
    max_idx = valid_indices[np.argmax(valid_avg)]
    min_idx = valid_indices[np.argmin(valid_avg)]

    # 闭眼程度 (EAR下降比例)
    left_closure = (left_max - left_min) / left_max if left_max > 1e-9 else 0
    right_closure = (right_max - right_min) / right_max if right_max > 1e-9 else 0

    # 时间分析
    total_frames = len(avg)
    duration_sec = total_frames / fps if fps > 0 else 0

    # 对称性分析
    ear_ratio_seq = valid_left / np.maximum(valid_right, 1e-9)
    symmetry_mean = float(np.mean(ear_ratio_seq))
    symmetry_std = float(np.std(ear_ratio_seq))

    # ========== 持续性不对称检测 ==========
    # 观察整段视频中左右眼EAR的持续差异
    # 如果一只眼睛总是比另一只闭合程度差，可能是患侧

    # 计算每帧左右眼EAR的差值
    ear_diff = valid_left - valid_right  # 正值表示左眼EAR更大（闭合更差）

    # 统计左眼EAR更大的帧数比例
    left_worse_ratio = np.mean(ear_diff > 0.02)  # 左眼闭合更差的帧比例
    right_worse_ratio = np.mean(ear_diff < -0.02)  # 右眼闭合更差的帧比例

    # 计算持续性不对称指标
    # 如果一侧持续性地闭合更差（>70%的帧），则认为可能是患侧
    persistent_asymmetry_side = 0  # 0=对称, 1=左侧持续差, 2=右侧持续差
    persistent_asymmetry_ratio = 0

    if left_worse_ratio > 0.7:
        persistent_asymmetry_side = 1
        persistent_asymmetry_ratio = left_worse_ratio
    elif right_worse_ratio > 0.7:
        persistent_asymmetry_side = 2
        persistent_asymmetry_ratio = right_worse_ratio

    return {
        "left_ear_max": left_max,
        "left_ear_min": left_min,
        "right_ear_max": right_max,
        "right_ear_min": right_min,
        "left_closure_ratio": left_closure,
        "right_closure_ratio": right_closure,
        "max_open_frame": int(max_idx),
        "min_close_frame": int(min_idx),
        "total_frames": total_frames,
        "duration_sec": duration_sec,
        "symmetry_ratio_mean": symmetry_mean,
        "symmetry_ratio_std": symmetry_std,
        "fps": fps,
        "scale": scale,
        "persistent_asymmetry_side": persistent_asymmetry_side,
        "persistent_asymmetry_ratio": persistent_asymmetry_ratio,
        "left_worse_frame_ratio": float(left_worse_ratio),
        "right_worse_frame_ratio": float(right_worse_ratio),
    }


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


def detect_palsy_side(dynamics: Dict[str, Any]) -> Dict[str, Any]:
    """
    从眨眼动作检测面瘫侧别 - 增强版

    原理:
    1. 面瘫侧眼睛闭合比例低
    2. 持续性不对称：整段视频中一只眼睛总是闭合程度差
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
    symmetry_ratio = dynamics.get("symmetry_ratio_mean", 1.0)

    # 持续性不对称指标
    persistent_side = dynamics.get("persistent_asymmetry_side", 0)
    persistent_ratio = dynamics.get("persistent_asymmetry_ratio", 0)
    left_worse_ratio = dynamics.get("left_worse_frame_ratio", 0)
    right_worse_ratio = dynamics.get("right_worse_frame_ratio", 0)

    result["evidence"] = {
        "left_closure_pct": left_closure * 100,
        "right_closure_pct": right_closure * 100,
        "symmetry_ratio_mean": symmetry_ratio,
        "persistent_asymmetry_side": persistent_side,
        "left_worse_frame_ratio": left_worse_ratio,
        "right_worse_frame_ratio": right_worse_ratio,
    }

    max_closure = max(left_closure, right_closure)

    if max_closure < 0.25:  # 闭合不足25%
        result["method"] = "insufficient"
        result["interpretation"] = f"眨眼幅度过小 (L={left_closure * 100:.1f}%, R={right_closure * 100:.1f}%)"
        result["evidence"]["status"] = "insufficient_movement"
        return result

    # ========== 方法1：持续性不对称（优先使用）==========
    # 如果整段视频中一只眼睛持续性地闭合更差，则认为是患侧
    if persistent_side != 0 and persistent_ratio > 0.7:
        result["method"] = "persistent_asymmetry"
        result["confidence"] = min(1.0, persistent_ratio)

        if persistent_side == 1:
            result["palsy_side"] = 1
            result["interpretation"] = (
                f"左眼持续闭合差 ({left_worse_ratio * 100:.0f}%帧中左眼EAR更大) → 左侧面瘫"
            )
        else:
            result["palsy_side"] = 2
            result["interpretation"] = (
                f"右眼持续闭合差 ({right_worse_ratio * 100:.0f}%帧中右眼EAR更大) → 右侧面瘫"
            )
        return result

    # ========== 方法2：闭合比例比较 ==========
    result["method"] = "closure_ratio"

    # 计算不对称比例
    asymmetry = abs(left_closure - right_closure) / max_closure
    result["confidence"] = min(1.0, asymmetry * 2.5)
    result["evidence"]["asymmetry_ratio"] = asymmetry

    if asymmetry < 0.15:
        result["palsy_side"] = 0
        result["interpretation"] = (
            f"双眼眨眼对称 (L={left_closure * 100:.1f}%, R={right_closure * 100:.1f}%, "
            f"差异{asymmetry * 100:.1f}%)"
        )
    elif left_closure < right_closure:
        # 左眼闭合程度低 -> 左侧面瘫
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
                            baseline_result: Optional[ActionResult] = None) -> Optional[ActionResult]:
    """处理自主眨眼"""
    return _process_blink(
        landmarks_seq, frames_seq, w, h, video_info, output_dir,
        action_name="VoluntaryEyeBlink",
        action_name_cn="自主眨眼",
        baseline_result=baseline_result
    )


def process_spontaneous_blink(landmarks_seq: List, frames_seq: List, w: int, h: int,
                              video_info: Dict[str, Any], output_dir: Path,
                              baseline_result: Optional[ActionResult] = None) -> Optional[ActionResult]:
    """处理自然眨眼"""
    return _process_blink(
        landmarks_seq, frames_seq, w, h, video_info, output_dir,
        action_name="SpontaneousEyeBlink",
        action_name_cn="自然眨眼",
        baseline_result=baseline_result
    )


def _process_blink(landmarks_seq: List, frames_seq: List, w: int, h: int,
                   video_info: Dict[str, Any], output_dir: Path,
                   action_name: str, action_name_cn: str,
                   baseline_result: Optional[ActionResult] = None) -> Optional[ActionResult]:
    """处理眨眼动作的通用函数"""
    if not landmarks_seq or not frames_seq:
        return None

    # 找峰值帧 (EAR最小)
    peak_idx = find_peak_frame_blink(landmarks_seq, frames_seq, w, h)
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

    # 提取EAR序列
    ear_seq = extract_ear_sequence(landmarks_seq, w, h)
    area_seq = extract_eye_area_sequence(landmarks_seq, w, h)

    # 分析眨眼动态
    dynamics = analyze_blink_dynamics(ear_seq, fps)

    # 获取面瘫侧别检测结果
    palsy_detection = detect_palsy_side(dynamics)

    # 检测联动
    synkinesis = detect_synkinesis_from_blink(baseline_result, result)
    result.synkinesis_scores = synkinesis

    # 存储动作特有指标
    result.action_specific = {
        "blink_dynamics": dynamics,
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

    # 绘制EAR曲线
    plot_ear_curve(ear_seq, area_seq, fps, peak_idx,
                   action_dir / "ear_curve.png", action_name)

    # 保存JSON
    import json
    with open(action_dir / "indicators.json", 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"    [OK] {action_name}: EAR L={result.left_ear:.4f} R={result.right_ear:.4f}")
    print(
        f"         Closure: L={dynamics.get('left_closure_ratio', 0) * 100:.1f}% R={dynamics.get('right_closure_ratio', 0) * 100:.1f}%")

    return result