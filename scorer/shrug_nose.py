#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShrugNose 动作处理模块
================================

分析皱鼻动作:
1. 鼻翼点到同侧眼角内眦点的距离 (关键帧检测 + 面瘫侧别判断)
2. 鼻翼位置变化
3. 联动运动检测

修复内容:
- 使用鼻翼到内眦距离替代错误的NLF计算
- 关键帧检测: 鼻翼-内眦距离最小的帧
- 面瘫侧别检测: 距离变化小的一侧为患侧

对应Sunnybrook: Snarl (LLA/LLS)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json

from clinical_base import (
    LM, pt2d, pts2d, dist, compute_ear, compute_eye_area,
    compute_mouth_metrics, compute_icd, extract_common_indicators,
    ActionResult, draw_polygon, compute_scale_to_baseline,
    compute_nose_midline_symmetry, compute_ala_canthus_change,
)

from thresholds import THR

ACTION_NAME = "ShrugNose"
ACTION_NAME_CN = "皱鼻"


def compute_ala_to_canthus_distance(landmarks, w: int, h: int, left: bool = True) -> float:
    """
    计算鼻翼点到同侧眼角内眦点的距离

    皱鼻动作时，鼻翼上提，此距离变小
    """
    if left:
        ala = pt2d(landmarks[LM.NOSE_ALA_L], w, h)
        canthus = pt2d(landmarks[LM.EYE_INNER_L], w, h)
    else:
        ala = pt2d(landmarks[LM.NOSE_ALA_R], w, h)
        canthus = pt2d(landmarks[LM.EYE_INNER_R], w, h)

    return dist(ala, canthus)


def find_peak_frame(landmarks_seq: List, frames_seq: List, w: int, h: int,
                    baseline_landmarks=None) -> int:
    """
    找皱鼻峰值帧

    使用双侧鼻翼-内眦距离之和最小的帧作为峰值帧
    皱鼻时鼻翼上提，距离变小
    """
    min_total_dist = float('inf')
    min_idx = 0

    for i, lm in enumerate(landmarks_seq):
        if lm is None:
            continue

        # 计算双侧鼻翼-内眦距离
        left_dist = compute_ala_to_canthus_distance(lm, w, h, left=True)
        right_dist = compute_ala_to_canthus_distance(lm, w, h, left=False)
        total_dist = left_dist + right_dist

        if total_dist < min_total_dist:
            min_total_dist = total_dist
            min_idx = i

    return min_idx


def extract_shrug_nose_sequences(
        landmarks_seq: List,
        w: int, h: int
) -> Dict[str, List[float]]:
    """
    提取皱鼻关键指标的时序序列

    Returns:
        包含鼻翼-内眦距离的时序数据
    """
    left_dist_seq = []
    right_dist_seq = []
    total_dist_seq = []

    for lm in landmarks_seq:
        if lm is None:
            left_dist_seq.append(np.nan)
            right_dist_seq.append(np.nan)
            total_dist_seq.append(np.nan)
        else:
            left_dist = compute_ala_to_canthus_distance(lm, w, h, left=True)
            right_dist = compute_ala_to_canthus_distance(lm, w, h, left=False)
            left_dist_seq.append(left_dist)
            right_dist_seq.append(right_dist)
            total_dist_seq.append(left_dist + right_dist)

    return {
        "Left Ala-Canthus": left_dist_seq,
        "Right Ala-Canthus": right_dist_seq,
        "Total Distance": total_dist_seq,
    }


def plot_shrug_nose_peak_selection(
        sequences: Dict[str, List[float]],
        fps: float,
        peak_idx: int,
        output_path: Path,
        baseline_values: Dict[str, float] = None
) -> None:
    """
    绘制皱鼻关键帧选择的可解释性曲线

    皱鼻选择标准: 双侧鼻翼-内眦距离之和最小的帧
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_frames = len(sequences["Left Ala-Canthus"])
    frames = np.arange(n_frames)
    time_sec = frames / fps if fps > 0 else frames
    x_label = 'Time (seconds)' if fps > 0 else 'Frame'
    peak_time = peak_idx / fps if fps > 0 else peak_idx

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # 上图: 鼻翼-内眦距离曲线
    ax1 = axes[0]
    ax1.plot(time_sec, sequences["Left Ala-Canthus"], 'b-', label='Left Ala-Canthus', linewidth=2)
    ax1.plot(time_sec, sequences["Right Ala-Canthus"], 'r-', label='Right Ala-Canthus', linewidth=2)

    # 绘制基线
    if baseline_values:
        if "left" in baseline_values:
            ax1.axhline(y=baseline_values["left"], color='blue', linestyle=':', alpha=0.5, label='Left Baseline')
        if "right" in baseline_values:
            ax1.axhline(y=baseline_values["right"], color='red', linestyle=':', alpha=0.5, label='Right Baseline')

    ax1.axvline(x=peak_time, color='black', linestyle='--', linewidth=2, alpha=0.7)

    # 标注峰值帧处的值
    left_at_peak = sequences["Left Ala-Canthus"][peak_idx] if peak_idx < len(sequences["Left Ala-Canthus"]) else 0
    right_at_peak = sequences["Right Ala-Canthus"][peak_idx] if peak_idx < len(sequences["Right Ala-Canthus"]) else 0

    if not np.isnan(left_at_peak):
        ax1.scatter([peak_time], [left_at_peak], color='blue', s=80, zorder=5, edgecolors='white', linewidths=1.5)
        ax1.annotate(f'L:{left_at_peak:.1f}', xy=(peak_time, left_at_peak),
                     xytext=(-50, 5), textcoords='offset points', fontsize=9, color='blue')
    if not np.isnan(right_at_peak):
        ax1.scatter([peak_time], [right_at_peak], color='red', s=80, zorder=5, edgecolors='white', linewidths=1.5)
        ax1.annotate(f'R:{right_at_peak:.1f}', xy=(peak_time, right_at_peak),
                     xytext=(5, 5), textcoords='offset points', fontsize=9, color='red')

    ax1.set_xlabel(x_label, fontsize=11)
    ax1.set_ylabel('Distance (pixels)', fontsize=11)
    ax1.set_title('Ala-Canthus Distance Over Time', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # 下图: 总距离曲线 (关键帧选择依据)
    ax2 = axes[1]
    total_arr = np.array(sequences["Total Distance"])
    ax2.plot(time_sec, total_arr, 'g-', label='Total Distance (selection criterion)', linewidth=2)

    ax2.axvline(x=peak_time, color='black', linestyle='--', linewidth=2, alpha=0.7)
    total_at_peak = total_arr[peak_idx] if peak_idx < len(total_arr) else 0
    ax2.scatter([peak_time], [total_at_peak], color='red', s=150, zorder=5,
                edgecolors='black', linewidths=2, marker='*', label=f'Peak Frame {peak_idx}')
    ax2.annotate(f'Min: {total_at_peak:.1f}px', xy=(peak_time, total_at_peak),
                 xytext=(10, 10), textcoords='offset points', fontsize=10, fontweight='bold')

    ax2.set_xlabel(x_label, fontsize=11)
    ax2.set_ylabel('Total Distance (pixels)', fontsize=11)
    ax2.set_title('ShrugNose Peak Selection: Minimum Total Ala-Canthus Distance', fontsize=13, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()


def compute_shrug_nose_metrics(landmarks, w: int, h: int,
                               baseline_landmarks=None) -> Dict[str, Any]:
    """计算皱鼻特有指标  - 使用统一 scale"""
    # 鼻翼到内眦距离
    left_ac_dist = compute_ala_to_canthus_distance(landmarks, w, h, left=True)
    right_ac_dist = compute_ala_to_canthus_distance(landmarks, w, h, left=False)

    # 鼻翼点位置
    left_ala = pt2d(landmarks[LM.NOSE_ALA_L], w, h)
    right_ala = pt2d(landmarks[LM.NOSE_ALA_R], w, h)

    # 内眦点位置
    left_canthus = pt2d(landmarks[LM.EYE_INNER_L], w, h)
    right_canthus = pt2d(landmarks[LM.EYE_INNER_R], w, h)

    # 鼻翼间距
    ala_width = dist(left_ala, right_ala)

    metrics = {
        "left_ala_canthus_dist": left_ac_dist,
        "right_ala_canthus_dist": right_ac_dist,
        "ala_canthus_ratio": left_ac_dist / right_ac_dist if right_ac_dist > 1e-9 else 1.0,
        "ala_width": ala_width,
        "points": {
            "left_ala": left_ala,
            "right_ala": right_ala,
            "left_canthus": left_canthus,
            "right_canthus": right_canthus,
        }
    }

    # 如果有基线，计算变化
    if baseline_landmarks is not None:
        # ========== 计算统一 scale ==========
        scale = compute_scale_to_baseline(landmarks, baseline_landmarks, w, h)
        metrics["scale"] = scale
        # ====================================

        baseline_left = compute_ala_to_canthus_distance(baseline_landmarks, w, h, left=True)
        baseline_right = compute_ala_to_canthus_distance(baseline_landmarks, w, h, left=False)
        baseline_left_ala = pt2d(baseline_landmarks[LM.NOSE_ALA_L], w, h)
        baseline_right_ala = pt2d(baseline_landmarks[LM.NOSE_ALA_R], w, h)

        metrics["baseline"] = {
            "left_ala_canthus_dist": baseline_left,
            "right_ala_canthus_dist": baseline_right,
            "ala_width": dist(baseline_left_ala, baseline_right_ala),
        }

        # ========== 缩放到 baseline 尺度后计算变化 ==========
        left_scaled = left_ac_dist * scale
        right_scaled = right_ac_dist * scale

        # 距离变化 (皱鼻时应为负值)
        metrics["left_change"] = left_scaled - baseline_left
        metrics["right_change"] = right_scaled - baseline_right
        # ===================================================

        # 变化百分比
        if baseline_left > 1e-9:
            metrics["left_change_percent"] = metrics["left_change"] / baseline_left * 100
        else:
            metrics["left_change_percent"] = 0

        if baseline_right > 1e-9:
            metrics["right_change_percent"] = metrics["right_change"] / baseline_right * 100
        else:
            metrics["right_change_percent"] = 0

        # 鼻翼垂直移动量 (缩放后)
        metrics["left_ala_vertical_move"] = (left_ala[1] - baseline_left_ala[1]) * scale
        metrics["right_ala_vertical_move"] = (right_ala[1] - baseline_right_ala[1]) * scale

        # ========== 计算鼻翼-内眦距离的变化量 ==========
        ala_change_data = compute_ala_canthus_change(landmarks, w, h, baseline_landmarks)
        metrics["ala_canthus_change"] = ala_change_data

    # ========== 鼻部面中线对称性（用于面瘫侧别判断）==========
    nose_symmetry = compute_nose_midline_symmetry(landmarks, w, h)
    metrics["nose_symmetry"] = nose_symmetry

    # ========== 鼻翼到眼部水平线的垂直距离（用于面瘫侧别判断）==========
    # 眼部水平线：双眼内眦连线
    # 计算方法：鼻翼点到内眦连线的垂直距离
    eye_line_y = (left_canthus[1] + right_canthus[1]) / 2  # 眼部水平线的y坐标（近似）

    # 左右鼻翼到眼部水平线的垂直距离
    # 注意：y坐标向下为正，所以鼻翼y - 眼部y = 正值表示鼻翼在眼睛下方
    left_ala_to_eye_line = left_ala[1] - eye_line_y
    right_ala_to_eye_line = right_ala[1] - eye_line_y

    metrics["ala_to_eye_line"] = {
        "eye_line_y": float(eye_line_y),
        "left_distance": float(left_ala_to_eye_line),
        "right_distance": float(right_ala_to_eye_line),
        "distance_diff": float(left_ala_to_eye_line - right_ala_to_eye_line),
    }

    # 计算鼻翼连线相对于眼部水平线的倾斜角度
    # 如果左鼻翼更高（y更小），角度为负；右鼻翼更高，角度为正
    ala_line_angle = np.degrees(np.arctan2(
        right_ala[1] - left_ala[1],  # y差值
        right_ala[0] - left_ala[0]  # x差值
    ))
    eye_line_angle = np.degrees(np.arctan2(
        right_canthus[1] - left_canthus[1],
        right_canthus[0] - left_canthus[0]
    ))
    # 鼻翼连线相对于眼部水平线的偏转角度
    tilt_angle = ala_line_angle - eye_line_angle

    metrics["ala_to_eye_line"]["ala_line_angle"] = float(ala_line_angle)
    metrics["ala_to_eye_line"]["eye_line_angle"] = float(eye_line_angle)
    metrics["ala_to_eye_line"]["tilt_angle"] = float(tilt_angle)

    return metrics


def detect_palsy_side(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    从皱鼻动作检测面瘫侧别 - 基于鼻翼-内眦距离变化

    改进:
    - 主要方法: 比较左右鼻翼-内眦距离的缩短量
    - 皱鼻时距离应减小，健侧减小更多（肌肉拉力强）
    - 面瘫侧减小量小（或不减小）
    """
    result = {
        "palsy_side": 0,
        "confidence": 0.0,
        "interpretation": "",
        "method": "",
        "evidence": {}
    }

    # 方法1（优先）: 距离变化量（如果有基线）
    if "ala_canthus_change" in metrics:
        change_data = metrics["ala_canthus_change"]

        left_reduction = change_data.get("left_reduction", 0)
        right_reduction = change_data.get("right_reduction", 0)

        result["method"] = "ala_canthus_change"
        result["evidence"]["left_distance"] = change_data.get("left_distance", 0)
        result["evidence"]["right_distance"] = change_data.get("right_distance", 0)
        result["evidence"]["baseline_left_distance"] = change_data.get("baseline_left_distance", 0)
        result["evidence"]["baseline_right_distance"] = change_data.get("baseline_right_distance", 0)
        result["evidence"]["left_reduction"] = left_reduction
        result["evidence"]["right_reduction"] = right_reduction

        max_reduction = max(abs(left_reduction), abs(right_reduction))

        if max_reduction < 3:  # 运动幅度过小（像素）
            result["evidence"]["status"] = "insufficient_movement"
            # 继续使用静态方法
        else:
            reduction_diff = abs(left_reduction - right_reduction)
            asymmetry = reduction_diff / max_reduction
            result["evidence"]["reduction_asymmetry"] = asymmetry
            result["evidence"]["left_reduction"] = left_reduction
            result["evidence"]["right_reduction"] = right_reduction
            result["evidence"]["max_reduction"] = max_reduction
            result["confidence"] = min(1.0, asymmetry * 2)

            if asymmetry < 0.08:
                result["palsy_side"] = 0
                result["interpretation"] = (
                    f"双侧鼻翼收缩对称 (L缩短{left_reduction:.1f}px, R缩短{right_reduction:.1f}px), "
                    f"不对称{asymmetry:.4%})"
                )
            elif left_reduction < right_reduction:
                # 左侧缩短量小 = 左侧收缩弱 = 左侧面瘫
                result["palsy_side"] = 1
                result["interpretation"] = (
                    f"左侧鼻翼收缩弱 (L缩短{left_reduction:.1f}px < R缩短{right_reduction:.1f}px), "
                    f"不对称{asymmetry:.4%}) → 左侧面瘫"
                )
            else:
                result["palsy_side"] = 2
                result["interpretation"] = (
                    f"右侧鼻翼收缩弱 (R缩短{right_reduction:.1f}px < L缩短{left_reduction:.1f}px), "
                    f"不对称{asymmetry:.4%}) → 右侧面瘫"
                )
            return result

    # 方法2: 静态距离比较（当前帧）
    # 皱鼻时距离大的一侧是面瘫侧（收缩弱）
    left_dist = metrics.get("left_ala_canthus_dist", 0)
    right_dist = metrics.get("right_ala_canthus_dist", 0)

    result["method"] = "ala_canthus_static"
    result["evidence"]["left_ala_canthus_dist"] = left_dist
    result["evidence"]["right_ala_canthus_dist"] = right_dist

    max_dist = max(left_dist, right_dist)
    if max_dist > 1e-6:
        dist_diff = abs(left_dist - right_dist)
        asymmetry = dist_diff / max_dist
        result["evidence"]["distance_asymmetry"] = asymmetry
        result["confidence"] = min(1.0, asymmetry * 3)

        if asymmetry < 0.08:
            result["palsy_side"] = 0
            result["interpretation"] = f"双侧鼻翼-内眦距离对称 (L={left_dist:.1f}px, R={right_dist:.1f}px)"
        elif left_dist > right_dist:
            # 左侧距离大 = 左侧收缩弱 = 左侧面瘫
            result["palsy_side"] = 1
            result["interpretation"] = f"左侧距离大 (L={left_dist:.1f}px > R={right_dist:.1f}px) → 左侧面瘫"
        else:
            result["palsy_side"] = 2
            result["interpretation"] = f"右侧距离大 (R={right_dist:.1f}px > L={left_dist:.1f}px) → 右侧面瘫"

    return result


def compute_severity_score(metrics: Dict[str, Any]) -> Tuple[int, str]:
    """
    计算动作严重度分数(医生标注标准)

    计算依据: 鼻翼 - 内眦距离变化的对称性

    修改: 调整阈值
    """
    ala_change = metrics.get("ala_canthus_change", {})

    left_reduction = ala_change.get("left_reduction", 0)
    right_reduction = ala_change.get("right_reduction", 0)

    abs_left = abs(left_reduction)
    abs_right = abs(right_reduction)
    max_reduction = max(abs_left, abs_right)
    min_reduction = min(abs_left, abs_right)

    # 检查是否有足够的运动
    if max_reduction < 3.0:
        return 1, f"运动幅度过小 (L={left_reduction:.1f}px, R={right_reduction:.1f}px)"

    # 计算对称性比值
    symmetry_ratio = min_reduction / max_reduction if max_reduction > 0 else 1.0

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

    基于鼻翼-内眦距离变化的程度和对称性

    评分标准:
    - 5=完整: 双侧对称且运动充分
    - 4=几乎完整: 轻度不对称或运动略有不足
    - 3=启动但不对称: 明显不对称但有运动
    - 2=轻微启动: 运动幅度很小
    - 1=无法启动: 几乎没有运动
    """
    if baseline_landmarks is not None and "left_change" in metrics:
        left_change = metrics["left_change"]
        right_change = metrics["right_change"]

        # 皱鼻时距离应该变小，所以收缩 = -change
        left_contraction = -left_change
        right_contraction = -right_change

        # 检查是否有明显运动
        max_contraction = max(left_contraction, right_contraction)
        min_contraction = min(left_contraction, right_contraction)

        if max_contraction < 2:  # 几乎没有收缩
            return 1, "无法启动运动 (鼻翼-内眦距离变化过小)"

        # 检查是否一侧反向运动
        if min_contraction < -1:  # 一侧反而变远
            return 2, "轻微启动 (单侧异常运动)"

        # 计算对称性
        if max_contraction > 1e-9:
            symmetry_ratio = min_contraction / max_contraction
        else:
            symmetry_ratio = 1.0

        if symmetry_ratio >= 0.85:
            if max_contraction > 8:
                return 5, "运动完整 (对称且幅度大)"
            elif max_contraction > 5:
                return 4, "几乎完整"
            else:
                return 3, "启动但幅度不足"
        elif symmetry_ratio >= 0.60:
            return 3, "启动但不对称"
        elif symmetry_ratio >= 0.30:
            return 2, "轻微启动"
        else:
            return 1, "无法启动"
    else:
        # 没有基线，使用静态比值
        ratio = metrics.get("ala_canthus_ratio", 1.0)
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
    """检测皱鼻时的联动运动"""
    synkinesis = {
        "eye_synkinesis": 0,
        "mouth_synkinesis": 0,
    }

    if baseline_result is None:
        return synkinesis

    # 检测眼部联动
    l_ear = compute_ear(current_landmarks, w, h, True)
    r_ear = compute_ear(current_landmarks, w, h, False)

    baseline_l_ear = baseline_result.left_ear
    baseline_r_ear = baseline_result.right_ear

    if baseline_l_ear > 1e-9 and baseline_r_ear > 1e-9:
        l_change = abs(l_ear - baseline_l_ear) / baseline_l_ear
        r_change = abs(r_ear - baseline_r_ear) / baseline_r_ear
        avg_change = (l_change + r_change) / 2

        if avg_change > 0.18:
            synkinesis["eye_synkinesis"] = 3
        elif avg_change > 0.10:
            synkinesis["eye_synkinesis"] = 2
        elif avg_change > 0.05:
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


def visualize_shrug_nose(frame: np.ndarray, landmarks, w: int, h: int,
                         result: ActionResult,
                         metrics: Dict[str, Any],
                         palsy_detection: Dict[str, Any]) -> np.ndarray:
    """可视化皱鼻指标"""
    img = frame.copy()

    # 获取点坐标
    points = metrics.get("points", {})
    left_ala = points.get("left_ala", pt2d(landmarks[LM.NOSE_ALA_L], w, h))
    right_ala = points.get("right_ala", pt2d(landmarks[LM.NOSE_ALA_R], w, h))
    left_canthus = points.get("left_canthus", pt2d(landmarks[LM.EYE_INNER_L], w, h))
    right_canthus = points.get("right_canthus", pt2d(landmarks[LM.EYE_INNER_R], w, h))

    # 绘制鼻翼-内眦连线 (关键测量线)
    cv2.line(img, (int(left_ala[0]), int(left_ala[1])),
             (int(left_canthus[0]), int(left_canthus[1])), (255, 0, 0), 3)
    cv2.line(img, (int(right_ala[0]), int(right_ala[1])),
             (int(right_canthus[0]), int(right_canthus[1])), (0, 165, 255), 3)

    # 绘制鼻翼点
    cv2.circle(img, (int(left_ala[0]), int(left_ala[1])), 6, (255, 0, 0), -1)
    cv2.circle(img, (int(right_ala[0]), int(right_ala[1])), 6, (0, 165, 255), -1)

    # 绘制内眦点
    cv2.circle(img, (int(left_canthus[0]), int(left_canthus[1])), 5, (255, 0, 0), 2)
    cv2.circle(img, (int(right_canthus[0]), int(right_canthus[1])), 5, (0, 165, 255), 2)

    # 绘制鼻翼连线
    cv2.line(img, (int(left_ala[0]), int(left_ala[1])),
             (int(right_ala[0]), int(right_ala[1])), (0, 255, 255), 2)

    # ========== 绘制眼部水平线 ==========
    cv2.line(img, (int(left_canthus[0]), int(left_canthus[1])),
             (int(right_canthus[0]), int(right_canthus[1])), (0, 255, 0), 2)

    # ========== 绘制鼻翼到眼部水平线的垂直距离 ==========
    if "ala_to_eye_line" in metrics:
        ala_eye = metrics["ala_to_eye_line"]
        eye_line_y = int(ala_eye["eye_line_y"])

        # 左鼻翼到眼部水平线的垂直线
        cv2.line(img, (int(left_ala[0]), int(left_ala[1])),
                 (int(left_ala[0]), eye_line_y), (255, 0, 0), 2, cv2.LINE_AA)
        # 右鼻翼到眼部水平线的垂直线
        cv2.line(img, (int(right_ala[0]), int(right_ala[1])),
                 (int(right_ala[0]), eye_line_y), (0, 165, 255), 2, cv2.LINE_AA)

        # 标注距离值
        left_dist = ala_eye["left_distance"]
        right_dist = ala_eye["right_distance"]
        cv2.putText(img, f"{left_dist:.0f}",
                    (int(left_ala[0]) - 30, int((left_ala[1] + eye_line_y) / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        cv2.putText(img, f"{right_dist:.0f}",
                    (int(right_ala[0]) + 5, int((right_ala[1] + eye_line_y) / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

    # 信息面板
    panel_h = 320
    cv2.rectangle(img, (5, 5), (420, panel_h), (0, 0, 0), -1)
    cv2.rectangle(img, (5, 5), (420, panel_h), (255, 255, 255), 1)

    y = 28
    cv2.putText(img, f"{ACTION_NAME} (ShrugNose)", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y += 28

    # ========== 显示鼻翼到眼部水平线距离 ==========
    if "ala_to_eye_line" in metrics:
        ala_eye = metrics["ala_to_eye_line"]
        cv2.putText(img, "=== Ala to Eye-Line Distance ===", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        y += 20

        left_dist = ala_eye["left_distance"]
        right_dist = ala_eye["right_distance"]
        tilt = ala_eye.get("tilt_angle", 0)

        # 距离长的用红色标记（患侧）
        left_color = (0, 0, 255) if left_dist > right_dist else (0, 255, 0)
        right_color = (0, 0, 255) if right_dist > left_dist else (0, 255, 0)

        cv2.putText(img, f"Left to Eye-Line: {left_dist:.1f}px", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, left_color, 1)
        y += 18
        cv2.putText(img, f"Right to Eye-Line: {right_dist:.1f}px", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, right_color, 1)
        y += 18
        cv2.putText(img, f"Tilt Angle: {tilt:+.1f} deg", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 22

    cv2.putText(img, "=== Ala-Canthus Distance ===", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    y += 20

    cv2.putText(img, f"Left: {metrics['left_ala_canthus_dist']:.1f}px", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
    y += 18

    cv2.putText(img, f"Right: {metrics['right_ala_canthus_dist']:.1f}px", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
    y += 18

    ratio = metrics['ala_canthus_ratio']
    ratio_color = (0, 255, 0) if 0.9 <= ratio <= 1.1 else (0, 0, 255)
    cv2.putText(img, f"Ratio (L/R): {ratio:.3f}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, ratio_color, 1)
    y += 22

    if "left_change" in metrics:
        cv2.putText(img, "=== Changes from Baseline ===", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        y += 20

        cv2.putText(img,
                    f"Left: {metrics['left_change']:+.1f}px ({metrics.get('left_change_percent', 0):+.1f}%)",
                    (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 18

        cv2.putText(img,
                    f"Right: {metrics['right_change']:+.1f}px ({metrics.get('right_change_percent', 0):+.1f}%)",
                    (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 22

    cv2.putText(img, f"Ala Width: {metrics['ala_width']:.1f}px", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    y += 25

    # 面瘫侧别检测结果
    palsy_side = palsy_detection.get("palsy_side", 0)
    palsy_text = {0: "No", 1: "Left", 2: "Right"}.get(palsy_side, "Unkown")
    palsy_color = (0, 255, 0) if palsy_side == 0 else (0, 0, 255)
    cv2.putText(img, f"Palsy Side: {palsy_text}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, palsy_color, 1)
    y += 25

    # Voluntary Score
    cv2.putText(img, f"Voluntary Score: {result.voluntary_movement_score}/5", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # 图例
    legend_y = panel_h + 15
    cv2.line(img, (15, legend_y), (45, legend_y), (255, 0, 0), 3)
    cv2.putText(img, "Left Ala-Canthus", (50, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.line(img, (200, legend_y), (230, legend_y), (0, 165, 255), 3)
    cv2.putText(img, "Right Ala-Canthus", (235, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    return img


def process(landmarks_seq: List, frames_seq: List, w: int, h: int,
            video_info: Dict[str, Any], output_dir: Path,
            baseline_result: Optional[ActionResult] = None,
            baseline_landmarks=None) -> Optional[ActionResult]:
    """处理ShrugNose动作"""
    if not landmarks_seq or not frames_seq:
        return None

    # 找峰值帧 (鼻翼-内眦距离最小)
    peak_idx = find_peak_frame(landmarks_seq, frames_seq, w, h, baseline_landmarks)
    peak_landmarks = landmarks_seq[peak_idx]
    peak_frame = frames_seq[peak_idx]

    # 提取时序序列用于可视化
    sequences = extract_shrug_nose_sequences(landmarks_seq, w, h)

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

    # 计算皱鼻特有指标
    metrics = compute_shrug_nose_metrics(peak_landmarks, w, h, baseline_landmarks)

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
        "ala_canthus_metrics": {
            "left_distance": metrics["left_ala_canthus_dist"],
            "right_distance": metrics["right_ala_canthus_dist"],
            "ratio": metrics["ala_canthus_ratio"],
            "ala_width": metrics["ala_width"],
        },
        "palsy_detection": palsy_detection,
        "voluntary_interpretation": interpretation,
        "synkinesis": synkinesis,
        "severity_score": severity_score,
        "severity_desc": severity_desc,
    }

    if "baseline" in metrics:
        result.action_specific["baseline"] = metrics["baseline"]
        result.action_specific["changes"] = {
            "left_change": metrics.get("left_change", 0),
            "right_change": metrics.get("right_change", 0),
            "left_change_percent": metrics.get("left_change_percent", 0),
            "right_change_percent": metrics.get("right_change_percent", 0),
        }

    # 创建输出目录
    action_dir = output_dir / ACTION_NAME
    action_dir.mkdir(parents=True, exist_ok=True)

    # 保存原始帧
    cv2.imwrite(str(action_dir / "peak_raw.jpg"), peak_frame)

    # 保存可视化
    vis = visualize_shrug_nose(peak_frame, peak_landmarks, w, h, result, metrics, palsy_detection)
    cv2.imwrite(str(action_dir / "peak_indicators.jpg"), vis)

    # 准备基线值
    baseline_values = None
    if baseline_landmarks is not None:
        baseline_left = compute_ala_to_canthus_distance(baseline_landmarks, w, h, left=True)
        baseline_right = compute_ala_to_canthus_distance(baseline_landmarks, w, h, left=False)
        baseline_values = {"left": baseline_left, "right": baseline_right}

    # 绘制关键帧选择曲线
    plot_shrug_nose_peak_selection(
        sequences,
        video_info.get("fps", 30.0),
        peak_idx,
        action_dir / "peak_selection_curve.png",
        baseline_values
    )

    # 保存JSON
    with open(action_dir / "indicators.json", 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    print(
        f"    [OK] {ACTION_NAME}: Ala-Canthus L={metrics['left_ala_canthus_dist']:.1f} R={metrics['right_ala_canthus_dist']:.1f}")
    if "left_change" in metrics:
        print(f"         Change L={metrics['left_change']:+.1f}px R={metrics['right_change']:+.1f}px")
    print(f"         Palsy Side: {palsy_detection.get('interpretation', 'N/A')}")
    print(f"         Voluntary Score: {result.voluntary_movement_score}/5 ({interpretation})")

    return result