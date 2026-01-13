#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
微笑动作处理模块
==========================

分析:
1. 嘴角位移和对称性
2. 口角角度变化
3. 运动幅度对比
4. 面瘫侧别检测
5. 联动运动检测 (眼部联动)

# 1. 关键帧改为嘴角到眼部水平线距离最小的帧
# 2. 面瘫侧别改为比较左右嘴角上提幅度（距离减小量）
对应Sunnybrook: Open mouth smile (ZYG/RIS)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import math
import json
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from clinical_base import (
    LM, pt2d, pts2d, dist, compute_ear, compute_eye_area,
    compute_mouth_metrics, compute_oral_angle, compute_icd,
    extract_common_indicators, compute_scale_to_baseline,
    ActionResult, OralAngleMeasure, draw_polygon,
    add_valid_region_shading, get_palsy_side_text,
    draw_palsy_side_label, compute_lip_midline_symmetry,
    compute_mouth_corner_to_eye_line_distance,
    compute_smile_excursion_by_eye_line,
)

from thresholds import THR

from sunnybrook_scorer import (
    VoluntaryMovementItem, compute_voluntary_score_from_ratio
)

ACTION_NAME = "Smile"
ACTION_NAME_CN = "微笑"


def find_peak_frame(landmarks_seq: List, frames_seq: List, w: int, h: int,
                    baseline_landmarks=None) -> Tuple[int, Dict[str, Any]]:
    """
    找微笑峰值帧 - 使用嘴角到眼部水平线距离

    改进:
    - 微笑时嘴角上提，到眼部水平线的距离减小
    - 取距离之和最小的帧作为峰值帧
    - 返回peak_debug用于可视化曲线

    Returns:
        (peak_idx, peak_debug): 峰值帧索引和调试信息
    """
    n_frames = len(landmarks_seq)
    if n_frames == 0:
        return 0, {"error": "empty_sequence"}

    # 收集时序数据
    left_dist_seq = []
    right_dist_seq = []
    total_dist_seq = []

    min_total_dist = float('inf')
    min_idx = 0

    for i, lm in enumerate(landmarks_seq):
        if lm is None:
            left_dist_seq.append(np.nan)
            right_dist_seq.append(np.nan)
            total_dist_seq.append(np.nan)
            continue

        # 计算左右嘴角到眼部水平线的距离
        left_result = compute_mouth_corner_to_eye_line_distance(lm, w, h, left=True)
        right_result = compute_mouth_corner_to_eye_line_distance(lm, w, h, left=False)

        left_dist = left_result["distance"]
        right_dist = right_result["distance"]
        total_dist = left_dist + right_dist

        left_dist_seq.append(left_dist)
        right_dist_seq.append(right_dist)
        total_dist_seq.append(total_dist)

        if total_dist < min_total_dist:
            min_total_dist = total_dist
            min_idx = i

    # 构建peak_debug字典
    peak_debug = {
        "left_dist": left_dist_seq,
        "right_dist": right_dist_seq,
        "total_dist": total_dist_seq,
        "peak_idx": min_idx,
        "peak_value": float(min_total_dist) if np.isfinite(min_total_dist) else None,
        "selection_criterion": "min_total_eye_line_distance",
    }

    return min_idx, peak_debug


def extract_smile_sequences(landmarks_seq: List, w: int, h: int) -> Dict[str, List[float]]:
    """
    提取微笑关键指标的时序序列

    改进: 添加嘴角到眼部水平线距离的序列
    """
    mouth_width_seq = []
    aoe_seq = []
    bof_seq = []
    left_eye_dist_seq = []
    right_eye_dist_seq = []
    total_eye_dist_seq = []

    for lm in landmarks_seq:
        if lm is None:
            mouth_width_seq.append(np.nan)
            aoe_seq.append(np.nan)
            bof_seq.append(np.nan)
            left_eye_dist_seq.append(np.nan)
            right_eye_dist_seq.append(np.nan)
            total_eye_dist_seq.append(np.nan)
        else:
            # 嘴宽
            l_corner = pt2d(lm[LM.MOUTH_L], w, h)
            r_corner = pt2d(lm[LM.MOUTH_R], w, h)
            width = dist(l_corner, r_corner)
            mouth_width_seq.append(width)

            # 口角角度
            oral = compute_oral_angle(lm, w, h)
            aoe_seq.append(oral.AOE_angle if oral else np.nan)
            bof_seq.append(oral.BOF_angle if oral else np.nan)

            # 嘴角到眼部水平线距离
            left_result = compute_mouth_corner_to_eye_line_distance(lm, w, h, left=True)
            right_result = compute_mouth_corner_to_eye_line_distance(lm, w, h, left=False)
            left_eye_dist_seq.append(left_result["distance"])
            right_eye_dist_seq.append(right_result["distance"])
            total_eye_dist_seq.append(left_result["distance"] + right_result["distance"])

    return {
        "Mouth Width": mouth_width_seq,
        "AOE (Right)": aoe_seq,
        "BOF (Left)": bof_seq,
        "Left Eye-Line Dist": left_eye_dist_seq,
        "Right Eye-Line Dist": right_eye_dist_seq,
        "Total Eye-Line Dist": total_eye_dist_seq,
    }


def plot_smile_peak_selection(
        peak_debug: Dict[str, Any],  # 改为接收peak_debug
        fps: float,
        output_path: Path,
        palsy_detection: Dict[str, Any] = None
) -> None:
    """
    绘制微笑关键帧选择的可解释性曲线。
    选择标准：嘴角到眼部水平线距离最小的帧。
    """

    left_dist = peak_debug.get("left_dist", [])
    right_dist = peak_debug.get("right_dist", [])
    total_dist = peak_debug.get("total_dist", [])
    peak_idx = peak_debug.get("peak_idx", 0)

    if not total_dist:
        return

    n_frames = len(total_dist)
    frames = np.arange(n_frames)
    time_sec = frames / fps if fps > 0 else frames
    x_label = 'Time (seconds)' if fps > 0 else 'Frame'
    peak_time = peak_idx / fps if fps > 0 else peak_idx

    plt.figure(figsize=(12, 6))

    # 绘制左右距离和总距离
    plt.plot(time_sec, left_dist, 'b-', label='Left Eye-Line Distance', linewidth=2, alpha=0.6)
    plt.plot(time_sec, right_dist, 'r-', label='Right Eye-Line Distance', linewidth=2, alpha=0.6)
    plt.plot(time_sec, total_dist, 'g-', label='Total Distance (Selection)', linewidth=2.5)

    # 标记峰值
    plt.axvline(x=peak_time, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Peak Frame {peak_idx}')
    if 0 <= peak_idx < n_frames and np.isfinite(total_dist[peak_idx]):
        peak_value = total_dist[peak_idx]
        plt.scatter([peak_time], [peak_value], color='red', s=150, zorder=5,
                    edgecolors='black', linewidths=1.5, marker='*',
                    label=f'Selected Peak (Dist: {peak_value:.1f})')

    title = "Smile Peak Selection: Min Eye-Line Distance"
    if palsy_detection:
        palsy_text = get_palsy_side_text(palsy_detection.get("palsy_side", 0))
        title += f' | Detected: {palsy_text}'

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(x_label, fontsize=11)
    plt.ylabel('Distance to Eye-Line (px, lower is better)', fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()


def compute_smile_metrics(landmarks, w: int, h: int,
                          baseline_landmarks=None) -> Dict[str, Any]:
    """计算微笑特有指标 - 使用统一 scale"""
    mouth = compute_mouth_metrics(landmarks, w, h)
    oral = compute_oral_angle(landmarks, w, h)

    # 嘴角位置
    left_corner = mouth["left_corner"]
    right_corner = mouth["right_corner"]

    # 嘴角高度 (相对于嘴中心)
    mouth_center_y = (left_corner[1] + right_corner[1]) / 2
    left_height_from_center = mouth_center_y - left_corner[1]  # 正值表示左嘴角较高
    right_height_from_center = mouth_center_y - right_corner[1]

    metrics = {
        "mouth_width": mouth["width"],
        "mouth_height": mouth["height"],
        "left_corner": left_corner,
        "right_corner": right_corner,
        "left_corner_height": left_height_from_center,
        "right_corner_height": right_height_from_center,
        "corner_height_diff": left_height_from_center - right_height_from_center,
        "oral_angle": {
            "AOE": oral.AOE_angle,
            "BOF": oral.BOF_angle,
            "diff": oral.angle_diff,
            "asymmetry": oral.angle_asymmetry,
        }
    }

    # 如果有基线，计算运动幅度（统一尺度）
    if baseline_landmarks is not None:
        # ========== 计算统一 scale（优化：传入预计算的 ICD）==========
        icd_current = compute_icd(landmarks, w, h)
        # 假设 baseline 的 icd 已在 NeutralFace 时计算并存储
        # 如果有传入 icd_base 参数，使用它；否则重新计算
        scale = compute_scale_to_baseline(landmarks, baseline_landmarks, w, h,
                                          icd_current=icd_current)
        metrics["scale"] = scale
        metrics["icd_current"] = icd_current

        baseline_mouth = compute_mouth_metrics(baseline_landmarks, w, h)
        baseline_oral = compute_oral_angle(baseline_landmarks, w, h)
        baseline_left = baseline_mouth["left_corner"]
        baseline_right = baseline_mouth["right_corner"]

        # ========== 缩放当前帧距离到 baseline 尺度 ==========
        # 嘴角位移（先计算原始位移，再缩放）
        left_excursion_raw = dist(left_corner, baseline_left)
        right_excursion_raw = dist(right_corner, baseline_right)

        # 缩放到 baseline 尺度
        left_excursion = left_excursion_raw * scale
        right_excursion = right_excursion_raw * scale

        # 嘴宽变化
        width_change = mouth["width"] * scale - baseline_mouth["width"]

        metrics["excursion"] = {
            "left_total": left_excursion,
            "right_total": right_excursion,
            "excursion_ratio": left_excursion / right_excursion if right_excursion > 1e-9 else 1.0,
            "baseline_width": baseline_mouth["width"],
            "width_change": width_change,
            # 保留原始值供调试
            "left_raw": left_excursion_raw,
            "right_raw": right_excursion_raw,
        }

        # 口角角度变化（角度不需要缩放）
        metrics["oral_angle_change"] = {
            "AOE_change": oral.AOE_angle - baseline_oral.AOE_angle,
            "BOF_change": oral.BOF_angle - baseline_oral.BOF_angle,
        }

        # ========== 计算嘴角到眼部水平线距离的变化 ==========
        eye_line_excursion = compute_smile_excursion_by_eye_line(landmarks, w, h, baseline_landmarks)
        metrics["eye_line_excursion"] = eye_line_excursion

    # ========== 嘴唇面中线对称性（用于面瘫侧别判断）==========
    lip_symmetry = compute_lip_midline_symmetry(landmarks, w, h)
    metrics["lip_symmetry"] = lip_symmetry

    return metrics


def detect_palsy_side(smile_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    从微笑动作检测面瘫侧别 - 基于嘴角到眼线距离

    核心逻辑:
    - 微笑时嘴角上提，到眼线的距离减小
    - 距离减小量小的一侧 = 嘴角上提弱 = 面瘫侧
    - 或者：当前距离大的一侧 = 嘴角位置低 = 面瘫侧
    """
    result = {
        "palsy_side": 0,
        "confidence": 0.0,
        "interpretation": "",
        "method": "",
        "evidence": {}
    }

    if "eye_line_excursion" not in smile_metrics:
        result["interpretation"] = "无eye_line数据"
        return result

    exc = smile_metrics["eye_line_excursion"]

    # 优先使用距离减小量（如果有基线）
    left_reduction = exc.get("left_reduction", None)
    right_reduction = exc.get("right_reduction", None)

    if left_reduction is not None and right_reduction is not None:
        effective_left = max(0, left_reduction)
        effective_right = max(0, right_reduction)
        max_reduction = max(effective_left, effective_right)

        if max_reduction > 3:  # 有一定的微笑幅度
            result["method"] = "eye_line_reduction"
            result["evidence"]["left_reduction"] = left_reduction
            result["evidence"]["right_reduction"] = right_reduction

            asymmetry = abs(effective_left - effective_right) / max_reduction
            result["evidence"]["reduction_asymmetry"] = asymmetry
            result["confidence"] = min(1.0, asymmetry * 2)

            if asymmetry < THR.SMILE_ASYM_SYMMETRIC:
                result["palsy_side"] = 0
                result["interpretation"] = (
                    f"双侧嘴角上提对称 (L减小{left_reduction:.1f}px, R减小{right_reduction:.1f}px)"
                )
            elif effective_left < effective_right:
                # 左侧距离减小量小 = 左嘴角上提弱 = 左侧面瘫
                result["palsy_side"] = 1
                result["interpretation"] = (
                    f"左嘴角上提弱 (L减小{left_reduction:.1f}px < R减小{right_reduction:.1f}px) → 左侧面瘫"
                )
            else:
                result["palsy_side"] = 2
                result["interpretation"] = (
                    f"右嘴角上提弱 (R减小{right_reduction:.1f}px < L减小{left_reduction:.1f}px) → 右侧面瘫"
                )
            return result

    # 没有基线时，使用当前距离比较
    left_dist = exc.get("left_distance", 0)
    right_dist = exc.get("right_distance", 0)

    if left_dist > 0 or right_dist > 0:
        result["method"] = "eye_line_distance"
        result["evidence"]["left_distance"] = left_dist
        result["evidence"]["right_distance"] = right_dist

        avg_dist = (left_dist + right_dist) / 2
        if avg_dist > 1:
            asymmetry = abs(left_dist - right_dist) / avg_dist
            result["evidence"]["distance_asymmetry"] = asymmetry
            result["confidence"] = min(1.0, asymmetry * 3)

            if asymmetry < THR.SMILE_ASYM_SYMMETRIC:
                result["palsy_side"] = 0
                result["interpretation"] = (
                    f"双侧嘴角到眼线距离对称 (L={left_dist:.1f}px, R={right_dist:.1f}px, "
                    f"不对称{asymmetry:.1%})"
                )
            elif left_dist > right_dist:
                # 左嘴角距离大（位置低，上提少）→ 左侧面瘫
                result["palsy_side"] = 1
                result["interpretation"] = (
                    f"左嘴角距离大 (L={left_dist:.1f}px > R={right_dist:.1f}px, "
                    f"不对称{asymmetry:.1%}) → 左侧面瘫"
                )
            else:
                result["palsy_side"] = 2
                result["interpretation"] = (
                    f"右嘴角距离大 (R={right_dist:.1f}px > L={left_dist:.1f}px, "
                    f"不对称{asymmetry:.1%}) → 右侧面瘫"
                )
            return result

    result["interpretation"] = "无有效数据"
    return result


def compute_severity_score(smile_metrics: Dict[str, Any]) -> Tuple[int, str]:
    """
    计算动作严重度分数(医生标注标准)

    计算依据: 嘴角到眼线距离的不对称度
    """
    if "eye_line_excursion" in smile_metrics:
        exc = smile_metrics["eye_line_excursion"]

        # 优先使用距离减小量的不对称度
        left_reduction = exc.get("left_reduction", None)
        right_reduction = exc.get("right_reduction", None)

        if left_reduction is not None and right_reduction is not None:
            effective_left = max(0, left_reduction)
            effective_right = max(0, right_reduction)
            max_reduction = max(effective_left, effective_right)

            if max_reduction > 3:
                asymmetry = abs(effective_left - effective_right) / max_reduction

                if asymmetry < 0.08:
                    return 1, f"正常 (不对称度{asymmetry:.2%})"
                elif asymmetry < 0.18:
                    return 2, f"轻度异常 (不对称度{asymmetry:.2%})"
                elif asymmetry < 0.30:
                    return 3, f"中度异常 (不对称度{asymmetry:.2%})"
                elif asymmetry < 0.45:
                    return 4, f"重度异常 (不对称度{asymmetry:.2%})"
                else:
                    return 5, f"完全面瘫 (不对称度{asymmetry:.2%})"

        # 回退到当前距离
        left_dist = exc.get("left_distance", 0)
        right_dist = exc.get("right_distance", 0)

        avg_dist = (left_dist + right_dist) / 2
        if avg_dist > 1:
            asymmetry = abs(left_dist - right_dist) / avg_dist

            if asymmetry < 0.03:
                return 1, f"正常 (不对称度{asymmetry:.1%})"
            elif asymmetry < 0.08:
                return 2, f"轻度异常 (不对称度{asymmetry:.1%})"
            elif asymmetry < 0.15:
                return 3, f"中度异常 (不对称度{asymmetry:.1%})"
            elif asymmetry < 0.25:
                return 4, f"重度异常 (不对称度{asymmetry:.1%})"
            else:
                return 5, f"完全面瘫 (不对称度{asymmetry:.1%})"

    return 1, "无有效数据"


def detect_synkinesis_from_smile(baseline_result: Optional[ActionResult],
                                 current_landmarks, w: int, h: int) -> Dict[str, int]:
    """检测微笑时的联动运动 (主要是眼部)"""
    synkinesis = {
        "eye_synkinesis": 0,
        "brow_synkinesis": 0,
    }

    if baseline_result is None:
        return synkinesis

    # 当前EAR
    l_ear = compute_ear(current_landmarks, w, h, True)
    r_ear = compute_ear(current_landmarks, w, h, False)

    # 基线EAR
    baseline_l_ear = baseline_result.left_ear
    baseline_r_ear = baseline_result.right_ear

    # 检测眼部联动 (微笑时眼睛变小)
    if baseline_l_ear > 1e-9 and baseline_r_ear > 1e-9:
        l_change = (baseline_l_ear - l_ear) / baseline_l_ear
        r_change = (baseline_r_ear - r_ear) / baseline_r_ear
        avg_change = (l_change + r_change) / 2

        if avg_change > 0.25:  # 眼睛明显变小
            synkinesis["eye_synkinesis"] = 3
        elif avg_change > 0.15:
            synkinesis["eye_synkinesis"] = 2
        elif avg_change > 0.08:
            synkinesis["eye_synkinesis"] = 1

    return synkinesis


def visualize_smile_indicators(frame: np.ndarray, landmarks, w: int, h: int,
                               result: ActionResult,
                               smile_metrics: Dict[str, Any],
                               palsy_detection: Dict[str, Any]) -> np.ndarray:
    """可视化微笑指标 - 字体放大版"""
    img = frame.copy()

    # 字体参数（放大4倍）
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE_TITLE = 1.4  # 标题字号
    FONT_SCALE_LARGE = 1.0  # 大字号
    FONT_SCALE_NORMAL = 0.9  # 普通字号
    THICKNESS_TITLE = 3
    THICKNESS_NORMAL = 2
    LINE_HEIGHT = 50  # 行高

    # ========== 在左上角绘制患侧标签 ==========
    img = draw_palsy_side_label(img, palsy_detection, x=20, y=70, font_scale=1.4)

    # 绘制嘴部轮廓
    draw_polygon(img, landmarks, w, h, LM.OUTER_LIP, (0, 255, 0), 3)

    # ========== 绘制眼部水平线和嘴角到眼线的距离 ==========
    if "eye_line_excursion" in smile_metrics:
        exc = smile_metrics["eye_line_excursion"]
        eye_line_y = exc.get("eye_line_y", None)
        left_corner = exc.get("left_corner", None)
        right_corner = exc.get("right_corner", None)
        left_dist = exc.get("left_distance", 0)
        right_dist = exc.get("right_distance", 0)

        if eye_line_y is not None:
            # 绘制眼部水平线（青色虚线）
            for x in range(0, w, 20):
                cv2.line(img, (x, int(eye_line_y)), (min(x + 10, w), int(eye_line_y)),
                         (255, 255, 0), 2)

            # 标注眼线
            cv2.putText(img, "Eye Line", (w - 150, int(eye_line_y) - 10),
                        FONT, 0.6, (255, 255, 0), 2)

        # 绘制嘴角到眼线的垂直距离线
        if left_corner is not None and eye_line_y is not None:
            lx, ly = int(left_corner[0]), int(left_corner[1])
            # 左嘴角到眼线的垂直线（蓝色）
            cv2.line(img, (lx, ly), (lx, int(eye_line_y)), (255, 0, 0), 2)
            cv2.circle(img, (lx, ly), 8, (255, 0, 0), -1)
            # 标注距离
            mid_y = (ly + int(eye_line_y)) // 2
            cv2.putText(img, f"L:{left_dist:.0f}", (lx - 60, mid_y),
                        FONT, 0.5, (255, 0, 0), 2)

        if right_corner is not None and eye_line_y is not None:
            rx, ry = int(right_corner[0]), int(right_corner[1])
            # 右嘴角到眼线的垂直线（红色）
            cv2.line(img, (rx, ry), (rx, int(eye_line_y)), (0, 0, 255), 2)
            cv2.circle(img, (rx, ry), 8, (0, 0, 255), -1)
            # 标注距离
            mid_y = (ry + int(eye_line_y)) // 2
            cv2.putText(img, f"R:{right_dist:.0f}", (rx + 10, mid_y),
                        FONT, 0.5, (0, 0, 255), 2)

    # 绘制嘴角点
    if result.oral_angle:
        oral = result.oral_angle
        cv2.circle(img, (int(oral.A[0]), int(oral.A[1])), 10, (0, 0, 255), -1)
        cv2.circle(img, (int(oral.B[0]), int(oral.B[1])), 10, (255, 0, 0), -1)
        cv2.circle(img, (int(oral.O[0]), int(oral.O[1])), 8, (255, 255, 255), -1)
        cv2.line(img, (int(oral.E[0]), int(oral.E[1])),
                 (int(oral.F[0]), int(oral.F[1])), (0, 255, 0), 3)
        cv2.line(img, (int(oral.O[0]), int(oral.O[1])),
                 (int(oral.A[0]), int(oral.A[1])), (0, 0, 255), 3)
        cv2.line(img, (int(oral.O[0]), int(oral.O[1])),
                 (int(oral.B[0]), int(oral.B[1])), (255, 0, 0), 3)

    # 信息面板
    panel_w, panel_h = 700, 550
    cv2.rectangle(img, (10, 100), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.rectangle(img, (10, 100), (panel_w, panel_h), (255, 255, 255), 2)

    y = 160
    cv2.putText(img, f"{result.action_name}", (25, y),
                FONT, FONT_SCALE_TITLE, (0, 255, 0), THICKNESS_TITLE)
    y += LINE_HEIGHT + 10

    cv2.putText(img, f"Mouth Width: {smile_metrics['mouth_width']:.1f}px", (25, y),
                FONT, FONT_SCALE_NORMAL, (255, 255, 255), THICKNESS_NORMAL)
    y += LINE_HEIGHT

    oral_angle = smile_metrics.get("oral_angle", {})
    cv2.putText(img, f"AOE(R): {oral_angle.get('AOE', 0):+.1f}  BOF(L): {oral_angle.get('BOF', 0):+.1f}", (25, y),
                FONT, FONT_SCALE_NORMAL, (255, 255, 255), THICKNESS_NORMAL)
    y += LINE_HEIGHT

    asym = oral_angle.get('asymmetry', 0)
    asym_color = (0, 255, 0) if asym < 5 else ((0, 165, 255) if asym < 10 else (0, 0, 255))
    cv2.putText(img, f"Asymmetry: {asym:.1f} deg", (25, y),
                FONT, FONT_SCALE_NORMAL, asym_color, THICKNESS_NORMAL)
    y += LINE_HEIGHT + 10

    # 嘴角到眼线距离
    if "eye_line_excursion" in smile_metrics:
        exc = smile_metrics["eye_line_excursion"]
        cv2.putText(img, "=== Corner-to-EyeLine ===", (25, y),
                    FONT, FONT_SCALE_NORMAL, (0, 255, 255), THICKNESS_NORMAL)
        y += LINE_HEIGHT

        left_dist = exc.get("left_distance", 0)
        right_dist = exc.get("right_distance", 0)

        # 距离更小的用绿色（健侧），更大的用红色（面瘫侧）
        left_color = (0, 255, 0) if left_dist <= right_dist else (0, 0, 255)
        right_color = (0, 255, 0) if right_dist <= left_dist else (0, 0, 255)

        cv2.putText(img, f"L dist: {left_dist:.1f}px", (25, y),
                    FONT, FONT_SCALE_NORMAL, left_color, THICKNESS_NORMAL)
        y += LINE_HEIGHT
        cv2.putText(img, f"R dist: {right_dist:.1f}px", (25, y),
                    FONT, FONT_SCALE_NORMAL, right_color, THICKNESS_NORMAL)
        y += LINE_HEIGHT

        # 如果有reduction数据也显示
        if "left_reduction" in exc:
            cv2.putText(img, f"L reduce: {exc['left_reduction']:.1f}px", (25, y),
                        FONT, FONT_SCALE_NORMAL, (255, 255, 255), THICKNESS_NORMAL)
            y += LINE_HEIGHT
            cv2.putText(img, f"R reduce: {exc['right_reduction']:.1f}px", (25, y),
                        FONT, FONT_SCALE_NORMAL, (255, 255, 255), THICKNESS_NORMAL)
            y += LINE_HEIGHT

    # 面瘫侧别检测结果
    palsy_side = palsy_detection.get("palsy_side", 0)
    palsy_text = {0: "Symmetric", 1: "Left Palsy", 2: "Right Palsy"}.get(palsy_side, "Unknown")
    palsy_color = (0, 255, 0) if palsy_side == 0 else (0, 0, 255)
    cv2.putText(img, f"Palsy: {palsy_text}", (25, y),
                FONT, FONT_SCALE_LARGE, palsy_color, THICKNESS_NORMAL)
    y += LINE_HEIGHT + 10

    # Voluntary Score
    cv2.putText(img, f"Voluntary Score: {result.voluntary_movement_score}/5", (25, y),
                FONT, FONT_SCALE_LARGE, (0, 255, 255), THICKNESS_TITLE)

    return img


def process(landmarks_seq: List, frames_seq: List, w: int, h: int,
            video_info: Dict[str, Any], output_dir: Path,
            baseline_result: Optional[ActionResult] = None,
            baseline_landmarks=None) -> Optional[ActionResult]:
    """处理微笑类动作的通用函数"""
    if not landmarks_seq or not frames_seq:
        return None

    # 找峰值帧（嘴角到眼线距离最小）
    peak_idx, peak_debug = find_peak_frame(landmarks_seq, frames_seq, w, h, baseline_landmarks)
    peak_landmarks = landmarks_seq[peak_idx]
    peak_frame = frames_seq[peak_idx]

    # 提取时序序列用于可视化
    sequences = extract_smile_sequences(landmarks_seq, w, h)

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
    extract_common_indicators(peak_landmarks, w, h, result)

    # 计算微笑特有指标
    smile_metrics = compute_smile_metrics(peak_landmarks, w, h, baseline_landmarks)

    # 检测面瘫侧别
    palsy_detection = detect_palsy_side(smile_metrics)

    # 检测联动
    synkinesis = detect_synkinesis_from_smile(baseline_result, peak_landmarks, w, h)
    result.synkinesis_scores = synkinesis

    # 计算严重度分数 (医生标注标准: 1=正常, 5=面瘫)
    severity_score, severity_desc = compute_severity_score(smile_metrics)

    # 计算Voluntary Movement评分
    if "excursion" in smile_metrics:
        exc_ratio = smile_metrics["excursion"]["excursion_ratio"]
        score, interp = compute_voluntary_score_from_ratio(exc_ratio)
        result.voluntary_movement_score = score
    else:
        # 没有基线时使用口角对称性
        oral = smile_metrics.get("oral_angle", {})
        asym = oral.get("asymmetry", 0)
        if asym < 3:
            result.voluntary_movement_score = 5
        elif asym < 6:
            result.voluntary_movement_score = 4
        elif asym < 10:
            result.voluntary_movement_score = 3
        elif asym < 15:
            result.voluntary_movement_score = 2
        else:
            result.voluntary_movement_score = 1

    # 存储动作特有指标
    result.action_specific = {
        "smile_metrics": {
            "mouth_width": smile_metrics["mouth_width"],
            "mouth_height": smile_metrics["mouth_height"],
            "oral_angle": smile_metrics["oral_angle"],
        },
        "palsy_detection": palsy_detection,
        "synkinesis": synkinesis,
        "voluntary_score": result.voluntary_movement_score,
        "severity_score": severity_score,
        "severity_desc": severity_desc,
        "peak_debug": peak_debug,
    }

    if "excursion" in smile_metrics:
        result.action_specific["excursion"] = smile_metrics["excursion"]

    # 创建输出目录
    action_dir = output_dir / ACTION_NAME
    action_dir.mkdir(parents=True, exist_ok=True)

    # 绘制关键帧选择曲线（添加患侧信息）
    plot_smile_peak_selection(
        peak_debug,
        video_info.get("fps", 30.0),
        action_dir / "peak_selection_curve.png",
        palsy_detection
    )

    # 保存原始帧
    cv2.imwrite(str(action_dir / "peak_raw.jpg"), peak_frame)

    # 保存可视化
    vis = visualize_smile_indicators(peak_frame, peak_landmarks, w, h, result,
                                     smile_metrics, palsy_detection)
    cv2.imwrite(str(action_dir / "peak_indicators.jpg"), vis)

    # 保存JSON
    with open(action_dir / "indicators.json", 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    oral = smile_metrics.get("oral_angle", {})
    print(f"    [OK] {ACTION_NAME}: Width={smile_metrics['mouth_width']:.1f}px, Asym={oral.get('asymmetry', 0):.1f}°")
    print(f"         Palsy: {palsy_detection.get('interpretation', 'N/A')}")
    print(f"         Voluntary Score: {result.voluntary_movement_score}/5")

    return result