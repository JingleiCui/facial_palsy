#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShrugNose 动作处理模块 (V2)
================================

核心指标：鼻翼到眼部水平线的垂直距离

测量方法：
1. 眼部水平线：双侧内眦点连线
2. 扩展眼线：以中点为中心扩展到2倍长度（用于可视化和归一化）
3. 垂直距离：从鼻翼点向眼线作垂线，取点到线的垂直距离

面瘫判断：
- 侧别：距离更长的一侧为患侧（不需要baseline）
- 程度：看距离变化量 + 左右偏差（需要baseline）

对应Sunnybrook: Snarl (鼻翼上提)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json

from clinical_base import (
    LM, pt2d, dist, compute_ear,
    compute_mouth_metrics, extract_common_indicators,
    ActionResult, compute_scale_to_baseline,
    draw_palsy_annotation_header,
)

from thresholds import THR

ACTION_NAME = "ShrugNose"
ACTION_NAME_CN = "皱鼻"

# OpenCV字体
FONT = cv2.FONT_HERSHEY_SIMPLEX


# =============================================================================
# 核心几何计算函数
# =============================================================================

def get_inner_canthi(landmarks, w: int, h: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """获取左右内眦点坐标"""
    pL = pt2d(landmarks[LM.EYE_INNER_L], w, h)
    pR = pt2d(landmarks[LM.EYE_INNER_R], w, h)
    return pL, pR


def get_ala_points(landmarks, w: int, h: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """获取左右鼻翼点坐标"""
    left_ala = pt2d(landmarks[LM.NOSE_ALA_L], w, h)
    right_ala = pt2d(landmarks[LM.NOSE_ALA_R], w, h)
    return left_ala, right_ala


def extend_segment_2x(pL: Tuple, pR: Tuple) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    将线段以中点为中心扩展到2倍长度

    原端点 = m ± v/2
    扩展端点 = m ± v
    """
    mx = (pL[0] + pR[0]) / 2.0
    my = (pL[1] + pR[1]) / 2.0
    vx = pR[0] - pL[0]
    vy = pR[1] - pL[1]

    ext_L = (mx - vx, my - vy)
    ext_R = (mx + vx, my + vy)
    return ext_L, ext_R


def project_point_to_line(p: Tuple, a: Tuple, b: Tuple) -> Tuple[float, float]:
    """
    计算点p到直线ab的投影垂足坐标

    Returns:
        垂足点坐标 (fx, fy)
    """
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    px, py = float(p[0]), float(p[1])

    vx, vy = bx - ax, by - ay
    denom = vx * vx + vy * vy

    if denom < 1e-8:
        return (ax, ay)

    t = ((px - ax) * vx + (py - ay) * vy) / denom
    fx = ax + t * vx
    fy = ay + t * vy
    return (fx, fy)


def perpendicular_distance(p: Tuple, a: Tuple, b: Tuple) -> float:
    """
    计算点p到直线ab的垂直距离

    Returns:
        垂直距离（像素，非负）
    """
    fx, fy = project_point_to_line(p, a, b)
    dx = float(p[0]) - fx
    dy = float(p[1]) - fy
    return float(np.sqrt(dx * dx + dy * dy))


def compute_eye_line_metrics(landmarks, w: int, h: int) -> Dict[str, Any]:
    """
    计算眼部水平线相关的所有指标

    Returns:
        {
            "inner_canthi": {"left": (x,y), "right": (x,y)},
            "eye_line_original": {"start": (x,y), "end": (x,y), "length": float},
            "eye_line_extended": {"start": (x,y), "end": (x,y), "length": float},
            "ala_points": {"left": (x,y), "right": (x,y)},
            "perpendicular": {
                "left": {"foot": (x,y), "distance_px": float, "distance_norm": float},
                "right": {"foot": (x,y), "distance_px": float, "distance_norm": float},
            },
            "asymmetry": float,
            "longer_side": str,
        }
    """
    # 1. 获取关键点
    left_canthus, right_canthus = get_inner_canthi(landmarks, w, h)
    left_ala, right_ala = get_ala_points(landmarks, w, h)

    # 2. 原始眼线长度
    original_length = dist(left_canthus, right_canthus)

    # 3. 扩展眼线（2倍长度）
    ext_L, ext_R = extend_segment_2x(left_canthus, right_canthus)
    extended_length = original_length * 2

    # 4. 计算垂直距离和垂足
    left_foot = project_point_to_line(left_ala, ext_L, ext_R)
    right_foot = project_point_to_line(right_ala, ext_L, ext_R)

    left_dist_px = perpendicular_distance(left_ala, ext_L, ext_R)
    right_dist_px = perpendicular_distance(right_ala, ext_L, ext_R)

    # 5. 归一化（除以扩展眼线长度）
    left_dist_norm = left_dist_px / extended_length if extended_length > 0 else 0
    right_dist_norm = right_dist_px / extended_length if extended_length > 0 else 0

    # 6. 计算不对称度
    max_dist = max(left_dist_px, right_dist_px)
    asymmetry = abs(left_dist_px - right_dist_px) / max_dist if max_dist > 1e-6 else 0

    # 7. 判断哪侧更长
    longer_side = "left" if left_dist_px > right_dist_px else ("right" if right_dist_px > left_dist_px else "equal")

    return {
        "inner_canthi": {"left": left_canthus, "right": right_canthus},
        "eye_line_original": {
            "start": left_canthus,
            "end": right_canthus,
            "length": original_length
        },
        "eye_line_extended": {
            "start": ext_L,
            "end": ext_R,
            "length": extended_length
        },
        "ala_points": {"left": left_ala, "right": right_ala},
        "perpendicular": {
            "left": {
                "foot": left_foot,
                "distance_px": left_dist_px,
                "distance_norm": left_dist_norm,
            },
            "right": {
                "foot": right_foot,
                "distance_px": right_dist_px,
                "distance_norm": right_dist_norm,
            },
        },
        "asymmetry": asymmetry,
        "longer_side": longer_side,
    }


# =============================================================================
# 关键帧检测
# =============================================================================

def find_peak_frame(landmarks_seq: List, frames_seq: List, w: int, h: int,
                    baseline_landmarks=None) -> Tuple[int, Dict[str, Any]]:
    """
    找皱鼻峰值帧

    皱鼻时鼻翼向上 → 垂直距离减小
    选择 (left_dist + right_dist) 最小的帧
    """
    n_frames = len(landmarks_seq)
    if n_frames == 0:
        return 0, {"error": "empty_sequence"}

    # 时序数据收集
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

        metrics = compute_eye_line_metrics(lm, w, h)

        left_dist = metrics["perpendicular"]["left"]["distance_px"]
        right_dist = metrics["perpendicular"]["right"]["distance_px"]
        total_dist = left_dist + right_dist

        left_dist_seq.append(left_dist)
        right_dist_seq.append(right_dist)
        total_dist_seq.append(total_dist)

        if total_dist < min_total_dist:
            min_total_dist = total_dist
            min_idx = i

    peak_debug = {
        "left_vertical_dist": left_dist_seq,
        "right_vertical_dist": right_dist_seq,
        "total_vertical_dist": total_dist_seq,
        "peak_idx": min_idx,
        "peak_value": float(min_total_dist) if np.isfinite(min_total_dist) else None,
        "selection_criterion": "min_total_vertical_distance",
    }

    return min_idx, peak_debug


# =============================================================================
# 面瘫侧别检测（不需要baseline）
# =============================================================================

def detect_palsy_side(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    面瘫侧别判断

    原理：
    - 鼻翼到眼部水平线的垂直距离
    - 距离长 = 鼻翼位置低 = 患侧（皱鼻时该侧上提能力差）
    - 不需要baseline
    """
    perp = metrics.get("perpendicular", {})
    left_dist = perp.get("left", {}).get("distance_px", 0)
    right_dist = perp.get("right", {}).get("distance_px", 0)
    asymmetry = metrics.get("asymmetry", 0)

    result = {
        "palsy_side": 0,
        "confidence": 0.0,
        "interpretation": "",
        "method": "perpendicular_distance_to_eye_line",
        "evidence": {
            "left_distance_px": float(left_dist),
            "right_distance_px": float(right_dist),
            "distance_diff_px": float(abs(left_dist - right_dist)),
            "asymmetry": float(asymmetry),
        }
    }

    # 检查有效性
    max_dist = max(left_dist, right_dist)
    if max_dist < 1e-6:
        result["interpretation"] = "距离数据无效"
        return result

    # 使用阈值判断
    threshold = THR.SHRUG_NOSE_STATIC_ASYMMETRY

    result["evidence"]["threshold"] = threshold
    result["evidence"]["threshold_status"] = "below" if asymmetry < threshold else "above"

    if asymmetry < threshold:
        result["palsy_side"] = 0
        result["confidence"] = 1.0 - (asymmetry / threshold)
        result["interpretation"] = (
            f"双侧对称: L={left_dist:.1f}px, R={right_dist:.1f}px, "
            f"不对称度={asymmetry:.2%} < 阈值{threshold:.2%}"
        )
    elif left_dist > right_dist:
        result["palsy_side"] = 1  # 左侧面瘫
        result["confidence"] = min(1.0, asymmetry / 0.3)  # 30%以上高置信
        result["interpretation"] = (
            f"左侧距离更长: L={left_dist:.1f}px > R={right_dist:.1f}px, "
            f"不对称度={asymmetry:.2%} → 左侧面瘫"
        )
    else:
        result["palsy_side"] = 2  # 右侧面瘫
        result["confidence"] = min(1.0, asymmetry / 0.3)
        result["interpretation"] = (
            f"右侧距离更长: R={right_dist:.1f}px > L={left_dist:.1f}px, "
            f"不对称度={asymmetry:.2%} → 右侧面瘫"
        )

    return result


# =============================================================================
# 面瘫程度判断（需要baseline）
# =============================================================================

def compute_severity_with_baseline(
        current_metrics: Dict[str, Any],
        baseline_metrics: Dict[str, Any],
        scale: float = 1.0
) -> Dict[str, Any]:
    """
    面瘫程度判断 - 需要baseline

    评估维度：
    1. 运动幅度：左右侧距离减小量（皱鼻时鼻翼上提，距离变小）
    2. 运动对称性：左右减小量的比值
    3. 当前位置不对称度
    """
    # 当前帧的垂直距离
    curr_left = current_metrics["perpendicular"]["left"]["distance_px"]
    curr_right = current_metrics["perpendicular"]["right"]["distance_px"]

    # 基线帧的垂直距离
    base_left = baseline_metrics["perpendicular"]["left"]["distance_px"]
    base_right = baseline_metrics["perpendicular"]["right"]["distance_px"]

    # 计算运动量（正值表示向上移动，即距离减小）
    # 注意要考虑scale（人脸远近变化）
    left_movement = base_left - (curr_left * scale)
    right_movement = base_right - (curr_right * scale)

    # 运动对称性
    max_movement = max(abs(left_movement), abs(right_movement))
    min_movement = min(abs(left_movement), abs(right_movement))

    movement_symmetry = min_movement / max_movement if max_movement > THR.SHRUG_NOSE_MIN_MOVEMENT else 1.0

    # 当前位置不对称度
    current_asymmetry = current_metrics["asymmetry"]

    # 综合判断严重等级
    severity_score = 1
    severity_desc = ""

    if max_movement < THR.SHRUG_NOSE_MIN_MOVEMENT:
        severity_score = 1
        severity_desc = f"运动幅度过小 (max={max_movement:.1f}px < {THR.SHRUG_NOSE_MIN_MOVEMENT}px)"

    elif movement_symmetry >= THR.SHRUG_NOSE_GRADE1_MOVEMENT_SYMMETRY and \
            current_asymmetry < THR.SHRUG_NOSE_GRADE1_POSITION_ASYMMETRY:
        severity_score = 1
        severity_desc = f"正常 (运动对称{movement_symmetry:.1%}, 位置不对称{current_asymmetry:.1%})"

    elif movement_symmetry >= THR.SHRUG_NOSE_GRADE2_MOVEMENT_SYMMETRY and \
            current_asymmetry < THR.SHRUG_NOSE_GRADE2_POSITION_ASYMMETRY:
        severity_score = 2
        severity_desc = f"轻度 (运动对称{movement_symmetry:.1%}, 位置不对称{current_asymmetry:.1%})"

    elif movement_symmetry >= THR.SHRUG_NOSE_GRADE3_MOVEMENT_SYMMETRY or \
            current_asymmetry < THR.SHRUG_NOSE_GRADE3_POSITION_ASYMMETRY:
        severity_score = 3
        severity_desc = f"中度 (运动对称{movement_symmetry:.1%}, 位置不对称{current_asymmetry:.1%})"

    elif movement_symmetry >= THR.SHRUG_NOSE_GRADE4_MOVEMENT_SYMMETRY or \
            current_asymmetry < THR.SHRUG_NOSE_GRADE4_POSITION_ASYMMETRY:
        severity_score = 4
        severity_desc = f"重度 (运动对称{movement_symmetry:.1%}, 位置不对称{current_asymmetry:.1%})"

    else:
        severity_score = 5
        severity_desc = f"完全麻痹 (运动对称{movement_symmetry:.1%}, 位置不对称{current_asymmetry:.1%})"

    return {
        "severity_score": severity_score,
        "severity_desc": severity_desc,
        "movement": {
            "left_px": float(left_movement),
            "right_px": float(right_movement),
            "diff_px": float(abs(left_movement - right_movement)),
            "symmetry": float(movement_symmetry),
        },
        "current_asymmetry": float(current_asymmetry),
        "scale": float(scale),
        "thresholds_used": {
            "min_movement": THR.SHRUG_NOSE_MIN_MOVEMENT,
            "grade1_movement_sym": THR.SHRUG_NOSE_GRADE1_MOVEMENT_SYMMETRY,
            "grade1_position_asym": THR.SHRUG_NOSE_GRADE1_POSITION_ASYMMETRY,
        }
    }


# =============================================================================
# 可视化
# =============================================================================

def visualize_shrug_nose(
        img: np.ndarray,
        landmarks,
        w: int, h: int,
        result: ActionResult,
        metrics: Dict[str, Any],
        palsy_detection: Dict[str, Any],
        severity_info: Dict[str, Any] = None
) -> np.ndarray:
    """
    可视化皱鼻指标

    绘制内容：
    1. 眼部水平线（扩展后的2倍长度）
    2. 左右鼻翼点
    3. 垂线（从鼻翼到眼线的垂足）
    4. 距离标注
    5. 信息面板
    """
    img = img.copy()

    # 获取几何数据
    eye_line_ext = metrics.get("eye_line_extended", {})
    eye_start = eye_line_ext.get("start", (0, 0))
    eye_end = eye_line_ext.get("end", (0, 0))

    ala_points = metrics.get("ala_points", {})
    left_ala = ala_points.get("left", (0, 0))
    right_ala = ala_points.get("right", (0, 0))

    perp = metrics.get("perpendicular", {})
    left_foot = perp.get("left", {}).get("foot", (0, 0))
    right_foot = perp.get("right", {}).get("foot", (0, 0))
    left_dist = perp.get("left", {}).get("distance_px", 0)
    right_dist = perp.get("right", {}).get("distance_px", 0)

    asymmetry = metrics.get("asymmetry", 0)

    # ========== 颜色定义 ==========
    COLOR_EYE_LINE = (0, 255, 0)  # 绿色 - 眼线
    COLOR_LEFT_ALA = (255, 100, 100)  # 蓝色 - 左鼻翼
    COLOR_RIGHT_ALA = (100, 100, 255)  # 红色 - 右鼻翼
    COLOR_PERP_LINE = (255, 255, 0)  # 青色 - 垂线
    COLOR_HIGHLIGHT = (0, 0, 255)  # 红色高亮 - 患侧

    # 判断哪侧更长（患侧）
    left_color = COLOR_HIGHLIGHT if left_dist > right_dist else COLOR_LEFT_ALA
    right_color = COLOR_HIGHLIGHT if right_dist > left_dist else COLOR_RIGHT_ALA

    # ========== 1. 绘制扩展眼部水平线 ==========
    cv2.line(img,
             (int(eye_start[0]), int(eye_start[1])),
             (int(eye_end[0]), int(eye_end[1])),
             COLOR_EYE_LINE, 2, cv2.LINE_AA)

    # 标注原始内眦点
    canthi = metrics.get("inner_canthi", {})
    left_canthus = canthi.get("left", (0, 0))
    right_canthus = canthi.get("right", (0, 0))
    cv2.circle(img, (int(left_canthus[0]), int(left_canthus[1])), 4, COLOR_EYE_LINE, -1)
    cv2.circle(img, (int(right_canthus[0]), int(right_canthus[1])), 4, COLOR_EYE_LINE, -1)

    # ========== 2. 绘制鼻翼点 ==========
    cv2.circle(img, (int(left_ala[0]), int(left_ala[1])), 6, left_color, -1)
    cv2.circle(img, (int(right_ala[0]), int(right_ala[1])), 6, right_color, -1)

    # ========== 3. 绘制垂线 ==========
    # 左侧垂线
    cv2.line(img,
             (int(left_ala[0]), int(left_ala[1])),
             (int(left_foot[0]), int(left_foot[1])),
             left_color, 2, cv2.LINE_AA)
    # 垂足点
    cv2.circle(img, (int(left_foot[0]), int(left_foot[1])), 4, left_color, -1)

    # 右侧垂线
    cv2.line(img,
             (int(right_ala[0]), int(right_ala[1])),
             (int(right_foot[0]), int(right_foot[1])),
             right_color, 2, cv2.LINE_AA)
    # 垂足点
    cv2.circle(img, (int(right_foot[0]), int(right_foot[1])), 4, right_color, -1)

    # ========== 4. 标注距离值 ==========
    # 左侧距离
    mid_left_y = (left_ala[1] + left_foot[1]) / 2
    cv2.putText(img, f"{left_dist:.1f}px",
                (int(left_ala[0]) - 70, int(mid_left_y)),
                FONT, 0.55, left_color, 2)

    # 右侧距离
    mid_right_y = (right_ala[1] + right_foot[1]) / 2
    cv2.putText(img, f"{right_dist:.1f}px",
                (int(right_ala[0]) + 10, int(mid_right_y)),
                FONT, 0.55, right_color, 2)

    # ========== 5. 患侧标注头部 ==========
    img, header_end_y = draw_palsy_annotation_header(img, palsy_detection, ACTION_NAME)

    # ========== 6. 信息面板 ==========
    panel_top = header_end_y + 10
    panel_height = 280 if severity_info else 200
    panel_bottom = panel_top + panel_height

    cv2.rectangle(img, (5, panel_top), (380, panel_bottom), (0, 0, 0), -1)
    cv2.rectangle(img, (5, panel_top), (380, panel_bottom), (255, 255, 255), 1)

    y = panel_top + 25
    cv2.putText(img, f"{ACTION_NAME_CN} (ShrugNose)", (15, y), FONT, 0.65, (0, 255, 0), 2)
    y += 30

    # 垂直距离部分
    cv2.putText(img, "=== Perpendicular Distance ===", (15, y), FONT, 0.5, (0, 255, 255), 1)
    y += 25

    cv2.putText(img, f"Left:  {left_dist:.1f} px", (15, y), FONT, 0.5, left_color, 1)
    y += 22

    cv2.putText(img, f"Right: {right_dist:.1f} px", (15, y), FONT, 0.5, right_color, 1)
    y += 22

    cv2.putText(img, f"Diff:  {abs(left_dist - right_dist):.1f} px ({asymmetry:.1%})",
                (15, y), FONT, 0.5, (255, 255, 255), 1)
    y += 30

    # 阈值信息
    threshold = THR.SHRUG_NOSE_STATIC_ASYMMETRY
    status = "PASS" if asymmetry < threshold else "FAIL"
    status_color = (0, 255, 0) if asymmetry < threshold else (0, 0, 255)
    cv2.putText(img, f"Threshold: {threshold:.1%} | {status}",
                (15, y), FONT, 0.5, status_color, 1)
    y += 30

    # 运动信息（如果有baseline）
    if severity_info:
        cv2.putText(img, "=== Movement (from baseline) ===", (15, y), FONT, 0.5, (0, 255, 255), 1)
        y += 25

        movement = severity_info.get("movement", {})
        cv2.putText(img, f"Left:  {movement.get('left_px', 0):+.1f} px", (15, y), FONT, 0.5, (255, 255, 255), 1)
        y += 22
        cv2.putText(img, f"Right: {movement.get('right_px', 0):+.1f} px", (15, y), FONT, 0.5, (255, 255, 255), 1)
        y += 22
        cv2.putText(img, f"Symmetry: {movement.get('symmetry', 0):.1%}", (15, y), FONT, 0.5, (255, 255, 255), 1)
        y += 25

        # 严重度
        severity_score = severity_info.get("severity_score", 1)
        cv2.putText(img, f"Severity: {severity_score}/5", (15, y), FONT, 0.55, (0, 255, 255), 2)

    # ========== 7. 图例 ==========
    legend_y = panel_bottom + 15
    cv2.line(img, (15, legend_y), (45, legend_y), COLOR_EYE_LINE, 3)
    cv2.putText(img, "Eye Line (2x)", (50, legend_y + 4), FONT, 0.35, (255, 255, 255), 1)

    cv2.line(img, (160, legend_y), (190, legend_y), COLOR_LEFT_ALA, 3)
    cv2.putText(img, "Left Ala", (195, legend_y + 4), FONT, 0.35, (255, 255, 255), 1)

    cv2.line(img, (280, legend_y), (310, legend_y), COLOR_RIGHT_ALA, 3)
    cv2.putText(img, "Right Ala", (315, legend_y + 4), FONT, 0.35, (255, 255, 255), 1)

    return img


# =============================================================================
# 曲线图
# =============================================================================

def plot_peak_selection_curve(
        peak_debug: Dict[str, Any],
        fps: float,
        output_path,
        baseline_values: Dict[str, float] = None,
        palsy_detection: Dict[str, Any] = None,
        threshold_info: Dict[str, Any] = None
) -> None:
    """
    绘制关键帧选择曲线

    包含：
    1. 左右垂直距离时序曲线
    2. 基线参考（如果有）
    3. 峰值标记
    4. 阈值线和判断结果
    """
    import matplotlib.pyplot as plt
    from clinical_base import get_palsy_side_text

    left_dist = peak_debug.get("left_vertical_dist", [])
    right_dist = peak_debug.get("right_vertical_dist", [])
    total_dist = peak_debug.get("total_vertical_dist", [])
    peak_idx = peak_debug.get("peak_idx", 0)

    if not total_dist:
        return

    n_frames = len(total_dist)
    frames = np.arange(n_frames)
    time_sec = frames / fps if fps > 0 else frames
    x_label = 'Time (seconds)' if fps > 0 else 'Frame'
    peak_time = peak_idx / fps if fps > 0 else peak_idx

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # ===== 上图：左右垂直距离 =====
    ax1 = axes[0]
    ax1.plot(time_sec, left_dist, 'b-', label='Left Ala → Eye Line', linewidth=2, alpha=0.8)
    ax1.plot(time_sec, right_dist, 'r-', label='Right Ala → Eye Line', linewidth=2, alpha=0.8)

    # 基线参考
    if baseline_values:
        if "left" in baseline_values:
            ax1.axhline(y=baseline_values["left"], color='blue', linestyle=':', alpha=0.5, label='Baseline Left')
        if "right" in baseline_values:
            ax1.axhline(y=baseline_values["right"], color='red', linestyle=':', alpha=0.5, label='Baseline Right')

    # 峰值标记
    ax1.axvline(x=peak_time, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    if 0 <= peak_idx < n_frames:
        if np.isfinite(left_dist[peak_idx]):
            ax1.scatter([peak_time], [left_dist[peak_idx]], color='blue', s=100, zorder=5, marker='o',
                        edgecolors='black')
        if np.isfinite(right_dist[peak_idx]):
            ax1.scatter([peak_time], [right_dist[peak_idx]], color='red', s=100, zorder=5, marker='o',
                        edgecolors='black')

    ax1.set_ylabel('Perpendicular Distance (px)', fontsize=11)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.4)
    ax1.set_title('Left/Right Ala to Eye-Line Distance', fontsize=12, fontweight='bold')

    # ===== 下图：不对称度 =====
    ax2 = axes[1]

    # 计算每帧的不对称度
    asymmetry_seq = []
    for l, r in zip(left_dist, right_dist):
        if np.isnan(l) or np.isnan(r):
            asymmetry_seq.append(np.nan)
        else:
            max_val = max(l, r)
            asym = abs(l - r) / max_val if max_val > 1e-6 else 0
            asymmetry_seq.append(asym * 100)  # 转为百分比

    ax2.plot(time_sec, asymmetry_seq, 'purple', linewidth=2, label='Asymmetry')

    # 阈值线
    threshold = THR.SHRUG_NOSE_STATIC_ASYMMETRY * 100
    ax2.axhline(y=threshold, color='red', linestyle='--', linewidth=2,
                label=f'Threshold ({threshold:.1f}%)')

    # 填充超过阈值的区域
    asymmetry_arr = np.array(asymmetry_seq)
    ax2.fill_between(time_sec, 0, asymmetry_arr,
                     where=asymmetry_arr > threshold,
                     color='red', alpha=0.3, label='Above Threshold')

    # 峰值标记
    ax2.axvline(x=peak_time, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    if 0 <= peak_idx < n_frames and np.isfinite(asymmetry_seq[peak_idx]):
        ax2.scatter([peak_time], [asymmetry_seq[peak_idx]], color='green', s=150, zorder=5,
                    marker='*', edgecolors='black', linewidths=1.5)

    ax2.set_xlabel(x_label, fontsize=11)
    ax2.set_ylabel('Asymmetry (%)', fontsize=11)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.4)

    # 总标题
    title = "ShrugNose: Perpendicular Distance Analysis"
    if palsy_detection:
        palsy_text = get_palsy_side_text(palsy_detection.get("palsy_side", 0))
        confidence = palsy_detection.get("confidence", 0)
        title += f' | {palsy_text} (conf={confidence:.2f})'

    fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()


# =============================================================================
# 联动检测
# =============================================================================

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

        if avg_change > THR.SHRUG_NOSE_SYNKINESIS_SEVERE:
            synkinesis["eye_synkinesis"] = 3
        elif avg_change > THR.SHRUG_NOSE_SYNKINESIS_MODERATE:
            synkinesis["eye_synkinesis"] = 2
        elif avg_change > THR.SHRUG_NOSE_SYNKINESIS_MILD:
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


# =============================================================================
# 主处理函数
# =============================================================================

def process(landmarks_seq: List, frames_seq: List, w: int, h: int,
            video_info: Dict[str, Any], output_dir: Path,
            baseline_result: Optional[ActionResult] = None,
            baseline_landmarks=None) -> Optional[ActionResult]:
    """处理ShrugNose动作"""

    if not landmarks_seq or not frames_seq:
        return None

    # 1. 找峰值帧
    peak_idx, peak_debug = find_peak_frame(landmarks_seq, frames_seq, w, h, baseline_landmarks)
    peak_landmarks = landmarks_seq[peak_idx]
    peak_frame = frames_seq[peak_idx]

    if peak_landmarks is None:
        return None

    # 2. 计算核心指标
    metrics = compute_eye_line_metrics(peak_landmarks, w, h)

    # 3. 面瘫侧别检测（不需要baseline）
    palsy_detection = detect_palsy_side(metrics)

    # 4. 面瘫程度判断（需要baseline）
    severity_info = None
    baseline_metrics = None

    if baseline_landmarks is not None:
        baseline_metrics = compute_eye_line_metrics(baseline_landmarks, w, h)
        scale = compute_scale_to_baseline(peak_landmarks, baseline_landmarks, w, h)
        severity_info = compute_severity_with_baseline(metrics, baseline_metrics, scale)

    # 5. 创建结果对象
    result = ActionResult(
        action_name=ACTION_NAME,
        action_name_cn=ACTION_NAME_CN,
        video_path=video_info.get("file_path", ""),
        total_frames=len(frames_seq),
        peak_frame_idx=peak_idx,
        image_size=(w, h),
        fps=video_info.get("fps", 30.0)
    )

    # 6. 提取通用指标
    extract_common_indicators(peak_landmarks, w, h, result, baseline_landmarks)

    # 7. 检测联动
    synkinesis = detect_synkinesis(baseline_result, peak_landmarks, w, h)
    result.synkinesis_scores = synkinesis

    # 8. 存储动作特有指标
    perp = metrics["perpendicular"]
    result.action_specific = {
        "measurement_method": "perpendicular_distance_to_extended_eye_line",
        "perpendicular_distance": {
            "left_px": perp["left"]["distance_px"],
            "right_px": perp["right"]["distance_px"],
            "left_norm": perp["left"]["distance_norm"],
            "right_norm": perp["right"]["distance_norm"],
            "diff_px": abs(perp["left"]["distance_px"] - perp["right"]["distance_px"]),
            "asymmetry": metrics["asymmetry"],
        },
        "eye_line": {
            "original_length": metrics["eye_line_original"]["length"],
            "extended_length": metrics["eye_line_extended"]["length"],
        },
        "palsy_detection": palsy_detection,
        "peak_debug": peak_debug,
    }

    if severity_info:
        result.action_specific["severity"] = severity_info
        result.action_specific["baseline"] = {
            "left_px": baseline_metrics["perpendicular"]["left"]["distance_px"],
            "right_px": baseline_metrics["perpendicular"]["right"]["distance_px"],
        }
        result.voluntary_movement_score = severity_info["severity_score"]
    else:
        result.voluntary_movement_score = 1  # 无baseline时无法评估

    # 9. 创建输出目录
    action_dir = output_dir / ACTION_NAME
    action_dir.mkdir(parents=True, exist_ok=True)

    # 10. 保存原始帧
    cv2.imwrite(str(action_dir / "peak_raw.jpg"), peak_frame)

    # 11. 保存可视化
    vis = visualize_shrug_nose(peak_frame, peak_landmarks, w, h, result,
                               metrics, palsy_detection, severity_info)
    cv2.imwrite(str(action_dir / "peak_indicators.jpg"), vis)

    # 12. 准备基线值用于曲线图
    baseline_values = None
    if baseline_metrics:
        baseline_values = {
            "left": baseline_metrics["perpendicular"]["left"]["distance_px"],
            "right": baseline_metrics["perpendicular"]["right"]["distance_px"],
        }

    # 13. 绘制曲线图
    plot_peak_selection_curve(
        peak_debug,
        video_info.get("fps", 30.0),
        action_dir / "peak_selection_curve.png",
        baseline_values,
        palsy_detection,
    )

    # 14. 保存JSON
    with open(action_dir / "indicators.json", 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    # 15. 打印结果
    left_px = perp["left"]["distance_px"]
    right_px = perp["right"]["distance_px"]
    asym = metrics["asymmetry"]

    print(f"    [OK] {ACTION_NAME}: L={left_px:.1f}px R={right_px:.1f}px Asym={asym:.1%}")
    print(f"         Palsy: {palsy_detection.get('interpretation', 'N/A')}")

    if severity_info:
        print(f"         Movement: L={severity_info['movement']['left_px']:+.1f}px "
              f"R={severity_info['movement']['right_px']:+.1f}px")
        print(f"         Severity: {severity_info['severity_score']}/5 - {severity_info['severity_desc']}")

    return result