#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LipPucker 动作处理模块
================================

分析撅嘴动作:
1. 嘴唇宽度变化
2. 嘴唇高度变化
3. 嘴角位置对称性
4. 口角角度变化
5. 面瘫侧别检测
6. 联动运动检测

新增指标:
- 左右嘴角到嘴唇中线的垂直距离及不对称程度
- 嘴唇中线偏离面中线的距离
- 左右嘴角到面中线的垂直距离及不对称程度

对应Sunnybrook: Lip pucker (OOS/OOI)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from clinical_base import (
    LM, pt2d, pt3d, pts2d, dist, polygon_area, compute_ear, compute_eye_area,
    compute_mouth_metrics, compute_oral_angle,
    compute_icd, extract_common_indicators,
    ActionResult, draw_polygon, compute_scale_to_baseline,
    kabsch_rigid_transform, apply_rigid_transform,
    compute_lip_midline_symmetry,
    compute_lip_midline_offset_from_face_midline,
    compute_lip_midline_center, compute_face_midline, draw_face_midline,
    draw_palsy_annotation_header, compute_lip_midline_angle,
)

from thresholds import THR

ACTION_NAME = "LipPucker"
ACTION_NAME_CN = "撅嘴"

# =============================================================================
# 统一的可视化参数（供所有动作模块使用）
# =============================================================================

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


# =============================================================================
# 几何辅助函数
# =============================================================================

def point_to_line_perpendicular_distance(point, line_p1, line_p2):
    """
    计算点到直线的垂直距离，以及垂足点

    Args:
        point: (x, y) 点坐标
        line_p1: (x, y) 直线上的第一个点
        line_p2: (x, y) 直线上的第二个点

    Returns:
        distance: 垂直距离（有符号，正值表示点在直线左侧）
        foot: 垂足点坐标
    """
    px, py = point
    x1, y1 = line_p1
    x2, y2 = line_p2

    # 直线向量
    dx = x2 - x1
    dy = y2 - y1

    # 直线长度的平方
    line_len_sq = dx * dx + dy * dy

    if line_len_sq < 1e-9:
        # 两点重合，返回点到点的距离
        dist_val = math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
        return dist_val, (x1, y1)

    # 参数t表示垂足在直线上的位置
    t = ((px - x1) * dx + (py - y1) * dy) / line_len_sq

    # 垂足坐标
    foot_x = x1 + t * dx
    foot_y = y1 + t * dy

    # 垂直距离（使用叉积判断方向）
    # 正值表示点在直线左侧（从p1看向p2的方向）
    cross = dx * (py - y1) - dy * (px - x1)
    signed_dist = cross / math.sqrt(line_len_sq)

    return signed_dist, (foot_x, foot_y)


def _mean_lip_z_aligned(landmarks, w: int, h: int, baseline_landmarks=None) -> float:
    """
    计算"唇部区域平均z"（更稳：先把当前帧刚体对齐到 baseline，再取 z）。
    z 越小（更负）表示越靠近镜头。
    """
    lip_indices = list(LM.OUTER_LIP) + list(LM.INNER_LIP)

    # 没 baseline：直接算当前帧平均z（不推荐但可跑）
    if baseline_landmarks is None:
        zs = []
        for idx in lip_indices:
            x, y, z = pt3d(landmarks[idx], w, h)
            zs.append(z)
        return float(np.mean(zs)) if zs else float("nan")

    stable_idx = [
        LM.EYE_INNER_L, LM.EYE_INNER_R,
        LM.EYE_OUTER_L, LM.EYE_OUTER_R,
        LM.NOSE_BRIDGE, LM.NOSE_TIP,
        LM.CHIN
    ]

    P, Q = [], []
    for i in stable_idx:
        px, py, pz = pt3d(landmarks[i], w, h)
        qx, qy, qz = pt3d(baseline_landmarks[i], w, h)
        P.append([px, py, pz])
        Q.append([qx, qy, qz])

    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    Rm, t = kabsch_rigid_transform(P, Q)
    if Rm is None:
        # 对齐失败就退化
        return _mean_lip_z_aligned(landmarks, w, h, baseline_landmarks=None)

    # 当前唇部点 -> 对齐到 baseline 坐标
    cur = []
    for idx in lip_indices:
        x, y, z = pt3d(landmarks[idx], w, h)
        cur.append([x, y, z])
    cur = np.asarray(cur, dtype=np.float64)
    cur_aligned = apply_rigid_transform(cur, Rm, t)

    return float(np.mean(cur_aligned[:, 2]))


# =============================================================================
# 曲线绘制函数
# =============================================================================
def plot_lip_pucker_peak_selection(
        peak_debug: Dict[str, Any],
        fps: float,
        output_path: Path,
        palsy_detection: Dict[str, Any] = None
) -> None:
    """
    绘制撅嘴（LipPucker）关键帧选择的可解释性曲线。
    """
    import matplotlib.pyplot as plt
    from clinical_base import add_valid_region_shading, get_palsy_side_text

    protrusion = peak_debug.get("protrusion", [])
    width_ratio = peak_debug.get("width_ratio", [])
    score = peak_debug.get("score", [])
    valid_mask = peak_debug.get("valid", None)
    peak_idx = peak_debug.get("peak_idx", 0)

    if not score:
        return

    n_frames = len(score)
    frames = np.arange(n_frames)
    time_sec = frames / fps if fps > 0 else frames
    x_label = 'Time (seconds)' if fps > 0 else 'Frame'
    peak_time = peak_idx / fps if fps > 0 else peak_idx

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1 = axes[0]
    if valid_mask is not None:
        add_valid_region_shading(ax1, valid_mask, time_sec)

    ax1.plot(time_sec, protrusion, 'b-', label='Lip Protrusion', linewidth=2, alpha=0.7)
    ax1.plot(time_sec, score, 'g-', label='Combined Score (Selection)', linewidth=2.5)

    ax1.axvline(x=peak_time, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    if 0 <= peak_idx < n_frames:
        peak_score = score[peak_idx]
        ax1.scatter([peak_time], [peak_score], color='red', s=150, zorder=5,
                    edgecolors='black', linewidths=1.5, marker='*', label=f'Peak (Score: {peak_score:.3f})')

    title = "LipPucker Peak Selection: Max Protrusion & Puckering Score"
    if palsy_detection:
        palsy_text = get_palsy_side_text(palsy_detection.get("palsy_side", 0))
        title += f' | Detected: {palsy_text}'

    ax1.set_title(title, fontsize=13, fontweight='bold')
    ax1.set_ylabel('Score / Protrusion', fontsize=11)
    ax1.legend()
    ax1.grid(True, alpha=0.4)

    ax2 = axes[1]
    if valid_mask is not None:
        add_valid_region_shading(ax2, valid_mask, time_sec)

    ax2.plot(time_sec, width_ratio, 'orange', label='Mouth Width Ratio', linewidth=2)
    ax2.axhline(y=1.0, color='gray', linestyle=':', label='Baseline Width')

    ax2.axvline(x=peak_time, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.set_title('Mouth Width Ratio (lower is better)', fontsize=11)
    ax2.set_xlabel(x_label, fontsize=11)
    ax2.set_ylabel('Width Ratio', fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()


def find_peak_frame(landmarks_seq: List, frames_seq: List, w: int, h: int,
                    baseline_landmarks=None) -> Tuple[int, Dict[str, Any]]:
    """撅嘴峰值帧选择"""
    n = len(landmarks_seq)
    if n == 0:
        return 0, {"error": "empty_sequence"}

    BASELINE_FRAMES = getattr(THR, 'LIP_PUCKER_BASELINE_FRAMES', 10)
    MOUTH_HEIGHT_THR = THR.MOUTH_HEIGHT

    def get_nose_tip_z(lm):
        if lm is None:
            return float("nan")
        try:
            _, _, z = pt3d(lm[LM.NOSE_TIP], w, h)
            return float(z)
        except:
            return float("nan")

    def get_lip_mean_z(lm):
        if lm is None:
            return float("nan")
        try:
            lip_idx = list(LM.OUTER_LIP) + list(LM.INNER_LIP)
            zs = [pt3d(lm[i], w, h)[2] for i in lip_idx]
            return float(np.mean(zs))
        except:
            return float("nan")

    def get_icd(lm):
        if lm is None:
            return float("nan")
        try:
            return float(compute_icd(lm, w, h))
        except:
            return float("nan")

    def get_mouth_width(lm):
        if lm is None:
            return float("nan")
        try:
            lc = pt2d(lm[LM.MOUTH_L], w, h)
            rc = pt2d(lm[LM.MOUTH_R], w, h)
            return float(dist(lc, rc))
        except:
            return float("nan")

    def get_mouth_height_norm(lm, icd_val):
        if lm is None or icd_val <= 0:
            return float("nan")
        try:
            top = pt2d(lm[LM.LIP_TOP], w, h)
            bot = pt2d(lm[LM.LIP_BOT], w, h)
            return float(dist(top, bot) / icd_val)
        except:
            return float("nan")

    nose_z_arr = np.array([get_nose_tip_z(lm) for lm in landmarks_seq], dtype=np.float64)
    lip_z_arr = np.array([get_lip_mean_z(lm) for lm in landmarks_seq], dtype=np.float64)
    icd_arr = np.array([get_icd(lm) for lm in landmarks_seq], dtype=np.float64)
    mouth_width_arr = np.array([get_mouth_width(lm) for lm in landmarks_seq], dtype=np.float64)
    mouth_height_arr = np.array([get_mouth_height_norm(lm, icd_arr[i]) for i, lm in enumerate(landmarks_seq)],
                                 dtype=np.float64)

    lip_rel_z_arr = lip_z_arr - nose_z_arr

    valid_init_idx = np.where(
        np.isfinite(lip_rel_z_arr[:BASELINE_FRAMES]) &
        np.isfinite(icd_arr[:BASELINE_FRAMES]) &
        np.isfinite(mouth_width_arr[:BASELINE_FRAMES])
    )[0]

    if len(valid_init_idx) == 0:
        return 0, {"error": "no_valid_frames", "method": "nose_relative_protrusion"}

    base_lip_rel_z = float(np.median(lip_rel_z_arr[valid_init_idx]))
    base_icd = float(np.median(icd_arr[valid_init_idx]))
    base_mouth_width = float(np.median(mouth_width_arr[valid_init_idx]))

    protrusion_arr = np.full(n, np.nan, dtype=np.float64)
    width_ratio_arr = np.full(n, np.nan, dtype=np.float64)
    score_arr = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        if np.isfinite(lip_rel_z_arr[i]):
            protrusion_arr[i] = (base_lip_rel_z - lip_rel_z_arr[i]) / base_icd

        if np.isfinite(mouth_width_arr[i]) and base_mouth_width > 0:
            width_ratio_arr[i] = mouth_width_arr[i] / base_mouth_width

        if np.isfinite(protrusion_arr[i]):
            score_arr[i] = protrusion_arr[i]
            if np.isfinite(width_ratio_arr[i]) and width_ratio_arr[i] < 1.0:
                width_bonus = (1.0 - width_ratio_arr[i]) * 0.5
                score_arr[i] += width_bonus

    valid = (np.isfinite(score_arr) & (mouth_height_arr <= MOUTH_HEIGHT_THR))

    if valid.sum() < 3:
        valid = np.isfinite(score_arr) & (mouth_height_arr <= MOUTH_HEIGHT_THR * 1.5)

    if valid.sum() < 1:
        valid = np.isfinite(score_arr)

    fallback = None
    if valid.sum() < 1:
        fallback = "no_valid_fallback"
        peak_idx = int(np.nanargmax(score_arr)) if np.any(np.isfinite(score_arr)) else 0
    else:
        cand = np.where(valid)[0]
        peak_idx = int(cand[int(np.nanargmax(score_arr[cand]))])

    peak_debug = {
        "nose_z": [float(v) if np.isfinite(v) else None for v in nose_z_arr],
        "lip_z": [float(v) if np.isfinite(v) else None for v in lip_z_arr],
        "lip_rel_z": [float(v) if np.isfinite(v) else None for v in lip_rel_z_arr],
        "protrusion": [float(v) if np.isfinite(v) else None for v in protrusion_arr],
        "width_ratio": [float(v) if np.isfinite(v) else None for v in width_ratio_arr],
        "score": [float(v) if np.isfinite(v) else None for v in score_arr],
        "mouth_height_norm": [float(v) if np.isfinite(v) else None for v in mouth_height_arr],
        "baseline": {
            "frames_used": int(len(valid_init_idx)),
            "lip_rel_z": float(base_lip_rel_z),
            "icd": float(base_icd),
            "mouth_width": float(base_mouth_width),
        },
        "thresholds": {"mouth_height_thr": float(MOUTH_HEIGHT_THR)},
        "valid": [bool(v) for v in valid],
        "peak_idx": int(peak_idx),
        "fallback": fallback,
        "method": "nose_relative_protrusion",
    }

    return peak_idx, peak_debug


def compute_lip_pucker_metrics(landmarks, w: int, h: int,
                               baseline_landmarks=None) -> Dict[str, Any]:
    """
    计算撅嘴特有指标

    新增指标:
    - corner_to_lip_midline: 左右嘴角到嘴唇中线的垂直距离及不对称程度
    - lip_midline_to_face_midline: 嘴唇中线偏离面中线的距离
    - corner_to_face_midline: 左右嘴角到面中线的垂直距离及不对称程度
    """
    mouth = compute_mouth_metrics(landmarks, w, h)
    oral = compute_oral_angle(landmarks, w, h)

    left_corner = mouth["left_corner"]
    right_corner = mouth["right_corner"]

    left_canthus = pt2d(landmarks[LM.EYE_INNER_L], w, h)
    right_canthus = pt2d(landmarks[LM.EYE_INNER_R], w, h)
    midline_x = (left_canthus[0] + right_canthus[0]) / 2

    left_to_midline = abs(midline_x - left_corner[0])
    right_to_midline = abs(right_corner[0] - midline_x)

    mouth_center_x = (left_corner[0] + right_corner[0]) / 2
    mouth_midline_offset = mouth_center_x - midline_x

    avg_y = (left_corner[1] + right_corner[1]) / 2
    left_height_diff = avg_y - left_corner[1]
    right_height_diff = avg_y - right_corner[1]

    metrics = {
        "mouth_width": float(mouth["width"]),
        "mouth_height": float(mouth["height"]),
        "width_height_ratio": float(mouth["width"] / mouth["height"]) if mouth["height"] > 1e-9 else 0,
        "left_corner": left_corner,
        "right_corner": right_corner,
        "midline_x": float(midline_x),
        "left_to_midline": float(left_to_midline),
        "right_to_midline": float(right_to_midline),
        "mouth_midline_offset": float(mouth_midline_offset),
        "corner_height_diff": float(left_height_diff - right_height_diff),
        "oral_angle": {
            "AOE": float(oral.AOE_angle),
            "BOF": float(oral.BOF_angle),
            "asymmetry": float(oral.angle_asymmetry),
        }
    }

    metrics["icd"] = compute_icd(landmarks, w, h)
    icd = metrics["icd"]

    # ========== 嘴唇中线的定义 ==========
    lip_top = pt2d(landmarks[LM.LIP_TOP_CENTER], w, h)
    lip_bot = pt2d(landmarks[LM.LIP_BOT_CENTER], w, h)
    lip_midline_p1 = lip_top
    lip_midline_p2 = lip_bot

    metrics["lip_midline"] = {"top": lip_top, "bottom": lip_bot}

    # ========== 面中线的定义 ==========
    face_midline = compute_face_midline(landmarks, w, h)
    if face_midline:
        face_midline_center = face_midline["center"]
        face_midline_direction = face_midline["direction"]
        if face_midline_direction:
            dx, dy = face_midline_direction
            face_midline_p1 = (face_midline_center[0] - dx * 100, face_midline_center[1] - dy * 100)
            face_midline_p2 = (face_midline_center[0] + dx * 100, face_midline_center[1] + dy * 100)
        else:
            face_midline_p1 = (face_midline_center[0], face_midline_center[1] - 100)
            face_midline_p2 = (face_midline_center[0], face_midline_center[1] + 100)
    else:
        face_midline_p1 = (midline_x, 0)
        face_midline_p2 = (midline_x, h)
        face_midline_center = (midline_x, h / 2)

    metrics["face_midline"] = {"center": face_midline_center, "p1": face_midline_p1, "p2": face_midline_p2}

    # ========== 左右嘴角到嘴唇中线的垂直距离 ==========
    left_dist_to_lip_midline, left_foot_lip = point_to_line_perpendicular_distance(
        left_corner, lip_midline_p1, lip_midline_p2
    )
    right_dist_to_lip_midline, right_foot_lip = point_to_line_perpendicular_distance(
        right_corner, lip_midline_p1, lip_midline_p2
    )

    left_dist_lip_abs = abs(left_dist_to_lip_midline)
    right_dist_lip_abs = abs(right_dist_to_lip_midline)

    max_dist_lip = max(left_dist_lip_abs, right_dist_lip_abs)
    lip_midline_asymmetry = abs(left_dist_lip_abs - right_dist_lip_abs) / max_dist_lip if max_dist_lip > 1e-6 else 0

    metrics["corner_to_lip_midline"] = {
        "left_distance": float(left_dist_lip_abs),
        "right_distance": float(right_dist_lip_abs),
        "left_signed": float(left_dist_to_lip_midline),
        "right_signed": float(right_dist_to_lip_midline),
        "left_foot": left_foot_lip,
        "right_foot": right_foot_lip,
        "asymmetry": float(lip_midline_asymmetry),
        "left_norm": float(left_dist_lip_abs / icd) if icd > 1e-6 else 0,
        "right_norm": float(right_dist_lip_abs / icd) if icd > 1e-6 else 0,
    }

    # ========== 嘴唇中线偏离面中线的距离 ==========
    lip_midline_center = ((lip_top[0] + lip_bot[0]) / 2, (lip_top[1] + lip_bot[1]) / 2)
    lip_to_face_dist, lip_to_face_foot = point_to_line_perpendicular_distance(
        lip_midline_center, face_midline_p1, face_midline_p2
    )

    metrics["lip_midline_to_face_midline"] = {
        "distance": float(abs(lip_to_face_dist)),
        "signed_distance": float(lip_to_face_dist),
        "norm": float(abs(lip_to_face_dist) / icd) if icd > 1e-6 else 0,
        "lip_midline_center": lip_midline_center,
        "foot_on_face_midline": lip_to_face_foot,
    }

    # ========== 左右嘴角到面中线的垂直距离 ==========
    left_dist_to_face_midline, left_foot_face = point_to_line_perpendicular_distance(
        left_corner, face_midline_p1, face_midline_p2
    )
    right_dist_to_face_midline, right_foot_face = point_to_line_perpendicular_distance(
        right_corner, face_midline_p1, face_midline_p2
    )

    left_dist_face_abs = abs(left_dist_to_face_midline)
    right_dist_face_abs = abs(right_dist_to_face_midline)

    max_dist_face = max(left_dist_face_abs, right_dist_face_abs)
    face_midline_asymmetry = abs(left_dist_face_abs - right_dist_face_abs) / max_dist_face if max_dist_face > 1e-6 else 0

    metrics["corner_to_face_midline"] = {
        "left_distance": float(left_dist_face_abs),
        "right_distance": float(right_dist_face_abs),
        "left_signed": float(left_dist_to_face_midline),
        "right_signed": float(right_dist_to_face_midline),
        "left_foot": left_foot_face,
        "right_foot": right_foot_face,
        "asymmetry": float(face_midline_asymmetry),
        "left_norm": float(left_dist_face_abs / icd) if icd > 1e-6 else 0,
        "right_norm": float(right_dist_face_abs / icd) if icd > 1e-6 else 0,
    }

    # ========== 计算被面中线分开的左右嘴唇内缘面积 ==========
    inner_lip_left_pts = np.array(pts2d(landmarks, LM.INNER_LIP_L, w, h), dtype=np.float32)
    inner_lip_right_pts = np.array(pts2d(landmarks, LM.INNER_LIP_R, w, h), dtype=np.float32)

    inner_left_area = float(abs(polygon_area(inner_lip_left_pts))) if len(inner_lip_left_pts) >= 3 else 0.0
    inner_right_area = float(abs(polygon_area(inner_lip_right_pts))) if len(inner_lip_right_pts) >= 3 else 0.0

    inner_area_diff = abs(inner_left_area - inner_right_area)
    max_inner_area = max(inner_left_area, inner_right_area)
    inner_area_ratio = min(inner_left_area, inner_right_area) / max_inner_area if max_inner_area > 1e-6 else 1.0

    icd_sq = icd * icd + 1e-9

    metrics["inner_lip_areas"] = {
        "left": float(inner_left_area),
        "right": float(inner_right_area),
        "diff": float(inner_area_diff),
        "ratio": float(inner_area_ratio),
        "diff_norm": float(inner_area_diff / icd_sq),
    }

    lip_symmetry = compute_lip_midline_symmetry(landmarks, w, h)
    metrics["lip_symmetry"] = lip_symmetry
    lip_angle = compute_lip_midline_angle(landmarks, w, h)
    metrics["lip_midline_angle"] = lip_angle

    lip_offset_data = compute_lip_midline_offset_from_face_midline(landmarks, w, h, baseline_landmarks)
    metrics["lip_midline_offset"] = lip_offset_data

    # 基线参考
    if baseline_landmarks is not None:
        scale = compute_scale_to_baseline(landmarks, baseline_landmarks, w, h)
        metrics["scale"] = scale

        baseline_mouth = compute_mouth_metrics(baseline_landmarks, w, h)
        baseline_oral = compute_oral_angle(baseline_landmarks, w, h)

        aoe_change = oral.AOE_angle - baseline_oral.AOE_angle
        bof_change = oral.BOF_angle - baseline_oral.BOF_angle

        scaled_width = mouth["width"] * scale
        width_change = scaled_width - baseline_mouth["width"]

        metrics["baseline"] = {
            "mouth_width": float(baseline_mouth["width"]),
            "AOE": float(baseline_oral.AOE_angle),
            "BOF": float(baseline_oral.BOF_angle),
        }

        metrics["movement"] = {"aoe_change": float(aoe_change), "bof_change": float(bof_change)}

        metrics["width_change"] = float(width_change)
        metrics["width_ratio"] = float(scaled_width / baseline_mouth["width"]) if baseline_mouth["width"] > 1e-9 else 1.0

        if baseline_mouth["width"] > 1e-9:
            metrics["width_change_percent"] = width_change / baseline_mouth["width"] * 100
        else:
            metrics["width_change_percent"] = 0

        lip_center = compute_lip_midline_center(landmarks, w, h)
        baseline_lip_center = compute_lip_midline_center(baseline_landmarks, w, h)

        left_corner = pt2d(landmarks[LM.MOUTH_CORNER_L], w, h)
        right_corner = pt2d(landmarks[LM.MOUTH_CORNER_R], w, h)

        baseline_left_corner = pt2d(baseline_landmarks[LM.MOUTH_CORNER_L], w, h)
        baseline_right_corner = pt2d(baseline_landmarks[LM.MOUTH_CORNER_R], w, h)

        def eucl_dist(p1, p2):
            return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        baseline_left_dist = eucl_dist(baseline_left_corner, baseline_lip_center)
        baseline_right_dist = eucl_dist(baseline_right_corner, baseline_lip_center)
        current_left_dist = eucl_dist(left_corner, lip_center)
        current_right_dist = eucl_dist(right_corner, lip_center)

        left_contraction = baseline_left_dist - current_left_dist
        right_contraction = baseline_right_dist - current_right_dist

        metrics["left_corner_contraction"] = left_contraction
        metrics["right_corner_contraction"] = right_contraction

        max_contraction = max(abs(left_contraction), abs(right_contraction))
        if max_contraction > 1:
            metrics["corner_contraction_asymmetry"] = abs(left_contraction - right_contraction) / max_contraction
        else:
            metrics["corner_contraction_asymmetry"] = 0

    return metrics


def detect_palsy_side(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """LipPucker 侧别判定（弱证据，不依赖 baseline）"""

    lip_offset = metrics.get("lip_midline_offset", {}) or {}
    current_offset = float(lip_offset.get("current_offset", 0.0) or 0.0)
    offset_norm = float(abs(lip_offset.get("offset_norm", 0.0) or 0.0))
    thr_offset= THR.LIP_PUCKER_OFFSET_THRESHOLD

    lip_angle = metrics.get("lip_midline_angle", {}) or {}
    angle = float(lip_angle.get("angle", 0.0) or 0.0)
    angle_direction = str(lip_angle.get("direction", "center") or "center")
    thr_angle = THR.LIP_PUCKER_MIDLINE_ANGLE_THRESHOLD

    evidence_used = {
        "current_offset_px": current_offset,
        "offset_norm": offset_norm,
        "thr_offset_norm": thr_offset,
        "midline_angle_deg": angle,
        "angle_direction": angle_direction,
        "thr_angle_deg": thr_angle,
    }

    result = {
        "palsy_side": 0,
        "confidence": 0.0,
        "interpretation": "",
        "method": "lip_offset_weak",
        "evidence": evidence_used,
        "evidence_used": evidence_used,
        "evidence_dump": {
            "lip_midline_offset": lip_offset,
            "lip_midline_angle": lip_angle,
            "corner_contraction_asymmetry": metrics.get("corner_contraction_asymmetry", None),
            "left_corner_contraction": metrics.get("left_corner_contraction", None),
            "right_corner_contraction": metrics.get("right_corner_contraction", None),
        }
    }

    trig_offset = (offset_norm >= thr_offset)
    trig_angle = (angle >= thr_angle)

    if not (trig_offset or trig_angle):
        result["palsy_side"] = 0
        conf = max(offset_norm / max(thr_offset, 1e-6), angle / max(thr_angle, 1e-6))
        result["confidence"] = float(min(1.0, conf) * 0.6)
        result["interpretation"] = f"撅嘴有效，但中线偏移/角度不足以判侧别 → 对称"
        return result

    if current_offset != 0:
        if current_offset > 0:
            result["palsy_side"] = 2
            result["interpretation"] = f"嘴唇偏向患者左侧(offset={current_offset:+.1f}px, norm={offset_norm:.3f}) → 右侧面瘫(弱证据)"
        else:
            result["palsy_side"] = 1
            result["interpretation"] = f"嘴唇偏向患者右侧(offset={current_offset:+.1f}px, norm={offset_norm:.3f}) → 左侧面瘫(弱证据)"
    else:
        if angle_direction == "left":
            result["palsy_side"] = 1
            result["interpretation"] = f"嘴唇中线向患者右侧倾斜(angle={angle:.1f}°, dir=left) → 左侧面瘫(弱证据)"
        else:
            result["palsy_side"] = 2
            result["interpretation"] = f"嘴唇中线向患者左侧倾斜(angle={angle:.1f}°, dir=right) → 右侧面瘫(弱证据)"

    conf_raw = max(offset_norm / max(thr_offset, 1e-6), angle / max(thr_angle, 1e-6))
    result["confidence"] = float(min(1.0, conf_raw) * 0.6)
    return result


def compute_severity_score(metrics: Dict[str, Any]) -> Tuple[int, str]:
    """计算动作严重度分数"""
    corner_asymmetry = metrics.get("corner_contraction_asymmetry", 0)

    if corner_asymmetry > 0:
        if corner_asymmetry < 0.10:
            return 1, f"正常 (收缩不对称{corner_asymmetry:.2%})"
        elif corner_asymmetry < 0.20:
            return 2, f"轻度异常 (收缩不对称{corner_asymmetry:.2%})"
        elif corner_asymmetry < 0.35:
            return 3, f"中度异常 (收缩不对称{corner_asymmetry:.2%})"
        elif corner_asymmetry < 0.50:
            return 4, f"重度异常 (收缩不对称{corner_asymmetry:.2%})"
        else:
            return 5, f"完全面瘫 (收缩不对称{corner_asymmetry:.2%})"

    lip_offset_data = metrics.get("lip_midline_offset", {})
    offset_norm = lip_offset_data.get("offset_norm")
    if offset_norm is None:
        offset_norm = lip_offset_data.get("offset_change_norm", 0) or 0
    current_offset = lip_offset_data.get("current_offset", 0)

    if offset_norm < 0.03:
        return 1, f"正常 (偏移{offset_norm:.2%}, {current_offset:+.1f}px)"
    elif offset_norm < 0.06:
        return 2, f"轻度异常 (偏移{offset_norm:.2%})"
    elif offset_norm < 0.10:
        return 3, f"中度异常 (偏移{offset_norm:.2%})"
    elif offset_norm < 0.15:
        return 4, f"重度异常 (偏移{offset_norm:.2%})"
    else:
        return 5, f"完全面瘫 (偏移{offset_norm:.2%})"


def compute_voluntary_score(metrics: Dict[str, Any], baseline_landmarks=None) -> Tuple[int, str]:
    """计算Voluntary Movement评分"""
    corner_diff = metrics.get("corner_height_diff", 0)
    oral_asym = metrics.get("oral_angle", {}).get("asymmetry", 0)

    if baseline_landmarks is not None and "width_ratio" in metrics:
        width_ratio = metrics.get("width_ratio", 1.0)
        corner_contraction_asym = metrics.get("corner_contraction_asymmetry", 0)

        if width_ratio > 0.95:
            return 1, "几乎无撅嘴动作 (嘴宽变化<5%)"

        if corner_contraction_asym < 0.15:
            if width_ratio < 0.75:
                return 5, "运动完整 (嘴宽收缩>25%, 对称)"
            elif width_ratio < 0.85:
                return 4, "几乎完整 (嘴宽收缩>15%, 对称)"
            else:
                return 3, "启动运动 (嘴宽收缩>5%)"
        elif corner_contraction_asym < 0.30:
            return 3, f"运动不对称 (收缩不对称{corner_contraction_asym:.1%})"
        elif corner_contraction_asym < 0.50:
            return 2, f"轻微启动 (收缩不对称{corner_contraction_asym:.1%})"
        else:
            return 1, f"无法启动 (收缩不对称{corner_contraction_asym:.1%})"

    if oral_asym < 5:
        return 5, "运动完整 (口角对称)"
    elif oral_asym < 10:
        return 4, "几乎完整"
    elif oral_asym < 15:
        return 3, "启动但不对称"
    elif oral_asym < 25:
        return 2, "轻微启动"
    else:
        return 1, "无法启动"


def detect_synkinesis(baseline_result: Optional[ActionResult],
                      current_landmarks, w: int, h: int) -> Dict[str, int]:
    """检测撅嘴时的联动运动"""
    synkinesis = {"eye_synkinesis": 0, "brow_synkinesis": 0}

    if baseline_result is None:
        return synkinesis

    l_ear = compute_ear(current_landmarks, w, h, True)
    r_ear = compute_ear(current_landmarks, w, h, False)

    baseline_l_ear = baseline_result.left_ear
    baseline_r_ear = baseline_result.right_ear

    if baseline_l_ear > 1e-9 and baseline_r_ear > 1e-9:
        l_change = abs(l_ear - baseline_l_ear) / baseline_l_ear
        r_change = abs(r_ear - baseline_r_ear) / baseline_r_ear
        avg_change = (l_change + r_change) / 2

        if avg_change > 0.15:
            synkinesis["eye_synkinesis"] = 3
        elif avg_change > 0.08:
            synkinesis["eye_synkinesis"] = 2
        elif avg_change > 0.04:
            synkinesis["eye_synkinesis"] = 1

    return synkinesis


def visualize_lip_pucker(frame: np.ndarray, landmarks, w: int, h: int,
                         result: ActionResult,
                         metrics: Dict[str, Any],
                         palsy_detection: Dict[str, Any]) -> np.ndarray:
    """可视化撅嘴指标 - 包含新增的几何指标"""
    img = frame.copy()

    img, header_end_y = draw_palsy_annotation_header(img, palsy_detection, ACTION_NAME)

    midline = compute_face_midline(landmarks, w, h)
    if midline:
        img = draw_face_midline(img, midline, color=(0, 255, 255), thickness=2, dashed=True)

    draw_polygon(img, landmarks, w, h, LM.OUTER_LIP, (0, 255, 0), 2)
    draw_polygon(img, landmarks, w, h, LM.INNER_LIP, (0, 200, 200), 2)
    draw_polygon(img, landmarks, w, h, LM.INNER_LIP_L, (255, 100, 100), 2)
    draw_polygon(img, landmarks, w, h, LM.INNER_LIP_R, (100, 100, 255), 2)

    left_corner = metrics["left_corner"]
    right_corner = metrics["right_corner"]
    cv2.circle(img, (int(left_corner[0]), int(left_corner[1])), 6, (255, 0, 0), -1)
    cv2.circle(img, (int(right_corner[0]), int(right_corner[1])), 6, (0, 0, 255), -1)

    cv2.line(img, (int(left_corner[0]), int(left_corner[1])),
             (int(right_corner[0]), int(right_corner[1])), (0, 255, 255), 2)

    if result.oral_angle:
        oral = result.oral_angle
        cv2.line(img, (int(oral.E[0]), int(oral.E[1])),
                 (int(oral.F[0]), int(oral.F[1])), (0, 255, 0), 1)

    # ========== 绘制嘴唇中线 ==========
    lip_top = pt2d(landmarks[LM.LIP_TOP_CENTER], w, h)
    lip_bot = pt2d(landmarks[LM.LIP_BOT_CENTER], w, h)

    cv2.line(img, (int(lip_top[0]), int(lip_top[1])),
             (int(lip_bot[0]), int(lip_bot[1])), (255, 255, 0), 3)
    cv2.circle(img, (int(lip_top[0]), int(lip_top[1])), 5, (255, 255, 0), -1)
    cv2.circle(img, (int(lip_bot[0]), int(lip_bot[1])), 5, (255, 255, 0), -1)

    # ========== 绘制嘴角到嘴唇中线的垂直距离 ==========
    corner_to_lip = metrics.get("corner_to_lip_midline", {})
    if corner_to_lip:
        left_foot = corner_to_lip.get("left_foot")
        right_foot = corner_to_lip.get("right_foot")

        if left_foot:
            cv2.line(img, (int(left_corner[0]), int(left_corner[1])),
                     (int(left_foot[0]), int(left_foot[1])), (255, 100, 100), 2)
            cv2.circle(img, (int(left_foot[0]), int(left_foot[1])), 4, (255, 100, 100), -1)

        if right_foot:
            cv2.line(img, (int(right_corner[0]), int(right_corner[1])),
                     (int(right_foot[0]), int(right_foot[1])), (100, 100, 255), 2)
            cv2.circle(img, (int(right_foot[0]), int(right_foot[1])), 4, (100, 100, 255), -1)

    # ========== 绘制嘴唇中线到面中线的偏移 ==========
    lip_to_face = metrics.get("lip_midline_to_face_midline", {})
    if lip_to_face:
        lip_center = lip_to_face.get("lip_midline_center")
        foot = lip_to_face.get("foot_on_face_midline")
        if lip_center and foot:
            dist_val = lip_to_face.get("distance", 0)
            if dist_val > 3:
                cv2.line(img, (int(lip_center[0]), int(lip_center[1])),
                         (int(foot[0]), int(foot[1])), (0, 165, 255), 2)
                cv2.circle(img, (int(lip_center[0]), int(lip_center[1])), 5, (0, 165, 255), -1)

    # ========== 绘制嘴角到面中线的距离标注 ==========
    corner_to_face = metrics.get("corner_to_face_midline", {})
    if corner_to_face:
        left_dist = corner_to_face.get("left_distance", 0)
        right_dist = corner_to_face.get("right_distance", 0)
        cv2.putText(img, f"L:{left_dist:.0f}", (int(left_corner[0]) - 50, int(left_corner[1]) + 20),
                    FONT, 0.5, (255, 100, 100), 1)
        cv2.putText(img, f"R:{right_dist:.0f}", (int(right_corner[0]) + 10, int(right_corner[1]) + 20),
                    FONT, 0.5, (100, 100, 255), 1)

    # ========== 信息面板 ==========
    panel_top = header_end_y + 10
    panel_h = panel_top + 480
    cv2.rectangle(img, (5, panel_top), (480, panel_h), (0, 0, 0), -1)
    cv2.rectangle(img, (5, panel_top), (480, panel_h), (255, 255, 255), 1)

    y = panel_top + 35
    cv2.putText(img, f"{ACTION_NAME}", (15, y), FONT, FONT_SCALE_TITLE, (0, 255, 0), THICKNESS_TITLE)
    y += LINE_HEIGHT

    cv2.putText(img, "=== Mouth Metrics ===", (15, y), FONT, FONT_SCALE_NORMAL, (0, 255, 255), THICKNESS_NORMAL)
    y += LINE_HEIGHT_SMALL

    cv2.putText(img, f"Width: {metrics['mouth_width']:.1f}px  Height: {metrics['mouth_height']:.1f}px", (15, y),
                FONT, FONT_SCALE_SMALL, (255, 255, 255), THICKNESS_THIN)
    y += LINE_HEIGHT_SMALL

    # ========== 嘴角到嘴唇中线的距离 ==========
    if corner_to_lip:
        cv2.putText(img, "=== Corner to Lip Midline ===", (15, y), FONT, FONT_SCALE_SMALL, (0, 255, 255), THICKNESS_THIN)
        y += LINE_HEIGHT_SMALL

        left_dist_lip = corner_to_lip.get("left_distance", 0)
        right_dist_lip = corner_to_lip.get("right_distance", 0)
        lip_asym = corner_to_lip.get("asymmetry", 0)

        cv2.putText(img, f"L: {left_dist_lip:.1f}px  R: {right_dist_lip:.1f}px", (15, y),
                    FONT, FONT_SCALE_SMALL, (255, 255, 255), THICKNESS_THIN)
        y += LINE_HEIGHT_SMALL - 5

        asym_color = (0, 0, 255) if lip_asym > 0.15 else (0, 255, 0)
        cv2.putText(img, f"Asymmetry: {lip_asym:.1%}", (15, y), FONT, FONT_SCALE_SMALL, asym_color, THICKNESS_THIN)
        y += LINE_HEIGHT_SMALL

    # ========== 嘴唇中线偏离面中线的距离 ==========
    if lip_to_face:
        cv2.putText(img, "=== Lip to Face Midline ===", (15, y), FONT, FONT_SCALE_SMALL, (0, 255, 255), THICKNESS_THIN)
        y += LINE_HEIGHT_SMALL

        lip_face_dist = lip_to_face.get("distance", 0)
        lip_face_signed = lip_to_face.get("signed_distance", 0)
        lip_face_norm = lip_to_face.get("norm", 0)

        direction = "L" if lip_face_signed > 0 else "R" if lip_face_signed < 0 else "C"
        dist_color = (0, 0, 255) if lip_face_dist > 5 else (255, 255, 255)
        cv2.putText(img, f"Offset: {lip_face_dist:.1f}px ({direction}) ({lip_face_norm:.1%} ICD)", (15, y),
                    FONT, FONT_SCALE_SMALL, dist_color, THICKNESS_THIN)
        y += LINE_HEIGHT_SMALL

    # ========== 嘴角到面中线的距离 ==========
    if corner_to_face:
        cv2.putText(img, "=== Corner to Face Midline ===", (15, y), FONT, FONT_SCALE_SMALL, (0, 255, 255), THICKNESS_THIN)
        y += LINE_HEIGHT_SMALL

        left_dist_face = corner_to_face.get("left_distance", 0)
        right_dist_face = corner_to_face.get("right_distance", 0)
        face_asym = corner_to_face.get("asymmetry", 0)

        cv2.putText(img, f"L: {left_dist_face:.1f}px  R: {right_dist_face:.1f}px", (15, y),
                    FONT, FONT_SCALE_SMALL, (255, 255, 255), THICKNESS_THIN)
        y += LINE_HEIGHT_SMALL - 5

        asym_color = (0, 0, 255) if face_asym > 0.10 else (0, 255, 0)
        cv2.putText(img, f"Asymmetry: {face_asym:.1%}", (15, y), FONT, FONT_SCALE_SMALL, asym_color, THICKNESS_THIN)
        y += LINE_HEIGHT_SMALL

    # ========== 内缘面积 ==========
    inner_areas = metrics.get("inner_lip_areas", {})
    if inner_areas:
        cv2.putText(img, "=== Inner Lip Area ===", (15, y), FONT, FONT_SCALE_SMALL, (0, 255, 255), THICKNESS_THIN)
        y += LINE_HEIGHT_SMALL

        left_area = inner_areas.get("left", 0)
        right_area = inner_areas.get("right", 0)
        area_ratio = inner_areas.get("ratio", 1.0)

        cv2.putText(img, f"L: {left_area:.0f}  R: {right_area:.0f}  Ratio: {area_ratio:.2f}", (15, y),
                    FONT, FONT_SCALE_SMALL, (255, 255, 255), THICKNESS_THIN)
        y += LINE_HEIGHT_SMALL

    # 口角对称性
    oral_asym = metrics.get("oral_angle", {}).get("asymmetry", 0)
    asym_color = (0, 255, 0) if oral_asym < 5 else (0, 165, 255) if oral_asym < 10 else (0, 0, 255)
    cv2.putText(img, f"Oral Asymmetry: {oral_asym:.1f} deg", (15, y), FONT, FONT_SCALE_SMALL, asym_color, THICKNESS_THIN)
    y += LINE_HEIGHT

    # Voluntary Score
    cv2.putText(img, f"Voluntary Score: {result.voluntary_movement_score}/5", (15, y),
                FONT, FONT_SCALE_LARGE, (0, 255, 255), THICKNESS_TITLE)

    # ========== 绘制嘴唇中线偏移（原有的）==========
    if "lip_midline_offset" in metrics:
        offset_data = metrics["lip_midline_offset"]
        lip_x = offset_data.get("lip_midline_x")
        lip_y = offset_data.get("lip_midline_y")
        lip_center_proj = offset_data.get("lip_center_proj")
        current_signed_dist = offset_data.get("current_signed_dist", offset_data.get("current_offset", 0))

        if lip_x is not None and lip_y is not None:
            cv2.circle(img, (int(lip_x), int(lip_y)), 6, (0, 255, 0), -1)

            if lip_center_proj is not None:
                proj_x, proj_y = lip_center_proj
                dist_val = abs(current_signed_dist)
                offset_color = (0, 0, 255) if dist_val > 8 else (0, 165, 255)

                cv2.line(img, (int(lip_x), int(lip_y)), (int(proj_x), int(proj_y)), offset_color, 2)
                cv2.circle(img, (int(proj_x), int(proj_y)), 4, offset_color, -1)

                if dist_val > 3:
                    direction = "L" if current_signed_dist > 0 else "R"
                    mid_x = (int(lip_x) + int(proj_x)) // 2
                    mid_y = (int(lip_y) + int(proj_y)) // 2
                    cv2.putText(img, f"{direction}{dist_val:.0f}", (mid_x + 3, mid_y - 5),
                                FONT, FONT_SCALE_SMALL, offset_color, THICKNESS_THIN)

    # 嘴唇中线角度
    lip_angle = metrics.get("lip_midline_angle", {})
    angle_deg = lip_angle.get("angle", 0)
    direction = lip_angle.get("direction", "center")
    if angle_deg > 0.1:
        mid_x = (int(lip_top[0]) + int(lip_bot[0])) // 2
        mid_y = (int(lip_top[1]) + int(lip_bot[1])) // 2
        cv2.putText(img, f"Lip Line: {angle_deg:.1f}deg ({direction})",
                    (mid_x + 10, mid_y), FONT, 0.5, (255, 255, 0), 2)

    return img


def process(landmarks_seq: List, frames_seq: List, w: int, h: int,
            video_info: Dict[str, Any], output_dir: Path,
            baseline_result: Optional[ActionResult] = None,
            baseline_landmarks=None) -> Optional[ActionResult]:
    """处理LipPucker动作"""
    if not landmarks_seq or not frames_seq:
        return None

    peak_idx, peak_debug = find_peak_frame(landmarks_seq, frames_seq, w, h, baseline_landmarks)
    peak_landmarks = landmarks_seq[peak_idx]
    peak_frame = frames_seq[peak_idx]

    if peak_landmarks is None:
        return None

    result = ActionResult(
        action_name=ACTION_NAME,
        action_name_cn=ACTION_NAME_CN,
        video_path=video_info.get("file_path", ""),
        total_frames=len(frames_seq),
        peak_frame_idx=peak_idx,
        image_size=(w, h),
        fps=video_info.get("fps", 30.0)
    )

    extract_common_indicators(peak_landmarks, w, h, result, baseline_landmarks)
    metrics = compute_lip_pucker_metrics(peak_landmarks, w, h, baseline_landmarks)
    palsy_detection = detect_palsy_side(metrics)
    score, interpretation = compute_voluntary_score(metrics, baseline_landmarks)
    result.voluntary_movement_score = score
    synkinesis = detect_synkinesis(baseline_result, peak_landmarks, w, h)
    result.synkinesis_scores = synkinesis
    severity_score, severity_desc = compute_severity_score(metrics)

    result.action_specific = {
        "mouth_metrics": {"width": metrics["mouth_width"], "height": metrics["mouth_height"]},
        "oral_angle": metrics["oral_angle"],
        "palsy_detection": palsy_detection,
        "voluntary_interpretation": interpretation,
        "synkinesis": synkinesis,
        "severity_score": severity_score,
        "severity_desc": severity_desc,
        "peak_debug": peak_debug,
        # ========== 新增指标输出 ==========
        "corner_to_lip_midline": metrics.get("corner_to_lip_midline", {}),
        "lip_midline_to_face_midline": metrics.get("lip_midline_to_face_midline", {}),
        "corner_to_face_midline": metrics.get("corner_to_face_midline", {}),
    }

    if "baseline" in metrics:
        result.action_specific["baseline"] = metrics["baseline"]
    if "movement" in metrics:
        result.action_specific["movement"] = metrics["movement"]

    action_dir = output_dir / ACTION_NAME
    action_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(action_dir / "peak_raw.jpg"), peak_frame)

    vis = visualize_lip_pucker(peak_frame, peak_landmarks, w, h, result, metrics, palsy_detection)
    cv2.imwrite(str(action_dir / "peak_indicators.jpg"), vis)

    if peak_debug:
        plot_lip_pucker_peak_selection(
            peak_debug,
            video_info.get("fps", 30.0),
            action_dir / "peak_selection_curve.png",
            palsy_detection
        )

    with open(action_dir / "indicators.json", 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    # 打印信息
    corner_to_lip = metrics.get("corner_to_lip_midline", {})
    corner_to_face = metrics.get("corner_to_face_midline", {})
    lip_to_face = metrics.get("lip_midline_to_face_midline", {})

    print(f"    [OK] {ACTION_NAME}: Width={metrics['mouth_width']:.1f}px")
    if corner_to_lip:
        print(f"         Corner→LipMidline: L={corner_to_lip.get('left_distance', 0):.1f}px "
              f"R={corner_to_lip.get('right_distance', 0):.1f}px "
              f"Asym={corner_to_lip.get('asymmetry', 0):.1%}")
    if lip_to_face:
        print(f"         LipMidline→FaceMidline: {lip_to_face.get('distance', 0):.1f}px "
              f"({lip_to_face.get('norm', 0):.1%} ICD)")
    if corner_to_face:
        print(f"         Corner→FaceMidline: L={corner_to_face.get('left_distance', 0):.1f}px "
              f"R={corner_to_face.get('right_distance', 0):.1f}px "
              f"Asym={corner_to_face.get('asymmetry', 0):.1%}")
    print(f"         Palsy: {palsy_detection.get('interpretation', 'N/A')}")
    print(f"         Voluntary Score: {result.voluntary_movement_score}/5 ({interpretation})")

    return result