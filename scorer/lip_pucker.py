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

修复内容:
- 移除错误的NLF分析
- 使用口角角度和嘴部收缩作为主要指标

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
    LM, pt2d, pt3d, pts2d, dist, compute_ear, compute_eye_area,
    compute_mouth_metrics, compute_oral_angle,
    compute_icd, extract_common_indicators,
    ActionResult, draw_polygon, compute_scale_to_baseline,
    kabsch_rigid_transform, apply_rigid_transform,
    compute_lip_midline_symmetry,
    compute_lip_midline_offset_from_face_midline,
    compute_lip_midline_center, compute_face_midline, draw_face_midline,
)

from thresholds import THR

ACTION_NAME = "LipPucker"
ACTION_NAME_CN = "撅嘴"


def _mean_lip_z_aligned(landmarks, w: int, h: int, baseline_landmarks=None) -> float:
    """
    计算“唇部区域平均z”（更稳：先把当前帧刚体对齐到 baseline，再取 z）。
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
    选择标准：嘴唇前突（protrusion）和嘴宽收缩（width ratio）的综合分数。
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

    # 上图：Protrusion 和 Score
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

    # 下图：Width Ratio
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


def compute_lip_midline_angle(landmarks, w: int, h: int) -> Dict[str, Any]:
    """
    计算嘴唇中线与面中线的角度

    原理:
    - 面中线: 鼻尖到下巴中点的连线方向 (垂直向下)
    - 嘴唇中线: 上唇中点到下唇中点的连线方向
    - 正常情况两线接近平行，面瘫时形成角度

    Args:
        landmarks: 面部关键点
        w, h: 图像尺寸

    Returns:
        {
            "angle": 夹角(度),
            "direction": 偏向 ("left", "right", "center"),
            "cross_product": 叉积值 (用于判断方向),
            "face_vec": 面中线向量,
            "lip_vec": 嘴唇中线向量,
        }
    """
    result = {
        "angle": 0.0,
        "direction": "center",
        "cross_product": 0.0,
        "face_vec": (0, 1),
        "lip_vec": (0, 1),
    }

    try:
        # 面中线: 鼻尖 -> 下巴中点 (归一化方向)
        nose_tip = pt2d(landmarks[LM.NOSE_TIP], w, h)

        # 获取下巴点 (LM.CHIN = 152)
        chin = pt2d(landmarks[LM.CHIN], w, h)

        face_vec = (chin[0] - nose_tip[0], chin[1] - nose_tip[1])
        result["face_vec"] = face_vec

        # 嘴唇中线: 上唇中点 -> 下唇中点
        lip_top = pt2d(landmarks[LM.LIP_TOP_CENTER], w, h)
        lip_bot = pt2d(landmarks[LM.LIP_BOT_CENTER], w, h)
        lip_vec = (lip_bot[0] - lip_top[0], lip_bot[1] - lip_top[1])
        result["lip_vec"] = lip_vec

        # 计算向量模
        def magnitude(v):
            return math.sqrt(v[0] ** 2 + v[1] ** 2)

        mag_face = magnitude(face_vec)
        mag_lip = magnitude(lip_vec)

        if mag_face < 1e-6 or mag_lip < 1e-6:
            return result

        # 计算夹角 (点积)
        dot = face_vec[0] * lip_vec[0] + face_vec[1] * lip_vec[1]
        cos_angle = dot / (mag_face * mag_lip)
        cos_angle = max(-1.0, min(1.0, cos_angle))  # 数值稳定
        angle = math.degrees(math.acos(cos_angle))
        result["angle"] = angle

        # 判断偏向 (叉积)
        # cross = face_x * lip_y - face_y * lip_x
        # cross > 0: lip_vec 在 face_vec 逆时针方向 (嘴唇偏左)
        # cross < 0: lip_vec 在 face_vec 顺时针方向 (嘴唇偏右)
        cross = face_vec[0] * lip_vec[1] - face_vec[1] * lip_vec[0]
        result["cross_product"] = cross

        # 设置方向阈值 (避免噪声)
        if abs(cross) < 1e-3:
            result["direction"] = "center"
        elif cross > 0:
            result["direction"] = "left"  # 嘴唇偏向左侧
        else:
            result["direction"] = "right"  # 嘴唇偏向右侧

    except Exception as e:
        result["error"] = str(e)

    return result


def find_peak_frame(landmarks_seq: List, frames_seq: List, w: int, h: int,
                    baseline_landmarks=None) -> Tuple[int, Dict[str, Any]]:
    """
    撅嘴峰值帧选择 - 使用嘴唇相对鼻尖的深度变化

    原理：
    - rel_z = mean_z(lip_region) - z(nose_tip)
    - 撅嘴时嘴唇前突，rel_z 变小（更靠近相机）
    - protrusion = (base_rel_z - current_rel_z) / ICD
    - 同时考虑嘴宽收缩（撅嘴时嘴变窄）

    内部baseline：用视频前 BASELINE_FRAMES 帧的中位数

    Args:
        landmarks_seq: 关键点序列
        frames_seq: 帧图像序列（未使用，保留接口兼容）
        w, h: 图像尺寸
        baseline_landmarks: 静息帧关键点（本方法不需要，但保留接口兼容）

    Returns:
        (peak_idx, peak_debug): 峰值帧索引和调试信息
    """
    n = len(landmarks_seq)
    if n == 0:
        return 0, {"error": "empty_sequence"}

    # 配置参数
    BASELINE_FRAMES = getattr(THR, 'LIP_PUCKER_BASELINE_FRAMES', 10)
    MOUTH_H_THR = THR.MOUTH_HEIGHT

    # ========== 辅助函数 ==========
    def _safe_pt3d(landmarks, idx: int):
        """安全获取3D点"""
        try:
            if landmarks is None or idx < 0 or idx >= len(landmarks):
                return None
            return np.asarray(pt3d(landmarks[idx], w, h), dtype=np.float64)
        except Exception:
            return None

    def _get_nose_z(landmarks) -> float:
        """获取鼻尖Z坐标"""
        p = _safe_pt3d(landmarks, LM.NOSE_TIP)
        return float(p[2]) if p is not None else np.nan

    def _get_lip_mean_z(landmarks) -> float:
        """获取嘴唇区域平均Z坐标"""
        lip_indices = list(LM.OUTER_LIP) + list(LM.INNER_LIP)
        zs = []
        for idx in lip_indices:
            p = _safe_pt3d(landmarks, idx)
            if p is not None:
                zs.append(float(p[2]))
        return float(np.mean(zs)) if zs else np.nan

    # ========== 第一遍扫描：收集所有帧的指标 ==========
    icd_arr = np.full(n, np.nan, dtype=np.float64)
    nose_z_arr = np.full(n, np.nan, dtype=np.float64)
    lip_z_arr = np.full(n, np.nan, dtype=np.float64)
    lip_rel_z_arr = np.full(n, np.nan, dtype=np.float64)
    mouth_width_arr = np.full(n, np.nan, dtype=np.float64)
    mouth_height_arr = np.full(n, np.nan, dtype=np.float64)

    for i, lm in enumerate(landmarks_seq):
        if lm is None:
            continue

        try:
            icd = float(max(compute_icd(lm, w, h), 1e-6))
        except Exception:
            continue

        icd_arr[i] = icd

        # 鼻尖和嘴唇Z坐标
        nose_z = _get_nose_z(lm)
        lip_z = _get_lip_mean_z(lm)

        nose_z_arr[i] = nose_z
        lip_z_arr[i] = lip_z

        # 相对鼻尖的深度
        if np.isfinite(nose_z) and np.isfinite(lip_z):
            lip_rel_z_arr[i] = lip_z - nose_z

        # 嘴部指标
        try:
            mouth = compute_mouth_metrics(lm, w, h)
            mouth_width_arr[i] = float(mouth["width"])
            mouth_height_arr[i] = float(mouth["height"])
        except Exception:
            pass

    # ========== 建立视频内部baseline ==========
    valid_init_mask = (np.isfinite(lip_rel_z_arr[:BASELINE_FRAMES]) &
                       np.isfinite(icd_arr[:BASELINE_FRAMES]) &
                       np.isfinite(mouth_width_arr[:BASELINE_FRAMES]))
    valid_init_idx = np.where(valid_init_mask)[0]

    if len(valid_init_idx) < 3:
        all_valid = (np.isfinite(lip_rel_z_arr) &
                     np.isfinite(icd_arr) &
                     np.isfinite(mouth_width_arr))
        valid_init_idx = np.where(all_valid)[0]

    if len(valid_init_idx) == 0:
        return 0, {"error": "no_valid_frames", "method": "nose_relative_protrusion"}

    base_lip_rel_z = float(np.median(lip_rel_z_arr[valid_init_idx]))
    base_icd = float(np.median(icd_arr[valid_init_idx]))
    base_mouth_width = float(np.median(mouth_width_arr[valid_init_idx]))

    # ========== 计算 protrusion 曲线 ==========
    protrusion_arr = np.full(n, np.nan, dtype=np.float64)
    width_ratio_arr = np.full(n, np.nan, dtype=np.float64)
    score_arr = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        # protrusion = (base_rel - current_rel) / ICD
        # 撅嘴时嘴唇前突，current_rel 变小，protrusion 变大
        if np.isfinite(lip_rel_z_arr[i]):
            protrusion_arr[i] = (base_lip_rel_z - lip_rel_z_arr[i]) / base_icd

        # 嘴宽比（撅嘴时嘴变窄）
        if np.isfinite(mouth_width_arr[i]) and base_mouth_width > 0:
            width_ratio_arr[i] = mouth_width_arr[i] / base_mouth_width

        # 综合评分：前突为主，嘴宽收缩为辅
        if np.isfinite(protrusion_arr[i]):
            score_arr[i] = protrusion_arr[i]
            # 加分：嘴变窄（width_ratio < 1）
            if np.isfinite(width_ratio_arr[i]) and width_ratio_arr[i] < 1.0:
                width_bonus = (1.0 - width_ratio_arr[i]) * 0.5
                score_arr[i] += width_bonus

    # ========== 门控：闭唇 ==========
    mouth_h_norm = mouth_height_arr / icd_arr

    valid = (np.isfinite(score_arr) & (mouth_h_norm <= MOUTH_H_THR))

    # 退化策略
    if valid.sum() < 3:
        valid = np.isfinite(score_arr) & (mouth_h_norm <= MOUTH_H_THR * 1.5)

    if valid.sum() < 1:
        valid = np.isfinite(score_arr)

    # 选峰值帧
    fallback = None
    if valid.sum() < 1:
        fallback = "no_valid_fallback"
        peak_idx = int(np.nanargmax(score_arr)) if np.any(np.isfinite(score_arr)) else 0
    else:
        cand = np.where(valid)[0]
        peak_idx = int(cand[int(np.nanargmax(score_arr[cand]))])

    # ========== 调试信息 ==========
    peak_debug = {
        # 原始Z坐标
        "nose_z": [float(v) if np.isfinite(v) else None for v in nose_z_arr],
        "lip_z": [float(v) if np.isfinite(v) else None for v in lip_z_arr],

        # 相对鼻尖的深度
        "lip_rel_z": [float(v) if np.isfinite(v) else None for v in lip_rel_z_arr],

        # protrusion 曲线
        "protrusion": [float(v) if np.isfinite(v) else None for v in protrusion_arr],

        # 嘴宽比
        "width_ratio": [float(v) if np.isfinite(v) else None for v in width_ratio_arr],

        # 综合评分
        "score": [float(v) if np.isfinite(v) else None for v in score_arr],

        # 门控曲线
        "mouth_height_norm": [float(v) if np.isfinite(v) else None for v in mouth_h_norm],

        # baseline信息
        "baseline": {
            "frames_used": int(len(valid_init_idx)),
            "lip_rel_z": float(base_lip_rel_z),
            "icd": float(base_icd),
            "mouth_width": float(base_mouth_width),
        },

        # 阈值
        "thresholds": {
            "mouth_h_thr": float(MOUTH_H_THR),
        },

        # 选帧结果
        "valid": [bool(v) for v in valid],
        "peak_idx": int(peak_idx),
        "fallback": fallback,
        "method": "nose_relative_protrusion",
    }

    return peak_idx, peak_debug


def compute_lip_pucker_metrics(landmarks, w: int, h: int,
                               baseline_landmarks=None) -> Dict[str, Any]:
    """
    计算撅嘴特有指标 - 增加面中线对称性和运动变化量
    """
    mouth = compute_mouth_metrics(landmarks, w, h)
    oral = compute_oral_angle(landmarks, w, h)

    # 嘴角位置
    left_corner = mouth["left_corner"]
    right_corner = mouth["right_corner"]

    # 面中线参考
    left_canthus = pt2d(landmarks[LM.EYE_INNER_L], w, h)
    right_canthus = pt2d(landmarks[LM.EYE_INNER_R], w, h)
    midline_x = (left_canthus[0] + right_canthus[0]) / 2

    # 左右嘴角到面中线的距离（保留用于参考）
    left_to_midline = abs(midline_x - left_corner[0])
    right_to_midline = abs(right_corner[0] - midline_x)

    # 嘴唇中心相对于面中线的偏移
    mouth_center_x = (left_corner[0] + right_corner[0]) / 2
    mouth_midline_offset = mouth_center_x - midline_x

    # 嘴角高度对称性
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

    # ========== 嘴唇面中线对称性（用于面瘫侧别判断）==========
    lip_symmetry = compute_lip_midline_symmetry(landmarks, w, h)
    metrics["lip_symmetry"] = lip_symmetry

    # 基线参考
    if baseline_landmarks is not None:
        scale = compute_scale_to_baseline(landmarks, baseline_landmarks, w, h)
        metrics["scale"] = scale

        baseline_mouth = compute_mouth_metrics(baseline_landmarks, w, h)
        baseline_oral = compute_oral_angle(baseline_landmarks, w, h)
        baseline_left = baseline_mouth["left_corner"]
        baseline_right = baseline_mouth["right_corner"]

        # 口角角度变化（保留用于参考）
        aoe_change = oral.AOE_angle - baseline_oral.AOE_angle
        bof_change = oral.BOF_angle - baseline_oral.BOF_angle

        # 嘴宽变化
        scaled_width = mouth["width"] * scale
        width_change = scaled_width - baseline_mouth["width"]

        metrics["baseline"] = {
            "mouth_width": float(baseline_mouth["width"]),
            "AOE": float(baseline_oral.AOE_angle),
            "BOF": float(baseline_oral.BOF_angle),
        }

        metrics["movement"] = {
            "aoe_change": float(aoe_change),
            "bof_change": float(bof_change),
        }

        metrics["width_change"] = float(width_change)
        metrics["width_ratio"] = float(scaled_width / baseline_mouth["width"]) if baseline_mouth["width"] > 1e-9 else 1.0

        # 保留原有的百分比计算
        if baseline_mouth["width"] > 1e-9:
            metrics["width_change_percent"] = width_change / baseline_mouth["width"] * 100
        else:
            metrics["width_change_percent"] = 0

        # ========== 计算嘴唇中线相对于面中线的偏移变化 ==========
        lip_offset_data = compute_lip_midline_offset_from_face_midline(landmarks, w, h, baseline_landmarks)
        metrics["lip_midline_offset"] = lip_offset_data

        # 嘴唇中心点
        lip_center = compute_lip_midline_center(landmarks, w, h)
        baseline_lip_center = compute_lip_midline_center(baseline_landmarks, w, h)

        # 当前嘴角位置
        left_corner = pt2d(landmarks[LM.MOUTH_CORNER_L], w, h)
        right_corner = pt2d(landmarks[LM.MOUTH_CORNER_R], w, h)

        # 基线嘴角位置
        baseline_left_corner = pt2d(baseline_landmarks[LM.MOUTH_CORNER_L], w, h)
        baseline_right_corner = pt2d(baseline_landmarks[LM.MOUTH_CORNER_R], w, h)

        # 计算嘴角到嘴唇中心的距离变化（收缩量）
        # 撅嘴时距离应该减小，所以是 baseline - current
        def dist(p1, p2):
            return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        baseline_left_dist = dist(baseline_left_corner, baseline_lip_center)
        baseline_right_dist = dist(baseline_right_corner, baseline_lip_center)
        current_left_dist = dist(left_corner, lip_center)
        current_right_dist = dist(right_corner, lip_center)

        left_contraction = baseline_left_dist - current_left_dist
        right_contraction = baseline_right_dist - current_right_dist

        metrics["left_corner_contraction"] = left_contraction
        metrics["right_corner_contraction"] = right_contraction

        # 计算不对称度
        max_contraction = max(abs(left_contraction), abs(right_contraction))
        if max_contraction > 1:
            metrics["corner_contraction_asymmetry"] = abs(left_contraction - right_contraction) / max_contraction
        else:
            metrics["corner_contraction_asymmetry"] = 0

        lip_angle = compute_lip_midline_angle(landmarks, w, h)
        metrics["lip_midline_angle"] = lip_angle

    return metrics


def detect_palsy_side(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    从撅嘴动作检测面瘫侧别

    判断逻辑:
    1. 嘴角收缩量比较: 收缩量小的一侧是患侧
    2. 嘴唇中线角度: 嘴唇偏向X侧 → X对侧是患侧
    3. 嘴唇中心偏移: 嘴唇偏向X侧 → X对侧是患侧
    4. 综合加权投票
    """
    result = {
        "palsy_side": 0,
        "confidence": 0.0,
        "interpretation": "",
        "method": "",
        "evidence": {}
    }

    votes = {0: 0.0, 1: 0.0, 2: 0.0}
    evidence = {}
    reasons = {1: [], 2: []}

    # ========== 方法1: 嘴角收缩量比较 ==========
    left_contraction = metrics.get("left_corner_contraction", 0)
    right_contraction = metrics.get("right_corner_contraction", 0)
    max_contraction = max(abs(left_contraction), abs(right_contraction))

    evidence["left_contraction"] = left_contraction
    evidence["right_contraction"] = right_contraction

    if max_contraction >= THR.LIP_PUCKER_PALSY_MIN_CONTRACTION:
        asymmetry = abs(left_contraction - right_contraction) / max_contraction
        evidence["contraction_asymmetry"] = asymmetry

        if asymmetry >= THR.LIP_PUCKER_PALSY_ASYMMETRY_THRESHOLD:
            if left_contraction < right_contraction:
                votes[1] += THR.LIP_PUCKER_CONTRACTION_WEIGHT
                reasons[1].append(f"L收缩弱{left_contraction:.1f}px")
                evidence["contraction_vote"] = 1
            else:
                votes[2] += THR.LIP_PUCKER_CONTRACTION_WEIGHT
                reasons[2].append(f"R收缩弱{right_contraction:.1f}px")
                evidence["contraction_vote"] = 2
            evidence["contraction_triggered"] = True

    # ========== 方法2: 嘴唇中线角度 ==========
    lip_angle = metrics.get("lip_midline_angle", {})
    angle = lip_angle.get("angle", 0)
    direction = lip_angle.get("direction", "center")

    evidence["midline_angle"] = angle
    evidence["angle_direction"] = direction

    if angle >= THR.LIP_PUCKER_MIDLINE_ANGLE_THRESHOLD:
        if direction == "left":
            votes[2] += THR.LIP_PUCKER_ANGLE_WEIGHT
            reasons[2].append(f"偏左{angle:.1f}°")
            evidence["angle_vote"] = 2
        elif direction == "right":
            votes[1] += THR.LIP_PUCKER_ANGLE_WEIGHT
            reasons[1].append(f"偏右{angle:.1f}°")
            evidence["angle_vote"] = 1
        evidence["angle_triggered"] = True

    # ========== 方法3: 嘴唇中心偏移 ==========
    lip_offset = metrics.get("lip_midline_offset", {})
    offset_norm = abs(lip_offset.get("offset_norm", 0) or 0)
    current_offset = lip_offset.get("current_offset", 0) or 0

    evidence["offset_norm"] = offset_norm
    evidence["current_offset"] = current_offset

    if offset_norm >= THR.LIP_PUCKER_PALSY_OFFSET_THRESHOLD:
        if current_offset > 0:
            votes[2] += THR.LIP_PUCKER_OFFSET_WEIGHT
            reasons[2].append(f"偏左{current_offset:+.1f}px")
            evidence["offset_vote"] = 2
        elif current_offset < 0:
            votes[1] += THR.LIP_PUCKER_OFFSET_WEIGHT
            reasons[1].append(f"偏右{abs(current_offset):.1f}px")
            evidence["offset_vote"] = 1
        evidence["offset_triggered"] = True

    # ========== 汇总 ==========
    evidence["votes"] = {"left": votes[1], "right": votes[2]}
    result["evidence"] = evidence

    min_conf = 0.12
    if votes[1] >= min_conf and votes[1] >= votes[2]:
        result["palsy_side"] = 1
        result["confidence"] = min(1.0, votes[1])
        result["interpretation"] = "左侧面瘫: " + ", ".join(reasons[1])
    elif votes[2] >= min_conf:
        result["palsy_side"] = 2
        result["confidence"] = min(1.0, votes[2])
        result["interpretation"] = "右侧面瘫: " + ", ".join(reasons[2])
    else:
        asym = evidence.get("contraction_asymmetry", 0)
        result["interpretation"] = f"对称 (收缩差{asym:.1%}, 角度{angle:.1f}°)"

    return result


def compute_severity_score(metrics: Dict[str, Any]) -> Tuple[int, str]:
    """
    计算动作严重度分数(医生标注标准)

    计算依据: 嘴角收缩的不对称程度
    """
    # 优先使用嘴角收缩不对称度
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

    # 回退到lip_midline_offset
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
    """
    计算Voluntary Movement评分

    基于嘴唇收缩程度和对称性

    评分标准:
    - 5=完整: 双侧对称且收缩明显
    - 4=几乎完整: 轻度不对称或收缩略有不足
    - 3=启动但不对称: 明显不对称但有运动
    - 2=轻微启动: 运动幅度很小
    - 1=无法启动: 几乎没有运动
    """
    # 检查口角对称性
    corner_diff = metrics.get("corner_height_diff", 0)
    oral_asym = metrics.get("oral_angle", {}).get("asymmetry", 0)

    if baseline_landmarks is not None and "width_ratio" in metrics:
        width_ratio = metrics["width_ratio"]

        # 撅嘴时宽度应该显著减小
        if width_ratio > 0.95:
            return 1, "无法启动运动 (宽度几乎无变化)"

        # 结合对称性评分
        if oral_asym < 3 and corner_diff < 3:
            if width_ratio < 0.70:
                return 5, "运动完整 (收缩明显且对称)"
            elif width_ratio < 0.80:
                return 4, "几乎完整"
            else:
                return 3, "启动但幅度不足"
        elif oral_asym < 6 and corner_diff < 6:
            if width_ratio < 0.75:
                return 4, "几乎完整 (轻度不对称)"
            else:
                return 3, "启动但不对称"
        elif oral_asym < 10:
            return 2, "轻微启动 (明显不对称)"
        else:
            return 1, "无法启动 (严重不对称)"
    else:
        # 没有基线，使用静态对称性
        if oral_asym < 3:
            return 5, "运动完整"
        elif oral_asym < 6:
            return 4, "几乎完整"
        elif oral_asym < 10:
            return 3, "启动但不对称"
        elif oral_asym < 15:
            return 2, "轻微启动"
        else:
            return 1, "无法启动"


def detect_synkinesis(baseline_result: Optional[ActionResult],
                      current_landmarks, w: int, h: int) -> Dict[str, int]:
    """检测撅嘴时的联动运动"""
    synkinesis = {
        "eye_synkinesis": 0,
        "brow_synkinesis": 0,
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
    """可视化撅嘴指标"""
    img = frame.copy()

    # ========== 添加面中线绘制 ==========
    midline = compute_face_midline(landmarks, w, h)
    if midline:
        img = draw_face_midline(img, midline, color=(0, 255, 255), thickness=2, dashed=True)

    # 绘制嘴部轮廓
    draw_polygon(img, landmarks, w, h, LM.OUTER_LIP, (0, 255, 0), 2)
    draw_polygon(img, landmarks, w, h, LM.INNER_LIP, (0, 200, 200), 1)

    # 绘制嘴角点
    left_corner = metrics["left_corner"]
    right_corner = metrics["right_corner"]
    cv2.circle(img, (int(left_corner[0]), int(left_corner[1])), 5, (255, 0, 0), -1)
    cv2.circle(img, (int(right_corner[0]), int(right_corner[1])), 5, (0, 0, 255), -1)

    # 绘制嘴宽测量线
    cv2.line(img, (int(left_corner[0]), int(left_corner[1])),
             (int(right_corner[0]), int(right_corner[1])), (0, 255, 255), 2)

    # 绘制口角角度参考线
    if result.oral_angle:
        oral = result.oral_angle
        cv2.line(img, (int(oral.E[0]), int(oral.E[1])),
                 (int(oral.F[0]), int(oral.F[1])), (0, 255, 0), 1)

    # 信息面板
    panel_h = 300
    cv2.rectangle(img, (5, 5), (380, panel_h), (0, 0, 0), -1)
    cv2.rectangle(img, (5, 5), (380, panel_h), (255, 255, 255), 1)

    y = 28
    cv2.putText(img, f"{ACTION_NAME}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y += 28

    cv2.putText(img, "=== Mouth Metrics ===", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    y += 20

    cv2.putText(img, f"Width: {metrics['mouth_width']:.1f}px", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    y += 18

    cv2.putText(img, f"Height: {metrics['mouth_height']:.1f}px", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    y += 18

    cv2.putText(img, f"W/H Ratio: {metrics['width_height_ratio']:.2f}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    y += 22

    if "width_change" in metrics:
        cv2.putText(img, "=== Changes from Baseline ===", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        y += 20

        cv2.putText(img,
                    f"Width Change: {metrics['width_change']:+.1f}px ({metrics.get('width_change_percent', 0):+.1f}%)",
                    (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 18

        cv2.putText(img, f"Width Ratio: {metrics.get('width_ratio', 1):.3f}", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 22

    # 口角对称性
    oral_asym = metrics.get("oral_angle", {}).get("asymmetry", 0)
    asym_color = (0, 255, 0) if oral_asym < 5 else (0, 165, 255) if oral_asym < 10 else (0, 0, 255)
    cv2.putText(img, f"Oral Asymmetry: {oral_asym:.1f} deg", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, asym_color, 1)
    y += 22

    # ========== 绘制嘴唇中线偏移 ==========
    if "lip_midline_offset" in metrics:
        offset_data = metrics["lip_midline_offset"]
        lip_x = offset_data.get("lip_midline_x")
        lip_y = offset_data.get("lip_midline_y")
        lip_center_proj = offset_data.get("lip_center_proj")
        current_signed_dist = offset_data.get("current_signed_dist",
                                              offset_data.get("current_offset", 0))

        if lip_x is not None and lip_y is not None:
            # 绘制嘴唇中心点（绿色）
            cv2.circle(img, (int(lip_x), int(lip_y)), 6, (0, 255, 0), -1)

            if lip_center_proj is not None:
                proj_x, proj_y = lip_center_proj
                dist = abs(current_signed_dist)
                offset_color = (0, 0, 255) if dist > 8 else (0, 165, 255)

                # 画垂线
                cv2.line(img, (int(lip_x), int(lip_y)),
                         (int(proj_x), int(proj_y)), offset_color, 2)
                cv2.circle(img, (int(proj_x), int(proj_y)), 4, offset_color, -1)

                # 标注
                if dist > 3:
                    direction = "L" if current_signed_dist > 0 else "R"
                    mid_x = (int(lip_x) + int(proj_x)) // 2
                    mid_y = (int(lip_y) + int(proj_y)) // 2
                    cv2.putText(img, f"{direction}{dist:.0f}",
                                (mid_x + 3, mid_y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, offset_color, 1)

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

    # 调用绘图函数
    if peak_debug:
        plot_lip_pucker_peak_selection(
            peak_debug,
            video_info.get("fps", 30.0),
            action_dir / "peak_selection_curve.png",
            palsy_detection
        )

    with open(action_dir / "indicators.json", 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"    [OK] {ACTION_NAME}: Width={metrics['mouth_width']:.1f}px")
    print(f"         Palsy: {palsy_detection.get('interpretation', 'N/A')}")
    print(f"         Voluntary Score: {result.voluntary_movement_score}/5 ({interpretation})")

    return result
