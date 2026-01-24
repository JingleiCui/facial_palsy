#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BlowCheek 动作处理模块 (修改版)
================================

核心改进:
=========
1. 面瘫侧别判断：使用嘴唇中线偏移面中线的距离
   - 鼓腮时，健侧脸颊能正常鼓起，会把嘴唇挤向患侧
   - 嘴唇偏向哪侧，哪侧就是患侧

2. 深度计算方式改为直观定义:
   - depth = nose_z - cheek_z (鼻尖z减去脸颊z)
   - 负值: 脸颊在鼻尖后面（还没鼓起）
   - 正值: 脸颊超过鼻尖（已经鼓起）
   - 值越大: 鼓腮越多

3. 不再依赖前10帧作为baseline:
   - 直接计算每帧的深度值
   - 峰值帧 = 深度值最大的帧
   - 避免"视频全程在鼓腮"的问题

4. 归一化到ICD，使不同患者可比较

分析内容:
1. 唇密封距离
2. 嘴部闭合程度
3. 口角对称性
4. 面瘫侧别检测
5. 联动运动检测

关键帧检测方法:
- 脸颊相对鼻尖深度最大的帧（鼓腮最明显）
"""

import cv2
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import numpy as np

from thresholds import THR

from clinical_base import (
    LM, pt2d, pt3d, pts2d, dist, compute_ear, compute_eye_area,
    compute_mouth_metrics, compute_oral_angle,
    compute_icd, polygon_area, extract_common_indicators,
    ActionResult, draw_polygon, draw_landmarks,
    compute_scale_to_baseline,
    kabsch_rigid_transform, apply_rigid_transform,
    add_valid_region_shading, get_palsy_side_text,
    draw_palsy_side_label, compute_lip_midline_symmetry,
    compute_lip_midline_offset_from_face_midline,
    compute_face_midline, draw_face_midline, compute_lip_shape_symmetry,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ACTION_NAME = "BlowCheek"
ACTION_NAME_CN = "鼓腮"

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


def _safe_pt3d(landmarks, idx: int, w: int, h: int):
    """安全获取3D坐标"""
    try:
        p = np.array(pt3d(landmarks[idx], w, h), dtype=np.float64)  # (x_px, y_px, z_px)
        if np.any(~np.isfinite(p)):
            return None
        return p
    except Exception:
        return None


def compute_cheek_depth(landmarks, w: int, h: int) -> Dict[str, Any]:
    """
    计算脸颊相对于鼻尖的深度 (直观定义)

    核心公式:
    =========
    depth = nose_z - cheek_z

    解释:
    - MediaPipe z坐标: z越小(更负) = 越靠近镜头
    - nose_z - cheek_z:
      - 静息时: 鼻尖靠近镜头(z小), 脸颊远离镜头(z大)
        例: nose_z=-50, cheek_z=-30 → depth = -50-(-30) = -20 (负值)
      - 鼓腮后: 脸颊向前凸出(z变小)
        例: nose_z=-50, cheek_z=-55 → depth = -50-(-55) = +5 (正值)

    直观理解:
    - 负值: 脸颊在鼻尖后面（还没鼓起）
    - 正值: 脸颊超过鼻尖（已经鼓起）
    - 值越大: 鼓腮越多

    归一化:
    - depth_norm = depth / ICD
    - 使得不同患者可比较

    Returns:
        {
            "left_depth": 左脸颊深度(像素),
            "right_depth": 右脸颊深度(像素),
            "left_depth_norm": 左脸颊深度(归一化到ICD),
            "right_depth_norm": 右脸颊深度(归一化到ICD),
            "mean_depth_norm": 平均深度(归一化),
            "max_depth_norm": 最大深度(归一化),
            "nose_z": 鼻尖z坐标,
            "left_cheek_z": 左脸颊平均z坐标,
            "right_cheek_z": 右脸颊平均z坐标,
            "icd": 内眦间距,
        }
    """
    result = {
        "left_depth": None, "right_depth": None,
        "left_depth_norm": None, "right_depth_norm": None,
        "mean_depth_norm": None, "max_depth_norm": None,
        "nose_z": None, "left_cheek_z": None, "right_cheek_z": None,
        "icd": None,
    }

    if landmarks is None:
        return result

    # 获取鼻尖z坐标
    nose_pt = _safe_pt3d(landmarks, LM.NOSE_TIP, w, h)
    if nose_pt is None:
        return result
    nose_z = nose_pt[2]

    # 获取左脸颊平均z坐标
    left_zs = []
    for idx in LM.BLOW_CHEEK_L:
        p = _safe_pt3d(landmarks, idx, w, h)
        if p is not None:
            left_zs.append(p[2])

    if not left_zs:
        return result
    left_cheek_z = np.mean(left_zs)

    # 获取右脸颊平均z坐标
    right_zs = []
    for idx in LM.BLOW_CHEEK_R:
        p = _safe_pt3d(landmarks, idx, w, h)
        if p is not None:
            right_zs.append(p[2])

    if not right_zs:
        return result
    right_cheek_z = np.mean(right_zs)

    # 计算ICD用于归一化
    icd = compute_icd(landmarks, w, h)
    icd = max(icd, 1e-6)

    # 计算深度 (nose_z - cheek_z)
    # 直观: 负值=脸颊在后面, 正值=脸颊超过鼻尖, 值越大=鼓腮越多
    left_depth = nose_z - left_cheek_z
    right_depth = nose_z - right_cheek_z

    # 归一化
    left_depth_norm = left_depth / icd
    right_depth_norm = right_depth / icd
    mean_depth_norm = (left_depth_norm + right_depth_norm) / 2
    max_depth_norm = max(left_depth_norm, right_depth_norm)

    result.update({
        "left_depth": float(left_depth),
        "right_depth": float(right_depth),
        "left_depth_norm": float(left_depth_norm),
        "right_depth_norm": float(right_depth_norm),
        "mean_depth_norm": float(mean_depth_norm),
        "max_depth_norm": float(max_depth_norm),
        "nose_z": float(nose_z),
        "left_cheek_z": float(left_cheek_z),
        "right_cheek_z": float(right_cheek_z),
        "icd": float(icd),
    })

    return result


def find_peak_frame(landmarks_seq: List, frames_seq: List, w: int, h: int,
                    baseline_landmarks=None) -> Tuple[int, Dict[str, Any]]:
    """
    鼓腮峰值帧选择 - 使用脸颊深度最大的帧

    核心改进:
    =========
    不再依赖前10帧作为baseline！
    - 直接计算每帧的深度值 (depth = nose_z - cheek_z)
    - 取深度最大的帧作为峰值帧
    - 这样即使"视频全程在鼓腮"也能找到最大鼓腮帧

    门控条件:
    - 嘴唇必须闭合 (mouth_height_norm < 阈值)
    - 避免选到张嘴的帧

    Returns:
        (peak_idx, peak_debug): 峰值帧索引和调试信息
    """
    n = len(landmarks_seq)
    if n == 0:
        return 0, {}

    # 配置参数
    MOUTH_HEIGHT_THR = THR.MOUTH_HEIGHT

    # 扫描所有帧，计算深度
    left_depth_seq = [np.nan] * n
    right_depth_seq = [np.nan] * n
    max_depth_seq = [np.nan] * n  # 用于选峰值
    mouth_height_norm_seq = [np.nan] * n

    for i, lm in enumerate(landmarks_seq):
        if lm is None:
            continue

        try:
            # 计算深度
            depth_data = compute_cheek_depth(lm, w, h)

            if depth_data["left_depth_norm"] is not None:
                left_depth_seq[i] = depth_data["left_depth_norm"]
                right_depth_seq[i] = depth_data["right_depth_norm"]
                max_depth_seq[i] = depth_data["max_depth_norm"]

            # 计算嘴唇高度（门控条件）
            icd = depth_data["icd"] or compute_icd(lm, w, h)
            upper = pt2d(lm[LM.LIP_TOP], w, h)
            lower = pt2d(lm[LM.LIP_BOT], w, h)
            mouth_height_norm_seq[i] = dist(upper, lower) / max(icd, 1e-9)

        except Exception:
            continue

    # 门控：嘴唇闭合
    valid = np.isfinite(mouth_height_norm_seq) & (np.array(mouth_height_norm_seq) <= MOUTH_HEIGHT_THR)

    # 选峰值帧（深度最大的帧）
    peak_idx = 0
    if np.sum(valid) > 0:
        valid_depths = np.array(max_depth_seq)
        valid_depths[~valid] = np.nan  # 不满足门控的设为nan
        if not np.all(np.isnan(valid_depths)):
            peak_idx = int(np.nanargmax(valid_depths))
    elif not np.all(np.isnan(max_depth_seq)):  # Fallback
        peak_idx = int(np.nanargmax(max_depth_seq))

    # 构建peak_debug
    peak_debug = {
        # 深度曲线（直观定义: 负=脸颊在后, 正=脸颊超前, 越大=鼓腮越多）
        "left_depth": [float(v) if np.isfinite(v) else None for v in left_depth_seq],
        "right_depth": [float(v) if np.isfinite(v) else None for v in right_depth_seq],
        "max_depth": [float(v) if np.isfinite(v) else None for v in max_depth_seq],
        # 为兼容旧代码，保留这些字段名
        "left_bulge": [float(v) if np.isfinite(v) else None for v in left_depth_seq],
        "right_bulge": [float(v) if np.isfinite(v) else None for v in right_depth_seq],
        "score": [float(v) if np.isfinite(v) else None for v in max_depth_seq],
        # 门控曲线
        "mouth_height_norm": [float(v) if np.isfinite(v) else None for v in mouth_height_norm_seq],
        # 阈值
        "thresholds": {
            "mouth_height_thr": float(MOUTH_HEIGHT_THR),
        },
        "valid": [bool(v) for v in valid],
        "peak_idx": peak_idx,
        # 说明
        "depth_definition": "depth = nose_z - cheek_z; 负值=脸颊在鼻尖后面; 正值=脸颊超过鼻尖; 越大=鼓腮越多",
    }

    return peak_idx, peak_debug


def plot_cheek_bulge_curve(
        peak_debug: Dict[str, Any],
        fps: float,
        output_path: Path,
        palsy_detection: Dict[str, Any] = None
) -> None:
    """
    绘制鼓腮（BlowCheek）关键帧选择的可解释性曲线。

    深度定义 (直观):
    - depth = nose_z - cheek_z
    - 负值: 脸颊在鼻尖后面
    - 正值: 脸颊超过鼻尖
    - 值越大: 鼓腮越多

    选择标准: 深度最大的帧
    """
    # 兼容新旧字段名
    left_depth = peak_debug.get("left_depth", peak_debug.get("left_bulge", []))
    right_depth = peak_debug.get("right_depth", peak_debug.get("right_bulge", []))
    max_depth = peak_debug.get("max_depth", peak_debug.get("score", []))
    valid_mask = peak_debug.get("valid", None)
    peak_idx = peak_debug.get("peak_idx", 0)

    if not max_depth:
        return

    n_frames = len(max_depth)
    frames = np.arange(n_frames)
    time_sec = frames / fps if fps > 0 else frames
    x_label = 'Time (seconds)' if fps > 0 else 'Frame'
    peak_time = peak_idx / fps if fps > 0 else peak_idx

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # 转换为numpy数组
    left_arr = np.array([v if v is not None else np.nan for v in left_depth])
    right_arr = np.array([v if v is not None else np.nan for v in right_depth])
    max_arr = np.array([v if v is not None else np.nan for v in max_depth])

    # ===== 上图: 深度曲线 =====
    ax1 = axes[0]

    if valid_mask is not None:
        add_valid_region_shading(ax1, valid_mask, time_sec)

    ax1.plot(time_sec, left_arr, 'b-', label='Left Cheek Depth', linewidth=2, alpha=0.6)
    ax1.plot(time_sec, right_arr, 'r-', label='Right Cheek Depth', linewidth=2, alpha=0.6)
    ax1.plot(time_sec, max_arr, 'g--', label='Max Depth (Selection)', linewidth=2.5)

    # 零线（鼻尖位置）
    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5, label='Nose tip level')

    # 标记峰值
    ax1.axvline(x=peak_time, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Peak Frame {peak_idx}')
    if 0 <= peak_idx < n_frames and np.isfinite(max_arr[peak_idx]):
        peak_value = max_arr[peak_idx]
        ax1.scatter([peak_time], [peak_value], color='red', s=150, zorder=5,
                    edgecolors='black', linewidths=1.5, marker='*',
                    label=f'Selected Peak ({peak_value:.3f})')

    title = "BlowCheek: Cheek Depth Relative to Nose"
    if palsy_detection:
        palsy_text = get_palsy_side_text(palsy_detection.get("palsy_side", 0))
        title += f' | Detected: {palsy_text}'

    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_xlabel(x_label, fontsize=11)
    ax1.set_ylabel('Depth / ICD (positive = bulging past nose)', fontsize=11)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.4)

    # 添加说明
    ax1.text(0.02, 0.02,
             "Depth = nose_z - cheek_z\nNegative = cheek behind nose\nPositive = cheek past nose",
             transform=ax1.transAxes, fontsize=9, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ===== 下图: 左右差异 =====
    ax2 = axes[1]

    diff_arr = left_arr - right_arr
    ax2.fill_between(time_sec, diff_arr, 0, where=diff_arr > 0, color='blue', alpha=0.3, label='Left > Right')
    ax2.fill_between(time_sec, diff_arr, 0, where=diff_arr < 0, color='red', alpha=0.3, label='Right > Left')
    ax2.plot(time_sec, diff_arr, 'purple', linewidth=1.5)
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1)

    if 0 <= peak_idx < n_frames:
        ax2.axvline(x=peak_time, linestyle='--', color='black', linewidth=2)

    ax2.set_title('Left - Right Depth Difference (positive = left bulges more)', fontsize=11)
    ax2.set_xlabel(x_label, fontsize=11)
    ax2.set_ylabel('Depth Difference', fontsize=11)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()


def _get_blow_cheek_polygons():
    """
    优先用 LM.BLOW_CHEEK_L/R；如果没有，就回退到 LM.CHEEK_L/R。
    """
    left = getattr(LM, "BLOW_CHEEK_L", None)
    right = getattr(LM, "BLOW_CHEEK_R", None)
    region_name = "BLOW_CHEEK"

    if left is None or right is None:
        left = getattr(LM, "CHEEK_L", None)
        right = getattr(LM, "CHEEK_R", None)
        region_name = "CHEEK"

    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)) or len(left) < 3 or len(right) < 3:
        return None, None, None

    return list(left), list(right), region_name


def _inner_lip_area_norm(lm, w: int, h: int, icd: float) -> float:
    """嘴唇内圈面积（归一化到 ICD^2）"""
    pts = pts2d(lm, LM.INNER_LIP, w, h)
    area = float(abs(polygon_area(pts)))
    denom = float(max(icd * icd, 1e-9))
    return area / denom


def compute_blow_cheek_metrics(landmarks, w: int, h: int, baseline_landmarks=None) -> Dict[str, Any]:
    """
    鼓腮动作指标 - 使用直观的深度定义
    """
    mouth = compute_mouth_metrics(landmarks, w, h)
    oral = compute_oral_angle(landmarks, w, h)

    # 面中线：使用双内眦中点
    left_canthus = pt2d(landmarks[LM.EYE_INNER_L], w, h)
    right_canthus = pt2d(landmarks[LM.EYE_INNER_R], w, h)
    midline_x = (left_canthus[0] + right_canthus[0]) / 2

    # 左右嘴角位置
    left_corner = mouth["left_corner"]
    right_corner = mouth["right_corner"]

    # 嘴角到面中线的距离
    left_to_midline = abs(midline_x - left_corner[0])
    right_to_midline = abs(right_corner[0] - midline_x)

    # 嘴唇中心相对于面中线的偏移
    mouth_center_x = (left_corner[0] + right_corner[0]) / 2
    mouth_midline_offset = mouth_center_x - midline_x

    metrics = {
        "mouth_width": mouth["width"],
        "mouth_height": mouth["height"],
        "midline_x": float(midline_x),
        "left_to_midline": float(left_to_midline),
        "right_to_midline": float(right_to_midline),
        "mouth_midline_offset": float(mouth_midline_offset),
        "left_corner": left_corner,
        "right_corner": right_corner,
        "oral_angle": {
            "AOE": oral.AOE_angle,
            "BOF": oral.BOF_angle,
            "asymmetry": oral.angle_asymmetry,
        }
    }

    # ========== 计算脸颊深度（直观定义）==========
    cheek_depth = compute_cheek_depth(landmarks, w, h)
    metrics["cheek_depth"] = cheek_depth
    metrics["icd"] = cheek_depth.get("icd") or compute_icd(landmarks, w, h)

    # ========== 嘴唇面中线对称性 ==========
    lip_symmetry = compute_lip_midline_symmetry(landmarks, w, h)
    metrics["lip_symmetry"] = lip_symmetry

    # 如果有基线，计算变化量
    if baseline_landmarks is not None:
        scale = compute_scale_to_baseline(landmarks, baseline_landmarks, w, h)
        metrics["scale"] = scale

        baseline_mouth = compute_mouth_metrics(baseline_landmarks, w, h)
        baseline_oral = compute_oral_angle(baseline_landmarks, w, h)

        # 口角角度变化
        aoe_change = oral.AOE_angle - baseline_oral.AOE_angle
        bof_change = oral.BOF_angle - baseline_oral.BOF_angle

        metrics["baseline"] = {
            "mouth_width": baseline_mouth["width"],
            "AOE": baseline_oral.AOE_angle,
            "BOF": baseline_oral.BOF_angle,
        }

        metrics["movement"] = {
            "aoe_change": float(aoe_change),
            "bof_change": float(bof_change),
        }

        scaled_width = mouth["width"] * scale
        metrics["mouth_width_change"] = scaled_width - baseline_mouth["width"]

        # 计算深度变化（如果需要）
        baseline_depth = compute_cheek_depth(baseline_landmarks, w, h)
        if baseline_depth["left_depth_norm"] is not None and cheek_depth["left_depth_norm"] is not None:
            metrics["depth_change"] = {
                "left": cheek_depth["left_depth_norm"] - baseline_depth["left_depth_norm"],
                "right": cheek_depth["right_depth_norm"] - baseline_depth["right_depth_norm"],
            }

    # ========== 计算嘴唇中线相对于面中线的偏移 ==========
    lip_offset_data = compute_lip_midline_offset_from_face_midline(
        landmarks, w, h, baseline_landmarks
    )
    metrics["lip_midline_offset"] = lip_offset_data

    # ========== 计算嘴唇形状对称性 ==========
    lip_shape_data = compute_lip_shape_symmetry(landmarks, w, h)
    metrics["lip_shape_symmetry"] = lip_shape_data

    return metrics


def detect_palsy_side(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    鼓腮动作面瘫侧别检测 - 基于嘴唇中线偏移

    核心逻辑:
    =========
    鼓腮时，健侧脸颊能正常鼓起，会把嘴唇挤向患侧。
    因此：嘴唇偏向哪侧，哪侧就是患侧。

    判断方式:
    - current_signed_dist > 0 (嘴唇偏向患者左侧) → 左侧是患侧
    - current_signed_dist < 0 (嘴唇偏向患者右侧) → 右侧是患侧

    这与其他动作（如LipPucker、ShowTeeth）的逻辑相反！
    其他动作是"被健侧拉向健侧"，鼓腮是"被健侧挤向患侧"。
    """
    result = {
        "palsy_side": 0,          # 0=对称, 1=左患侧, 2=右患侧
        "confidence": 0.0,
        "interpretation": "",
        "method": "lip_offset_blow_cheek",  # 标记这是鼓腮特有的逻辑
        "evidence": {},
        "votes": [],
        "final_decision": {},
    }

    # ========== 获取嘴唇中线偏移数据 ==========
    if "lip_midline_offset" not in metrics:
        result["interpretation"] = "无嘴唇偏移数据"
        return result

    offset_data = metrics["lip_midline_offset"]

    # 获取有符号距离（正值=偏向患者左侧，负值=偏向患者右侧）
    current_signed_dist = offset_data.get("current_signed_dist",
                                          offset_data.get("current_offset", 0))
    offset_norm = offset_data.get("offset_norm", 0)
    icd = offset_data.get("icd", metrics.get("icd", 1))

    # 如果没有归一化值，手动计算
    if offset_norm == 0 and icd > 1e-6:
        offset_norm = abs(current_signed_dist) / icd

    # 阈值
    threshold = THR.MOUTH_CENTER_PALSY_OFFSET
    is_abnormal = offset_norm > threshold

    # 记录证据
    result["evidence"]["lip_offset"] = {
        "raw_value_px": float(current_signed_dist),
        "normalized_value": float(offset_norm),
        "threshold": float(threshold),
        "is_abnormal": is_abnormal,
        "description": (
            f"嘴唇偏移: {current_signed_dist:+.1f}px ({offset_norm:.4%} ICD), "
            f"阈值: {threshold:.3%}"
        ),
        "logic": "鼓腮时嘴唇被挤向患侧：偏向哪侧，哪侧就是患侧",
    }

    result["final_decision"] = {
        "offset_signed": float(current_signed_dist),
        "offset_norm": float(offset_norm),
        "is_abnormal": is_abnormal,
    }

    # ========== 判断面瘫侧别 ==========
    if not is_abnormal:
        result["palsy_side"] = 0
        result["confidence"] = 0.0
        result["interpretation"] = (
            f"嘴唇位置对称 (偏移{current_signed_dist:+.1f}px = {offset_norm:.4%} ICD ≤ {threshold:.3%})"
        )
        return result

    # 计算置信度
    conf = min(1.0, (offset_norm - threshold) / max(threshold, 1e-6))
    result["confidence"] = float(conf)

    # 鼓腮特有逻辑：嘴唇被拉向健侧
    if current_signed_dist < 0:
        # 嘴唇偏向患者右侧 = 被拉向健侧 = 左侧是患侧
        result["palsy_side"] = 1
        result["interpretation"] = (
            f"左侧面瘫：嘴唇被挤向左侧 ({current_signed_dist:+.1f}px = {offset_norm:.4%} ICD > {threshold:.3%})"
        )
    else:
        # 嘴唇偏向患者左侧 = 被拉向健侧  = 右侧是患侧
        result["palsy_side"] = 2
        result["interpretation"] = (
            f"右侧面瘫：嘴唇被挤向右侧 ({current_signed_dist:+.1f}px = {offset_norm:.4%} ICD > {threshold:.3%})"
        )

    return result


def compute_voluntary_score(metrics: Dict[str, Any], baseline_landmarks=None) -> Tuple[int, str]:
    """
    计算自主运动评分

    基于脸颊深度的对称性
    """
    if "cheek_depth" not in metrics:
        return 3, "无深度数据"

    depth_data = metrics["cheek_depth"]
    left_depth = depth_data.get("left_depth_norm")
    right_depth = depth_data.get("right_depth_norm")

    if left_depth is None or right_depth is None:
        return 3, "深度数据无效"

    # 计算对称性比例
    max_depth = max(abs(left_depth), abs(right_depth))
    min_depth = min(abs(left_depth), abs(right_depth))

    if max_depth < 0.001:
        return 1, "几乎无鼓腮动作"

    symmetry_ratio = min_depth / max_depth

    if symmetry_ratio >= 0.85:
        return 5, "完全对称鼓腮"
    elif symmetry_ratio >= 0.70:
        return 4, "基本对称"
    elif symmetry_ratio >= 0.50:
        return 3, "轻度不对称"
    elif symmetry_ratio >= 0.25:
        return 2, "明显不对称"
    else:
        return 1, "严重不对称"


def compute_severity_score(metrics: Dict[str, Any]) -> Tuple[int, str]:
    """
    计算严重度分数

    基于嘴唇偏移程度
    """
    if "lip_midline_offset" not in metrics:
        return 1, "无数据"

    offset_data = metrics["lip_midline_offset"]
    offset_norm = offset_data.get("offset_norm", 0)

    if offset_norm < 0.02:
        return 1, f"正常 (偏移{offset_norm:.1%} ICD)"
    elif offset_norm < 0.04:
        return 2, f"轻度 (偏移{offset_norm:.1%} ICD)"
    elif offset_norm < 0.06:
        return 3, f"中度 (偏移{offset_norm:.1%} ICD)"
    elif offset_norm < 0.10:
        return 4, f"重度 (偏移{offset_norm:.1%} ICD)"
    else:
        return 5, f"完全面瘫 (偏移{offset_norm:.1%} ICD)"


def detect_synkinesis(baseline_result: Optional[ActionResult],
                      current_landmarks, w: int, h: int) -> Dict[str, int]:
    """检测联动运动"""
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

        if avg_change > 0.18:
            synkinesis["eye_synkinesis"] = 3
        elif avg_change > 0.10:
            synkinesis["eye_synkinesis"] = 2
        elif avg_change > 0.05:
            synkinesis["eye_synkinesis"] = 1

    return synkinesis


def _get_blow_cheek_vis_indices() -> list:
    """鼓腮可视化：嘴唇轮廓 + 鼓腮区域点"""
    idx = []

    for name in ("OUTER_LIP", "INNER_LIP"):
        v = getattr(LM, name, None)
        if isinstance(v, (list, tuple)):
            idx += list(v)

    for name in ("BLOW_CHEEK_L", "BLOW_CHEEK_R"):
        v = getattr(LM, name, None)
        if isinstance(v, (list, tuple)):
            idx += list(v)

    return sorted(set(int(i) for i in idx if isinstance(i, (int, np.integer))))


BLOW_CHEEK_VIS_INDICES = _get_blow_cheek_vis_indices()


def visualize_blow_cheek(frame, landmarks, metrics: Dict[str, Any], w: int, h: int,
                         palsy_detection: Dict[str, Any] = None):
    """可视化鼓腮指标"""
    img = frame.copy()

    # 添加患侧标签
    img = draw_palsy_side_label(img, palsy_detection, x=20, y=70, font_scale=1.4)

    # 面中线
    midline = compute_face_midline(landmarks, w, h)
    if midline:
        img = draw_face_midline(img, midline, color=(0, 255, 255), thickness=2, dashed=True)

    # 关键点
    draw_landmarks(img, landmarks, w, h, BLOW_CHEEK_VIS_INDICES, radius=2)

    # 脸颊区域
    left_idx, right_idx, region = _get_blow_cheek_polygons()
    if left_idx is not None:
        left_pts = pts2d(landmarks, left_idx, w, h).astype(np.int32)
        right_pts = pts2d(landmarks, right_idx, w, h).astype(np.int32)
        overlay = img.copy()
        cv2.fillPoly(overlay, [left_pts], (0, 255, 255))
        cv2.fillPoly(overlay, [right_pts], (255, 255, 0))
        img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)
        cv2.polylines(img, [left_pts], True, (0, 255, 255), 3)
        cv2.polylines(img, [right_pts], True, (255, 255, 0), 3)

    # 信息面板
    y = 140

    # ========== 嘴唇偏移信息（主要指标）==========
    if "lip_midline_offset" in metrics:
        offset_data = metrics["lip_midline_offset"]
        current_signed_dist = offset_data.get("current_signed_dist",
                                              offset_data.get("current_offset", 0))
        offset_norm = offset_data.get("offset_norm", 0)

        # 偏移方向和颜色
        if abs(current_signed_dist) > 3:
            direction = "→ Left" if current_signed_dist > 0 else "→ Right"
            offset_color = (0, 0, 255)  # 红色表示有偏移
        else:
            direction = "Center"
            offset_color = (0, 255, 0)  # 绿色表示居中

        cv2.putText(img, f"Lip Offset: {current_signed_dist:+.1f}px ({offset_norm:.1%} ICD) {direction}", (30, y),
                    FONT, FONT_SCALE_NORMAL, offset_color, THICKNESS_NORMAL)
        y += LINE_HEIGHT

        # 可视化嘴唇中心和偏移线
        lip_x = offset_data.get("lip_midline_x")
        lip_y = offset_data.get("lip_midline_y")
        lip_center_proj = offset_data.get("lip_center_proj")

        if lip_x is not None and lip_y is not None:
            cv2.circle(img, (int(lip_x), int(lip_y)), 8, (0, 255, 0), -1)
            cv2.putText(img, "Lip", (int(lip_x) + 10, int(lip_y) - 5),
                        FONT, 0.5, (0, 255, 0), 2)

            if lip_center_proj is not None:
                proj_x, proj_y = lip_center_proj
                dist_val = abs(current_signed_dist)

                cv2.line(img, (int(lip_x), int(lip_y)),
                         (int(proj_x), int(proj_y)), offset_color, 3)
                cv2.circle(img, (int(proj_x), int(proj_y)), 5, offset_color, -1)

                if dist_val > 3:
                    direction_text = "L" if current_signed_dist > 0 else "R"
                    mid_x = (int(lip_x) + int(proj_x)) // 2
                    mid_y = (int(lip_y) + int(proj_y)) // 2
                    cv2.putText(img, f"{direction_text} {dist_val:.1f}px",
                                (mid_x + 5, mid_y - 10), FONT, 0.6, offset_color, 2)

    # 判断逻辑说明
    cv2.putText(img, "Logic: Lip pushed to palsy side", (30, y),
                FONT, FONT_SCALE_SMALL, (200, 200, 200), THICKNESS_THIN)
    y += LINE_HEIGHT

    # ========== 脸颊深度信息（参考）==========
    cheek = metrics.get("cheek_depth", {})
    ld = cheek.get("left_depth_norm", float("nan"))
    rd = cheek.get("right_depth_norm", float("nan"))

    cv2.putText(img, f"Cheek Depth L/R: {ld:+.4f} / {rd:+.4f}", (30, y),
                FONT, FONT_SCALE_SMALL, (180, 180, 180), THICKNESS_THIN)
    y += LINE_HEIGHT

    oral = metrics.get("oral_angle", {})
    cv2.putText(img, f"AOE/BOF: {oral.get('AOE', 0):.2f} / {oral.get('BOF', 0):.2f}", (30, y),
                FONT, FONT_SCALE_SMALL, (180, 180, 180), THICKNESS_THIN)
    y += LINE_HEIGHT + 10

    # 唇中线
    lip_top = pt2d(landmarks[LM.LIP_TOP_CENTER], w, h)
    lip_bot = pt2d(landmarks[LM.LIP_BOT_CENTER], w, h)
    cv2.line(img,
             (int(lip_top[0]), int(lip_top[1])),
             (int(lip_bot[0]), int(lip_bot[1])),
             (255, 255, 0), 3)
    cv2.circle(img, (int(lip_top[0]), int(lip_top[1])), 5, (255, 255, 0), -1)
    cv2.circle(img, (int(lip_bot[0]), int(lip_bot[1])), 5, (255, 255, 0), -1)

    return img


def process(landmarks_seq: List, frames_seq: List, w: int, h: int,
            video_info: Dict[str, Any], output_dir: Path,
            baseline_result: Optional[ActionResult] = None,
            baseline_landmarks=None) -> Optional[ActionResult]:
    """处理BlowCheek动作"""
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
    metrics = compute_blow_cheek_metrics(peak_landmarks, w, h, baseline_landmarks)
    palsy_detection = detect_palsy_side(metrics)
    score, interpretation = compute_voluntary_score(metrics, baseline_landmarks)
    result.voluntary_movement_score = score
    synkinesis = detect_synkinesis(baseline_result, peak_landmarks, w, h)
    result.synkinesis_scores = synkinesis
    severity_score, severity_desc = compute_severity_score(metrics)

    result.action_specific = {
        "mouth_metrics": {"width": metrics["mouth_width"], "height": metrics["mouth_height"]},
        "oral_angle": metrics["oral_angle"],
        "cheek_depth": metrics["cheek_depth"],
        "lip_midline_offset": metrics.get("lip_midline_offset", {}),
        "palsy_detection": palsy_detection,
        "voluntary_interpretation": interpretation,
        "synkinesis": synkinesis,
        "severity_score": severity_score,
        "severity_desc": severity_desc,
        "peak_debug": peak_debug,
    }

    action_dir = output_dir / ACTION_NAME
    action_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(action_dir / "peak_raw.jpg"), peak_frame)

    # 绘图
    plot_cheek_bulge_curve(
        peak_debug,
        video_info.get("fps", 30.0),
        action_dir / "peak_selection_curve.png",
        palsy_detection
    )

    vis = visualize_blow_cheek(peak_frame, peak_landmarks, metrics, w, h, palsy_detection)
    cv2.imwrite(str(action_dir / "peak_indicators.jpg"), vis)

    with open(action_dir / "indicators.json", 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    # 打印信息
    offset = metrics.get("lip_midline_offset", {})
    offset_val = offset.get("current_signed_dist", 0)
    offset_norm = offset.get("offset_norm", 0)
    print(f"    [OK] {ACTION_NAME}: Lip offset={offset_val:+.1f}px ({offset_norm:.2%} ICD)")
    print(f"         Palsy: {palsy_detection.get('interpretation', 'N/A')}")
    print(f"         Voluntary Score: {result.voluntary_movement_score}/5 ({interpretation})")

    return result
