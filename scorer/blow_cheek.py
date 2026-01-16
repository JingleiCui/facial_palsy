#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BlowCheek 动作处理模块
================================

分析鼓腮动作:
1. 唇密封距离
2. 嘴部闭合程度
3. 口角对称性
4. 面瘫侧别检测
5. 联动运动检测

关键帧检测方法:
- 鼓腮时嘴唇紧闭，鼓腮体积最大
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


def plot_cheek_bulge_curve(
        peak_debug: Dict[str, Any],
        fps: float,
        output_path: Path,
        palsy_detection: Dict[str, Any] = None
) -> None:
    """
    绘制鼓腮（BlowCheek）关键帧选择的可解释性曲线。
    选择标准：左右脸颊相对鼻尖的凸出程度（bulge）最大。
    """
    import matplotlib.pyplot as plt
    from clinical_base import add_valid_region_shading, get_palsy_side_text

    left_bulge = peak_debug.get("left_bulge", [])
    right_bulge = peak_debug.get("right_bulge", [])
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

    plt.figure(figsize=(12, 6))

    if valid_mask is not None:
        add_valid_region_shading(plt.gca(), valid_mask, time_sec)

    plt.plot(time_sec, left_bulge, 'b-', label='Left Cheek Bulge', linewidth=2, alpha=0.6)
    plt.plot(time_sec, right_bulge, 'r-', label='Right Cheek Bulge', linewidth=2, alpha=0.6)
    plt.plot(time_sec, score, 'g-', label='Combined Score (Selection)', linewidth=2.5)

    plt.axvline(x=peak_time, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Peak Frame {peak_idx}')
    if 0 <= peak_idx < n_frames:
        peak_score = score[peak_idx]
        plt.scatter([peak_time], [peak_score], color='red', s=150, zorder=5,
                    edgecolors='black', linewidths=1.5, marker='*', label=f'Selected Peak (Score: {peak_score:.3f})')

    title = "BlowCheek Peak Selection: Max Cheek Bulge"
    if palsy_detection:
        palsy_text = get_palsy_side_text(palsy_detection.get("palsy_side", 0))
        title += f' | Detected: {palsy_text}'

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(x_label, fontsize=11)
    plt.ylabel('Bulge Score (higher is better)', fontsize=11)
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()


def _safe_pt3d(landmarks, idx: int, w: int, h: int):
    try:
        p = np.array(pt3d(landmarks[idx], w, h), dtype=np.float64)  # (x_px, y_px, z_px)
        if np.any(~np.isfinite(p)):
            return None
        return p
    except Exception:
        return None


def _mean_region_z_aligned(landmarks, indices: List[int], w: int, h: int,
                           baseline_landmarks=None) -> float:
    """
    计算某个区域(一圈点)在“对齐到baseline坐标系后”的平均z。
    z 越小(更负) => 越靠近镜头
    """
    if landmarks is None or len(indices) == 0:
        return float("nan")

    # 没 baseline：退化为直接平均当前帧z（不推荐，但保证能跑）
    if baseline_landmarks is None:
        zs = []
        for idx in indices:
            p = _safe_pt3d(landmarks, idx, w, h)
            if p is not None:
                zs.append(float(p[2]))
        return float(np.mean(zs)) if zs else float("nan")

    # 用稳定点做刚体对齐：把当前帧对齐到 baseline
    stable_idx = [
        LM.EYE_INNER_L, LM.EYE_INNER_R,
        LM.EYE_OUTER_L, LM.EYE_OUTER_R,
        LM.NOSE_BRIDGE, LM.NOSE_TIP,
        LM.CHIN
    ]

    P, Q = [], []
    for i in stable_idx:
        p = _safe_pt3d(landmarks, i, w, h)
        q = _safe_pt3d(baseline_landmarks, i, w, h)
        if p is not None and q is not None:
            P.append(p);
            Q.append(q)

    if len(P) < 3 or len(Q) < 3:
        return _mean_region_z_aligned(landmarks, indices, w, h, baseline_landmarks=None)

    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    R, t = kabsch_rigid_transform(P, Q)
    if R is None:
        return _mean_region_z_aligned(landmarks, indices, w, h, baseline_landmarks=None)

    region = []
    for idx in indices:
        p = _safe_pt3d(landmarks, idx, w, h)
        if p is not None:
            region.append(p)

    if len(region) == 0:
        return float("nan")

    region = np.asarray(region, dtype=np.float64)
    region_aligned = apply_rigid_transform(region, R, t)
    return float(np.mean(region_aligned[:, 2]))


def compute_cheek_depth_delta(landmarks, w: int, h: int, baseline_landmarks=None) -> Dict[str, Any]:
    """
    鼓腮“深度代理”：左右脸颊分别计算 (base_z - aligned_z)/ICD
    值越大 => 该侧脸颊越“凸出/靠近镜头”
    """
    if baseline_landmarks is None:
        return {
            "left_delta_norm": None, "right_delta_norm": None, "mean_delta_norm": None,
            "left_z_aligned": None, "right_z_aligned": None,
            "base_left_z": None, "base_right_z": None,
            "icd": None
        }

    icd = float(compute_icd(baseline_landmarks, w, h))
    icd = max(icd, 1e-6)

    # baseline 的区域平均z（baseline坐标系下，不需要对齐）
    base_left_z = _mean_region_z_aligned(baseline_landmarks, LM.BLOW_CHEEK_L, w, h, baseline_landmarks=None)
    base_right_z = _mean_region_z_aligned(baseline_landmarks, LM.BLOW_CHEEK_R, w, h, baseline_landmarks=None)

    # 当前帧：对齐到baseline后再算区域平均z
    left_z_aligned = _mean_region_z_aligned(landmarks, LM.BLOW_CHEEK_L, w, h, baseline_landmarks=baseline_landmarks)
    right_z_aligned = _mean_region_z_aligned(landmarks, LM.BLOW_CHEEK_R, w, h, baseline_landmarks=baseline_landmarks)

    if not np.isfinite(base_left_z) or not np.isfinite(base_right_z) or not np.isfinite(
            left_z_aligned) or not np.isfinite(right_z_aligned):
        return {
            "left_delta_norm": None, "right_delta_norm": None, "mean_delta_norm": None,
            "left_z_aligned": float(left_z_aligned), "right_z_aligned": float(right_z_aligned),
            "base_left_z": float(base_left_z), "base_right_z": float(base_right_z),
            "icd": float(icd)
        }

    left_delta_norm = float((base_left_z - left_z_aligned) / icd)
    right_delta_norm = float((base_right_z - right_z_aligned) / icd)
    mean_delta_norm = float((left_delta_norm + right_delta_norm) / 2.0)

    return {
        "left_delta_norm": left_delta_norm,
        "right_delta_norm": right_delta_norm,
        "mean_delta_norm": mean_delta_norm,
        "left_z_aligned": float(left_z_aligned),
        "right_z_aligned": float(right_z_aligned),
        "base_left_z": float(base_left_z),
        "base_right_z": float(base_right_z),
        "icd": float(icd)
    }


def _save_cheek_depth_curve_png(png_path,
                                left_series: List[float],
                                right_series: List[float],
                                mean_series: List[float],
                                score_series: List[float],
                                peak_idx: int,
                                valid_mask: List[bool] = None,
                                palsy_detection: Dict[str, Any] = None) -> None:
    """输出左右/平均脸颊深度曲线，并标注峰值帧和valid区域"""
    if not left_series or not right_series or not mean_series:
        return

    fig, ax = plt.subplots(figsize=(16, 6))  # 增加宽度

    xs = np.array(list(range(len(mean_series))))

    # 标注 valid/invalid 区域
    if valid_mask is not None:
        add_valid_region_shading(ax, valid_mask, xs)

    ax.plot(xs, left_series, 'b-', label="Left cheek (delta_z/ICD)", linewidth=2)
    ax.plot(xs, right_series, 'r-', label="Right cheek (delta_z/ICD)", linewidth=2)

    # 绘制原始 mean_delta
    ax.plot(xs, mean_series, 'g-', label="Mean (delta_z/ICD) - Raw", linewidth=2, alpha=0.7)

    # 标注原始mean_delta的最高点
    max_mean_idx = np.nanargmax(mean_series)
    ax.scatter([max_mean_idx], [mean_series[max_mean_idx]],
               color='green', s=100, zorder=4, marker='o',
               edgecolors='black', linewidths=1, label='Max mean')

    # （score_smooth的最高点）
    ax.axvline(peak_idx, linestyle="--", color='black', linewidth=2)
    ax.scatter([peak_idx], [score_series[peak_idx]],
               color='red', s=150, zorder=5, marker='*',
               edgecolors='black', linewidths=2, label='Selected peak')

    peak_idx = int(max(0, min(peak_idx, len(xs) - 1)))
    ax.axvline(peak_idx, linestyle="--", color='black', linewidth=2)
    if mean_series[peak_idx] is not None and not np.isnan(mean_series[peak_idx]):
        ax.scatter([peak_idx], [mean_series[peak_idx]], color='red', s=150,
                   zorder=5, marker='*', edgecolors='black', linewidths=2)

    title = "BlowCheek depth curve: (baseline_z - aligned_z) / ICD"
    if palsy_detection:
        palsy_text = get_palsy_side_text(palsy_detection.get("palsy_side", 0))
        title += f' | Detected: {palsy_text}'
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel("Frame", fontsize=11)
    ax.set_ylabel("Normalized depth delta (bigger = more bulge)", fontsize=11)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(png_path), dpi=160)
    plt.close()


# =============================================================================
# 新的曲线绘制函数
# =============================================================================

def _save_cheek_bulge_curve_png(png_path,
                                peak_debug: Dict[str, Any],
                                palsy_detection: Dict[str, Any] = None) -> None:
    """
    绘制鼓腮 bulge 曲线（相对鼻尖深度变化）

    Args:
        png_path: 输出路径
        peak_debug: find_peak_frame 返回的调试信息
        palsy_detection: 面瘫检测结果
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 从 clinical_base 导入
    from clinical_base import add_valid_region_shading, get_palsy_side_text

    left_bulge = peak_debug.get("left_bulge", [])
    right_bulge = peak_debug.get("right_bulge", [])
    mean_bulge = peak_debug.get("mean_bulge", [])
    valid_mask = peak_debug.get("valid", None)
    peak_idx = peak_debug.get("peak_idx", 0)
    baseline = peak_debug.get("baseline", {})

    if not left_bulge or not right_bulge:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    n = len(mean_bulge)
    xs = np.arange(n)

    # 转换为 numpy 数组
    left_arr = np.array([v if v is not None else np.nan for v in left_bulge])
    right_arr = np.array([v if v is not None else np.nan for v in right_bulge])
    mean_arr = np.array([v if v is not None else np.nan for v in mean_bulge])

    # ===== 上图：Bulge 曲线 =====
    ax1 = axes[0]

    # 标注有效区域
    if valid_mask is not None:
        add_valid_region_shading(ax1, valid_mask, xs)

    ax1.plot(xs, left_arr, 'b-', label="Left cheek bulge", linewidth=2)
    ax1.plot(xs, right_arr, 'r-', label="Right cheek bulge", linewidth=2)
    ax1.plot(xs, mean_arr, 'g--', label="Mean bulge", linewidth=1.5, alpha=0.7)

    # 标注峰值帧
    if 0 <= peak_idx < n:
        ax1.axvline(peak_idx, linestyle="--", color='black', linewidth=2, label='Peak frame')
        if np.isfinite(mean_arr[peak_idx]):
            ax1.scatter([peak_idx], [mean_arr[peak_idx]], color='red', s=150,
                        zorder=5, marker='*', edgecolors='black', linewidths=2)

    # 标题
    title = "BlowCheek: Cheek bulge relative to nose (higher = more bulge)"
    if palsy_detection:
        palsy_text = get_palsy_side_text(palsy_detection.get("palsy_side", 0))
        title += f' | Detected: {palsy_text}'

    ax1.set_title(title, fontsize=12, fontweight='bold')
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Bulge = (base_rel_z - current_rel_z) / ICD")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 添加baseline信息
    if baseline:
        info_text = (f"Baseline: {baseline.get('frames_used', 0)} frames\n"
                     f"Base L_rel_z: {baseline.get('left_rel_z', 0):.4f}\n"
                     f"Base R_rel_z: {baseline.get('right_rel_z', 0):.4f}")
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=9,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ===== 下图：左右差异 =====
    ax2 = axes[1]

    diff_arr = left_arr - right_arr
    ax2.fill_between(xs, diff_arr, 0, where=diff_arr > 0, color='blue', alpha=0.3, label='Left > Right')
    ax2.fill_between(xs, diff_arr, 0, where=diff_arr < 0, color='red', alpha=0.3, label='Right > Left')
    ax2.plot(xs, diff_arr, 'purple', linewidth=1.5)
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1)

    if 0 <= peak_idx < n:
        ax2.axvline(peak_idx, linestyle='--', color='black', linewidth=2)

    ax2.set_title('Left - Right Bulge Difference (positive = left bulges more)', fontsize=11)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Bulge Difference')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(png_path), dpi=150)
    plt.close()


def _inner_lip_area_norm(lm, w: int, h: int, icd: float) -> float:
    """
    嘴唇内圈面积（归一化到 ICD^2）
    闭嘴时该值很小；张嘴会明显变大。
    """
    pts = pts2d(lm, LM.INNER_LIP, w, h)  # 你的 clinical_base 里已经有 LM.INNER_LIP
    area = float(abs(polygon_area(pts)))
    denom = float(max(icd * icd, 1e-9))
    return area / denom


def find_peak_frame(landmarks_seq: List, frames_seq: List, w: int, h: int,
                    baseline_landmarks=None) -> Tuple[int, Dict[str, Any]]:
    """
    鼓腮峰值帧选择 - 使用脸颊相对鼻尖的深度变化。
    """
    n = len(landmarks_seq)
    if n == 0:
        return 0, {}

    # 配置参数
    BASELINE_FRAMES = getattr(THR, 'BLOW_CHEEK_BASELINE_FRAMES', 10)
    MOUTH_HEIGHT_THR = THR.MOUTH_HEIGHT

    # 辅助函数
    def _get_rel_z(lm, region_indices):
        nose_z = pt3d(lm[LM.NOSE_TIP], w, h)[2]
        region_pts_3d = [pt3d(lm[i], w, h) for i in region_indices]
        region_z = np.mean([p[2] for p in region_pts_3d])
        return region_z - nose_z

    def _get_mouth_height_norm(lm, icd):
        upper = pt2d(lm[LM.LIP_TOP], w, h)
        lower = pt2d(lm[LM.LIP_BOT], w, h)
        return dist(upper, lower) / max(icd, 1e-9)

    # 扫描所有帧
    icd_arr, left_rel_z, right_rel_z, mouth_height_norm = [np.nan] * n, [np.nan] * n, [np.nan] * n, [np.nan] * n
    for i, lm in enumerate(landmarks_seq):
        if lm is None: continue
        try:
            icd = compute_icd(lm, w, h)
            icd_arr[i] = icd
            left_rel_z[i] = _get_rel_z(lm, LM.BLOW_CHEEK_L)
            right_rel_z[i] = _get_rel_z(lm, LM.BLOW_CHEEK_R)
            mouth_height_norm[i] = _get_mouth_height_norm(lm, icd)
        except Exception:
            continue

    # 建立内部baseline
    valid_init = np.isfinite(left_rel_z[:BASELINE_FRAMES]) & np.isfinite(right_rel_z[:BASELINE_FRAMES]) & np.isfinite(
        icd_arr[:BASELINE_FRAMES])
    if np.sum(valid_init) < 3:
        valid_init = np.isfinite(left_rel_z) & np.isfinite(right_rel_z) & np.isfinite(icd_arr)

    if np.sum(valid_init) == 0: return 0, {"error": "no valid frames"}

    base_l_rel_z = np.median([x for x in left_rel_z[:BASELINE_FRAMES] if np.isfinite(x)])
    base_r_rel_z = np.median([x for x in right_rel_z[:BASELINE_FRAMES] if np.isfinite(x)])
    base_icd = np.median([x for x in icd_arr[:BASELINE_FRAMES] if np.isfinite(x)])

    # 计算bulge和score
    left_bulge, right_bulge, score = [np.nan] * n, [np.nan] * n, [np.nan] * n
    for i in range(n):
        if np.isfinite(left_rel_z[i]):
            left_bulge[i] = (base_l_rel_z - left_rel_z[i]) / base_icd
        if np.isfinite(right_rel_z[i]):
            right_bulge[i] = (base_r_rel_z - right_rel_z[i]) / base_icd
        if np.isfinite(left_bulge[i]) and np.isfinite(right_bulge[i]):
            score[i] = max(left_bulge[i], right_bulge[i])

    # 门控：嘴唇闭合
    valid = np.isfinite(mouth_height_norm) & (np.array(mouth_height_norm) <= MOUTH_HEIGHT_THR)

    # 选峰值
    peak_idx = 0
    if np.sum(valid) > 0:
        valid_scores = np.array(score)[valid]
        valid_indices = np.where(valid)[0]
        if len(valid_scores) > 0 and not np.all(np.isnan(valid_scores)):
            peak_idx = int(valid_indices[np.nanargmax(valid_scores)])
    elif len(score) > 0 and not np.all(np.isnan(score)):  # Fallback
        peak_idx = int(np.nanargmax(score))

    peak_debug = {
        "left_bulge": left_bulge,
        "right_bulge": right_bulge,
        "score": score,
        # 门控曲线
        "mouth_height_norm": [float(v) if np.isfinite(v) else None for v in mouth_height_norm],
        # 阈值
        "thresholds": {
            "mouth_height_thr": float(MOUTH_HEIGHT_THR),
        },
        "valid": [bool(v) for v in valid],
        "peak_idx": peak_idx,
        "baseline": {"left_rel_z": base_l_rel_z, "right_rel_z": base_r_rel_z},
    }
    return peak_idx, peak_debug


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


def _minmax01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    v = x[np.isfinite(x)]
    if v.size == 0:
        return np.full_like(x, np.nan, dtype=np.float64)
    mn, mx = float(np.nanmin(v)), float(np.nanmax(v))
    if abs(mx - mn) < 1e-12:
        out = np.zeros_like(x, dtype=np.float64)
        out[~np.isfinite(x)] = np.nan
        return out
    return (x - mn) / (mx - mn)


def compute_blow_cheek_metrics(landmarks, w: int, h: int, baseline_landmarks=None) -> Dict[str, Any]:
    """
    鼓腮动作指标 - 增加面中线对称性分析
    """
    mouth = compute_mouth_metrics(landmarks, w, h)
    oral = compute_oral_angle(landmarks, w, h)

    # ========== 面中线对称性计算 ==========
    left_corner = mouth["left_corner"]
    right_corner = mouth["right_corner"]

    # 面中线：使用双内眦中点
    left_canthus = pt2d(landmarks[LM.EYE_INNER_L], w, h)
    right_canthus = pt2d(landmarks[LM.EYE_INNER_R], w, h)
    midline_x = (left_canthus[0] + right_canthus[0]) / 2

    # 左右嘴角到面中线的距离（保留用于参考，不用于判断）
    left_to_midline = abs(midline_x - left_corner[0])
    right_to_midline = abs(right_corner[0] - midline_x)

    # 嘴唇中心相对于面中线的偏移
    mouth_center_x = (left_corner[0] + right_corner[0]) / 2
    mouth_midline_offset = mouth_center_x - midline_x
    # ==========================================

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

    metrics["icd"] = compute_icd(landmarks, w, h)

    # 计算脸颊深度
    cheek_depth = compute_cheek_depth_delta(landmarks, w, h, baseline_landmarks)
    metrics["cheek_depth"] = cheek_depth

    # ========== 嘴唇面中线对称性（用于面瘫侧别判断）==========
    lip_symmetry = compute_lip_midline_symmetry(landmarks, w, h)
    metrics["lip_symmetry"] = lip_symmetry

    # 如果有基线
    if baseline_landmarks is not None:
        scale = compute_scale_to_baseline(landmarks, baseline_landmarks, w, h)
        metrics["scale"] = scale

        baseline_mouth = compute_mouth_metrics(baseline_landmarks, w, h)
        baseline_oral = compute_oral_angle(baseline_landmarks, w, h)

        # 口角角度变化（保留用于参考）
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

        # ========== 计算嘴唇中线相对于面中线的偏移变化 ==========
        lip_offset_data = compute_lip_midline_offset_from_face_midline(landmarks, w, h, baseline_landmarks)
        metrics["lip_midline_offset"] = lip_offset_data
    else:
        # 无baseline时也计算当前帧的lip_midline_offset（用于severity_score）
        lip_offset_data = compute_lip_midline_offset_from_face_midline(landmarks, w, h, None)
        metrics["lip_midline_offset"] = lip_offset_data

    # ========= 使用向量化函数计算嘴唇偏移 ==========
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
    从鼓腮动作检测面瘫侧别

    核心改进：
    1. 使用向量化面中线（处理头部倾斜）
    2. 多指标综合判断
    3. 详细记录每个指标的value vs threshold

    指标体系：
    1. 嘴唇中心偏移（主要）: 嘴唇中心到面中线的垂直距离
    2. 嘴唇形状对称性（辅助）: 左右嘴唇展开度对比
    3. 脸颊膨胀对称性（辅助）: 左右脸颊鼓起程度对比

    """
    result = {
        "palsy_side": 0,
        "confidence": 0.0,
        "interpretation": "",
        "method": "multi_indicator",
        "evidence": {},
        "votes": [],
        "final_decision": {},
    }

    votes = []  # 收集各指标的投票

    # ========== 指标1: 嘴唇中心偏移（主要指标）==========
    if "lip_midline_offset" in metrics:
        offset_data = metrics["lip_midline_offset"]

        # 获取关键数据
        current_signed_dist = offset_data.get("current_signed_dist",
                                              offset_data.get("current_offset", 0))
        offset_norm = offset_data.get("offset_norm", 0)
        icd = offset_data.get("icd", metrics.get("icd", 1))

        # 如果没有offset_norm，手动计算
        if offset_norm == 0 and icd > 1e-6:
            offset_norm = abs(current_signed_dist) / icd

        # 阈值
        threshold = THR.MOUTH_CENTER_PALSY_OFFSET  # 0.020
        is_abnormal = offset_norm > threshold

        # 判断方向
        if not is_abnormal:
            contribution = "None"
            vote_side = 0
            vote_conf = 0.0
        elif current_signed_dist > 0:
            # 偏向患者左侧 = 右侧面瘫
            contribution = "Right"
            vote_side = 2
            vote_conf = min(1.0, offset_norm / threshold - 1.0 + 0.5)
        else:
            contribution = "Left"
            vote_side = 1
            vote_conf = min(1.0, offset_norm / threshold - 1.0 + 0.5)

        # 记录证据
        result["evidence"]["lip_offset"] = {
            "raw_value_px": float(current_signed_dist),
            "normalized_value": float(offset_norm),
            "threshold": float(threshold),
            "is_abnormal": is_abnormal,
            "contribution": contribution,
            "weight": 0.50,  # 主要指标，权重50%
            "description": f"嘴唇中心偏移: {current_signed_dist:+.1f}px ({offset_norm:.2%} ICD), 阈值: {threshold:.2%}"
        }

        if vote_side != 0:
            votes.append({
                "indicator": "lip_offset",
                "side": vote_side,
                "confidence": vote_conf,
                "weight": 0.50,
            })

    # ========== 指标2: 脸颊膨胀对称性 ==========
    if "bulge" in metrics:
        bulge = metrics["bulge"]
        left_bulge = bulge.get("left", 0)
        right_bulge = bulge.get("right", 0)

        # 计算不对称度
        max_bulge = max(abs(left_bulge), abs(right_bulge))
        if max_bulge > 0.001:  # 有明显鼓腮动作
            bulge_diff = left_bulge - right_bulge
            bulge_asym = abs(bulge_diff) / max_bulge

            threshold = THR.BLOW_CHEEK_ASYM_THRESHOLD  # 0.10
            is_abnormal = bulge_asym > threshold

            if not is_abnormal:
                contribution = "None"
                vote_side = 0
                vote_conf = 0.0
            elif left_bulge < right_bulge:
                # 左侧鼓起更小 = 左侧面瘫
                contribution = "Left"
                vote_side = 1
                vote_conf = min(1.0, bulge_asym / threshold - 1.0 + 0.3)
            else:
                contribution = "Right"
                vote_side = 2
                vote_conf = min(1.0, bulge_asym / threshold - 1.0 + 0.3)

            result["evidence"]["cheek_bulge"] = {
                "left_bulge": float(left_bulge),
                "right_bulge": float(right_bulge),
                "asymmetry_ratio": float(bulge_asym),
                "threshold": float(threshold),
                "is_abnormal": is_abnormal,
                "contribution": contribution,
                "weight": 0.30,
                "description": f"脸颊膨胀: L={left_bulge:.3f}, R={right_bulge:.3f}, 不对称={bulge_asym:.2%}, 阈值={threshold:.2%}"
            }

            if vote_side != 0:
                votes.append({
                    "indicator": "cheek_bulge",
                    "side": vote_side,
                    "confidence": vote_conf,
                    "weight": 0.30,
                })

    # ========== 指标3: 嘴唇形状对称性 ==========
    if "lip_shape_symmetry" in metrics:
        shape_data = metrics["lip_shape_symmetry"]
        shape_asym = shape_data.get("shape_asymmetry", 0)
        palsy_suggestion = shape_data.get("palsy_suggestion", 0)

        threshold = 0.10  # 10%
        is_abnormal = shape_asym > threshold

        if not is_abnormal:
            contribution = "None"
        elif palsy_suggestion == 1:
            contribution = "Left"
        elif palsy_suggestion == 2:
            contribution = "Right"
        else:
            contribution = "None"

        result["evidence"]["lip_shape"] = {
            "left_avg_dist": float(shape_data.get("left_avg_dist", 0)),
            "right_avg_dist": float(shape_data.get("right_avg_dist", 0)),
            "asymmetry_ratio": float(shape_asym),
            "threshold": float(threshold),
            "is_abnormal": is_abnormal,
            "contribution": contribution,
            "weight": 0.20,
            "description": f"嘴唇形状不对称: {shape_asym:.2%}, 阈值: {threshold:.2%}"
        }

        if is_abnormal and palsy_suggestion != 0:
            votes.append({
                "indicator": "lip_shape",
                "side": palsy_suggestion,
                "confidence": min(1.0, shape_asym / threshold - 1.0 + 0.3),
                "weight": 0.20,
            })

    # ========== 汇总投票 ==========
    result["votes"] = votes

    if not votes:
        result["palsy_side"] = 0
        result["confidence"] = 0.0
        result["interpretation"] = "各指标均未检测到明显不对称"
        result["final_decision"] = {
            "left_score": 0.0,
            "right_score": 0.0,
            "decision_reason": "No significant asymmetry detected",
        }
        return result

    # 加权计算左右得分
    left_score = 0.0
    right_score = 0.0
    total_weight = 0.0

    for vote in votes:
        weight = vote["weight"] * vote["confidence"]
        total_weight += weight
        if vote["side"] == 1:
            left_score += weight
        elif vote["side"] == 2:
            right_score += weight

    # 归一化
    if total_weight > 0:
        left_score /= total_weight
        right_score /= total_weight

    result["final_decision"] = {
        "left_score": float(left_score),
        "right_score": float(right_score),
        "total_weight": float(total_weight),
    }

    # 决策
    if left_score > right_score * 1.2:
        result["palsy_side"] = 1
        result["confidence"] = left_score
        result["interpretation"] = f"左侧面瘫 (L={left_score:.2f} vs R={right_score:.2f})"
        result["final_decision"]["decision_reason"] = "Left score significantly higher"
    elif right_score > left_score * 1.2:
        result["palsy_side"] = 2
        result["confidence"] = right_score
        result["interpretation"] = f"右侧面瘫 (R={right_score:.2f} vs L={left_score:.2f})"
        result["final_decision"]["decision_reason"] = "Right score significantly higher"
    else:
        result["palsy_side"] = 0
        result["confidence"] = max(left_score, right_score)
        result["interpretation"] = f"无法确定 (L={left_score:.2f} vs R={right_score:.2f})"
        result["final_decision"]["decision_reason"] = "Scores too close to determine"

    return result


def compute_severity_score(metrics: Dict[str, Any]) -> Tuple[int, str]:
    """
    计算动作严重度分数 (医生标注标准)

    医生标注标准:
    - 1 = 正常 (对称性好)
    - 2 = 轻度异常
    - 3 = 中度异常
    - 4 = 重度异常
    - 5 = 完全面瘫 (严重不对称)

    计算依据: 嘴唇中线偏移量 (归一化到ICD)
    """
    lip_offset_data = metrics.get("lip_midline_offset", {})
    offset_norm = lip_offset_data.get("offset_norm", 0) or 0
    current_offset = lip_offset_data.get("current_offset", 0)

    if offset_norm < 0.03:
        return 1, f"正常 (偏移{offset_norm:.2%}, {current_offset:+.2f}px)"
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

    基于嘴部闭合程度和对称性

    评分标准:
    - 5=完整: 双侧对称且运动充分
    - 4=几乎完整: 轻度不对称或运动略有不足
    - 3=启动但不对称: 明显不对称但有运动
    - 2=轻微启动: 运动幅度很小
    - 1=无法启动: 几乎没有运动
    """
    oral = metrics.get("oral_angle", {})
    oral_asym = oral.get("asymmetry", 0)

    if baseline_landmarks is not None and "excursion_ratio" in metrics:
        # 使用运动幅度评估
        exc_ratio = metrics["excursion_ratio"]
        left_exc = metrics.get("left_excursion", 0)
        right_exc = metrics.get("right_excursion", 0)

        # 检查是否有运动
        max_exc = max(left_exc, right_exc)
        if max_exc < 2:
            return 1, "无法启动运动 (运动幅度过小)"

        # 计算对称性
        asymmetry = abs(exc_ratio - 1.0)

        if asymmetry < 0.15:
            if max_exc > 8:
                return 5, "运动完整 (对称性>85%)"
            elif max_exc > 5:
                return 4, "几乎完整"
            else:
                return 3, "启动但幅度不足"
        elif asymmetry < 0.30:
            return 3, "启动但不对称"
        elif asymmetry < 0.50:
            return 2, "轻微启动"
        else:
            return 1, "无法启动"
    else:
        # 没有基线，使用静态口角对称性
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
    """检测鼓腮时的联动运动"""
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
    """
    鼓腮可视化：只画嘴唇轮廓 + 鼓腮区域点（BLOW_CHEEK_L/R）
    """
    idx = []

    for name in ("OUTER_LIP", "INNER_LIP"):
        v = getattr(LM, name, None)
        if isinstance(v, (list, tuple)):
            idx += list(v)

    # 鼓腮区域 BLOW_CHEEK_L/R
    for name in ("BLOW_CHEEK_L", "BLOW_CHEEK_R"):
        v = getattr(LM, name, None)
        if isinstance(v, (list, tuple)):
            idx += list(v)

    # 去重 + 排序
    return sorted(set(int(i) for i in idx if isinstance(i, (int, np.integer))))


BLOW_CHEEK_VIS_INDICES = _get_blow_cheek_vis_indices()


def visualize_blow_cheek(frame, landmarks, metrics: Dict[str, Any], w: int, h: int,
                         palsy_detection: Dict[str, Any] = None):
    """可视化鼓腮指标 - 字体放大版"""
    img = frame.copy()

    # 添加患侧标签
    img = draw_palsy_side_label(img, palsy_detection, x=20, y=70, font_scale=1.4)

    # ========== 添加面中线绘制 ==========
    midline = compute_face_midline(landmarks, w, h)
    if midline:
        img = draw_face_midline(img, midline, color=(0, 255, 255), thickness=2, dashed=True)

    # 画关键点
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
    cheek = metrics.get("cheek_depth", {})
    ld = cheek.get("left_delta_norm", float("nan"))
    rd = cheek.get("right_delta_norm", float("nan"))
    md = cheek.get("mean_delta_norm", float("nan"))

    cv2.putText(img, f"CheekDepth L/R: {ld:+.4f} / {rd:+.4f}", (30, y),
                FONT, FONT_SCALE_NORMAL, (255, 255, 255), THICKNESS_NORMAL)
    y += LINE_HEIGHT
    cv2.putText(img, f"CheekDepth Mean: {md:+.4f}", (30, y),
                FONT, FONT_SCALE_NORMAL, (255, 255, 255), THICKNESS_NORMAL)
    y += LINE_HEIGHT

    oral = metrics.get("oral_angle", {})
    cv2.putText(img, f"AOE/BOF: {oral.get('AOE', 0):.2f} / {oral.get('BOF', 0):.2f}", (30, y),
                FONT, FONT_SCALE_NORMAL, (255, 255, 255), THICKNESS_NORMAL)
    y += LINE_HEIGHT + 10

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
            cv2.circle(img, (int(lip_x), int(lip_y)), 8, (0, 255, 0), -1)
            cv2.putText(img, "Lip", (int(lip_x) + 10, int(lip_y) - 5),
                        FONT, 0.5, (0, 255, 0), 2)

            if lip_center_proj is not None:
                proj_x, proj_y = lip_center_proj
                dist = abs(current_signed_dist)
                offset_color = (0, 0, 255) if dist > 10 else (0, 165, 255)

                # 画垂线
                cv2.line(img, (int(lip_x), int(lip_y)),
                         (int(proj_x), int(proj_y)), offset_color, 3)
                cv2.circle(img, (int(proj_x), int(proj_y)), 5, offset_color, -1)

                # 标注
                if dist > 3:
                    direction = "L" if current_signed_dist > 0 else "R"
                    mid_x = (int(lip_x) + int(proj_x)) // 2
                    mid_y = (int(lip_y) + int(proj_y)) // 2
                    cv2.putText(img, f"{direction} {dist:.1f}px",
                                (mid_x + 5, mid_y - 10), FONT, 0.6, offset_color, 2)

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

    # 调用绘图函数
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

    print(f"    [OK] {ACTION_NAME}: Mouth Height={metrics['mouth_height']:.1f}px")
    print(f"         Palsy: {palsy_detection.get('interpretation', 'N/A')}")
    print(f"         Voluntary Score: {result.voluntary_movement_score}/5 ({interpretation})")

    return result
