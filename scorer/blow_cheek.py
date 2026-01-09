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
)
from diagnosis_visualization import add_diagnosis_overlay, draw_diagnosis_badge
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ACTION_NAME = "BlowCheek"
ACTION_NAME_CN = "鼓腮"


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


def _estimate_face_plane(landmarks, w: int, h: int):
    """用双内眦+下巴估计面部参考平面，返回 (origin, unit_normal)，并保证法向朝向鼻尖外侧。"""
    pL = _safe_pt3d(landmarks, LM.EYE_INNER_L, w, h)
    pR = _safe_pt3d(landmarks, LM.EYE_INNER_R, w, h)
    pC = _safe_pt3d(landmarks, LM.CHIN, w, h)
    pN = _safe_pt3d(landmarks, LM.NOSE_TIP, w, h)
    if pL is None or pR is None or pC is None or pN is None:
        return None, None

    v1 = pR - pL
    v2 = pC - pL
    n = np.cross(v1, v2)
    norm = float(np.linalg.norm(n))
    if norm < 1e-8:
        return None, None
    n = n / norm

    origin = (pL + pR + pC) / 3.0

    # 让法向指向“面部外侧”（鼻尖方向）
    if float(np.dot(n, (pN - origin))) < 0:
        n = -n

    return origin, n


def _tri_area_3d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))


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
    MOUTH_SEAL_THR = THR.MOUTH_SEAL

    # 辅助函数
    def _get_rel_z(lm, region_indices):
        nose_z = pt3d(lm[LM.NOSE_TIP], w, h)[2]
        region_pts_3d = [pt3d(lm[i], w, h) for i in region_indices]
        region_z = np.mean([p[2] for p in region_pts_3d])
        return region_z - nose_z

    def _get_seal_norm(lm, icd):
        upper = pt2d(lm[LM.LIP_TOP], w, h)
        lower = pt2d(lm[LM.LIP_BOT], w, h)
        return dist(upper, lower) / max(icd, 1e-9)

    # 扫描所有帧
    icd_arr, left_rel_z, right_rel_z, seal_norm = [np.nan] * n, [np.nan] * n, [np.nan] * n, [np.nan] * n
    for i, lm in enumerate(landmarks_seq):
        if lm is None: continue
        try:
            icd = compute_icd(lm, w, h)
            icd_arr[i] = icd
            left_rel_z[i] = _get_rel_z(lm, LM.BLOW_CHEEK_L)
            right_rel_z[i] = _get_rel_z(lm, LM.BLOW_CHEEK_R)
            seal_norm[i] = _get_seal_norm(lm, icd)
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
            score[i] = left_bulge[i] + right_bulge[i]

    # 门控：嘴唇闭合
    valid = np.isfinite(seal_norm) & (np.array(seal_norm) <= MOUTH_SEAL_THR)

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

    return metrics


def detect_palsy_side(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    从鼓腮动作检测面瘫侧别 - 基于嘴唇中线偏移

    核心逻辑:
    - 拟合嘴唇中线，计算相对于面中线的偏移
    - 面中线是双眼内眦连线的中垂线
    - 嘴唇被健侧肌肉拉扯，偏向健侧
    - 嘴唇偏向哪侧 → 对侧是面瘫侧

    指标优先级:
    1. 嘴唇中线偏移（最直观）
    2. 口角角度（作为验证）
    """
    result = {
        "palsy_side": 0,
        "confidence": 0.0,
        "interpretation": "",
        "method": "",
        "evidence": {}
    }

    # ========== 方法1（优先）: 嘴唇中线偏移 ==========
    if "lip_midline_offset" in metrics:
        offset_data = metrics["lip_midline_offset"]

        current_offset = offset_data.get("current_offset", 0)
        face_midline_x = offset_data.get("face_midline_x", 0)
        lip_midline_x = offset_data.get("lip_midline_x", 0)

        result["evidence"]["face_midline_x"] = face_midline_x
        result["evidence"]["lip_midline_x"] = lip_midline_x
        result["evidence"]["current_offset"] = current_offset

        # 计算归一化偏移
        icd = metrics.get("icd", 1)
        if icd < 1e-6:
            icd = 1
        offset_norm = abs(current_offset) / icd
        result["evidence"]["offset_norm"] = offset_norm

        # 判断阈值：偏移超过ICD的2.5%
        if offset_norm > 0.025:
            result["method"] = "lip_midline_offset"
            result["confidence"] = min(1.0, offset_norm * 5)

            if current_offset > 0:
                # 嘴唇中线偏向左侧（图像右侧）= 被左侧拉 = 右侧面瘫
                result["palsy_side"] = 2
                result["interpretation"] = (
                    f"嘴唇中线偏向左侧 ({current_offset:+.1f}px, {offset_norm:.4%}) → 右侧面瘫"
                )
            else:
                # 嘴唇中线偏向右侧（图像左侧）= 被右侧拉 = 左侧面瘫
                result["palsy_side"] = 1
                result["interpretation"] = (
                    f"嘴唇中线偏向右侧 ({current_offset:+.1f}px, {offset_norm:.4%}) → 左侧面瘫"
                )
            return result

    # 未检测到明显不对称
    result["method"] = "none"
    result["interpretation"] = (f"各指标均未检测到明显不对称, {offset_norm:.4%}")
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

    # 字体参数
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE_TITLE = 1.4
    FONT_SCALE_NORMAL = 0.9
    THICKNESS_TITLE = 3
    THICKNESS_NORMAL = 2
    LINE_HEIGHT = 50

    # 患侧标签
    img = draw_palsy_side_label(img, palsy_detection, x=20, y=70, font_scale=1.4)

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

    # ========== 绘制嘴唇对称性证据 ==========
    if "lip_symmetry" in metrics:
        lip_sym = metrics["lip_symmetry"]
        cv2.putText(img, "=== Lip Symmetry Evidence ===", (30, y),
                    FONT, FONT_SCALE_NORMAL, (0, 255, 255), THICKNESS_NORMAL)
        y += LINE_HEIGHT

        left_dist = lip_sym["left_to_midline"]
        right_dist = lip_sym["right_to_midline"]
        lip_offset = lip_sym["lip_offset"]
        asymmetry_ratio = lip_sym["asymmetry_ratio"]

        # 左侧距离（距离大的用绿色表示健侧拉力强）
        left_color = (0, 255, 0) if left_dist >= right_dist else (0, 0, 255)
        cv2.putText(img, f"L to midline: {left_dist:.1f}px", (30, y),
                    FONT, FONT_SCALE_NORMAL, left_color, THICKNESS_NORMAL)
        y += LINE_HEIGHT

        # 右侧距离
        right_color = (0, 255, 0) if right_dist >= left_dist else (0, 0, 255)
        cv2.putText(img, f"R to midline: {right_dist:.1f}px", (30, y),
                    FONT, FONT_SCALE_NORMAL, right_color, THICKNESS_NORMAL)
        y += LINE_HEIGHT

        # 偏移方向
        direction = "Left" if lip_offset > 0 else "Right"
        cv2.putText(img, f"Lip Offset: {lip_offset:+.1f}px ({direction})", (30, y),
                    FONT, FONT_SCALE_NORMAL, (255, 255, 255), THICKNESS_NORMAL)
        y += LINE_HEIGHT

        # 不对称比例
        cv2.putText(img, f"Asymmetry: {asymmetry_ratio * 100:.1f}%", (30, y),
                    FONT, FONT_SCALE_NORMAL, (255, 255, 255), THICKNESS_NORMAL)
        y += LINE_HEIGHT

        # ========== 绘制面中线和嘴唇中线 ==========
        if "lip_midline_offset" in metrics:
            offset_data = metrics["lip_midline_offset"]
            face_midline_x = offset_data.get("face_midline_x", None)
            lip_midline_x = offset_data.get("lip_midline_x", None)
            lip_midline_y = offset_data.get("lip_midline_y", None)

            if face_midline_x is not None:
                face_midline_x_int = int(face_midline_x)

                # 面中线的起点和终点（从眉间到下巴）
                # 获取内眦y坐标作为参考
                left_canthus = pt2d(landmarks[LM.EYE_INNER_L], w, h)
                right_canthus = pt2d(landmarks[LM.EYE_INNER_R], w, h)
                eye_y = int((left_canthus[1] + right_canthus[1]) / 2)

                midline_start_y = max(20, eye_y - 80)  # 从眼睛上方开始
                midline_end_y = min(h - 20, eye_y + 300)  # 到下巴位置

                # 绘制面中线（青色虚线）
                for yy in range(midline_start_y, midline_end_y, 15):
                    cv2.line(img, (face_midline_x_int, yy),
                             (face_midline_x_int, min(yy + 8, midline_end_y)),
                             (255, 255, 0), 2)
                cv2.putText(img, "Face Mid", (face_midline_x_int + 5, midline_start_y + 20),
                            FONT, 0.5, (255, 255, 0), 2)

            # ========== 绘制嘴唇中线 ==========
            if lip_midline_x is not None and lip_midline_y is not None:
                lip_midline_x_int = int(lip_midline_x)
                lip_midline_y_int = int(lip_midline_y)

                # 获取嘴唇中线的四个点
                lip_top_center = pt2d(landmarks[LM.LIP_TOP_CENTER], w, h)
                lip_top = pt2d(landmarks[LM.LIP_TOP], w, h)
                lip_bot = pt2d(landmarks[LM.LIP_BOT], w, h)
                lip_bot_center = pt2d(landmarks[LM.LIP_BOT_CENTER], w, h)

                # 绘制嘴唇中线（绿色实线）
                lip_midline_pts = np.array([
                    [int(lip_top_center[0]), int(lip_top_center[1])],
                    [int(lip_top[0]), int(lip_top[1])],
                    [int(lip_bot[0]), int(lip_bot[1])],
                    [int(lip_bot_center[0]), int(lip_bot_center[1])]
                ], dtype=np.int32)
                cv2.polylines(img, [lip_midline_pts], False, (0, 255, 0), 3)

                # 绘制嘴唇中线中心点
                cv2.circle(img, (lip_midline_x_int, lip_midline_y_int), 8, (0, 255, 0), -1)
                cv2.putText(img, "Lip Mid", (lip_midline_x_int + 10, lip_midline_y_int),
                            FONT, 0.5, (0, 255, 0), 2)

                # ========== 绘制偏移指示线 ==========
                if face_midline_x is not None:
                    face_midline_x_int = int(face_midline_x)
                    offset = lip_midline_x - face_midline_x
                    offset_color = (0, 0, 255) if abs(offset) > 10 else (0, 255, 255)
                    cv2.line(img, (lip_midline_x_int, lip_midline_y_int),
                             (face_midline_x_int, lip_midline_y_int), offset_color, 3)

                    # 标注偏移值
                    mid_x = (lip_midline_x_int + face_midline_x_int) // 2
                    direction = "L" if offset > 0 else "R"
                    cv2.putText(img, f"{abs(offset):.0f}px({direction})",
                                (mid_x - 40, lip_midline_y_int - 15),
                                FONT, 0.6, offset_color, 2)

        elif "midline_x" in metrics:
            # 回退到旧的简化版
            midline_x = int(metrics["midline_x"])
            for yy in range(0, h, 10):
                cv2.line(img, (midline_x, yy), (midline_x, min(yy + 5, h)), (0, 255, 255), 1)
            cv2.putText(img, "Mid", (midline_x + 5, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # ========== 绘制嘴角到中线的连线（保留原功能作为补充） ==========
        if "left_corner" in metrics and "lip_midline_offset" not in metrics:
            if "midline_x" in metrics:
                left_corner = metrics["left_corner"]
                right_corner = metrics["right_corner"]
                midline_x = int(metrics["midline_x"])

                cv2.line(img, (int(left_corner[0]), int(left_corner[1])),
                         (midline_x, int(left_corner[1])), (255, 0, 0), 2)
                cv2.line(img, (int(right_corner[0]), int(right_corner[1])),
                         (midline_x, int(right_corner[1])), (0, 165, 255), 2)

    # 面瘫侧别
    if palsy_detection:
        palsy_side = palsy_detection.get("palsy_side", 0)
        palsy_text = {0: "Symmetric", 1: "Left Palsy", 2: "Right Palsy"}.get(palsy_side, "Unknown")
        palsy_color = (0, 255, 0) if palsy_side == 0 else (0, 0, 255)
        cv2.putText(img, f"Palsy: {palsy_text}", (30, y), FONT, FONT_SCALE_NORMAL, palsy_color, THICKNESS_NORMAL)

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
