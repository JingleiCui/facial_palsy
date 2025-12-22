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

from clinical_base import (
    LM, pt2d, pt3d, pts2d, dist, compute_ear, compute_eye_area,
    compute_mouth_metrics, compute_oral_angle,
    compute_icd, polygon_area, extract_common_indicators, compute_lip_seal_distance,
    ActionResult, draw_polygon, draw_landmarks
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACTION_NAME = "BlowCheek"
ACTION_NAME_CN = "鼓腮"


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


def _kabsch_rigid(P: np.ndarray, Q: np.ndarray):
    """
    求刚体变换：把 P(当前帧稳定点) 对齐到 Q(基线稳定点)
    返回 R(3x3), t(3,)
    """
    if P is None or Q is None:
        return None, None
    if P.shape[0] < 3 or Q.shape[0] < 3:
        return None, None

    Pc = P.mean(axis=0)
    Qc = Q.mean(axis=0)
    X = P - Pc
    Y = Q - Qc

    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # 防止反射
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = Qc - (R @ Pc)
    return R, t


def _apply_rt(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """对 (N,3) 点集应用刚体变换"""
    return (R @ points.T).T + t

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
            P.append(p); Q.append(q)

    if len(P) < 3 or len(Q) < 3:
        return _mean_region_z_aligned(landmarks, indices, w, h, baseline_landmarks=None)

    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    R, t = _kabsch_rigid(P, Q)
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
    region_aligned = _apply_rt(region, R, t)
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

    if not np.isfinite(base_left_z) or not np.isfinite(base_right_z) or not np.isfinite(left_z_aligned) or not np.isfinite(right_z_aligned):
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


def _save_cheek_depth_curve_png(png_path: Path,
                                left_series: List[float],
                                right_series: List[float],
                                mean_series: List[float],
                                peak_idx: int) -> None:
    """输出左右/平均脸颊深度曲线，并标注峰值帧"""
    if not left_series or not right_series or not mean_series:
        return

    xs = list(range(len(mean_series)))
    plt.figure()
    plt.plot(xs, left_series, label="Left cheek (delta_z/ICD)")
    plt.plot(xs, right_series, label="Right cheek (delta_z/ICD)")
    plt.plot(xs, mean_series, label="Mean (delta_z/ICD)")

    peak_idx = int(max(0, min(peak_idx, len(xs) - 1)))
    plt.axvline(peak_idx, linestyle="--")
    plt.scatter([peak_idx], [mean_series[peak_idx]])

    plt.title("BlowCheek depth curve: (baseline_z - aligned_z) / ICD")
    plt.xlabel("Frame")
    plt.ylabel("Normalized depth delta (bigger = more bulge)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(png_path), dpi=160)  # PNG，无质量压缩那种损失
    plt.close()


def find_peak_frame(
    landmarks_seq, w: int, h: int,
    baseline_landmarks=None,
    seal_thr: float = 0.03,      # lip_seal_total / ICD   （你日志里现在用的）
    mouth_thr: float = 0.025,    # mouth_height / ICD
    smooth_win: int = 7
) -> Tuple[int, Dict[str, Any]]:
    """
    鼓腮峰值帧（更符合你要的“鼓起来=更靠近镜头”）：
    1) 闭唇门控 + 不张口门控（防止“张嘴被选中”）
    2) 左右脸颊分别算深度代理：delta=(base_z - aligned_z)/ICD，越大越鼓
    3) 峰值打分：0.6*mean + 0.4*min(L,R) （兼顾“整体鼓起”与“单侧面瘫”）
    4) 输出debug曲线供可解释性
    """
    n = len(landmarks_seq)
    if n == 0:
        return 0, {}

    icd_arr = np.full(n, np.nan, dtype=np.float64)
    seal_arr = np.full(n, np.nan, dtype=np.float64)
    mouth_h_arr = np.full(n, np.nan, dtype=np.float64)

    left_arr = np.full(n, np.nan, dtype=np.float64)   # left delta_z/ICD
    right_arr = np.full(n, np.nan, dtype=np.float64)  # right delta_z/ICD
    mean_arr = np.full(n, np.nan, dtype=np.float64)   # mean delta_z/ICD
    score_arr = np.full(n, np.nan, dtype=np.float64)

    for i, lm in enumerate(landmarks_seq):
        if lm is None:
            continue

        # ICD 用 baseline 的更合理，但为了门控这里用当前帧也行
        try:
            icd = float(compute_icd(lm, w, h))
        except Exception:
            icd = np.nan
        icd = max(icd, 1e-6) if np.isfinite(icd) else np.nan
        icd_arr[i] = icd

        try:
            seal_arr[i] = float(compute_lip_seal_distance(lm, w, h)["total_distance"])
        except Exception:
            pass

        try:
            mouth_h_arr[i] = float(compute_mouth_metrics(lm, w, h)["height"])
        except Exception:
            pass

        # 左右脸颊深度代理（核心）
        d = compute_cheek_depth_delta(lm, w, h, baseline_landmarks=baseline_landmarks)
        if d.get("left_delta_norm") is not None:
            left_arr[i] = float(d["left_delta_norm"])
        if d.get("right_delta_norm") is not None:
            right_arr[i] = float(d["right_delta_norm"])
        if d.get("mean_delta_norm") is not None:
            mean_arr[i] = float(d["mean_delta_norm"])

        if np.isfinite(left_arr[i]) and np.isfinite(right_arr[i]):
            score_arr[i] = 0.6 * mean_arr[i] + 0.4 * min(left_arr[i], right_arr[i])

    # 平滑（滑动平均，忽略nan）
    def smooth(x, win):
        if win <= 1:
            return x.copy()
        y = np.full_like(x, np.nan, dtype=np.float64)
        half = win // 2
        for k in range(len(x)):
            a = max(0, k - half)
            b = min(len(x), k + half + 1)
            seg = x[a:b]
            if np.isfinite(seg).any():
                y[k] = float(np.nanmean(seg))
        return y

    score_s = smooth(score_arr, smooth_win)

    # 门控：必须闭唇 + 不能张口
    seal_ok = (seal_arr / np.maximum(icd_arr, 1e-6)) <= seal_thr
    mouth_ok = (mouth_h_arr / np.maximum(icd_arr, 1e-6)) <= mouth_thr

    valid = np.isfinite(score_s) & np.isfinite(seal_arr) & np.isfinite(mouth_h_arr) & seal_ok & mouth_ok

    # 如果门控过严：退化为只闭唇
    if valid.sum() < 3:
        valid = np.isfinite(score_s) & np.isfinite(seal_arr) & seal_ok

    # 再不行：选“唇封闭最好”那一帧
    if valid.sum() == 0:
        idx = int(np.nanargmin(seal_arr)) if np.isfinite(seal_arr).sum() > 0 else 0
        debug = {
            "left_delta_norm": left_arr.tolist(),
            "right_delta_norm": right_arr.tolist(),
            "mean_delta_norm": mean_arr.tolist(),
            "score": score_s.tolist(),
            "seal_norm": (seal_arr / np.maximum(icd_arr, 1e-6)).tolist(),
            "mouth_norm": (mouth_h_arr / np.maximum(icd_arr, 1e-6)).tolist(),
            "valid": valid.tolist(),
            "fallback": "min_lip_seal"
        }
        return idx, debug

    candidates = np.where(valid)[0]
    best = candidates[int(np.nanargmax(score_s[candidates]))]

    debug = {
        "left_delta_norm": left_arr.tolist(),
        "right_delta_norm": right_arr.tolist(),
        "mean_delta_norm": mean_arr.tolist(),
        "score": score_s.tolist(),
        "seal_norm": (seal_arr / np.maximum(icd_arr, 1e-6)).tolist(),
        "mouth_norm": (mouth_h_arr / np.maximum(icd_arr, 1e-6)).tolist(),
        "valid": valid.tolist(),
        "seal_thr": float(seal_thr),
        "mouth_thr": float(mouth_thr),
        "smooth_win": int(smooth_win),
        "fallback": None
    }
    return int(best), debug


def _smooth_1d_nan(x: np.ndarray, win: int = 5) -> np.ndarray:
    """对一维序列做简单滑动平均（忽略 NaN）。"""
    if win <= 1:
        return x.copy()
    n = len(x)
    y = np.full(n, np.nan, dtype=np.float64)
    half = win // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        seg = x[lo:hi]
        if np.all(np.isnan(seg)):
            continue
        y[i] = np.nanmean(seg)
    return y

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
    鼓腮动作指标：
    - lip_seal: 唇封闭距离
    - mouth: 嘴部宽度等
    - oral_angle: AOE/BOF角（存成dict，避免 .get 报错）
    - cheek_area: 脸颊区域面积（归一化 + 左右差异 + 相对静息变化）
    """
    lip_seal = compute_lip_seal_distance(landmarks, w, h)
    mouth = compute_mouth_metrics(landmarks, w, h)

    oral_obj = compute_oral_angle(landmarks, w, h)
    oral = {
        "AOE": float(oral_obj.AOE_angle),
        "BOF": float(oral_obj.BOF_angle),
        "asymmetry": float(oral_obj.angle_asymmetry),
        "diff": float(oral_obj.angle_diff),
    }

    cheek_depth = compute_cheek_depth_delta(landmarks, w, h, baseline_landmarks=baseline_landmarks)

    metrics: Dict[str, Any] = {
        "lip_seal": lip_seal,
        "mouth_width": float(mouth["width"]),
        "mouth_height": float(mouth["height"]),
        "lip_seal_total_distance": float(lip_seal["total_distance"]),
        "oral_angle": oral,
        "cheek_depth": cheek_depth,
    }

    if baseline_landmarks is not None:
        base_lip = compute_lip_seal_distance(baseline_landmarks, w, h)
        base_mouth = compute_mouth_metrics(baseline_landmarks, w, h)
        base_oral_obj = compute_oral_angle(baseline_landmarks, w, h)
        base_oral = {
            "AOE": float(base_oral_obj.AOE_angle),
            "BOF": float(base_oral_obj.BOF_angle),
            "asymmetry": float(base_oral_obj.angle_asymmetry),
            "diff": float(base_oral_obj.angle_diff),
        }

        corner_shift = abs(mouth["left_corner"][0] - base_mouth["left_corner"][0]) + \
                       abs(mouth["right_corner"][0] - base_mouth["right_corner"][0])

        metrics["baseline"] = {
            "lip_seal_total_distance": float(base_lip["total_distance"]),
            "mouth_width": float(base_mouth["width"]),
            "oral_asymmetry": float(base_oral["asymmetry"]),
            "corner_shift": float(corner_shift),
            "cheek_depth_baseline": {
                "base_left_z": cheek_depth.get("base_left_z"),
                "base_right_z": cheek_depth.get("base_right_z"),
            },
        }

    return metrics


def detect_palsy_side(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    从鼓腮动作检测面瘫侧别

    原理:
    1. 面瘫侧嘴角运动幅度小
    2. 口角角度不对称

    Returns:
        Dict包含:
        - palsy_side: 0=无/对称, 1=左, 2=右
        - confidence: 置信度
        - interpretation: 解释
    """
    result = {"palsy_side": 0, "confidence": 0.0, "interpretation": ""}

    # 使用运动幅度作为主要指标
    if "left_excursion" in metrics and "right_excursion" in metrics:
        left_exc = metrics["left_excursion"]
        right_exc = metrics["right_excursion"]

        if max(left_exc, right_exc) < 2:  # 运动幅度太小
            # 检查口角角度
            oral = metrics.get("oral_angle", {})
            asym = oral.get("asymmetry", 0)
            if asym < 3:
                result["interpretation"] = "双侧对称"
            else:
                result["interpretation"] = "运动幅度过小，难以判断"
            return result

        exc_ratio = metrics["excursion_ratio"]

        if abs(exc_ratio - 1.0) < 0.15:
            result["palsy_side"] = 0
            result["confidence"] = 1.0 - abs(exc_ratio - 1.0)
            result["interpretation"] = f"双侧运动对称 (比值={exc_ratio:.2f})"
        elif left_exc < right_exc:
            result["palsy_side"] = 1
            result["confidence"] = min(1.0, abs(exc_ratio - 1.0))
            result["interpretation"] = f"左侧运动较弱 (L={left_exc:.1f}px < R={right_exc:.1f}px)"
        else:
            result["palsy_side"] = 2
            result["confidence"] = min(1.0, abs(exc_ratio - 1.0))
            result["interpretation"] = f"右侧运动较弱 (R={right_exc:.1f}px < L={left_exc:.1f}px)"
    else:
        # 没有基线，使用口角对称性
        oral = metrics.get("oral_angle", {})
        asym = oral.get("asymmetry", 0)
        aoe = oral.get("AOE", 0)
        bof = oral.get("BOF", 0)

        if asym < 3:
            result["palsy_side"] = 0
            result["confidence"] = 1.0 - asym / 10
            result["interpretation"] = f"口角对称 (不对称度={asym:.1f}°)"
        elif aoe < bof:
            result["palsy_side"] = 2
            result["confidence"] = min(1.0, asym / 15)
            result["interpretation"] = f"右口角位置较低 (AOE={aoe:+.1f}°)"
        else:
            result["palsy_side"] = 1
            result["confidence"] = min(1.0, asym / 15)
            result["interpretation"] = f"左口角位置较低 (BOF={bof:+.1f}°)"

    return result


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

def visualize_blow_cheek(frame, landmarks, metrics: Dict[str, Any], w: int, h: int):
    img = frame.copy()

    # 画全脸关键点（你原来就有）
    draw_landmarks(img, landmarks, w, h, BLOW_CHEEK_VIS_INDICES, radius=1)

    # 嘴唇/封闭点
    lip_seal = metrics.get("lip_seal", {})
    upper = lip_seal.get("upper_lip_point", None)
    lower = lip_seal.get("lower_lip_point", None)
    left_corner = lip_seal.get("left_corner", None)
    right_corner = lip_seal.get("right_corner", None)

    if upper is not None and lower is not None:
        cv2.line(img, tuple(upper), tuple(lower), (255, 255, 255), 2)
    if left_corner is not None and right_corner is not None:
        cv2.line(img, tuple(left_corner), tuple(right_corner), (255, 255, 255), 2)

    # 画左右脸颊区域（BLOW_CHEEK_L/R）
    left_idx, right_idx, region = _get_blow_cheek_polygons()
    if left_idx is not None:
        left_pts = pts2d(landmarks, left_idx, w, h).astype(np.int32)
        right_pts = pts2d(landmarks, right_idx, w, h).astype(np.int32)

        overlay = img.copy()
        cv2.fillPoly(overlay, [left_pts], (0, 255, 255))   # 黄
        cv2.fillPoly(overlay, [right_pts], (255, 255, 0))  # 青
        img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)

        cv2.polylines(img, [left_pts], True, (0, 255, 255), 2)
        cv2.polylines(img, [right_pts], True, (255, 255, 0), 2)

        cv2.putText(img, f"CheekRegion: {region}", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 写指标文字
    cheek = metrics.get("cheek_depth", {})
    ld = cheek.get("left_delta_norm", float("nan"))
    rd = cheek.get("right_delta_norm", float("nan"))
    md = cheek.get("mean_delta_norm", float("nan"))

    cv2.putText(img, f"CheekDepthΔ/ICD L/R: {ld:+.4f} / {rd:+.4f}", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"CheekDepthΔ/ICD Mean: {md:+.4f}", (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"LipSealDist: {metrics.get('lip_seal_total_distance', 0):.2f}px", (30, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    oral = metrics.get("oral_angle", {})
    cv2.putText(img, f"AOE/BOF: {oral.get('AOE', 0):.2f} / {oral.get('BOF', 0):.2f}", (30, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return img


def process(landmarks_seq: List, frames_seq: List, w: int, h: int,
            video_info: Dict[str, Any], output_dir: Path,
            baseline_result: Optional[ActionResult] = None,
            baseline_landmarks=None) -> Optional[ActionResult]:
    """处理BlowCheek动作"""
    if not landmarks_seq or not frames_seq:
        return None

    # 找峰值帧
    peak_idx, peak_debug = find_peak_frame(
        landmarks_seq, w, h,
        baseline_landmarks=baseline_landmarks,
        seal_thr=0.13,
        mouth_thr=0.125,
        smooth_win=7
    )
    peak_landmarks = landmarks_seq[peak_idx]
    peak_frame = frames_seq[peak_idx]

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

    # 计算鼓腮特有指标
    metrics = compute_blow_cheek_metrics(peak_landmarks, w, h, baseline_landmarks)

    # 检测面瘫侧别
    palsy_detection = detect_palsy_side(metrics)

    # 计算Voluntary Movement评分
    score, interpretation = compute_voluntary_score(metrics, baseline_landmarks)
    result.voluntary_movement_score = score

    # 检测联动
    synkinesis = detect_synkinesis(baseline_result, peak_landmarks, w, h)
    result.synkinesis_scores = synkinesis


    result.action_specific = {
        "lip_seal": metrics["lip_seal"],
        "mouth_width": metrics["mouth_width"],
        "mouth_height": metrics["mouth_height"],
        "oral_angle": metrics["oral_angle"],
        "cheek_depth": metrics.get("cheek_depth", {}),
    }

    if "baseline" in metrics:
        result.action_specific["baseline"] = metrics["baseline"]
        result.action_specific["changes"] = {
            "seal_change": metrics.get("seal_change", 0),
            "mouth_width_change": metrics.get("mouth_width_change", 0),
        }

    # 创建输出目录
    action_dir = output_dir / ACTION_NAME
    action_dir.mkdir(parents=True, exist_ok=True)

    # 保存原始帧
    cv2.imwrite(str(action_dir / "peak_raw.jpg"), peak_frame)

    # 保存可视化
    vis = visualize_blow_cheek(peak_frame, peak_landmarks, metrics, w, h)
    cv2.imwrite(str(action_dir / "peak_indicators.jpg"), vis)

    # 保存左右脸颊深度曲线（可解释性）
    if peak_debug:
        left_series = peak_debug.get("left_delta_norm", [])
        right_series = peak_debug.get("right_delta_norm", [])
        mean_series = peak_debug.get("mean_delta_norm", [])
        _save_cheek_depth_curve_png(action_dir / "cheek_depth_curve.png",
                                    left_series, right_series, mean_series, peak_idx)

    # 写入 debug，方便你回看为什么选中这帧
    if result.action_specific is None:
        result.action_specific = {}
    result.action_specific["peak_debug"] = peak_debug

    # 保存JSON
    with open(action_dir / "indicators.json", 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"    [OK] {ACTION_NAME}: Lip Seal={metrics['lip_seal']['total_distance']:.1f}px")
    print(f"         Palsy: {palsy_detection.get('interpretation', 'N/A')}")
    print(f"         Voluntary Score: {result.voluntary_movement_score}/5 ({interpretation})")

    return result
