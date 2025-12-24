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
    ActionResult, draw_polygon, draw_landmarks,
    compute_scale_to_baseline,
    kabsch_rigid_transform, apply_rigid_transform,
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

def _inner_lip_area_norm(lm, w: int, h: int, icd: float) -> float:
    """
    嘴唇内圈面积（归一化到 ICD^2）
    闭嘴时该值很小；张嘴会明显变大。
    """
    pts = pts2d(lm, LM.INNER_LIP, w, h)  # 你的 clinical_base 里已经有 LM.INNER_LIP
    area = float(abs(polygon_area(pts)))
    denom = float(max(icd * icd, 1e-9))
    return area / denom


def find_peak_frame(
    landmarks_seq: List,
    frames_seq: List,   # 保留参数，但不再用它做“囤帧”
    w: int,
    h: int,
    baseline_landmarks=None,
    seal_thr: float = 0.63,          # lip_seal_total / ICD  越小越闭合
    mouth_thr: float = 0.125,        # mouth_height / ICD    越小越不张口
    smooth_win: int = 7,
    inner_area_inc_thr: float = 0.30,   #  你提出：内圈面积相对静息增幅>30% => 认为没闭嘴
    inner_area_base_eps: float = 1e-5,  # baseline面积太小防止ratio爆炸
) -> Tuple[int, Dict[str, Any]]:
    """
    鼓腮关键帧选择（最新版）：
    1) 计算左右脸颊“深度鼓起代理”曲线：delta = (baseline_z - aligned_z)/ICD
       - aligned_z：当前帧刚体对齐到baseline后，脸颊区域平均z
       - ICD归一化：跨人/跨距离更稳
    2) 门控过滤：闭唇(seal_norm<=thr) + 不张口(mouth_norm<=thr) + 嘴唇内圈面积增幅(<=thr)
    3) 打分：score = 0.6*mean_delta + 0.4*min(L,R)（兼顾整体鼓起 & 单侧塌陷）
    4) 在valid帧里取 argmax(score) 作为 peak_idx
    5) 返回 peak_debug（用于画曲线、调参、解释）
    """

    n = len(landmarks_seq)
    if n == 0:
        return 0, {}

    # --------- 小工具：兼容 dict / dataclass ----------
    def _get(obj, key, default=np.nan):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    # --------- 3D点安全读取 ----------
    def _safe_pt3d(landmarks, idx: int):
        try:
            if landmarks is None:
                return None
            if idx < 0 or idx >= len(landmarks):
                return None
            return np.asarray(pt3d(landmarks[idx], w, h), dtype=np.float64)
        except Exception:
            return None

    # --------- Kabsch刚体对齐（当前 -> baseline） ----------
    def _kabsch_rigid(P: np.ndarray, Q: np.ndarray):
        """
        求R,t，使 R@P + t ~= Q
        P,Q: (N,3)
        """
        if P.shape[0] < 3:
            return None, None
        Pc = P.mean(axis=0)
        Qc = Q.mean(axis=0)
        X = P - Pc
        Y = Q - Qc
        H = X.T @ Y
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = Qc - R @ Pc
        return R, t

    def _apply_rt(points: np.ndarray, R: np.ndarray, t: np.ndarray):
        return (points @ R.T) + t

    # --------- 区域平均z：对齐到baseline后再算 ----------
    def _mean_region_z_aligned(curr_landmarks, indices: List[int], baseline_landmarks):
        if curr_landmarks is None or baseline_landmarks is None:
            return np.nan

        # 选择稳定点做刚体对齐（你可以按你LM实际情况调整这组点）
        stable_idx = []
        for name in ["EYE_INNER_L", "EYE_INNER_R", "EYE_OUTER_L", "EYE_OUTER_R",
                     "NOSE_BRIDGE", "NOSE_TIP", "CHIN"]:
            if hasattr(LM, name):
                stable_idx.append(getattr(LM, name))

        P, Q = [], []
        for i in stable_idx:
            p = _safe_pt3d(curr_landmarks, i)
            q = _safe_pt3d(baseline_landmarks, i)
            if p is not None and q is not None:
                P.append(p); Q.append(q)

        if len(P) < 3:
            # 对齐失败就退化为直接平均当前z（不理想但保证能跑）
            zs = []
            for idx in indices:
                p = _safe_pt3d(curr_landmarks, idx)
                if p is not None:
                    zs.append(float(p[2]))
            return float(np.mean(zs)) if zs else np.nan

        P = np.asarray(P, dtype=np.float64)
        Q = np.asarray(Q, dtype=np.float64)
        R, t = _kabsch_rigid(P, Q)
        if R is None:
            zs = []
            for idx in indices:
                p = _safe_pt3d(curr_landmarks, idx)
                if p is not None:
                    zs.append(float(p[2]))
            return float(np.mean(zs)) if zs else np.nan

        region = []
        for idx in indices:
            p = _safe_pt3d(curr_landmarks, idx)
            if p is not None:
                region.append(p)
        if len(region) == 0:
            return np.nan

        region = np.asarray(region, dtype=np.float64)
        region_aligned = _apply_rt(region, R, t)
        return float(np.mean(region_aligned[:, 2]))

    # --------- 嘴唇内圈面积（归一化到ICD^2） ----------
    def _inner_lip_area_norm(lm, icd: float) -> float:
        try:
            pts = pts2d(lm, LM.INNER_LIP, w, h)
            area = float(abs(polygon_area(pts)))
            denom = float(max(icd * icd, 1e-9))
            return area / denom
        except Exception:
            return np.nan

    # --------- baseline 的脸颊区域平均z（baseline本身不需要对齐） ----------
    def _mean_region_z_baseline(baseline_landmarks, indices: List[int]):
        zs = []
        for idx in indices:
            p = _safe_pt3d(baseline_landmarks, idx)
            if p is not None:
                zs.append(float(p[2]))
        return float(np.mean(zs)) if zs else np.nan

    # =============== 逐帧计算曲线 ===============
    icd_arr = np.full(n, np.nan, dtype=np.float64)
    seal_norm_arr = np.full(n, np.nan, dtype=np.float64)
    mouth_norm_arr = np.full(n, np.nan, dtype=np.float64)

    left_delta = np.full(n, np.nan, dtype=np.float64)
    right_delta = np.full(n, np.nan, dtype=np.float64)
    mean_delta = np.full(n, np.nan, dtype=np.float64)
    score_raw = np.full(n, np.nan, dtype=np.float64)

    inner_area_norm_arr = np.full(n, np.nan, dtype=np.float64)
    inner_area_inc_arr = np.full(n, np.nan, dtype=np.float64)

    # baseline：脸颊区域z（baseline坐标系）
    base_left_z = np.nan
    base_right_z = np.nan
    base_icd = np.nan
    base_inner_area = np.nan

    if baseline_landmarks is not None:
        try:
            base_icd = float(max(compute_icd(baseline_landmarks, w, h), 1e-6))
        except Exception:
            base_icd = np.nan
        base_left_z = _mean_region_z_baseline(baseline_landmarks, LM.BLOW_CHEEK_L)
        base_right_z = _mean_region_z_baseline(baseline_landmarks, LM.BLOW_CHEEK_R)
        base_inner_area = _inner_lip_area_norm(baseline_landmarks, base_icd if np.isfinite(base_icd) else 1.0)

    # baseline 内圈面积防爆
    if not np.isfinite(base_inner_area):
        base_inner_area = inner_area_base_eps
    base_inner_area = float(max(base_inner_area, inner_area_base_eps))

    for i, lm in enumerate(landmarks_seq):
        if lm is None:
            continue

        try:
            icd = float(max(compute_icd(lm, w, h), 1e-6))
        except Exception:
            icd = np.nan
        icd_arr[i] = icd

        # 闭唇：lip seal
        try:
            seal = compute_lip_seal_distance(lm, w, h)
            seal_total = float(_get(seal, "total_distance", np.nan))
            if np.isfinite(seal_total) and np.isfinite(icd):
                seal_norm_arr[i] = seal_total / icd
        except Exception:
            pass

        # 张口：mouth height
        try:
            mouth = compute_mouth_metrics(lm, w, h)
            mouth_h = float(_get(mouth, "height", np.nan))
            if np.isfinite(mouth_h) and np.isfinite(icd):
                mouth_norm_arr[i] = mouth_h / icd
        except Exception:
            pass

        # 嘴唇内圈面积（闭嘴辅助门控）
        if np.isfinite(icd):
            inner_area_norm_arr[i] = _inner_lip_area_norm(lm, icd)
            if np.isfinite(inner_area_norm_arr[i]):
                inner_area_inc_arr[i] = (inner_area_norm_arr[i] / base_inner_area) - 1.0

        # 核心：左右脸颊深度代理（对齐到baseline）
        if baseline_landmarks is not None and np.isfinite(base_left_z) and np.isfinite(base_right_z) and np.isfinite(icd):
            lz_aligned = _mean_region_z_aligned(lm, LM.BLOW_CHEEK_L, baseline_landmarks)
            rz_aligned = _mean_region_z_aligned(lm, LM.BLOW_CHEEK_R, baseline_landmarks)

            if np.isfinite(lz_aligned):
                left_delta[i] = (base_left_z - lz_aligned) / icd
            if np.isfinite(rz_aligned):
                right_delta[i] = (base_right_z - rz_aligned) / icd

            if np.isfinite(left_delta[i]) and np.isfinite(right_delta[i]):
                mean_delta[i] = 0.5 * (left_delta[i] + right_delta[i])
                score_raw[i] = 0.6 * mean_delta[i] + 0.4 * min(left_delta[i], right_delta[i])

    # =============== 平滑 score（避免抖动） ===============
    def _smooth_nan(x: np.ndarray, win: int):
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

    score_s = _smooth_nan(score_raw, smooth_win)

    # =============== 门控 valid：闭唇 + 不张口 + 内圈面积不过大 ===============
    seal_ok = np.isfinite(seal_norm_arr) & (seal_norm_arr <= seal_thr)
    mouth_ok = np.isfinite(mouth_norm_arr) & (mouth_norm_arr <= mouth_thr)

    # 闭嘴判断：内圈面积增幅不能过大
    inner_ok = np.isfinite(inner_area_inc_arr) & (inner_area_inc_arr <= inner_area_inc_thr)

    valid = np.isfinite(score_s) & seal_ok & mouth_ok & inner_ok

    # 退化策略1：如果 valid 太少，放宽“内圈面积门控”
    if valid.sum() < 3:
        valid = np.isfinite(score_s) & seal_ok & mouth_ok

    # 退化策略2：再不行，只要求闭唇
    if valid.sum() < 1:
        valid = np.isfinite(score_s) & seal_ok

    # 退化策略3：最后兜底——选 seal_norm 最小的帧
    fallback = None
    if valid.sum() < 1:
        fallback = "min_seal_norm"
        if np.isfinite(seal_norm_arr).any():
            peak_idx = int(np.nanargmin(seal_norm_arr))
        else:
            peak_idx = 0
    else:
        cand = np.where(valid)[0]
        peak_idx = int(cand[int(np.nanargmax(score_s[cand]))])

    # =============== peak_debug：保证字段齐全 ===============
    peak_debug = {
        # 深度曲线
        "left_delta_norm": left_delta.tolist(),
        "right_delta_norm": right_delta.tolist(),
        "mean_delta_norm": mean_delta.tolist(),

        # 实际用于选峰值的score（平滑后）
        "score": score_s.tolist(),
        "score_raw": score_raw.tolist(),

        # 三个门控曲线
        "seal_norm": seal_norm_arr.tolist(),
        "mouth_norm": mouth_norm_arr.tolist(),
        "inner_lip_area_norm": inner_area_norm_arr.tolist(),
        "inner_lip_area_inc": inner_area_inc_arr.tolist(),

        # 门控阈值
        "seal_thr": float(seal_thr),
        "mouth_thr": float(mouth_thr),
        "inner_area_inc_thr": float(inner_area_inc_thr),
        "inner_area_base": float(base_inner_area),
        "smooth_win": int(smooth_win),

        # baseline信息（方便你核对符号/幅度）
        "base_left_z": float(base_left_z) if np.isfinite(base_left_z) else None,
        "base_right_z": float(base_right_z) if np.isfinite(base_right_z) else None,
        "base_icd": float(base_icd) if np.isfinite(base_icd) else None,

        # valid mask + 选帧结果
        "valid": valid.tolist(),
        "peak_idx": int(peak_idx),
        "fallback": fallback,
        "selected_by": "argmax(score)"
    }

    return peak_idx, peak_debug


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
    - 使用统一 scale
    - lip_seal: 唇封闭距离
    - mouth: 嘴部宽度等
    - oral_angle: AOE/BOF角（存成dict，避免 .get 报错）
    - cheek_area: 脸颊区域面积（归一化 + 左右差异 + 相对静息变化）
    """
    mouth = compute_mouth_metrics(landmarks, w, h)
    oral = compute_oral_angle(landmarks, w, h)
    lip_seal = compute_lip_seal_distance(landmarks, w, h)

    metrics = {
        "mouth_width": mouth["width"],
        "mouth_height": mouth["height"],
        "lip_seal": lip_seal,
        "lip_seal_total_distance": lip_seal["total_distance"],
        "oral_angle": {
            "AOE": oral.AOE_angle,
            "BOF": oral.BOF_angle,
            "asymmetry": oral.angle_asymmetry,
        }
    }

    # 计算脸颊深度
    cheek_depth = compute_cheek_depth_delta(landmarks, w, h, baseline_landmarks)
    metrics["cheek_depth"] = cheek_depth

    # 如果有基线
    if baseline_landmarks is not None:
        # ========== 计算统一 scale ==========
        scale = compute_scale_to_baseline(landmarks, baseline_landmarks, w, h)
        metrics["scale"] = scale
        # ====================================

        baseline_mouth = compute_mouth_metrics(baseline_landmarks, w, h)
        baseline_seal = compute_lip_seal_distance(baseline_landmarks, w, h)

        # ========== 缩放到 baseline 尺度 ==========
        scaled_width = mouth["width"] * scale
        scaled_seal = lip_seal["total_distance"] * scale
        # ==========================================

        metrics["baseline"] = {
            "mouth_width": baseline_mouth["width"],
            "lip_seal_total": baseline_seal["total_distance"],
        }

        metrics["seal_change"] = scaled_seal - baseline_seal["total_distance"]
        metrics["mouth_width_change"] = scaled_width - baseline_mouth["width"]

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
    peak_idx, peak_debug = find_peak_frame(landmarks_seq, frames_seq, w, h, baseline_landmarks=baseline_landmarks)
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
