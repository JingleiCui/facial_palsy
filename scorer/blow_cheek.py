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
import numpy as np
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


def _volume_proxy_from_polygon_3d(poly_pts: np.ndarray, origin: np.ndarray, normal: np.ndarray) -> float:
    """
    poly_pts: (N,3) 3D多边形点（按轮廓顺序）
    体积代理：sum( A_tri * mean(pos_dist_to_plane) )
    """
    n = poly_pts.shape[0]
    if n < 3:
        return float("nan")

    p0 = poly_pts[0]
    vol = 0.0
    for i in range(1, n - 1):
        a, b, c = p0, poly_pts[i], poly_pts[i + 1]
        A = _tri_area_3d(a, b, c)
        d1 = float(np.dot((a - origin), normal))
        d2 = float(np.dot((b - origin), normal))
        d3 = float(np.dot((c - origin), normal))
        # 只取向外鼓出的部分（正值）
        d_mean = (max(d1, 0.0) + max(d2, 0.0) + max(d3, 0.0)) / 3.0
        vol += A * d_mean
    return float(vol)


def compute_cheek_volume(landmarks, w: int, h: int, baseline_landmarks=None):
    """
    鼓腮体积代理（左右分开 + 总分），并做 ICD^3 归一化，方便跨帧/跨视频稳定。
    """
    left_idx = getattr(LM, "BLOW_CHEEK_L", None)
    right_idx = getattr(LM, "BLOW_CHEEK_R", None)
    if not isinstance(left_idx, (list, tuple)) or not isinstance(right_idx, (list, tuple)):
        return {"score": None}

    origin, normal = _estimate_face_plane(landmarks, w, h)
    if origin is None:
        return {"score": None}

    def gather(indices):
        pts = []
        for idx in indices:
            p = _safe_pt3d(landmarks, int(idx), w, h)
            if p is not None:
                pts.append(p)
        if len(pts) < 3:
            return None
        return np.stack(pts, axis=0)

    L = gather(left_idx)
    R = gather(right_idx)
    if L is None or R is None:
        return {"score": None}

    volL = _volume_proxy_from_polygon_3d(L, origin, normal)
    volR = _volume_proxy_from_polygon_3d(R, origin, normal)

    icd = float(compute_icd(landmarks, w, h))
    denom = max(icd, 1e-6) ** 3  # 体积归一化
    volL_n = volL / denom
    volR_n = volR / denom

    score = 0.5 * (volL_n + volR_n)
    asym = abs(volL_n - volR_n)

    out = {
        "left_vol": volL, "right_vol": volR,
        "left_norm": volL_n, "right_norm": volR_n,
        "score": score, "asymmetry": asym,
    }

    if baseline_landmarks is not None:
        base = compute_cheek_volume(baseline_landmarks, w, h, baseline_landmarks=None)
        if base.get("score") is not None:
            out["baseline_score"] = base["score"]
            out["score_change"] = out["score"] - base["score"]
            out["left_change"] = out["left_norm"] - base.get("left_norm", 0.0)
            out["right_change"] = out["right_norm"] - base.get("right_norm", 0.0)

    return out


def find_peak_frame(
    landmarks_seq, w: int, h: int,
    baseline_landmarks=None,
    seal_thr: float = 0.03,      # lip_seal_total / ICD
    mouth_thr: float = 0.025,    # mouth_height / ICD
    smooth_win: int = 7
) -> int:
    """
    鼓腮峰值帧：强制闭唇门控 + 3D体积代理Δ最大
    - seal_thr/mouth_thr 是关键：防止“嘴张开但被选中”
    """
    n = len(landmarks_seq)
    if n == 0:
        return 0

    icd_arr = np.full(n, np.nan, dtype=np.float64)
    seal_arr = np.full(n, np.nan, dtype=np.float64)
    mouth_h_arr = np.full(n, np.nan, dtype=np.float64)
    vol_arr = np.full(n, np.nan, dtype=np.float64)

    for i, lm in enumerate(landmarks_seq):
        if lm is None:
            continue
        icd = float(compute_icd(lm, w, h))
        icd_arr[i] = icd

        try:
            seal_arr[i] = float(compute_lip_seal_distance(lm, w, h)["total_distance"])
        except Exception:
            pass

        try:
            mouth_h_arr[i] = float(compute_mouth_metrics(lm, w, h)["height"])
        except Exception:
            pass

        try:
            m = compute_cheek_volume(lm, w, h, baseline_landmarks=None)
            vol_arr[i] = float(m["score"]) if m.get("score") is not None else np.nan
        except Exception:
            pass

    # baseline：优先使用 NeutralFace 的 baseline_landmarks；否则用前10%
    if baseline_landmarks is not None:
        base = compute_cheek_volume(baseline_landmarks, w, h, baseline_landmarks=None)
        base_score = float(base["score"]) if base.get("score") is not None else float(np.nan)
    else:
        k = max(3, int(0.1 * n))
        base_score = float(np.nanmedian(vol_arr[:k]))

    delta = vol_arr - base_score

    # 简单平滑（滑动平均，忽略nan）
    def smooth(x, win):
        if win <= 1:
            return x
        y = np.full_like(x, np.nan, dtype=np.float64)
        half = win // 2
        for i in range(len(x)):
            lo = max(0, i - half)
            hi = min(len(x), i + half + 1)
            seg = x[lo:hi]
            if np.all(np.isnan(seg)):
                continue
            y[i] = np.nanmean(seg)
        return y

    delta_s = smooth(delta, smooth_win)

    # 门控：必须闭唇 + 口高不能大（防止张嘴）
    seal_ok = (seal_arr / np.maximum(icd_arr, 1e-6)) <= seal_thr
    mouth_ok = (mouth_h_arr / np.maximum(icd_arr, 1e-6)) <= mouth_thr
    valid = np.isfinite(delta_s) & np.isfinite(seal_arr) & np.isfinite(mouth_h_arr) & seal_ok & mouth_ok

    if valid.sum() < 3:
        # 门控太严格时：放宽到只要求闭唇（仍然防张嘴）
        valid = np.isfinite(delta_s) & np.isfinite(seal_arr) & seal_ok

    if valid.sum() == 0:
        # 再不行就回退：找唇封闭最好的那帧
        idx = int(np.nanargmin(seal_arr)) if np.isfinite(seal_arr).sum() > 0 else 0
        return idx

    candidates = np.where(valid)[0]
    best = candidates[int(np.nanargmax(delta_s[candidates]))]
    return int(best)


def compute_cheek_bulge(landmarks, w: int, h: int,
                               baseline_landmarks=None) -> Dict[str, Any]:
    """计算“鼓腮凸出/体积代理”指标（基于 3D 点到面部参考平面的投影距离）。

    思路：
    - 用(双内眦 + 下巴)估计面部参考平面；
    - 计算左右面颊点( LM.CHEEK_L / LM.CHEEK_R )沿平面法向的投影距离；
    - 距离越大 -> 面颊越“鼓”。
    """
    centroid, n = _estimate_face_plane(landmarks, w, h)
    if centroid is None or n is None:
        return {"score": None, "left_mean": None, "right_mean": None, "asymmetry": None}

    left_pts = []
    right_pts = []
    for idx in LM.BLOW_CHEEK_L:
        p = _safe_pt3d(landmarks, idx, w, h)
        if p is not None:
            left_pts.append(p)
    for idx in LM.BLOW_CHEEK_R:
        p = _safe_pt3d(landmarks, idx, w, h)
        if p is not None:
            right_pts.append(p)

    if len(left_pts) == 0 or len(right_pts) == 0:
        return {"score": None, "left_mean": None, "right_mean": None, "asymmetry": None}

    left_pts = np.stack(left_pts, axis=0)
    right_pts = np.stack(right_pts, axis=0)

    left_d = (left_pts - centroid) @ n
    right_d = (right_pts - centroid) @ n

    left_mean = float(np.mean(left_d))
    right_mean = float(np.mean(right_d))
    score = float(0.5 * (left_mean + right_mean))
    asym = float(abs(left_mean - right_mean))

    out = {
        "score": score,
        "left_mean": left_mean,
        "right_mean": right_mean,
        "asymmetry": asym,
    }

    if baseline_landmarks is not None:
        base = compute_cheek_bulge(baseline_landmarks, w, h, baseline_landmarks=None)
        if base.get("score") is not None and out["score"] is not None:
            out["baseline_score"] = base["score"]
            out["score_change"] = out["score"] - base["score"]
            out["left_change"] = out["left_mean"] - base.get("left_mean", 0.0)
            out["right_change"] = out["right_mean"] - base.get("right_mean", 0.0)

    return out


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

    cheek_volume = compute_cheek_volume(landmarks, w, h, baseline_landmarks=baseline_landmarks)

    metrics: Dict[str, Any] = {
        "lip_seal": lip_seal,
        "mouth_width": float(mouth["width"]),
        "mouth_height": float(mouth["height"]),
        "lip_seal_total_distance": float(lip_seal["total_distance"]),
        "oral_angle": oral,
        "cheek_volume": cheek_volume,
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
            "cheek_area_score": float(compute_cheek_volume(baseline_landmarks, w, h)["score"]),
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
    cheek = metrics.get("cheek_area", {})
    ln = cheek.get("left_norm", float("nan"))
    rn = cheek.get("right_norm", float("nan"))
    asym = cheek.get("asymmetry", float("nan"))
    delta = None
    if isinstance(cheek.get("change", None), dict):
        delta = cheek["change"].get("score_delta", None)

    cv2.putText(img, f"CheekNorm L/R: {ln:.4f} / {rn:.4f}", (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"CheekAsym: {asym:.4f}", (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if delta is not None:
        cv2.putText(img, f"CheekDelta(vs neutral): {float(delta):+.4f}", (30, 150),
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
    peak_idx = find_peak_frame(
        landmarks_seq, w, h,
        baseline_landmarks=baseline_landmarks,
        seal_thr=0.03,
        mouth_thr=0.025,
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
        "cheek_area": metrics.get("cheek_area", {}),  # 用面积
        "palsy_detection": palsy_detection,
        "voluntary_interpretation": interpretation,
        "synkinesis": synkinesis,
    }

    if "baseline" in metrics:
        result.action_specific["baseline"] = metrics["baseline"]
        result.action_specific["changes"] = {
            "seal_change": metrics.get("seal_change", 0),
            "mouth_width_change": metrics.get("mouth_width_change", 0),
            "cheek_area_change": metrics.get("cheek_area_change", None),  # 用面积
        }

    # 创建输出目录
    action_dir = output_dir / ACTION_NAME
    action_dir.mkdir(parents=True, exist_ok=True)

    # 保存原始帧
    cv2.imwrite(str(action_dir / "peak_raw.jpg"), peak_frame)

    # 保存可视化
    vis = visualize_blow_cheek(peak_frame, peak_landmarks, metrics, w, h)
    cv2.imwrite(str(action_dir / "peak_indicators.jpg"), vis)

    # 保存JSON
    with open(action_dir / "indicators.json", 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"    [OK] {ACTION_NAME}: Lip Seal={metrics['lip_seal']['total_distance']:.1f}px")
    print(f"         Palsy: {palsy_detection.get('interpretation', 'N/A')}")
    print(f"         Voluntary Score: {result.voluntary_movement_score}/5 ({interpretation})")

    return result
