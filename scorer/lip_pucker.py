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
    compute_mouth_metrics, compute_oral_angle, compute_lip_seal_distance,
    compute_icd, extract_common_indicators,
    ActionResult, draw_polygon, compute_scale_to_baseline,
    kabsch_rigid_transform, apply_rigid_transform,
    compute_lip_midline_symmetry,
    compute_lip_midline_offset_from_face_midline,
    compute_lip_midline_center,
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


def _save_lip_z_curve_png(png_path, z_delta_norm, peak_idx: int):
    """画嘴唇区域 z 轴运动曲线（相对基线的 delta_z/ICD）"""
    if not z_delta_norm:
        return
    plt.figure()
    plt.plot(list(range(len(z_delta_norm))), z_delta_norm)
    plt.axvline(int(peak_idx), linestyle="--")
    plt.title("Lip region depth curve (baseline_z - current_z) / ICD")
    plt.xlabel("Frame")
    plt.ylabel("Delta depth (normalized)")
    plt.tight_layout()
    plt.savefig(str(png_path), dpi=150)
    plt.close()


def find_peak_frame(landmarks_seq: List, frames_seq: List, w: int, h: int,
                    baseline_landmarks=None) -> Tuple[int, Dict[str, Any]]:
    """
    撅嘴峰值帧：
    - 主：嘴宽相对 baseline 明显变小
    - 辅：唇部区域更靠近镜头（baseline_z - current_z 更大）
    - 门控：不张嘴 + 唇密封（避免误选说话/张口帧）
    """
    n = len(landmarks_seq)
    if n == 0:
        return 0, {}

    # baseline mouth width / ICD / baseline lip z
    if baseline_landmarks is not None:
        base_mouth = compute_mouth_metrics(baseline_landmarks, w, h)
        base_width = float(base_mouth["width"])
        base_icd = float(compute_icd(baseline_landmarks, w, h))
        base_z = _mean_lip_z_aligned(baseline_landmarks, w, h, baseline_landmarks=None)
    else:
        # 没 baseline 就用第一帧当 baseline（退化）
        first = next((lm for lm in landmarks_seq if lm is not None), None)
        base_mouth = compute_mouth_metrics(first, w, h) if first is not None else {"width": 1.0}
        base_width = float(base_mouth["width"])
        base_icd = float(compute_icd(first, w, h)) if first is not None else 1.0
        base_z = _mean_lip_z_aligned(first, w, h, baseline_landmarks=None) if first is not None else 0.0

    base_width = max(base_width, 1e-6)
    base_icd = max(base_icd, 1e-6)

    # 门控阈值（你可以按数据再调）
    mouth_h_thr = THR.MOUTH_HEIGHT
    seal_thr = THR.MOUTH_SEAL

    best_score = -1e18
    best_idx = 0

    width_ratio_list = []
    z_delta_norm_list = []
    score_list = []
    valid_list = []

    for i, lm in enumerate(landmarks_seq):
        if lm is None:
            width_ratio_list.append(float("nan"))
            z_delta_norm_list.append(float("nan"))
            score_list.append(float("nan"))
            valid_list.append(False)
            continue

        mouth = compute_mouth_metrics(lm, w, h)
        width = float(mouth["width"])
        height = float(mouth["height"])

        width_ratio = width / base_width
        mouth_h_ratio = height / base_icd

        # 唇密封（越小越闭合）
        seal = compute_lip_seal_distance(lm, w, h)
        seal_ratio = float(seal["middle_distance"]) / base_icd

        # 唇部深度：对齐后取唇部平均z（越小越靠前）
        cur_z = _mean_lip_z_aligned(lm, w, h, baseline_landmarks=baseline_landmarks)
        # delta_z：baseline_z - current_z （越大表示越“靠前”）
        z_delta = float(base_z - cur_z)
        z_delta_norm = z_delta / base_icd

        # 门控：不张口 + 唇闭合（避免嘴张开误选）
        valid = (mouth_h_ratio <= mouth_h_thr) and (seal_ratio <= seal_thr)

        # 评分：嘴唇前突为主
        score = z_delta_norm

        # 如果没通过门控，强惩罚
        if not valid:
            score -= 10.0

        width_ratio_list.append(width_ratio)
        z_delta_norm_list.append(z_delta_norm)
        score_list.append(score)
        valid_list.append(valid)

        if score > best_score:
            best_score = score
            best_idx = i

    debug = {
        "width_ratio": width_ratio_list,
        "lip_z_delta_norm": z_delta_norm_list,
        "score": score_list,
        "valid": valid_list,
        "best_score": float(best_score)
    }
    return best_idx, debug


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

    return metrics


def detect_palsy_side(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    从撅嘴动作检测面瘫侧别 - 基于嘴角收缩程度比较

    核心逻辑:
    - 撅嘴时口轮匝肌收缩，嘴唇向中间聚拢
    - 比较左右嘴角的收缩量
    - 收缩量小的一侧 = 面瘫侧（肌肉无力）
    """
    result = {
        "palsy_side": 0,
        "confidence": 0.0,
        "interpretation": "",
        "method": "",
        "evidence": {}
    }

    # ========== 方法: 嘴角收缩量比较 ==========
    left_contraction = metrics.get("left_corner_contraction", 0)
    right_contraction = metrics.get("right_corner_contraction", 0)

    result["evidence"]["left_contraction"] = left_contraction
    result["evidence"]["right_contraction"] = right_contraction

    max_contraction = max(abs(left_contraction), abs(right_contraction))

    if max_contraction < 2:  # 运动幅度过小（像素）
        # 回退到lip_midline_offset
        if "lip_midline_offset" in metrics:
            offset_data = metrics["lip_midline_offset"]
            current_offset = offset_data.get("current_offset", 0)
            icd = metrics.get("icd", 1)
            if icd < 1e-6:
                icd = 1
            offset_norm = abs(current_offset) / icd
            result["evidence"]["offset_norm"] = offset_norm

            if offset_norm > 0.020:
                result["method"] = "lip_midline_offset"
                result["confidence"] = min(1.0, offset_norm * 15)

                if current_offset > 0:
                    result["palsy_side"] = 2
                    result["interpretation"] = f"嘴唇偏向左侧 ({current_offset:+.1f}px) → 右侧面瘫"
                else:
                    result["palsy_side"] = 1
                    result["interpretation"] = f"嘴唇偏向右侧 ({current_offset:+.1f}px) → 左侧面瘫"
                return result

        result["method"] = "none"
        result["interpretation"] = "运动幅度过小，无法判断"
        return result

    # 计算不对称度
    asymmetry = abs(left_contraction - right_contraction) / max_contraction
    result["evidence"]["contraction_asymmetry"] = asymmetry
    result["method"] = "corner_contraction"
    result["confidence"] = min(1.0, asymmetry * 2)

    if asymmetry < 0.15:
        result["palsy_side"] = 0
        result["interpretation"] = (
            f"双侧嘴角收缩对称 (L={left_contraction:.1f}px, R={right_contraction:.1f}px, "
            f"不对称{asymmetry:.1%})"
        )
    elif left_contraction < right_contraction:
        # 左侧收缩量小 = 左侧肌肉弱 = 左侧面瘫
        result["palsy_side"] = 1
        result["interpretation"] = (
            f"左侧嘴角收缩弱 (L={left_contraction:.1f}px < R={right_contraction:.1f}px, "
            f"不对称{asymmetry:.1%}) → 左侧面瘫"
        )
    else:
        result["palsy_side"] = 2
        result["interpretation"] = (
            f"右侧嘴角收缩弱 (R={right_contraction:.1f}px < L={left_contraction:.1f}px, "
            f"不对称{asymmetry:.1%}) → 右侧面瘫"
        )

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

    # ========== 绘制嘴唇对称性证据 ==========
    if "lip_symmetry" in metrics:
        lip_sym = metrics["lip_symmetry"]
        cv2.putText(img, "=== Lip Symmetry Evidence ===", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        y += 20

        left_dist = lip_sym["left_to_midline"]
        right_dist = lip_sym["right_to_midline"]
        lip_offset = lip_sym["lip_offset"]
        asymmetry_ratio = lip_sym["asymmetry_ratio"]

        # 左侧距离
        left_color = (0, 255, 0) if left_dist >= right_dist else (0, 0, 255)
        cv2.putText(img, f"L to mid: {left_dist:.1f}px", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, left_color, 1)
        y += 18

        # 右侧距离
        right_color = (0, 255, 0) if right_dist >= left_dist else (0, 0, 255)
        cv2.putText(img, f"R to mid: {right_dist:.1f}px", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, right_color, 1)
        y += 18

        # 偏移方向
        direction = "L" if lip_offset > 0 else "R"
        cv2.putText(img, f"Offset: {lip_offset:+.1f}px ({direction})", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 18

        # 不对称比例
        cv2.putText(img, f"Asym: {asymmetry_ratio * 100:.1f}%", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 22

    # 面瘫侧别检测结果
    palsy_side = palsy_detection.get("palsy_side", 0)
    palsy_text = {0: "Symmetry", 1: "Left", 2: "Right"}.get(palsy_side, "Unkown")
    palsy_color = (0, 255, 0) if palsy_side == 0 else (0, 0, 255)
    cv2.putText(img, f"Palsy Side: {palsy_text}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, palsy_color, 1)
    y += 25

    # Voluntary Score
    cv2.putText(img, f"Voluntary Score: {result.voluntary_movement_score}/5", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # ========== 绘制面中线和嘴唇中线 ==========
    if "lip_midline_offset" in metrics:
        offset_data = metrics["lip_midline_offset"]
        face_midline_x = offset_data.get("face_midline_x", None)
        lip_midline_x = offset_data.get("lip_midline_x", None)
        lip_midline_y = offset_data.get("lip_midline_y", None)

        if face_midline_x is not None:
            face_midline_x_int = int(face_midline_x)

            # 面中线的起点和终点
            # 获取内眦y坐标作为参考
            left_canthus = pt2d(landmarks[LM.EYE_INNER_L], w, h)
            right_canthus = pt2d(landmarks[LM.EYE_INNER_R], w, h)
            eye_y = int((left_canthus[1] + right_canthus[1]) / 2)

            midline_start_y = max(20, eye_y - 80)
            midline_end_y = min(h - 20, eye_y + 300)

            # 绘制面中线（青色虚线）
            for yy in range(midline_start_y, midline_end_y, 15):
                cv2.line(img, (face_midline_x_int, yy),
                         (face_midline_x_int, min(yy + 8, midline_end_y)),
                         (255, 255, 0), 2)
            cv2.putText(img, "Face Mid", (face_midline_x_int + 5, midline_start_y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

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
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # ========== 绘制偏移指示线 ==========
            if face_midline_x is not None:
                face_midline_x_int = int(face_midline_x)
                offset = lip_midline_x - face_midline_x
                offset_color = (0, 0, 255) if abs(offset) > 10 else (0, 255, 255)
                cv2.line(img, (lip_midline_x_int, lip_midline_y_int),
                         (face_midline_x_int, lip_midline_y_int), offset_color, 3)

                mid_x = (lip_midline_x_int + face_midline_x_int) // 2
                direction = "L" if offset > 0 else "R"
                cv2.putText(img, f"{abs(offset):.0f}px({direction})",
                            (mid_x - 40, lip_midline_y_int - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, offset_color, 2)

    elif "midline_x" in metrics:
        midline_x = int(metrics["midline_x"])
        for yy in range(0, h, 10):
            cv2.line(img, (midline_x, yy), (midline_x, min(yy + 5, h)), (0, 255, 255), 1)
        cv2.putText(img, "Mid", (midline_x + 5, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # ========== 绘制嘴角到中线的连线 ==========
    if "left_corner" in metrics and "midline_x" in metrics:
        left_corner = metrics["left_corner"]
        right_corner = metrics["right_corner"]
        midline_x = int(metrics["midline_x"])

        # 左嘴角到中线的水平线（蓝色）
        cv2.line(img, (int(left_corner[0]), int(left_corner[1])),
                 (midline_x, int(left_corner[1])), (255, 0, 0), 1)
        # 右嘴角到中线的水平线（橙色）
        cv2.line(img, (int(right_corner[0]), int(right_corner[1])),
                 (midline_x, int(right_corner[1])), (0, 165, 255), 1)

    return img


def process(landmarks_seq: List, frames_seq: List, w: int, h: int,
            video_info: Dict[str, Any], output_dir: Path,
            baseline_result: Optional[ActionResult] = None,
            baseline_landmarks=None) -> Optional[ActionResult]:
    """处理LipPucker动作"""
    if not landmarks_seq or not frames_seq:
        return None

    # 找峰值帧 (嘴宽最小)
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

    # 计算撅嘴特有指标
    metrics = compute_lip_pucker_metrics(peak_landmarks, w, h, baseline_landmarks)

    # 检测面瘫侧别
    palsy_detection = detect_palsy_side(metrics)

    # 计算Voluntary Movement评分
    score, interpretation = compute_voluntary_score(metrics, baseline_landmarks)
    result.voluntary_movement_score = score

    # 检测联动
    synkinesis = detect_synkinesis(baseline_result, peak_landmarks, w, h)
    result.synkinesis_scores = synkinesis

    # 计算严重度分数 (医生标注标准: 1=正常, 5=面瘫)
    severity_score, severity_desc = compute_severity_score(metrics)

    # 存储动作特有指标
    result.action_specific = {
        "mouth_metrics": {
            "width": metrics["mouth_width"],
            "height": metrics["mouth_height"],
            "width_height_ratio": metrics["width_height_ratio"],
        },
        "oral_angle": metrics["oral_angle"],
        "midline_x": metrics.get("midline_x", 0),
        "left_to_midline": metrics.get("left_to_midline", 0),
        "right_to_midline": metrics.get("right_to_midline", 0),
        "palsy_detection": palsy_detection,
        "voluntary_interpretation": interpretation,
        "synkinesis": synkinesis,
        "severity_score": severity_score,
        "severity_desc": severity_desc,
    }

    if "baseline" in metrics:
        result.action_specific["baseline"] = metrics["baseline"]

    if "movement" in metrics:
        result.action_specific["movement"] = metrics["movement"]

    if "width_change" in metrics:
        result.action_specific["changes"] = {
            "width_change": metrics["width_change"],
            "width_ratio": metrics.get("width_ratio", 1.0),
        }

    # 创建输出目录
    action_dir = output_dir / ACTION_NAME
    action_dir.mkdir(parents=True, exist_ok=True)

    # 保存原始帧
    cv2.imwrite(str(action_dir / "peak_raw.jpg"), peak_frame)

    # 保存可视化
    vis = visualize_lip_pucker(peak_frame, peak_landmarks, w, h, result, metrics, palsy_detection)
    cv2.imwrite(str(action_dir / "peak_indicators.jpg"), vis)

    # 保存嘴唇Z轴曲线（相对基线）
    if peak_debug and "lip_z_delta_norm" in peak_debug:
        _save_lip_z_curve_png(action_dir / "mouth_z_curve.png",
                              peak_debug["lip_z_delta_norm"],
                              peak_idx)

    # 也把 debug 写进 json（可解释性）
    result.action_specific["peak_debug"] = peak_debug

    # 保存JSON
    with open(action_dir / "indicators.json", 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"    [OK] {ACTION_NAME}: Width={metrics['mouth_width']:.1f}px")
    if "width_change_percent" in metrics:
        print(f"         Width Change: {metrics['width_change_percent']:+.1f}%")
    print(f"         Palsy: {palsy_detection.get('interpretation', 'N/A')}")
    print(f"         Voluntary Score: {result.voluntary_movement_score}/5 ({interpretation})")

    return result