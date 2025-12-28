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
    mouth_h_thr = 0.32  # mouth_height / ICD 过大 => 张口
    seal_thr = 0.3  # lip_seal_total / ICD 过大 => 唇不闭合

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
        seal_ratio = float(seal["total_distance"]) / base_icd

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

        # 嘴角位移（总位移）
        left_displacement = dist(left_corner, baseline_left) * scale
        right_displacement = dist(right_corner, baseline_right) * scale

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

        metrics["left_corner_displacement"] = float(left_displacement)
        metrics["right_corner_displacement"] = float(right_displacement)
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

    return metrics


def detect_palsy_side(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    从撅嘴动作检测面瘫侧别 - 优先使用嘴角位移对称性

    核心洞察:
    - 嘴唇偏移方向不确定（与BlowCheek相同问题）
    - 嘴角位移对称性更可靠：面瘫侧口轮匝肌无力，收缩位移小

    指标优先级:
    1. 嘴角位移对称性 - 最可靠，直接反映肌肉力量
    2. 口角角度 - 作为验证
    3. 嘴唇偏移 - 方向不确定，仅当其他指标不可用时使用
    """
    result = {
        "palsy_side": 0,
        "confidence": 0.0,
        "interpretation": "",
        "method": "",
        "evidence": {}
    }

    # ========== 方法1（最可靠）: 嘴角位移对称性 ==========
    # 原理：撅嘴时口轮匝肌收缩，嘴角向内位移
    # 面瘫侧肌肉无力，位移量小
    left_disp = metrics.get("left_corner_displacement", None)
    right_disp = metrics.get("right_corner_displacement", None)

    if left_disp is not None and right_disp is not None:
        result["evidence"]["left_corner_disp"] = left_disp
        result["evidence"]["right_corner_disp"] = right_disp

        # 取绝对值（位移可能是负的，表示向内）
        left_abs = abs(left_disp)
        right_abs = abs(right_disp)
        max_disp = max(left_abs, right_abs)

        if max_disp > 3:  # 有明显的撅嘴动作（位移>3px）
            # 计算不对称性
            if max_disp > 1e-6:
                asymmetry = abs(left_abs - right_abs) / max_disp
                result["evidence"]["corner_asymmetry"] = asymmetry

                if asymmetry > 0.25:  # 明显不对称 (>25%)
                    result["method"] = "corner_displacement"
                    result["confidence"] = min(1.0, asymmetry * 2)

                    # 修正逻辑：面瘫侧嘴角被健侧牵拉，位移更大
                    if left_abs > right_abs:
                        # 左嘴角位移多 → 左侧面瘫（被右侧牵拉）
                        result["palsy_side"] = 1
                        result["interpretation"] = (
                            f"左嘴角位移多 (L={left_disp:.1f}px > R={right_disp:.1f}px, "
                            f"不对称{asymmetry:.1%}) → 左侧面瘫"
                        )
                    else:
                        # 右嘴角位移多 → 右侧面瘫（被左侧牵拉）
                        result["palsy_side"] = 2
                        result["interpretation"] = (
                            f"右嘴角位移多 (R={right_disp:.1f}px > L={left_disp:.1f}px, "
                            f"不对称{asymmetry:.1%}) → 右侧面瘫"
                        )
                    return result
                else:
                    result["evidence"]["corner_status"] = "symmetric"

    # ========== 方法2: 口角角度 ==========
    oral_angle = metrics.get("oral_angle", {})
    aoe = oral_angle.get("AOE", 0)
    bof = oral_angle.get("BOF", 0)
    angle_diff = abs(aoe - bof)

    result["evidence"]["AOE_right"] = aoe
    result["evidence"]["BOF_left"] = bof
    result["evidence"]["angle_diff"] = angle_diff

    if angle_diff > 3:
        result["method"] = "oral_angle"
        result["confidence"] = min(1.0, angle_diff / 15)

        if aoe < bof:
            result["palsy_side"] = 2
            result["interpretation"] = f"右口角低 (AOE={aoe:.1f}° < BOF={bof:.1f}°) → 右侧面瘫"
        else:
            result["palsy_side"] = 1
            result["interpretation"] = f"左口角低 (BOF={bof:.1f}° < AOE={aoe:.1f}°) → 左侧面瘫"
        return result

    # ========== 方法3: 嘴唇中线偏移（不可靠，最后使用） ==========
    if "lip_midline_offset" in metrics:
        offset_data = metrics["lip_midline_offset"]

        if "offset_change" in offset_data:
            offset_change = offset_data.get("offset_change", 0)
            offset_change_norm = offset_data.get("offset_change_norm", 0)

            result["evidence"]["lip_offset_change"] = offset_change
            result["evidence"]["lip_offset_norm"] = offset_change_norm

            if offset_change_norm > 0.02:
                result["method"] = "lip_offset_unreliable"
                result["confidence"] = min(0.5, offset_change_norm * 10)

                if offset_change > 0:
                    result["palsy_side"] = 2
                    result["interpretation"] = (
                        f"嘴唇向左偏移 ({offset_change:+.1f}px) → 可能右侧面瘫 "
                        f"(注意：此指标在撅嘴中不完全可靠)"
                    )
                else:
                    result["palsy_side"] = 1
                    result["interpretation"] = (
                        f"嘴唇向右偏移 ({offset_change:+.1f}px) → 可能左侧面瘫 "
                        f"(注意：此指标在撅嘴中不完全可靠)"
                    )
                return result

    # 未检测到明显不对称
    result["method"] = "none"
    result["interpretation"] = "各指标均未检测到明显不对称"
    return result


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

    # ========== 绘制面中线（虚线） ==========
    if "midline_x" in metrics:
        midline_x = int(metrics["midline_x"])
        # 绘制虚线
        for yy in range(0, h, 10):
            cv2.line(img, (midline_x, yy), (midline_x, min(yy + 5, h)), (0, 255, 255), 1)
        # 标注
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