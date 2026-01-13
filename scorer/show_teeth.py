#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShowTeeth 动作处理模块
======================

分析露齿动作:
1. 嘴角位移和对称性
2. 口角角度变化
3. 上唇提升程度
4. 嘴部开口/露齿幅度 (以内唇圈面积近似)
5. 联动运动检测 (眼部联动)

对应Sunnybrook: Open mouth smile (ZYG/RIS)

与Smile模块的区别:
- ShowTeeth更强调上唇提升和牙齿暴露
- 在Sunnybrook评分中优先使用ShowTeeth作为OpenMouthSmile
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json

from clinical_base import (
    LM, pt2d, pts2d, dist, polygon_area, compute_ear, compute_eye_area,
    compute_mouth_metrics, compute_oral_angle,
    compute_icd, extract_common_indicators,
    ActionResult, OralAngleMeasure, draw_polygon,
    compute_scale_to_baseline, draw_palsy_side_label,
    compute_lip_midline_offset_from_face_midline,
    compute_mouth_corner_to_eye_line_distance, compute_mouth_metrics,
    compute_face_midline, draw_face_midline,
)

from sunnybrook_scorer import (
    VoluntaryMovementItem, compute_voluntary_score_from_ratio
)

from thresholds import THR

ACTION_NAME = "ShowTeeth"
ACTION_NAME_CN = "露齿"


def find_peak_frame(landmarks_seq: List, frames_seq: List, w: int, h: int,
                    baseline_landmarks=None) -> Tuple[int, Dict[str, Any]]:
    """
    找露齿峰值帧 - 使用口内缘面积最大

    改进:
    - 露齿时嘴巴张开，内唇面积增大
    - 取内唇面积最大的帧作为峰值帧
    - 返回peak_debug用于可视化曲线

    Returns:
        (peak_idx, peak_debug): 峰值帧索引和调试信息
    """
    n_frames = len(landmarks_seq)
    if n_frames == 0:
        return 0, {"error": "empty_sequence"}

    # 收集时序数据
    inner_area_seq = []
    inner_area_norm_seq = []  # ICD归一化后的面积

    max_area = -1
    max_idx = 0

    for i, lm in enumerate(landmarks_seq):
        if lm is None:
            inner_area_seq.append(np.nan)
            inner_area_norm_seq.append(np.nan)
            continue

        # 计算内唇面积
        inner_pts = np.array(pts2d(lm, LM.INNER_LIP, w, h), dtype=np.float32)
        area = float(abs(polygon_area(inner_pts))) if len(inner_pts) >= 3 else 0.0

        # ICD归一化
        icd = float(compute_icd(lm, w, h))
        area_norm = area / (icd * icd + 1e-9)

        inner_area_seq.append(area)
        inner_area_norm_seq.append(area_norm)

        # 使用归一化面积选峰值（更稳定）
        if area_norm > max_area:
            max_area = area_norm
            max_idx = i

    # 构建peak_debug字典
    peak_debug = {
        "inner_area": inner_area_seq,
        "inner_area_norm": inner_area_norm_seq,
        "peak_idx": max_idx,
        "peak_value": float(max_area) if np.isfinite(max_area) else None,
        "selection_criterion": "max_inner_mouth_area",
    }

    return max_idx, peak_debug


def extract_show_teeth_sequences(landmarks_seq: List, w: int, h: int) -> Dict[str, List[float]]:
    """
    提取露齿关键指标的时序序列

    Returns:
        包含内唇面积和口角角度的时序数据
    """
    inner_area_seq = []
    aoe_seq = []
    bof_seq = []

    for lm in landmarks_seq:
        if lm is None:
            inner_area_seq.append(np.nan)
            aoe_seq.append(np.nan)
            bof_seq.append(np.nan)
        else:
            # 内唇面积
            inner_pts = np.array(pts2d(lm, LM.INNER_LIP, w, h), dtype=np.float32)
            area = float(abs(polygon_area(inner_pts))) if len(inner_pts) >= 3 else 0.0
            # ICD归一化
            icd = float(compute_icd(lm, w, h))
            normalized_area = area / (icd * icd + 1e-9)
            inner_area_seq.append(normalized_area)

            oral = compute_oral_angle(lm, w, h)
            aoe_seq.append(oral.AOE_angle if oral else np.nan)
            bof_seq.append(oral.BOF_angle if oral else np.nan)

    return {
        "Inner Mouth Area (norm)": inner_area_seq,
        "AOE (Right)": aoe_seq,
        "BOF (Left)": bof_seq,
    }


def plot_show_teeth_peak_selection(
        peak_debug: Dict[str, Any],
        fps: float,
        output_path,
        palsy_detection: Dict[str, Any] = None
) -> None:
    """绘制ShowTeeth关键帧选择曲线 - 更新版"""
    import matplotlib.pyplot as plt
    from clinical_base import get_palsy_side_text

    inner_area = peak_debug.get("inner_area", [])
    inner_area_norm = peak_debug.get("inner_area_norm", [])
    peak_idx = peak_debug.get("peak_idx", 0)

    if not inner_area_norm:
        return

    n_frames = len(inner_area_norm)
    frames = np.arange(n_frames)
    time_sec = frames / fps if fps > 0 else frames
    x_label = 'Time (seconds)' if fps > 0 else 'Frame'
    peak_time = peak_idx / fps if fps > 0 else peak_idx

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # 上图：归一化内唇面积
    ax1 = axes[0]
    ax1.plot(time_sec, inner_area_norm, 'g-', label='Inner Mouth Area (norm)', linewidth=2.5)
    ax1.axvline(x=peak_time, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

    if 0 <= peak_idx < n_frames and np.isfinite(inner_area_norm[peak_idx]):
        ax1.scatter([peak_time], [inner_area_norm[peak_idx]], color='red', s=150, zorder=5,
                    edgecolors='black', linewidths=1.5, marker='*',
                    label=f'Peak Frame {peak_idx}')

    title = "ShowTeeth Peak Selection: Max Inner Mouth Area"
    if palsy_detection:
        palsy_text = get_palsy_side_text(palsy_detection.get("palsy_side", 0))
        title += f' | Detected: {palsy_text}'

    ax1.set_title(title, fontsize=13, fontweight='bold')
    ax1.set_ylabel('Normalized Area', fontsize=11)
    ax1.legend()
    ax1.grid(True, alpha=0.4)

    # 下图：原始内唇面积
    ax2 = axes[1]
    ax2.plot(time_sec, inner_area, 'purple', label='Inner Mouth Area (px²)', linewidth=2)
    ax2.axvline(x=peak_time, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.set_title('Inner Mouth Area Over Time', fontsize=11)
    ax2.set_xlabel(x_label, fontsize=11)
    ax2.set_ylabel('Area (px²)', fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()


def compute_show_teeth_metrics(landmarks, w: int, h: int,
                               baseline_landmarks=None) -> Dict[str, Any]:
    """
    计算露齿(ShowTeeth)特有指标 - 增加嘴角位移计算
    """
    mouth = compute_mouth_metrics(landmarks, w, h)
    oral = compute_oral_angle(landmarks, w, h)

    # 内唇缘开口面积
    inner_pts = np.array(pts2d(landmarks, LM.INNER_LIP, w, h), dtype=np.float32)
    inner_mouth_area = float(abs(polygon_area(inner_pts))) if len(inner_pts) >= 3 else 0.0

    # 嘴角位置
    left_corner = mouth["left_corner"]
    right_corner = mouth["right_corner"]

    # 嘴角高度 (相对于嘴中心)
    mouth_center_y = (left_corner[1] + right_corner[1]) / 2
    left_height_from_center = mouth_center_y - left_corner[1]
    right_height_from_center = mouth_center_y - right_corner[1]

    # 面中线参考（使用双内眦中点）
    left_canthus = pt2d(landmarks[LM.EYE_INNER_L], w, h)
    right_canthus = pt2d(landmarks[LM.EYE_INNER_R], w, h)
    midline_x = (left_canthus[0] + right_canthus[0]) / 2

    # 嘴角相对于面中线的水平距离
    left_to_midline = midline_x - left_corner[0]  # 左嘴角到中线（正值）
    right_to_midline = right_corner[0] - midline_x  # 右嘴角到中线（正值）

    metrics: Dict[str, Any] = {
        "mouth_width": float(mouth["width"]),
        "mouth_height": float(mouth["height"]),
        "inner_mouth_area": float(inner_mouth_area),
        "left_corner": left_corner,
        "right_corner": right_corner,
        "left_height_from_center": float(left_height_from_center),
        "right_height_from_center": float(right_height_from_center),
        "corner_height_diff": float(left_height_from_center - right_height_from_center),
        "midline_x": float(midline_x),
        "left_to_midline": float(left_to_midline),
        "right_to_midline": float(right_to_midline),
        "midline_symmetry_ratio": float(left_to_midline / right_to_midline) if right_to_midline > 1e-9 else 1.0,
        "oral_angle": {
            "AOE": float(getattr(oral, "AOE_angle", 0.0) or 0.0),
            "BOF": float(getattr(oral, "BOF_angle", 0.0) or 0.0),
            "asymmetry": float(getattr(oral, "angle_asymmetry", 0.0) or 0.0),
        },
    }

    # 基线参考 - 计算嘴角位移
    if baseline_landmarks is not None:
        scale = compute_scale_to_baseline(landmarks, baseline_landmarks, w, h)
        metrics["scale"] = scale

        baseline_mouth = compute_mouth_metrics(baseline_landmarks, w, h)
        baseline_left = baseline_mouth["left_corner"]
        baseline_right = baseline_mouth["right_corner"]
        baseline_oral = compute_oral_angle(baseline_landmarks, w, h)

        # 嘴角位移（缩放到基线尺度）
        left_excursion_raw = dist(left_corner, baseline_left)
        right_excursion_raw = dist(right_corner, baseline_right)
        left_excursion = left_excursion_raw * scale
        right_excursion = right_excursion_raw * scale

        # 嘴宽变化
        width_change = mouth["width"] * scale - baseline_mouth["width"]

        metrics["excursion"] = {
            "left_total": float(left_excursion),
            "right_total": float(right_excursion),
            "excursion_ratio": float(left_excursion / right_excursion) if right_excursion > 1e-9 else 1.0,
            "baseline_width": float(baseline_mouth["width"]),
            "width_change": float(width_change),
        }

        # 口角角度变化
        metrics["oral_angle_change"] = {
            "AOE_change": float(oral.AOE_angle - baseline_oral.AOE_angle),
            "BOF_change": float(oral.BOF_angle - baseline_oral.BOF_angle),
        }

        # 嘴角高度变化
        baseline_center_y = (baseline_left[1] + baseline_right[1]) / 2
        baseline_left_height = baseline_center_y - baseline_left[1]
        baseline_right_height = baseline_center_y - baseline_right[1]

        metrics["height_change"] = {
            "left": float((left_height_from_center - baseline_left_height) * scale),
            "right": float((right_height_from_center - baseline_right_height) * scale),
        }

        # ========== 嘴唇中线偏移（用于面瘫侧别判断，有基线版本）==========
        lip_offset_data = compute_lip_midline_offset_from_face_midline(landmarks, w, h, baseline_landmarks)
        metrics["lip_midline_offset"] = lip_offset_data
    else:
        # ========== 没有基线时也计算嘴唇中线偏移 ==========
        lip_offset_data = compute_lip_midline_offset_from_face_midline(landmarks, w, h, None)
        metrics["lip_midline_offset"] = lip_offset_data

    return metrics


def detect_palsy_side(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    从露齿动作检测面瘫侧别 - 基于嘴唇中线偏移

    原理:
    - 面瘫侧肌肉无力，嘴巴整体被健侧肌肉拉向健侧
    - 嘴唇中线相对于面中线会偏向健侧

    重要改进:
    - 直接看当前帧嘴唇中线偏离面中线的距离和方向
    - 嘴唇偏向哪侧 → 对侧是面瘫侧
    """
    result = {
        "palsy_side": 0,
        "confidence": 0.0,
        "interpretation": "",
        "method": "",
        "evidence": {}
    }

    lip_offset_data = metrics.get("lip_midline_offset", {})

    if lip_offset_data:
        current_offset = lip_offset_data.get("current_offset", 0)
        face_midline_x = lip_offset_data.get("face_midline_x", 0)
        lip_midline_x = lip_offset_data.get("lip_midline_x", 0)

        # 获取归一化偏移（兼容有/无基线两种情况）
        # 无基线时用 offset_norm，有基线时用 offset_change_norm
        offset_norm = lip_offset_data.get("offset_norm", None)
        if offset_norm is None:
            offset_norm = lip_offset_data.get("offset_change_norm", 0)

        result["method"] = "lip_midline_offset"
        result["evidence"]["current_offset"] = current_offset
        result["evidence"]["offset_norm"] = offset_norm
        result["evidence"]["face_midline_x"] = face_midline_x
        result["evidence"]["lip_midline_x"] = lip_midline_x

        # 置信度计算
        result["confidence"] = min(1.0, offset_norm * 10)

        # 阈值调整
        if offset_norm < 0.025:
            result["palsy_side"] = 0
            result["interpretation"] = (
                f"嘴唇中线居中 (偏移{current_offset:.1f}px, {offset_norm:.1%})"
            )
        elif current_offset > 0:
            # 嘴唇偏向左侧 = 被左侧（健侧）拉动 = 右侧面瘫
            result["palsy_side"] = 2
            result["interpretation"] = (
                f"嘴唇偏向左侧 ({current_offset:+.1f}px, {offset_norm:.1%}) → 右侧面瘫"
            )
        else:
            # 嘴唇偏向右侧 = 被右侧（健侧）拉动 = 左侧面瘫
            result["palsy_side"] = 1
            result["interpretation"] = (
                f"嘴唇偏向右侧 ({current_offset:+.1f}px, {offset_norm:.1%}) → 左侧面瘫"
            )
        return result

    return result


def compute_severity_score(metrics: Dict[str, Any]) -> Tuple[int, str]:
    """
    计算动作严重度分数(医生标注标准)

    修改: 大幅提高阈值，解决高估问题
    """
    lip_offset_data = metrics.get("lip_midline_offset", {})
    offset_norm = lip_offset_data.get("offset_norm")
    if offset_norm is None:
        offset_norm = lip_offset_data.get("offset_change_norm", 0) or 0
    current_offset = lip_offset_data.get("current_offset", 0)

    # 阈值大幅提高，解决高估问题
    # GT分布: 1=26例, 2=9例, 3=19例, 4=27例, 5=3例
    if offset_norm < 0.05:  # 原0.03，提高
        return 1, f"正常 (偏移{offset_norm:.4%}, {current_offset:+.4f}px)"
    elif offset_norm < 0.10:  # 原0.06，提高
        return 2, f"轻度异常 (偏移{offset_norm:.4%})"
    elif offset_norm < 0.18:  # 原0.10，大幅提高
        return 3, f"中度异常 (偏移{offset_norm:.4%})"
    elif offset_norm < 0.28:  # 原0.15，大幅提高
        return 4, f"重度异常 (偏移{offset_norm:.4%})"
    else:
        return 5, f"完全面瘫 (偏移{offset_norm:.4%})"


def compute_voluntary_score(metrics: Dict[str, Any], baseline_landmarks=None) -> Tuple[int, str]:
    """
    计算Voluntary Movement评分

    基于嘴角位移对称性和口角角度对称性
    """
    oral_asym = metrics.get("oral_angle", {}).get("asymmetry", 0)

    if baseline_landmarks is not None and "excursion" in metrics:
        exc = metrics["excursion"]
        exc_ratio = exc["excursion_ratio"]

        # 检查是否有明显运动
        max_exc = max(exc["left_total"], exc["right_total"])
        if max_exc < 3:
            return 1, "无法启动运动 (位移过小)"

        # 使用位移比值评分
        deviation = abs(exc_ratio - 1.0)

        if deviation <= 0.10 and oral_asym < 5:
            return 5, "运动完整 (对称性优秀)"
        elif deviation <= 0.20 and oral_asym < 8:
            return 4, "几乎完整 (轻度不对称)"
        elif deviation <= 0.35 and oral_asym < 12:
            return 3, "启动但不对称"
        elif deviation <= 0.50:
            return 2, "轻微启动 (明显不对称)"
        else:
            return 1, "无法启动 (严重不对称)"
    else:
        # 没有基线，使用口角对称性
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
    """检测露齿时的联动运动 (主要是眼部)"""
    synkinesis = {
        "eye_synkinesis": 0,
        "brow_synkinesis": 0,
    }

    if baseline_result is None:
        return synkinesis

    # 当前EAR
    l_ear = compute_ear(current_landmarks, w, h, True)
    r_ear = compute_ear(current_landmarks, w, h, False)

    # 基线EAR
    baseline_l_ear = baseline_result.left_ear
    baseline_r_ear = baseline_result.right_ear

    # 检测眼部联动 (露齿时眼睛变小)
    if baseline_l_ear > 1e-9 and baseline_r_ear > 1e-9:
        l_change = (baseline_l_ear - l_ear) / baseline_l_ear
        r_change = (baseline_r_ear - r_ear) / baseline_r_ear
        avg_change = (l_change + r_change) / 2

        if avg_change > 0.25:  # 眼睛明显变小
            synkinesis["eye_synkinesis"] = 3
        elif avg_change > 0.15:
            synkinesis["eye_synkinesis"] = 2
        elif avg_change > 0.08:
            synkinesis["eye_synkinesis"] = 1

    return synkinesis


def visualize_show_teeth(frame: np.ndarray, landmarks, w: int, h: int,
                         result: ActionResult,
                         metrics: Dict[str, Any],
                         palsy_detection: Dict[str, Any] = None) -> np.ndarray:
    """可视化露齿指标 - 增加左右内唇面积区域可视化"""
    img = frame.copy()

    # 绘制面中线
    midline = compute_face_midline(landmarks, w, h)
    if midline:
        img = draw_face_midline(img, midline, color=(0, 255, 255), thickness=2, dashed=True)

    # 字体参数
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE_TITLE = 1.4
    FONT_SCALE_NORMAL = 0.9
    THICKNESS_TITLE = 3
    THICKNESS_NORMAL = 2
    LINE_HEIGHT = 45

    # 患侧标签
    if palsy_detection:
        img = draw_palsy_side_label(img, palsy_detection, x=20, y=70, font_scale=1.4)

    # ========== 绘制嘴唇中线和偏移 ==========
    lip_offset_data = metrics.get("lip_midline_offset", {})
    if lip_offset_data:
        lip_midline_x = lip_offset_data.get("lip_midline_x", 0)
        lip_midline_y = lip_offset_data.get("lip_midline_y", 0)
        lip_center_proj = lip_offset_data.get("lip_center_proj")
        current_signed_dist = lip_offset_data.get("current_signed_dist",
                                                  lip_offset_data.get("current_offset", 0))

        # 绘制嘴唇中线点（绿色圆点）
        cv2.circle(img, (int(lip_midline_x), int(lip_midline_y)), 8, (0, 255, 0), -1)

        # 绘制偏移连线（垂线到面中线）
        if lip_center_proj is not None:
            proj_x, proj_y = lip_center_proj
            dist = abs(current_signed_dist)
            offset_color = (0, 0, 255) if dist > 10 else (0, 165, 255)

            # 画垂线：从嘴唇中心到面中线投影点
            cv2.line(img, (int(lip_midline_x), int(lip_midline_y)),
                     (int(proj_x), int(proj_y)), offset_color, 3)

            # 画投影点
            cv2.circle(img, (int(proj_x), int(proj_y)), 5, offset_color, -1)

            # 标注偏移方向和距离
            direction = "L" if current_signed_dist > 0 else "R" if current_signed_dist < 0 else ""
            if dist > 3:
                mid_x = (int(lip_midline_x) + int(proj_x)) // 2
                mid_y = (int(lip_midline_y) + int(proj_y)) // 2
                cv2.putText(img, f"{direction} {dist:.1f}px",
                            (mid_x + 5, mid_y - 10), FONT, 0.8, offset_color, 2)

    # 绘制嘴部轮廓
    draw_polygon(img, landmarks, w, h, LM.OUTER_LIP, (0, 255, 0), 3)

    # 绘制嘴角点
    if result.oral_angle:
        oral = result.oral_angle
        cv2.circle(img, (int(oral.A[0]), int(oral.A[1])), 10, (0, 0, 255), -1)
        cv2.circle(img, (int(oral.B[0]), int(oral.B[1])), 10, (255, 0, 0), -1)

    # ========== 信息面板 ==========
    panel_w, panel_h = 700, 650  # 增加高度以显示更多信息
    cv2.rectangle(img, (10, 100), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.rectangle(img, (10, 100), (panel_w, panel_h), (255, 255, 255), 2)

    y = 160
    cv2.putText(img, f"{ACTION_NAME}", (25, y), FONT, FONT_SCALE_TITLE, (0, 255, 0), THICKNESS_TITLE)
    y += LINE_HEIGHT + 10

    cv2.putText(img, f"Width: {metrics['mouth_width']:.1f}px  Height: {metrics['mouth_height']:.1f}px", (25, y),
                FONT, FONT_SCALE_NORMAL, (255, 255, 255), THICKNESS_NORMAL)
    y += LINE_HEIGHT

    oral_angle = metrics.get("oral_angle", {})
    cv2.putText(img, f"AOE(R): {oral_angle.get('AOE', 0):+.1f}  BOF(L): {oral_angle.get('BOF', 0):+.1f}", (25, y),
                FONT, FONT_SCALE_NORMAL, (255, 255, 255), THICKNESS_NORMAL)
    y += LINE_HEIGHT

    asym = oral_angle.get('asymmetry', 0)
    asym_color = (0, 255, 0) if asym < 5 else ((0, 165, 255) if asym < 10 else (0, 0, 255))
    cv2.putText(img, f"Angle Asymmetry: {asym:.1f} deg", (25, y), FONT, FONT_SCALE_NORMAL, asym_color, THICKNESS_NORMAL)
    y += LINE_HEIGHT + 10

    # ========== 显示嘴唇中线偏移信息 ==========
    lip_offset_data = metrics.get("lip_midline_offset", {})
    if lip_offset_data:
        current_offset = lip_offset_data.get("current_offset", 0)

        cv2.putText(img, f"Lip Midline Offset:", (25, y), FONT, FONT_SCALE_NORMAL, (0, 255, 255), THICKNESS_NORMAL)
        y += LINE_HEIGHT

        # 当前偏移
        offset_color = (0, 255, 0) if abs(current_offset) < 10 else (0, 165, 255) if abs(current_offset) < 20 else (
            0, 0, 255)
        direction = "Left" if current_offset > 0 else "Right" if current_offset < 0 else "Center"
        cv2.putText(img, f"  Current: {current_offset:+.1f}px ({direction})", (25, y),
                    FONT, FONT_SCALE_NORMAL, offset_color, THICKNESS_NORMAL)
        y += LINE_HEIGHT

        # 如果有基线，显示变化量
        if "offset_change" in lip_offset_data:
            baseline_offset = lip_offset_data.get("baseline_offset", 0)
            offset_change = lip_offset_data.get("offset_change", 0)
            offset_change_norm = lip_offset_data.get("offset_change_norm", 0)

            cv2.putText(img, f"  Baseline: {baseline_offset:+.1f}px", (25, y),
                        FONT, FONT_SCALE_NORMAL, (200, 200, 200), THICKNESS_NORMAL)
            y += LINE_HEIGHT

            # 偏移变化量（关键指标）
            change_color = (0, 255, 0) if abs(offset_change_norm) < 0.02 else (0, 0, 255)
            change_dir = "→Left" if offset_change > 0 else "→Right" if offset_change < 0 else ""
            cv2.putText(img, f"  Change: {offset_change:+.1f}px {change_dir}", (25, y),
                        FONT, FONT_SCALE_NORMAL, change_color, THICKNESS_NORMAL)
            y += LINE_HEIGHT + 10

    if "excursion" in metrics:
        exc = metrics["excursion"]
        cv2.putText(img, f"Excursion L: {exc['left_total']:.1f}px  R: {exc['right_total']:.1f}px", (25, y),
                    FONT, FONT_SCALE_NORMAL, (255, 255, 255), THICKNESS_NORMAL)
        y += LINE_HEIGHT
        cv2.putText(img, f"Ratio: {exc['excursion_ratio']:.3f}", (25, y),
                    FONT, FONT_SCALE_NORMAL, (255, 255, 255), THICKNESS_NORMAL)
        y += LINE_HEIGHT + 10

    # 面瘫侧别
    if palsy_detection:
        palsy_side = palsy_detection.get("palsy_side", 0)
        palsy_text = {0: "Symmetric", 1: "Left Palsy", 2: "Right Palsy"}.get(palsy_side, "Unknown")
        palsy_color = (0, 255, 0) if palsy_side == 0 else (0, 0, 255)
        cv2.putText(img, f"Palsy: {palsy_text}", (25, y), FONT, FONT_SCALE_NORMAL, palsy_color, THICKNESS_NORMAL)
        y += LINE_HEIGHT

    cv2.putText(img, f"Voluntary Score: {result.voluntary_movement_score}/5", (25, y),
                FONT, FONT_SCALE_TITLE, (0, 255, 255), THICKNESS_TITLE)

    return img


def process(landmarks_seq: List, frames_seq: List, w: int, h: int,
            video_info: Dict[str, Any], output_dir: Path,
            baseline_result: Optional[ActionResult] = None,
            baseline_landmarks=None) -> Optional[ActionResult]:
    """处理ShowTeeth动作"""
    if not landmarks_seq or not frames_seq:
        return None

    # 找峰值帧
    peak_idx, peak_debug = find_peak_frame(landmarks_seq, frames_seq, w, h)
    peak_landmarks = landmarks_seq[peak_idx]
    peak_frame = frames_seq[peak_idx]

    # 提取时序序列用于可视化
    sequences = extract_show_teeth_sequences(landmarks_seq, w, h)

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

    # 计算露齿特有指标
    metrics = compute_show_teeth_metrics(peak_landmarks, w, h, baseline_landmarks)

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
        "show_teeth_metrics": metrics,
        "palsy_detection": palsy_detection,
        "voluntary_interpretation": interpretation,
        "synkinesis": synkinesis,
        "severity_score": severity_score,
        "severity_desc": severity_desc,
        "peak_debug": peak_debug,
    }

    # 创建输出目录
    action_dir = output_dir / ACTION_NAME
    action_dir.mkdir(parents=True, exist_ok=True)

    # 保存原始帧
    cv2.imwrite(str(action_dir / "peak_raw.jpg"), peak_frame)

    # 保存可视化
    vis = visualize_show_teeth(peak_frame, peak_landmarks, w, h, result, metrics, palsy_detection)
    cv2.imwrite(str(action_dir / "peak_indicators.jpg"), vis)

    # 绘制关键帧选择曲线
    plot_show_teeth_peak_selection(
        peak_debug,
        video_info.get("fps", 30.0),
        action_dir / "peak_selection_curve.png",
        palsy_detection
    )

    # 保存JSON
    with open(action_dir / "indicators.json", 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"         Palsy: {palsy_detection.get('interpretation', 'N/A')}")
    oral = metrics.get("oral_angle", {})
    print(f"    [OK] {ACTION_NAME}: Width={metrics['mouth_width']:.1f}px, Asym={oral.get('asymmetry', 0):.1f}°")
    if "excursion" in metrics:
        exc = metrics["excursion"]
        print(
            f"         Excursion L={exc['left_total']:.1f}px R={exc['right_total']:.1f}px Ratio={exc['excursion_ratio']:.3f}")
    print(f"         Voluntary Score: {result.voluntary_movement_score}/5 ({interpretation})")

    return result