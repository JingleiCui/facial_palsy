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
    compute_lip_midline_symmetry,
)

from sunnybrook_scorer import (
    VoluntaryMovementItem, compute_voluntary_score_from_ratio
)

from thresholds import THR

ACTION_NAME = "ShowTeeth"
ACTION_NAME_CN = "露齿"


def find_peak_frame(landmarks_seq: List, frames_seq: List, w: int, h: int) -> int:
    """找露齿峰值帧

    露齿的“峰值”不一定来自“上唇提升”，也可能来自“下颌下压/下唇下拉”。
    因此这里用 **内唇缘一圈(Inner Lip)围成的开口面积** 作为露齿强度的近似指标：
    - 开口面积越大，通常牙齿暴露越明显
    - 为减轻前后距离变化的影响，对面积做 ICD² 归一化

    注意：这仍是“牙齿暴露”的几何代理指标。若后续接入牙齿/口腔分割，可进一步更准。
    """
    best_score = -1.0
    best_idx = 0

    for i, lm in enumerate(landmarks_seq):
        if lm is None:
            continue

        # 1) 内唇缘面积
        inner_pts = np.array(pts2d(lm, LM.INNER_LIP, w, h), dtype=np.float32)
        area = float(abs(polygon_area(inner_pts))) if len(inner_pts) >= 3 else 0.0

        # 2) ICD²归一化，降低面部远近变化带来的面积缩放
        icd = float(compute_icd(lm, w, h))
        score = area / (icd * icd + 1e-9)

        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx


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
        sequences: Dict[str, List[float]],
        fps: float,
        peak_idx: int,
        output_path: Path
) -> None:
    """
    绘制露齿关键帧选择的可解释性曲线

    露齿选择标准: 内唇缘开口面积(ICD归一化)最大的帧
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    n_frames = len(sequences["Inner Mouth Area (norm)"])
    frames = np.arange(n_frames)
    time_sec = frames / fps if fps > 0 else frames
    x_label = 'Time (seconds)' if fps > 0 else 'Frame'
    peak_time = peak_idx / fps if fps > 0 else peak_idx

    # 上图: 内唇面积曲线 (关键帧选择依据)
    ax1 = axes[0]
    area_seq = sequences["Inner Mouth Area (norm)"]
    ax1.plot(time_sec, area_seq, 'purple', label='Inner Mouth Area (ICD² normalized)', linewidth=2)

    # 标注峰值帧
    ax1.axvline(x=peak_time, color='black', linestyle='--', linewidth=2, alpha=0.7)
    area_at_peak = area_seq[peak_idx] if peak_idx < len(area_seq) else 0
    ax1.scatter([peak_time], [area_at_peak], color='red', s=150, zorder=5,
                edgecolors='black', linewidths=2, marker='*', label=f'Peak Frame {peak_idx}')
    ax1.annotate(f'Max: {area_at_peak:.3f}', xy=(peak_time, area_at_peak),
                 xytext=(10, -20), textcoords='offset points', fontsize=10, fontweight='bold')

    ax1.set_xlabel(x_label, fontsize=11)
    ax1.set_ylabel('Normalized Area', fontsize=11)
    ax1.set_title('ShowTeeth Peak Selection: Maximum Inner Lip Area (ICD² normalized)', fontsize=13, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # 下图: 口角角度 (用于患侧判断)
    ax2 = axes[1]
    ax2.plot(time_sec, sequences["AOE (Right)"], 'r-', label='AOE (Right)', linewidth=2)
    ax2.plot(time_sec, sequences["BOF (Left)"], 'b-', label='BOF (Left)', linewidth=2)
    ax2.axvline(x=peak_time, color='black', linestyle='--', linewidth=2, alpha=0.7, label=f'Peak Frame {peak_idx}')
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

    ax2.set_xlabel(x_label, fontsize=11)
    ax2.set_ylabel('Oral Angle (degrees)', fontsize=11)
    ax2.set_title('Oral Commissure Angles (for Palsy Detection)', fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
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

    # ========== 嘴唇面中线对称性（用于面瘫侧别判断）==========
    lip_symmetry = compute_lip_midline_symmetry(landmarks, w, h)
    metrics["lip_symmetry"] = lip_symmetry

    return metrics


def detect_palsy_side(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    从露齿动作检测面瘫侧别

    核心原理:
    - 面瘫侧肌肉瘫痪，健侧肌肉收缩把嘴唇拉向健侧
    - 直接比较峰值帧嘴唇区域相对于面中线的对称性
    - 嘴唇偏向的那侧是健侧，另一侧是面瘫侧

    方法:
    1. 主要：嘴唇区域面中线对称性（lip_symmetry）
    2. 辅助：口角角度（oral_angle）
    """
    result = {
        "palsy_side": 0,
        "confidence": 0.0,
        "interpretation": "",
        "method": "",
        "evidence": {}
    }

    # ========== 方法1：嘴唇面中线对称性（优先使用）==========
    if "lip_symmetry" in metrics:
        lip_sym = metrics["lip_symmetry"]
        left_dist = lip_sym["left_to_midline"]
        right_dist = lip_sym["right_to_midline"]
        lip_offset = lip_sym["lip_offset"]
        asymmetry_ratio = lip_sym["asymmetry_ratio"]

        result["method"] = "lip_midline_symmetry"
        result["evidence"] = {
            "left_to_midline": left_dist,
            "right_to_midline": right_dist,
            "lip_offset": lip_offset,
            "asymmetry_ratio": asymmetry_ratio,
            "symmetry_ratio": lip_sym["symmetry_ratio"],
        }

        # 置信度基于不对称程度
        result["confidence"] = min(1.0, asymmetry_ratio * 3.0)

        if asymmetry_ratio < 0.08:  # 8%以内认为对称
            result["palsy_side"] = 0
            result["interpretation"] = f"嘴唇对称 (L={left_dist:.1f}px, R={right_dist:.1f}px, 偏移{lip_offset:+.1f}px)"
        elif lip_offset > 0:
            # 嘴唇偏向左侧（患者左侧）= 被左侧拉 = 右侧面瘫
            result["palsy_side"] = 2
            result["interpretation"] = f"嘴唇偏左 (L={left_dist:.1f}px > R={right_dist:.1f}px) → 右侧面瘫"
        else:
            # 嘴唇偏向右侧（患者右侧）= 被右侧拉 = 左侧面瘫
            result["palsy_side"] = 1
            result["interpretation"] = f"嘴唇偏右 (R={right_dist:.1f}px > L={left_dist:.1f}px) → 左侧面瘫"

        return result

    # ========== 方法2（退化）：口角角度比较 ==========
    oral = metrics.get("oral_angle", {})
    aoe = oral.get("AOE", 0)  # 右侧口角角度
    bof = oral.get("BOF", 0)  # 左侧口角角度

    result["method"] = "oral_angle"
    angle_diff = abs(aoe - bof)
    result["confidence"] = min(1.0, angle_diff / 15)
    result["evidence"] = {
        "AOE_right": aoe,
        "BOF_left": bof,
        "angle_diff": angle_diff,
    }

    if angle_diff < 3:  # 3度以内认为对称
        result["palsy_side"] = 0
        result["interpretation"] = f"口角对称 (AOE={aoe:+.1f}°, BOF={bof:+.1f}°)"
    elif aoe < bof:
        # 右口角角度更小（位置更低） -> 右侧面瘫
        result["palsy_side"] = 2
        result["interpretation"] = f"右口角较低 (AOE={aoe:+.1f}° < BOF={bof:+.1f}°)"
    else:
        # 左口角角度更小 -> 左侧面瘫
        result["palsy_side"] = 1
        result["interpretation"] = f"左口角较低 (BOF={bof:+.1f}° < AOE={aoe:+.1f}°)"

    return result


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
    """可视化露齿指标 - 字体放大版"""
    img = frame.copy()

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

    # 绘制嘴部轮廓
    draw_polygon(img, landmarks, w, h, LM.OUTER_LIP, (0, 255, 0), 3)
    draw_polygon(img, landmarks, w, h, LM.INNER_LIP, (0, 200, 200), 2)

    # 绘制嘴角点
    if result.oral_angle:
        oral = result.oral_angle
        cv2.circle(img, (int(oral.A[0]), int(oral.A[1])), 10, (0, 0, 255), -1)
        cv2.circle(img, (int(oral.B[0]), int(oral.B[1])), 10, (255, 0, 0), -1)

    # 信息面板
    panel_w, panel_h = 700, 550
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
    cv2.putText(img, f"Asymmetry: {asym:.1f} deg", (25, y), FONT, FONT_SCALE_NORMAL, asym_color, THICKNESS_NORMAL)
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

    # 找峰值帧 (嘴宽最大)
    peak_idx = find_peak_frame(landmarks_seq, frames_seq, w, h)
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

    # 存储动作特有指标
    result.action_specific = {
        "show_teeth_metrics": metrics,
        "palsy_detection": palsy_detection,
        "voluntary_interpretation": interpretation,
        "synkinesis": synkinesis,
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
        sequences,
        video_info.get("fps", 30.0),
        peak_idx,
        action_dir / "peak_selection_curve.png"
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