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
    """计算露齿(ShowTeeth)特有指标 - 使用统一 scale

    关键：用“内唇缘开口面积”作为露齿/张口幅度的几何代理指标。
    鼻唇沟(NLF)相关几何指标已禁用（当前实现为鼻翼-嘴角连线，不等价于真实鼻唇沟）。
    """
    mouth = compute_mouth_metrics(landmarks, w, h)
    oral = compute_oral_angle(landmarks, w, h)

    # 内唇缘开口面积 (牙齿暴露的几何代理)
    inner_pts = np.array(pts2d(landmarks, LM.INNER_LIP, w, h), dtype=np.float32)
    inner_mouth_area = float(abs(polygon_area(inner_pts))) if len(inner_pts) >= 3 else 0.0

    # 嘴角位置
    left_corner = mouth["left_corner"]
    right_corner = mouth["right_corner"]

    # 嘴角高度 (相对于嘴中心)
    mouth_center_y = (left_corner[1] + right_corner[1]) / 2
    left_height_from_center = mouth_center_y - left_corner[1]   # 正值: 左嘴角更高
    right_height_from_center = mouth_center_y - right_corner[1]

    # 上唇/下唇中心位置（用于描述上唇提升或下颌下压）
    lip_top = mouth["top_center"]
    lip_bottom = mouth["bottom_center"]

    metrics: Dict[str, Any] = {
        "mouth_width": float(mouth["width"]),
        "mouth_height": float(mouth["height"]),
        "mouth_opening_ratio": float(mouth["height"] / (mouth["width"] + 1e-9)),
        "inner_mouth_area": float(inner_mouth_area),
        "left_corner": left_corner,
        "right_corner": right_corner,
        "left_height_from_center": float(left_height_from_center),
        "right_height_from_center": float(right_height_from_center),
        "lip_top_y": float(lip_top[1]),
        "lip_bottom_y": float(lip_bottom[1]),
        "oral_angle": {
            "AOE": float(getattr(oral, "AOE_angle", 0.0) or 0.0),
            "BOF": float(getattr(oral, "BOF_angle", 0.0) or 0.0),
            "asymmetry": float(getattr(oral, "angle_asymmetry", 0.0) or 0.0),
        },
        "nlf_disabled": True,
    }

    # 基线参考
    if baseline_landmarks is not None:
        # ========== 计算统一 scale ==========
        scale = compute_scale_to_baseline(landmarks, baseline_landmarks, w, h)
        metrics["scale"] = scale
        # ====================================

        baseline_mouth = compute_mouth_metrics(baseline_landmarks, w, h)
        baseline_inner_pts = np.array(pts2d(baseline_landmarks, LM.INNER_LIP, w, h), dtype=np.float32)
        baseline_inner_area = float(abs(polygon_area(baseline_inner_pts))) if len(baseline_inner_pts) >= 3 else 0.0

        # ========== 缩放到 baseline 尺度 ==========
        # 面积需要用 scale² 缩放
        scaled_inner_area = inner_mouth_area * (scale ** 2)
        scaled_width = mouth["width"] * scale
        scaled_height = mouth["height"] * scale

        metrics["baseline"] = {
            "mouth_width": float(baseline_mouth["width"]),
            "mouth_height": float(baseline_mouth["height"]),
            "inner_mouth_area": float(baseline_inner_area),
        }
        metrics["delta"] = {
            "mouth_width": float(scaled_width - baseline_mouth["width"]),
            "mouth_height": float(scaled_height - baseline_mouth["height"]),
            "inner_mouth_area": float(scaled_inner_area - baseline_inner_area),
        }

    return metrics


def detect_palsy_side(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    从露齿动作检测面瘫侧别

    原理: 与微笑类似
    1. 面瘫侧口角运动幅度小
    2. 面瘫侧口角角度低
    """
    result = {"palsy_side": 0, "confidence": 0.0, "interpretation": ""}

    oral = metrics.get("oral_angle", {})
    aoe = oral.get("AOE", 0)
    bof = oral.get("BOF", 0)

    if "excursion" in metrics:
        exc = metrics["excursion"]
        left_exc = exc["left_total"]
        right_exc = exc["right_total"]
        max_exc = max(left_exc, right_exc)

        if max_exc < 1:
            result["interpretation"] = "露齿运动幅度过小"
            return result

        asymmetry = abs(left_exc - right_exc) / max_exc
        result["confidence"] = min(1.0, asymmetry * 2.5)
        result["asymmetry_ratio"] = asymmetry

        if asymmetry < 0.10:
            result["palsy_side"] = 0
            result["interpretation"] = f"双侧对称 (差异{asymmetry * 100:.1f}%)"
        elif left_exc < right_exc:
            result["palsy_side"] = 1
            result["interpretation"] = f"左侧运动弱 (L={left_exc:.1f} < R={right_exc:.1f})"
        else:
            result["palsy_side"] = 2
            result["interpretation"] = f"右侧运动弱 (R={right_exc:.1f} < L={left_exc:.1f})"
    else:
        angle_diff = abs(aoe - bof)
        result["confidence"] = min(1.0, angle_diff / 15)

        if angle_diff < 3:
            result["palsy_side"] = 0
            result["interpretation"] = f"口角对称 (差{angle_diff:.1f}°)"
        elif aoe < bof:
            result["palsy_side"] = 2
            result["interpretation"] = f"右口角下垂 (AOE={aoe:+.1f}°)"
        else:
            result["palsy_side"] = 1
            result["interpretation"] = f"左口角下垂 (BOF={bof:+.1f}°)"

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