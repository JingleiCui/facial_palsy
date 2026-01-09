#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CloseEye 动作处理模块
==============================

分析闭眼动作:
1. 眼睛闭合程度 (EAR)
2. 眼睛面积变化
3. 左右眼对称性
4. 面瘫侧别检测 - 检测两只眼是否都闭上
5. 联动运动检测

修复内容:
- 添加面瘫侧别检测：基于两只眼是否都能完全闭合
- 面瘫侧的眼睛无法完全闭合(EAR较大)

面瘫侧别编码: 0=无/对称, 1=左, 2=右

对应Sunnybrook: Gentle Eye closure (OCS)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from clinical_base import (
    LM, pt2d, pts2d, dist, compute_ear, compute_eye_area,
    compute_palpebral_height, compute_mouth_metrics,
    compute_icd, extract_common_indicators,
    ActionResult, draw_polygon, compute_scale_to_baseline,
    draw_palsy_side_label, compute_eye_closure_by_area,
    compute_eye_closure_sequence, compute_eye_synchrony,
    compute_eye_symmetry_at_peak,
)

from thresholds import THR


def find_peak_frame_close_eye(landmarks_seq: List, frames_seq: List, w: int, h: int,
                              baseline_landmarks=None) -> int:
    """
    找闭眼峰值帧 - 使用眼睛面积

    改进: 使用眼睛面积相对于基线的闭合程度来找峰值帧
    """
    if baseline_landmarks is not None:
        # 有基线：使用面积闭合度
        baseline_left_area, _ = compute_eye_area(baseline_landmarks, w, h, True)
        baseline_right_area, _ = compute_eye_area(baseline_landmarks, w, h, False)

        max_closure = -float('inf')
        max_idx = 0

        for i, lm in enumerate(landmarks_seq):
            if lm is None:
                continue

            scale = compute_scale_to_baseline(lm, baseline_landmarks, w, h)

            l_area, _ = compute_eye_area(lm, w, h, True)
            r_area, _ = compute_eye_area(lm, w, h, False)
            scaled_l_area = l_area * (scale ** 2)
            scaled_r_area = r_area * (scale ** 2)

            l_closure = 1.0 - (scaled_l_area / baseline_left_area) if baseline_left_area > 1e-6 else 0
            r_closure = 1.0 - (scaled_r_area / baseline_right_area) if baseline_right_area > 1e-6 else 0

            avg_closure = (max(0, min(1, l_closure)) + max(0, min(1, r_closure))) / 2

            if avg_closure > max_closure:
                max_closure = avg_closure
                max_idx = i

        return max_idx
    else:
        # 没有基线：使用面积最小的帧
        min_area = float('inf')
        min_idx = 0

        for i, lm in enumerate(landmarks_seq):
            if lm is None:
                continue
            l_area, _ = compute_eye_area(lm, w, h, True)
            r_area, _ = compute_eye_area(lm, w, h, False)
            avg_area = (l_area + r_area) / 2
            if avg_area < min_area:
                min_area = avg_area
                min_idx = i

        return min_idx


def extract_eye_sequence(landmarks_seq: List, w: int, h: int) -> Dict[str, Dict[str, List[float]]]:
    """提取整个序列的眼睛指标"""
    left_ear_seq = []
    right_ear_seq = []
    left_area_seq = []
    right_area_seq = []

    for lm in landmarks_seq:
        if lm is None:
            left_ear_seq.append(np.nan)
            right_ear_seq.append(np.nan)
            left_area_seq.append(np.nan)
            right_area_seq.append(np.nan)
        else:
            l_ear = compute_ear(lm, w, h, True)
            r_ear = compute_ear(lm, w, h, False)
            l_area, _ = compute_eye_area(lm, w, h, True)
            r_area, _ = compute_eye_area(lm, w, h, False)
            left_ear_seq.append(l_ear)
            right_ear_seq.append(r_ear)
            left_area_seq.append(l_area)
            right_area_seq.append(r_area)

    return {
        "ear": {
            "left": left_ear_seq,
            "right": right_ear_seq,
            "average": [(l + r) / 2 if not (np.isnan(l) or np.isnan(r)) else np.nan
                        for l, r in zip(left_ear_seq, right_ear_seq)]
        },
        "area": {
            "left": left_area_seq,
            "right": right_area_seq
        }
    }


def detect_palsy_side_from_closure(peak_closure_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    从闭眼动作检测面瘫侧别 - 基于眼睛面积

    原理: 使用compute_eye_closure_by_area的结果来判断面瘫侧

    Args:
        peak_closure_data: 峰值帧的眼睛闭合数据（来自compute_eye_closure_by_area）
    """
    result = {
        "palsy_side": 0,
        "confidence": 0.0,
        "interpretation": "",
        "method": "area_closure",
        "evidence": {}
    }

    if peak_closure_data is None:
        result["interpretation"] = "无闭眼数据"
        return result

    left_closure = peak_closure_data.get("left_closure_ratio", 0)
    right_closure = peak_closure_data.get("right_closure_ratio", 0)

    result["evidence"] = {
        "left_area": peak_closure_data.get("left_area", 0),
        "right_area": peak_closure_data.get("right_area", 0),
        "left_closure_pct": left_closure * 100,
        "right_closure_pct": right_closure * 100,
    }

    if "baseline_left_area" in peak_closure_data:
        result["evidence"]["baseline_left_area"] = peak_closure_data["baseline_left_area"]
        result["evidence"]["baseline_right_area"] = peak_closure_data["baseline_right_area"]

    max_closure = max(left_closure, right_closure)

    if max_closure < THR.EYE_CLOSURE_RATIO_PARTIAL:  # 闭合不足
        result["interpretation"] = f"闭眼幅度过小 (L={left_closure * 100:.1f}%, R={right_closure * 100:.1f}%)"
        result["evidence"]["status"] = "insufficient_closure"
        return result

    # 计算不对称比例
    asymmetry = abs(left_closure - right_closure) / max_closure
    result["confidence"] = min(1.0, asymmetry * 5)
    result["evidence"]["asymmetry_ratio"] = asymmetry
    result["evidence"]["left_closure_pct"] = left_closure * 100
    result["evidence"]["right_closure_pct"] = right_closure * 100

    if asymmetry < THR.EYE_SYMMETRY_NORMAL:  # 以内认为对称
        result["palsy_side"] = 0
        result["interpretation"] = (
            f"双眼闭合对称 (L={left_closure * 100:.1f}%, R={right_closure * 100:.1f}%, "
            f"差异{asymmetry * 100:.4f}%)"
        )
    elif left_closure < right_closure:
        # 左眼闭合程度低 -> 左侧面瘫
        result["palsy_side"] = 1
        result["interpretation"] = (
            f"左眼闭合差 (L={left_closure * 100:.1f}% < R={right_closure * 100:.1f}%) → 左侧面瘫"
        )
    else:
        result["palsy_side"] = 2
        result["interpretation"] = (
            f"右眼闭合差 (R={right_closure * 100:.1f}% < L={left_closure * 100:.1f}%) → 右侧面瘫"
        )

    return result


def compute_close_eye_metrics(landmarks, w: int, h: int,
                              baseline_landmarks=None) -> Dict[str, Any]:
    """计算闭眼特有指标 - 使用统一 scale"""
    # 当前EAR和面积
    l_ear = compute_ear(landmarks, w, h, True)
    r_ear = compute_ear(landmarks, w, h, False)
    l_area, _ = compute_eye_area(landmarks, w, h, True)
    r_area, _ = compute_eye_area(landmarks, w, h, False)
    l_height = compute_palpebral_height(landmarks, w, h, True)
    r_height = compute_palpebral_height(landmarks, w, h, False)

    metrics = {
        "left_ear": l_ear,
        "right_ear": r_ear,
        "ear_ratio": l_ear / r_ear if r_ear > 1e-9 else 1.0,
        "left_area": l_area,
        "right_area": r_area,
        "area_ratio": l_area / r_area if r_area > 1e-9 else 1.0,
        "left_palpebral_height": l_height,
        "right_palpebral_height": r_height,
        "height_ratio": l_height / r_height if r_height > 1e-9 else 1.0,
    }

    # 如果有基线，计算闭合程度
    if baseline_landmarks is not None:
        # ========== 计算统一 scale ==========
        scale = compute_scale_to_baseline(landmarks, baseline_landmarks, w, h)
        metrics["scale"] = scale
        # ====================================

        baseline_l_ear = compute_ear(baseline_landmarks, w, h, True)
        baseline_r_ear = compute_ear(baseline_landmarks, w, h, False)
        baseline_l_height = compute_palpebral_height(baseline_landmarks, w, h, True)
        baseline_r_height = compute_palpebral_height(baseline_landmarks, w, h, False)

        # EAR 是比值，不需要缩放
        # 眼睑裂高度需要缩放
        l_height_scaled = l_height * scale
        r_height_scaled = r_height * scale

        # 闭合比例 (当前/基线)
        left_closure_ratio = l_ear / baseline_l_ear if baseline_l_ear > 1e-9 else 1.0
        right_closure_ratio = r_ear / baseline_r_ear if baseline_r_ear > 1e-9 else 1.0

        # 闭合百分比 (1 - ratio) * 100%
        metrics["left_closure_percent"] = (1 - left_closure_ratio) * 100
        metrics["right_closure_percent"] = (1 - right_closure_ratio) * 100
        metrics["left_closure_ratio"] = left_closure_ratio
        metrics["right_closure_ratio"] = right_closure_ratio

        # 高度变化（缩放后）
        metrics["left_height_change"] = l_height_scaled - baseline_l_height
        metrics["right_height_change"] = r_height_scaled - baseline_r_height

    # ========== 使用眼睛面积计算闭合度 ==========
    area_closure = compute_eye_closure_by_area(landmarks, w, h, baseline_landmarks)
    metrics["area_closure"] = area_closure

    # 使用面积闭合度作为主要的闭合指标
    if "left_closure_ratio" in area_closure:
        metrics["left_closure_ratio"] = area_closure["left_closure_ratio"]
        metrics["right_closure_ratio"] = area_closure["right_closure_ratio"]
        metrics["left_closure_percent"] = area_closure["left_closure_ratio"] * 100
        metrics["right_closure_percent"] = area_closure["right_closure_ratio"] * 100

    # 面瘫侧检测（使用面积方法）
    palsy_detection = detect_palsy_side_from_closure(area_closure)
    metrics["palsy_detection"] = palsy_detection

    return metrics


def compute_voluntary_score(metrics: Dict[str, Any], baseline_landmarks=None) -> Tuple[int, str]:
    """
    计算Voluntary Movement评分

    基于闭眼程度的对称性

    评分标准:
    - 5=完整: 双眼完全闭合且对称
    - 4=几乎完整: 闭合良好，轻度不对称
    - 3=启动但不对称: 有闭眼动作但明显不对称
    - 2=轻微启动: 闭眼动作很弱
    - 1=无法启动: 几乎无闭眼动作
    """
    if baseline_landmarks is not None and "left_closure_percent" in metrics:
        left_closure = metrics["left_closure_percent"]
        right_closure = metrics["right_closure_percent"]

        # 检查是否有闭眼
        max_closure = max(left_closure, right_closure)
        min_closure = min(left_closure, right_closure)

        if max_closure < 30:  # 几乎没闭眼
            return 1, "无法启动运动 (闭眼不足30%)"

        # 计算对称性
        if max_closure < 1e-9:
            symmetry_ratio = 1.0
        else:
            symmetry_ratio = min_closure / max_closure

        if symmetry_ratio >= 0.85:
            if max_closure >= 90:
                return 5, "运动完整 (双眼完全闭合且对称)"
            elif max_closure >= 70:
                return 4, "几乎完整 (闭合良好)"
            else:
                return 3, "启动但幅度不足"
        elif symmetry_ratio >= 0.60:
            return 3, "启动但不对称"
        elif symmetry_ratio >= 0.30:
            return 2, "轻微启动"
        else:
            return 1, "无法启动"
    else:
        # 没有基线，使用静态EAR比值
        ratio = metrics.get("ear_ratio", 1.0)
        deviation = abs(ratio - 1.0)

        if deviation <= 0.10:
            return 5, "运动完整"
        elif deviation <= 0.20:
            return 4, "几乎完整"
        elif deviation <= 0.35:
            return 3, "启动但不对称"
        elif deviation <= 0.50:
            return 2, "轻微启动"
        else:
            return 1, "无法启动"


def compute_severity_score(metrics: Dict[str, Any]) -> Tuple[int, str]:
    """
    计算动作严重度分数(医生标注标准)

    计算依据: 闭眼程度的对称性

    修改: 调整阈值使判定更严格
    """
    # 优先使用闭合百分比
    left_closure = metrics.get("left_closure_percent")
    right_closure = metrics.get("right_closure_percent")

    # 如果没有闭合百分比数据，尝试从area_closure获取
    if left_closure is None or right_closure is None:
        area_closure = metrics.get("area_closure", {})
        left_ratio = area_closure.get("left_closure_ratio", 0)
        right_ratio = area_closure.get("right_closure_ratio", 0)
        left_closure = left_ratio * 100
        right_closure = right_ratio * 100

    # 如果仍然没有数据，使用面积比
    if left_closure == 0 and right_closure == 0:
        area_ratio = metrics.get("area_ratio", 1.0)
        deviation = abs(area_ratio - 1.0)

        if deviation <= 0.08:
            return 1, f"对称性良好 (area_ratio={area_ratio:.3f})"
        elif deviation <= 0.15:
            return 2, f"轻度不对称 (area_ratio={area_ratio:.3f})"
        elif deviation <= 0.25:
            return 3, f"中度不对称 (area_ratio={area_ratio:.3f})"
        elif deviation <= 0.40:
            return 4, f"重度不对称 (area_ratio={area_ratio:.3f})"
        else:
            return 5, f"严重不对称 (area_ratio={area_ratio:.3f})"

    max_closure = max(left_closure, right_closure)
    min_closure = min(left_closure, right_closure)

    # 检查是否有足够的闭眼动作
    if max_closure < 20:
        return 1, f"闭眼幅度过小 (L={left_closure:.1f}%, R={right_closure:.1f}%)"

    # 计算对称性比值
    symmetry_ratio = min_closure / max_closure if max_closure > 0 else 1.0

    # 阈值调整 - 更严格
    if symmetry_ratio >= 0.95:
        return 1, f"正常 (对称性{symmetry_ratio:.2%})"
    elif symmetry_ratio >= 0.85:
        return 2, f"轻度异常 (对称性{symmetry_ratio:.2%})"
    elif symmetry_ratio >= 0.70:
        return 3, f"中度异常 (对称性{symmetry_ratio:.2%})"
    elif symmetry_ratio >= 0.50:
        return 4, f"重度异常 (对称性{symmetry_ratio:.2%})"
    else:
        return 5, f"完全面瘫 (对称性{symmetry_ratio:.2%})"


def detect_synkinesis(baseline_result: Optional[ActionResult],
                      current_landmarks, w: int, h: int) -> Dict[str, int]:
    """检测闭眼时的联动运动"""
    synkinesis = {
        "mouth_synkinesis": 0,
        "cheek_synkinesis": 0,
    }

    if baseline_result is None:
        return synkinesis

    # 检测嘴部联动
    mouth = compute_mouth_metrics(current_landmarks, w, h)
    baseline_mouth_w = baseline_result.mouth_width

    if baseline_mouth_w > 1e-9:
        mouth_change = abs(mouth["width"] - baseline_mouth_w) / baseline_mouth_w
        if mouth_change > 0.15:
            synkinesis["mouth_synkinesis"] = 3
        elif mouth_change > 0.08:
            synkinesis["mouth_synkinesis"] = 2
        elif mouth_change > 0.04:
            synkinesis["mouth_synkinesis"] = 1

    return synkinesis


def plot_eye_curve(eye_seq: Dict, fps: float, peak_idx: int,
                   output_path: Path, action_name: str,
                   valid_mask: List[bool] = None,
                   palsy_detection: Dict[str, Any] = None) -> None:
    """绘制眼睛变化曲线"""
    from clinical_base import add_valid_region_shading, get_palsy_side_text

    fig, axes = plt.subplots(2, 1, figsize=(16, 9))  # 增加宽度

    frames = np.arange(len(eye_seq["ear"]["left"]))
    time_sec = frames / fps if fps > 0 else frames

    # 上图: EAR曲线
    ax1 = axes[0]

    if valid_mask is not None:
        add_valid_region_shading(ax1, valid_mask, time_sec)

    ax1.plot(time_sec, eye_seq["ear"]["left"], 'b-', label='Left Eye EAR', linewidth=2)
    ax1.plot(time_sec, eye_seq["ear"]["right"], 'r-', label='Right Eye EAR', linewidth=2)
    ax1.plot(time_sec, eye_seq["ear"]["average"], 'g--', label='Average EAR', linewidth=1.5, alpha=0.7)
    ax1.axvline(x=time_sec[peak_idx], color='k', linestyle='--', alpha=0.5, label=f'Peak Frame ({peak_idx})')
    ax1.axhline(y=THR.EYE_CLOSURE_EAR, color='orange', linestyle=':', alpha=0.7, label='Closure Threshold')
    ax1.set_xlabel('Time (seconds)' if fps > 0 else 'Frame', fontsize=11)
    ax1.set_ylabel('Eye Aspect Ratio (EAR)', fontsize=11)

    title = f'{action_name} - Eye Aspect Ratio Over Time'
    if palsy_detection:
        palsy_text = get_palsy_side_text(palsy_detection.get("palsy_side", 0))
        title += f' | Detected: {palsy_text}'
    ax1.set_title(title, fontsize=13)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 下图: 面积曲线
    ax2 = axes[1]

    if valid_mask is not None:
        add_valid_region_shading(ax2, valid_mask, time_sec)

    ax2.plot(time_sec, eye_seq["area"]["left"], 'b-', label='Left Eye Area', linewidth=2)
    ax2.plot(time_sec, eye_seq["area"]["right"], 'r-', label='Right Eye Area', linewidth=2)
    ax2.axvline(x=time_sec[peak_idx], color='k', linestyle='--', alpha=0.5, label=f'Peak Frame ({peak_idx})')
    ax2.set_xlabel('Time (seconds)' if fps > 0 else 'Frame', fontsize=11)
    ax2.set_ylabel('Eye Area (pixels²)', fontsize=11)
    ax2.set_title(f'{action_name} - Eye Area Over Time', fontsize=13)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()


def visualize_close_eye(frame: np.ndarray, landmarks, w: int, h: int,
                        result: ActionResult,
                        metrics: Dict[str, Any]) -> np.ndarray:
    """可视化闭眼指标 - 字体放大版"""
    img = frame.copy()

    # 字体参数
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE_TITLE = 1.4
    FONT_SCALE_NORMAL = 0.9
    THICKNESS_TITLE = 3
    THICKNESS_NORMAL = 2
    LINE_HEIGHT = 45

    # 患侧标签
    palsy_detection = metrics.get("palsy_detection", {})
    img = draw_palsy_side_label(img, palsy_detection, x=20, y=70, font_scale=1.4)

    # 绘制眼部轮廓
    draw_polygon(img, landmarks, w, h, LM.EYE_CONTOUR_L, (255, 0, 0), 3)
    draw_polygon(img, landmarks, w, h, LM.EYE_CONTOUR_R, (0, 165, 255), 3)

    # 信息面板
    panel_w, panel_h = 700, 600
    cv2.rectangle(img, (10, 100), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.rectangle(img, (10, 100), (panel_w, panel_h), (255, 255, 255), 2)

    y = 160
    cv2.putText(img, f"{result.action_name}", (25, y), FONT, FONT_SCALE_TITLE, (0, 255, 0), THICKNESS_TITLE)
    y += LINE_HEIGHT + 15

    cv2.putText(img, "=== Eye Aspect Ratio ===", (25, y), FONT, FONT_SCALE_NORMAL, (0, 255, 255), THICKNESS_NORMAL)
    y += LINE_HEIGHT

    cv2.putText(img, f"Left EAR: {metrics['left_ear']:.4f}", (25, y), FONT, FONT_SCALE_NORMAL, (255, 0, 0),
                THICKNESS_NORMAL)
    y += LINE_HEIGHT

    cv2.putText(img, f"Right EAR: {metrics['right_ear']:.4f}", (25, y), FONT, FONT_SCALE_NORMAL, (0, 165, 255),
                THICKNESS_NORMAL)
    y += LINE_HEIGHT

    ratio = metrics['ear_ratio']
    ratio_color = (0, 255, 0) if 0.85 <= ratio <= 1.15 else (0, 0, 255)
    cv2.putText(img, f"EAR Ratio: {ratio:.3f}", (25, y), FONT, FONT_SCALE_NORMAL, ratio_color, THICKNESS_NORMAL)
    y += LINE_HEIGHT + 10

    if "left_closure_percent" in metrics:
        cv2.putText(img, "=== Closure from Baseline ===", (25, y), FONT, FONT_SCALE_NORMAL, (0, 255, 255),
                    THICKNESS_NORMAL)
        y += LINE_HEIGHT
        cv2.putText(img,
                    f"Left: {metrics['left_closure_percent']:.1f}%  Right: {metrics['right_closure_percent']:.1f}%",
                    (25, y), FONT, FONT_SCALE_NORMAL, (255, 255, 255), THICKNESS_NORMAL)
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


def process_close_eye_softly(landmarks_seq: List, frames_seq: List, w: int, h: int,
                             video_info: Dict[str, Any], output_dir: Path,
                             baseline_result: Optional[ActionResult] = None,
                             baseline_landmarks=None) -> Optional[ActionResult]:
    """处理CloseEyeSoftly动作"""
    return _process_close_eye(
        landmarks_seq, frames_seq, w, h, video_info, output_dir,
        action_name="CloseEyeSoftly",
        action_name_cn="轻闭眼",
        baseline_result=baseline_result,
        baseline_landmarks=baseline_landmarks
    )


def process_close_eye_hardly(landmarks_seq: List, frames_seq: List, w: int, h: int,
                             video_info: Dict[str, Any], output_dir: Path,
                             baseline_result: Optional[ActionResult] = None,
                             baseline_landmarks=None) -> Optional[ActionResult]:
    """处理CloseEyeHardly动作"""
    return _process_close_eye(
        landmarks_seq, frames_seq, w, h, video_info, output_dir,
        action_name="CloseEyeHardly",
        action_name_cn="用力闭眼",
        baseline_result=baseline_result,
        baseline_landmarks=baseline_landmarks
    )


def _process_close_eye(landmarks_seq: List, frames_seq: List, w: int, h: int,
                       video_info: Dict[str, Any], output_dir: Path,
                       action_name: str, action_name_cn: str,
                       baseline_result: Optional[ActionResult] = None,
                       baseline_landmarks=None) -> Optional[ActionResult]:
    """处理闭眼动作的通用函数"""
    if not landmarks_seq or not frames_seq:
        return None

    # 计算整个视频的闭合度序列
    closure_data = compute_eye_closure_sequence(landmarks_seq, w, h, baseline_landmarks)

    # 找峰值帧 (眼睛面积最小/闭合度最大)
    peak_idx = find_peak_frame_close_eye(landmarks_seq, frames_seq, w, h, baseline_landmarks)
    peak_landmarks = landmarks_seq[peak_idx]
    peak_frame = frames_seq[peak_idx]

    # 计算同步度（整个视频）
    synchrony = compute_eye_synchrony(
        closure_data["left_closure"],
        closure_data["right_closure"]
    )

    # 计算峰值帧对称度
    left_closure_at_peak = closure_data["left_closure"][peak_idx]
    right_closure_at_peak = closure_data["right_closure"][peak_idx]
    symmetry_at_peak = compute_eye_symmetry_at_peak(left_closure_at_peak, right_closure_at_peak)

    if peak_landmarks is None:
        return None

    fps = video_info.get("fps", 30.0)

    # 创建结果对象
    result = ActionResult(
        action_name=action_name,
        action_name_cn=action_name_cn,
        video_path=video_info.get("file_path", ""),
        total_frames=len(frames_seq),
        peak_frame_idx=peak_idx,
        image_size=(w, h),
        fps=fps
    )

    # 提取通用指标
    extract_common_indicators(peak_landmarks, w, h, result, baseline_landmarks)

    # 计算闭眼特有指标
    metrics = compute_close_eye_metrics(peak_landmarks, w, h, baseline_landmarks)

    # 提取眼睛变化序列
    eye_seq = extract_eye_sequence(landmarks_seq, w, h)

    # 计算Voluntary Movement评分
    score, interpretation = compute_voluntary_score(metrics, baseline_landmarks)
    result.voluntary_movement_score = score

    # 检测联动
    synkinesis = detect_synkinesis(baseline_result, peak_landmarks, w, h)
    result.synkinesis_scores = synkinesis

    # 面瘫侧别检测
    palsy_detection = {
        "palsy_side": symmetry_at_peak["palsy_side"],
        "confidence": 1.0 - symmetry_at_peak["symmetry_score"],
        "interpretation": symmetry_at_peak["interpretation"],
        "method": "area_closure_symmetry",
    }

    # 计算严重度分数 (医生标注标准: 1=正常, 5=面瘫)
    severity_score, severity_desc = compute_severity_score(metrics)

    # 存储动作特有指标
    result.action_specific = {
        "close_eye_metrics": {
            "left_ear": metrics["left_ear"],
            "right_ear": metrics["right_ear"],
            "ear_ratio": metrics["ear_ratio"],
            "left_area": metrics["left_area"],
            "right_area": metrics["right_area"],
            "area_ratio": metrics["area_ratio"],
        },
        "eye_sequence": {
            "ear_left": [float(x) if not np.isnan(x) else None for x in eye_seq["ear"]["left"]],
            "ear_right": [float(x) if not np.isnan(x) else None for x in eye_seq["ear"]["right"]],
            "area_left": [float(x) if not np.isnan(x) else None for x in eye_seq["area"]["left"]],
            "area_right": [float(x) if not np.isnan(x) else None for x in eye_seq["area"]["right"]],
        },
        # 闭合度序列（用于曲线绘制）
        "closure_sequence": {
            "left": [float(v) if np.isfinite(v) else None for v in closure_data["left_closure"]],
            "right": [float(v) if np.isfinite(v) else None for v in closure_data["right_closure"]],
        },

        # 基线信息
        "baseline": {
            "left_area": float(closure_data["baseline_left_area"]),
            "right_area": float(closure_data["baseline_right_area"]),
        },

        # 峰值帧信息
        "peak_frame": {
            "idx": int(peak_idx),
            "left_closure": float(left_closure_at_peak),
            "right_closure": float(right_closure_at_peak),
            "left_area": float(closure_data["left_area"][peak_idx]),
            "right_area": float(closure_data["right_area"][peak_idx]),
        },

        # 对称度分析
        "symmetry_analysis": symmetry_at_peak,

        # 同步度分析
        "synchrony_analysis": synchrony,

        # 面瘫检测
        "palsy_detection": palsy_detection,

        # 诊断汇总
        "diagnosis_summary": {
            "palsy_side": palsy_detection["palsy_side"],
            "symmetry_score": symmetry_at_peak["symmetry_score"],
            "synchrony_score": synchrony["synchrony_score"],
            "pearson_correlation": synchrony["pearson_correlation"],
        }
    }

    if "left_closure_percent" in metrics:
        result.action_specific["closure_metrics"] = {
            "left_closure_percent": metrics["left_closure_percent"],
            "right_closure_percent": metrics["right_closure_percent"],
            "left_closure_ratio": metrics["left_closure_ratio"],
            "right_closure_ratio": metrics["right_closure_ratio"],
        }

    # 创建输出目录
    action_dir = output_dir / action_name
    action_dir.mkdir(parents=True, exist_ok=True)

    # 保存原始帧
    cv2.imwrite(str(action_dir / "peak_raw.jpg"), peak_frame)

    # 保存可视化
    vis = visualize_close_eye(peak_frame, peak_landmarks, w, h, result, metrics)
    cv2.imwrite(str(action_dir / "peak_indicators.jpg"), vis)

    # 绘制眼睛变化曲线
    plot_eye_curve(eye_seq, fps, peak_idx, action_dir / "eye_curve.png", action_name,
                   valid_mask=None,  # close_eye 没有 valid 逻辑
                   palsy_detection=palsy_detection)

    # 保存JSON
    with open(action_dir / "indicators.json", 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"    [OK] {action_name}: EAR L={metrics['left_ear']:.4f} R={metrics['right_ear']:.4f}")
    if "left_closure_percent" in metrics:
        print(f"         Closure L={metrics['left_closure_percent']:.1f}% R={metrics['right_closure_percent']:.1f}%")
    print(f"         Palsy: {palsy_detection.get('interpretation', 'N/A')}")
    print(f"         Voluntary Score: {result.voluntary_movement_score}/5 ({interpretation})")

    return result
