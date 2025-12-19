#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RaiseEyebrow 动作处理模块
=========================

分析抬眉/皱额动作:
1. 眉眼距
2. 眉眼距变化度
3. 双侧眉眼距比和变化度比
4. 联动运动检测

对应Sunnybrook: Brow (Forehead wrinkle)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json

from clinical_base import (
    LM, pt2d, pts2d, dist, compute_ear, compute_eye_area,
    compute_brow_height, compute_brow_position, compute_mouth_metrics,
    compute_icd, extract_common_indicators,
    compute_brow_eye_distance, compute_brow_eye_distance_ratio,
    compute_brow_eye_distance_change, compute_brow_eye_distance_change_ratio,
    compute_brow_centroid,
    ActionResult, draw_polygon, draw_landmarks
)

ACTION_NAME = "RaiseEyebrow"
ACTION_NAME_CN = "抬眉/皱额"


def find_peak_frame(landmarks_seq: List, frames_seq: List, w: int, h: int,
                    baseline_landmarks=None) -> int:
    """
    找抬眉峰值帧

    使用眉眼距最大的帧作为峰值帧
    如果有基线，使用眉眼距变化度最大的帧
    """
    max_bed = -1.0
    max_idx = 0

    for i, lm in enumerate(landmarks_seq):
        if lm is None:
            continue

        if baseline_landmarks is not None:
            # 使用变化度
            left_change = compute_brow_eye_distance_change(lm, baseline_landmarks, w, h, left=True)
            right_change = compute_brow_eye_distance_change(lm, baseline_landmarks, w, h, left=False)
            # 取两侧变化度的平均
            avg_change = (left_change["change"] + right_change["change"]) / 2
            if avg_change > max_bed:
                max_bed = avg_change
                max_idx = i
        else:
            # 使用绝对值
            bed_result = compute_brow_eye_distance_ratio(lm, w, h)
            avg_bed = (bed_result["left_distance"] + bed_result["right_distance"]) / 2
            if avg_bed > max_bed:
                max_bed = avg_bed
                max_idx = i

    return max_idx


def compute_raise_eyebrow_metrics(landmarks, w: int, h: int,
                                  baseline_landmarks=None) -> Dict[str, Any]:
    """计算抬眉特有指标"""
    # 当前眉眼距
    bed_result = compute_brow_eye_distance_ratio(landmarks, w, h)

    metrics = {
        "left_brow_eye_distance": bed_result["left_distance"],
        "right_brow_eye_distance": bed_result["right_distance"],
        "brow_eye_distance_ratio": bed_result["ratio"],
        "left_eye_inner": bed_result["left_eye_inner"],
        "right_eye_inner": bed_result["right_eye_inner"],
        "left_brow_centroid": bed_result["left_brow_centroid"],
        "right_brow_centroid": bed_result["right_brow_centroid"],
    }

    # 如果有基线，计算变化度
    if baseline_landmarks is not None:
        bedc_result = compute_brow_eye_distance_change_ratio(landmarks, baseline_landmarks, w, h)
        metrics["left_change"] = bedc_result["left_change"]
        metrics["right_change"] = bedc_result["right_change"]
        metrics["change_ratio"] = bedc_result["ratio"]
        metrics["left_baseline_distance"] = bedc_result["left_baseline_distance"]
        metrics["right_baseline_distance"] = bedc_result["right_baseline_distance"]

        # 计算变化百分比
        if bedc_result["left_baseline_distance"] > 1e-9:
            metrics["left_change_percent"] = bedc_result["left_change"] / bedc_result["left_baseline_distance"] * 100
        else:
            metrics["left_change_percent"] = 0

        if bedc_result["right_baseline_distance"] > 1e-9:
            metrics["right_change_percent"] = bedc_result["right_change"] / bedc_result["right_baseline_distance"] * 100
        else:
            metrics["right_change_percent"] = 0

    return metrics


def compute_voluntary_score(metrics: Dict[str, Any], baseline_landmarks=None) -> Tuple[int, str]:
    """
    计算Voluntary Movement评分

    基于眉眼距变化度的对称性
    """
    if baseline_landmarks is not None and "change_ratio" in metrics:
        ratio = metrics["change_ratio"]
        left_change = metrics.get("left_change", 0)
        right_change = metrics.get("right_change", 0)

        # 首先检查是否有明显的运动
        min_change = min(abs(left_change), abs(right_change))
        max_change = max(abs(left_change), abs(right_change))

        if max_change < 3:  # 几乎没有运动
            return 1, "无法启动运动 (变化度过小)"

        # 基于比值的对称性评分
        if min_change < 1e-9:
            # 一侧完全没动
            return 1, "无法启动运动 (单侧无运动)"

        # 计算对称比 (较小/较大)
        symmetry_ratio = min_change / max_change

        if symmetry_ratio >= 0.90:
            return 5, "运动完整 (对称性>90%)"
        elif symmetry_ratio >= 0.75:
            return 4, "几乎完整 (对称性75-90%)"
        elif symmetry_ratio >= 0.50:
            return 3, "启动但不对称 (对称性50-75%)"
        elif symmetry_ratio >= 0.25:
            return 2, "轻微启动 (对称性25-50%)"
        else:
            return 1, "无法启动 (对称性<25%)"
    else:
        # 没有基线，使用静态比值
        ratio = metrics.get("brow_eye_distance_ratio", 1.0)
        deviation = abs(ratio - 1.0)

        if deviation <= 0.05:
            return 5, "运动完整"
        elif deviation <= 0.10:
            return 4, "几乎完整"
        elif deviation <= 0.20:
            return 3, "启动但不对称"
        elif deviation <= 0.35:
            return 2, "轻微启动"
        else:
            return 1, "无法启动"


def detect_synkinesis(baseline_result: Optional[ActionResult],
                      current_landmarks, w: int, h: int) -> Dict[str, int]:
    """检测抬眉时的联动运动"""
    synkinesis = {
        "eye_synkinesis": 0,
        "mouth_synkinesis": 0,
    }

    if baseline_result is None:
        return synkinesis

    # 检测眼部联动 (抬眉时眼睛变大)
    l_ear = compute_ear(current_landmarks, w, h, True)
    r_ear = compute_ear(current_landmarks, w, h, False)

    baseline_l_ear = baseline_result.left_ear
    baseline_r_ear = baseline_result.right_ear

    if baseline_l_ear > 1e-9 and baseline_r_ear > 1e-9:
        l_change = (l_ear - baseline_l_ear) / baseline_l_ear
        r_change = (r_ear - baseline_r_ear) / baseline_r_ear
        avg_change = (l_change + r_change) / 2

        if abs(avg_change) > 0.20:
            synkinesis["eye_synkinesis"] = 3
        elif abs(avg_change) > 0.12:
            synkinesis["eye_synkinesis"] = 2
        elif abs(avg_change) > 0.06:
            synkinesis["eye_synkinesis"] = 1

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


def visualize_brow_eye_distance(frame: np.ndarray, landmarks, w: int, h: int,
                                result: ActionResult,
                                metrics: Dict[str, Any],
                                baseline_metrics: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """可视化眉眼距"""
    img = frame.copy()

    # 绘制眉毛轮廓
    draw_polygon(img, landmarks, w, h, LM.BROW_L, (255, 100, 100), 2, False)
    draw_polygon(img, landmarks, w, h, LM.BROW_R, (100, 165, 255), 2, False)

    # 绘制眼部轮廓
    draw_polygon(img, landmarks, w, h, LM.EYE_CONTOUR_L, (255, 0, 0), 1)
    draw_polygon(img, landmarks, w, h, LM.EYE_CONTOUR_R, (0, 165, 255), 1)

    # 获取关键点
    left_eye_inner = metrics["left_eye_inner"]
    right_eye_inner = metrics["right_eye_inner"]
    left_brow_centroid = metrics["left_brow_centroid"]
    right_brow_centroid = metrics["right_brow_centroid"]

    # 绘制眉毛质心 (红色)
    cv2.circle(img, (int(left_brow_centroid[0]), int(left_brow_centroid[1])), 6, (0, 0, 255), -1)
    cv2.circle(img, (int(right_brow_centroid[0]), int(right_brow_centroid[1])), 6, (0, 0, 255), -1)

    # 绘制眼内眦点 (蓝色)
    cv2.circle(img, (int(left_eye_inner[0]), int(left_eye_inner[1])), 5, (255, 0, 0), -1)
    cv2.circle(img, (int(right_eye_inner[0]), int(right_eye_inner[1])), 5, (255, 0, 0), -1)

    # 绘制眉眼距连线 (黄色)
    cv2.line(img, (int(left_eye_inner[0]), int(left_eye_inner[1])),
             (int(left_brow_centroid[0]), int(left_brow_centroid[1])), (0, 255, 255), 2)
    cv2.line(img, (int(right_eye_inner[0]), int(right_eye_inner[1])),
             (int(right_brow_centroid[0]), int(right_brow_centroid[1])), (0, 255, 255), 2)

    # 信息面板
    panel_h = 280
    cv2.rectangle(img, (5, 5), (380, panel_h), (0, 0, 0), -1)
    cv2.rectangle(img, (5, 5), (380, panel_h), (255, 255, 255), 1)

    y = 28
    cv2.putText(img, f"{ACTION_NAME}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y += 28

    cv2.putText(img, "=== Brow-Eye Distance ===", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    y += 20

    cv2.putText(img, f"Left: {metrics['left_brow_eye_distance']:.1f}px", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    y += 18

    cv2.putText(img, f"Right: {metrics['right_brow_eye_distance']:.1f}px", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    y += 18

    ratio = metrics['brow_eye_distance_ratio']
    ratio_color = (0, 255, 0) if 0.9 <= ratio <= 1.1 else (0, 165, 255) if 0.8 <= ratio <= 1.2 else (0, 0, 255)
    cv2.putText(img, f"Ratio: {ratio:.3f}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, ratio_color, 1)
    y += 25

    if "left_change" in metrics:
        cv2.putText(img, "=== Distance Change ===", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        y += 20

        left_change = metrics["left_change"]
        right_change = metrics["right_change"]

        cv2.putText(img, f"Left: {left_change:+.1f}px ({metrics.get('left_change_percent', 0):+.1f}%)", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 18

        cv2.putText(img, f"Right: {right_change:+.1f}px ({metrics.get('right_change_percent', 0):+.1f}%)",
                    (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 18

        change_ratio = metrics.get("change_ratio", 1.0)
        if not np.isinf(change_ratio):
            cv2.putText(img, f"Change Ratio: {change_ratio:.3f}", (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 25

    # Voluntary Score
    cv2.putText(img, f"Voluntary Score: {result.voluntary_movement_score}/5", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # 图例
    legend_y = panel_h + 15
    cv2.circle(img, (20, legend_y), 5, (0, 0, 255), -1)
    cv2.putText(img, "Brow Centroid", (35, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.circle(img, (150, legend_y), 5, (255, 0, 0), -1)
    cv2.putText(img, "Eye Inner", (165, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.line(img, (260, legend_y), (290, legend_y), (0, 255, 255), 2)
    cv2.putText(img, "BED", (295, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    return img


def process(landmarks_seq: List, frames_seq: List, w: int, h: int,
            video_info: Dict[str, Any], output_dir: Path,
            baseline_result: Optional[ActionResult] = None,
            baseline_landmarks=None) -> Optional[ActionResult]:
    """
    处理RaiseEyebrow动作

    Args:
        landmarks_seq: landmarks序列
        frames_seq: 帧序列
        w, h: 图像尺寸
        video_info: 视频信息
        output_dir: 输出目录
        baseline_result: NeutralFace的结果 (用于联动检测)
        baseline_landmarks: NeutralFace的landmarks (用于变化度计算)

    Returns:
        ActionResult 或 None
    """
    if not landmarks_seq or not frames_seq:
        return None

    # 找峰值帧
    peak_idx = find_peak_frame(landmarks_seq, frames_seq, w, h, baseline_landmarks)
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

    # 计算抬眉特有指标
    metrics = compute_raise_eyebrow_metrics(peak_landmarks, w, h, baseline_landmarks)

    # 计算Voluntary Movement评分
    score, interpretation = compute_voluntary_score(metrics, baseline_landmarks)
    result.voluntary_movement_score = score

    # 检测联动
    synkinesis = detect_synkinesis(baseline_result, peak_landmarks, w, h)
    result.synkinesis_scores = synkinesis

    # 存储动作特有指标
    result.action_specific = {
        "brow_eye_metrics": {
            "left_brow_eye_distance": metrics["left_brow_eye_distance"],
            "right_brow_eye_distance": metrics["right_brow_eye_distance"],
            "brow_eye_distance_ratio": metrics["brow_eye_distance_ratio"],
        },
        "voluntary_interpretation": interpretation,
        "synkinesis": synkinesis,
    }

    if "left_change" in metrics:
        result.action_specific["brow_eye_metrics"].update({
            "left_change": metrics["left_change"],
            "right_change": metrics["right_change"],
            "change_ratio": metrics.get("change_ratio", 1.0),
            "left_change_percent": metrics.get("left_change_percent", 0),
            "right_change_percent": metrics.get("right_change_percent", 0),
        })

    # 创建输出目录
    action_dir = output_dir / ACTION_NAME
    action_dir.mkdir(parents=True, exist_ok=True)

    # 保存原始帧
    cv2.imwrite(str(action_dir / "peak_raw.jpg"), peak_frame)

    # 保存可视化
    baseline_metrics = None
    if baseline_landmarks is not None:
        baseline_metrics = compute_raise_eyebrow_metrics(baseline_landmarks, w, h, None)

    vis = visualize_brow_eye_distance(peak_frame, peak_landmarks, w, h, result, metrics, baseline_metrics)
    cv2.imwrite(str(action_dir / "peak_brow_eye_distance.jpg"), vis)

    # 保存JSON
    with open(action_dir / "indicators.json", 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    print(
        f"    [OK] {ACTION_NAME}: BED L={metrics['left_brow_eye_distance']:.1f} R={metrics['right_brow_eye_distance']:.1f}")
    if "left_change" in metrics:
        print(f"         Change L={metrics['left_change']:+.1f}px R={metrics['right_change']:+.1f}px")
    print(f"         Voluntary Score: {result.voluntary_movement_score}/5 ({interpretation})")

    return result