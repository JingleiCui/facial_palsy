#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
微笑动作处理模块
==========================

分析:
1. 嘴角位移和对称性
2. 口角角度变化
3. 运动幅度对比
4. 面瘫侧别检测
5. 联动运动检测 (眼部联动)

修复内容:
- 移除错误的NLF分析
- 使用口角角度和运动幅度作为主要指标
- 添加面瘫侧别检测

对应Sunnybrook: Open mouth smile (ZYG/RIS)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import math
import json

from clinical_base import (
    LM, pt2d, pts2d, dist, compute_ear, compute_eye_area,
    compute_mouth_metrics, compute_oral_angle, compute_icd,
    extract_common_indicators,
    ActionResult, OralAngleMeasure, draw_polygon
)

from sunnybrook_scorer import (
    VoluntaryMovementItem, compute_voluntary_score_from_ratio
)


def find_peak_frame_smile(landmarks_seq: List, frames_seq: List, w: int, h: int) -> int:
    """找微笑峰值帧 (嘴宽最大)"""
    max_width = -1.0
    max_idx = 0

    for i, lm in enumerate(landmarks_seq):
        if lm is None:
            continue
        l_corner = pt2d(lm[LM.MOUTH_L], w, h)
        r_corner = pt2d(lm[LM.MOUTH_R], w, h)
        width = dist(l_corner, r_corner)
        if width > max_width:
            max_width = width
            max_idx = i

    return max_idx


def compute_smile_metrics(landmarks, w: int, h: int,
                          baseline_landmarks=None) -> Dict[str, Any]:
    """计算微笑特有指标"""
    mouth = compute_mouth_metrics(landmarks, w, h)
    oral = compute_oral_angle(landmarks, w, h)

    # 嘴角位置
    left_corner = mouth["left_corner"]
    right_corner = mouth["right_corner"]

    # 嘴角高度 (相对于嘴中心)
    mouth_center_y = (left_corner[1] + right_corner[1]) / 2
    left_height_from_center = mouth_center_y - left_corner[1]  # 正值表示左嘴角较高
    right_height_from_center = mouth_center_y - right_corner[1]

    metrics = {
        "mouth_width": mouth["width"],
        "mouth_height": mouth["height"],
        "left_corner": left_corner,
        "right_corner": right_corner,
        "left_corner_height": left_height_from_center,
        "right_corner_height": right_height_from_center,
        "corner_height_diff": left_height_from_center - right_height_from_center,
        "oral_angle": {
            "AOE": oral.AOE_angle,
            "BOF": oral.BOF_angle,
            "diff": oral.angle_diff,
            "asymmetry": oral.angle_asymmetry,
        }
    }

    # 如果有基线，计算运动幅度
    if baseline_landmarks is not None:
        baseline_mouth = compute_mouth_metrics(baseline_landmarks, w, h)
        baseline_oral = compute_oral_angle(baseline_landmarks, w, h)
        baseline_left = baseline_mouth["left_corner"]
        baseline_right = baseline_mouth["right_corner"]

        # 嘴角位移
        left_excursion = dist(left_corner, baseline_left)
        right_excursion = dist(right_corner, baseline_right)

        # 水平和垂直分量
        left_horizontal = left_corner[0] - baseline_left[0]
        left_vertical = baseline_left[1] - left_corner[1]  # 向上为正
        right_horizontal = right_corner[0] - baseline_right[0]
        right_vertical = baseline_right[1] - right_corner[1]

        metrics["excursion"] = {
            "left_total": left_excursion,
            "right_total": right_excursion,
            "excursion_ratio": left_excursion / right_excursion if right_excursion > 1e-9 else 1.0,
            "left_horizontal": left_horizontal,
            "left_vertical": left_vertical,
            "right_horizontal": right_horizontal,
            "right_vertical": right_vertical,
            "baseline_width": baseline_mouth["width"],
            "width_change": mouth["width"] - baseline_mouth["width"],
        }

        # 口角角度变化
        metrics["oral_angle_change"] = {
            "AOE_change": oral.AOE_angle - baseline_oral.AOE_angle,
            "BOF_change": oral.BOF_angle - baseline_oral.BOF_angle,
        }

    return metrics


def detect_palsy_side(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    从微笑动作检测面瘫侧别

    原理:
    1. 面瘫侧口角运动幅度小
    2. 面瘫侧口角角度低(下垂)

    Returns:
        Dict包含:
        - palsy_side: 0=无/对称, 1=左, 2=右
        - confidence: 置信度
        - interpretation: 解释
    """
    result = {"palsy_side": 0, "confidence": 0.0, "interpretation": ""}

    oral = metrics.get("oral_angle", {})
    aoe = oral.get("AOE", 0)  # 右侧口角角度
    bof = oral.get("BOF", 0)  # 左侧口角角度
    asymmetry = oral.get("asymmetry", 0)

    # 使用运动幅度作为主要指标
    if "excursion" in metrics:
        exc = metrics["excursion"]
        left_exc = exc["left_total"]
        right_exc = exc["right_total"]

        if max(left_exc, right_exc) < 3:  # 运动幅度太小
            result["interpretation"] = "微笑运动幅度过小"
            return result

        exc_ratio = exc["excursion_ratio"]

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
        # 没有基线，使用口角角度
        if asymmetry < 3:
            result["palsy_side"] = 0
            result["confidence"] = 1.0 - asymmetry / 10
            result["interpretation"] = f"口角对称 (不对称度={asymmetry:.1f}°)"
        elif aoe < bof:
            result["palsy_side"] = 2
            result["confidence"] = min(1.0, asymmetry / 15)
            result["interpretation"] = f"右口角位置较低 (AOE={aoe:+.1f}° < BOF={bof:+.1f}°)"
        else:
            result["palsy_side"] = 1
            result["confidence"] = min(1.0, asymmetry / 15)
            result["interpretation"] = f"左口角位置较低 (BOF={bof:+.1f}° < AOE={aoe:+.1f}°)"

    return result


def detect_synkinesis_from_smile(baseline_result: Optional[ActionResult],
                                 current_landmarks, w: int, h: int) -> Dict[str, int]:
    """检测微笑时的联动运动 (主要是眼部)"""
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

    # 检测眼部联动 (微笑时眼睛变小)
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


def visualize_smile_indicators(frame: np.ndarray, landmarks, w: int, h: int,
                               result: ActionResult,
                               smile_metrics: Dict[str, Any],
                               palsy_detection: Dict[str, Any]) -> np.ndarray:
    """可视化微笑指标"""
    img = frame.copy()

    # 绘制嘴部轮廓
    draw_polygon(img, landmarks, w, h, LM.OUTER_LIP, (0, 255, 0), 2)

    # 绘制嘴角点
    if result.oral_angle:
        oral = result.oral_angle
        cv2.circle(img, (int(oral.A[0]), int(oral.A[1])), 6, (0, 0, 255), -1)
        cv2.circle(img, (int(oral.B[0]), int(oral.B[1])), 6, (255, 0, 0), -1)
        cv2.circle(img, (int(oral.O[0]), int(oral.O[1])), 4, (255, 255, 255), -1)

        # 绘制EF水平参考线
        cv2.line(img, (int(oral.E[0]), int(oral.E[1])),
                 (int(oral.F[0]), int(oral.F[1])), (0, 255, 0), 2)

        # 绘制O到A和O到B的连线
        cv2.line(img, (int(oral.O[0]), int(oral.O[1])),
                 (int(oral.A[0]), int(oral.A[1])), (0, 0, 255), 2)
        cv2.line(img, (int(oral.O[0]), int(oral.O[1])),
                 (int(oral.B[0]), int(oral.B[1])), (255, 0, 0), 2)

    # 信息面板
    panel_h = 300
    cv2.rectangle(img, (5, 5), (400, panel_h), (0, 0, 0), -1)
    cv2.rectangle(img, (5, 5), (400, panel_h), (255, 255, 255), 1)

    y = 25
    cv2.putText(img, f"{result.action_name}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y += 30

    cv2.putText(img, f"Mouth Width: {smile_metrics['mouth_width']:.1f}px", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y += 22

    oral_angle = smile_metrics.get("oral_angle", {})
    cv2.putText(img, f"AOE(R): {oral_angle.get('AOE', 0):+.1f}  BOF(L): {oral_angle.get('BOF', 0):+.1f}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y += 22

    asym = oral_angle.get('asymmetry', 0)
    asym_color = (0, 255, 0) if asym < 5 else ((0, 165, 255) if asym < 10 else (0, 0, 255))
    cv2.putText(img, f"Asymmetry: {asym:.1f} deg", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, asym_color, 1)
    y += 25

    # 运动幅度
    if "excursion" in smile_metrics:
        exc = smile_metrics["excursion"]
        cv2.putText(img, "=== Excursion ===", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y += 22
        cv2.putText(img, f"L: {exc['left_total']:.1f}px  R: {exc['right_total']:.1f}px", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 18
        cv2.putText(img, f"Ratio: {exc['excursion_ratio']:.3f}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 18
        cv2.putText(img, f"Width Change: {exc['width_change']:+.1f}px", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 22

    # 面瘫侧别检测结果
    palsy_side = palsy_detection.get("palsy_side", 0)
    palsy_text = {0: "无/对称", 1: "左侧", 2: "右侧"}.get(palsy_side, "未知")
    palsy_color = (0, 255, 0) if palsy_side == 0 else (0, 0, 255)
    cv2.putText(img, f"Palsy Side: {palsy_text}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, palsy_color, 1)
    y += 25

    # Voluntary Score
    cv2.putText(img, f"Voluntary Score: {result.voluntary_movement_score}/5", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return img


def process_smile(landmarks_seq: List, frames_seq: List, w: int, h: int,
                  video_info: Dict[str, Any], output_dir: Path,
                  baseline_result: Optional[ActionResult] = None,
                  baseline_landmarks=None) -> Optional[ActionResult]:
    """处理Smile动作"""
    return _process_smile_action(
        landmarks_seq, frames_seq, w, h, video_info, output_dir,
        action_name="Smile",
        action_name_cn="微笑",
        baseline_result=baseline_result,
        baseline_landmarks=baseline_landmarks
    )


def process_show_teeth(landmarks_seq: List, frames_seq: List, w: int, h: int,
                       video_info: Dict[str, Any], output_dir: Path,
                       baseline_result: Optional[ActionResult] = None,
                       baseline_landmarks=None) -> Optional[ActionResult]:
    """处理ShowTeeth动作"""
    return _process_smile_action(
        landmarks_seq, frames_seq, w, h, video_info, output_dir,
        action_name="ShowTeeth",
        action_name_cn="露齿",
        baseline_result=baseline_result,
        baseline_landmarks=baseline_landmarks
    )


def _process_smile_action(landmarks_seq: List, frames_seq: List, w: int, h: int,
                          video_info: Dict[str, Any], output_dir: Path,
                          action_name: str, action_name_cn: str,
                          baseline_result: Optional[ActionResult] = None,
                          baseline_landmarks=None) -> Optional[ActionResult]:
    """处理微笑类动作的通用函数"""
    if not landmarks_seq or not frames_seq:
        return None

    # 找峰值帧
    peak_idx = find_peak_frame_smile(landmarks_seq, frames_seq, w, h)
    peak_landmarks = landmarks_seq[peak_idx]
    peak_frame = frames_seq[peak_idx]

    if peak_landmarks is None:
        return None

    # 创建结果对象
    result = ActionResult(
        action_name=action_name,
        action_name_cn=action_name_cn,
        video_path=video_info.get("file_path", ""),
        total_frames=len(frames_seq),
        peak_frame_idx=peak_idx,
        image_size=(w, h),
        fps=video_info.get("fps", 30.0)
    )

    # 提取通用指标
    extract_common_indicators(peak_landmarks, w, h, result)

    # 计算微笑特有指标
    smile_metrics = compute_smile_metrics(peak_landmarks, w, h, baseline_landmarks)

    # 检测面瘫侧别
    palsy_detection = detect_palsy_side(smile_metrics)

    # 检测联动
    synkinesis = detect_synkinesis_from_smile(baseline_result, peak_landmarks, w, h)
    result.synkinesis_scores = synkinesis

    # 计算Voluntary Movement评分
    if "excursion" in smile_metrics:
        exc_ratio = smile_metrics["excursion"]["excursion_ratio"]
        score, interp = compute_voluntary_score_from_ratio(exc_ratio)
        result.voluntary_movement_score = score
    else:
        # 没有基线时使用口角对称性
        oral = smile_metrics.get("oral_angle", {})
        asym = oral.get("asymmetry", 0)
        if asym < 3:
            result.voluntary_movement_score = 5
        elif asym < 6:
            result.voluntary_movement_score = 4
        elif asym < 10:
            result.voluntary_movement_score = 3
        elif asym < 15:
            result.voluntary_movement_score = 2
        else:
            result.voluntary_movement_score = 1

    # 存储动作特有指标
    result.action_specific = {
        "smile_metrics": {
            "mouth_width": smile_metrics["mouth_width"],
            "mouth_height": smile_metrics["mouth_height"],
            "oral_angle": smile_metrics["oral_angle"],
        },
        "palsy_detection": palsy_detection,
        "synkinesis": synkinesis,
        "voluntary_score": result.voluntary_movement_score,
    }

    if "excursion" in smile_metrics:
        result.action_specific["excursion"] = smile_metrics["excursion"]

    # 创建输出目录
    action_dir = output_dir / action_name
    action_dir.mkdir(parents=True, exist_ok=True)

    # 保存原始帧
    cv2.imwrite(str(action_dir / "peak_raw.jpg"), peak_frame)

    # 保存可视化
    vis = visualize_smile_indicators(peak_frame, peak_landmarks, w, h, result,
                                     smile_metrics, palsy_detection)
    cv2.imwrite(str(action_dir / "peak_indicators.jpg"), vis)

    # 保存JSON
    with open(action_dir / "indicators.json", 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    oral = smile_metrics.get("oral_angle", {})
    print(f"    [OK] {action_name}: Width={smile_metrics['mouth_width']:.1f}px, Asym={oral.get('asymmetry', 0):.1f}°")
    print(f"         Palsy: {palsy_detection.get('interpretation', 'N/A')}")
    print(f"         Voluntary Score: {result.voluntary_movement_score}/5")

    return result