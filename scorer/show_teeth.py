#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShowTeeth 动作处理模块
======================

分析露齿动作:
1. 嘴角位移和对称性
2. 口角角度变化
3. 上唇提升程度
4. 鼻唇沟变化
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
    LM, pt2d, pts2d, dist, compute_ear, compute_eye_area,
    compute_mouth_metrics, compute_oral_angle, compute_nlf_length,
    compute_icd, extract_common_indicators,
    ActionResult, OralAngleMeasure, draw_polygon
)

from sunnybrook_scorer import (
    VoluntaryMovementItem, compute_voluntary_score_from_ratio
)

ACTION_NAME = "ShowTeeth"
ACTION_NAME_CN = "露齿"


def find_peak_frame(landmarks_seq: List, frames_seq: List, w: int, h: int) -> int:
    """
    找露齿峰值帧

    使用嘴宽最大的帧 (与Smile类似，但更强调张嘴程度)
    """
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


def compute_show_teeth_metrics(landmarks, w: int, h: int,
                               baseline_landmarks=None) -> Dict[str, Any]:
    """计算露齿特有指标"""
    mouth = compute_mouth_metrics(landmarks, w, h)
    oral = compute_oral_angle(landmarks, w, h)

    # 嘴角位置
    left_corner = mouth["left_corner"]
    right_corner = mouth["right_corner"]

    # 嘴角高度 (相对于嘴中心)
    mouth_center_y = (left_corner[1] + right_corner[1]) / 2
    left_height_from_center = mouth_center_y - left_corner[1]  # 正值表示左嘴角较高
    right_height_from_center = mouth_center_y - right_corner[1]

    # 上唇提升量 (通过上唇中心位置评估)
    lip_top = mouth["top_center"]

    # NLF长度
    left_nlf = compute_nlf_length(landmarks, w, h, left=True)
    right_nlf = compute_nlf_length(landmarks, w, h, left=False)

    metrics = {
        "mouth_width": mouth["width"],
        "mouth_height": mouth["height"],
        "mouth_opening_ratio": mouth["height"] / mouth["width"] if mouth["width"] > 1e-9 else 0,
        "left_corner": left_corner,
        "right_corner": right_corner,
        "lip_top": lip_top,
        "left_corner_height": left_height_from_center,
        "right_corner_height": right_height_from_center,
        "corner_height_diff": left_height_from_center - right_height_from_center,
        "left_nlf": left_nlf,
        "right_nlf": right_nlf,
        "nlf_ratio": left_nlf / right_nlf if right_nlf > 1e-9 else 1.0,
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
        baseline_left = baseline_mouth["left_corner"]
        baseline_right = baseline_mouth["right_corner"]
        baseline_top = baseline_mouth["top_center"]
        baseline_left_nlf = compute_nlf_length(baseline_landmarks, w, h, left=True)
        baseline_right_nlf = compute_nlf_length(baseline_landmarks, w, h, left=False)

        # 嘴角位移
        left_excursion = dist(left_corner, baseline_left)
        right_excursion = dist(right_corner, baseline_right)

        # 水平和垂直分量
        left_horizontal = left_corner[0] - baseline_left[0]
        left_vertical = baseline_left[1] - left_corner[1]  # 向上为正
        right_horizontal = right_corner[0] - baseline_right[0]
        right_vertical = baseline_right[1] - right_corner[1]

        # 上唇提升量
        lip_lift = baseline_top[1] - lip_top[1]  # 向上为正

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
            "lip_lift": lip_lift,
        }

        metrics["baseline"] = {
            "mouth_width": baseline_mouth["width"],
            "mouth_height": baseline_mouth["height"],
            "left_nlf": baseline_left_nlf,
            "right_nlf": baseline_right_nlf,
        }

        metrics["nlf_change"] = {
            "left": left_nlf - baseline_left_nlf,
            "right": right_nlf - baseline_right_nlf,
        }

    return metrics


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
                         metrics: Dict[str, Any]) -> np.ndarray:
    """可视化露齿指标"""
    img = frame.copy()

    # 绘制嘴部轮廓
    draw_polygon(img, landmarks, w, h, LM.OUTER_LIP, (0, 255, 0), 2)
    draw_polygon(img, landmarks, w, h, LM.INNER_LIP, (0, 200, 200), 1)

    # 绘制嘴角点
    if result.oral_angle:
        oral = result.oral_angle
        cv2.circle(img, (int(oral.A[0]), int(oral.A[1])), 6, (0, 0, 255), -1)  # 右嘴角 红色
        cv2.circle(img, (int(oral.B[0]), int(oral.B[1])), 6, (255, 0, 0), -1)  # 左嘴角 蓝色
        cv2.circle(img, (int(oral.O[0]), int(oral.O[1])), 4, (255, 255, 255), -1)  # 中心点 白色

        # 绘制EF水平参考线
        cv2.line(img, (int(oral.E[0]), int(oral.E[1])),
                 (int(oral.F[0]), int(oral.F[1])), (0, 255, 0), 2)

        # 绘制O到A和O到B的连线
        cv2.line(img, (int(oral.O[0]), int(oral.O[1])),
                 (int(oral.A[0]), int(oral.A[1])), (0, 0, 255), 2)
        cv2.line(img, (int(oral.O[0]), int(oral.O[1])),
                 (int(oral.B[0]), int(oral.B[1])), (255, 0, 0), 2)

    # 绘制鼻唇沟
    l_ala = pt2d(landmarks[LM.NOSE_ALA_L], w, h)
    r_ala = pt2d(landmarks[LM.NOSE_ALA_R], w, h)
    l_mouth = pt2d(landmarks[LM.MOUTH_L], w, h)
    r_mouth = pt2d(landmarks[LM.MOUTH_R], w, h)
    cv2.line(img, (int(l_ala[0]), int(l_ala[1])), (int(l_mouth[0]), int(l_mouth[1])), (255, 100, 100), 2)
    cv2.line(img, (int(r_ala[0]), int(r_ala[1])), (int(r_mouth[0]), int(r_mouth[1])), (100, 165, 255), 2)

    # 信息面板
    panel_h = 300
    cv2.rectangle(img, (5, 5), (400, panel_h), (0, 0, 0), -1)
    cv2.rectangle(img, (5, 5), (400, panel_h), (255, 255, 255), 1)

    y = 28
    cv2.putText(img, f"{ACTION_NAME}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y += 28

    cv2.putText(img, "=== Mouth Metrics ===", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    y += 20

    cv2.putText(img, f"Width: {metrics['mouth_width']:.1f}px  Height: {metrics['mouth_height']:.1f}px", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    y += 18

    cv2.putText(img, f"Opening Ratio: {metrics['mouth_opening_ratio']:.3f}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    y += 22

    # 口角角度
    oral_angle = metrics.get("oral_angle", {})
    cv2.putText(img, f"AOE(R): {oral_angle.get('AOE', 0):+.1f}  BOF(L): {oral_angle.get('BOF', 0):+.1f}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    y += 18

    asym = oral_angle.get('asymmetry', 0)
    asym_color = (0, 255, 0) if asym < 5 else ((0, 165, 255) if asym < 10 else (0, 0, 255))
    cv2.putText(img, f"Asymmetry: {asym:.1f} deg", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, asym_color, 1)
    y += 22

    # NLF
    cv2.putText(img, f"NLF L: {metrics['left_nlf']:.1f}px  R: {metrics['right_nlf']:.1f}px", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    y += 18

    nlf_ratio = metrics['nlf_ratio']
    nlf_color = (0, 255, 0) if 0.9 <= nlf_ratio <= 1.1 else (0, 0, 255)
    cv2.putText(img, f"NLF Ratio: {nlf_ratio:.3f}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, nlf_color, 1)
    y += 22

    # 运动幅度
    if "excursion" in metrics:
        exc = metrics["excursion"]
        cv2.putText(img, "=== Excursion ===", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        y += 20

        cv2.putText(img, f"L: {exc['left_total']:.1f}px  R: {exc['right_total']:.1f}px", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 18

        exc_ratio = exc['excursion_ratio']
        exc_color = (0, 255, 0) if 0.85 <= exc_ratio <= 1.15 else (0, 0, 255)
        cv2.putText(img, f"Ratio: {exc_ratio:.3f}", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, exc_color, 1)
        y += 18

        cv2.putText(img, f"Width Change: {exc['width_change']:+.1f}px", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 18

        if "lip_lift" in exc:
            cv2.putText(img, f"Lip Lift: {exc['lip_lift']:+.1f}px", (15, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            y += 22

    # Voluntary Score
    cv2.putText(img, f"Voluntary Score: {result.voluntary_movement_score}/5", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # 图例
    legend_y = panel_h + 15
    cv2.circle(img, (20, legend_y), 5, (0, 0, 255), -1)
    cv2.putText(img, "Right corner (A)", (30, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.circle(img, (160, legend_y), 5, (255, 0, 0), -1)
    cv2.putText(img, "Left corner (B)", (170, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

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

    # 计算Voluntary Movement评分
    score, interpretation = compute_voluntary_score(metrics, baseline_landmarks)
    result.voluntary_movement_score = score

    # 检测联动
    synkinesis = detect_synkinesis(baseline_result, peak_landmarks, w, h)
    result.synkinesis_scores = synkinesis

    # 存储动作特有指标
    result.action_specific = {
        "show_teeth_metrics": metrics,
        "voluntary_interpretation": interpretation,
        "synkinesis": synkinesis,
    }

    # 创建输出目录
    action_dir = output_dir / ACTION_NAME
    action_dir.mkdir(parents=True, exist_ok=True)

    # 保存原始帧
    cv2.imwrite(str(action_dir / "peak_raw.jpg"), peak_frame)

    # 保存可视化
    vis = visualize_show_teeth(peak_frame, peak_landmarks, w, h, result, metrics)
    cv2.imwrite(str(action_dir / "peak_indicators.jpg"), vis)

    # 保存JSON
    with open(action_dir / "indicators.json", 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    oral = metrics.get("oral_angle", {})
    print(f"    [OK] {ACTION_NAME}: Width={metrics['mouth_width']:.1f}px, Asym={oral.get('asymmetry', 0):.1f}°")
    if "excursion" in metrics:
        exc = metrics["excursion"]
        print(
            f"         Excursion L={exc['left_total']:.1f}px R={exc['right_total']:.1f}px Ratio={exc['excursion_ratio']:.3f}")
    print(f"         Voluntary Score: {result.voluntary_movement_score}/5 ({interpretation})")

    return result