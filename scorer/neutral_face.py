#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NeutralFace 动作处理模块
========================

静息面分析 - 用于:
1. 提取基线指标
2. 计算Resting Symmetry评分
3. 作为其他动作的参考基准
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from clinical_base import (
    LM, pt2d, pts2d, dist, compute_ear, compute_eye_area,
    compute_palpebral_height, compute_palpebral_width,
    compute_brow_height, compute_brow_position,
    compute_mouth_metrics, compute_oral_angle, compute_nlf_length,
    compute_icd, extract_common_indicators,
    ActionResult, OralAngleMeasure,
    draw_text_with_background, draw_landmarks, draw_polygon, compute_brow_eye_distance
)

from sunnybrook_scorer import (
    RestingSymmetry, compute_resting_symmetry
)

ACTION_NAME = "NeutralFace"
ACTION_NAME_CN = "静息面"


def find_peak_frame(landmarks_seq: List, frames_seq: List, w: int, h: int) -> int:
    """
    找静息峰值帧

    使用min(left_ear, right_ear)确保两只眼睛都睁开
    """
    max_min_ear = -1.0
    max_idx = 0

    for i, lm in enumerate(landmarks_seq):
        if lm is None:
            continue
        l_ear = compute_ear(lm, w, h, True)
        r_ear = compute_ear(lm, w, h, False)
        min_ear = min(l_ear, r_ear)
        if min_ear > max_min_ear:
            max_min_ear = min_ear
            max_idx = i

    return max_idx


def extract_action_specific_indicators(landmarks, w: int, h: int,
                                       result: ActionResult) -> None:
    """提取NeutralFace特有指标"""
    # NeutralFace主要用于基线，大部分指标已在common中提取
    # 这里添加一些特有的分析

    result.action_specific = {
        "purpose": "baseline",
        "description": "用作其他动作的基线参考",
        "eye_symmetry": {
            "ear_left": result.left_ear,
            "ear_right": result.right_ear,
            "ear_ratio": result.left_ear / result.right_ear if result.right_ear > 1e-9 else 1.0,
            "area_ratio": result.eye_area_ratio,
            "height_ratio": result.palpebral_height_ratio,
        },
        "mouth_symmetry": {
            "oral_angle_diff": result.oral_angle.angle_diff if result.oral_angle else 0,
            "oral_angle_asymmetry": result.oral_angle.angle_asymmetry if result.oral_angle else 0,
        },
        "face_symmetry": {
            "nlf_ratio": result.nlf_ratio,
            "brow_height_ratio": result.brow_height_ratio,
        }
    }


def compute_resting_symmetry_from_result(result: ActionResult) -> RestingSymmetry:
    """从ActionResult计算RestingSymmetry"""
    oral = result.oral_angle
    return compute_resting_symmetry(
        palpebral_height_ratio=result.palpebral_height_ratio,
        nlf_ratio=result.nlf_ratio,
        oral_angle_diff=oral.angle_diff if oral else 0,
        aoe_angle=oral.AOE_angle if oral else 0,
        bof_angle=oral.BOF_angle if oral else 0
    )


def visualize_indicators(frame: np.ndarray, landmarks, w: int, h: int,
                         result: ActionResult) -> np.ndarray:
    """可视化NeutralFace指标"""
    img = frame.copy()

    # 绘制眼部轮廓
    draw_polygon(img, landmarks, w, h, LM.EYE_CONTOUR_L, (255, 0, 0), 2)
    draw_polygon(img, landmarks, w, h, LM.EYE_CONTOUR_R, (0, 165, 255), 2)

    # 绘制眉毛
    draw_polygon(img, landmarks, w, h, LM.BROW_L, (255, 100, 100), 1, False)
    draw_polygon(img, landmarks, w, h, LM.BROW_R, (100, 165, 255), 1, False)

    # 绘制嘴部轮廓
    draw_polygon(img, landmarks, w, h, LM.OUTER_LIP, (200, 200, 200), 1)

    # 绘制鼻唇沟
    l_ala = pt2d(landmarks[LM.NOSE_ALA_L], w, h)
    r_ala = pt2d(landmarks[LM.NOSE_ALA_R], w, h)
    l_mouth = pt2d(landmarks[LM.MOUTH_L], w, h)
    r_mouth = pt2d(landmarks[LM.MOUTH_R], w, h)
    cv2.line(img, (int(l_ala[0]), int(l_ala[1])), (int(l_mouth[0]), int(l_mouth[1])), (255, 0, 0), 2)
    cv2.line(img, (int(r_ala[0]), int(r_ala[1])), (int(r_mouth[0]), int(r_mouth[1])), (0, 165, 255), 2)

    # 口角点
    if result.oral_angle:
        oral = result.oral_angle
        cv2.circle(img, (int(oral.A[0]), int(oral.A[1])), 5, (0, 0, 255), -1)
        cv2.circle(img, (int(oral.B[0]), int(oral.B[1])), 5, (255, 0, 0), -1)
        cv2.circle(img, (int(oral.O[0]), int(oral.O[1])), 4, (255, 255, 255), -1)
        cv2.line(img, (int(oral.E[0]), int(oral.E[1])), (int(oral.F[0]), int(oral.F[1])), (0, 255, 0), 1)

    # 信息面板
    y = 25
    cv2.putText(img, f"{ACTION_NAME}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y += 30

    cv2.putText(img, f"ICD: {result.icd:.1f}px", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    y += 18

    cv2.putText(img, f"Eye Area L/R: {result.eye_area_ratio:.3f}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    y += 18

    cv2.putText(img, f"EAR L:{result.left_ear:.3f} R:{result.right_ear:.3f}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    y += 18

    cv2.putText(img, f"Palpebral H Ratio: {result.palpebral_height_ratio:.3f}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    y += 18

    cv2.putText(img, f"Brow H Ratio: {result.brow_height_ratio:.3f}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    y += 18

    if result.oral_angle:
        cv2.putText(img, f"AOE(R):{result.oral_angle.AOE_angle:+.1f} BOF(L):{result.oral_angle.BOF_angle:+.1f}",
                    (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 18

    cv2.putText(img, f"NLF Ratio: {result.nlf_ratio:.3f}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return img


def visualize_resting_symmetry(frame: np.ndarray, landmarks, w: int, h: int,
                               result: ActionResult,
                               resting: RestingSymmetry) -> np.ndarray:
    """可视化Resting Symmetry评估"""
    img = frame.copy()

    # 绘制眼部
    draw_polygon(img, landmarks, w, h, LM.EYE_CONTOUR_L, (255, 0, 0), 2)
    draw_polygon(img, landmarks, w, h, LM.EYE_CONTOUR_R, (0, 165, 255), 2)

    # 绘制鼻唇沟
    l_ala = pt2d(landmarks[LM.NOSE_ALA_L], w, h)
    r_ala = pt2d(landmarks[LM.NOSE_ALA_R], w, h)
    l_mouth = pt2d(landmarks[LM.MOUTH_L], w, h)
    r_mouth = pt2d(landmarks[LM.MOUTH_R], w, h)
    cv2.line(img, (int(l_ala[0]), int(l_ala[1])), (int(l_mouth[0]), int(l_mouth[1])), (255, 0, 0), 2)
    cv2.line(img, (int(r_ala[0]), int(r_ala[1])), (int(r_mouth[0]), int(r_mouth[1])), (0, 165, 255), 2)

    # 口角
    if result.oral_angle:
        oral = result.oral_angle
        cv2.circle(img, (int(oral.A[0]), int(oral.A[1])), 5, (0, 0, 255), -1)
        cv2.circle(img, (int(oral.B[0]), int(oral.B[1])), 5, (255, 0, 0), -1)

    # 信息面板
    panel_h = 240
    cv2.rectangle(img, (5, 5), (420, panel_h), (0, 0, 0), -1)
    cv2.rectangle(img, (5, 5), (420, panel_h), (255, 255, 255), 1)

    y = 28
    cv2.putText(img, "RESTING SYMMETRY (Sunnybrook)", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    y += 30

    # Eye
    eye_color = (0, 255, 0) if resting.eye.score == 0 else (0, 0, 255)
    cv2.putText(img, f"Eye: {resting.eye.status_cn} (score={resting.eye.score})", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, eye_color, 1)
    y += 18
    cv2.putText(img, f"  {resting.eye.threshold_info}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    y += 22

    # Cheek
    cheek_color = (0, 255, 0) if resting.cheek.score == 0 else (
        (0, 165, 255) if resting.cheek.score == 1 else (0, 0, 255))
    cv2.putText(img, f"Cheek: {resting.cheek.status_cn} (score={resting.cheek.score})", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, cheek_color, 1)
    y += 18
    cv2.putText(img, f"  {resting.cheek.threshold_info}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    y += 22

    # Mouth
    mouth_color = (0, 255, 0) if resting.mouth.score == 0 else (0, 0, 255)
    cv2.putText(img, f"Mouth: {resting.mouth.status_cn} (score={resting.mouth.score})", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, mouth_color, 1)
    y += 18
    cv2.putText(img, f"  {resting.mouth.threshold_info}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    y += 25

    # 分隔线
    cv2.line(img, (15, y - 5), (405, y - 5), (100, 100, 100), 1)
    y += 12

    # 总分
    total_color = (0, 255, 0) if resting.raw_score == 0 else (0, 0, 255)
    cv2.putText(img, f"Raw: {resting.raw_score} x 5 = {resting.total_score} (Resting Score)", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, total_color, 2)
    y += 25

    # 患侧
    side_color = (0, 255, 255) if "Uncertain" not in resting.affected_side else (128, 128, 128)
    cv2.putText(img, f"Affected Side: {resting.affected_side}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, side_color, 1)

    return img


def process(landmarks_seq: List, frames_seq: List, w: int, h: int,
            video_info: Dict[str, Any], output_dir: Path) -> Optional[ActionResult]:
    """
    处理NeutralFace动作

    Args:
        landmarks_seq: landmarks序列
        frames_seq: 帧序列
        w, h: 图像尺寸
        video_info: 视频信息
        output_dir: 输出目录

    Returns:
        ActionResult 或 None
    """
    if not landmarks_seq or not frames_seq:
        return None

    # 找峰值帧
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
    extract_common_indicators(peak_landmarks, w, h, result)

    # ========== 存储 baseline ICD 和关键距离 ==========
    icd_base = compute_icd(peak_landmarks, w, h)

    # 存储 baseline 距离（原始像素值，作为后续动作的参考）
    baseline_distances = {
        "icd": icd_base,
        "left_palpebral_height": compute_palpebral_height(peak_landmarks, w, h, left=True),
        "right_palpebral_height": compute_palpebral_height(peak_landmarks, w, h, left=False),
        "left_brow_eye_distance": compute_brow_eye_distance(peak_landmarks, w, h, left=True),
        "right_brow_eye_distance": compute_brow_eye_distance(peak_landmarks, w, h, left=False),
        "mouth_width": result.mouth_width,
        "mouth_height": result.mouth_height,
        "left_nlf_length": result.left_nlf_length,
        "right_nlf_length": result.right_nlf_length,
    }

    # 提取动作特有指标
    extract_action_specific_indicators(peak_landmarks, w, h, result)

    # 计算Resting Symmetry
    resting = compute_resting_symmetry_from_result(result)
    result.action_specific["resting_symmetry"] = resting.to_dict()

    # ========== 存储 baseline 距离 ==========
    result.action_specific["baseline_distances"] = baseline_distances

    # 创建输出目录
    action_dir = output_dir / ACTION_NAME
    action_dir.mkdir(parents=True, exist_ok=True)

    # 保存原始帧
    cv2.imwrite(str(action_dir / "peak_raw.jpg"), peak_frame)

    # 保存可视化
    vis_indicators = visualize_indicators(peak_frame, peak_landmarks, w, h, result)
    cv2.imwrite(str(action_dir / "peak_indicators.jpg"), vis_indicators)

    vis_resting = visualize_resting_symmetry(peak_frame, peak_landmarks, w, h, result, resting)
    cv2.imwrite(str(action_dir / "resting_symmetry.jpg"), vis_resting)

    # 保存JSON
    import json
    with open(action_dir / "indicators.json", 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"    [OK] {ACTION_NAME}: EAR L={result.left_ear:.3f} R={result.right_ear:.3f}")
    print(f"         Resting Score: {resting.total_score}, Affected: {resting.affected_side}")

    return result