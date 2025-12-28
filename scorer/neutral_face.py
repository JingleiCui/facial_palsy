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
    draw_text_with_background, draw_landmarks, draw_polygon, compute_brow_eye_distance,
)

from sunnybrook_scorer import (
    RestingSymmetry, compute_resting_symmetry
)

from thresholds import THR

ACTION_NAME = "NeutralFace"
ACTION_NAME_CN = "静息面"


def find_peak_frame(landmarks_seq: List, frames_seq: List, w: int, h: int) -> int:
    """
    找静息峰值帧

    标准：
    1. 眼睛睁开（EAR > 阈值）
    2. 嘴巴闭合（嘴高/嘴宽 < 阈值）
    3. 眉毛稳定（眉毛不要在运动过程中）
    4. 稳定区间（避免眨眼、张嘴或眉毛运动的瞬间）
    """
    from thresholds import THR

    n_frames = len(landmarks_seq)
    if n_frames == 0:
        return 0

    # 计算每帧的评分
    scores = []
    ear_values = []
    mouth_ratios = []
    brow_heights_left = []  # 左眉高度（眉眼距）
    brow_heights_right = []  # 右眉高度（眉眼距）

    for i, lm in enumerate(landmarks_seq):
        if lm is None:
            scores.append(-999)
            ear_values.append(0)
            mouth_ratios.append(1)
            brow_heights_left.append(0)
            brow_heights_right.append(0)
            continue

        # 眼睛睁开程度
        l_ear = compute_ear(lm, w, h, True)
        r_ear = compute_ear(lm, w, h, False)
        min_ear = min(l_ear, r_ear)
        avg_ear = (l_ear + r_ear) / 2

        # 嘴巴闭合程度
        mouth = compute_mouth_metrics(lm, w, h)
        mouth_ratio = mouth["height"] / mouth["width"] if mouth["width"] > 1e-9 else 0

        # 眉毛高度（眉眼距）
        # 使用简单的眉毛中心到眼睛内眦的Y轴距离
        left_brow_height = compute_brow_height(lm, w, h, left=True)
        right_brow_height = compute_brow_height(lm, w, h, left=False)

        ear_values.append(avg_ear)
        mouth_ratios.append(mouth_ratio)
        brow_heights_left.append(left_brow_height)
        brow_heights_right.append(right_brow_height)

        # 评分：眼睛越睁开越好，嘴巴越闭越好
        score = 0.0

        # 眼睛分数（EAR > 0.20 为睁开，越大越好）
        if min_ear > 0.20:
            score += min_ear * 10
        else:
            score -= (0.20 - min_ear) * 20  # 惩罚闭眼

        # 嘴巴分数（高宽比 < 0.15 为闭合，越小越好）
        if mouth_ratio < 0.15:
            score += (0.15 - mouth_ratio) * 10
        else:
            score -= (mouth_ratio - 0.15) * 20  # 惩罚张嘴

        scores.append(score)

    # 寻找稳定区间（避免快速变化的帧）
    # 使用滑动窗口计算稳定性
    window_size = min(5, n_frames // 3)
    if window_size < 3:
        window_size = 3

    stability_scores = []
    for i in range(n_frames):
        start = max(0, i - window_size // 2)
        end = min(n_frames, i + window_size // 2 + 1)

        window_ears = ear_values[start:end]
        window_mouths = mouth_ratios[start:end]
        window_brow_left = brow_heights_left[start:end]
        window_brow_right = brow_heights_right[start:end]

        # 计算窗口内的变化（标准差）
        ear_std = np.std(window_ears) if len(window_ears) > 1 else 0
        mouth_std = np.std(window_mouths) if len(window_mouths) > 1 else 0
        brow_left_std = np.std(window_brow_left) if len(window_brow_left) > 1 else 0
        brow_right_std = np.std(window_brow_right) if len(window_brow_right) > 1 else 0
        brow_std = (brow_left_std + brow_right_std) / 2  # 平均眉毛变化

        # 变化越小，稳定性越好
        # 加入眉毛稳定性权重
        stability = 1.0 / (1.0 + ear_std * 10 + mouth_std * 5 + brow_std * 3)
        stability_scores.append(stability)

    # 最终分数 = 基础分数 * 稳定性
    final_scores = [s * stab for s, stab in zip(scores, stability_scores)]

    # 找最高分
    best_idx = int(np.argmax(final_scores))

    return best_idx


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


def detect_resting_palsy(result: ActionResult) -> Dict[str, Any]:
    """
    检测静息状态下的面瘫侧别 - 增强版

    原理：即使在静息状态，面瘫侧也可能表现出：
    1. 眼睑裂不对称（高度或面积）
    2. 鼻唇沟变浅
    3. 口角下垂
    4. 眉毛高度不对称
    5. 面部各器官相对于面中线不对称

    增强：使用更多指标和加权投票机制
    """
    detection = {
        "palsy_side": 0,
        "confidence": 0.0,
        "interpretation": "",
        "method": "resting_symmetry_enhanced",
        "evidence": {}
    }

    # 收集证据 (side, weight, description)
    votes = []

    # ========== 1. 眼睛面积比 ==========
    area_ratio = result.eye_area_ratio
    if area_ratio is not None:
        detection["evidence"]["eye_area_ratio"] = area_ratio
        # 面积比显著偏离1表示不对称
        deviation = abs(area_ratio - 1.0)
        if deviation > 0.10:  # 10%以上差异
            weight = min(1.0, deviation * 2)  # 权重随差异增大
            if area_ratio < 1.0:
                # 左眼面积小 - 可能是左侧面瘫（眼睑下垂）或右侧（眼睑提升）
                # 静息态时，面瘫侧眼睑可能松弛下垂，导致面积略小
                votes.append((1, weight * 0.3, f"左眼面积较小 (ratio={area_ratio:.3f})"))
            else:
                votes.append((2, weight * 0.3, f"右眼面积较小 (ratio={area_ratio:.3f})"))

    # ========== 2. 眼睑裂高度比 ==========
    palp_ratio = result.palpebral_height_ratio
    if palp_ratio is not None:
        detection["evidence"]["palpebral_height_ratio"] = palp_ratio
        deviation = abs(palp_ratio - 1.0)
        if deviation > 0.12:  # 12%以上差异
            weight = min(1.0, deviation * 2)
            if palp_ratio < 1.0:
                votes.append((1, weight * 0.4, f"左眼睑裂高度较小 (ratio={palp_ratio:.3f})"))
            else:
                votes.append((2, weight * 0.4, f"右眼睑裂高度较小 (ratio={palp_ratio:.3f})"))

    # ========== 3. 鼻唇沟长度比 ==========
    nlf_ratio = result.nlf_ratio
    if nlf_ratio is not None:
        detection["evidence"]["nlf_ratio"] = nlf_ratio
        deviation = abs(nlf_ratio - 1.0)
        if deviation > 0.12:
            weight = min(1.0, deviation * 2)
            if nlf_ratio < 1.0:
                # 左侧NLF较短（变浅）-> 左侧面瘫
                votes.append((1, weight * 0.5, f"左侧鼻唇沟变浅 (ratio={nlf_ratio:.3f})"))
            else:
                votes.append((2, weight * 0.5, f"右侧鼻唇沟变浅 (ratio={nlf_ratio:.3f})"))

    # ========== 4. 口角角度 ==========
    if result.oral_angle:
        aoe = result.oral_angle.AOE_angle
        bof = result.oral_angle.BOF_angle
        angle_diff = abs(aoe - bof)
        detection["evidence"]["AOE_right"] = aoe
        detection["evidence"]["BOF_left"] = bof
        detection["evidence"]["oral_angle_diff"] = angle_diff

        if angle_diff > 3:
            weight = min(1.0, angle_diff / 15)
            if aoe < bof:
                # 右口角更低（角度更小/更负）
                votes.append((2, weight * 0.6, f"右口角下垂 (AOE={aoe:+.1f}°, BOF={bof:+.1f}°)"))
            else:
                votes.append((1, weight * 0.6, f"左口角下垂 (BOF={bof:+.1f}°, AOE={aoe:+.1f}°)"))

    # ========== 5. 眉高比 ==========
    brow_ratio = result.brow_height_ratio
    if brow_ratio is not None:
        detection["evidence"]["brow_height_ratio"] = brow_ratio
        deviation = abs(brow_ratio - 1.0)
        if deviation > 0.12:
            weight = min(1.0, deviation * 2)
            if brow_ratio < 1.0:
                # 左侧眉毛较低 -> 可能是左侧面瘫
                votes.append((1, weight * 0.3, f"左眉较低 (ratio={brow_ratio:.3f})"))
            else:
                votes.append((2, weight * 0.3, f"右眉较低 (ratio={brow_ratio:.3f})"))

    # ========== 综合投票 ==========
    if not votes:
        detection["palsy_side"] = 0
        detection["confidence"] = 0.0
        detection["interpretation"] = "静息状态面部对称，无明显面瘫迹象"
        return detection

    # 加权投票
    side_weights = {0: 0.0, 1: 0.0, 2: 0.0}
    for side, weight, _ in votes:
        side_weights[side] += weight

    total_weight = sum(side_weights.values())
    if total_weight < 0.3:  # 总权重太小
        detection["palsy_side"] = 0
        detection["confidence"] = 0.2
        detection["interpretation"] = "静息状态轻微不对称，但不足以判断面瘫侧别"
        return detection

    # 归一化
    for k in side_weights:
        side_weights[k] /= total_weight

    # 选择得票最高的
    final_side = max(side_weights, key=side_weights.get)
    final_conf = side_weights[final_side]

    detection["palsy_side"] = final_side
    detection["confidence"] = final_conf
    detection["vote_weights"] = side_weights
    detection["vote_details"] = [(side, weight, desc) for side, weight, desc in votes]

    if final_side == 0:
        detection["interpretation"] = "静息状态面部基本对称"
    elif final_side == 1:
        evidence_list = [desc for side, _, desc in votes if side == 1]
        detection["interpretation"] = f"静息态左侧面瘫迹象 (置信度{final_conf * 100:.0f}%): " + "; ".join(
            evidence_list[:2])
    else:
        evidence_list = [desc for side, _, desc in votes if side == 2]
        detection["interpretation"] = f"静息态右侧面瘫迹象 (置信度{final_conf * 100:.0f}%): " + "; ".join(
            evidence_list[:2])

    return detection


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

    # 检测静息状态面瘫
    resting_palsy = detect_resting_palsy(result)
    result.action_specific["resting_palsy_detection"] = resting_palsy
    result.action_specific["palsy_detection"] = resting_palsy

    # ========== 存储 baseline ICD 和关键距离 ==========
    icd_base = compute_icd(peak_landmarks, w, h)
    left_eye_area, _ = compute_eye_area(peak_landmarks, w, h, left=True)
    right_eye_area, _ = compute_eye_area(peak_landmarks, w, h, left=False)
    left_palp_width = compute_palpebral_width(peak_landmarks, w, h, left=True)
    right_palp_width = compute_palpebral_width(peak_landmarks, w, h, left=False)

    baseline_distances = {
        "icd": icd_base,
        "left_palpebral_height": compute_palpebral_height(peak_landmarks, w, h, left=True),
        "right_palpebral_height": compute_palpebral_height(peak_landmarks, w, h, left=False),
        "left_palpebral_width": left_palp_width,
        "right_palpebral_width": right_palp_width,
        "left_eye_area": left_eye_area,
        "right_eye_area": right_eye_area,
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