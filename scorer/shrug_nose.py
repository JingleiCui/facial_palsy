#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShrugNose 动作处理模块
======================

分析皱鼻动作:
1. 鼻翼位置变化
2. 鼻唇沟变化
3. 上唇位置变化
4. 联动运动检测

对应Sunnybrook: Snarl (LLA/LLS)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json

from clinical_base import (
    LM, pt2d, pts2d, dist, compute_ear, compute_eye_area,
    compute_mouth_metrics, compute_nlf_length, compute_nose_wrinkle_metrics,
    compute_icd, extract_common_indicators,
    ActionResult, draw_polygon
)

ACTION_NAME = "ShrugNose"
ACTION_NAME_CN = "皱鼻"


def find_peak_frame(landmarks_seq: List, frames_seq: List, w: int, h: int,
                    baseline_landmarks=None) -> int:
    """
    找皱鼻峰值帧

    皱鼻时鼻翼上提，鼻唇沟变短
    使用鼻唇沟长度最小的帧
    """
    min_nlf = float('inf')
    min_idx = 0

    for i, lm in enumerate(landmarks_seq):
        if lm is None:
            continue

        # 计算双侧NLF平均值
        left_nlf = compute_nlf_length(lm, w, h, left=True)
        right_nlf = compute_nlf_length(lm, w, h, left=False)
        avg_nlf = (left_nlf + right_nlf) / 2

        if avg_nlf < min_nlf:
            min_nlf = avg_nlf
            min_idx = i

    return min_idx


def compute_shrug_nose_metrics(landmarks, w: int, h: int,
                               baseline_landmarks=None) -> Dict[str, Any]:
    """计算皱鼻特有指标"""
    # NLF长度
    left_nlf = compute_nlf_length(landmarks, w, h, left=True)
    right_nlf = compute_nlf_length(landmarks, w, h, left=False)

    # 鼻部指标
    nose_metrics = compute_nose_wrinkle_metrics(landmarks, w, h)

    metrics = {
        "left_nlf": left_nlf,
        "right_nlf": right_nlf,
        "nlf_ratio": left_nlf / right_nlf if right_nlf > 1e-9 else 1.0,
        "avg_nlf": (left_nlf + right_nlf) / 2,
        "ala_width": nose_metrics["ala_width"],
        "left_ala_to_bridge": nose_metrics["left_ala_to_bridge"],
        "right_ala_to_bridge": nose_metrics["right_ala_to_bridge"],
        "ala_ratio": nose_metrics["ala_ratio"],
        "nose_points": {
            "tip": nose_metrics["nose_tip"],
            "bridge": nose_metrics["nose_bridge"],
            "left_ala": nose_metrics["left_ala"],
            "right_ala": nose_metrics["right_ala"],
        }
    }

    # 如果有基线，计算变化
    if baseline_landmarks is not None:
        baseline_left_nlf = compute_nlf_length(baseline_landmarks, w, h, left=True)
        baseline_right_nlf = compute_nlf_length(baseline_landmarks, w, h, left=False)
        baseline_nose = compute_nose_wrinkle_metrics(baseline_landmarks, w, h)

        metrics["baseline"] = {
            "left_nlf": baseline_left_nlf,
            "right_nlf": baseline_right_nlf,
            "avg_nlf": (baseline_left_nlf + baseline_right_nlf) / 2,
            "ala_width": baseline_nose["ala_width"],
        }

        # NLF变化 (皱鼻时应为负值，表示NLF缩短)
        metrics["left_nlf_change"] = left_nlf - baseline_left_nlf
        metrics["right_nlf_change"] = right_nlf - baseline_right_nlf

        # NLF变化比例
        if baseline_left_nlf > 1e-9:
            metrics["left_nlf_change_percent"] = (left_nlf - baseline_left_nlf) / baseline_left_nlf * 100
        else:
            metrics["left_nlf_change_percent"] = 0

        if baseline_right_nlf > 1e-9:
            metrics["right_nlf_change_percent"] = (right_nlf - baseline_right_nlf) / baseline_right_nlf * 100
        else:
            metrics["right_nlf_change_percent"] = 0

        # 鼻翼宽度变化
        metrics["ala_width_change"] = nose_metrics["ala_width"] - baseline_nose["ala_width"]

    return metrics


def compute_voluntary_score(metrics: Dict[str, Any], baseline_landmarks=None) -> Tuple[int, str]:
    """
    计算Voluntary Movement评分

    基于NLF变化的程度和对称性
    """
    if baseline_landmarks is not None and "left_nlf_change" in metrics:
        left_change = metrics["left_nlf_change"]
        right_change = metrics["right_nlf_change"]

        # 皱鼻时NLF应该变短 (负变化)
        left_contraction = -left_change
        right_contraction = -right_change

        # 检查是否有明显运动
        max_contraction = max(left_contraction, right_contraction)
        min_contraction = min(left_contraction, right_contraction)

        if max_contraction < 2:  # 几乎没有收缩
            return 1, "无法启动运动 (NLF变化过小)"

        # 计算对称性
        if min_contraction < 0:  # 一侧反而变长
            return 2, "轻微启动 (单侧异常)"

        if max_contraction < 1e-9:
            symmetry_ratio = 1.0
        else:
            symmetry_ratio = min_contraction / max_contraction

        if symmetry_ratio >= 0.85:
            if max_contraction > 8:
                return 5, "运动完整 (对称且幅度大)"
            elif max_contraction > 5:
                return 4, "几乎完整"
            else:
                return 3, "启动但幅度不足"
        elif symmetry_ratio >= 0.60:
            return 3, "启动但不对称"
        elif symmetry_ratio >= 0.30:
            return 2, "轻微启动"
        else:
            return 1, "无法启动"
    else:
        # 没有基线，使用静态比值
        ratio = metrics.get("nlf_ratio", 1.0)
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
    """检测皱鼻时的联动运动"""
    synkinesis = {
        "eye_synkinesis": 0,
        "mouth_synkinesis": 0,
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

        if avg_change > 0.18:
            synkinesis["eye_synkinesis"] = 3
        elif avg_change > 0.10:
            synkinesis["eye_synkinesis"] = 2
        elif avg_change > 0.05:
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


def visualize_shrug_nose(frame: np.ndarray, landmarks, w: int, h: int,
                         result: ActionResult,
                         metrics: Dict[str, Any]) -> np.ndarray:
    """可视化皱鼻指标"""
    img = frame.copy()

    # 绘制鼻唇沟
    l_ala = pt2d(landmarks[LM.NOSE_ALA_L], w, h)
    r_ala = pt2d(landmarks[LM.NOSE_ALA_R], w, h)
    l_mouth = pt2d(landmarks[LM.MOUTH_L], w, h)
    r_mouth = pt2d(landmarks[LM.MOUTH_R], w, h)

    cv2.line(img, (int(l_ala[0]), int(l_ala[1])), (int(l_mouth[0]), int(l_mouth[1])), (255, 0, 0), 3)
    cv2.line(img, (int(r_ala[0]), int(r_ala[1])), (int(r_mouth[0]), int(r_mouth[1])), (0, 165, 255), 3)

    # 绘制鼻翼点
    cv2.circle(img, (int(l_ala[0]), int(l_ala[1])), 6, (255, 0, 0), -1)
    cv2.circle(img, (int(r_ala[0]), int(r_ala[1])), 6, (0, 165, 255), -1)

    # 绘制鼻翼连线
    cv2.line(img, (int(l_ala[0]), int(l_ala[1])),
             (int(r_ala[0]), int(r_ala[1])), (0, 255, 255), 2)

    # 绘制嘴角点
    cv2.circle(img, (int(l_mouth[0]), int(l_mouth[1])), 5, (255, 0, 0), -1)
    cv2.circle(img, (int(r_mouth[0]), int(r_mouth[1])), 5, (0, 165, 255), -1)

    # 信息面板
    panel_h = 260
    cv2.rectangle(img, (5, 5), (380, panel_h), (0, 0, 0), -1)
    cv2.rectangle(img, (5, 5), (380, panel_h), (255, 255, 255), 1)

    y = 28
    cv2.putText(img, f"{ACTION_NAME}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y += 28

    cv2.putText(img, "=== Nasolabial Fold (NLF) ===", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    y += 20

    cv2.putText(img, f"Left NLF: {metrics['left_nlf']:.1f}px", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
    y += 18

    cv2.putText(img, f"Right NLF: {metrics['right_nlf']:.1f}px", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
    y += 18

    ratio = metrics['nlf_ratio']
    ratio_color = (0, 255, 0) if 0.9 <= ratio <= 1.1 else (0, 0, 255)
    cv2.putText(img, f"NLF Ratio: {ratio:.3f}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, ratio_color, 1)
    y += 22

    if "left_nlf_change" in metrics:
        cv2.putText(img, "=== NLF Changes from Baseline ===", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        y += 20

        cv2.putText(img,
                    f"Left: {metrics['left_nlf_change']:+.1f}px ({metrics.get('left_nlf_change_percent', 0):+.1f}%)",
                    (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 18

        cv2.putText(img,
                    f"Right: {metrics['right_nlf_change']:+.1f}px ({metrics.get('right_nlf_change_percent', 0):+.1f}%)",
                    (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 22

    cv2.putText(img, f"Ala Width: {metrics['ala_width']:.1f}px", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    y += 25

    # Voluntary Score
    cv2.putText(img, f"Voluntary Score: {result.voluntary_movement_score}/5", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    return img


def process(landmarks_seq: List, frames_seq: List, w: int, h: int,
            video_info: Dict[str, Any], output_dir: Path,
            baseline_result: Optional[ActionResult] = None,
            baseline_landmarks=None) -> Optional[ActionResult]:
    """处理ShrugNose动作"""
    if not landmarks_seq or not frames_seq:
        return None

    # 找峰值帧 (NLF最短)
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

    # 计算皱鼻特有指标
    metrics = compute_shrug_nose_metrics(peak_landmarks, w, h, baseline_landmarks)

    # 计算Voluntary Movement评分
    score, interpretation = compute_voluntary_score(metrics, baseline_landmarks)
    result.voluntary_movement_score = score

    # 检测联动
    synkinesis = detect_synkinesis(baseline_result, peak_landmarks, w, h)
    result.synkinesis_scores = synkinesis

    # 存储动作特有指标
    result.action_specific = {
        "shrug_nose_metrics": metrics,
        "voluntary_interpretation": interpretation,
        "synkinesis": synkinesis,
    }

    # 创建输出目录
    action_dir = output_dir / ACTION_NAME
    action_dir.mkdir(parents=True, exist_ok=True)

    # 保存原始帧
    cv2.imwrite(str(action_dir / "peak_raw.jpg"), peak_frame)

    # 保存可视化
    vis = visualize_shrug_nose(peak_frame, peak_landmarks, w, h, result, metrics)
    cv2.imwrite(str(action_dir / "peak_indicators.jpg"), vis)

    # 保存JSON
    with open(action_dir / "indicators.json", 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"    [OK] {ACTION_NAME}: NLF L={metrics['left_nlf']:.1f} R={metrics['right_nlf']:.1f}")
    if "left_nlf_change" in metrics:
        print(f"         Change L={metrics['left_nlf_change']:+.1f}px R={metrics['right_nlf_change']:+.1f}px")
    print(f"         Voluntary Score: {result.voluntary_movement_score}/5 ({interpretation})")

    return result