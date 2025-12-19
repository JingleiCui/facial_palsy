#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BlowCheek 动作处理模块
======================

分析鼓腮动作:
1. 唇密封距离 (关键帧检测)
2. 面颊膨胀度
3. 嘴部闭合程度
4. 联动运动检测

关键帧检测方法:
- upper_lip: 0 (上唇上边界) 到 13 (上唇下边界)
- lower_lip: 14 (下唇上边界) 到 17 (下唇下边界)
- 鼓腮时嘴唇紧闭，上述两距离之和最小
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json

from clinical_base import (
    LM, pt2d, pts2d, dist, compute_ear, compute_eye_area,
    compute_mouth_metrics, compute_oral_angle, compute_nlf_length,
    compute_icd, extract_common_indicators, compute_lip_seal_distance,
    ActionResult, draw_polygon, draw_landmarks
)

ACTION_NAME = "BlowCheek"
ACTION_NAME_CN = "鼓腮"


def find_peak_frame(landmarks_seq: List, frames_seq: List, w: int, h: int) -> int:
    """
    找鼓腮峰值帧

    使用唇密封距离最小的帧作为峰值帧
    唇密封距离 = 上唇(0-13)距离 + 下唇(14-17)距离
    """
    min_seal_dist = float('inf')
    min_idx = 0

    for i, lm in enumerate(landmarks_seq):
        if lm is None:
            continue

        seal_result = compute_lip_seal_distance(lm, w, h)
        total_dist = seal_result["total_distance"]

        if total_dist < min_seal_dist:
            min_seal_dist = total_dist
            min_idx = i

    return min_idx


def compute_blow_cheek_metrics(landmarks, w: int, h: int,
                               baseline_landmarks=None) -> Dict[str, Any]:
    """计算鼓腮特有指标"""
    # 唇密封距离
    seal_result = compute_lip_seal_distance(landmarks, w, h)

    # 嘴部指标
    mouth = compute_mouth_metrics(landmarks, w, h)

    # 面颊区域分析 (使用鼻唇沟长度作为间接指标)
    left_nlf = compute_nlf_length(landmarks, w, h, left=True)
    right_nlf = compute_nlf_length(landmarks, w, h, left=False)

    metrics = {
        "lip_seal": {
            "upper_distance": seal_result["upper_distance"],
            "lower_distance": seal_result["lower_distance"],
            "total_distance": seal_result["total_distance"],
            "upper_outer": seal_result["upper_outer"],
            "upper_inner": seal_result["upper_inner"],
            "lower_inner": seal_result["lower_inner"],
            "lower_outer": seal_result["lower_outer"],
        },
        "mouth_width": mouth["width"],
        "mouth_height": mouth["height"],
        "left_nlf": left_nlf,
        "right_nlf": right_nlf,
        "nlf_ratio": left_nlf / right_nlf if right_nlf > 1e-9 else 1.0,
    }

    # 如果有基线，计算变化
    if baseline_landmarks is not None:
        baseline_seal = compute_lip_seal_distance(baseline_landmarks, w, h)
        baseline_mouth = compute_mouth_metrics(baseline_landmarks, w, h)
        baseline_left_nlf = compute_nlf_length(baseline_landmarks, w, h, left=True)
        baseline_right_nlf = compute_nlf_length(baseline_landmarks, w, h, left=False)

        metrics["baseline"] = {
            "lip_seal_total": baseline_seal["total_distance"],
            "mouth_width": baseline_mouth["width"],
            "left_nlf": baseline_left_nlf,
            "right_nlf": baseline_right_nlf,
        }

        # 变化量
        metrics["seal_change"] = seal_result["total_distance"] - baseline_seal["total_distance"]
        metrics["mouth_width_change"] = mouth["width"] - baseline_mouth["width"]
        metrics["left_nlf_change"] = left_nlf - baseline_left_nlf
        metrics["right_nlf_change"] = right_nlf - baseline_right_nlf

    return metrics


def compute_voluntary_score(metrics: Dict[str, Any], baseline_landmarks=None) -> Tuple[int, str]:
    """
    计算Voluntary Movement评分

    基于嘴部闭合程度和面颊膨胀的对称性
    """
    if baseline_landmarks is not None and "baseline" in metrics:
        # 使用NLF变化的对称性评估
        left_change = abs(metrics.get("left_nlf_change", 0))
        right_change = abs(metrics.get("right_nlf_change", 0))

        # 检查是否有运动
        if left_change < 2 and right_change < 2:
            return 1, "无法启动运动 (NLF变化过小)"

        # 计算对称性
        max_change = max(left_change, right_change)
        min_change = min(left_change, right_change)

        if max_change < 1e-9:
            symmetry_ratio = 1.0
        else:
            symmetry_ratio = min_change / max_change

        if symmetry_ratio >= 0.85:
            return 5, "运动完整 (对称性>85%)"
        elif symmetry_ratio >= 0.70:
            return 4, "几乎完整 (对称性70-85%)"
        elif symmetry_ratio >= 0.50:
            return 3, "启动但不对称 (对称性50-70%)"
        elif symmetry_ratio >= 0.25:
            return 2, "轻微启动 (对称性25-50%)"
        else:
            return 1, "无法启动 (对称性<25%)"
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
    """检测鼓腮时的联动运动"""
    synkinesis = {
        "eye_synkinesis": 0,
        "brow_synkinesis": 0,
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

    return synkinesis


def visualize_blow_cheek(frame: np.ndarray, landmarks, w: int, h: int,
                         result: ActionResult,
                         metrics: Dict[str, Any]) -> np.ndarray:
    """可视化鼓腮指标"""
    img = frame.copy()

    # 绘制嘴部轮廓
    draw_polygon(img, landmarks, w, h, LM.OUTER_LIP, (0, 255, 0), 2)

    # 绘制唇密封距离测量线
    seal = metrics["lip_seal"]

    # 上唇线 (0-13): 绿色
    upper_outer = seal["upper_outer"]
    upper_inner = seal["upper_inner"]
    cv2.line(img, (int(upper_outer[0]), int(upper_outer[1])),
             (int(upper_inner[0]), int(upper_inner[1])), (0, 255, 0), 3)
    cv2.circle(img, (int(upper_outer[0]), int(upper_outer[1])), 5, (255, 255, 0), -1)
    cv2.circle(img, (int(upper_inner[0]), int(upper_inner[1])), 5, (255, 255, 0), -1)

    # 下唇线 (14-17): 黄色
    lower_inner = seal["lower_inner"]
    lower_outer = seal["lower_outer"]
    cv2.line(img, (int(lower_inner[0]), int(lower_inner[1])),
             (int(lower_outer[0]), int(lower_outer[1])), (0, 255, 255), 3)
    cv2.circle(img, (int(lower_inner[0]), int(lower_inner[1])), 5, (0, 255, 255), -1)
    cv2.circle(img, (int(lower_outer[0]), int(lower_outer[1])), 5, (0, 255, 255), -1)

    # 绘制鼻唇沟
    l_ala = pt2d(landmarks[LM.NOSE_ALA_L], w, h)
    r_ala = pt2d(landmarks[LM.NOSE_ALA_R], w, h)
    l_mouth = pt2d(landmarks[LM.MOUTH_L], w, h)
    r_mouth = pt2d(landmarks[LM.MOUTH_R], w, h)
    cv2.line(img, (int(l_ala[0]), int(l_ala[1])), (int(l_mouth[0]), int(l_mouth[1])), (255, 100, 100), 2)
    cv2.line(img, (int(r_ala[0]), int(r_ala[1])), (int(r_mouth[0]), int(r_mouth[1])), (100, 165, 255), 2)

    # 信息面板
    panel_h = 260
    cv2.rectangle(img, (5, 5), (380, panel_h), (0, 0, 0), -1)
    cv2.rectangle(img, (5, 5), (380, panel_h), (255, 255, 255), 1)

    y = 28
    cv2.putText(img, f"{ACTION_NAME} - {ACTION_NAME_CN}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y += 28

    cv2.putText(img, "=== Lip Seal Distance ===", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    y += 20

    cv2.putText(img, f"Upper (0-13): {seal['upper_distance']:.1f}px", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    y += 18

    cv2.putText(img, f"Lower (14-17): {seal['lower_distance']:.1f}px", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    y += 18

    cv2.putText(img, f"Total: {seal['total_distance']:.1f}px", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    y += 25

    cv2.putText(img, "=== Mouth & NLF ===", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    y += 20

    cv2.putText(img, f"Mouth W: {metrics['mouth_width']:.1f}px  H: {metrics['mouth_height']:.1f}px", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    y += 18

    cv2.putText(img, f"NLF L: {metrics['left_nlf']:.1f}px  R: {metrics['right_nlf']:.1f}px", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    y += 18

    ratio = metrics['nlf_ratio']
    ratio_color = (0, 255, 0) if 0.9 <= ratio <= 1.1 else (0, 0, 255)
    cv2.putText(img, f"NLF Ratio: {ratio:.3f}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, ratio_color, 1)
    y += 25

    # Voluntary Score
    cv2.putText(img, f"Voluntary Score: {result.voluntary_movement_score}/5", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # 图例
    legend_y = panel_h + 15
    cv2.line(img, (15, legend_y), (45, legend_y), (0, 255, 0), 3)
    cv2.putText(img, "Upper lip (0-13)", (50, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.line(img, (180, legend_y), (210, legend_y), (0, 255, 255), 3)
    cv2.putText(img, "Lower lip (14-17)", (215, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    return img


def process(landmarks_seq: List, frames_seq: List, w: int, h: int,
            video_info: Dict[str, Any], output_dir: Path,
            baseline_result: Optional[ActionResult] = None,
            baseline_landmarks=None) -> Optional[ActionResult]:
    """处理BlowCheek动作"""
    if not landmarks_seq or not frames_seq:
        return None

    # 找峰值帧 (唇密封距离最小)
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

    # 计算鼓腮特有指标
    metrics = compute_blow_cheek_metrics(peak_landmarks, w, h, baseline_landmarks)

    # 计算Voluntary Movement评分
    score, interpretation = compute_voluntary_score(metrics, baseline_landmarks)
    result.voluntary_movement_score = score

    # 检测联动
    synkinesis = detect_synkinesis(baseline_result, peak_landmarks, w, h)
    result.synkinesis_scores = synkinesis

    # 存储动作特有指标
    result.action_specific = {
        "lip_seal": metrics["lip_seal"],
        "mouth_width": metrics["mouth_width"],
        "mouth_height": metrics["mouth_height"],
        "left_nlf": metrics["left_nlf"],
        "right_nlf": metrics["right_nlf"],
        "nlf_ratio": metrics["nlf_ratio"],
        "voluntary_interpretation": interpretation,
        "synkinesis": synkinesis,
    }

    if "baseline" in metrics:
        result.action_specific["baseline"] = metrics["baseline"]
        result.action_specific["changes"] = {
            "seal_change": metrics.get("seal_change", 0),
            "mouth_width_change": metrics.get("mouth_width_change", 0),
            "left_nlf_change": metrics.get("left_nlf_change", 0),
            "right_nlf_change": metrics.get("right_nlf_change", 0),
        }

    # 创建输出目录
    action_dir = output_dir / ACTION_NAME
    action_dir.mkdir(parents=True, exist_ok=True)

    # 保存原始帧
    cv2.imwrite(str(action_dir / "peak_raw.jpg"), peak_frame)

    # 保存可视化
    vis = visualize_blow_cheek(peak_frame, peak_landmarks, w, h, result, metrics)
    cv2.imwrite(str(action_dir / "peak_indicators.jpg"), vis)

    # 保存JSON
    with open(action_dir / "indicators.json", 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"    [OK] {ACTION_NAME}: Lip Seal={metrics['lip_seal']['total_distance']:.1f}px")
    print(f"         NLF L={metrics['left_nlf']:.1f} R={metrics['right_nlf']:.1f} Ratio={metrics['nlf_ratio']:.3f}")
    print(f"         Voluntary Score: {result.voluntary_movement_score}/5 ({interpretation})")

    return result