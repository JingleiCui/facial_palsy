#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BlowCheek 动作处理模块
================================

分析鼓腮动作:
1. 唇密封距离 (关键帧检测)
2. 嘴部闭合程度
3. 口角对称性
4. 面瘫侧别检测
5. 联动运动检测

修复内容:
- 移除错误的NLF分析
- 使用唇密封距离和口角对称性作为主要指标

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
    compute_mouth_metrics, compute_oral_angle,
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

    # 口角角度
    oral = compute_oral_angle(landmarks, w, h)

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
        "left_corner": mouth["left_corner"],
        "right_corner": mouth["right_corner"],
        "oral_angle": {
            "AOE": oral.AOE_angle,
            "BOF": oral.BOF_angle,
            "asymmetry": oral.angle_asymmetry,
        }
    }

    # 如果有基线，计算变化
    if baseline_landmarks is not None:
        baseline_seal = compute_lip_seal_distance(baseline_landmarks, w, h)
        baseline_mouth = compute_mouth_metrics(baseline_landmarks, w, h)
        baseline_oral = compute_oral_angle(baseline_landmarks, w, h)

        metrics["baseline"] = {
            "lip_seal_total": baseline_seal["total_distance"],
            "mouth_width": baseline_mouth["width"],
        }

        # 变化量
        metrics["seal_change"] = seal_result["total_distance"] - baseline_seal["total_distance"]
        metrics["mouth_width_change"] = mouth["width"] - baseline_mouth["width"]

        # 嘴角位移
        left_corner = mouth["left_corner"]
        right_corner = mouth["right_corner"]
        baseline_left = baseline_mouth["left_corner"]
        baseline_right = baseline_mouth["right_corner"]

        left_excursion = dist(left_corner, baseline_left)
        right_excursion = dist(right_corner, baseline_right)
        metrics["left_excursion"] = left_excursion
        metrics["right_excursion"] = right_excursion
        metrics["excursion_ratio"] = left_excursion / right_excursion if right_excursion > 1e-9 else 1.0

    return metrics


def detect_palsy_side(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    从鼓腮动作检测面瘫侧别

    原理:
    1. 面瘫侧嘴角运动幅度小
    2. 口角角度不对称

    Returns:
        Dict包含:
        - palsy_side: 0=无/对称, 1=左, 2=右
        - confidence: 置信度
        - interpretation: 解释
    """
    result = {"palsy_side": 0, "confidence": 0.0, "interpretation": ""}

    # 使用运动幅度作为主要指标
    if "left_excursion" in metrics and "right_excursion" in metrics:
        left_exc = metrics["left_excursion"]
        right_exc = metrics["right_excursion"]

        if max(left_exc, right_exc) < 2:  # 运动幅度太小
            # 检查口角角度
            oral = metrics.get("oral_angle", {})
            asym = oral.get("asymmetry", 0)
            if asym < 3:
                result["interpretation"] = "双侧对称"
            else:
                result["interpretation"] = "运动幅度过小，难以判断"
            return result

        exc_ratio = metrics["excursion_ratio"]

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
        # 没有基线，使用口角对称性
        oral = metrics.get("oral_angle", {})
        asym = oral.get("asymmetry", 0)
        aoe = oral.get("AOE", 0)
        bof = oral.get("BOF", 0)

        if asym < 3:
            result["palsy_side"] = 0
            result["confidence"] = 1.0 - asym / 10
            result["interpretation"] = f"口角对称 (不对称度={asym:.1f}°)"
        elif aoe < bof:
            result["palsy_side"] = 2
            result["confidence"] = min(1.0, asym / 15)
            result["interpretation"] = f"右口角位置较低 (AOE={aoe:+.1f}°)"
        else:
            result["palsy_side"] = 1
            result["confidence"] = min(1.0, asym / 15)
            result["interpretation"] = f"左口角位置较低 (BOF={bof:+.1f}°)"

    return result


def compute_voluntary_score(metrics: Dict[str, Any], baseline_landmarks=None) -> Tuple[int, str]:
    """
    计算Voluntary Movement评分

    基于嘴部闭合程度和对称性

    评分标准:
    - 5=完整: 双侧对称且运动充分
    - 4=几乎完整: 轻度不对称或运动略有不足
    - 3=启动但不对称: 明显不对称但有运动
    - 2=轻微启动: 运动幅度很小
    - 1=无法启动: 几乎没有运动
    """
    oral = metrics.get("oral_angle", {})
    oral_asym = oral.get("asymmetry", 0)

    if baseline_landmarks is not None and "excursion_ratio" in metrics:
        # 使用运动幅度评估
        exc_ratio = metrics["excursion_ratio"]
        left_exc = metrics.get("left_excursion", 0)
        right_exc = metrics.get("right_excursion", 0)

        # 检查是否有运动
        max_exc = max(left_exc, right_exc)
        if max_exc < 2:
            return 1, "无法启动运动 (运动幅度过小)"

        # 计算对称性
        asymmetry = abs(exc_ratio - 1.0)

        if asymmetry < 0.15:
            if max_exc > 8:
                return 5, "运动完整 (对称性>85%)"
            elif max_exc > 5:
                return 4, "几乎完整"
            else:
                return 3, "启动但幅度不足"
        elif asymmetry < 0.30:
            return 3, "启动但不对称"
        elif asymmetry < 0.50:
            return 2, "轻微启动"
        else:
            return 1, "无法启动"
    else:
        # 没有基线，使用静态口角对称性
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
                         metrics: Dict[str, Any],
                         palsy_detection: Dict[str, Any]) -> np.ndarray:
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

    # 绘制嘴角点
    left_corner = metrics["left_corner"]
    right_corner = metrics["right_corner"]
    cv2.circle(img, (int(left_corner[0]), int(left_corner[1])), 5, (255, 0, 0), -1)
    cv2.circle(img, (int(right_corner[0]), int(right_corner[1])), 5, (0, 0, 255), -1)

    # 信息面板
    panel_h = 300
    cv2.rectangle(img, (5, 5), (400, panel_h), (0, 0, 0), -1)
    cv2.rectangle(img, (5, 5), (400, panel_h), (255, 255, 255), 1)

    y = 28
    cv2.putText(img, f"{ACTION_NAME}", (15, y),
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

    cv2.putText(img, "=== Mouth Metrics ===", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    y += 20

    cv2.putText(img, f"Mouth W: {metrics['mouth_width']:.1f}px  H: {metrics['mouth_height']:.1f}px", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    y += 22

    # 口角对称性
    oral = metrics.get("oral_angle", {})
    oral_asym = oral.get("asymmetry", 0)
    asym_color = (0, 255, 0) if oral_asym < 5 else (0, 165, 255) if oral_asym < 10 else (0, 0, 255)
    cv2.putText(img, f"Oral Asymmetry: {oral_asym:.1f} deg", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, asym_color, 1)
    y += 22

    # 面瘫侧别检测结果
    palsy_side = palsy_detection.get("palsy_side", 0)
    palsy_text = {0: "无/对称", 1: "左侧", 2: "右侧"}.get(palsy_side, "未知")
    palsy_color = (0, 255, 0) if palsy_side == 0 else (0, 0, 255)
    cv2.putText(img, f"Palsy Side: {palsy_text}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, palsy_color, 1)
    y += 25

    # Voluntary Score
    cv2.putText(img, f"Voluntary Score: {result.voluntary_movement_score}/5", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # 图例
    legend_y = panel_h + 15
    cv2.line(img, (15, legend_y), (45, legend_y), (0, 255, 0), 3)
    cv2.putText(img, "Upper lip (0-13)", (50, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.line(img, (200, legend_y), (230, legend_y), (0, 255, 255), 3)
    cv2.putText(img, "Lower lip (14-17)", (235, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

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
        "lip_seal": metrics["lip_seal"],
        "mouth_width": metrics["mouth_width"],
        "mouth_height": metrics["mouth_height"],
        "oral_angle": metrics["oral_angle"],
        "palsy_detection": palsy_detection,
        "voluntary_interpretation": interpretation,
        "synkinesis": synkinesis,
    }

    if "baseline" in metrics:
        result.action_specific["baseline"] = metrics["baseline"]
        result.action_specific["changes"] = {
            "seal_change": metrics.get("seal_change", 0),
            "mouth_width_change": metrics.get("mouth_width_change", 0),
        }

    # 创建输出目录
    action_dir = output_dir / ACTION_NAME
    action_dir.mkdir(parents=True, exist_ok=True)

    # 保存原始帧
    cv2.imwrite(str(action_dir / "peak_raw.jpg"), peak_frame)

    # 保存可视化
    vis = visualize_blow_cheek(peak_frame, peak_landmarks, w, h, result, metrics, palsy_detection)
    cv2.imwrite(str(action_dir / "peak_indicators.jpg"), vis)

    # 保存JSON
    with open(action_dir / "indicators.json", 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"    [OK] {ACTION_NAME}: Lip Seal={metrics['lip_seal']['total_distance']:.1f}px")
    print(f"         Palsy: {palsy_detection.get('interpretation', 'N/A')}")
    print(f"         Voluntary Score: {result.voluntary_movement_score}/5 ({interpretation})")

    return result