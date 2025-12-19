#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShrugNose 动作处理模块
================================

分析皱鼻动作:
1. 鼻翼点到同侧眼角内眦点的距离 (关键帧检测 + 面瘫侧别判断)
2. 鼻翼位置变化
3. 联动运动检测

修复内容:
- 使用鼻翼到内眦距离替代错误的NLF计算
- 关键帧检测: 鼻翼-内眦距离最小的帧
- 面瘫侧别检测: 距离变化小的一侧为患侧

对应Sunnybrook: Snarl (LLA/LLS)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json

from clinical_base import (
    LM, pt2d, pts2d, dist, compute_ear, compute_eye_area,
    compute_mouth_metrics, compute_icd, extract_common_indicators,
    ActionResult, draw_polygon
)

ACTION_NAME = "ShrugNose"
ACTION_NAME_CN = "皱鼻"


def compute_ala_to_canthus_distance(landmarks, w: int, h: int, left: bool = True) -> float:
    """
    计算鼻翼点到同侧眼角内眦点的距离

    皱鼻动作时，鼻翼上提，此距离变小
    """
    if left:
        ala = pt2d(landmarks[LM.NOSE_ALA_L], w, h)
        canthus = pt2d(landmarks[LM.EYE_INNER_L], w, h)
    else:
        ala = pt2d(landmarks[LM.NOSE_ALA_R], w, h)
        canthus = pt2d(landmarks[LM.EYE_INNER_R], w, h)

    return dist(ala, canthus)


def find_peak_frame(landmarks_seq: List, frames_seq: List, w: int, h: int,
                    baseline_landmarks=None) -> int:
    """
    找皱鼻峰值帧

    使用双侧鼻翼-内眦距离之和最小的帧作为峰值帧
    皱鼻时鼻翼上提，距离变小
    """
    min_total_dist = float('inf')
    min_idx = 0

    for i, lm in enumerate(landmarks_seq):
        if lm is None:
            continue

        # 计算双侧鼻翼-内眦距离
        left_dist = compute_ala_to_canthus_distance(lm, w, h, left=True)
        right_dist = compute_ala_to_canthus_distance(lm, w, h, left=False)
        total_dist = left_dist + right_dist

        if total_dist < min_total_dist:
            min_total_dist = total_dist
            min_idx = i

    return min_idx


def compute_shrug_nose_metrics(landmarks, w: int, h: int,
                               baseline_landmarks=None) -> Dict[str, Any]:
    """计算皱鼻特有指标"""
    # 鼻翼到内眦距离
    left_ac_dist = compute_ala_to_canthus_distance(landmarks, w, h, left=True)
    right_ac_dist = compute_ala_to_canthus_distance(landmarks, w, h, left=False)

    # 鼻翼点位置
    left_ala = pt2d(landmarks[LM.NOSE_ALA_L], w, h)
    right_ala = pt2d(landmarks[LM.NOSE_ALA_R], w, h)

    # 内眦点位置
    left_canthus = pt2d(landmarks[LM.EYE_INNER_L], w, h)
    right_canthus = pt2d(landmarks[LM.EYE_INNER_R], w, h)

    # 鼻翼间距
    ala_width = dist(left_ala, right_ala)

    metrics = {
        "left_ala_canthus_dist": left_ac_dist,
        "right_ala_canthus_dist": right_ac_dist,
        "ala_canthus_ratio": left_ac_dist / right_ac_dist if right_ac_dist > 1e-9 else 1.0,
        "ala_width": ala_width,
        "points": {
            "left_ala": left_ala,
            "right_ala": right_ala,
            "left_canthus": left_canthus,
            "right_canthus": right_canthus,
        }
    }

    # 如果有基线，计算变化
    if baseline_landmarks is not None:
        baseline_left = compute_ala_to_canthus_distance(baseline_landmarks, w, h, left=True)
        baseline_right = compute_ala_to_canthus_distance(baseline_landmarks, w, h, left=False)
        baseline_left_ala = pt2d(baseline_landmarks[LM.NOSE_ALA_L], w, h)
        baseline_right_ala = pt2d(baseline_landmarks[LM.NOSE_ALA_R], w, h)

        metrics["baseline"] = {
            "left_ala_canthus_dist": baseline_left,
            "right_ala_canthus_dist": baseline_right,
            "ala_width": dist(baseline_left_ala, baseline_right_ala),
        }

        # 距离变化 (皱鼻时应为负值，表示距离变小)
        metrics["left_change"] = left_ac_dist - baseline_left
        metrics["right_change"] = right_ac_dist - baseline_right

        # 变化百分比
        if baseline_left > 1e-9:
            metrics["left_change_percent"] = (left_ac_dist - baseline_left) / baseline_left * 100
        else:
            metrics["left_change_percent"] = 0

        if baseline_right > 1e-9:
            metrics["right_change_percent"] = (right_ac_dist - baseline_right) / baseline_right * 100
        else:
            metrics["right_change_percent"] = 0

        # 鼻翼垂直移动量 (Y轴，向上为负)
        metrics["left_ala_vertical_move"] = left_ala[1] - baseline_left_ala[1]
        metrics["right_ala_vertical_move"] = right_ala[1] - baseline_right_ala[1]

    return metrics


def detect_palsy_side(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    从皱鼻动作检测面瘫侧别

    原理: 面瘫侧的鼻翼无法充分上提，鼻翼-内眦距离变化小

    Returns:
        Dict包含:
        - palsy_side: 0=无/对称, 1=左, 2=右
        - confidence: 置信度
        - interpretation: 解释
    """
    result = {"palsy_side": 0, "confidence": 0.0, "interpretation": ""}

    if "left_change" not in metrics or "right_change" not in metrics:
        result["interpretation"] = "无基线对比数据"
        return result

    left_change = metrics["left_change"]
    right_change = metrics["right_change"]

    # 皱鼻时距离应该变小，所以change应该是负值
    # 取绝对值表示收缩幅度
    left_contraction = -left_change  # 正值表示收缩
    right_contraction = -right_change

    max_contraction = max(left_contraction, right_contraction)
    min_contraction = min(left_contraction, right_contraction)

    # 判断是否有明显运动
    if max_contraction < 3:  # 像素阈值
        result["interpretation"] = "皱鼻运动幅度过小，无法判断"
        return result

    # 计算不对称性
    if max_contraction > 1e-9:
        asymmetry_ratio = (max_contraction - min_contraction) / max_contraction
    else:
        asymmetry_ratio = 0

    if asymmetry_ratio < 0.15:
        result["palsy_side"] = 0
        result["confidence"] = 1.0 - asymmetry_ratio
        result["interpretation"] = f"双侧皱鼻对称 (不对称比={asymmetry_ratio:.1%})"
    elif left_contraction < right_contraction:
        result["palsy_side"] = 1  # 左侧收缩弱 -> 左侧面瘫
        result["confidence"] = min(1.0, asymmetry_ratio)
        result[
            "interpretation"] = f"左侧鼻翼收缩较弱 (L收缩={left_contraction:.1f}px < R收缩={right_contraction:.1f}px)"
    else:
        result["palsy_side"] = 2  # 右侧收缩弱 -> 右侧面瘫
        result["confidence"] = min(1.0, asymmetry_ratio)
        result[
            "interpretation"] = f"右侧鼻翼收缩较弱 (R收缩={right_contraction:.1f}px < L收缩={left_contraction:.1f}px)"

    return result


def compute_voluntary_score(metrics: Dict[str, Any], baseline_landmarks=None) -> Tuple[int, str]:
    """
    计算Voluntary Movement评分

    基于鼻翼-内眦距离变化的程度和对称性

    评分标准:
    - 5=完整: 双侧对称且运动充分
    - 4=几乎完整: 轻度不对称或运动略有不足
    - 3=启动但不对称: 明显不对称但有运动
    - 2=轻微启动: 运动幅度很小
    - 1=无法启动: 几乎没有运动
    """
    if baseline_landmarks is not None and "left_change" in metrics:
        left_change = metrics["left_change"]
        right_change = metrics["right_change"]

        # 皱鼻时距离应该变小，所以收缩 = -change
        left_contraction = -left_change
        right_contraction = -right_change

        # 检查是否有明显运动
        max_contraction = max(left_contraction, right_contraction)
        min_contraction = min(left_contraction, right_contraction)

        if max_contraction < 2:  # 几乎没有收缩
            return 1, "无法启动运动 (鼻翼-内眦距离变化过小)"

        # 检查是否一侧反向运动
        if min_contraction < -1:  # 一侧反而变远
            return 2, "轻微启动 (单侧异常运动)"

        # 计算对称性
        if max_contraction > 1e-9:
            symmetry_ratio = min_contraction / max_contraction
        else:
            symmetry_ratio = 1.0

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
        ratio = metrics.get("ala_canthus_ratio", 1.0)
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
                         metrics: Dict[str, Any],
                         palsy_detection: Dict[str, Any]) -> np.ndarray:
    """可视化皱鼻指标"""
    img = frame.copy()

    # 获取点坐标
    points = metrics.get("points", {})
    left_ala = points.get("left_ala", pt2d(landmarks[LM.NOSE_ALA_L], w, h))
    right_ala = points.get("right_ala", pt2d(landmarks[LM.NOSE_ALA_R], w, h))
    left_canthus = points.get("left_canthus", pt2d(landmarks[LM.EYE_INNER_L], w, h))
    right_canthus = points.get("right_canthus", pt2d(landmarks[LM.EYE_INNER_R], w, h))

    # 绘制鼻翼-内眦连线 (关键测量线)
    cv2.line(img, (int(left_ala[0]), int(left_ala[1])),
             (int(left_canthus[0]), int(left_canthus[1])), (255, 0, 0), 3)
    cv2.line(img, (int(right_ala[0]), int(right_ala[1])),
             (int(right_canthus[0]), int(right_canthus[1])), (0, 165, 255), 3)

    # 绘制鼻翼点
    cv2.circle(img, (int(left_ala[0]), int(left_ala[1])), 6, (255, 0, 0), -1)
    cv2.circle(img, (int(right_ala[0]), int(right_ala[1])), 6, (0, 165, 255), -1)

    # 绘制内眦点
    cv2.circle(img, (int(left_canthus[0]), int(left_canthus[1])), 5, (255, 0, 0), 2)
    cv2.circle(img, (int(right_canthus[0]), int(right_canthus[1])), 5, (0, 165, 255), 2)

    # 绘制鼻翼连线
    cv2.line(img, (int(left_ala[0]), int(left_ala[1])),
             (int(right_ala[0]), int(right_ala[1])), (0, 255, 255), 2)

    # 信息面板
    panel_h = 320
    cv2.rectangle(img, (5, 5), (420, panel_h), (0, 0, 0), -1)
    cv2.rectangle(img, (5, 5), (420, panel_h), (255, 255, 255), 1)

    y = 28
    cv2.putText(img, f"{ACTION_NAME} (ShrugNose)", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y += 28

    cv2.putText(img, "=== Ala-Canthus Distance ===", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    y += 20

    cv2.putText(img, f"Left: {metrics['left_ala_canthus_dist']:.1f}px", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1)
    y += 18

    cv2.putText(img, f"Right: {metrics['right_ala_canthus_dist']:.1f}px", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
    y += 18

    ratio = metrics['ala_canthus_ratio']
    ratio_color = (0, 255, 0) if 0.9 <= ratio <= 1.1 else (0, 0, 255)
    cv2.putText(img, f"Ratio (L/R): {ratio:.3f}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, ratio_color, 1)
    y += 22

    if "left_change" in metrics:
        cv2.putText(img, "=== Changes from Baseline ===", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        y += 20

        cv2.putText(img,
                    f"Left: {metrics['left_change']:+.1f}px ({metrics.get('left_change_percent', 0):+.1f}%)",
                    (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 18

        cv2.putText(img,
                    f"Right: {metrics['right_change']:+.1f}px ({metrics.get('right_change_percent', 0):+.1f}%)",
                    (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 22

    cv2.putText(img, f"Ala Width: {metrics['ala_width']:.1f}px", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    y += 25

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
    cv2.line(img, (15, legend_y), (45, legend_y), (255, 0, 0), 3)
    cv2.putText(img, "Left Ala-Canthus", (50, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    cv2.line(img, (200, legend_y), (230, legend_y), (0, 165, 255), 3)
    cv2.putText(img, "Right Ala-Canthus", (235, legend_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    return img


def process(landmarks_seq: List, frames_seq: List, w: int, h: int,
            video_info: Dict[str, Any], output_dir: Path,
            baseline_result: Optional[ActionResult] = None,
            baseline_landmarks=None) -> Optional[ActionResult]:
    """处理ShrugNose动作"""
    if not landmarks_seq or not frames_seq:
        return None

    # 找峰值帧 (鼻翼-内眦距离最小)
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
        "ala_canthus_metrics": {
            "left_distance": metrics["left_ala_canthus_dist"],
            "right_distance": metrics["right_ala_canthus_dist"],
            "ratio": metrics["ala_canthus_ratio"],
            "ala_width": metrics["ala_width"],
        },
        "palsy_detection": palsy_detection,
        "voluntary_interpretation": interpretation,
        "synkinesis": synkinesis,
    }

    if "baseline" in metrics:
        result.action_specific["baseline"] = metrics["baseline"]
        result.action_specific["changes"] = {
            "left_change": metrics.get("left_change", 0),
            "right_change": metrics.get("right_change", 0),
            "left_change_percent": metrics.get("left_change_percent", 0),
            "right_change_percent": metrics.get("right_change_percent", 0),
        }

    # 创建输出目录
    action_dir = output_dir / ACTION_NAME
    action_dir.mkdir(parents=True, exist_ok=True)

    # 保存原始帧
    cv2.imwrite(str(action_dir / "peak_raw.jpg"), peak_frame)

    # 保存可视化
    vis = visualize_shrug_nose(peak_frame, peak_landmarks, w, h, result, metrics, palsy_detection)
    cv2.imwrite(str(action_dir / "peak_indicators.jpg"), vis)

    # 保存JSON
    with open(action_dir / "indicators.json", 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    print(
        f"    [OK] {ACTION_NAME}: Ala-Canthus L={metrics['left_ala_canthus_dist']:.1f} R={metrics['right_ala_canthus_dist']:.1f}")
    if "left_change" in metrics:
        print(f"         Change L={metrics['left_change']:+.1f}px R={metrics['right_change']:+.1f}px")
    print(f"         Palsy Side: {palsy_detection.get('interpretation', 'N/A')}")
    print(f"         Voluntary Score: {result.voluntary_movement_score}/5 ({interpretation})")

    return result