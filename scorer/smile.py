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
    extract_common_indicators, compute_scale_to_baseline,
    ActionResult, OralAngleMeasure, draw_polygon,
)

from sunnybrook_scorer import (
    VoluntaryMovementItem, compute_voluntary_score_from_ratio
)


def find_peak_frame(landmarks_seq: List, frames_seq: List, w: int, h: int) -> int:
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


def extract_smile_sequences(landmarks_seq: List, w: int, h: int) -> Dict[str, List[float]]:
    """
    提取微笑关键指标的时序序列

    Returns:
        包含嘴宽和口角角度的时序数据
    """
    mouth_width_seq = []
    aoe_seq = []
    bof_seq = []

    for lm in landmarks_seq:
        if lm is None:
            mouth_width_seq.append(np.nan)
            aoe_seq.append(np.nan)
            bof_seq.append(np.nan)
        else:
            l_corner = pt2d(lm[LM.MOUTH_L], w, h)
            r_corner = pt2d(lm[LM.MOUTH_R], w, h)
            width = dist(l_corner, r_corner)
            mouth_width_seq.append(width)

            oral = compute_oral_angle(lm, w, h)
            aoe_seq.append(oral.AOE_angle if oral else np.nan)
            bof_seq.append(oral.BOF_angle if oral else np.nan)

    return {
        "Mouth Width": mouth_width_seq,
        "AOE (Right)": aoe_seq,
        "BOF (Left)": bof_seq,
    }


def plot_smile_peak_selection(
        sequences: Dict[str, List[float]],
        fps: float,
        peak_idx: int,
        output_path: Path,
        action_name: str = "Smile"
) -> None:
    """
    绘制微笑关键帧选择的可解释性曲线

    微笑选择标准: 嘴宽最大的帧
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    n_frames = len(sequences["Mouth Width"])
    frames = np.arange(n_frames)
    time_sec = frames / fps if fps > 0 else frames
    x_label = 'Time (seconds)' if fps > 0 else 'Frame'
    peak_time = peak_idx / fps if fps > 0 else peak_idx

    # 上图: 嘴宽曲线 (关键帧选择依据)
    ax1 = axes[0]
    mouth_width = sequences["Mouth Width"]
    ax1.plot(time_sec, mouth_width, 'g-', label='Mouth Width', linewidth=2)

    # 标注峰值帧
    ax1.axvline(x=peak_time, color='black', linestyle='--', linewidth=2, alpha=0.7)
    width_at_peak = mouth_width[peak_idx] if peak_idx < len(mouth_width) else 0
    ax1.scatter([peak_time], [width_at_peak], color='red', s=150, zorder=5,
                edgecolors='black', linewidths=2, marker='*', label=f'Peak Frame {peak_idx}')
    ax1.annotate(f'Max: {width_at_peak:.1f}px', xy=(peak_time, width_at_peak),
                 xytext=(10, -20), textcoords='offset points', fontsize=10, fontweight='bold')

    ax1.set_xlabel(x_label, fontsize=11)
    ax1.set_ylabel('Mouth Width (pixels)', fontsize=11)
    ax1.set_title(f'{action_name} Peak Selection: Maximum Mouth Width', fontsize=13, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # 下图: 口角角度 (用于患侧判断)
    ax2 = axes[1]
    ax2.plot(time_sec, sequences["AOE (Right)"], 'r-', label='AOE (Right)', linewidth=2)
    ax2.plot(time_sec, sequences["BOF (Left)"], 'b-', label='BOF (Left)', linewidth=2)
    ax2.axvline(x=peak_time, color='black', linestyle='--', linewidth=2, alpha=0.7, label=f'Peak Frame {peak_idx}')
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1, alpha=0.5)

    # 在峰值帧标注角度值
    aoe_at_peak = sequences["AOE (Right)"][peak_idx] if peak_idx < len(sequences["AOE (Right)"]) else 0
    bof_at_peak = sequences["BOF (Left)"][peak_idx] if peak_idx < len(sequences["BOF (Left)"]) else 0
    if not np.isnan(aoe_at_peak):
        ax2.scatter([peak_time], [aoe_at_peak], color='red', s=80, zorder=5, edgecolors='white', linewidths=1.5)
        ax2.annotate(f'{aoe_at_peak:+.1f}°', xy=(peak_time, aoe_at_peak),
                     xytext=(5, 5), textcoords='offset points', fontsize=9, color='red')
    if not np.isnan(bof_at_peak):
        ax2.scatter([peak_time], [bof_at_peak], color='blue', s=80, zorder=5, edgecolors='white', linewidths=1.5)
        ax2.annotate(f'{bof_at_peak:+.1f}°', xy=(peak_time, bof_at_peak),
                     xytext=(5, -15), textcoords='offset points', fontsize=9, color='blue')

    ax2.set_xlabel(x_label, fontsize=11)
    ax2.set_ylabel('Oral Angle (degrees)', fontsize=11)
    ax2.set_title('Oral Commissure Angles (for Palsy Detection)', fontsize=12)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()


def compute_smile_metrics(landmarks, w: int, h: int,
                          baseline_landmarks=None) -> Dict[str, Any]:
    """计算微笑特有指标 - 使用统一 scale"""
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

    # 如果有基线，计算运动幅度（统一尺度）
    if baseline_landmarks is not None:
        # ========== 计算统一 scale（优化：传入预计算的 ICD）==========
        icd_current = compute_icd(landmarks, w, h)
        # 假设 baseline 的 icd 已在 NeutralFace 时计算并存储
        # 如果有传入 icd_base 参数，使用它；否则重新计算
        scale = compute_scale_to_baseline(landmarks, baseline_landmarks, w, h,
                                          icd_current=icd_current)
        metrics["scale"] = scale
        metrics["icd_current"] = icd_current

        baseline_mouth = compute_mouth_metrics(baseline_landmarks, w, h)
        baseline_oral = compute_oral_angle(baseline_landmarks, w, h)
        baseline_left = baseline_mouth["left_corner"]
        baseline_right = baseline_mouth["right_corner"]

        # ========== 缩放当前帧距离到 baseline 尺度 ==========
        # 嘴角位移（先计算原始位移，再缩放）
        left_excursion_raw = dist(left_corner, baseline_left)
        right_excursion_raw = dist(right_corner, baseline_right)

        # 缩放到 baseline 尺度
        left_excursion = left_excursion_raw * scale
        right_excursion = right_excursion_raw * scale

        # 嘴宽变化
        width_change = mouth["width"] * scale - baseline_mouth["width"]

        metrics["excursion"] = {
            "left_total": left_excursion,
            "right_total": right_excursion,
            "excursion_ratio": left_excursion / right_excursion if right_excursion > 1e-9 else 1.0,
            "baseline_width": baseline_mouth["width"],
            "width_change": width_change,
            # 保留原始值供调试
            "left_raw": left_excursion_raw,
            "right_raw": right_excursion_raw,
        }

        # 口角角度变化（角度不需要缩放）
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
    return process(
        landmarks_seq, frames_seq, w, h, video_info, output_dir,
        action_name="Smile",
        action_name_cn="微笑",
        baseline_result=baseline_result,
        baseline_landmarks=baseline_landmarks
    )

def process(landmarks_seq: List, frames_seq: List, w: int, h: int,
                          video_info: Dict[str, Any], output_dir: Path,
                          action_name: str, action_name_cn: str,
                          baseline_result: Optional[ActionResult] = None,
                          baseline_landmarks=None) -> Optional[ActionResult]:
    """处理微笑类动作的通用函数"""
    if not landmarks_seq or not frames_seq:
        return None

    # 找峰值帧
    peak_idx = find_peak_frame(landmarks_seq, frames_seq, w, h)
    peak_landmarks = landmarks_seq[peak_idx]
    peak_frame = frames_seq[peak_idx]

    # 提取时序序列用于可视化
    sequences = extract_smile_sequences(landmarks_seq, w, h)

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

    # 绘制关键帧选择曲线
    plot_smile_peak_selection(
        sequences,
        video_info.get("fps", 30.0),
        peak_idx,
        action_dir / "peak_selection_curve.png",
        action_name
    )

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