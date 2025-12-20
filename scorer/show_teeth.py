#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ShowTeeth 动作处理模块
======================

分析露齿动作:
1. 嘴角位移和对称性
2. 口角角度变化
3. 上唇提升程度
4. 嘴部开口/露齿幅度 (以内唇圈面积近似)
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
    LM, pt2d, pts2d, dist, polygon_area, compute_ear, compute_eye_area,
    compute_mouth_metrics, compute_oral_angle,
    compute_icd, extract_common_indicators,
    ActionResult, OralAngleMeasure, draw_polygon
)

from sunnybrook_scorer import (
    VoluntaryMovementItem, compute_voluntary_score_from_ratio
)

ACTION_NAME = "ShowTeeth"
ACTION_NAME_CN = "露齿"


def find_peak_frame(landmarks_seq: List, frames_seq: List, w: int, h: int) -> int:
    """找露齿峰值帧

    露齿的“峰值”不一定来自“上唇提升”，也可能来自“下颌下压/下唇下拉”。
    因此这里用 **内唇缘一圈(Inner Lip)围成的开口面积** 作为露齿强度的近似指标：
    - 开口面积越大，通常牙齿暴露越明显
    - 为减轻前后距离变化的影响，对面积做 ICD² 归一化

    注意：这仍是“牙齿暴露”的几何代理指标。若后续接入牙齿/口腔分割，可进一步更准。
    """
    best_score = -1.0
    best_idx = 0

    for i, lm in enumerate(landmarks_seq):
        if lm is None:
            continue

        # 1) 内唇缘面积
        inner_pts = np.array(pts2d(lm, LM.INNER_LIP, w, h), dtype=np.float32)
        area = float(abs(polygon_area(inner_pts))) if len(inner_pts) >= 3 else 0.0

        # 2) ICD²归一化，降低面部远近变化带来的面积缩放
        icd = float(compute_icd(lm, w, h))
        score = area / (icd * icd + 1e-9)

        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx

def compute_show_teeth_metrics(landmarks, w: int, h: int,
                               baseline_landmarks=None) -> Dict[str, Any]:
    """计算露齿(ShowTeeth)特有指标

    关键：用“内唇缘开口面积”作为露齿/张口幅度的几何代理指标。
    鼻唇沟(NLF)相关几何指标已禁用（当前实现为鼻翼-嘴角连线，不等价于真实鼻唇沟）。
    """
    mouth = compute_mouth_metrics(landmarks, w, h)
    oral = compute_oral_angle(landmarks, w, h)

    # 内唇缘开口面积 (牙齿暴露的几何代理)
    inner_pts = np.array(pts2d(landmarks, LM.INNER_LIP, w, h), dtype=np.float32)
    inner_mouth_area = float(abs(polygon_area(inner_pts))) if len(inner_pts) >= 3 else 0.0

    # 嘴角位置
    left_corner = mouth["left_corner"]
    right_corner = mouth["right_corner"]

    # 嘴角高度 (相对于嘴中心)
    mouth_center_y = (left_corner[1] + right_corner[1]) / 2
    left_height_from_center = mouth_center_y - left_corner[1]   # 正值: 左嘴角更高
    right_height_from_center = mouth_center_y - right_corner[1]

    # 上唇/下唇中心位置（用于描述上唇提升或下颌下压）
    lip_top = mouth["top_center"]
    lip_bottom = mouth["bottom_center"]

    metrics: Dict[str, Any] = {
        "mouth_width": float(mouth["width"]),
        "mouth_height": float(mouth["height"]),
        "mouth_opening_ratio": float(mouth["height"] / (mouth["width"] + 1e-9)),
        "inner_mouth_area": float(inner_mouth_area),
        "left_corner": left_corner,
        "right_corner": right_corner,
        "left_height_from_center": float(left_height_from_center),
        "right_height_from_center": float(right_height_from_center),
        "lip_top_y": float(lip_top[1]),
        "lip_bottom_y": float(lip_bottom[1]),
        "oral_angle": {
            "AOE": float(getattr(oral, "AOE_angle", 0.0) or 0.0),
            "BOF": float(getattr(oral, "BOF_angle", 0.0) or 0.0),
            "asymmetry": float(getattr(oral, "angle_asymmetry", 0.0) or 0.0),
        },
        "nlf_disabled": True,
    }

    # 基线参考 (NeutralFace)
    if baseline_landmarks is not None:
        baseline_mouth = compute_mouth_metrics(baseline_landmarks, w, h)

        baseline_inner_pts = np.array(pts2d(baseline_landmarks, LM.INNER_LIP, w, h), dtype=np.float32)
        baseline_inner_area = float(abs(polygon_area(baseline_inner_pts))) if len(baseline_inner_pts) >= 3 else 0.0

        metrics["baseline"] = {
            "mouth_width": float(baseline_mouth["width"]),
            "mouth_height": float(baseline_mouth["height"]),
            "mouth_opening_ratio": float(baseline_mouth["height"] / (baseline_mouth["width"] + 1e-9)),
            "inner_mouth_area": float(baseline_inner_area),
        }
        metrics["delta"] = {
            "mouth_width": float(mouth["width"] - baseline_mouth["width"]),
            "mouth_height": float(mouth["height"] - baseline_mouth["height"]),
            "mouth_opening_ratio": float(mouth["height"] / (mouth["width"] + 1e-9)),
            "inner_mouth_area": float(inner_mouth_area - baseline_inner_area),
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
    # 鼻唇沟（NLF）几何指标已禁用：目前实现为“鼻翼-嘴角连线”，不等价于真实鼻唇沟纹理/沟壑。

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
    # NLF（禁用）
    cv2.putText(img, "NLF: disabled", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    y += 18

    # 露齿/张口幅度（内唇缘开口面积）
    cv2.putText(img, f"Inner Mouth Area: {metrics.get('inner_mouth_area', 0.0):.1f}px^2", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
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