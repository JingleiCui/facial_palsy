#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LipPucker 动作处理模块
================================

分析撅嘴动作:
1. 嘴唇宽度变化
2. 嘴唇高度变化
3. 嘴角位置对称性
4. 口角角度变化
5. 面瘫侧别检测
6. 联动运动检测

修复内容:
- 移除错误的NLF分析
- 使用口角角度和嘴部收缩作为主要指标

对应Sunnybrook: Lip pucker (OOS/OOI)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from clinical_base import (
    LM, pt2d, pt3d, pts2d, dist, compute_ear, compute_eye_area,
    compute_mouth_metrics, compute_oral_angle, compute_lip_seal_distance,
    compute_icd, extract_common_indicators,
    ActionResult, draw_polygon
)

ACTION_NAME = "LipPucker"
ACTION_NAME_CN = "撅嘴"


def _kabsch_rigid(P: np.ndarray, Q: np.ndarray):
    """把当前帧稳定点 P 刚体对齐到基线稳定点 Q，返回 R,t"""
    if P is None or Q is None or P.shape[0] < 3 or Q.shape[0] < 3:
        return None, None
    Pc = P.mean(axis=0); Qc = Q.mean(axis=0)
    X = P - Pc; Y = Q - Qc
    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = Qc - (R @ Pc)
    return R, t


def _apply_rt(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (R @ points.T).T + t


def _mean_lip_z_aligned(landmarks, w: int, h: int, baseline_landmarks=None) -> float:
    """
    计算“唇部区域平均z”（更稳：先把当前帧刚体对齐到 baseline，再取 z）。
    z 越小（更负）表示越靠近镜头。
    """
    lip_indices = list(LM.OUTER_LIP) + list(LM.INNER_LIP)

    # 没 baseline：直接算当前帧平均z（不推荐但可跑）
    if baseline_landmarks is None:
        zs = []
        for idx in lip_indices:
            x, y, z = pt3d(landmarks[idx], w, h)
            zs.append(z)
        return float(np.mean(zs)) if zs else float("nan")

    stable_idx = [
        LM.EYE_INNER_L, LM.EYE_INNER_R,
        LM.EYE_OUTER_L, LM.EYE_OUTER_R,
        LM.NOSE_BRIDGE, LM.NOSE_TIP,
        LM.CHIN
    ]

    P, Q = [], []
    for i in stable_idx:
        px, py, pz = pt3d(landmarks[i], w, h)
        qx, qy, qz = pt3d(baseline_landmarks[i], w, h)
        P.append([px, py, pz])
        Q.append([qx, qy, qz])

    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    Rm, t = _kabsch_rigid(P, Q)
    if Rm is None:
        # 对齐失败就退化
        return _mean_lip_z_aligned(landmarks, w, h, baseline_landmarks=None)

    # 当前唇部点 -> 对齐到 baseline 坐标
    cur = []
    for idx in lip_indices:
        x, y, z = pt3d(landmarks[idx], w, h)
        cur.append([x, y, z])
    cur = np.asarray(cur, dtype=np.float64)
    cur_aligned = _apply_rt(cur, Rm, t)

    return float(np.mean(cur_aligned[:, 2]))


def _save_lip_z_curve_png(png_path, z_delta_norm, peak_idx: int):
    """画嘴唇区域 z 轴运动曲线（相对基线的 delta_z/ICD）"""
    if not z_delta_norm:
        return
    plt.figure()
    plt.plot(list(range(len(z_delta_norm))), z_delta_norm)
    plt.axvline(int(peak_idx), linestyle="--")
    plt.title("Lip region depth curve (baseline_z - current_z) / ICD")
    plt.xlabel("Frame")
    plt.ylabel("Delta depth (normalized)")
    plt.tight_layout()
    plt.savefig(str(png_path), dpi=150)
    plt.close()

def find_peak_frame(landmarks_seq: List, frames_seq: List, w: int, h: int,
                    baseline_landmarks=None) -> Tuple[int, Dict[str, Any]]:
    """
    撅嘴峰值帧：
    - 主：嘴宽相对 baseline 明显变小
    - 辅：唇部区域更靠近镜头（baseline_z - current_z 更大）
    - 门控：不张嘴 + 唇密封（避免误选说话/张口帧）
    """
    n = len(landmarks_seq)
    if n == 0:
        return 0, {}

    # baseline mouth width / ICD / baseline lip z
    if baseline_landmarks is not None:
        base_mouth = compute_mouth_metrics(baseline_landmarks, w, h)
        base_width = float(base_mouth["width"])
        base_icd = float(compute_icd(baseline_landmarks, w, h))
        base_z = _mean_lip_z_aligned(baseline_landmarks, w, h, baseline_landmarks=None)
    else:
        # 没 baseline 就用第一帧当 baseline（退化）
        first = next((lm for lm in landmarks_seq if lm is not None), None)
        base_mouth = compute_mouth_metrics(first, w, h) if first is not None else {"width": 1.0}
        base_width = float(base_mouth["width"])
        base_icd = float(compute_icd(first, w, h)) if first is not None else 1.0
        base_z = _mean_lip_z_aligned(first, w, h, baseline_landmarks=None) if first is not None else 0.0

    base_width = max(base_width, 1e-6)
    base_icd = max(base_icd, 1e-6)

    # 门控阈值（你可以按数据再调）
    mouth_h_thr = 0.32      # mouth_height / ICD 过大 => 张口
    seal_thr = 0.3         # lip_seal_total / ICD 过大 => 唇不闭合

    best_score = -1e18
    best_idx = 0

    width_ratio_list = []
    z_delta_norm_list = []
    score_list = []
    valid_list = []

    for i, lm in enumerate(landmarks_seq):
        if lm is None:
            width_ratio_list.append(float("nan"))
            z_delta_norm_list.append(float("nan"))
            score_list.append(float("nan"))
            valid_list.append(False)
            continue

        mouth = compute_mouth_metrics(lm, w, h)
        width = float(mouth["width"])
        height = float(mouth["height"])

        width_ratio = width / base_width
        mouth_h_ratio = height / base_icd

        # 唇密封（越小越闭合）
        seal = compute_lip_seal_distance(lm, w, h)
        seal_ratio = float(seal["total_distance"]) / base_icd

        # 唇部深度：对齐后取唇部平均z（越小越靠前）
        cur_z = _mean_lip_z_aligned(lm, w, h, baseline_landmarks=baseline_landmarks)
        # delta_z：baseline_z - current_z （越大表示越“靠前”）
        z_delta = float(base_z - cur_z)
        z_delta_norm = z_delta / base_icd

        # 门控：不张口 + 唇闭合（避免嘴张开误选）
        valid = (mouth_h_ratio <= mouth_h_thr) and (seal_ratio <= seal_thr)

        # 联合评分：宽度收缩为主，前突为辅
        # width_ratio 越小越好 => (1 - width_ratio) 越大越好
        score = (1.2 * (1.0 - width_ratio)) + (0.8 * z_delta_norm)

        # 如果没通过门控，强惩罚
        if not valid:
            score -= 10.0

        width_ratio_list.append(width_ratio)
        z_delta_norm_list.append(z_delta_norm)
        score_list.append(score)
        valid_list.append(valid)

        if score > best_score:
            best_score = score
            best_idx = i

    debug = {
        "width_ratio": width_ratio_list,
        "lip_z_delta_norm": z_delta_norm_list,
        "score": score_list,
        "valid": valid_list,
        "best_score": float(best_score)
    }
    return best_idx, debug


def compute_lip_pucker_metrics(landmarks, w: int, h: int,
                               baseline_landmarks=None) -> Dict[str, Any]:
    """计算撅嘴特有指标"""
    mouth = compute_mouth_metrics(landmarks, w, h)
    oral = compute_oral_angle(landmarks, w, h)

    # 嘴角位置
    left_corner = mouth["left_corner"]
    right_corner = mouth["right_corner"]

    # 嘴角水平对称性
    avg_y = (left_corner[1] + right_corner[1]) / 2
    left_diff = left_corner[1] - avg_y
    right_diff = right_corner[1] - avg_y

    metrics = {
        "mouth_width": mouth["width"],
        "mouth_height": mouth["height"],
        "width_height_ratio": mouth["width"] / mouth["height"] if mouth["height"] > 1e-9 else 0,
        "left_corner": left_corner,
        "right_corner": right_corner,
        "corner_height_diff": abs(left_corner[1] - right_corner[1]),
        "oral_angle": {
            "AOE": oral.AOE_angle,
            "BOF": oral.BOF_angle,
            "asymmetry": oral.angle_asymmetry,
        }
    }

    # 如果有基线，计算变化
    if baseline_landmarks is not None:
        baseline_mouth = compute_mouth_metrics(baseline_landmarks, w, h)
        baseline_oral = compute_oral_angle(baseline_landmarks, w, h)
        baseline_left = baseline_mouth["left_corner"]
        baseline_right = baseline_mouth["right_corner"]

        metrics["baseline"] = {
            "mouth_width": baseline_mouth["width"],
            "mouth_height": baseline_mouth["height"],
        }

        # 变化量
        metrics["width_change"] = mouth["width"] - baseline_mouth["width"]
        metrics["height_change"] = mouth["height"] - baseline_mouth["height"]

        # 变化百分比
        if baseline_mouth["width"] > 1e-9:
            metrics["width_change_percent"] = (mouth["width"] - baseline_mouth["width"]) / baseline_mouth["width"] * 100
        else:
            metrics["width_change_percent"] = 0

        # 收缩比例 (撅嘴时应该<1)
        metrics["width_ratio"] = mouth["width"] / baseline_mouth["width"] if baseline_mouth["width"] > 1e-9 else 1.0

        # 嘴角位移
        left_excursion = dist(left_corner, baseline_left)
        right_excursion = dist(right_corner, baseline_right)
        metrics["left_excursion"] = left_excursion
        metrics["right_excursion"] = right_excursion
        metrics["excursion_ratio"] = left_excursion / right_excursion if right_excursion > 1e-9 else 1.0

    return metrics


def detect_palsy_side(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    从撅嘴动作检测面瘫侧别

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

    基于嘴唇收缩程度和对称性

    评分标准:
    - 5=完整: 双侧对称且收缩明显
    - 4=几乎完整: 轻度不对称或收缩略有不足
    - 3=启动但不对称: 明显不对称但有运动
    - 2=轻微启动: 运动幅度很小
    - 1=无法启动: 几乎没有运动
    """
    # 检查口角对称性
    corner_diff = metrics.get("corner_height_diff", 0)
    oral_asym = metrics.get("oral_angle", {}).get("asymmetry", 0)

    if baseline_landmarks is not None and "width_ratio" in metrics:
        width_ratio = metrics["width_ratio"]

        # 撅嘴时宽度应该显著减小
        if width_ratio > 0.95:
            return 1, "无法启动运动 (宽度几乎无变化)"

        # 结合对称性评分
        if oral_asym < 3 and corner_diff < 3:
            if width_ratio < 0.70:
                return 5, "运动完整 (收缩明显且对称)"
            elif width_ratio < 0.80:
                return 4, "几乎完整"
            else:
                return 3, "启动但幅度不足"
        elif oral_asym < 6 and corner_diff < 6:
            if width_ratio < 0.75:
                return 4, "几乎完整 (轻度不对称)"
            else:
                return 3, "启动但不对称"
        elif oral_asym < 10:
            return 2, "轻微启动 (明显不对称)"
        else:
            return 1, "无法启动 (严重不对称)"
    else:
        # 没有基线，使用静态对称性
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
    """检测撅嘴时的联动运动"""
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

        if avg_change > 0.15:
            synkinesis["eye_synkinesis"] = 3
        elif avg_change > 0.08:
            synkinesis["eye_synkinesis"] = 2
        elif avg_change > 0.04:
            synkinesis["eye_synkinesis"] = 1

    return synkinesis


def visualize_lip_pucker(frame: np.ndarray, landmarks, w: int, h: int,
                         result: ActionResult,
                         metrics: Dict[str, Any],
                         palsy_detection: Dict[str, Any]) -> np.ndarray:
    """可视化撅嘴指标"""
    img = frame.copy()

    # 绘制嘴部轮廓
    draw_polygon(img, landmarks, w, h, LM.OUTER_LIP, (0, 255, 0), 2)
    draw_polygon(img, landmarks, w, h, LM.INNER_LIP, (0, 200, 200), 1)

    # 绘制嘴角点
    left_corner = metrics["left_corner"]
    right_corner = metrics["right_corner"]
    cv2.circle(img, (int(left_corner[0]), int(left_corner[1])), 5, (255, 0, 0), -1)
    cv2.circle(img, (int(right_corner[0]), int(right_corner[1])), 5, (0, 0, 255), -1)

    # 绘制嘴宽测量线
    cv2.line(img, (int(left_corner[0]), int(left_corner[1])),
             (int(right_corner[0]), int(right_corner[1])), (0, 255, 255), 2)

    # 绘制口角角度参考线
    if result.oral_angle:
        oral = result.oral_angle
        cv2.line(img, (int(oral.E[0]), int(oral.E[1])),
                 (int(oral.F[0]), int(oral.F[1])), (0, 255, 0), 1)

    # 信息面板
    panel_h = 300
    cv2.rectangle(img, (5, 5), (380, panel_h), (0, 0, 0), -1)
    cv2.rectangle(img, (5, 5), (380, panel_h), (255, 255, 255), 1)

    y = 28
    cv2.putText(img, f"{ACTION_NAME}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y += 28

    cv2.putText(img, "=== Mouth Metrics ===", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    y += 20

    cv2.putText(img, f"Width: {metrics['mouth_width']:.1f}px", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    y += 18

    cv2.putText(img, f"Height: {metrics['mouth_height']:.1f}px", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    y += 18

    cv2.putText(img, f"W/H Ratio: {metrics['width_height_ratio']:.2f}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    y += 22

    if "width_change" in metrics:
        cv2.putText(img, "=== Changes from Baseline ===", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        y += 20

        cv2.putText(img,
                    f"Width Change: {metrics['width_change']:+.1f}px ({metrics.get('width_change_percent', 0):+.1f}%)",
                    (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 18

        cv2.putText(img, f"Width Ratio: {metrics.get('width_ratio', 1):.3f}", (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 22

    # 口角对称性
    oral_asym = metrics.get("oral_angle", {}).get("asymmetry", 0)
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

    return img


def process(landmarks_seq: List, frames_seq: List, w: int, h: int,
            video_info: Dict[str, Any], output_dir: Path,
            baseline_result: Optional[ActionResult] = None,
            baseline_landmarks=None) -> Optional[ActionResult]:
    """处理LipPucker动作"""
    if not landmarks_seq or not frames_seq:
        return None

    # 找峰值帧 (嘴宽最小)
    peak_idx, peak_debug = find_peak_frame(landmarks_seq, frames_seq, w, h, baseline_landmarks=baseline_landmarks)
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

    # 计算撅嘴特有指标
    metrics = compute_lip_pucker_metrics(peak_landmarks, w, h, baseline_landmarks)

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
        "lip_pucker_metrics": {
            "mouth_width": metrics["mouth_width"],
            "mouth_height": metrics["mouth_height"],
            "width_height_ratio": metrics["width_height_ratio"],
            "oral_angle": metrics["oral_angle"],
        },
        "palsy_detection": palsy_detection,
        "voluntary_interpretation": interpretation,
        "synkinesis": synkinesis,
    }

    if "baseline" in metrics:
        result.action_specific["baseline"] = metrics["baseline"]
        result.action_specific["changes"] = {
            "width_change": metrics.get("width_change", 0),
            "width_change_percent": metrics.get("width_change_percent", 0),
            "width_ratio": metrics.get("width_ratio", 1),
        }

    # 创建输出目录
    action_dir = output_dir / ACTION_NAME
    action_dir.mkdir(parents=True, exist_ok=True)

    # 保存原始帧
    cv2.imwrite(str(action_dir / "peak_raw.jpg"), peak_frame)

    # 保存可视化
    vis = visualize_lip_pucker(peak_frame, peak_landmarks, w, h, result, metrics, palsy_detection)
    cv2.imwrite(str(action_dir / "peak_indicators.jpg"), vis)

    # 保存嘴唇Z轴曲线（相对基线）
    if peak_debug and "lip_z_delta_norm" in peak_debug:
        _save_lip_z_curve_png(action_dir / "mouth_z_curve.png",
                              peak_debug["lip_z_delta_norm"],
                              peak_idx)

    # 也把 debug 写进 json（可解释性）
    result.action_specific["peak_debug"] = peak_debug

    # 保存JSON
    with open(action_dir / "indicators.json", 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"    [OK] {ACTION_NAME}: Width={metrics['mouth_width']:.1f}px")
    if "width_change_percent" in metrics:
        print(f"         Width Change: {metrics['width_change_percent']:+.1f}%")
    print(f"         Palsy: {palsy_detection.get('interpretation', 'N/A')}")
    print(f"         Voluntary Score: {result.voluntary_movement_score}/5 ({interpretation})")

    return result