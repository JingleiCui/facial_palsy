#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
临床分级主运行脚本
==================

整合所有动作模块，执行完整的面瘫评估：
1. 处理11个标准动作视频
2. 计算完整Sunnybrook评分
3. 生成详细的HTML报告
4. 输出可视化结果和JSON数据

使用方法:
    python run_clinical_grading.py
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from clinical_base import (
    LM, LandmarkExtractor, ActionResult,
    db_fetch_examinations, db_fetch_videos_for_exam, db_fetch_labels,
    compute_ear, extract_common_indicators
)

from sunnybrook_scorer import (
    RestingSymmetry, VoluntaryMovement, VoluntaryMovementItem,
    Synkinesis, SynkinesisItem, SunnybrookScore,
    compute_resting_symmetry, compute_voluntary_score_from_ratio,
    compute_sunnybrook_composite, SUNNYBROOK_EXPRESSION_MAPPING
)

# 导入动作模块
import neutral_face
import eye_blink
import smile

# 尝试导入其他动作模块（如果存在）
try:
    import action_close_eye
except ImportError:
    action_close_eye = None

try:
    import action_raise_eyebrow
except ImportError:
    action_raise_eyebrow = None

# =============================================================================
# 配置参数
# =============================================================================

DATABASE_PATH = r"/Users/cuijinglei/PycharmProjects/medicalProject/facialPalsy/facialPalsy.db"
MEDIAPIPE_MODEL_PATH = r"/Users/cuijinglei/PycharmProjects/medicalProject/models/face_landmarker.task"
OUTPUT_DIR = r"/Users/cuijinglei/Documents/facialPalsy/HGFA/clinical_grading"
PATIENT_LIMIT = None
TARGET_EXAM_ID = None

# =============================================================================
# 动作处理器映射
# =============================================================================

# 所有11个动作
ALL_ACTIONS = [
    "NeutralFace",
    "Smile",
    "ShowTeeth",
    "RaiseEyebrow",
    "CloseEyeSoftly",
    "CloseEyeHardly",
    "VoluntaryEyeBlink",
    "SpontaneousEyeBlink",
    "LipPucker",
    "BlowCheek",
    "ShrugNose"
]


def find_peak_frame_generic(landmarks_seq, frames_seq, w, h, action_name):
    """通用峰值帧查找"""
    if action_name == "NeutralFace":
        return neutral_face.find_peak_frame(landmarks_seq, frames_seq, w, h)
    elif action_name in ["Smile", "ShowTeeth"]:
        return smile.find_peak_frame_smile(landmarks_seq, frames_seq, w, h)
    elif action_name in ["VoluntaryEyeBlink", "SpontaneousEyeBlink", "CloseEyeSoftly", "CloseEyeHardly"]:
        return eye_blink.find_peak_frame_blink(landmarks_seq, frames_seq, w, h)
    elif action_name == "RaiseEyebrow":
        # 找眉毛最高的帧
        from clinical_base import compute_brow_height
        max_brow = -1.0
        max_idx = 0
        for i, lm in enumerate(landmarks_seq):
            if lm is None:
                continue
            l_brow = compute_brow_height(lm, w, h, True)
            r_brow = compute_brow_height(lm, w, h, False)
            avg = (l_brow + r_brow) / 2
            if avg > max_brow:
                max_brow = avg
                max_idx = i
        return max_idx
    elif action_name == "LipPucker":
        # 找嘴最窄的帧
        from clinical_base import compute_mouth_metrics
        min_width = float('inf')
        min_idx = 0
        for i, lm in enumerate(landmarks_seq):
            if lm is None:
                continue
            mouth = compute_mouth_metrics(lm, w, h)
            if mouth["width"] < min_width:
                min_width = mouth["width"]
                min_idx = i
        return min_idx
    else:
        # 默认: 使用NeutralFace的方法
        return neutral_face.find_peak_frame(landmarks_seq, frames_seq, w, h)


def process_action_generic(landmarks_seq, frames_seq, w, h, video_info, output_dir,
                           action_name, baseline_result=None, baseline_landmarks=None):
    """通用动作处理"""
    if action_name == "NeutralFace":
        return neutral_face.process(landmarks_seq, frames_seq, w, h, video_info, output_dir)
    elif action_name == "Smile":
        return smile.process_smile(landmarks_seq, frames_seq, w, h, video_info, output_dir,
                                          baseline_result, baseline_landmarks)
    elif action_name == "ShowTeeth":
        return smile.process_show_teeth(landmarks_seq, frames_seq, w, h, video_info, output_dir,
                                               baseline_result, baseline_landmarks)
    elif action_name == "VoluntaryEyeBlink":
        return eye_blink.process_voluntary_blink(landmarks_seq, frames_seq, w, h, video_info, output_dir,
                                                        baseline_result)
    elif action_name == "SpontaneousEyeBlink":
        return eye_blink.process_spontaneous_blink(landmarks_seq, frames_seq, w, h, video_info, output_dir,
                                                          baseline_result)
    else:
        # 其他动作使用通用处理
        return process_generic_action(landmarks_seq, frames_seq, w, h, video_info, output_dir,
                                      action_name, baseline_result)


def process_generic_action(landmarks_seq, frames_seq, w, h, video_info, output_dir,
                           action_name, baseline_result=None):
    """通用动作处理（用于没有专门模块的动作）"""
    if not landmarks_seq or not frames_seq:
        return None

    # 找峰值帧
    peak_idx = find_peak_frame_generic(landmarks_seq, frames_seq, w, h, action_name)
    peak_landmarks = landmarks_seq[peak_idx]
    peak_frame = frames_seq[peak_idx]

    if peak_landmarks is None:
        return None

    # 动作名称映射
    action_cn_map = {
        "RaiseEyebrow": "皱额",
        "CloseEyeSoftly": "轻闭眼",
        "CloseEyeHardly": "用力闭眼",
        "LipPucker": "撅嘴",
        "BlowCheek": "鼓腮",
        "ShrugNose": "皱鼻",
    }

    result = ActionResult(
        action_name=action_name,
        action_name_cn=action_cn_map.get(action_name, action_name),
        video_path=video_info.get("file_path", ""),
        total_frames=len(frames_seq),
        peak_frame_idx=peak_idx,
        image_size=(w, h),
        fps=video_info.get("fps", 30.0)
    )

    # 提取通用指标
    extract_common_indicators(peak_landmarks, w, h, result)

    # 计算Voluntary Movement评分
    if baseline_result:
        # 根据动作类型选择比较指标
        if action_name == "RaiseEyebrow":
            ratio = result.brow_height_ratio
        elif action_name in ["CloseEyeSoftly", "CloseEyeHardly"]:
            # 闭眼程度比较
            ratio = result.left_ear / result.right_ear if result.right_ear > 1e-9 else 1.0
        elif action_name == "LipPucker":
            baseline_width = baseline_result.mouth_width
            if baseline_width > 1e-9:
                ratio = result.mouth_width / baseline_width
            else:
                ratio = 1.0
        else:
            ratio = 1.0

        score, interp = compute_voluntary_score_from_ratio(ratio)
        result.voluntary_movement_score = score

    # 检测联动
    if baseline_result:
        synkinesis = {}

        # 检测眼部联动（用于嘴部动作）
        if action_name in ["LipPucker", "BlowCheek", "ShrugNose"]:
            l_ear_change = abs(result.left_ear - baseline_result.left_ear)
            r_ear_change = abs(result.right_ear - baseline_result.right_ear)
            avg_change = (l_ear_change + r_ear_change) / 2

            if avg_change > 0.15:
                synkinesis["eye_synkinesis"] = 3
            elif avg_change > 0.10:
                synkinesis["eye_synkinesis"] = 2
            elif avg_change > 0.05:
                synkinesis["eye_synkinesis"] = 1
            else:
                synkinesis["eye_synkinesis"] = 0

        # 检测嘴部联动（用于眼部动作）
        if action_name in ["RaiseEyebrow", "CloseEyeSoftly", "CloseEyeHardly"]:
            mouth_change = abs(result.mouth_width - baseline_result.mouth_width)
            if baseline_result.mouth_width > 1e-9:
                mouth_ratio = mouth_change / baseline_result.mouth_width
                if mouth_ratio > 0.20:
                    synkinesis["mouth_synkinesis"] = 3
                elif mouth_ratio > 0.10:
                    synkinesis["mouth_synkinesis"] = 2
                elif mouth_ratio > 0.05:
                    synkinesis["mouth_synkinesis"] = 1
                else:
                    synkinesis["mouth_synkinesis"] = 0

        result.synkinesis_scores = synkinesis

    # 创建输出目录
    action_dir = output_dir / action_name
    action_dir.mkdir(parents=True, exist_ok=True)

    # 保存原始帧
    cv2.imwrite(str(action_dir / "peak_raw.jpg"), peak_frame)

    # 简单可视化
    vis = visualize_generic_action(peak_frame, peak_landmarks, w, h, result)
    cv2.imwrite(str(action_dir / "peak_indicators.jpg"), vis)

    # 保存JSON
    with open(action_dir / "indicators.json", 'w', encoding='utf-8') as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

    print(f"    [OK] {action_name}: EAR L={result.left_ear:.3f} R={result.right_ear:.3f}")

    return result


def visualize_generic_action(frame, landmarks, w, h, result):
    """通用动作可视化"""
    from clinical_base import draw_polygon, pt2d

    img = frame.copy()

    # 绘制眼部
    draw_polygon(img, landmarks, w, h, LM.EYE_CONTOUR_L, (255, 0, 0), 2)
    draw_polygon(img, landmarks, w, h, LM.EYE_CONTOUR_R, (0, 165, 255), 2)

    # 绘制眉毛
    draw_polygon(img, landmarks, w, h, LM.BROW_L, (255, 100, 100), 1, False)
    draw_polygon(img, landmarks, w, h, LM.BROW_R, (100, 165, 255), 1, False)

    # 绘制嘴部
    draw_polygon(img, landmarks, w, h, LM.OUTER_LIP, (0, 255, 0), 2)

    # 信息面板
    y = 25
    cv2.putText(img, f"{result.action_name} - {result.action_name_cn}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y += 30

    cv2.putText(img, f"EAR L:{result.left_ear:.3f} R:{result.right_ear:.3f}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y += 22

    cv2.putText(img, f"Eye Area Ratio: {result.eye_area_ratio:.3f}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y += 22

    cv2.putText(img, f"Brow H Ratio: {result.brow_height_ratio:.3f}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y += 22

    cv2.putText(img, f"Mouth W: {result.mouth_width:.1f}px", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y += 22

    if result.oral_angle:
        cv2.putText(img, f"AOE:{result.oral_angle.AOE_angle:+.1f} BOF:{result.oral_angle.BOF_angle:+.1f}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 22

    cv2.putText(img, f"Voluntary Score: {result.voluntary_movement_score}/5", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    return img


# =============================================================================
# Sunnybrook评分计算
# =============================================================================

def calculate_sunnybrook_from_results(action_results: Dict[str, ActionResult]) -> Optional[SunnybrookScore]:
    """从动作结果计算完整Sunnybrook评分"""

    # 1. Resting Symmetry (从NeutralFace)
    if "NeutralFace" not in action_results:
        return None

    neutral = action_results["NeutralFace"]
    resting_data = neutral.action_specific.get("resting_symmetry", {})

    if not resting_data:
        # 重新计算
        oral = neutral.oral_angle
        resting = compute_resting_symmetry(
            palpebral_height_ratio=neutral.palpebral_height_ratio,
            nlf_ratio=neutral.nlf_ratio,
            oral_angle_diff=oral.angle_diff if oral else 0,
            aoe_angle=oral.AOE_angle if oral else 0,
            bof_angle=oral.BOF_angle if oral else 0
        )
    else:
        # 从存储的数据重建
        from sunnybrook_scorer import RestingSymmetryItem
        resting = RestingSymmetry(
            eye=RestingSymmetryItem(
                region="Eye", region_cn=resting_data["eye"]["region_cn"],
                status=resting_data["eye"]["status"], status_cn=resting_data["eye"]["status_cn"],
                score=resting_data["eye"]["score"], measurement=resting_data["eye"]["measurement"],
                threshold_info=resting_data["eye"]["threshold_info"]
            ),
            cheek=RestingSymmetryItem(
                region="Cheek", region_cn=resting_data["cheek"]["region_cn"],
                status=resting_data["cheek"]["status"], status_cn=resting_data["cheek"]["status_cn"],
                score=resting_data["cheek"]["score"], measurement=resting_data["cheek"]["measurement"],
                threshold_info=resting_data["cheek"]["threshold_info"]
            ),
            mouth=RestingSymmetryItem(
                region="Mouth", region_cn=resting_data["mouth"]["region_cn"],
                status=resting_data["mouth"]["status"], status_cn=resting_data["mouth"]["status_cn"],
                score=resting_data["mouth"]["score"], measurement=resting_data["mouth"]["measurement"],
                threshold_info=resting_data["mouth"]["threshold_info"]
            ),
            raw_score=resting_data["raw_score"],
            total_score=resting_data["total_score"],
            affected_side=resting_data["affected_side"]
        )

    # 2. Voluntary Movement (从5个标准表情)
    vol_items = []

    # Brow -> RaiseEyebrow
    if "RaiseEyebrow" in action_results:
        r = action_results["RaiseEyebrow"]
        vol_items.append(VoluntaryMovementItem(
            expression="Brow", expression_cn="皱额/抬眉",
            left_value=r.left_brow_height, right_value=r.right_brow_height,
            ratio=r.brow_height_ratio,
            score=r.voluntary_movement_score,
            interpretation=""
        ))
    else:
        vol_items.append(VoluntaryMovementItem(
            expression="Brow", expression_cn="皱额/抬眉",
            left_value=0, right_value=0, ratio=1.0, score=5, interpretation="未评估"
        ))

    # Gentle Eye closure -> CloseEyeSoftly
    if "CloseEyeSoftly" in action_results:
        r = action_results["CloseEyeSoftly"]
        vol_items.append(VoluntaryMovementItem(
            expression="GentleEyeClosure", expression_cn="轻闭眼",
            left_value=r.left_ear, right_value=r.right_ear,
            ratio=r.left_ear / r.right_ear if r.right_ear > 1e-9 else 1.0,
            score=r.voluntary_movement_score,
            interpretation=""
        ))
    else:
        vol_items.append(VoluntaryMovementItem(
            expression="GentleEyeClosure", expression_cn="轻闭眼",
            left_value=0, right_value=0, ratio=1.0, score=5, interpretation="未评估"
        ))

    # Open mouth smile -> Smile or ShowTeeth
    smile_result = action_results.get("Smile") or action_results.get("ShowTeeth")
    if smile_result:
        oral = smile_result.oral_angle
        vol_items.append(VoluntaryMovementItem(
            expression="OpenMouthSmile", expression_cn="露齿微笑",
            left_value=oral.BOF_angle if oral else 0,
            right_value=oral.AOE_angle if oral else 0,
            ratio=1.0,  # 使用评分直接
            score=smile_result.voluntary_movement_score,
            interpretation=""
        ))
    else:
        vol_items.append(VoluntaryMovementItem(
            expression="OpenMouthSmile", expression_cn="露齿微笑",
            left_value=0, right_value=0, ratio=1.0, score=5, interpretation="未评估"
        ))

    # Snarl -> ShrugNose
    if "ShrugNose" in action_results:
        r = action_results["ShrugNose"]
        vol_items.append(VoluntaryMovementItem(
            expression="Snarl", expression_cn="皱鼻",
            left_value=r.nlf_ratio, right_value=1.0,
            ratio=r.nlf_ratio,
            score=r.voluntary_movement_score,
            interpretation=""
        ))
    else:
        vol_items.append(VoluntaryMovementItem(
            expression="Snarl", expression_cn="皱鼻",
            left_value=0, right_value=0, ratio=1.0, score=5, interpretation="未评估"
        ))

    # Lip pucker -> LipPucker
    if "LipPucker" in action_results:
        r = action_results["LipPucker"]
        vol_items.append(VoluntaryMovementItem(
            expression="LipPucker", expression_cn="撅嘴",
            left_value=r.mouth_width, right_value=r.mouth_width,
            ratio=1.0,
            score=r.voluntary_movement_score,
            interpretation=""
        ))
    else:
        vol_items.append(VoluntaryMovementItem(
            expression="LipPucker", expression_cn="撅嘴",
            left_value=0, right_value=0, ratio=1.0, score=5, interpretation="未评估"
        ))

    raw_sum = sum(item.score for item in vol_items)
    voluntary = VoluntaryMovement(
        items=vol_items,
        raw_sum=raw_sum,
        total_score=raw_sum * 4  # 满分 5×5×4 = 100
    )

    # 3. Synkinesis (从所有有联动检测的动作)
    syn_items = []

    for action_name, result in action_results.items():
        if action_name == "NeutralFace":
            continue

        if result.synkinesis_scores:
            total_syn = sum(result.synkinesis_scores.values())
            syn_items.append(SynkinesisItem(
                expression=action_name,
                expression_cn=result.action_name_cn,
                eye_synkinesis=result.synkinesis_scores.get("eye_synkinesis", 0),
                mouth_synkinesis=result.synkinesis_scores.get("mouth_synkinesis", 0),
                total_score=total_syn,
                interpretation=""
            ))

    synkinesis = Synkinesis(
        items=syn_items,
        total_score=sum(item.total_score for item in syn_items)
    )

    # 4. 计算Composite Score
    return compute_sunnybrook_composite(resting, voluntary, synkinesis)


# =============================================================================
# HTML报告生成
# =============================================================================

def generate_html_report(exam_id: str, patient_id: str,
                         action_results: Dict[str, ActionResult],
                         sunnybrook: Optional[SunnybrookScore],
                         ground_truth: Dict[str, Any],
                         output_dir: Path) -> None:
    """生成详细HTML报告"""

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>面部指标分析报告 - {exam_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1600px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #666; margin-top: 30px; border-left: 4px solid #2196F3; padding-left: 10px; }}
        h3 {{ color: #888; margin-top: 20px; }}
        .info-box {{ background: #e8f5e9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .ground-truth {{ background: #fff3e0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .sunnybrook-summary {{ background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
        .score-box {{ display: inline-block; padding: 15px 25px; margin: 10px; border-radius: 8px; text-align: center; }}
        .score-label {{ font-size: 0.9em; color: #666; margin-bottom: 5px; }}
        .score-value {{ font-size: 1.8em; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background: #1976d2; color: white; }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        .score-0 {{ color: #4CAF50; font-weight: bold; }}
        .score-1 {{ color: #FF9800; font-weight: bold; }}
        .score-2, .score-3 {{ color: #f44336; font-weight: bold; }}
        .action-card {{ border: 1px solid #ddd; margin: 15px 0; padding: 20px; border-radius: 8px; background: #fafafa; }}
        .action-title {{ font-weight: bold; font-size: 1.3em; color: #1976d2; margin-bottom: 15px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; }}
        .metric-card {{ background: white; padding: 12px; border-radius: 5px; border: 1px solid #e0e0e0; }}
        .metric-label {{ color: #666; font-size: 0.85em; margin-bottom: 3px; }}
        .metric-value {{ font-size: 1.1em; font-weight: bold; color: #333; }}
        .images {{ display: flex; gap: 15px; margin-top: 15px; flex-wrap: wrap; }}
        .images img {{ max-width: 350px; border: 1px solid #ddd; border-radius: 5px; }}
        .formula {{ background: #f5f5f5; padding: 10px; border-radius: 5px; font-family: monospace; margin-top: 10px; }}
        .grade-box {{ background: #4CAF50; color: white; padding: 20px; border-radius: 10px; text-align: center; margin-top: 15px; }}
        .grade-box.warning {{ background: #FF9800; }}
        .grade-box.danger {{ background: #f44336; }}
    </style>
</head>
<body>
<div class="container">
    <h1>🏥 面部指标分析报告</h1>

    <div class="info-box">
        <strong>检查ID:</strong> {exam_id}<br>
        <strong>患者ID:</strong> {patient_id}<br>
        <strong>分析时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
        <strong>处理动作数:</strong> {len(action_results)}
    </div>

    <div class="ground-truth">
        <h3>📋 医生标注 (Ground Truth)</h3>
        <strong>面瘫:</strong> {'是' if ground_truth.get('has_palsy') else '否'}<br>
        <strong>患侧:</strong> {ground_truth.get('palsy_side', 'N/A')}<br>
        <strong>HB分级:</strong> {ground_truth.get('hb_grade', 'N/A')}<br>
        <strong>Sunnybrook评分:</strong> {ground_truth.get('sunnybrook_score', 'N/A')}
    </div>
"""

    # Sunnybrook评分汇总
    if sunnybrook:
        grade_class = ""
        if sunnybrook.composite_score < 50:
            grade_class = "danger"
        elif sunnybrook.composite_score < 70:
            grade_class = "warning"

        html += f"""
    <div class="sunnybrook-summary">
        <h2>📊 Sunnybrook 面神经分级评分</h2>

        <div style="display: flex; flex-wrap: wrap; justify-content: center;">
            <div class="score-box" style="background: #ffcdd2;">
                <div class="score-label">Resting Symmetry</div>
                <div class="score-value">{sunnybrook.resting_score}</div>
                <div style="font-size: 0.8em; color: #666;">(0-20)</div>
            </div>
            <div class="score-box" style="background: #c8e6c9;">
                <div class="score-label">Voluntary Movement</div>
                <div class="score-value">{sunnybrook.voluntary_score}</div>
                <div style="font-size: 0.8em; color: #666;">(20-100)</div>
            </div>
            <div class="score-box" style="background: #ffe0b2;">
                <div class="score-label">Synkinesis</div>
                <div class="score-value">{sunnybrook.synkinesis_score}</div>
                <div style="font-size: 0.8em; color: #666;">(0-15)</div>
            </div>
        </div>

        <div class="grade-box {grade_class}">
            <div style="font-size: 2.5em; font-weight: bold;">Composite: {sunnybrook.composite_score}</div>
            <div style="font-size: 1.2em; margin-top: 10px;">Grade {sunnybrook.grade}: {sunnybrook.grade_description}</div>
        </div>

        <div class="formula">
            公式: Composite = Voluntary({sunnybrook.voluntary_score}) - Resting({sunnybrook.resting_score}) - Synkinesis({sunnybrook.synkinesis_score}) = {sunnybrook.composite_score}
        </div>
    </div>
"""

        # Resting Symmetry 详细表格
        rs = sunnybrook.resting_symmetry
        html += f"""
    <h2>1️⃣ Resting Symmetry (静息对称性)</h2>
    <table>
        <tr>
            <th>部位</th>
            <th>状态</th>
            <th>测量值</th>
            <th>评分</th>
            <th>说明</th>
        </tr>
        <tr>
            <td>👁️ Eye (眼/睑裂)</td>
            <td>{rs.eye.status_cn}</td>
            <td>{rs.eye.measurement:.3f}</td>
            <td class="score-{rs.eye.score}">{rs.eye.score}</td>
            <td>{rs.eye.threshold_info}</td>
        </tr>
        <tr>
            <td>😊 Cheek (颊/鼻唇沟)</td>
            <td>{rs.cheek.status_cn}</td>
            <td>{rs.cheek.measurement:.3f}</td>
            <td class="score-{rs.cheek.score}">{rs.cheek.score}</td>
            <td>{rs.cheek.threshold_info}</td>
        </tr>
        <tr>
            <td>👄 Mouth (嘴)</td>
            <td>{rs.mouth.status_cn}</td>
            <td>{rs.mouth.measurement:.1f}°</td>
            <td class="score-{rs.mouth.score}">{rs.mouth.score}</td>
            <td>{rs.mouth.threshold_info}</td>
        </tr>
        <tr style="background: #e3f2fd; font-weight: bold;">
            <td colspan="3">Total (Raw Score × 5)</td>
            <td>{rs.raw_score} × 5 = {rs.total_score}</td>
            <td>判断患侧: {rs.affected_side}</td>
        </tr>
    </table>
"""

        # Voluntary Movement 详细表格
        vm = sunnybrook.voluntary_movement
        html += f"""
    <h2>2️⃣ Symmetry of Voluntary Movement (主动运动对称性)</h2>
    <p>评分标准: 1=无法启动, 2=轻微启动, 3=启动但不对称, 4=几乎完整, 5=完整</p>
    <table>
        <tr>
            <th>表情</th>
            <th>对应动作</th>
            <th>左侧测量</th>
            <th>右侧测量</th>
            <th>比值</th>
            <th>评分 (1-5)</th>
        </tr>
"""
        for item in vm.items:
            html += f"""
        <tr>
            <td>{item.expression_cn}</td>
            <td>{item.expression}</td>
            <td>{item.left_value:.3f}</td>
            <td>{item.right_value:.3f}</td>
            <td>{item.ratio:.3f}</td>
            <td class="score-{5 - item.score if item.score < 4 else 0}">{item.score}</td>
        </tr>
"""
        html += f"""
        <tr style="background: #e3f2fd; font-weight: bold;">
            <td colspan="5">Total (Sum × 4)</td>
            <td>{vm.raw_sum} × 4 = {vm.total_score}</td>
        </tr>
    </table>
"""

        # Synkinesis 详细表格
        syn = sunnybrook.synkinesis
        html += f"""
    <h2>3️⃣ Synkinesis (联动运动)</h2>
    <p>评分标准: 0=无联动, 1=轻度, 2=中度, 3=重度</p>
    <table>
        <tr>
            <th>表情</th>
            <th>眼部联动</th>
            <th>嘴部联动</th>
            <th>总分</th>
        </tr>
"""
        if syn.items:
            for item in syn.items:
                html += f"""
        <tr>
            <td>{item.expression_cn}</td>
            <td class="score-{item.eye_synkinesis}">{item.eye_synkinesis}</td>
            <td class="score-{item.mouth_synkinesis}">{item.mouth_synkinesis}</td>
            <td>{item.total_score}</td>
        </tr>
"""
        else:
            html += """
        <tr>
            <td colspan="4" style="text-align: center; color: #666;">未检测到联动运动</td>
        </tr>
"""
        html += f"""
        <tr style="background: #e3f2fd; font-weight: bold;">
            <td colspan="3">Total</td>
            <td>{syn.total_score}</td>
        </tr>
    </table>
"""

    # 各动作详细结果
    html += """
    <h2>📹 各动作详细分析</h2>
"""

    for action_name, result in action_results.items():
        oral = result.oral_angle

        html += f"""
    <div class="action-card">
        <div class="action-title">{action_name} - {result.action_name_cn}</div>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">峰值帧</div>
                <div class="metric-value">{result.peak_frame_idx} / {result.total_frames}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">ICD</div>
                <div class="metric-value">{result.icd:.1f}px</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">EAR Left</div>
                <div class="metric-value">{result.left_ear:.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">EAR Right</div>
                <div class="metric-value">{result.right_ear:.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Eye Area Left</div>
                <div class="metric-value">{result.left_eye_area:.1f}px²</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Eye Area Right</div>
                <div class="metric-value">{result.right_eye_area:.1f}px²</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Eye Area Ratio</div>
                <div class="metric-value">{result.eye_area_ratio:.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Palpebral H Ratio</div>
                <div class="metric-value">{result.palpebral_height_ratio:.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Brow H Ratio</div>
                <div class="metric-value">{result.brow_height_ratio:.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Mouth Width</div>
                <div class="metric-value">{result.mouth_width:.1f}px</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">NLF Ratio</div>
                <div class="metric-value">{result.nlf_ratio:.3f}</div>
            </div>
"""
        if oral:
            html += f"""
            <div class="metric-card">
                <div class="metric-label">AOE (Right)</div>
                <div class="metric-value">{oral.AOE_angle:+.2f}°</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">BOF (Left)</div>
                <div class="metric-value">{oral.BOF_angle:+.2f}°</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Oral Asymmetry</div>
                <div class="metric-value">{oral.angle_asymmetry:.2f}°</div>
            </div>
"""

        html += f"""
            <div class="metric-card">
                <div class="metric-label">Voluntary Score</div>
                <div class="metric-value">{result.voluntary_movement_score}/5</div>
            </div>
        </div>

        <div class="images">
            <img src="{action_name}/peak_raw.jpg" alt="原始帧">
            <img src="{action_name}/peak_indicators.jpg" alt="指标可视化">
"""

        # 如果有EAR曲线
        if action_name in ["VoluntaryEyeBlink", "SpontaneousEyeBlink"]:
            html += f'            <img src="{action_name}/ear_curve.png" alt="EAR曲线">\n'

        # 如果有Resting Symmetry可视化
        if action_name == "NeutralFace":
            html += f'            <img src="{action_name}/resting_symmetry.jpg" alt="Resting Symmetry">\n'

        html += """        </div>
    </div>
"""

    html += """
</div>
</body>
</html>
"""

    with open(output_dir / "report.html", 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"  [OK] HTML报告: {output_dir / 'report.html'}")


# =============================================================================
# 主处理函数
# =============================================================================

def process_examination(examination: Dict[str, Any], db_path: str,
                        output_dir: Path, extractor: LandmarkExtractor) -> Dict[str, Any]:
    """处理单个检查"""
    exam_id = examination["examination_id"]
    patient_id = examination["patient_id"]

    print(f"\n{'=' * 60}")
    print(f"处理检查: {exam_id}")
    print(f"{'=' * 60}")

    # 获取视频和标签
    videos = db_fetch_videos_for_exam(db_path, exam_id)
    labels = db_fetch_labels(db_path, exam_id)

    print(f"找到 {len(videos)} 个动作视频")
    print(f"医生标注: {labels}")

    exam_output_dir = output_dir / exam_id
    exam_output_dir.mkdir(parents=True, exist_ok=True)

    action_results: Dict[str, ActionResult] = {}
    baseline_result = None
    baseline_landmarks = None

    # 首先处理NeutralFace获取基线
    if "NeutralFace" in videos:
        video_info = videos["NeutralFace"]
        video_path = video_info["file_path"]

        print(f"\n  处理基线动作: NeutralFace")

        if os.path.exists(video_path):
            landmarks_seq, frames_seq = extractor.extract_sequence(
                video_path,
                video_info.get("start_frame", 0),
                video_info.get("end_frame", None)
            )

            if landmarks_seq and frames_seq:
                h, w = frames_seq[0].shape[:2]
                result = neutral_face.process(landmarks_seq, frames_seq, w, h,
                                                     video_info, exam_output_dir)
                if result:
                    action_results["NeutralFace"] = result
                    baseline_result = result

                    # 保存基线landmarks
                    peak_idx = result.peak_frame_idx
                    baseline_landmarks = landmarks_seq[peak_idx]

    # 处理其他动作
    for action_name, video_info in videos.items():
        if action_name == "NeutralFace":
            continue

        video_path = video_info["file_path"]
        print(f"\n  处理动作: {action_name}")

        if not os.path.exists(video_path):
            print(f"    [!] 视频不存在: {video_path}")
            continue

        landmarks_seq, frames_seq = extractor.extract_sequence(
            video_path,
            video_info.get("start_frame", 0),
            video_info.get("end_frame", None)
        )

        if not landmarks_seq or not frames_seq:
            print(f"    [!] 无法提取landmarks")
            continue

        h, w = frames_seq[0].shape[:2]

        result = process_action_generic(
            landmarks_seq, frames_seq, w, h, video_info, exam_output_dir,
            action_name, baseline_result, baseline_landmarks
        )

        if result:
            action_results[action_name] = result

    # 计算Sunnybrook评分
    sunnybrook = calculate_sunnybrook_from_results(action_results)

    if sunnybrook:
        print(f"\n  === Sunnybrook评分 ===")
        print(f"  Resting: {sunnybrook.resting_score}")
        print(f"  Voluntary: {sunnybrook.voluntary_score}")
        print(f"  Synkinesis: {sunnybrook.synkinesis_score}")
        print(f"  Composite: {sunnybrook.composite_score}")
        print(f"  Grade: {sunnybrook.grade} - {sunnybrook.grade_description}")

    # 生成HTML报告
    generate_html_report(exam_id, patient_id, action_results, sunnybrook, labels, exam_output_dir)

    # 保存汇总JSON
    summary = {
        "examination_id": exam_id,
        "patient_id": patient_id,
        "ground_truth": labels,
        "actions": {name: result.to_dict() for name, result in action_results.items()},
    }

    if sunnybrook:
        summary["sunnybrook"] = sunnybrook.to_dict()

    with open(exam_output_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


# =============================================================================
# 主函数
# =============================================================================

def main():
    print("=" * 70)
    print("面部临床分级系统 - 完整Sunnybrook评分")
    print("=" * 70)

    print(f"\n配置:")
    print(f"  数据库: {DATABASE_PATH}")
    print(f"  模型: {MEDIAPIPE_MODEL_PATH}")
    print(f"  输出: {OUTPUT_DIR}")

    if not os.path.exists(DATABASE_PATH):
        print(f"\n[ERROR] 数据库不存在: {DATABASE_PATH}")
        return

    if not os.path.exists(MEDIAPIPE_MODEL_PATH):
        print(f"\n[ERROR] MediaPipe模型不存在: {MEDIAPIPE_MODEL_PATH}")
        return

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n获取检查记录...")
    examinations = db_fetch_examinations(DATABASE_PATH, TARGET_EXAM_ID, PATIENT_LIMIT)
    print(f"找到 {len(examinations)} 个检查记录")

    if not examinations:
        print("[ERROR] 没有有效的检查记录")
        return

    print(f"\n初始化MediaPipe...")

    all_results = []

    with LandmarkExtractor(MEDIAPIPE_MODEL_PATH) as extractor:
        for i, exam in enumerate(examinations):
            print(f"\n[{i + 1}/{len(examinations)}]", end="")
            result = process_examination(exam, DATABASE_PATH, output_dir, extractor)
            all_results.append(result)

    print(f"\n\n{'=' * 70}")
    print("处理完成!")
    print(f"{'=' * 70}")
    print(f"处理了 {len(all_results)} 个检查")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()