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
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from clinical_base import (
    LM, LandmarkExtractor, ActionResult,
    db_fetch_examinations, db_fetch_videos_for_exam, db_fetch_labels,
    compute_ear, extract_common_indicators,
)

from sunnybrook_scorer import (
    RestingSymmetry, VoluntaryMovement, VoluntaryMovementItem,
    Synkinesis, SynkinesisItem, SunnybrookScore,
    compute_resting_symmetry, compute_voluntary_score_from_ratio,
    compute_sunnybrook_composite, SUNNYBROOK_EXPRESSION_MAPPING
)

from thresholds import THR

# 导入动作模块
import neutral_face
import eye_blink
import close_eye
import smile
import show_teeth
import raise_eyebrow
import lip_pucker
import blow_cheek
import shrug_nose

# =============================================================================
# 配置参数
# =============================================================================

DATABASE_PATH = r"/Users/cuijinglei/PycharmProjects/medicalProject/facial_palsy/facialPalsy.db"
MEDIAPIPE_MODEL_PATH = r"/Users/cuijinglei/PycharmProjects/medicalProject/models/face_landmarker.task"
OUTPUT_DIR = r"/Users/cuijinglei/Documents/facialPalsy/HGFA/clinical_grading"
PATIENT_LIMIT = None
TARGET_EXAM_ID = None

# =============================================================================
# 调试筛选：只分析特定患者/特定检查（其余跳过）
# =============================================================================
# 1) 只跑指定患者（常用）
TARGET_PATIENT_IDS = []  # "XW000264", "XW000304", "XW000312"]

# 2) 只跑指定检查ID（优先级更高）
TARGET_EXAM_IDS = []

ENABLED_ACTIONS = [
    "NeutralFace",  # 基线（如果REUSE_BASELINE=False会自动添加）
    "ShowTeeth",
]

# 是否复用已有的 NeutralFace 结果（用于调试其他动作时跳过基线重算）
# True: 从已有的 indicators.json 加载基线
# False: 每次都重新运行 NeutralFace
REUSE_BASELINE = True

# 是否跳过已存在的动作结果（增量更新模式）
# True: 如果动作结果已存在，跳过该动作
# False: 总是重新处理所有指定的动作
SKIP_EXISTING_ACTIONS = False

# =============================================================================
# 并行配置（多CPU加速）
# =============================================================================
USE_MULTIPROCESS = True
CPU_N = os.cpu_count()
MAX_WORKERS = 5

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


def find_peak_frame_generic(landmarks_seq, frames_seq, w, h, action_name, baseline_landmarks=None):
    """通用峰值帧查找

    说明：
    - 优先调用各动作模块自带的峰值检测逻辑（与最新版动作代码保持一致）
    """
    if action_name == "NeutralFace":
        return neutral_face.find_peak_frame(landmarks_seq, frames_seq, w, h)
    elif action_name == "Smile":
        return smile.find_peak_frame_smile(landmarks_seq, frames_seq, w, h)
    elif action_name == "ShowTeeth":
        return show_teeth.find_peak_frame(landmarks_seq, frames_seq, w, h)
    elif action_name in ["VoluntaryEyeBlink", "SpontaneousEyeBlink"]:
        return eye_blink.find_peak_frame_blink(landmarks_seq, frames_seq, w, h)
    elif action_name in ["CloseEyeSoftly", "CloseEyeHardly"]:
        return close_eye.find_peak_frame_close_eye(landmarks_seq, frames_seq, w, h)
    elif action_name == "RaiseEyebrow":
        return raise_eyebrow.find_peak_frame(landmarks_seq, frames_seq, w, h, baseline_landmarks)
    elif action_name == "LipPucker":
        return lip_pucker.find_peak_frame(landmarks_seq, frames_seq, w, h)
    elif action_name == "BlowCheek":
        return blow_cheek.find_peak_frame(landmarks_seq, frames_seq, w, h)
    elif action_name == "ShrugNose":
        return shrug_nose.find_peak_frame(landmarks_seq, frames_seq, w, h, baseline_landmarks)
    else:
        # 默认: 使用NeutralFace的方法
        return neutral_face.find_peak_frame(landmarks_seq, frames_seq, w, h)


def process_action_generic(landmarks_seq, frames_seq, w, h, video_info, output_dir,
                           action_name, baseline_result=None, baseline_landmarks=None):
    """动作处理入口（与最新版动作代码对齐）

    所有动作全部调用各自模块的 process / process_xxx 函数
    """
    if action_name == "NeutralFace":
        return neutral_face.process(landmarks_seq, frames_seq, w, h, video_info, output_dir)

    # --- Smile ---
    elif action_name == "Smile":
        return smile.process(
            landmarks_seq, frames_seq, w, h, video_info, output_dir,
            baseline_result, baseline_landmarks
        )

    # --- ShowTeeth (独立模块) ---
    elif action_name == "ShowTeeth":
        return show_teeth.process(
            landmarks_seq, frames_seq, w, h, video_info, output_dir,
            baseline_result, baseline_landmarks
        )

    # --- Eye Blink ---
    elif action_name == "VoluntaryEyeBlink":
        return eye_blink.process_voluntary_blink(
            landmarks_seq, frames_seq, w, h, video_info, output_dir,
            baseline_result, baseline_landmarks
        )
    elif action_name == "SpontaneousEyeBlink":
        return eye_blink.process_spontaneous_blink(
            landmarks_seq, frames_seq, w, h, video_info, output_dir,
            baseline_result, baseline_landmarks
        )

    # --- Close Eye ---
    elif action_name == "CloseEyeSoftly":
        return close_eye.process_close_eye_softly(
            landmarks_seq, frames_seq, w, h, video_info, output_dir,
            baseline_result=baseline_result,
            baseline_landmarks=baseline_landmarks
        )
    elif action_name == "CloseEyeHardly":
        return close_eye.process_close_eye_hardly(
            landmarks_seq, frames_seq, w, h, video_info, output_dir,
            baseline_result=baseline_result,
            baseline_landmarks=baseline_landmarks
        )

    # --- Other Voluntary Movements ---
    elif action_name == "RaiseEyebrow":
        return raise_eyebrow.process(
            landmarks_seq, frames_seq, w, h, video_info, output_dir,
            baseline_result=baseline_result,
            baseline_landmarks=baseline_landmarks
        )
    elif action_name == "LipPucker":
        return lip_pucker.process(
            landmarks_seq, frames_seq, w, h, video_info, output_dir,
            baseline_result=baseline_result,
            baseline_landmarks=baseline_landmarks
        )
    elif action_name == "BlowCheek":
        return blow_cheek.process(
            landmarks_seq, frames_seq, w, h, video_info, output_dir,
            baseline_result=baseline_result,
            baseline_landmarks=baseline_landmarks
        )
    elif action_name == "ShrugNose":
        return shrug_nose.process(
            landmarks_seq, frames_seq, w, h, video_info, output_dir,
            baseline_result=baseline_result,
            baseline_landmarks=baseline_landmarks
        )

    # fallback
    return process_generic_action(
        landmarks_seq, frames_seq, w, h, video_info, output_dir,
        action_name, baseline_result, baseline_landmarks
    )


def process_generic_action(landmarks_seq, frames_seq, w, h, video_info, output_dir,
                           action_name, baseline_result=None, baseline_landmarks=None):
    """通用动作处理（用于没有专门模块的动作）"""
    if not landmarks_seq or not frames_seq:
        return None

    # 找峰值帧
    peak_idx = find_peak_frame_generic(landmarks_seq, frames_seq, w, h, action_name, baseline_landmarks)
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
    extract_common_indicators(peak_landmarks, w, h, result, baseline_landmarks)

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
    """
    基于各动作的ActionResult，计算Sunnybrook评分（严格按Sunnybrook 5个主动运动项 + 5个联动项）。

    说明：
    - Resting Symmetry：仅使用 NeutralFace
    - Voluntary Movement：Brow、GentleEyeClosure、OpenMouthSmile、Snarl、LipPucker（5项，Sum×4）
    - Synkinesis：同上5项（每项0-3，总分0-15）
    """
    if not action_results or "NeutralFace" not in action_results:
        return None

    # =========================
    # 1) Resting Symmetry
    # =========================
    neutral = action_results["NeutralFace"]

    # oral_angle_diff：用 oral_angle 的 AOE/BOF 计算
    oral = getattr(neutral, "oral_angle", None)
    aoe = float(getattr(oral, "AOE_angle", 0.0) or 0.0) if oral else 0.0
    bof = float(getattr(oral, "BOF_angle", 0.0) or 0.0) if oral else 0.0
    oral_angle_diff = abs(aoe - bof)

    resting = compute_resting_symmetry(
        palpebral_height_ratio=float(getattr(neutral, "palpebral_height_ratio", 1.0) or 1.0),
        nlf_ratio=float(getattr(neutral, "nlf_ratio", 1.0) or 1.0),
        oral_angle_diff=float(oral_angle_diff),
        aoe_angle=float(aoe),
        bof_angle=float(bof),
    )

    # =========================
    # 2) Voluntary Movement (5 items)
    # =========================
    vol_items: List[VoluntaryMovementItem] = []

    # Brow -> RaiseEyebrow
    brow_result = action_results.get("RaiseEyebrow")
    if brow_result:
        # 优先用"变化量"做对称性（更符合"运动幅度"）
        l = getattr(brow_result, "left_brow_eye_distance_change", None)
        r = getattr(brow_result, "right_brow_eye_distance_change", None)

        if l is not None and r is not None and (abs(l) > 1e-9 or abs(r) > 1e-9):
            ratio = (float(l) / float(r)) if abs(float(r)) > 1e-9 else (float("inf") if float(l) > 0 else 1.0)
        else:
            ratio = float(brow_result.brow_height_ratio or 1.0)

        score, interp = compute_voluntary_score_from_ratio(ratio)
        vol_items.append(VoluntaryMovementItem(
            expression="Brow", expression_cn="皱额/抬眉",
            left_value=float(getattr(brow_result, "left_brow_height", 0.0) or 0.0),
            right_value=float(getattr(brow_result, "right_brow_height", 0.0) or 0.0),
            ratio=float(ratio) if np.isfinite(ratio) else 999.0,
            score=int(brow_result.voluntary_movement_score or score),
            interpretation=str(interp or "")
        ))
    else:
        vol_items.append(VoluntaryMovementItem(
            expression="Brow", expression_cn="皱额/抬眉",
            left_value=0.0, right_value=0.0, ratio=1.0, score=5, interpretation="未评估"
        ))

    # Gentle eye closure -> CloseEyeSoftly
    gentle = action_results.get("CloseEyeSoftly")
    if gentle:
        ratio = float(gentle.eye_area_ratio or 1.0)
        score, interp = compute_voluntary_score_from_ratio(ratio)
        vol_items.append(VoluntaryMovementItem(
            expression="GentleEyeClosure", expression_cn="轻闭眼",
            left_value=float(gentle.left_eye_area or 0.0),
            right_value=float(gentle.right_eye_area or 0.0),
            ratio=float(ratio),
            score=int(gentle.voluntary_movement_score or score),
            interpretation=str(interp or "")
        ))
    else:
        vol_items.append(VoluntaryMovementItem(
            expression="GentleEyeClosure", expression_cn="轻闭眼",
            left_value=0.0, right_value=0.0, ratio=1.0, score=5, interpretation="未评估"
        ))

    # Open mouth smile -> 优先 ShowTeeth，其次 Smile
    smile_result = action_results.get("ShowTeeth") or action_results.get("Smile")
    if smile_result:
        oral = smile_result.oral_angle
        vol_items.append(VoluntaryMovementItem(
            expression="OpenMouthSmile", expression_cn="露齿微笑",
            left_value=float(oral.BOF_angle if oral else 0.0),
            right_value=float(oral.AOE_angle if oral else 0.0),
            ratio=1.0,  # 该项直接用动作评分（避免角度符号导致比值误导）
            score=int(smile_result.voluntary_movement_score or 5),
            interpretation=""
        ))
    else:
        vol_items.append(VoluntaryMovementItem(
            expression="OpenMouthSmile", expression_cn="露齿微笑",
            left_value=0.0, right_value=0.0, ratio=1.0, score=5, interpretation="未评估"
        ))

    # Snarl -> ShrugNose
    snarl = action_results.get("ShrugNose")
    if snarl:
        ratio = float(snarl.nlf_ratio or 1.0)
        score, interp = compute_voluntary_score_from_ratio(ratio)
        vol_items.append(VoluntaryMovementItem(
            expression="Snarl", expression_cn="皱鼻",
            left_value=float(snarl.left_nlf_length or 0.0),
            right_value=float(snarl.right_nlf_length or 0.0),
            ratio=float(ratio),
            score=int(snarl.voluntary_movement_score or score),
            interpretation=str(interp or "")
        ))
    else:
        vol_items.append(VoluntaryMovementItem(
            expression="Snarl", expression_cn="皱鼻",
            left_value=0.0, right_value=0.0, ratio=1.0, score=5, interpretation="未评估"
        ))

    # Lip pucker -> LipPucker
    pucker = action_results.get("LipPucker")
    if pucker:
        vol_items.append(VoluntaryMovementItem(
            expression="LipPucker", expression_cn="撅嘴",
            left_value=float(pucker.mouth_width or 0.0),
            right_value=float(pucker.mouth_width or 0.0),
            ratio=1.0,
            score=int(pucker.voluntary_movement_score or 5),
            interpretation=""
        ))
    else:
        vol_items.append(VoluntaryMovementItem(
            expression="LipPucker", expression_cn="撅嘴",
            left_value=0.0, right_value=0.0, ratio=1.0, score=5, interpretation="未评估"
        ))

    # VoluntaryMovement 需要 raw_sum 和 total_score
    vol_raw_sum = int(sum(int(it.score) for it in vol_items))
    vol_total_score = int(vol_raw_sum * 4)
    voluntary = VoluntaryMovement(items=vol_items, raw_sum=vol_raw_sum, total_score=vol_total_score)

    # =========================
    # 3) Synkinesis (严格5项，0-15)
    # =========================
    def _get_syn(action_name: str) -> Tuple[int, int, int, str]:
        r = action_results.get(action_name)
        if not r or not getattr(r, "synkinesis_scores", None):
            return 0, 0, 0, "未检测"
        eye = int(r.synkinesis_scores.get("eye_synkinesis", 0) or 0)
        mouth = int(r.synkinesis_scores.get("mouth_synkinesis", 0) or 0)
        # 单项总分按0-3：取更严重的联动（避免eye+mouth叠加>3）
        total = max(eye, mouth)
        interp = str(getattr(r, "synkinesis_interpretation", "") or f"eye={eye}, mouth={mouth}")
        return eye, mouth, total, interp

    syn_items: List[SynkinesisItem] = []
    syn_map = [
        ("Brow", "RaiseEyebrow"),
        ("GentleEyeClosure", "CloseEyeSoftly"),
        ("OpenMouthSmile", "ShowTeeth" if "ShowTeeth" in action_results else "Smile"),
        ("Snarl", "ShrugNose"),
        ("LipPucker", "LipPucker"),
    ]

    for expr, act in syn_map:
        cn = SUNNYBROOK_EXPRESSION_MAPPING.get(expr, {}).get("cn", expr)
        eye, mouth, total, interp = _get_syn(act)
        syn_items.append(SynkinesisItem(
            expression=expr,
            expression_cn=str(cn),
            eye_synkinesis=int(eye),
            mouth_synkinesis=int(mouth),
            total_score=int(total),
            interpretation=str(interp)
        ))

    syn_total_score = int(sum(int(it.total_score) for it in syn_items))
    synkinesis = Synkinesis(items=syn_items, total_score=syn_total_score)

    # =========================
    # 4) Composite
    # =========================
    score = compute_sunnybrook_composite(resting, voluntary, synkinesis)
    return score


def _palsy_side_to_text(side_code: Any) -> str:
    """0/1/2 -> 无/左/右（兼容None/空字符串）"""
    try:
        side_int = int(side_code)
    except Exception:
        side_int = 0
    return {0: "无", 1: "左", 2: "右"}.get(side_int, "无")


def infer_palsy_and_side(action_results: Dict[str, ActionResult]) -> Dict[str, Any]:
    """
    综合 11 个动作的"是否面瘫 + 患侧投票"。

    优先使用每个动作模块的 palsy_detection 结果（保存在 action_specific 中），
    这样可以确保报告和 indicators.json 的结果一致。

    返回结构直接给HTML使用：
    - has_palsy / palsy_side / confidence
    - left_score / right_score / votes / top_evidence
    """
    weights = {
        "SpontaneousEyeBlink": 1.0,
        "VoluntaryEyeBlink": 1.1,
        "CloseEyeSoftly": 1.4,
        "CloseEyeHardly": 1.6,
        "RaiseEyebrow": 1.0,
        "Smile": 1.3,
        "ShowTeeth": 1.3,
        "LipPucker": 1.0,
        "ShrugNose": 1.0,
        "BlowCheek": 1.1,
        "NeutralFace": 0.4,  # 静息只提示异常，不定向
    }

    def _clip01(x: float) -> float:
        return float(max(0.0, min(1.0, x)))

    def _vote_record(action: str, side: int, strength: float, region: str, reason: str, metric: Dict[str, Any] = None):
        return {
            "action": action,
            "side": int(side),  # 0=中立,1=左弱,2=右弱
            "side_text": _palsy_side_to_text(side) if side != 0 else "中立",
            "strength": float(strength),
            "weight": float(weights.get(action, 1.0)),
            "region": str(region),
            "reason": str(reason),
            "metric": metric or {}
        }

    votes: List[Dict[str, Any]] = []

    # ========== 优先使用各动作模块的 palsy_detection 结果 ==========
    for act_name, res in action_results.items():
        if not res or not res.action_specific:
            continue

        palsy_det = res.action_specific.get("palsy_detection", {})
        if not palsy_det:
            continue

        palsy_side = palsy_det.get("palsy_side", 0)
        confidence = palsy_det.get("confidence", 0.0)
        method = palsy_det.get("method", "")
        interpretation = palsy_det.get("interpretation", "")
        evidence = palsy_det.get("evidence", {})

        if palsy_side != 0 and confidence > 0.05:
            # 根据动作类型确定区域
            if act_name in ["SpontaneousEyeBlink", "VoluntaryEyeBlink", "CloseEyeSoftly", "CloseEyeHardly"]:
                region = "眼"
            elif act_name == "RaiseEyebrow":
                region = "额"
            elif act_name in ["Smile", "ShowTeeth"]:
                region = "口"
            elif act_name in ["ShrugNose", "BlowCheek", "LipPucker"]:
                region = "中面"
            elif act_name == "NeutralFace":
                region = "静息"
            else:
                region = "其他"

            votes.append(_vote_record(
                act_name, palsy_side, confidence, region,
                f"{method}: {interpretation}",
                evidence
            ))

    # ========== 汇总 ==========
    left_score = 0.0
    right_score = 0.0
    for v in votes:
        w = float(v["weight"])
        s = float(v["strength"])
        if v["side"] == 1:
            left_score += w * s
        elif v["side"] == 2:
            right_score += w * s

    total = left_score + right_score
    if total < 0.3:
        has_palsy = False
        palsy_side = 0
        palsy_side_text = "无"
        confidence = 1.0 - total
    else:
        has_palsy = True
        if left_score > right_score * 1.2:
            palsy_side = 1
            palsy_side_text = "左"
        elif right_score > left_score * 1.2:
            palsy_side = 2
            palsy_side_text = "右"
        else:
            palsy_side = 0
            palsy_side_text = "不确定"
        confidence = _clip01(abs(left_score - right_score) / max(total, 1e-9))

    # 排序证据
    votes_sorted = sorted(votes, key=lambda x: float(x["weight"]) * float(x["strength"]), reverse=True)
    top_evidence = votes_sorted[:5]

    return {
        "has_palsy": has_palsy,
        "palsy_side": palsy_side,
        "palsy_side_text": palsy_side_text,
        "confidence": confidence,
        "left_score": left_score,
        "right_score": right_score,
        "votes": votes,
        "top_evidence": top_evidence,
    }


def generate_html_report(exam_id: str, patient_id: str,
                         action_results: Dict[str, ActionResult],
                         sunnybrook: Optional[SunnybrookScore],
                         ground_truth: Dict[str, Any],
                         prediction: Optional[Dict[str, Any]],
                         output_dir: Path) -> None:
    """生成详细HTML报告（含：Sunnybrook + 11动作综合投票与证据叠加图）"""

    action_name_map = {
        "NeutralFace": "静息面",
        "SpontaneousEyeBlink": "自然眨眼",
        "VoluntaryEyeBlink": "自主眨眼",
        "CloseEyeSoftly": "轻闭眼",
        "CloseEyeHardly": "用力闭眼",
        "RaiseEyebrow": "皱额/抬眉",
        "Smile": "微笑",
        "ShrugNose": "皱鼻",
        "ShowTeeth": "露齿",
        "BlowCheek": "鼓腮",
        "LipPucker": "撅嘴",
    }

    open_mouth_used = "ShowTeeth" if "ShowTeeth" in action_results else "Smile"
    voluntary_used_effective = {"RaiseEyebrow", "CloseEyeSoftly", open_mouth_used, "ShrugNose", "LipPucker"}
    syn_used_effective = voluntary_used_effective.copy()

    action_focus = {
        "NeutralFace": "静息对称性与基线：睑裂、鼻唇沟、口角下垂/偏斜等。",
        "SpontaneousEyeBlink": "自然眨眼是否完整/对称；是否伴随口部联动。",
        "VoluntaryEyeBlink": "自主眨眼启动能力与闭合幅度对称性；联动表现。",
        "CloseEyeSoftly": "轻闭眼闭合不全（滞睑/轻度无力）最敏感。",
        "CloseEyeHardly": "用力闭眼反映眼轮匝肌力量，常用于区分中重度。",
        "RaiseEyebrow": "额肌功能：抬眉幅度左右差异，反映上面部运动。",
        "Smile": "口角牵拉/上抬幅度左右差异，反映下/中面部运动。",
        "ShrugNose": "鼻翼/鼻唇沟牵拉幅度左右差异，反映中面部运动与联动。",
        "ShowTeeth": "露齿微笑（Sunnybrook的OpenMouthSmile）：口角牵拉与上唇提升。",
        "BlowCheek": "闭唇与鼓腮充气能力（漏气/一侧塌陷）；辅助下脸评估。",
        "LipPucker": "口轮匝肌收缩（撅嘴）对称性；口角偏斜/下垂可作为弱证据。",
    }

    vote_by_action: Dict[str, Dict[str, Any]] = {}
    if prediction and isinstance(prediction.get("votes"), list):
        for v in prediction["votes"]:
            act = v.get("action")
            if not act:
                continue
            score = float(v.get("weight", 1.0)) * float(v.get("strength", 0.0))
            if act not in vote_by_action or score > float(vote_by_action[act].get("_score", -1.0)):
                v2 = dict(v)
                v2["_score"] = score
                vote_by_action[act] = v2

    gt_has = "是" if int(ground_truth.get("has_palsy", 0) or 0) == 1 else "否"
    gt_side_code = ground_truth.get("palsy_side", 0)
    gt_side_text = _palsy_side_to_text(gt_side_code)

    pred_has = "—"
    pred_side_text = "—"
    pred_conf = "—"
    pred_left = 0.0
    pred_right = 0.0
    pred_top = []
    if prediction:
        pred_has = "是" if prediction.get("has_palsy") else "否"
        pred_side_text = prediction.get("palsy_side_text", "无")
        pred_conf = f"{float(prediction.get('confidence', 0.0)):.2f}"
        pred_left = float(prediction.get("left_score", 0.0))
        pred_right = float(prediction.get("right_score", 0.0))
        pred_top = prediction.get("top_evidence", []) or []

    def _collect_extra_synkinesis():
        extras = []
        for act, r in action_results.items():
            if act == "NeutralFace":
                continue
            if act in syn_used_effective:
                continue
            if not getattr(r, "synkinesis_scores", None):
                continue
            eye = int(r.synkinesis_scores.get("eye_synkinesis", 0) or 0)
            mouth = int(r.synkinesis_scores.get("mouth_synkinesis", 0) or 0)
            total = max(eye, mouth)
            if total <= 0:
                continue
            extras.append((act, eye, mouth, total))
        extras.sort(key=lambda x: x[3], reverse=True)
        return extras

    extra_syn = _collect_extra_synkinesis()

    def _bar(value: float, max_value: float) -> str:
        v = max(0.0, float(value))
        mv = max(1e-9, float(max_value))
        pct = max(0.0, min(100.0, 100.0 * v / mv))
        return f'<div class="bar"><div class="barfill" style="width:{pct:.1f}%"></div></div>'

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>面部指标分析报告 - {exam_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1600px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .summary {{ background: #ecf0f1; padding: 15px; border-radius: 8px; margin: 15px 0; }}
        .metric-box {{ display: inline-block; background: #3498db; color: white; padding: 10px 15px; border-radius: 6px; margin: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        th {{ background: #3498db; color: white; }}
        .action-section {{ background: #fafafa; border: 1px solid #e0e0e0; border-radius: 8px; padding: 15px; margin: 15px 0; }}
        .images {{ display: flex; gap: 10px; flex-wrap: wrap; justify-content: flex-start; }}
        .images img {{ max-width: 480px; border-radius: 6px; border: 1px solid #ddd; }}
        .tip {{ background: #fff8e1; border-left: 5px solid #f1c40f; padding: 10px; margin: 10px 0; }}
        .bar {{ width: 260px; height: 12px; background: #e5e7eb; border-radius: 10px; overflow: hidden; display: inline-block; vertical-align: middle; }}
        .barfill {{ height: 100%; background: #e74c3c; }}
        .small {{ font-size: 12px; color: #555; }}
        .tag {{ display:inline-block; padding:2px 8px; border-radius: 10px; background:#eef2ff; margin-left:6px; font-size: 12px; }}
    </style>
</head>
<body>
<div class="container">
    <h1>🏥 面部指标分析报告</h1>

    <div class="summary">
        <div><b>检查ID:</b> {exam_id}</div>
        <div><b>患者ID:</b> {patient_id}</div>
        <div><b>分析时间:</b> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
        <div><b>处理动作数:</b> {len(action_results)}</div>
    </div>

    <h2>📋 医生标注 (Ground Truth)</h2>
    <div class="summary">
        <div class="metric-box">面瘫: {gt_has}</div>
        <div class="metric-box">患侧: {gt_side_code} ({gt_side_text})</div>
        <div class="metric-box">HB分级: {ground_truth.get('hb_grade', '—')}</div>
        <div class="metric-box">Sunnybrook评分: {ground_truth.get('sunnybrook_score', '—')}</div>
    </div>

    <h2>🧠 综合判定（11动作投票 + 证据叠加图）</h2>
    <div class="summary">
        <div class="metric-box">预测面瘫: {pred_has}</div>
        <div class="metric-box">预测患侧: {pred_side_text}</div>
        <div class="metric-box">置信度: {pred_conf}</div>
        <div style="margin-top:10px;">
            <div><b>左侧累计证据:</b> {pred_left:.2f} {_bar(pred_left, max(pred_left, pred_right, 1.0))}</div>
            <div><b>右侧累计证据:</b> {pred_right:.2f} {_bar(pred_right, max(pred_left, pred_right, 1.0))}</div>
            <div class="small">说明：证据来自10个运动动作的"方向+强度+权重"投票；静息仅用于异常提示，不强行定向。</div>
        </div>
    </div>

    <div class="tip">
        <b>Top 证据（按权重×强度排序）</b><br/>
        {"".join([f"• {action_name_map.get(e.get('action', ''), e.get('action', ''))}：{e.get('side_text', '')}（{e.get('region', '')}）— {e.get('reason', '')}<br/>" for e in pred_top]) if pred_top else "暂无（未提供综合投票结果）"}
    </div>

    <h2>🧾 动作清单与用途</h2>
    <table>
        <tr><th>动作</th><th>中文</th><th>参与Resting</th><th>参与Voluntary(5项)</th><th>参与Synkinesis(5项)</th><th>关注点</th></tr>
        {"".join([
        f"<tr>"
        f"<td>{a}</td>"
        f"<td>{action_name_map.get(a, a)}</td>"
        f"<td>{'✓' if a == 'NeutralFace' else ''}</td>"
        f"<td>{'✓' if a in voluntary_used_effective else ''}</td>"
        f"<td>{'✓' if a in syn_used_effective else ''}</td>"
        f"<td style='text-align:left'>{action_focus.get(a, '')}</td>"
        f"</tr>"
        for a in action_name_map.keys()
    ])}
    </table>

    <div class="tip small">
        Sunnybrook正式统计只使用：Resting(静息1项) + Voluntary(5项) + Synkinesis(5项)。
        本报告会把其余录制动作（眨眼/用力闭眼/鼓腮等）全部展示出来，但会明确标注"未参与Sunnybrook计分"。
    </div>

    <h2>📊 Sunnybrook 面神经分级评分</h2>
"""

    if sunnybrook:
        html += f"""
    <div class="summary">
        <div class="metric-box">Resting Symmetry: {sunnybrook.resting_score}</div>
        <div class="metric-box">Voluntary Movement: {sunnybrook.voluntary_score}</div>
        <div class="metric-box">Synkinesis: {sunnybrook.synkinesis_score}</div>
        <div class="metric-box">Composite: {sunnybrook.composite_score}</div>
        <div class="metric-box">Grade {sunnybrook.grade}: {sunnybrook.grade_description}</div>
        <div class="small">公式: Composite = Voluntary({sunnybrook.voluntary_score}) - Resting({sunnybrook.resting_score}) - Synkinesis({sunnybrook.synkinesis_score}) = {sunnybrook.composite_score}</div>
    </div>

    <h2>1️⃣ Resting Symmetry (静息对称性)</h2>
    <table>
        <tr><th>部位</th><th>状态</th><th>测量值</th><th>评分</th><th>说明</th></tr>
        {"".join([
            f"<tr>"
            f"<td>{it.region_cn}</td>"
            f"<td>{it.status_cn}</td>"
            f"<td>{(f'{it.measurement:.1f}°' if it.region == 'Mouth' else f'{it.measurement:.3f}')}</td>"
            f"<td class='score-{it.score}'>{it.score}</td>"
            f"<td>{it.threshold_info}</td>"
            f"</tr>"
            for it in (
                sunnybrook.resting_symmetry.eye,
                sunnybrook.resting_symmetry.cheek,
                sunnybrook.resting_symmetry.mouth
            )
        ])}
        <tr>
          <td colspan="5">
            <b>Total (Raw Score × 5):</b>
            {sunnybrook.resting_symmetry.raw_score} × 5 = {sunnybrook.resting_symmetry.total_score}
            （判断患侧: {sunnybrook.resting_symmetry.affected_side}）
          </td>
        </tr>
    </table>

    <h2>2️⃣ Symmetry of Voluntary Movement (主动运动对称性)</h2>
    <div class="tip small">
        计分只使用5个动作：抬眉、轻闭眼、露齿微笑（ShowTeeth优先）、皱鼻、撅嘴。其余运动动作（眨眼、用力闭眼、鼓腮等）仅展示，不参与Sunnybrook分数。
    </div>
    <table>
        <tr><th>表情</th><th>对应动作</th><th>左侧测量</th><th>右侧测量</th><th>比值</th><th>评分 (1-5)</th></tr>
        {"".join([f"<tr><td>{it.expression_cn}</td><td>{it.expression}</td><td>{it.left_value:.3f}</td><td>{it.right_value:.3f}</td><td>{it.ratio:.3f}</td><td>{it.score}</td></tr>" for it in sunnybrook.voluntary_movement.items])}
        <tr><td colspan="6"><b>Total (Sum × 4):</b> {sunnybrook.voluntary_movement.raw_sum} × 4 = {sunnybrook.voluntary_movement.total_score}</td></tr>
    </table>

    <h2>3️⃣ Synkinesis (联动运动)</h2>
    <div class="tip small">
        Sunnybrook正式联动分数只统计5项（与Voluntary相同）。下面会额外展示其它动作的联动检测结果，但不计入0-15总分。
    </div>
    <table>
        <tr><th>表情</th><th>眼部联动</th><th>嘴部联动</th><th>单项总分(0-3)</th></tr>
        {"".join([f"<tr><td>{it.expression_cn}</td><td>{it.eye_synkinesis}</td><td>{it.mouth_synkinesis}</td><td>{it.total_score}</td></tr>" for it in sunnybrook.synkinesis.items])}
        <tr><td colspan="4"><b>Total:</b> {sunnybrook.synkinesis_score} (0-15)</td></tr>
    </table>
"""
        if extra_syn:
            html += """
    <h3>扩展联动结果（不计入Sunnybrook）</h3>
    <table>
        <tr><th>动作</th><th>眼部联动</th><th>嘴部联动</th><th>单项总分(0-3)</th></tr>
"""
            for act, eye, mouth, total in extra_syn:
                html += f"<tr><td>{action_name_map.get(act, act)}</td><td>{eye}</td><td>{mouth}</td><td>{total}</td></tr>"
            html += "</table>"
    else:
        html += '<div class="tip">未能计算Sunnybrook评分（缺少NeutralFace或关键动作结果）。</div>'

    html += "<h2>📹 各动作详细分析</h2>"

    action_order = [
        "NeutralFace",
        "SpontaneousEyeBlink",
        "VoluntaryEyeBlink",
        "CloseEyeSoftly",
        "CloseEyeHardly",
        "RaiseEyebrow",
        "Smile",
        "ShrugNose",
        "ShowTeeth",
        "BlowCheek",
        "LipPucker",
    ]
    for action_name in action_order:
        if action_name not in action_results:
            continue
        result = action_results[action_name]
        cn = action_name_map.get(action_name, action_name)
        action_dir = output_dir / action_name

        v = vote_by_action.get(action_name)
        if v:
            vote_line = (
                f"本动作投票：{v.get('side_text', '中立')} <span class='tag'>{v.get('region', '')}</span> "
                f"强度={float(v.get('strength', 0.0)):.2f} 权重={float(v.get('weight', 1.0)):.2f}<br/>"
                f"<span class='small'>{v.get('reason', '')}</span>"
            )
        else:
            vote_line = "本动作投票：—（无/中立）"

        def _img_tag(rel_path: str, alt: str) -> str:
            p = action_dir / rel_path
            if p.exists():
                return f'<img src="{action_name}/{rel_path}" alt="{alt}"/>'
            return ""

        raw_img = _img_tag("peak_raw.jpg", "原始帧")
        ind_img = _img_tag("peak_indicators.jpg", "指标可视化")
        rest_img = _img_tag("resting_symmetry.jpg", "Resting Symmetry")
        ear_curve = _img_tag("ear_curve.png", "EAR曲线")
        eye_curve = _img_tag("eye_curve.png", "眼睛曲线")
        cheek_curve = _img_tag("cheek_curve.png", "鼓腮曲线")
        brow_curve = _img_tag("brow_curve.png", "眉眼距曲线")

        oral_asym = result.oral_angle.angle_asymmetry if result.oral_angle else 0.0

        html += f"""
    <div class="action-section">
        <h3>{action_name} - {cn}</h3>
        <div class="tip">{vote_line}</div>
        <table>
            <tr><th>指标</th><th>数值</th></tr>
            <tr><td>峰值帧</td><td>{result.peak_frame_idx} / {result.total_frames}</td></tr>
            <tr><td>ICD</td><td>{(result.icd or 0.0):.1f}px</td></tr>
            <tr><td>EAR Left / Right</td><td>{(result.left_ear or 0.0):.4f} / {(result.right_ear or 0.0):.4f}</td></tr>
            <tr><td>Eye Area Left / Right</td><td>{(result.left_eye_area or 0.0):.1f}px² / {(result.right_eye_area or 0.0):.1f}px²</td></tr>
            <tr><td>Eye Area Ratio</td><td>{(result.eye_area_ratio or 0.0):.3f}</td></tr>
            <tr><td>Palpebral H Ratio</td><td>{(result.palpebral_height_ratio or 0.0):.3f}</td></tr>
            <tr><td>Brow H Ratio</td><td>{(result.brow_height_ratio or 0.0):.3f}</td></tr>
            <tr><td>Mouth Width</td><td>{(result.mouth_width or 0.0):.1f}px</td></tr>
            <tr><td>NLF Ratio</td><td>{(result.nlf_ratio or 0.0):.3f}</td></tr>
            <tr><td>AOE/BOF (Right/Left)</td><td>{(result.oral_angle.AOE_angle if result.oral_angle else 0.0):+.2f}° / {(result.oral_angle.BOF_angle if result.oral_angle else 0.0):+.2f}°</td></tr>
            <tr><td>Oral Asymmetry</td><td>{oral_asym:.2f}°</td></tr>
            <tr><td>Voluntary Score</td><td>{result.voluntary_movement_score or 0}/5 {"<span class='tag'>计分动作</span>" if action_name in voluntary_used_effective else "<span class='tag'>展示</span>"}</td></tr>
        </table>

        <div class="images">
            {raw_img}
            {ind_img}
            {brow_curve}
            {ear_curve}
            {eye_curve}
            {cheek_curve}
            {rest_img}
        </div>
    </div>
"""

    html += """
</div>
</body>
</html>
"""

    report_path = output_dir / "report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"[OK] HTML报告已生成: {report_path}")


def load_existing_baseline(exam_output_dir: Path) -> Tuple[Optional[ActionResult], Optional[Any]]:
    """
    从已有的 NeutralFace 结果加载基线

    Returns:
        (baseline_result, baseline_landmarks) 或 (None, None)
    """
    neutral_dir = exam_output_dir / "NeutralFace"
    indicators_path = neutral_dir / "indicators.json"

    if not indicators_path.exists():
        print(f"    [!] 未找到已有基线: {indicators_path}")
        return None, None

    try:
        with open(indicators_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 重建 ActionResult
        result = ActionResult(
            action_name="NeutralFace",
            action_name_cn="静息面",
            video_path=data.get("video_path", ""),
            total_frames=data.get("total_frames", 0),
            peak_frame_idx=data.get("peak_frame_idx", 0),
            image_size=tuple(data.get("image_size", (0, 0))),
            fps=data.get("fps", 30.0)
        )

        # 恢复关键属性
        for key in ["left_ear", "right_ear", "left_eye_area", "right_eye_area",
                    "mouth_width", "mouth_height", "left_brow_height", "right_brow_height",
                    "left_nlf_length", "right_nlf_length", "brow_height_ratio",
                    "eye_area_ratio", "nlf_ratio", "voluntary_movement_score"]:
            if key in data:
                setattr(result, key, data[key])

        print(f"    [OK] 复用已有基线: {indicators_path}")

        # 注意：baseline_landmarks 无法从 JSON 恢复
        # 如果需要 baseline_landmarks，必须重新处理 NeutralFace
        return result, None

    except Exception as e:
        print(f"    [!] 加载基线失败: {e}")
        return None, None


def should_process_action(action_name: str, enabled_actions: Optional[List[str]]) -> bool:
    """判断是否应该处理该动作"""
    if enabled_actions is None or len(enabled_actions) == 0:
        return True  # 空列表表示处理所有动作
    return action_name in enabled_actions


def action_result_exists(exam_output_dir: Path, action_name: str) -> bool:
    """检查动作结果是否已存在"""
    indicators_path = exam_output_dir / action_name / "indicators.json"
    return indicators_path.exists()


# =============================================================================
# 主处理函数
# =============================================================================

def process_examination(examination: Dict[str, Any], db_path: str,
                        output_dir: Path, extractor: LandmarkExtractor,
                        enabled_actions: Optional[List[str]] = None,
                        reuse_baseline: bool = False,
                        skip_existing: bool = False) -> Dict[str, Any]:
    """
    处理单个检查

    Args:
        enabled_actions: 要处理的动作列表，None 或 [] 表示全部
        reuse_baseline: 是否复用已有的 NeutralFace 结果
        skip_existing: 是否跳过已存在的动作结果
    """
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
    need_neutral = (
            not reuse_baseline or
            should_process_action("NeutralFace", enabled_actions) or
            not action_result_exists(exam_output_dir, "NeutralFace")
    )

    if "NeutralFace" in videos and need_neutral:
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

    elif reuse_baseline:
        # 尝试复用已有基线
        print(f"\n  尝试复用已有基线...")
        baseline_result, baseline_landmarks = load_existing_baseline(exam_output_dir)
        if baseline_result:
            action_results["NeutralFace"] = baseline_result
        else:
            # 复用失败，需要重新处理
            if "NeutralFace" in videos:
                print(f"  复用失败，重新处理 NeutralFace...")
                video_info = videos["NeutralFace"]
                video_path = video_info["file_path"]

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
                            peak_idx = result.peak_frame_idx
                            baseline_landmarks = landmarks_seq[peak_idx]

    # ========== 处理其他动作（移到 if-elif 块外面！）==========
    for action_name, video_info in videos.items():
        if action_name == "NeutralFace":
            continue

        # 检查是否应该处理该动作
        if not should_process_action(action_name, enabled_actions):
            print(f"\n  跳过动作 (未启用): {action_name}")
            continue

        # 检查是否跳过已存在的结果
        if skip_existing and action_result_exists(exam_output_dir, action_name):
            print(f"\n  跳过动作 (结果已存在): {action_name}")
            # 尝试加载已有结果
            try:
                with open(exam_output_dir / action_name / "indicators.json", 'r') as f:
                    existing_data = json.load(f)
                # 简单创建一个占位结果（用于报告生成）
                existing_result = ActionResult(
                    action_name=action_name,
                    action_name_cn=existing_data.get("action_name_cn", action_name),
                    video_path=existing_data.get("video_path", ""),
                    total_frames=existing_data.get("total_frames", 0),
                    peak_frame_idx=existing_data.get("peak_frame_idx", 0),
                    image_size=tuple(existing_data.get("image_size", (0, 0))),
                    fps=existing_data.get("fps", 30.0)
                )
                action_results[action_name] = existing_result
            except:
                pass
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

    prediction = infer_palsy_and_side(action_results)

    # summary 里也建议存一份
    summary = {
        "exam_id": exam_id,
        "patient_id": patient_id,
        "analysis_time": datetime.now().isoformat(),
        "ground_truth": labels,
        "sunnybrook": sunnybrook.to_dict() if sunnybrook else None,
        "actions": {name: result.to_dict() for name, result in action_results.items()},
        "prediction": prediction,
    }

    generate_html_report(
        exam_id, patient_id,
        action_results,
        sunnybrook,
        labels,
        prediction,
        exam_output_dir
    )

    if sunnybrook:
        summary["sunnybrook"] = sunnybrook.to_dict()

    with open(exam_output_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def _process_exam_worker(args):
    """
    子进程执行单个检查的处理。
    """
    exam, db_path, output_dir_str, model_path, enabled_actions, reuse_baseline, skip_existing = args
    output_dir = Path(output_dir_str)

    with LandmarkExtractor(model_path) as extractor:
        return process_examination(
            exam, db_path, output_dir, extractor,
            enabled_actions=enabled_actions,
            reuse_baseline=reuse_baseline,
            skip_existing=skip_existing
        )


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
    # ===== 调试过滤：只跑指定 exam / patient =====
    if TARGET_EXAM_IDS:
        allow = set(TARGET_EXAM_IDS)
        before = len(examinations)
        examinations = [e for e in examinations if e.get("examination_id") in allow]
        print(f"[DEBUG] 仅分析指定检查ID：{sorted(allow)} | {before} -> {len(examinations)}")
    elif TARGET_PATIENT_IDS:
        allow = set(TARGET_PATIENT_IDS)
        before = len(examinations)
        examinations = [e for e in examinations if e.get("patient_id") in allow]
        print(f"[DEBUG] 仅分析指定患者：{sorted(allow)} | {before} -> {len(examinations)}")

    print(f"找到 {len(examinations)} 个检查记录")

    if not examinations:
        print("[ERROR] 没有有效的检查记录")
        return

    print(f"\n初始化MediaPipe...")

    all_results = [None] * len(examinations)

    if USE_MULTIPROCESS and len(examinations) > 1:
        print(f"\n启用多进程并行: workers={MAX_WORKERS}, exams={len(examinations)}")

        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        tasks = [(exam, DATABASE_PATH, str(output_dir), MEDIAPIPE_MODEL_PATH, ENABLED_ACTIONS, REUSE_BASELINE,
                  SKIP_EXISTING_ACTIONS) for exam in examinations]

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
            future_map = {pool.submit(_process_exam_worker, tasks[i]): i for i in range(len(tasks))}

            done = 0
            for fut in as_completed(future_map):
                idx = future_map[fut]
                done += 1
                try:
                    res = fut.result()
                    all_results[idx] = res
                    print(
                        f"\n[{done}/{len(examinations)}] 完成: {examinations[idx].get('examination_id', 'unknown') if isinstance(examinations[idx], dict) else 'exam'}")
                except Exception as e:
                    # 不中断全局：记录错误继续跑其他检查
                    print(f"\n[{done}/{len(examinations)}] 失败: idx={idx}, err={e}")
                    all_results[idx] = {
                        "error": str(e),
                        "exam": examinations[idx] if isinstance(examinations[idx],
                                                                (str, int, dict, list, tuple)) else "unserializable"
                    }

        all_results = [r for r in all_results if r is not None]

    else:
        # 只有1个检查时，多进程收益不大，避免额外开销
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