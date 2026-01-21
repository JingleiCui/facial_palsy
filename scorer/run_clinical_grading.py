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
    compute_ear, extract_common_indicators, draw_polygon,
    make_json_serializable,
)

from sunnybrook_scorer import (
    RestingSymmetry, VoluntaryMovement, VoluntaryMovementItem,
    Synkinesis, SynkinesisItem, SunnybrookScore,
    compute_resting_symmetry, compute_voluntary_score_from_ratio,
    compute_sunnybrook_composite, SUNNYBROOK_EXPRESSION_MAPPING
)
from session_diagnosis import compute_session_diagnosis
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
TARGET_PATIENT_IDS = []
# 2) 只跑指定检查ID（优先级更高）
TARGET_EXAM_IDS = []

ENABLED_ACTIONS = [
"NeutralFace",
"ShrugNose",
]

# ENABLED_ACTIONS = [
#     "NeutralFace",
#     "CloseEyeSoftly",
#     "CloseEyeHardly",
#     "VoluntaryEyeBlink",
#     "SpontaneousEyeBlink",
#     "RaiseEyebrow",
#     "Smile",
#     "ShrugNose",
#     "ShowTeeth",
#     "BlowCheek",
#     "LipPucker",
# ]

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
MAX_WORKERS = 6

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


def process_action_generic(landmarks_seq, frames_seq, w, h, video_info, output_dir,
                           action_name, baseline_result=None, baseline_landmarks=None):
    """
    动作处理的统一入口。
    根据 action_name 分发到各自独立的、规范化的处理模块。
    """
    # 动作处理器映射
    PROCESSOR_MAP = {
        "NeutralFace": neutral_face.process,
        "Smile": smile.process,
        "ShowTeeth": show_teeth.process,
        "RaiseEyebrow": raise_eyebrow.process,
        "CloseEyeSoftly": close_eye.process_close_eye_softly,
        "CloseEyeHardly": close_eye.process_close_eye_hardly,
        "VoluntaryEyeBlink": eye_blink.process_voluntary_blink,
        "SpontaneousEyeBlink": eye_blink.process_spontaneous_blink,
        "LipPucker": lip_pucker.process,
        "BlowCheek": blow_cheek.process,
        "ShrugNose": shrug_nose.process,
    }

    processor = PROCESSOR_MAP.get(action_name)
    if not processor:
        print(f"    [!] 未找到动作 '{action_name}' 的处理器，跳过。")
        return None

    # 调用相应的 process 函数
    return processor(
        landmarks_seq, frames_seq, w, h, video_info, output_dir,
        baseline_result=baseline_result,
        baseline_landmarks=baseline_landmarks
    )


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


def compute_full_diagnosis(
        action_results,
        sunnybrook_score_obj
):
    """
    计算完整的Session级诊断

    这是对 infer_palsy_and_side 的替换，整合了:
    - 患侧投票
    - Sunnybrook评分
    - HB分级推导
    - 一致性检查

    Returns:
        SessionDiagnosis 对象
    """
    return compute_session_diagnosis(
        action_results=action_results,
        sunnybrook_score_obj=sunnybrook_score_obj
    )


def generate_html_report(exam_id: str, patient_id: str,
                         action_results: Dict[str, ActionResult],
                         sunnybrook: Optional[SunnybrookScore],
                         ground_truth: Dict[str, Any],
                         diagnosis,
                         output_dir: Path) -> None:
    """生成详细HTML报告（含：Sunnybrook + 11动作综合投票与证据叠加图）"""

    action_name_map = {
        "NeutralFace": "静息",
        "SpontaneousEyeBlink": "自然眨眼",
        "VoluntaryEyeBlink": "自主眨眼",
        "CloseEyeSoftly": "轻轻闭眼",
        "CloseEyeHardly": "用力闭眼",
        "RaiseEyebrow": "皱额/抬眉",
        "Smile": "微笑",
        "ShrugNose": "皱鼻",
        "ShowTeeth": "露齿",
        "BlowCheek": "鼓腮",
        "LipPucker": "撅嘴",
    }

    # Ground Truth
    gt_has = "是" if int(ground_truth.get("has_palsy", 0) or 0) == 1 else "否"
    gt_side_code = ground_truth.get("palsy_side", 0)
    gt_side_text = {0: "无", 1: "左", 2: "右"}.get(gt_side_code, "无")
    gt_hb = ground_truth.get('hb_grade', '—')
    gt_sb = ground_truth.get('sunnybrook_score', '—')

    # Prediction (from SessionDiagnosis)
    if diagnosis:
        pred_has = "是" if diagnosis.has_palsy else "否"
        pred_side = diagnosis.palsy_side
        pred_side_text = diagnosis.palsy_side_text
        pred_hb = diagnosis.hb_grade
        pred_hb_desc = diagnosis.hb_description
        pred_sb = diagnosis.sunnybrook_score
        pred_conf = diagnosis.confidence
        pred_left = diagnosis.left_score
        pred_right = diagnosis.right_score
        top_evidence = diagnosis.top_evidence
        checks = diagnosis.consistency_checks
        adjustments = diagnosis.adjustments_made
        interpretation = diagnosis.interpretation
    else:
        pred_has = "—"
        pred_side_text = "—"
        pred_hb = "—"
        pred_hb_desc = ""
        pred_sb = "—"
        pred_conf = 0
        pred_left = 0
        pred_right = 0
        top_evidence = []
        checks = []
        adjustments = []
        interpretation = ""

    # 比较结果
    def _match_badge(pred, gt, label):
        if pred == "—" or gt == "—":
            return f'<span class="badge badge-gray">{label}: ?</span>'
        elif str(pred) == str(gt):
            return f'<span class="badge badge-green">{label}: ✓</span>'
        else:
            return f'<span class="badge badge-red">{label}: ✗</span>'

    match_palsy = _match_badge(pred_has, gt_has, "面瘫")
    match_side = _match_badge(pred_side_text, gt_side_text, "患侧")
    match_hb = _match_badge(str(pred_hb) if pred_hb != "—" else "—", str(gt_hb), "HB")

    # 进度条辅助函数
    def _bar(value, max_value, color="#e74c3c"):
        v = max(0.0, float(value))
        mv = max(1e-9, float(max_value))
        pct = max(0.0, min(100.0, 100.0 * v / mv))
        return f'<div class="bar"><div class="barfill" style="width:{pct:.1f}%; background:{color}"></div></div>'

    html = f"""<!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>面部指标分析报告 - {exam_id}</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f0f2f5; }}
            .container {{ max-width: 1600px; margin: 0 auto; background: white; padding: 25px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #1a365d; border-bottom: 3px solid #3182ce; padding-bottom: 12px; }}
            h2 {{ color: #2c5282; margin-top: 35px; border-left: 4px solid #3182ce; padding-left: 12px; }}
            h3 {{ color: #4a5568; }}

            /* 诊断卡片 */
            .diagnosis-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 12px; padding: 25px; margin: 20px 0; }}
            .diagnosis-card h2 {{ color: white; border: none; margin-top: 0; }}
            .diagnosis-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin-top: 20px; }}
            .metric-card {{ background: rgba(255,255,255,0.15); border-radius: 10px; padding: 18px; text-align: center; backdrop-filter: blur(5px); }}
            .metric-label {{ font-size: 13px; opacity: 0.9; margin-bottom: 8px; }}
            .metric-value {{ font-size: 28px; font-weight: bold; }}
            .metric-sub {{ font-size: 12px; opacity: 0.8; margin-top: 5px; }}

            /* 比较区域 */
            .comparison {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
            .compare-box {{ background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 20px; }}
            .compare-box h3 {{ margin-top: 0; color: #2d3748; }}
            .compare-box.ground-truth {{ border-left: 4px solid #38a169; }}
            .compare-box.prediction {{ border-left: 4px solid #3182ce; }}

            /* 徽章 */
            .badge {{ display: inline-block; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: bold; margin: 3px; }}
            .badge-green {{ background: #c6f6d5; color: #22543d; }}
            .badge-red {{ background: #fed7d7; color: #822727; }}
            .badge-gray {{ background: #e2e8f0; color: #4a5568; }}
            .badge-blue {{ background: #bee3f8; color: #2a4365; }}
            .badge-yellow {{ background: #fefcbf; color: #744210; }}

            /* 证据条 */
            .bar {{ width: 200px; height: 12px; background: #e2e8f0; border-radius: 6px; overflow: hidden; display: inline-block; vertical-align: middle; }}
            .barfill {{ height: 100%; background: #e53e3e; transition: width 0.3s; }}

            /* 一致性检查 */
            .check-list {{ list-style: none; padding: 0; }}
            .check-list li {{ padding: 8px 12px; margin: 5px 0; border-radius: 6px; display: flex; align-items: center; }}
            .check-list li.passed {{ background: #f0fff4; border-left: 3px solid #38a169; }}
            .check-list li.failed {{ background: #fff5f5; border-left: 3px solid #e53e3e; }}
            .check-list li.warning {{ background: #fffaf0; border-left: 3px solid #dd6b20; }}
            .check-icon {{ margin-right: 10px; font-size: 16px; }}

            /* 表格 */
            table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
            th, td {{ border: 1px solid #e2e8f0; padding: 10px; text-align: center; }}
            th {{ background: #3182ce; color: white; }}
            tr:nth-child(even) {{ background: #f7fafc; }}

            /* 动作卡片 */
            .action-section {{ background: #fafafa; border: 1px solid #e2e8f0; border-radius: 10px; padding: 20px; margin: 20px 0; }}
            .action-section:hover {{ box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
            .images {{ display: flex; gap: 12px; flex-wrap: wrap; justify-content: flex-start; margin-top: 15px; }}
            .images img {{ max-width: 480px; border-radius: 8px; border: 1px solid #e2e8f0; }}

            /* 提示框 */
            .tip {{ background: #fffff0; border-left: 4px solid #ecc94b; padding: 12px 15px; margin: 15px 0; border-radius: 0 6px 6px 0; }}
            .tip.info {{ background: #ebf8ff; border-color: #3182ce; }}
            .tip.warning {{ background: #fffaf0; border-color: #dd6b20; }}
            .tip.error {{ background: #fff5f5; border-color: #e53e3e; }}

            /* 解释文本 */
            .interpretation {{ background: #f7fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px; white-space: pre-line; font-family: monospace; font-size: 13px; }}

            .small {{ font-size: 12px; color: #718096; }}
            .tag {{ display: inline-block; padding: 2px 8px; border-radius: 12px; background: #ebf4ff; margin-left: 6px; font-size: 11px; color: #3182ce; }}

            /* 响应式 */
            @media (max-width: 768px) {{
                .comparison {{ grid-template-columns: 1fr; }}
                .diagnosis-grid {{ grid-template-columns: repeat(2, 1fr); }}
            }}
        </style>
    </head>
    <body>
    <div class="container">
        <h1>🏥 面瘫智能评估报告</h1>

        <div style="background: #f7fafc; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
            <div><b>检查ID:</b> {exam_id}</div>
            <div><b>患者ID:</b> {patient_id}</div>
            <div><b>分析时间:</b> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
            <div><b>处理动作数:</b> {len(action_results)}</div>
        </div>

        <!-- ==================== Session诊断摘要卡片 ==================== -->
        <div class="diagnosis-card">
            <h2>🎯 Session诊断结果</h2>
            <div class="diagnosis-grid">
                <div class="metric-card">
                    <div class="metric-label">面瘫判定</div>
                    <div class="metric-value">{pred_has}</div>
                    <div class="metric-sub">置信度: {pred_conf:.0%}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">患侧</div>
                    <div class="metric-value">{pred_side_text}</div>
                    <div class="metric-sub">L:{pred_left:.2f} R:{pred_right:.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">HB分级</div>
                    <div class="metric-value">Grade {pred_hb}</div>
                    <div class="metric-sub">{pred_hb_desc.split('(')[0] if pred_hb_desc else ''}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Sunnybrook</div>
                    <div class="metric-value">{pred_sb}</div>
                    <div class="metric-sub">{diagnosis.voluntary_score if diagnosis else 0} - {diagnosis.resting_score if diagnosis else 0} - {diagnosis.synkinesis_score if diagnosis else 0}</div>
                </div>
            </div>
        </div>

        <!-- ==================== GT vs Prediction 比较 ==================== -->
        <h2>📊 Ground Truth vs Prediction</h2>
        <div class="comparison">
            <div class="compare-box ground-truth">
                <h3>📋 医生标注 (Ground Truth)</h3>
                <table>
                    <tr><td><b>面瘫</b></td><td>{gt_has}</td></tr>
                    <tr><td><b>患侧</b></td><td>{gt_side_code} ({gt_side_text})</td></tr>
                    <tr><td><b>HB分级</b></td><td>{gt_hb}</td></tr>
                    <tr><td><b>Sunnybrook</b></td><td>{gt_sb}</td></tr>
                </table>
            </div>
            <div class="compare-box prediction">
                <h3>🤖 系统预测 (Prediction)</h3>
                <table>
                    <tr><td><b>面瘫</b></td><td>{pred_has}</td></tr>
                    <tr><td><b>患侧</b></td><td>{pred_side} ({pred_side_text})</td></tr>
                    <tr><td><b>HB分级</b></td><td>{pred_hb}</td></tr>
                    <tr><td><b>Sunnybrook</b></td><td>{pred_sb}</td></tr>
                </table>
            </div>
        </div>

        <div style="text-align: center; margin: 15px 0;">
            <b>匹配结果:</b> {match_palsy} {match_side} {match_hb}
        </div>

        <!-- ==================== 一致性检查 ==================== -->
        <h2>✅ 一致性检查</h2>
        <ul class="check-list">
    """

    # 添加一致性检查结果
    for check in checks:
        status_class = "passed" if check.passed else ("warning" if check.severity == "warning" else "failed")
        icon = "✓" if check.passed else ("⚠" if check.severity == "warning" else "✗")
        html += f'<li class="{status_class}"><span class="check-icon">{icon}</span><b>{check.rule_name}:</b> {check.message}</li>'

    if not checks:
        html += '<li class="passed"><span class="check-icon">✓</span>所有一致性检查通过</li>'

    html += """
        </ul>
    """

    # 如果有调整，显示调整说明
    if adjustments:
        html += '<div class="tip warning"><b>已做出的调整:</b><ul>'
        for adj in adjustments:
            html += f'<li>{adj}</li>'
        html += '</ul></div>'

    # 诊断解释
    if interpretation:
        html += f"""
        <h2>📝 诊断解释</h2>
        <div class="interpretation">{interpretation}</div>
    """

    # 证据投票
    html += """
        <h2>🗳️ 动作投票证据</h2>
        <div style="margin-bottom: 15px;">
            <div><b>左侧累计证据:</b> {pred_left:.2f} {_bar(pred_left, max(pred_left, pred_right, 1.0), '#3182ce')}</div>
            <div><b>右侧累计证据:</b> {pred_right:.2f} {_bar(pred_right, max(pred_left, pred_right, 1.0), '#e53e3e')}</div>
        </div>
    """

    # Top证据
    if top_evidence:
        html += """
        <div class="tip info">
            <b>Top 5 证据:</b><br/>
    """
        for i, e in enumerate(top_evidence[:5]):
            html += f'<span class="badge badge-blue">{i + 1}</span> {e.action_cn} ({e.region}): {e.side_text}侧弱, 权重×置信={e.weighted_score:.2f}<br/>'
        html += '</div>'

    # Sunnybrook详细评分
    if sunnybrook:
        html += f"""
        <h2>📊 Sunnybrook详细评分</h2>
        <div style="background: #f7fafc; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
            <span class="badge badge-blue">Resting: {sunnybrook.resting_score}</span>
            <span class="badge badge-green">Voluntary: {sunnybrook.voluntary_score}</span>
            <span class="badge badge-yellow">Synkinesis: {sunnybrook.synkinesis_score}</span>
            <span class="badge" style="background: #667eea; color: white;">Composite: {sunnybrook.composite_score}</span>
            <div class="small" style="margin-top: 10px;">公式: {sunnybrook.voluntary_score} - {sunnybrook.resting_score} - {sunnybrook.synkinesis_score} = {sunnybrook.composite_score}</div>
        </div>

        <h3>1️⃣ Resting Symmetry (静息对称性)</h3>
        <table>
            <tr><th>部位</th><th>状态</th><th>测量值</th><th>评分</th></tr>
    """
        for it in [sunnybrook.resting_symmetry.eye, sunnybrook.resting_symmetry.cheek,
                   sunnybrook.resting_symmetry.mouth]:
            html += f"<tr><td>{it.region_cn}</td><td>{it.status_cn}</td><td>{it.measurement:.3f}</td><td>{it.score}</td></tr>"
        html += f"""
            <tr><td colspan="4"><b>Total:</b> {sunnybrook.resting_symmetry.raw_score} × 5 = {sunnybrook.resting_symmetry.total_score}</td></tr>
        </table>

        <h3>2️⃣ Voluntary Movement (主动运动)</h3>
        <table>
            <tr><th>表情</th><th>左侧</th><th>右侧</th><th>比值</th><th>评分</th></tr>
    """
        for it in sunnybrook.voluntary_movement.items:
            html += f"<tr><td>{it.expression_cn}</td><td>{it.left_value:.3f}</td><td>{it.right_value:.3f}</td><td>{it.ratio:.3f}</td><td>{it.score}</td></tr>"
        html += f"""
            <tr><td colspan="5"><b>Total:</b> {sunnybrook.voluntary_movement.raw_sum} × 4 = {sunnybrook.voluntary_movement.total_score}</td></tr>
        </table>

        <h3>3️⃣ Synkinesis (联动运动)</h3>
        <table>
            <tr><th>表情</th><th>眼联动</th><th>嘴联动</th><th>评分</th></tr>
    """
        for it in sunnybrook.synkinesis.items:
            html += f"<tr><td>{it.expression_cn}</td><td>{it.eye_synkinesis}</td><td>{it.mouth_synkinesis}</td><td>{it.total_score}</td></tr>"
        html += f"""
            <tr><td colspan="4"><b>Total:</b> {sunnybrook.synkinesis_score}</td></tr>
        </table>
    """

    # 各动作详细分析
    html += """
        <h2>📹 各动作详细分析</h2>
    """

    action_order = [
        "NeutralFace", "SpontaneousEyeBlink", "VoluntaryEyeBlink",
        "CloseEyeSoftly", "CloseEyeHardly", "RaiseEyebrow",
        "Smile", "ShrugNose", "ShowTeeth", "BlowCheek", "LipPucker",
    ]

    for action_name in action_order:
        if action_name not in action_results:
            continue
        result = action_results[action_name]
        cn = action_name_map.get(action_name, action_name)
        action_dir = output_dir / action_name

        # 获取动作的诊断信息
        action_spec = getattr(result, 'action_specific', {}) or {}
        palsy_det = action_spec.get('palsy_detection', {})
        act_palsy_side = palsy_det.get('palsy_side', 0)
        act_confidence = palsy_det.get('confidence', 0)
        act_severity = action_spec.get('severity_score', 0)
        act_voluntary = action_spec.get('voluntary_score', result.voluntary_movement_score or 0)

        palsy_text = {0: "Symmetric", 1: "Left Palsy", 2: "Right Palsy"}.get(act_palsy_side, "Unknown")
        palsy_badge_class = "badge-green" if act_palsy_side == 0 else "badge-red"

        def _img_tag(rel_path, alt):
            p = action_dir / rel_path
            if p.exists():
                return f'<img src="{action_name}/{rel_path}" alt="{alt}"/>'
            return ""

        html += f"""
        <div class="action-section">
            <h3>{action_name} - {cn}</h3>
            <div style="margin-bottom: 10px;">
                <span class="badge {palsy_badge_class}">{palsy_text}</span>
                <span class="badge badge-blue">Severity: {act_severity}/5</span>
                <span class="badge badge-yellow">Voluntary: {act_voluntary}/5</span>
                <span class="small">Confidence: {act_confidence:.0%}</span>
            </div>
            <table>
                <tr><th>指标</th><th>数值</th><th>指标</th><th>数值</th></tr>
                <tr>
                    <td>峰值帧</td><td>{result.peak_frame_idx}/{result.total_frames}</td>
                    <td>ICD</td><td>{(result.icd or 0):.1f}px</td>
                </tr>
                <tr>
                    <td>EAR Left</td><td>{(result.left_ear or 0):.4f}</td>
                    <td>EAR Right</td><td>{(result.right_ear or 0):.4f}</td>
                </tr>
                <tr>
                    <td>Eye Area Ratio</td><td>{(result.eye_area_ratio or 0):.3f}</td>
                    <td>Brow H Ratio</td><td>{(result.brow_height_ratio or 0):.3f}</td>
                </tr>
                <tr>
                    <td>Mouth Width</td><td>{(result.mouth_width or 0):.1f}px</td>
                    <td>NLF Ratio</td><td>{(result.nlf_ratio or 0):.3f}</td>
                </tr>
            </table>
            <div class="images">
                {_img_tag("peak_raw.jpg", "原始帧")}
                {_img_tag("peak_indicators.jpg", "指标可视化")}
                {_img_tag("peak_selection_curve.png", "峰值选择曲线")}
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


def load_baseline_from_existing_json(exam_output_dir: Path) -> Tuple[Optional[ActionResult], Optional[Any]]:
    """
    从已有 NeutralFace/indicators.json 加载基线数值（用于只跑部分动作时复用）
    注意：baseline_landmarks 无法从 JSON 恢复，所以第二个返回值仍为 None。
    但我们会把完整 baseline 数值缓存到 baseline_result.action_specific["baseline_cache"]。
    """
    neutral_dir = exam_output_dir / "NeutralFace"
    indicators_path = neutral_dir / "indicators.json"

    if not indicators_path.exists():
        print(f"    [!] 未找到已有基线: {indicators_path}")
        return None, None

    try:
        with open(indicators_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 重建 ActionResult
        img_size = data.get("image_size", {}) or {}
        w = int(img_size.get("width", 0) or 0)
        h = int(img_size.get("height", 0) or 0)

        result = ActionResult(
            action_name="NeutralFace",
            action_name_cn="静息面",
            video_path=str(data.get("video_path", "") or ""),
            total_frames=int(data.get("total_frames", 0) or 0),
            peak_frame_idx=int(data.get("peak_frame_idx", 0) or 0),
            image_size=(w, h),
            fps=float(data.get("fps", 30.0) or 30.0),
        )

        # icd
        if "icd" in data:
            result.icd = float(data["icd"] or 0.0)

        # eye / brow / mouth / nlf
        eye = data.get("eye", {}) or {}
        brow = data.get("brow", {}) or {}
        mouth = data.get("mouth", {}) or {}
        nlf = data.get("nlf", {}) or {}

        result.left_eye_area = float(eye.get("left_area", 0.0) or 0.0)
        result.right_eye_area = float(eye.get("right_area", 0.0) or 0.0)
        result.eye_area_ratio = float(eye.get("area_ratio", 1.0) or 1.0)
        result.left_ear = float(eye.get("left_ear", 0.0) or 0.0)
        result.right_ear = float(eye.get("right_ear", 0.0) or 0.0)
        result.left_palpebral_height = float(eye.get("left_palpebral_height", 0.0) or 0.0)
        result.right_palpebral_height = float(eye.get("right_palpebral_height", 0.0) or 0.0)
        result.palpebral_height_ratio = float(eye.get("palpebral_height_ratio", 1.0) or 1.0)
        result.left_palpebral_width = float(eye.get("left_palpebral_width", 0.0) or 0.0)
        result.right_palpebral_width = float(eye.get("right_palpebral_width", 0.0) or 0.0)

        result.left_brow_height = float(brow.get("left_height", 0.0) or 0.0)
        result.right_brow_height = float(brow.get("right_height", 0.0) or 0.0)
        result.brow_height_ratio = float(brow.get("height_ratio", 1.0) or 1.0)
        result.left_brow_position = brow.get("left_position", None)
        result.right_brow_position = brow.get("right_position", None)

        # brow eye distance（如果有）
        result.left_brow_eye_distance = float(brow.get("left_brow_eye_distance", 0.0) or 0.0)
        result.right_brow_eye_distance = float(brow.get("right_brow_eye_distance", 0.0) or 0.0)
        result.brow_eye_distance_ratio = float(brow.get("brow_eye_distance_ratio", 1.0) or 1.0)
        result.left_brow_eye_distance_change = float(brow.get("left_brow_eye_distance_change", 0.0) or 0.0)
        result.right_brow_eye_distance_change = float(brow.get("right_brow_eye_distance_change", 0.0) or 0.0)
        result.brow_eye_distance_change_ratio = float(brow.get("brow_eye_distance_change_ratio", 1.0) or 1.0)

        result.mouth_width = float(mouth.get("width", 0.0) or 0.0)
        result.mouth_height = float(mouth.get("height", 0.0) or 0.0)

        result.left_nlf_length = float(nlf.get("left_length", 0.0) or 0.0)
        result.right_nlf_length = float(nlf.get("right_length", 0.0) or 0.0)
        result.nlf_ratio = float(nlf.get("ratio", 1.0) or 1.0)

        # 把完整 baseline JSON 缓存起来，供后续动作（无 baseline_landmarks 时）使用
        result.action_specific["baseline_cache"] = data

        print(f"    [OK] 复用已有基线(JSON): {indicators_path}")
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
    print(f"医生标注: {labels}\n")

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
        baseline_result, baseline_landmarks = load_baseline_from_existing_json(exam_output_dir)
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
    diagnosis = compute_session_diagnosis(action_results)
    palsy_side = diagnosis.palsy_side if diagnosis else 0
    sunnybrook = calculate_sunnybrook_from_results(action_results)

    # summary 更新
    summary = {
        "exam_id": exam_id,
        "patient_id": patient_id,
        "analysis_time": datetime.now().isoformat(),
        "ground_truth": labels,
        "sunnybrook": sunnybrook.to_dict() if sunnybrook else None,
        "diagnosis": diagnosis.to_dict() if diagnosis else None,
        "actions": {name: result.to_dict() for name, result in action_results.items()},
    }

    # 生成HTML报告
    generate_html_report(
        exam_id, patient_id,
        action_results,
        sunnybrook,
        labels,
        diagnosis,
        exam_output_dir
    )

    if sunnybrook:
        summary["sunnybrook"] = sunnybrook.to_dict()

    with open(exam_output_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(make_json_serializable(summary), f, indent=2, ensure_ascii=False)

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
    print("[DEBUG] running file:", __file__)
    print("[DEBUG] ENABLED_ACTIONS:", ENABLED_ACTIONS)
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
        all_results = []
        with LandmarkExtractor(MEDIAPIPE_MODEL_PATH) as extractor:
            for i, exam in enumerate(examinations):
                print(f"\n[{i + 1}/{len(examinations)}]", end="")

                result = process_examination(
                    exam,
                    DATABASE_PATH,
                    output_dir,
                    extractor,
                    enabled_actions=ENABLED_ACTIONS,
                    reuse_baseline=REUSE_BASELINE,
                    skip_existing=SKIP_EXISTING_ACTIONS,
                )
                all_results.append(result)

    print(f"\n\n{'=' * 70}")
    print("处理完成!")
    print(f"{'=' * 70}")
    print(f"处理了 {len(all_results)} 个检查")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()