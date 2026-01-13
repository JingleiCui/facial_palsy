#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recompute Diagnosis (recompute_diagnosis.py)
============================================

功能：
1. 读取已有的 action 结果 (indicators.json)
2. 使用 session_diagnosis.py 中的最新逻辑重新计算 Session 级诊断
3. 更新 summary.json
4. 输出准确率统计报告

用途：
调整诊断逻辑/阈值后，无需重新分析视频即可快速验证效果。
"""

import os
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import sys

# 引入项目模块
sys.path.insert(0, str(Path(__file__).parent))
from session_diagnosis import compute_session_diagnosis, SessionDiagnosis
from sunnybrook_scorer import SunnybrookScore, RestingSymmetry, VoluntaryMovement, Synkinesis, \
    RestingSymmetryItem, VoluntaryMovementItem, SynkinesisItem, compute_sunnybrook_composite
from clinical_base import ActionResult, OralAngleMeasure

# 配置
DATA_DIR = Path("/Users/cuijinglei/Documents/facialPalsy/HGFA/clinical_grading")

ACTIONS = [
    "NeutralFace", "SpontaneousEyeBlink", "VoluntaryEyeBlink",
    "CloseEyeSoftly", "CloseEyeHardly", "RaiseEyebrow",
    "Smile", "ShrugNose", "ShowTeeth", "BlowCheek", "LipPucker",
]


def dict_to_action_result(data: Dict[str, Any]) -> ActionResult:
    """将 indicators.json 的字典内容还原为 ActionResult 对象"""
    # 还原 OralAngleMeasure
    oral_angle = None
    if "oral_angle" in data:
        oa = data["oral_angle"]
        # 注意: 这里只还原了必要的字段用于诊断
        oral_angle = OralAngleMeasure(
            A=(0, 0), B=(0, 0), C=(0, 0), D=(0, 0), E=(0, 0), F=(0, 0), O=(0, 0),  # 坐标不重要
            AOE_angle=oa.get("AOE_angle_deg", 0),
            BOF_angle=oa.get("BOF_angle_deg", 0),
            angle_diff=oa.get("angle_diff", 0),
            angle_asymmetry=oa.get("angle_asymmetry", 0)
        )

    # 创建对象
    res = ActionResult(
        action_name=data.get("action_name", ""),
        action_name_cn=data.get("action_name_cn", ""),
        video_path=data.get("video_path", ""),
        total_frames=data.get("total_frames", 0),
        peak_frame_idx=data.get("peak_frame_idx", 0),
        image_size=tuple(data.get("image_size", {}).values()) if isinstance(data.get("image_size"), dict) else (0, 0),
        fps=data.get("fps", 30.0)
    )

    # 填充关键属性
    res.action_specific = data.get("action_specific", {})
    res.voluntary_movement_score = data.get("voluntary_movement_score", 5)
    res.synkinesis_scores = data.get("synkinesis_scores", {})
    res.oral_angle = oral_angle

    # 填充通用指标 (SessionDiagnosis 可能用到)
    res.icd = data.get("icd", 0)
    res.mouth_width = data.get("mouth", {}).get("width", 0)
    res.left_ear = data.get("eye", {}).get("left_ear", 0)
    res.right_ear = data.get("eye", {}).get("right_ear", 0)
    res.left_brow_height = data.get("brow", {}).get("left_height", 0)
    res.right_brow_height = data.get("brow", {}).get("right_height", 0)
    res.brow_height_ratio = data.get("brow", {}).get("height_ratio", 1.0)
    res.nlf_ratio = data.get("nlf", {}).get("ratio", 1.0)
    res.palpebral_height_ratio = data.get("eye", {}).get("palpebral_height_ratio", 1.0)

    # 特殊处理 RaiseEyebrow 的变化量
    if res.action_name == "RaiseEyebrow":
        brow_metrics = res.action_specific.get("brow_eye_metrics", {})
        res.left_brow_eye_distance_change = brow_metrics.get("left_change", 0)
        res.right_brow_eye_distance_change = brow_metrics.get("right_change", 0)

    return res


def reconstruct_sunnybrook(data: Dict[str, Any]) -> Optional[SunnybrookScore]:
    """从 summary.json 的字典重建 SunnybrookScore 对象"""
    if not data:
        return None

    try:
        # 重建 Resting
        r_data = data.get("resting_symmetry", {})
        resting = RestingSymmetry(
            eye=RestingSymmetryItem(**r_data.get("eye", {})),
            cheek=RestingSymmetryItem(**r_data.get("cheek", {})),
            mouth=RestingSymmetryItem(**r_data.get("mouth", {})),
            raw_score=r_data.get("raw_score", 0),
            total_score=r_data.get("total_score", 0),
            affected_side=r_data.get("affected_side", "")
        )

        # 重建 Voluntary
        v_data = data.get("voluntary_movement", {})
        v_items = [VoluntaryMovementItem(**item) for item in v_data.get("items", [])]
        voluntary = VoluntaryMovement(
            items=v_items,
            raw_sum=v_data.get("raw_sum", 0),
            total_score=v_data.get("total_score", 0)
        )

        # 重建 Synkinesis
        s_data = data.get("synkinesis", {})
        s_items = [SynkinesisItem(**item) for item in s_data.get("items", [])]
        synkinesis = Synkinesis(
            items=s_items,
            total_score=s_data.get("total_score", 0)
        )

        return SunnybrookScore(
            resting_symmetry=resting,
            voluntary_movement=voluntary,
            synkinesis=synkinesis,
            resting_score=data.get("scores", {}).get("resting_score", 0),
            voluntary_score=data.get("scores", {}).get("voluntary_score", 0),
            synkinesis_score=data.get("scores", {}).get("synkinesis_score", 0),
            composite_score=data.get("scores", {}).get("composite_score", 0)
        )
    except Exception as e:
        print(f"[WARN] Failed to reconstruct Sunnybrook object: {e}")
        return None


def main():
    print("=" * 60)
    print("RECOMPUTE DIAGNOSIS & STATISTICS")
    print(f"Data Dir: {DATA_DIR}")
    print("=" * 60)

    # 查找所有检查目录 (包含 summary.json 的目录)
    exam_dirs = [p.parent for p in DATA_DIR.rglob("summary.json")]
    print(f"Found {len(exam_dirs)} examinations.")

    stats = {
        "total": 0,
        "has_palsy_correct": 0,
        "side_correct": 0,
        "side_wrong": 0,
        "side_fn": 0,
        "side_fp": 0,
        "hb_exact": 0,
        "hb_within1": 0,
    }

    results_buffer = []

    for exam_dir in exam_dirs:
        exam_id = exam_dir.name

        # 1. 加载 summary.json 获取 GT 和 旧的 Sunnybrook (如果需要)
        with open(exam_dir / "summary.json", 'r', encoding='utf-8') as f:
            summary = json.load(f)

        gt = summary.get("ground_truth", {})

        # 2. 加载各动作的 indicators.json
        action_results = {}
        for action in ACTIONS:
            json_path = exam_dir / action / "indicators.json"
            if json_path.exists():
                with open(json_path, 'r', encoding='utf-8') as f:
                    action_data = json.load(f)
                    action_results[action] = dict_to_action_result(action_data)

        # 3. 重建 Sunnybrook 对象 (通常不需要重算 SB，除非 sunnybrook_scorer 也改了)
        # 如果你想重算 SB，需要调用 calculate_sunnybrook_from_results
        sb_obj = reconstruct_sunnybrook(summary.get("sunnybrook", {}))

        # 4. 核心：重新运行 Session Diagnosis
        diagnosis = compute_session_diagnosis(action_results, sb_obj)

        # 5. 更新 summary
        summary["diagnosis"] = diagnosis.to_dict()

        # 保存回文件 (可选，方便后续 collect_keyframes 读取)
        with open(exam_dir / "summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # 6. 实时统计
        stats["total"] += 1

        # Has Palsy
        gt_has = bool(gt.get("has_palsy", 0))
        pred_has = diagnosis.has_palsy
        if gt_has == pred_has:
            stats["has_palsy_correct"] += 1

        # Palsy Side
        gt_side = gt.get("palsy_side", 0)
        pred_side = diagnosis.palsy_side

        if gt_side == pred_side:
            if gt_side != 0: stats["side_correct"] += 1
        else:
            if gt_side != 0 and pred_side == 0:
                stats["side_fn"] += 1  # 漏检
            elif gt_side == 0 and pred_side != 0:
                stats["side_fp"] += 1  # 误检
            elif gt_side != 0 and pred_side != 0:
                stats["side_wrong"] += 1  # 侧别错

        # HB Grade
        gt_hb = gt.get("hb_grade")
        pred_hb = diagnosis.hb_grade
        if gt_hb is not None:
            if gt_hb == pred_hb:
                stats["hb_exact"] += 1
            if abs(gt_hb - pred_hb) <= 1:
                stats["hb_within1"] += 1

        # 记录错误详情
        if gt_side != 0 and pred_side != gt_side:
            results_buffer.append({
                "id": exam_id,
                "gt": f"{gt_side} (HB{gt_hb})",
                "pred": f"{pred_side} (HB{pred_hb})",
                "votes": len(diagnosis.votes),
                "top_vote": f"{diagnosis.votes[0].action}:{diagnosis.votes[0].side}" if diagnosis.votes else "None",
                "conf": f"{diagnosis.confidence:.2f}"
            })

    # === 输出统计 ===
    print("\n" + "=" * 60)
    print("RECOMPUTED STATISTICS")
    print("=" * 60)
    print(f"Total Exams: {stats['total']}")

    acc_has = stats['has_palsy_correct'] / stats['total'] if stats['total'] else 0
    print(f"Has Palsy Acc: {acc_has:.1%} ({stats['has_palsy_correct']}/{stats['total']})")

    total_palsy = stats['side_correct'] + stats['side_wrong'] + stats['side_fn']
    acc_strict = stats['side_correct'] / total_palsy if total_palsy else 0
    acc_relax = (stats['side_correct'] + stats['side_fn']) / total_palsy if total_palsy else 0  # 注意定义
    # 你的 Relaxed 定义：OK + FN / Total (预测对称也算正确? 这通常用于排除误报，但在这里 FN 是漏检)
    # 通常 Relaxed 指的是 Side Correct + Side Wrong (检测出面瘫但侧别错了也算检测出)
    # 或者 指的是 Side Correct (Strict)

    print(f"\nPalsy Side (Palsy Cases Only: {total_palsy})")
    print(f"  Strict Acc (Correct): {acc_strict:.1%} ({stats['side_correct']})")
    print(f"  Missed (FN):          {stats['side_fn']}")
    print(f"  Wrong Side:           {stats['side_wrong']}")
    print(f"  False Positive:       {stats['side_fp']}")

    print(f"\nHB Grade")
    print(f"  Exact Match: {stats['hb_exact'] / stats['total']:.1%}")
    print(f"  Within ±1:   {stats['hb_within1'] / stats['total']:.1%}")

    print("\n" + "=" * 60)
    print("ERROR SAMPLES (Top 10)")
    print(f"{'Exam ID':<25} {'GT':<10} {'Pred':<10} {'Votes':<5} {'Top Vote'}")
    print("-" * 60)
    for r in results_buffer[:15]:
        print(f"{r['id']:<25} {r['gt']:<10} {r['pred']:<10} {r['votes']:<5} {r['top_vote']}")


if __name__ == "__main__":
    main()
