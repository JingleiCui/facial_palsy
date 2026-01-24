#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recompute Diagnosis (recompute_diagnosis.py)
============================================

功能：
1. 从数据库读取最新的 ground_truth 标签（确保Excel修改后能同步）
2. 读取已有的 action 结果 (indicators.json)
3. 使用 session_diagnosis.py 中的最新逻辑重新计算 Session 级诊断
4. 更新 summary.json（包括 ground_truth 和 diagnosis）
5. 输出准确率统计报告
"""

import os
import json
import sqlite3
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

# ============================================================
# ★★★ 配置 - 请确认路径正确 ★★★
# ============================================================

DATA_DIR = Path("/Users/cuijinglei/Documents/facial_palsy/HGFA/clinical_grading")
DB_PATH = Path("/Users/cuijinglei/PycharmProjects/medicalProject/facial_palsy/facialPalsy.db")

# 是否从数据库更新 ground_truth
UPDATE_GT_FROM_DB = True

# ★★★ 调试模式：设为 True 会打印详细的匹配信息 ★★★
DEBUG_MODE = True

ACTIONS = [
    "NeutralFace", "SpontaneousEyeBlink", "VoluntaryEyeBlink",
    "CloseEyeSoftly", "CloseEyeHardly", "RaiseEyebrow",
    "Smile", "ShrugNose", "ShowTeeth", "BlowCheek", "LipPucker",
]


# ============================================================
# 数据库标签读取
# ============================================================

def load_labels_from_db(db_path: Path) -> Dict[str, Dict[str, Any]]:
    """从数据库加载所有检查的标签"""
    if not db_path.exists():
        print(f"❌ 数据库文件不存在: {db_path}")
        print(f"   请检查路径是否正确!")
        return {}

    print(f"📂 连接数据库: {db_path}")

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # 检查表是否存在
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='examination_labels'")
    if not cursor.fetchone():
        print("❌ 数据库中没有 examination_labels 表")
        print("   请先运行 import_labels.py 导入标签!")
        conn.close()
        return {}

    cursor.execute('''
        SELECT 
            examination_id,
            has_palsy,
            palsy_side,
            hb_grade,
            sunnybrook_score,
            label_source
        FROM examination_labels
    ''')

    labels = {}
    for row in cursor.fetchall():
        exam_id, has_palsy, palsy_side, hb_grade, sunnybrook, source = row
        labels[exam_id] = {
            "has_palsy": has_palsy,
            "palsy_side": palsy_side,
            "hb_grade": hb_grade,
            "sunnybrook_score": sunnybrook,
            "label_source": source,
        }

    conn.close()

    print(f"✅ 从数据库加载了 {len(labels)} 条标签")

    # 打印所有数据库中的 examination_id（调试用）
    if DEBUG_MODE:
        print("\n📋 数据库中所有 examination_id:")
        for i, (k, v) in enumerate(sorted(labels.items())):
            print(f"   {i+1:3d}. {k} -> side={v['palsy_side']}, HB={v['hb_grade']}")
        print()

    return labels


def match_exam_id_to_db(exam_dir_name: str, db_labels: Dict[str, Dict], debug: bool = False) -> Optional[str]:
    """
    将目录名匹配到数据库的 examination_id
    """
    # 1. 精确匹配
    if exam_dir_name in db_labels:
        if debug:
            print(f"      ✓ 精确匹配成功: {exam_dir_name}")
        return exam_dir_name

    # 2. 前缀匹配
    for db_id in db_labels.keys():
        # 尝试两种方向的前缀匹配
        if db_id.startswith(exam_dir_name) or exam_dir_name.startswith(db_id):
            if debug:
                print(f"      ✓ 前缀匹配成功: {exam_dir_name} -> {db_id}")
            return db_id

        # 更宽松：比较前两部分 (patient_date)
        dir_parts = exam_dir_name.split('_')
        db_parts = db_id.split('_')

        if len(dir_parts) >= 2 and len(db_parts) >= 2:
            # 比较患者ID
            if dir_parts[0] == db_parts[0]:
                # 比较日期前8位（YYYYMMDD）
                dir_date = dir_parts[1].replace('-', '')[:8]
                db_date = db_parts[1].replace('-', '')[:8]
                if dir_date == db_date:
                    if debug:
                        print(f"      ✓ 宽松匹配成功: {exam_dir_name} -> {db_id}")
                    return db_id

    if debug:
        print(f"      ✗ 匹配失败: {exam_dir_name}")
        # 打印可能的候选
        dir_parts = exam_dir_name.split('_')
        if len(dir_parts) >= 1:
            patient_id = dir_parts[0]
            candidates = [k for k in db_labels.keys() if k.startswith(patient_id)]
            if candidates:
                print(f"        可能的候选（同患者ID）: {candidates}")

    return None


# ============================================================
# ActionResult 重建
# ============================================================

def dict_to_action_result(data: Dict[str, Any]) -> ActionResult:
    """将 indicators.json 的字典内容还原为 ActionResult 对象"""
    oral_angle = None
    if "oral_angle" in data:
        oa = data["oral_angle"]
        oral_angle = OralAngleMeasure(
            A=(0, 0), B=(0, 0), C=(0, 0), D=(0, 0), E=(0, 0), F=(0, 0), O=(0, 0),
            AOE_angle=oa.get("AOE_angle_deg", 0),
            BOF_angle=oa.get("BOF_angle_deg", 0),
            angle_diff=oa.get("angle_diff", 0),
            angle_asymmetry=oa.get("angle_asymmetry", 0)
        )

    res = ActionResult(
        action_name=data.get("action_name", ""),
        action_name_cn=data.get("action_name_cn", ""),
        video_path=data.get("video_path", ""),
        total_frames=data.get("total_frames", 0),
        peak_frame_idx=data.get("peak_frame_idx", 0),
        image_size=tuple(data.get("image_size", {}).values()) if isinstance(data.get("image_size"), dict) else (0, 0),
        fps=data.get("fps", 30.0)
    )

    res.action_specific = data.get("action_specific", {})
    res.voluntary_movement_score = data.get("voluntary_movement_score", 5)
    res.synkinesis_scores = data.get("synkinesis_scores", {})
    res.oral_angle = oral_angle

    res.icd = data.get("icd", 0)
    res.mouth_width = data.get("mouth", {}).get("width", 0)
    res.left_ear = data.get("eye", {}).get("left_ear", 0)
    res.right_ear = data.get("eye", {}).get("right_ear", 0)
    res.left_brow_height = data.get("brow", {}).get("left_height", 0)
    res.right_brow_height = data.get("brow", {}).get("right_height", 0)
    res.brow_height_ratio = data.get("brow", {}).get("height_ratio", 1.0)
    res.nlf_ratio = data.get("nlf", {}).get("ratio", 1.0)
    res.palpebral_height_ratio = data.get("eye", {}).get("palpebral_height_ratio", 1.0)

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
        r_data = data.get("resting_symmetry", {})
        resting = RestingSymmetry(
            eye=RestingSymmetryItem(**r_data.get("eye", {})),
            cheek=RestingSymmetryItem(**r_data.get("cheek", {})),
            mouth=RestingSymmetryItem(**r_data.get("mouth", {})),
            raw_score=r_data.get("raw_score", 0),
            total_score=r_data.get("total_score", 0),
            affected_side=r_data.get("affected_side", "")
        )

        v_data = data.get("voluntary_movement", {})
        v_items = [VoluntaryMovementItem(**item) for item in v_data.get("items", [])]
        voluntary = VoluntaryMovement(
            items=v_items,
            raw_sum=v_data.get("raw_sum", 0),
            total_score=v_data.get("total_score", 0)
        )

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


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 70)
    print("RECOMPUTE DIAGNOSIS & UPDATE GROUND TRUTH FROM DATABASE")
    print("=" * 70)
    print(f"Data Dir:           {DATA_DIR}")
    print(f"Database:           {DB_PATH}")
    print(f"Update GT from DB:  {UPDATE_GT_FROM_DB}")
    print(f"Debug Mode:         {DEBUG_MODE}")
    print("=" * 70 + "\n")

    # 检查路径
    if not DATA_DIR.exists():
        print(f"❌ 数据目录不存在: {DATA_DIR}")
        return

    # 1. 从数据库加载最新标签
    db_labels = {}
    if UPDATE_GT_FROM_DB:
        db_labels = load_labels_from_db(DB_PATH)
        if not db_labels:
            print("\n⚠️  数据库标签为空，将不更新 ground_truth")
            print("   请先运行: python import_labels.py")
            return

    # 2. 查找所有检查目录
    exam_dirs = sorted([p.parent for p in DATA_DIR.rglob("summary.json")])
    print(f"📁 Found {len(exam_dirs)} examinations.\n")

    if not exam_dirs:
        print("❌ 没有找到任何 summary.json 文件")
        return

    stats = {
        "total": 0,
        "gt_updated": 0,
        "gt_not_found": 0,
        "has_palsy_correct": 0,
        "side_correct": 0,
        "side_wrong": 0,
        "side_fn": 0,
        "side_fp": 0,
        "hb_exact": 0,
        "hb_within1": 0,
        "hb_total": 0,
    }

    results_buffer = []
    gt_update_details = []
    not_found_list = []  # 记录匹配失败的

    print("=" * 70)
    print("处理每个检查...")
    print("=" * 70)

    for exam_dir in exam_dirs:
        exam_dir_name = exam_dir.name

        if DEBUG_MODE:
            print(f"\n📂 处理: {exam_dir_name}")

        # 1. 加载 summary.json
        summary_path = exam_dir / "summary.json"
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)

        # 保存旧的 ground_truth
        old_gt = summary.get("ground_truth", {}).copy()

        if DEBUG_MODE:
            print(f"   旧GT: side={old_gt.get('palsy_side')}, HB={old_gt.get('hb_grade')}")

        # 2. ★★★ 从数据库更新 ground_truth ★★★
        gt = summary.get("ground_truth", {})

        if UPDATE_GT_FROM_DB and db_labels:
            matched_db_id = match_exam_id_to_db(exam_dir_name, db_labels, debug=DEBUG_MODE)

            if matched_db_id:
                db_gt = db_labels[matched_db_id]

                # 构建新的 ground_truth
                new_gt = {
                    "has_palsy": db_gt["has_palsy"],
                    "palsy_side": db_gt["palsy_side"],
                    "hb_grade": db_gt["hb_grade"],
                    "sunnybrook_score": db_gt["sunnybrook_score"],
                    "label_source": db_gt["label_source"],
                    "db_examination_id": matched_db_id,
                }

                if DEBUG_MODE:
                    print(f"   新GT: side={new_gt['palsy_side']}, HB={new_gt['hb_grade']}")

                # 检查是否有变化
                if (old_gt.get("palsy_side") != new_gt["palsy_side"] or
                        old_gt.get("hb_grade") != new_gt["hb_grade"] or
                        old_gt.get("has_palsy") != new_gt["has_palsy"]):
                    gt_update_details.append({
                        "exam": exam_dir_name,
                        "old": f"side={old_gt.get('palsy_side')}, HB={old_gt.get('hb_grade')}",
                        "new": f"side={new_gt['palsy_side']}, HB={new_gt['hb_grade']}",
                    })
                    if DEBUG_MODE:
                        print(f"   ⚡ GT有变化!")

                # ★★★ 更新 summary 中的 ground_truth ★★★
                gt = new_gt
                summary["ground_truth"] = gt
                stats["gt_updated"] += 1
            else:
                stats["gt_not_found"] += 1
                not_found_list.append(exam_dir_name)

        # 3. 加载各动作的 indicators.json
        action_results = {}
        for action in ACTIONS:
            json_path = exam_dir / action / "indicators.json"
            if json_path.exists():
                with open(json_path, 'r', encoding='utf-8') as f:
                    action_data = json.load(f)
                    action_results[action] = dict_to_action_result(action_data)

        # 4. 重建 Sunnybrook 对象
        sb_obj = reconstruct_sunnybrook(summary.get("sunnybrook", {}))

        # 5. 重新运行 Session Diagnosis
        diagnosis = compute_session_diagnosis(action_results, sb_obj)

        # 6. 更新 summary 的 diagnosis
        summary["diagnosis"] = diagnosis.to_dict()

        # 7. ★★★ 保存回文件 ★★★
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        if DEBUG_MODE:
            print(f"   ✅ 已保存 summary.json")

        # 8. 统计
        stats["total"] += 1

        gt_has = bool(gt.get("has_palsy", 0))
        pred_has = diagnosis.has_palsy
        if gt_has == pred_has:
            stats["has_palsy_correct"] += 1

        gt_side = gt.get("palsy_side", 0) or 0
        pred_side = diagnosis.palsy_side

        if gt_side == pred_side:
            if gt_side != 0:
                stats["side_correct"] += 1
        else:
            if gt_side != 0 and pred_side == 0:
                stats["side_fn"] += 1
            elif gt_side == 0 and pred_side != 0:
                stats["side_fp"] += 1
            elif gt_side != 0 and pred_side != 0:
                stats["side_wrong"] += 1

        gt_hb = gt.get("hb_grade")
        pred_hb = diagnosis.hb_grade
        if gt_hb is not None:
            stats["hb_total"] += 1
            if gt_hb == pred_hb:
                stats["hb_exact"] += 1
            if abs(gt_hb - pred_hb) <= 1:
                stats["hb_within1"] += 1

        if gt_side != 0 and pred_side != gt_side:
            results_buffer.append({
                "id": exam_dir_name,
                "gt": f"side={gt_side}, HB={gt_hb}",
                "pred": f"side={pred_side}, HB={pred_hb}",
                "votes": len(diagnosis.votes),
                "top_vote": f"{diagnosis.votes[0].action}:{diagnosis.votes[0].side}" if diagnosis.votes else "None",
            })

    # === 输出匹配失败的列表 ===
    if not_found_list:
        print("\n" + "=" * 70)
        print(f"⚠️  匹配失败的检查 ({len(not_found_list)} 个)")
        print("=" * 70)
        for name in not_found_list[:20]:
            print(f"   - {name}")
        if len(not_found_list) > 20:
            print(f"   ... 还有 {len(not_found_list) - 20} 个")

    # === 输出 GT 更新详情 ===
    if gt_update_details:
        print("\n" + "=" * 70)
        print(f"📝 GROUND TRUTH 有变化 ({len(gt_update_details)} 个)")
        print("=" * 70)
        for detail in gt_update_details[:20]:
            print(f"  {detail['exam']}")
            print(f"    旧: {detail['old']}")
            print(f"    新: {detail['new']}")
        if len(gt_update_details) > 20:
            print(f"  ... 还有 {len(gt_update_details) - 20} 个")

    # === 输出统计 ===
    print("\n" + "=" * 70)
    print("📊 STATISTICS")
    print("=" * 70)
    print(f"Total Exams:        {stats['total']}")
    print(f"GT Updated:         {stats['gt_updated']}")
    print(f"GT Not Found in DB: {stats['gt_not_found']}")

    acc_has = stats['has_palsy_correct'] / stats['total'] if stats['total'] else 0
    print(f"\nHas Palsy Accuracy: {acc_has:.1%} ({stats['has_palsy_correct']}/{stats['total']})")

    total_palsy = stats['side_correct'] + stats['side_wrong'] + stats['side_fn']
    acc_strict = stats['side_correct'] / total_palsy if total_palsy else 0

    print(f"\nPalsy Side (Palsy Cases: {total_palsy})")
    print(f"  ✓ Correct:     {stats['side_correct']} ({acc_strict:.1%})")
    print(f"  ✗ Wrong Side:  {stats['side_wrong']}")
    print(f"  ✗ Missed (FN): {stats['side_fn']}")
    print(f"  ✗ False Pos:   {stats['side_fp']}")

    if stats['hb_total'] > 0:
        print(f"\nHB Grade (has GT: {stats['hb_total']})")
        print(f"  Exact Match: {stats['hb_exact']} ({stats['hb_exact'] / stats['hb_total']:.1%})")
        print(f"  Within ±1:   {stats['hb_within1']} ({stats['hb_within1'] / stats['hb_total']:.1%})")

    if results_buffer:
        print("\n" + "=" * 70)
        print(f"ERROR SAMPLES ({len(results_buffer)} total)")
        print("=" * 70)
        print(f"{'Exam ID':<35} {'GT':<18} {'Pred':<18} {'Votes'}")
        print("-" * 70)
        for r in results_buffer[:15]:
            print(f"{r['id']:<35} {r['gt']:<18} {r['pred']:<18} {r['votes']}")

    print("\n" + "=" * 70)
    print("✅ Done! All summary.json files have been updated.")
    print("=" * 70)


if __name__ == "__main__":
    main()