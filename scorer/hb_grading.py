#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
House-Brackmann 分级计算模块
==========================================

基于HB官方定义的5个评估项（a1-a5），映射到11个标准动作，
按顺序逐步计算最终HB等级。

临床理论依据：
1985年 House & Brackmann 原始论文定义了5个评估维度：
- a1: Gross function - General (整体功能)
- a2: Gross function - At rest (静息状态)
- a3: Motion - Forehead (额部运动)
- a4: Motion - Eyes (眼睛)
- a5: Motion - Mouth (口部)

最终HB等级 = max(a1, a2, a3, a4, a5)
"""

from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass
import numpy as np


# =============================================================================
# 数据结构定义
# =============================================================================

@dataclass
class HBComponentGrade:
    """单个HB评估项的结果"""
    component: str  # a1, a2, a3, a4, a5
    component_cn: str  # 中文名称
    grade: int  # 1-6
    grade_description: str  # 等级描述
    evidence: Dict[str, Any]  # 判断依据
    source_actions: List[str]  # 使用的动作


@dataclass
class HBGradingResult:
    """完整HB分级结果"""
    final_grade: int  # 最终HB等级 (I-VI)
    final_description: str  # 等级描述

    # 5个评估项的详细结果
    a1_general: HBComponentGrade  # 整体功能
    a2_at_rest: HBComponentGrade  # 静息状态
    a3_forehead: HBComponentGrade  # 额部运动
    a4_eyes: HBComponentGrade  # 眼睛
    a5_mouth: HBComponentGrade  # 口部

    palsy_side: int  # 面瘫侧别 (0=无, 1=左, 2=右)
    confidence: float  # 置信度
    decision_path: List[str]  # 决策路径记录

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_grade": self.final_grade,
            "final_description": self.final_description,
            "component_grades": {
                "a1_general": self.a1_general.__dict__,
                "a2_at_rest": self.a2_at_rest.__dict__,
                "a3_forehead": self.a3_forehead.__dict__,
                "a4_eyes": self.a4_eyes.__dict__,
                "a5_mouth": self.a5_mouth.__dict__,
            },
            "palsy_side": self.palsy_side,
            "confidence": self.confidence,
            "decision_path": self.decision_path,
        }


# =============================================================================
# HB等级描述映射
# =============================================================================

HB_DESCRIPTIONS = {
    1: "I - Normal (正常)",
    2: "II - Slight dysfunction (轻度功能障碍)",
    3: "III - Moderate dysfunction (中度功能障碍)",
    4: "IV - Moderately severe dysfunction (中重度功能障碍)",
    5: "V - Severe dysfunction (重度功能障碍)",
    6: "VI - Total paralysis (完全麻痹)",
}


# =============================================================================
# 阈值定义 (可根据数据集调整)
# =============================================================================

class HBThresholds:
    """HB分级专用阈值"""

    # ========== a2: At Rest 静息状态阈值 ==========
    # 眼睑裂高度比偏差
    REST_PALP_NORMAL = 0.08  # < 8% → Grade I/II
    REST_PALP_MILD = 0.15  # 8-15% → Grade IV
    REST_PALP_MODERATE = 0.25  # 15-25% → Grade V

    # 鼻唇沟长度比偏差
    REST_NLF_NORMAL = 0.10
    REST_NLF_MILD = 0.20
    REST_NLF_MODERATE = 0.35

    # 口角角度差(度)
    REST_ORAL_ANGLE_NORMAL = 3.0
    REST_ORAL_ANGLE_MILD = 6.0
    REST_ORAL_ANGLE_MODERATE = 10.0

    # ========== a3: Forehead 额部运动阈值 ==========
    # 眉眼距变化的对称比 (min_change / max_change)
    FOREHEAD_NORMAL = 0.85  # > 85% → Grade I
    FOREHEAD_GOOD = 0.60  # 60-85% → Grade II
    FOREHEAD_SLIGHT = 0.30  # 30-60% → Grade III
    # < 30% 或无运动 → Grade IV

    # 最小有效运动量(像素，需根据分辨率调整)
    FOREHEAD_MIN_MOVEMENT = 3.0

    # ========== a4: Eyes 眼睛闭合阈值 ==========
    # 闭合度 (closure_ratio)
    EYE_CLOSURE_COMPLETE = 0.85  # > 85% 视为完全闭合
    EYE_CLOSURE_PARTIAL = 0.50  # > 50% 视为部分闭合
    EYE_CLOSURE_MINIMAL = 0.20  # > 20% 视为有运动

    # 闭合对称比 (min_closure / max_closure)
    EYE_SYMMETRY_NORMAL = 0.85  # > 85% → 正常
    EYE_SYMMETRY_MILD = 0.60  # 60-85% → 轻度不对称

    # ========== a5: Mouth 口部运动阈值 ==========
    # 嘴角运动幅度对称比 (excursion_ratio)
    MOUTH_SYMMETRY_NORMAL = 0.85  # > 85% → Grade I
    MOUTH_SYMMETRY_SLIGHT = 0.70  # 70-85% → Grade II
    MOUTH_SYMMETRY_WEAK = 0.50  # 50-70% → Grade III
    MOUTH_SYMMETRY_ASYM = 0.30  # 30-50% → Grade IV
    MOUTH_SYMMETRY_MINIMAL = 0.15  # 15-30% → Grade V
    # < 15% → Grade VI

    # 口角角度不对称(度)
    MOUTH_ANGLE_NORMAL = 3.0
    MOUTH_ANGLE_SLIGHT = 6.0
    MOUTH_ANGLE_WEAK = 10.0
    MOUTH_ANGLE_ASYM = 15.0


THR_HB = HBThresholds()


# =============================================================================
# Step 1: 确定面瘫侧别 (前置条件)
# =============================================================================

def determine_palsy_side(action_results: Dict) -> Tuple[int, float, List[str]]:
    """
    综合所有动作结果确定面瘫侧别

    这是HB分级的前置步骤，因为后续所有评估都基于患侧/健侧对比

    Returns:
        (palsy_side, confidence, evidence_list)
        palsy_side: 0=无面瘫/对称, 1=左侧面瘫, 2=右侧面瘫
    """
    votes = []  # (side, weight, reason)
    evidence = []

    # 收集各动作的面瘫侧别判断
    for action_name, result in action_results.items():
        if result is None:
            continue

        action_spec = getattr(result, 'action_specific', {})
        palsy_detection = action_spec.get('palsy_detection', {})

        if palsy_detection:
            side = palsy_detection.get('palsy_side', 0)
            conf = palsy_detection.get('confidence', 0.5)

            if side != 0:
                # 根据动作类型给予不同权重
                weight = 1.0
                if action_name in ['CloseEyeHardly', 'CloseEyeSoftly']:
                    weight = 1.5  # 闭眼动作权重更高
                elif action_name in ['ShowTeeth', 'Smile']:
                    weight = 1.3
                elif action_name == 'RaiseEyebrow':
                    weight = 1.2

                votes.append((side, weight * conf, f"{action_name}: side={side}, conf={conf:.2f}"))
                evidence.append(f"{action_name} → {'左侧' if side == 1 else '右侧'}面瘫 (conf={conf:.2f})")

    if not votes:
        return 0, 0.0, ["未检测到面瘫证据"]

    # 加权投票
    left_score = sum(w for s, w, _ in votes if s == 1)
    right_score = sum(w for s, w, _ in votes if s == 2)
    total_score = left_score + right_score

    if total_score < 0.5:  # 置信度太低
        return 0, 0.0, evidence + ["总置信度过低，判定为对称"]

    if left_score > right_score * 1.5:
        confidence = left_score / total_score
        return 1, confidence, evidence
    elif right_score > left_score * 1.5:
        confidence = right_score / total_score
        return 2, confidence, evidence
    else:
        # 两侧得分接近，可能是对称或双侧
        return 0, 0.5, evidence + ["左右得分接近，判定为对称或双侧"]


# =============================================================================
# Step 2: a2 - At Rest (静息状态评估)
# =============================================================================

def compute_a2_at_rest(action_results: Dict, palsy_side: int) -> HBComponentGrade:
    """
    计算 a2: Gross function - At rest

    使用动作: NeutralFace

    评估指标:
    - 眼睑裂高度比 (palpebral_height_ratio)
    - 鼻唇沟长度比 (nlf_ratio)
    - 口角角度差 (oral_angle.angle_diff)

    Grade映射 (注意: 没有Grade III):
    - a2=1 → Grade I (Normal facial function in all areas)
    - a2=2 → Grade II (Normal symmetry and tone)
    - a2=4 → Grade IV (Slight asymmetry with normal tone)
    - a2=5 → Grade V (Asymmetry)
    - a2=6 → Grade VI (No movement - 面部松弛)
    """
    evidence = {}
    decision_path = []

    neutral = action_results.get("NeutralFace")
    if neutral is None:
        return HBComponentGrade(
            component="a2", component_cn="静息状态",
            grade=2, grade_description="Normal symmetry (缺少数据，默认)",
            evidence={"error": "NeutralFace数据缺失"},
            source_actions=[]
        )

    # 提取指标
    palp_ratio = getattr(neutral, 'palpebral_height_ratio', 1.0) or 1.0
    nlf_ratio = getattr(neutral, 'nlf_ratio', 1.0) or 1.0
    oral_angle = getattr(neutral, 'oral_angle', None)
    oral_diff = oral_angle.angle_diff if oral_angle else 0.0

    evidence = {
        "palpebral_height_ratio": palp_ratio,
        "nlf_ratio": nlf_ratio,
        "oral_angle_diff": oral_diff,
    }

    # 计算各指标的偏差
    palp_deviation = abs(palp_ratio - 1.0)
    nlf_deviation = abs(nlf_ratio - 1.0)

    decision_path.append(f"眼睑裂比={palp_ratio:.3f} (偏差{palp_deviation:.1%})")
    decision_path.append(f"鼻唇沟比={nlf_ratio:.3f} (偏差{nlf_deviation:.1%})")
    decision_path.append(f"口角角度差={oral_diff:.1f}°")

    # 判断逻辑
    # Grade VI: 面部完全松弛（极端情况）
    if palp_deviation > 0.40 and nlf_deviation > 0.40:
        grade = 6
        desc = "Asymmetry with flaccidity (严重不对称伴松弛)"
        decision_path.append("→ 严重不对称 → Grade VI")

    # Grade V: 明显不对称
    elif (palp_deviation > THR_HB.REST_PALP_MODERATE or
          nlf_deviation > THR_HB.REST_NLF_MODERATE or
          oral_diff > THR_HB.REST_ORAL_ANGLE_MODERATE):
        grade = 5
        desc = "Asymmetry (明显不对称)"
        decision_path.append("→ 明显不对称 → Grade V")

    # Grade IV: 轻度不对称但张力正常
    elif (palp_deviation > THR_HB.REST_PALP_MILD or
          nlf_deviation > THR_HB.REST_NLF_MILD or
          oral_diff > THR_HB.REST_ORAL_ANGLE_MILD):
        grade = 4
        desc = "Slight asymmetry with normal tone (轻度不对称，张力正常)"
        decision_path.append("→ 轻度不对称 → Grade IV")

    # Grade II: 基本对称，张力正常
    elif (palp_deviation > THR_HB.REST_PALP_NORMAL or
          nlf_deviation > THR_HB.REST_NLF_NORMAL or
          oral_diff > THR_HB.REST_ORAL_ANGLE_NORMAL):
        grade = 2
        desc = "Normal symmetry and tone (对称，张力正常)"
        decision_path.append("→ 接近对称 → Grade II")

    # Grade I: 完全正常
    else:
        grade = 1
        desc = "Normal facial function in all areas (完全正常)"
        decision_path.append("→ 完全对称 → Grade I")

    evidence["decision_path"] = decision_path

    return HBComponentGrade(
        component="a2", component_cn="静息状态",
        grade=grade, grade_description=desc,
        evidence=evidence,
        source_actions=["NeutralFace"]
    )


# =============================================================================
# Step 3: a3 - Forehead (额部运动评估)
# =============================================================================

def compute_a3_forehead(action_results: Dict, palsy_side: int) -> HBComponentGrade:
    """
    计算 a3: Motion - Forehead

    使用动作: RaiseEyebrow

    评估指标:
    - 眉眼距变化量 (left_change, right_change)
    - 变化量对称比 (min/max)

    Grade映射:
    - a3=1 → Grade I (Normal)
    - a3=2 → Grade II (Moderate to good)
    - a3=3 → Grade III (Slight to moderate)
    - a3=4 → Grade IV (No movement)
    """
    evidence = {}
    decision_path = []

    raise_eyebrow = action_results.get("RaiseEyebrow")
    if raise_eyebrow is None:
        return HBComponentGrade(
            component="a3", component_cn="额部运动",
            grade=2, grade_description="Moderate to good (缺少数据，默认)",
            evidence={"error": "RaiseEyebrow数据缺失"},
            source_actions=[]
        )

    action_spec = getattr(raise_eyebrow, 'action_specific', {})
    brow_metrics = action_spec.get('brow_eye_metrics', {})

    left_change = abs(brow_metrics.get('left_change', 0) or 0)
    right_change = abs(brow_metrics.get('right_change', 0) or 0)

    evidence = {
        "left_change": left_change,
        "right_change": right_change,
    }

    decision_path.append(f"左侧眉眼距变化={left_change:.1f}px")
    decision_path.append(f"右侧眉眼距变化={right_change:.1f}px")

    # 获取患侧和健侧的运动量
    if palsy_side == 1:
        affected_change, healthy_change = left_change, right_change
    elif palsy_side == 2:
        affected_change, healthy_change = right_change, left_change
    else:
        affected_change = min(left_change, right_change)
        healthy_change = max(left_change, right_change)

    evidence["affected_change"] = affected_change
    evidence["healthy_change"] = healthy_change

    # 计算对称比
    max_change = max(left_change, right_change)
    min_change = min(left_change, right_change)

    if max_change < THR_HB.FOREHEAD_MIN_MOVEMENT:
        # 双侧都几乎没有运动
        grade = 4
        desc = "No movement (无运动)"
        decision_path.append(f"→ 双侧运动幅度过小 (max={max_change:.1f}px) → Grade IV")
    else:
        symmetry_ratio = min_change / max_change
        evidence["symmetry_ratio"] = symmetry_ratio
        decision_path.append(f"对称比={symmetry_ratio:.1%}")

        if symmetry_ratio >= THR_HB.FOREHEAD_NORMAL:
            grade = 1
            desc = "Normal (正常)"
            decision_path.append("→ 高度对称 → Grade I")
        elif symmetry_ratio >= THR_HB.FOREHEAD_GOOD:
            grade = 2
            desc = "Moderate to good (中等至良好)"
            decision_path.append("→ 良好对称 → Grade II")
        elif symmetry_ratio >= THR_HB.FOREHEAD_SLIGHT:
            grade = 3
            desc = "Slight to moderate (轻度至中等)"
            decision_path.append("→ 轻度不对称 → Grade III")
        else:
            grade = 4
            desc = "No movement (患侧无运动)"
            decision_path.append("→ 严重不对称或患侧无运动 → Grade IV")

    evidence["decision_path"] = decision_path

    return HBComponentGrade(
        component="a3", component_cn="额部运动",
        grade=grade, grade_description=desc,
        evidence=evidence,
        source_actions=["RaiseEyebrow"]
    )


# =============================================================================
# Step 4: a4 - Eyes (眼睛闭合评估) - 最关键的分级依据
# =============================================================================

def compute_a4_eyes(action_results: Dict, palsy_side: int) -> HBComponentGrade:
    """
    计算 a4: Motion - Eyes

    使用动作: CloseEyeSoftly (轻闭眼) + CloseEyeHardly (用力闭眼)

    这是HB分级中最关键的指标！
    - 轻闭眼完全闭合 → 可能Grade I/II
    - 轻闭眼不完全但用力可闭合 → Grade III
    - 用力也不能完全闭合 → Grade IV或更高

    评估指标:
    - 患侧闭合度 (closure_ratio)
    - 闭合对称比

    Grade映射 (注意: 没有Grade V):
    - a4=1 → Grade I (Normal)
    - a4=2 → Grade II (Complete closure, minimum effort)
    - a4=3 → Grade III (Complete closure, effort)
    - a4=4 → Grade IV (Incomplete closure)
    - a4=6 → Grade VI (No movement)
    """
    evidence = {}
    decision_path = []

    soft_close = action_results.get("CloseEyeSoftly")
    hard_close = action_results.get("CloseEyeHardly")

    if soft_close is None and hard_close is None:
        return HBComponentGrade(
            component="a4", component_cn="眼睛",
            grade=2, grade_description="Complete closure, minimum effort (缺少数据，默认)",
            evidence={"error": "CloseEye数据缺失"},
            source_actions=[]
        )

    # 提取闭合度数据
    def get_closure_data(result):
        if result is None:
            return None, None, None, None

        action_spec = getattr(result, 'action_specific', {})

        # 优先从closure_metrics获取
        closure_metrics = action_spec.get('closure_metrics', {})
        if closure_metrics:
            left = closure_metrics.get('left_closure_ratio', 0) or 0
            right = closure_metrics.get('right_closure_ratio', 0) or 0
        else:
            # 备选从peak_frame获取
            peak_frame = action_spec.get('peak_frame', {})
            left = peak_frame.get('left_closure', 0) or 0
            right = peak_frame.get('right_closure', 0) or 0

        # 确定患侧和健侧
        if palsy_side == 1:
            affected, healthy = left, right
        elif palsy_side == 2:
            affected, healthy = right, left
        else:
            affected = min(left, right)
            healthy = max(left, right)

        return left, right, affected, healthy

    soft_left, soft_right, soft_affected, soft_healthy = get_closure_data(soft_close)
    hard_left, hard_right, hard_affected, hard_healthy = get_closure_data(hard_close)

    evidence = {
        "soft_closure": {"left": soft_left, "right": soft_right,
                         "affected": soft_affected, "healthy": soft_healthy},
        "hard_closure": {"left": hard_left, "right": hard_right,
                         "affected": hard_affected, "healthy": hard_healthy},
    }

    if soft_affected is not None:
        decision_path.append(f"轻闭眼: 患侧闭合度={soft_affected:.1%}, 健侧={soft_healthy:.1%}")
    if hard_affected is not None:
        decision_path.append(f"用力闭眼: 患侧闭合度={hard_affected:.1%}, 健侧={hard_healthy:.1%}")

    # ========== 核心判断逻辑 ==========

    # 情况1: 有用力闭眼数据
    if hard_affected is not None:

        # Grade VI: 用力闭眼也几乎没有运动
        if hard_affected < THR_HB.EYE_CLOSURE_MINIMAL:
            grade = 6
            desc = "No movement (无运动)"
            decision_path.append(f"→ 用力闭眼患侧闭合度极低 ({hard_affected:.1%}) → Grade VI")

        # Grade IV: 用力也不能完全闭合
        elif hard_affected < THR_HB.EYE_CLOSURE_COMPLETE:
            grade = 4
            desc = "Incomplete closure (不能完全闭合)"
            decision_path.append(f"→ 用力闭眼患侧仍不完全 ({hard_affected:.1%} < 85%) → Grade IV")

        # 用力可以完全闭合
        else:
            # 检查轻闭眼情况来区分 Grade II vs III
            if soft_affected is not None:
                if soft_affected >= THR_HB.EYE_CLOSURE_COMPLETE:
                    # 轻闭眼就能完全闭合
                    # 进一步检查对称性
                    if soft_healthy > 1e-6:
                        symmetry = soft_affected / soft_healthy
                    else:
                        symmetry = 1.0

                    if symmetry >= THR_HB.EYE_SYMMETRY_NORMAL:
                        grade = 1
                        desc = "Normal (正常)"
                        decision_path.append(f"→ 轻闭眼完全且对称 (对称比={symmetry:.1%}) → Grade I")
                    else:
                        grade = 2
                        desc = "Complete closure, minimum effort (完全闭合，轻微用力)"
                        decision_path.append(f"→ 轻闭眼完全但略不对称 → Grade II")
                else:
                    # 轻闭眼不完全，但用力可以闭合 → Grade III
                    grade = 3
                    desc = "Complete closure, effort (完全闭合，需用力)"
                    decision_path.append(
                        f"→ 轻闭眼不完全 ({soft_affected:.1%})，用力可闭合 ({hard_affected:.1%}) → Grade III"
                    )
            else:
                # 没有轻闭眼数据，仅基于用力闭眼判断
                if hard_healthy > 1e-6:
                    symmetry = hard_affected / hard_healthy
                else:
                    symmetry = 1.0

                if symmetry >= THR_HB.EYE_SYMMETRY_NORMAL:
                    grade = 2
                    desc = "Complete closure, minimum effort (完全闭合)"
                    decision_path.append(f"→ 用力闭眼完全且对称 → Grade II")
                else:
                    grade = 3
                    desc = "Complete closure, effort (完全闭合，略不对称)"
                    decision_path.append(f"→ 用力闭眼完全但不对称 → Grade III")

    # 情况2: 只有轻闭眼数据
    elif soft_affected is not None:
        if soft_affected < THR_HB.EYE_CLOSURE_MINIMAL:
            grade = 6
            desc = "No movement (无运动)"
            decision_path.append(f"→ 轻闭眼患侧几乎无运动 → Grade VI")
        elif soft_affected < THR_HB.EYE_CLOSURE_PARTIAL:
            grade = 4
            desc = "Incomplete closure (不能完全闭合)"
            decision_path.append(f"→ 轻闭眼患侧闭合度低 ({soft_affected:.1%}) → Grade IV")
        elif soft_affected < THR_HB.EYE_CLOSURE_COMPLETE:
            grade = 3
            desc = "Complete closure, effort (需用力才能闭合)"
            decision_path.append(f"→ 轻闭眼不完全 ({soft_affected:.1%}) → Grade III")
        else:
            if soft_healthy > 1e-6:
                symmetry = soft_affected / soft_healthy
            else:
                symmetry = 1.0

            if symmetry >= THR_HB.EYE_SYMMETRY_NORMAL:
                grade = 1
                desc = "Normal (正常)"
                decision_path.append("→ 轻闭眼完全且对称 → Grade I")
            else:
                grade = 2
                desc = "Complete closure, minimum effort (轻微不对称)"
                decision_path.append("→ 轻闭眼完全但略不对称 → Grade II")

    else:
        grade = 2
        desc = "Complete closure (缺少数据，默认)"
        decision_path.append("→ 缺少闭眼数据 → 默认Grade II")

    evidence["decision_path"] = decision_path

    source_actions = []
    if soft_close is not None:
        source_actions.append("CloseEyeSoftly")
    if hard_close is not None:
        source_actions.append("CloseEyeHardly")

    return HBComponentGrade(
        component="a4", component_cn="眼睛",
        grade=grade, grade_description=desc,
        evidence=evidence,
        source_actions=source_actions
    )


# =============================================================================
# Step 5: a5 - Mouth (口部运动评估)
# =============================================================================

def compute_a5_mouth(action_results: Dict, palsy_side: int) -> HBComponentGrade:
    """
    计算 a5: Motion - Mouth

    使用动作: ShowTeeth (优先) 或 Smile
    辅助动作: LipPucker, BlowCheek, ShrugNose

    评估指标:
    - 嘴角运动幅度对称比 (excursion_ratio)
    - 口角角度不对称度
    - 嘴唇中心偏移

    Grade映射:
    - a5=1 → Grade I (Normal)
    - a5=2 → Grade II (Slight asymmetry)
    - a5=3 → Grade III (Slightly weak with maximum effort)
    - a5=4 → Grade IV (Asymmetric with maximum effort)
    - a5=5 → Grade V (Slight movement)
    - a5=6 → Grade VI (No movement)
    """
    evidence = {}
    decision_path = []

    # 优先使用ShowTeeth，其次Smile
    primary_result = action_results.get("ShowTeeth") or action_results.get("Smile")

    if primary_result is None:
        return HBComponentGrade(
            component="a5", component_cn="口部",
            grade=2, grade_description="Slight asymmetry (缺少数据，默认)",
            evidence={"error": "ShowTeeth/Smile数据缺失"},
            source_actions=[]
        )

    action_name = primary_result.action_name
    action_spec = getattr(primary_result, 'action_specific', {})

    # 提取运动幅度数据
    excursion = action_spec.get('excursion', {})
    eye_line_exc = action_spec.get('eye_line_excursion', {})

    # 方法1: 使用excursion_ratio
    exc_ratio = excursion.get('excursion_ratio')

    # 方法2: 使用eye_line_excursion
    if exc_ratio is None and eye_line_exc:
        left_red = abs(eye_line_exc.get('left_reduction', 0) or 0)
        right_red = abs(eye_line_exc.get('right_reduction', 0) or 0)

        if palsy_side == 1:
            affected_exc, healthy_exc = left_red, right_red
        elif palsy_side == 2:
            affected_exc, healthy_exc = right_red, left_red
        else:
            affected_exc = min(left_red, right_red)
            healthy_exc = max(left_red, right_red)

        if healthy_exc > 1e-6:
            exc_ratio = affected_exc / healthy_exc
        else:
            exc_ratio = 1.0 if affected_exc < 1e-6 else 0.0

        evidence["eye_line_excursion"] = {
            "left_reduction": left_red,
            "right_reduction": right_red,
            "affected": affected_exc,
            "healthy": healthy_exc,
        }

    # 方法3: 使用口角角度
    oral_angle = getattr(primary_result, 'oral_angle', None)
    if oral_angle:
        oral_diff = oral_angle.angle_diff
        oral_asym = oral_angle.angle_asymmetry
        evidence["oral_angle"] = {
            "angle_diff": oral_diff,
            "asymmetry": oral_asym,
        }
    else:
        oral_diff = 0
        oral_asym = 0

    evidence["excursion_ratio"] = exc_ratio

    if exc_ratio is not None:
        decision_path.append(f"运动幅度对称比={exc_ratio:.1%}")
    decision_path.append(f"口角角度差={oral_diff:.1f}°")

    # ========== 判断逻辑 ==========

    # 综合评判：优先使用excursion_ratio，辅以oral_angle
    if exc_ratio is not None:
        if exc_ratio >= THR_HB.MOUTH_SYMMETRY_NORMAL:
            grade = 1
            desc = "Normal (正常)"
            decision_path.append("→ 高度对称 → Grade I")
        elif exc_ratio >= THR_HB.MOUTH_SYMMETRY_SLIGHT:
            grade = 2
            desc = "Slight asymmetry (轻微不对称)"
            decision_path.append("→ 轻微不对称 → Grade II")
        elif exc_ratio >= THR_HB.MOUTH_SYMMETRY_WEAK:
            grade = 3
            desc = "Slightly weak with maximum effort (用力时轻度无力)"
            decision_path.append("→ 明显不对称 → Grade III")
        elif exc_ratio >= THR_HB.MOUTH_SYMMETRY_ASYM:
            grade = 4
            desc = "Asymmetric with maximum effort (用力时明显不对称)"
            decision_path.append("→ 严重不对称 → Grade IV")
        elif exc_ratio >= THR_HB.MOUTH_SYMMETRY_MINIMAL:
            grade = 5
            desc = "Slight movement (仅有轻微运动)"
            decision_path.append("→ 仅有微量运动 → Grade V")
        else:
            grade = 6
            desc = "No movement (无运动)"
            decision_path.append("→ 几乎无运动 → Grade VI")

    # 备选：使用口角角度判断
    else:
        if oral_diff < THR_HB.MOUTH_ANGLE_NORMAL:
            grade = 1
            desc = "Normal (正常)"
            decision_path.append("→ 口角对称 → Grade I")
        elif oral_diff < THR_HB.MOUTH_ANGLE_SLIGHT:
            grade = 2
            desc = "Slight asymmetry (轻微不对称)"
            decision_path.append("→ 口角轻微不对称 → Grade II")
        elif oral_diff < THR_HB.MOUTH_ANGLE_WEAK:
            grade = 3
            desc = "Slightly weak (轻度无力)"
            decision_path.append("→ 口角明显不对称 → Grade III")
        elif oral_diff < THR_HB.MOUTH_ANGLE_ASYM:
            grade = 4
            desc = "Asymmetric (明显不对称)"
            decision_path.append("→ 口角严重不对称 → Grade IV")
        else:
            grade = 5
            desc = "Slight movement (仅有轻微运动)"
            decision_path.append("→ 口角极度不对称 → Grade V")

    evidence["decision_path"] = decision_path

    source_actions = [action_name]

    return HBComponentGrade(
        component="a5", component_cn="口部",
        grade=grade, grade_description=desc,
        evidence=evidence,
        source_actions=source_actions
    )


# =============================================================================
# Step 6: a1 - General (整体功能评估) - 综合所有动作
# =============================================================================

def compute_a1_general(
        action_results: Dict,
        palsy_side: int,
        a2_grade: int,
        a3_grade: int,
        a4_grade: int,
        a5_grade: int
) -> HBComponentGrade:
    """
    计算 a1: Gross function - General

    这是整体印象评分，综合考虑:
    1. 其他4个评估项的结果
    2. 联动运动(synkinesis)的严重程度
    3. 面部挛缩(contracture)的存在
    4. 整体视觉印象

    Grade映射:
    - a1=1 → Grade I (Normal facial function in all areas)
    - a1=2 → Grade II (Slight weakness on close inspection; may have very slight synkinesis)
    - a1=3 → Grade III (Obvious but not disfiguring; noticeable synkinesis/contracture)
    - a1=4 → Grade IV (Obvious weakness and/or disfiguring asymmetry)
    - a1=5 → Grade V (Only barely perceptible motion)
    - a1=6 → Grade VI (No movement)
    """
    evidence = {}
    decision_path = []

    # 基础等级 = 其他4项的最大值
    base_grade = max(a2_grade, a3_grade, a4_grade, a5_grade)
    evidence["component_grades"] = {
        "a2_at_rest": a2_grade,
        "a3_forehead": a3_grade,
        "a4_eyes": a4_grade,
        "a5_mouth": a5_grade,
    }
    evidence["base_grade"] = base_grade
    decision_path.append(f"组件等级: a2={a2_grade}, a3={a3_grade}, a4={a4_grade}, a5={a5_grade}")
    decision_path.append(f"基础等级 = max = {base_grade}")

    # 检查联动运动
    total_synkinesis = 0
    synkinesis_details = {}

    for action_name, result in action_results.items():
        if result is None:
            continue
        syn_scores = getattr(result, 'synkinesis_scores', {})
        if syn_scores:
            eye_syn = syn_scores.get('eye_synkinesis', 0) or 0
            mouth_syn = syn_scores.get('mouth_synkinesis', 0) or 0
            max_syn = max(eye_syn, mouth_syn)
            if max_syn > 0:
                synkinesis_details[action_name] = {
                    "eye": eye_syn, "mouth": mouth_syn, "max": max_syn
                }
                total_synkinesis += max_syn

    evidence["synkinesis"] = synkinesis_details

    # 联动运动对等级的影响
    avg_synkinesis = total_synkinesis / max(len(synkinesis_details), 1) if synkinesis_details else 0

    if avg_synkinesis >= 2.5:
        decision_path.append(f"严重联动运动 (avg={avg_synkinesis:.1f})")
        # 严重联动不会降低等级，但会限制恢复到Grade I
        if base_grade <= 2:
            base_grade = max(base_grade, 2)
            decision_path.append("→ 联动限制等级 ≥ II")
    elif avg_synkinesis >= 1.5:
        decision_path.append(f"中度联动运动 (avg={avg_synkinesis:.1f})")
    elif avg_synkinesis >= 0.5:
        decision_path.append(f"轻度联动运动 (avg={avg_synkinesis:.1f})")

    grade = base_grade

    # 等级描述
    descriptions = {
        1: "Normal facial function in all areas (所有区域功能正常)",
        2: "Slight weakness on close inspection (仔细观察可见轻微无力)",
        3: "Obvious but not disfiguring difference (明显但不毁容的差异)",
        4: "Obvious weakness and/or disfiguring asymmetry (明显无力或毁容性不对称)",
        5: "Only barely perceptible motion (仅有勉强可见的运动)",
        6: "No movement (无运动)",
    }

    desc = descriptions.get(grade, f"Grade {grade}")
    decision_path.append(f"→ 最终a1等级 = {grade}")
    evidence["decision_path"] = decision_path

    return HBComponentGrade(
        component="a1", component_cn="整体功能",
        grade=grade, grade_description=desc,
        evidence=evidence,
        source_actions=list(action_results.keys())
    )


# =============================================================================
# 主函数: 计算完整HB分级
# =============================================================================

def compute_hb_grade(action_results: Dict) -> HBGradingResult:
    """
    计算完整的House-Brackmann分级

    处理流程:
    1. 确定面瘫侧别 (prerequisite)
    2. 计算 a2: At Rest (静息状态)
    3. 计算 a3: Forehead (额部运动)
    4. 计算 a4: Eyes (眼睛) - 最关键
    5. 计算 a5: Mouth (口部)
    6. 计算 a1: General (整体功能)
    7. 最终HB等级 = max(a1, a2, a3, a4, a5)

    Args:
        action_results: Dict[str, ActionResult] - 所有动作的处理结果

    Returns:
        HBGradingResult - 完整的HB分级结果
    """
    decision_path = []

    # ========== Step 1: 确定面瘫侧别 ==========
    palsy_side, side_confidence, side_evidence = determine_palsy_side(action_results)
    decision_path.append(f"Step 1: 面瘫侧别 = {['无/对称', '左侧', '右侧'][palsy_side]} (conf={side_confidence:.2f})")
    decision_path.extend([f"  - {e}" for e in side_evidence[:3]])  # 最多显示3条证据

    # ========== Step 2: a2 - At Rest ==========
    a2_result = compute_a2_at_rest(action_results, palsy_side)
    decision_path.append(f"Step 2: a2 (静息) = Grade {a2_result.grade}")

    # ========== Step 3: a3 - Forehead ==========
    a3_result = compute_a3_forehead(action_results, palsy_side)
    decision_path.append(f"Step 3: a3 (额部) = Grade {a3_result.grade}")

    # ========== Step 4: a4 - Eyes (最关键) ==========
    a4_result = compute_a4_eyes(action_results, palsy_side)
    decision_path.append(f"Step 4: a4 (眼睛) = Grade {a4_result.grade} ★")

    # ========== Step 5: a5 - Mouth ==========
    a5_result = compute_a5_mouth(action_results, palsy_side)
    decision_path.append(f"Step 5: a5 (口部) = Grade {a5_result.grade}")

    # ========== Step 6: a1 - General ==========
    a1_result = compute_a1_general(
        action_results, palsy_side,
        a2_result.grade, a3_result.grade, a4_result.grade, a5_result.grade
    )
    decision_path.append(f"Step 6: a1 (整体) = Grade {a1_result.grade}")

    # ========== Step 7: 最终HB等级 ==========
    final_grade = max(
        a1_result.grade,
        a2_result.grade,
        a3_result.grade,
        a4_result.grade,
        a5_result.grade
    )

    # 确定最差的评估项
    worst_components = []
    for name, result in [
        ("a1", a1_result), ("a2", a2_result),
        ("a3", a3_result), ("a4", a4_result), ("a5", a5_result)
    ]:
        if result.grade == final_grade:
            worst_components.append(f"{name}({result.component_cn})")

    decision_path.append("=" * 50)
    decision_path.append(f"最终HB等级 = max(a1,a2,a3,a4,a5) = Grade {final_grade}")
    decision_path.append(f"最差评估项: {', '.join(worst_components)}")

    final_description = HB_DESCRIPTIONS.get(final_grade, f"Grade {final_grade}")

    return HBGradingResult(
        final_grade=final_grade,
        final_description=final_description,
        a1_general=a1_result,
        a2_at_rest=a2_result,
        a3_forehead=a3_result,
        a4_eyes=a4_result,
        a5_mouth=a5_result,
        palsy_side=palsy_side,
        confidence=side_confidence,
        decision_path=decision_path
    )


# =============================================================================
# 便捷函数: 打印HB分级结果
# =============================================================================

def print_hb_result(result: HBGradingResult):
    """打印HB分级结果"""
    print("\n" + "=" * 70)
    print("House-Brackmann 分级结果")
    print("=" * 70)

    print(f"\n【最终等级】HB Grade {result.final_grade}")
    print(f"  {result.final_description}")

    print(f"\n【面瘫侧别】{['无面瘫/对称', '左侧面瘫', '右侧面瘫'][result.palsy_side]}")
    print(f"  置信度: {result.confidence:.1%}")

    print("\n【各评估项详情】")
    print("-" * 50)

    for comp in [result.a1_general, result.a2_at_rest,
                 result.a3_forehead, result.a4_eyes, result.a5_mouth]:
        marker = "★" if comp.grade == result.final_grade else " "
        print(f"  {marker} {comp.component} ({comp.component_cn}): Grade {comp.grade}")
        print(f"      {comp.grade_description}")
        print(f"      来源动作: {', '.join(comp.source_actions)}")

    print("\n【决策路径】")
    print("-" * 50)
    for step in result.decision_path:
        print(f"  {step}")


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    # 模拟测试数据
    print("HB Grading Module")
    print("请在 session_diagnosis.py 中调用 compute_hb_grade() 函数")