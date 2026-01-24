#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Session级诊断模块 - Session Diagnosis Module
=============================================

综合11个动作的分析结果，输出Session级诊断：
- has_palsy: 是否面瘫
- palsy_side: 面瘫侧别 (0=无, 1=左, 2=右)
- hb_grade: House-Brackmann分级 (1-6)
- sunnybrook_score: Sunnybrook综合评分 (0-100)

包含临床规则约束，确保诊断结果的医学一致性。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import json
from thresholds import THR

# 导入HB分级模块
from hb_grading import (
    compute_hb_grade,
    HBGradingResult,
    print_hb_result
)

# =============================================================================
# 常量定义
# =============================================================================

# HB分级描述
HB_DESCRIPTIONS = {
    1: "Normal (正常)",
    2: "Slight dysfunction (轻度功能障碍)",
    3: "Moderate dysfunction (中度功能障碍)",
    4: "Moderately severe dysfunction (中重度功能障碍)",
    5: "Severe dysfunction (重度功能障碍)",
    6: "Total paralysis (完全麻痹)",
}

# 严重度描述
SEVERITY_DESCRIPTIONS = {
    1: "Normal (正常)",
    2: "Mild (轻度)",
    3: "Moderate (中度)",
    4: "Mod-Severe (中重度)",
    5: "Severe (重度)",
}

# 动作权重（用于投票）
ACTION_WEIGHTS = {
    "NeutralFace": 0.0,  # 静息只作参考
    "SpontaneousEyeBlink": 1,
    "VoluntaryEyeBlink": 1,
    "CloseEyeSoftly": 1,
    "CloseEyeHardly": 0.5,
    "RaiseEyebrow": 1.3,
    "Smile": 0.5,
    "ShrugNose": 0.5,
    "ShowTeeth": 1.5,
    "BlowCheek": 0.5,
    "LipPucker": 0.5,
}

# 动作对应的面部区域
ACTION_REGIONS = {
    "SpontaneousEyeBlink": "眼",
    "VoluntaryEyeBlink": "眼",
    "CloseEyeSoftly": "眼",
    "CloseEyeHardly": "眼",
    "RaiseEyebrow": "额",
    "Smile": "口",
    "ShowTeeth": "口",
    "ShrugNose": "中面",
    "BlowCheek": "中面",
    "LipPucker": "口",
    "NeutralFace": "静息",
}


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class ConsistencyCheck:
    """一致性检查结果"""
    rule_id: str
    rule_name: str
    passed: bool
    message: str
    severity: str = "info"  # info, warning, error


@dataclass
class VoteRecord:
    """投票记录"""
    action: str
    action_cn: str
    side: int  # 0=中立, 1=左弱, 2=右弱
    side_text: str
    confidence: float
    weight: float
    region: str
    reason: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def weighted_score(self) -> float:
        return self.weight * self.confidence


@dataclass
class SessionDiagnosis:
    """Session级诊断结果"""

    # === 核心诊断 ===
    has_palsy: bool
    palsy_side: int  # 0=无, 1=左, 2=右
    palsy_side_text: str
    hb_grade: int  # 1-6
    hb_description: str
    sunnybrook_score: int  # Composite score (0-100)

    # === 置信度 ===
    confidence: float  # 整体置信度 0-1
    palsy_side_confidence: float

    # === Sunnybrook详情 ===
    resting_score: int = 0  # 0-20
    voluntary_score: int = 0  # 20-100
    synkinesis_score: int = 0  # 0-15

    # === 投票与证据 ===
    left_score: float = 0.0  # 左侧累计证据
    right_score: float = 0.0  # 右侧累计证据
    votes: List[VoteRecord] = field(default_factory=list)
    top_evidence: List[VoteRecord] = field(default_factory=list)

    # === 一致性检查 ===
    consistency_checks: List[ConsistencyCheck] = field(default_factory=list)
    adjustments_made: List[str] = field(default_factory=list)

    # === HB 分级证据 ===
    hb_evidence: Dict[str, Any] = field(default_factory=dict)

    # === HB 组件分级 ===
    hb_component_grades: Dict[str, Any] = field(default_factory=dict)
    hb_worst_component: str = ""
    hb_decision_path: List[str] = field(default_factory=list)

    # === 可解释性 ===
    interpretation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转为字典（用于JSON序列化）"""
        return {
            "has_palsy": self.has_palsy,
            "palsy_side": self.palsy_side,
            "palsy_side_text": self.palsy_side_text,
            "hb_grade": self.hb_grade,
            "hb_description": self.hb_description,
            "sunnybrook_score": self.sunnybrook_score,
            "confidence": self.confidence,
            "palsy_side_confidence": self.palsy_side_confidence,
            "resting_score": self.resting_score,
            "voluntary_score": self.voluntary_score,
            "synkinesis_score": self.synkinesis_score,
            "left_score": self.left_score,
            "right_score": self.right_score,
            "votes": [
                {
                    "action": v.action,
                    "action_cn": v.action_cn,
                    "side": v.side,
                    "side_text": v.side_text,
                    "confidence": v.confidence,
                    "weight": v.weight,
                    "region": v.region,
                    "reason": v.reason,
                }
                for v in self.votes
            ],
            "top_evidence": [
                {
                    "action": v.action,
                    "action_cn": v.action_cn,
                    "side_text": v.side_text,
                    "region": v.region,
                    "reason": v.reason,
                    "weighted_score": v.weighted_score,
                }
                for v in self.top_evidence
            ],
            "hb_evidence": self.hb_evidence,
            "hb_component_grades": self.hb_component_grades,
            "hb_worst_component": self.hb_worst_component,
            "hb_decision_path": self.hb_decision_path,
            "consistency_checks": [
                {
                    "rule_id": c.rule_id,
                    "rule_name": c.rule_name,
                    "passed": c.passed,
                    "message": c.message,
                    "severity": c.severity,
                }
                for c in self.consistency_checks
            ],
            "adjustments_made": self.adjustments_made,
            "interpretation": self.interpretation,
        }


# =============================================================================
# Sunnybrook 相关计算
# =============================================================================

def calculate_sunnybrook_from_results(action_results: Dict, palsy_side: int):
    """
    从动作结果计算Sunnybrook评分

    这是一个简化版本，返回一个带有必要属性的对象

    Args:
        action_results: 动作结果字典
        palsy_side: 面瘫侧别 (0=无, 1=左, 2=右)

    Returns:
        SunnybrookScore对象（简化版）
    """

    @dataclass
    class RestingSymmetry:
        total_score: int = 0

    @dataclass
    class VoluntaryMovement:
        total_score: int = 0

    @dataclass
    class Synkinesis:
        total_score: int = 0

    @dataclass
    class SunnybrookScore:
        resting_symmetry: RestingSymmetry = field(default_factory=RestingSymmetry)
        voluntary_movement: VoluntaryMovement = field(default_factory=VoluntaryMovement)
        synkinesis: Synkinesis = field(default_factory=Synkinesis)
        composite_score: int = 100

    if palsy_side == 0:
        # 无面瘫，返回满分
        return SunnybrookScore(
            resting_symmetry=RestingSymmetry(total_score=0),
            voluntary_movement=VoluntaryMovement(total_score=100),
            synkinesis=Synkinesis(total_score=0),
            composite_score=100
        )

    # === 计算 Resting Symmetry (0-20, 越高越差) ===
    resting_score = 0
    neutral = action_results.get("NeutralFace")
    if neutral:
        # 检查静息状态不对称
        palp_ratio = getattr(neutral, 'palpebral_height_ratio', 1.0) or 1.0
        nlf_ratio = getattr(neutral, 'nlf_ratio', 1.0) or 1.0
        oral_angle = getattr(neutral, 'oral_angle', None)

        palp_dev = abs(palp_ratio - 1.0)
        nlf_dev = abs(nlf_ratio - 1.0)
        oral_diff = oral_angle.angle_diff if oral_angle else 0

        # 眼睑裂不对称 (0-4)
        if palp_dev > 0.25:
            resting_score += 4
        elif palp_dev > 0.15:
            resting_score += 2
        elif palp_dev > 0.08:
            resting_score += 1

        # 鼻唇沟不对称 (0-8)
        if nlf_dev > 0.30:
            resting_score += 8
        elif nlf_dev > 0.20:
            resting_score += 4
        elif nlf_dev > 0.10:
            resting_score += 2

        # 口角不对称 (0-8)
        if oral_diff > 15:
            resting_score += 8
        elif oral_diff > 10:
            resting_score += 4
        elif oral_diff > 5:
            resting_score += 2

    # === 计算 Voluntary Movement (20-100, 越高越好) ===
    voluntary_score = 0
    voluntary_items = 0

    # Brow (RaiseEyebrow) - 最高20分
    raise_eyebrow = action_results.get("RaiseEyebrow")
    if raise_eyebrow:
        vol_score = getattr(raise_eyebrow, 'voluntary_movement_score', 3) or 3
        voluntary_score += vol_score * 4  # 1-5 → 4-20
        voluntary_items += 1

    # Eye closure (CloseEyeSoftly) - 最高20分
    close_eye = action_results.get("CloseEyeSoftly") or action_results.get("CloseEyeHardly")
    if close_eye:
        vol_score = getattr(close_eye, 'voluntary_movement_score', 3) or 3
        voluntary_score += vol_score * 4
        voluntary_items += 1

    # Smile/ShowTeeth - 最高20分
    smile = action_results.get("ShowTeeth") or action_results.get("Smile")
    if smile:
        vol_score = getattr(smile, 'voluntary_movement_score', 3) or 3
        voluntary_score += vol_score * 4
        voluntary_items += 1

    # Snarl (ShrugNose) - 最高20分
    snarl = action_results.get("ShrugNose")
    if snarl:
        vol_score = getattr(snarl, 'voluntary_movement_score', 3) or 3
        voluntary_score += vol_score * 4
        voluntary_items += 1

    # LipPucker - 最高20分
    pucker = action_results.get("LipPucker")
    if pucker:
        vol_score = getattr(pucker, 'voluntary_movement_score', 3) or 3
        voluntary_score += vol_score * 4
        voluntary_items += 1

    # 归一化到20-100范围
    if voluntary_items > 0:
        voluntary_score = int(20 + (voluntary_score / voluntary_items) * 16)
    else:
        voluntary_score = 60  # 默认中等

    voluntary_score = min(100, max(20, voluntary_score))

    # === 计算 Synkinesis (0-15, 越高越差) ===
    synkinesis_score = 0
    for action_name, result in action_results.items():
        if result is None:
            continue
        syn_scores = getattr(result, 'synkinesis_scores', {}) or {}
        eye_syn = syn_scores.get('eye_synkinesis', 0) or 0
        mouth_syn = syn_scores.get('mouth_synkinesis', 0) or 0
        synkinesis_score += max(eye_syn, mouth_syn)

    synkinesis_score = min(15, synkinesis_score)

    # === Composite Score ===
    composite = voluntary_score - resting_score - synkinesis_score
    composite = max(0, min(100, composite))

    return SunnybrookScore(
        resting_symmetry=RestingSymmetry(total_score=resting_score),
        voluntary_movement=VoluntaryMovement(total_score=voluntary_score),
        synkinesis=Synkinesis(total_score=synkinesis_score),
        composite_score=composite
    )


# =============================================================================
# 核心计算函数
# =============================================================================
def sunnybrook_to_hb(composite_score: int) -> int:
    """
    Sunnybrook Composite Score → House-Brackmann Grade (简化映射)

    仅作为兜底参考，主要 HB 分级由 compute_hb_grade_clinical 决策树计算

    映射关系（基于临床文献）:
    - 90-100 → HB I  (正常)
    - 70-89  → HB II (轻度)
    - 50-69  → HB III(中度)
    - 30-49  → HB IV (中重度)
    - 10-29  → HB V  (重度)
    - <10    → HB VI (完全麻痹)
    """
    if composite_score >= 90:
        return 1
    elif composite_score >= 70:
        return 2
    elif composite_score >= 50:
        return 3
    elif composite_score >= 30:
        return 4
    elif composite_score >= 10:
        return 5
    else:
        return 6


def hb_to_sunnybrook_range(hb_grade: int) -> Tuple[int, int]:
    """HB分级 → Sunnybrook分数范围"""
    ranges = {
        1: (90, 100),
        2: (70, 89),
        3: (50, 69),
        4: (30, 49),
        5: (10, 29),
        6: (0, 9),
    }
    return ranges.get(hb_grade, (0, 100))


def voluntary_to_severity(voluntary_score: int) -> int:
    """
    Voluntary Movement Score (1-5) → Severity Score (1-5)

    互为倒数关系:
    - voluntary=5 (运动完整) → severity=1 (正常)
    - voluntary=1 (无法启动) → severity=5 (重度)
    """
    return max(1, min(5, 6 - voluntary_score))


def severity_to_voluntary(severity_score: int) -> int:
    """Severity Score → Voluntary Movement Score"""
    return max(1, min(5, 6 - severity_score))


def palsy_side_to_text(side: int) -> str:
    """患侧代码转文本"""
    return {0: "无", 1: "左", 2: "右"}.get(side, "无")


def get_action_cn_name(action: str) -> str:
    """获取动作中文名"""
    names = {
        "NeutralFace": "静息面",
        "SpontaneousEyeBlink": "自然眨眼",
        "VoluntaryEyeBlink": "自主眨眼",
        "CloseEyeSoftly": "轻闭眼",
        "CloseEyeHardly": "用力闭眼",
        "RaiseEyebrow": "抬眉",
        "Smile": "微笑",
        "ShowTeeth": "露齿",
        "ShrugNose": "皱鼻",
        "BlowCheek": "鼓腮",
        "LipPucker": "撅嘴",
    }
    return names.get(action, action)


# =============================================================================
# 投票算法
# =============================================================================

def collect_votes(action_results: Dict) -> List[VoteRecord]:
    """
    从各动作结果中收集患侧投票

    Args:
        action_results: {action_name: ActionResult}

    Returns:
        投票记录列表
    """
    votes = []

    for action_name, result in action_results.items():
        if result is None:
            continue

        # 获取action_specific中的palsy_detection
        action_specific = getattr(result, 'action_specific', None) or {}
        palsy_det = action_specific.get('palsy_detection', {})

        if not palsy_det:
            continue

        side = palsy_det.get('palsy_side', 0)
        confidence = palsy_det.get('confidence', 0.0)
        method = palsy_det.get('method', '')
        interpretation = palsy_det.get('interpretation', '')
        evidence = palsy_det.get('evidence', {})

        # 只收集有方向判断的投票
        if side != 0 and confidence > 0.05:
            vote = VoteRecord(
                action=action_name,
                action_cn=get_action_cn_name(action_name),
                side=side,
                side_text=palsy_side_to_text(side),
                confidence=confidence,
                weight=ACTION_WEIGHTS.get(action_name, 1.0),
                region=ACTION_REGIONS.get(action_name, "其他"),
                reason=f"{method}: {interpretation}" if method else interpretation,
                evidence=evidence,
            )
            votes.append(vote)

    return votes


def aggregate_votes(votes: List[VoteRecord]) -> Tuple[float, float, int, float]:
    """
    汇总投票结果

    Returns:
        (left_score, right_score, final_side, confidence)
    """
    left_score = 0.0
    right_score = 0.0

    for v in votes:
        if v.side == 1:
            left_score += v.weighted_score
        elif v.side == 2:
            right_score += v.weighted_score

    total = left_score + right_score

    # 确定最终患侧
    if total < 0.3:
        # 证据不足，判为无面瘫
        final_side = 0
        confidence = 1.0 - total
    else:
        if left_score > right_score * 1.2:
            final_side = 1
        elif right_score > left_score * 1.2:
            final_side = 2
        else:
            final_side = 0  # 不确定

        # 置信度 = 优势比
        confidence = abs(left_score - right_score) / max(total, 1e-9)
        confidence = min(1.0, confidence)

    return left_score, right_score, final_side, confidence


# =============================================================================
# 一致性检查
# =============================================================================

def check_consistency(
        has_palsy: bool,
        palsy_side: int,
        hb_grade: int,
        sunnybrook_score: int,
        votes: List[VoteRecord]
) -> Tuple[List[ConsistencyCheck], List[str]]:
    """
    检查诊断结果的一致性

    Returns:
        (检查结果列表, 调整说明列表)
    """
    checks = []
    adjustments = []

    # R1: 健康人标签一致性
    if not has_palsy:
        if palsy_side != 0:
            checks.append(ConsistencyCheck(
                rule_id="R1a",
                rule_name="健康人患侧",
                passed=False,
                message=f"has_palsy=False 但 palsy_side={palsy_side}，已调整为0",
                severity="warning"
            ))
            adjustments.append("R1a: palsy_side 调整为 0")
        else:
            checks.append(ConsistencyCheck(
                rule_id="R1a",
                rule_name="健康人患侧",
                passed=True,
                message="palsy_side=0 ✓"
            ))

        if hb_grade != 1:
            checks.append(ConsistencyCheck(
                rule_id="R1b",
                rule_name="健康人HB",
                passed=False,
                message=f"has_palsy=False 但 hb_grade={hb_grade}，已调整为1",
                severity="warning"
            ))
            adjustments.append(f"R1b: hb_grade 从 {hb_grade} 调整为 1")
        else:
            checks.append(ConsistencyCheck(
                rule_id="R1b",
                rule_name="健康人HB",
                passed=True,
                message="hb_grade=1 ✓"
            ))

        if sunnybrook_score < 90:
            checks.append(ConsistencyCheck(
                rule_id="R1c",
                rule_name="健康人Sunnybrook",
                passed=False,
                message=f"has_palsy=False 但 sunnybrook={sunnybrook_score}<90，存在冲突",
                severity="warning"
            ))
        else:
            checks.append(ConsistencyCheck(
                rule_id="R1c",
                rule_name="健康人Sunnybrook",
                passed=True,
                message=f"sunnybrook={sunnybrook_score}≥90 ✓"
            ))

    # R2: 面瘫必有患侧
    if has_palsy and palsy_side == 0:
        checks.append(ConsistencyCheck(
            rule_id="R2",
            rule_name="面瘫患侧",
            passed=False,
            message="has_palsy=True 但 palsy_side=0，患侧方向不确定",
            severity="warning"
        ))
    elif has_palsy:
        checks.append(ConsistencyCheck(
            rule_id="R2",
            rule_name="面瘫患侧",
            passed=True,
            message=f"palsy_side={palsy_side} ({palsy_side_to_text(palsy_side)}) ✓"
        ))

        # R3: HB-Sunnybrook合理性检查
        expected_hb = sunnybrook_to_hb(sunnybrook_score)
        hb_diff = abs(hb_grade - expected_hb)

        if hb_diff == 0:
            checks.append(ConsistencyCheck(
                rule_id="R3",
                rule_name="HB-Sunnybrook兼容",
                passed=True,
                message=f"临床HB={hb_grade} 与 Sunnybrook映射={expected_hb} 完全一致 ✓"
            ))
        elif hb_diff <= 2:
            checks.append(ConsistencyCheck(
                rule_id="R3",
                rule_name="HB-Sunnybrook兼容",
                passed=True,
                message=f"临床HB={hb_grade} 与 Sunnybrook映射={expected_hb} 相差{hb_diff}级（正常）",
                severity="info"
            ))
        else:
            checks.append(ConsistencyCheck(
                rule_id="R3",
                rule_name="HB-Sunnybrook兼容",
                passed=False,
                message=f"临床HB={hb_grade} 与 Sunnybrook映射={expected_hb} 相差{hb_diff}级，请检查",
                severity="warning"
            ))

    # R4: 投票方向一致性
    if votes:
        left_votes = sum(1 for v in votes if v.side == 1)
        right_votes = sum(1 for v in votes if v.side == 2)

        if left_votes > 0 and right_votes > 0:
            ratio = min(left_votes, right_votes) / max(left_votes, right_votes)
            if ratio > 0.5:
                checks.append(ConsistencyCheck(
                    rule_id="R4",
                    rule_name="投票方向一致",
                    passed=False,
                    message=f"投票存在冲突: 左={left_votes}, 右={right_votes}",
                    severity="warning"
                ))
            else:
                checks.append(ConsistencyCheck(
                    rule_id="R4",
                    rule_name="投票方向一致",
                    passed=True,
                    message=f"投票方向基本一致: 左={left_votes}, 右={right_votes}"
                ))
        else:
            checks.append(ConsistencyCheck(
                rule_id="R4",
                rule_name="投票方向一致",
                passed=True,
                message=f"投票方向一致: 左={left_votes}, 右={right_votes} ✓"
            ))

    return checks, adjustments


# =============================================================================
# 主诊断函数
# =============================================================================

def compute_session_diagnosis(
        action_results: Dict,
        sunnybrook_score_obj=None,  # SunnybrookScore对象
) -> SessionDiagnosis:
    """
    计算Session级诊断结果

    Args:
        action_results: {action_name: ActionResult}
        sunnybrook_score_obj: SunnybrookScore对象（可选，如已计算）

    Returns:
        SessionDiagnosis 对象
    """
    # ========== 1. 使用新的HB分级计算 ==========
    hb_result = compute_hb_grade(action_results)

    # 提取核心结果
    palsy_side = hb_result.palsy_side
    palsy_side_confidence = hb_result.confidence
    hb_grade = hb_result.final_grade
    hb_description = hb_result.final_description

    # 判断是否有面瘫
    has_palsy = (palsy_side != 0)

    # 面瘫侧别文本
    palsy_side_text = palsy_side_to_text(palsy_side)

    # ========== 2. 计算 Sunnybrook 评分 ==========
    sunnybrook = calculate_sunnybrook_from_results(action_results, palsy_side)
    sunnybrook_composite = sunnybrook.composite_score if sunnybrook else 100

    # ========== 3. 收集投票 ==========
    votes = collect_votes(action_results)
    left_score, right_score, _, _ = aggregate_votes(votes)

    # 获取top证据
    top_evidence = sorted(votes, key=lambda x: x.weighted_score, reverse=True)[:5]

    # ========== 4. 构建HB组件详情 ==========
    component_details = {
        "a1_general": {
            "grade": hb_result.a1_general.grade,
            "description": hb_result.a1_general.grade_description,
        },
        "a2_at_rest": {
            "grade": hb_result.a2_at_rest.grade,
            "description": hb_result.a2_at_rest.grade_description,
        },
        "a3_forehead": {
            "grade": hb_result.a3_forehead.grade,
            "description": hb_result.a3_forehead.grade_description,
        },
        "a4_eyes": {
            "grade": hb_result.a4_eyes.grade,
            "description": hb_result.a4_eyes.grade_description,
        },
        "a5_mouth": {
            "grade": hb_result.a5_mouth.grade,
            "description": hb_result.a5_mouth.grade_description,
        },
    }

    # 确定最差的评估项
    worst_component = max(
        [("a1", hb_result.a1_general.grade),
         ("a2", hb_result.a2_at_rest.grade),
         ("a3", hb_result.a3_forehead.grade),
         ("a4", hb_result.a4_eyes.grade),
         ("a5", hb_result.a5_mouth.grade)],
        key=lambda x: x[1]
    )[0]

    # ========== 5. 一致性检查 ==========
    checks, adjustments = check_consistency(
        has_palsy, palsy_side, hb_grade, sunnybrook_composite, votes
    )

    # ========== 6. 生成解释 ==========
    interpretation = generate_interpretation(
        has_palsy, palsy_side, hb_grade, sunnybrook_composite,
        sunnybrook.resting_symmetry.total_score,
        sunnybrook.voluntary_movement.total_score,
        sunnybrook.synkinesis.total_score,
        votes
    )

    # ========== 7. 创建 SessionDiagnosis 对象 ==========
    diagnosis = SessionDiagnosis(
        has_palsy=has_palsy,
        palsy_side=palsy_side,
        palsy_side_text=palsy_side_text,
        hb_grade=hb_grade,
        hb_description=hb_description,
        sunnybrook_score=sunnybrook_composite,
        confidence=palsy_side_confidence,
        palsy_side_confidence=palsy_side_confidence,
        resting_score=sunnybrook.resting_symmetry.total_score,
        voluntary_score=sunnybrook.voluntary_movement.total_score,
        synkinesis_score=sunnybrook.synkinesis.total_score,
        left_score=left_score,
        right_score=right_score,
        votes=votes,
        top_evidence=top_evidence,
        consistency_checks=checks,
        adjustments_made=adjustments,
        hb_evidence={},
        hb_component_grades=component_details,
        hb_worst_component=worst_component,
        hb_decision_path=hb_result.decision_path,
        interpretation=interpretation,
    )

    # ========== 8. 打印调试信息 ==========
    print_hb_result(hb_result)

    return diagnosis


def generate_interpretation(
        has_palsy: bool,
        palsy_side: int,
        hb_grade: int,
        composite_score: int,
        resting_score: int,
        voluntary_score: int,
        synkinesis_score: int,
        votes: List[VoteRecord]
) -> str:
    """生成诊断解释文本"""

    lines = []

    if not has_palsy:
        lines.append("综合评估结果：未检测到明显面瘫表现。")
        lines.append(f"Sunnybrook综合评分 {composite_score} 分，属于正常范围。")
    else:
        side_text = palsy_side_to_text(palsy_side)
        lines.append(f"综合评估结果：检测到面瘫，患侧为{side_text}侧。")
        lines.append(f"House-Brackmann分级：{hb_grade}级 ({HB_DESCRIPTIONS.get(hb_grade, '')})")
        lines.append(f"Sunnybrook综合评分：{composite_score} 分")
        lines.append(f"  - 静息对称性扣分: {resting_score}")
        lines.append(f"  - 主动运动得分: {voluntary_score}")
        lines.append(f"  - 联动运动扣分: {synkinesis_score}")

    if votes:
        lines.append("")
        lines.append("主要证据来源：")
        for i, v in enumerate(sorted(votes, key=lambda x: x.weighted_score, reverse=True)[:3]):
            lines.append(f"  {i + 1}. {v.action_cn} ({v.region}): {v.side_text}侧弱, 置信度{v.confidence:.2f}")

    return "\n".join(lines)


# =============================================================================
# 辅助函数：用于动作模块
# =============================================================================

def compute_action_severity(
        voluntary_score: int = None,
        asymmetry_ratio: float = None,
        method: str = "voluntary"
) -> Tuple[int, str]:
    """
    计算动作严重度分数

    Args:
        voluntary_score: Voluntary Movement Score (1-5)
        asymmetry_ratio: 对称性偏差 (0-1, 0=完全对称)
        method: 计算方法 ("voluntary" 或 "asymmetry")

    Returns:
        (severity_score, severity_description)
    """
    if method == "voluntary" and voluntary_score is not None:
        severity = voluntary_to_severity(voluntary_score)
    elif asymmetry_ratio is not None:
        # 基于对称性偏差计算
        if asymmetry_ratio < 0.05:
            severity = 1
        elif asymmetry_ratio < 0.15:
            severity = 2
        elif asymmetry_ratio < 0.30:
            severity = 3
        elif asymmetry_ratio < 0.50:
            severity = 4
        else:
            severity = 5
    else:
        severity = 1  # 默认正常

    description = SEVERITY_DESCRIPTIONS.get(severity, "Unknown")
    return severity, description


def standardize_action_output(
        result,
        palsy_side: int,
        palsy_confidence: float,
        severity_score: int,
        voluntary_score: int,
        method: str = "",
        interpretation: str = "",
        evidence: Dict = None
) -> Dict[str, Any]:
    """
    标准化动作输出格式

    用于确保所有动作模块输出一致的 action_specific 结构

    Returns:
        标准化的 action_specific 字典
    """
    severity_desc = SEVERITY_DESCRIPTIONS.get(severity_score, "Unknown")

    # 确定显示颜色
    if palsy_side == 0:
        palsy_text = "Symmetric"
        color = (0, 255, 0)  # 绿色
    elif palsy_side == 1:
        palsy_text = "Left Palsy"
        color = (0, 0, 255)  # 红色
    else:
        palsy_text = "Right Palsy"
        color = (0, 0, 255)  # 红色

    return {
        "palsy_detection": {
            "palsy_side": palsy_side,
            "confidence": palsy_confidence,
            "method": method,
            "interpretation": interpretation,
            "evidence": evidence or {},
        },
        "severity_score": severity_score,
        "severity_description": severity_desc,
        "voluntary_score": voluntary_score,
        "display": {
            "palsy_text": palsy_text,
            "severity_text": f"{severity_score}/5 ({severity_desc})",
            "color": color,
        }
    }


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    print("Session Diagnosis Module")
    print("=" * 50)

    print("\nSeverity ↔ Voluntary 转换测试:")
    for vol in [1, 2, 3, 4, 5]:
        sev = voluntary_to_severity(vol)
        back = severity_to_voluntary(sev)
        print(f"  Voluntary {vol} → Severity {sev} → Voluntary {back}")

    print("\nSunnybrook → HB 映射测试:")
    for score in [95, 80, 60, 40, 20, 5]:
        hb = sunnybrook_to_hb(score)
        print(f"  Sunnybrook {score} → HB Grade {hb}")