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
    "CloseEyeHardly": 0.8,
    "RaiseEyebrow": 1,
    "Smile": 1,
    "ShrugNose": 1.0,
    "ShowTeeth": 1.3,
    "BlowCheek": 0.8,
    "LipPucker": 0.7,
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
    resting_score: int  # 0-20
    voluntary_score: int  # 20-100
    synkinesis_score: int  # 0-15

    # === 投票与证据 ===
    left_score: float  # 左侧累计证据
    right_score: float  # 右侧累计证据
    votes: List[VoteRecord] = field(default_factory=list)
    top_evidence: List[VoteRecord] = field(default_factory=list)

    # === 一致性检查 ===
    consistency_checks: List[ConsistencyCheck] = field(default_factory=list)
    adjustments_made: List[str] = field(default_factory=list)

    # === HB 分级证据 ===
    hb_evidence: Dict[str, Any] = field(default_factory=dict)

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


# =============================================================================
# 临床定义的 HB 分级（决策树）
# =============================================================================

def compute_hb_grade_clinical(
        action_results: Dict,
        palsy_side: int,
        sunnybrook_composite: int = None
) -> Tuple[int, str, Dict[str, Any]]:
    """
    基于临床定义的 House-Brackmann 分级（决策树方法）

    HB 量表核心区分逻辑：
    - Grade I: 所有功能正常
    - Grade II: 轻微不对称，眼闭合，额部有运动
    - Grade III: 眼睑能完全闭合，但额部无/微动
    - Grade IV: 眼睑无法完全闭合（分水岭！）
    - Grade V: 仅有微量运动
    - Grade VI: 完全无运动

    Args:
        action_results: {action_name: ActionResult}
        palsy_side: 患侧 (0=无, 1=左, 2=右)
        sunnybrook_composite: Sunnybrook综合分（用于辅助判断）

    Returns:
        (hb_grade, description, evidence_dict)
    """
    evidence = {
        "method": "clinical_decision_tree",
        "palsy_side": palsy_side,
        "eye_closure": None,
        "forehead_movement": None,
        "mouth_movement": None,
        "decision_path": []
    }

    # === 如果没有面瘫，直接返回 Grade I ===
    if palsy_side == 0:
        evidence["decision_path"].append("No palsy detected → Grade I")
        return 1, "Normal (正常)", evidence

    # === Step 1: 获取关键指标 ===

    # 1.1 眼睑闭合度 (CloseEyeHardly)
    eye_closure_affected = None
    close_eye_result = action_results.get("CloseEyeHardly")
    if close_eye_result is not None:
        action_spec = getattr(close_eye_result, 'action_specific', {})

        # 优先从 closure_metrics 获取
        closure_metrics = action_spec.get("closure_metrics", {})
        if closure_metrics:
            if palsy_side == 1:  # 左侧面瘫
                eye_closure_affected = closure_metrics.get("left_closure_ratio", None)
            else:  # 右侧面瘫
                eye_closure_affected = closure_metrics.get("right_closure_ratio", None)

        # 备选：从 peak_frame 获取
        if eye_closure_affected is None:
            peak_frame = action_spec.get("peak_frame", {})
            if palsy_side == 1:
                eye_closure_affected = peak_frame.get("left_closure", None)
            else:
                eye_closure_affected = peak_frame.get("right_closure", None)

    evidence["eye_closure"] = eye_closure_affected

    # 1.2 额部运动 (RaiseEyebrow)
    forehead_movement_affected = None
    forehead_movement_healthy = None
    raise_eyebrow_result = action_results.get("RaiseEyebrow")
    if raise_eyebrow_result is not None:
        action_spec = getattr(raise_eyebrow_result, 'action_specific', {})
        brow_metrics = action_spec.get("brow_eye_metrics", {})

        left_change = brow_metrics.get("left_change", 0)
        right_change = brow_metrics.get("right_change", 0)

        if palsy_side == 1:  # 左侧面瘫
            forehead_movement_affected = abs(left_change)
            forehead_movement_healthy = abs(right_change)
        else:  # 右侧面瘫
            forehead_movement_affected = abs(right_change)
            forehead_movement_healthy = abs(left_change)

    evidence["forehead_movement"] = {
        "affected": forehead_movement_affected,
        "healthy": forehead_movement_healthy
    }

    # 1.3 嘴角运动 (Smile 或 ShowTeeth)
    mouth_movement_affected = None
    mouth_movement_healthy = None

    for action_name in ["Smile", "ShowTeeth"]:
        action_result = action_results.get(action_name)
        if action_result is not None:
            action_spec = getattr(action_result, 'action_specific', {})

            # 从 eye_line_excursion 获取嘴角运动
            if "eye_line_excursion" in action_spec:
                exc = action_spec["eye_line_excursion"]
                left_reduction = exc.get("left_reduction", 0)
                right_reduction = exc.get("right_reduction", 0)
            elif "excursion" in action_spec:
                exc = action_spec.get("excursion", {})
                left_reduction = exc.get("left_total", 0)
                right_reduction = exc.get("right_total", 0)
            else:
                continue

            if palsy_side == 1:
                mouth_movement_affected = abs(left_reduction) if left_reduction else 0
                mouth_movement_healthy = abs(right_reduction) if right_reduction else 0
            else:
                mouth_movement_affected = abs(right_reduction) if right_reduction else 0
                mouth_movement_healthy = abs(left_reduction) if left_reduction else 0

            if mouth_movement_affected is not None:
                break  # 找到一个有效的就行

    evidence["mouth_movement"] = {
        "affected": mouth_movement_affected,
        "healthy": mouth_movement_healthy
    }

    # === Step 2: 决策树判断 ===

    # 阈值定义（可调整）
    EYE_CLOSURE_COMPLETE = 0.85  # 闭合度 >= 85% 视为完全闭合
    FOREHEAD_MINIMAL = 3.0  # 额部变化 < 3px 视为无运动
    FOREHEAD_WEAK = 8.0  # 额部变化 < 8px 视为微弱
    MOUTH_MINIMAL = 5.0  # 嘴角运动 < 5px 视为微量

    # === 检查是否全瘫 (Grade VI) ===
    total_movement = 0
    movement_count = 0

    if forehead_movement_affected is not None:
        total_movement += forehead_movement_affected
        movement_count += 1
    if mouth_movement_affected is not None:
        total_movement += mouth_movement_affected
        movement_count += 1
    if eye_closure_affected is not None:
        # 闭合度转换为"运动量"：完全不闭合=0运动
        eye_movement = eye_closure_affected * 50  # 缩放到可比较范围
        total_movement += eye_movement
        movement_count += 1

    avg_movement = total_movement / movement_count if movement_count > 0 else 0

    if movement_count >= 2 and avg_movement < 2.0:
        evidence["decision_path"].append(f"Total movement very low ({avg_movement:.1f}) → Grade VI")
        return 6, "Total paralysis (完全麻痹)", evidence

    # === 检查眼睑闭合 (Grade IV 的分水岭) ===
    if eye_closure_affected is not None:
        if eye_closure_affected < EYE_CLOSURE_COMPLETE:
            # 眼睛无法完全闭合 → 进入 Grade IV-V 范围
            evidence["decision_path"].append(
                f"Eye closure incomplete ({eye_closure_affected:.1%} < {EYE_CLOSURE_COMPLETE:.0%})"
            )

            # 区分 IV vs V：看嘴角是否有一定运动
            if mouth_movement_affected is not None and mouth_movement_affected > MOUTH_MINIMAL:
                evidence["decision_path"].append(
                    f"Mouth movement present ({mouth_movement_affected:.1f}px) → Grade IV"
                )
                return 4, "Moderately severe dysfunction (中重度功能障碍)", evidence
            else:
                # 嘴角也几乎不动
                evidence["decision_path"].append(
                    f"Mouth movement minimal ({mouth_movement_affected}px) → Grade V"
                )
                return 5, "Severe dysfunction (重度功能障碍)", evidence
        else:
            evidence["decision_path"].append(
                f"Eye closure complete ({eye_closure_affected:.1%} >= {EYE_CLOSURE_COMPLETE:.0%})"
            )
    else:
        # 没有闭眼数据，使用 Sunnybrook 辅助判断
        evidence["decision_path"].append("No eye closure data, using Sunnybrook fallback")
        if sunnybrook_composite is not None and sunnybrook_composite < 40:
            return 4, "Moderately severe dysfunction (中重度功能障碍)", evidence

    # === 眼睛能闭合 → 检查额部运动 (Grade III 的分水岭) ===
    if forehead_movement_affected is not None:
        # 计算患侧相对健侧的运动比例
        if forehead_movement_healthy and forehead_movement_healthy > FOREHEAD_MINIMAL:
            forehead_ratio = forehead_movement_affected / forehead_movement_healthy
        else:
            forehead_ratio = 1.0 if forehead_movement_affected > FOREHEAD_MINIMAL else 0.0

        if forehead_movement_affected < FOREHEAD_MINIMAL or forehead_ratio < 0.25:
            evidence["decision_path"].append(
                f"Forehead movement minimal (affected={forehead_movement_affected:.1f}px, "
                f"ratio={forehead_ratio:.1%}) → Grade III"
            )
            return 3, "Moderate dysfunction (中度功能障碍)", evidence
        elif forehead_ratio < 0.60:
            evidence["decision_path"].append(
                f"Forehead movement weak (ratio={forehead_ratio:.1%}) → between II-III"
            )
            # 边界情况，检查嘴角对称性
            if mouth_movement_affected is not None and mouth_movement_healthy is not None:
                if mouth_movement_healthy > MOUTH_MINIMAL:
                    mouth_ratio = mouth_movement_affected / mouth_movement_healthy
                    if mouth_ratio < 0.5:
                        return 3, "Moderate dysfunction (中度功能障碍)", evidence
            return 3, "Moderate dysfunction (中度功能障碍)", evidence
        else:
            evidence["decision_path"].append(
                f"Forehead movement present (ratio={forehead_ratio:.1%})"
            )
    else:
        evidence["decision_path"].append("No forehead data")

    # === 眼睛能闭合，额部有运动 → Grade I 或 II ===
    # 通过整体对称性区分

    asymmetry_indicators = []

    if forehead_movement_affected is not None and forehead_movement_healthy is not None:
        if forehead_movement_healthy > FOREHEAD_MINIMAL:
            forehead_asym = 1.0 - (forehead_movement_affected / forehead_movement_healthy)
            asymmetry_indicators.append(max(0, forehead_asym))

    if mouth_movement_affected is not None and mouth_movement_healthy is not None:
        if mouth_movement_healthy > MOUTH_MINIMAL:
            mouth_asym = 1.0 - (mouth_movement_affected / mouth_movement_healthy)
            asymmetry_indicators.append(max(0, mouth_asym))

    if asymmetry_indicators:
        avg_asymmetry = sum(asymmetry_indicators) / len(asymmetry_indicators)
        evidence["overall_asymmetry"] = avg_asymmetry

        if avg_asymmetry < 0.10:
            evidence["decision_path"].append(f"Minimal asymmetry ({avg_asymmetry:.1%}) → Grade I")
            return 1, "Normal (正常)", evidence
        else:
            evidence["decision_path"].append(f"Mild asymmetry ({avg_asymmetry:.1%}) → Grade II")
            return 2, "Slight dysfunction (轻度功能障碍)", evidence

    # 默认
    evidence["decision_path"].append("Insufficient data → default Grade II")
    return 2, "Slight dysfunction (轻度功能障碍)", evidence


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
        # 注意：HB 由临床决策树计算，可能与 Sunnybrook 映射有差异，这是正常的
        expected_hb = sunnybrook_to_hb(sunnybrook_score)
        hb_diff = abs(hb_grade - expected_hb)

        if hb_diff == 0:
            checks.append(ConsistencyCheck(
                rule_id="R3",
                rule_name="HB-Sunnybrook兼容",
                passed=True,
                message=f"临床HB={hb_grade} 与 Sunnybrook映射={expected_hb} 完全一致 ✓"
            ))
        elif (hb_diff == 1 or hb_diff == 2 or hb_diff == 3):
            # 差1级是可接受的（临床决策树更精细）
            checks.append(ConsistencyCheck(
                rule_id="R3",
                rule_name="HB-Sunnybrook兼容",
                passed=True,
                message=f"临床HB={hb_grade} 与 Sunnybrook映射={expected_hb} 相差{hb_diff}级（正常）",
                severity="info"
            ))
        else:
            # 差4级以上需要关注
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
            # 存在冲突投票
            ratio = min(left_votes, right_votes) / max(left_votes, right_votes)
            if ratio > 0.5:  # 冲突严重
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
    # === Step 1: 获取Sunnybrook评分 ===
    if sunnybrook_score_obj is not None:
        resting_score = sunnybrook_score_obj.resting_score
        voluntary_score = sunnybrook_score_obj.voluntary_score
        synkinesis_score = sunnybrook_score_obj.synkinesis_score
        composite_score = sunnybrook_score_obj.composite_score
    else:
        # 如果没有传入，使用默认值（健康人）
        resting_score = 0
        voluntary_score = 100
        synkinesis_score = 0
        composite_score = 100

    # === Step 3: 收集投票并确定患侧 ===
    votes = collect_votes(action_results)
    left_score, right_score, voted_side, side_confidence = aggregate_votes(votes)

    # === Step 4: 确定是否面瘫 ===
    # 主要依据: Sunnybrook < 90 或 HB > 1
    # 辅助依据: 投票总分 > 阈值
    total_vote_score = left_score + right_score

    if composite_score >= 90 and total_vote_score < 0.5:
        has_palsy = False
    elif composite_score < 70 or total_vote_score > 1.5:
        has_palsy = True
    else:
        # 边界情况：综合判断
        has_palsy = (composite_score < 85) or (total_vote_score > 0.8)

    # === Step 5: 确定最终患侧 ===
    if not has_palsy:
        palsy_side = 0
    else:
        palsy_side = voted_side
        # 如果投票不确定，尝试从Resting Symmetry获取
        if palsy_side == 0 and sunnybrook_score_obj is not None:
            resting = sunnybrook_score_obj.resting_symmetry
            if hasattr(resting, 'affected_side'):
                affected = resting.affected_side
                if "Left" in affected:
                    palsy_side = 1
                elif "Right" in affected:
                    palsy_side = 2

    palsy_side_text = palsy_side_to_text(palsy_side)

    # === Step 5.5: 使用临床决策树计算 HB 分级 ===
    hb_grade, hb_description, hb_evidence = compute_hb_grade_clinical(
        action_results=action_results,
        palsy_side=palsy_side,
        sunnybrook_composite=composite_score
    )

    # === Step 6: 一致性检查与调整 ===
    checks, adjustments = check_consistency(
        has_palsy, palsy_side, hb_grade, composite_score, votes
    )

    # 如果没有面瘫，强制 Grade I
    if not has_palsy:
        palsy_side = 0
        palsy_side_text = "无"
        hb_grade = 1
        hb_description = HB_DESCRIPTIONS[1]

    # === Step 7: 计算整体置信度 ===
    if not has_palsy:
        confidence = max(0.5, 1.0 - total_vote_score)
    else:
        # 置信度 = Sunnybrook证据 × 投票一致性
        sb_confidence = 1.0 - abs(composite_score - 50) / 50  # 越极端越确信
        sb_confidence = max(0.3, min(1.0, sb_confidence))
        confidence = (sb_confidence + side_confidence) / 2

    # === Step 8: 生成解释文本 ===
    interpretation = generate_interpretation(
        has_palsy, palsy_side, hb_grade, composite_score,
        resting_score, voluntary_score, synkinesis_score,
        votes
    )

    # === Step 9: 排序证据 ===
    sorted_votes = sorted(votes, key=lambda v: v.weighted_score, reverse=True)
    top_evidence = sorted_votes[:5]

    return SessionDiagnosis(
        has_palsy=has_palsy,
        palsy_side=palsy_side,
        palsy_side_text=palsy_side_text,
        hb_grade=hb_grade,
        hb_description=hb_description,
        hb_evidence=hb_evidence,
        sunnybrook_score=composite_score,
        confidence=confidence,
        palsy_side_confidence=side_confidence,
        resting_score=resting_score,
        voluntary_score=voluntary_score,
        synkinesis_score=synkinesis_score,
        left_score=left_score,
        right_score=right_score,
        votes=votes,
        top_evidence=top_evidence,
        consistency_checks=checks,
        adjustments_made=adjustments,
        interpretation=interpretation,
    )


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
    # 测试 sunnybrook_to_hb

    print("\nSeverity ↔ Voluntary 转换测试:")
    for vol in [1, 2, 3, 4, 5]:
        sev = voluntary_to_severity(vol)
        back = severity_to_voluntary(sev)
        print(f"  Voluntary {vol} → Severity {sev} → Voluntary {back}")