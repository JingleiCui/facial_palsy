#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
面瘫检测器 (Palsy Detector)
===========================

综合11个动作的指标，判断:
1. 是否存在面瘫
2. 患侧是哪一侧 (Left/Right)
3. 置信度和关键证据

检测策略:
- 多证据融合: 从每个动作提取异常证据
- 按区域聚合: Eye/Brow/Mouth/Cheek/Nose
- 加权投票: 根据严重程度(mild/moderate/severe)赋予不同权重
- 可解释性输出: 提供关键证据列表
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum


# =============================================================================
# 数据类定义
# =============================================================================

class Severity(Enum):
    """严重程度"""
    NONE = 0
    MILD = 1
    MODERATE = 2
    SEVERE = 3


@dataclass
class EvidenceItem:
    """单条证据"""
    action_name: str  # 来源动作
    action_name_cn: str  # 动作中文名
    indicator: str  # 指标名称
    indicator_cn: str  # 指标中文名
    region: str  # 区域: eye/brow/mouth/cheek/nose
    severity: Severity  # 严重程度
    affected_side: str  # 受影响侧: left/right/bilateral

    # 测量值
    left_value: float = 0.0
    right_value: float = 0.0
    ratio: float = 1.0
    threshold_info: str = ""

    # 权重分数
    score: int = 0  # mild=1, moderate=2, severe=3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_name": self.action_name,
            "action_name_cn": self.action_name_cn,
            "indicator": self.indicator,
            "indicator_cn": self.indicator_cn,
            "region": self.region,
            "severity": self.severity.name,
            "affected_side": self.affected_side,
            "left_value": self.left_value,
            "right_value": self.right_value,
            "ratio": self.ratio,
            "threshold_info": self.threshold_info,
            "score": self.score,
        }


@dataclass
class RegionEvidence:
    """区域证据汇总"""
    region: str
    region_cn: str
    left_score: int = 0  # 左侧异常累积分
    right_score: int = 0  # 右侧异常累积分
    evidence_count: int = 0  # 证据数量
    evidences: List[EvidenceItem] = field(default_factory=list)
    conclusion: str = ""  # 结论: normal/left_affected/right_affected

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region": self.region,
            "region_cn": self.region_cn,
            "left_score": self.left_score,
            "right_score": self.right_score,
            "evidence_count": self.evidence_count,
            "conclusion": self.conclusion,
            "evidences": [e.to_dict() for e in self.evidences],
        }


@dataclass
class PalsyDetectionResult:
    """面瘫检测结果"""
    has_palsy: bool = False
    affected_side: str = "Unknown"  # Left/Right/Bilateral/Unknown
    confidence: float = 0.0  # 置信度 0-100%

    # 各区域证据
    eye_evidence: Optional[RegionEvidence] = None
    brow_evidence: Optional[RegionEvidence] = None
    mouth_evidence: Optional[RegionEvidence] = None
    cheek_evidence: Optional[RegionEvidence] = None
    nose_evidence: Optional[RegionEvidence] = None

    # 汇总
    left_total_score: int = 0
    right_total_score: int = 0
    total_evidence_count: int = 0

    # 关键证据 (按严重程度排序)
    key_evidences: List[EvidenceItem] = field(default_factory=list)

    # 诊断说明
    diagnosis_summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_palsy": self.has_palsy,
            "affected_side": self.affected_side,
            "confidence": self.confidence,
            "left_total_score": self.left_total_score,
            "right_total_score": self.right_total_score,
            "total_evidence_count": self.total_evidence_count,
            "diagnosis_summary": self.diagnosis_summary,
            "regions": {
                "eye": self.eye_evidence.to_dict() if self.eye_evidence else None,
                "brow": self.brow_evidence.to_dict() if self.brow_evidence else None,
                "mouth": self.mouth_evidence.to_dict() if self.mouth_evidence else None,
                "cheek": self.cheek_evidence.to_dict() if self.cheek_evidence else None,
                "nose": self.nose_evidence.to_dict() if self.nose_evidence else None,
            },
            "key_evidences": [e.to_dict() for e in self.key_evidences],
        }


# =============================================================================
# 阈值配置
# =============================================================================

RATIO_THRESHOLDS = {
    "ear_ratio": {"mild": 0.12, "moderate": 0.22, "severe": 0.38},
    "eye_area_ratio": {"mild": 0.12, "moderate": 0.22, "severe": 0.38},
    "palpebral_height_ratio": {"mild": 0.15, "moderate": 0.28, "severe": 0.45},
    "brow_height_ratio": {"mild": 0.12, "moderate": 0.22, "severe": 0.38},
    "brow_eye_distance_ratio": {"mild": 0.15, "moderate": 0.28, "severe": 0.45},
    "nlf_ratio": {"mild": 0.12, "moderate": 0.22, "severe": 0.38},
    "excursion_ratio": {"mild": 0.15, "moderate": 0.28, "severe": 0.45},
}

ANGLE_THRESHOLDS = {
    "oral_angle_asymmetry": {"mild": 5.0, "moderate": 10.0, "severe": 16.0},
}

CLOSURE_THRESHOLDS = {
    "closure_ratio_diff": {"mild": 0.15, "moderate": 0.30, "severe": 0.50},
}


# =============================================================================
# 辅助函数
# =============================================================================

def classify_severity(deviation: float, thresholds: Dict[str, float]) -> Tuple[Severity, int]:
    """根据偏离程度分类严重度"""
    if deviation >= thresholds["severe"]:
        return Severity.SEVERE, 3
    elif deviation >= thresholds["moderate"]:
        return Severity.MODERATE, 2
    elif deviation >= thresholds["mild"]:
        return Severity.MILD, 1
    return Severity.NONE, 0


def determine_affected_side_from_ratio(ratio: float, is_lower_better: bool = False) -> str:
    """从比值判断受影响侧"""
    if abs(ratio - 1.0) < 0.02:
        return "bilateral"

    if is_lower_better:
        if ratio > 1.0:
            return "left"
        else:
            return "right"
    else:
        if ratio < 1.0:
            return "left"
        else:
            return "right"


# =============================================================================
# 证据提取函数
# =============================================================================

def extract_evidence_from_neutral_face(result, baseline_result=None) -> List[EvidenceItem]:
    """从NeutralFace提取证据"""
    evidences = []
    action_name = result.action_name
    action_name_cn = result.action_name_cn

    # 睑裂高度比
    ratio = result.palpebral_height_ratio
    deviation = abs(ratio - 1.0)
    severity, score = classify_severity(deviation, RATIO_THRESHOLDS["palpebral_height_ratio"])

    if severity != Severity.NONE:
        affected = determine_affected_side_from_ratio(ratio)
        evidences.append(EvidenceItem(
            action_name=action_name, action_name_cn=action_name_cn,
            indicator="palpebral_height_ratio", indicator_cn="睑裂高度比",
            region="eye", severity=severity, affected_side=affected,
            left_value=result.left_palpebral_height, right_value=result.right_palpebral_height,
            ratio=ratio, threshold_info=f"偏离{deviation:.2f}", score=score,
        ))

    # 眼睛面积比
    ratio = result.eye_area_ratio
    deviation = abs(ratio - 1.0)
    severity, score = classify_severity(deviation, RATIO_THRESHOLDS["eye_area_ratio"])

    if severity != Severity.NONE:
        affected = determine_affected_side_from_ratio(ratio)
        evidences.append(EvidenceItem(
            action_name=action_name, action_name_cn=action_name_cn,
            indicator="eye_area_ratio", indicator_cn="眼睛面积比",
            region="eye", severity=severity, affected_side=affected,
            left_value=result.left_eye_area, right_value=result.right_eye_area,
            ratio=ratio, threshold_info=f"偏离{deviation:.2f}", score=score,
        ))

    # EAR比
    if result.right_ear > 1e-9:
        ratio = result.left_ear / result.right_ear
        deviation = abs(ratio - 1.0)
        severity, score = classify_severity(deviation, RATIO_THRESHOLDS["ear_ratio"])

        if severity != Severity.NONE:
            affected = determine_affected_side_from_ratio(ratio)
            evidences.append(EvidenceItem(
                action_name=action_name, action_name_cn=action_name_cn,
                indicator="ear_ratio", indicator_cn="眼睛睁开比(EAR)",
                region="eye", severity=severity, affected_side=affected,
                left_value=result.left_ear, right_value=result.right_ear,
                ratio=ratio, threshold_info=f"L={result.left_ear:.3f} R={result.right_ear:.3f}", score=score,
            ))

    # 眉毛高度比
    ratio = result.brow_height_ratio
    deviation = abs(ratio - 1.0)
    severity, score = classify_severity(deviation, RATIO_THRESHOLDS["brow_height_ratio"])

    if severity != Severity.NONE:
        affected = determine_affected_side_from_ratio(ratio)
        evidences.append(EvidenceItem(
            action_name=action_name, action_name_cn=action_name_cn,
            indicator="brow_height_ratio", indicator_cn="眉毛高度比",
            region="brow", severity=severity, affected_side=affected,
            left_value=result.left_brow_height, right_value=result.right_brow_height,
            ratio=ratio, threshold_info=f"偏离{deviation:.2f}", score=score,
        ))

    # 鼻唇沟比
    ratio = result.nlf_ratio
    deviation = abs(ratio - 1.0)
    severity, score = classify_severity(deviation, RATIO_THRESHOLDS["nlf_ratio"])

    if severity != Severity.NONE:
        affected = determine_affected_side_from_ratio(ratio)
        evidences.append(EvidenceItem(
            action_name=action_name, action_name_cn=action_name_cn,
            indicator="nlf_ratio", indicator_cn="鼻唇沟比",
            region="cheek", severity=severity, affected_side=affected,
            left_value=result.left_nlf_length, right_value=result.right_nlf_length,
            ratio=ratio, threshold_info=f"L={result.left_nlf_length:.1f} R={result.right_nlf_length:.1f}", score=score,
        ))

    # 口角角度不对称
    if result.oral_angle:
        oral = result.oral_angle
        asymmetry = oral.angle_asymmetry
        severity, score = classify_severity(asymmetry, ANGLE_THRESHOLDS["oral_angle_asymmetry"])

        if severity != Severity.NONE:
            if oral.BOF_angle < oral.AOE_angle - 2:
                affected = "left"
            elif oral.AOE_angle < oral.BOF_angle - 2:
                affected = "right"
            else:
                affected = "bilateral"

            evidences.append(EvidenceItem(
                action_name=action_name, action_name_cn=action_name_cn,
                indicator="oral_angle_asymmetry", indicator_cn="口角角度不对称",
                region="mouth", severity=severity, affected_side=affected,
                left_value=oral.BOF_angle, right_value=oral.AOE_angle,
                ratio=1.0, threshold_info=f"BOF(L)={oral.BOF_angle:+.1f}° AOE(R)={oral.AOE_angle:+.1f}°", score=score,
            ))

    return evidences


def extract_evidence_from_smile(result, baseline_result=None) -> List[EvidenceItem]:
    """从Smile/ShowTeeth提取证据"""
    evidences = []
    action_name = result.action_name
    action_name_cn = result.action_name_cn

    # 口角角度不对称
    if result.oral_angle:
        oral = result.oral_angle
        asymmetry = oral.angle_asymmetry
        severity, score = classify_severity(asymmetry, ANGLE_THRESHOLDS["oral_angle_asymmetry"])

        if severity != Severity.NONE:
            if oral.BOF_angle < oral.AOE_angle - 2:
                affected = "left"
            elif oral.AOE_angle < oral.BOF_angle - 2:
                affected = "right"
            else:
                affected = "bilateral"

            evidences.append(EvidenceItem(
                action_name=action_name, action_name_cn=action_name_cn,
                indicator="smile_oral_asymmetry", indicator_cn="微笑口角不对称",
                region="mouth", severity=severity, affected_side=affected,
                left_value=oral.BOF_angle, right_value=oral.AOE_angle,
                ratio=1.0, threshold_info=f"差值={asymmetry:.1f}°", score=score,
            ))

    # 运动幅度比
    if result.action_specific:
        smile_metrics = result.action_specific.get("smile_metrics", {})
        excursion = smile_metrics.get("excursion", {})
        if excursion:
            exc_ratio = excursion.get("excursion_ratio", 1.0)
            deviation = abs(exc_ratio - 1.0)
            severity, score = classify_severity(deviation, RATIO_THRESHOLDS["excursion_ratio"])

            if severity != Severity.NONE:
                affected = determine_affected_side_from_ratio(exc_ratio)
                evidences.append(EvidenceItem(
                    action_name=action_name, action_name_cn=action_name_cn,
                    indicator="smile_excursion_ratio", indicator_cn="微笑运动幅度比",
                    region="mouth", severity=severity, affected_side=affected,
                    left_value=excursion.get("left_total", 0), right_value=excursion.get("right_total", 0),
                    ratio=exc_ratio,
                    threshold_info=f"L={excursion.get('left_total', 0):.1f}px R={excursion.get('right_total', 0):.1f}px",
                    score=score,
                ))

    # NLF比
    ratio = result.nlf_ratio
    deviation = abs(ratio - 1.0)
    severity, score = classify_severity(deviation, RATIO_THRESHOLDS["nlf_ratio"])

    if severity != Severity.NONE:
        affected = determine_affected_side_from_ratio(ratio)
        evidences.append(EvidenceItem(
            action_name=action_name, action_name_cn=action_name_cn,
            indicator="smile_nlf_ratio", indicator_cn="微笑时鼻唇沟比",
            region="cheek", severity=severity, affected_side=affected,
            left_value=result.left_nlf_length, right_value=result.right_nlf_length,
            ratio=ratio, threshold_info=f"L={result.left_nlf_length:.1f} R={result.right_nlf_length:.1f}", score=score,
        ))

    return evidences


def extract_evidence_from_close_eye(result, baseline_result=None) -> List[EvidenceItem]:
    """从CloseEyeSoftly/CloseEyeHardly提取证据"""
    evidences = []
    action_name = result.action_name
    action_name_cn = result.action_name_cn

    # 闭眼时的EAR比
    if result.right_ear > 1e-9:
        ratio = result.left_ear / result.right_ear
        deviation = abs(ratio - 1.0)
        severity, score = classify_severity(deviation, RATIO_THRESHOLDS["ear_ratio"])

        if severity != Severity.NONE:
            affected = determine_affected_side_from_ratio(ratio, is_lower_better=True)
            evidences.append(EvidenceItem(
                action_name=action_name, action_name_cn=action_name_cn,
                indicator="close_eye_ear_ratio", indicator_cn="闭眼时EAR比",
                region="eye", severity=severity, affected_side=affected,
                left_value=result.left_ear, right_value=result.right_ear,
                ratio=ratio, threshold_info=f"L={result.left_ear:.4f} R={result.right_ear:.4f}", score=score,
            ))

    # 闭眼完整度
    if result.action_specific:
        metrics = result.action_specific.get("close_eye_metrics", {})
        l_closure = metrics.get("left_closure_ratio", 0)
        r_closure = metrics.get("right_closure_ratio", 0)

        if l_closure > 0 and r_closure > 0:
            closure_diff = abs(l_closure - r_closure)
            severity, score = classify_severity(closure_diff, CLOSURE_THRESHOLDS["closure_ratio_diff"])

            if severity != Severity.NONE:
                if l_closure < r_closure:
                    affected = "left"
                else:
                    affected = "right"

                evidences.append(EvidenceItem(
                    action_name=action_name, action_name_cn=action_name_cn,
                    indicator="eye_closure_diff", indicator_cn="闭眼完整度差异",
                    region="eye", severity=severity, affected_side=affected,
                    left_value=l_closure, right_value=r_closure,
                    ratio=l_closure / r_closure if r_closure > 1e-9 else 1.0,
                    threshold_info=f"L={l_closure * 100:.1f}% R={r_closure * 100:.1f}%", score=score,
                ))

    return evidences


def extract_evidence_from_raise_eyebrow(result, baseline_result=None) -> List[EvidenceItem]:
    """从RaiseEyebrow提取证据"""
    evidences = []
    action_name = result.action_name
    action_name_cn = result.action_name_cn

    # 眉毛高度比
    ratio = result.brow_height_ratio
    deviation = abs(ratio - 1.0)
    severity, score = classify_severity(deviation, RATIO_THRESHOLDS["brow_height_ratio"])

    if severity != Severity.NONE:
        affected = determine_affected_side_from_ratio(ratio)
        evidences.append(EvidenceItem(
            action_name=action_name, action_name_cn=action_name_cn,
            indicator="raise_brow_height_ratio", indicator_cn="皱额时眉高比",
            region="brow", severity=severity, affected_side=affected,
            left_value=result.left_brow_height, right_value=result.right_brow_height,
            ratio=ratio, threshold_info=f"L={result.left_brow_height:.1f} R={result.right_brow_height:.1f}",
            score=score,
        ))

    return evidences


def extract_evidence_from_lip_pucker(result, baseline_result=None) -> List[EvidenceItem]:
    """从LipPucker提取证据"""
    evidences = []
    action_name = result.action_name
    action_name_cn = result.action_name_cn

    if result.oral_angle:
        oral = result.oral_angle
        asymmetry = oral.angle_asymmetry
        severity, score = classify_severity(asymmetry, ANGLE_THRESHOLDS["oral_angle_asymmetry"])

        if severity != Severity.NONE:
            if oral.BOF_angle < oral.AOE_angle - 2:
                affected = "left"
            elif oral.AOE_angle < oral.BOF_angle - 2:
                affected = "right"
            else:
                affected = "bilateral"

            evidences.append(EvidenceItem(
                action_name=action_name, action_name_cn=action_name_cn,
                indicator="lip_pucker_asymmetry", indicator_cn="撅嘴口角不对称",
                region="mouth", severity=severity, affected_side=affected,
                left_value=oral.BOF_angle, right_value=oral.AOE_angle,
                ratio=1.0, threshold_info=f"差值={asymmetry:.1f}°", score=score,
            ))

    return evidences


def extract_evidence_from_shrug_nose(result, baseline_result=None) -> List[EvidenceItem]:
    """从ShrugNose提取证据"""
    evidences = []
    action_name = result.action_name
    action_name_cn = result.action_name_cn

    ratio = result.nlf_ratio
    deviation = abs(ratio - 1.0)
    severity, score = classify_severity(deviation, RATIO_THRESHOLDS["nlf_ratio"])

    if severity != Severity.NONE:
        affected = determine_affected_side_from_ratio(ratio)
        evidences.append(EvidenceItem(
            action_name=action_name, action_name_cn=action_name_cn,
            indicator="shrug_nlf_ratio", indicator_cn="皱鼻时鼻唇沟比",
            region="nose", severity=severity, affected_side=affected,
            left_value=result.left_nlf_length, right_value=result.right_nlf_length,
            ratio=ratio, threshold_info=f"L={result.left_nlf_length:.1f} R={result.right_nlf_length:.1f}", score=score,
        ))

    return evidences


def extract_evidence_from_blow_cheek(result, baseline_result=None) -> List[EvidenceItem]:
    """从BlowCheek提取证据"""
    evidences = []
    action_name = result.action_name
    action_name_cn = result.action_name_cn

    ratio = result.nlf_ratio
    deviation = abs(ratio - 1.0)
    severity, score = classify_severity(deviation, RATIO_THRESHOLDS["nlf_ratio"])

    if severity != Severity.NONE:
        affected = determine_affected_side_from_ratio(ratio)
        evidences.append(EvidenceItem(
            action_name=action_name, action_name_cn=action_name_cn,
            indicator="blow_nlf_ratio", indicator_cn="鼓腮时鼻唇沟比",
            region="cheek", severity=severity, affected_side=affected,
            left_value=result.left_nlf_length, right_value=result.right_nlf_length,
            ratio=ratio, threshold_info=f"L={result.left_nlf_length:.1f} R={result.right_nlf_length:.1f}", score=score,
        ))

    return evidences


def extract_evidence_from_eye_blink(result, baseline_result=None) -> List[EvidenceItem]:
    """从VoluntaryEyeBlink/SpontaneousEyeBlink提取证据"""
    evidences = []
    action_name = result.action_name
    action_name_cn = result.action_name_cn

    if result.right_ear > 1e-9:
        ratio = result.left_ear / result.right_ear
        deviation = abs(ratio - 1.0)
        severity, score = classify_severity(deviation, RATIO_THRESHOLDS["ear_ratio"])

        if severity != Severity.NONE:
            affected = determine_affected_side_from_ratio(ratio, is_lower_better=True)
            evidences.append(EvidenceItem(
                action_name=action_name, action_name_cn=action_name_cn,
                indicator="blink_ear_ratio", indicator_cn="眨眼时EAR比",
                region="eye", severity=severity, affected_side=affected,
                left_value=result.left_ear, right_value=result.right_ear,
                ratio=ratio, threshold_info=f"L={result.left_ear:.4f} R={result.right_ear:.4f}", score=score,
            ))

    return evidences


# 动作名到提取函数的映射
EVIDENCE_EXTRACTORS = {
    "NeutralFace": extract_evidence_from_neutral_face,
    "Smile": extract_evidence_from_smile,
    "ShowTeeth": extract_evidence_from_smile,
    "CloseEyeSoftly": extract_evidence_from_close_eye,
    "CloseEyeHardly": extract_evidence_from_close_eye,
    "RaiseEyebrow": extract_evidence_from_raise_eyebrow,
    "LipPucker": extract_evidence_from_lip_pucker,
    "ShrugNose": extract_evidence_from_shrug_nose,
    "BlowCheek": extract_evidence_from_blow_cheek,
    "VoluntaryEyeBlink": extract_evidence_from_eye_blink,
    "SpontaneousEyeBlink": extract_evidence_from_eye_blink,
}


def aggregate_region_evidence(all_evidence: List[EvidenceItem], region: str, region_cn: str) -> RegionEvidence:
    """聚合某区域的所有证据"""
    region_evidences = [e for e in all_evidence if e.region == region]

    left_score = 0
    right_score = 0

    for e in region_evidences:
        if e.affected_side == "left":
            left_score += e.score
        elif e.affected_side == "right":
            right_score += e.score
        else:
            left_score += e.score // 2
            right_score += e.score // 2

    if left_score == 0 and right_score == 0:
        conclusion = "normal"
    elif left_score > right_score:
        conclusion = "left_affected"
    elif right_score > left_score:
        conclusion = "right_affected"
    else:
        conclusion = "bilateral"

    return RegionEvidence(
        region=region, region_cn=region_cn,
        left_score=left_score, right_score=right_score,
        evidence_count=len(region_evidences), evidences=region_evidences,
        conclusion=conclusion,
    )


def detect_facial_palsy(action_results: Dict[str, Any]) -> PalsyDetectionResult:
    """综合所有动作结果检测面瘫"""
    all_evidence: List[EvidenceItem] = []
    baseline_result = action_results.get("NeutralFace")

    for action_name, result in action_results.items():
        extractor = EVIDENCE_EXTRACTORS.get(action_name)
        if extractor:
            evidences = extractor(result, baseline_result)
            all_evidence.extend(evidences)

    # 按区域聚合
    eye_evidence = aggregate_region_evidence(all_evidence, "eye", "眼部")
    brow_evidence = aggregate_region_evidence(all_evidence, "brow", "眉部")
    mouth_evidence = aggregate_region_evidence(all_evidence, "mouth", "嘴部")
    cheek_evidence = aggregate_region_evidence(all_evidence, "cheek", "面颊")
    nose_evidence = aggregate_region_evidence(all_evidence, "nose", "鼻部")

    # 汇总分数
    left_total = sum([
        eye_evidence.left_score, brow_evidence.left_score,
        mouth_evidence.left_score, cheek_evidence.left_score,
        nose_evidence.left_score,
    ])

    right_total = sum([
        eye_evidence.right_score, brow_evidence.right_score,
        mouth_evidence.right_score, cheek_evidence.right_score,
        nose_evidence.right_score,
    ])

    total_score = left_total + right_total
    total_evidence_count = len(all_evidence)

    # 判断是否面瘫
    if total_score < 3:
        has_palsy = False
        affected_side = "None"
        confidence = max(0, 100 - total_score * 20)
        diagnosis = "未检测到明显面瘫迹象"
    elif abs(left_total - right_total) < 2:
        if total_score >= 6:
            has_palsy = True
            affected_side = "Bilateral"
            confidence = min(90, 50 + total_score * 5)
            diagnosis = "检测到双侧面部异常"
        else:
            has_palsy = False
            affected_side = "Uncertain"
            confidence = 50
            diagnosis = "证据不足,无法确定"
    else:
        has_palsy = True
        if left_total > right_total:
            affected_side = "Left"
            diff = left_total - right_total
        else:
            affected_side = "Right"
            diff = right_total - left_total

        confidence = min(95, 60 + diff * 5 + total_evidence_count * 2)
        side_cn = "左" if affected_side == "Left" else "右"
        diagnosis = f"检测到{side_cn}侧面瘫,累积异常分数:{max(left_total, right_total)}"

    # 关键证据
    key_evidences = sorted(all_evidence, key=lambda e: e.score, reverse=True)[:10]

    return PalsyDetectionResult(
        has_palsy=has_palsy,
        affected_side=affected_side,
        confidence=confidence,
        eye_evidence=eye_evidence,
        brow_evidence=brow_evidence,
        mouth_evidence=mouth_evidence,
        cheek_evidence=cheek_evidence,
        nose_evidence=nose_evidence,
        left_total_score=left_total,
        right_total_score=right_total,
        total_evidence_count=total_evidence_count,
        key_evidences=key_evidences,
        diagnosis_summary=diagnosis,
    )


def generate_palsy_summary_text(result: PalsyDetectionResult) -> str:
    """生成面瘫检测的文本摘要"""
    lines = [
        "=" * 50,
        "面瘫检测结果",
        "=" * 50,
        f"检测到面瘫: {'是' if result.has_palsy else '否'}",
        f"判断患侧: {result.affected_side}",
        f"置信度: {result.confidence:.1f}%",
        "",
        f"左侧异常累积分: {result.left_total_score}",
        f"右侧异常累积分: {result.right_total_score}",
        f"证据总数: {result.total_evidence_count}",
        "",
        "诊断说明:",
        result.diagnosis_summary,
        "",
    ]

    if result.key_evidences:
        lines.append("关键证据:")
        for i, e in enumerate(result.key_evidences[:5], 1):
            lines.append(f"  {i}. [{e.action_name_cn}] {e.indicator_cn}")
            lines.append(f"     严重度: {e.severity.name}, 患侧: {e.affected_side}")
            lines.append(f"     {e.threshold_info}")

    lines.append("=" * 50)
    return "\n".join(lines)