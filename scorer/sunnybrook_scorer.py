#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sunnybrook 面神经评分系统
=========================

基于 Ross et al. (1996) 的Sunnybrook面神经分级系统

组成部分:
1. Resting Symmetry (静息对称性): 0-20分
2. Symmetry of Voluntary Movement (主动运动对称性): 20-100分
3. Synkinesis (联动运动): 0-15分

Composite Score = Voluntary Movement Score - Resting Symmetry Score - Synkinesis Score
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import json


# =============================================================================
# Resting Symmetry (静息对称性)
# =============================================================================

@dataclass
class RestingSymmetryItem:
    """静息对称性单项评估"""
    region: str  # 区域: Eye, Cheek, Mouth
    region_cn: str  # 区域中文
    status: str  # 状态描述
    status_cn: str  # 状态中文
    score: int  # 评分
    measurement: float  # 测量值
    threshold_info: str  # 阈值说明


@dataclass
class RestingSymmetry:
    """静息对称性评估 (Sunnybrook Resting Symmetry)

    评分标准:
    - Eye: Normal(0), Narrow(1), Wide(1), Eyelid Surgery(1)
    - Cheek (NLF): Normal(0), Absent(2), Less pronounced(1), More pronounced(1)
    - Mouth: Normal(0), Corner dropped(1), Corner pulled up/out(1)

    Total Score = (Eye + Cheek + Mouth) × 5
    Range: 0-20
    """
    eye: RestingSymmetryItem
    cheek: RestingSymmetryItem
    mouth: RestingSymmetryItem

    raw_score: int  # 原始分 (0-4)
    total_score: int  # Sunnybrook分数 = raw × 5 (0-20)
    affected_side: str  # 判断的患侧

    def to_dict(self) -> Dict[str, Any]:
        return {
            "eye": {
                "region": self.eye.region,
                "region_cn": self.eye.region_cn,
                "status": self.eye.status,
                "status_cn": self.eye.status_cn,
                "score": self.eye.score,
                "measurement": self.eye.measurement,
                "threshold_info": self.eye.threshold_info,
            },
            "cheek": {
                "region": self.cheek.region,
                "region_cn": self.cheek.region_cn,
                "status": self.cheek.status,
                "status_cn": self.cheek.status_cn,
                "score": self.cheek.score,
                "measurement": self.cheek.measurement,
                "threshold_info": self.cheek.threshold_info,
            },
            "mouth": {
                "region": self.mouth.region,
                "region_cn": self.mouth.region_cn,
                "status": self.mouth.status,
                "status_cn": self.mouth.status_cn,
                "score": self.mouth.score,
                "measurement": self.mouth.measurement,
                "threshold_info": self.mouth.threshold_info,
            },
            "raw_score": self.raw_score,
            "total_score": self.total_score,
            "affected_side": self.affected_side,
        }


def compute_resting_symmetry(palpebral_height_ratio: float,
                             nlf_ratio: float,
                             oral_angle_diff: float,
                             aoe_angle: float,
                             bof_angle: float) -> RestingSymmetry:
    """
    计算静息对称性评估

    Args:
        palpebral_height_ratio: 睑裂高度比 (左/右)
        nlf_ratio: 鼻唇沟长度比 (左/右)
        oral_angle_diff: 口角角度差 (BOF - AOE)
        aoe_angle: 右口角角度
        bof_angle: 左口角角度
    """
    # ========== Eye (眼) ==========
    # Normal: 0.85-1.15, Narrow: <0.85 or >1.15
    if 0.85 <= palpebral_height_ratio <= 1.15:
        eye_status = "Normal"
        eye_status_cn = "正常"
        eye_score = 0
        eye_threshold = "比值在0.85-1.15之间"
    elif palpebral_height_ratio < 0.85:
        eye_status = "Narrow (Left)"
        eye_status_cn = "左侧眼裂缩窄"
        eye_score = 1
        eye_threshold = f"比值{palpebral_height_ratio:.3f}<0.85"
    else:
        eye_status = "Wide (Left)"
        eye_status_cn = "左侧眼裂增宽"
        eye_score = 1
        eye_threshold = f"比值{palpebral_height_ratio:.3f}>1.15"

    eye_item = RestingSymmetryItem(
        region="Eye", region_cn="眼(睑裂)",
        status=eye_status, status_cn=eye_status_cn,
        score=eye_score, measurement=palpebral_height_ratio,
        threshold_info=eye_threshold
    )

    # ========== Cheek (颊/鼻唇沟) ==========
    # Normal: 0.85-1.15
    # Less pronounced: 0.75-0.85 or 1.15-1.25
    # Absent: <0.75 or >1.25
    if 0.85 <= nlf_ratio <= 1.15:
        cheek_status = "Normal"
        cheek_status_cn = "正常"
        cheek_score = 0
        cheek_threshold = "比值在0.85-1.15之间"
    elif 0.75 <= nlf_ratio < 0.85:
        cheek_status = "Less pronounced (Left)"
        cheek_status_cn = "左侧鼻唇沟不明显"
        cheek_score = 1
        cheek_threshold = f"比值{nlf_ratio:.3f}在0.75-0.85之间"
    elif 1.15 < nlf_ratio <= 1.25:
        cheek_status = "Less pronounced (Right)"
        cheek_status_cn = "右侧鼻唇沟不明显"
        cheek_score = 1
        cheek_threshold = f"比值{nlf_ratio:.3f}在1.15-1.25之间"
    elif nlf_ratio < 0.75:
        cheek_status = "Absent (Left)"
        cheek_status_cn = "左侧鼻唇沟消失"
        cheek_score = 2
        cheek_threshold = f"比值{nlf_ratio:.3f}<0.75"
    else:
        cheek_status = "More pronounced (Left)"
        cheek_status_cn = "左侧鼻唇沟过深"
        cheek_score = 1
        cheek_threshold = f"比值{nlf_ratio:.3f}>1.25"

    cheek_item = RestingSymmetryItem(
        region="Cheek", region_cn="颊(鼻唇沟)",
        status=cheek_status, status_cn=cheek_status_cn,
        score=cheek_score, measurement=nlf_ratio,
        threshold_info=cheek_threshold
    )

    # ========== Mouth (嘴) ==========
    # Normal: |diff| <= 5°
    # Corner dropped/pulled: |diff| > 5°
    MOUTH_THRESHOLD = 5.0
    if abs(oral_angle_diff) <= MOUTH_THRESHOLD:
        mouth_status = "Normal"
        mouth_status_cn = "正常"
        mouth_score = 0
        mouth_threshold = f"角度差{oral_angle_diff:+.1f}°在±{MOUTH_THRESHOLD}°之内"
    elif oral_angle_diff < -MOUTH_THRESHOLD:
        # 左口角比右口角低更多 -> 左侧下垂
        mouth_status = "Corner dropped (Left)"
        mouth_status_cn = "左侧口角下垂"
        mouth_score = 1
        mouth_threshold = f"左{bof_angle:+.1f}° vs 右{aoe_angle:+.1f}°"
    else:
        # 左口角比右口角高 -> 右侧下垂(或左侧上提)
        mouth_status = "Corner dropped (Right)"
        mouth_status_cn = "右侧口角下垂"
        mouth_score = 1
        mouth_threshold = f"左{bof_angle:+.1f}° vs 右{aoe_angle:+.1f}°"

    mouth_item = RestingSymmetryItem(
        region="Mouth", region_cn="嘴",
        status=mouth_status, status_cn=mouth_status_cn,
        score=mouth_score, measurement=oral_angle_diff,
        threshold_info=mouth_threshold
    )

    # ========== 计算总分和判断患侧 ==========
    raw_score = eye_score + cheek_score + mouth_score
    total_score = raw_score * 5

    # 判断患侧
    left_signs = 0
    right_signs = 0

    if "Left" in eye_status:
        if "Narrow" in eye_status:
            left_signs += 1
        else:
            right_signs += 1

    if "Left" in cheek_status:
        if "Less" in cheek_status or "Absent" in cheek_status:
            left_signs += 1
        else:
            right_signs += 1
    if "Right" in cheek_status:
        right_signs += 1

    if "Left" in mouth_status:
        left_signs += 1
    if "Right" in mouth_status:
        right_signs += 1

    if left_signs > right_signs:
        affected_side = "Left (左侧)"
    elif right_signs > left_signs:
        affected_side = "Right (右侧)"
    else:
        affected_side = "Uncertain (不确定)"

    return RestingSymmetry(
        eye=eye_item,
        cheek=cheek_item,
        mouth=mouth_item,
        raw_score=raw_score,
        total_score=total_score,
        affected_side=affected_side
    )


# =============================================================================
# Symmetry of Voluntary Movement (主动运动对称性)
# =============================================================================

@dataclass
class VoluntaryMovementItem:
    """主动运动单项评估

    评分标准 (1-5):
    1 = Unable to initiate movement (无法启动运动)
    2 = Initiates slight movement (轻微启动)
    3 = Initiates movement with mild asymmetry (启动但不对称)
    4 = Movement almost complete (几乎完整)
    5 = Movement complete (完整)
    """
    expression: str  # 表情名称
    expression_cn: str  # 表情中文

    left_value: float  # 左侧测量值
    right_value: float  # 右侧测量值
    ratio: float  # 比值

    baseline_left: Optional[float] = None  # 基线左侧值
    baseline_right: Optional[float] = None  # 基线右侧值
    excursion_left: Optional[float] = None  # 左侧运动幅度
    excursion_right: Optional[float] = None  # 右侧运动幅度
    excursion_ratio: Optional[float] = None  # 运动幅度比

    score: int = 5  # 评分 (1-5)
    interpretation: str = ""  # 解释


@dataclass
class VoluntaryMovement:
    """主动运动对称性评估 (Sunnybrook Voluntary Movement)

    标准表情:
    - Brow (FRD): 皱额/抬眉
    - Gentle Eye closure (OCS): 轻闭眼
    - Open mouth smile (ZYG/RIS): 露齿微笑
    - Snarl (LLA/LLS): 耸鼻/龇牙
    - Lip pucker (OOS/OOI): 撅嘴

    Total Score = Sum × 4
    Range: 20-100 (满分100)
    """
    items: List[VoluntaryMovementItem]

    raw_sum: int  # 原始分总和 (5-25)
    total_score: int  # Sunnybrook分数 = sum × 4 (20-100)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "items": [
                {
                    "expression": item.expression,
                    "expression_cn": item.expression_cn,
                    "left_value": item.left_value,
                    "right_value": item.right_value,
                    "ratio": item.ratio,
                    "baseline_left": item.baseline_left,
                    "baseline_right": item.baseline_right,
                    "excursion_left": item.excursion_left,
                    "excursion_right": item.excursion_right,
                    "excursion_ratio": item.excursion_ratio,
                    "score": item.score,
                    "interpretation": item.interpretation,
                }
                for item in self.items
            ],
            "raw_sum": self.raw_sum,
            "total_score": self.total_score,
        }


def compute_voluntary_score_from_ratio(ratio: float,
                                       excursion_ratio: Optional[float] = None) -> Tuple[int, str]:
    """
    根据比值计算主动运动评分

    Args:
        ratio: 左/右比值 (接近1.0为对称)
        excursion_ratio: 运动幅度比值 (可选)

    Returns:
        (score, interpretation)
    """
    # 使用偏离1.0的程度来评分
    deviation = abs(ratio - 1.0)

    if deviation <= 0.05:
        return 5, "Movement complete (运动完整)"
    elif deviation <= 0.15:
        return 4, "Movement almost complete (几乎完整)"
    elif deviation <= 0.30:
        return 3, "Initiates with mild asymmetry (轻度不对称)"
    elif deviation <= 0.50:
        return 2, "Initiates slight movement (轻微运动)"
    else:
        return 1, "Unable to initiate movement (无法启动)"


def compute_voluntary_score_from_excursion(excursion_ratio: float) -> Tuple[int, str]:
    """
    根据运动幅度比计算Sunnybrook Voluntary Movement评分

    excursion_ratio = 患侧运动幅度 / 健侧运动幅度

    Sunnybrook原始定义:
    - Score 5: ≥95% (Complete)
    - Score 4: 75-94% (Almost complete)
    - Score 3: 50-74% (Mild asymmetry)
    - Score 2: 25-49% (Slight movement)
    - Score 1: <25% (Unable to initiate)
    """
    if excursion_ratio >= 0.95:
        return 5, "Movement complete (运动完整)"
    elif excursion_ratio >= 0.75:
        return 4, "Movement almost complete (几乎完整)"
    elif excursion_ratio >= 0.50:
        return 3, "Initiates with mild asymmetry (轻度不对称)"
    elif excursion_ratio >= 0.25:
        return 2, "Initiates slight movement (轻微运动)"
    else:
        return 1, "Unable to initiate movement (无法启动)"


# Sunnybrook标准表情到我们的动作映射
SUNNYBROOK_EXPRESSION_MAPPING = {
    "Brow": {
        "action": "RaiseEyebrow",
        "cn": "皱额/抬眉",
        "metric": "brow_height",
        "description": "Forehead wrinkle"
    },
    "GentleEyeClosure": {
        "action": "CloseEyeSoftly",
        "cn": "轻闭眼",
        "metric": "eye_closure",
        "description": "Gentle eye closure"
    },
    "OpenMouthSmile": {
        "action": "Smile",  # 或 ShowTeeth
        "cn": "露齿微笑",
        "metric": "smile",
        "description": "Open mouth smile"
    },
    "Snarl": {
        "action": "ShrugNose",
        "cn": "耸鼻",
        "metric": "nose_wrinkle",
        "description": "Snarl/nose wrinkle"
    },
    "LipPucker": {
        "action": "LipPucker",
        "cn": "撅嘴",
        "metric": "lip_pucker",
        "description": "Lip pucker"
    }
}


# =============================================================================
# Synkinesis (联动运动)
# =============================================================================

@dataclass
class SynkinesisItem:
    """联动运动单项评估

    评分标准 (0-3):
    0 = None (无联动)
    1 = Mild (轻度联动)
    2 = Moderate (中度联动)
    3 = Severe (重度联动/畸形)
    """
    expression: str  # 评估时的表情
    expression_cn: str

    # 各区域联动程度
    brow_synkinesis: int = 0  # 眉部联动
    eye_synkinesis: int = 0  # 眼部联动
    cheek_synkinesis: int = 0  # 面颊联动
    mouth_synkinesis: int = 0  # 嘴部联动
    chin_synkinesis: int = 0  # 下巴联动

    total_score: int = 0
    interpretation: str = ""
    detail: str = ""


@dataclass
class Synkinesis:
    """联动运动评估 (Sunnybrook Synkinesis)

    对每个标准表情，评估非目标区域的联动情况

    Total Score = Sum of all synkinesis scores
    Range: 0-15
    """
    items: List[SynkinesisItem]

    total_score: int  # 总分 (0-15)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "items": [
                {
                    "expression": item.expression,
                    "expression_cn": item.expression_cn,
                    "brow_synkinesis": item.brow_synkinesis,
                    "eye_synkinesis": item.eye_synkinesis,
                    "cheek_synkinesis": item.cheek_synkinesis,
                    "mouth_synkinesis": item.mouth_synkinesis,
                    "chin_synkinesis": item.chin_synkinesis,
                    "total_score": item.total_score,
                    "interpretation": item.interpretation,
                    "detail": item.detail,
                }
                for item in self.items
            ],
            "total_score": self.total_score,
        }


def detect_synkinesis_level(change: float, thresholds: Tuple[float, float, float]) -> int:
    """
    检测联动程度

    Args:
        change: 变化量 (通常是与基线的差值比例)
        thresholds: (mild, moderate, severe) 阈值

    Returns:
        0-3 的联动程度
    """
    mild, moderate, severe = thresholds

    if change >= severe:
        return 3
    elif change >= moderate:
        return 2
    elif change >= mild:
        return 1
    else:
        return 0


# =============================================================================
# Sunnybrook 综合评分
# =============================================================================

@dataclass
class SunnybrookScore:
    """完整的Sunnybrook评分"""
    resting_symmetry: RestingSymmetry
    voluntary_movement: VoluntaryMovement
    synkinesis: Synkinesis

    resting_score: int  # 0-20
    voluntary_score: int  # 20-100
    synkinesis_score: int  # 0-15
    composite_score: int  # = voluntary - resting - synkinesis

    # 等级判断
    grade: str = ""
    grade_description: str = ""

    def __post_init__(self):
        # 根据composite score判断等级
        if self.composite_score >= 90:
            self.grade = "I"
            self.grade_description = "Normal (正常)"
        elif self.composite_score >= 70:
            self.grade = "II"
            self.grade_description = "Slight dysfunction (轻度功能障碍)"
        elif self.composite_score >= 50:
            self.grade = "III"
            self.grade_description = "Moderate dysfunction (中度功能障碍)"
        elif self.composite_score >= 30:
            self.grade = "IV"
            self.grade_description = "Moderately severe dysfunction (中重度功能障碍)"
        elif self.composite_score >= 10:
            self.grade = "V"
            self.grade_description = "Severe dysfunction (重度功能障碍)"
        else:
            self.grade = "VI"
            self.grade_description = "Total paralysis (完全麻痹)"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resting_symmetry": self.resting_symmetry.to_dict(),
            "voluntary_movement": self.voluntary_movement.to_dict(),
            "synkinesis": self.synkinesis.to_dict(),
            "scores": {
                "resting_score": self.resting_score,
                "voluntary_score": self.voluntary_score,
                "synkinesis_score": self.synkinesis_score,
                "composite_score": self.composite_score,
            },
            "grade": self.grade,
            "grade_description": self.grade_description,
            "formula": f"Composite = {self.voluntary_score} - {self.resting_score} - {self.synkinesis_score} = {self.composite_score}"
        }


def compute_sunnybrook_composite(resting: RestingSymmetry,
                                 voluntary: VoluntaryMovement,
                                 synkinesis: Synkinesis) -> SunnybrookScore:
    """计算完整的Sunnybrook评分"""
    composite = voluntary.total_score - resting.total_score - synkinesis.total_score

    return SunnybrookScore(
        resting_symmetry=resting,
        voluntary_movement=voluntary,
        synkinesis=synkinesis,
        resting_score=resting.total_score,
        voluntary_score=voluntary.total_score,
        synkinesis_score=synkinesis.total_score,
        composite_score=composite
    )