# -*- coding: utf-8 -*-
"""
House-Brackmann 面部神经分级系统评分器
=====================================

House-Brackmann (HB) 分级系统是临床最常用的面瘫评估标准，
被美国耳鼻喉-头颈外科学会面神经委员会推荐为"金标准"。

分级标准:
- I级 (Normal): 正常，各区面肌运动正常
- II级 (Mild): 轻度功能异常，仔细检查可见轻度面肌无力
- III级 (Moderate): 中度功能异常，明显面肌无力但无面部变形
- IV级 (Moderate-Severe): 中重度功能异常，明显面肌无力和/或面部变形
- V级 (Severe): 重度功能异常，仅有几乎不能察觉的面部运动
- VI级 (Complete): 完全麻痹，无运动

参考论文:
- Automatic Facial Paralysis Evaluation Augmented by a Cascaded Encoder Network Structure
- UPFPSG: A New Benchmark for Unilateral Peripheral Facial Paralysis Severity Grading
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np


class HBGrade(Enum):
    """House-Brackmann分级"""
    GRADE_I = 1  # Normal
    GRADE_II = 2  # Mild dysfunction
    GRADE_III = 3  # Moderate dysfunction
    GRADE_IV = 4  # Moderate-severe dysfunction
    GRADE_V = 5  # Severe dysfunction
    GRADE_VI = 6  # Total paralysis


class FacialNerveBranch(Enum):
    """面神经分支"""
    TEMPORAL = "temporal"  # 颞支 - 额肌、眼轮匝肌上部
    ZYGOMATIC = "zygomatic"  # 颧支 - 眼轮匝肌下部
    BUCCAL = "buccal"  # 颊支 - 颧大肌、提上唇肌等
    MARGINAL_MANDIBULAR = "marginal_mandibular"  # 下颌缘支 - 降下唇肌、颏肌
    CERVICAL = "cervical"  # 颈支 - 颈阔肌


@dataclass
class BranchAssessment:
    """单个神经分支评估结果"""
    branch: FacialNerveBranch
    function_pct: float  # 功能百分比 (0-100%)
    grade: HBGrade  # 该分支的HB分级
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def description(self) -> str:
        """分级描述"""
        descriptions = {
            HBGrade.GRADE_I: "正常",
            HBGrade.GRADE_II: "轻度功能异常",
            HBGrade.GRADE_III: "中度功能异常",
            HBGrade.GRADE_IV: "中重度功能异常",
            HBGrade.GRADE_V: "重度功能异常",
            HBGrade.GRADE_VI: "完全麻痹",
        }
        return descriptions.get(self.grade, "未知")


@dataclass
class HBResult:
    """House-Brackmann综合评分结果"""
    grade: HBGrade
    composite_function_pct: float  # 综合功能百分比

    # 各分支评估
    temporal_branch: Optional[BranchAssessment] = None
    zygomatic_branch: Optional[BranchAssessment] = None
    buccal_branch: Optional[BranchAssessment] = None
    marginal_mandibular_branch: Optional[BranchAssessment] = None

    # 静态和动态特征
    resting_symmetry_score: float = 0.0  # 静息对称性 (0-1, 1=对称)
    voluntary_movement_score: float = 0.0  # 自主运动能力 (0-1, 1=正常)
    synkinesis_score: float = 0.0  # 联带运动 (0-1, 0=无)

    # 置信度和依据
    confidence: float = 0.0
    affected_side: str = "none"  # left/right/none
    evidence: Dict[str, Any] = field(default_factory=dict)

    @property
    def grade_roman(self) -> str:
        """返回罗马数字分级"""
        return ["", "I", "II", "III", "IV", "V", "VI"][self.grade.value]

    @property
    def description(self) -> str:
        """分级描述"""
        descriptions = {
            HBGrade.GRADE_I: "正常 - 各区面肌运动正常",
            HBGrade.GRADE_II: "轻度功能异常 - 仔细检查可见轻度面肌无力，可有轻微联带运动",
            HBGrade.GRADE_III: "中度功能异常 - 明显面肌无力但无面部变形，联带运动明显",
            HBGrade.GRADE_IV: "中重度功能异常 - 明显面肌无力和/或面部变形",
            HBGrade.GRADE_V: "重度功能异常 - 仅有几乎不能察觉的面部运动",
            HBGrade.GRADE_VI: "完全麻痹 - 无运动",
        }
        return descriptions.get(self.grade, "未知")

    @property
    def clinical_features(self) -> Dict[str, str]:
        """各分级的临床特征"""
        features = {
            HBGrade.GRADE_I: {
                "gross": "正常",
                "at_rest": "面部对称，肌张力正常",
                "forehead": "正常",
                "eye": "用力/不用力均能完全闭合",
                "mouth": "正常对称"
            },
            HBGrade.GRADE_II: {
                "gross": "仔细检查时有轻度面肌无力",
                "at_rest": "面部对称，肌张力正常",
                "forehead": "正常功能",
                "eye": "稍用力即可完全闭合",
                "mouth": "轻度不对称"
            },
            HBGrade.GRADE_III: {
                "gross": "明显面肌无力，但无面部变形",
                "at_rest": "面部对称，肌张力正常",
                "forehead": "减弱运动",
                "eye": "用力后可完全闭合",
                "mouth": "用最大力后轻度不对称"
            },
            HBGrade.GRADE_IV: {
                "gross": "明显面肌无力和/或面部变形",
                "at_rest": "面部对称，肌张力正常",
                "forehead": "无运动",
                "eye": "闭眼不完全",
                "mouth": "用最大力后不对称"
            },
            HBGrade.GRADE_V: {
                "gross": "仅有几乎不能察觉的面部运动",
                "at_rest": "面部不对称",
                "forehead": "无运动",
                "eye": "闭眼不完全",
                "mouth": "轻微运动"
            },
            HBGrade.GRADE_VI: {
                "gross": "完全麻痹",
                "at_rest": "面部不对称",
                "forehead": "无运动",
                "eye": "无运动",
                "mouth": "无运动"
            },
        }
        return features.get(self.grade, {})


class HouseBrackmannScorer:
    """
    House-Brackmann评分计算器

    使用方法:
    ```python
    scorer = HouseBrackmannScorer()

    result = scorer.compute_score(
        neutral_indicators=neutral_indicators,
        action_results={
            'RaiseEyebrow': raise_brow_indicators,
            'CloseEyeHardly': close_eye_indicators,
            'Smile': smile_indicators,
            'ShowTeeth': show_teeth_indicators,
        }
    )

    print(f"HB分级: {result.grade_roman} ({result.description})")
    ```
    """

    # ============ 功能百分比 → HB分级 阈值 ============
    # 参考 UPFPSG 论文 Table I

    THRESHOLD_GRADE_I = 100.0  # 100% = Grade I (Normal)
    THRESHOLD_GRADE_II = 75.0  # 75-99% = Grade II (Mild)
    THRESHOLD_GRADE_III = 50.0  # 50-74% = Grade III (Moderate)
    THRESHOLD_GRADE_IV = 25.0  # 25-49% = Grade IV (Moderate-Severe)
    THRESHOLD_GRADE_V = 0.0  # 1-24% = Grade V (Severe)
    # 0% = Grade VI (Complete)

    # ============ 分支权重 ============
    # 综合评分时各分支的权重
    BRANCH_WEIGHTS = {
        FacialNerveBranch.TEMPORAL: 0.25,  # 颞支 (额部)
        FacialNerveBranch.ZYGOMATIC: 0.25,  # 颧支 (眼部)
        FacialNerveBranch.BUCCAL: 0.30,  # 颊支 (中面部)
        FacialNerveBranch.MARGINAL_MANDIBULAR: 0.20  # 下颌缘支 (口部)
    }

    # ============ 动作 → 分支映射 ============
    ACTION_TO_BRANCH = {
        'RaiseEyebrow': FacialNerveBranch.TEMPORAL,
        'CloseEyeSoftly': FacialNerveBranch.ZYGOMATIC,
        'CloseEyeHardly': FacialNerveBranch.ZYGOMATIC,
        'VoluntaryEyeBlink': FacialNerveBranch.ZYGOMATIC,
        'Smile': FacialNerveBranch.BUCCAL,
        'ShowTeeth': FacialNerveBranch.BUCCAL,
        'ShrugNose': FacialNerveBranch.BUCCAL,
        'LipPucker': FacialNerveBranch.MARGINAL_MANDIBULAR,
        'BlowCheek': FacialNerveBranch.MARGINAL_MANDIBULAR,
    }

    def __init__(self):
        pass

    def function_pct_to_grade(self, function_pct: float) -> HBGrade:
        """
        将功能百分比映射到HB分级

        P_a/h: 患侧相对健侧的运动功能百分比

        - Grade I: P = 100%
        - Grade II: 75% < P ≤ 99%
        - Grade III: 50% < P ≤ 75%
        - Grade IV: 25% < P ≤ 50%
        - Grade V: 0% < P ≤ 25%
        - Grade VI: P = 0%
        """
        if function_pct >= 99.0:
            return HBGrade.GRADE_I
        elif function_pct >= self.THRESHOLD_GRADE_II:
            return HBGrade.GRADE_II
        elif function_pct >= self.THRESHOLD_GRADE_III:
            return HBGrade.GRADE_III
        elif function_pct >= self.THRESHOLD_GRADE_IV:
            return HBGrade.GRADE_IV
        elif function_pct > 0:
            return HBGrade.GRADE_V
        else:
            return HBGrade.GRADE_VI

    def compute_branch_function(
            self,
            branch: FacialNerveBranch,
            action_indicators: Dict[str, Dict[str, float]],
            neutral_indicators: Dict[str, float] = None,
            affected_side: str = None
    ) -> BranchAssessment:
        """
        计算单个神经分支的功能

        Args:
            branch: 神经分支
            action_indicators: 该分支相关动作的指标
            neutral_indicators: 静息帧指标
            affected_side: 患侧

        Returns:
            BranchAssessment
        """
        evidence = {'branch': branch.value, 'actions': list(action_indicators.keys())}

        if not action_indicators:
            # 无数据，假设正常
            return BranchAssessment(
                branch=branch,
                function_pct=100.0,
                grade=HBGrade.GRADE_I,
                evidence={**evidence, 'note': '无动作数据，假设正常'}
            )

        # 根据分支选择合适的指标计算方法
        if branch == FacialNerveBranch.TEMPORAL:
            function_pct = self._compute_temporal_function(action_indicators, neutral_indicators)
        elif branch == FacialNerveBranch.ZYGOMATIC:
            function_pct = self._compute_zygomatic_function(action_indicators, neutral_indicators)
        elif branch == FacialNerveBranch.BUCCAL:
            function_pct = self._compute_buccal_function(action_indicators, neutral_indicators)
        elif branch == FacialNerveBranch.MARGINAL_MANDIBULAR:
            function_pct = self._compute_marginal_function(action_indicators, neutral_indicators)
        else:
            function_pct = 100.0

        grade = self.function_pct_to_grade(function_pct)

        evidence['function_pct'] = function_pct
        evidence['grade'] = grade.value

        return BranchAssessment(
            branch=branch,
            function_pct=function_pct,
            grade=grade,
            evidence=evidence
        )

    def _compute_temporal_function(
            self,
            action_indicators: Dict[str, Dict[str, float]],
            neutral_indicators: Dict[str, float] = None
    ) -> float:
        """
        计算颞支功能 (Temporal Branch)

        主要动作: RaiseEyebrow (抬眉)
        评估: 额部运动能力

        HB特征:
        - I/II: 额部正常功能
        - III: 额部减弱
        - IV/V/VI: 额部无运动
        """
        raise_brow = action_indicators.get('RaiseEyebrow', {})

        if not raise_brow:
            return 100.0

        # 获取抬眉指标
        lift_ratio = raise_brow.get('lift_ratio', 1.0)
        lift_asymmetry = raise_brow.get('lift_asymmetry', 0.0)
        function_pct = raise_brow.get('function_pct', None)

        if function_pct is not None:
            return function_pct

        # 从比值计算功能百分比
        # lift_ratio = 患侧/健侧, 越接近1越好
        if lift_ratio > 1:
            lift_ratio = 1 / lift_ratio  # 规范化为 <=1

        function_pct = lift_ratio * 100

        return max(0, min(100, function_pct))

    def _compute_zygomatic_function(
            self,
            action_indicators: Dict[str, Dict[str, float]],
            neutral_indicators: Dict[str, float] = None
    ) -> float:
        """
        计算颧支功能 (Zygomatic Branch)

        主要动作: CloseEyeSoftly, CloseEyeHardly
        评估: 眼部闭合能力

        HB特征:
        - II: 稍用力完全闭合
        - III: 用力完全闭合
        - IV: 闭眼不完全
        - V/VI: 闭眼不完全
        """
        # 优先使用用力闭眼
        close_eye = action_indicators.get('CloseEyeHardly',
                                          action_indicators.get('CloseEyeSoftly', {}))

        if not close_eye:
            return 100.0

        # 获取闭眼指标
        closure_ratio = close_eye.get('closure_ratio', 1.0)
        both_complete = close_eye.get('both_complete_closure', True)
        left_complete = close_eye.get('left_complete_closure', True)
        right_complete = close_eye.get('right_complete_closure', True)
        function_pct = close_eye.get('function_pct', None)

        if function_pct is not None:
            return function_pct

        # 基于闭合情况计算
        if both_complete:
            # 两眼都能完全闭合
            # 但可能有不对称
            asymmetry = abs(1 - closure_ratio)
            function_pct = max(50, 100 - asymmetry * 50)
        elif left_complete or right_complete:
            # 一侧能完全闭合，一侧不能
            function_pct = 40  # Grade IV
        else:
            # 两侧都不能完全闭合
            function_pct = 20  # Grade V

        return max(0, min(100, function_pct))

    def _compute_buccal_function(
            self,
            action_indicators: Dict[str, Dict[str, float]],
            neutral_indicators: Dict[str, float] = None
    ) -> float:
        """
        计算颊支功能 (Buccal Branch)

        主要动作: Smile, ShowTeeth, ShrugNose
        评估: 中面部运动能力 (微笑、露齿、皱鼻)

        HB特征:
        - I/II: 口角正常/轻度不对称
        - III: 用力后轻度不对称
        - IV: 用力后不对称
        - V: 口角轻微运动
        - VI: 无运动
        """
        function_pcts = []

        # Smile
        smile = action_indicators.get('Smile', {})
        if smile:
            ratio = smile.get('oral_excursion_ratio', smile.get('smile_ratio', 1.0))
            pct = smile.get('function_pct', None)
            if pct is None:
                if ratio > 1:
                    ratio = 1 / ratio
                pct = ratio * 100
            function_pcts.append(pct)

        # ShowTeeth
        show_teeth = action_indicators.get('ShowTeeth', {})
        if show_teeth:
            ratio = show_teeth.get('teeth_exposure_ratio', 1.0)
            pct = show_teeth.get('function_pct', None)
            if pct is None:
                if ratio > 1:
                    ratio = 1 / ratio
                pct = ratio * 100
            function_pcts.append(pct)

        # ShrugNose
        shrug_nose = action_indicators.get('ShrugNose', {})
        if shrug_nose:
            ratio = shrug_nose.get('nostril_flare_ratio', 1.0)
            pct = shrug_nose.get('function_pct', None)
            if pct is None:
                if isinstance(ratio, float) and ratio > 0:
                    if ratio > 1:
                        ratio = 1 / ratio
                    pct = ratio * 100
                else:
                    pct = 100
            function_pcts.append(pct)

        if not function_pcts:
            return 100.0

        # 取平均或最差值
        return np.mean(function_pcts)

    def _compute_marginal_function(
            self,
            action_indicators: Dict[str, Dict[str, float]],
            neutral_indicators: Dict[str, float] = None
    ) -> float:
        """
        计算下颌缘支功能 (Marginal Mandibular Branch)

        主要动作: LipPucker, BlowCheek
        评估: 口轮匝肌功能
        """
        function_pcts = []

        # LipPucker
        lip_pucker = action_indicators.get('LipPucker', {})
        if lip_pucker:
            symmetry = lip_pucker.get('pucker_symmetry', 1.0)
            pct = lip_pucker.get('function_pct', None)
            if pct is None:
                if symmetry > 1:
                    symmetry = 1 / symmetry
                pct = symmetry * 100
            function_pcts.append(pct)

        # BlowCheek
        blow_cheek = action_indicators.get('BlowCheek', {})
        if blow_cheek:
            ratio = blow_cheek.get('cheek_volume_ratio', 1.0)
            pct = blow_cheek.get('function_pct', None)
            if pct is None:
                if ratio > 1:
                    ratio = 1 / ratio
                pct = ratio * 100
            function_pcts.append(pct)

        if not function_pcts:
            return 100.0

        return np.mean(function_pcts)

    def compute_resting_symmetry(
            self,
            neutral_indicators: Dict[str, float]
    ) -> float:
        """
        计算静息对称性评分 (0-1, 1=完全对称)

        HB标准中:
        - I-IV级: 静息时面部对称，肌张力正常
        - V-VI级: 静息时面部不对称
        """
        if not neutral_indicators:
            return 1.0

        # 各部位对称性
        eye_ratio = neutral_indicators.get('eye_area_ratio', 1.0)
        nlf_ratio = neutral_indicators.get('nlf_length_ratio', 1.0)
        oral_diff = neutral_indicators.get('oral_height_diff', 0.0)

        # 计算各部位的对称性分数
        eye_sym = 1.0 - min(1.0, abs(1.0 - eye_ratio))
        nlf_sym = 1.0 - min(1.0, abs(1.0 - nlf_ratio))
        mouth_sym = 1.0 - min(1.0, abs(oral_diff) * 10)  # 放大口角差异

        # 加权平均
        symmetry = 0.35 * eye_sym + 0.35 * nlf_sym + 0.30 * mouth_sym

        return max(0, min(1, symmetry))

    def compute_score(
            self,
            neutral_indicators: Dict[str, float],
            action_results: Dict[str, Dict[str, float]],
            affected_side: str = None
    ) -> HBResult:
        """
        计算House-Brackmann综合评分

        Args:
            neutral_indicators: NeutralFace指标
            action_results: 各动作指标字典
            affected_side: 患侧 (可选)

        Returns:
            HBResult
        """
        evidence = {}

        # 1. 计算静息对称性
        resting_symmetry = self.compute_resting_symmetry(neutral_indicators)
        evidence['resting_symmetry'] = resting_symmetry

        # 2. 判断患侧 (如果未指定)
        if affected_side is None:
            affected_side = self._determine_affected_side(neutral_indicators)
        evidence['affected_side'] = affected_side

        # 3. 按分支整理动作
        branch_actions = {
            FacialNerveBranch.TEMPORAL: {},
            FacialNerveBranch.ZYGOMATIC: {},
            FacialNerveBranch.BUCCAL: {},
            FacialNerveBranch.MARGINAL_MANDIBULAR: {},
        }

        for action_name, indicators in action_results.items():
            branch = self.ACTION_TO_BRANCH.get(action_name)
            if branch is not None:
                branch_actions[branch][action_name] = indicators

        # 4. 计算各分支功能
        temporal = self.compute_branch_function(
            FacialNerveBranch.TEMPORAL,
            branch_actions[FacialNerveBranch.TEMPORAL],
            neutral_indicators, affected_side
        )

        zygomatic = self.compute_branch_function(
            FacialNerveBranch.ZYGOMATIC,
            branch_actions[FacialNerveBranch.ZYGOMATIC],
            neutral_indicators, affected_side
        )

        buccal = self.compute_branch_function(
            FacialNerveBranch.BUCCAL,
            branch_actions[FacialNerveBranch.BUCCAL],
            neutral_indicators, affected_side
        )

        marginal = self.compute_branch_function(
            FacialNerveBranch.MARGINAL_MANDIBULAR,
            branch_actions[FacialNerveBranch.MARGINAL_MANDIBULAR],
            neutral_indicators, affected_side
        )

        # 5. 计算加权综合功能百分比
        composite_pct = (
                self.BRANCH_WEIGHTS[FacialNerveBranch.TEMPORAL] * temporal.function_pct +
                self.BRANCH_WEIGHTS[FacialNerveBranch.ZYGOMATIC] * zygomatic.function_pct +
                self.BRANCH_WEIGHTS[FacialNerveBranch.BUCCAL] * buccal.function_pct +
                self.BRANCH_WEIGHTS[FacialNerveBranch.MARGINAL_MANDIBULAR] * marginal.function_pct
        )

        # 6. 确定最终HB分级
        # 考虑静息对称性的影响
        if resting_symmetry < 0.7:
            # 静息不对称，至少Grade V
            grade = HBGrade.GRADE_V
            if composite_pct < 5:
                grade = HBGrade.GRADE_VI
        else:
            grade = self.function_pct_to_grade(composite_pct)

        # 7. 计算置信度
        confidence = self._compute_confidence(
            temporal, zygomatic, buccal, marginal, resting_symmetry
        )

        evidence['composite_function_pct'] = composite_pct
        evidence['branch_function_pcts'] = {
            'temporal': temporal.function_pct,
            'zygomatic': zygomatic.function_pct,
            'buccal': buccal.function_pct,
            'marginal_mandibular': marginal.function_pct,
        }

        return HBResult(
            grade=grade,
            composite_function_pct=composite_pct,
            temporal_branch=temporal,
            zygomatic_branch=zygomatic,
            buccal_branch=buccal,
            marginal_mandibular_branch=marginal,
            resting_symmetry_score=resting_symmetry,
            voluntary_movement_score=composite_pct / 100,
            confidence=confidence,
            affected_side=affected_side,
            evidence=evidence
        )

    def _determine_affected_side(
            self,
            indicators: Dict[str, float]
    ) -> str:
        """根据静态指标判断患侧"""
        if not indicators:
            return "none"

        votes = {"left": 0, "right": 0}

        # 眼部: 面积小的一侧
        eye_ratio = indicators.get('eye_area_ratio', 1.0)
        if eye_ratio < 0.9:
            votes["left"] += 1
        elif eye_ratio > 1.1:
            votes["right"] += 1

        # 鼻唇沟: 短/浅的一侧
        nlf_ratio = indicators.get('nlf_length_ratio', 1.0)
        if nlf_ratio < 0.9:
            votes["left"] += 1
        elif nlf_ratio > 1.1:
            votes["right"] += 1

        # 口角: 低的一侧
        oral_diff = indicators.get('oral_height_diff', 0)
        if oral_diff < -0.015:
            votes["left"] += 1
        elif oral_diff > 0.015:
            votes["right"] += 1

        if votes["left"] > votes["right"]:
            return "left"
        elif votes["right"] > votes["left"]:
            return "right"
        else:
            return "none"

    def _compute_confidence(
            self,
            temporal: BranchAssessment,
            zygomatic: BranchAssessment,
            buccal: BranchAssessment,
            marginal: BranchAssessment,
            resting_symmetry: float
    ) -> float:
        """计算评分置信度"""
        # 基于各分支评分的一致性
        grades = [
            temporal.grade.value,
            zygomatic.grade.value,
            buccal.grade.value,
            marginal.grade.value
        ]

        # 计算标准差，越小越一致
        std = np.std(grades)
        consistency = 1.0 - min(1.0, std / 3.0)

        # 基础置信度
        confidence = 0.6 + 0.4 * consistency

        return min(1.0, confidence)


# ============ 便捷函数 ============

def compute_hb_grade(
        neutral_indicators: Dict[str, float],
        action_results: Dict[str, Dict[str, float]],
        affected_side: str = None
) -> HBResult:
    """
    便捷函数：计算House-Brackmann分级
    """
    scorer = HouseBrackmannScorer()
    return scorer.compute_score(neutral_indicators, action_results, affected_side)


def grade_to_text(grade: HBGrade) -> str:
    """将HB分级转换为文本描述"""
    texts = {
        HBGrade.GRADE_I: "I级 (正常)",
        HBGrade.GRADE_II: "II级 (轻度)",
        HBGrade.GRADE_III: "III级 (中度)",
        HBGrade.GRADE_IV: "IV级 (中重度)",
        HBGrade.GRADE_V: "V级 (重度)",
        HBGrade.GRADE_VI: "VI级 (完全麻痹)",
    }
    return texts.get(grade, "未知")


if __name__ == "__main__":
    # 测试代码
    print("House-Brackmann Scorer - 单元测试")

    # 模拟数据: 中度面瘫
    neutral = {
        'eye_area_ratio': 0.82,
        'nlf_length_ratio': 0.85,
        'oral_height_diff': -0.025,
    }

    actions = {
        'RaiseEyebrow': {'lift_ratio': 0.6},
        'CloseEyeHardly': {'closure_ratio': 0.7, 'both_complete_closure': True},
        'Smile': {'oral_excursion_ratio': 0.65},
        'ShrugNose': {'nostril_flare_ratio': 0.7},
        'LipPucker': {'pucker_symmetry': 0.75},
        'BlowCheek': {'cheek_volume_ratio': 0.8},
    }

    result = compute_hb_grade(neutral, actions)

    print(f"\n=== House-Brackmann 评分结果 ===")
    print(f"综合分级: {result.grade_roman} ({result.description})")
    print(f"综合功能百分比: {result.composite_function_pct:.1f}%")
    print(f"患侧: {result.affected_side}")
    print(f"置信度: {result.confidence:.2f}")

    print(f"\n各分支评估:")
    print(f"  颞支 (额部): {result.temporal_branch.function_pct:.1f}% - {result.temporal_branch.description}")
    print(f"  颧支 (眼部): {result.zygomatic_branch.function_pct:.1f}% - {result.zygomatic_branch.description}")
    print(f"  颊支 (中面部): {result.buccal_branch.function_pct:.1f}% - {result.buccal_branch.description}")
    print(
        f"  下颌缘支 (口部): {result.marginal_mandibular_branch.function_pct:.1f}% - {result.marginal_mandibular_branch.description}")

    print(f"\n静息对称性: {result.resting_symmetry_score:.2f}")

    print(f"\n临床特征:")
    for key, value in result.clinical_features.items():
        print(f"  {key}: {value}")