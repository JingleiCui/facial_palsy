# -*- coding: utf-8 -*-
"""
Sunnybrook 面部神经分级系统评分器
================================

Sunnybrook评分系统包含三个部分:
1. 静态对称性评分 (Resting Symmetry Score): 0-20分, 越低越好
2. 自主运动评分 (Voluntary Movement Score): 20-100分, 越高越好
3. 联带运动评分 (Synkinesis Score): 0-15分, 越低越好

综合评分 = 自主运动评分 - 静态对称性评分 - 联带运动评分
范围: 0-100分, 越高越好

参考论文:
- Inter and Intra-rater Reliability of Modified House-Brackmann and Sunnybrook
  Facial Nerve Grading Systems in Post Parotidectomy Patients
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np


class RestingEyeStatus(Enum):
    """静息眼部状态"""
    NORMAL = 0
    NARROW = 1  # 缩窄
    WIDE = 1  # 增宽
    SURGERY = 1  # 眼睑手术


class RestingCheekStatus(Enum):
    """静息鼻唇沟状态"""
    NORMAL = 0
    LESS_PRONOUNCED = 1  # 不明显
    ABSENT = 2  # 消失
    MORE_PRONOUNCED = 1  # 过于明显


class RestingMouthStatus(Enum):
    """静息口部状态"""
    NORMAL = 0
    CORNER_DROPPED = 1  # 口角下垂
    CORNER_PULLED = 1  # 口角上提


class VoluntaryMovementLevel(Enum):
    """自主运动等级"""
    UNABLE = 1  # 无法启动运动
    INITIATES_SLIGHT = 2  # 启动轻微运动
    INITIATES_WITH_MOVEMENT = 3  # 有运动但不完全
    ALMOST_COMPLETE = 4  # 运动接近完全
    COMPLETE = 5  # 正常完全运动


class SynkinesisLevel(Enum):
    """联带运动等级"""
    NONE = 0  # 无
    MILD = 1  # 轻度
    MODERATE_OBVIOUS = 2  # 中度明显
    SEVERE_DISFIGURING = 3  # 重度毁容


@dataclass
class RestingSymmetryResult:
    """静态对称性评分结果"""
    eye_status: RestingEyeStatus
    eye_score: int
    eye_evidence: Dict[str, Any]

    cheek_status: RestingCheekStatus
    cheek_score: int
    cheek_evidence: Dict[str, Any]

    mouth_status: RestingMouthStatus
    mouth_score: int
    mouth_evidence: Dict[str, Any]

    total_raw: int = 0  # 原始总分 (Eye + Cheek + Mouth)
    total_weighted: int = 0  # 加权总分 = total_raw × 5

    def __post_init__(self):
        self.total_raw = self.eye_score + self.cheek_score + self.mouth_score
        self.total_weighted = self.total_raw * 5


@dataclass
class VoluntaryMovementResult:
    """自主运动评分结果"""
    brow_level: VoluntaryMovementLevel
    brow_score: int
    brow_evidence: Dict[str, Any]

    eye_closure_level: VoluntaryMovementLevel
    eye_closure_score: int
    eye_closure_evidence: Dict[str, Any]

    smile_level: VoluntaryMovementLevel
    smile_score: int
    smile_evidence: Dict[str, Any]

    snarl_level: VoluntaryMovementLevel
    snarl_score: int
    snarl_evidence: Dict[str, Any]

    lip_pucker_level: VoluntaryMovementLevel
    lip_pucker_score: int
    lip_pucker_evidence: Dict[str, Any]

    total_raw: int = 0  # 原始总分
    total_weighted: int = 0  # 加权总分 = total_raw × 4

    def __post_init__(self):
        self.total_raw = (self.brow_score + self.eye_closure_score +
                          self.smile_score + self.snarl_score + self.lip_pucker_score)
        self.total_weighted = self.total_raw * 4


@dataclass
class SynkinesisResult:
    """联带运动评分结果"""
    brow_synkinesis: SynkinesisLevel
    brow_synkinesis_score: int
    brow_synkinesis_evidence: Dict[str, Any]

    eye_closure_synkinesis: SynkinesisLevel
    eye_closure_synkinesis_score: int
    eye_closure_synkinesis_evidence: Dict[str, Any]

    smile_synkinesis: SynkinesisLevel
    smile_synkinesis_score: int
    smile_synkinesis_evidence: Dict[str, Any]

    snarl_synkinesis: SynkinesisLevel
    snarl_synkinesis_score: int
    snarl_synkinesis_evidence: Dict[str, Any]

    lip_pucker_synkinesis: SynkinesisLevel
    lip_pucker_synkinesis_score: int
    lip_pucker_synkinesis_evidence: Dict[str, Any]

    total_score: int = 0

    def __post_init__(self):
        self.total_score = (self.brow_synkinesis_score + self.eye_closure_synkinesis_score +
                            self.smile_synkinesis_score + self.snarl_synkinesis_score +
                            self.lip_pucker_synkinesis_score)


@dataclass
class SunnybrookResult:
    """Sunnybrook综合评分结果"""
    resting_symmetry: RestingSymmetryResult
    voluntary_movement: VoluntaryMovementResult
    synkinesis: SynkinesisResult

    composite_score: float = 0.0  # 综合评分
    affected_side: str = "none"  # 患侧 (left/right/none)
    confidence: float = 0.0  # 置信度

    def __post_init__(self):
        # 综合评分 = 自主运动 - 静态对称性 - 联带运动
        self.composite_score = (self.voluntary_movement.total_weighted -
                                self.resting_symmetry.total_weighted -
                                self.synkinesis.total_score)
        self.composite_score = max(0, min(100, self.composite_score))


class SunnybrookScorer:
    """
    Sunnybrook评分计算器

    使用方法:
    ```python
    scorer = SunnybrookScorer()

    # 从动作指标计算评分
    result = scorer.compute_score(
        neutral_indicators=neutral_indicators,
        action_results={
            'RaiseEyebrow': raise_brow_indicators,
            'CloseEyeSoftly': close_eye_indicators,
            'Smile': smile_indicators,
            'ShrugNose': shrug_nose_indicators,
            'LipPucker': lip_pucker_indicators,
        }
    )

    print(f"综合评分: {result.composite_score}")
    ```
    """

    # ============ 阈值配置 ============

    # 眼部静态对称性阈值
    EYE_NORMAL_LOW = 0.85  # 面积比 >= 0.85 认为对称
    EYE_NORMAL_HIGH = 1.15  # 面积比 <= 1.15 认为对称
    EYE_NARROW_THRESHOLD = 0.85  # < 0.85 认为缩窄
    EYE_WIDE_THRESHOLD = 1.15  # > 1.15 认为增宽

    # 鼻唇沟静态对称性阈值
    CHEEK_NORMAL_LOW = 0.85
    CHEEK_NORMAL_HIGH = 1.15
    CHEEK_LESS_LOW = 0.70
    CHEEK_LESS_HIGH = 1.30
    CHEEK_ABSENT_LOW = 0.60  # < 0.60 认为消失
    CHEEK_ABSENT_HIGH = 1.60  # > 1.60 认为消失

    # 口部静态对称性阈值 (归一化口角高度差)
    MOUTH_NORMAL_THRESHOLD = 0.015  # |diff| <= 0.015 认为对称
    MOUTH_ABNORMAL_THRESHOLD = 0.015  # |diff| > 0.015 认为异常

    # 自主运动功能百分比阈值
    VOLUNTARY_UNABLE_THRESHOLD = 5  # < 5% 无法启动
    VOLUNTARY_SLIGHT_THRESHOLD = 25  # 5-25% 轻微运动
    VOLUNTARY_PARTIAL_THRESHOLD = 50  # 25-50% 部分运动
    VOLUNTARY_ALMOST_THRESHOLD = 75  # 50-75% 接近完全
    VOLUNTARY_COMPLETE_THRESHOLD = 75  # >= 75% 完全运动

    # 联带运动阈值
    SYNKINESIS_NONE_THRESHOLD = 0.05
    SYNKINESIS_MILD_THRESHOLD = 0.15
    SYNKINESIS_MODERATE_THRESHOLD = 0.30

    def __init__(self):
        pass

    # ============ 静态对称性评分 ============

    def compute_resting_symmetry(
            self,
            neutral_indicators: Dict[str, float]
    ) -> RestingSymmetryResult:
        """
        计算静态对称性评分

        基于NeutralFace动作的指标

        Args:
            neutral_indicators: NeutralFace动作提取的指标字典

        Returns:
            RestingSymmetryResult
        """
        # 1. Eye评分
        eye_status, eye_score, eye_evidence = self._score_resting_eye(neutral_indicators)

        # 2. Cheek (鼻唇沟) 评分
        cheek_status, cheek_score, cheek_evidence = self._score_resting_cheek(neutral_indicators)

        # 3. Mouth评分
        mouth_status, mouth_score, mouth_evidence = self._score_resting_mouth(neutral_indicators)

        return RestingSymmetryResult(
            eye_status=eye_status,
            eye_score=eye_score,
            eye_evidence=eye_evidence,
            cheek_status=cheek_status,
            cheek_score=cheek_score,
            cheek_evidence=cheek_evidence,
            mouth_status=mouth_status,
            mouth_score=mouth_score,
            mouth_evidence=mouth_evidence
        )

    def _score_resting_eye(
            self,
            indicators: Dict[str, float]
    ) -> Tuple[RestingEyeStatus, int, Dict[str, Any]]:
        """
        评估静息眼部状态

        指标:
        - eye_area_ratio: 左眼面积/右眼面积
        - palpebral_height_ratio: 左睑裂高/右睑裂高 (如果有)
        """
        # 优先使用面积比，也可以使用高度比
        area_ratio = indicators.get('eye_area_ratio', 1.0)
        height_ratio = indicators.get('palpebral_height_ratio', None)

        # 使用的主要指标
        ratio = area_ratio
        ratio_name = 'eye_area_ratio'

        # 如果有高度比，取平均
        if height_ratio is not None:
            left_h = indicators.get('left_palpebral_height', 0)
            right_h = indicators.get('right_palpebral_height', 0)
            if right_h > 0:
                height_ratio = left_h / right_h
                ratio = (area_ratio + height_ratio) / 2

        evidence = {
            'eye_area_ratio': area_ratio,
            'computed_ratio': ratio,
            'threshold_normal_low': self.EYE_NORMAL_LOW,
            'threshold_normal_high': self.EYE_NORMAL_HIGH,
        }

        if self.EYE_NORMAL_LOW <= ratio <= self.EYE_NORMAL_HIGH:
            status = RestingEyeStatus.NORMAL
            score = 0
            evidence['interpretation'] = '眼睛对称，静态正常'
        elif ratio < self.EYE_NORMAL_LOW:
            status = RestingEyeStatus.NARROW
            score = 1
            evidence['interpretation'] = f'左眼相对较小(比例{ratio:.3f}<{self.EYE_NORMAL_LOW})，可能患侧眼裂缩窄'
        else:  # ratio > self.EYE_NORMAL_HIGH
            status = RestingEyeStatus.WIDE
            score = 1
            evidence['interpretation'] = f'左眼相对较大(比例{ratio:.3f}>{self.EYE_NORMAL_HIGH})，可能患侧眼裂增宽'

        return status, score, evidence

    def _score_resting_cheek(
            self,
            indicators: Dict[str, float]
    ) -> Tuple[RestingCheekStatus, int, Dict[str, Any]]:
        """
        评估静息鼻唇沟状态

        指标:
        - nlf_length_ratio: 左鼻唇沟长度/右鼻唇沟长度
        - nlf_depth_ratio: 鼻唇沟深度比 (如果有)
        """
        length_ratio = indicators.get('nlf_length_ratio', 1.0)
        depth_ratio = indicators.get('nlf_depth_ratio', None)
        depth_proxy_ratio = indicators.get('nlf_depth_proxy_ratio', None)

        # 综合考虑长度和深度
        ratio = length_ratio
        if depth_proxy_ratio is not None:
            # 结合深度代理
            ratio = (length_ratio + depth_proxy_ratio) / 2

        evidence = {
            'nlf_length_ratio': length_ratio,
            'nlf_depth_proxy_ratio': depth_proxy_ratio,
            'computed_ratio': ratio,
            'threshold_normal_low': self.CHEEK_NORMAL_LOW,
            'threshold_normal_high': self.CHEEK_NORMAL_HIGH,
        }

        # 判断状态
        if self.CHEEK_NORMAL_LOW <= ratio <= self.CHEEK_NORMAL_HIGH:
            status = RestingCheekStatus.NORMAL
            score = 0
            evidence['interpretation'] = '鼻唇沟左右对称，静态正常'
        elif (self.CHEEK_LESS_LOW <= ratio < self.CHEEK_NORMAL_LOW) or \
                (self.CHEEK_NORMAL_HIGH < ratio <= self.CHEEK_LESS_HIGH):
            # 判断是不明显还是过于明显
            if ratio < 1.0:
                status = RestingCheekStatus.LESS_PRONOUNCED
                evidence['interpretation'] = f'左侧鼻唇沟相对不明显(比例{ratio:.3f})'
            else:
                status = RestingCheekStatus.MORE_PRONOUNCED
                evidence['interpretation'] = f'左侧鼻唇沟相对过于明显(比例{ratio:.3f})'
            score = 1
        elif ratio < self.CHEEK_LESS_LOW or ratio > self.CHEEK_LESS_HIGH:
            status = RestingCheekStatus.ABSENT
            score = 2
            if ratio < self.CHEEK_LESS_LOW:
                evidence['interpretation'] = f'左侧鼻唇沟消失/极不对称(比例{ratio:.3f}<{self.CHEEK_LESS_LOW})'
            else:
                evidence['interpretation'] = f'右侧鼻唇沟消失/极不对称(比例{ratio:.3f}>{self.CHEEK_LESS_HIGH})'
        else:
            status = RestingCheekStatus.NORMAL
            score = 0
            evidence['interpretation'] = '鼻唇沟状态正常'

        return status, score, evidence

    def _score_resting_mouth(
            self,
            indicators: Dict[str, float]
    ) -> Tuple[RestingMouthStatus, int, Dict[str, Any]]:
        """
        评估静息口部状态

        指标:
        - oral_height_diff: 口角高度差 (left_y - right_y，归一化)
                           正值 = 左口角更高; 负值 = 右口角更高
        """
        height_diff = indicators.get('oral_height_diff', 0.0)
        angle_diff = indicators.get('oral_angle_diff', None)

        evidence = {
            'oral_height_diff': height_diff,
            'oral_angle_diff': angle_diff,
            'threshold': self.MOUTH_NORMAL_THRESHOLD,
        }

        # 判断状态
        if abs(height_diff) <= self.MOUTH_NORMAL_THRESHOLD:
            status = RestingMouthStatus.NORMAL
            score = 0
            evidence['interpretation'] = '口角位置对称，静态正常'
        elif height_diff < -self.MOUTH_NORMAL_THRESHOLD:
            # 负值 = 右口角更高 = 左口角下垂
            status = RestingMouthStatus.CORNER_DROPPED
            score = 1
            evidence['interpretation'] = f'左口角下垂(高度差{height_diff:.4f})'
            evidence['affected_side'] = 'left'
        else:  # height_diff > self.MOUTH_NORMAL_THRESHOLD
            # 正值 = 左口角更高，可能是右侧口角下垂或左侧上提
            status = RestingMouthStatus.CORNER_DROPPED  # 通常是对侧下垂
            score = 1
            evidence['interpretation'] = f'右口角下垂(高度差{height_diff:.4f})'
            evidence['affected_side'] = 'right'

        return status, score, evidence

    # ============ 自主运动评分 ============

    def compute_voluntary_movement(
            self,
            action_results: Dict[str, Dict[str, float]],
            neutral_indicators: Dict[str, float] = None,
            affected_side: str = None
    ) -> VoluntaryMovementResult:
        """
        计算自主运动评分

        Args:
            action_results: 各动作的指标字典
                - 'RaiseEyebrow': 眉部动作指标
                - 'CloseEyeSoftly'/'CloseEyeHardly': 闭眼动作指标
                - 'Smile'/'ShowTeeth': 微笑动作指标
                - 'ShrugNose': 皱鼻动作指标
                - 'LipPucker': 撅嘴动作指标
            neutral_indicators: NeutralFace基准指标
            affected_side: 患侧 ('left'/'right')

        Returns:
            VoluntaryMovementResult
        """
        # 1. Brow (FRD) - RaiseEyebrow
        brow_indicators = action_results.get('RaiseEyebrow', {})
        brow_level, brow_score, brow_evidence = self._score_voluntary_brow(
            brow_indicators, neutral_indicators, affected_side
        )

        # 2. Eye closure (OCS) - CloseEyeSoftly或CloseEyeHardly
        eye_indicators = action_results.get('CloseEyeSoftly',
                                            action_results.get('CloseEyeHardly', {}))
        eye_level, eye_score, eye_evidence = self._score_voluntary_eye_closure(
            eye_indicators, neutral_indicators, affected_side
        )

        # 3. Open mouth smile (ZYG/RIS) - Smile或ShowTeeth
        smile_indicators = action_results.get('Smile',
                                              action_results.get('ShowTeeth', {}))
        smile_level, smile_score, smile_evidence = self._score_voluntary_smile(
            smile_indicators, neutral_indicators, affected_side
        )

        # 4. Snarl (LLA/LLS) - ShrugNose
        snarl_indicators = action_results.get('ShrugNose', {})
        snarl_level, snarl_score, snarl_evidence = self._score_voluntary_snarl(
            snarl_indicators, neutral_indicators, affected_side
        )

        # 5. Lip pucker (OOS/OOI) - LipPucker
        lip_indicators = action_results.get('LipPucker', {})
        lip_level, lip_score, lip_evidence = self._score_voluntary_lip_pucker(
            lip_indicators, neutral_indicators, affected_side
        )

        return VoluntaryMovementResult(
            brow_level=brow_level,
            brow_score=brow_score,
            brow_evidence=brow_evidence,
            eye_closure_level=eye_level,
            eye_closure_score=eye_score,
            eye_closure_evidence=eye_evidence,
            smile_level=smile_level,
            smile_score=smile_score,
            smile_evidence=smile_evidence,
            snarl_level=snarl_level,
            snarl_score=snarl_score,
            snarl_evidence=snarl_evidence,
            lip_pucker_level=lip_level,
            lip_pucker_score=lip_score,
            lip_pucker_evidence=lip_evidence
        )

    def _function_pct_to_level(self, function_pct: float) -> Tuple[VoluntaryMovementLevel, int]:
        """将功能百分比映射到自主运动等级"""
        if function_pct < self.VOLUNTARY_UNABLE_THRESHOLD:
            return VoluntaryMovementLevel.UNABLE, 1
        elif function_pct < self.VOLUNTARY_SLIGHT_THRESHOLD:
            return VoluntaryMovementLevel.INITIATES_SLIGHT, 2
        elif function_pct < self.VOLUNTARY_PARTIAL_THRESHOLD:
            return VoluntaryMovementLevel.INITIATES_WITH_MOVEMENT, 3
        elif function_pct < self.VOLUNTARY_ALMOST_THRESHOLD:
            return VoluntaryMovementLevel.ALMOST_COMPLETE, 4
        else:
            return VoluntaryMovementLevel.COMPLETE, 5

    def _compute_function_pct_from_ratio(
            self,
            ratio: float,
            affected_side: str = None
    ) -> float:
        """
        从左右比值计算功能百分比

        假设比值=1为完全对称，偏离1的程度代表功能损失
        """
        if ratio is None:
            return 100.0

        # 计算不对称度
        asymmetry = abs(1.0 - ratio)

        # 功能百分比 = 1 - 不对称度
        function_pct = max(0, (1.0 - asymmetry)) * 100

        return function_pct

    def _score_voluntary_brow(
            self,
            indicators: Dict[str, float],
            neutral_indicators: Dict[str, float] = None,
            affected_side: str = None
    ) -> Tuple[VoluntaryMovementLevel, int, Dict[str, Any]]:
        """评估眉部(抬眉)自主运动"""

        evidence = {'action': 'RaiseEyebrow', 'branch': 'Temporal'}

        if not indicators:
            # 无数据,假设正常
            return VoluntaryMovementLevel.COMPLETE, 5, {
                **evidence,
                'interpretation': '无数据，默认正常',
                'function_pct': 100.0
            }

        # 获取抬眉比值
        lift_ratio = indicators.get('lift_ratio', 1.0)
        lift_asymmetry = indicators.get('lift_asymmetry', 0.0)
        function_pct = indicators.get('function_pct', None)

        # 如果动作已计算function_pct,直接使用
        if function_pct is None:
            function_pct = self._compute_function_pct_from_ratio(lift_ratio)

        level, score = self._function_pct_to_level(function_pct)

        evidence.update({
            'lift_ratio': lift_ratio,
            'lift_asymmetry': lift_asymmetry,
            'function_pct': function_pct,
            'level': level.name,
            'score': score,
            'interpretation': f'抬眉功能{function_pct:.1f}%，评分{score}/5'
        })

        return level, score, evidence

    def _score_voluntary_eye_closure(
            self,
            indicators: Dict[str, float],
            neutral_indicators: Dict[str, float] = None,
            affected_side: str = None
    ) -> Tuple[VoluntaryMovementLevel, int, Dict[str, Any]]:
        """评估闭眼自主运动"""

        evidence = {'action': 'CloseEyeSoftly/CloseEyeHardly', 'branch': 'Zygomatic'}

        if not indicators:
            return VoluntaryMovementLevel.COMPLETE, 5, {
                **evidence,
                'interpretation': '无数据，默认正常',
                'function_pct': 100.0
            }

        # 获取闭眼指标
        closure_ratio = indicators.get('closure_ratio', 1.0)
        left_complete = indicators.get('left_complete_closure', True)
        right_complete = indicators.get('right_complete_closure', True)
        both_complete = indicators.get('both_complete_closure', True)
        function_pct = indicators.get('function_pct', None)

        # 计算功能百分比
        if function_pct is None:
            # 基于闭合完整性
            if both_complete:
                function_pct = 100.0
            elif left_complete or right_complete:
                # 一侧能闭合
                function_pct = 75.0 * closure_ratio if closure_ratio <= 1 else 75.0 / closure_ratio
            else:
                # 两侧都不能完全闭合
                function_pct = 50.0 * min(closure_ratio, 1.0 / closure_ratio)

        level, score = self._function_pct_to_level(function_pct)

        evidence.update({
            'closure_ratio': closure_ratio,
            'left_complete_closure': left_complete,
            'right_complete_closure': right_complete,
            'both_complete_closure': both_complete,
            'function_pct': function_pct,
            'level': level.name,
            'score': score,
            'interpretation': f'闭眼功能{function_pct:.1f}%，评分{score}/5'
        })

        return level, score, evidence

    def _score_voluntary_smile(
            self,
            indicators: Dict[str, float],
            neutral_indicators: Dict[str, float] = None,
            affected_side: str = None
    ) -> Tuple[VoluntaryMovementLevel, int, Dict[str, Any]]:
        """评估微笑自主运动"""

        evidence = {'action': 'Smile/ShowTeeth', 'branch': 'Buccal'}

        if not indicators:
            return VoluntaryMovementLevel.COMPLETE, 5, {
                **evidence,
                'interpretation': '无数据，默认正常',
                'function_pct': 100.0
            }

        # 获取微笑指标
        oral_excursion_ratio = indicators.get('oral_excursion_ratio',
                                              indicators.get('smile_ratio', 1.0))
        smile_asymmetry = indicators.get('smile_asymmetry', 0.0)
        function_pct = indicators.get('function_pct', None)

        if function_pct is None:
            function_pct = self._compute_function_pct_from_ratio(oral_excursion_ratio)

        level, score = self._function_pct_to_level(function_pct)

        evidence.update({
            'oral_excursion_ratio': oral_excursion_ratio,
            'smile_asymmetry': smile_asymmetry,
            'function_pct': function_pct,
            'level': level.name,
            'score': score,
            'interpretation': f'微笑功能{function_pct:.1f}%，评分{score}/5'
        })

        return level, score, evidence

    def _score_voluntary_snarl(
            self,
            indicators: Dict[str, float],
            neutral_indicators: Dict[str, float] = None,
            affected_side: str = None
    ) -> Tuple[VoluntaryMovementLevel, int, Dict[str, Any]]:
        """评估皱鼻自主运动"""

        evidence = {'action': 'ShrugNose', 'branch': 'Buccal'}

        if not indicators:
            return VoluntaryMovementLevel.COMPLETE, 5, {
                **evidence,
                'interpretation': '无数据，默认正常',
                'function_pct': 100.0
            }

        # 获取皱鼻指标
        nostril_ratio = indicators.get('nostril_flare_ratio',
                                       indicators.get('nostril_asymmetry', 1.0))
        function_pct = indicators.get('function_pct', None)

        if function_pct is None:
            if isinstance(nostril_ratio, float) and nostril_ratio != 0:
                function_pct = self._compute_function_pct_from_ratio(nostril_ratio)
            else:
                function_pct = 100.0

        level, score = self._function_pct_to_level(function_pct)

        evidence.update({
            'nostril_ratio': nostril_ratio,
            'function_pct': function_pct,
            'level': level.name,
            'score': score,
            'interpretation': f'皱鼻功能{function_pct:.1f}%，评分{score}/5'
        })

        return level, score, evidence

    def _score_voluntary_lip_pucker(
            self,
            indicators: Dict[str, float],
            neutral_indicators: Dict[str, float] = None,
            affected_side: str = None
    ) -> Tuple[VoluntaryMovementLevel, int, Dict[str, Any]]:
        """评估撅嘴自主运动"""

        evidence = {'action': 'LipPucker', 'branch': 'Marginal Mandibular'}

        if not indicators:
            return VoluntaryMovementLevel.COMPLETE, 5, {
                **evidence,
                'interpretation': '无数据，默认正常',
                'function_pct': 100.0
            }

        # 获取撅嘴指标
        pucker_symmetry = indicators.get('pucker_symmetry', 1.0)
        mouth_aspect_ratio = indicators.get('mouth_aspect_ratio', None)
        function_pct = indicators.get('function_pct', None)

        if function_pct is None:
            function_pct = self._compute_function_pct_from_ratio(pucker_symmetry)

        level, score = self._function_pct_to_level(function_pct)

        evidence.update({
            'pucker_symmetry': pucker_symmetry,
            'mouth_aspect_ratio': mouth_aspect_ratio,
            'function_pct': function_pct,
            'level': level.name,
            'score': score,
            'interpretation': f'撅嘴功能{function_pct:.1f}%，评分{score}/5'
        })

        return level, score, evidence

    # ============ 联带运动评分 ============

    def compute_synkinesis(
            self,
            action_results: Dict[str, Dict[str, float]]
    ) -> SynkinesisResult:
        """
        计算联带运动评分

        联带运动指做一个动作时，其他面部区域出现不自主的运动

        Args:
            action_results: 各动作的指标字典

        Returns:
            SynkinesisResult
        """
        # 各动作的联带运动评分
        brow_syn, brow_score, brow_ev = self._score_synkinesis_brow(
            action_results.get('RaiseEyebrow', {})
        )

        eye_syn, eye_score, eye_ev = self._score_synkinesis_eye(
            action_results.get('CloseEyeSoftly',
                               action_results.get('CloseEyeHardly', {}))
        )

        smile_syn, smile_score, smile_ev = self._score_synkinesis_smile(
            action_results.get('Smile',
                               action_results.get('ShowTeeth', {}))
        )

        snarl_syn, snarl_score, snarl_ev = self._score_synkinesis_snarl(
            action_results.get('ShrugNose', {})
        )

        lip_syn, lip_score, lip_ev = self._score_synkinesis_lip(
            action_results.get('LipPucker', {})
        )

        return SynkinesisResult(
            brow_synkinesis=brow_syn,
            brow_synkinesis_score=brow_score,
            brow_synkinesis_evidence=brow_ev,
            eye_closure_synkinesis=eye_syn,
            eye_closure_synkinesis_score=eye_score,
            eye_closure_synkinesis_evidence=eye_ev,
            smile_synkinesis=smile_syn,
            smile_synkinesis_score=smile_score,
            smile_synkinesis_evidence=smile_ev,
            snarl_synkinesis=snarl_syn,
            snarl_synkinesis_score=snarl_score,
            snarl_synkinesis_evidence=snarl_ev,
            lip_pucker_synkinesis=lip_syn,
            lip_pucker_synkinesis_score=lip_score,
            lip_pucker_synkinesis_evidence=lip_ev
        )

    def _synkinesis_index_to_level(
            self,
            index: float
    ) -> Tuple[SynkinesisLevel, int]:
        """将联带运动指数映射到等级"""
        if index < self.SYNKINESIS_NONE_THRESHOLD:
            return SynkinesisLevel.NONE, 0
        elif index < self.SYNKINESIS_MILD_THRESHOLD:
            return SynkinesisLevel.MILD, 1
        elif index < self.SYNKINESIS_MODERATE_THRESHOLD:
            return SynkinesisLevel.MODERATE_OBVIOUS, 2
        else:
            return SynkinesisLevel.SEVERE_DISFIGURING, 3

    def _score_synkinesis_brow(
            self,
            indicators: Dict[str, float]
    ) -> Tuple[SynkinesisLevel, int, Dict[str, Any]]:
        """评估抬眉时的联带运动(眼/嘴是否不自主运动)"""

        # 检测抬眉时眼睛或嘴角是否有异常运动
        eye_synkinesis = indicators.get('eye_synkinesis_index', 0.0)
        mouth_synkinesis = indicators.get('mouth_synkinesis_index', 0.0)

        # 取最大值
        synkinesis_index = max(eye_synkinesis, mouth_synkinesis)

        level, score = self._synkinesis_index_to_level(synkinesis_index)

        evidence = {
            'action': 'RaiseEyebrow',
            'eye_synkinesis_index': eye_synkinesis,
            'mouth_synkinesis_index': mouth_synkinesis,
            'max_synkinesis_index': synkinesis_index,
            'level': level.name,
            'score': score
        }

        return level, score, evidence

    def _score_synkinesis_eye(
            self,
            indicators: Dict[str, float]
    ) -> Tuple[SynkinesisLevel, int, Dict[str, Any]]:
        """评估闭眼时的联带运动(嘴角是否被牵拉)"""

        mouth_synkinesis = indicators.get('mouth_synkinesis_index',
                                          indicators.get('left_eye_synkinesis', 0.0))

        synkinesis_index = mouth_synkinesis
        level, score = self._synkinesis_index_to_level(synkinesis_index)

        evidence = {
            'action': 'CloseEyeSoftly/CloseEyeHardly',
            'mouth_synkinesis_index': mouth_synkinesis,
            'synkinesis_index': synkinesis_index,
            'level': level.name,
            'score': score
        }

        return level, score, evidence

    def _score_synkinesis_smile(
            self,
            indicators: Dict[str, float]
    ) -> Tuple[SynkinesisLevel, int, Dict[str, Any]]:
        """评估微笑时的联带运动(眼睛是否眯起)"""

        eye_synkinesis = indicators.get('eye_synkinesis_index',
                                        indicators.get('left_eye_synkinesis', 0.0))

        synkinesis_index = eye_synkinesis
        level, score = self._synkinesis_index_to_level(synkinesis_index)

        evidence = {
            'action': 'Smile/ShowTeeth',
            'eye_synkinesis_index': eye_synkinesis,
            'synkinesis_index': synkinesis_index,
            'level': level.name,
            'score': score
        }

        return level, score, evidence

    def _score_synkinesis_snarl(
            self,
            indicators: Dict[str, float]
    ) -> Tuple[SynkinesisLevel, int, Dict[str, Any]]:
        """评估皱鼻时的联带运动"""

        synkinesis_index = indicators.get('synkinesis_index', 0.0)
        level, score = self._synkinesis_index_to_level(synkinesis_index)

        evidence = {
            'action': 'ShrugNose',
            'synkinesis_index': synkinesis_index,
            'level': level.name,
            'score': score
        }

        return level, score, evidence

    def _score_synkinesis_lip(
            self,
            indicators: Dict[str, float]
    ) -> Tuple[SynkinesisLevel, int, Dict[str, Any]]:
        """评估撅嘴时的联带运动"""

        synkinesis_index = indicators.get('synkinesis_index', 0.0)
        level, score = self._synkinesis_index_to_level(synkinesis_index)

        evidence = {
            'action': 'LipPucker',
            'synkinesis_index': synkinesis_index,
            'level': level.name,
            'score': score
        }

        return level, score, evidence

    # ============ 综合评分 ============

    def compute_score(
            self,
            neutral_indicators: Dict[str, float],
            action_results: Dict[str, Dict[str, float]],
            affected_side: str = None
    ) -> SunnybrookResult:
        """
        计算Sunnybrook综合评分

        Args:
            neutral_indicators: NeutralFace指标
            action_results: 各动作指标字典
            affected_side: 患侧(可选，如果None则自动判断)

        Returns:
            SunnybrookResult
        """
        # 1. 静态对称性评分
        resting = self.compute_resting_symmetry(neutral_indicators)

        # 2. 判断患侧(如果未指定)
        if affected_side is None:
            affected_side = self._determine_affected_side(neutral_indicators, resting)

        # 3. 自主运动评分
        voluntary = self.compute_voluntary_movement(
            action_results, neutral_indicators, affected_side
        )

        # 4. 联带运动评分
        synkinesis = self.compute_synkinesis(action_results)

        # 5. 计算置信度
        confidence = self._compute_confidence(resting, voluntary, synkinesis)

        return SunnybrookResult(
            resting_symmetry=resting,
            voluntary_movement=voluntary,
            synkinesis=synkinesis,
            affected_side=affected_side,
            confidence=confidence
        )

    def _determine_affected_side(
            self,
            indicators: Dict[str, float],
            resting: RestingSymmetryResult
    ) -> str:
        """根据静态对称性判断患侧"""
        votes = {"left": 0, "right": 0}

        # 眼部
        eye_ratio = indicators.get('eye_area_ratio', 1.0)
        if eye_ratio < 0.9:
            votes["left"] += 1
        elif eye_ratio > 1.1:
            votes["right"] += 1

        # 鼻唇沟
        nlf_ratio = indicators.get('nlf_length_ratio', 1.0)
        if nlf_ratio < 0.9:
            votes["left"] += 1
        elif nlf_ratio > 1.1:
            votes["right"] += 1

        # 口角
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
            resting: RestingSymmetryResult,
            voluntary: VoluntaryMovementResult,
            synkinesis: SynkinesisResult
    ) -> float:
        """计算评分置信度"""
        # 基于各项评分的一致性
        # 简单实现：如果静态和动态评分都指向同一结论，置信度高

        confidence = 0.8  # 基础置信度

        # 如果静态有明显异常
        if resting.total_raw > 0:
            confidence += 0.1

        # 如果自主运动有明显问题
        if voluntary.total_raw < 25:
            confidence += 0.1

        return min(1.0, confidence)


# ============ 便捷函数 ============

def compute_sunnybrook_score(
        neutral_indicators: Dict[str, float],
        action_results: Dict[str, Dict[str, float]],
        affected_side: str = None
) -> SunnybrookResult:
    """
    便捷函数：计算Sunnybrook评分
    """
    scorer = SunnybrookScorer()
    return scorer.compute_score(neutral_indicators, action_results, affected_side)


if __name__ == "__main__":
    # 测试代码
    print("Sunnybrook Scorer - 单元测试")

    # 模拟数据
    neutral = {
        'eye_area_ratio': 0.85,  # 左眼稍小
        'nlf_length_ratio': 0.88,  # 左鼻唇沟稍短
        'oral_height_diff': -0.02,  # 左口角稍低
    }

    actions = {
        'RaiseEyebrow': {'lift_ratio': 0.7, 'function_pct': 70},
        'CloseEyeSoftly': {'closure_ratio': 0.8, 'both_complete_closure': True},
        'Smile': {'oral_excursion_ratio': 0.75},
        'ShrugNose': {'nostril_flare_ratio': 0.8},
        'LipPucker': {'pucker_symmetry': 0.85},
    }

    result = compute_sunnybrook_score(neutral, actions)

    print(f"\n静态对称性评分: {result.resting_symmetry.total_weighted}/20")
    print(f"  Eye: {result.resting_symmetry.eye_score}")
    print(f"  Cheek: {result.resting_symmetry.cheek_score}")
    print(f"  Mouth: {result.resting_symmetry.mouth_score}")

    print(f"\n自主运动评分: {result.voluntary_movement.total_weighted}/100")
    print(f"  Brow: {result.voluntary_movement.brow_score}/5")
    print(f"  Eye: {result.voluntary_movement.eye_closure_score}/5")
    print(f"  Smile: {result.voluntary_movement.smile_score}/5")
    print(f"  Snarl: {result.voluntary_movement.snarl_score}/5")
    print(f"  Lip: {result.voluntary_movement.lip_pucker_score}/5")

    print(f"\n联带运动评分: {result.synkinesis.total_score}/15")

    print(f"\n综合评分: {result.composite_score}/100")
    print(f"患侧: {result.affected_side}")
    print(f"置信度: {result.confidence:.2f}")