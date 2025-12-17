"""
特征整合器 - 利用现有 actions/*.py 的指标
将动作特定的归一化指标转换为固定顺序的特征向量。

更新说明 (2025-12):
- 对齐最新 11 个动作代码：NeutralFace、SpontaneousEyeBlink、VoluntaryEyeBlink、
  CloseEyeSoftly、CloseEyeHardly、RaiseEyebrow、Smile、ShrugNose、ShowTeeth、
  BlowCheek、LipPucker
- 关键指标字段名按最新 action 文件输出更新（例如 left_eye_openness / icd 等）
- 增加 schema 自动扩展：如果动作输出了未在列表中的新字段，会按稳定顺序追加，
  以避免“字段名变更导致全 0”的问题
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Any, Optional

# 导入所有动作类（actions 目录应为 Python package）
from facialPalsy.actions.neutral_face import NeutralFaceAction
from facialPalsy.actions.spontaneous_eye_blink import SpontaneousEyeBlinkAction
from facialPalsy.actions.voluntary_eye_blink import VoluntaryEyeBlinkAction
from facialPalsy.actions.close_eye_softly import CloseEyeSoftlyAction
from facialPalsy.actions.close_eye_hardly import CloseEyeHardlyAction
from facialPalsy.actions.raise_eyebrow import RaiseEyebrowAction
from facialPalsy.actions.smile import SmileAction
from facialPalsy.actions.shrug_nose import ShrugNoseAction
from facialPalsy.actions.show_teeth import ShowTeethAction
from facialPalsy.actions.blow_cheek import BlowCheekAction
from facialPalsy.actions.lip_pucker import LipPuckerAction


class ActionFeatureIntegrator:
    """
    整合各个动作的特定特征

    核心思想:
    1) 每个动作保留自己的特定特征（动作关注点不同）
    2) 将各动作的归一化指标 + 动态特征转换为固定维度的特征向量
    3) 保持动作特异性，同时便于模型训练
    """

    def __init__(self, auto_extend_schema: bool = True):
        """初始化 11 个动作检测器"""
        self.action_detectors = {
            'NeutralFace': NeutralFaceAction(),
            'SpontaneousEyeBlink': SpontaneousEyeBlinkAction(),
            'VoluntaryEyeBlink': VoluntaryEyeBlinkAction(),
            'CloseEyeSoftly': CloseEyeSoftlyAction(),
            'CloseEyeHardly': CloseEyeHardlyAction(),
            'RaiseEyebrow': RaiseEyebrowAction(),
            'Smile': SmileAction(),
            'ShrugNose': ShrugNoseAction(),
            'ShowTeeth': ShowTeethAction(),
            'BlowCheek': BlowCheekAction(),
            'LipPucker': LipPuckerAction(),
        }

        self.auto_extend_schema = auto_extend_schema

        # 每个动作的关键指标定义（固定顺序）
        self.action_key_indicators: Dict[str, Dict[str, List[str]]] = self._define_key_indicators()

    def _define_key_indicators(self) -> Dict[str, Dict[str, List[str]]]:
        """
        定义每个动作的关键指标（字段名必须与 action 输出一致）。

        注意：
        - 对于动作代码升级导致字段变化的情况，建议只在这里更新即可。
        - 如果 auto_extend_schema=True，会把 action 输出但未在此列出的字段追加到末尾，
          从而避免信息丢失（维度会增加，但顺序稳定）。
        """
        return {
            # ========== Neutral (静息) ==========
            # 对齐 neutral_face.py 的输出字段（包含 icd、*_norm、ratio、diff 等）
            'NeutralFace': {
                'static': [
                    'icd',
                    'left_eye_area_norm', 'right_eye_area_norm', 'eye_area_ratio', 'eye_asymmetry',
                    'left_eye_openness', 'right_eye_openness', 'openness_ratio', 'left_eye_closure', 'right_eye_closure',
                    'left_palpebral_length', 'right_palpebral_length',
                    'left_brow_height_norm', 'right_brow_height_norm', 'brow_height_ratio',
                    'mouth_width_norm', 'mouth_height_norm', 'oral_height_diff', 'oral_angle_diff',
                    'nlf_length_ratio', 'nlf_depth_ratio',
                    'face_symmetry_score',
                ],
                'dynamic': []
            },

            # ========== Eye Blinks ==========
            # 下面三个动作（自然眨眼/自主眨眼/闭眼）通常共享类似的眼部字段。
            # 若眨眼动作代码字段不同，会通过 auto_extend_schema 自动补齐。
            'SpontaneousEyeBlink': {
                'static': [
                    'left_eye_openness', 'right_eye_openness',
                    'left_eye_closure', 'right_eye_closure',
                    'closure_ratio', 'closure_diff',
                    'function_pct', 'icd',
                ],
                'dynamic': [
                    'left_motion_range', 'right_motion_range',
                    'left_mean_velocity', 'right_mean_velocity',
                    'left_max_velocity', 'right_max_velocity',
                    'left_smoothness', 'right_smoothness',
                    'motion_asymmetry',
                ]
            },
            'VoluntaryEyeBlink': {
                'static': [
                    'left_eye_openness', 'right_eye_openness',
                    'left_eye_closure', 'right_eye_closure',
                    'closure_ratio', 'closure_diff',
                    'function_pct', 'icd',
                ],
                'dynamic': [
                    'left_motion_range', 'right_motion_range',
                    'left_mean_velocity', 'right_mean_velocity',
                    'left_max_velocity', 'right_max_velocity',
                    'left_smoothness', 'right_smoothness',
                    'motion_asymmetry',
                ]
            },

            'CloseEyeSoftly': {
                'static': [
                    'left_eye_openness', 'right_eye_openness',
                    'left_eye_closure', 'right_eye_closure',
                    'left_complete_closure', 'right_complete_closure',
                    'left_ear', 'right_ear',
                    'openness_ratio', 'closure_ratio', 'closure_diff',
                    'function_pct', 'icd',
                ],
                'dynamic': [
                    'left_motion_range', 'right_motion_range',
                    'left_mean_velocity', 'right_mean_velocity',
                    'left_max_velocity', 'right_max_velocity',
                    'left_smoothness', 'right_smoothness',
                    'motion_asymmetry',
                ]
            },
            'CloseEyeHardly': {
                'static': [
                    'left_eye_openness', 'right_eye_openness',
                    'left_eye_closure', 'right_eye_closure',
                    'left_complete_closure', 'right_complete_closure',
                    'left_ear', 'right_ear',
                    'openness_ratio', 'closure_ratio', 'closure_diff',
                    'function_pct', 'icd',
                ],
                'dynamic': [
                    'left_motion_range', 'right_motion_range',
                    'left_mean_velocity', 'right_mean_velocity',
                    'left_max_velocity', 'right_max_velocity',
                    'left_smoothness', 'right_smoothness',
                    'motion_asymmetry',
                ]
            },

            # ========== Brow ==========
            'RaiseEyebrow': {
                'static': [
                    'left_brow_height', 'right_brow_height',
                    'left_brow_height_norm', 'right_brow_height_norm',
                    'brow_height_ratio',
                    'left_brow_lift', 'right_brow_lift',
                    'lift_ratio', 'lift_asymmetry',
                    'function_pct',
                    'left_eye_openness', 'right_eye_openness',
                    'eye_synkinesis',
                    'icd',
                ],
                'dynamic': [
                    'left_motion_range', 'right_motion_range',
                    'left_mean_velocity', 'right_mean_velocity',
                    'left_smoothness', 'right_smoothness',
                    'motion_asymmetry',
                ]
            },

            # ========== Mouth / Mid-face ==========
            # 下面三项（Smile / ShowTeeth / ShrugNose）字段可能因代码版本不同变化较大，
            # 因此保留旧字段 + auto_extend_schema 补齐。
            'Smile': {
                'static': [
                    'mouth_width', 'mouth_height', 'mouth_aspect_ratio',
                    'left_mouth_corner_height', 'right_mouth_corner_height',
                    'mouth_corner_height_diff',
                    'left_mouth_eye_dist', 'right_mouth_eye_dist',
                    'mouth_corner_symmetry',
                    'nasolabial_fold_depth_left', 'nasolabial_fold_depth_right'
                ],
                'dynamic': [
                    'mouth_motion_range',
                    'mouth_mean_velocity',
                    'mouth_smoothness'
                ]
            },

            'ShrugNose': {
                'static': [
                    'left_nostril_area', 'right_nostril_area',
                    'nostril_area_ratio',
                    'left_nasolabial_fold_depth', 'right_nasolabial_fold_depth',
                    'nasolabial_fold_ratio'
                ],
                'dynamic': []
            },

            'ShowTeeth': {
                'static': [
                    'mouth_width', 'mouth_height',
                    'upper_lip_exposure', 'lower_lip_exposure',
                    'teeth_visible_area',
                    'mouth_corner_height_diff',
                    'lip_symmetry'
                ],
                'dynamic': []
            },

            'BlowCheek': {
                'static': [
                    'face_width', 'face_width_norm',
                    'left_expansion', 'right_expansion',
                    'left_expansion_norm', 'right_expansion_norm',
                    'expansion_ratio',
                    'mouth_seal',
                    'function_pct',
                    'icd',
                ],
                'dynamic': []
            },

            'LipPucker': {
                'static': [
                    'mouth_width', 'mouth_height',
                    'mouth_width_norm', 'mouth_height_norm',
                    'width_change',
                    'contraction_ratio',
                    'oral_height_diff',
                    'left_oral_angle', 'right_oral_angle',
                    'function_pct',
                    'icd',
                ],
                'dynamic': []
            },
        }

    def _ensure_schema(
        self,
        action_name: str,
        normalized_indicators: Dict[str, float],
        normalized_dynamic_features: Dict[str, float],
    ) -> Dict[str, List[str]]:
        """
        确保 schema 覆盖动作输出字段：
        - 若 schema 中不存在该动作：用当前字段生成并缓存
        - 若 auto_extend_schema=True：把当前输出中未列出的字段追加（按字典序稳定）
        """
        if action_name not in self.action_key_indicators:
            # 兜底：完全基于当前输出生成 schema（顺序稳定）
            static_keys = sorted(list(normalized_indicators.keys()))
            dynamic_keys = sorted(list(normalized_dynamic_features.keys()))
            self.action_key_indicators[action_name] = {'static': static_keys, 'dynamic': dynamic_keys}
            return self.action_key_indicators[action_name]

        if not self.auto_extend_schema:
            return self.action_key_indicators[action_name]

        schema = self.action_key_indicators[action_name]
        static_list = schema['static']
        dynamic_list = schema['dynamic']

        # 追加未覆盖字段（稳定排序）
        new_static = sorted([k for k in normalized_indicators.keys() if k not in static_list])
        new_dynamic = sorted([k for k in normalized_dynamic_features.keys() if k not in dynamic_list])

        if new_static:
            static_list.extend(new_static)
        if new_dynamic:
            dynamic_list.extend(new_dynamic)

        return schema

    def extract_action_features(
        self,
        action_name: str,
        normalized_indicators: Optional[Dict[str, float]],
        normalized_dynamic_features: Optional[Dict[str, float]],
    ) -> np.ndarray:
        """
        从动作的归一化指标中提取关键特征

        Args:
            action_name: 动作名称
            normalized_indicators: 归一化后的静态指标字典
            normalized_dynamic_features: 归一化后的动态特征字典

        Returns:
            numpy array: 该动作的特征向量（固定顺序）
        """
        normalized_indicators = normalized_indicators or {}
        normalized_dynamic_features = normalized_dynamic_features or {}

        schema = self._ensure_schema(action_name, normalized_indicators, normalized_dynamic_features)

        static_features: List[float] = []
        for k in schema['static']:
            static_features.append(float(normalized_indicators.get(k, 0.0)))

        dynamic_features: List[float] = []
        for k in schema['dynamic']:
            dynamic_features.append(float(normalized_dynamic_features.get(k, 0.0)))

        return np.array(static_features + dynamic_features, dtype=np.float32)

    def get_feature_dimension(self, action_name: str) -> int:
        """获取某个动作的特征维度"""
        if action_name not in self.action_key_indicators:
            return 0
        schema = self.action_key_indicators[action_name]
        return len(schema['static']) + len(schema['dynamic'])

    def get_all_feature_dimensions(self) -> Dict[str, int]:
        """获取所有动作的特征维度"""
        return {a: self.get_feature_dimension(a) for a in self.action_key_indicators.keys()}
