"""
特征整合器 - 利用现有actions/*.py的指标
将动作特定的原始指标转换为统一的特征向量
"""
import numpy as np
import sys


# 导入所有动作类
from actions.neutral_face import NeutralFaceAction
from actions.spontaneous_eye_blink import SpontaneousEyeBlinkAction
from actions.voluntary_eye_blink import VoluntaryEyeBlinkAction
from actions.close_eye_softly import CloseEyeSoftlyAction
from actions.close_eye_hardly import CloseEyeHardlyAction
from actions.raise_eyebrow import RaiseEyebrowAction
from actions.smile import SmileAction
from actions.shrug_nose import ShrugNoseAction
from actions.show_teeth import ShowTeethAction
from actions.blow_cheek import BlowCheekAction
from actions.lip_pucker import LipPuckerAction


class ActionFeatureIntegrator:
    """
    整合各个动作的特定特征

    核心思想:
    1. 每个动作保留自己的特定特征(因为每个动作关注点不同)
    2. 将各动作的归一化指标+动态特征转换为固定维度的特征向量
    3. 保持动作特异性,同时便于模型训练
    """

    def __init__(self):
        """初始化11个动作检测器"""
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

        # 每个动作的关键指标定义
        self.action_key_indicators = self._define_key_indicators()

    def _define_key_indicators(self):
        """
        定义每个动作的关键指标

        这些指标是从各动作的extract_indicators()和extract_dynamic_features()中
        精选出来的最有判别力的特征
        """
        return {
            'NeutralFace': {
                'static': [
                    'left_eye_opening', 'right_eye_opening', 'eye_opening_ratio',
                    'mouth_width', 'mouth_height',
                    'left_eyebrow_eye_dist', 'right_eyebrow_eye_dist',
                    'face_symmetry_score'
                ],
                'dynamic': []  # 静息帧通常没有动态特征
            },

            'Smile': {
                'static': [
                    'mouth_width', 'mouth_height', 'mouth_aspect_ratio',
                    'left_mouth_corner_height', 'right_mouth_corner_height',
                    'mouth_corner_height_diff',  # 关键!患侧判断
                    'left_mouth_eye_dist', 'right_mouth_eye_dist',
                    'mouth_corner_symmetry',
                    'nasolabial_fold_depth_left', 'nasolabial_fold_depth_right'
                ],
                'dynamic': [
                    'mouth_motion_range',
                    'mouth_mean_velocity',
                    'mouth_max_velocity',
                    'mouth_smoothness'
                ]
            },

            'CloseEyeHardly': {
                'static': [
                    'left_eye_opening', 'right_eye_opening',
                    'left_eye_closure', 'right_eye_closure',  # 与静息帧对比
                    'left_eye_closure_pct', 'right_eye_closure_pct',
                    'eye_closure_ratio',  # 对称性
                    'left_eye_area', 'right_eye_area',
                    'eye_shape_disparity'
                ],
                'dynamic': [
                    'left_eye_motion_range', 'right_eye_motion_range',
                    'left_eye_mean_velocity', 'right_eye_mean_velocity',
                    'left_eye_max_velocity', 'right_eye_max_velocity',
                    'left_eye_smoothness', 'right_eye_smoothness'
                ]
            },

            'CloseEyeSoftly': {
                'static': [
                    'left_eye_opening', 'right_eye_opening',
                    'left_eye_closure', 'right_eye_closure',
                    'eye_closure_ratio',
                    'left_eye_contraction', 'right_eye_contraction'
                ],
                'dynamic': [
                    'left_eye_motion_range', 'right_eye_motion_range',
                    'left_eye_smoothness', 'right_eye_smoothness'
                ]
            },

            'RaiseEyebrow': {
                'static': [
                    'left_eyebrow_eye_dist', 'right_eyebrow_eye_dist',
                    'left_eyebrow_dist_change', 'right_eyebrow_dist_change',  # 与静息帧对比
                    'left_eyebrow_lift', 'right_eyebrow_lift',
                    'eyebrow_change_ratio', 'eyebrow_lift_ratio',
                    'eyebrow_height_diff',
                    'eyebrow_shape_disparity'
                ],
                'dynamic': [
                    'eyebrow_motion_range',
                    'eyebrow_mean_velocity',
                    'eyebrow_smoothness'
                ]
            },

            'ShowTeeth': {
                'static': [
                    'mouth_width', 'mouth_height',
                    'upper_lip_exposure', 'lower_lip_exposure',
                    'teeth_visible_area',
                    'mouth_corner_height_diff',
                    'lip_symmetry'
                ],
                'dynamic': [
                    'mouth_motion_range',
                    'mouth_mean_velocity',
                    'mouth_smoothness'
                ]
            },

            'ShrugNose': {
                'static': [
                    'nose_wrinkle_depth',
                    'left_nasolabial_depth', 'right_nasolabial_depth',
                    'nose_tip_movement',
                    'upper_lip_lift'
                ],
                'dynamic': [
                    'nose_motion_range',
                    'nose_smoothness'
                ]
            },

            'BlowCheek': {
                'static': [
                    'left_cheek_expansion', 'right_cheek_expansion',
                    'cheek_expansion_ratio',
                    'mouth_seal_tightness',
                    'face_width_increase'
                ],
                'dynamic': [
                    'cheek_motion_range',
                    'mouth_smoothness'
                ]
            },

            'LipPucker': {
                'static': [
                    'mouth_width', 'mouth_height',
                    'lip_protrusion',  # Z轴特征!
                    'mouth_corner_convergence',
                    'upper_lip_lower_lip_distance'
                ],
                'dynamic': [
                    'mouth_motion_range',
                    'mouth_smoothness'
                ]
            },

            'SpontaneousEyeBlink': {
                'static': [
                    'left_eye_opening', 'right_eye_opening',
                    'left_eye_closure', 'right_eye_closure',
                    'eye_closure_ratio'
                ],
                'dynamic': [
                    'left_eye_motion_range', 'right_eye_motion_range',
                    'left_eye_mean_velocity', 'right_eye_mean_velocity',
                    'blink_speed',  # 眨眼特定的速度指标
                    'left_eye_smoothness', 'right_eye_smoothness'
                ]
            },

            'VoluntaryEyeBlink': {
                'static': [
                    'left_eye_opening', 'right_eye_opening',
                    'left_eye_closure', 'right_eye_closure',
                    'eye_closure_ratio'
                ],
                'dynamic': [
                    'left_eye_motion_range', 'right_eye_motion_range',
                    'left_eye_smoothness', 'right_eye_smoothness'
                ]
            }
        }

    def extract_action_features(self, action_name, normalized_indicators,
                                normalized_dynamic_features):
        """
        从动作的归一化指标中提取关键特征

        Args:
            action_name: 动作名称
            normalized_indicators: 归一化后的静态指标字典
            normalized_dynamic_features: 归一化后的动态特征字典

        Returns:
            numpy array: 该动作的特征向量
        """
        if action_name not in self.action_key_indicators:
            raise ValueError(f"Unknown action: {action_name}")

        key_indicators = self.action_key_indicators[action_name]

        # 提取静态特征
        static_features = []
        for indicator_name in key_indicators['static']:
            value = normalized_indicators.get(indicator_name, 0.0)
            static_features.append(value)

        # 提取动态特征
        dynamic_features = []
        for indicator_name in key_indicators['dynamic']:
            value = normalized_dynamic_features.get(indicator_name, 0.0)
            dynamic_features.append(value)

        # 合并
        combined_features = np.array(static_features + dynamic_features, dtype=np.float32)

        return combined_features

    def get_feature_dimension(self, action_name):
        """获取某个动作的特征维度"""
        key_indicators = self.action_key_indicators[action_name]
        static_dim = len(key_indicators['static'])
        dynamic_dim = len(key_indicators['dynamic'])
        return static_dim + dynamic_dim

    def get_all_feature_dimensions(self):
        """获取所有动作的特征维度"""
        dims = {}
        for action_name in self.action_key_indicators.keys():
            dims[action_name] = self.get_feature_dimension(action_name)
        return dims


def print_feature_summary():
    """打印特征维度总结"""
    integrator = ActionFeatureIntegrator()

    print("\n" + "=" * 80)
    print("各动作的特征维度总结")
    print("=" * 80)
    print(f"{'动作名称':<25} {'静态特征':<10} {'动态特征':<10} {'总维度':<10}")
    print("-" * 80)

    total_static = 0
    total_dynamic = 0

    for action_name, key_indicators in integrator.action_key_indicators.items():
        static_dim = len(key_indicators['static'])
        dynamic_dim = len(key_indicators['dynamic'])
        total_dim = static_dim + dynamic_dim

        total_static += static_dim
        total_dynamic += dynamic_dim

        print(f"{action_name:<25} {static_dim:<10} {dynamic_dim:<10} {total_dim:<10}")

    print("-" * 80)
    print(f"{'总计':<25} {total_static:<10} {total_dynamic:<10} {total_static + total_dynamic:<10}")
    print("=" * 80)

    print("\n优势:")
    print("1. ✅ 每个动作的特征都是针对该动作特点定制的")
    print("2. ✅ 保留了原有代码的动作特异性")
    print("3. ✅ 静态特征 + 动态特征组合")
    print("4. ✅ 利用了与静息帧的对比信息")
    print("5. ✅ 特征维度可变,更灵活")


if __name__ == '__main__':
    print_feature_summary()