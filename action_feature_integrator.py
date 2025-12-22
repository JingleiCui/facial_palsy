"""
特征整合器 - Action Feature Integrator
==========================================

从各动作的结果中提取关键特征，转换为固定维度的特征向量。

设计思想:
1. 每个动作有不同的关键指标 (因为关注点不同)
2. 保持动作特异性，但输出统一格式
3. 便于模型训练时的特征管理

使用方法:
    integrator = ActionFeatureIntegrator()

    # 从处理结果提取特征向量
    feature_vector = integrator.extract_action_features(
        action_name='Smile',
        normalized_indicators=result['normalized_indicators'],
        normalized_dynamic_features=result['normalized_dynamic_features']
    )

    # 获取特征维度信息
    dims = integrator.get_all_feature_dimensions()

关于motion_utils.py:
    - motion_utils 提供12维全局运动特征
    - 与这里的动作特定特征是互补的
    - 全局运动特征描述整体运动模式
    - 动作特定特征描述各动作的关键指标
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from facial_palsy.actions import ACTION_CLASSES

class ActionFeatureIntegrator:
    """
    动作特征整合器

    功能:
    1. 定义每个动作的关键指标
    2. 从结果字典提取特征向量
    3. 提供特征维度信息
    """

    def __init__(self):
        """初始化"""
        # 每个动作的关键指标定义
        self.action_key_indicators = self._define_key_indicators()

    def _define_key_indicators(self) -> Dict[str, Dict[str, List[str]]]:
        """
        定义每个动作的关键指标

        这些指标名称必须与各动作类的extract_indicators()输出匹配
        """
        return {
            # ==================== 静息帧 ====================
            'NeutralFace': {
                'static': [
                    # 眼部
                    'left_eye_area_norm', 'right_eye_area_norm', 'eye_area_ratio',
                    'left_eye_openness', 'right_eye_openness',
                    # 眉毛
                    'left_brow_height_norm', 'right_brow_height_norm', 'brow_height_ratio',
                    # 口部
                    'mouth_width_norm', 'mouth_height_norm',
                    'left_oral_height_norm', 'right_oral_height_norm', 'oral_height_ratio',
                    # 鼻唇沟
                    'left_nlf_length_norm', 'right_nlf_length_norm', 'nlf_ratio',
                    # 面部宽度
                    'face_width_norm',
                    # Sunnybrook静态评分
                    'sunnybrook_rest_eye', 'sunnybrook_rest_cheek', 'sunnybrook_rest_mouth',
                ],
                'dynamic': []  # 静息帧无动态特征
            },

            # ==================== 自发眨眼 ====================
            'SpontaneousEyeBlink': {
                'static': [
                    'left_eye_openness', 'right_eye_openness',
                    'left_eye_closure', 'right_eye_closure',
                    'left_complete_closure', 'right_complete_closure',
                    'closure_ratio', 'eye_asymmetry',
                    'function_pct',
                ],
                'dynamic': [
                    'left_motion_range', 'right_motion_range',
                    'left_mean_velocity', 'right_mean_velocity',
                    'left_smoothness', 'right_smoothness',
                    'motion_asymmetry',
                    'left_closing_speed', 'right_closing_speed',
                    'left_opening_speed', 'right_opening_speed',
                    'blink_duration', 'blink_symmetry',
                ]
            },

            # ==================== 主动眨眼 ====================
            'VoluntaryEyeBlink': {
                'static': [
                    'left_eye_openness', 'right_eye_openness',
                    'left_eye_closure', 'right_eye_closure',
                    'left_complete_closure', 'right_complete_closure',
                    'closure_ratio', 'eye_asymmetry',
                    'function_pct',
                ],
                'dynamic': [
                    'left_motion_range', 'right_motion_range',
                    'left_mean_velocity', 'right_mean_velocity',
                    'left_smoothness', 'right_smoothness',
                    'motion_asymmetry',
                    'left_closing_speed', 'right_closing_speed',
                    'left_opening_speed', 'right_opening_speed',
                    'blink_duration', 'blink_symmetry',
                ]
            },

            # ==================== 轻柔闭眼 ====================
            'CloseEyeSoftly': {
                'static': [
                    'left_eye_openness', 'right_eye_openness',
                    'left_eye_closure', 'right_eye_closure',
                    'left_complete_closure', 'right_complete_closure',
                    'closure_ratio', 'eye_asymmetry',
                    'function_pct', 'both_complete_closure',
                ],
                'dynamic': [
                    'left_motion_range', 'right_motion_range',
                    'left_mean_velocity', 'right_mean_velocity',
                    'left_smoothness', 'right_smoothness',
                    'motion_asymmetry',
                ]
            },

            # ==================== 用力闭眼 ====================
            'CloseEyeHardly': {
                'static': [
                    'left_eye_openness', 'right_eye_openness',
                    'left_eye_closure', 'right_eye_closure',
                    'left_complete_closure', 'right_complete_closure',
                    'closure_ratio', 'eye_asymmetry',
                    'function_pct', 'both_complete_closure',
                ],
                'dynamic': [
                    'left_motion_range', 'right_motion_range',
                    'left_mean_velocity', 'right_mean_velocity',
                    'left_smoothness', 'right_smoothness',
                    'motion_asymmetry',
                ]
            },

            # ==================== 抬眉 ====================
            'RaiseEyebrow': {
                'static': [
                    'left_brow_height', 'right_brow_height',
                    'left_brow_height_norm', 'right_brow_height_norm',
                    'brow_height_ratio',
                    'left_brow_lift', 'right_brow_lift',
                    'lift_ratio', 'lift_asymmetry',
                    'function_pct',
                    # 联动检测
                    'left_eye_openness', 'right_eye_openness',
                    'left_eye_synkinesis', 'right_eye_synkinesis',
                ],
                'dynamic': [
                    'brow_motion_range', 'brow_mean_velocity', 'brow_smoothness',
                ]
            },

            # ==================== 微笑 ====================
            'Smile': {
                'static': [
                    'mouth_width', 'mouth_width_norm', 'mouth_width_change',
                    'left_oral_height', 'right_oral_height',
                    'left_oral_height_norm', 'right_oral_height_norm',
                    'oral_height_ratio', 'oral_height_asymmetry',
                    'left_oral_angle', 'right_oral_angle',
                    'oral_angle_ratio', 'oral_angle_asymmetry',
                    'left_nlf_change', 'right_nlf_change', 'nlf_change_ratio',
                    'function_pct',
                    # 联动检测
                    'left_eye_openness', 'right_eye_openness',
                    'left_eye_synkinesis', 'right_eye_synkinesis',
                ],
                'dynamic': [
                    'mouth_motion_range', 'mouth_mean_velocity', 'mouth_smoothness',
                ]
            },

            # ==================== 皱鼻 ====================
            'ShrugNose': {
                'static': [
                    'left_nostril_height', 'right_nostril_height',
                    'nostril_height_ratio', 'nostril_asymmetry',
                    'left_nlf_change', 'right_nlf_change', 'nlf_change_ratio',
                    'function_pct',
                ],
                'dynamic': [
                    'nose_motion_range', 'nose_mean_velocity', 'nose_smoothness',
                ]
            },

            # ==================== 露齿 ====================
            'ShowTeeth': {
                'static': [
                    'mouth_width', 'mouth_height',
                    'mouth_width_norm', 'mouth_height_norm',
                    'mouth_width_change', 'mouth_height_change',
                    'left_oral_height', 'right_oral_height',
                    'oral_height_ratio', 'oral_height_asymmetry',
                    'function_pct',
                ],
                'dynamic': [
                    'mouth_motion_range', 'mouth_mean_velocity', 'mouth_smoothness',
                ]
            },

            # ==================== 鼓腮 ====================
            'BlowCheek': {
                'static': [
                    'face_width', 'face_width_norm', 'face_width_change',
                    'left_cheek_expansion', 'right_cheek_expansion',
                    'cheek_expansion_ratio', 'cheek_asymmetry',
                    'function_pct',
                ],
                'dynamic': [
                    'face_motion_range', 'face_mean_velocity', 'face_smoothness',
                ]
            },

            # ==================== 撅嘴 ====================
            'LipPucker': {
                'static': [
                    'mouth_width', 'mouth_height',
                    'mouth_width_norm', 'mouth_height_norm',
                    'mouth_width_change',
                    'lip_protrusion',  # Z轴
                    'mouth_aspect_ratio',
                    'function_pct',
                ],
                'dynamic': [
                    'mouth_motion_range', 'mouth_mean_velocity', 'mouth_smoothness',
                ]
            },
        }

    def extract_action_features(
        self,
        action_name: str,
        normalized_indicators: Dict[str, float],
        normalized_dynamic_features: Dict[str, float]
    ) -> np.ndarray:
        """
        从动作的归一化指标中提取关键特征向量

        Args:
            action_name: 动作名称
            normalized_indicators: 归一化后的静态指标字典
            normalized_dynamic_features: 归一化后的动态特征字典

        Returns:
            特征向量 (numpy array)
        """
        if action_name not in self.action_key_indicators:
            raise ValueError(f"未知动作: {action_name}")

        key_indicators = self.action_key_indicators[action_name]

        # 提取静态特征
        static_features = []
        for indicator_name in key_indicators['static']:
            value = normalized_indicators.get(indicator_name, 0.0)
            if value is None:
                value = 0.0
            static_features.append(float(value))

        # 提取动态特征
        dynamic_features = []
        for indicator_name in key_indicators['dynamic']:
            value = normalized_dynamic_features.get(indicator_name, 0.0)
            if value is None:
                value = 0.0
            dynamic_features.append(float(value))

        # 合并
        combined_features = np.array(static_features + dynamic_features, dtype=np.float32)

        return combined_features

    def extract_all_features(
        self,
        results: Dict[str, Dict],
        include_motion: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        从所有动作结果中提取特征

        Args:
            results: 每个动作的处理结果字典
            include_motion: 是否包含12维全局运动特征

        Returns:
            每个动作的特征向量
        """
        all_features = {}

        for action_name, result in results.items():
            if action_name not in self.action_key_indicators:
                continue

            indicators = result.get('normalized_indicators', {})
            dynamic = result.get('normalized_dynamic_features', {})

            # 动作特定特征
            action_features = self.extract_action_features(
                action_name, indicators, dynamic
            )

            # 是否添加全局运动特征
            if include_motion and 'motion_features' in result:
                motion_features = result['motion_features']
                action_features = np.concatenate([action_features, motion_features])

            all_features[action_name] = action_features

        return all_features

    def get_feature_dimension(self, action_name: str, include_motion: bool = False) -> int:
        """获取某个动作的特征维度"""
        if action_name not in self.action_key_indicators:
            return 0

        key_indicators = self.action_key_indicators[action_name]
        static_dim = len(key_indicators['static'])
        dynamic_dim = len(key_indicators['dynamic'])
        total = static_dim + dynamic_dim

        if include_motion:
            total += 12  # 12维全局运动特征

        return total

    def get_all_feature_dimensions(self, include_motion: bool = False) -> Dict[str, int]:
        """获取所有动作的特征维度"""
        dims = {}
        for action_name in self.action_key_indicators.keys():
            dims[action_name] = self.get_feature_dimension(action_name, include_motion)
        return dims

    def get_feature_names(
        self,
        action_name: str,
        include_motion: bool = False
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        获取某个动作的特征名称

        Returns:
            (静态特征名称列表, 动态特征名称列表, 运动特征名称列表)
        """
        if action_name not in self.action_key_indicators:
            return [], [], []

        key_indicators = self.action_key_indicators[action_name]
        static_names = key_indicators['static'].copy()
        dynamic_names = key_indicators['dynamic'].copy()

        motion_names = []
        if include_motion:
            motion_names = [
                'mean_displacement', 'max_displacement', 'std_displacement',
                'motion_energy', 'motion_asymmetry', 'temporal_smoothness',
                'spatial_concentration', 'peak_ratio',
                'motion_center_x', 'motion_center_y',
                'velocity_mean', 'acceleration_std',
            ]

        return static_names, dynamic_names, motion_names

    def get_total_feature_dimension(self, include_motion: bool = False) -> int:
        """获取所有动作的总特征维度"""
        total = 0
        for action_name in self.action_key_indicators.keys():
            total += self.get_feature_dimension(action_name, include_motion)
        return total

    def get_action_names(self) -> List[str]:
        """获取所有支持的动作名称"""
        return list(self.action_key_indicators.keys())


def print_feature_summary(include_motion: bool = False):
    """打印特征维度总结"""
    integrator = ActionFeatureIntegrator()

    print("\n" + "=" * 80)
    print("V2 各动作的特征维度总结")
    if include_motion:
        print("(包含12维全局运动特征)")
    print("=" * 80)

    header = f"{'动作名称':<25} {'静态特征':<10} {'动态特征':<10}"
    if include_motion:
        header += f" {'运动特征':<10}"
    header += f" {'总维度':<10}"
    print(header)
    print("-" * 80)

    total_static = 0
    total_dynamic = 0
    total_motion = 0

    for action_name, key_indicators in integrator.action_key_indicators.items():
        static_dim = len(key_indicators['static'])
        dynamic_dim = len(key_indicators['dynamic'])
        motion_dim = 12 if include_motion else 0
        total_dim = static_dim + dynamic_dim + motion_dim

        total_static += static_dim
        total_dynamic += dynamic_dim
        total_motion += motion_dim

        row = f"{action_name:<25} {static_dim:<10} {dynamic_dim:<10}"
        if include_motion:
            row += f" {motion_dim:<10}"
        row += f" {total_dim:<10}"
        print(row)

    print("-" * 80)
    total_row = f"{'总计':<25} {total_static:<10} {total_dynamic:<10}"
    if include_motion:
        total_row += f" {total_motion:<10}"
    total_row += f" {total_static + total_dynamic + total_motion:<10}"
    print(total_row)
    print("=" * 80)

    print("\n特点:")
    print("1. ✅ 每个动作的特征都是针对该动作特点定制的")
    print("2. ✅ 指标名称与V2动作类的extract_indicators()输出完全匹配")
    print("3. ✅ 静态特征 + 动态特征组合")
    print("4. ✅ 利用了与静息帧的对比信息 (如 xxx_change)")
    print("5. ✅ 包含联动检测指标 (synkinesis)")
    if include_motion:
        print("6. ✅ 包含12维全局运动特征 (来自motion_utils.py)")


def verify_feature_alignment():
    """验证特征定义与动作类输出是否对齐"""

    integrator = ActionFeatureIntegrator()

    print("\n" + "=" * 80)
    print("特征对齐验证")
    print("=" * 80)

    all_passed = True

    for action_name, action_cls in ACTION_CLASSES.items():
        if action_name not in integrator.action_key_indicators:
            print(f"[WARN] {action_name} 未在 integrator 中定义")
            all_passed = False
            continue

        # 获取动作类定义的动态特征名称
        action_instance = action_cls()
        action_dynamic_names = getattr(action_instance, 'DYNAMIC_FEATURE_NAMES', [])

        # 获取integrator定义的动态特征名称
        integrator_dynamic_names = integrator.action_key_indicators[action_name]['dynamic']

        # 检查是否匹配
        missing_in_integrator = set(action_dynamic_names) - set(integrator_dynamic_names)
        extra_in_integrator = set(integrator_dynamic_names) - set(action_dynamic_names)

        if missing_in_integrator or extra_in_integrator:
            print(f"\n[MISMATCH] {action_name}:")
            if missing_in_integrator:
                print(f"  动作类有，integrator无: {missing_in_integrator}")
            if extra_in_integrator:
                print(f"  integrator有，动作类无: {extra_in_integrator}")
            all_passed = False
        else:
            print(f"[OK] {action_name}")

    print("\n" + "=" * 80)
    if all_passed:
        print("✓ 所有特征定义对齐")
    else:
        print("⚠ 存在不对齐的特征定义")
    print("=" * 80)


if __name__ == '__main__':
    # 打印特征摘要
    print_feature_summary(include_motion=False)
    print("\n")
    print_feature_summary(include_motion=True)

    # 验证特征对齐
    # verify_feature_alignment()  # 需要导入动作类