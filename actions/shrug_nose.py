"""
耸鼻动作 (ShrugNose)
与静息帧对比，计算上唇上提和鼻翼变化
"""
import numpy as np
from .base_action import BaseAction
from ..core.geometry_utils import *


class ShrugNoseAction(BaseAction):
    """耸鼻动作"""

    def detect_peak_frame(self, landmarks_seq, w, h):
        """
        检测峰值帧: 上唇上提最大
        """
        valid_indices = [i for i, lm in enumerate(landmarks_seq) if lm is not None]
        if not valid_indices:
            return 0

        # 以第一帧为基准
        baseline_lm = landmarks_seq[valid_indices[0]]
        baseline_upper_lip = get_point(baseline_lm, UPPER_LIP_CENTER, w, h)[1]
        baseline_unit = get_unit_length(baseline_lm, w, h)

        max_lift_normalized = 0
        peak_idx = valid_indices[0]

        for idx in valid_indices[1:]:
            lm = landmarks_seq[idx]
            current_upper_lip = get_point(lm, UPPER_LIP_CENTER, w, h)[1]
            current_unit = get_unit_length(lm, w, h)

            lift = (baseline_upper_lip - current_upper_lip) / current_unit

            if lift > max_lift_normalized:
                max_lift_normalized = lift
                peak_idx = idx

        return peak_idx

    def extract_indicators(self, landmarks, w, h, neutral_indicators=None):
        """
        提取耸鼻指标，并与静息帧对比
        """
        indicators = {}

        # === 一级指标：当前状态 ===

        # 上唇位置
        upper_lip_center = get_point(landmarks, UPPER_LIP_CENTER, w, h)
        upper_lip_bottom = get_point(landmarks, UPPER_LIP_BOTTOM_CENTER, w, h)
        indicators['upper_lip_y'] = upper_lip_center[1]
        indicators['upper_lip_bottom_y'] = upper_lip_bottom[1]

        # 鼻尖和上唇距离
        nose_tip = get_point(landmarks, NOSE_TIP, w, h)
        indicators['nose_upper_lip_dist'] = dist.euclidean(nose_tip, upper_lip_center)

        # 鼻翼宽度（耸鼻时可能变化）
        nose_left = get_point(landmarks, 392, w, h)
        nose_right = get_point(landmarks, 166, w, h)
        indicators['nose_width'] = dist.euclidean(nose_left, nose_right)

        # 上唇高度
        indicators['upper_lip_height'] = abs(upper_lip_center[1] - upper_lip_bottom[1])

        # 嘴部
        mouth_left = get_point(landmarks, MOUTH_CORNER_LEFT, w, h)
        mouth_right = get_point(landmarks, MOUTH_CORNER_RIGHT, w, h)
        indicators['mouth_width'] = dist.euclidean(mouth_left, mouth_right)
        indicators['mouth_height'] = get_mouth_height(landmarks, w, h)

        # 唇部面积
        indicators['upper_lip_left_area'] = get_lip_area(landmarks, w, h, upper=True, left=True)
        indicators['upper_lip_right_area'] = get_lip_area(landmarks, w, h, upper=True, left=False)

        # 鼻唇沟（用鼻翼到嘴角距离）
        indicators['nose_mouth_left_dist'] = dist.euclidean(nose_left, mouth_left)
        indicators['nose_mouth_right_dist'] = dist.euclidean(nose_right, mouth_right)

        # === 二级指标：与静息帧对比 ===

        if neutral_indicators:
            # 上唇上提幅度
            indicators['upper_lip_lift'] = (neutral_indicators['upper_lip_y'] -
                                           indicators['upper_lip_y'])
            indicators['upper_lip_lift_pct'] = (indicators['upper_lip_lift'] /
                                               (neutral_indicators['nose_upper_lip_dist'] + 1e-6))

            # 鼻唇距变化
            indicators['nose_lip_dist_change'] = (neutral_indicators['nose_upper_lip_dist'] -
                                                 indicators['nose_upper_lip_dist'])

            # 鼻翼宽度变化
            indicators['nose_width_change'] = (indicators['nose_width'] -
                                              neutral_indicators.get('nose_width', indicators['nose_width']))

            # 上唇高度变化
            indicators['upper_lip_height_change'] = (indicators['upper_lip_height'] -
                                                     neutral_indicators.get('upper_lip_height', indicators['upper_lip_height']))

            # 鼻唇沟深度变化
            indicators['nasolabial_fold_left_change'] = (indicators['nose_mouth_left_dist'] -
                                                          neutral_indicators['nose_mouth_left_dist'])
            indicators['nasolabial_fold_right_change'] = (indicators['nose_mouth_right_dist'] -
                                                           neutral_indicators['nose_mouth_right_dist'])

            # 对称性评估
            if indicators['nasolabial_fold_right_change'] != 0:
                indicators['nasolabial_change_ratio'] = (indicators['nasolabial_fold_left_change'] /
                                                          indicators['nasolabial_fold_right_change'])
            else:
                indicators['nasolabial_change_ratio'] = 1.0

        # === 二级指标：当前状态对称性 ===

        indicators['upper_lip_area_ratio'] = (indicators['upper_lip_left_area'] /
                                             (indicators['upper_lip_right_area'] + 1e-6))
        indicators['nasolabial_fold_ratio'] = (indicators['nose_mouth_left_dist'] /
                                               (indicators['nose_mouth_right_dist'] + 1e-6))

        # 形状对称性
        upper_lip_left_points = get_points(landmarks, UPPER_LIP_LEFT_POINTS, w, h)
        upper_lip_right_points = get_points(landmarks, UPPER_LIP_RIGHT_POINTS, w, h)
        indicators['upper_lip_shape_disparity'] = get_shape_disparity(upper_lip_left_points, upper_lip_right_points)

        return indicators