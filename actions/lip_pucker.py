"""
噘嘴动作 (LipPucker)
与静息帧对比，计算嘴部前伸程度
"""
import numpy as np
from .base_action import BaseAction
from ..core.geometry_utils import *


class LipPuckerAction(BaseAction):
    """噘嘴动作"""

    def detect_peak_frame(self, landmarks_seq, w, h):
        """
        检测峰值帧: 嘴部高度最大
        """
        valid_indices = [i for i, lm in enumerate(landmarks_seq) if lm is not None]
        if not valid_indices:
            return 0

        max_height_normalized = 0
        peak_idx = valid_indices[0]

        for idx in valid_indices:
            lm = landmarks_seq[idx]
            unit_length = get_unit_length(lm, w, h)
            mouth_h = get_mouth_height(lm, w, h) / unit_length

            if mouth_h > max_height_normalized:
                max_height_normalized = mouth_h
                peak_idx = idx

        return peak_idx

    def extract_indicators(self, landmarks, w, h, neutral_indicators=None):
        """
        提取噘嘴指标，并与静息帧对比
        """
        indicators = {}

        # === 一级指标：当前状态 ===

        indicators['mouth_width'] = get_mouth_width(landmarks, w, h)
        indicators['mouth_height'] = get_mouth_height(landmarks, w, h)
        indicators['mouth_area'] = get_mouth_area(landmarks, w, h)

        # 唇部中心点
        upper_lip_center = get_point(landmarks, UPPER_LIP_CENTER, w, h)
        lower_lip_center = get_point(landmarks, LOWER_LIP_BOTTOM_CENTER, w, h)

        # 唇部到鼻尖距离（噘嘴时前伸）
        nose_tip = get_point(landmarks, NOSE_TIP, w, h)
        indicators['upper_lip_nose_dist'] = dist.euclidean(upper_lip_center, nose_tip)
        indicators['lower_lip_nose_dist'] = dist.euclidean(lower_lip_center, nose_tip)

        # 嘴角到中心距离
        mouth_left = get_point(landmarks, MOUTH_CORNER_LEFT, w, h)
        mouth_right = get_point(landmarks, MOUTH_CORNER_RIGHT, w, h)
        mouth_center_x = (mouth_left[0] + mouth_right[0]) / 2
        mouth_center_y = (mouth_left[1] + mouth_right[1]) / 2
        mouth_center = (mouth_center_x, mouth_center_y)

        indicators['mouth_left_center_dist'] = dist.euclidean(mouth_left, mouth_center)
        indicators['mouth_right_center_dist'] = dist.euclidean(mouth_right, mouth_center)

        # 唇部面积
        indicators['upper_lip_left_area'] = get_lip_area(landmarks, w, h, upper=True, left=True)
        indicators['upper_lip_right_area'] = get_lip_area(landmarks, w, h, upper=True, left=False)

        # === 二级指标：与静息帧对比 ===

        if neutral_indicators:
            # 嘴部尺寸变化
            indicators['mouth_width_change'] = (indicators['mouth_width'] -
                                               neutral_indicators['mouth_width'])
            indicators['mouth_height_change'] = (indicators['mouth_height'] -
                                                neutral_indicators['mouth_height'])

            # 噘嘴程度（宽度减小，高度增加）
            indicators['pucker_width_ratio'] = (indicators['mouth_width'] /
                                               (neutral_indicators['mouth_width'] + 1e-6))
            indicators['pucker_height_ratio'] = (indicators['mouth_height'] /
                                                (neutral_indicators['mouth_height'] + 1e-6))

            # 唇部前伸度（鼻唇距减小）
            indicators['upper_lip_protrusion'] = (neutral_indicators['nose_upper_lip_dist'] -
                                                 indicators['upper_lip_nose_dist'])

            # 嘴角收缩度
            indicators['mouth_corner_retraction_left'] = (neutral_indicators['mouth_width'] / 2 -
                                                          indicators['mouth_left_center_dist'])
            indicators['mouth_corner_retraction_right'] = (neutral_indicators['mouth_width'] / 2 -
                                                           indicators['mouth_right_center_dist'])

            # 对称性
            if indicators['mouth_corner_retraction_right'] != 0:
                indicators['corner_retraction_ratio'] = (indicators['mouth_corner_retraction_left'] /
                                                          indicators['mouth_corner_retraction_right'])
            else:
                indicators['corner_retraction_ratio'] = 1.0

        # === 二级指标：当前状态对称性 ===

        indicators['upper_lip_area_ratio'] = (indicators['upper_lip_left_area'] /
                                             (indicators['upper_lip_right_area'] + 1e-6))

        # 嘴角高度差
        indicators['mouth_corner_height_diff'] = abs(mouth_left[1] - mouth_right[1])

        # 形状对称性
        upper_lip_left_points = get_points(landmarks, UPPER_LIP_LEFT_POINTS, w, h)
        upper_lip_right_points = get_points(landmarks, UPPER_LIP_RIGHT_POINTS, w, h)
        indicators['upper_lip_shape_disparity'] = get_shape_disparity(upper_lip_left_points, upper_lip_right_points)

        return indicators