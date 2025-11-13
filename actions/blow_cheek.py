"""
鼓腮动作 (BlowCheek)
与静息帧对比，计算嘴部闭合和腮部膨胀
"""
import numpy as np
from .base_action import BaseAction
from ..core.geometry_utils import *


class BlowCheekAction(BaseAction):
    """鼓腮动作"""

    def detect_peak_frame(self, landmarks_seq, w, h):
        """
        检测峰值帧: 嘴部高度最小（闭合）
        """
        valid_indices = [i for i, lm in enumerate(landmarks_seq) if lm is not None]
        if not valid_indices:
            return 0

        min_height_normalized = float('inf')
        peak_idx = valid_indices[0]

        for idx in valid_indices:
            lm = landmarks_seq[idx]
            unit_length = get_unit_length(lm, w, h)
            mouth_h = get_mouth_height(lm, w, h) / unit_length

            if mouth_h < min_height_normalized:
                min_height_normalized = mouth_h
                peak_idx = idx

        return peak_idx

    def extract_indicators(self, landmarks, w, h, neutral_indicators=None):
        """
        提取鼓腮指标，并与静息帧对比
        """
        indicators = {}

        # === 一级指标：当前状态 ===

        indicators['mouth_width'] = get_mouth_width(landmarks, w, h)
        indicators['mouth_height'] = get_mouth_height(landmarks, w, h)
        indicators['mouth_area'] = get_mouth_area(landmarks, w, h)

        # 单侧唇部面积
        indicators['upper_lip_left_area'] = get_lip_area(landmarks, w, h, upper=True, left=True)
        indicators['upper_lip_right_area'] = get_lip_area(landmarks, w, h, upper=True, left=False)
        indicators['lower_lip_left_area'] = get_lip_area(landmarks, w, h, upper=False, left=True)
        indicators['lower_lip_right_area'] = get_lip_area(landmarks, w, h, upper=False, left=False)

        # 嘴角位置
        mouth_left = get_point(landmarks, MOUTH_CORNER_LEFT, w, h)
        mouth_right = get_point(landmarks, MOUTH_CORNER_RIGHT, w, h)
        indicators['mouth_corner_left_y'] = mouth_left[1]
        indicators['mouth_corner_right_y'] = mouth_right[1]

        # === 二级指标：与静息帧对比 ===

        if neutral_indicators:
            # 嘴部尺寸变化
            indicators['mouth_width_change'] = (indicators['mouth_width'] -
                                               neutral_indicators['mouth_width'])
            indicators['mouth_height_change'] = (indicators['mouth_height'] -
                                                neutral_indicators['mouth_height'])
            indicators['mouth_area_change'] = (indicators['mouth_area'] -
                                              neutral_indicators['mouth_area'])

            # 唇部面积变化（鼓腮时可能膨胀）
            indicators['upper_lip_left_area_change'] = (indicators['upper_lip_left_area'] -
                                                        neutral_indicators['upper_lip_left_area'])
            indicators['upper_lip_right_area_change'] = (indicators['upper_lip_right_area'] -
                                                         neutral_indicators['upper_lip_right_area'])

            # 嘴角位置变化（Y坐标）
            indicators['mouth_corner_left_y_change'] = (indicators['mouth_corner_left_y'] -
                                                        neutral_indicators['mouth_corner_left_y'])
            indicators['mouth_corner_right_y_change'] = (indicators['mouth_corner_right_y'] -
                                                         neutral_indicators['mouth_corner_right_y'])

            # 对称性评估
            if indicators['upper_lip_right_area_change'] != 0:
                indicators['upper_lip_area_change_ratio'] = (indicators['upper_lip_left_area_change'] /
                                                              indicators['upper_lip_right_area_change'])
            else:
                indicators['upper_lip_area_change_ratio'] = 1.0

        # === 二级指标：当前状态对称性 ===

        indicators['upper_lip_area_ratio'] = (indicators['upper_lip_left_area'] /
                                             (indicators['upper_lip_right_area'] + 1e-6))
        indicators['lower_lip_area_ratio'] = (indicators['lower_lip_left_area'] /
                                             (indicators['lower_lip_right_area'] + 1e-6))

        # 嘴角高度差
        indicators['mouth_corner_height_diff'] = abs(mouth_left[1] - mouth_right[1])

        # 唇部形状对称性
        upper_lip_left_points = get_points(landmarks, UPPER_LIP_LEFT_POINTS, w, h)
        upper_lip_right_points = get_points(landmarks, UPPER_LIP_RIGHT_POINTS, w, h)
        indicators['upper_lip_shape_disparity'] = get_shape_disparity(upper_lip_left_points, upper_lip_right_points)

        lower_lip_left_points = get_points(landmarks, LOWER_LIP_LEFT_POINTS, w, h)
        lower_lip_right_points = get_points(landmarks, LOWER_LIP_RIGHT_POINTS, w, h)
        indicators['lower_lip_shape_disparity'] = get_shape_disparity(lower_lip_left_points, lower_lip_right_points)

        # 角度
        h_line = get_horizontal_line(landmarks, w, h)
        indicators['mouth_horizontal_angle'] = get_horizontal_angle(h_line, mouth_left, mouth_right)

        return indicators