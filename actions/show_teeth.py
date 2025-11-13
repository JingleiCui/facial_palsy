"""
露齿动作 (ShowTeeth)
与静息帧对比，计算嘴部展开和上唇上提
"""
import numpy as np
from .base_action import BaseAction
from ..core.geometry_utils import *


class ShowTeethAction(BaseAction):
    """露齿动作"""

    def detect_peak_frame(self, landmarks_seq, w, h):
        """
        检测峰值帧: 嘴角水平角度最大
        """
        valid_indices = [i for i, lm in enumerate(landmarks_seq) if lm is not None]
        if not valid_indices:
            return 0

        max_angle = 0
        peak_idx = valid_indices[0]

        for idx in valid_indices:
            lm = landmarks_seq[idx]
            h_line = get_horizontal_line(lm, w, h)
            mouth_left = get_point(lm, MOUTH_CORNER_LEFT, w, h)
            mouth_right = get_point(lm, MOUTH_CORNER_RIGHT, w, h)
            angle = abs(get_horizontal_angle(h_line, mouth_left, mouth_right))

            if angle > max_angle:
                max_angle = angle
                peak_idx = idx

        return peak_idx

    def extract_indicators(self, landmarks, w, h, neutral_indicators=None):
        """
        提取露齿指标，并与静息帧对比
        """
        indicators = {}

        # === 一级指标：当前状态 ===

        indicators['mouth_width'] = get_mouth_width(landmarks, w, h)
        indicators['mouth_height'] = get_mouth_height(landmarks, w, h)
        indicators['mouth_area'] = get_mouth_area(landmarks, w, h)

        # 嘴角位置
        mouth_left = get_point(landmarks, MOUTH_CORNER_LEFT, w, h)
        mouth_right = get_point(landmarks, MOUTH_CORNER_RIGHT, w, h)
        indicators['mouth_corner_left_y'] = mouth_left[1]
        indicators['mouth_corner_right_y'] = mouth_right[1]

        # 上唇位置（露齿时上提）
        upper_lip_center = get_point(landmarks, UPPER_LIP_CENTER, w, h)
        upper_lip_bottom = get_point(landmarks, UPPER_LIP_BOTTOM_CENTER, w, h)
        indicators['upper_lip_y'] = upper_lip_center[1]
        indicators['upper_lip_bottom_y'] = upper_lip_bottom[1]

        # 上唇到鼻尖距离
        nose_tip = get_point(landmarks, NOSE_TIP, w, h)
        indicators['upper_lip_nose_dist'] = dist.euclidean(upper_lip_center, nose_tip)
        indicators['upper_lip_bottom_nose_dist'] = dist.euclidean(upper_lip_bottom, nose_tip)

        # 唇部面积
        indicators['upper_lip_left_area'] = get_lip_area(landmarks, w, h, upper=True, left=True)
        indicators['upper_lip_right_area'] = get_lip_area(landmarks, w, h, upper=True, left=False)
        indicators['lower_lip_left_area'] = get_lip_area(landmarks, w, h, upper=False, left=True)
        indicators['lower_lip_right_area'] = get_lip_area(landmarks, w, h, upper=False, left=False)

        # 眼部（露齿时可能眯眼）
        indicators['left_eye_opening'] = get_eye_opening(landmarks, w, h, left=True)
        indicators['right_eye_opening'] = get_eye_opening(landmarks, w, h, left=False)
        indicators['left_eye_area'] = get_eye_area(landmarks, w, h, left=True)
        indicators['right_eye_area'] = get_eye_area(landmarks, w, h, left=False)

        # === 二级指标：与静息帧对比 ===

        if neutral_indicators:
            # 嘴部展开度
            indicators['mouth_width_change'] = (indicators['mouth_width'] -
                                               neutral_indicators['mouth_width'])
            indicators['mouth_height_change'] = (indicators['mouth_height'] -
                                                neutral_indicators['mouth_height'])
            indicators['mouth_area_change'] = (indicators['mouth_area'] -
                                              neutral_indicators['mouth_area'])

            # 展开幅度百分比
            indicators['mouth_width_expansion_pct'] = (indicators['mouth_width_change'] /
                                                       (neutral_indicators['mouth_width'] + 1e-6))

            # 上唇上提幅度（Y坐标减小）
            indicators['upper_lip_lift'] = (neutral_indicators['upper_lip_y'] -
                                           indicators['upper_lip_y'])
            indicators['upper_lip_lift_pct'] = (indicators['upper_lip_lift'] /
                                               (neutral_indicators['nose_upper_lip_dist'] + 1e-6))

            # 牙龈暴露度（上唇底部到鼻尖距离减小）
            indicators['gum_exposure'] = (neutral_indicators['nose_upper_lip_dist'] -
                                         indicators['upper_lip_bottom_nose_dist'])

            # 嘴角位置变化
            indicators['mouth_corner_left_y_change'] = (indicators['mouth_corner_left_y'] -
                                                        neutral_indicators['mouth_corner_left_y'])
            indicators['mouth_corner_right_y_change'] = (indicators['mouth_corner_right_y'] -
                                                         neutral_indicators['mouth_corner_right_y'])

            # 对称性评估
            if indicators['mouth_corner_right_y_change'] != 0:
                indicators['mouth_corner_lift_ratio'] = (indicators['mouth_corner_left_y_change'] /
                                                          indicators['mouth_corner_right_y_change'])
            else:
                indicators['mouth_corner_lift_ratio'] = 1.0

            # 眼部变化（眯眼程度）
            indicators['left_eye_opening_change'] = (indicators['left_eye_opening'] -
                                                     neutral_indicators['left_eye_opening'])
            indicators['right_eye_opening_change'] = (indicators['right_eye_opening'] -
                                                      neutral_indicators['right_eye_opening'])

        # === 二级指标：当前状态对称性 ===

        indicators['upper_lip_area_ratio'] = (indicators['upper_lip_left_area'] /
                                             (indicators['upper_lip_right_area'] + 1e-6))
        indicators['lower_lip_area_ratio'] = (indicators['lower_lip_left_area'] /
                                             (indicators['lower_lip_right_area'] + 1e-6))
        indicators['eye_opening_ratio'] = (indicators['left_eye_opening'] /
                                          (indicators['right_eye_opening'] + 1e-6))

        # 嘴角高度差
        indicators['mouth_corner_height_diff'] = abs(mouth_left[1] - mouth_right[1])

        # 形状对称性
        upper_lip_left_points = get_points(landmarks, UPPER_LIP_LEFT_POINTS, w, h)
        upper_lip_right_points = get_points(landmarks, UPPER_LIP_RIGHT_POINTS, w, h)
        indicators['upper_lip_shape_disparity'] = get_shape_disparity(upper_lip_left_points, upper_lip_right_points)

        # 角度
        h_line = get_horizontal_line(landmarks, w, h)
        indicators['mouth_horizontal_angle'] = get_horizontal_angle(h_line, mouth_left, mouth_right)

        return indicators