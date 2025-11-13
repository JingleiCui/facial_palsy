"""
微笑动作 (Smile)
与静息帧对比，计算嘴角展开和上扬幅度
"""
import numpy as np
from .base_action import BaseAction
from ..core.geometry_utils import *


class SmileAction(BaseAction):
    """微笑动作"""

    def detect_peak_frame(self, landmarks_seq, w, h):
        """
        检测峰值帧: 嘴角宽度最大
        """
        valid_indices = [i for i, lm in enumerate(landmarks_seq) if lm is not None]
        if not valid_indices:
            return 0

        max_width_normalized = 0
        peak_idx = valid_indices[0]

        for idx in valid_indices:
            lm = landmarks_seq[idx]
            unit_length = get_unit_length(lm, w, h)
            mouth_w = get_mouth_width(lm, w, h) / unit_length

            if mouth_w > max_width_normalized:
                max_width_normalized = mouth_w
                peak_idx = idx

        return peak_idx

    def extract_indicators(self, landmarks, w, h, neutral_indicators=None):
        """
        提取微笑指标，并与静息帧对比
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

        # 嘴角到鼻尖距离
        nose_tip = get_point(landmarks, NOSE_TIP, w, h)
        indicators['mouth_left_nose_dist'] = dist.euclidean(mouth_left, nose_tip)
        indicators['mouth_right_nose_dist'] = dist.euclidean(mouth_right, nose_tip)

        # 唇部面积
        indicators['upper_lip_left_area'] = get_lip_area(landmarks, w, h, upper=True, left=True)
        indicators['upper_lip_right_area'] = get_lip_area(landmarks, w, h, upper=True, left=False)
        indicators['lower_lip_left_area'] = get_lip_area(landmarks, w, h, upper=False, left=True)
        indicators['lower_lip_right_area'] = get_lip_area(landmarks, w, h, upper=False, left=False)

        # 眼部（微笑时可能眯眼）
        indicators['left_eye_opening'] = get_eye_opening(landmarks, w, h, left=True)
        indicators['right_eye_opening'] = get_eye_opening(landmarks, w, h, left=False)
        indicators['left_eye_area'] = get_eye_area(landmarks, w, h, left=True)
        indicators['right_eye_area'] = get_eye_area(landmarks, w, h, left=False)

        # 鼻唇沟
        nose_left = get_point(landmarks, 392, w, h)
        nose_right = get_point(landmarks, 166, w, h)
        indicators['nasolabial_fold_left'] = dist.euclidean(nose_left, mouth_left)
        indicators['nasolabial_fold_right'] = dist.euclidean(nose_right, mouth_right)

        # === 二级指标：与静息帧对比 ===

        if neutral_indicators:
            # 嘴部展开幅度
            indicators['mouth_width_change'] = (indicators['mouth_width'] -
                                               neutral_indicators['mouth_width'])
            indicators['mouth_width_expansion_pct'] = (indicators['mouth_width_change'] /
                                                       (neutral_indicators['mouth_width'] + 1e-6))

            indicators['mouth_height_change'] = (indicators['mouth_height'] -
                                                neutral_indicators['mouth_height'])
            indicators['mouth_area_change'] = (indicators['mouth_area'] -
                                              neutral_indicators['mouth_area'])

            # 嘴角上扬幅度（Y坐标减小）
            indicators['mouth_corner_left_lift'] = (neutral_indicators['mouth_corner_left_y'] -
                                                   indicators['mouth_corner_left_y'])
            indicators['mouth_corner_right_lift'] = (neutral_indicators['mouth_corner_right_y'] -
                                                    indicators['mouth_corner_right_y'])

            # 上扬幅度百分比
            indicators['mouth_corner_left_lift_pct'] = (indicators['mouth_corner_left_lift'] /
                                                        (neutral_indicators['nose_mouth_left_dist'] + 1e-6))
            indicators['mouth_corner_right_lift_pct'] = (indicators['mouth_corner_right_lift'] /
                                                         (neutral_indicators['nose_mouth_right_dist'] + 1e-6))

            # 对称性评估（上扬幅度比）
            if indicators['mouth_corner_right_lift'] != 0:
                indicators['smile_lift_ratio'] = (indicators['mouth_corner_left_lift'] /
                                                  indicators['mouth_corner_right_lift'])
            else:
                indicators['smile_lift_ratio'] = 1.0

            # 鼻唇沟加深程度
            indicators['nasolabial_fold_left_change'] = (indicators['nasolabial_fold_left'] -
                                                          neutral_indicators['nose_mouth_left_dist'])
            indicators['nasolabial_fold_right_change'] = (indicators['nasolabial_fold_right'] -
                                                           neutral_indicators['nose_mouth_right_dist'])

            # 眼部变化（眯眼程度）
            indicators['left_eye_opening_change'] = (indicators['left_eye_opening'] -
                                                     neutral_indicators['left_eye_opening'])
            indicators['right_eye_opening_change'] = (indicators['right_eye_opening'] -
                                                      neutral_indicators['right_eye_opening'])

            indicators['left_eye_area_change'] = (indicators['left_eye_area'] -
                                                  neutral_indicators['left_eye_area'])
            indicators['right_eye_area_change'] = (indicators['right_eye_area'] -
                                                   neutral_indicators['right_eye_area'])

            # 眼部变化对称性
            if indicators['right_eye_opening_change'] != 0:
                indicators['eye_change_ratio'] = (indicators['left_eye_opening_change'] /
                                                  indicators['right_eye_opening_change'])
            else:
                indicators['eye_change_ratio'] = 1.0

        # === 二级指标：当前状态对称性 ===

        indicators['upper_lip_area_ratio'] = (indicators['upper_lip_left_area'] /
                                             (indicators['upper_lip_right_area'] + 1e-6))
        indicators['lower_lip_area_ratio'] = (indicators['lower_lip_left_area'] /
                                             (indicators['lower_lip_right_area'] + 1e-6))
        indicators['eye_opening_ratio'] = (indicators['left_eye_opening'] /
                                          (indicators['right_eye_opening'] + 1e-6))
        indicators['eye_area_ratio'] = (indicators['left_eye_area'] /
                                       (indicators['right_eye_area'] + 1e-6))

        # 嘴角高度差
        indicators['mouth_corner_height_diff'] = abs(mouth_left[1] - mouth_right[1])

        # 形状对称性
        upper_lip_left_points = get_points(landmarks, UPPER_LIP_LEFT_POINTS, w, h)
        upper_lip_right_points = get_points(landmarks, UPPER_LIP_RIGHT_POINTS, w, h)
        indicators['upper_lip_shape_disparity'] = get_shape_disparity(upper_lip_left_points, upper_lip_right_points)

        left_eye_points = get_points(landmarks, LEFT_EYE_POINTS, w, h)
        right_eye_points = get_points(landmarks, RIGHT_EYE_POINTS, w, h)
        indicators['eye_shape_disparity'] = get_shape_disparity(left_eye_points, right_eye_points)

        # 角度
        h_line = get_horizontal_line(landmarks, w, h)
        indicators['mouth_horizontal_angle'] = get_horizontal_angle(h_line, mouth_left, mouth_right)

        return indicators