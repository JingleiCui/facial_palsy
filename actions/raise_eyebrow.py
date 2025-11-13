"""
抬眉动作 (RaiseEyebrow)
与静息帧对比，计算眉眼距变化
"""
import numpy as np
from .base_action import BaseAction
from ..core.geometry_utils import *


class RaiseEyebrowAction(BaseAction):
    """抬眉动作"""

    def detect_peak_frame(self, landmarks_seq, w, h):
        """
        检测峰值帧: 眉眼距最大的帧
        """
        valid_indices = [i for i, lm in enumerate(landmarks_seq) if lm is not None]
        if not valid_indices:
            return 0

        max_distance_normalized = 0
        peak_idx = valid_indices[0]

        for idx in valid_indices:
            lm = landmarks_seq[idx]
            unit_length = get_unit_length(lm, w, h)

            left_dist = get_eyebrow_eye_distance(lm, w, h, left=True) / unit_length
            right_dist = get_eyebrow_eye_distance(lm, w, h, left=False) / unit_length
            avg_dist_normalized = (left_dist + right_dist) / 2

            if avg_dist_normalized > max_distance_normalized:
                max_distance_normalized = avg_dist_normalized
                peak_idx = idx

        return peak_idx

    def extract_indicators(self, landmarks, w, h, neutral_indicators=None):
        """
        提取抬眉指标，并与静息帧对比

        Args:
            landmarks: 当前帧landmarks
            w, h: 图像尺寸
            neutral_indicators: 静息帧的指标（原始像素值）
        """
        indicators = {}

        # === 一级指标：当前状态 ===

        indicators['left_eyebrow_eye_dist'] = get_eyebrow_eye_distance(landmarks, w, h, left=True)
        indicators['right_eyebrow_eye_dist'] = get_eyebrow_eye_distance(landmarks, w, h, left=False)

        left_eyebrow_center = get_eyebrow_center(landmarks, w, h, left=True)
        right_eyebrow_center = get_eyebrow_center(landmarks, w, h, left=False)
        indicators['left_eyebrow_y'] = left_eyebrow_center[1]
        indicators['right_eyebrow_y'] = right_eyebrow_center[1]

        # 眼部（抬眉时可能变化）
        indicators['left_eye_opening'] = get_eye_opening(landmarks, w, h, left=True)
        indicators['right_eye_opening'] = get_eye_opening(landmarks, w, h, left=False)
        indicators['left_eye_area'] = get_eye_area(landmarks, w, h, left=True)
        indicators['right_eye_area'] = get_eye_area(landmarks, w, h, left=False)

        # 额头到眉毛距离
        forehead = get_point(landmarks, FOREHEAD, w, h)
        indicators['left_forehead_eyebrow_dist'] = dist.euclidean(forehead, left_eyebrow_center)
        indicators['right_forehead_eyebrow_dist'] = dist.euclidean(forehead, right_eyebrow_center)

        # === 二级指标：与静息帧对比（如果提供）===

        if neutral_indicators:
            # 眉眼距变化（绝对值，像素）
            indicators['left_eyebrow_dist_change'] = (indicators['left_eyebrow_eye_dist'] -
                                                      neutral_indicators['left_eyebrow_eye_dist'])
            indicators['right_eyebrow_dist_change'] = (indicators['right_eyebrow_eye_dist'] -
                                                       neutral_indicators['right_eyebrow_eye_dist'])

            # 眉毛上抬幅度（Y坐标减小表示上抬）
            indicators['left_eyebrow_lift'] = (neutral_indicators['left_eyebrow_y'] -
                                               indicators['left_eyebrow_y'])
            indicators['right_eyebrow_lift'] = (neutral_indicators['right_eyebrow_y'] -
                                                indicators['right_eyebrow_y'])

            # 眼睛开合度变化
            indicators['left_eye_opening_change'] = (indicators['left_eye_opening'] -
                                                     neutral_indicators['left_eye_opening'])
            indicators['right_eye_opening_change'] = (indicators['right_eye_opening'] -
                                                      neutral_indicators['right_eye_opening'])

            # 变化量比例（对称性评估）
            if indicators['right_eyebrow_dist_change'] != 0:
                indicators['eyebrow_change_ratio'] = (indicators['left_eyebrow_dist_change'] /
                                                      indicators['right_eyebrow_dist_change'])
            else:
                indicators['eyebrow_change_ratio'] = 1.0

            if indicators['right_eyebrow_lift'] != 0:
                indicators['eyebrow_lift_ratio'] = (indicators['left_eyebrow_lift'] /
                                                    indicators['right_eyebrow_lift'])
            else:
                indicators['eyebrow_lift_ratio'] = 1.0

        # === 二级指标：当前状态的对称性 ===

        indicators['eyebrow_dist_ratio'] = (indicators['left_eyebrow_eye_dist'] /
                                           (indicators['right_eyebrow_eye_dist'] + 1e-6))
        indicators['eye_opening_ratio'] = (indicators['left_eye_opening'] /
                                          (indicators['right_eye_opening'] + 1e-6))
        indicators['eye_area_ratio'] = (indicators['left_eye_area'] /
                                       (indicators['right_eye_area'] + 1e-6))

        # 眉毛高度差
        indicators['eyebrow_height_diff'] = abs(left_eyebrow_center[1] - right_eyebrow_center[1])

        # 眉毛形状对称性
        left_eyebrow_points = get_points(landmarks, EYEBROW_LEFT_POINTS, w, h)
        right_eyebrow_points = get_points(landmarks, EYEBROW_RIGHT_POINTS, w, h)
        indicators['eyebrow_shape_disparity'] = get_shape_disparity(left_eyebrow_points, right_eyebrow_points)

        # 角度
        h_line = get_horizontal_line(landmarks, w, h)
        indicators['eyebrow_horizontal_angle'] = get_horizontal_angle(h_line, left_eyebrow_center, right_eyebrow_center)

        return indicators