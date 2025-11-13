"""
主动眨眼动作 (VoluntaryEyeBlink)
与静息帧对比，计算眨眼闭合程度
"""
import numpy as np
from .base_action import BaseAction
from ..core.geometry_utils import *


class VoluntaryEyeBlinkAction(BaseAction):
    """主动眨眼动作"""

    def detect_peak_frame(self, landmarks_seq, w, h):
        """
        检测峰值帧: 眼睛开合度最小
        """
        valid_indices = [i for i, lm in enumerate(landmarks_seq) if lm is not None]
        if not valid_indices:
            return 0

        min_opening_normalized = float('inf')
        peak_idx = valid_indices[0]

        for idx in valid_indices:
            lm = landmarks_seq[idx]
            unit_length = get_unit_length(lm, w, h)

            left_eye = get_eye_opening(lm, w, h, left=True) / unit_length
            right_eye = get_eye_opening(lm, w, h, left=False) / unit_length
            avg_opening_normalized = (left_eye + right_eye) / 2

            if avg_opening_normalized < min_opening_normalized:
                min_opening_normalized = avg_opening_normalized
                peak_idx = idx

        return peak_idx

    def extract_indicators(self, landmarks, w, h, neutral_indicators=None):
        """
        提取主动眨眼指标，并与静息帧对比
        """
        indicators = {}

        # === 一级指标：当前状态 ===

        indicators['left_eye_opening'] = get_eye_opening(landmarks, w, h, left=True)
        indicators['right_eye_opening'] = get_eye_opening(landmarks, w, h, left=False)
        indicators['left_eye_area'] = get_eye_area(landmarks, w, h, left=True)
        indicators['right_eye_area'] = get_eye_area(landmarks, w, h, left=False)
        indicators['left_eye_length'] = get_eye_length(landmarks, w, h, left=True)
        indicators['right_eye_length'] = get_eye_length(landmarks, w, h, left=False)

        # 眉眼距（主动眨眼时眉毛可能更用力下压）
        indicators['left_eyebrow_eye_dist'] = get_eyebrow_eye_distance(landmarks, w, h, left=True)
        indicators['right_eyebrow_eye_dist'] = get_eyebrow_eye_distance(landmarks, w, h, left=False)

        # === 二级指标：与静息帧对比 ===

        if neutral_indicators:
            # 眨眼闭合度
            indicators['left_eye_closure'] = (neutral_indicators['left_eye_opening'] -
                                             indicators['left_eye_opening'])
            indicators['right_eye_closure'] = (neutral_indicators['right_eye_opening'] -
                                              indicators['right_eye_opening'])

            # 闭合完成度百分比
            indicators['left_eye_closure_pct'] = (indicators['left_eye_closure'] /
                                                  (neutral_indicators['left_eye_opening'] + 1e-6))
            indicators['right_eye_closure_pct'] = (indicators['right_eye_closure'] /
                                                   (neutral_indicators['right_eye_opening'] + 1e-6))

            # 面积减少
            indicators['left_eye_area_reduction'] = (neutral_indicators['left_eye_area'] -
                                                     indicators['left_eye_area'])
            indicators['right_eye_area_reduction'] = (neutral_indicators['right_eye_area'] -
                                                      indicators['right_eye_area'])

            indicators['left_eye_area_reduction_pct'] = (indicators['left_eye_area_reduction'] /
                                                         (neutral_indicators['left_eye_area'] + 1e-6))
            indicators['right_eye_area_reduction_pct'] = (indicators['right_eye_area_reduction'] /
                                                          (neutral_indicators['right_eye_area'] + 1e-6))

            # 眉眼距变化（主动眨眼时眉毛下压）
            indicators['left_eyebrow_dist_change'] = (indicators['left_eyebrow_eye_dist'] -
                                                      neutral_indicators['left_eyebrow_eye_dist'])
            indicators['right_eyebrow_dist_change'] = (indicators['right_eyebrow_eye_dist'] -
                                                       neutral_indicators['right_eyebrow_eye_dist'])

            # 对称性评估
            if indicators['right_eye_closure'] != 0:
                indicators['eye_closure_ratio'] = (indicators['left_eye_closure'] /
                                                   indicators['right_eye_closure'])
            else:
                indicators['eye_closure_ratio'] = 1.0

            if indicators['right_eyebrow_dist_change'] != 0:
                indicators['eyebrow_change_ratio'] = (indicators['left_eyebrow_dist_change'] /
                                                      indicators['right_eyebrow_dist_change'])
            else:
                indicators['eyebrow_change_ratio'] = 1.0

        # === 二级指标：当前状态对称性 ===

        indicators['eye_opening_ratio'] = (indicators['left_eye_opening'] /
                                          (indicators['right_eye_opening'] + 1e-6))
        indicators['eye_area_ratio'] = (indicators['left_eye_area'] /
                                       (indicators['right_eye_area'] + 1e-6))
        indicators['eye_length_ratio'] = (indicators['left_eye_length'] /
                                         (indicators['right_eye_length'] + 1e-6))
        indicators['eyebrow_dist_ratio'] = (indicators['left_eyebrow_eye_dist'] /
                                           (indicators['right_eyebrow_eye_dist'] + 1e-6))

        # 眼睑收缩度
        indicators['left_eye_contraction'] = (indicators['left_eye_opening'] /
                                             (indicators['left_eye_length'] + 1e-6))
        indicators['right_eye_contraction'] = (indicators['right_eye_opening'] /
                                              (indicators['right_eye_length'] + 1e-6))

        # 形状对称性
        left_eye_points = get_points(landmarks, LEFT_EYE_POINTS, w, h)
        right_eye_points = get_points(landmarks, RIGHT_EYE_POINTS, w, h)
        indicators['eye_shape_disparity'] = get_shape_disparity(left_eye_points, right_eye_points)

        # 角度
        h_line = get_horizontal_line(landmarks, w, h)
        left_inner = get_point(landmarks, EYE_INNER_CANTHUS_LEFT, w, h)
        right_inner = get_point(landmarks, EYE_INNER_CANTHUS_RIGHT, w, h)
        indicators['eye_horizontal_angle'] = get_horizontal_angle(h_line, left_inner, right_inner)

        return indicators