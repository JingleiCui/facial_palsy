"""
轻闭眼动作 (CloseEyeSoftly)
与静息帧对比，计算部分闭合程度
"""
import numpy as np
from .base_action import BaseAction
from ..core.geometry_utils import *


class CloseEyeSoftlyAction(BaseAction):
    """轻闭眼动作"""

    def detect_peak_frame(self, landmarks_seq, w, h):
        """
        检测峰值帧: 前30%最小值的中位数帧
        """
        valid_indices = [i for i, lm in enumerate(landmarks_seq) if lm is not None]
        if not valid_indices:
            return 0

        eye_openings = []
        for idx in valid_indices:
            lm = landmarks_seq[idx]
            unit_length = get_unit_length(lm, w, h)
            left_eye = get_eye_opening(lm, w, h, left=True) / unit_length
            right_eye = get_eye_opening(lm, w, h, left=False) / unit_length
            avg_opening = (left_eye + right_eye) / 2
            eye_openings.append((idx, avg_opening))

        eye_openings.sort(key=lambda x: x[1])

        top_30_percent = int(len(eye_openings) * 0.3)
        if top_30_percent < 1:
            top_30_percent = 1

        median_idx = top_30_percent // 2
        return eye_openings[median_idx][0]

    def extract_indicators(self, landmarks, w, h, neutral_indicators=None):
        """
        提取轻闭眼指标，并与静息帧对比
        """
        indicators = {}

        # === 一级指标：当前状态 ===

        indicators['left_eye_opening'] = get_eye_opening(landmarks, w, h, left=True)
        indicators['right_eye_opening'] = get_eye_opening(landmarks, w, h, left=False)
        indicators['left_eye_area'] = get_eye_area(landmarks, w, h, left=True)
        indicators['right_eye_area'] = get_eye_area(landmarks, w, h, left=False)
        indicators['left_eye_length'] = get_eye_length(landmarks, w, h, left=True)
        indicators['right_eye_length'] = get_eye_length(landmarks, w, h, left=False)

        # === 二级指标：与静息帧对比 ===

        if neutral_indicators:
            # 眼睛闭合度
            indicators['left_eye_closure'] = (neutral_indicators['left_eye_opening'] -
                                             indicators['left_eye_opening'])
            indicators['right_eye_closure'] = (neutral_indicators['right_eye_opening'] -
                                              indicators['right_eye_opening'])

            # 闭合完成度百分比
            indicators['left_eye_closure_pct'] = (indicators['left_eye_closure'] /
                                                  (neutral_indicators['left_eye_opening'] + 1e-6))
            indicators['right_eye_closure_pct'] = (indicators['right_eye_closure'] /
                                                   (neutral_indicators['right_eye_opening'] + 1e-6))

            # 面积减少百分比
            indicators['left_eye_area_reduction_pct'] = ((neutral_indicators['left_eye_area'] -
                                                          indicators['left_eye_area']) /
                                                         (neutral_indicators['left_eye_area'] + 1e-6))
            indicators['right_eye_area_reduction_pct'] = ((neutral_indicators['right_eye_area'] -
                                                           indicators['right_eye_area']) /
                                                          (neutral_indicators['right_eye_area'] + 1e-6))

            # 闭合对称性
            if indicators['right_eye_closure'] != 0:
                indicators['eye_closure_ratio'] = (indicators['left_eye_closure'] /
                                                   indicators['right_eye_closure'])
            else:
                indicators['eye_closure_ratio'] = 1.0

        # === 二级指标：当前状态对称性 ===

        indicators['eye_opening_ratio'] = (indicators['left_eye_opening'] /
                                          (indicators['right_eye_opening'] + 1e-6))
        indicators['eye_area_ratio'] = (indicators['left_eye_area'] /
                                       (indicators['right_eye_area'] + 1e-6))

        # 眼睑收缩度
        indicators['left_eye_contraction'] = (indicators['left_eye_opening'] /
                                             (indicators['left_eye_length'] + 1e-6))
        indicators['right_eye_contraction'] = (indicators['right_eye_opening'] /
                                              (indicators['right_eye_length'] + 1e-6))

        # 形状对称性
        left_eye_points = get_points(landmarks, LEFT_EYE_POINTS, w, h)
        right_eye_points = get_points(landmarks, RIGHT_EYE_POINTS, w, h)
        indicators['eye_shape_disparity'] = get_shape_disparity(left_eye_points, right_eye_points)

        return indicators