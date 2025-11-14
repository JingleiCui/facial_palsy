"""
静息动作 (NeutralFace)
作为所有动作的基准参考
"""
import numpy as np
from .base_action import BaseAction
from ..core.geometry_utils import *


class NeutralFaceAction(BaseAction):
    """静息动作 - 基准状态"""

    def detect_peak_frame(self, landmarks_seq, w, h):
        """
        检测峰值帧: 取中间40%最稳定的帧
        """
        valid_indices = [i for i, lm in enumerate(landmarks_seq) if lm is not None]
        if not valid_indices:
            return 0

        total = len(valid_indices)
        start_idx = int(total * 0.3)
        end_idx = int(total * 0.7)

        if end_idx <= start_idx:
            return valid_indices[total // 2]

        middle_indices = valid_indices[start_idx:end_idx]

        # 计算每帧的归一化眼睛开合度
        eye_openings_norm = []
        for idx in middle_indices:
            lm = landmarks_seq[idx]
            unit_length = get_unit_length(lm, w, h)
            left_eye = get_eye_opening(lm, w, h, left=True) / unit_length
            right_eye = get_eye_opening(lm, w, h, left=False) / unit_length
            eye_openings_norm.append((left_eye + right_eye) / 2)

        # 找最稳定的帧
        if len(eye_openings_norm) < 3:
            return middle_indices[len(middle_indices) // 2]

        stabilities = []
        for i in range(1, len(eye_openings_norm) - 1):
            diff = (abs(eye_openings_norm[i] - eye_openings_norm[i-1]) +
                    abs(eye_openings_norm[i] - eye_openings_norm[i+1]))
            stabilities.append(diff)

        most_stable_idx = np.argmin(stabilities) + 1
        return middle_indices[most_stable_idx]

    def extract_indicators(self, landmarks, w, h, neutral_indicators=None):
        """
        提取静息状态的一级指标（原始像素值）
        这些指标将作为其他动作的基准
        """
        indicators = {}

        # === 一级指标：基础测量（原始像素值）===

        # 眼部
        indicators['left_eye_opening'] = get_eye_opening(landmarks, w, h, left=True)
        indicators['right_eye_opening'] = get_eye_opening(landmarks, w, h, left=False)
        indicators['left_eye_length'] = get_eye_length(landmarks, w, h, left=True)
        indicators['right_eye_length'] = get_eye_length(landmarks, w, h, left=False)
        indicators['left_eye_area'] = get_eye_area(landmarks, w, h, left=True)
        indicators['right_eye_area'] = get_eye_area(landmarks, w, h, left=False)

        # 眉毛
        indicators['left_eyebrow_eye_dist'] = get_eyebrow_eye_distance(landmarks, w, h, left=True)
        indicators['right_eyebrow_eye_dist'] = get_eyebrow_eye_distance(landmarks, w, h, left=False)

        left_eyebrow_center = get_eyebrow_center(landmarks, w, h, left=True)
        right_eyebrow_center = get_eyebrow_center(landmarks, w, h, left=False)
        indicators['left_eyebrow_y'] = left_eyebrow_center[1]
        indicators['right_eyebrow_y'] = right_eyebrow_center[1]

        # 嘴部
        indicators['mouth_width'] = get_mouth_width(landmarks, w, h)
        indicators['mouth_height'] = get_mouth_height(landmarks, w, h)
        indicators['mouth_area'] = get_mouth_area(landmarks, w, h)

        mouth_left = get_point(landmarks, MOUTH_CORNER_LEFT, w, h)
        mouth_right = get_point(landmarks, MOUTH_CORNER_RIGHT, w, h)
        indicators['mouth_corner_left_y'] = mouth_left[1]
        indicators['mouth_corner_right_y'] = mouth_right[1]

        # 上唇
        upper_lip_center = get_point(landmarks, UPPER_LIP_CENTER, w, h)
        indicators['upper_lip_y'] = upper_lip_center[1]

        indicators['upper_lip_left_area'] = get_lip_area(landmarks, w, h, upper=True, left=True)
        indicators['upper_lip_right_area'] = get_lip_area(landmarks, w, h, upper=True, left=False)

        # 下唇
        indicators['lower_lip_left_area'] = get_lip_area(landmarks, w, h, upper=False, left=True)
        indicators['lower_lip_right_area'] = get_lip_area(landmarks, w, h, upper=False, left=False)

        # 鼻唇距
        nose_tip = get_point(landmarks, NOSE_TIP, w, h)
        indicators['nose_upper_lip_dist'] = dist.euclidean(nose_tip, upper_lip_center)
        indicators['nose_mouth_left_dist'] = dist.euclidean(nose_tip, mouth_left)
        indicators['nose_mouth_right_dist'] = dist.euclidean(nose_tip, mouth_right)

        # === 二级指标：对称性比例 ===

        indicators['eye_opening_ratio'] = (indicators['left_eye_opening'] /
                                          (indicators['right_eye_opening'] + 1e-6))
        indicators['eye_area_ratio'] = (indicators['left_eye_area'] /
                                       (indicators['right_eye_area'] + 1e-6))
        indicators['eyebrow_dist_ratio'] = (indicators['left_eyebrow_eye_dist'] /
                                           (indicators['right_eyebrow_eye_dist'] + 1e-6))
        indicators['upper_lip_area_ratio'] = (indicators['upper_lip_left_area'] /
                                             (indicators['upper_lip_right_area'] + 1e-6))
        indicators['lower_lip_area_ratio'] = (indicators['lower_lip_left_area'] /
                                             (indicators['lower_lip_right_area'] + 1e-6))

        # 嘴角高度差
        indicators['mouth_corner_height_diff'] = abs(mouth_left[1] - mouth_right[1])

        # 眉毛高度差
        indicators['eyebrow_height_diff'] = abs(left_eyebrow_center[1] - right_eyebrow_center[1])

        # === 二级指标：形状对称性（Procrustes）===

        left_eye_points = get_points(landmarks, LEFT_EYE_POINTS, w, h)
        right_eye_points = get_points(landmarks, RIGHT_EYE_POINTS, w, h)
        indicators['eye_shape_disparity'] = get_shape_disparity(left_eye_points, right_eye_points)

        upper_lip_left_points = get_points(landmarks, UPPER_LIP_LEFT_POINTS, w, h)
        upper_lip_right_points = get_points(landmarks, UPPER_LIP_RIGHT_POINTS, w, h)
        indicators['upper_lip_shape_disparity'] = get_shape_disparity(upper_lip_left_points, upper_lip_right_points)

        lower_lip_left_points = get_points(landmarks, LOWER_LIP_LEFT_POINTS, w, h)
        lower_lip_right_points = get_points(landmarks, LOWER_LIP_RIGHT_POINTS, w, h)
        indicators['lower_lip_shape_disparity'] = get_shape_disparity(lower_lip_left_points, lower_lip_right_points)

        # === 二级指标：角度 ===

        h_line = get_horizontal_line(landmarks, w, h)

        left_inner = get_point(landmarks, EYE_INNER_CANTHUS_LEFT, w, h)
        right_inner = get_point(landmarks, EYE_INNER_CANTHUS_RIGHT, w, h)
        indicators['eye_horizontal_angle'] = get_horizontal_angle(h_line, left_inner, right_inner)

        indicators['mouth_horizontal_angle'] = get_horizontal_angle(h_line, mouth_left, mouth_right)

        indicators['eyebrow_horizontal_angle'] = get_horizontal_angle(h_line, left_eyebrow_center, right_eyebrow_center)

        return indicators