"""
自然眨眼动作 (SpontaneousEyeBlink)
峰值帧检测逻辑: 检测眼睛开合度最小的帧
"""
import numpy as np
from base_action import BaseAction
from ..core.geometry_utils import *


class SpontaneousEyeBlinkAction(BaseAction):
    """自然眨眼动作"""

    def detect_peak_frame(self, landmarks_seq, w, h):
        """
        检测峰值帧: 眼睛开合度最小的帧

        Args:
            landmarks_seq: landmarks序列
            w: 图像宽度
            h: 图像高度

        Returns:
            int: 峰值帧索引
        """
        valid_indices = [i for i, lm in enumerate(landmarks_seq) if lm is not None]

        if not valid_indices:
            return 0

        # 计算每帧的平均眼睛开合度
        min_opening = float('inf')
        peak_idx = valid_indices[0]

        for idx in valid_indices:
            lm = landmarks_seq[idx]
            left_eye = get_eye_opening(lm, w, h, left=True)
            right_eye = get_eye_opening(lm, w, h, left=False)
            avg_opening = (left_eye + right_eye) / 2

            if avg_opening < min_opening:
                min_opening = avg_opening
                peak_idx = idx

        return peak_idx
