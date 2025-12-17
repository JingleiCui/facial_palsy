"""
ShowTeeth - 露齿
"""

import numpy as np
from typing import Dict, List, Optional, Any
import cv2

from .base_action import BaseAction, NeutralBaseline
from ..core.geometry_utils import (
    LM, compute_icd, measure_oral, measure_nlf,
    pts2d, pt2d, dist
)

class ShowTeethAction(BaseAction):
    """露齿动作"""

    ACTION_NAME = "ShowTeeth"
    ACTION_NAME_CN = "露齿"

    def find_peak_frame(
            self,
            landmarks_seq: List,
            w: int, h: int,
            **kwargs
    ) -> int:
        """找嘴高最大帧 (露齿时嘴张得最大)"""
        max_height = -1.0
        max_idx = 0

        for i, lm in enumerate(landmarks_seq):
            if lm is None:
                continue

            lip_top = pt2d(lm[LM.LIP_TOP], w, h)
            lip_bot = pt2d(lm[LM.LIP_BOT], w, h)
            mouth_height = dist(lip_top, lip_bot)

            if mouth_height > max_height:
                max_height = mouth_height
                max_idx = i

        return max_idx

    def extract_indicators(
            self,
            landmarks,
            w: int, h: int,
            neutral_baseline: Optional[NeutralBaseline] = None
    ) -> Dict[str, float]:
        """提取露齿指标"""
        icd = compute_icd(landmarks, w, h)
        oral = measure_oral(landmarks, w, h, icd)

        # 嘴高变化
        baseline_mouth_height = neutral_baseline.mouth_height if neutral_baseline else None
        if baseline_mouth_height is not None and baseline_mouth_height > 1e-6:
            height_change = (oral.mouth_height - baseline_mouth_height) / icd
            height_change_ratio = oral.mouth_height / baseline_mouth_height
        else:
            height_change = 0.0
            height_change_ratio = 1.0

        # 口角对称性
        function_pct = 1.0 - min(1.0, abs(oral.height_diff) * 5)  # 放大差异
        function_pct = max(0.0, function_pct)

        return {
            'mouth_width': oral.mouth_width,
            'mouth_height': oral.mouth_height,
            'mouth_width_norm': oral.mouth_width / icd if icd > 0 else 0,
            'mouth_height_norm': oral.mouth_height / icd if icd > 0 else 0,
            'height_change': height_change,
            'height_change_ratio': height_change_ratio,
            'oral_height_diff': oral.height_diff,
            'left_oral_angle': oral.left_angle,
            'right_oral_angle': oral.right_angle,
            'function_pct': function_pct,
            'icd': icd,
        }

    def visualize_peak_frame(
            self,
            frame: np.ndarray,
            landmarks,
            indicators: Dict,
            w: int, h: int
    ) -> np.ndarray:
        img = frame.copy()

        # 绘制嘴部
        l_corner = tuple(map(int, pt2d(landmarks[LM.MOUTH_L], w, h)))
        r_corner = tuple(map(int, pt2d(landmarks[LM.MOUTH_R], w, h)))
        lip_top = tuple(map(int, pt2d(landmarks[LM.LIP_TOP], w, h)))
        lip_bot = tuple(map(int, pt2d(landmarks[LM.LIP_BOT], w, h)))

        cv2.circle(img, l_corner, 5, (255, 0, 0), -1)
        cv2.circle(img, r_corner, 5, (0, 165, 255), -1)
        cv2.line(img, lip_top, lip_bot, (0, 255, 0), 2)

        # 文字
        y = 30
        cv2.putText(img, f"{self.ACTION_NAME}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y += 35
        cv2.putText(img, f"Mouth Height: {indicators.get('mouth_height_norm', 0):.3f}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 25
        cv2.putText(img, f"Height Diff: {indicators.get('oral_height_diff', 0):.3f}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return img