"""
ShrugNose - 皱鼻 (Sunnybrook B4)
"""

import numpy as np
from typing import Dict, List, Optional, Any
import cv2

from .base_action import BaseAction, NeutralBaseline
from ..core.geometry_utils import (
    LM, compute_icd, measure_oral, measure_nlf,
    pts2d, pt2d, dist
)


class ShrugNoseAction(BaseAction):
    """皱鼻动作 (Sunnybrook随意运动B4)"""

    ACTION_NAME = "ShrugNose"
    ACTION_NAME_CN = "皱鼻"

    def find_peak_frame(
            self,
            landmarks_seq: List,
            w: int, h: int,
            **kwargs
    ) -> int:
        """找鼻唇距最小帧 (皱鼻时鼻子和上唇接近)"""
        min_dist = float('inf')
        min_idx = 0

        for i, lm in enumerate(landmarks_seq):
            if lm is None:
                continue

            nose_tip = pt2d(lm[LM.NOSE_TIP], w, h)
            lip_top = pt2d(lm[LM.LIP_TOP], w, h)
            nose_lip_dist = dist(nose_tip, lip_top)

            if nose_lip_dist < min_dist:
                min_dist = nose_lip_dist
                min_idx = i

        return min_idx

    def extract_indicators(
            self,
            landmarks,
            w: int, h: int,
            neutral_baseline: Optional[NeutralBaseline] = None
    ) -> Dict[str, float]:
        """提取皱鼻指标"""
        icd = compute_icd(landmarks, w, h)

        # 鼻唇距
        nose_tip = pt2d(landmarks[LM.NOSE_TIP], w, h)
        lip_top = pt2d(landmarks[LM.LIP_TOP], w, h)
        nose_lip_dist = dist(nose_tip, lip_top)
        nose_lip_dist_norm = nose_lip_dist / icd if icd > 0 else 0

        # 鼻唇沟
        nlf = measure_nlf(landmarks, w, h, icd)

        # 上唇上提量
        baseline_mouth_height = neutral_baseline.mouth_height if neutral_baseline else None
        oral = measure_oral(landmarks, w, h, icd)

        if baseline_mouth_height is not None and baseline_mouth_height > 1e-6:
            lip_lift = (baseline_mouth_height - oral.mouth_height) / icd
        else:
            lip_lift = 0.0

        # 功能百分比
        # 皱鼻主要看鼻唇沟的变化
        if nlf.right_length > 1e-6:
            function_pct = min(nlf.left_length, nlf.right_length) / max(nlf.left_length, nlf.right_length)
        else:
            function_pct = 1.0

        return {
            'nose_lip_dist': nose_lip_dist,
            'nose_lip_dist_norm': nose_lip_dist_norm,
            'left_nlf_length_norm': nlf.left_length_norm,
            'right_nlf_length_norm': nlf.right_length_norm,
            'nlf_length_ratio': nlf.length_ratio,
            'lip_lift': lip_lift,
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

        # 绘制鼻唇距线
        nose_tip = tuple(map(int, pt2d(landmarks[LM.NOSE_TIP], w, h)))
        lip_top = tuple(map(int, pt2d(landmarks[LM.LIP_TOP], w, h)))
        cv2.line(img, nose_tip, lip_top, (0, 255, 0), 2)
        cv2.circle(img, nose_tip, 5, (255, 0, 0), -1)
        cv2.circle(img, lip_top, 5, (0, 165, 255), -1)

        # 绘制鼻唇沟
        l_ala = tuple(map(int, pt2d(landmarks[LM.NOSE_ALA_L], w, h)))
        r_ala = tuple(map(int, pt2d(landmarks[LM.NOSE_ALA_R], w, h)))
        l_mouth = tuple(map(int, pt2d(landmarks[LM.MOUTH_L], w, h)))
        r_mouth = tuple(map(int, pt2d(landmarks[LM.MOUTH_R], w, h)))
        cv2.line(img, l_ala, l_mouth, (255, 0, 0), 2)
        cv2.line(img, r_ala, r_mouth, (0, 165, 255), 2)

        # 文字
        y = 30
        cv2.putText(img, f"{self.ACTION_NAME}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y += 35
        cv2.putText(img, f"Nose-Lip: {indicators.get('nose_lip_dist_norm', 0):.3f}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 25
        cv2.putText(img, f"NLF Ratio: {indicators.get('nlf_length_ratio', 1):.3f}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return img