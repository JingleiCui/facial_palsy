"""
BlowCheek - 鼓腮
"""

import numpy as np
from typing import Dict, List, Optional, Any
import cv2

from .base_action import BaseAction, NeutralBaseline
from ..core.geometry_utils import (
    LM, compute_icd, measure_oral, measure_nlf,
    pts2d, pt2d, dist
)

class BlowCheekAction(BaseAction):
    """鼓腮动作"""

    ACTION_NAME = "BlowCheek"
    ACTION_NAME_CN = "鼓腮"

    def find_peak_frame(
            self,
            landmarks_seq: List,
            w: int, h: int,
            **kwargs
    ) -> int:
        """找面宽最大帧 (鼓腮时面部膨胀)"""
        max_width = -1.0
        max_idx = 0

        for i, lm in enumerate(landmarks_seq):
            if lm is None:
                continue

            # 面宽 (左右面部轮廓的距离)
            l_face = pts2d(lm, LM.FACE_CONTOUR_L, w, h)
            r_face = pts2d(lm, LM.FACE_CONTOUR_R, w, h)
            face_width = np.max(r_face[:, 0]) - np.min(l_face[:, 0])

            if face_width > max_width:
                max_width = face_width
                max_idx = i

        return max_idx

    def extract_indicators(
            self,
            landmarks,
            w: int, h: int,
            neutral_baseline: Optional[NeutralBaseline] = None
    ) -> Dict[str, float]:
        """提取鼓腮指标"""
        icd = compute_icd(landmarks, w, h)

        # 面宽
        l_face = pts2d(landmarks, LM.FACE_CONTOUR_L, w, h)
        r_face = pts2d(landmarks, LM.FACE_CONTOUR_R, w, h)
        face_width = np.max(r_face[:, 0]) - np.min(l_face[:, 0])
        face_width_norm = face_width / icd if icd > 0 else 0

        # 面颊膨胀 (左右面颊点到中线的距离)
        l_cheek = pts2d(landmarks, LM.CHEEK_L, w, h)
        r_cheek = pts2d(landmarks, LM.CHEEK_R, w, h)

        # 中线X坐标
        midline_x = (np.mean(l_cheek[:, 0]) + np.mean(r_cheek[:, 0])) / 2

        l_expansion = midline_x - np.min(l_cheek[:, 0])
        r_expansion = np.max(r_cheek[:, 0]) - midline_x

        l_expansion_norm = l_expansion / icd if icd > 0 else 0
        r_expansion_norm = r_expansion / icd if icd > 0 else 0

        # 膨胀比
        if r_expansion > 1e-6:
            expansion_ratio = l_expansion / r_expansion
        else:
            expansion_ratio = 1.0

        # 嘴闭合程度 (鼓腮时嘴应该闭紧)
        oral = measure_oral(landmarks, w, h, icd)
        mouth_seal = 1.0 - min(1.0, oral.mouth_height / icd * 10)  # 越小越紧
        mouth_seal = max(0.0, mouth_seal)

        # 功能百分比
        min_exp = min(l_expansion, r_expansion)
        max_exp = max(l_expansion, r_expansion)
        function_pct = min_exp / max_exp if max_exp > 1e-6 else 1.0

        return {
            'face_width': face_width,
            'face_width_norm': face_width_norm,
            'left_expansion': l_expansion,
            'right_expansion': r_expansion,
            'left_expansion_norm': l_expansion_norm,
            'right_expansion_norm': r_expansion_norm,
            'expansion_ratio': expansion_ratio,
            'mouth_seal': mouth_seal,
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

        # 绘制面部轮廓
        l_face = pts2d(landmarks, LM.FACE_CONTOUR_L, w, h).astype(np.int32)
        r_face = pts2d(landmarks, LM.FACE_CONTOUR_R, w, h).astype(np.int32)
        cv2.polylines(img, [l_face], False, (255, 0, 0), 2)
        cv2.polylines(img, [r_face], False, (0, 165, 255), 2)

        # 绘制面颊点
        l_cheek = pts2d(landmarks, LM.CHEEK_L, w, h).astype(np.int32)
        r_cheek = pts2d(landmarks, LM.CHEEK_R, w, h).astype(np.int32)
        for pt in l_cheek:
            cv2.circle(img, tuple(pt), 3, (255, 0, 0), -1)
        for pt in r_cheek:
            cv2.circle(img, tuple(pt), 3, (0, 165, 255), -1)

        # 文字
        y = 30
        cv2.putText(img, f"{self.ACTION_NAME}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y += 35
        cv2.putText(img, f"Face Width: {indicators.get('face_width_norm', 0):.3f}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 25
        cv2.putText(img, f"Expansion Ratio: {indicators.get('expansion_ratio', 1):.3f}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 25
        cv2.putText(img, f"Mouth Seal: {indicators.get('mouth_seal', 0):.1%}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return img