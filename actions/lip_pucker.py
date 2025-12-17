"""
LipPucker - 撅嘴 (Sunnybrook B5)
"""

import numpy as np
from typing import Dict, List, Optional, Any
import cv2

# ========== 第1步：导入常量 ==========
# 从constants模块导入所有需要的常量
from ..core.constants import LM, Colors, Thresholds

# ========== 第2步：导入几何工具 ==========
# 根据动作需要选择性导入
from ..core.geometry_utils import (
    compute_icd,           # ICD计算
    compute_ear,           # EAR计算
    measure_eyes,          # 眼部测量
    measure_oral,          # 口角测量
    measure_nlf,           # 鼻唇沟测量
    measure_brow,          # 眉毛测量
    find_max_ear_frame,    # 找最大EAR帧
    find_min_ear_frame,    # 找最小EAR帧
    compute_openness_curve,  # 计算睁眼度曲线
    pts2d,                 # 批量获取2D坐标
    pt2d,                  # 单点2D坐标
    dist,                  # 距离计算
)

# ========== 第3步：导入基类 ==========
from .base_action import BaseAction, ActionResult, NeutralBaseline

class LipPuckerAction(BaseAction):
    """撅嘴动作 (Sunnybrook随意运动B5)"""

    ACTION_NAME = "LipPucker"
    ACTION_NAME_CN = "撅嘴"

    def find_peak_frame(
            self,
            landmarks_seq: List,
            w: int, h: int,
            **kwargs
    ) -> int:
        """找嘴宽最小帧 (撅嘴时嘴收缩)"""
        min_width = float('inf')
        min_idx = 0

        for i, lm in enumerate(landmarks_seq):
            if lm is None:
                continue

            l_corner = pt2d(lm[LM.MOUTH_L], w, h)
            r_corner = pt2d(lm[LM.MOUTH_R], w, h)
            mouth_width = dist(l_corner, r_corner)

            if mouth_width < min_width:
                min_width = mouth_width
                min_idx = i

        return min_idx

    def extract_indicators(
            self,
            landmarks,
            w: int, h: int,
            neutral_baseline: Optional[NeutralBaseline] = None
    ) -> Dict[str, float]:
        """提取撅嘴指标"""
        icd = compute_icd(landmarks, w, h)
        oral = measure_oral(landmarks, w, h, icd)

        # 嘴宽变化 (撅嘴时应该变窄)
        baseline_mouth_width = neutral_baseline.mouth_width if neutral_baseline else None
        if baseline_mouth_width is not None and baseline_mouth_width > 1e-6:
            width_change = (baseline_mouth_width - oral.mouth_width) / icd  # 正=收缩
            contraction_ratio = 1.0 - (oral.mouth_width / baseline_mouth_width)  # 收缩百分比
        else:
            width_change = 0.0
            contraction_ratio = 0.0

        contraction_ratio = max(0.0, contraction_ratio)

        # 口角对称性
        function_pct = 1.0 - min(1.0, abs(oral.height_diff) * 5)
        function_pct = max(0.0, function_pct)

        return {
            'mouth_width': oral.mouth_width,
            'mouth_height': oral.mouth_height,
            'mouth_width_norm': oral.mouth_width / icd if icd > 0 else 0,
            'mouth_height_norm': oral.mouth_height / icd if icd > 0 else 0,
            'width_change': width_change,
            'contraction_ratio': contraction_ratio,
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
        cv2.circle(img, l_corner, 5, (255, 0, 0), -1)
        cv2.circle(img, r_corner, 5, (0, 165, 255), -1)
        cv2.line(img, l_corner, r_corner, (0, 255, 0), 2)

        # 文字
        y = 30
        cv2.putText(img, f"{self.ACTION_NAME}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y += 35
        cv2.putText(img, f"Mouth Width: {indicators.get('mouth_width_norm', 0):.3f}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 25
        cv2.putText(img, f"Contraction: {indicators.get('contraction_ratio', 0):.1%}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 25
        cv2.putText(img, f"Height Diff: {indicators.get('oral_height_diff', 0):.3f}",
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return img