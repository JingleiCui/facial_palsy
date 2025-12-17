"""
CloseEyeHardly - 用力闭眼 (HB分级关键)

关键帧: 最小EAR帧 (眼睛闭得最紧的帧)

核心指标:
- 睁眼度 (openness): 相对静息帧的眼睛开合程度
- 闭拢度 (closure): 1 - 睁眼度
- 完全闭眼 (complete_closure): 睁眼度 <= 6.25%

HB分级关键: CloseEyeHardly能否完全闭眼决定Grade III/IV分界
"""

import numpy as np
from typing import Dict, List, Optional, Any
import cv2

from .base_action import BaseAction, ActionResult, NeutralBaseline
from ..core.geometry_utils import (
    LM, compute_icd, compute_ear,
    measure_eyes, find_min_ear_frame, compute_openness_curve,
    pts2d, pt2d
)
from .close_eye_softly import CloseEyeSoftlyAction

class CloseEyeHardlyAction(CloseEyeSoftlyAction):
    """
    用力闭眼动作 (HB分级关键)

    继承自CloseEyeSoftly，关键帧选择和指标计算相同
    区别在于临床意义:
    - CloseEyeHardly无法完全闭眼 → HB Grade IV
    - CloseEyeHardly可以完全闭眼 → HB Grade III或更好
    """

    ACTION_NAME = "CloseEyeHardly"
    ACTION_NAME_CN = "用力闭眼"

    def _build_interpretability(
            self,
            landmarks_seq: List,
            w: int, h: int,
            peak_idx: int,
            indicators: Dict,
            neutral_baseline: Optional[NeutralBaseline]
    ) -> Dict[str, Any]:
        """构建可解释性数据 - 增加HB分级相关信息"""
        result = super()._build_interpretability(
            landmarks_seq, w, h, peak_idx, indicators, neutral_baseline
        )

        # HB分级关键: 用力闭眼能否完全闭合
        l_complete = indicators.get('left_complete_closure', 0)
        r_complete = indicators.get('right_complete_closure', 0)

        # 判断闭眼完全性
        both_complete = l_complete and r_complete

        result['hb_critical'] = {
            'left_complete': l_complete,
            'right_complete': r_complete,
            'both_complete': both_complete,
            'hb_indicator': 'Grade III or better' if both_complete else 'Grade IV or worse',
        }

        return result