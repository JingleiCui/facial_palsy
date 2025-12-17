"""
CloseEyeSoftly - 轻轻闭眼 (Sunnybrook随意运动)

关键帧: 最小EAR帧 (眼睛闭得最紧的帧)

核心指标:
- 睁眼度 (openness): 相对静息帧的眼睛开合程度
- 闭拢度 (closure): 1 - 睁眼度
- 完全闭眼 (complete_closure): 睁眼度 <= 6.25%
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

class CloseEyeSoftlyAction(BaseAction):
    """轻轻闭眼动作 (Sunnybrook随意运动B2)"""

    ACTION_NAME = "CloseEyeSoftly"
    ACTION_NAME_CN = "轻闭眼"

    def find_peak_frame(
            self,
            landmarks_seq: List,
            w: int, h: int,
            **kwargs
    ) -> int:
        """找最小EAR帧"""
        peak_idx, _, _ = find_min_ear_frame(landmarks_seq, w, h)
        return peak_idx

    def extract_indicators(
            self,
            landmarks,
            w: int, h: int,
            neutral_baseline: Optional[NeutralBaseline] = None
    ) -> Dict[str, float]:
        """提取闭眼指标"""
        icd = compute_icd(landmarks, w, h)

        # 获取基准值
        baseline_l_area = neutral_baseline.left_eye_area if neutral_baseline else None
        baseline_r_area = neutral_baseline.right_eye_area if neutral_baseline else None
        baseline_l_palp = neutral_baseline.left_palpebral_length if neutral_baseline else None
        baseline_r_palp = neutral_baseline.right_palpebral_length if neutral_baseline else None

        # 眼部测量 (使用基准)
        eyes = measure_eyes(
            landmarks, w, h,
            baseline_l_area, baseline_r_area,
            baseline_l_palp, baseline_r_palp
        )

        # 闭眼程度对称性
        closure_diff = abs(eyes.left.closure - eyes.right.closure)

        # 功能百分比 (用于Severity计算)
        # 对于闭眼: 闭拢度越高=功能越好
        # 患侧闭眼不完全，所以患侧closure较低
        if eyes.right.closure > 1e-6:
            closure_ratio = eyes.left.closure / eyes.right.closure
        else:
            closure_ratio = float('inf') if eyes.left.closure > 1e-6 else 1.0

        # 功能百分比 (弱侧/强侧)
        min_closure = min(eyes.left.closure, eyes.right.closure)
        max_closure = max(eyes.left.closure, eyes.right.closure)
        if max_closure > 1e-6:
            function_pct = min_closure / max_closure
        else:
            function_pct = 1.0

        return {
            # 眼部
            'left_eye_openness': eyes.left.openness,
            'right_eye_openness': eyes.right.openness,
            'left_eye_closure': eyes.left.closure,
            'right_eye_closure': eyes.right.closure,
            'left_complete_closure': float(eyes.left.complete_closure),
            'right_complete_closure': float(eyes.right.complete_closure),

            # EAR
            'left_ear': eyes.left.ear,
            'right_ear': eyes.right.ear,

            # 对称性
            'openness_ratio': eyes.openness_ratio,
            'closure_ratio': closure_ratio,
            'closure_diff': closure_diff,

            # 功能评估
            'function_pct': function_pct,

            # ICD
            'icd': icd,
        }

    def extract_dynamic_features(
            self,
            landmarks_seq: List,
            w: int, h: int,
            fps: float = 30.0,
            neutral_baseline: Optional[NeutralBaseline] = None
    ) -> Dict[str, float]:
        """提取动态特征 - 闭眼过程的时序信息"""
        n = len(landmarks_seq)
        if n < 2:
            return {}

        # 计算睁眼度序列
        baseline_l_area = neutral_baseline.left_eye_area if neutral_baseline else None
        baseline_r_area = neutral_baseline.right_eye_area if neutral_baseline else None
        baseline_l_palp = neutral_baseline.left_palpebral_length if neutral_baseline else None
        baseline_r_palp = neutral_baseline.right_palpebral_length if neutral_baseline else None

        l_open, r_open = compute_openness_curve(
            landmarks_seq, w, h,
            baseline_l_area or 1.0, baseline_r_area or 1.0,
            baseline_l_palp or 1.0, baseline_r_palp or 1.0
        )

        # 运动范围 (最大-最小)
        l_range = np.max(l_open) - np.min(l_open)
        r_range = np.max(r_open) - np.min(r_open)

        # 速度 (睁眼度变化率)
        dt = 1.0 / fps
        l_velocity = np.abs(np.diff(l_open)) / dt
        r_velocity = np.abs(np.diff(r_open)) / dt

        l_mean_vel = np.mean(l_velocity) if len(l_velocity) > 0 else 0
        r_mean_vel = np.mean(r_velocity) if len(r_velocity) > 0 else 0
        l_max_vel = np.max(l_velocity) if len(l_velocity) > 0 else 0
        r_max_vel = np.max(r_velocity) if len(r_velocity) > 0 else 0

        # 平滑度 (速度的标准差，越小越平滑)
        l_smoothness = 1.0 / (1.0 + np.std(l_velocity)) if len(l_velocity) > 0 else 1.0
        r_smoothness = 1.0 / (1.0 + np.std(r_velocity)) if len(r_velocity) > 0 else 1.0

        # 运动不对称性
        motion_asymmetry = abs(l_range - r_range) / max(l_range, r_range, 1e-6)

        return {
            'left_motion_range': l_range,
            'right_motion_range': r_range,
            'left_mean_velocity': l_mean_vel,
            'right_mean_velocity': r_mean_vel,
            'left_max_velocity': l_max_vel,
            'right_max_velocity': r_max_vel,
            'left_smoothness': l_smoothness,
            'right_smoothness': r_smoothness,
            'motion_asymmetry': motion_asymmetry,
        }

    def _build_interpretability(
            self,
            landmarks_seq: List,
            w: int, h: int,
            peak_idx: int,
            indicators: Dict,
            neutral_baseline: Optional[NeutralBaseline]
    ) -> Dict[str, Any]:
        """构建可解释性数据"""
        # 计算睁眼度曲线
        baseline_l_area = neutral_baseline.left_eye_area if neutral_baseline else None
        baseline_r_area = neutral_baseline.right_eye_area if neutral_baseline else None
        baseline_l_palp = neutral_baseline.left_palpebral_length if neutral_baseline else None
        baseline_r_palp = neutral_baseline.right_palpebral_length if neutral_baseline else None

        l_open, r_open = compute_openness_curve(
            landmarks_seq, w, h,
            baseline_l_area or 1.0, baseline_r_area or 1.0,
            baseline_l_palp or 1.0, baseline_r_palp or 1.0
        )

        return {
            'peak_frame_idx': peak_idx,
            'total_frames': len(landmarks_seq),
            'peak_reason': 'min_ear',
            'left_openness_curve': l_open,
            'right_openness_curve': r_open,
            'left_complete_closure': indicators.get('left_complete_closure', 0),
            'right_complete_closure': indicators.get('right_complete_closure', 0),
        }

    def visualize_peak_frame(
            self,
            frame: np.ndarray,
            landmarks,
            indicators: Dict,
            w: int, h: int
    ) -> np.ndarray:
        """可视化峰值帧"""
        img = frame.copy()

        # 绘制眼部轮廓
        l_eye_pts = pts2d(landmarks, LM.EYE_CONTOUR_L, w, h).astype(np.int32)
        r_eye_pts = pts2d(landmarks, LM.EYE_CONTOUR_R, w, h).astype(np.int32)

        # 根据闭眼程度选择颜色
        l_color = (0, 255, 0) if indicators.get('left_complete_closure', 0) else (0, 0, 255)
        r_color = (0, 255, 0) if indicators.get('right_complete_closure', 0) else (0, 0, 255)

        cv2.polylines(img, [l_eye_pts], True, l_color, 2)
        cv2.polylines(img, [r_eye_pts], True, r_color, 2)

        # 文字标注
        y = 30
        cv2.putText(img, f"{self.ACTION_NAME}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y += 35

        # 睁眼度
        l_open = indicators.get('left_eye_openness', 0)
        r_open = indicators.get('right_eye_openness', 0)
        cv2.putText(img, f"L Openness: {l_open:.1%}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, l_color, 2)
        y += 25
        cv2.putText(img, f"R Openness: {r_open:.1%}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, r_color, 2)
        y += 25

        # 闭拢度
        l_close = indicators.get('left_eye_closure', 0)
        r_close = indicators.get('right_eye_closure', 0)
        cv2.putText(img, f"L Closure: {l_close:.1%}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, l_color, 2)
        y += 25
        cv2.putText(img, f"R Closure: {r_close:.1%}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, r_color, 2)
        y += 25

        # 完全闭眼状态
        l_complete = "✓" if indicators.get('left_complete_closure', 0) else "✗"
        r_complete = "✓" if indicators.get('right_complete_closure', 0) else "✗"
        cv2.putText(img, f"L Complete: {l_complete}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, l_color, 2)
        y += 25
        cv2.putText(img, f"R Complete: {r_complete}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, r_color, 2)

        return img