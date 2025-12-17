"""
RaiseEyebrow (抬眉) 动作
========================

评估面神经颞支功能

关键帧: 眉眼距最大的帧 (眉毛抬得最高)

核心指标:
- 眉眼距变化量 (相对静息帧)
- 眉毛抬起的左右不对称性
- 联动检测: 抬眉时眼睛可能睁大

注意: 这是唯一可能导致睁眼度>1的动作
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

class RaiseEyebrowAction(BaseAction):
    """抬眉动作 (Sunnybrook随意运动B1)"""

    ACTION_NAME = "RaiseEyebrow"
    ACTION_NAME_CN = "抬眉"

    def find_peak_frame(
        self,
        landmarks_seq: List,
        w: int, h: int,
        **kwargs
    ) -> int:
        """找眉眼距最大帧"""
        max_dist = -1.0
        max_idx = 0

        for i, lm in enumerate(landmarks_seq):
            if lm is None:
                continue

            icd = compute_icd(lm, w, h)
            brow = measure_brow(lm, w, h, icd)

            # 使用两侧眉眼距的平均值
            avg_height = (brow.left_height + brow.right_height) / 2

            if avg_height > max_dist:
                max_dist = avg_height
                max_idx = i

        return max_idx

    def extract_indicators(
        self,
        landmarks,
        w: int, h: int,
        neutral_baseline: Optional[NeutralBaseline] = None
    ) -> Dict[str, float]:
        """提取抬眉指标"""
        icd = compute_icd(landmarks, w, h)

        # 获取基准
        baseline_l_brow = neutral_baseline.left_brow_height if neutral_baseline else None
        baseline_r_brow = neutral_baseline.right_brow_height if neutral_baseline else None
        baseline_l_area = neutral_baseline.left_eye_area if neutral_baseline else None
        baseline_r_area = neutral_baseline.right_eye_area if neutral_baseline else None
        baseline_l_palp = neutral_baseline.left_palpebral_length if neutral_baseline else None
        baseline_r_palp = neutral_baseline.right_palpebral_length if neutral_baseline else None

        # 眉毛测量
        brow = measure_brow(landmarks, w, h, icd, baseline_l_brow, baseline_r_brow)

        # 眼部测量 (检测联动: 抬眉时可能睁大眼睛)
        eyes = measure_eyes(
            landmarks, w, h,
            baseline_l_area, baseline_r_area,
            baseline_l_palp, baseline_r_palp
        )

        # 眉毛抬起量 (归一化)
        l_lift_norm = brow.left_lift  # 已经除以ICD
        r_lift_norm = brow.right_lift

        # 抬起量比值 (功能百分比)
        if r_lift_norm > 1e-6:
            lift_ratio = l_lift_norm / r_lift_norm
        elif l_lift_norm > 1e-6:
            lift_ratio = float('inf')
        else:
            lift_ratio = 1.0

        # 功能百分比 (弱侧/强侧)
        min_lift = min(l_lift_norm, r_lift_norm)
        max_lift = max(l_lift_norm, r_lift_norm)
        if max_lift > 1e-6:
            function_pct = min_lift / max_lift
        else:
            function_pct = 1.0

        # 抬眉不对称度
        lift_asymmetry = abs(l_lift_norm - r_lift_norm)

        return {
            # 眉毛
            'left_brow_height': brow.left_height,
            'right_brow_height': brow.right_height,
            'left_brow_height_norm': brow.left_height_norm,
            'right_brow_height_norm': brow.right_height_norm,
            'brow_height_ratio': brow.height_ratio,

            # 抬起量 (相对静息帧)
            'left_brow_lift': l_lift_norm,
            'right_brow_lift': r_lift_norm,
            'lift_ratio': lift_ratio,
            'lift_asymmetry': lift_asymmetry,

            # 功能评估
            'function_pct': function_pct,

            # 联动检测: 眼睛睁大程度
            'left_eye_openness': eyes.left.openness,
            'right_eye_openness': eyes.right.openness,
            'eye_synkinesis': max(eyes.left.openness, eyes.right.openness) > 1.1,  # 超过110%视为联动

            'icd': icd,
        }

    def extract_dynamic_features(
        self,
        landmarks_seq: List,
        w: int, h: int,
        fps: float = 30.0,
        neutral_baseline: Optional[NeutralBaseline] = None
    ) -> Dict[str, float]:
        """提取动态特征"""
        n = len(landmarks_seq)
        if n < 2:
            return {}

        # 计算眉眼距序列
        l_heights = np.zeros(n)
        r_heights = np.zeros(n)

        for i, lm in enumerate(landmarks_seq):
            if lm is None:
                continue
            icd = compute_icd(lm, w, h)
            brow = measure_brow(lm, w, h, icd)
            l_heights[i] = brow.left_height_norm
            r_heights[i] = brow.right_height_norm

        # 运动范围
        l_range = np.max(l_heights) - np.min(l_heights)
        r_range = np.max(r_heights) - np.min(r_heights)

        # 速度
        dt = 1.0 / fps
        l_velocity = np.abs(np.diff(l_heights)) / dt
        r_velocity = np.abs(np.diff(r_heights)) / dt

        l_mean_vel = np.mean(l_velocity) if len(l_velocity) > 0 else 0
        r_mean_vel = np.mean(r_velocity) if len(r_velocity) > 0 else 0

        # 平滑度
        l_smoothness = 1.0 / (1.0 + np.std(l_velocity)) if len(l_velocity) > 0 else 1.0
        r_smoothness = 1.0 / (1.0 + np.std(r_velocity)) if len(r_velocity) > 0 else 1.0

        # 运动不对称性
        motion_asymmetry = abs(l_range - r_range) / max(l_range, r_range, 1e-6)

        return {
            'left_motion_range': l_range,
            'right_motion_range': r_range,
            'left_mean_velocity': l_mean_vel,
            'right_mean_velocity': r_mean_vel,
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
        # 计算眉眼距曲线
        n = len(landmarks_seq)
        l_brow_curve = np.zeros(n)
        r_brow_curve = np.zeros(n)

        for i, lm in enumerate(landmarks_seq):
            if lm is None:
                continue
            icd = compute_icd(lm, w, h)
            brow = measure_brow(lm, w, h, icd)
            l_brow_curve[i] = brow.left_height_norm
            r_brow_curve[i] = brow.right_height_norm

        # 睁眼度曲线 (检测联动)
        baseline_l_area = neutral_baseline.left_eye_area if neutral_baseline else 1.0
        baseline_r_area = neutral_baseline.right_eye_area if neutral_baseline else 1.0
        baseline_l_palp = neutral_baseline.left_palpebral_length if neutral_baseline else 1.0
        baseline_r_palp = neutral_baseline.right_palpebral_length if neutral_baseline else 1.0

        l_open, r_open = compute_openness_curve(
            landmarks_seq, w, h,
            baseline_l_area, baseline_r_area,
            baseline_l_palp, baseline_r_palp
        )

        return {
            'peak_frame_idx': peak_idx,
            'total_frames': n,
            'peak_reason': 'max_brow_height',
            'left_brow_curve': l_brow_curve,
            'right_brow_curve': r_brow_curve,
            'left_openness_curve': l_open,
            'right_openness_curve': r_open,
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

        # 绘制眉毛
        l_brow_pts = pts2d(landmarks, LM.BROW_L, w, h).astype(np.int32)
        r_brow_pts = pts2d(landmarks, LM.BROW_R, w, h).astype(np.int32)
        cv2.polylines(img, [l_brow_pts], False, (255, 0, 0), 2)
        cv2.polylines(img, [r_brow_pts], False, (0, 165, 255), 2)

        # 绘制眉眼连线
        l_brow_center = tuple(map(int, pt2d(landmarks[LM.BROW_CENTER_L], w, h)))
        r_brow_center = tuple(map(int, pt2d(landmarks[LM.BROW_CENTER_R], w, h)))
        l_eye_inner = tuple(map(int, pt2d(landmarks[LM.EYE_INNER_L], w, h)))
        r_eye_inner = tuple(map(int, pt2d(landmarks[LM.EYE_INNER_R], w, h)))

        cv2.line(img, l_brow_center, l_eye_inner, (255, 0, 0), 1)
        cv2.line(img, r_brow_center, r_eye_inner, (0, 165, 255), 1)

        # 绘制眼部轮廓
        l_eye_pts = pts2d(landmarks, LM.EYE_CONTOUR_L, w, h).astype(np.int32)
        r_eye_pts = pts2d(landmarks, LM.EYE_CONTOUR_R, w, h).astype(np.int32)
        cv2.polylines(img, [l_eye_pts], True, (255, 0, 0), 1)
        cv2.polylines(img, [r_eye_pts], True, (0, 165, 255), 1)

        # 文字标注
        y = 30
        cv2.putText(img, f"{self.ACTION_NAME}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y += 35

        # 眉毛抬起量
        l_lift = indicators.get('left_brow_lift', 0)
        r_lift = indicators.get('right_brow_lift', 0)
        cv2.putText(img, f"L Lift: {l_lift:.3f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        y += 25
        cv2.putText(img, f"R Lift: {r_lift:.3f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        y += 25

        # 功能百分比
        func_pct = indicators.get('function_pct', 1.0)
        cv2.putText(img, f"Function: {func_pct:.1%}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 25

        # 联动检测
        synk = indicators.get('eye_synkinesis', False)
        synk_text = "Yes" if synk else "No"
        synk_color = (0, 255, 255) if synk else (0, 255, 0)
        cv2.putText(img, f"Synkinesis: {synk_text}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, synk_color, 2)
        y += 25

        # 眼睛睁大程度
        l_open = indicators.get('left_eye_openness', 1.0)
        r_open = indicators.get('right_eye_openness', 1.0)
        cv2.putText(img, f"L Eye Open: {l_open:.1%}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        y += 20
        cv2.putText(img, f"R Eye Open: {r_open:.1%}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        return img