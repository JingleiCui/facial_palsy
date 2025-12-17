"""
Smile (微笑) 动作
=================

评估面神经颊支功能

关键帧: 嘴角拉伸最大的帧 (嘴宽最大)

核心指标:
- 嘴角高度差: 患侧嘴角下垂
- 嘴角上提量: 相对静息帧的变化
- 鼻唇沟变化: 微笑时鼻唇沟会加深

联动检测:
- 微笑时眼睛可能眯起 (正常联动)
- 面瘫恢复期可能出现口-眼联动
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

class SmileAction(BaseAction):
    """微笑动作 (Sunnybrook随意运动B3)"""

    ACTION_NAME = "Smile"
    ACTION_NAME_CN = "微笑"

    def find_peak_frame(
        self,
        landmarks_seq: List,
        w: int, h: int,
        **kwargs
    ) -> int:
        """找嘴宽最大帧"""
        max_width = -1.0
        max_idx = 0

        for i, lm in enumerate(landmarks_seq):
            if lm is None:
                continue

            l_corner = pt2d(lm[LM.MOUTH_L], w, h)
            r_corner = pt2d(lm[LM.MOUTH_R], w, h)
            mouth_width = dist(l_corner, r_corner)

            if mouth_width > max_width:
                max_width = mouth_width
                max_idx = i

        return max_idx

    def extract_indicators(
        self,
        landmarks,
        w: int, h: int,
        neutral_baseline: Optional[NeutralBaseline] = None
    ) -> Dict[str, float]:
        """提取微笑指标"""
        icd = compute_icd(landmarks, w, h)

        # 口角测量
        oral = measure_oral(landmarks, w, h, icd)

        # 鼻唇沟测量
        nlf = measure_nlf(landmarks, w, h, icd)

        # 基准值
        baseline_mouth_width = neutral_baseline.mouth_width if neutral_baseline else None
        baseline_mouth_height = neutral_baseline.mouth_height if neutral_baseline else None
        baseline_l_nlf = neutral_baseline.left_nlf_length if neutral_baseline else None
        baseline_r_nlf = neutral_baseline.right_nlf_length if neutral_baseline else None

        baseline_l_area = neutral_baseline.left_eye_area if neutral_baseline else None
        baseline_r_area = neutral_baseline.right_eye_area if neutral_baseline else None
        baseline_l_palp = neutral_baseline.left_palpebral_length if neutral_baseline else None
        baseline_r_palp = neutral_baseline.right_palpebral_length if neutral_baseline else None

        # 嘴角变化量
        if baseline_mouth_width is not None and baseline_mouth_width > 1e-6:
            width_change = (oral.mouth_width - baseline_mouth_width) / icd
            width_change_ratio = oral.mouth_width / baseline_mouth_width
        else:
            width_change = 0.0
            width_change_ratio = 1.0

        # 嘴角上提量 (Y坐标减小=上提)
        # 需要计算相对于静息帧的变化，这里简化为口角角度
        l_lift = oral.left_angle  # 正=上提
        r_lift = oral.right_angle

        # 功能百分比 (弱侧/强侧的抬起程度)
        # 对于微笑: 正角度表示嘴角上提
        min_lift = min(l_lift, r_lift)
        max_lift = max(l_lift, r_lift)
        if max_lift > 0 and min_lift > 0:
            function_pct = min_lift / max_lift
        elif max_lift > 0:
            # 有一侧上提，另一侧没有或下垂
            function_pct = 0.0
        else:
            function_pct = 1.0  # 两侧都没上提

        # 鼻唇沟变化
        if baseline_l_nlf is not None and baseline_l_nlf > 1e-6:
            l_nlf_change = (nlf.left_length - baseline_l_nlf) / icd
        else:
            l_nlf_change = 0.0

        if baseline_r_nlf is not None and baseline_r_nlf > 1e-6:
            r_nlf_change = (nlf.right_length - baseline_r_nlf) / icd
        else:
            r_nlf_change = 0.0

        # 眼部测量 (联动检测)
        eyes = measure_eyes(
            landmarks, w, h,
            baseline_l_area, baseline_r_area,
            baseline_l_palp, baseline_r_palp
        )

        # 眼睛眯起检测 (睁眼度 < 0.9 视为眯眼)
        eye_squint = eyes.left.openness < 0.9 or eyes.right.openness < 0.9

        return {
            # 嘴部
            'mouth_width': oral.mouth_width,
            'mouth_height': oral.mouth_height,
            'mouth_width_norm': oral.mouth_width / icd if icd > 0 else 0,
            'mouth_height_norm': oral.mouth_height / icd if icd > 0 else 0,
            'width_change': width_change,
            'width_change_ratio': width_change_ratio,

            # 口角
            'left_oral_angle': oral.left_angle,
            'right_oral_angle': oral.right_angle,
            'oral_height_diff': oral.height_diff,
            'oral_angle_diff': oral.left_angle - oral.right_angle,

            # 嘴角对称性
            'left_corner_lift': l_lift,
            'right_corner_lift': r_lift,

            # 鼻唇沟
            'left_nlf_length_norm': nlf.left_length_norm,
            'right_nlf_length_norm': nlf.right_length_norm,
            'nlf_length_ratio': nlf.length_ratio,
            'left_nlf_change': l_nlf_change,
            'right_nlf_change': r_nlf_change,

            # 功能评估
            'function_pct': function_pct,

            # 联动检测
            'left_eye_openness': eyes.left.openness,
            'right_eye_openness': eyes.right.openness,
            'eye_squint': float(eye_squint),

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

        # 计算嘴宽序列
        mouth_widths = np.zeros(n)

        for i, lm in enumerate(landmarks_seq):
            if lm is None:
                continue
            l_corner = pt2d(lm[LM.MOUTH_L], w, h)
            r_corner = pt2d(lm[LM.MOUTH_R], w, h)
            mouth_widths[i] = dist(l_corner, r_corner)

        # 归一化
        icd_avg = np.mean([compute_icd(lm, w, h) for lm in landmarks_seq if lm is not None])
        mouth_widths_norm = mouth_widths / icd_avg if icd_avg > 0 else mouth_widths

        # 运动范围
        motion_range = np.max(mouth_widths_norm) - np.min(mouth_widths_norm)

        # 速度
        dt = 1.0 / fps
        velocity = np.abs(np.diff(mouth_widths_norm)) / dt
        mean_velocity = np.mean(velocity) if len(velocity) > 0 else 0
        max_velocity = np.max(velocity) if len(velocity) > 0 else 0

        # 平滑度
        smoothness = 1.0 / (1.0 + np.std(velocity)) if len(velocity) > 0 else 1.0

        return {
            'mouth_motion_range': motion_range,
            'mouth_mean_velocity': mean_velocity,
            'mouth_max_velocity': max_velocity,
            'mouth_smoothness': smoothness,
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
        n = len(landmarks_seq)

        # 嘴宽曲线
        mouth_width_curve = np.zeros(n)
        for i, lm in enumerate(landmarks_seq):
            if lm is None:
                continue
            l_corner = pt2d(lm[LM.MOUTH_L], w, h)
            r_corner = pt2d(lm[LM.MOUTH_R], w, h)
            mouth_width_curve[i] = dist(l_corner, r_corner)

        # 睁眼度曲线 (联动检测)
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
            'peak_reason': 'max_mouth_width',
            'mouth_width_curve': mouth_width_curve,
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

        # 绘制嘴角
        l_corner = tuple(map(int, pt2d(landmarks[LM.MOUTH_L], w, h)))
        r_corner = tuple(map(int, pt2d(landmarks[LM.MOUTH_R], w, h)))

        # 根据高度差选择颜色
        height_diff = indicators.get('oral_height_diff', 0)
        if height_diff > 0.02:
            l_color = (0, 0, 255)  # 红=下垂
            r_color = (0, 255, 0)
        elif height_diff < -0.02:
            l_color = (0, 255, 0)
            r_color = (0, 0, 255)
        else:
            l_color = (0, 255, 0)
            r_color = (0, 255, 0)

        cv2.circle(img, l_corner, 8, l_color, -1)
        cv2.circle(img, r_corner, 8, r_color, -1)
        cv2.line(img, l_corner, r_corner, (255, 255, 255), 1)

        # 绘制口裂水平线
        midline_y = int((l_corner[1] + r_corner[1]) / 2)
        cv2.line(img, (l_corner[0] - 20, midline_y), (r_corner[0] + 20, midline_y),
                (128, 128, 128), 1)

        # 绘制鼻唇沟
        l_ala = tuple(map(int, pt2d(landmarks[LM.NOSE_ALA_L], w, h)))
        r_ala = tuple(map(int, pt2d(landmarks[LM.NOSE_ALA_R], w, h)))
        cv2.line(img, l_ala, l_corner, (255, 0, 0), 2)
        cv2.line(img, r_ala, r_corner, (0, 165, 255), 2)

        # 文字标注
        y = 30
        cv2.putText(img, f"{self.ACTION_NAME}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y += 35

        # 口角角度
        l_angle = indicators.get('left_oral_angle', 0)
        r_angle = indicators.get('right_oral_angle', 0)
        cv2.putText(img, f"L Angle: {l_angle:.1f}°", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, l_color, 2)
        y += 25
        cv2.putText(img, f"R Angle: {r_angle:.1f}°", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, r_color, 2)
        y += 25

        # 高度差
        cv2.putText(img, f"Height Diff: {height_diff:.3f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 25

        # 功能百分比
        func_pct = indicators.get('function_pct', 1.0)
        cv2.putText(img, f"Function: {func_pct:.1%}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 25

        # 联动检测
        squint = indicators.get('eye_squint', 0)
        squint_text = "Yes" if squint else "No"
        cv2.putText(img, f"Eye Squint: {squint_text}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return img