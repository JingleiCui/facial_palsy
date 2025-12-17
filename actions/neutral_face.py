"""
NeutralFace (静息帧) 动作
=========================

静息帧是所有其他动作的基准，用于:
1. 计算基准眼裂面积 (睁眼度计算的分母)
2. 计算基准眉眼距 (眉毛抬起的参考)
3. Sunnybrook表A静态对称性评估
4. 确定患侧

关键帧选择: 使用最大EAR帧 (眼睛睁得最大的帧)
"""

import numpy as np
from typing import Dict, List, Optional, Any
import cv2

# 从constants导入常量
from ..core.constants import LM, Colors, Thresholds

# 从geometry_utils导入测量函数
from ..core.geometry_utils import (
    compute_icd, compute_ear,
    measure_eyes, measure_oral, measure_nlf, measure_brow,
    find_max_ear_frame, pts2d, pt2d
)

# 从base_action导入基类
from .base_action import BaseAction, ActionResult, NeutralBaseline


class NeutralFaceAction(BaseAction):
    """静息帧动作"""

    ACTION_NAME = "NeutralFace"
    ACTION_NAME_CN = "静息"

    def find_peak_frame(
        self,
        landmarks_seq: List,
        w: int, h: int,
        **kwargs
    ) -> int:
        """
        找最大EAR帧作为关键帧

        原因: 静息时应该是自然睁眼状态，EAR最大代表眼睛睁得最开
        """
        peak_idx, _, _ = find_max_ear_frame(landmarks_seq, w, h)
        return peak_idx

    def extract_indicators(
        self,
        landmarks,
        w: int, h: int,
        neutral_baseline: Optional[NeutralBaseline] = None
    ) -> Dict[str, float]:
        """
        提取静息帧指标

        静息帧本身就是基准，所以不需要neutral_baseline

        返回指标:
        - 眼部: 眼裂面积、眼睑裂长度、EAR、面积比、睁眼度
        - 口角: 嘴宽、嘴高、口角角度、高度差
        - 鼻唇沟: 长度、深度代理
        - 眉毛: 眉眼距
        - 对称性评分
        """
        icd = compute_icd(landmarks, w, h)

        # 眼部测量 (无基准)
        eyes = measure_eyes(landmarks, w, h)

        # 口角测量
        oral = measure_oral(landmarks, w, h, icd)

        # 鼻唇沟测量
        nlf = measure_nlf(landmarks, w, h, icd)

        # 眉毛测量 (无基准)
        brow = measure_brow(landmarks, w, h, icd)

        # 面部对称性评分 (综合各部位不对称度)
        eye_asymmetry = eyes.asymmetry
        oral_asymmetry = abs(oral.height_diff)
        nlf_asymmetry = abs(1.0 - nlf.length_ratio)
        brow_asymmetry = abs(1.0 - brow.height_ratio)

        # 综合对称性评分 (0-1, 1=完全对称)
        face_symmetry_score = 1.0 - (
            0.3 * min(1.0, eye_asymmetry) +
            0.3 * min(1.0, oral_asymmetry * 5) +  # 放大口角差异
            0.2 * min(1.0, nlf_asymmetry) +
            0.2 * min(1.0, brow_asymmetry)
        )
        face_symmetry_score = max(0.0, face_symmetry_score)

        return {
            # ICD (基准)
            'icd': icd,

            # 左眼
            'left_eye_area': eyes.left.area_raw,
            'left_eye_area_norm': eyes.left.area_norm,
            'left_palpebral_length': eyes.left.palpebral_length,
            'left_palpebral_height': eyes.left.palpebral_height,
            'left_eye_openness': eyes.left.openness,
            'left_eye_closure': eyes.left.closure,
            'left_eye_ear': eyes.left.ear,

            # 右眼
            'right_eye_area': eyes.right.area_raw,
            'right_eye_area_norm': eyes.right.area_norm,
            'right_palpebral_length': eyes.right.palpebral_length,
            'right_palpebral_height': eyes.right.palpebral_height,
            'right_eye_openness': eyes.right.openness,
            'right_eye_closure': eyes.right.closure,
            'right_eye_ear': eyes.right.ear,

            # 眼部对称性
            'eye_area_ratio': eyes.area_ratio,
            'eye_asymmetry': eyes.asymmetry,

            # 口角
            'mouth_width': oral.mouth_width,
            'mouth_height': oral.mouth_height,
            'mouth_width_norm': oral.mouth_width / icd if icd > 0 else 0,
            'mouth_height_norm': oral.mouth_height / icd if icd > 0 else 0,
            'left_oral_angle': oral.left_angle,
            'right_oral_angle': oral.right_angle,
            'oral_height_diff': oral.height_diff,
            'oral_angle_diff': oral.left_angle - oral.right_angle,

            # 鼻唇沟
            'left_nlf_length': nlf.left_length,
            'right_nlf_length': nlf.right_length,
            'left_nlf_length_norm': nlf.left_length_norm,
            'right_nlf_length_norm': nlf.right_length_norm,
            'nlf_length_ratio': nlf.length_ratio,
            'left_nlf_depth_proxy': nlf.left_depth_proxy,
            'right_nlf_depth_proxy': nlf.right_depth_proxy,
            'nlf_depth_ratio': nlf.depth_ratio,

            # 眉毛
            'left_brow_height': brow.left_height,
            'right_brow_height': brow.right_height,
            'left_brow_height_norm': brow.left_height_norm,
            'right_brow_height_norm': brow.right_height_norm,
            'brow_height_ratio': brow.height_ratio,

            # 综合对称性
            'face_symmetry_score': face_symmetry_score,
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
        # EAR曲线
        left_ear_curve = np.zeros(len(landmarks_seq))
        right_ear_curve = np.zeros(len(landmarks_seq))

        for i, lm in enumerate(landmarks_seq):
            if lm is None:
                continue
            left_ear_curve[i] = compute_ear(lm, w, h, True)
            right_ear_curve[i] = compute_ear(lm, w, h, False)

        return {
            'peak_frame_idx': peak_idx,
            'total_frames': len(landmarks_seq),
            'left_ear_curve': left_ear_curve,
            'right_ear_curve': right_ear_curve,
            'peak_reason': 'max_ear',

            # Sunnybrook表A静态评分 (0=正常)
            'sunnybrook_static': {
                'eye_score': 1 if indicators['eye_asymmetry'] > Thresholds.EYE_ASYMMETRY_THRESHOLD else 0,
                'cheek_score': self._compute_cheek_score(indicators),
                'mouth_score': 1 if abs(indicators['oral_height_diff']) > Thresholds.ORAL_HEIGHT_DIFF_THRESHOLD else 0,
            },
        }

    def _compute_cheek_score(self, indicators: Dict) -> int:
        """计算颊部(鼻唇沟)评分"""
        ratio = indicators.get('nlf_length_ratio', 1.0)
        if 0.90 <= ratio <= 1.10:
            return 0  # 正常
        elif ratio < 0.75 or ratio > 1.33:
            return 2  # 消失/严重
        else:
            return 1  # 不明显/轻度

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
        cv2.polylines(img, [l_eye_pts], True, Colors.LEFT_COLOR, 2)
        cv2.polylines(img, [r_eye_pts], True, Colors.RIGHT_COLOR, 2)

        # 绘制鼻唇沟线
        l_ala = tuple(map(int, pt2d(landmarks[LM.NOSE_ALA_L], w, h)))
        r_ala = tuple(map(int, pt2d(landmarks[LM.NOSE_ALA_R], w, h)))
        l_mouth = tuple(map(int, pt2d(landmarks[LM.MOUTH_L], w, h)))
        r_mouth = tuple(map(int, pt2d(landmarks[LM.MOUTH_R], w, h)))
        cv2.line(img, l_ala, l_mouth, Colors.LEFT_COLOR, 2)
        cv2.line(img, r_ala, r_mouth, Colors.RIGHT_COLOR, 2)

        # 绘制嘴角
        cv2.circle(img, l_mouth, 5, Colors.LEFT_COLOR, -1)
        cv2.circle(img, r_mouth, 5, Colors.RIGHT_COLOR, -1)

        # 绘制ICD线
        l_inner = tuple(map(int, pt2d(landmarks[LM.EYE_INNER_L], w, h)))
        r_inner = tuple(map(int, pt2d(landmarks[LM.EYE_INNER_R], w, h)))
        cv2.line(img, l_inner, r_inner, Colors.GRAY_COLOR, 1)

        # 文字标注
        y = 30
        cv2.putText(img, "NeutralFace", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, Colors.NORMAL_COLOR, 2)
        y += 30
        cv2.putText(img, f"ICD: {indicators['icd']:.1f}px", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, Colors.NEUTRAL_COLOR, 1)
        y += 25
        cv2.putText(img, f"Eye Ratio: {indicators['eye_area_ratio']:.3f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, Colors.NEUTRAL_COLOR, 1)
        y += 25
        cv2.putText(img, f"Symmetry: {indicators['face_symmetry_score']:.3f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, Colors.NEUTRAL_COLOR, 1)
        y += 25

        # 眼裂面积
        cv2.putText(img, f"L Eye Area: {indicators['left_eye_area']:.0f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.LEFT_COLOR, 1)
        y += 20
        cv2.putText(img, f"R Eye Area: {indicators['right_eye_area']:.0f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, Colors.RIGHT_COLOR, 1)

        return img

    def get_baseline_dict(self, indicators: Dict) -> Dict:
        """
        从indicators提取基准数据字典

        供其他动作使用
        """
        return {
            'icd': indicators.get('icd', 0),
            'left_eye_area': indicators.get('left_eye_area', 0),
            'right_eye_area': indicators.get('right_eye_area', 0),
            'left_palpebral_length': indicators.get('left_palpebral_length', 0),
            'right_palpebral_length': indicators.get('right_palpebral_length', 0),
            'left_brow_height': indicators.get('left_brow_height', 0),
            'right_brow_height': indicators.get('right_brow_height', 0),
            'mouth_width': indicators.get('mouth_width', 0),
            'mouth_height': indicators.get('mouth_height', 0),
            'left_nlf_length': indicators.get('left_nlf_length', 0),
            'right_nlf_length': indicators.get('right_nlf_length', 0),
        }