"""
特征提取器 - 统一的静态和动态特征提取逻辑
"""
import numpy as np
from geometry_utils import *
from config import STATIC_FEATURE_DIM, DYNAMIC_FEATURE_DIM


class FeatureExtractor:
    """特征提取器 - 提取静态和动态几何特征"""

    @staticmethod
    def extract_static_features(landmarks, w, h, unit_length):
        """
        提取静态几何特征（从峰值帧）

        Args:
            landmarks: MediaPipe face landmarks对象
            w: 图像宽度
            h: 图像高度
            unit_length: 单位长度（用于归一化）

        Returns:
            np.ndarray: (32,) 静态特征向量
        """
        features = []

        # 1-2. 眼睛开合度（左右，归一化）
        left_eye_opening = get_eye_opening(landmarks, w, h, left=True)
        right_eye_opening = get_eye_opening(landmarks, w, h, left=False)
        features.append(left_eye_opening / unit_length if unit_length > 0 else 0)
        features.append(right_eye_opening / unit_length if unit_length > 0 else 0)

        # 3-4. 眼睛长度（左右，归一化）
        left_eye_length = get_eye_length(landmarks, w, h, left=True)
        right_eye_length = get_eye_length(landmarks, w, h, left=False)
        features.append(left_eye_length / unit_length if unit_length > 0 else 0)
        features.append(right_eye_length / unit_length if unit_length > 0 else 0)

        # 5-6. 眉眼距（左右，归一化）
        left_eyebrow_dist = get_eyebrow_eye_distance(landmarks, w, h, left=True)
        right_eyebrow_dist = get_eyebrow_eye_distance(landmarks, w, h, left=False)
        features.append(left_eyebrow_dist / unit_length if unit_length > 0 else 0)
        features.append(right_eyebrow_dist / unit_length if unit_length > 0 else 0)

        # 7. 嘴角宽度（归一化）
        mouth_width = get_mouth_width(landmarks, w, h)
        features.append(mouth_width / unit_length if unit_length > 0 else 0)

        # 8. 嘴部高度（归一化）
        mouth_height = get_mouth_height(landmarks, w, h)
        features.append(mouth_height / unit_length if unit_length > 0 else 0)

        # 9-16. 对称性指标
        eye_opening_ratio = left_eye_opening / right_eye_opening if right_eye_opening > 0 else 1.0
        eye_length_ratio = left_eye_length / right_eye_length if right_eye_length > 0 else 1.0
        eyebrow_ratio = left_eyebrow_dist / right_eyebrow_dist if right_eyebrow_dist > 0 else 1.0

        features.extend([
            eye_opening_ratio,  # 9. 眼开合度比例
            eye_length_ratio,  # 10. 眼长度比例
            eyebrow_ratio,  # 11. 眉眼距比例
            abs(eye_opening_ratio - 1.0),  # 12. 眼开合度偏离对称
            abs(eye_length_ratio - 1.0),  # 13. 眼长度偏离对称
            abs(eyebrow_ratio - 1.0),  # 14. 眉眼距偏离对称
            (left_eye_opening + left_eye_length) / (right_eye_opening + right_eye_length)
            if (right_eye_opening + right_eye_length) > 0 else 1.0,  # 15. 综合对称度
            abs((left_eye_opening - right_eye_opening) / unit_length)
            if unit_length > 0 else 0,  # 16. 眼开合度绝对差异
        ])

        # 补齐到32维
        while len(features) < STATIC_FEATURE_DIM:
            features.append(0.0)

        return np.array(features[:STATIC_FEATURE_DIM], dtype=np.float32)

    @staticmethod
    def extract_dynamic_features(landmarks_seq, w, h, unit_length, fps):
        """
        提取动态几何特征（从整个视频序列）

        Args:
            landmarks_seq: landmarks序列
            w: 图像宽度
            h: 图像高度
            unit_length: 单位长度（用于归一化）
            fps: 帧率

        Returns:
            np.ndarray: (16,) 动态特征向量
        """
        features = []

        # 过滤掉None
        valid_landmarks = [lm for lm in landmarks_seq if lm is not None]

        if len(valid_landmarks) < 2:
            return np.zeros(DYNAMIC_FEATURE_DIM, dtype=np.float32)

        # 计算时序测量值
        left_eye_openings = [get_eye_opening(lm, w, h, left=True) for lm in valid_landmarks]
        right_eye_openings = [get_eye_opening(lm, w, h, left=False) for lm in valid_landmarks]
        mouth_widths = [get_mouth_width(lm, w, h) for lm in valid_landmarks]
        eyebrow_dists_left = [get_eyebrow_eye_distance(lm, w, h, left=True) for lm in valid_landmarks]

        # 1-4. 运动范围（归一化）
        features.append(compute_motion_range(left_eye_openings, unit_length))
        features.append(compute_motion_range(right_eye_openings, unit_length))
        features.append(compute_motion_range(mouth_widths, unit_length))
        features.append(compute_motion_range(eyebrow_dists_left, unit_length))

        # 5-8. 平均速度（归一化）
        features.append(compute_mean_velocity(left_eye_openings, fps, unit_length))
        features.append(compute_mean_velocity(right_eye_openings, fps, unit_length))
        features.append(compute_mean_velocity(mouth_widths, fps, unit_length))
        features.append(compute_mean_velocity(eyebrow_dists_left, fps, unit_length))

        # 9-12. 最大速度（归一化）
        features.append(compute_max_velocity(left_eye_openings, fps, unit_length))
        features.append(compute_max_velocity(right_eye_openings, fps, unit_length))
        features.append(compute_max_velocity(mouth_widths, fps, unit_length))
        features.append(compute_max_velocity(eyebrow_dists_left, fps, unit_length))

        # 13-16. 运动平滑度
        features.append(compute_smoothness(left_eye_openings))
        features.append(compute_smoothness(right_eye_openings))
        features.append(compute_smoothness(mouth_widths))
        features.append(compute_smoothness(eyebrow_dists_left))

        return np.array(features[:DYNAMIC_FEATURE_DIM], dtype=np.float32)