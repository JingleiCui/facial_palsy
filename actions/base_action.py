"""
动作基类 - 统一的处理流程，支持与静息帧对比
"""
from abc import ABC, abstractmethod
import numpy as np
from ..core.geometry_utils import *


class BaseAction(ABC):
    """动作基类"""

    def __init__(self, action_name, action_config):
        """
        初始化

        Args:
            action_name: 动作名称（英文）
            action_config: 动作配置字典
        """
        self.action_name = action_name
        self.action_config = action_config
        self.action_id = action_config['action_id']
        self.name_cn = action_config['name_cn']

    @abstractmethod
    def detect_peak_frame(self, landmarks_seq, w, h):
        """
        检测峰值帧 - 子类必须实现

        注意：在比较时，临时计算unit_length并归一化

        Args:
            landmarks_seq: landmarks序列
            w: 图像宽度
            h: 图像高度

        Returns:
            int: 峰值帧索引
        """
        pass

    @abstractmethod
    def extract_indicators(self, landmarks, w, h, neutral_indicators=None):
        """
        提取指标 - 子类必须实现

        返回原始指标（像素值），并与静息帧对比（如果提供）

        Args:
            landmarks: MediaPipe face landmarks对象
            w: 图像宽度
            h: 图像高度
            neutral_indicators: 静息帧的指标（原始像素值），用于对比

        Returns:
            dict: 原始指标字典
        """
        pass

    def extract_dynamic_features(self, landmarks_seq, w, h, fps):
        """
        提取动态特征（可选，子类可重写）

        Args:
            landmarks_seq: landmarks序列
            w: 图像宽度
            h: 图像高度
            fps: 帧率

        Returns:
            dict: 动态特征字典
        """
        dynamic_features = {}

        valid_landmarks = [lm for lm in landmarks_seq if lm is not None]

        if len(valid_landmarks) < 2:
            return dynamic_features

        # 计算时序测量值
        left_eye_openings = [get_eye_opening(lm, w, h, left=True)
                            for lm in valid_landmarks]
        right_eye_openings = [get_eye_opening(lm, w, h, left=False)
                             for lm in valid_landmarks]
        mouth_widths = [get_mouth_width(lm, w, h)
                       for lm in valid_landmarks]
        eyebrow_dists_left = [get_eyebrow_eye_distance(lm, w, h, left=True)
                             for lm in valid_landmarks]

        # 运动范围（原始像素值）
        dynamic_features['left_eye_motion_range'] = compute_motion_range(left_eye_openings)
        dynamic_features['right_eye_motion_range'] = compute_motion_range(right_eye_openings)
        dynamic_features['mouth_motion_range'] = compute_motion_range(mouth_widths)
        dynamic_features['eyebrow_motion_range'] = compute_motion_range(eyebrow_dists_left)

        # 平均速度（原始像素/秒）
        dynamic_features['left_eye_mean_velocity'] = compute_mean_velocity(left_eye_openings, fps)
        dynamic_features['right_eye_mean_velocity'] = compute_mean_velocity(right_eye_openings, fps)
        dynamic_features['mouth_mean_velocity'] = compute_mean_velocity(mouth_widths, fps)
        dynamic_features['eyebrow_mean_velocity'] = compute_mean_velocity(eyebrow_dists_left, fps)

        # 最大速度（原始像素/秒）
        dynamic_features['left_eye_max_velocity'] = compute_max_velocity(left_eye_openings, fps)
        dynamic_features['right_eye_max_velocity'] = compute_max_velocity(right_eye_openings, fps)
        dynamic_features['mouth_max_velocity'] = compute_max_velocity(mouth_widths, fps)
        dynamic_features['eyebrow_max_velocity'] = compute_max_velocity(eyebrow_dists_left, fps)

        # 平滑度（jerk，无单位）
        dynamic_features['left_eye_smoothness'] = compute_smoothness(left_eye_openings)
        dynamic_features['right_eye_smoothness'] = compute_smoothness(right_eye_openings)
        dynamic_features['mouth_smoothness'] = compute_smoothness(mouth_widths)
        dynamic_features['eyebrow_smoothness'] = compute_smoothness(eyebrow_dists_left)

        return dynamic_features

    def process(self, landmarks_seq, frames_seq, w, h, fps, neutral_indicators=None):
        """
        完整处理流程

        Args:
            landmarks_seq: landmarks序列
            frames_seq: frames序列
            w: 图像宽度
            h: 图像高度
            fps: 帧率
            neutral_indicators: 静息帧的指标（原始像素值），用于对比

        Returns:
            dict: {
                'peak_frame_idx': int,
                'peak_frame': np.ndarray,
                'peak_landmarks': landmarks,
                'unit_length': float,
                'raw_indicators': dict,
                'normalized_indicators': dict,
                'dynamic_features': dict,
                'normalized_dynamic_features': dict
            }
        """
        # 1. 检测峰值帧
        peak_idx = self.detect_peak_frame(landmarks_seq, w, h)

        # 2. 获取峰值帧
        peak_landmarks = landmarks_seq[peak_idx]
        peak_frame = frames_seq[peak_idx]

        if peak_landmarks is None:
            return None

        # 3. 计算单位长度
        unit_length = get_unit_length(peak_landmarks, w, h)

        # 4. 提取原始指标（可能包含与静息帧的对比）
        raw_indicators = self.extract_indicators(peak_landmarks, w, h, neutral_indicators)

        # 5. 统一归一化静态指标
        normalized_indicators = normalize_indicators(raw_indicators, unit_length)

        # 6. 提取动态特征
        dynamic_features = self.extract_dynamic_features(landmarks_seq, w, h, fps)

        # 7. 归一化动态特征
        normalized_dynamic_features = normalize_indicators(dynamic_features, unit_length)

        return {
            'peak_frame_idx': peak_idx,
            'peak_frame': peak_frame,
            'peak_landmarks': peak_landmarks,
            'unit_length': unit_length,
            'raw_indicators': raw_indicators,
            'normalized_indicators': normalized_indicators,
            'dynamic_features': dynamic_features,
            'normalized_dynamic_features': normalized_dynamic_features
        }

    def __str__(self):
        return f"{self.action_name} ({self.name_cn})"