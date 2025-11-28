"""
运动特征计算工具 - Motion Feature Utilities
============================================

为 geometry_utils.py 补充运动特征计算相关的函数和常量。
用于从landmarks序列计算12维运动特征向量。

设计原则:
1. 复用 geometry_utils.py 中已有的关键点索引
2. 不重复定义关键点
3. 提供纯函数式接口，便于在 video_pipeline.py 中调用
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

# 导入已有的关键点定义
from .geometry_utils import (
    LEFT_EYE_POINTS, RIGHT_EYE_POINTS,
    EYEBROW_LEFT_POINTS, EYEBROW_RIGHT_POINTS,
    UPPER_LIP_LEFT_POINTS, UPPER_LIP_RIGHT_POINTS,
    LOWER_LIP_LEFT_POINTS, LOWER_LIP_RIGHT_POINTS,
    LEFT_CHEEK, RIGHT_CHEEK,
    get_point, get_unit_length
)

# =============================================================================
# 面部区域索引 (用于运动不对称性计算)
# 基于 MediaPipe 478点模型
# =============================================================================

# 左右对称区域对 - 复用已有定义并补充腮部
MOTION_SYMMETRIC_REGIONS = {
    'left_eye': LEFT_EYE_POINTS,
    'right_eye': RIGHT_EYE_POINTS,
    'left_brow': EYEBROW_LEFT_POINTS,
    'right_brow': EYEBROW_RIGHT_POINTS,
    # 腮部区域 (脸颊)
    'left_cheek': LEFT_CHEEK,
    'right_cheek': RIGHT_CHEEK,
}

# 左右对应区域对
SYMMETRIC_PAIRS = [
    ('left_eye', 'right_eye'),
    ('left_brow', 'right_brow'),
    ('left_cheek', 'right_cheek'),
]


# =============================================================================
# 运动指标数据结构
# =============================================================================

@dataclass
class MotionMetrics:
    """
    运动量化指标

    包含12维特征:
        0: mean_displacement     - 平均位移
        1: max_displacement      - 最大位移
        2: std_displacement      - 位移标准差
        3: motion_energy         - 运动能量
        4: motion_asymmetry      - 运动不对称性
        5: temporal_smoothness   - 时间平滑度
        6: spatial_concentration - 空间集中度
        7: peak_ratio            - 峰值区域比例
        8-9: motion_center       - 运动重心
        10: velocity_mean        - 平均速度
        11: acceleration_std     - 加速度变化
    """
    mean_displacement: float = 0.0
    max_displacement: float = 0.0
    std_displacement: float = 0.0
    motion_energy: float = 0.0
    motion_asymmetry: float = 0.0
    temporal_smoothness: float = 0.0
    spatial_concentration: float = 0.0
    peak_ratio: float = 0.0
    motion_center: Tuple[float, float] = (0.5, 0.5)
    velocity_mean: float = 0.0
    acceleration_std: float = 0.0

    def to_feature_vector(self) -> np.ndarray:
        """转换为12维特征向量 (归一化到0-1范围)"""
        features = np.zeros(12, dtype=np.float32)

        # 位移统计
        features[0] = min(self.mean_displacement / 50.0, 1.0)
        features[1] = min(self.max_displacement / 200.0, 1.0)
        features[2] = min(self.std_displacement / 30.0, 1.0)
        features[3] = min(self.motion_energy / 10000.0, 1.0)

        # 不对称性
        features[4] = min(self.motion_asymmetry, 1.0)

        # 时间/空间特性
        features[5] = self.temporal_smoothness
        features[6] = self.spatial_concentration
        features[7] = self.peak_ratio

        # 运动重心
        features[8] = self.motion_center[0]
        features[9] = self.motion_center[1]

        # 速度/加速度
        features[10] = min(self.velocity_mean / 20.0, 1.0)
        features[11] = min(self.acceleration_std / 10.0, 1.0)

        return features


# =============================================================================
# 坐标转换函数
# =============================================================================

def landmarks_to_array(landmarks, w: int, h: int) -> Optional[np.ndarray]:
    """
    将MediaPipe landmarks对象转换为numpy数组

    Args:
        landmarks: MediaPipe face_landmarks对象 (478点)
        w: 图像宽度
        h: 图像高度

    Returns:
        coords: (478, 2) ndarray 或 None
    """
    if landmarks is None:
        return None

    try:
        n_points = len(landmarks)
        coords = np.array(
            [[landmarks[i].x * w, landmarks[i].y * h] for i in range(n_points)],
            dtype=np.float32
        )
        return coords
    except Exception:
        return None


# =============================================================================
# 运动计算函数
# =============================================================================

def compute_displacement(coords_list: List[np.ndarray]) -> np.ndarray:
    """
    计算每个关键点的累计位移

    Args:
        coords_list: 坐标数组序列 [(N, 2), ...]

    Returns:
        displacement: (N,) 每个点的累计位移
    """
    if len(coords_list) < 2:
        n = coords_list[0].shape[0] if coords_list else 478
        return np.zeros(n, dtype=np.float32)

    n_landmarks = coords_list[0].shape[0]
    displacement = np.zeros(n_landmarks, dtype=np.float32)

    for i in range(len(coords_list) - 1):
        curr = coords_list[i]
        next_coords = coords_list[i + 1]
        n = min(curr.shape[0], next_coords.shape[0], n_landmarks)
        diff = np.linalg.norm(next_coords[:n] - curr[:n], axis=1)
        displacement[:n] += diff

    return displacement


def compute_velocity_acceleration(
        coords_list: List[np.ndarray],
        fps: float = 30.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算速度和加速度

    Returns:
        velocities: (frames-1,) 每帧的平均速度
        accelerations: (frames-2,) 每帧的平均加速度
    """
    if len(coords_list) < 2:
        return np.array([]), np.array([])

    dt = 1.0 / fps

    # 计算每帧的平均速度
    velocities = []
    for i in range(len(coords_list) - 1):
        curr = coords_list[i]
        next_coords = coords_list[i + 1]
        n = min(curr.shape[0], next_coords.shape[0])
        v = np.linalg.norm(next_coords[:n] - curr[:n], axis=1).mean() / dt
        velocities.append(v)

    velocities = np.array(velocities)

    # 计算加速度
    if len(velocities) < 2:
        return velocities, np.array([])

    accelerations = np.diff(velocities) / dt

    return velocities, accelerations


def compute_asymmetry(displacement: np.ndarray) -> float:
    """
    计算运动不对称性 (左右差异)

    Args:
        displacement: (N,) 每个关键点的位移

    Returns:
        asymmetry: 0-1, 0表示完全对称
    """
    asymmetries = []

    for left_region, right_region in SYMMETRIC_PAIRS:
        left_indices = MOTION_SYMMETRIC_REGIONS.get(left_region, [])
        right_indices = MOTION_SYMMETRIC_REGIONS.get(right_region, [])

        # 过滤有效索引
        left_valid = [i for i in left_indices if i < len(displacement)]
        right_valid = [i for i in right_indices if i < len(displacement)]

        if not left_valid or not right_valid:
            continue

        left_motion = np.mean(displacement[left_valid])
        right_motion = np.mean(displacement[right_valid])

        total = left_motion + right_motion
        if total > 1e-6:
            diff = abs(left_motion - right_motion) / total
            asymmetries.append(diff)

    return float(np.mean(asymmetries)) if asymmetries else 0.0


def compute_motion_center(
        displacement: np.ndarray,
        coords: np.ndarray
) -> Tuple[float, float]:
    """
    计算运动重心 (位移加权坐标)

    Returns:
        (cx, cy) 归一化坐标 (0-1)
    """
    if len(displacement) == 0 or len(coords) == 0:
        return (0.5, 0.5)

    n = min(len(displacement), len(coords))
    total = displacement[:n].sum()

    if total < 1e-6:
        return (0.5, 0.5)

    weights = displacement[:n]
    coords_n = coords[:n]

    cx = np.sum(weights * coords_n[:, 0]) / total
    cy = np.sum(weights * coords_n[:, 1]) / total

    # 归一化到0-1
    x_min, x_max = coords_n[:, 0].min(), coords_n[:, 0].max()
    y_min, y_max = coords_n[:, 1].min(), coords_n[:, 1].max()

    if x_max > x_min:
        cx = (cx - x_min) / (x_max - x_min)
    else:
        cx = 0.5

    if y_max > y_min:
        cy = (cy - y_min) / (y_max - y_min)
    else:
        cy = 0.5

    return (float(np.clip(cx, 0, 1)), float(np.clip(cy, 0, 1)))


def compute_spatial_entropy(displacement: np.ndarray) -> float:
    """
    计算空间熵 (运动集中度)

    Returns:
        entropy: 0-1, 0表示运动高度集中在少数点
    """
    total = displacement.sum()
    if total < 1e-6:
        return 0.0

    probs = displacement / total
    probs = probs[probs > 1e-10]

    entropy = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(len(displacement))

    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def compute_temporal_smoothness(coords_list: List[np.ndarray]) -> float:
    """
    计算时间平滑度 (运动连续性)

    Returns:
        smoothness: 0-1, 1表示非常平滑
    """
    if len(coords_list) < 3:
        return 0.0

    frame_disps = []
    for i in range(len(coords_list) - 1):
        d = np.linalg.norm(coords_list[i + 1] - coords_list[i], axis=1).mean()
        frame_disps.append(d)

    if len(frame_disps) < 2:
        return 0.0

    mean_d = np.mean(frame_disps)
    std_d = np.std(frame_disps)

    if mean_d < 1e-6:
        return 1.0

    cv = std_d / mean_d  # 变异系数
    return float(1.0 / (1.0 + cv))


# =============================================================================
# 主函数：从landmarks序列计算运动特征
# =============================================================================

def compute_motion_features(
        landmarks_seq: List,
        w: int,
        h: int,
        fps: float = 30.0
) -> np.ndarray:
    """
    从landmarks序列计算12维运动特征

    Args:
        landmarks_seq: MediaPipe landmarks对象列表
        w: 图像宽度
        h: 图像高度
        fps: 视频帧率

    Returns:
        features: (12,) 运动特征向量
    """
    metrics = MotionMetrics()

    # 转换为坐标数组，跳过None
    coords_list = []
    for lm in landmarks_seq:
        coords = landmarks_to_array(lm, w, h)
        if coords is not None:
            coords_list.append(coords)

    # 最少需要3帧
    if len(coords_list) < 3:
        return metrics.to_feature_vector()

    try:
        # 1. 计算位移
        displacement = compute_displacement(coords_list)

        # 2. 基础统计
        metrics.mean_displacement = float(np.mean(displacement))
        metrics.max_displacement = float(np.max(displacement))
        metrics.std_displacement = float(np.std(displacement))
        metrics.motion_energy = float(np.sum(displacement ** 2))

        # 3. 速度和加速度
        velocities, accelerations = compute_velocity_acceleration(coords_list, fps)
        if len(velocities) > 0:
            metrics.velocity_mean = float(np.mean(velocities))
        if len(accelerations) > 0:
            metrics.acceleration_std = float(np.std(accelerations))

        # 4. 时间平滑度
        metrics.temporal_smoothness = compute_temporal_smoothness(coords_list)

        # 5. 空间集中度
        metrics.spatial_concentration = compute_spatial_entropy(displacement)

        # 6. 运动不对称性
        metrics.motion_asymmetry = compute_asymmetry(displacement)

        # 7. 峰值比例
        threshold = np.percentile(displacement, 75)
        metrics.peak_ratio = float(np.sum(displacement > threshold) / len(displacement))

        # 8. 运动重心
        metrics.motion_center = compute_motion_center(displacement, coords_list[0])

    except Exception as e:
        print(f"  [WARN] 运动特征计算异常: {e}")

    return metrics.to_feature_vector()