"""
运动特征计算工具 - Motion Feature Utilities
============================================

从landmarks序列计算12维全局运动特征向量。

这些特征是全局性的运动描述，与动作特定的dynamic_features不同：
- 动作特定的dynamic_features: 关注特定区域（眼、口等）的运动
- 全局motion_features: 描述整体运动模式（位移、速度、对称性等）

两者可以并行使用，互为补充。

特征维度 (12维):
    0: mean_displacement     - 平均位移
    1: max_displacement      - 最大位移
    2: std_displacement      - 位移标准差
    3: motion_energy         - 运动能量
    4: motion_asymmetry      - 运动不对称性
    5: temporal_smoothness   - 时间平滑度
    6: spatial_concentration - 空间集中度
    7: peak_ratio            - 峰值区域比例
    8: motion_center_x       - 运动重心X
    9: motion_center_y       - 运动重心Y
    10: velocity_mean        - 平均速度
    11: acceleration_std     - 加速度变化
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

# 从constants模块导入常量
from .constants import MotionRegions, MOTION_FEATURE_NAMES


# =============================================================================
# 运动指标数据结构
# =============================================================================

@dataclass
class MotionMetrics:
    """
    运动量化指标 (12维)
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

        # 位移统计 (归一化)
        features[0] = min(self.mean_displacement / 50.0, 1.0)
        features[1] = min(self.max_displacement / 200.0, 1.0)
        features[2] = min(self.std_displacement / 30.0, 1.0)
        features[3] = min(self.motion_energy / 10000.0, 1.0)

        # 不对称性 (已经是0-1)
        features[4] = min(self.motion_asymmetry, 1.0)

        # 时间/空间特性 (已经是0-1)
        features[5] = self.temporal_smoothness
        features[6] = self.spatial_concentration
        features[7] = self.peak_ratio

        # 运动重心 (已经是0-1)
        features[8] = self.motion_center[0]
        features[9] = self.motion_center[1]

        # 速度/加速度 (归一化)
        features[10] = min(self.velocity_mean / 20.0, 1.0)
        features[11] = min(self.acceleration_std / 10.0, 1.0)

        return features

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            'mean_displacement': self.mean_displacement,
            'max_displacement': self.max_displacement,
            'std_displacement': self.std_displacement,
            'motion_energy': self.motion_energy,
            'motion_asymmetry': self.motion_asymmetry,
            'temporal_smoothness': self.temporal_smoothness,
            'spatial_concentration': self.spatial_concentration,
            'peak_ratio': self.peak_ratio,
            'motion_center_x': self.motion_center[0],
            'motion_center_y': self.motion_center[1],
            'velocity_mean': self.velocity_mean,
            'acceleration_std': self.acceleration_std,
        }


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
        coords: (N, 2) ndarray 或 None
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
    n = len(displacement)

    for left_region, right_region in MotionRegions.SYMMETRIC_PAIRS:
        left_indices = [i for i in MotionRegions.ALL_REGIONS.get(left_region, []) if i < n]
        right_indices = [i for i in MotionRegions.ALL_REGIONS.get(right_region, []) if i < n]

        if not left_indices or not right_indices:
            continue

        left_motion = np.mean(displacement[left_indices])
        right_motion = np.mean(displacement[right_indices])

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
        n = min(coords_list[i].shape[0], coords_list[i + 1].shape[0])
        d = np.linalg.norm(coords_list[i + 1][:n] - coords_list[i][:n], axis=1).mean()
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


def compute_motion_features_detailed(
    landmarks_seq: List,
    w: int,
    h: int,
    fps: float = 30.0
) -> Tuple[np.ndarray, dict]:
    """
    计算12维运动特征并返回详细指标

    Args:
        landmarks_seq: MediaPipe landmarks对象列表
        w: 图像宽度
        h: 图像高度
        fps: 视频帧率

    Returns:
        features: (12,) 运动特征向量
        metrics_dict: 详细指标字典
    """
    metrics = MotionMetrics()

    # 转换为坐标数组
    coords_list = []
    for lm in landmarks_seq:
        coords = landmarks_to_array(lm, w, h)
        if coords is not None:
            coords_list.append(coords)

    if len(coords_list) < 3:
        return metrics.to_feature_vector(), metrics.to_dict()

    try:
        displacement = compute_displacement(coords_list)

        metrics.mean_displacement = float(np.mean(displacement))
        metrics.max_displacement = float(np.max(displacement))
        metrics.std_displacement = float(np.std(displacement))
        metrics.motion_energy = float(np.sum(displacement ** 2))

        velocities, accelerations = compute_velocity_acceleration(coords_list, fps)
        if len(velocities) > 0:
            metrics.velocity_mean = float(np.mean(velocities))
        if len(accelerations) > 0:
            metrics.acceleration_std = float(np.std(accelerations))

        metrics.temporal_smoothness = compute_temporal_smoothness(coords_list)
        metrics.spatial_concentration = compute_spatial_entropy(displacement)
        metrics.motion_asymmetry = compute_asymmetry(displacement)

        threshold = np.percentile(displacement, 75)
        metrics.peak_ratio = float(np.sum(displacement > threshold) / len(displacement))
        metrics.motion_center = compute_motion_center(displacement, coords_list[0])

    except Exception as e:
        print(f"  [WARN] 运动特征计算异常: {e}")

    return metrics.to_feature_vector(), metrics.to_dict()


# =============================================================================
# 辅助函数
# =============================================================================

def get_motion_feature_names() -> List[str]:
    """返回12维运动特征的名称列表"""
    return MOTION_FEATURE_NAMES.copy()