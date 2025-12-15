# -*- coding: utf-8 -*-
"""
Eyelid Angle (ELA) Based Blink Detector
基于眼睑角度的眨眼检测系统

参考论文: Blinking Beyond EAR: A Stable Eyelid Angle Metric
核心功能:
1. ELA计算 - 基于3D landmarks的眼睑角度
2. 眨眼检测 - 使用k-means聚类
3. 特征提取 - closing/closed/reopening durations等
4. 可视化 - ELA时序图+眨眼阶段标注
"""

import os
import re
import json
import sqlite3
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")  # 多进程 + 无界面环境下安全绘图
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.cluster import KMeans

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from concurrent.futures import ProcessPoolExecutor, as_completed


# ==================== JSON 序列化辅助函数 ====================

def convert_numpy_types(obj):
    """
    递归转换 numpy 类型为 Python 原生类型，用于 JSON 序列化
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'PingFang SC']
matplotlib.rcParams['axes.unicode_minus'] = False

# ==================== MediaPipe 眼睑Landmark索引 ====================

# 左眼上眼睑的7个点（从内眼角到外眼角）
LEFT_EYE_UPPER = [398, 384, 385, 386, 387, 388, 466]

# 左眼下眼睑的7个点（从内眼角到外眼角）
LEFT_EYE_LOWER = [382, 381, 380, 374, 373, 390, 249]

# 右眼上眼睑的7个点（从内眼角到外眼角）
RIGHT_EYE_UPPER = [173, 157, 158, 159, 160, 161, 246]
# 右眼下眼睑的7个点（从内眼角到外眼角）
RIGHT_EYE_LOWER = [155, 154, 153, 145, 144, 163, 7]

# 眼内眦索引（用于归一化）
LEFT_INNER_CANTHUS = 362
RIGHT_INNER_CANTHUS = 133


# ==================== 数据结构 ====================

@dataclass
class BlinkEvent:
    """单次眨眼事件"""
    start_idx: int  # 眨眼开始帧索引
    end_idx: int  # 眨眼结束帧索引
    min_idx: int  # ELA最小值帧索引

    # 时间特征
    closing_duration: float  # 闭眼持续时间(秒)
    closed_duration: float  # 闭合持续时间(秒)
    reopening_duration: float  # 睁眼持续时间(秒)

    # 幅度特征
    amplitude: float  # 相对幅度
    ela_start: float  # 起始ELA
    ela_min: float  # 最小ELA
    ela_end: float  # 结束ELA

    # 速度特征
    max_closing_velocity: float  # 最大闭眼速度
    max_reopening_velocity: float  # 最大睁眼速度
    amplitude_velocity_ratio: float  # 幅度/速度比

    # 其他特征
    previous_time: Optional[float] = None  # 距上次眨眼时间
    normal_area: Optional[float] = None  # 归一化面积

    @property
    def duration(self) -> float:
        """计算眨眼总持续时间（秒）= closing + closed + reopening"""
        return self.closing_duration + self.closed_duration + self.reopening_duration


@dataclass
class ELASignal:
    """ELA信号数据"""
    raw: np.ndarray  # 原始ELA信号
    filtered: np.ndarray  # 滤波后的ELA信号
    derivative: np.ndarray  # 导数
    fps: float  # 帧率
    timestamps: np.ndarray  # 时间戳

    # 左右眼分别的ELA
    left_ela: Optional[np.ndarray] = None
    right_ela: Optional[np.ndarray] = None


# ==================== ELA计算核心函数 ====================

def normalize_z_coordinate(z_raw: float, transform_matrix: np.ndarray) -> float:
    """
    归一化Z坐标（论文公式1）

    Args:
        z_raw: MediaPipe原始z坐标
        transform_matrix: MediaPipe的变换矩阵

    Returns:
        归一化后的z坐标
    """
    if transform_matrix is not None and transform_matrix.shape[0] > 2:
        return 1.7 * z_raw * transform_matrix[2, 2]
    return z_raw


def fit_plane_to_landmarks(landmarks_3d: np.ndarray) -> np.ndarray:
    """
    对3D landmarks拟合平面，返回法向量

    使用SVD方法拟合平面：
    1. 中心化数据
    2. SVD分解
    3. 最小奇异值对应的向量即为法向量

    Args:
        landmarks_3d: [N, 3] N个3D landmarks

    Returns:
        [3,] 单位法向量
    """
    # 中心化
    centroid = np.mean(landmarks_3d, axis=0)
    centered = landmarks_3d - centroid

    # SVD分解
    U, S, Vt = np.linalg.svd(centered.T, full_matrices=False)

    # 法向量是最小奇异值对应的向量
    normal = U[:, -1]

    # 标准化方向：使用cross product判断
    # 确保法向量方向一致（从内眼角指向外眼角）
    if len(centered) > 1:
        direction_vec = centered[1] - centered[0]
        cross_prod = np.cross(direction_vec, normal)
        if np.dot(normal, cross_prod) < 0:
            normal = -normal

    return normal / (np.linalg.norm(normal) + 1e-8)


def calculate_ela_for_eye(
        upper_landmarks: np.ndarray,
        lower_landmarks: np.ndarray
) -> float:
    """
    计算单眼的ELA（眼睑角度）

    Args:
        upper_landmarks: [7, 3] 上眼睑的7个3D点
        lower_landmarks: [7, 3] 下眼睑的7个3D点

    Returns:
        ELA角度（度数）
    """
    # 拟合上下眼睑平面
    normal_upper = fit_plane_to_landmarks(upper_landmarks)
    normal_lower = fit_plane_to_landmarks(lower_landmarks)

    # 计算两个平面法向量的夹角（论文公式2）
    cos_angle = np.clip(np.dot(normal_upper, normal_lower), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def get_unit_length(landmarks, image_width: int, image_height: int) -> float:
    """
    获取归一化单位长度（双侧眼内眦距离）

    Args:
        landmarks: MediaPipe 478个landmarks
        image_width: 图像宽度
        image_height: 图像高度

    Returns:
        单位长度（像素）
    """
    left_canthus = landmarks[LEFT_INNER_CANTHUS]
    right_canthus = landmarks[RIGHT_INNER_CANTHUS]

    left_x = left_canthus.x * image_width
    left_y = left_canthus.y * image_height
    right_x = right_canthus.x * image_width
    right_y = right_canthus.y * image_height

    distance = np.sqrt((right_x - left_x) ** 2 + (right_y - left_y) ** 2)
    return distance


def extract_3d_landmarks(
        landmarks,
        indices: List[int],
        image_width: int,
        image_height: int,
        aspect_ratio: float = 1.0
) -> np.ndarray:
    """
    提取指定索引的3D landmarks

    Args:
        landmarks: MediaPipe检测到的478个landmarks
        indices: 要提取的landmark索引列表
        image_width: 图像宽度
        image_height: 图像高度
        aspect_ratio: 图像宽高比

    Returns:
        [N, 3] 3D坐标数组
    """
    coords = []
    for idx in indices:
        lm = landmarks[idx]
        x = lm.x * image_width
        y = lm.y * image_height / aspect_ratio  # Y坐标归一化
        z = lm.z * image_width  # Z坐标缩放（论文的启发式方法）
        coords.append([x, y, z])

    return np.array(coords, dtype=np.float32)


def calculate_combined_ela(
        left_ela: float,
        right_ela: float,
        yaw_angle: float
) -> float:
    """
    结合左右眼ELA，考虑头部yaw旋转（论文公式3）

    使用sigmoid权重：
    - 头部左转时，右眼权重增加
    - 头部右转时，左眼权重增加

    Args:
        left_ela: 左眼ELA
        right_ela: 右眼ELA
        yaw_angle: 头部yaw角度（弧度）

    Returns:
        组合后的ELA
    """

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    # 论文中的缩放因子
    weight_left = sigmoid(-4 * yaw_angle)
    weight_right = sigmoid(4 * yaw_angle)

    combined = weight_left * left_ela + weight_right * right_ela
    return combined


# ==================== ELA信号处理 ====================

def smooth_ela_signal(ela_raw: np.ndarray, fps: float) -> np.ndarray:
    """
    使用高斯滤波平滑ELA信号（论文公式4-5）

    标准差与帧率成正比：σ = FPS/30

    Args:
        ela_raw: 原始ELA信号
        fps: 视频帧率

    Returns:
        平滑后的ELA信号
    """
    sigma = fps / 30.0  # 论文的公式
    ela_filtered = gaussian_filter1d(ela_raw, sigma=sigma)
    return ela_filtered


def compute_derivative(signal: np.ndarray) -> np.ndarray:
    """
    计算信号的导数（中心差分）

    Args:
        signal: 输入信号

    Returns:
        导数信号
    """
    derivative = np.zeros_like(signal)

    # 中心差分
    derivative[1:-1] = (signal[2:] - signal[:-2]) / 2.0

    # 边界使用前向/后向差分
    derivative[0] = signal[1] - signal[0]
    derivative[-1] = signal[-1] - signal[-2]

    return derivative


# ==================== 眨眼检测 ====================

def detect_blinks_kmeans(
        ela_filtered: np.ndarray,
        derivative: np.ndarray,
        fps: float,
        min_blink_duration: float = 0.05,  # 最小眨眼持续时间(秒)
        max_blink_duration: float = 0.8  # 最大眨眼持续时间(秒)
) -> List[BlinkEvent]:
    """
    使用k-means聚类检测眨眼事件（论文Section III-4）

    步骤：
    1. 对导数的极值点进行k-means聚类（2类）
    2. 配对下降沿和上升沿
    3. 提取时间特征

    Args:
        ela_filtered: 平滑后的ELA信号
        derivative: ELA导数
        fps: 帧率
        min_blink_duration: 最小眨眼持续时间
        max_blink_duration: 最大眨眼持续时间

    Returns:
        检测到的眨眼事件列表
    """
    # 1. 找到导数的极值点
    neg_peaks, _ = find_peaks(-derivative, height=0)  # 负峰（下降沿）
    pos_peaks, _ = find_peaks(derivative, height=0)  # 正峰（上升沿）

    if len(neg_peaks) < 2 or len(pos_peaks) < 2:
        return []

    # 2. K-means聚类（2类：噪声 vs 真实眨眼）
    neg_values = -derivative[neg_peaks]
    pos_values = derivative[pos_peaks]

    # 对负峰聚类
    if len(neg_peaks) >= 2:
        kmeans_neg = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels_neg = kmeans_neg.fit_predict(neg_values.reshape(-1, 1))
        # 选择幅度较大的类作为眨眼
        cluster_means = [neg_values[labels_neg == i].mean() for i in range(2)]
        blink_cluster_neg = np.argmax(cluster_means)
        blink_neg_peaks = neg_peaks[labels_neg == blink_cluster_neg]
    else:
        blink_neg_peaks = neg_peaks

    # 对正峰聚类
    if len(pos_peaks) >= 2:
        kmeans_pos = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels_pos = kmeans_pos.fit_predict(pos_values.reshape(-1, 1))
        cluster_means = [pos_values[labels_pos == i].mean() for i in range(2)]
        blink_cluster_pos = np.argmax(cluster_means)
        blink_pos_peaks = pos_peaks[labels_pos == blink_cluster_pos]
    else:
        blink_pos_peaks = pos_peaks

    # 3. 配对下降沿和上升沿
    blinks = []
    for neg_idx in blink_neg_peaks:
        # 找到neg_idx之后最近的pos_idx
        following_pos = blink_pos_peaks[blink_pos_peaks > neg_idx]
        if len(following_pos) == 0:
            continue

        pos_idx = following_pos[0]

        # 检查眨眼持续时间
        duration = (pos_idx - neg_idx) / fps
        if duration < min_blink_duration or duration > max_blink_duration:
            continue

        # 找到最小ELA点
        min_idx = neg_idx + np.argmin(ela_filtered[neg_idx:pos_idx + 1])

        # 提取眨眼特征
        blink = extract_blink_features(
            ela_filtered, derivative,
            neg_idx, min_idx, pos_idx, fps
        )

        if blink is not None:
            blinks.append(blink)

    # 计算相邻眨眼的时间间隔
    for i in range(1, len(blinks)):
        blinks[i].previous_time = (blinks[i].start_idx - blinks[i - 1].start_idx) / fps

    return blinks


def summarize_blink_sequence(
        ela_signal: "ELASignal",
        blinks: List["BlinkEvent"]
) -> Dict[str, float]:
    """
    对整段眨眼序列做一个全局统计，方便后续写入 JSON / 画图。

    返回字段示例：
        {
            "num_blinks": 12,
            "duration_sec": 28.3,
            "blink_rate_per_minute": 25.4,
            "mean_duration": 0.18,
            "median_duration": 0.17,
            "max_duration": 0.32
        }
    """
    if ela_signal.timestamps is None or len(ela_signal.timestamps) == 0:
        return {
            "num_blinks": 0,
            "duration_sec": 0.0,
            "blink_rate_per_minute": 0.0,
            "mean_duration": 0.0,
            "median_duration": 0.0,
            "max_duration": 0.0,
        }

    num_blinks = len(blinks)
    t0 = float(ela_signal.timestamps[0])
    t1 = float(ela_signal.timestamps[-1])
    duration_sec = max(t1 - t0, 0.0)

    if duration_sec > 0 and num_blinks > 0:
        blink_rate_per_minute = num_blinks / duration_sec * 60.0
    else:
        blink_rate_per_minute = 0.0

    if num_blinks > 0:
        durations = np.array([b.duration for b in blinks], dtype=np.float32)
        mean_duration = float(np.mean(durations))
        median_duration = float(np.median(durations))
        max_duration = float(np.max(durations))
    else:
        mean_duration = median_duration = max_duration = 0.0

    return {
        "num_blinks": int(num_blinks),
        "duration_sec": float(duration_sec),
        "blink_rate_per_minute": float(blink_rate_per_minute),
        "mean_duration": float(mean_duration),
        "median_duration": float(median_duration),
        "max_duration": float(max_duration),
    }


def extract_blink_features(
        ela_filtered: np.ndarray,
        derivative: np.ndarray,
        start_idx: int,
        min_idx: int,
        end_idx: int,
        fps: float
) -> Optional[BlinkEvent]:
    """
    提取单次眨眼的特征（论文Table I）

    Args:
        ela_filtered: 平滑后的ELA信号
        derivative: 导数
        start_idx: 眨眼开始索引
        min_idx: ELA最小值索引
        end_idx: 眨眼结束索引
        fps: 帧率

    Returns:
        BlinkEvent对象，失败返回None
    """
    if start_idx >= min_idx or min_idx >= end_idx:
        return None

    # 扩展搜索范围，找到真正的起点和终点
    # 向前找到最近的局部最大值
    search_start = max(0, start_idx - int(0.5 * fps))  # 向前搜索0.5秒
    local_max_before = search_start + np.argmax(ela_filtered[search_start:start_idx + 1])

    # 向后找到最近的局部最大值
    search_end = min(len(ela_filtered), end_idx + int(0.5 * fps))
    local_max_after = end_idx + np.argmax(ela_filtered[end_idx:search_end])

    # 使用切线交点法计算精确的时间边界（论文Fig. 3）
    ela_start = ela_filtered[local_max_before]
    ela_min = ela_filtered[min_idx]
    ela_end = ela_filtered[local_max_after]

    # 闭眼阶段：从start到min
    closing_phase = ela_filtered[local_max_before:min_idx + 1]
    max_closing_vel = np.max(-derivative[local_max_before:min_idx + 1])

    # 睁眼阶段：从min到end
    reopening_phase = ela_filtered[min_idx:local_max_after + 1]
    max_reopening_vel = np.max(derivative[min_idx:local_max_after + 1])

    # 使用切线交点法计算持续时间
    # 简化版本：直接使用峰值导数点
    t1 = local_max_before / fps
    t2 = min_idx / fps
    t3 = local_max_after / fps

    closing_duration = t2 - t1
    closed_duration = 0.05  # 简化：假设闭合持续时间很短
    reopening_duration = t3 - t2

    # 幅度特征
    amplitude = (ela_start - ela_min) / (ela_start + 1e-6)

    # 幅度/速度比（论文Table I）
    av_ratio = (ela_end - ela_min) / (max_reopening_vel + 1e-6)

    # 归一化面积（论文Table I）
    area = np.sum(ela_end - reopening_phase)
    normal_area = area / ((ela_end - ela_min) * 2 * reopening_duration + 1e-6)

    return BlinkEvent(
        start_idx=local_max_before,
        end_idx=local_max_after,
        min_idx=min_idx,
        closing_duration=closing_duration,
        closed_duration=closed_duration,
        reopening_duration=reopening_duration,
        amplitude=amplitude,
        ela_start=ela_start,
        ela_min=ela_min,
        ela_end=ela_end,
        max_closing_velocity=max_closing_vel,
        max_reopening_velocity=max_reopening_vel,
        amplitude_velocity_ratio=av_ratio,
        normal_area=normal_area
    )


# ==================== 视频处理 ====================

def process_video_ela(
        video_path: str,
        model_path: str,
        start_frame: int = 0,
        end_frame: Optional[int] = None
) -> Optional[ELASignal]:
    """
    处理视频，提取ELA信号

    Args:
        video_path: 视频路径
        model_path: MediaPipe模型路径
        start_frame: 起始帧
        end_frame: 结束帧（None表示处理到视频末尾）

    Returns:
        ELASignal对象，失败返回None
    """
    # 初始化MediaPipe
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=True,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if end_frame is None:
        end_frame = total_frames

    # 跳转到起始帧
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    print(f"📹 处理视频: {os.path.basename(video_path)}")
    print(f"   帧率: {fps:.2f} FPS")
    print(f"   分辨率: {width}x{height}")
    print(f"   处理范围: {start_frame}-{end_frame}")

    left_elas = []
    right_elas = []
    combined_elas = []
    frame_idx = start_frame

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        # 转换为RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 检测landmarks
        detection_result = detector.detect(mp_image)

        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]

            # 提取左眼上下眼睑3D landmarks
            left_upper_3d = extract_3d_landmarks(
                landmarks, LEFT_EYE_UPPER, width, height, height / width
            )
            left_lower_3d = extract_3d_landmarks(
                landmarks, LEFT_EYE_LOWER, width, height, height / width
            )

            # 提取右眼上下眼睑3D landmarks
            right_upper_3d = extract_3d_landmarks(
                landmarks, RIGHT_EYE_UPPER, width, height, height / width
            )
            right_lower_3d = extract_3d_landmarks(
                landmarks, RIGHT_EYE_LOWER, width, height, height / width
            )

            # 计算左右眼ELA
            left_ela = calculate_ela_for_eye(left_upper_3d, left_lower_3d)
            right_ela = calculate_ela_for_eye(right_upper_3d, right_lower_3d)

            # 获取yaw角度（简化：假设为0）
            # 实际应该从detection_result.facial_transformation_matrixes提取
            yaw_angle = 0.0

            # 组合ELA
            combined_ela = calculate_combined_ela(left_ela, right_ela, yaw_angle)

            left_elas.append(left_ela)
            right_elas.append(right_ela)
            combined_elas.append(combined_ela)
        else:
            # 未检测到人脸，使用前一帧的值或NaN
            if len(combined_elas) > 0:
                left_elas.append(left_elas[-1])
                right_elas.append(right_elas[-1])
                combined_elas.append(combined_elas[-1])
            else:
                left_elas.append(np.nan)
                right_elas.append(np.nan)
                combined_elas.append(np.nan)

        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"   处理进度: {frame_idx - start_frame}/{end_frame - start_frame}")

    cap.release()

    if len(combined_elas) == 0:
        print("❌ 未检测到任何人脸")
        return None

    # 转换为numpy数组
    raw_ela = np.array(combined_elas, dtype=np.float32)
    left_ela_array = np.array(left_elas, dtype=np.float32)
    right_ela_array = np.array(right_elas, dtype=np.float32)

    # 处理NaN值（线性插值）
    if np.any(np.isnan(raw_ela)):
        valid_indices = ~np.isnan(raw_ela)
        if np.sum(valid_indices) > 1:
            raw_ela = np.interp(
                np.arange(len(raw_ela)),
                np.where(valid_indices)[0],
                raw_ela[valid_indices]
            )

    # 平滑信号
    filtered_ela = smooth_ela_signal(raw_ela, fps)

    # 计算导数
    derivative = compute_derivative(filtered_ela)

    # 时间戳
    timestamps = np.arange(len(raw_ela)) / fps

    print(f"✅ ELA信号提取完成: {len(raw_ela)} 帧")

    return ELASignal(
        raw=raw_ela,
        filtered=filtered_ela,
        derivative=derivative,
        fps=fps,
        timestamps=timestamps,
        left_ela=left_ela_array,
        right_ela=right_ela_array
    )


# ==================== 可视化 ====================

def visualize_ela_with_blinks(
        ela_signal: ELASignal,
        blinks: List[BlinkEvent],
        output_path: str,
        title: str = "ELA Signal with Blink Detection"
):
    """
    可视化ELA信号和检测到的眨眼事件

    创建3个子图：
    1. ELA原始信号 + 滤波信号 + 眨眼标注
    2. ELA导数 + 眨眼边界
    3. 眨眼各阶段详细标注（放大图）

    Args:
        ela_signal: ELA信号
        blinks: 检测到的眨眼列表
        output_path: 输出图像路径
        title: 图表标题
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    time = ela_signal.timestamps

    # ========== 子图1: ELA信号 + 眨眼标注 ==========
    ax1 = axes[0]

    # 绘制原始和滤波信号
    ax1.plot(time, ela_signal.raw, 'lightgray',
             linewidth=0.8, alpha=0.6, label='Raw ELA')
    ax1.plot(time, ela_signal.filtered, 'darkblue',
             linewidth=1.5, label='Filtered ELA')

    # 标注每个眨眼事件
    for i, blink in enumerate(blinks):
        start_time = time[blink.start_idx]
        end_time = time[blink.end_idx]
        min_time = time[blink.min_idx]

        # 眨眼区间背景
        ax1.axvspan(start_time, end_time, alpha=0.2, color='yellow')

        # 最小点
        ax1.plot(min_time, blink.ela_min, 'ro', markersize=6)

        # 标注眨眼编号
        ax1.text(min_time, blink.ela_min - 5, f'#{i + 1}',
                 ha='center', va='top', fontsize=8, color='red')

    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('ELA (degrees)', fontsize=11)
    ax1.set_title(f'{title}\n检测到 {len(blinks)} 次眨眼',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # ========== 子图2: 导数 + 眨眼边界 ==========
    ax2 = axes[1]

    ax2.plot(time, ela_signal.derivative, 'green', linewidth=1.0)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    # 标注眨眼的起止点
    for blink in blinks:
        start_time = time[blink.start_idx]
        end_time = time[blink.end_idx]

        # 起点（最大负导数）
        ax2.plot(time[blink.start_idx],
                 ela_signal.derivative[blink.start_idx],
                 'rv', markersize=8, label='Start' if blink == blinks[0] else '')

        # 终点（最大正导数）
        ax2.plot(time[blink.end_idx],
                 ela_signal.derivative[blink.end_idx],
                 'r^', markersize=8, label='End' if blink == blinks[0] else '')

    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('dELA/dt (deg/frame)', fontsize=11)
    ax2.set_title('ELA Derivative (用于检测眨眼边界)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # ========== 子图3: 眨眼阶段详细标注（选择前几个眨眼放大显示） ==========
    ax3 = axes[2]

    # 选择前3个眨眼事件进行详细展示
    demo_blinks = blinks[:min(3, len(blinks))]

    for i, blink in enumerate(demo_blinks):
        # 扩展显示范围
        margin = int(0.3 * ela_signal.fps)  # 前后各0.3秒
        plot_start = max(0, blink.start_idx - margin)
        plot_end = min(len(time), blink.end_idx + margin)

        plot_time = time[plot_start:plot_end] - time[blink.start_idx]  # 相对时间
        plot_ela = ela_signal.filtered[plot_start:plot_end]

        # 偏移以分离多个眨眼
        offset = i * 15

        # 绘制信号
        ax3.plot(plot_time, plot_ela + offset, linewidth=2,
                 label=f'Blink #{i + 1}')

        # 标注三个阶段
        t1 = time[blink.start_idx] - time[blink.start_idx]  # =0
        t2 = time[blink.min_idx] - time[blink.start_idx]
        t3 = time[blink.end_idx] - time[blink.start_idx]

        # 闭眼阶段
        ax3.axvspan(t1, t2, alpha=0.2, color='red')
        ax3.text((t1 + t2) / 2, blink.ela_start + offset + 2,
                 f'Closing\n{blink.closing_duration:.3f}s',
                 ha='center', va='bottom', fontsize=9, color='darkred')

        # 睁眼阶段
        ax3.axvspan(t2, t3, alpha=0.2, color='green')
        ax3.text((t2 + t3) / 2, blink.ela_end + offset + 2,
                 f'Reopening\n{blink.reopening_duration:.3f}s',
                 ha='center', va='bottom', fontsize=9, color='darkgreen')

        # 关键点标注
        ax3.plot(t1, blink.ela_start + offset, 'go', markersize=8)
        ax3.plot(t2, blink.ela_min + offset, 'ro', markersize=8)
        ax3.plot(t3, blink.ela_end + offset, 'go', markersize=8)

    ax3.set_xlabel('Relative Time (s)', fontsize=11)
    ax3.set_ylabel('ELA (degrees) + offset', fontsize=11)
    ax3.set_title('眨眼阶段详细标注 (Closing → Closed → Reopening)',
                  fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"✅ 可视化结果已保存: {output_path}")


def create_blink_summary_figure(
        blinks: List[BlinkEvent],
        fps: float,
        output_path: str
):
    """
    创建眨眼特征统计汇总图

    包含：
    1. 眨眼频率直方图
    2. closing/closed/reopening持续时间箱线图
    3. 幅度-速度散点图

    Args:
        blinks: 眨眼事件列表
        fps: 帧率
        output_path: 输出路径
    """
    if len(blinks) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ========== 子图1: 眨眼频率直方图 ==========
    ax1 = axes[0, 0]
    if len(blinks) > 1:
        intervals = [b.previous_time for b in blinks[1:] if b.previous_time is not None]
        if intervals:
            ax1.hist(intervals, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
            ax1.axvline(np.mean(intervals), color='red', linestyle='--',
                        linewidth=2, label=f'Mean: {np.mean(intervals):.2f}s')
            ax1.set_xlabel('Inter-Blink Interval (s)', fontsize=11)
            ax1.set_ylabel('Frequency', fontsize=11)
            ax1.set_title('眨眼频率分布', fontsize=12, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

    # ========== 子图2: 持续时间箱线图 ==========
    ax2 = axes[0, 1]

    closing_durs = [b.closing_duration * 1000 for b in blinks]  # 转换为ms
    closed_durs = [b.closed_duration * 1000 for b in blinks]
    reopening_durs = [b.reopening_duration * 1000 for b in blinks]

    data_to_plot = [closing_durs, closed_durs, reopening_durs]
    labels = ['Closing', 'Closed', 'Reopening']

    bp = ax2.boxplot(data_to_plot, labels=labels, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax2.set_ylabel('Duration (ms)', fontsize=11)
    ax2.set_title('眨眼各阶段持续时间', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # ========== 子图3: 幅度-速度散点图 ==========
    ax3 = axes[1, 0]

    amplitudes = [b.amplitude for b in blinks]
    velocities = [b.max_reopening_velocity for b in blinks]

    ax3.scatter(amplitudes, velocities, c='coral', s=50, alpha=0.6, edgecolors='black')
    ax3.set_xlabel('Amplitude (relative)', fontsize=11)
    ax3.set_ylabel('Max Reopening Velocity (deg/frame)', fontsize=11)
    ax3.set_title('幅度 vs 速度', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # ========== 子图4: 统计表格 ==========
    ax4 = axes[1, 1]
    ax4.axis('off')

    # 计算统计量
    stats_data = [
        ['指标', '均值', '标准差', 'Min', 'Max'],
        ['眨眼次数', f'{len(blinks)}', '-', '-', '-'],
        ['Closing (ms)', f'{np.mean(closing_durs):.1f}',
         f'{np.std(closing_durs):.1f}',
         f'{np.min(closing_durs):.1f}', f'{np.max(closing_durs):.1f}'],
        ['Closed (ms)', f'{np.mean(closed_durs):.1f}',
         f'{np.std(closed_durs):.1f}',
         f'{np.min(closed_durs):.1f}', f'{np.max(closed_durs):.1f}'],
        ['Reopening (ms)', f'{np.mean(reopening_durs):.1f}',
         f'{np.std(reopening_durs):.1f}',
         f'{np.min(reopening_durs):.1f}', f'{np.max(reopening_durs):.1f}'],
        ['Amplitude', f'{np.mean(amplitudes):.3f}',
         f'{np.std(amplitudes):.3f}',
         f'{np.min(amplitudes):.3f}', f'{np.max(amplitudes):.3f}'],
    ]

    table = ax4.table(cellText=stats_data, cellLoc='center',
                      loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # 设置表头样式
    for i in range(5):
        cell = table[(0, i)]
        cell.set_facecolor('#4CAF50')
        cell.set_text_props(weight='bold', color='white')

    ax4.set_title('眨眼特征统计', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"✅ 特征统计图已保存: {output_path}")


# ==================== 主处理函数 ====================

def _safe_prefix(
        examination_id: Optional[int],
        action_name: Optional[str]
) -> str:
    """
    把 exam_id / action_name 拼成一个安全的前缀，用来命名 PNG/JSON 文件。
    """
    parts = []
    if examination_id is not None:
        parts.append(f"{examination_id}")
    if action_name:
        parts.append(action_name)

    raw = "_".join(parts) if parts else "UNKNOWN"
    raw = raw.replace(" ", "_")
    return re.sub(r"[^0-9A-Za-z_\-]+", "", raw)


def analyze_video_blinks(
        video_path: str,
        model_path: str,
        output_dir: str,
        patient_id: str = "UNKNOWN",
        action_name: str = "UnknownAction",
        examination_id: Optional[int] = None,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        fps: Optional[float] = None,
) -> Optional[Dict]:
    """
    单个视频的眨眼分析入口：
      1. 调用 process_video_ela 提取 ELA 信号；
      2. 用 KMeans 做眨眼检测；
      3. 计算统计特征；
      4. 输出两张图 + 一个 JSON 结果文件；
      5. 返回一个 Python dict，供批处理统计用。

    注意：
      - start_frame / end_frame 用于只分析关键动作时间段；
      - fps 如果传入，则用于统计（否则用视频里的 fps）。
    """
    os.makedirs(output_dir, exist_ok=True)
    prefix = _safe_prefix(examination_id, action_name)

    print(f"\n[INFO] 分析视频: {prefix}")
    print(f"       路径: {video_path}")
    print(f"       帧范围: [{start_frame}, {('end' if end_frame is None else end_frame)}]")

    # 1. 提取 ELA 信号
    ela_signal = process_video_ela(
        video_path=video_path,
        model_path=model_path,
        start_frame=start_frame,
        end_frame=end_frame
    )

    if ela_signal is None or ela_signal.filtered is None or len(ela_signal.filtered) == 0:
        print(f"[WARN] {prefix} - ELA 信号为空，跳过。")
        return None

    # 如果调用者给了 fps，则覆盖；否则使用视频中检测到的 fps
    if fps is not None and fps > 0:
        ela_signal.fps = float(fps)

    # 2. 眨眼检测
    blinks = detect_blinks_kmeans(
        ela_filtered=ela_signal.filtered,
        derivative=ela_signal.derivative,
        fps=ela_signal.fps
    )

    # 3. 统计特征
    stats = summarize_blink_sequence(ela_signal, blinks)

    # 4. 可视化：ELA + 眨眼标记
    vis_path = os.path.join(output_dir, f"{prefix}_ela_blinks.png")
    try:
        visualize_ela_with_blinks(ela_signal, blinks, vis_path)
    except Exception as e:
        print(f"[WARN] {prefix} - 绘制 ELA 曲线图失败: {e}")

    # 5. 可视化：眨眼总结图（直方图 / 箱线图等）
    summary_path = os.path.join(output_dir, f"{prefix}_blink_summary.png")
    try:
        create_blink_summary_figure(blinks, ela_signal.fps, summary_path)
    except Exception as e:
        print(f"[WARN] {prefix} - 绘制总结图失败: {e}")

    # 6. JSON 结果
    result = {
        "video_path": video_path,
        "examination_id": examination_id,
        "patient_id": patient_id,
        "action_name": action_name,
        "start_frame": int(start_frame),
        "end_frame": None if end_frame is None else int(end_frame),
        "fps": float(ela_signal.fps),
        "num_frames": int(len(ela_signal.filtered)),
        "num_blinks": int(stats["num_blinks"]),
        "duration_sec": float(stats["duration_sec"]),
        "blink_rate_per_minute": float(stats["blink_rate_per_minute"]),
        "blink_stats": stats,
        "blinks": [asdict(b) for b in blinks],
    }

    # 转换所有 numpy 类型为 Python 原生类型
    result = convert_numpy_types(result)

    json_path = os.path.join(output_dir, f"{prefix}_blinks.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[OK] {prefix} - 眨眼数: {stats['num_blinks']}, 频率: {stats['blink_rate_per_minute']:.2f} 次/分钟")
    print(f"     JSON: {json_path}")
    print(f"     图像: {vis_path}")
    print(f"           {summary_path}")

    return result


def load_videos_from_database(
        db_path: str,
        action_filter: Optional[List[str]] = None,
        limit: Optional[int] = None
) -> List[Dict]:
    """
    从 facialPalsy.db 中读取需要做眨眼分析的视频列表。

    关联的表：
        - video_files       (视频路径、帧范围、fps 等)
        - examinations      (patient_id)
        - action_types      (action_name_en)

    参数:
        action_filter: 只处理某些动作，例如 ["SpontaneousEyeBlink", "VoluntaryEyeBlink"]
        limit        : 仅取前 N 个样本，用于调试

    返回:
        每个元素是一个 dict，包含：
            {
                "examination_id", "patient_id",
                "action_name", "file_path",
                "start_frame", "end_frame", "fps"
            }
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    base_sql = """
        SELECT
            vf.examination_id,
            e.patient_id,
            at.action_name_en,
            vf.file_path,
            vf.start_frame,
            vf.end_frame,
            vf.fps
        FROM video_files AS vf
        LEFT JOIN examinations AS e ON vf.examination_id = e.examination_id
        LEFT JOIN action_types AS at ON vf.action_id = at.action_id
        WHERE vf.file_exists = 1
    """

    params: List = []

    if action_filter:
        placeholders = ",".join("?" for _ in action_filter)
        base_sql += f" AND at.action_name_en IN ({placeholders})"
        params.extend(action_filter)

    base_sql += " ORDER BY vf.examination_id"

    cursor.execute(base_sql, params)
    rows = cursor.fetchall()
    conn.close()

    if limit is not None:
        rows = rows[:limit]

    videos: List[Dict] = []
    for (exam_id, patient_id, action_name,
         file_path, start_frame, end_frame, fps) in rows:

        videos.append({
            "examination_id": str(exam_id) if exam_id is not None else "exam_id",
            "patient_id": str(patient_id) if patient_id is not None else "UNKNOWN",
            "action_name": str(action_name) if action_name is not None else "UnknownAction",
            "file_path": file_path,
            "start_frame": 0 if start_frame is None else int(start_frame),
            "end_frame": None if end_frame is None else int(end_frame),
            "fps": None if fps is None else float(fps),
        })

    return videos


def _worker_process_one(args: Dict) -> Dict:
    """
    多进程 worker：处理单个视频，并包装异常。
    """
    try:
        result = analyze_video_blinks(
            video_path=args["file_path"],
            model_path=args["model_path"],
            output_dir=args["output_dir"],
            patient_id=args["patient_id"],
            action_name=args["action_name"],
            examination_id=args["examination_id"],
            start_frame=args["start_frame"],
            end_frame=args["end_frame"],
            fps=args["fps"],
        )
        return {
            "success": result is not None,
            "error": None,
            "meta": args,
            "result": result,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "meta": args,
            "result": None,
        }


def batch_process_database(
        db_path: str,
        model_path: str,
        output_dir: str,
        action_filter: Optional[List[str]] = None,
        limit: Optional[int] = None,
        use_multiprocessing: bool = True,
        num_workers: Optional[int] = None,
):
    """
    从数据库中批量读取视频，做眨眼检测 + 可视化。

    参数：
        db_path          : facialPalsy.db 的路径
        model_path       : MediaPipe face_landmarker.task 模型路径
        output_dir       : 所有结果（json + png）的输出目录
        action_filter    : 指定只处理哪些动作
        limit            : 仅处理前 N 个样本（调试用）
        use_multiprocessing: 是否启用多进程
        num_workers      : 进程数（None=自动=CPU 核数）
    """
    os.makedirs(output_dir, exist_ok=True)

    print("============================================================")
    print("批量眨眼分析 - 从数据库加载视频")
    print("============================================================")
    print(f"数据库: {db_path}")
    print(f"模型   : {model_path}")
    print(f"输出目录: {output_dir}")

    if action_filter:
        print(f"动作过滤: {action_filter}")
    if limit is not None:
        print(f"样本上限: {limit}")

    videos = load_videos_from_database(db_path, action_filter=action_filter, limit=limit)
    if not videos:
        print("[WARN] 数据库中没有找到符合条件的视频。")
        return

    print(f"[INFO] 共 {len(videos)} 个视频待处理。")

    # 给每个 task 填充模型路径和输出目录
    tasks: List[Dict] = []
    for v in videos:
        one = dict(v)
        one["model_path"] = model_path
        one["output_dir"] = output_dir
        tasks.append(one)

    results_summary = []

    if use_multiprocessing:
        if num_workers is None or num_workers <= 0:
            # 不指定的话，默认用 CPU 逻辑核数
            try:
                import multiprocessing
                num_workers = multiprocessing.cpu_count()
            except Exception:
                num_workers = 4

        print(f"[INFO] 启用多进程，进程数: {num_workers}")

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_task = {
                executor.submit(_worker_process_one, t): t for t in tasks
            }

            for idx, future in enumerate(as_completed(future_to_task), start=1):
                t = future_to_task[future]
                meta = {
                    "examination_id": t["examination_id"],
                    "patient_id": t["patient_id"],
                    "action_name": t["action_name"],
                }
                try:
                    out = future.result()
                except Exception as e:
                    print(f"[ERROR] 视频 {meta} 处理失败（future 异常）: {e}")
                    results_summary.append({
                        **meta,
                        "success": False,
                        "error": str(e),
                    })
                    continue

                if out["success"]:
                    r = out["result"]
                    print(f"[{idx}/{len(tasks)}] OK - {meta}")
                    results_summary.append({
                        **meta,
                        "success": True,
                        "num_blinks": r["num_blinks"],
                        "blink_rate_per_minute": r["blink_rate_per_minute"],
                    })
                else:
                    print(f"[{idx}/{len(tasks)}] FAIL - {meta} - {out['error']}")
                    results_summary.append({
                        **meta,
                        "success": False,
                        "error": out["error"],
                    })

    else:
        print("[INFO] 使用单进程顺序处理（调试或排查错误时使用）。")
        for idx, t in enumerate(tasks, start=1):
            meta = {
                "examination_id": t["examination_id"],
                "patient_id": t["patient_id"],
                "action_name": t["action_name"],
            }
            try:
                out = _worker_process_one(t)
            except Exception as e:
                print(f"[ERROR] 视频 {meta} 处理失败（直接异常）: {e}")
                results_summary.append({
                    **meta,
                    "success": False,
                    "error": str(e),
                })
                continue

            if out["success"]:
                r = out["result"]
                print(f"[{idx}/{len(tasks)}] OK - {meta}")
                results_summary.append({
                    **meta,
                    "success": True,
                    "num_blinks": r["num_blinks"],
                    "blink_rate_per_minute": r["blink_rate_per_minute"],
                })
            else:
                print(f"[{idx}/{len(tasks)}] FAIL - {meta} - {out['error']}")
                results_summary.append({
                    **meta,
                    "success": False,
                    "error": out["error"],
                })

    # 写一个整体 summary
    summary_path = os.path.join(output_dir, "z_blink_batch_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)

    print("============================================================")
    print(f"批量眨眼分析完成，summary 写入: {summary_path}")
    print("============================================================")


if __name__ == "__main__":
    DB_PATH = "/Users/cuijinglei/PycharmProjects/medicalProject/facialPalsy/facialPalsy.db"
    MODEL_PATH = "/Users/cuijinglei/PycharmProjects/medicalProject/models/face_landmarker.task"
    OUTPUT_DIR = "/Users/cuijinglei/Documents/facialPalsy/HGFA/eyelid_blink_analysis"

    # 只分析眨眼相关动作（可以按需修改或扩展）
    ACTION_FILTER = None

    LIMIT = None

    batch_process_database(
        db_path=DB_PATH,
        model_path=MODEL_PATH,
        output_dir=OUTPUT_DIR,
        action_filter=ACTION_FILTER,
        limit=LIMIT,
        use_multiprocessing=True,
        num_workers=8,
    )