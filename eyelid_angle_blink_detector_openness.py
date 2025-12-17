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



# ==================== Openness（眼睛开合度）参数（推荐） ====================
# Openness 计算：上/下眼睑“垂直距离” / “眼裂宽度”，再做个体自适应归一化到 0~1
OPENNESS_SMOOTH_SIGMA_SEC = 0.02       # 平滑窗口（秒）：20ms 左右
OPENNESS_NORM_P_OPEN = 95             # 认为“最睁开”的分位数
OPENNESS_NORM_P_CLOSED = 5            # 认为“最闭合”的分位数

# 眨眼检测（基于 closure = 1 - openness）
BLINK_MIN_DURATION_SEC = 0.05
BLINK_MAX_DURATION_SEC = 0.80
BLINK_MIN_DISTANCE_SEC = 0.08         # 两次眨眼最小间隔（秒）
BLINK_PROMINENCE = 0.10               # 峰值显著性（越大越严格）
BLINK_CROSS_LEVEL = 0.15              # 用于找 start/end 的阈值（closure）
BLINK_CLOSED_LEVEL = 0.65             # 判断“完全闭合/闭合保持”的阈值（closure）

# 左右眼同步判定：两眼眨眼最小点（min_idx）时间差阈值
MAX_LR_PAIR_DELTA_SEC = 0.15

# ==================== MediaPipe FaceLandmarker 缓存（每进程一次） ====================

_FACE_LANDMARKER_CACHE = {}
# VIDEO 模式下，detect_for_video() 要求 timestamp_ms 在同一 detector 生命周期内严格递增。
# 为了在一个进程里复用 detector（加速）且跨视频不报错，我们为每个 model_path 维护全局时间戳。
_FACE_LANDMARKER_LAST_TS_MS: Dict[str, int] = {}

def get_face_landmarker(model_path: str):
    """获取 FaceLandmarker（VIDEO 模式），并做进程内缓存。"""
    global _FACE_LANDMARKER_CACHE
    if model_path in _FACE_LANDMARKER_CACHE:
        return _FACE_LANDMARKER_CACHE[model_path]

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    _FACE_LANDMARKER_CACHE[model_path] = detector
    return detector


def _interp_nan_1d(arr: np.ndarray) -> np.ndarray:
    """线性插值填充 1D 数组中的 NaN。"""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return arr
    if not np.any(np.isnan(arr)):
        return arr
    x = np.arange(arr.size)
    mask = ~np.isnan(arr)
    if mask.sum() < 2:
        # 太少有效点：全部用 0
        return np.nan_to_num(arr, nan=0.0)
    arr2 = arr.copy()
    arr2[~mask] = np.interp(x[~mask], x[mask], arr[mask])
    return arr2


def _smooth_signal(arr: np.ndarray, fps: float, sigma_sec: float = OPENNESS_SMOOTH_SIGMA_SEC) -> np.ndarray:
    """按“秒”尺度做高斯平滑（更符合不同 FPS 的一致性）。"""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size < 3:
        return arr
    sigma = max(0.5, float(sigma_sec) * float(fps))  # sigma=样本数
    return gaussian_filter1d(arr, sigma=sigma, mode="nearest")


def _normalize_openness_by_percentiles(raw: np.ndarray,
                                      p_open: float = OPENNESS_NORM_P_OPEN,
                                      p_closed: float = OPENNESS_NORM_P_CLOSED) -> np.ndarray:
    """把 openness ratio 做个体自适应归一化到 0~1。"""
    raw = np.asarray(raw, dtype=np.float32)
    raw = _interp_nan_1d(raw)
    if raw.size == 0:
        return raw
    hi = float(np.nanpercentile(raw, p_open))
    lo = float(np.nanpercentile(raw, p_closed))
    denom = hi - lo
    if denom < 1e-6:
        # 极端情况：几乎不变
        base = hi if abs(hi) > 1e-6 else 1.0
        norm = raw / base
    else:
        norm = (raw - lo) / denom
    return np.clip(norm, 0.0, 1.2).astype(np.float32)

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
    """眼部时序信号数据（推荐使用 Openness：眼睛开合度）

    - Openness：0~1，数值越大表示越“睁开”，越小表示越“闭合”
    - 本文件为了兼容旧代码，仍保留 left_ela/right_ela 字段（只有 compute_ela=True 才会生成）
    """
    raw: np.ndarray  # 主信号（默认=左右 Openness 的平均）
    filtered: np.ndarray  # 平滑后的主信号
    derivative: np.ndarray  # 主信号导数（每帧差分）
    fps: float  # 帧率
    timestamps: np.ndarray  # 时间戳（秒）

    # 旧字段（兼容）：左右眼分别的 ELA（可选）
    left_ela: Optional[np.ndarray] = None
    right_ela: Optional[np.ndarray] = None

    # 新增：左右眼 Openness（更直观、更稳定）
    left_openness_raw: Optional[np.ndarray] = None   # 原始 ratio（未归一化）
    right_openness_raw: Optional[np.ndarray] = None
    left_openness: Optional[np.ndarray] = None       # 0~1（个体自适应归一化）
    right_openness: Optional[np.ndarray] = None


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


def extract_2d_landmarks(
        landmarks,
        indices: List[int],
        image_width: int,
        image_height: int,
        aspect_ratio: float = 1.0
) -> np.ndarray:
    """
    提取指定索引的2D landmarks（像素坐标）

    Returns:
        [N, 2] 2D坐标数组
    """
    coords = []
    for idx in indices:
        lm = landmarks[idx]
        x = lm.x * image_width
        y = lm.y * image_height / aspect_ratio
        coords.append([x, y])
    return np.array(coords, dtype=np.float32)


def calculate_eye_openness_ratio(upper_xy: np.ndarray, lower_xy: np.ndarray) -> float:
    """计算单眼 Openness ratio（更直观的开合度）。

    思路（易解释）：
    - 取多对“上眼睑点 - 下眼睑点”的距离，求平均：代表“眼睛竖向张开程度”
    - 再除以“眼裂宽度”（眼角到眼角的距离，近似用眼睑点序列两端距离）：做尺度归一化

    返回：
        openness_ratio（未归一化到 0~1，随人/镜头略变，但稳定、可比较）
    """
    upper_xy = np.asarray(upper_xy, dtype=np.float32)
    lower_xy = np.asarray(lower_xy, dtype=np.float32)
    if upper_xy.shape != lower_xy.shape or upper_xy.ndim != 2 or upper_xy.shape[1] != 2:
        return float("nan")

    # 垂直张开：多对上/下点的距离平均
    vertical = np.linalg.norm(upper_xy - lower_xy, axis=1)
    v_mean = float(np.mean(vertical))

    # 眼裂宽度：用序列两端近似（内外眼角方向）
    w1 = float(np.linalg.norm(upper_xy[0] - upper_xy[-1]))
    w2 = float(np.linalg.norm(lower_xy[0] - lower_xy[-1]))
    width = 0.5 * (w1 + w2)
    if width < 1e-6:
        return float("nan")

    return v_mean / width


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


# ==================== Openness 眨眼检测（推荐） ====================

def _blink_event_to_dict(blink: "BlinkEvent", eye: str) -> Dict:
    """把 BlinkEvent 转成 dict，并补上 duration + eye 字段（更完整）。"""
    d = asdict(blink)
    d["duration"] = float(blink.duration)
    d["eye"] = eye
    # 额外补一个更直观的“闭合幅度百分比”
    # amplitude = closure_peak（0~1），转换成百分比更好理解
    d["closure_amplitude_pct"] = float(blink.amplitude * 100.0)
    # 同时补上最小 Openness（更直观）
    d["min_openness"] = float(blink.ela_min)  # 这里 ela_min 存的是 openness_min（见下面的构造）
    return d


def detect_blinks_openness(
        openness: np.ndarray,
        fps: float,
        min_blink_duration: float = BLINK_MIN_DURATION_SEC,
        max_blink_duration: float = BLINK_MAX_DURATION_SEC,
        min_distance_sec: float = BLINK_MIN_DISTANCE_SEC,
        prominence: float = BLINK_PROMINENCE,
        cross_level: float = BLINK_CROSS_LEVEL,
        closed_level: float = BLINK_CLOSED_LEVEL,
) -> List["BlinkEvent"]:
    """
    基于 Openness（0~1）的眨眼检测。

    直观解释：
    - openness 越小 -> 越闭眼
    - closure = 1 - openness 越大 -> 越闭眼
    - 找 closure 的峰值（对应眨眼最闭合的时刻）
    - 再用阈值（cross_level）向左右扩展得到 start/end

    返回：
        BlinkEvent 列表（其中：
            - ela_start/ela_min/ela_end 实际存的是 openness_start / openness_min / openness_end
            - amplitude 存的是 closure_peak = (1 - openness_min) ，范围 0~1
        ）
    """
    openness = np.asarray(openness, dtype=np.float32)
    if openness.size < 5:
        return []

    # 平滑一下再检测（避免噪声峰）
    op_s = _smooth_signal(_interp_nan_1d(openness), fps)
    op_s = np.clip(op_s, 0.0, 1.2)

    closure = 1.0 - np.clip(op_s, 0.0, 1.0)  # 0~1
    distance = max(1, int(min_distance_sec * fps))

    peaks, props = find_peaks(closure, prominence=prominence, distance=distance)
    if peaks is None or len(peaks) == 0:
        return []

    # derivative 转成“每秒变化量”，更好解释速度
    d_op = compute_derivative(op_s) * float(fps)

    blinks: List[BlinkEvent] = []
    for pk in peaks:
        # 以 cross_level 找 start/end
        if closure[pk] < cross_level:
            continue

        s = pk
        while s > 0 and closure[s] >= cross_level:
            s -= 1
        start_idx = max(0, s)

        e = pk
        n = closure.size
        while e < n - 1 and closure[e] >= cross_level:
            e += 1
        end_idx = min(n - 1, e)

        min_idx = int(pk)

        duration = (end_idx - start_idx) / float(fps)
        if duration < min_blink_duration or duration > max_blink_duration:
            continue

        # 分阶段时长
        closing_duration = (min_idx - start_idx) / float(fps)
        reopening_duration = (end_idx - min_idx) / float(fps)

        # 闭合保持：closure >= closed_level 的累计时间
        seg = closure[start_idx:end_idx + 1]
        closed_duration = float(np.sum(seg >= closed_level) / float(fps))

        # 关键数值（更直观都用 openness）
        op_start = float(op_s[start_idx])
        op_min = float(op_s[min_idx])
        op_end = float(op_s[end_idx])

        amplitude = float(1.0 - np.clip(op_min, 0.0, 1.0))  # 0~1 的闭合幅度

        # 速度：closing 用 -d_op 最大值，reopening 用 d_op 最大值
        max_closing_vel = float(np.max(-d_op[start_idx:min_idx + 1])) if min_idx > start_idx else 0.0
        max_reopening_vel = float(np.max(d_op[min_idx:end_idx + 1])) if end_idx > min_idx else 0.0

        # 一个简单比值：幅度 / 速度（只用于相对比较，非必须）
        av_ratio = float(amplitude / (max(max_closing_vel, max_reopening_vel, 1e-6)))

        # 用“闭合面积”（closure 曲线下面积）做一个直观强度指标
        normal_area = float(np.trapz(seg, dx=1.0 / float(fps)))

        blink = BlinkEvent(
            start_idx=int(start_idx),
            min_idx=int(min_idx),
            end_idx=int(end_idx),
            closing_duration=float(closing_duration),
            closed_duration=float(closed_duration),
            reopening_duration=float(reopening_duration),
            amplitude=float(amplitude),
            ela_start=float(op_start),
            ela_min=float(op_min),
            ela_end=float(op_end),
            max_closing_velocity=float(max_closing_vel),
            max_reopening_velocity=float(max_reopening_vel),
            amplitude_velocity_ratio=float(av_ratio),
            normal_area=float(normal_area),
        )
        blinks.append(blink)

    # 按时间排序
    blinks.sort(key=lambda b: b.min_idx)
    return blinks


def pair_left_right_blinks(
        left_blinks: List["BlinkEvent"],
        right_blinks: List["BlinkEvent"],
        fps: float,
        max_delta_sec: float = MAX_LR_PAIR_DELTA_SEC,
) -> Dict:
    """把左右眼眨眼按“最闭合时刻(min_idx)”做同步配对（贪心匹配）。"""
    if not left_blinks or not right_blinks:
        return {
            "pairs": [],
            "unmatched_left": list(range(len(left_blinks))),
            "unmatched_right": list(range(len(right_blinks))),
        }

    left_times = np.array([b.min_idx / float(fps) for b in left_blinks], dtype=np.float32)
    right_times = np.array([b.min_idx / float(fps) for b in right_blinks], dtype=np.float32)

    used_r = set()
    pairs = []
    unmatched_left = []

    for i, t in enumerate(left_times):
        # 找最近的未使用的 right
        diffs = np.abs(right_times - t)
        j = int(np.argmin(diffs))
        if j in used_r or float(diffs[j]) > float(max_delta_sec):
            unmatched_left.append(i)
            continue
        used_r.add(j)
        pairs.append({
            "left_index": int(i),
            "right_index": int(j),
            "delta_sec": float(right_times[j] - t),  # right - left
            "abs_delta_sec": float(abs(right_times[j] - t)),
        })

    unmatched_right = [j for j in range(len(right_blinks)) if j not in used_r]

    # 同步统计
    if pairs:
        abs_d = np.array([p["abs_delta_sec"] for p in pairs], dtype=np.float32)
        sync = {
            "paired_blinks": int(len(pairs)),
            "mean_abs_delta_ms": float(np.mean(abs_d) * 1000.0),
            "median_abs_delta_ms": float(np.median(abs_d) * 1000.0),
            "max_abs_delta_ms": float(np.max(abs_d) * 1000.0),
        }
    else:
        sync = {
            "paired_blinks": 0,
            "mean_abs_delta_ms": 0.0,
            "median_abs_delta_ms": 0.0,
            "max_abs_delta_ms": 0.0,
        }

    return {
        "pairs": pairs,
        "unmatched_left": unmatched_left,
        "unmatched_right": unmatched_right,
        "synchrony": sync,
    }


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
        end_frame: Optional[int] = None,
        compute_ela: bool = False,
) -> Optional[ELASignal]:
    """
    处理视频，提取眼部时序信号。

    ✅ 推荐默认：Openness（眼睛开合度）——更直观、更稳定、更容易解释
    - Openness_raw: (上/下眼睑平均距离) / (眼裂宽度)
    - Openness: 对 Openness_raw 做个体自适应归一化到 0~1

    可选 compute_ela=True：同时计算论文里的 ELA（角度），用于对照/调试（更慢）。

    Returns:
        ELASignal:
            - raw/filtered/derivative: 默认用“左右 Openness 平均”作为主信号
            - left_openness/right_openness: 左/右眼 Openness（0~1）
            - left_ela/right_ela: 仅 compute_ela=True 才有
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频不存在: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspect_ratio = height / max(1, width)

    if end_frame is None:
        end_frame = total_frames
    end_frame = min(end_frame, total_frames)
    start_frame = max(0, start_frame)

    # 跳转到起始帧
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    print(f"📹 处理视频: {os.path.basename(video_path)}")
    print(f"   帧率: {fps:.2f} FPS")
    print(f"   分辨率: {width}x{height}")
    print(f"   处理范围: {start_frame}-{end_frame}")

    detector = get_face_landmarker(model_path)

    left_openness_raw_list = []
    right_openness_raw_list = []

    left_elas = []      # 可选
    right_elas = []     # 可选

    frame_idx = start_frame
    # VIDEO 模式要求 timestamp_ms 单调递增（同一个 detector 贯穿多段视频时也一样）
    # 由于本脚本会在同一进程内复用 detector（加速），如果跨视频 timestamp 重新从 0 开始，会触发：
    #   Input timestamp must be monotonically increasing.
    # 所以这里为每个 model_path 维护一个全局 base_ts_ms，保证跨视频也严格递增。
    global _FACE_LANDMARKER_LAST_TS_MS
    base_ts_ms = int(_FACE_LANDMARKER_LAST_TS_MS.get(model_path, -1)) + 1
    prev_ts_ms = base_ts_ms - 1
    frame_counter = 0  # 相对当前视频的帧计数
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        ts_ms = base_ts_ms + int(round(frame_counter * 1000.0 / fps))
        if ts_ms <= prev_ts_ms:
            ts_ms = prev_ts_ms + 1
        prev_ts_ms = ts_ms
        detection_result = detector.detect_for_video(mp_image, ts_ms)

        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]

            # ---- Openness（2D）----
            left_upper_xy = extract_2d_landmarks(landmarks, LEFT_EYE_UPPER, width, height, aspect_ratio)
            left_lower_xy = extract_2d_landmarks(landmarks, LEFT_EYE_LOWER, width, height, aspect_ratio)
            right_upper_xy = extract_2d_landmarks(landmarks, RIGHT_EYE_UPPER, width, height, aspect_ratio)
            right_lower_xy = extract_2d_landmarks(landmarks, RIGHT_EYE_LOWER, width, height, aspect_ratio)

            left_open_raw = calculate_eye_openness_ratio(left_upper_xy, left_lower_xy)
            right_open_raw = calculate_eye_openness_ratio(right_upper_xy, right_lower_xy)

            left_openness_raw_list.append(left_open_raw)
            right_openness_raw_list.append(right_open_raw)

            # ---- 可选：ELA（3D）----
            if compute_ela:
                left_upper_3d = extract_3d_landmarks(landmarks, LEFT_EYE_UPPER, width, height, aspect_ratio)
                left_lower_3d = extract_3d_landmarks(landmarks, LEFT_EYE_LOWER, width, height, aspect_ratio)
                right_upper_3d = extract_3d_landmarks(landmarks, RIGHT_EYE_UPPER, width, height, aspect_ratio)
                right_lower_3d = extract_3d_landmarks(landmarks, RIGHT_EYE_LOWER, width, height, aspect_ratio)

                left_ela = calculate_ela_for_eye(left_upper_3d, left_lower_3d)
                right_ela = calculate_ela_for_eye(right_upper_3d, right_lower_3d)
                left_elas.append(left_ela)
                right_elas.append(right_ela)
            else:
                left_elas.append(np.nan)
                right_elas.append(np.nan)

        else:
            # 未检测到人脸：沿用上一帧（更稳），没有上一帧就 NaN
            if len(left_openness_raw_list) > 0:
                left_openness_raw_list.append(left_openness_raw_list[-1])
                right_openness_raw_list.append(right_openness_raw_list[-1])
                left_elas.append(left_elas[-1])
                right_elas.append(right_elas[-1])
            else:
                left_openness_raw_list.append(np.nan)
                right_openness_raw_list.append(np.nan)
                left_elas.append(np.nan)
                right_elas.append(np.nan)
        frame_counter += 1
        frame_idx += 1

    # 更新该进程中此 detector 的全局时间戳，保证跨视频继续递增
    _FACE_LANDMARKER_LAST_TS_MS[model_path] = int(prev_ts_ms)

    cap.release()

    n = len(left_openness_raw_list)
    if n == 0:
        return None

    # ---- Openness 后处理：插值、平滑、归一化 ----
    left_raw = np.array(left_openness_raw_list, dtype=np.float32)
    right_raw = np.array(right_openness_raw_list, dtype=np.float32)

    left_raw = _interp_nan_1d(left_raw)
    right_raw = _interp_nan_1d(right_raw)

    left_raw_s = _smooth_signal(left_raw, fps)
    right_raw_s = _smooth_signal(right_raw, fps)

    left_open = _normalize_openness_by_percentiles(left_raw_s)
    right_open = _normalize_openness_by_percentiles(right_raw_s)

    # 主信号：左右平均（你也可以在后续直接用 left_open/right_open 分别检测）
    combined_open = 0.5 * (left_open + right_open)
    combined_open_s = _smooth_signal(combined_open, fps)

    derivative = compute_derivative(combined_open_s)

    timestamps = (np.arange(n, dtype=np.float32) + float(start_frame)) / float(fps)

    # 可选：ELA 输出
    left_ela_arr = np.array(left_elas, dtype=np.float32) if compute_ela else None
    right_ela_arr = np.array(right_elas, dtype=np.float32) if compute_ela else None

    return ELASignal(
        raw=combined_open.astype(np.float32),
        filtered=combined_open_s.astype(np.float32),
        derivative=derivative.astype(np.float32),
        fps=fps,
        timestamps=timestamps,
        left_ela=left_ela_arr,
        right_ela=right_ela_arr,
        left_openness_raw=left_raw,
        right_openness_raw=right_raw,
        left_openness=left_open,
        right_openness=right_open,
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

def visualize_openness_lr_with_blinks(
        ela_signal: ELASignal,
        left_blinks: List["BlinkEvent"],
        right_blinks: List["BlinkEvent"],
        pair_info: Dict,
        output_path: str,
        title_prefix: str = "",
):
    """可视化左右眼 Openness + 检测到的眨眼区间 + 同步时间差分布。"""
    if ela_signal.left_openness is None or ela_signal.right_openness is None:
        return

    t = ela_signal.timestamps
    left = np.asarray(ela_signal.left_openness, dtype=np.float32)
    right = np.asarray(ela_signal.right_openness, dtype=np.float32)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    def _plot_eye(ax, signal, blinks, eye_name: str):
        ax.plot(t, signal, linewidth=1.5, label=f"{eye_name} Openness")
        # 眨眼区间（start-end）涂底色，min 点打标
        for b in blinks:
            ts = t[b.start_idx]
            te = t[b.end_idx]
            tm = t[b.min_idx]
            ax.axvspan(ts, te, alpha=0.15)
            ax.scatter([tm], [signal[b.min_idx]], s=18)

        ax.set_ylim(-0.05, 1.1)
        ax.set_ylabel("Openness (0~1)")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right")

    _plot_eye(axes[0], left, left_blinks, "Left")
    _plot_eye(axes[1], right, right_blinks, "Right")

    # 同步时间差直方图
    pairs = pair_info.get("pairs", [])
    if pairs:
        deltas = np.array([p["delta_sec"] for p in pairs], dtype=np.float32) * 1000.0
        axes[2].hist(deltas, bins=30)
        axes[2].set_ylabel("Count")
        axes[2].set_xlabel("Right - Left (ms)")
        axes[2].grid(True, alpha=0.2)

        sync = pair_info.get("synchrony", {})
        txt = (
            f"Paired: {sync.get('paired_blinks', 0)} | "
            f"Mean |Δ|: {sync.get('mean_abs_delta_ms', 0.0):.1f} ms | "
            f"Median |Δ|: {sync.get('median_abs_delta_ms', 0.0):.1f} ms"
        )
        axes[2].set_title(txt)
    else:
        axes[2].text(0.5, 0.5, "No paired blinks", ha="center", va="center")
        axes[2].set_axis_off()

    main_title = (title_prefix + " " if title_prefix else "") + "Eye Openness (Left/Right) + Blink Detection"
    fig.suptitle(main_title, fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def create_openness_blink_summary_figure(
        left_blinks: List["BlinkEvent"],
        right_blinks: List["BlinkEvent"],
        pair_info: Dict,
        fps: float,
        output_path: str,
        title_prefix: str = "",
):
    """生成更直观的汇总图：左右眼眨眼次数/频率、时长分布、闭合幅度分布、同步差。"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1) 眨眼次数 & 频率
    total_time_min = None
    # 这里用“最大帧索引 / fps”近似时长
    # 如果没有眨眼，就按 1 分钟避免除零（只影响显示）
    max_idx = 0
    for b in (left_blinks + right_blinks):
        max_idx = max(max_idx, b.end_idx)
    total_time_sec = max(1e-6, (max_idx / float(fps)))
    total_time_min = total_time_sec / 60.0

    l_cnt = len(left_blinks)
    r_cnt = len(right_blinks)
    axes[0, 0].bar(["Left", "Right"], [l_cnt, r_cnt])
    axes[0, 0].set_title("Blink Count")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].grid(True, axis="y", alpha=0.2)

    axes[0, 1].bar(["Left", "Right"], [l_cnt / total_time_min, r_cnt / total_time_min])
    axes[0, 1].set_title("Blink Rate (per minute)")
    axes[0, 1].set_ylabel("Blinks/min")
    axes[0, 1].grid(True, axis="y", alpha=0.2)

    # 2) 时长分布
    l_dur = [b.duration for b in left_blinks]
    r_dur = [b.duration for b in right_blinks]
    if l_dur or r_dur:
        if l_dur:
            axes[1, 0].hist(l_dur, bins=25, alpha=0.6, label="Left")
        if r_dur:
            axes[1, 0].hist(r_dur, bins=25, alpha=0.6, label="Right")
        axes[1, 0].set_title("Blink Duration (s)")
        axes[1, 0].set_xlabel("Seconds")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.2)
    else:
        axes[1, 0].text(0.5, 0.5, "No blinks", ha="center", va="center")
        axes[1, 0].set_axis_off()

    # 3) 闭合幅度分布（closure_peak）
    l_amp = [b.amplitude for b in left_blinks]
    r_amp = [b.amplitude for b in right_blinks]
    if l_amp or r_amp:
        if l_amp:
            axes[1, 1].hist(np.array(l_amp) * 100.0, bins=25, alpha=0.6, label="Left")
        if r_amp:
            axes[1, 1].hist(np.array(r_amp) * 100.0, bins=25, alpha=0.6, label="Right")
        axes[1, 1].set_title("Closure Amplitude (%)")
        axes[1, 1].set_xlabel("% (higher = more closed)")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.2)
    else:
        axes[1, 1].text(0.5, 0.5, "No blinks", ha="center", va="center")
        axes[1, 1].set_axis_off()

    title = (title_prefix + " " if title_prefix else "") + "Blink Summary (Openness)"
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)

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
    单个视频的眨眼分析入口（改为更直观的 Openness）：

    1) 提取左右眼 Openness（0~1）
    2) 左右眼分别做眨眼检测（基于 closure = 1 - openness）
    3) 左右配对，得到同步时间差
    4) 输出图 + JSON（左右眼分别统计）

    注意：
    - 参数 fps 仅保留兼容（实际以视频文件读取到的 fps 为准）
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频不存在: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    # 文件名前缀
    prefix = _safe_prefix(examination_id, action_name)

    # ========== 1) 提取 Openness 信号 ==========
    ela_signal = process_video_ela(
        video_path=video_path,
        model_path=model_path,
        start_frame=start_frame,
        end_frame=end_frame,
        compute_ela=False,
    )
    if ela_signal is None or ela_signal.left_openness is None or ela_signal.right_openness is None:
        return None

    fps_real = float(ela_signal.fps)

    # ========== 2) 左右眼分别检测眨眼 ==========
    left_blinks = detect_blinks_openness(ela_signal.left_openness, fps_real)
    right_blinks = detect_blinks_openness(ela_signal.right_openness, fps_real)

    # ========== 3) 左右配对（同步） ==========
    pair_info = pair_left_right_blinks(left_blinks, right_blinks, fps_real)

    # ========== 4) 统计 ==========
    left_stats = summarize_blink_sequence(ela_signal, left_blinks)
    right_stats = summarize_blink_sequence(ela_signal, right_blinks)

    # 平均值作为整体参考
    num_blinks_avg = int(round((len(left_blinks) + len(right_blinks)) / 2.0))
    blink_rate_avg = float((left_stats.get("blink_rate_per_minute", 0.0) + right_stats.get("blink_rate_per_minute", 0.0)) / 2.0)

    # ========== 5) 可视化 ==========
    openness_plot_path = os.path.join(output_dir, f"{prefix}_openness_lr.png")
    summary_plot_path = os.path.join(output_dir, f"{prefix}_blink_summary_openness.png")

    title_prefix = f"{patient_id} | {action_name}"
    visualize_openness_lr_with_blinks(
        ela_signal=ela_signal,
        left_blinks=left_blinks,
        right_blinks=right_blinks,
        pair_info=pair_info,
        output_path=openness_plot_path,
        title_prefix=title_prefix,
    )
    create_openness_blink_summary_figure(
        left_blinks=left_blinks,
        right_blinks=right_blinks,
        pair_info=pair_info,
        fps=fps_real,
        output_path=summary_plot_path,
        title_prefix=title_prefix,
    )

    # ========== 6) 组织输出 JSON ==========
    result = {
        "patient_id": patient_id,
        "action_name": action_name,
        "examination_id": examination_id,
        "video_path": video_path,
        "start_frame": int(start_frame),
        "end_frame": int(end_frame) if end_frame is not None else None,
        "fps": float(fps_real),
        "signal_type": "openness",
        "num_blinks": int(num_blinks_avg),
        "blink_rate_per_minute": float(blink_rate_avg),

        "left": left_stats,
        "right": right_stats,
        "synchrony": pair_info.get("synchrony", {}),

        "pairs": pair_info.get("pairs", []),
        "unmatched_left": pair_info.get("unmatched_left", []),
        "unmatched_right": pair_info.get("unmatched_right", []),

        "left_blinks": [_blink_event_to_dict(b, "left") for b in left_blinks],
        "right_blinks": [_blink_event_to_dict(b, "right") for b in right_blinks],

        "plots": {
            "openness_lr": openness_plot_path,
            "summary": summary_plot_path,
        },
    }

    # 保存 JSON
    json_path = os.path.join(output_dir, f"{prefix}_blink_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(convert_numpy_types(result), f, ensure_ascii=False, indent=2)

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

                        # 兼容旧字段（整体参考）
                        "num_blinks": r.get("num_blinks", 0),
                        "blink_rate_per_minute": r.get("blink_rate_per_minute", 0.0),

                        # 新增：左右眼分别
                        "num_blinks_left": r.get("left", {}).get("num_blinks", 0),
                        "num_blinks_right": r.get("right", {}).get("num_blinks", 0),
                        "blink_rate_left": r.get("left", {}).get("blink_rate_per_minute", 0.0),
                        "blink_rate_right": r.get("right", {}).get("blink_rate_per_minute", 0.0),

                        # 新增：同步
                        "paired_blinks": r.get("synchrony", {}).get("paired_blinks", 0),
                        "mean_abs_delta_ms": r.get("synchrony", {}).get("mean_abs_delta_ms", 0.0),
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

                        # 兼容旧字段（整体参考）
                        "num_blinks": r.get("num_blinks", 0),
                        "blink_rate_per_minute": r.get("blink_rate_per_minute", 0.0),

                        # 新增：左右眼分别
                        "num_blinks_left": r.get("left", {}).get("num_blinks", 0),
                        "num_blinks_right": r.get("right", {}).get("num_blinks", 0),
                        "blink_rate_left": r.get("left", {}).get("blink_rate_per_minute", 0.0),
                        "blink_rate_right": r.get("right", {}).get("blink_rate_per_minute", 0.0),

                        # 新增：同步
                        "paired_blinks": r.get("synchrony", {}).get("paired_blinks", 0),
                        "mean_abs_delta_ms": r.get("synchrony", {}).get("mean_abs_delta_ms", 0.0),
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
    OUTPUT_DIR = "/Users/cuijinglei/Documents/facialPalsy/HGFA/eyelid_blink_openness"

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