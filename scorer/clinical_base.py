#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
临床分级基础模块 - Clinical Grading Base Module
================================================

包含:
- Landmark索引定义
- 基础几何计算函数
- 数据类定义
- 数据库操作函数
- 可视化基础函数

"""

import os
import sqlite3
import json
import cv2
import numpy as np
import math
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from thresholds import THR

# =============================================================================
# Landmark 索引定义 (MediaPipe 478点)
# =============================================================================

class LM:
    """MediaPipe FaceLandmarker 478点索引"""

    # ========== 眼部 ==========
    EYE_INNER_L = 362  # 左眼内眦
    EYE_INNER_R = 133  # 右眼内眦
    EYE_OUTER_L = 263  # 左眼外眦
    EYE_OUTER_R = 33  # 右眼外眦

    EYE_TOP_L = 386  # 左眼上睑
    EYE_BOT_L = 374  # 左眼下睑
    EYE_TOP_R = 159  # 右眼上睑
    EYE_BOT_R = 145  # 右眼下睑

    # 眼轮廓点
    EYE_CONTOUR_L = [263, 466, 388, 387, 386, 385, 384, 398,
                     362, 382, 381, 380, 374, 373, 390, 249]
    EYE_CONTOUR_R = [33, 246, 161, 160, 159, 158, 157, 173,
                     133, 155, 154, 153, 145, 144, 163, 7]

    # EAR计算点
    EAR_L = [263, 386, 387, 362, 373, 374]
    EAR_R = [33, 159, 158, 133, 144, 145]

    # ========== 眉毛 ==========
    BROW_L = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
    BROW_R = [107, 66, 105, 63, 70, 46, 53, 52, 65, 55]
    BROW_CENTER_L = 282
    BROW_CENTER_R = 52
    BROW_INNER_L = 336
    BROW_INNER_R = 107
    BROW_OUTER_L = 285
    BROW_OUTER_R = 55

    # ========== 嘴部 ==========
    MOUTH_L = 291  # 左嘴角
    MOUTH_R = 61  # 右嘴角
    LIP_TOP_CENTER = 0  # 上唇上边界中心
    LIP_TOP = 13  # 上唇下边界中心
    LIP_BOT = 14  # 下唇上边界中心
    LIP_BOT_CENTER = 17  # 下唇下边界中心

    # 嘴唇轮廓
    OUTER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
                 375, 321, 405, 314, 17, 84, 181, 91, 146]
    INNER_LIP = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
                 415, 310, 311, 312, 13, 82, 81, 80, 191]

    # ========== 嘴唇左右区域（用于面中线对称性分析）==========
    # 患者左侧嘴唇（图像右侧，x值较大）
    UPPER_LIP_L = [267, 269, 270, 409, 291, 308, 415, 310, 311, 312]
    LOWER_LIP_L = [317, 402, 318, 324, 308, 291, 375, 321, 405, 314]
    LIP_L = [267, 269, 270, 409, 291, 308, 415, 310, 311, 312,
             317, 402, 318, 324, 375, 321, 405, 314]
    INNER_LIP_L = [13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14]
    MOUTH_CORNER_L = 291

    # 患者右侧嘴唇（图像左侧，x值较小）
    UPPER_LIP_R = [37, 39, 40, 185, 61, 78, 191, 80, 81, 82]
    LOWER_LIP_R = [87, 178, 88, 95, 78, 61, 146, 91, 181, 84]
    LIP_R = [37, 39, 40, 185, 61, 78, 191, 80, 81, 82,
             87, 178, 88, 95, 146, 91, 181, 84]
    INNER_LIP_R = [13,  82,  81,  80, 191, 78,  95,  88, 178,  87, 14]
    MOUTH_CORNER_R = 61

    # ========== 口角角度 ==========
    ORAL_CORNER_R = 78  # A点: 右嘴角
    ORAL_CORNER_L = 308  # B点: 左嘴角
    LIP_PEAK_R = 37  # C点: 右上唇峰
    LIP_PEAK_L = 267  # D点: 左上唇峰
    ORAL_E = 82  # E点: mouth orifice右侧
    ORAL_F = 312  # F点: mouth orifice左侧

    # ========== 面中线参考点 ==========
    FOREHEAD = 10
    NOSE_TIP = 4
    CHIN = 152

    # ========== 鼻部 ==========
    NOSE_ALA_L = 294  # 左鼻翼
    NOSE_ALA_R = 64  # 右鼻翼
    NOSE_TIP_TOP = 4  # 鼻尖
    NOSE_BRIDGE = 168  # 鼻梁

    # ========== 面颊 ==========
    CHEEK_L = [425, 426, 427, 411, 280]
    CHEEK_R = [205, 206, 207, 187, 50]

    BLOW_CHEEK_L = [280, 376, 433, 367, 364, 287, 410, 423, 266, ]
    BLOW_CHEEK_R = [50, 147, 213, 138, 135, 57, 92, 203, 36, ]

    # =============================================================================


# JSON 序列化安全转换（处理 numpy.bool_ / numpy.float32 等）
# =============================================================================
def make_json_serializable(obj: Any) -> Any:
    """递归将对象转换为 JSON 可序列化的 Python 原生类型。"""
    # numpy 标量
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)

    # numpy 数组
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # Path
    if isinstance(obj, Path):
        return str(obj)

    # dict / list / tuple
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]

    return obj


# =============================================================================
# 统一尺度归一化 (Scale Normalization)
# =============================================================================

def compute_icd_cached(landmarks, w: int, h: int, cache: dict = None, key: str = None) -> float:
    """
    计算 ICD，支持缓存避免重复计算

    Args:
        landmarks: 面部关键点
        w, h: 图像尺寸
        cache: 缓存字典（可选）
        key: 缓存键名（可选）

    Returns:
        ICD 值
    """
    if cache is not None and key is not None and key in cache:
        return cache[key]

    icd = compute_icd(landmarks, w, h)

    if cache is not None and key is not None:
        cache[key] = icd

    return icd


def compute_scale_to_baseline(current_landmarks, baseline_landmarks, w: int, h: int,
                              icd_base: float = None, icd_current: float = None) -> float:
    """
    计算将当前帧缩放到 baseline 尺度的比例因子（优化版）

    scale = ICD_base / ICD_current

    用法: 当前帧距离 × scale = baseline尺度下的距离

    Args:
        current_landmarks: 当前帧 landmarks
        baseline_landmarks: 静息帧 landmarks
        w, h: 图像尺寸
        icd_base: 预计算的基线 ICD（可选，避免重复计算）
        icd_current: 预计算的当前帧 ICD（可选）

    Returns:
        scale 比例因子 (ICD_base / ICD_current)
    """
    if baseline_landmarks is None or current_landmarks is None:
        return 1.0

    # 使用传入的 ICD 值，或计算
    if icd_base is None:
        icd_base = compute_icd(baseline_landmarks, w, h)
    if icd_current is None:
        icd_current = compute_icd(current_landmarks, w, h)

    if icd_current < 1e-6:
        return 1.0

    return icd_base / icd_current


def scale_distance(distance: float, scale: float) -> float:
    """将距离缩放到 baseline 尺度"""
    return distance * scale


def scale_point(point: Tuple[float, float], scale: float,
                anchor: Tuple[float, float] = None) -> Tuple[float, float]:
    """
    将点坐标缩放到 baseline 尺度

    Args:
        point: 当前帧坐标
        scale: 缩放比例
        anchor: 缩放锚点（默认为原点）

    Returns:
        缩放后的坐标
    """
    if anchor is None:
        return (point[0] * scale, point[1] * scale)
    else:
        dx = point[0] - anchor[0]
        dy = point[1] - anchor[1]
        return (anchor[0] + dx * scale, anchor[1] + dy * scale)


class ScaledMetrics:
    """
    统一尺度的指标容器

    存储已缩放到 baseline 尺度的指标，方便直接对比
    """

    def __init__(self, scale: float = 1.0):
        self.scale = scale
        self._raw = {}  # 原始值
        self._scaled = {}  # 缩放后的值

    def add(self, name: str, raw_value: float) -> float:
        """添加一个距离指标，自动缩放"""
        self._raw[name] = raw_value
        scaled = raw_value * self.scale
        self._scaled[name] = scaled
        return scaled

    def get_raw(self, name: str) -> float:
        return self._raw.get(name, 0.0)

    def get_scaled(self, name: str) -> float:
        return self._scaled.get(name, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scale": self.scale,
            "raw": self._raw.copy(),
            "scaled": self._scaled.copy()
        }


# =============================================================================
# 批量处理函数 - NumPy 向量化
# =============================================================================

def batch_compute_ear(landmarks_seq: list, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    批量计算整个序列的 EAR 值（向量化优化）

    Args:
        landmarks_seq: landmarks 序列
        w, h: 图像尺寸

    Returns:
        (left_ear_array, right_ear_array)
    """
    n = len(landmarks_seq)
    left_ear = np.full(n, np.nan)
    right_ear = np.full(n, np.nan)

    for i, lm in enumerate(landmarks_seq):
        if lm is not None:
            left_ear[i] = compute_ear(lm, w, h, True)
            right_ear[i] = compute_ear(lm, w, h, False)

    return left_ear, right_ear


def batch_compute_mouth_width(landmarks_seq: list, w: int, h: int) -> np.ndarray:
    """
    批量计算嘴宽

    Returns:
        mouth_width_array
    """
    n = len(landmarks_seq)
    mouth_width = np.full(n, np.nan)

    for i, lm in enumerate(landmarks_seq):
        if lm is not None:
            l_corner = pt2d(lm[LM.MOUTH_L], w, h)
            r_corner = pt2d(lm[LM.MOUTH_R], w, h)
            mouth_width[i] = dist(l_corner, r_corner)

    return mouth_width


def batch_compute_palpebral_height(landmarks_seq: list, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    批量计算眼睑裂高度

    Returns:
        (left_height_array, right_height_array)
    """
    n = len(landmarks_seq)
    left_height = np.full(n, np.nan)
    right_height = np.full(n, np.nan)

    for i, lm in enumerate(landmarks_seq):
        if lm is not None:
            left_height[i] = compute_palpebral_height(lm, w, h, True)
            right_height[i] = compute_palpebral_height(lm, w, h, False)

    return left_height, right_height


def find_peak_by_metric(metric_array: np.ndarray, mode: str = 'max') -> int:
    """
    根据指标数组找峰值帧索引

    Args:
        metric_array: 指标数组
        mode: 'max' 或 'min'

    Returns:
        峰值帧索引
    """
    valid_mask = np.isfinite(metric_array)
    if not np.any(valid_mask):
        return 0

    if mode == 'max':
        return int(np.nanargmax(metric_array))
    else:
        return int(np.nanargmin(metric_array))


# =============================================================================
# 基础几何计算函数
# =============================================================================

def pt2d(lm, w: int, h: int) -> Tuple[float, float]:
    """将归一化坐标转换为像素坐标"""
    return (lm.x * w, lm.y * h)


def pt3d(lm, w: int, h: int) -> Tuple[float, float, float]:
    """将归一化坐标转换为3D像素坐标"""
    return (lm.x * w, lm.y * h, lm.z * w)


def pts2d(landmarks, indices: List[int], w: int, h: int) -> np.ndarray:
    """批量转换点坐标"""
    return np.array([pt2d(landmarks[i], w, h) for i in indices])


def dist(p1, p2) -> float:
    """计算两点间的欧氏距离"""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def dist3d(p1, p2) -> float:
    """计算3D两点间的欧氏距离"""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


def polygon_area(points: np.ndarray) -> float:
    """计算多边形面积 (Shoelace公式)"""
    n = len(points)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i, 0] * points[j, 1]
        area -= points[j, 0] * points[i, 1]
    return abs(area) / 2.0


def point_to_line_signed_distance(point: Tuple[float, float],
                                  line_p1: Tuple[float, float],
                                  line_p2: Tuple[float, float]) -> float:
    """
    计算点到直线的有符号距离

    在图像坐标系中(Y轴向下):
    - 返回正值: 点在直线下方
    - 返回负值: 点在直线上方
    """
    dx = line_p2[0] - line_p1[0]
    dy = line_p2[1] - line_p1[1]
    line_len = math.sqrt(dx * dx + dy * dy)
    if line_len < 1e-9:
        return 0.0
    px = point[0] - line_p1[0]
    py = point[1] - line_p1[1]
    cross = dx * py - dy * px
    return cross / line_len


def angle_between_vectors(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """计算两个向量之间的夹角(度), 返回0-180度"""
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    if mag1 < 1e-9 or mag2 < 1e-9:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def compute_centroid(points: np.ndarray) -> Tuple[float, float]:
    """计算点集的质心"""
    return (float(np.mean(points[:, 0])), float(np.mean(points[:, 1])))


# 建议添加到 clinical_base.py 或 geometry_utils.py

def kabsch_rigid_transform(P: np.ndarray, Q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算刚体变换：把 P(当前帧稳定点) 对齐到 Q(基线稳定点)

    Args:
        P: 当前帧点集 (N, 3)
        Q: 基线帧点集 (N, 3)

    Returns:
        R: 旋转矩阵 (3, 3)
        t: 平移向量 (3,)
    """
    if P is None or Q is None or P.shape[0] < 3 or Q.shape[0] < 3:
        return None, None

    Pc = P.mean(axis=0)
    Qc = Q.mean(axis=0)
    X = P - Pc
    Y = Q - Qc

    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # 防止反射
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = Qc - (R @ Pc)
    return R, t


def apply_rigid_transform(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """对 (N,3) 点集应用刚体变换"""
    return (R @ points.T).T + t


# =============================================================================
# 通用指标计算函数
# =============================================================================

def compute_icd(landmarks, w: int, h: int) -> float:
    """计算内眦间距 (Inter-Canthal Distance)"""
    l_inner = pt2d(landmarks[LM.EYE_INNER_L], w, h)
    r_inner = pt2d(landmarks[LM.EYE_INNER_R], w, h)
    return dist(l_inner, r_inner)


def compute_eye_area(landmarks, w: int, h: int, left: bool = True) -> Tuple[float, np.ndarray]:
    """计算眼睛面积"""
    contour = LM.EYE_CONTOUR_L if left else LM.EYE_CONTOUR_R
    points = pts2d(landmarks, contour, w, h)
    return polygon_area(points), points


def compute_ear(landmarks, w: int, h: int, left: bool = True) -> float:
    """计算眼睛纵横比 (Eye Aspect Ratio)"""
    pts_idx = LM.EAR_L if left else LM.EAR_R
    p = [pt2d(landmarks[i], w, h) for i in pts_idx]
    v1 = dist(p[1], p[5])
    v2 = dist(p[2], p[4])
    h_dist = dist(p[0], p[3])
    if h_dist < 1e-9:
        return 0.0
    return (v1 + v2) / (2.0 * h_dist)


def compute_palpebral_height(landmarks, w: int, h: int, left: bool = True) -> float:
    """计算眼睑裂高度"""
    if left:
        top = pt2d(landmarks[LM.EYE_TOP_L], w, h)
        bot = pt2d(landmarks[LM.EYE_BOT_L], w, h)
    else:
        top = pt2d(landmarks[LM.EYE_TOP_R], w, h)
        bot = pt2d(landmarks[LM.EYE_BOT_R], w, h)
    return dist(top, bot)


def compute_palpebral_width(landmarks, w: int, h: int, left: bool = True) -> float:
    """计算眼睑裂宽度"""
    if left:
        inner = pt2d(landmarks[LM.EYE_INNER_L], w, h)
        outer = pt2d(landmarks[LM.EYE_OUTER_L], w, h)
    else:
        inner = pt2d(landmarks[LM.EYE_INNER_R], w, h)
        outer = pt2d(landmarks[LM.EYE_OUTER_R], w, h)
    return dist(inner, outer)


def compute_brow_height(landmarks, w: int, h: int, left: bool = True) -> float:
    """计算眉毛高度 (眉毛中心到眼睛内眦的Y轴距离)"""
    if left:
        brow = pt2d(landmarks[LM.BROW_CENTER_L], w, h)
        eye = pt2d(landmarks[LM.EYE_INNER_L], w, h)
    else:
        brow = pt2d(landmarks[LM.BROW_CENTER_R], w, h)
        eye = pt2d(landmarks[LM.EYE_INNER_R], w, h)
    return abs(brow[1] - eye[1])


def compute_brow_position(landmarks, w: int, h: int, left: bool = True) -> Dict[str, float]:
    """计算眉毛位置详细信息"""
    if left:
        inner = pt2d(landmarks[LM.BROW_INNER_L], w, h)
        center = pt2d(landmarks[LM.BROW_CENTER_L], w, h)
        outer = pt2d(landmarks[LM.BROW_OUTER_L], w, h)
        eye_inner = pt2d(landmarks[LM.EYE_INNER_L], w, h)
    else:
        inner = pt2d(landmarks[LM.BROW_INNER_R], w, h)
        center = pt2d(landmarks[LM.BROW_CENTER_R], w, h)
        outer = pt2d(landmarks[LM.BROW_OUTER_R], w, h)
        eye_inner = pt2d(landmarks[LM.EYE_INNER_R], w, h)

    return {
        "inner_height": abs(inner[1] - eye_inner[1]),
        "center_height": abs(center[1] - eye_inner[1]),
        "outer_height": abs(outer[1] - eye_inner[1]),
        "brow_width": dist(inner, outer),
        "inner_y": inner[1],
        "center_y": center[1],
        "outer_y": outer[1]
    }


# =============================================================================
# 眉眼距相关指标
# =============================================================================

def compute_brow_centroid(landmarks, w: int, h: int, left: bool = True) -> Tuple[float, float]:
    """
    计算眉毛质心

    使用眉毛所有关键点计算质心
    """
    brow_indices = LM.BROW_L if left else LM.BROW_R
    brow_points = pts2d(landmarks, brow_indices, w, h)
    return compute_centroid(brow_points)


def compute_brow_eye_distance(landmarks, w: int, h: int, left: bool = True) -> Dict[str, Any]:
    """
    计算眉眼距

    定义：
    - 眉眼距 = “眉毛质心” 到 “眼部水平线(双眼内眦点连线)” 的垂直距离（像素）
    - 同时给出眼部水平线用于可视化：方向取内眦连线，线段端点延长到眉毛最左(300)与最右(70)的对齐范围

    Returns:
        Dict包含:
        - distance: 眉眼距(垂直距离, px)
        - brow_centroid: 眉毛质心
        - eye_inner_l / eye_inner_r: 左/右内眦点
        - foot: 垂足点（眉毛质心投影到眼水平线上的点）
        - eye_line_p0 / eye_line_p1: 画线用端点（与眉毛左右极值对齐后的线段端点）
        - brow_extreme_left / brow_extreme_right: 眉毛极值点(300/70)
        - eye_inner: 为兼容旧代码保留（left=True 返回左内眦；否则右内眦）
    """
    # 眼内眦点（用于定义“眼部水平线”）
    eye_inner_l = pt2d(landmarks[LM.EYE_INNER_L], w, h)
    eye_inner_r = pt2d(landmarks[LM.EYE_INNER_R], w, h)

    # 眉毛质心（左右分别算）
    brow_centroid = compute_brow_centroid(landmarks, w, h, left)

    x1, y1 = float(eye_inner_l[0]), float(eye_inner_l[1])
    x2, y2 = float(eye_inner_r[0]), float(eye_inner_r[1])
    cx, cy = float(brow_centroid[0]), float(brow_centroid[1])

    vx, vy = (x2 - x1), (y2 - y1)
    denom = vx * vx + vy * vy

    # 防止极端退化
    if denom < 1e-9:
        # 退化时回退为“到对应内眦点的距离”
        eye_inner = eye_inner_l if left else eye_inner_r
        return {
            "distance": dist(eye_inner, brow_centroid),
            "eye_inner": eye_inner,
            "brow_centroid": brow_centroid,
            "eye_inner_l": eye_inner_l,
            "eye_inner_r": eye_inner_r,
            "foot": eye_inner,
            "eye_line_p0": eye_inner_l,
            "eye_line_p1": eye_inner_r,
            "brow_extreme_left": pt2d(landmarks[300], w, h),
            "brow_extreme_right": pt2d(landmarks[70], w, h),
        }

    # 垂足（投影点）
    t = ((cx - x1) * vx + (cy - y1) * vy) / denom
    foot = (x1 + t * vx, y1 + t * vy)

    # 垂直距离（点到直线距离）
    # |v x (c - p1)| / |v|
    distance = abs(vx * (cy - y1) - vy * (cx - x1)) / (denom ** 0.5)

    # 眉毛左右极值点（用于“画眼水平线的端点对齐范围”）
    brow_extreme_left = pt2d(landmarks[300], w, h)  # 眉毛最左点
    brow_extreme_right = pt2d(landmarks[70], w, h)  # 眉毛最右点

    def _point_on_eye_line_aligned_to_anchor(anchor_xy):
        ax, ay = float(anchor_xy[0]), float(anchor_xy[1])
        # 用变化更大的轴求参数，避免除零
        if abs(vx) >= abs(vy) and abs(vx) > 1e-6:
            tt = (ax - x1) / vx
        elif abs(vy) > 1e-6:
            tt = (ay - y1) / vy
        else:
            tt = 0.0
        return (x1 + tt * vx, y1 + tt * vy)

    eye_line_p0 = _point_on_eye_line_aligned_to_anchor(brow_extreme_left)
    eye_line_p1 = _point_on_eye_line_aligned_to_anchor(brow_extreme_right)

    eye_inner = eye_inner_l if left else eye_inner_r

    return {
        "distance": float(distance),
        "eye_inner": eye_inner,
        "brow_centroid": brow_centroid,

        "eye_inner_l": eye_inner_l,
        "eye_inner_r": eye_inner_r,
        "foot": (float(foot[0]), float(foot[1])),

        "eye_line_p0": (float(eye_line_p0[0]), float(eye_line_p0[1])),
        "eye_line_p1": (float(eye_line_p1[0]), float(eye_line_p1[1])),

        "brow_extreme_left": brow_extreme_left,
        "brow_extreme_right": brow_extreme_right,
    }


def compute_brow_eye_distance_ratio(landmarks, w: int, h: int) -> Dict[str, Any]:
    """
    计算双侧眉眼距比

    左侧眉眼距 / 右侧眉眼距

    Returns:
        Dict包含:
        - ratio: 眉眼距比值
        - left_distance: 左侧眉眼距
        - right_distance: 右侧眉眼距
        - left_eye_inner: 左眼内眦坐标
        - right_eye_inner: 右眼内眦坐标
        - left_brow_centroid: 左眉毛质心坐标
        - right_brow_centroid: 右眉毛质心坐标
    """
    left_result = compute_brow_eye_distance(landmarks, w, h, left=True)
    right_result = compute_brow_eye_distance(landmarks, w, h, left=False)

    left_dist = left_result["distance"]
    right_dist = right_result["distance"]

    ratio = left_dist / right_dist if right_dist > 1e-9 else 1.0

    return {
        "ratio": ratio,
        "left_distance": left_dist,
        "right_distance": right_dist,
        "left_eye_inner": left_result["eye_inner"],
        "right_eye_inner": right_result["eye_inner"],
        "left_brow_centroid": left_result["brow_centroid"],
        "right_brow_centroid": right_result["brow_centroid"]
    }


def compute_brow_eye_distance_change(current_landmarks, baseline_landmarks,
                                     w: int, h: int, left: bool = True) -> Dict[str, Any]:
    """
    计算眉眼距变化度

    当前帧眉眼距 - 基线帧眉眼距

    Args:
        current_landmarks: 当前帧landmarks
        baseline_landmarks: 基线帧landmarks (通常是NeutralFace)
        w, h: 图像尺寸
        left: 是否左侧

    Returns:
        Dict包含:
        - change: 变化度 (像素), 正值表示眉眼距增大(眉毛上抬)
        - current_distance: 当前眉眼距
        - baseline_distance: 基线眉眼距
    """
    current_result = compute_brow_eye_distance(current_landmarks, w, h, left)
    baseline_result = compute_brow_eye_distance(baseline_landmarks, w, h, left)

    change = current_result["distance"] - baseline_result["distance"]

    return {
        "change": change,
        "current_distance": current_result["distance"],
        "baseline_distance": baseline_result["distance"],
        "current_eye_inner": current_result["eye_inner"],
        "current_brow_centroid": current_result["brow_centroid"],
        "baseline_eye_inner": baseline_result["eye_inner"],
        "baseline_brow_centroid": baseline_result["brow_centroid"]
    }


def compute_brow_eye_distance_change_ratio(current_landmarks, baseline_landmarks,
                                           w: int, h: int) -> Dict[str, Any]:
    """
    计算双侧眉眼距变化度比

    左侧眉眼距变化度 / 右侧眉眼距变化度

    Args:
        current_landmarks: 当前帧landmarks
        baseline_landmarks: 基线帧landmarks
        w, h: 图像尺寸

    Returns:
        Dict包含:
        - ratio: 变化度比值
        - left_change: 左侧变化度
        - right_change: 右侧变化度
    """
    left_result = compute_brow_eye_distance_change(current_landmarks, baseline_landmarks, w, h, left=True)
    right_result = compute_brow_eye_distance_change(current_landmarks, baseline_landmarks, w, h, left=False)

    left_change = left_result["change"]
    right_change = right_result["change"]

    # 比值计算：需要处理分母为0或接近0的情况
    if abs(right_change) > 1e-9:
        ratio = left_change / right_change
    elif abs(left_change) < 1e-9:
        ratio = 1.0  # 两边都没变化
    else:
        ratio = float('inf') if left_change > 0 else float('-inf')  # 一边变化一边不变

    return {
        "ratio": ratio,
        "left_change": left_change,
        "right_change": right_change,
        "left_current_distance": left_result["current_distance"],
        "left_baseline_distance": left_result["baseline_distance"],
        "right_current_distance": right_result["current_distance"],
        "right_baseline_distance": right_result["baseline_distance"]
    }


# =============================================================================
# 嘴部指标
# =============================================================================

def compute_mouth_metrics(landmarks, w: int, h: int) -> Dict[str, Any]:
    """计算嘴部度量"""
    l_corner = pt2d(landmarks[LM.MOUTH_L], w, h)
    r_corner = pt2d(landmarks[LM.MOUTH_R], w, h)
    lip_top = pt2d(landmarks[LM.LIP_TOP], w, h)
    lip_bot = pt2d(landmarks[LM.LIP_BOT], w, h)

    return {
        "width": dist(l_corner, r_corner),
        "height": dist(lip_top, lip_bot),
        "left_corner": l_corner,
        "right_corner": r_corner,
        "top_center": lip_top,
        "bottom_center": lip_bot
    }


def compute_lip_seal_distance(landmarks, w: int, h: int) -> Dict[str, Any]:
    """
    计算唇密封距离 (用于BlowCheek关键帧检测)

    计算方法:
    - 上唇:  LIP_TOP (13) 到 下唇: LIP_BOT (14) 的距离
    - 总距离: 上唇到下唇距离

    鼓腮时嘴唇紧闭，此距离最小
    """
    upper_inner = pt2d(landmarks[LM.LIP_TOP], w, h)  # 13
    lower_inner = pt2d(landmarks[LM.LIP_BOT], w, h)  # 14

    middle_dist = dist(upper_inner, lower_inner)

    return {
        "middle_distance": middle_dist,
        "upper_inner": upper_inner,
        "lower_inner": lower_inner,
    }


def compute_nlf_length(landmarks, w: int, h: int, left: bool = True) -> float:
    """计算鼻唇沟长度"""
    if left:
        ala = pt2d(landmarks[LM.NOSE_ALA_L], w, h)
        corner = pt2d(landmarks[LM.MOUTH_L], w, h)
    else:
        ala = pt2d(landmarks[LM.NOSE_ALA_R], w, h)
        corner = pt2d(landmarks[LM.MOUTH_R], w, h)
    return dist(ala, corner)


def compute_nose_wrinkle_metrics(landmarks, w: int, h: int) -> Dict[str, Any]:
    """
    计算鼻皱纹相关指标 (用于ShrugNose)

    主要测量鼻翼与鼻梁的相对位置变化
    """
    nose_tip = pt2d(landmarks[LM.NOSE_TIP], w, h)
    nose_bridge = pt2d(landmarks[LM.NOSE_BRIDGE], w, h)
    left_ala = pt2d(landmarks[LM.NOSE_ALA_L], w, h)
    right_ala = pt2d(landmarks[LM.NOSE_ALA_R], w, h)

    # 鼻翼间距
    ala_width = dist(left_ala, right_ala)

    # 鼻翼到鼻梁的距离
    left_ala_to_bridge = dist(left_ala, nose_bridge)
    right_ala_to_bridge = dist(right_ala, nose_bridge)

    return {
        "ala_width": ala_width,
        "left_ala_to_bridge": left_ala_to_bridge,
        "right_ala_to_bridge": right_ala_to_bridge,
        "ala_ratio": left_ala_to_bridge / right_ala_to_bridge if right_ala_to_bridge > 1e-9 else 1.0,
        "nose_tip": nose_tip,
        "nose_bridge": nose_bridge,
        "left_ala": left_ala,
        "right_ala": right_ala
    }


# =============================================================================
# 口角角度计算 (Oral Commissure Lift论文)
# =============================================================================

@dataclass
class OralAngleMeasure:
    """口角角度测量结果"""
    A: Tuple[float, float]  # 右嘴角
    B: Tuple[float, float]  # 左嘴角
    C: Tuple[float, float]  # 右上唇峰
    D: Tuple[float, float]  # 左上唇峰
    E: Tuple[float, float]  # mouth orifice右侧
    F: Tuple[float, float]  # mouth orifice左侧
    O: Tuple[float, float]  # E和F的中点

    AOE_angle: float  # 右口角角度: 负值=下垂, 正值=上提
    BOF_angle: float  # 左口角角度: 负值=下垂, 正值=上提
    angle_diff: float
    angle_asymmetry: float

    A_dist_to_EF: float = 0.0
    B_dist_to_EF: float = 0.0


def compute_oral_angle(landmarks, w: int, h: int) -> OralAngleMeasure:
    """计算口角角度"""
    A = pt2d(landmarks[LM.ORAL_CORNER_R], w, h)
    B = pt2d(landmarks[LM.ORAL_CORNER_L], w, h)
    C = pt2d(landmarks[LM.LIP_PEAK_R], w, h)
    D = pt2d(landmarks[LM.LIP_PEAK_L], w, h)
    E = pt2d(landmarks[LM.ORAL_E], w, h)
    F = pt2d(landmarks[LM.ORAL_F], w, h)
    O = ((E[0] + F[0]) / 2.0, (E[1] + F[1]) / 2.0)

    A_dist = point_to_line_signed_distance(A, E, F)
    B_dist = point_to_line_signed_distance(B, E, F)

    OA_horiz = abs(A[0] - O[0])
    OB_horiz = abs(B[0] - O[0])

    if OA_horiz > 1e-9:
        AOE_angle = -math.degrees(math.atan(A_dist / OA_horiz))
    else:
        AOE_angle = -90.0 if A_dist > 0 else (90.0 if A_dist < 0 else 0.0)

    if OB_horiz > 1e-9:
        BOF_angle = -math.degrees(math.atan(B_dist / OB_horiz))
    else:
        BOF_angle = -90.0 if B_dist > 0 else (90.0 if B_dist < 0 else 0.0)

    return OralAngleMeasure(
        A=A, B=B, C=C, D=D, E=E, F=F, O=O,
        AOE_angle=AOE_angle, BOF_angle=BOF_angle,
        angle_diff=BOF_angle - AOE_angle,
        angle_asymmetry=abs(BOF_angle - AOE_angle),
        A_dist_to_EF=A_dist, B_dist_to_EF=B_dist
    )


def compute_face_midline(landmarks, w: int, h: int) -> Dict[str, Any]:
    """
    计算面中线（双眼内眦中点的中垂线）

    Returns:
        {
            "center": (cx, cy),           # 内眦中点
            "direction": (nx, ny),         # 中垂线单位方向向量
            "top_point": (tx, ty),         # 面中线顶端点
            "bottom_point": (bx, by),      # 面中线底端点
            "icd": float,                  # 内眦距（归一化基准）
        }
    """
    # 获取内眦点
    left_canthus = pt2d(landmarks[LM.EYE_INNER_L], w, h)  # 362
    right_canthus = pt2d(landmarks[LM.EYE_INNER_R], w, h)  # 133

    lx, ly = left_canthus
    rx, ry = right_canthus

    # 中点
    cx, cy = (lx + rx) / 2, (ly + ry) / 2

    # 内眦距
    icd = math.sqrt((rx - lx) ** 2 + (ry - ly) ** 2)

    # 连线方向
    dx, dy = rx - lx, ry - ly

    # 中垂线方向（法向量，归一化）
    length = math.sqrt(dx ** 2 + dy ** 2)
    if length < 1e-6:
        return None
    nx, ny = -dy / length, dx / length  # 旋转90度

    # 计算上下端点（延伸到图像边界）
    # 向上延伸到眉毛上方，向下延伸到下巴
    extend_up = cy + 50  # 向上延伸
    extend_down = h - cy  # 向下延伸

    top_point = (cx + nx * extend_up, cy + ny * extend_up)
    bottom_point = (cx - nx * extend_down, cy - ny * extend_down)

    return {
        "center": (cx, cy),
        "direction": (nx, ny),
        "top_point": top_point,
        "bottom_point": bottom_point,
        "icd": icd,
    }


def draw_face_midline(img: np.ndarray, midline: Dict[str, Any],
                      color=(0, 255, 255), thickness=2, dashed=True) -> np.ndarray:
    """
    在图像上绘制面中线

    Args:
        img: 输入图像
        midline: compute_face_midline的返回值
        color: 颜色 (BGR)
        thickness: 线宽
        dashed: 是否使用虚线
    """
    result = img.copy()

    if midline is None:
        return result

    top = midline["top_point"]
    bottom = midline["bottom_point"]
    center = midline["center"]

    tx, ty = int(top[0]), int(top[1])
    bx, by = int(bottom[0]), int(bottom[1])
    cx, cy = int(center[0]), int(center[1])

    if dashed:
        # 绘制虚线
        dash_length = 10
        gap_length = 5

        # 计算总长度和方向
        total_length = math.sqrt((bx - tx) ** 2 + (by - ty) ** 2)
        dx = (bx - tx) / total_length
        dy = (by - ty) / total_length

        current_length = 0
        while current_length < total_length:
            start_x = tx + dx * current_length
            start_y = ty + dy * current_length
            end_length = min(current_length + dash_length, total_length)
            end_x = tx + dx * end_length
            end_y = ty + dy * end_length

            cv2.line(result,
                     (int(start_x), int(start_y)),
                     (int(end_x), int(end_y)),
                     color, thickness)

            current_length += dash_length + gap_length
    else:
        cv2.line(result, (tx, ty), (bx, by), color, thickness)

    # 标注中心点
    cv2.circle(result, (cx, cy), 4, color, -1)

    # 标注"Face Midline"
    cv2.putText(result, "Face Mid", (cx + 5, ty + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return result


# =============================================================================
# 数据库操作函数
# =============================================================================

def db_fetch_examinations(db_path: str, target_exam_id: Optional[str] = None,
                          limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """获取检查记录列表"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if target_exam_id:
        cursor.execute("""
            SELECT examination_id, patient_id, capture_datetime, has_labels, has_videos
            FROM examinations WHERE examination_id = ?
        """, (target_exam_id,))
    else:
        cursor.execute("""
            SELECT examination_id, patient_id, capture_datetime, has_labels, has_videos
            FROM examinations WHERE has_videos = 1 AND is_valid = 1
            ORDER BY capture_datetime DESC
        """)

    rows = cursor.fetchall()
    conn.close()

    exams = [{"examination_id": r[0], "patient_id": r[1], "capture_datetime": r[2],
              "has_labels": r[3], "has_videos": r[4]} for r in rows]

    if limit is not None:
        exams = exams[:int(limit)]
    return exams


def db_fetch_videos_for_exam(db_path: str, examination_id: str) -> Dict[str, Dict[str, Any]]:
    """获取检查的所有视频"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT v.video_id, v.action_id, v.file_path, v.start_frame, v.end_frame,
               v.fps, v.video_file_index, at.action_name_en, at.action_name_cn
        FROM video_files v
        LEFT JOIN action_types at ON v.action_id = at.action_id
        WHERE v.examination_id = ? AND v.file_exists = 1
        ORDER BY at.display_order ASC, v.video_file_index ASC
    """, (examination_id,))
    rows = cursor.fetchall()
    conn.close()

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        video_id, action_id, file_path, start_frame, end_frame, fps, video_file_index, action_en, action_cn = row
        if not action_en:
            continue
        action_en = str(action_en).strip()
        grouped.setdefault(action_en, []).append({
            "video_id": int(video_id), "action_id": int(action_id),
            "action_name_en": action_en, "action_name_cn": action_cn or action_en,
            "file_path": file_path,
            "start_frame": int(start_frame) if start_frame is not None else 0,
            "end_frame": int(end_frame) if end_frame is not None else None,
            "fps": float(fps) if fps is not None else 30.0,
            "video_file_index": int(video_file_index) if video_file_index is not None else 0,
        })

    selected = {}
    for action_en, candidates in grouped.items():
        candidates_sorted = sorted(candidates, key=lambda x: x.get("video_file_index", 0))
        selected[action_en] = candidates_sorted[0]
    return selected


def db_fetch_labels(db_path: str, examination_id: str) -> Dict[str, Any]:
    """获取检查的标签"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT has_palsy, palsy_side, hb_grade, sunnybrook_score
        FROM examination_labels WHERE examination_id = ?
    """, (examination_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return {}
    return {"has_palsy": row[0], "palsy_side": row[1], "hb_grade": row[2], "sunnybrook_score": row[3]}


class LazyVideoFrames:
    """
    懒加载视频帧序列：不把所有帧读进内存。
    只缓存第一帧（用于 w/h），其它帧按需从磁盘读取。
    用法保持和 list 一样：frames_seq[i]、len(frames_seq)
    """

    def __init__(self, video_path: str, start_frame: int, count: int, first_frame=None):
        self.video_path = video_path
        self.start_frame = int(start_frame)
        self.count = int(count)
        self._first_frame = first_frame  # 只缓存1张，不会爆内存

    def __len__(self):
        return self.count

    def __bool__(self):
        return self.count > 0

    def __getitem__(self, idx: int):
        if isinstance(idx, slice):
            # 很少用到；需要的话你可以按 slice 逐帧读取
            start, stop, step = idx.indices(self.count)
            return [self[i] for i in range(start, stop, step)]

        if idx < 0:
            idx = self.count + idx
        if idx < 0 or idx >= self.count:
            raise IndexError("LazyVideoFrames index out of range")

        # 0号帧优先用缓存（避免重复解码）
        if idx == 0 and self._first_frame is not None:
            return self._first_frame

        frame_no = self.start_frame + idx
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None
        return frame


# =============================================================================
# Landmark提取器
# =============================================================================

class LandmarkExtractor:
    """MediaPipe面部关键点提取器"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.landmarker = None

        try:
            import mediapipe as mp
            self.mp = mp
            BaseOptions = mp.tasks.BaseOptions
            FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode
            self.options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.IMAGE,
                num_faces=1
            )
        except ImportError:
            raise ImportError("请安装 mediapipe: pip install mediapipe")

    def __enter__(self):
        FaceLandmarker = self.mp.tasks.vision.FaceLandmarker
        self.landmarker = FaceLandmarker.create_from_options(self.options)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.landmarker:
            self.landmarker.close()

    def extract_from_frame(self, frame: np.ndarray):
        """从单帧提取landmarks"""
        if self.landmarker is None:
            FaceLandmarker = self.mp.tasks.vision.FaceLandmarker
            self.landmarker = FaceLandmarker.create_from_options(self.options)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.landmarker.detect(mp_image)

        if results and results.face_landmarks and len(results.face_landmarks) > 0:
            return results.face_landmarks[0]
        return None

    def extract_sequence(self, video_path: str, start_frame: int = 0,
                         end_frame: Optional[int] = None) -> Tuple[List, List]:
        """
        从视频提取 landmarks 序列。
        不再把所有帧存进 frames_seq（会爆内存）
        frames_seq 返回 LazyVideoFrames：需要哪一帧就读哪一帧
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if end_frame is None or end_frame >= total_frames:
            end_frame = total_frames - 1

        start_frame = max(0, int(start_frame))
        end_frame = min(int(end_frame), total_frames - 1)
        if end_frame < start_frame:
            cap.release()
            return [], []

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        landmarks_seq = []
        first_frame_cache = None
        current_frame = start_frame
        read_count = 0

        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # 只缓存第一帧：给外面 frames_seq[0].shape 用
            if read_count == 0:
                first_frame_cache = frame.copy()

            landmarks = self.extract_from_frame(frame)
            landmarks_seq.append(landmarks)

            current_frame += 1
            read_count += 1

        cap.release()

        frames_seq = LazyVideoFrames(
            video_path=video_path,
            start_frame=start_frame,
            count=len(landmarks_seq),
            first_frame=first_frame_cache
        )
        return landmarks_seq, frames_seq


# =============================================================================
# 基础可视化函数
# =============================================================================

def draw_text_with_background(img: np.ndarray, text: str, pos: Tuple[int, int],
                              font_scale: float = 0.5, color: Tuple[int, int, int] = (255, 255, 255),
                              bg_color: Tuple[int, int, int] = (0, 0, 0),
                              thickness: int = 1) -> None:
    """绘制带背景的文字"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x - 2, y - text_h - 2), (x + text_w + 2, y + baseline + 2), bg_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness)


def draw_landmarks(img: np.ndarray, landmarks, w: int, h: int,
                   indices: List[int], color: Tuple[int, int, int] = (0, 255, 0),
                   radius: int = 2) -> None:
    """绘制指定的landmark点"""
    for idx in indices:
        pt = pt2d(landmarks[idx], w, h)
        cv2.circle(img, (int(pt[0]), int(pt[1])), radius, color, -1)


def draw_polygon(img: np.ndarray, landmarks, w: int, h: int,
                 indices: List[int], color: Tuple[int, int, int] = (0, 255, 0),
                 thickness: int = 1, closed: bool = True) -> None:
    """绘制多边形轮廓"""
    pts = pts2d(landmarks, indices, w, h).astype(np.int32)
    cv2.polylines(img, [pts], closed, color, thickness)


# =============================================================================
# 数据类 - 动作处理结果
# =============================================================================

@dataclass
class ActionResult:
    """单个动作的处理结果"""
    action_name: str
    action_name_cn: str
    video_path: str
    total_frames: int
    peak_frame_idx: int
    image_size: Tuple[int, int]
    fps: float

    # 通用指标
    icd: float = 0.0

    # 眼部指标
    left_eye_area: float = 0.0
    right_eye_area: float = 0.0
    eye_area_ratio: float = 1.0
    left_ear: float = 0.0
    right_ear: float = 0.0
    left_palpebral_height: float = 0.0
    right_palpebral_height: float = 0.0
    palpebral_height_ratio: float = 1.0
    left_palpebral_width: float = 0.0
    right_palpebral_width: float = 0.0

    # 眉毛指标
    left_brow_height: float = 0.0
    right_brow_height: float = 0.0
    brow_height_ratio: float = 1.0
    left_brow_position: Optional[Dict] = None
    right_brow_position: Optional[Dict] = None

    # 眉眼距指标
    left_brow_eye_distance: float = 0.0
    right_brow_eye_distance: float = 0.0
    brow_eye_distance_ratio: float = 1.0
    left_brow_eye_distance_change: float = 0.0
    right_brow_eye_distance_change: float = 0.0
    brow_eye_distance_change_ratio: float = 1.0

    # 嘴部指标
    mouth_width: float = 0.0
    mouth_height: float = 0.0
    oral_angle: Optional[OralAngleMeasure] = None

    # 鼻唇沟
    left_nlf_length: float = 0.0
    right_nlf_length: float = 0.0
    nlf_ratio: float = 1.0

    # 动作特定指标
    action_specific: Dict[str, Any] = field(default_factory=dict)

    # Sunnybrook评分相关
    voluntary_movement_score: int = 5  # 1-5
    synkinesis_scores: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            "action_name": self.action_name,
            "action_name_cn": self.action_name_cn,
            "video_path": self.video_path,
            "total_frames": self.total_frames,
            "peak_frame_idx": self.peak_frame_idx,
            "image_size": {"width": self.image_size[0], "height": self.image_size[1]},
            "fps": self.fps,
            "icd": self.icd,
            "eye": {
                "left_area": self.left_eye_area,
                "right_area": self.right_eye_area,
                "area_ratio": self.eye_area_ratio,
                "left_ear": self.left_ear,
                "right_ear": self.right_ear,
                "left_palpebral_height": self.left_palpebral_height,
                "right_palpebral_height": self.right_palpebral_height,
                "palpebral_height_ratio": self.palpebral_height_ratio,
                "left_palpebral_width": self.left_palpebral_width,
                "right_palpebral_width": self.right_palpebral_width,
            },
            "brow": {
                "left_height": self.left_brow_height,
                "right_height": self.right_brow_height,
                "height_ratio": self.brow_height_ratio,
                "left_position": self.left_brow_position,
                "right_position": self.right_brow_position,
                "left_brow_eye_distance": self.left_brow_eye_distance,
                "right_brow_eye_distance": self.right_brow_eye_distance,
                "brow_eye_distance_ratio": self.brow_eye_distance_ratio,
                "left_brow_eye_distance_change": self.left_brow_eye_distance_change,
                "right_brow_eye_distance_change": self.right_brow_eye_distance_change,
                "brow_eye_distance_change_ratio": self.brow_eye_distance_change_ratio,
            },
            "mouth": {
                "width": self.mouth_width,
                "height": self.mouth_height,
            },
            "nlf": {
                "left_length": self.left_nlf_length,
                "right_length": self.right_nlf_length,
                "ratio": self.nlf_ratio,
            },
            "action_specific": self.action_specific,
            "voluntary_movement_score": self.voluntary_movement_score,
            "synkinesis_scores": self.synkinesis_scores,
        }

        if self.oral_angle:
            result["oral_angle"] = {
                "AOE_angle_deg": self.oral_angle.AOE_angle,
                "BOF_angle_deg": self.oral_angle.BOF_angle,
                "angle_diff": self.oral_angle.angle_diff,
                "angle_asymmetry": self.oral_angle.angle_asymmetry,
                "A_dist_to_EF": self.oral_angle.A_dist_to_EF,
                "B_dist_to_EF": self.oral_angle.B_dist_to_EF,
                "points": {
                    "A": list(self.oral_angle.A),
                    "B": list(self.oral_angle.B),
                    "E": list(self.oral_angle.E),
                    "F": list(self.oral_angle.F),
                    "O": list(self.oral_angle.O),
                }
            }

        return make_json_serializable(result)


def extract_common_indicators(landmarks, w: int, h: int, result: ActionResult,
                              baseline_landmarks=None) -> None:
    """
    提取通用指标到ActionResult

    Args:
        landmarks: 当前帧landmarks
        w, h: 图像尺寸
        result: ActionResult对象
        baseline_landmarks: 基线帧landmarks (用于计算变化度)
    """
    result.icd = compute_icd(landmarks, w, h)

    result.left_eye_area, _ = compute_eye_area(landmarks, w, h, left=True)
    result.right_eye_area, _ = compute_eye_area(landmarks, w, h, left=False)
    result.eye_area_ratio = result.left_eye_area / result.right_eye_area if result.right_eye_area > 1e-9 else 1.0

    result.left_ear = compute_ear(landmarks, w, h, left=True)
    result.right_ear = compute_ear(landmarks, w, h, left=False)

    result.left_palpebral_height = compute_palpebral_height(landmarks, w, h, left=True)
    result.right_palpebral_height = compute_palpebral_height(landmarks, w, h, left=False)
    result.palpebral_height_ratio = result.left_palpebral_height / result.right_palpebral_height if result.right_palpebral_height > 1e-9 else 1.0

    result.left_palpebral_width = compute_palpebral_width(landmarks, w, h, left=True)
    result.right_palpebral_width = compute_palpebral_width(landmarks, w, h, left=False)

    result.left_brow_height = compute_brow_height(landmarks, w, h, left=True)
    result.right_brow_height = compute_brow_height(landmarks, w, h, left=False)
    result.brow_height_ratio = result.left_brow_height / result.right_brow_height if result.right_brow_height > 1e-9 else 1.0

    result.left_brow_position = compute_brow_position(landmarks, w, h, left=True)
    result.right_brow_position = compute_brow_position(landmarks, w, h, left=False)

    # 眉眼距
    bed_result = compute_brow_eye_distance_ratio(landmarks, w, h)
    result.left_brow_eye_distance = bed_result["left_distance"]
    result.right_brow_eye_distance = bed_result["right_distance"]
    result.brow_eye_distance_ratio = bed_result["ratio"]

    # 眉眼距变化度 (需要基线)
    if baseline_landmarks is not None:
        bedc_result = compute_brow_eye_distance_change_ratio(landmarks, baseline_landmarks, w, h)
        result.left_brow_eye_distance_change = bedc_result["left_change"]
        result.right_brow_eye_distance_change = bedc_result["right_change"]
        result.brow_eye_distance_change_ratio = bedc_result["ratio"]

    mouth = compute_mouth_metrics(landmarks, w, h)
    result.mouth_width = mouth["width"]
    result.mouth_height = mouth["height"]

    result.oral_angle = compute_oral_angle(landmarks, w, h)

    result.left_nlf_length = compute_nlf_length(landmarks, w, h, left=True)
    result.right_nlf_length = compute_nlf_length(landmarks, w, h, left=False)
    result.nlf_ratio = result.left_nlf_length / result.right_nlf_length if result.right_nlf_length > 1e-9 else 1.0


# =============================================================================
# 鼻翼到内眦距离计算 (用于ShrugNose)
# =============================================================================

def compute_ala_to_canthus_distance(landmarks, w: int, h: int, left: bool = True) -> float:
    """
    计算鼻翼点到同侧眼角内眦点的距离

    皱鼻动作时，鼻翼上提，此距离变小

    Args:
        landmarks: MediaPipe landmarks
        w, h: 图像尺寸
        left: True=左侧, False=右侧

    Returns:
        距离 (像素)
    """
    if left:
        ala = pt2d(landmarks[LM.NOSE_ALA_L], w, h)
        canthus = pt2d(landmarks[LM.EYE_INNER_L], w, h)
    else:
        ala = pt2d(landmarks[LM.NOSE_ALA_R], w, h)
        canthus = pt2d(landmarks[LM.EYE_INNER_R], w, h)

    return dist(ala, canthus)


def compute_ala_canthus_metrics(landmarks, w: int, h: int,
                                baseline_landmarks=None) -> Dict[str, Any]:
    """
    计算双侧鼻翼到内眦距离的指标

    用于ShrugNose动作分析

    Returns:
        Dict包含:
        - left_distance: 左侧距离
        - right_distance: 右侧距离
        - ratio: 左/右比值
        - 如果有基线: left_change, right_change, left_change_percent, right_change_percent
    """
    left_dist = compute_ala_to_canthus_distance(landmarks, w, h, left=True)
    right_dist = compute_ala_to_canthus_distance(landmarks, w, h, left=False)

    metrics = {
        "left_distance": left_dist,
        "right_distance": right_dist,
        "ratio": left_dist / right_dist if right_dist > 1e-9 else 1.0,
    }

    if baseline_landmarks is not None:
        baseline_left = compute_ala_to_canthus_distance(baseline_landmarks, w, h, left=True)
        baseline_right = compute_ala_to_canthus_distance(baseline_landmarks, w, h, left=False)

        metrics["baseline_left"] = baseline_left
        metrics["baseline_right"] = baseline_right

        # 变化量 (皱鼻时应为负值，表示距离变小)
        metrics["left_change"] = left_dist - baseline_left
        metrics["right_change"] = right_dist - baseline_right

        # 变化百分比
        if baseline_left > 1e-9:
            metrics["left_change_percent"] = (left_dist - baseline_left) / baseline_left * 100
        else:
            metrics["left_change_percent"] = 0

        if baseline_right > 1e-9:
            metrics["right_change_percent"] = (right_dist - baseline_right) / baseline_right * 100
        else:
            metrics["right_change_percent"] = 0

    return metrics


# =============================================================================
# 从闭眼动作检测面瘫侧别
# =============================================================================

def detect_palsy_side_from_eye_closure(left_ear: float, right_ear: float,
                                       baseline_left_ear: float = None,
                                       baseline_right_ear: float = None,
                                       closure_threshold: float = 0.15) -> Dict[str, Any]:
    """
    从闭眼动作检测面瘫侧别

    原理: 面瘫侧的眼睛无法完全闭合，EAR值较大

    Args:
        left_ear: 左眼EAR值 (闭眼时)
        right_ear: 右眼EAR值 (闭眼时)
        baseline_left_ear: 基线左眼EAR值 (可选)
        baseline_right_ear: 基线右眼EAR值 (可选)
        closure_threshold: 闭合阈值，EAR低于此值认为完全闭合

    Returns:
        Dict包含:
        - palsy_side: 面瘫侧别 (0=无/对称, 1=左, 2=右)
        - confidence: 置信度 (0-1)
        - left_closed: 左眼是否闭合
        - right_closed: 右眼是否闭合
        - interpretation: 解释文字
    """
    left_closed = left_ear < closure_threshold
    right_closed = right_ear < closure_threshold

    result = {
        "left_ear": left_ear,
        "right_ear": right_ear,
        "left_closed": left_closed,
        "right_closed": right_closed,
        "closure_threshold": closure_threshold,
    }

    if baseline_left_ear is not None and baseline_right_ear is not None:
        # 使用闭合比例
        left_closure_ratio = left_ear / baseline_left_ear if baseline_left_ear > 1e-9 else 1.0
        right_closure_ratio = right_ear / baseline_right_ear if baseline_right_ear > 1e-9 else 1.0
        result["left_closure_ratio"] = left_closure_ratio
        result["right_closure_ratio"] = right_closure_ratio

    # 判断面瘫侧别
    if left_closed and right_closed:
        # 两眼都能闭合
        # 看哪只眼的EAR更大（闭合更不完全）
        ear_diff = abs(left_ear - right_ear)
        max_ear = max(left_ear, right_ear)

        if ear_diff / max_ear < 0.15 if max_ear > 1e-9 else True:
            result["palsy_side"] = 0
            result["confidence"] = 1.0 - (ear_diff / max_ear if max_ear > 1e-9 else 0)
            result["interpretation"] = "双眼对称闭合"
        elif left_ear > right_ear:
            result["palsy_side"] = 1  # 左侧面瘫
            result["confidence"] = min(1.0, ear_diff / max_ear)
            result["interpretation"] = f"左眼闭合较弱 (EAR L={left_ear:.3f} > R={right_ear:.3f})"
        else:
            result["palsy_side"] = 2  # 右侧面瘫
            result["confidence"] = min(1.0, ear_diff / max_ear)
            result["interpretation"] = f"右眼闭合较弱 (EAR R={right_ear:.3f} > L={left_ear:.3f})"

    elif left_closed and not right_closed:
        # 只有左眼能闭合 -> 右侧面瘫
        result["palsy_side"] = 2
        result["confidence"] = min(1.0, (right_ear - closure_threshold) / closure_threshold)
        result["interpretation"] = f"右眼无法闭合 (EAR={right_ear:.3f}>{closure_threshold})"

    elif right_closed and not left_closed:
        # 只有右眼能闭合 -> 左侧面瘫
        result["palsy_side"] = 1
        result["confidence"] = min(1.0, (left_ear - closure_threshold) / closure_threshold)
        result["interpretation"] = f"左眼无法闭合 (EAR={left_ear:.3f}>{closure_threshold})"

    else:
        # 两眼都无法完全闭合
        ear_diff = abs(left_ear - right_ear)
        if ear_diff < 0.05:
            result["palsy_side"] = 0
            result["confidence"] = 0.5
            result["interpretation"] = "双眼均无法完全闭合，可能为双侧面瘫或其他原因"
        elif left_ear > right_ear:
            result["palsy_side"] = 1
            result["confidence"] = min(1.0, ear_diff / max(left_ear, right_ear))
            result["interpretation"] = f"双眼均无法完全闭合，左眼更差 (L={left_ear:.3f}, R={right_ear:.3f})"
        else:
            result["palsy_side"] = 2
            result["confidence"] = min(1.0, ear_diff / max(left_ear, right_ear))
            result["interpretation"] = f"双眼均无法完全闭合，右眼更差 (R={right_ear:.3f}, L={left_ear:.3f})"

    return result


# =============================================================================
# 从嘴角运动检测面瘫侧别
# =============================================================================

def detect_palsy_side_from_mouth(oral_angle, baseline_oral_angle=None) -> Dict[str, Any]:
    """
    从嘴角运动检测面瘫侧别

    Args:
        oral_angle: OralAngleMeasure对象
        baseline_oral_angle: 基线OralAngleMeasure对象 (可选)

    Returns:
        Dict包含:
        - palsy_side: 面瘫侧别 (0=无/对称, 1=左, 2=右)
        - confidence: 置信度
        - interpretation: 解释
    """
    if oral_angle is None:
        return {
            "palsy_side": 0,
            "confidence": 0.0,
            "interpretation": "无口角角度数据"
        }

    aoe = oral_angle.AOE_angle  # 右侧口角角度
    bof = oral_angle.BOF_angle  # 左侧口角角度
    asymmetry = oral_angle.angle_asymmetry

    result = {
        "aoe_angle": aoe,
        "bof_angle": bof,
        "asymmetry": asymmetry,
    }

    if baseline_oral_angle is not None:
        # 计算运动幅度变化
        baseline_aoe = baseline_oral_angle.AOE_angle
        baseline_bof = baseline_oral_angle.BOF_angle

        aoe_change = aoe - baseline_aoe
        bof_change = bof - baseline_bof

        result["aoe_change"] = aoe_change
        result["bof_change"] = bof_change

    # 判断面瘫侧别
    if asymmetry < 3:  # 对称性良好
        result["palsy_side"] = 0
        result["confidence"] = 1.0 - asymmetry / 10
        result["interpretation"] = f"口角对称 (不对称度={asymmetry:.1f}°)"
    elif asymmetry < 10:  # 轻度不对称
        # AOE负值表示右口角下垂，BOF负值表示左口角下垂
        # 口角下垂的一侧是患侧
        if aoe < bof:  # 右口角更低
            result["palsy_side"] = 2
            result["confidence"] = min(1.0, asymmetry / 15)
            result["interpretation"] = f"右口角位置较低 (AOE={aoe:+.1f}° < BOF={bof:+.1f}°)"
        else:
            result["palsy_side"] = 1
            result["confidence"] = min(1.0, asymmetry / 15)
            result["interpretation"] = f"左口角位置较低 (BOF={bof:+.1f}° < AOE={aoe:+.1f}°)"
    else:  # 明显不对称
        if aoe < bof:
            result["palsy_side"] = 2
            result["confidence"] = min(1.0, asymmetry / 20)
            result["interpretation"] = f"右口角明显下垂 (AOE={aoe:+.1f}°, BOF={bof:+.1f}°)"
        else:
            result["palsy_side"] = 1
            result["confidence"] = min(1.0, asymmetry / 20)
            result["interpretation"] = f"左口角明显下垂 (BOF={bof:+.1f}°, AOE={aoe:+.1f}°)"

    return result


# =============================================================================
# 从运动幅度比较检测面瘫侧别
# =============================================================================

def detect_palsy_side_from_excursion(left_excursion: float, right_excursion: float,
                                     threshold_ratio: float = 0.15) -> Dict[str, Any]:
    """
    从双侧运动幅度检测面瘫侧别

    适用于: Smile, ShowTeeth, RaiseEyebrow等有明显双侧运动的动作

    Args:
        left_excursion: 左侧运动幅度
        right_excursion: 右侧运动幅度
        threshold_ratio: 判断不对称的阈值比例

    Returns:
        Dict包含:
        - palsy_side: 面瘫侧别
        - confidence: 置信度
        - interpretation: 解释
    """
    max_exc = max(left_excursion, right_excursion)
    min_exc = min(left_excursion, right_excursion)

    if max_exc < 1e-9:
        return {
            "palsy_side": 0,
            "confidence": 0.0,
            "left_excursion": left_excursion,
            "right_excursion": right_excursion,
            "interpretation": "无明显运动"
        }

    asymmetry_ratio = (max_exc - min_exc) / max_exc

    result = {
        "left_excursion": left_excursion,
        "right_excursion": right_excursion,
        "asymmetry_ratio": asymmetry_ratio,
    }

    if asymmetry_ratio < threshold_ratio:
        result["palsy_side"] = 0
        result["confidence"] = 1.0 - asymmetry_ratio / threshold_ratio
        result["interpretation"] = f"双侧运动对称 (不对称比={asymmetry_ratio:.1%})"
    elif left_excursion < right_excursion:
        result["palsy_side"] = 1  # 左侧运动弱 -> 左侧面瘫
        result["confidence"] = min(1.0, asymmetry_ratio)
        result["interpretation"] = f"左侧运动较弱 (L={left_excursion:.1f} < R={right_excursion:.1f})"
    else:
        result["palsy_side"] = 2  # 右侧运动弱 -> 右侧面瘫
        result["confidence"] = min(1.0, asymmetry_ratio)
        result["interpretation"] = f"右侧运动较弱 (R={right_excursion:.1f} < L={left_excursion:.1f})"

    return result


# =============================================================================
# 通用绘图辅助函数
# =============================================================================

def add_valid_region_shading(ax, valid_mask: List[bool], time_or_frames: np.ndarray,
                             alpha: float = 0.15) -> None:
    """
    在曲线图中标注 valid/invalid 区域

    Args:
        ax: matplotlib axes
        valid_mask: valid 状态的布尔列表
        time_or_frames: x 轴数值（时间或帧号）
        alpha: 无效区域的透明度
    """
    if valid_mask is None or len(valid_mask) == 0:
        return

    valid_arr = np.array(valid_mask, dtype=bool)
    n = len(valid_arr)
    if n != len(time_or_frames):
        return

    # 找出 invalid 区域并着色
    in_invalid = False
    start_idx = 0

    for i in range(n):
        if not valid_arr[i] and not in_invalid:
            # 进入 invalid 区域
            start_idx = i
            in_invalid = True
        elif valid_arr[i] and in_invalid:
            # 离开 invalid 区域
            ax.axvspan(time_or_frames[start_idx], time_or_frames[i - 1],
                       color='red', alpha=alpha, label='Invalid' if start_idx == 0 else None)
            in_invalid = False

    # 处理末尾的 invalid 区域
    if in_invalid:
        ax.axvspan(time_or_frames[start_idx], time_or_frames[n - 1],
                   color='red', alpha=alpha)


def get_palsy_side_text(palsy_side: int, confidence: float = None) -> str:
    """
    根据患侧编码返回显示文本

    Args:
        palsy_side: 0=无/对称, 1=左侧, 2=右侧
        confidence: 置信度（可选）

    Returns:
        显示用的文本
    """
    side_map = {0: "Normal/Symmetric", 1: "Left Palsy", 2: "Right Palsy"}
    text = side_map.get(palsy_side, "Unknown")
    if confidence is not None and confidence > 0:
        text += f" ({confidence * 100:.0f}%)"
    return text


def get_palsy_side_color(palsy_side: int) -> Tuple[int, int, int]:
    """
    根据患侧编码返回显示颜色 (BGR格式)

    Args:
        palsy_side: 0=无/对称, 1=左侧, 2=右侧

    Returns:
        BGR颜色元组
    """
    color_map = {
        0: (0, 255, 0),  # 绿色 - 正常
        1: (0, 0, 255),  # 红色 - 左侧面瘫
        2: (255, 0, 0),  # 蓝色 - 右侧面瘫
    }
    return color_map.get(palsy_side, (255, 255, 255))


def draw_palsy_side_label(img: np.ndarray, palsy_detection: Dict[str, Any],
                          x: int = 10, y: int = 60, font_scale: float = 1.2) -> np.ndarray:
    """
    在图像左上角绘制患侧标签（放大4倍）

    Args:
        img: 图像
        palsy_detection: 包含 palsy_side, confidence, interpretation 的字典
        x, y: 标签位置
        font_scale: 字体大小（默认1.2，约为之前的2倍）

    Returns:
        绘制后的图像
    """
    if not palsy_detection:
        return img

    palsy_side = palsy_detection.get("palsy_side", 0)
    confidence = palsy_detection.get("confidence", 0)
    interpretation = palsy_detection.get("interpretation", "")

    color = get_palsy_side_color(palsy_side)
    text = get_palsy_side_text(palsy_side, confidence)

    # 使用更大的字体
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 3

    # 计算文本大小
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # 绘制背景框
    padding = 10
    cv2.rectangle(img, (x - padding, y - text_h - padding),
                  (x + text_w + padding, y + padding), (0, 0, 0), -1)
    cv2.rectangle(img, (x - padding, y - text_h - padding),
                  (x + text_w + padding, y + padding), color, 3)

    # 绘制文本
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness)

    return img


# =============================================================================
# 通用面瘫侧别检测 - 基于比值的无阈值方法
# =============================================================================

def detect_palsy_side_by_ratio(left_value: float, right_value: float,
                               metric_name: str = "metric",
                               higher_is_worse: bool = False) -> Dict[str, Any]:
    """
    通用面瘫侧别检测 - 基于左右比值，无需硬编码阈值

    原理: 直接比较左右差异，差异越大置信度越高

    Args:
        left_value: 左侧指标值
        right_value: 右侧指标值
        metric_name: 指标名称（用于解释）
        higher_is_worse: True表示值越大越差(如EAR闭眼时), False表示值越小越差(如运动幅度)

    Returns:
        Dict包含:
        - palsy_side: 0=对称, 1=左侧面瘫, 2=右侧面瘫
        - confidence: 置信度 (0-1)，基于左右差异比例
        - interpretation: 解释文字
        - left_value, right_value: 原始值
        - asymmetry_ratio: 不对称比例
    """
    result = {
        "left_value": left_value,
        "right_value": right_value,
        "metric_name": metric_name,
    }

    max_val = max(abs(left_value), abs(right_value))
    min_val = min(abs(left_value), abs(right_value))

    if max_val < 1e-9:
        result["palsy_side"] = 0
        result["confidence"] = 0.0
        result["asymmetry_ratio"] = 0.0
        result["interpretation"] = f"{metric_name}: 无有效数据"
        return result

    # 计算不对称比例 (0-1范围，越大越不对称)
    asymmetry_ratio = (max_val - min_val) / max_val
    result["asymmetry_ratio"] = asymmetry_ratio

    # 置信度直接由不对称比例决定，归一化到0-1
    # 使用sigmoid函数使得小差异时置信度低，大差异时置信度高
    confidence = min(1.0, asymmetry_ratio * 2)  # 50%差异 -> 100%置信度
    result["confidence"] = confidence

    # 判断面瘫侧别
    if asymmetry_ratio < 0.10:  # 10%以内认为对称
        result["palsy_side"] = 0
        result["interpretation"] = f"{metric_name}: 左右对称 (差异{asymmetry_ratio * 100:.1f}%)"
    else:
        if higher_is_worse:
            # 值越大越差（如闭眼时EAR）
            if left_value > right_value:
                result["palsy_side"] = 1
                result["interpretation"] = f"{metric_name}: 左侧较差 (L={left_value:.3f} > R={right_value:.3f})"
            else:
                result["palsy_side"] = 2
                result["interpretation"] = f"{metric_name}: 右侧较差 (R={right_value:.3f} > L={left_value:.3f})"
        else:
            # 值越小越差（如运动幅度）
            if left_value < right_value:
                result["palsy_side"] = 1
                result["interpretation"] = f"{metric_name}: 左侧运动弱 (L={left_value:.3f} < R={right_value:.3f})"
            else:
                result["palsy_side"] = 2
                result["interpretation"] = f"{metric_name}: 右侧运动弱 (R={right_value:.3f} < L={left_value:.3f})"

    return result


def combine_palsy_detections(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    综合多个指标的面瘫检测结果

    使用加权投票机制，权重为各指标的置信度

    Args:
        detections: 多个detect_palsy_side_by_ratio的结果列表

    Returns:
        综合后的面瘫检测结果
    """
    if not detections:
        return {"palsy_side": 0, "confidence": 0.0, "interpretation": "无检测数据"}

    # 加权投票
    votes = {0: 0.0, 1: 0.0, 2: 0.0}
    total_weight = 0.0

    details = []
    for det in detections:
        side = det.get("palsy_side", 0)
        conf = det.get("confidence", 0.0)
        votes[side] += conf
        total_weight += conf
        details.append(det.get("interpretation", ""))

    if total_weight < 1e-9:
        return {"palsy_side": 0, "confidence": 0.0, "interpretation": "所有指标无明显差异"}

    # 归一化投票
    for k in votes:
        votes[k] /= total_weight

    # 选择得票最高的
    final_side = max(votes, key=votes.get)
    final_conf = votes[final_side]

    side_names = {0: "对称", 1: "左侧", 2: "右侧"}

    return {
        "palsy_side": final_side,
        "confidence": final_conf,
        "interpretation": f"综合判断: {side_names[final_side]}面瘫 (置信度{final_conf * 100:.1f}%)",
        "vote_details": votes,
        "indicator_details": details,
    }


# =============================================================================
# 嘴唇区域面中线对称性分析（用于面瘫侧别判断）
# =============================================================================

def compute_lip_region_centroid(landmarks, w: int, h: int, left: bool = True) -> Tuple[float, float]:
    """
    计算左侧或右侧嘴唇区域的质心

    Args:
        landmarks: 关键点
        w, h: 图像宽高
        left: True=患者左侧嘴唇, False=患者右侧嘴唇

    Returns:
        (cx, cy): 质心坐标
    """
    indices = LM.LIP_L if left else LM.LIP_R
    points = pts2d(landmarks, indices, w, h)
    return compute_centroid(points)


def compute_lip_midline_symmetry(landmarks, w: int, h: int) -> Dict[str, Any]:
    """
    计算嘴唇相对于面中线的对称性

    核心原理:
    - 面中线: 双内眦中点的x坐标
    - 左侧嘴唇区域质心到面中线的水平距离
    - 右侧嘴唇区域质心到面中线的水平距离
    - 如果嘴唇被拉向一侧，该侧距离会增大，另一侧减小

    判断逻辑:
    - 嘴唇偏向健侧（被健侧肌肉拉动）
    - 所以：偏向的那侧是健侧，另一侧是面瘫侧

    Returns:
        Dict包含:
        - midline_x: 面中线x坐标
        - left_centroid: 左侧嘴唇质心 (x, y)
        - right_centroid: 右侧嘴唇质心 (x, y)
        - left_to_midline: 左侧嘴唇质心到中线的水平距离（正值）
        - right_to_midline: 右侧嘴唇质心到中线的水平距离（正值）
        - lip_offset: 嘴唇整体偏移 = (左侧距离 - 右侧距离)
                      正值表示偏向左侧（患者左侧），负值表示偏向右侧（患者右侧）
        - symmetry_ratio: 对称比 = min(L,R) / max(L,R)，越接近1越对称
        - asymmetry_ratio: 不对称比 = |L-R| / max(L,R)
        - palsy_side_suggestion: 根据偏移建议的面瘫侧 (0=对称, 1=左侧面瘫, 2=右侧面瘫)
    """
    # 面中线（双内眦中点）
    left_canthus = pt2d(landmarks[LM.EYE_INNER_L], w, h)
    right_canthus = pt2d(landmarks[LM.EYE_INNER_R], w, h)
    midline_x = (left_canthus[0] + right_canthus[0]) / 2

    # 左右嘴唇质心
    left_centroid = compute_lip_region_centroid(landmarks, w, h, left=True)
    right_centroid = compute_lip_region_centroid(landmarks, w, h, left=False)

    # 到面中线的水平距离
    # 患者左侧嘴唇在图像右侧，x > midline_x
    # 患者右侧嘴唇在图像左侧，x < midline_x
    left_to_midline = abs(left_centroid[0] - midline_x)
    right_to_midline = abs(right_centroid[0] - midline_x)

    # 嘴唇整体偏移
    # 如果左侧距离 > 右侧距离，说明嘴唇偏向左侧（患者左侧）
    lip_offset = left_to_midline - right_to_midline

    # 对称比和不对称比
    max_dist = max(left_to_midline, right_to_midline)
    min_dist = min(left_to_midline, right_to_midline)
    symmetry_ratio = min_dist / max_dist if max_dist > 1e-6 else 1.0
    asymmetry_ratio = abs(lip_offset) / max_dist if max_dist > 1e-6 else 0

    # 面瘫侧建议
    # 嘴唇偏向健侧，所以偏向的那侧是健侧，另一侧是面瘫侧
    if asymmetry_ratio < 0.08:  # 偏移小于8%认为对称
        palsy_side_suggestion = 0
    elif lip_offset > 0:
        # 嘴唇偏向左侧（患者左侧）= 被左侧拉 = 左侧是健侧 = 右侧面瘫
        palsy_side_suggestion = 2
    else:
        # 嘴唇偏向右侧（患者右侧）= 被右侧拉 = 右侧是健侧 = 左侧面瘫
        palsy_side_suggestion = 1

    return {
        "midline_x": float(midline_x),
        "left_centroid": (float(left_centroid[0]), float(left_centroid[1])),
        "right_centroid": (float(right_centroid[0]), float(right_centroid[1])),
        "left_to_midline": float(left_to_midline),
        "right_to_midline": float(right_to_midline),
        "lip_offset": float(lip_offset),
        "symmetry_ratio": float(symmetry_ratio),
        "asymmetry_ratio": float(asymmetry_ratio),
        "palsy_side_suggestion": palsy_side_suggestion,
    }


def compute_nose_midline_symmetry(landmarks, w: int, h: int) -> Dict[str, Any]:
    """
    计算鼻翼-内眦距离相对于面中线的对称性（用于ShrugNose面瘫侧判断）

    核心原理:
    - 皱鼻时，鼻翼向上向内收缩，鼻翼-内眦距离减小
    - 面瘫侧鼻翼运动弱，鼻翼-内眦距离变化小（距离更大）
    - 在峰值帧直接比较左右鼻翼到各自内眦的距离

    判断逻辑:
    - 皱鼻时距离应该变小
    - 距离大的一侧是面瘫侧（收缩弱）

    Returns:
        Dict包含:
        - left_ala_canthus_dist: 左鼻翼到左内眦距离
        - right_ala_canthus_dist: 右鼻翼到右内眦距离
        - distance_diff: 左 - 右 距离差
        - asymmetry_ratio: 不对称比 = |L-R| / max(L,R)
        - symmetry_ratio: 对称比 = min(L,R) / max(L,R)
        - palsy_side_suggestion: 距离更大的一侧是面瘫侧（收缩弱）
    """
    # 鼻翼和内眦位置
    left_ala = pt2d(landmarks[LM.NOSE_ALA_L], w, h)
    right_ala = pt2d(landmarks[LM.NOSE_ALA_R], w, h)
    left_canthus = pt2d(landmarks[LM.EYE_INNER_L], w, h)
    right_canthus = pt2d(landmarks[LM.EYE_INNER_R], w, h)

    # 鼻翼到内眦距离
    left_dist = dist(left_ala, left_canthus)
    right_dist = dist(right_ala, right_canthus)

    # 距离差和对称比
    distance_diff = left_dist - right_dist
    max_dist = max(left_dist, right_dist)
    min_dist = min(left_dist, right_dist)
    asymmetry_ratio = abs(distance_diff) / max_dist if max_dist > 1e-6 else 0
    symmetry_ratio = min_dist / max_dist if max_dist > 1e-6 else 1.0

    # 面瘫侧建议
    # 皱鼻时距离应该变小，距离大的一侧是面瘫侧（收缩弱）
    if asymmetry_ratio < 0.08:  # 8%以内认为对称
        palsy_side_suggestion = 0
    elif left_dist > right_dist:
        # 左侧距离大 = 左侧收缩弱 = 左侧面瘫
        palsy_side_suggestion = 1
    else:
        # 右侧距离大 = 右侧收缩弱 = 右侧面瘫
        palsy_side_suggestion = 2

    return {
        "left_ala_canthus_dist": float(left_dist),
        "right_ala_canthus_dist": float(right_dist),
        "distance_diff": float(distance_diff),
        "asymmetry_ratio": float(asymmetry_ratio),
        "symmetry_ratio": float(symmetry_ratio),
        "palsy_side_suggestion": palsy_side_suggestion,
    }


def compute_eye_closure_by_area(landmarks, w: int, h: int,
                                baseline_landmarks=None) -> Dict[str, Any]:
    """
    使用眼睛面积计算闭眼程度（解决天生大小眼的问题）

    原理:
    - 用当前眼睛面积 / 基线眼睛面积 来计算闭合度
    - 比值越小表示闭合程度越高
    - 相比EAR更能反映真实的眼睛开合状态

    Args:
        landmarks: 当前帧landmarks
        w, h: 图像尺寸
        baseline_landmarks: 基线帧landmarks（NeutralFace）

    Returns:
        Dict包含:
        - left_area, right_area: 当前帧眼睛面积
        - baseline_left_area, baseline_right_area: 基线眼睛面积
        - left_open_ratio, right_open_ratio: 睁眼比例 (0-1, 越大越睁开)
        - left_closure_ratio, right_closure_ratio: 闭眼比例 (0-1, 越大越闭合)
        - closure_asymmetry: 闭眼不对称度
        - palsy_side_suggestion: 面瘫侧建议 (闭合差的一侧)
    """
    # 当前帧眼睛面积
    left_area, _ = compute_eye_area(landmarks, w, h, left=True)
    right_area, _ = compute_eye_area(landmarks, w, h, left=False)

    result = {
        "left_area": float(left_area),
        "right_area": float(right_area),
    }

    if baseline_landmarks is None:
        # 没有基线，使用面积比作为替代
        max_area = max(left_area, right_area)
        if max_area < 1e-6:
            result["left_open_ratio"] = 1.0
            result["right_open_ratio"] = 1.0
            result["left_closure_ratio"] = 0.0
            result["right_closure_ratio"] = 0.0
            result["closure_asymmetry"] = 0.0
            result["palsy_side_suggestion"] = 0
            return result

        # 假设面积更大的眼睛是"更睁开"的
        result["left_open_ratio"] = left_area / max_area
        result["right_open_ratio"] = right_area / max_area
        result["left_closure_ratio"] = 1 - result["left_open_ratio"]
        result["right_closure_ratio"] = 1 - result["right_open_ratio"]

        # 面积更大的眼睛可能是闭合更差的（面瘫侧）
        area_ratio = left_area / right_area if right_area > 1e-6 else 1.0
        if abs(area_ratio - 1.0) < 0.15:
            result["closure_asymmetry"] = 0.0
            result["palsy_side_suggestion"] = 0
        elif left_area > right_area:
            result["closure_asymmetry"] = (left_area - right_area) / max_area
            result["palsy_side_suggestion"] = 1  # 左眼面积大=左眼闭合差=左侧面瘫
        else:
            result["closure_asymmetry"] = (right_area - left_area) / max_area
            result["palsy_side_suggestion"] = 2
        return result

    # 有基线：计算相对于基线的闭合程度
    # 需要进行尺度归一化
    scale = compute_scale_to_baseline(landmarks, baseline_landmarks, w, h)

    baseline_left_area, _ = compute_eye_area(baseline_landmarks, w, h, left=True)
    baseline_right_area, _ = compute_eye_area(baseline_landmarks, w, h, left=False)

    result["baseline_left_area"] = float(baseline_left_area)
    result["baseline_right_area"] = float(baseline_right_area)
    result["scale"] = float(scale)

    # 将当前面积换算到基线尺度 (面积用scale²)
    scaled_left_area = left_area * (scale ** 2)
    scaled_right_area = right_area * (scale ** 2)

    result["scaled_left_area"] = float(scaled_left_area)
    result["scaled_right_area"] = float(scaled_right_area)

    # 计算睁眼比例 = 当前面积 / 基线面积
    left_open_ratio = scaled_left_area / baseline_left_area if baseline_left_area > 1e-6 else 1.0
    right_open_ratio = scaled_right_area / baseline_right_area if baseline_right_area > 1e-6 else 1.0

    # 闭眼比例 = 1 - 睁眼比例 (限制在0-1范围)
    left_closure_ratio = max(0.0, min(1.0, 1.0 - left_open_ratio))
    right_closure_ratio = max(0.0, min(1.0, 1.0 - right_open_ratio))

    result["left_open_ratio"] = float(left_open_ratio)
    result["right_open_ratio"] = float(right_open_ratio)
    result["left_closure_ratio"] = float(left_closure_ratio)
    result["right_closure_ratio"] = float(right_closure_ratio)

    # 闭合不对称度
    max_closure = max(left_closure_ratio, right_closure_ratio)
    if max_closure < 0.1:
        # 两眼都几乎没闭合
        result["closure_asymmetry"] = 0.0
        result["palsy_side_suggestion"] = 0
    else:
        closure_diff = abs(left_closure_ratio - right_closure_ratio)
        result["closure_asymmetry"] = closure_diff / max_closure

        # 闭合比例低的一侧是面瘫侧
        if closure_diff < 0.15 * max_closure:  # 15%以内认为对称
            result["palsy_side_suggestion"] = 0
        elif left_closure_ratio < right_closure_ratio:
            result["palsy_side_suggestion"] = 1  # 左眼闭合差=左侧面瘫
        else:
            result["palsy_side_suggestion"] = 2  # 右眼闭合差=右侧面瘫

    return result


def compute_inner_lip_area(landmarks, w: int, h: int, left: bool = True) -> float:
    """
    计算左侧或右侧内唇面积

    Args:
        landmarks: 关键点
        w, h: 图像尺寸
        left: True=左侧内唇, False=右侧内唇

    Returns:
        内唇面积（像素平方）
    """
    indices = LM.INNER_LIP_L if left else LM.INNER_LIP_R
    points = pts2d(landmarks, indices, w, h)
    return polygon_area(points)


def compute_inner_lip_symmetry(landmarks, w: int, h: int,
                               baseline_landmarks=None) -> Dict[str, Any]:
    """
    计算左右内唇面积对称性（用于ShowTeeth面瘫侧判断）

    原理:
    - 露齿时，健侧肌肉拉力强，内唇面积增加更多
    - 面瘫侧肌肉无力，内唇面积增加少

    Args:
        landmarks: 当前帧landmarks
        w, h: 图像尺寸
        baseline_landmarks: 基线帧landmarks

    Returns:
        Dict包含:
        - left_area, right_area: 当前帧左右内唇面积
        - area_ratio: 左/右面积比
        - area_diff: 左-右面积差
        - baseline_left_area, baseline_right_area: 基线面积（如有）
        - left_change, right_change: 面积变化量（如有基线）
        - left_change_ratio, right_change_ratio: 面积变化比例
        - palsy_side_suggestion: 面瘫侧建议（面积增加少的一侧）
    """
    left_area = compute_inner_lip_area(landmarks, w, h, left=True)
    right_area = compute_inner_lip_area(landmarks, w, h, left=False)

    result = {
        "left_area": float(left_area),
        "right_area": float(right_area),
        "area_ratio": float(left_area / right_area) if right_area > 1e-6 else 1.0,
        "area_diff": float(left_area - right_area),
    }

    if baseline_landmarks is None:
        # 没有基线，直接比较当前面积
        max_area = max(left_area, right_area)
        if max_area < 1e-6:
            result["asymmetry_ratio"] = 0.0
            result["palsy_side_suggestion"] = 0
            return result

        asymmetry = abs(left_area - right_area) / max_area
        result["asymmetry_ratio"] = float(asymmetry)

        if asymmetry < 0.15:
            result["palsy_side_suggestion"] = 0
        elif left_area < right_area:
            # 左侧面积小 = 左侧肌肉弱 = 左侧面瘫
            result["palsy_side_suggestion"] = 1
        else:
            result["palsy_side_suggestion"] = 2
        return result

    # 有基线：计算变化量
    scale = compute_scale_to_baseline(landmarks, baseline_landmarks, w, h)

    baseline_left_area = compute_inner_lip_area(baseline_landmarks, w, h, left=True)
    baseline_right_area = compute_inner_lip_area(baseline_landmarks, w, h, left=False)

    result["baseline_left_area"] = float(baseline_left_area)
    result["baseline_right_area"] = float(baseline_right_area)
    result["scale"] = float(scale)

    # 换算到基线尺度
    scaled_left_area = left_area * (scale ** 2)
    scaled_right_area = right_area * (scale ** 2)

    # 面积变化量
    left_change = scaled_left_area - baseline_left_area
    right_change = scaled_right_area - baseline_right_area

    result["left_change"] = float(left_change)
    result["right_change"] = float(right_change)

    # 面积变化比例
    left_change_ratio = left_change / baseline_left_area if baseline_left_area > 1e-6 else 0.0
    right_change_ratio = right_change / baseline_right_area if baseline_right_area > 1e-6 else 0.0

    result["left_change_ratio"] = float(left_change_ratio)
    result["right_change_ratio"] = float(right_change_ratio)

    # 面瘫侧判断：面积增加少的一侧是面瘫侧
    max_change = max(abs(left_change), abs(right_change))
    if max_change < 1e-6:
        result["change_asymmetry"] = 0.0
        result["palsy_side_suggestion"] = 0
    else:
        change_diff = abs(left_change - right_change)
        result["change_asymmetry"] = float(change_diff / max_change)

        if change_diff < 0.15 * max_change:
            result["palsy_side_suggestion"] = 0
        elif left_change < right_change:
            # 左侧变化小 = 左侧肌肉弱 = 左侧面瘫
            result["palsy_side_suggestion"] = 1
        else:
            result["palsy_side_suggestion"] = 2

    return result


def compute_lip_midline_center(landmarks, w: int, h: int) -> Tuple[float, float]:
    """
    计算嘴唇中线的中心点

    使用上唇中线点(0, 13)和下唇中线点(14, 17)拟合嘴唇中线，返回中心点

    Args:
        landmarks: 关键点
        w, h: 图像尺寸

    Returns:
        (x, y): 嘴唇中线中心点坐标
    """
    # 上唇中线点
    lip_top_center = pt2d(landmarks[LM.LIP_TOP_CENTER], w, h)  # 0
    lip_top = pt2d(landmarks[LM.LIP_TOP], w, h)  # 13

    # 下唇中线点
    lip_bot = pt2d(landmarks[LM.LIP_BOT], w, h)  # 14
    lip_bot_center = pt2d(landmarks[LM.LIP_BOT_CENTER], w, h)  # 17

    # 计算四个点的质心作为嘴唇中线中心
    center_x = (lip_top_center[0] + lip_top[0] + lip_bot[0] + lip_bot_center[0]) / 4
    center_y = (lip_top_center[1] + lip_top[1] + lip_bot[1] + lip_bot_center[1]) / 4

    return (center_x, center_y)


def compute_lip_midline_offset_from_face_midline(landmarks, w: int, h: int,
                                                 baseline_landmarks=None) -> Dict[str, Any]:
    """
    计算嘴唇中线相对于面中线的偏移（用于BlowCheek/LipPucker面瘫侧判断）

    原理:
    - 面中线: 双内眦中点的x坐标
    - 嘴唇中线: 上下唇中线点的平均x坐标
    - 面瘫时嘴唇被健侧肌肉拉动，偏向健侧
    - 和基线相比，偏移变化方向指示面瘫侧

    Args:
        landmarks: 当前帧landmarks
        w, h: 图像尺寸
        baseline_landmarks: 基线帧landmarks

    Returns:
        Dict包含:
        - face_midline_x: 面中线x坐标
        - lip_midline_x: 嘴唇中线x坐标
        - current_offset: 当前偏移 (lip_x - face_x, 正=偏左，负=偏右)
        - baseline_offset: 基线偏移（如有）
        - offset_change: 偏移变化量（如有基线）
        - palsy_side_suggestion: 面瘫侧建议
    """
    # 面中线
    left_canthus = pt2d(landmarks[LM.EYE_INNER_L], w, h)
    right_canthus = pt2d(landmarks[LM.EYE_INNER_R], w, h)
    face_midline_x = (left_canthus[0] + right_canthus[0]) / 2

    # 嘴唇中线中心
    lip_center = compute_lip_midline_center(landmarks, w, h)
    lip_midline_x = lip_center[0]

    # 当前偏移
    current_offset = lip_midline_x - face_midline_x

    result = {
        "face_midline_x": float(face_midline_x),
        "lip_midline_x": float(lip_midline_x),
        "lip_midline_y": float(lip_center[1]),
        "current_offset": float(current_offset),
    }

    if baseline_landmarks is None:
        # 没有基线，直接根据当前偏移判断
        icd = compute_icd(landmarks, w, h)
        offset_norm = abs(current_offset) / icd if icd > 1e-6 else 0
        result["offset_norm"] = float(offset_norm)

        if offset_norm < 0.025:  # 以内认为对称
            result["palsy_side_suggestion"] = 0
        elif current_offset > 0:
            # 偏向患者左侧 = 被左侧拉 = 左侧健侧 = 右侧面瘫
            result["palsy_side_suggestion"] = 2
        else:
            result["palsy_side_suggestion"] = 1
        return result

    # 有基线：计算偏移变化
    baseline_lip_center = compute_lip_midline_center(baseline_landmarks, w, h)
    baseline_left_canthus = pt2d(baseline_landmarks[LM.EYE_INNER_L], w, h)
    baseline_right_canthus = pt2d(baseline_landmarks[LM.EYE_INNER_R], w, h)
    baseline_face_midline_x = (baseline_left_canthus[0] + baseline_right_canthus[0]) / 2

    baseline_offset = baseline_lip_center[0] - baseline_face_midline_x

    result["baseline_offset"] = float(baseline_offset)

    # 偏移变化量（考虑尺度）
    scale = compute_scale_to_baseline(landmarks, baseline_landmarks, w, h)
    scaled_current_offset = current_offset * scale

    offset_change = scaled_current_offset - baseline_offset
    result["offset_change"] = float(offset_change)
    result["scale"] = float(scale)

    # 归一化偏移变化
    icd_base = compute_icd(baseline_landmarks, w, h)
    offset_change_norm = abs(offset_change) / icd_base if icd_base > 1e-6 else 0
    result["offset_change_norm"] = float(offset_change_norm)

    # 面瘫侧判断：偏移变化方向
    if offset_change_norm < 0.02:  # 2%以内认为无变化
        result["palsy_side_suggestion"] = 0
    elif offset_change > 0:
        # 相对基线偏向左侧 = 被左侧拉 = 右侧面瘫
        result["palsy_side_suggestion"] = 2
    else:
        result["palsy_side_suggestion"] = 1

    return result


def compute_mouth_corner_to_eye_line_distance(landmarks, w: int, h: int,
                                              left: bool = True) -> Dict[str, Any]:
    """
    计算嘴角到眼部水平线的距离（用于Smile关键帧选择和面瘫侧判断）

    原理:
    - 眼部水平线: 双内眦连线
    - 微笑时嘴角上提，嘴角到眼部水平线的距离减小
    - 距离减小最多的帧是微笑峰值帧

    Args:
        landmarks: 关键点
        w, h: 图像尺寸
        left: True=左嘴角, False=右嘴角

    Returns:
        Dict包含:
        - corner: 嘴角坐标
        - eye_line_y: 眼部水平线y坐标（内眦平均y）
        - distance: 嘴角到眼部水平线的垂直距离
    """
    # 眼部水平线（双内眦平均y坐标）
    left_canthus = pt2d(landmarks[LM.EYE_INNER_L], w, h)
    right_canthus = pt2d(landmarks[LM.EYE_INNER_R], w, h)
    eye_line_y = (left_canthus[1] + right_canthus[1]) / 2

    # 嘴角
    if left:
        corner = pt2d(landmarks[LM.MOUTH_L], w, h)
    else:
        corner = pt2d(landmarks[LM.MOUTH_R], w, h)

    # 垂直距离（在图像坐标系中，y向下为正，所以corner_y > eye_line_y表示嘴角在眼睛下方）
    distance = corner[1] - eye_line_y

    return {
        "corner": corner,
        "eye_line_y": float(eye_line_y),
        "distance": float(distance),
    }


def compute_smile_excursion_by_eye_line(landmarks, w: int, h: int,
                                        baseline_landmarks=None) -> Dict[str, Any]:
    """
    计算微笑运动幅度（基于嘴角到眼部水平线的距离变化）

    原理:
    - 微笑时嘴角上提，到眼部水平线的距离减小
    - 健侧距离减小更多（嘴角上提更高）
    - 面瘫侧距离减小较少

    Args:
        landmarks: 当前帧landmarks
        w, h: 图像尺寸
        baseline_landmarks: 基线帧landmarks

    Returns:
        Dict包含:
        - left_distance, right_distance: 当前帧左右嘴角到眼线距离
        - baseline_left_distance, baseline_right_distance: 基线距离（如有）
        - left_change, right_change: 距离变化量（负值=上提）
        - excursion_ratio: 运动比例（min/max变化量）
        - palsy_side_suggestion: 面瘫侧建议（变化小的一侧）
    """
    left_result = compute_mouth_corner_to_eye_line_distance(landmarks, w, h, left=True)
    right_result = compute_mouth_corner_to_eye_line_distance(landmarks, w, h, left=False)

    result = {
        "left_distance": left_result["distance"],
        "right_distance": right_result["distance"],
        "eye_line_y": left_result["eye_line_y"],
        "left_corner": left_result["corner"],
        "right_corner": right_result["corner"],
    }

    if baseline_landmarks is None:
        # 没有基线，直接比较当前距离
        dist_diff = abs(left_result["distance"] - right_result["distance"])
        avg_dist = (left_result["distance"] + right_result["distance"]) / 2

        if avg_dist < 1e-6:
            result["asymmetry_ratio"] = 0.0
            result["palsy_side_suggestion"] = 0
        else:
            result["asymmetry_ratio"] = float(dist_diff / avg_dist)

            if result["asymmetry_ratio"] < 0.10:
                result["palsy_side_suggestion"] = 0
            elif left_result["distance"] > right_result["distance"]:
                # 左嘴角距离大（没上提）= 左侧面瘫
                result["palsy_side_suggestion"] = 1
            else:
                result["palsy_side_suggestion"] = 2
        return result

    # 有基线：计算变化量
    scale = compute_scale_to_baseline(landmarks, baseline_landmarks, w, h)

    baseline_left = compute_mouth_corner_to_eye_line_distance(baseline_landmarks, w, h, left=True)
    baseline_right = compute_mouth_corner_to_eye_line_distance(baseline_landmarks, w, h, left=False)

    result["baseline_left_distance"] = baseline_left["distance"]
    result["baseline_right_distance"] = baseline_right["distance"]
    result["scale"] = float(scale)

    # 换算到基线尺度后的距离
    scaled_left_dist = left_result["distance"] * scale
    scaled_right_dist = right_result["distance"] * scale

    # 变化量（负值表示距离减小=嘴角上提）
    left_change = scaled_left_dist - baseline_left["distance"]
    right_change = scaled_right_dist - baseline_right["distance"]

    result["left_change"] = float(left_change)
    result["right_change"] = float(right_change)

    # 运动比例
    # 微笑时两侧都应该是负值（距离减小）
    # 面瘫侧减小得少（负值绝对值小，或为正值）
    left_reduction = -left_change  # 转为正值表示减小量
    right_reduction = -right_change

    result["left_reduction"] = float(left_reduction)
    result["right_reduction"] = float(right_reduction)

    max_reduction = max(left_reduction, right_reduction)
    min_reduction = min(left_reduction, right_reduction)

    if max_reduction < 1e-6:
        result["excursion_ratio"] = 1.0
        result["palsy_side_suggestion"] = 0
    else:
        result["excursion_ratio"] = float(min_reduction / max_reduction) if max_reduction > 1e-6 else 1.0
        reduction_diff = abs(left_reduction - right_reduction)

        if reduction_diff < 0.15 * max_reduction:
            result["palsy_side_suggestion"] = 0
        elif left_reduction < right_reduction:
            # 左侧减小量小（嘴角没上提）= 左侧面瘫
            result["palsy_side_suggestion"] = 1
        else:
            result["palsy_side_suggestion"] = 2

    return result


def compute_ala_canthus_change(landmarks, w: int, h: int,
                               baseline_landmarks=None) -> Dict[str, Any]:
    """
    计算鼻翼-内眦距离变化量（用于ShrugNose面瘫侧判断）

    原理:
    - 皱鼻时鼻翼上提向内收缩，鼻翼-内眦距离减小
    - 健侧距离减小更多
    - 面瘫侧距离减小较少

    Args:
        landmarks: 当前帧landmarks
        w, h: 图像尺寸
        baseline_landmarks: 基线帧landmarks

    Returns:
        Dict包含:
        - left_distance, right_distance: 当前帧距离
        - baseline_left_distance, baseline_right_distance: 基线距离
        - left_change, right_change: 变化量（负值=距离减小）
        - left_change_ratio, right_change_ratio: 变化比例
        - palsy_side_suggestion: 面瘫侧建议（变化小的一侧）
    """
    # 当前帧距离
    left_ala = pt2d(landmarks[LM.NOSE_ALA_L], w, h)
    right_ala = pt2d(landmarks[LM.NOSE_ALA_R], w, h)
    left_canthus = pt2d(landmarks[LM.EYE_INNER_L], w, h)
    right_canthus = pt2d(landmarks[LM.EYE_INNER_R], w, h)

    left_dist = dist(left_ala, left_canthus)
    right_dist = dist(right_ala, right_canthus)

    result = {
        "left_distance": float(left_dist),
        "right_distance": float(right_dist),
        "distance_ratio": float(left_dist / right_dist) if right_dist > 1e-6 else 1.0,
    }

    if baseline_landmarks is None:
        # 没有基线，直接比较当前距离（距离大=收缩弱=面瘫）
        max_dist = max(left_dist, right_dist)
        if max_dist < 1e-6:
            result["asymmetry_ratio"] = 0.0
            result["palsy_side_suggestion"] = 0
        else:
            diff = abs(left_dist - right_dist)
            result["asymmetry_ratio"] = float(diff / max_dist)

            if result["asymmetry_ratio"] < 0.10:
                result["palsy_side_suggestion"] = 0
            elif left_dist > right_dist:
                result["palsy_side_suggestion"] = 1  # 左侧距离大=左侧收缩弱=左侧面瘫
            else:
                result["palsy_side_suggestion"] = 2
        return result

    # 有基线：计算变化量
    scale = compute_scale_to_baseline(landmarks, baseline_landmarks, w, h)

    baseline_left_ala = pt2d(baseline_landmarks[LM.NOSE_ALA_L], w, h)
    baseline_right_ala = pt2d(baseline_landmarks[LM.NOSE_ALA_R], w, h)
    baseline_left_canthus = pt2d(baseline_landmarks[LM.EYE_INNER_L], w, h)
    baseline_right_canthus = pt2d(baseline_landmarks[LM.EYE_INNER_R], w, h)

    baseline_left_dist = dist(baseline_left_ala, baseline_left_canthus)
    baseline_right_dist = dist(baseline_right_ala, baseline_right_canthus)

    result["baseline_left_distance"] = float(baseline_left_dist)
    result["baseline_right_distance"] = float(baseline_right_dist)
    result["scale"] = float(scale)

    # 换算到基线尺度
    scaled_left_dist = left_dist * scale
    scaled_right_dist = right_dist * scale

    # 变化量（负值=距离减小=收缩）
    left_change = scaled_left_dist - baseline_left_dist
    right_change = scaled_right_dist - baseline_right_dist

    result["left_change"] = float(left_change)
    result["right_change"] = float(right_change)

    # 变化比例
    left_change_ratio = left_change / baseline_left_dist if baseline_left_dist > 1e-6 else 0.0
    right_change_ratio = right_change / baseline_right_dist if baseline_right_dist > 1e-6 else 0.0

    result["left_change_ratio"] = float(left_change_ratio)
    result["right_change_ratio"] = float(right_change_ratio)

    # 面瘫侧判断：皱鼻时距离应减小，减小少的是面瘫侧
    left_reduction = -left_change  # 正值表示减小量
    right_reduction = -right_change

    result["left_reduction"] = float(left_reduction)
    result["right_reduction"] = float(right_reduction)

    max_reduction = max(left_reduction, right_reduction)

    if max_reduction < 1e-6:
        result["reduction_asymmetry"] = 0.0
        result["palsy_side_suggestion"] = 0
    else:
        reduction_diff = abs(left_reduction - right_reduction)
        result["reduction_asymmetry"] = float(reduction_diff / max_reduction)

        if reduction_diff < 0.15 * max_reduction:
            result["palsy_side_suggestion"] = 0
        elif left_reduction < right_reduction:
            # 左侧减小量小=左侧收缩弱=左侧面瘫
            result["palsy_side_suggestion"] = 1
        else:
            result["palsy_side_suggestion"] = 2

    return result


# =============================================================================
# 眼睛闭合度序列计算
# =============================================================================

def compute_eye_closure_sequence(landmarks_seq: List, w: int, h: int,
                                 baseline_landmarks=None) -> Dict[str, Any]:
    """
    计算整个视频序列的眼睛闭合度

    闭合度定义: closure_ratio = 1 - (current_area / baseline_area)
    - 0 = 完全睁开（面积等于基线）
    - 1 = 完全闭合（面积为0）

    Args:
        landmarks_seq: 关键点序列
        w, h: 图像尺寸
        baseline_landmarks: 基线关键点（静息帧）

    Returns:
        {
            "left_closure": [闭合度序列],
            "right_closure": [闭合度序列],
            "left_area": [面积序列],
            "right_area": [面积序列],
            "baseline_left_area": 基线左眼面积,
            "baseline_right_area": 基线右眼面积,
            "baseline_source": "neutral" | "video_init",
        }
    """
    n = len(landmarks_seq)

    left_area_seq = []
    right_area_seq = []
    left_closure_seq = []
    right_closure_seq = []

    # 获取基线面积
    baseline_source = "neutral"
    if baseline_landmarks is not None:
        baseline_left_area, _ = compute_eye_area(baseline_landmarks, w, h, True)
        baseline_right_area, _ = compute_eye_area(baseline_landmarks, w, h, False)
    else:
        # 用视频前10帧的最大面积作为baseline（假设开始时眼睛睁开）
        baseline_source = "video_init"
        init_left = []
        init_right = []
        for lm in landmarks_seq[:min(10, n)]:
            if lm is not None:
                la, _ = compute_eye_area(lm, w, h, True)
                ra, _ = compute_eye_area(lm, w, h, False)
                if la > 0:
                    init_left.append(la)
                if ra > 0:
                    init_right.append(ra)

        # 使用90百分位数而非最大值，更鲁棒
        baseline_left_area = float(np.percentile(init_left, 90)) if init_left else 1.0
        baseline_right_area = float(np.percentile(init_right, 90)) if init_right else 1.0

    # 确保baseline有效
    baseline_left_area = max(baseline_left_area, 1e-6)
    baseline_right_area = max(baseline_right_area, 1e-6)

    for lm in landmarks_seq:
        if lm is None:
            left_area_seq.append(np.nan)
            right_area_seq.append(np.nan)
            left_closure_seq.append(np.nan)
            right_closure_seq.append(np.nan)
        else:
            # 计算尺度因子（补偿距离变化）
            if baseline_landmarks is not None:
                scale = compute_scale_to_baseline(lm, baseline_landmarks, w, h)
            else:
                scale = 1.0

            l_area, _ = compute_eye_area(lm, w, h, True)
            r_area, _ = compute_eye_area(lm, w, h, False)

            # 尺度校正
            scaled_l = l_area * (scale ** 2)
            scaled_r = r_area * (scale ** 2)

            left_area_seq.append(float(scaled_l))
            right_area_seq.append(float(scaled_r))

            # 闭合度 = 1 - (当前面积/基线面积)
            l_closure = 1.0 - (scaled_l / baseline_left_area)
            r_closure = 1.0 - (scaled_r / baseline_right_area)

            # 限制在 0-1 范围
            left_closure_seq.append(float(max(0, min(1, l_closure))))
            right_closure_seq.append(float(max(0, min(1, r_closure))))

    return {
        "left_closure": left_closure_seq,
        "right_closure": right_closure_seq,
        "left_area": left_area_seq,
        "right_area": right_area_seq,
        "baseline_left_area": float(baseline_left_area),
        "baseline_right_area": float(baseline_right_area),
        "baseline_source": baseline_source,
    }


# =============================================================================
# 眼睛同步度计算
# =============================================================================

def compute_eye_synchrony(left_closure_seq: List[float],
                          right_closure_seq: List[float]) -> Dict[str, Any]:
    """
    计算左右眼的同步度

    衡量两只眼睛运动的时间同步性和幅度一致性

    指标：
    1. Pearson相关系数：时间同步性（-1~1，越接近1越同步）
    2. 峰值时间差：两眼达到最大闭合的帧差
    3. 闭合度差异：幅度一致性

    Args:
        left_closure_seq: 左眼闭合度序列
        right_closure_seq: 右眼闭合度序列

    Returns:
        {
            "pearson_correlation": 相关系数,
            "peak_frame_diff": 峰值帧差异,
            "left_peak_frame": 左眼峰值帧,
            "right_peak_frame": 右眼峰值帧,
            "mean_closure_diff": 平均闭合度差异,
            "max_closure_diff": 最大闭合度差异,
            "synchrony_score": 综合同步度得分 (0-1),
            "valid_frames": 有效帧数,
        }
    """
    left_arr = np.array(left_closure_seq, dtype=np.float64)
    right_arr = np.array(right_closure_seq, dtype=np.float64)

    # 过滤无效值
    valid_mask = np.isfinite(left_arr) & np.isfinite(right_arr)
    valid_count = int(valid_mask.sum())

    if valid_count < 5:
        return {
            "pearson_correlation": 0.0,
            "peak_frame_diff": 0,
            "left_peak_frame": 0,
            "right_peak_frame": 0,
            "mean_closure_diff": 0.0,
            "max_closure_diff": 0.0,
            "synchrony_score": 0.5,
            "valid_frames": valid_count,
            "status": "insufficient_data",
        }

    left_valid = left_arr[valid_mask]
    right_valid = right_arr[valid_mask]

    # 1. Pearson 相关系数
    if np.std(left_valid) > 1e-6 and np.std(right_valid) > 1e-6:
        pearson_corr = float(np.corrcoef(left_valid, right_valid)[0, 1])
        if np.isnan(pearson_corr):
            pearson_corr = 1.0
    else:
        pearson_corr = 1.0  # 无变化时认为同步

    # 2. 峰值帧差异
    # 使用整个序列（包含nan的位置也要考虑）
    left_peak_idx = int(np.nanargmax(left_arr)) if np.any(np.isfinite(left_arr)) else 0
    right_peak_idx = int(np.nanargmax(right_arr)) if np.any(np.isfinite(right_arr)) else 0
    peak_frame_diff = abs(left_peak_idx - right_peak_idx)

    # 3. 闭合度差异
    closure_diff = np.abs(left_valid - right_valid)
    mean_closure_diff = float(np.mean(closure_diff))
    max_closure_diff = float(np.max(closure_diff))

    # 4. 综合同步度得分
    # 权重：相关性(50%) + 峰值同步(30%) + 幅度一致(20%)
    corr_score = (pearson_corr + 1) / 2  # 映射到 0-1
    peak_score = max(0, 1 - peak_frame_diff / 10)  # 10帧以上差异得0分
    diff_score = max(0, 1 - mean_closure_diff * 5)  # 20%平均差异得0分

    synchrony_score = 0.5 * corr_score + 0.3 * peak_score + 0.2 * diff_score

    return {
        "pearson_correlation": float(pearson_corr),
        "peak_frame_diff": int(peak_frame_diff),
        "left_peak_frame": int(left_peak_idx),
        "right_peak_frame": int(right_peak_idx),
        "mean_closure_diff": float(mean_closure_diff),
        "max_closure_diff": float(max_closure_diff),
        "synchrony_score": float(synchrony_score),
        "valid_frames": valid_count,
        "status": "ok",
    }


# =============================================================================
# 峰值帧眼睛对称度计算
# =============================================================================

def compute_eye_symmetry_at_peak(left_closure: float, right_closure: float,
                                 threshold: float = 0.15) -> Dict[str, Any]:
    """
    计算峰值帧的左右眼对称度

    Args:
        left_closure: 左眼闭合度 (0-1)
        right_closure: 右眼闭合度 (0-1)
        threshold: 不对称判定阈值（默认15%）

    Returns:
        {
            "left_closure": 左眼闭合度,
            "right_closure": 右眼闭合度,
            "closure_diff": 闭合度差值（绝对值）,
            "asymmetry_ratio": 不对称比例,
            "symmetry_score": 对称度得分 (0-1, 1=完全对称),
            "palsy_side": 患侧判断 (0=对称, 1=左, 2=右),
            "interpretation": 文字解释,
        }
    """
    # 处理无效值
    if not np.isfinite(left_closure) or not np.isfinite(right_closure):
        return {
            "left_closure": float(left_closure) if np.isfinite(left_closure) else 0.0,
            "right_closure": float(right_closure) if np.isfinite(right_closure) else 0.0,
            "closure_diff": 0.0,
            "asymmetry_ratio": 0.0,
            "symmetry_score": 1.0,
            "palsy_side": 0,
            "interpretation": "数据无效",
        }

    closure_diff = abs(left_closure - right_closure)
    max_closure = max(left_closure, right_closure)

    # 闭合幅度太小，无法判断
    if max_closure < 0.15:
        return {
            "left_closure": float(left_closure),
            "right_closure": float(right_closure),
            "closure_diff": float(closure_diff),
            "asymmetry_ratio": 0.0,
            "symmetry_score": 1.0,
            "palsy_side": 0,
            "interpretation": f"闭眼幅度过小 (L={left_closure:.1%}, R={right_closure:.1%})，无法判断",
        }

    asymmetry_ratio = closure_diff / max_closure
    symmetry_score = max(0, 1 - asymmetry_ratio)

    # 判断患侧
    if asymmetry_ratio < threshold:
        palsy_side = 0
        interpretation = f"双眼闭合对称 (L={left_closure:.1%}, R={right_closure:.1%}, 差异{asymmetry_ratio:.1%})"
    elif left_closure < right_closure:
        # 左眼闭合程度低 = 左眼闭不上 = 左侧面瘫
        palsy_side = 1
        interpretation = f"左眼闭合较差 (L={left_closure:.1%} < R={right_closure:.1%}) → 左侧面瘫"
    else:
        palsy_side = 2
        interpretation = f"右眼闭合较差 (R={right_closure:.1%} < L={left_closure:.1%}) → 右侧面瘫"

    return {
        "left_closure": float(left_closure),
        "right_closure": float(right_closure),
        "closure_diff": float(closure_diff),
        "asymmetry_ratio": float(asymmetry_ratio),
        "symmetry_score": float(symmetry_score),
        "palsy_side": int(palsy_side),
        "interpretation": interpretation,
    }


# =============================================================================
# 综合眼睛分析
# =============================================================================

def analyze_eye_closure_full(landmarks_seq: List, w: int, h: int,
                             baseline_landmarks=None,
                             peak_idx: int = None) -> Dict[str, Any]:
    """
    完整的眼睛闭合分析（整合闭合度序列、同步度、对称度）

    Args:
        landmarks_seq: 关键点序列
        w, h: 图像尺寸
        baseline_landmarks: 基线关键点
        peak_idx: 峰值帧索引（如果不提供，自动找最大闭合帧）

    Returns:
        完整的分析结果，包含所有指标
    """
    # 1. 计算闭合度序列
    closure_data = compute_eye_closure_sequence(landmarks_seq, w, h, baseline_landmarks)

    # 2. 自动找峰值帧（如果没提供）
    if peak_idx is None:
        left_arr = np.array(closure_data["left_closure"])
        right_arr = np.array(closure_data["right_closure"])
        avg_arr = (left_arr + right_arr) / 2
        peak_idx = int(np.nanargmax(avg_arr)) if np.any(np.isfinite(avg_arr)) else 0

    # 3. 计算同步度
    synchrony = compute_eye_synchrony(
        closure_data["left_closure"],
        closure_data["right_closure"]
    )

    # 4. 计算峰值帧对称度
    if 0 <= peak_idx < len(closure_data["left_closure"]):
        left_at_peak = closure_data["left_closure"][peak_idx]
        right_at_peak = closure_data["right_closure"][peak_idx]
    else:
        left_at_peak = 0.0
        right_at_peak = 0.0

    symmetry = compute_eye_symmetry_at_peak(left_at_peak, right_at_peak)

    # 5. 综合诊断
    diagnosis = {
        "palsy_side": symmetry["palsy_side"],
        "confidence": 1.0 - symmetry["symmetry_score"],
        "symmetry_score": symmetry["symmetry_score"],
        "synchrony_score": synchrony["synchrony_score"],
        "method": "area_closure_analysis",
    }

    return {
        "closure_sequence": {
            "left": [float(v) if np.isfinite(v) else None for v in closure_data["left_closure"]],
            "right": [float(v) if np.isfinite(v) else None for v in closure_data["right_closure"]],
        },
        "area_sequence": {
            "left": [float(v) if np.isfinite(v) else None for v in closure_data["left_area"]],
            "right": [float(v) if np.isfinite(v) else None for v in closure_data["right_area"]],
        },
        "baseline": {
            "left_area": closure_data["baseline_left_area"],
            "right_area": closure_data["baseline_right_area"],
            "source": closure_data["baseline_source"],
        },
        "peak_frame": {
            "idx": int(peak_idx),
            "left_closure": float(left_at_peak),
            "right_closure": float(right_at_peak),
        },
        "symmetry_analysis": symmetry,
        "synchrony_analysis": synchrony,
        "diagnosis": diagnosis,
    }


# === Add these new functions to the end of clinical_base.py ===

def detect_synkinesis_from_mouth(baseline_result: Optional[ActionResult],
                                 current_landmarks, w: int, h: int) -> Dict[str, int]:
    """检测嘴部动作时的眼部联动"""
    synkinesis = {"eye_synkinesis": 0, "brow_synkinesis": 0}
    if baseline_result is None: return synkinesis

    l_ear, r_ear = compute_ear(current_landmarks, w, h, True), compute_ear(current_landmarks, w, h, False)
    bl_ear, br_ear = baseline_result.left_ear, baseline_result.right_ear

    if bl_ear > 1e-9 and br_ear > 1e-9:
        l_change = (bl_ear - l_ear) / bl_ear
        r_change = (br_ear - r_ear) / br_ear
        avg_change = (l_change + r_change) / 2

        if avg_change > THR.MOUTH_SYNKINESIS_EYE_SEVERE:
            synkinesis["eye_synkinesis"] = 3
        elif avg_change > THR.MOUTH_SYNKINESIS_EYE_MODERATE:
            synkinesis["eye_synkinesis"] = 2
        elif avg_change > THR.MOUTH_SYNKINESIS_EYE_MILD:
            synkinesis["eye_synkinesis"] = 1
    return synkinesis


def detect_synkinesis_from_brow(baseline_result: Optional[ActionResult],
                                current_landmarks, w: int, h: int) -> Dict[str, int]:
    """检测抬眉时的嘴部联动"""
    synkinesis = {"mouth_synkinesis": 0}
    if baseline_result is None: return synkinesis

    mouth = compute_mouth_metrics(current_landmarks, w, h)
    if baseline_result.mouth_width > 1e-9:
        mouth_change = abs(mouth["width"] - baseline_result.mouth_width) / baseline_result.mouth_width
        if mouth_change > THR.RAISE_EYEBROW_SYNKINESIS_MOUTH_SEVERE:
            synkinesis["mouth_synkinesis"] = 3
        elif mouth_change > THR.RAISE_EYEBROW_SYNKINESIS_MOUTH_MODERATE:
            synkinesis["mouth_synkinesis"] = 2
        elif mouth_change > THR.RAISE_EYEBROW_SYNKINESIS_MOUTH_MILD:
            synkinesis["mouth_synkinesis"] = 1
    return synkinesis


def compute_voluntary_score_from_excursion(excursion_metrics: Dict[str, Any]) -> Tuple[int, str]:
    """根据运动幅度计算Voluntary Score"""
    left_exc = excursion_metrics.get("left_reduction") or excursion_metrics.get("left_total", 0)
    right_exc = excursion_metrics.get("right_reduction") or excursion_metrics.get("right_total", 0)

    max_exc = max(left_exc, right_exc)
    if max_exc < 3: return 1, "运动幅度过小"

    symmetry_ratio = min(left_exc, right_exc) / max_exc if max_exc > 1e-6 else 1.0

    if symmetry_ratio >= THR.SMILE_VOL_SYMM_COMPLETE:
        return 5, "运动完整"
    elif symmetry_ratio >= THR.SMILE_VOL_SYMM_ALMOST:
        return 4, "几乎完整"
    elif symmetry_ratio >= THR.SMILE_VOL_SYMM_INITIATE:
        return 3, "启动但不对称"
    elif symmetry_ratio >= THR.SMILE_VOL_SYMM_SLIGHT:
        return 2, "轻微启动"
    else:
        return 1, "无法启动"

