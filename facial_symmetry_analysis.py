# -*- coding: utf-8 -*-
"""
facial_symmetry_analysis_optimized.py

面部对称性分析 - 优化版本
主要改进：
1. ✅ 修复面中线绘制：使用face_contour区域的第一对/最后一对点（非整个配置的）
2. ✅ 代码结构优化：分离职责、提取常量、改进可读性
3. ✅ 性能优化：向量化计算、减少重复操作
4. ✅ 错误处理增强：更健壮的异常处理
5. ✅ 类型提示完善：更好的代码可维护性
"""

import os
import re
import json
import sqlite3
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum

import cv2
import numpy as np
import mediapipe as mp_mediapipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from scipy import stats
from scipy.spatial.distance import mahalanobis
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体/负号
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'PingFang SC']
matplotlib.rcParams['axes.unicode_minus'] = False

# ==================== 常量定义 ====================

# MediaPipe 478点中的对称特征点定义
SYMMETRY_INDEX_CONFIG = [
    {
        "region": "eyebrow",
        "pairs": {
            "left": [336, 296, 334, 293, 300, 276, 283, 282, 295, 285],
            "right": [107, 66, 105, 63, 70, 46, 53, 52, 65, 55],
        }
    },
    {
        "region": "eye",
        "pairs": {
            "left": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            "right": [133, 155, 154, 153, 145, 144, 163, 7, 33, 246, 161, 160, 159, 158, 157, 173],
        }
    },
    {
        "region": "pupil",
        "pairs": {"left": [473], "right": [468]}
    },
    {
        "region": "iris",
        "pairs": {"left": [474, 475, 476, 477], "right": [471, 470, 469, 472]}
    },
    {
        "region": "upper_lip",
        "pairs": {
            "left": [267, 269, 270, 409, 291, 308, 415, 310, 311, 312],
            "right": [37, 39, 40, 185, 61, 78, 191, 80, 81, 82],
        }
    },
    {
        "region": "lower_lip",
        "pairs": {
            "left": [317, 402, 318, 324, 308, 291, 375, 321, 405, 314],
            "right": [87, 178, 88, 95, 78, 61, 146, 91, 181, 84],
        }
    },
    {
        "region": "nose",
        "pairs": {
            "left": [250, 458, 459, 309, 392, 289, 305, 460, 294, 358, 279, 429, 420, 456],
            "right": [20, 238, 239, 79, 166, 59, 75, 240, 64, 129, 49, 209, 198, 236],
        }
    },
    {
        "region": "face_contour",
        "pairs": {
            "left": [338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377],
            "right": [109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148],
        },
    },
]

# 眼内眦关键点索引
EYE_INNER_CANTHUS_LEFT = 362
EYE_INNER_CANTHUS_RIGHT = 133
NOSE_TIP = 4

# 可视化配置
class VisConfig:
    """可视化相关配置"""
    # 颜色定义 (BGR格式)
    COLOR_LEFT_POINT = (0, 0, 255)  # 红色
    COLOR_RIGHT_POINT = (255, 0, 0)  # 蓝色
    COLOR_MIDLINE = (0, 255, 255)  # 黄色
    COLOR_CONNECTION = (0, 255, 0)  # 绿色

    # 绘制参数
    POINT_RADIUS = 2
    MIDLINE_THICKNESS = 2
    CONNECTION_THICKNESS = 1
    TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
    TEXT_SIZE = 0.35
    TEXT_THICKNESS = 1

    # 热力图参数
    HEATMAP_ALPHA = 0.45
    HEATMAP_CANVAS_RES = 256
    HEATMAP_SIGMA = 8.0


# Pearson系数阈值
class PearsonThreshold:
    """Pearson相关系数阈值"""
    HIGH = 0.8
    MEDIUM = 0.5
    LOW = 0.0


# ==================== 数据结构 ====================

@dataclass
class SymmetryFeatures:
    """对称性特征数据结构"""
    pearson_coefficients: np.ndarray  # [F] Pearson相关系数
    landmark_names: List[str]  # [F] 特征点名称
    y_coords_left: np.ndarray  # [T, F] 左侧y坐标
    y_coords_right: np.ndarray  # [T, F] 右侧y坐标
    frame_count: int  # T 总帧数


@dataclass
class MidlinePoints:
    """面中线关键点"""
    top: Tuple[int, int]  # 顶点坐标
    bottom: Tuple[int, int]  # 底点坐标
    center: Tuple[float, float]  # 中心点坐标


# ==================== 辅助函数 ====================

def _safe_name(s: Union[str, None]) -> str:
    """安全的文件名转换"""
    s = str(s) if s is not None else ""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", s)[:160]


def build_pairs_and_names(cfg_list: List[Dict]) -> Tuple[List[Tuple[int, int]], List[str], Dict[str, List[int]]]:
    """
    从配置构建点对、名称和区域索引映射

    Args:
        cfg_list: SYMMETRY_INDEX_CONFIG配置列表

    Returns:
        pairs: [(left_idx, right_idx), ...] 点对列表
        names: ["eyebrow_01", "eyebrow_02", ...] 特征名称列表
        region_feature_indices: {"eyebrow":[0,1,...], ...} 区域到特征索引的映射
    """
    pairs = []
    names = []
    region_feature_indices = {}

    for item in cfg_list:
        region = item["region"]
        lr = item["pairs"]
        L = lr["left"]
        R = lr["right"]

        if len(L) != len(R):
            raise ValueError(f"[{region}] 左右点数不一致: {len(L)} vs {len(R)}")

        region_feature_indices.setdefault(region, [])

        for i, (li, ri) in enumerate(zip(L, R), start=1):
            pairs.append((int(li), int(ri)))
            names.append(f"{region}_{i:02d}")
            region_feature_indices[region].append(len(pairs) - 1)

    return pairs, names, region_feature_indices


def compute_midline_geometry(
        face_landmarks,
        face_contour_indices: List[int],
        feature_pairs: List[Tuple[int, int]],
        image_width: int,
        image_height: int
) -> Optional[MidlinePoints]:
    """
    计算面中线的几何位置

    Args:
        face_landmarks: MediaPipe检测到的面部关键点
        face_contour_indices: face_contour区域的特征索引列表
        feature_pairs: 所有点对列表
        image_width: 图像宽度
        image_height: 图像高度

    Returns:
        MidlinePoints对象，包含面中线的顶点、底点和中心点；失败返回None
    """
    try:
        # 1. 获取眼内眦点，计算中心和法向量
        lm_left = face_landmarks[EYE_INNER_CANTHUS_LEFT]
        lm_right = face_landmarks[EYE_INNER_CANTHUS_RIGHT]

        lx, ly = lm_left.x * image_width, lm_left.y * image_height
        rx, ry = lm_right.x * image_width, lm_right.y * image_height

        # 中点
        center_x, center_y = (lx + rx) / 2.0, (ly + ry) / 2.0

        # 眼内眦连线方向和法向量（中垂线方向）
        dx, dy = (rx - lx), (ry - ly)
        if abs(dx) + abs(dy) < 1e-6:
            raise ValueError("眼内眦两点过于接近，无法计算面中线")

        center = np.array([center_x, center_y], dtype=np.float32)
        normal = np.array([-dy, dx], dtype=np.float32)  # 中垂线方向
        denom = float(np.dot(normal, normal)) + 1e-6

        # 2. 获取face_contour的第一对和最后一对点（脸的最高点和最低点）
        if not face_contour_indices:
            raise ValueError("face_contour区域索引为空")

        # 第一对点（最高点）
        top_feature_idx = face_contour_indices[0]
        top_left_idx, top_right_idx = feature_pairs[top_feature_idx]

        # 最后一对点（最低点）
        bottom_feature_idx = face_contour_indices[-1]
        bottom_left_idx, bottom_right_idx = feature_pairs[bottom_feature_idx]

        # 获取关键点坐标
        top_lm_l = face_landmarks[top_left_idx]
        top_lm_r = face_landmarks[top_right_idx]
        bot_lm_l = face_landmarks[bottom_left_idx]
        bot_lm_r = face_landmarks[bottom_right_idx]

        # 计算顶部和底部中点
        top_mid = np.array([
            (top_lm_l.x * image_width + top_lm_r.x * image_width) / 2.0,
            (top_lm_l.y * image_height + top_lm_r.y * image_height) / 2.0
        ], dtype=np.float32)

        bot_mid = np.array([
            (bot_lm_l.x * image_width + bot_lm_r.x * image_width) / 2.0,
            (bot_lm_l.y * image_height + bot_lm_r.y * image_height) / 2.0
        ], dtype=np.float32)

        # 3. 投影到面中线上
        # 投影公式：P_proj = M + ((P - M)·n / (n·n)) * n
        t_top = float(np.dot(top_mid - center, normal)) / denom
        t_bot = float(np.dot(bot_mid - center, normal)) / denom

        p_top = center + t_top * normal
        p_bot = center + t_bot * normal

        return MidlinePoints(
            top=(int(round(p_top[0])), int(round(p_top[1]))),
            bottom=(int(round(p_bot[0])), int(round(p_bot[1]))),
            center=(float(center_x), float(center_y))
        )

    except Exception as e:
        print(f"[WARN] 计算面中线几何失败: {e}")
        return None


# ==================== 主类 ====================

class FacialSymmetryAnalyzer:
    """面部对称性分析器 - 优化版本"""

    def __init__(
            self,
            db_path: str,
            model_path: str = '/Users/cuijinglei/PycharmProjects/medicalProject/models/face_landmarker.task',
            verbose: bool = True,
    ):
        """
        初始化分析器

        Args:
            db_path: 数据库路径
            model_path: MediaPipe模型路径
            verbose: 是否打印详细信息
        """
        self.db_path = db_path
        self.model_path = model_path
        self.verbose = bool(verbose)

        # 生成点对、名称和区域索引映射
        self.feature_pairs, self.feature_names, self.region_feature_indices = (
            build_pairs_and_names(SYMMETRY_INDEX_CONFIG)
        )
        self.n_features = len(self.feature_pairs)

        # 调试信息：最不对称帧
        self._debug_frame: Optional[np.ndarray] = None
        self._debug_landmarks = None
        self._debug_frame_abs_index: Optional[int] = None
        self._debug_asym_score: Optional[float] = None
        self._debug_pair_absdy: Optional[np.ndarray] = None

        if self.verbose:
            print("✅ 面部对称性分析器初始化完成")
            print(f"   - 对称点对数: {self.n_features}")
            print(f"   - 区域数: {len(self.region_feature_indices)}")
            print(f"   - 数据库: {db_path}")
            print(f"   - Landmarker模型: {model_path}")

    def _create_landmarker(self) -> vision.FaceLandmarker:
        """创建MediaPipe FaceLandmarker实例"""
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        return vision.FaceLandmarker.create_from_options(options)

    def _reset_debug_info(self):
        """重置调试信息"""
        self._debug_frame = None
        self._debug_landmarks = None
        self._debug_frame_abs_index = None
        self._debug_asym_score = None
        self._debug_pair_absdy = None

    def _update_most_asymmetric_frame(
            self,
            frame: np.ndarray,
            face_landmarks,
            frame_abs_idx: int,
            pair_absdy: np.ndarray
    ):
        """
        更新最不对称帧信息

        Args:
            frame: 当前帧图像
            face_landmarks: 当前帧的面部关键点
            frame_abs_idx: 帧的绝对索引
            pair_absdy: 每对点的|Δy|值
        """
        asym_score = float(pair_absdy.sum())
        if (self._debug_asym_score is None) or (asym_score > self._debug_asym_score):
            self._debug_asym_score = asym_score
            self._debug_frame = frame.copy()
            self._debug_landmarks = face_landmarks
            self._debug_frame_abs_index = int(frame_abs_idx)
            self._debug_pair_absdy = pair_absdy.copy()

    def extract_landmarks_from_video(
            self,
            video_path: str,
            start_frame: Optional[int] = None,
            end_frame: Optional[int] = None,
            fps: Optional[float] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        从视频中提取面部关键点坐标

        Args:
            video_path: 视频文件路径
            start_frame: 起始帧（包含）
            end_frame: 结束帧（不包含）
            fps: 视频帧率（用于计算timestamp）

        Returns:
            (left_coords, right_coords): 左右对称点坐标，形状 [T, F, 3]
            如果失败返回 (None, None)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        # 处理帧范围
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        start_frame = max(0, start_frame or 0)
        end_frame = min(total_frames, end_frame or total_frames)

        if start_frame >= end_frame:
            raise ValueError(f"无效的帧范围: start={start_frame}, end={end_frame}")

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # 创建landmarker（每个视频独立实例，避免timestamp冲突）
        landmarker = self._create_landmarker()

        coords_list = []
        self._reset_debug_info()

        processed_idx = 0
        last_timestamp = -1
        frame_abs_idx = start_frame

        try:
            while cap.isOpened() and frame_abs_idx < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                # 转换为RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp_mediapipe.Image(
                    image_format=mp_mediapipe.ImageFormat.SRGB,
                    data=rgb_frame
                )

                # 计算单调递增的timestamp
                if fps and fps > 0:
                    timestamp_ms = int(processed_idx * 1000.0 / float(fps))
                else:
                    timestamp_ms = processed_idx * 33  # 默认30fps

                if timestamp_ms <= last_timestamp:
                    timestamp_ms = last_timestamp + 1
                last_timestamp = timestamp_ms
                processed_idx += 1

                # 检测面部关键点
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.face_landmarks:
                    face_landmarks = result.face_landmarks[0]

                    # 组装坐标并计算不对称性
                    coords = []
                    pair_absdy = np.empty((self.n_features,), dtype=np.float32)

                    for j, (left_idx, right_idx) in enumerate(self.feature_pairs):
                        left_lm = face_landmarks[left_idx]
                        right_lm = face_landmarks[right_idx]

                        coords.append([
                            [left_lm.x, left_lm.y, left_lm.z],
                            [right_lm.x, right_lm.y, right_lm.z]
                        ])
                        pair_absdy[j] = abs(float(left_lm.y) - float(right_lm.y))

                    coords_list.append(coords)

                    # 更新最不对称帧
                    self._update_most_asymmetric_frame(
                        frame, face_landmarks, frame_abs_idx, pair_absdy
                    )

                frame_abs_idx += 1

        finally:
            cap.release()
            try:
                landmarker.close()
            except Exception:
                pass

        if not coords_list:
            return None, None

        # 转换为numpy数组
        coords_array = np.array(coords_list, dtype=np.float32)  # [T, F, 2, 3]
        left_coords = coords_array[:, :, 0, :]  # [T, F, 3]
        right_coords = coords_array[:, :, 1, :]  # [T, F, 3]

        return left_coords, right_coords

    def draw_landmark_pairs(
            self,
            image: np.ndarray,
            face_landmarks,
            image_width: int,
            image_height: int
    ) -> np.ndarray:
        """
        在图像上绘制对称点对和连线

        Args:
            image: 输入图像
            face_landmarks: MediaPipe面部关键点
            image_width: 图像宽度
            image_height: 图像高度

        Returns:
            绘制后的图像
        """
        vis = image.copy()

        for (left_idx, right_idx) in self.feature_pairs:
            left_lm = face_landmarks[left_idx]
            right_lm = face_landmarks[right_idx]

            # 转换为像素坐标
            lx = int(left_lm.x * image_width)
            ly = int(left_lm.y * image_height)
            rx = int(right_lm.x * image_width)
            ry = int(right_lm.y * image_height)

            # 绘制连接线
            cv2.line(
                vis, (lx, ly), (rx, ry),
                VisConfig.COLOR_CONNECTION,
                VisConfig.CONNECTION_THICKNESS
            )

            # 绘制左侧点（红色）和索引
            cv2.circle(
                vis, (lx, ly),
                VisConfig.POINT_RADIUS,
                VisConfig.COLOR_LEFT_POINT,
                -1
            )
            cv2.putText(
                vis, str(left_idx), (lx + 2, ly - 2),
                VisConfig.TEXT_FONT,
                VisConfig.TEXT_SIZE,
                VisConfig.COLOR_LEFT_POINT,
                VisConfig.TEXT_THICKNESS
            )

            # 绘制右侧点（蓝色）和索引
            cv2.circle(
                vis, (rx, ry),
                VisConfig.POINT_RADIUS,
                VisConfig.COLOR_RIGHT_POINT,
                -1
            )
            cv2.putText(
                vis, str(right_idx), (rx + 2, ry - 2),
                VisConfig.TEXT_FONT,
                VisConfig.TEXT_SIZE,
                VisConfig.COLOR_RIGHT_POINT,
                VisConfig.TEXT_THICKNESS
            )

        return vis

    def draw_midline(
            self,
            image: np.ndarray,
            midline: MidlinePoints
    ) -> np.ndarray:
        """
        在图像上绘制面中线

        Args:
            image: 输入图像
            midline: 面中线关键点

        Returns:
            绘制后的图像
        """
        vis = image.copy()

        # 绘制面中线
        cv2.line(
            vis,
            midline.top,
            midline.bottom,
            VisConfig.COLOR_MIDLINE,
            VisConfig.MIDLINE_THICKNESS
        )

        # 绘制端点标记
        cv2.circle(vis, midline.top, 4, VisConfig.COLOR_MIDLINE, -1)
        cv2.circle(vis, midline.bottom, 4, VisConfig.COLOR_MIDLINE, -1)

        return vis

    def save_index_check_image(
            self,
            frame_bgr: np.ndarray,
            face_landmarks,
            output_path: str
    ):
        """
        保存索引检查图（包含点对连线、索引标注和面中线）

        Args:
            frame_bgr: BGR格式的帧图像
            face_landmarks: MediaPipe面部关键点
            output_path: 输出路径
        """
        if frame_bgr is None or face_landmarks is None:
            if self.verbose:
                print("  ⚠️ 跳过index_check图像：无有效帧或关键点")
            return

        h, w = frame_bgr.shape[:2]

        # 1. 绘制所有点对和连线
        vis = self.draw_landmark_pairs(frame_bgr, face_landmarks, w, h)

        # 2. 计算并绘制面中线
        face_contour_indices = self.region_feature_indices.get("face_contour", [])
        if face_contour_indices:
            midline = compute_midline_geometry(
                face_landmarks,
                face_contour_indices,
                self.feature_pairs,
                w, h
            )

            if midline:
                vis = self.draw_midline(vis, midline)
            else:
                if self.verbose:
                    print("  ⚠️ 面中线计算失败，跳过绘制")
        else:
            if self.verbose:
                print("  ⚠️ 未找到face_contour区域，跳过面中线绘制")

        # 3. 保存图像
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(str(output_path), vis)

        if self.verbose:
            print(f"  ✓ Index check图像已保存: {output_path}")

    def calculate_pearson_coefficients(
            self,
            left_coords: np.ndarray,
            right_coords: np.ndarray,
            use_y_only: bool = True
    ) -> SymmetryFeatures:
        """
        计算左右对称点y坐标的Pearson相关系数

        Args:
            left_coords: 左侧坐标 [T, F, 3]
            right_coords: 右侧坐标 [T, F, 3]
            use_y_only: 是否只使用y坐标

        Returns:
            SymmetryFeatures对象
        """
        if left_coords is None or right_coords is None:
            raise ValueError("未检测到人脸关键点")

        n_frames, n_features, _ = left_coords.shape

        # 提取y坐标（索引1）
        y_left = left_coords[:, :, 1]  # [T, F]
        y_right = right_coords[:, :, 1]  # [T, F]

        # 计算每对点的Pearson相关系数
        pearson_coeffs = np.zeros((n_features,), dtype=np.float32)

        for i in range(n_features):
            a = y_left[:, i]
            b = y_right[:, i]

            # 处理常数序列
            if np.std(a) < 1e-8 or np.std(b) < 1e-8:
                pearson_coeffs[i] = np.nan
            else:
                try:
                    corr, _ = stats.pearsonr(a, b)
                    pearson_coeffs[i] = float(corr)
                except Exception:
                    pearson_coeffs[i] = np.nan

        return SymmetryFeatures(
            pearson_coefficients=pearson_coeffs,
            landmark_names=list(self.feature_names),
            y_coords_left=y_left,
            y_coords_right=y_right,
            frame_count=n_frames
        )

    @staticmethod
    def _rolling_corr_cumsum(
            y_left: np.ndarray,
            y_right: np.ndarray,
            window: int
    ) -> np.ndarray:
        """
        向量化计算滑动窗口Pearson相关系数

        Args:
            y_left: 左侧y坐标 [T, F]
            y_right: 右侧y坐标 [T, F]
            window: 滑动窗口大小

        Returns:
            rolling_corr: [T, F] 滑动窗口相关系数，前window-1行为NaN
        """
        yL = y_left.astype(np.float64, copy=False)
        yR = y_right.astype(np.float64, copy=False)
        T, F = yL.shape

        window = int(max(3, window))
        if T < window:
            return np.full((T, F), np.nan, dtype=np.float32)

        def cumsum_pad(x):
            """在数组前添加零行后计算累积和"""
            return np.vstack([np.zeros((1, F), dtype=np.float64), np.cumsum(x, axis=0)])

        # 计算累积和
        cL = cumsum_pad(yL)
        cR = cumsum_pad(yR)
        cLL = cumsum_pad(yL * yL)
        cRR = cumsum_pad(yR * yR)
        cLR = cumsum_pad(yL * yR)

        # 滑动窗口统计量
        sumL = cL[window:] - cL[:-window]
        sumR = cR[window:] - cR[:-window]
        sumLL = cLL[window:] - cLL[:-window]
        sumRR = cRR[window:] - cRR[:-window]
        sumLR = cLR[window:] - cLR[:-window]

        # 均值
        meanL = sumL / window
        meanR = sumR / window

        # 协方差和方差
        cov = (sumLR / window) - (meanL * meanR)
        varL = (sumLL / window) - (meanL * meanL)
        varR = (sumRR / window) - (meanR * meanR)

        # Pearson相关系数
        denom = np.sqrt(np.maximum(varL, 0.0) * np.maximum(varR, 0.0))
        corr = np.divide(
            cov, denom,
            out=np.full_like(cov, np.nan),
            where=(denom > 1e-12)
        )

        # 填充结果
        out = np.full((T, F), np.nan, dtype=np.float32)
        out[window - 1:] = corr.astype(np.float32)

        return out

    def compute_region_timeseries(
            self,
            features: SymmetryFeatures,
            rolling_window: int = 15
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        计算区域级别的时间序列数据

        Args:
            features: 对称性特征
            rolling_window: 滑动窗口大小

        Returns:
            包含abs_diff和rolling_corr的字典
        """
        y_left = features.y_coords_left  # [T, F]
        y_right = features.y_coords_right
        T, F = y_left.shape

        # 计算绝对差值和滑动相关系数
        abs_diff = np.abs(y_left - y_right).astype(np.float32)  # [T, F]
        rolling_corr = self._rolling_corr_cumsum(y_left, y_right, rolling_window)  # [T, F]

        # 按区域聚合
        region_abs = {}
        region_corr = {}

        for region, idxs in self.region_feature_indices.items():
            idxs = list(idxs)
            if idxs:
                region_abs[region] = abs_diff[:, idxs].mean(axis=1)
                region_corr[region] = np.nanmean(rolling_corr[:, idxs], axis=1)
            else:
                region_abs[region] = np.full((T,), np.nan, np.float32)
                region_corr[region] = np.full((T,), np.nan, np.float32)

        return {
            "abs_diff": region_abs,
            "rolling_corr": region_corr,
            "abs_diff_raw": abs_diff,
            "rolling_corr_raw": rolling_corr
        }

    def save_region_timeseries_plot(
            self,
            ts: Dict[str, Dict[str, np.ndarray]],
            fps: float,
            title: str,
            save_path_base: str
    ) -> Tuple[str, str]:
        """
        保存区域时间序列图

        Args:
            ts: compute_region_timeseries的返回值
            fps: 视频帧率
            title: 图表标题
            save_path_base: 保存路径基础名

        Returns:
            (abs_diff图路径, rolling_corr图路径)
        """
        region_abs = ts["abs_diff"]
        region_corr = ts["rolling_corr"]

        # 获取时间轴
        any_region = next(iter(region_abs.keys()))
        T = len(region_abs[any_region])

        if fps and fps > 0:
            t = np.arange(T) / float(fps)
            xlabel = "Time (s)"
        else:
            t = np.arange(T)
            xlabel = "Frame"

        os.makedirs(os.path.dirname(save_path_base), exist_ok=True)

        # 1. 绝对差值图
        fig = plt.figure(figsize=(14, 6))
        ax = plt.gca()

        for region, y in region_abs.items():
            ax.plot(t, y, linewidth=1.6, label=region)

        ax.set_title(f"{title}\nRegion Asymmetry (abs(yL-yR), lower=better)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("abs(yL - yR)")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2, fontsize=9)

        out1 = save_path_base.replace(".png", "_region_absdiff.png")
        fig.savefig(out1, dpi=160, bbox_inches="tight")
        plt.close(fig)

        # 2. 滑动相关系数图
        fig = plt.figure(figsize=(14, 6))
        ax = plt.gca()

        for region, y in region_corr.items():
            ax.plot(t, y, linewidth=1.6, label=region)

        ax.set_title(f"{title}\nRegion Rolling Pearson (higher=better)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Rolling Pearson r")
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2, fontsize=9)

        out2 = save_path_base.replace(".png", "_region_rollingcorr.png")
        fig.savefig(out2, dpi=160, bbox_inches="tight")
        plt.close(fig)

        if self.verbose:
            print(f"  ✓ Region time-series saved: {out1}")
            print(f"  ✓ Region time-series saved: {out2}")

        return out1, out2

    @staticmethod
    def _fast_splat_heatmap(
            w: int,
            h: int,
            pts_xy: np.ndarray,
            vals: np.ndarray,
            canvas_res: int = 256,
            sigma: float = 8.0
    ) -> np.ndarray:
        """
        快速生成热力图（比griddata快很多）

        Args:
            w: 目标图像宽度
            h: 目标图像高度
            pts_xy: 点坐标 [N, 2]，像素坐标系
            vals: 点的值 [N]
            canvas_res: 画布分辨率
            sigma: 高斯模糊标准差

        Returns:
            heat_bgr: BGR格式的热力图，大小(h, w)
        """
        canvas_res = int(max(64, canvas_res))
        sx = canvas_res / float(w)
        sy = canvas_res / float(h)

        acc = np.zeros((canvas_res, canvas_res), dtype=np.float32)
        cnt = np.zeros((canvas_res, canvas_res), dtype=np.float32)

        # 将点"撒"到画布上
        for (x, y), v in zip(pts_xy, vals):
            cx = int(np.clip(x * sx, 0, canvas_res - 1))
            cy = int(np.clip(y * sy, 0, canvas_res - 1))
            cv2.circle(acc, (cx, cy), 2, float(v), -1)
            cv2.circle(cnt, (cx, cy), 2, 1.0, -1)

        # 归一化并模糊
        heat = acc / (cnt + 1e-6)
        heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=sigma, sigmaY=sigma)

        # 归一化到[0, 255]
        vmin = float(np.min(heat))
        vmax = float(np.max(heat))
        heat = (heat - vmin) / (vmax - vmin + 1e-6)
        heat_u8 = np.clip(heat * 255.0, 0, 255).astype(np.uint8)

        # 应用colormap并调整大小
        heat_bgr_small = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        heat_bgr = cv2.resize(heat_bgr_small, (w, h), interpolation=cv2.INTER_CUBIC)

        return heat_bgr

    def save_overlay_asymmetry_heatmap(
            self,
            frame_bgr: np.ndarray,
            face_landmarks,
            pair_values: np.ndarray,
            save_path: str,
            alpha: float = None,
            canvas_res: int = None
    ):
        """
        保存叠加不对称热力图

        Args:
            frame_bgr: BGR格式的帧图像
            face_landmarks: MediaPipe面部关键点
            pair_values: 每对点的分数（通常为|Δy|）
            save_path: 保存路径
            alpha: 热力图透明度
            canvas_res: 画布分辨率
        """
        if frame_bgr is None or face_landmarks is None:
            if self.verbose:
                print("  ⚠️ 跳过overlay热力图：无有效帧或关键点")
            return

        alpha = alpha or VisConfig.HEATMAP_ALPHA
        canvas_res = canvas_res or VisConfig.HEATMAP_CANVAS_RES

        h, w = frame_bgr.shape[:2]

        # 收集点坐标和值
        pts = []
        vals = []

        for (li, ri), s in zip(self.feature_pairs, pair_values):
            for idx in (li, ri):
                lm = face_landmarks[idx]
                pts.append([lm.x * w, lm.y * h])
                vals.append(float(s))

        pts = np.asarray(pts, dtype=np.float32)
        vals = np.asarray(vals, dtype=np.float32)

        # 生成热力图并叠加
        heat_bgr = self._fast_splat_heatmap(
            w=w, h=h, pts_xy=pts, vals=vals,
            canvas_res=canvas_res,
            sigma=VisConfig.HEATMAP_SIGMA
        )
        overlay = cv2.addWeighted(frame_bgr, 1 - alpha, heat_bgr, alpha, 0)

        # 保存
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, overlay)

        if self.verbose:
            print(f"  ✓ Overlay heatmap saved: {save_path}")

    def visualize_symmetry_heatmap(
            self,
            features: SymmetryFeatures,
            title: str,
            save_path: str,
    ):
        """
        可视化对称性热力图（Pearson系数柱状图 + 空间分布 + 统计摘要）

        Args:
            features: 对称性特征
            title: 图表标题
            save_path: 保存路径
        """
        fig = plt.figure(figsize=(16, 8))

        # 1) Pearson系数柱状图
        ax1 = plt.subplot(1, 3, 1)
        coeffs = features.pearson_coefficients

        # 颜色映射
        colors = []
        for coef in coeffs:
            if np.isnan(coef):
                colors.append('gray')
            elif coef > PearsonThreshold.HIGH:
                colors.append('darkblue')
            elif coef > PearsonThreshold.MEDIUM:
                colors.append('lightblue')
            elif coef > PearsonThreshold.LOW:
                colors.append('yellow')
            else:
                colors.append('red')

        bars = ax1.barh(range(self.n_features), coeffs, color=colors)
        ax1.set_yticks(range(self.n_features))
        ax1.set_yticklabels(features.landmark_names, fontsize=9)
        ax1.set_xlabel('Pearson Correlation Coefficient', fontsize=11)
        ax1.set_title('对称性系数\n(蓝色=高对称, 红色=不对称)', fontsize=10)
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        ax1.axvline(x=PearsonThreshold.HIGH, color='green', linestyle='--', linewidth=0.8, alpha=0.5)
        ax1.set_xlim(-1, 1)
        ax1.grid(axis='x', alpha=0.3)

        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, coeffs)):
            if np.isnan(val):
                label = "nan"
                xpos = 0.05
                ha = "left"
            else:
                label = f"{val:.3f}"
                xpos = (val + 0.05) if val > 0 else (val - 0.05)
                ha = "left" if val > 0 else "right"
            ax1.text(xpos, i, label, va='center', ha=ha, fontsize=8)

        # 2) 空间分布图
        ax2 = plt.subplot(1, 3, 2)

        # 按平均y坐标排序
        avg_y = (features.y_coords_left.mean(axis=0) + features.y_coords_right.mean(axis=0)) / 2
        y_norm = (avg_y - avg_y.min()) / (avg_y.max() - avg_y.min() + 1e-6)

        x = np.linspace(-1, 1, self.n_features)
        y = y_norm

        scatter = ax2.scatter(
            x, y,
            c=np.nan_to_num(coeffs, nan=0.0),
            s=500,
            cmap='RdYlBu',
            vmin=-1, vmax=1,
            edgecolors='black',
            linewidth=1.5
        )

        # 添加索引标签
        for i, (xi, yi) in enumerate(zip(x, y)):
            ax2.annotate(
                f'{i + 1}', (xi, yi),
                ha='center', va='center',
                fontsize=9, fontweight='bold'
            )

        ax2.set_xlim(-1.2, 1.2)
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_xlabel('左 ← | → 右', fontsize=11)
        ax2.set_ylabel('上 ← | → 下', fontsize=11)
        ax2.set_title('面部对称性空间分布', fontsize=11)
        ax2.set_aspect('equal')

        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Pearson Coefficient', fontsize=10)

        # 3) 统计摘要
        ax3 = plt.subplot(1, 3, 3)
        ax3.axis('off')

        # 计算统计量
        valid = coeffs[np.isfinite(coeffs)]
        mean_corr = float(np.mean(valid)) if valid.size else float("nan")
        std_corr = float(np.std(valid)) if valid.size else float("nan")
        min_corr = float(np.min(valid)) if valid.size else float("nan")
        max_corr = float(np.max(valid)) if valid.size else float("nan")

        def mean_by_region(region: str) -> float:
            """计算区域平均Pearson系数"""
            idxs = self.region_feature_indices.get(region, [])
            vals = coeffs[idxs] if idxs else np.array([], dtype=np.float32)
            vals = vals[np.isfinite(vals)]
            return float(np.mean(vals)) if vals.size else float("nan")

        # 按区域统计
        eyebrow_corr = mean_by_region("eyebrow")
        eye_corr = mean_by_region("eye")
        upper_lip_corr = mean_by_region("upper_lip")
        lower_lip_corr = mean_by_region("lower_lip")
        nose_corr = mean_by_region("nose")

        # 按阈值分类
        high_sym = int(np.sum((coeffs > PearsonThreshold.HIGH) & np.isfinite(coeffs)))
        medium_sym = int(np.sum(
            (coeffs > PearsonThreshold.MEDIUM) &
            (coeffs <= PearsonThreshold.HIGH) &
            np.isfinite(coeffs)
        ))
        low_sym = int(np.sum((coeffs <= PearsonThreshold.MEDIUM) & np.isfinite(coeffs)))

        # 生成摘要文本
        summary_text = f"""
Statistical Summary
{'=' * 30}

Overall Symmetry:
  Mean Correlation: {mean_corr:.3f} ± {std_corr:.3f}
  Min/Max: {min_corr:.3f} / {max_corr:.3f}
  Total Features: {self.n_features}

Region-wise Mean:
  Eyebrow: {eyebrow_corr:.3f}
  Eye: {eye_corr:.3f}
  Upper Lip: {upper_lip_corr:.3f}
  Lower Lip: {lower_lip_corr:.3f}
  Nose: {nose_corr:.3f}

Symmetry Distribution:
  High (r > 0.8): {high_sym}
  Medium (0.5 < r ≤ 0.8): {medium_sym}
  Low (r ≤ 0.5): {low_sym}

Frames Analyzed: {features.frame_count}
"""

        ax3.text(
            0.1, 0.95, summary_text,
            transform=ax3.transAxes,
            fontsize=11,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        )

        plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()

        # 保存
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        if self.verbose:
            print(f"  ✓ Symmetry heatmap saved: {save_path}")

    def analyze_single_video(
            self,
            video_path: str,
            action_name: str,
            output_dir: str,
            start_frame: Optional[int] = None,
            end_frame: Optional[int] = None,
            fps: Optional[float] = None,
            patient_id: Optional[str] = None,
            examination_id: Optional[str] = None,
            video_id: Optional[int] = None,
    ) -> Dict:
        """
        分析单个视频（完整pipeline）

        Args:
            video_path: 视频文件路径
            action_name: 动作名称
            output_dir: 输出目录（所有结果统一放在这个目录下，不再为每个视频创建子文件夹）
            start_frame: 起始帧
            end_frame: 结束帧
            fps: 帧率
            patient_id: 患者ID
            examination_id: 检查ID
            video_id: 视频ID

        Returns:
            结果字典
        """
        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"分析视频: {patient_id or 'Unknown'} - {action_name}")
            print(f"{'=' * 60}")

        # 确保输出目录存在（不再创建子文件夹）
        os.makedirs(output_dir, exist_ok=True)

        # 用“患者_检查_动作”作为文件名前缀，避免同一目录下冲突
        safe_patient = _safe_name(patient_id)
        safe_exam = _safe_name(examination_id)
        safe_action = _safe_name(action_name)
        base_filename = f"{safe_patient}_{safe_exam}_{safe_action}" if (safe_patient or safe_exam) else safe_action

        # 1. 提取关键点
        if self.verbose:
            print("  → 提取面部关键点...")

        left_coords, right_coords = self.extract_landmarks_from_video(
            video_path,
            start_frame=start_frame,
            end_frame=end_frame,
            fps=fps
        )

        if left_coords is None or right_coords is None:
            error_msg = "未检测到人脸关键点"
            if self.verbose:
                print(f"  ❌ {error_msg}")
            return {
                "video_path": video_path,
                "action_name": action_name,
                "patient_id": patient_id,
                "examination_id": examination_id,
                "video_id": video_id,
                "status": "failed",
                "error": error_msg
            }

        # 2. 计算Pearson系数
        if self.verbose:
            print("  → 计算Pearson相关系数...")

        features = self.calculate_pearson_coefficients(left_coords, right_coords)

        # 3. 计算区域时间序列
        if self.verbose:
            print("  → 计算区域时间序列...")

        ts = self.compute_region_timeseries(features, rolling_window=15)

        # -------- 各种可视化输出，全部直接写在 output_dir 下 --------

        # 1) Index check 图像（最不对称帧）
        if self._debug_frame is not None and self._debug_landmarks is not None:
            if self.verbose:
                print(
                    f"  → 保存 index_check 图像（最不对称帧: frame#{self._debug_frame_abs_index}, "
                    f"score={self._debug_asym_score:.4f}）..."
                )
            index_check_path = os.path.join(output_dir, f"{base_filename}_index_check.png")
            self.save_index_check_image(
                self._debug_frame,
                self._debug_landmarks,
                index_check_path
            )

        # 2) Overlay 热力图（用 |Δy| 做不对称程度）
        if self._debug_frame is not None and self._debug_landmarks is not None and self._debug_pair_absdy is not None:
            if self.verbose:
                print("  → 保存 overlay 热力图...")
            overlay_path = os.path.join(output_dir, f"{base_filename}_overlay.png")
            self.save_overlay_asymmetry_heatmap(
                self._debug_frame,
                self._debug_landmarks,
                self._debug_pair_absdy,
                overlay_path
            )

        # 3) 对称性热力图（Pearson 柱状图 + 空间分布 + 统计摘要）
        if self.verbose:
            print("  → 保存 symmetry 热力图...")
        heatmap_path = os.path.join(output_dir, f"{base_filename}_symmetry_heatmap.png")
        self.visualize_symmetry_heatmap(
            features,
            title=f"{patient_id} - {action_name}",
            save_path=heatmap_path
        )

        # 4) 区域时间序列图（abs_diff / rolling_corr）
        if self.verbose:
            print("  → 保存 region 时间序列图...")
        ts_base = os.path.join(output_dir, f"{base_filename}_timeseries.png")
        self.save_region_timeseries_plot(
            ts,
            fps=fps or 30.0,
            title=f"{patient_id} - {action_name}",
            save_path_base=ts_base
        )

        # 5) 保存统计结果 JSON
        if self.verbose:
            print("  → 保存统计结果 JSON...")
        stats_data = {
            "video_info": {
                "video_path": video_path,
                "video_id": video_id,
                "patient_id": patient_id,
                "examination_id": examination_id,
                "action_name": action_name,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "fps": fps,
            },
            "analysis_results": {
                "frame_count": features.frame_count,
                "n_features": self.n_features,
                "pearson_coefficients": features.pearson_coefficients.tolist(),
                "landmark_names": features.landmark_names,
                "overall_mean_pearson": float(np.nanmean(features.pearson_coefficients)),
                "overall_std_pearson": float(np.nanstd(features.pearson_coefficients)),
            },
            "most_asymmetric_frame": {
                "frame_index": self._debug_frame_abs_index,
                "asymmetry_score": self._debug_asym_score,
            } if self._debug_frame_abs_index is not None else None,
        }

        stats_path = os.path.join(output_dir, f"{base_filename}_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats_data, f, ensure_ascii=False, indent=2)

        if self.verbose:
            print(f"  ✓ 统计结果已保存: {stats_path}")
            print(f"\n✅ 视频分析完成: {base_filename}（输出目录: {output_dir}）")

        return {
            "video_path": video_path,
            "video_id": video_id,
            "patient_id": patient_id,
            "examination_id": examination_id,
            "action_name": action_name,
            "output_folder": output_dir,  # 这里也改成输出主目录
            "status": "success",
            "mean_pearson": float(np.nanmean(features.pearson_coefficients)),
        }

    def batch_process_database(
            self,
            output_dir: str,
            limit: Optional[int] = None,
            action_filter: Optional[List[str]] = None,
            use_multiprocessing: bool = True,
            num_workers: Optional[int] = None,
    ) -> List[Dict]:
        """
        批量处理数据库中的视频

        Args:
            output_dir: 输出目录
            limit: 限制处理数量
            action_filter: 动作名称过滤列表
            use_multiprocessing: 是否使用多进程
            num_workers: 进程数（None则自动选择）

        Returns:
            结果列表
        """
        # 从数据库查询视频
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
            SELECT
                v.video_id,
                v.examination_id,
                v.action_id,
                v.file_path,
                v.start_frame,
                v.end_frame,
                v.fps,
                at.action_name_en,
                e.patient_id
            FROM video_files v
            LEFT JOIN examinations e ON v.examination_id = e.examination_id
            LEFT JOIN action_types at ON v.action_id = at.action_id
            WHERE v.file_exists = 1
        """

        if action_filter:
            placeholders = ",".join(["?"] * len(action_filter))
            query += f" AND at.action_name_en IN ({placeholders})"
            cursor.execute(query, action_filter)
        else:
            cursor.execute(query)

        videos = cursor.fetchall()
        conn.close()

        if limit:
            videos = videos[:int(limit)]

        print(f"\n{'=' * 60}")
        print("批量处理模式")
        print(f"总视频数: {len(videos)}")
        print(f"输出目录: {output_dir}")
        print(f"多进程: {use_multiprocessing}")
        print(f"{'=' * 60}\n")

        # 准备任务列表
        tasks = []
        for (video_id, exam_id, action_id, file_path,
             start_frame, end_frame, fps, action_name, patient_id) in videos:
            if not file_path or not os.path.exists(file_path):
                continue

            tasks.append({
                "db_path": self.db_path,
                "model_path": self.model_path,
                "video_id": int(video_id),
                "video_path": file_path,
                "action_name": action_name,
                "output_dir": output_dir,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "fps": fps,
                "patient_id": patient_id,
                "examination_id": exam_id,
            })

        if not tasks:
            print("⚠️ 没有可处理的视频任务")
            return []

        # 自动确定进程数
        if num_workers is None:
            cpu = os.cpu_count() or 8
            num_workers = min(8, cpu)

        results = []
        errors = []

        # 多进程处理
        if use_multiprocessing and num_workers > 1:
            try:
                import multiprocessing as mp
                if mp.get_start_method(allow_none=True) is None:
                    mp.set_start_method("spawn", force=True)
            except RuntimeError:
                pass

            print(f"🚀 使用多进程: num_workers={num_workers}, tasks={len(tasks)}")

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futs = [executor.submit(_worker_analyze_one, task) for task in tasks]
                total = len(futs)
                done = 0

                for fut in as_completed(futs):
                    done += 1
                    try:
                        ok, payload = fut.result()
                        if ok:
                            results.append(payload)
                        else:
                            errors.append(payload)
                    except Exception as e:
                        errors.append({"error": str(e)})

                    if done % 10 == 0 or done == total:
                        print(f"进度: {done}/{total} | 成功: {len(results)} | 失败: {len(errors)}")

        else:
            # 单进程处理
            print(f"🧵 单进程顺序处理: tasks={len(tasks)}")
            old_verbose = self.verbose
            self.verbose = True

            for i, task in enumerate(tasks, 1):
                print(f"[{i}/{len(tasks)}] {task['patient_id']} - {task['action_name']}")
                try:
                    r = self.analyze_single_video(
                        video_path=task["video_path"],
                        action_name=task["action_name"],
                        output_dir=task["output_dir"],
                        start_frame=task["start_frame"],
                        end_frame=task["end_frame"],
                        fps=task["fps"],
                        patient_id=task["patient_id"],
                        examination_id=task["examination_id"],
                        video_id=task["video_id"],
                    )
                    results.append(r)
                except Exception as e:
                    errors.append({
                        "video_id": task["video_id"],
                        "video_path": task["video_path"],
                        "error": str(e)
                    })
                    print(f"  ❌ 错误: {e}")

            self.verbose = old_verbose

        # 保存汇总结果
        summary = {
            "success": results,
            "errors": errors,
            "total_tasks": len(tasks),
            "success_count": len(results),
            "error_count": len(errors),
        }

        summary_path = os.path.join(output_dir, "z_batch_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n{'=' * 60}")
        print("✅ 批量处理完成!")
        print(f"   成功处理: {len(results)}/{len(tasks)}")
        print(f"   失败: {len(errors)}")
        print(f"   汇总结果: {summary_path}")
        print(f"{'=' * 60}\n")

        return results

    def compare_actions(
            self,
            video_paths: Dict[str, str],
            patient_id: str,
            output_path: str
    ):
        """
        比较同一患者不同动作的对称性

        Args:
            video_paths: {action_name: video_path} 字典
            patient_id: 患者ID
            output_path: 输出路径
        """
        print(f"\n{'=' * 60}")
        print(f"动作对比分析: {patient_id}")
        print(f"{'=' * 60}")

        features_dict = {}
        for action_name, video_path in video_paths.items():
            print(f"\n处理动作: {action_name}")
            left_coords, right_coords = self.extract_landmarks_from_video(video_path)

            if left_coords is not None and right_coords is not None:
                features = self.calculate_pearson_coefficients(left_coords, right_coords)
                features_dict[action_name] = features
            else:
                print(f"  ⚠️ 跳过动作 {action_name}：未检测到人脸")

        if not features_dict:
            print("  ❌ 没有成功处理的动作")
            return

        # 绘制对比图
        n_actions = len(features_dict)
        fig, axes = plt.subplots(1, n_actions, figsize=(6 * n_actions, 6))
        if n_actions == 1:
            axes = [axes]

        for idx, (action_name, features) in enumerate(features_dict.items()):
            ax = axes[idx]
            coeffs = features.pearson_coefficients

            # 颜色映射
            colors = []
            for coef in coeffs:
                if np.isnan(coef):
                    colors.append('gray')
                elif coef > PearsonThreshold.HIGH:
                    colors.append('darkblue')
                elif coef > PearsonThreshold.MEDIUM:
                    colors.append('lightblue')
                elif coef > PearsonThreshold.LOW:
                    colors.append('yellow')
                else:
                    colors.append('red')

            ax.barh(range(self.n_features), coeffs, color=colors)
            ax.set_yticks(range(self.n_features))
            ax.set_yticklabels(features.landmark_names, fontsize=8)
            ax.set_xlabel('Pearson Coefficient', fontsize=10)
            ax.set_title(
                f'{action_name}\n平均: {np.nanmean(coeffs):.3f}',
                fontsize=11, fontweight='bold'
            )
            ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
            ax.axvline(x=PearsonThreshold.HIGH, color='green', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.set_xlim(-1, 1)
            ax.grid(axis='x', alpha=0.3)

        plt.suptitle(
            f'患者 {patient_id} - 不同动作对称性对比',
            fontsize=14, fontweight='bold', y=0.98
        )
        plt.tight_layout()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"\n✓ 对比图已保存: {output_path}")


# ==================== Multiprocessing Worker ====================

def _worker_analyze_one(task: Dict):
    """
    子进程worker函数

    Args:
        task: 任务字典

    Returns:
        (成功标志, 结果数据)
    """
    try:
        analyzer = FacialSymmetryAnalyzer(
            db_path=task["db_path"],
            model_path=task["model_path"],
            verbose=False  # 子进程不打印详细信息
        )

        res = analyzer.analyze_single_video(
            video_path=task["video_path"],
            action_name=task["action_name"],
            output_dir=task["output_dir"],
            start_frame=task["start_frame"],
            end_frame=task["end_frame"],
            fps=task["fps"],
            patient_id=task["patient_id"],
            examination_id=task["examination_id"],
            video_id=task.get("video_id"),
        )
        return True, res

    except Exception as e:
        return False, {
            "video_id": task.get("video_id"),
            "video_path": task.get("video_path"),
            "patient_id": task.get("patient_id"),
            "examination_id": task.get("examination_id"),
            "action_name": task.get("action_name"),
            "error": str(e),
        }


# ==================== Main ====================

def main():
    """主函数示例"""
    analyzer = FacialSymmetryAnalyzer(
        db_path='/facial_palsy/facialPalsy.db'
    )

    analyzer.batch_process_database(
        output_dir='/Users/cuijinglei/Documents/facial_palsy/HGFA/symmetry_analysis',
        limit=None,
        action_filter=None,
        use_multiprocessing=True,
        num_workers=8,
    )


if __name__ == '__main__':
    main()