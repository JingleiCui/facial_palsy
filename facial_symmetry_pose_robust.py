# -*- coding: utf-8 -*-
"""
facial_symmetry_pose_robust.py

面部对称性分析 - 3D姿态不敏感版本（优化版）

主要改进：
1. ✅ 多进程并行处理，充分利用多CPU
2. ✅ 简化文件命名：examination_id_action_name_xxx.png
3. ✅ 扁平化目录结构：所有文件直接保存在output_dir下
5. ✅ 使用MediaPipe 3D坐标构建姿态不敏感的对称性指标
"""

import os
import re
import json
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union

import cv2
import numpy as np
import mediapipe as mp_mediapipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 中文字体/负号
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'PingFang SC']
matplotlib.rcParams['axes.unicode_minus'] = False

# ==================== 常量定义 ====================

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

# 面部关键点索引
EYE_INNER_CANTHUS_LEFT = 362
EYE_INNER_CANTHUS_RIGHT = 133
NOSE_TIP = 4
FOREHEAD_CENTER = 10
CHIN = 152

# ==================== [新增] 标准3D人脸模型用于PnP解算 ====================
# 选取6个关键点：鼻尖, 下巴, 左眼外角, 右眼外角, 左嘴角, 右嘴角
# MediaPipe索引: Nose=4, Chin=152, LeftEye=33(用户右眼), RightEye=263(用户左眼), MouthL=61, MouthR=291
# 注意：这里使用的是基于通用人脸比例的3D坐标
GENERIC_FACE_3D = np.array([
    (0.0, 0.0, 0.0),  # Nose Tip (Index 4)
    (0.0, -330.0, -65.0),  # Chin (Index 152)
    (-225.0, 170.0, -135.0),  # Left Eye Left Corner (Index 33 - MP Right Eye)
    (225.0, 170.0, -135.0),  # Right Eye Right Corner (Index 263 - MP Left Eye)
    (-150.0, -150.0, -125.0),  # Left Mouth Corner (Index 61)
    (150.0, -150.0, -125.0)  # Right Mouth Corner (Index 291)
], dtype=np.float64)

# MediaPipe中对应的关键点索引 (注意：MediaPipe左右与人脸左右是镜像的，需仔细对应)
# 图像坐标系：x向右，y向下。
# 关键点选取：4(鼻), 152(下巴), 33(画面左侧眼角), 263(画面右侧眼角), 61(画面左嘴角), 291(画面右嘴角)
PNP_INDICES = [4, 152, 33, 263, 61, 291]

TH_LOW = 0.10
TH_HIGH = 0.50

class VisConfig:
    """可视化相关配置"""
    COLOR_LEFT_POINT = (0, 0, 255)  # 红色 (BGR)
    COLOR_RIGHT_POINT = (255, 0, 0)  # 蓝色 (BGR)
    COLOR_MIDLINE = (0, 255, 255)  # 黄色 (BGR)
    COLOR_CONNECTION = (0, 255, 0)  # 绿色 (BGR)
    POINT_RADIUS = 2
    MIDLINE_THICKNESS = 2
    CONNECTION_THICKNESS = 1
    TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
    TEXT_SIZE = 1
    TEXT_THICKNESS = 1


# ==================== 数据结构 ====================

@dataclass
class HeadPose:
    """头部姿态"""
    roll: float
    pitch: float
    yaw: float
    rotation_matrix: Optional[np.ndarray] = None
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None

    def __str__(self):
        return f"Roll:{self.roll:.1f}° Pitch:{self.pitch:.1f}° Yaw:{self.yaw:.1f}°"


@dataclass
class SymmetryMetrics3D:
    """3D对称性指标"""
    midplane_asymmetry: np.ndarray
    euclidean_3d: np.ndarray
    mirror_deviation: np.ndarray
    y_diff_abs: np.ndarray
    normalization_distance: float


@dataclass
class Symmetry3DFeatures:
    """3D对称性特征数据结构"""
    metrics_per_frame: List[SymmetryMetrics3D]
    head_poses: List[HeadPose]
    coords_left_3d: np.ndarray
    coords_right_3d: np.ndarray
    landmark_names: List[str]
    frame_count: int
    mean_midplane_asymmetry: np.ndarray = None
    mean_euclidean_3d: np.ndarray = None

    def __post_init__(self):
        if self.metrics_per_frame:
            all_midplane = np.array([m.midplane_asymmetry for m in self.metrics_per_frame])
            all_euclidean = np.array([m.euclidean_3d for m in self.metrics_per_frame])
            self.mean_midplane_asymmetry = np.mean(all_midplane, axis=0)
            self.mean_euclidean_3d = np.mean(all_euclidean, axis=0)


@dataclass
class MidlinePoints:
    """面中线关键点"""
    top: Tuple[int, int]
    bottom: Tuple[int, int]
    center: Tuple[float, float]


# ==================== 辅助函数 ====================

def _safe_name(s: Union[str, None]) -> str:
    """安全的文件名转换"""
    s = str(s) if s is not None else ""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", s)[:80]


def build_pairs_and_names(cfg_list: List[Dict]) -> Tuple[List[Tuple[int, int]], List[str], Dict[str, List[int]]]:
    """从配置构建点对、名称和区域索引映射"""
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

def estimate_head_pose_from_landmarks(face_landmarks, image_width: int, image_height: int) -> HeadPose:
    """
    使用 solvePnP 估计头部精确姿态（Roll / Pitch / Yaw）
    返回角度单位：度，右手坐标系
    """
    # 1. 构造 2D image_points
    image_points = []
    for idx in PNP_INDICES:
        lm = face_landmarks[idx]
        image_points.append([lm.x * image_width, lm.y * image_height])
    image_points = np.array(image_points, dtype=np.float64)

    model_points = GENERIC_FACE_3D  # [6, 3]

    # 2. 相机内参：简单假设 fx = fy = image_width，主点在中心
    focal_length = image_width
    center = (image_width / 2.0, image_height / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1.0]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        # 失败就返回 0 姿态，rotation_matrix=None
        return HeadPose(roll=0.0, pitch=0.0, yaw=0.0, rotation_matrix=None)

    # 3. rvec -> 旋转矩阵
    rmat, _ = cv2.Rodrigues(rvec)

    # 4. 用 decomposeProjectionMatrix 提取欧拉角（OpenCV 返回单位就是度）
    proj_matrix = np.hstack((rmat, tvec))
    euler = cv2.decomposeProjectionMatrix(proj_matrix)[6]  # shape (3, 1)
    pitch, yaw, roll = [float(a) for a in euler]  # 度

    # 这里不再强行 clip 到 [-90, 90]，保留真实值，方便你诊断
    return HeadPose(
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        rotation_matrix=rmat,
        rvec=rvec,
        tvec=tvec,
    )


def align_landmarks(coords_3d, rmat):
    """[新增] 将3D坐标根据头部旋转矩阵进行矫正（回正）"""
    # 坐标中心化（相对于鼻尖或质心，这里简化假设coords已经相对鼻尖）
    # P_aligned = R^T * P_original
    if rmat is None: return coords_3d
    return np.dot(coords_3d, rmat)  # rmat是旋转到相机的，rmat.T是旋转回正，但在MP坐标系下直接乘即可近似


def compute_3d_symmetry_metrics(
        left_coords_3d: np.ndarray,
        right_coords_3d: np.ndarray,
        rotation_matrix: np.ndarray,  # [新增] 传入旋转矩阵
        normalization_dist: float
) -> SymmetryMetrics3D:
    """[修改] 计算经过姿态矫正的对称性指标"""

    # 1. 刚体矫正 (Alignment) - 消除歪头对Y轴差值的影响
    # 注意：输入的coords_3d 应该是相对于鼻尖(0,0,0)的相对坐标
    left_aligned = align_landmarks(left_coords_3d, rotation_matrix)
    right_aligned = align_landmarks(right_coords_3d, rotation_matrix)

    # 2. 计算镜像偏差 (Mirror Deviation)
    # 在矫正坐标系下，左脸镜像 = x取反
    left_mirrored = left_aligned.copy()
    left_mirrored[:, 0] = -left_mirrored[:, 0]  # Flip X

    mirror_diff = left_mirrored - right_aligned
    mirror_deviation = np.linalg.norm(mirror_diff, axis=1)  # 3D距离

    # 3. 计算垂直高度差 (Y-Diff) - 使用矫正后的坐标
    y_diff_abs = np.abs(left_aligned[:, 1] - right_aligned[:, 1])

    # 4. 中轴面距离差 (Midplane) - 矫正后X轴距离差
    midplane_asymmetry = np.abs(np.abs(left_aligned[:, 0]) - np.abs(right_aligned[:, 0]))

    # 5. 归一化 (使用瞳孔间距 IPD)
    scale = normalization_dist + 1e-8

    return SymmetryMetrics3D(
        midplane_asymmetry=(midplane_asymmetry / scale).astype(np.float32),
        euclidean_3d=(np.linalg.norm(left_aligned - right_aligned, axis=1) / scale).astype(np.float32),
        mirror_deviation=(mirror_deviation / scale).astype(np.float32),  # 核心指标
        y_diff_abs=(y_diff_abs / scale).astype(np.float32),  # 现在准确了
        normalization_distance=float(normalization_dist)
    )

def compute_midline_geometry(
        face_landmarks,
        face_contour_indices: List[int],
        feature_pairs: List[Tuple[int, int]],
        image_width: int,
        image_height: int
) -> Optional[MidlinePoints]:
    """计算面中线的几何位置"""
    try:
        lm_left = face_landmarks[EYE_INNER_CANTHUS_LEFT]
        lm_right = face_landmarks[EYE_INNER_CANTHUS_RIGHT]

        lx, ly = lm_left.x * image_width, lm_left.y * image_height
        rx, ry = lm_right.x * image_width, lm_right.y * image_height

        center_x, center_y = (lx + rx) / 2.0, (ly + ry) / 2.0
        dx, dy = (rx - lx), (ry - ly)
        if abs(dx) + abs(dy) < 1e-6:
            raise ValueError("眼内眦两点过于接近")

        center = np.array([center_x, center_y], dtype=np.float32)
        normal = np.array([-dy, dx], dtype=np.float32)
        denom = float(np.dot(normal, normal)) + 1e-6

        if not face_contour_indices:
            raise ValueError("face_contour区域索引为空")

        top_feature_idx = face_contour_indices[0]
        top_left_idx, top_right_idx = feature_pairs[top_feature_idx]

        bottom_feature_idx = face_contour_indices[-1]
        bottom_left_idx, bottom_right_idx = feature_pairs[bottom_feature_idx]

        top_lm_l = face_landmarks[top_left_idx]
        top_lm_r = face_landmarks[top_right_idx]
        bot_lm_l = face_landmarks[bottom_left_idx]
        bot_lm_r = face_landmarks[bottom_right_idx]

        top_mid = np.array([
            (top_lm_l.x * image_width + top_lm_r.x * image_width) / 2.0,
            (top_lm_l.y * image_height + top_lm_r.y * image_height) / 2.0
        ], dtype=np.float32)

        bot_mid = np.array([
            (bot_lm_l.x * image_width + bot_lm_r.x * image_width) / 2.0,
            (bot_lm_l.y * image_height + bot_lm_r.y * image_height) / 2.0
        ], dtype=np.float32)

        t_top = float(np.dot(top_mid - center, normal)) / denom
        t_bot = float(np.dot(bot_mid - center, normal)) / denom

        p_top = center + t_top * normal
        p_bot = center + t_bot * normal

        return MidlinePoints(
            top=(int(round(p_top[0])), int(round(p_top[1]))),
            bottom=(int(round(p_bot[0])), int(round(p_bot[1]))),
            center=(float(center_x), float(center_y))
        )
    except Exception:
        return None


# ==================== 主类 ====================

class FacialSymmetryAnalyzer3D:
    """面部对称性分析器 - 3D姿态不敏感版本"""

    def __init__(
            self,
            db_path: str,
            model_path: str = '/Users/cuijinglei/PycharmProjects/medicalProject/models/face_landmarker.task',
            verbose: bool = True,
    ):
        self.db_path = db_path
        self.model_path = model_path
        self.verbose = bool(verbose)

        self.feature_pairs, self.feature_names, self.region_feature_indices = (
            build_pairs_and_names(SYMMETRY_INDEX_CONFIG)
        )
        self.n_features = len(self.feature_pairs)

        # 调试信息
        self._debug_frame: Optional[np.ndarray] = None
        self._debug_landmarks = None
        self._debug_frame_idx: Optional[int] = None
        self._debug_metrics: Optional[SymmetryMetrics3D] = None

        if self.verbose:
            print("✅ 3D面部对称性分析器初始化完成")
            print(f"   - 对称点对数: {self.n_features}")
            print(f"   - 区域数: {len(self.region_feature_indices)}")

    def _create_landmarker(self) -> vision.FaceLandmarker:
        """创建MediaPipe FaceLandmarker实例"""
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )
        return vision.FaceLandmarker.create_from_options(options)

    def _reset_debug_info(self):
        """重置调试信息"""
        self._debug_frame = None
        self._debug_landmarks = None
        self._debug_frame_idx = None
        self._debug_metrics = None

    def extract_3d_features_from_video(
            self,
            video_path: str,
            start_frame: Optional[int] = None,
            end_frame: Optional[int] = None,
            fps: Optional[float] = None,
    ) -> Optional[Symmetry3DFeatures]:
        """从视频中提取3D面部特征和对称性指标"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        start_frame = max(0, start_frame or 0)
        end_frame = min(total_frames, end_frame or total_frames)

        if start_frame >= end_frame:
            raise ValueError(f"无效的帧范围: start={start_frame}, end={end_frame}")

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        landmarker = self._create_landmarker()

        coords_left_list = []
        coords_right_list = []
        metrics_list = []
        head_poses_list = []

        self._reset_debug_info()
        max_asymmetry_score = -1

        processed_idx = 0
        last_timestamp = -1
        frame_abs_idx = start_frame

        try:
            while cap.isOpened() and frame_abs_idx < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                h, w = frame.shape[:2]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp_mediapipe.Image(
                    image_format=mp_mediapipe.ImageFormat.SRGB,
                    data=rgb_frame
                )

                if fps and fps > 0:
                    timestamp_ms = int(processed_idx * 1000.0 / float(fps))
                else:
                    timestamp_ms = processed_idx * 33

                if timestamp_ms <= last_timestamp:
                    timestamp_ms = last_timestamp + 1
                last_timestamp = timestamp_ms
                processed_idx += 1

                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.face_landmarks:
                    face_landmarks = result.face_landmarks[0]

                    head_pose = estimate_head_pose_from_landmarks(face_landmarks, w, h)
                    # 准备3D坐标 (相对于鼻尖，用于Metric计算)
                    nose = face_landmarks[4]
                    origin = np.array([nose.x, nose.y, nose.z])

                    left_coords_rel = []
                    right_coords_rel = []
                    for li, ri in self.feature_pairs:
                        l = face_landmarks[li]
                        r = face_landmarks[ri]
                        left_coords_rel.append([l.x - origin[0], l.y - origin[1], l.z - origin[2]])
                        right_coords_rel.append([r.x - origin[0], r.y - origin[1], r.z - origin[2]])

                    left_np = np.array(left_coords_rel)
                    right_np = np.array(right_coords_rel)

                    # 计算IPD用于归一化
                    l_inner = np.array([face_landmarks[362].x, face_landmarks[362].y, face_landmarks[362].z])
                    r_inner = np.array([face_landmarks[133].x, face_landmarks[133].y, face_landmarks[133].z])
                    ipd = np.linalg.norm(l_inner - r_inner)

                    # [修改] 计算鲁棒指标
                    metrics = compute_3d_symmetry_metrics(left_np, right_np, head_pose.rotation_matrix, ipd)

                    coords_left_list.append(left_coords_rel)
                    coords_right_list.append(right_coords_rel)
                    metrics_list.append(metrics)
                    head_poses_list.append(head_pose)

                    current_asym_score = np.sum(metrics.mirror_deviation)
                    if current_asym_score > max_asymmetry_score:
                        max_asymmetry_score = current_asym_score
                        self._debug_frame = frame.copy()
                        self._debug_landmarks = face_landmarks
                        self._debug_metrics = metrics
                        self._debug_pose = head_pose  # 记得保存pose用于画图

                frame_abs_idx += 1

        finally:
            cap.release()
            try:
                landmarker.close()
            except:
                pass

        if not coords_left_list:
            return None

        return Symmetry3DFeatures(
            metrics_per_frame=metrics_list,
            head_poses=head_poses_list,
            coords_left_3d=np.array(coords_left_list),
            coords_right_3d=np.array(coords_right_list),
            landmark_names=list(self.feature_names),
            frame_count=len(metrics_list)
        )

    def draw_landmark_pairs(
            self,
            image: np.ndarray,
            face_landmarks,
            image_width: int,
            image_height: int
    ) -> np.ndarray:
        """在图像上绘制对称点对和连线（用于index_check）"""
        vis = image.copy()

        for (left_idx, right_idx) in self.feature_pairs:
            left_lm = face_landmarks[left_idx]
            right_lm = face_landmarks[right_idx]

            lx = int(left_lm.x * image_width)
            ly = int(left_lm.y * image_height)
            rx = int(right_lm.x * image_width)
            ry = int(right_lm.y * image_height)

            cv2.line(vis, (lx, ly), (rx, ry), VisConfig.COLOR_CONNECTION, VisConfig.CONNECTION_THICKNESS)
            cv2.circle(vis, (lx, ly), VisConfig.POINT_RADIUS, VisConfig.COLOR_LEFT_POINT, -1)
            cv2.circle(vis, (rx, ry), VisConfig.POINT_RADIUS, VisConfig.COLOR_RIGHT_POINT, -1)

            cv2.putText(vis, str(left_idx), (lx + 2, ly - 2),
                        VisConfig.TEXT_FONT, VisConfig.TEXT_SIZE, VisConfig.COLOR_LEFT_POINT, VisConfig.TEXT_THICKNESS)
            cv2.putText(vis, str(right_idx), (rx + 2, ry - 2),
                        VisConfig.TEXT_FONT, VisConfig.TEXT_SIZE, VisConfig.COLOR_RIGHT_POINT, VisConfig.TEXT_THICKNESS)

        return vis

    def save_index_check_image(
            self,
            frame_bgr: np.ndarray,
            face_landmarks,
            output_path: str
    ):
        """保存index_check图像（包含点对连线、索引标注和面中线）"""
        if frame_bgr is None or face_landmarks is None:
            return

        h, w = frame_bgr.shape[:2]
        vis = self.draw_landmark_pairs(frame_bgr, face_landmarks, w, h)

        face_contour_indices = self.region_feature_indices.get("face_contour", [])
        if face_contour_indices:
            midline = compute_midline_geometry(
                face_landmarks, face_contour_indices, self.feature_pairs, w, h
            )
            if midline:
                cv2.line(vis, midline.top, midline.bottom, VisConfig.COLOR_MIDLINE, VisConfig.MIDLINE_THICKNESS)
                cv2.circle(vis, midline.top, 4, VisConfig.COLOR_MIDLINE, -1)
                cv2.circle(vis, midline.bottom, 4, VisConfig.COLOR_MIDLINE, -1)

        cv2.imwrite(str(output_path), vis)

    def draw_landmark_pairs_with_asymmetry(
            self,
            image: np.ndarray,
            face_landmarks,
            metrics: SymmetryMetrics3D,
            image_width: int,
            image_height: int
    ) -> np.ndarray:
        """在图像上绘制对称点对，颜色深浅表示不对称程度"""
        vis = image.copy()

        asymmetry = metrics.mirror_deviation
        max_asym = max(asymmetry.max(), 0.01)

        for j, (left_idx, right_idx) in enumerate(self.feature_pairs):
            left_lm = face_landmarks[left_idx]
            right_lm = face_landmarks[right_idx]

            lx = int(left_lm.x * image_width)
            ly = int(left_lm.y * image_height)
            rx = int(right_lm.x * image_width)
            ry = int(right_lm.y * image_height)

            asym_ratio = asymmetry[j] / max_asym
            color = (0, int(255 * (1 - asym_ratio)), int(255 * asym_ratio))

            cv2.line(vis, (lx, ly), (rx, ry), color, 1)
            cv2.circle(vis, (lx, ly), 2, VisConfig.COLOR_LEFT_POINT, -1)
            cv2.circle(vis, (rx, ry), 2, VisConfig.COLOR_RIGHT_POINT, -1)

        return vis

    def visualize_3d_comparison(
            self,
            features: Symmetry3DFeatures,
            title: str,
            output_path: str
    ):
        """创建3D对称性分析可视化对比图"""
        fig = plt.figure(figsize=(16, 12))

        n_frames = features.frame_count
        n_features = len(features.landmark_names)

        y_diff_all = np.array([m.y_diff_abs for m in features.metrics_per_frame])
        midplane_all = np.array([m.midplane_asymmetry for m in features.metrics_per_frame])
        mirror_all = np.array([m.mirror_deviation for m in features.metrics_per_frame])
        euclidean_all = np.array([m.euclidean_3d for m in features.metrics_per_frame])

        y_diff_mean = y_diff_all.mean(axis=0)
        midplane_mean = midplane_all.mean(axis=0)
        mirror_mean = mirror_all.mean(axis=0)
        euclidean_mean = euclidean_all.mean(axis=0)

        # 1. 柱状图对比
        ax1 = fig.add_subplot(2, 2, 1)
        x = np.arange(n_features)
        width = 0.2

        ax1.bar(x - 1.5 * width, y_diff_mean, width, label='|Δy| (原始)', alpha=0.8, color='red')
        ax1.bar(x - 0.5 * width, midplane_mean, width, label='中轴面距离差', alpha=0.8, color='blue')
        ax1.bar(x + 0.5 * width, mirror_mean, width, label='镜像偏差', alpha=0.8, color='green')
        ax1.bar(x + 1.5 * width, euclidean_mean, width, label='3D欧几里得', alpha=0.8, color='orange')

        ax1.set_xlabel('特征点对')
        ax1.set_ylabel('归一化不对称度')
        ax1.set_title('各指标对比（时间平均）')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.set_xticks(x[::5])
        ax1.set_xticklabels([features.landmark_names[i] for i in x[::5]], rotation=45, ha='right', fontsize=7)
        ax1.grid(axis='y', alpha=0.3)

        # 2. 头部姿态时间序列
        ax2 = fig.add_subplot(2, 2, 2)
        t = np.arange(n_frames)

        rolls = [p.roll for p in features.head_poses]
        pitches = [p.pitch for p in features.head_poses]
        yaws = [p.yaw for p in features.head_poses]

        ax2.plot(t, rolls, label='Roll', linewidth=1.5, color='red')
        ax2.plot(t, pitches, label='Pitch', linewidth=1.5, color='green')
        ax2.plot(t, yaws, label='Yaw', linewidth=1.5, color='blue')

        ax2.set_xlabel('帧')
        ax2.set_ylabel('角度 (°)')
        ax2.set_title('头部姿态变化')
        ax2.legend(loc='upper right')
        ax2.grid(alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # 3. 全局时间序列
        ax3 = fig.add_subplot(2, 2, 3)

        y_diff_ts = y_diff_all.mean(axis=1)
        midplane_ts = midplane_all.mean(axis=1)
        mirror_ts = mirror_all.mean(axis=1)

        ax3.plot(t, y_diff_ts, label='|Δy| (原始)', linewidth=1.5, color='red', alpha=0.8)
        ax3.plot(t, midplane_ts, label='中轴面距离差', linewidth=1.5, color='blue', alpha=0.8)
        ax3.plot(t, mirror_ts, label='镜像偏差', linewidth=1.5, color='green', alpha=0.8)

        ax3.set_xlabel('帧')
        ax3.set_ylabel('归一化不对称度')
        ax3.set_title('全局不对称度时间序列')
        ax3.legend(loc='upper right')
        ax3.grid(alpha=0.3)

        # 4. 相关性散点图
        ax4 = fig.add_subplot(2, 2, 4)

        ax4.scatter(np.abs(rolls), y_diff_ts, alpha=0.5, label='|Δy| vs |Roll|', color='red', s=20)
        ax4.scatter(np.abs(rolls), mirror_ts, alpha=0.5, label='镜像偏差 vs |Roll|', color='green', s=20)

        ax4.set_xlabel('|Roll| (°)')
        ax4.set_ylabel('不对称度')
        ax4.set_title('姿态与不对称度的相关性')
        ax4.legend(loc='upper right')
        ax4.grid(alpha=0.3)

        if len(rolls) > 3:
            corr_y_roll = np.corrcoef(np.abs(rolls), y_diff_ts)[0, 1]
            corr_mirror_roll = np.corrcoef(np.abs(rolls), mirror_ts)[0, 1]
            ax4.text(0.05, 0.95, f'r(|Δy|,|Roll|)={corr_y_roll:.3f}\nr(Mirror,|Roll|)={corr_mirror_roll:.3f}',
                     transform=ax4.transAxes, fontsize=9, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle(f'{title}\n3D姿态不敏感对称性分析', fontsize=12, fontweight='bold')
        plt.tight_layout()

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def visualize_region_comparison(
            self,
            features: Symmetry3DFeatures,
            title: str,
            output_path: str
    ):
        """按区域可视化对称性分析结果"""
        regions = list(self.region_feature_indices.keys())
        n_regions = len(regions)

        region_y_diff = {}
        region_mirror = {}

        for region, indices in self.region_feature_indices.items():
            y_diff_all = np.array([m.y_diff_abs[indices] for m in features.metrics_per_frame])
            mirror_all = np.array([m.mirror_deviation[indices] for m in features.metrics_per_frame])

            region_y_diff[region] = y_diff_all.mean()
            region_mirror[region] = mirror_all.mean()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        x = np.arange(n_regions)
        width = 0.35

        ax1.bar(x - width / 2, [region_y_diff[r] for r in regions], width,
                label='|Δy| (原始)', color='red', alpha=0.8)
        ax1.bar(x + width / 2, [region_mirror[r] for r in regions], width,
                label='镜像偏差 (3D)', color='green', alpha=0.8)

        ax1.set_xlabel('面部区域')
        ax1.set_ylabel('归一化不对称度')
        ax1.set_title('各区域不对称度对比')
        ax1.set_xticks(x)
        ax1.set_xticklabels(regions, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # 雷达图
        angles = np.linspace(0, 2 * np.pi, n_regions, endpoint=False).tolist()
        angles += angles[:1]

        y_diff_values = [region_y_diff[r] for r in regions] + [region_y_diff[regions[0]]]
        mirror_values = [region_mirror[r] for r in regions] + [region_mirror[regions[0]]]

        ax2 = fig.add_subplot(122, projection='polar')
        ax2.plot(angles, y_diff_values, 'o-', linewidth=2, label='|Δy| (原始)', color='red')
        ax2.fill(angles, y_diff_values, alpha=0.25, color='red')
        ax2.plot(angles, mirror_values, 'o-', linewidth=2, label='镜像偏差 (3D)', color='green')
        ax2.fill(angles, mirror_values, alpha=0.25, color='green')

        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(regions)
        ax2.set_title('区域不对称度雷达图')
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        plt.suptitle(f'{title}\n区域对称性分析', fontsize=12, fontweight='bold')
        plt.tight_layout()

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def draw_pose_axes(self, img, head_pose: HeadPose, face_landmarks, img_w, img_h):
        """
        在鼻尖位置画出 XYZ 坐标轴
        X: 红（向右），Y: 绿（向下），Z: 蓝（朝外）
        """
        if head_pose.rotation_matrix is None: return img

        nose = face_landmarks[NOSE_TIP]

        focal_length = img_w
        center = (img_w / 2.0, img_h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1.0]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        axis_len = 80.0  # 轴长度，可以按图像尺寸调

        axis_points_3d = np.float32([
            [0, 0, 0],  # origin
            [axis_len, 0, 0],  # X
            [0, axis_len, 0],  # Y
            [0, 0, axis_len],  # Z
        ]).reshape(-1, 3)

        img_points, _ = cv2.projectPoints(axis_points_3d, head_pose.rvec, head_pose.tvec, camera_matrix, dist_coeffs)

        p0 = tuple(img_points[0].ravel().astype(int))
        px = tuple(img_points[1].ravel().astype(int))
        py = tuple(img_points[2].ravel().astype(int))
        pz = tuple(img_points[3].ravel().astype(int))

        out = img.copy()
        cv2.line(out, p0, px, (255, 0, 255), 2)  # X
        cv2.line(out, p0, py, (0, 255, 0), 2)  # Y - 绿
        cv2.line(out, p0, pz, (255, 0, 0), 2)  # Z - 蓝
        cv2.putText(out, "X", px, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        cv2.putText(out, "Y", py, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        cv2.putText(out, "Z", pz, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
        cv2.putText(out, f"Roll:{head_pose.roll:.1f}  Pitch:{head_pose.pitch:.1f}  Yaw:{head_pose.yaw:.1f}",
                    (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2)
        return out

    def draw_asymmetry_heatmap(
            self,
            img: np.ndarray,
            pairs: List[Tuple[int, int]],
            left_lms_2d: np.ndarray,
            right_lms_2d: np.ndarray,
            metrics: SymmetryMetrics3D,
            img_w: int,
            img_h: int,
    ) -> np.ndarray:
        """
        颜色编码连线：基于 mirror_deviation（已按 IPD 归一化）
        Green: < 2% IPD (正常)
        Yellow: 2% - 5% IPD (轻度)
        Red: > 5% IPD (明显)
        """
        overlay = img.copy()

        for i, (li, ri) in enumerate(pairs):
            lx = int(left_lms_2d[i, 0] * img_w)
            ly = int(left_lms_2d[i, 1] * img_h)
            rx = int(right_lms_2d[i, 0] * img_w)
            ry = int(right_lms_2d[i, 1] * img_h)

            score = float(metrics.mirror_deviation[i])  # 已归一化

            if score < TH_LOW:
                color = (0, 255, 0)
                thickness = 1
            elif score < TH_HIGH:
                color = (0, 255, 255)
                thickness = 2
            else:
                color = (0, 0, 255)
                thickness = 3

            cv2.line(overlay, (lx, ly), (rx, ry), color, thickness, cv2.LINE_AA)

        cv2.putText(overlay, "Asymmetry vs IPD (Lower is Better)", (10, img_h - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(overlay,
                    f"Green: < {TH_LOW * 100:.1f}%   Yellow: {TH_LOW * 100:.1f}-{TH_HIGH * 100:.1f}%   Red: > {TH_HIGH * 100:.1f}%",
                    (10, img_h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return overlay

    def draw_mirror_comparison(
            self,
            img: np.ndarray,
            pairs: List[Tuple[int, int]],
            left_lms_2d: np.ndarray,
            right_lms_2d: np.ndarray,
            face_landmarks,
            img_w: int,
            img_h: int,
    ) -> np.ndarray:
        """
        镜像对比图：
        - 蓝点：真实右侧点
        - 红点：以鼻尖为轴镜像过来的左侧点
        用于直观看“镜像偏差”
        """
        vis = img.copy()
        nose_x = face_landmarks[NOSE_TIP].x * img_w

        for i, (li, ri) in enumerate(pairs):
            rx = int(right_lms_2d[i, 0] * img_w)
            ry = int(right_lms_2d[i, 1] * img_h)
            lx = int(left_lms_2d[i, 0] * img_w)
            ly = int(left_lms_2d[i, 1] * img_h)

            mirror_lx = int(nose_x + (nose_x - lx))
            mirror_ly = ly

            # 真实右点（蓝）
            cv2.circle(vis, (rx, ry), 2, (255, 0, 0), -1)
            # 镜像左点（红）
            cv2.circle(vis, (mirror_lx, mirror_ly), 2, (0, 0, 255), -1)
            # 连接线
            cv2.line(vis, (rx, ry), (mirror_lx, mirror_ly), (255, 255, 255), 1)

        cv2.putText(vis, "Mirror Comparison: Blue=Real Right, Red=Mirrored Left",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return vis

    def analyze_single_video(
            self,
            video_path: str,
            output_dir: str,
            start_frame: Optional[int] = None,
            end_frame: Optional[int] = None,
            fps: Optional[float] = None,
            video_id: Optional[int] = None,
            patient_id: Optional[str] = None,
            examination_id: Optional[int] = None,
            action_name: Optional[str] = None,
    ) -> Dict:
        """
        分析单个视频的面部对称性

        文件命名格式: {examination_id}_{action_name}_{type}.png
        所有文件直接保存在output_dir下，不创建子文件夹
        """
        # 构建文件名前缀：examination_id_action_name
        safe_exam = _safe_name(str(examination_id)) if examination_id else "exam"
        safe_action = _safe_name(action_name) if action_name else "action"
        prefix = f"{safe_exam}_{safe_action}"

        if self.verbose:
            print(f"  处理: {prefix}")

        # 提取3D特征
        features = self.extract_3d_features_from_video(video_path, start_frame, end_frame, fps)

        if features is None:
            raise ValueError("未检测到人脸关键点")

        os.makedirs(output_dir, exist_ok=True)

        # 1. 保存 index_check 图像
        if self._debug_frame is not None and self._debug_landmarks is not None:
            index_check_path = os.path.join(output_dir, f"{prefix}_index_check.png")
            self.save_index_check_image(self._debug_frame, self._debug_landmarks, index_check_path)

        # 2. 保存不对称程度热力图 & 镜像对比图
        if self._debug_frame is not None and self._debug_landmarks is not None and self._debug_metrics is not None:
            h, w = self._debug_frame.shape[:2]

            # 2D 关键点坐标（归一化到 [0,1]，后面乘 w/h）
            left_2d = np.array(
                [[self._debug_landmarks[li].x, self._debug_landmarks[li].y]
                 for (li, ri) in self.feature_pairs],
                dtype=np.float32
            )
            right_2d = np.array(
                [[self._debug_landmarks[ri].x, self._debug_landmarks[ri].y]
                 for (li, ri) in self.feature_pairs],
                dtype=np.float32
            )

            # 头部姿态（PnP）
            pose = estimate_head_pose_from_landmarks(self._debug_landmarks, w, h)

            # 2.1 先画坐标轴
            img_axes = self.draw_pose_axes(self._debug_frame.copy(), pose, self._debug_landmarks, w, h)

            # 2.2 在坐标轴图上叠加基于 IPD 的颜色热力图
            img_heat = self.draw_asymmetry_heatmap(
                img_axes,
                self.feature_pairs,
                left_2d,
                right_2d,
                self._debug_metrics,
                w,
                h,
            )

            # 2.3 再画一张镜像对比图
            img_mirror = self.draw_mirror_comparison(
                self._debug_frame.copy(),
                self.feature_pairs,
                left_2d,
                right_2d,
                self._debug_landmarks,
                w,
                h,
            )

            # 如果你仍然想画面中线，可以在 img_heat 上再叠一条：
            face_contour_indices = self.region_feature_indices.get("face_contour", [])
            if face_contour_indices:
                midline = compute_midline_geometry(
                    self._debug_landmarks, face_contour_indices, self.feature_pairs, w, h
                )
                if midline:
                    cv2.line(img_heat, midline.top, midline.bottom, VisConfig.COLOR_MIDLINE, 2)

            # 保存两张图
            asymmetry_path = os.path.join(output_dir, f"{prefix}_asymmetry.png")
            mirror_path = os.path.join(output_dir, f"{prefix}_mirror.png")
            cv2.imwrite(asymmetry_path, img_heat)
            cv2.imwrite(mirror_path, img_mirror)

        # 3. 保存3D对比分析图
        comparison_path = os.path.join(output_dir, f"{prefix}_3d_comparison.png")
        self.visualize_3d_comparison(
            features,
            f"Exam {examination_id} - {action_name}",
            comparison_path
        )

        # 4. 保存区域对比图
        region_path = os.path.join(output_dir, f"{prefix}_region.png")
        self.visualize_region_comparison(
            features,
            f"Exam {examination_id} - {action_name}",
            region_path
        )

        # 5. 构建结果数据
        result = {
            "video_id": video_id,
            "video_path": video_path,
            "patient_id": patient_id,
            "examination_id": examination_id,
            "action_name": action_name,
            "frame_count": features.frame_count,
            "metrics_summary": {
                "y_diff_mean": float(np.mean([m.y_diff_abs.mean() for m in features.metrics_per_frame])),
                "mirror_deviation_mean": float(
                    np.mean([m.mirror_deviation.mean() for m in features.metrics_per_frame])),
                "midplane_asymmetry_mean": float(
                    np.mean([m.midplane_asymmetry.mean() for m in features.metrics_per_frame])),
                "euclidean_3d_mean": float(np.mean([m.euclidean_3d.mean() for m in features.metrics_per_frame])),
            },
            "pose_summary": {
                "roll_mean": float(np.mean([p.roll for p in features.head_poses])),
                "roll_std": float(np.std([p.roll for p in features.head_poses])),
                "pitch_mean": float(np.mean([p.pitch for p in features.head_poses])),
                "yaw_mean": float(np.mean([p.yaw for p in features.head_poses])),
            },
            "region_metrics": {}
        }

        for region, indices in self.region_feature_indices.items():
            y_diff = np.mean([m.y_diff_abs[indices].mean() for m in features.metrics_per_frame])
            mirror = np.mean([m.mirror_deviation[indices].mean() for m in features.metrics_per_frame])
            result["region_metrics"][region] = {
                "y_diff_mean": float(y_diff),
                "mirror_deviation_mean": float(mirror)
            }

        # 6. 保存JSON
        json_path = os.path.join(output_dir, f"{prefix}_metrics.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return result

    def batch_process_database(
            self,
            output_dir: str,
            limit: Optional[int] = None,
            action_filter: Optional[List[str]] = None,
            use_multiprocessing: bool = True,
            num_workers: Optional[int] = None,
    ) -> List[Dict]:
        """
        批量处理数据库中的所有视频（支持多进程）

        Args:
            output_dir: 输出目录
            limit: 处理数量上限
            action_filter: 按动作英文名过滤
            use_multiprocessing: 是否使用多进程
            num_workers: 进程数（None则自动选择）
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
            SELECT 
                vf.video_id,
                vf.examination_id,
                vf.action_id,
                vf.file_path,
                vf.start_frame,
                vf.end_frame,
                vf.fps,
                at.action_name_en,
                e.patient_id
            FROM video_files vf
            JOIN action_types at ON vf.action_id = at.action_id
            JOIN examinations e ON vf.examination_id = e.examination_id
            WHERE vf.file_exists = 1
        """

        if action_filter:
            placeholders = ",".join(["?"] * len(action_filter))
            query += f" AND at.action_name_en IN ({placeholders})"
            cursor.execute(query, action_filter)
        else:
            cursor.execute(query)

        videos = cursor.fetchall()
        conn.close()

        if limit is not None:
            videos = videos[:limit]

        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "=" * 60)
        print("3D 姿态鲁棒面部对称性 - 批量处理模式")
        print(f"总视频数: {len(videos)}")
        print(f"输出目录: {output_dir}")
        print(f"多进程: {use_multiprocessing}")
        print("=" * 60 + "\n")

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
                futs = [executor.submit(_worker_analyze_one_3d, task) for task in tasks]
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
                print(f"[{i}/{len(tasks)}] exam={task['examination_id']} action={task['action_name']}")
                try:
                    r = self.analyze_single_video(
                        video_path=task["video_path"],
                        output_dir=task["output_dir"],
                        start_frame=task["start_frame"],
                        end_frame=task["end_frame"],
                        fps=task["fps"],
                        video_id=task["video_id"],
                        patient_id=task["patient_id"],
                        examination_id=task["examination_id"],
                        action_name=task["action_name"],
                    )
                    results.append(r)
                except Exception as e:
                    errors.append({
                        "video_id": task["video_id"],
                        "video_path": task["video_path"],
                        "examination_id": task["examination_id"],
                        "action_name": task["action_name"],
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


# ==================== Multiprocessing Worker ====================

def _worker_analyze_one_3d(task: Dict):
    """
    子进程worker函数（3D版本）

    Args:
        task: 任务字典

    Returns:
        (成功标志, 结果数据)
    """
    try:
        analyzer = FacialSymmetryAnalyzer3D(
            db_path=task["db_path"],
            model_path=task["model_path"],
            verbose=False  # 子进程不打印详细信息
        )

        res = analyzer.analyze_single_video(
            video_path=task["video_path"],
            output_dir=task["output_dir"],
            start_frame=task["start_frame"],
            end_frame=task["end_frame"],
            fps=task["fps"],
            video_id=task.get("video_id"),
            patient_id=task.get("patient_id"),
            examination_id=task.get("examination_id"),
            action_name=task.get("action_name"),
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
    """主函数"""
    # ========= 路径配置 =========
    db_path = "/Users/cuijinglei/PycharmProjects/medicalProject/facialPalsy/facialPalsy.db"
    model_path = "/Users/cuijinglei/PycharmProjects/medicalProject/models/face_landmarker.task"
    output_dir = "/Users/cuijinglei/Documents/facialPalsy/HGFA/symmetry_pose_robust_3d"
    # ============================

    analyzer = FacialSymmetryAnalyzer3D(
        db_path=db_path,
        model_path=model_path,
        verbose=True
    )

    analyzer.batch_process_database(
        output_dir=output_dir,
        limit=None,
        action_filter=None,
        use_multiprocessing=True,
        num_workers=8,
    )

    print("✅ 3D 姿态鲁棒面部对称性批量分析已完成")


if __name__ == '__main__':
    main()