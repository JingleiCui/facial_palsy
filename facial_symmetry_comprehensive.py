# -*- coding: utf-8 -*-
"""
facial_palsy_analysis_comprehensive.py

综合面部麻痹分析系统 - 整合两篇论文方法

论文1: "Numerical Approach to Facial Palsy Using a Novel Registration Method with 3D Facial Landmark" (Sensors 2022)
论文2: "Assessing 3D volumetric asymmetry in facial palsy patients via advanced multi-view landmarks and radial curves" (MVA 2025)

核心功能：
1. 3D配准方法 (Scale Matching + Global Registration + Point-to-Plane ICP)
2. 距离对称性、角度对称性、地标运动量分析
3. 区域级面部分析 (按面神经分支划分)
4. 临床分级评估 (HB, Sunnybrook, eFACE)
5. 可解释性可视化

作者: Rennie
日期: 2025-12-11
"""

import os
import re
import json
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum
import warnings

import cv2
import numpy as np
import mediapipe as mp_mediapipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy import ndimage
from scipy.spatial import procrustes
from scipy.spatial.transform import Rotation

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors

# 中文字体配置
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'PingFang SC']
matplotlib.rcParams['axes.unicode_minus'] = False


# ==================== 枚举定义 ====================

class PalsySide(Enum):
    """面瘫侧别"""
    NONE = "None"
    LEFT = "Left"
    RIGHT = "Right"
    BILATERAL = "Bilateral"


class HBGrade(Enum):
    """House-Brackmann分级"""
    GRADE_1 = 1  # 正常
    GRADE_2 = 2  # 轻度
    GRADE_3 = 3  # 中度
    GRADE_4 = 4  # 中重度
    GRADE_5 = 5  # 重度
    GRADE_6 = 6  # 完全麻痹


class FacialNerveBranch(Enum):
    """面神经分支"""
    TEMPORAL = "Temporal"  # 颞支 - 额肌
    ZYGOMATIC = "Zygomatic"  # 颧支 - 眼轮匝肌
    BUCCAL = "Buccal"  # 颊支 - 颊肌、提上唇肌
    MARGINAL_MANDIBULAR = "Marginal_Mandibular"  # 下颌缘支 - 降口角肌
    CERVICAL = "Cervical"  # 颈支 - 颈阔肌


# ==================== 常量定义 ====================

# 按面神经分支划分的肌肉区域和对应关键点
# 参考 Kim et al. 2022 的17个肌肉区域定义
FACIAL_MUSCLE_GROUPS = {
    # 1. 颞支区域 (Temporal branch)
    "Frontalis": {
        "branch": FacialNerveBranch.TEMPORAL,
        "description": "额肌 - 抬眉",
        "pairs": {
            "left": [336, 296, 334, 293, 300, 276],
            "right": [107, 66, 105, 63, 70, 46],
        }
    },
    "Corrugator": {
        "branch": FacialNerveBranch.TEMPORAL,
        "description": "皱眉肌",
        "pairs": {
            "left": [285],
            "right": [55],
        }
    },
    "Procerus": {
        "branch": FacialNerveBranch.TEMPORAL,
        "description": "降眉间肌",
        "pairs": {
            "left": [9],
            "right": [9],  # 中线单点
        }
    },

    # 2. 颧支区域 (Zygomatic branch)
    "Orbicularis_Oculi": {
        "branch": FacialNerveBranch.ZYGOMATIC,
        "description": "眼轮匝肌 - 闭眼",
        "pairs": {
            "left": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            "right": [133, 155, 154, 153, 145, 144, 163, 7, 33, 246, 161, 160, 159, 158, 157, 173],
        }
    },
    "Zygomaticus_Minor": {
        "branch": FacialNerveBranch.ZYGOMATIC,
        "description": "颧小肌",
        "pairs": {
            "left": [355, 329, 277],
            "right": [126, 100, 47],
        }
    },
    "Zygomaticus_Major": {
        "branch": FacialNerveBranch.ZYGOMATIC,
        "description": "颧大肌 - 微笑",
        "pairs": {
            "left": [361, 323, 454],
            "right": [132, 93, 234],
        }
    },

    # 3. 颊支区域 (Buccal branch)
    "Levator_Labii_Superioris": {
        "branch": FacialNerveBranch.BUCCAL,
        "description": "提上唇肌",
        "pairs": {
            "left": [250, 458, 459, 309, 392],
            "right": [20, 238, 239, 79, 166],
        }
    },
    "Nasalis": {
        "branch": FacialNerveBranch.BUCCAL,
        "description": "鼻肌 - 皱鼻",
        "pairs": {
            "left": [289, 305, 460, 294, 358, 279],
            "right": [59, 75, 240, 64, 129, 49],
        }
    },
    "Buccinator": {
        "branch": FacialNerveBranch.BUCCAL,
        "description": "颊肌 - 鼓腮",
        "pairs": {
            "left": [330, 280, 411, 427],
            "right": [101, 50, 187, 207],
        }
    },
    "Orbicularis_Oris_Upper": {
        "branch": FacialNerveBranch.BUCCAL,
        "description": "口轮匝肌上部",
        "pairs": {
            "left": [267, 269, 270, 409, 291, 308, 415, 310, 311, 312],
            "right": [37, 39, 40, 185, 61, 78, 191, 80, 81, 82],
        }
    },
    "Orbicularis_Oris_Lower": {
        "branch": FacialNerveBranch.BUCCAL,
        "description": "口轮匝肌下部",
        "pairs": {
            "left": [317, 402, 318, 324, 308, 291, 375, 321, 405, 314],
            "right": [87, 178, 88, 95, 78, 61, 146, 91, 181, 84],
        }
    },
    "Risorius": {
        "branch": FacialNerveBranch.BUCCAL,
        "description": "笑肌",
        "pairs": {
            "left": [391, 393, 322],
            "right": [165, 167, 92],
        }
    },

    # 4. 下颌缘支区域 (Marginal mandibular branch)
    "Depressor_Anguli_Oris": {
        "branch": FacialNerveBranch.MARGINAL_MANDIBULAR,
        "description": "降口角肌",
        "pairs": {
            "left": [375, 291, 409, 270, 269, 267, 0, 17],
            "right": [146, 61, 185, 40, 39, 37, 0, 17],
        }
    },
    "Mentalis": {
        "branch": FacialNerveBranch.MARGINAL_MANDIBULAR,
        "description": "颏肌",
        "pairs": {
            "left": [377, 152, 378, 379, 365, 397],
            "right": [148, 152, 149, 150, 136, 172],
        }
    },

    # 5. 颈支区域 (Cervical branch)
    "Platysma": {
        "branch": FacialNerveBranch.CERVICAL,
        "description": "颈阔肌",
        "pairs": {
            "left": [288, 397, 365, 379, 378],
            "right": [58, 172, 136, 150, 149],
        }
    },

    # 辅助区域
    "Nose_Tip": {
        "branch": None,  # 不属于特定分支，用于参考
        "description": "鼻尖 - 参考点",
        "pairs": {
            "left": [4, 5, 6, 122, 188, 114, 217, 126, 142, 129, 358, 327, 326, 2, 97, 99],
            "right": [4, 5, 6, 351, 412, 343, 437, 355, 371, 358, 129, 98, 97, 2, 326, 328],
        }
    },
    "Face_Contour": {
        "branch": None,
        "description": "面部轮廓",
        "pairs": {
            "left": [338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377],
            "right": [109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148],
        }
    },
    "Masseter": {
        "branch": None,
        "description": "咬肌",
        "pairs": {
            "left": [454, 356, 389, 251, 284, 332, 297, 338, 389],
            "right": [234, 127, 162, 21, 54, 103, 67, 109, 162],
        }
    },
    "Temporalis": {
        "branch": FacialNerveBranch.TEMPORAL,
        "description": "颞肌",
        "pairs": {
            "left": [251, 284, 332, 297, 338, 10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132],
            "right": [21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361],
        }
    },
}

# 标准动作与评估的面神经分支映射
ACTION_BRANCH_MAPPING = {
    "RaiseEyebrow": [FacialNerveBranch.TEMPORAL],
    "CloseEyes": [FacialNerveBranch.ZYGOMATIC],
    "CloseEyesTightly": [FacialNerveBranch.ZYGOMATIC],
    "Smile": [FacialNerveBranch.ZYGOMATIC, FacialNerveBranch.BUCCAL],
    "ShowTeeth": [FacialNerveBranch.BUCCAL],
    "PuffCheeks": [FacialNerveBranch.BUCCAL],
    "WhistlePout": [FacialNerveBranch.BUCCAL],
    "Pout": [FacialNerveBranch.BUCCAL],
    "Grimace": [FacialNerveBranch.MARGINAL_MANDIBULAR, FacialNerveBranch.BUCCAL],
    "WrinkleNose": [FacialNerveBranch.BUCCAL],
    "DropJaw": [FacialNerveBranch.MARGINAL_MANDIBULAR],
}

# 关键点索引
EYE_INNER_CANTHUS_LEFT = 362
EYE_INNER_CANTHUS_RIGHT = 133
NOSE_TIP = 4
FOREHEAD_CENTER = 10
CHIN = 152
LEFT_IRIS_CENTER = 473
RIGHT_IRIS_CENTER = 468

# PnP解算用的通用3D人脸模型
GENERIC_FACE_3D = np.array([
    (0.0, 0.0, 0.0),  # Nose Tip (Index 4)
    (0.0, -330.0, -65.0),  # Chin (Index 152)
    (-225.0, 170.0, -135.0),  # Left Eye Corner (Index 33)
    (225.0, 170.0, -135.0),  # Right Eye Corner (Index 263)
    (-150.0, -150.0, -125.0),  # Left Mouth Corner (Index 61)
    (150.0, -150.0, -125.0)  # Right Mouth Corner (Index 291)
], dtype=np.float64)

PNP_INDICES = [4, 152, 33, 263, 61, 291]

# 阈值设置
TH_NORMAL = 0.05  # 正常阈值
TH_MILD = 0.10  # 轻度阈值
TH_MODERATE = 0.20  # 中度阈值
TH_SEVERE = 0.35  # 重度阈值


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
class SymmetryMetrics:
    """对称性指标 - 整合Kim et al.的三种方法"""
    # 1. 距离对称性 (Distance Symmetry)
    distance_symmetry: np.ndarray  # 镜像点对间的3D距离

    # 2. 角度对称性 (Angle Symmetry) - 余弦相似度
    angle_symmetry: np.ndarray  # 中矢面法向量与点对向量的余弦相似度

    # 3. 地标运动量 (用于动态分析)
    # 注：这个在比较两个表情时计算

    # 其他派生指标
    midplane_asymmetry: np.ndarray  # 到中矢面的距离差
    euclidean_3d: np.ndarray  # 3D欧几里得距离
    mirror_deviation: np.ndarray  # 镜像偏差
    y_diff_abs: np.ndarray  # Y轴差异

    # 归一化参数
    normalization_distance: float  # 归一化距离(内眦间距)


@dataclass
class MuscleGroupMetrics:
    """单个肌肉组的对称性指标"""
    muscle_name: str
    branch: Optional[FacialNerveBranch]

    # 平均对称性指标
    mean_distance_symmetry: float
    mean_angle_symmetry: float
    mean_mirror_deviation: float

    # 左右侧分别的运动幅度
    left_movement: float = 0.0
    right_movement: float = 0.0
    movement_asymmetry: float = 0.0

    # 严重度评级
    severity_score: float = 0.0  # 0-1，越大越严重


@dataclass
class BranchAnalysis:
    """面神经分支分析结果"""
    branch: FacialNerveBranch
    muscle_metrics: List[MuscleGroupMetrics]

    # 分支级别聚合指标
    mean_asymmetry: float = 0.0
    severity_score: float = 0.0
    affected_side: Optional[PalsySide] = None


@dataclass
class ActionAnalysis:
    """单个动作的分析结果"""
    action_name: str
    frame_count: int

    # 帧级别指标
    metrics_per_frame: List[SymmetryMetrics]
    head_poses: List[HeadPose]

    # 肌肉组级别指标
    muscle_metrics: Dict[str, MuscleGroupMetrics]

    # 分支级别指标
    branch_metrics: Dict[FacialNerveBranch, BranchAnalysis]

    # 动作级别聚合
    overall_asymmetry: float = 0.0
    affected_side: Optional[PalsySide] = None
    severity_score: float = 0.0


@dataclass
class SessionDiagnosis:
    """检查级诊断结果"""
    patient_id: Optional[int]
    examination_id: int

    # 各动作分析
    action_analyses: Dict[str, ActionAnalysis]

    # 诊断结果
    has_palsy: bool = False
    palsy_side: PalsySide = PalsySide.NONE

    # 临床分级
    hb_grade: HBGrade = HBGrade.GRADE_1
    sunnybrook_score: float = 100.0
    eface_score: float = 100.0

    # 各分支功能状态
    branch_status: Dict[FacialNerveBranch, float] = field(default_factory=dict)

    # 置信度
    confidence: float = 0.0


@dataclass
class LandmarkMovement:
    """地标运动量分析 (Kim et al. 方法3)"""
    left_movements: np.ndarray  # 左侧地标运动量
    right_movements: np.ndarray  # 右侧地标运动量
    movement_asymmetry: np.ndarray  # |左-右|

    # 按肌肉组汇总
    muscle_movement_asymmetry: Dict[str, float] = field(default_factory=dict)


# ==================== 配准方法 (Kim et al. 2022) ====================

class RegistrationMethod:
    """
    3D地标配准方法

    实现 Kim et al. (2022) 论文中的配准流程:
    1. Scale Matching - 尺度匹配
    2. Global Registration - 全局配准
    3. Point-to-Plane ICP - 精细配准
    """

    def __init__(self, max_iterations: int = 100, tolerance: float = 1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance

    def scale_matching(
            self,
            source: np.ndarray,
            target: np.ndarray,
            reference_idx: Optional[int] = None
    ) -> Tuple[np.ndarray, float]:
        """
        尺度匹配

        Args:
            source: 源点云 [N, 3]
            target: 目标点云 [N, 3]
            reference_idx: 用于计算尺度因子的参考点索引

        Returns:
            scaled_target: 缩放后的目标点云
            scale_factor: 缩放因子
        """
        if reference_idx is None:
            # 使用所有点的平均距离
            source_dists = np.linalg.norm(source, axis=1)
            target_dists = np.linalg.norm(target, axis=1)
            scale_factor = np.mean(source_dists) / (np.mean(target_dists) + 1e-8)
        else:
            # 使用特定参考点
            source_dist = np.linalg.norm(source[reference_idx])
            target_dist = np.linalg.norm(target[reference_idx])
            scale_factor = source_dist / (target_dist + 1e-8)

        scaled_target = target * scale_factor
        return scaled_target, scale_factor

    def global_registration(
            self,
            source: np.ndarray,
            target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        全局配准 - 使用Procrustes分析

        Args:
            source: 源点云 [N, 3]
            target: 目标点云 [N, 3]

        Returns:
            R: 旋转矩阵 [3, 3]
            t: 平移向量 [3,]
            transformed_target: 变换后的目标点云
        """
        # 中心化
        source_centroid = np.mean(source, axis=0)
        target_centroid = np.mean(target, axis=0)

        source_centered = source - source_centroid
        target_centered = target - target_centroid

        # SVD求解最优旋转
        H = target_centered.T @ source_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # 确保正交
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        # 计算平移
        t = source_centroid - R @ target_centroid

        # 变换目标点云
        transformed_target = (R @ target.T).T + t

        return R, t, transformed_target

    def point_to_plane_icp(
            self,
            source: np.ndarray,
            target: np.ndarray,
            initial_R: Optional[np.ndarray] = None,
            initial_t: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Point-to-Plane ICP精细配准

        Args:
            source: 源点云 [N, 3]
            target: 目标点云 [N, 3]
            initial_R: 初始旋转矩阵
            initial_t: 初始平移向量

        Returns:
            R: 最终旋转矩阵
            t: 最终平移向量
            transformed_target: 变换后的目标点云
            rmse: 最终RMSE
        """
        if initial_R is None:
            initial_R = np.eye(3)
        if initial_t is None:
            initial_t = np.zeros(3)

        R = initial_R.copy()
        t = initial_t.copy()
        transformed = (R @ target.T).T + t

        prev_rmse = np.inf

        for iteration in range(self.max_iterations):
            # 计算点到点距离
            diff = source - transformed
            distances = np.linalg.norm(diff, axis=1)
            rmse = np.sqrt(np.mean(distances ** 2))

            # 检查收敛
            if abs(prev_rmse - rmse) < self.tolerance:
                break
            prev_rmse = rmse

            # 估计法向量 (简化：使用相邻点近似)
            normals = self._estimate_normals(source)

            # 构建线性系统 (Point-to-Plane)
            A = []
            b = []
            for i in range(len(source)):
                n = normals[i]
                p = transformed[i]
                q = source[i]

                # 线性化旋转
                cross_prod = np.cross(p, n)
                A.append(np.concatenate([cross_prod, n]))
                b.append(np.dot(q - p, n))

            A = np.array(A)
            b = np.array(b)

            # 求解
            try:
                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

                # 提取旋转和平移增量
                alpha, beta, gamma = x[:3]
                delta_t = x[3:]

                # 更新变换
                dR = Rotation.from_rotvec([alpha, beta, gamma]).as_matrix()
                R = dR @ R
                t = dR @ t + delta_t

                transformed = (R @ target.T).T + t
            except:
                break

        return R, t, transformed, rmse

    def _estimate_normals(self, points: np.ndarray, k: int = 5) -> np.ndarray:
        """估计点云法向量（简化版本）"""
        n_points = len(points)
        normals = np.zeros_like(points)

        for i in range(n_points):
            # 找k个最近邻
            dists = np.linalg.norm(points - points[i], axis=1)
            neighbors_idx = np.argsort(dists)[1:k + 1]
            neighbors = points[neighbors_idx]

            # PCA估计法向量
            centered = neighbors - points[i]
            cov = centered.T @ centered
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            normals[i] = eigenvectors[:, 0]  # 最小特征值对应的特征向量

        return normals

    def register(
            self,
            source: np.ndarray,
            target: np.ndarray
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        完整配准流程

        Args:
            source: 源点云 (参考)
            target: 目标点云 (待配准)

        Returns:
            transformation_matrix: 4x4变换矩阵
            rmse: 配准误差
            transformed_target: 配准后的目标点云
        """
        best_rmse = np.inf
        best_transform = np.eye(4)
        best_transformed = target.copy()

        # 尝试不同参考点进行配准
        n_points = len(source)
        test_indices = [None] + list(range(0, n_points, max(1, n_points // 10)))

        for ref_idx in test_indices:
            # 1. 尺度匹配
            scaled_target, scale = self.scale_matching(source, target, ref_idx)

            # 2. 全局配准
            R_global, t_global, transformed_global = self.global_registration(
                source, scaled_target
            )

            # 3. Point-to-Plane ICP
            R_final, t_final, transformed_final, rmse = self.point_to_plane_icp(
                source, transformed_global, R_global, t_global
            )

            if rmse < best_rmse:
                best_rmse = rmse
                best_transformed = transformed_final

                # 构建4x4变换矩阵
                T = np.eye(4)
                T[:3, :3] = R_final * scale
                T[:3, 3] = t_final
                best_transform = T

        return best_transform, best_rmse, best_transformed


# ==================== 对称性计算方法 ====================

class SymmetryAnalyzer:
    """
    对称性分析器 - 实现Kim et al.的三种数值方法
    """

    def __init__(self):
        self.registration = RegistrationMethod()

    def compute_midsagittal_plane(
            self,
            landmarks_3d: np.ndarray,
            left_iris_idx: int = LEFT_IRIS_CENTER,
            right_iris_idx: int = RIGHT_IRIS_CENTER
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算面部中矢面

        基于双眼虹膜中心定义中矢面：
        - 法向量：连接双眼的向量
        - 过点：双眼中点

        Returns:
            normal: 法向量 [3,]
            point: 平面上的点 [3,]
        """
        # 使用内眦点（更稳定）
        left_point = landmarks_3d[EYE_INNER_CANTHUS_LEFT]
        right_point = landmarks_3d[EYE_INNER_CANTHUS_RIGHT]

        # 连接双眼的向量即为法向量
        normal = right_point - left_point
        normal = normal / (np.linalg.norm(normal) + 1e-8)

        # 中点
        midpoint = (left_point + right_point) / 2

        return normal, midpoint

    def compute_distance_symmetry(
            self,
            left_coords: np.ndarray,
            right_coords: np.ndarray,
            midsagittal_normal: np.ndarray
    ) -> np.ndarray:
        """
        计算距离对称性 (Kim et al. 公式4)

        将左侧点通过中矢面镜像，计算与右侧点的距离

        Args:
            left_coords: 左侧点坐标 [N, 3]
            right_coords: 右侧点坐标 [N, 3]
            midsagittal_normal: 中矢面法向量

        Returns:
            distances: 镜像点与右侧点的距离 [N,]
        """
        # 镜像左侧点
        # 镜像公式: p' = p - 2 * (p · n) * n
        n = midsagittal_normal
        dots = np.dot(left_coords, n)
        left_mirrored = left_coords - 2 * np.outer(dots, n)

        # 计算镜像点与右侧点的距离
        distances = np.linalg.norm(left_mirrored - right_coords, axis=1)

        return distances

    def compute_angle_symmetry(
            self,
            left_coords: np.ndarray,
            right_coords: np.ndarray,
            midsagittal_normal: np.ndarray
    ) -> np.ndarray:
        """
        计算角度对称性 (Kim et al. 公式6)

        使用余弦相似度衡量点对向量与中矢面法向量的一致性
        完美对称时，点对向量应与法向量平行

        Args:
            left_coords: 左侧点坐标 [N, 3]
            right_coords: 右侧点坐标 [N, 3]
            midsagittal_normal: 中矢面法向量

        Returns:
            cosine_similarities: 余弦相似度 [N,]，1表示完美对称
        """
        # 点对向量
        pair_vectors = right_coords - left_coords

        # 归一化
        norms = np.linalg.norm(pair_vectors, axis=1, keepdims=True) + 1e-8
        pair_vectors_normalized = pair_vectors / norms

        # 余弦相似度
        cosine_similarities = np.abs(np.dot(pair_vectors_normalized, midsagittal_normal))

        return cosine_similarities

    def compute_landmark_movements(
            self,
            neutral_left: np.ndarray,
            neutral_right: np.ndarray,
            expression_left: np.ndarray,
            expression_right: np.ndarray
    ) -> LandmarkMovement:
        """
        计算地标运动量 (Kim et al. 公式7-8)

        比较中性表情和目标表情之间的地标运动

        Args:
            neutral_left: 中性表情左侧点
            neutral_right: 中性表情右侧点
            expression_left: 目标表情左侧点
            expression_right: 目标表情右侧点

        Returns:
            LandmarkMovement: 运动量分析结果
        """
        # 配准中性和表情数据
        # 首先配准左侧
        neutral_combined = np.vstack([neutral_left, neutral_right])
        expression_combined = np.vstack([expression_left, expression_right])

        _, _, expression_registered = self.registration.register(
            neutral_combined, expression_combined
        )

        n_points = len(neutral_left)
        expression_left_reg = expression_registered[:n_points]
        expression_right_reg = expression_registered[n_points:]

        # 计算运动量
        left_movements = np.linalg.norm(expression_left_reg - neutral_left, axis=1)
        right_movements = np.linalg.norm(expression_right_reg - neutral_right, axis=1)

        # 运动不对称性
        movement_asymmetry = np.abs(left_movements - right_movements)

        return LandmarkMovement(
            left_movements=left_movements,
            right_movements=right_movements,
            movement_asymmetry=movement_asymmetry
        )

    def compute_full_metrics(
            self,
            left_coords_3d: np.ndarray,
            right_coords_3d: np.ndarray,
            rotation_matrix: Optional[np.ndarray] = None,
            normalization_dist: float = 1.0
    ) -> SymmetryMetrics:
        """
        计算完整的对称性指标

        Args:
            left_coords_3d: 左侧3D坐标 [N, 3]
            right_coords_3d: 右侧3D坐标 [N, 3]
            rotation_matrix: 头部旋转矩阵（用于姿态校正）
            normalization_dist: 归一化距离（内眦间距）

        Returns:
            SymmetryMetrics: 完整的对称性指标
        """
        # 姿态校正
        if rotation_matrix is not None:
            left_aligned = (rotation_matrix.T @ left_coords_3d.T).T
            right_aligned = (rotation_matrix.T @ right_coords_3d.T).T
        else:
            left_aligned = left_coords_3d
            right_aligned = right_coords_3d

        # 计算中矢面
        # 简化：假设校正后X轴是左右方向
        midsagittal_normal = np.array([1.0, 0.0, 0.0])

        # 1. 距离对称性
        distance_symmetry = self.compute_distance_symmetry(
            left_aligned, right_aligned, midsagittal_normal
        )

        # 2. 角度对称性
        angle_symmetry = self.compute_angle_symmetry(
            left_aligned, right_aligned, midsagittal_normal
        )

        # 3. 镜像偏差
        left_mirrored = left_aligned.copy()
        left_mirrored[:, 0] = -left_mirrored[:, 0]
        mirror_deviation = np.linalg.norm(left_mirrored - right_aligned, axis=1)

        # 4. 中轴面距离差
        midplane_asymmetry = np.abs(np.abs(left_aligned[:, 0]) - np.abs(right_aligned[:, 0]))

        # 5. Y轴差异
        y_diff_abs = np.abs(left_aligned[:, 1] - right_aligned[:, 1])

        # 6. 3D欧几里得距离
        euclidean_3d = np.linalg.norm(left_aligned - right_aligned, axis=1)

        # 归一化
        scale = normalization_dist + 1e-8

        return SymmetryMetrics(
            distance_symmetry=distance_symmetry / scale,
            angle_symmetry=angle_symmetry,
            midplane_asymmetry=midplane_asymmetry / scale,
            euclidean_3d=euclidean_3d / scale,
            mirror_deviation=mirror_deviation / scale,
            y_diff_abs=y_diff_abs / scale,
            normalization_distance=normalization_dist
        )


# ==================== 临床分级计算 ====================

class ClinicalGrading:
    """
    临床分级计算

    基于对称性指标计算:
    - House-Brackmann 分级
    - Sunnybrook 评分
    - eFACE 评分
    """

    def __init__(self):
        # HB分级阈值（基于文献和经验）
        self.hb_thresholds = {
            HBGrade.GRADE_1: 0.0,  # 正常
            HBGrade.GRADE_2: 0.05,  # 轻度
            HBGrade.GRADE_3: 0.15,  # 中度
            HBGrade.GRADE_4: 0.30,  # 中重度
            HBGrade.GRADE_5: 0.50,  # 重度
            HBGrade.GRADE_6: 0.70,  # 完全麻痹
        }

    def compute_hb_grade(
            self,
            branch_scores: Dict[FacialNerveBranch, float],
            action_scores: Dict[str, float]
    ) -> Tuple[HBGrade, float]:
        """
        计算House-Brackmann分级

        HB分级标准:
        - Grade I: 正常功能
        - Grade II: 轻度功能障碍
        - Grade III: 中度功能障碍
        - Grade IV: 中重度功能障碍
        - Grade V: 重度功能障碍
        - Grade VI: 完全麻痹

        Args:
            branch_scores: 各分支的不对称评分
            action_scores: 各动作的不对称评分

        Returns:
            hb_grade: HB分级
            confidence: 置信度
        """
        if not branch_scores and not action_scores:
            return HBGrade.GRADE_1, 0.0

        # 综合评分
        all_scores = list(branch_scores.values()) + list(action_scores.values())

        # 使用最大值和平均值的加权组合
        max_score = max(all_scores) if all_scores else 0
        mean_score = np.mean(all_scores) if all_scores else 0

        # 加权综合 (最大值权重更高，因为单侧功能障碍也应反映)
        composite_score = 0.6 * max_score + 0.4 * mean_score

        # 根据阈值确定分级
        grade = HBGrade.GRADE_1
        for g in [HBGrade.GRADE_6, HBGrade.GRADE_5, HBGrade.GRADE_4,
                  HBGrade.GRADE_3, HBGrade.GRADE_2]:
            if composite_score >= self.hb_thresholds[g]:
                grade = g
                break

        # 计算置信度（基于分数与阈值的距离）
        if grade == HBGrade.GRADE_6:
            confidence = min(1.0, composite_score / self.hb_thresholds[HBGrade.GRADE_6])
        else:
            lower_th = self.hb_thresholds[grade]
            next_grade = HBGrade(grade.value + 1) if grade.value < 6 else grade
            upper_th = self.hb_thresholds[next_grade]

            range_size = upper_th - lower_th
            position = composite_score - lower_th
            confidence = 1.0 - abs(position / range_size - 0.5) * 2

        return grade, confidence

    def compute_sunnybrook_score(
            self,
            resting_symmetry: float,
            voluntary_movement: Dict[str, float],
            synkinesis: Dict[str, float]
    ) -> float:
        """
        计算Sunnybrook评分

        Sunnybrook评分 = 自主运动评分 - 静息对称性评分 - 联动评分
        范围: 0-100，越高越好

        Args:
            resting_symmetry: 静息对称性评分 (0-20)
            voluntary_movement: 各动作的自主运动评分
            synkinesis: 联动评分

        Returns:
            sunnybrook_score: Sunnybrook总评分
        """
        # 静息对称性评分 (0-20，0最好)
        resting_score = min(20, resting_symmetry * 20)

        # 自主运动评分 (0-100)
        if voluntary_movement:
            # 将不对称度转换为运动能力评分
            movement_scores = []
            for action, asymmetry in voluntary_movement.items():
                # 不对称度越低，运动能力越强
                movement_score = max(0, 1.0 - asymmetry) * 20
                movement_scores.append(movement_score)
            voluntary_score = np.mean(movement_scores) * 5  # 缩放到0-100
        else:
            voluntary_score = 100

        # 联动评分 (0-15)
        if synkinesis:
            synkinesis_score = min(15, np.mean(list(synkinesis.values())) * 15)
        else:
            synkinesis_score = 0

        # 综合评分
        sunnybrook = voluntary_score - resting_score - synkinesis_score

        return max(0, min(100, sunnybrook))

    def compute_eface_score(
            self,
            static_symmetry: Dict[str, float],
            dynamic_symmetry: Dict[str, float],
            synkinesis: Dict[str, float]
    ) -> float:
        """
        计算eFACE评分

        eFACE: 基于静态和动态面部特征的综合评分
        范围: 0-100，越高越好

        Args:
            static_symmetry: 静态对称性指标
            dynamic_symmetry: 动态对称性指标
            synkinesis: 联动指标

        Returns:
            eface_score: eFACE总评分
        """
        scores = []

        # 静态评分 (权重30%)
        if static_symmetry:
            static_score = 100 * (1 - np.mean(list(static_symmetry.values())))
            scores.append(0.3 * static_score)

        # 动态评分 (权重50%)
        if dynamic_symmetry:
            dynamic_score = 100 * (1 - np.mean(list(dynamic_symmetry.values())))
            scores.append(0.5 * dynamic_score)

        # 联动扣分 (权重20%)
        if synkinesis:
            synkinesis_penalty = np.mean(list(synkinesis.values())) * 100
            scores.append(0.2 * (100 - synkinesis_penalty))

        if not scores:
            return 100.0

        return max(0, min(100, sum(scores) / (0.3 + 0.5 + 0.2)))


# ==================== 可视化方法 ====================

class FacialPalsyVisualizer:
    """面部麻痹可视化器"""

    def __init__(self):
        # 颜色映射
        self.severity_cmap = plt.cm.RdYlGn_r  # 红黄绿反转
        self.branch_colors = {
            FacialNerveBranch.TEMPORAL: '#FF6B6B',
            FacialNerveBranch.ZYGOMATIC: '#4ECDC4',
            FacialNerveBranch.BUCCAL: '#45B7D1',
            FacialNerveBranch.MARGINAL_MANDIBULAR: '#96CEB4',
            FacialNerveBranch.CERVICAL: '#FFEAA7',
        }

    def draw_symmetry_heatmap(
            self,
            image: np.ndarray,
            landmarks,
            metrics: SymmetryMetrics,
            muscle_groups: Dict,
            output_path: str
    ):
        """
        绘制对称性热图

        在面部图像上用颜色编码显示各区域的不对称程度
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        h, w = image.shape[:2]

        # 原图 + 标注
        ax1 = axes[0]
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax1.set_title('原始图像 + 地标', fontsize=12)
        ax1.axis('off')

        # 绘制所有对称点对
        for muscle_name, config in muscle_groups.items():
            pairs = config['pairs']
            branch = config.get('branch')
            color = self.branch_colors.get(branch, '#888888') if branch else '#888888'

            left_indices = pairs['left']
            right_indices = pairs['right']

            for li, ri in zip(left_indices, right_indices):
                if li >= len(landmarks) or ri >= len(landmarks):
                    continue
                lm_l = landmarks[li]
                lm_r = landmarks[ri]

                lx, ly = int(lm_l.x * w), int(lm_l.y * h)
                rx, ry = int(lm_r.x * w), int(lm_r.y * h)

                ax1.plot([lx, rx], [ly, ry], color=color, alpha=0.6, linewidth=0.5)
                ax1.scatter([lx], [ly], c='red', s=3, zorder=5)
                ax1.scatter([rx], [ry], c='blue', s=3, zorder=5)

        # 距离对称性热图
        ax2 = axes[1]
        ax2.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax2.set_title('距离对称性热图', fontsize=12)
        ax2.axis('off')

        # 用颜色编码连线
        max_asym = max(metrics.mirror_deviation.max(), 0.01)
        idx = 0
        for muscle_name, config in muscle_groups.items():
            pairs = config['pairs']
            left_indices = pairs['left']
            right_indices = pairs['right']

            for li, ri in zip(left_indices, right_indices):
                if li >= len(landmarks) or ri >= len(landmarks) or idx >= len(metrics.mirror_deviation):
                    idx += 1
                    continue

                lm_l = landmarks[li]
                lm_r = landmarks[ri]

                lx, ly = int(lm_l.x * w), int(lm_l.y * h)
                rx, ry = int(lm_r.x * w), int(lm_r.y * h)

                asym_ratio = min(1.0, metrics.mirror_deviation[idx] / max_asym)
                color = self.severity_cmap(asym_ratio)

                ax2.plot([lx, rx], [ly, ry], color=color, linewidth=2, alpha=0.8)
                idx += 1

        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap=self.severity_cmap,
                                   norm=plt.Normalize(vmin=0, vmax=max_asym))
        plt.colorbar(sm, ax=ax2, label='不对称度', shrink=0.8)

        # 角度对称性热图
        ax3 = axes[2]
        ax3.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax3.set_title('角度对称性热图', fontsize=12)
        ax3.axis('off')

        idx = 0
        for muscle_name, config in muscle_groups.items():
            pairs = config['pairs']
            left_indices = pairs['left']
            right_indices = pairs['right']

            for li, ri in zip(left_indices, right_indices):
                if li >= len(landmarks) or ri >= len(landmarks) or idx >= len(metrics.angle_symmetry):
                    idx += 1
                    continue

                lm_l = landmarks[li]
                lm_r = landmarks[ri]

                lx, ly = int(lm_l.x * w), int(lm_l.y * h)
                rx, ry = int(lm_r.x * w), int(lm_r.y * h)

                # 角度对称性：1是完美对称，越小越不对称
                asym_ratio = 1.0 - metrics.angle_symmetry[idx]
                color = self.severity_cmap(asym_ratio)

                ax3.plot([lx, rx], [ly, ry], color=color, linewidth=2, alpha=0.8)
                idx += 1

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def draw_branch_analysis(
            self,
            branch_metrics: Dict[FacialNerveBranch, BranchAnalysis],
            output_path: str
    ):
        """
        绘制面神经分支分析图

        显示各分支的功能状态
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 柱状图
        ax1 = axes[0]
        branches = list(branch_metrics.keys())
        scores = [branch_metrics[b].severity_score for b in branches]
        colors = [self.branch_colors.get(b, '#888888') for b in branches]

        bars = ax1.bar([b.value for b in branches], scores, color=colors, alpha=0.8)
        ax1.set_xlabel('面神经分支')
        ax1.set_ylabel('不对称评分')
        ax1.set_title('各面神经分支功能状态')
        ax1.set_ylim(0, 1)

        # 添加阈值线
        ax1.axhline(y=TH_NORMAL, color='green', linestyle='--', alpha=0.5, label='正常阈值')
        ax1.axhline(y=TH_MILD, color='yellow', linestyle='--', alpha=0.5, label='轻度阈值')
        ax1.axhline(y=TH_MODERATE, color='orange', linestyle='--', alpha=0.5, label='中度阈值')
        ax1.axhline(y=TH_SEVERE, color='red', linestyle='--', alpha=0.5, label='重度阈值')
        ax1.legend(loc='upper right', fontsize=8)

        plt.xticks(rotation=45, ha='right')

        # 雷达图
        ax2 = fig.add_subplot(122, projection='polar')

        n_branches = len(branches)
        angles = np.linspace(0, 2 * np.pi, n_branches, endpoint=False).tolist()
        angles += angles[:1]

        scores_radar = scores + [scores[0]]

        ax2.plot(angles, scores_radar, 'o-', linewidth=2, color='#FF6B6B')
        ax2.fill(angles, scores_radar, alpha=0.25, color='#FF6B6B')

        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels([b.value for b in branches])
        ax2.set_ylim(0, 1)
        ax2.set_title('分支功能雷达图')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def draw_clinical_report(
            self,
            diagnosis: SessionDiagnosis,
            output_path: str
    ):
        """
        绘制临床诊断报告
        """
        fig = plt.figure(figsize=(16, 12))

        # 标题
        fig.suptitle(f'面部麻痹分析报告\n患者ID: {diagnosis.patient_id or "N/A"} | '
                     f'检查ID: {diagnosis.examination_id}',
                     fontsize=14, fontweight='bold')

        # 1. 诊断摘要
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.axis('off')

        summary_text = f"""
诊断结果
─────────────────
是否面瘫: {'是' if diagnosis.has_palsy else '否'}
面瘫侧别: {diagnosis.palsy_side.value}

临床分级
─────────────────
HB分级: Grade {diagnosis.hb_grade.value}
Sunnybrook: {diagnosis.sunnybrook_score:.1f}
eFACE: {diagnosis.eface_score:.1f}

置信度: {diagnosis.confidence:.1%}
        """
        ax1.text(0.1, 0.9, summary_text, transform=ax1.transAxes,
                 fontsize=11, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 2. 分支状态柱状图
        ax2 = fig.add_subplot(2, 3, 2)
        if diagnosis.branch_status:
            branches = list(diagnosis.branch_status.keys())
            scores = [diagnosis.branch_status[b] for b in branches]
            colors = [self.branch_colors.get(b, '#888888') for b in branches]

            ax2.barh([b.value for b in branches], scores, color=colors, alpha=0.8)
            ax2.set_xlabel('功能损伤程度')
            ax2.set_title('各分支功能状态')
            ax2.set_xlim(0, 1)
            ax2.axvline(x=TH_MILD, color='yellow', linestyle='--', alpha=0.7)
            ax2.axvline(x=TH_MODERATE, color='orange', linestyle='--', alpha=0.7)

        # 3. 动作分析
        ax3 = fig.add_subplot(2, 3, 3)
        if diagnosis.action_analyses:
            actions = list(diagnosis.action_analyses.keys())
            asymmetries = [diagnosis.action_analyses[a].overall_asymmetry for a in actions]

            ax3.barh(actions, asymmetries, color='#45B7D1', alpha=0.8)
            ax3.set_xlabel('不对称度')
            ax3.set_title('各动作分析')
            ax3.set_xlim(0, max(asymmetries) * 1.2 if asymmetries else 1)

        # 4. HB分级图解
        ax4 = fig.add_subplot(2, 3, 4)
        grades = [1, 2, 3, 4, 5, 6]
        grade_names = ['I\n正常', 'II\n轻度', 'III\n中度', 'IV\n中重度', 'V\n重度', 'VI\n完全']
        grade_colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c', '#9b59b6', '#34495e']

        bars = ax4.bar(grades, [1] * 6, color=grade_colors, alpha=0.5)
        bars[diagnosis.hb_grade.value - 1].set_alpha(1.0)
        bars[diagnosis.hb_grade.value - 1].set_edgecolor('black')
        bars[diagnosis.hb_grade.value - 1].set_linewidth(3)

        ax4.set_xticks(grades)
        ax4.set_xticklabels(grade_names)
        ax4.set_title('House-Brackmann分级')
        ax4.set_ylim(0, 1.5)
        ax4.set_yticks([])

        # 5. Sunnybrook评分仪表盘
        ax5 = fig.add_subplot(2, 3, 5)
        self._draw_gauge(ax5, diagnosis.sunnybrook_score, 'Sunnybrook', max_val=100)

        # 6. eFACE评分仪表盘
        ax6 = fig.add_subplot(2, 3, 6)
        self._draw_gauge(ax6, diagnosis.eface_score, 'eFACE', max_val=100)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _draw_gauge(self, ax, value, title, max_val=100):
        """绘制仪表盘"""
        ax.set_aspect('equal')

        # 背景圆弧
        theta = np.linspace(0, np.pi, 100)
        r = 1
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        ax.fill_between(x, 0, y, alpha=0.2, color='gray')

        # 根据值填充
        fill_ratio = value / max_val
        fill_theta = np.linspace(0, np.pi * fill_ratio, 100)
        fill_x = r * np.cos(fill_theta)
        fill_y = r * np.sin(fill_theta)

        # 颜色根据值变化
        if fill_ratio > 0.7:
            color = '#2ecc71'  # 绿色
        elif fill_ratio > 0.4:
            color = '#f1c40f'  # 黄色
        else:
            color = '#e74c3c'  # 红色

        ax.fill_between(fill_x, 0, fill_y, alpha=0.7, color=color)

        # 添加文字
        ax.text(0, 0.4, f'{value:.0f}', ha='center', va='center', fontsize=24, fontweight='bold')
        ax.text(0, -0.1, title, ha='center', va='center', fontsize=12)

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.3, 1.2)
        ax.axis('off')


# ==================== 主分析器类 ====================

class ComprehensiveFacialPalsyAnalyzer:
    """
    综合面部麻痹分析器

    整合:
    1. Kim et al. (2022) 的3D配准和对称性方法
    2. Büchner et al. (2025) 的体积分析方法
    3. 临床分级系统 (HB, Sunnybrook, eFACE)
    """

    def __init__(
            self,
            db_path: str,
            model_path: str,
            verbose: bool = True
    ):
        self.db_path = db_path
        self.model_path = model_path
        self.verbose = verbose

        # 初始化各组件
        self.symmetry_analyzer = SymmetryAnalyzer()
        self.clinical_grader = ClinicalGrading()
        self.visualizer = FacialPalsyVisualizer()
        self.registration = RegistrationMethod()

        # 构建点对映射
        self._build_feature_pairs()

        if self.verbose:
            print("=" * 60)
            print("综合面部麻痹分析系统初始化完成")
            print(f"  - 肌肉组数量: {len(FACIAL_MUSCLE_GROUPS)}")
            print(f"  - 面神经分支: {len(FacialNerveBranch)}")
            print(f"  - 特征点对数: {self.n_features}")
            print("=" * 60)

    def _build_feature_pairs(self):
        """构建特征点对和区域映射"""
        self.feature_pairs = []
        self.feature_names = []
        self.muscle_feature_indices = {}
        self.branch_feature_indices = {}

        for muscle_name, config in FACIAL_MUSCLE_GROUPS.items():
            pairs = config['pairs']
            branch = config.get('branch')

            left_indices = pairs['left']
            right_indices = pairs['right']

            if len(left_indices) != len(right_indices):
                # 对于中线单点，跳过
                continue

            self.muscle_feature_indices[muscle_name] = []

            if branch:
                if branch not in self.branch_feature_indices:
                    self.branch_feature_indices[branch] = []

            for i, (li, ri) in enumerate(zip(left_indices, right_indices)):
                self.feature_pairs.append((int(li), int(ri)))
                self.feature_names.append(f"{muscle_name}_{i + 1:02d}")

                idx = len(self.feature_pairs) - 1
                self.muscle_feature_indices[muscle_name].append(idx)

                if branch:
                    self.branch_feature_indices[branch].append(idx)

        self.n_features = len(self.feature_pairs)

    def _create_landmarker(self) -> vision.FaceLandmarker:
        """创建MediaPipe FaceLandmarker"""
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
        )
        return vision.FaceLandmarker.create_from_options(options)

    def _estimate_head_pose(
            self,
            face_landmarks,
            image_width: int,
            image_height: int
    ) -> HeadPose:
        """使用solvePnP估计头部姿态"""
        image_points = []
        for idx in PNP_INDICES:
            lm = face_landmarks[idx]
            image_points.append([lm.x * image_width, lm.y * image_height])
        image_points = np.array(image_points, dtype=np.float64)

        focal_length = image_width
        center = (image_width / 2.0, image_height / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1.0]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            GENERIC_FACE_3D,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return HeadPose(roll=0.0, pitch=0.0, yaw=0.0, rotation_matrix=None)

        rmat, _ = cv2.Rodrigues(rvec)
        proj_matrix = np.hstack((rmat, tvec))
        euler = cv2.decomposeProjectionMatrix(proj_matrix)[6]
        pitch, yaw, roll = [float(a) for a in euler]

        return HeadPose(
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            rotation_matrix=rmat,
            rvec=rvec,
            tvec=tvec,
        )

    def _extract_3d_coords(
            self,
            face_landmarks
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """提取3D坐标并计算归一化距离"""
        nose = face_landmarks[NOSE_TIP]
        origin = np.array([nose.x, nose.y, nose.z])

        left_coords = []
        right_coords = []

        for li, ri in self.feature_pairs:
            l = face_landmarks[li]
            r = face_landmarks[ri]
            left_coords.append([l.x - origin[0], l.y - origin[1], l.z - origin[2]])
            right_coords.append([r.x - origin[0], r.y - origin[1], r.z - origin[2]])

        left_np = np.array(left_coords)
        right_np = np.array(right_coords)

        # 计算内眦间距作为归一化距离
        l_inner = np.array([
            face_landmarks[EYE_INNER_CANTHUS_LEFT].x,
            face_landmarks[EYE_INNER_CANTHUS_LEFT].y,
            face_landmarks[EYE_INNER_CANTHUS_LEFT].z
        ])
        r_inner = np.array([
            face_landmarks[EYE_INNER_CANTHUS_RIGHT].x,
            face_landmarks[EYE_INNER_CANTHUS_RIGHT].y,
            face_landmarks[EYE_INNER_CANTHUS_RIGHT].z
        ])
        normalization_dist = np.linalg.norm(l_inner - r_inner)

        return left_np, right_np, normalization_dist

    def analyze_video(
            self,
            video_path: str,
            start_frame: Optional[int] = None,
            end_frame: Optional[int] = None,
            fps: Optional[float] = None,
            action_name: Optional[str] = None
    ) -> Optional[ActionAnalysis]:
        """
        分析单个视频

        Args:
            video_path: 视频路径
            start_frame: 起始帧
            end_frame: 结束帧
            fps: 帧率
            action_name: 动作名称

        Returns:
            ActionAnalysis: 动作分析结果
        """
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

        # 存储帧级数据
        metrics_list = []
        head_poses = []
        left_coords_all = []
        right_coords_all = []

        # 调试信息
        debug_frame = None
        debug_landmarks = None
        debug_metrics = None
        max_asymmetry = -1

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

                    # 估计头部姿态
                    head_pose = self._estimate_head_pose(face_landmarks, w, h)
                    head_poses.append(head_pose)

                    # 提取3D坐标
                    left_np, right_np, norm_dist = self._extract_3d_coords(face_landmarks)
                    left_coords_all.append(left_np)
                    right_coords_all.append(right_np)

                    # 计算对称性指标
                    metrics = self.symmetry_analyzer.compute_full_metrics(
                        left_np, right_np,
                        head_pose.rotation_matrix,
                        norm_dist
                    )
                    metrics_list.append(metrics)

                    # 记录最大不对称帧用于可视化
                    current_asym = np.sum(metrics.mirror_deviation)
                    if current_asym > max_asymmetry:
                        max_asymmetry = current_asym
                        debug_frame = frame.copy()
                        debug_landmarks = face_landmarks
                        debug_metrics = metrics

                frame_abs_idx += 1

        finally:
            cap.release()
            try:
                landmarker.close()
            except:
                pass

        if not metrics_list:
            return None

        # 计算肌肉组级别指标
        muscle_metrics = self._compute_muscle_metrics(metrics_list)

        # 计算分支级别指标
        branch_metrics = self._compute_branch_metrics(muscle_metrics)

        # 计算动作级别聚合
        overall_asymmetry = np.mean([m.mean_mirror_deviation for m in muscle_metrics.values()])

        # 判断受累侧
        affected_side = self._determine_affected_side(left_coords_all, right_coords_all)

        return ActionAnalysis(
            action_name=action_name or os.path.basename(video_path),
            frame_count=len(metrics_list),
            metrics_per_frame=metrics_list,
            head_poses=head_poses,
            muscle_metrics=muscle_metrics,
            branch_metrics=branch_metrics,
            overall_asymmetry=overall_asymmetry,
            affected_side=affected_side,
            severity_score=min(1.0, overall_asymmetry / TH_SEVERE)
        ), debug_frame, debug_landmarks, debug_metrics

    def _compute_muscle_metrics(
            self,
            metrics_list: List[SymmetryMetrics]
    ) -> Dict[str, MuscleGroupMetrics]:
        """计算肌肉组级别的指标"""
        muscle_metrics = {}

        for muscle_name, indices in self.muscle_feature_indices.items():
            if not indices:
                continue

            config = FACIAL_MUSCLE_GROUPS.get(muscle_name, {})
            branch = config.get('branch')

            # 提取该肌肉组的指标
            distance_syms = []
            angle_syms = []
            mirror_devs = []

            for metrics in metrics_list:
                for idx in indices:
                    if idx < len(metrics.distance_symmetry):
                        distance_syms.append(metrics.distance_symmetry[idx])
                        angle_syms.append(metrics.angle_symmetry[idx])
                        mirror_devs.append(metrics.mirror_deviation[idx])

            if not mirror_devs:
                continue

            mean_dist_sym = np.mean(distance_syms)
            mean_angle_sym = np.mean(angle_syms)
            mean_mirror_dev = np.mean(mirror_devs)

            # 计算严重度评分
            severity = min(1.0, mean_mirror_dev / TH_SEVERE)

            muscle_metrics[muscle_name] = MuscleGroupMetrics(
                muscle_name=muscle_name,
                branch=branch,
                mean_distance_symmetry=mean_dist_sym,
                mean_angle_symmetry=mean_angle_sym,
                mean_mirror_deviation=mean_mirror_dev,
                severity_score=severity
            )

        return muscle_metrics

    def _compute_branch_metrics(
            self,
            muscle_metrics: Dict[str, MuscleGroupMetrics]
    ) -> Dict[FacialNerveBranch, BranchAnalysis]:
        """计算分支级别的指标"""
        branch_metrics = {}

        for branch in FacialNerveBranch:
            # 收集该分支的所有肌肉指标
            branch_muscles = [
                mm for mm in muscle_metrics.values()
                if mm.branch == branch
            ]

            if not branch_muscles:
                continue

            mean_asymmetry = np.mean([m.mean_mirror_deviation for m in branch_muscles])
            severity = np.mean([m.severity_score for m in branch_muscles])

            branch_metrics[branch] = BranchAnalysis(
                branch=branch,
                muscle_metrics=branch_muscles,
                mean_asymmetry=mean_asymmetry,
                severity_score=severity
            )

        return branch_metrics

    def _determine_affected_side(
            self,
            left_coords_all: List[np.ndarray],
            right_coords_all: List[np.ndarray]
    ) -> PalsySide:
        """
        判断受累侧

        通过比较左右两侧的运动幅度来判断
        运动幅度较小的一侧为受累侧
        """
        if not left_coords_all or not right_coords_all:
            return PalsySide.NONE

        left_array = np.array(left_coords_all)
        right_array = np.array(right_coords_all)

        # 计算时间维度的运动幅度（标准差）
        left_movement = np.std(left_array, axis=0).mean()
        right_movement = np.std(right_array, axis=0).mean()

        # 计算运动不对称比例
        total = left_movement + right_movement + 1e-8
        left_ratio = left_movement / total
        right_ratio = right_movement / total

        # 如果差异不明显，认为无面瘫或双侧
        diff = abs(left_ratio - right_ratio)

        if diff < 0.1:
            return PalsySide.NONE
        elif left_ratio < right_ratio:
            return PalsySide.LEFT
        else:
            return PalsySide.RIGHT

    def compute_session_diagnosis(
            self,
            action_analyses: Dict[str, ActionAnalysis]
    ) -> SessionDiagnosis:
        """
        计算检查级诊断

        基于所有动作的分析结果，得出综合诊断
        """
        if not action_analyses:
            return SessionDiagnosis(
                patient_id=None,
                examination_id=0,
                action_analyses={}
            )

        # 收集所有分支的评分
        branch_scores = {}
        for branch in FacialNerveBranch:
            branch_asymmetries = []
            for analysis in action_analyses.values():
                if branch in analysis.branch_metrics:
                    branch_asymmetries.append(
                        analysis.branch_metrics[branch].mean_asymmetry
                    )
            if branch_asymmetries:
                branch_scores[branch] = np.mean(branch_asymmetries)

        # 收集动作评分
        action_scores = {
            name: analysis.overall_asymmetry
            for name, analysis in action_analyses.items()
        }

        # 计算HB分级
        hb_grade, confidence = self.clinical_grader.compute_hb_grade(
            branch_scores, action_scores
        )

        # 判断是否面瘫
        has_palsy = hb_grade.value > 1

        # 判断面瘫侧别
        side_votes = [a.affected_side for a in action_analyses.values()
                      if a.affected_side != PalsySide.NONE]
        if side_votes:
            from collections import Counter
            palsy_side = Counter(side_votes).most_common(1)[0][0]
        else:
            palsy_side = PalsySide.NONE

        # 计算Sunnybrook和eFACE
        # 简化：使用动作不对称度作为代理
        resting_sym = action_scores.get('Neutral', 0.0) if 'Neutral' in action_scores else 0.0
        sunnybrook = self.clinical_grader.compute_sunnybrook_score(
            resting_sym, action_scores, {}
        )

        eface = self.clinical_grader.compute_eface_score(
            {'resting': resting_sym},
            action_scores,
            {}
        )

        return SessionDiagnosis(
            patient_id=None,
            examination_id=0,
            action_analyses=action_analyses,
            has_palsy=has_palsy,
            palsy_side=palsy_side,
            hb_grade=hb_grade,
            sunnybrook_score=sunnybrook,
            eface_score=eface,
            branch_status=branch_scores,
            confidence=confidence
        )

    def analyze_and_visualize(
            self,
            video_path: str,
            output_dir: str,
            examination_id: int = 0,
            action_name: Optional[str] = None,
            start_frame: Optional[int] = None,
            end_frame: Optional[int] = None,
            fps: Optional[float] = None,
    ) -> Dict:
        """
        分析视频并生成可视化结果

        Args:
            video_path: 视频路径
            output_dir: 输出目录
            examination_id: 检查ID
            action_name: 动作名称
            start_frame: 起始帧
            end_frame: 结束帧
            fps: 帧率

        Returns:
            结果字典
        """
        os.makedirs(output_dir, exist_ok=True)

        # 构建文件名前缀
        safe_action = re.sub(r'[^0-9A-Za-z._-]+', '_', action_name or 'unknown')[:50]
        prefix = f"{examination_id}_{safe_action}"

        if self.verbose:
            print(f"\n分析视频: {video_path}")
            print(f"  动作: {action_name}")

        # 分析视频
        result = self.analyze_video(
            video_path, start_frame, end_frame, fps, action_name
        )

        if result is None:
            return {"status": "failed", "error": "无法检测到人脸"}

        analysis, debug_frame, debug_landmarks, debug_metrics = result

        # 生成可视化
        if debug_frame is not None and debug_landmarks is not None:
            # 1. 对称性热图
            heatmap_path = os.path.join(output_dir, f"{prefix}_symmetry_heatmap.png")
            self.visualizer.draw_symmetry_heatmap(
                debug_frame, debug_landmarks, debug_metrics,
                FACIAL_MUSCLE_GROUPS, heatmap_path
            )

        # 2. 分支分析图
        if analysis.branch_metrics:
            branch_path = os.path.join(output_dir, f"{prefix}_branch_analysis.png")
            self.visualizer.draw_branch_analysis(analysis.branch_metrics, branch_path)

        # 3. 保存指标JSON
        metrics_path = os.path.join(output_dir, f"{prefix}_metrics.json")
        metrics_dict = {
            "action_name": analysis.action_name,
            "frame_count": analysis.frame_count,
            "overall_asymmetry": float(analysis.overall_asymmetry),
            "affected_side": analysis.affected_side.value if analysis.affected_side else "None",
            "severity_score": float(analysis.severity_score),
            "muscle_metrics": {
                name: {
                    "mean_distance_symmetry": float(m.mean_distance_symmetry),
                    "mean_angle_symmetry": float(m.mean_angle_symmetry),
                    "mean_mirror_deviation": float(m.mean_mirror_deviation),
                    "severity_score": float(m.severity_score),
                    "branch": m.branch.value if m.branch else None
                }
                for name, m in analysis.muscle_metrics.items()
            },
            "branch_metrics": {
                b.value: {
                    "mean_asymmetry": float(bm.mean_asymmetry),
                    "severity_score": float(bm.severity_score)
                }
                for b, bm in analysis.branch_metrics.items()
            }
        }

        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, ensure_ascii=False, indent=2)

        if self.verbose:
            print(f"  ✅ 分析完成")
            print(f"     - 总帧数: {analysis.frame_count}")
            print(f"     - 整体不对称度: {analysis.overall_asymmetry:.4f}")
            print(f"     - 受累侧: {analysis.affected_side.value if analysis.affected_side else 'None'}")
            print(f"     - 严重度评分: {analysis.severity_score:.4f}")

        return {
            "status": "success",
            "analysis": analysis,
            "output_files": {
                "heatmap": heatmap_path if debug_frame is not None else None,
                "branch_analysis": branch_path if analysis.branch_metrics else None,
                "metrics": metrics_path
            }
        }


# ==================== 批量处理和数据库交互 ====================

def batch_process_database(
        analyzer: ComprehensiveFacialPalsyAnalyzer,
        output_dir: str,
        limit: Optional[int] = None,
        action_filter: Optional[List[str]] = None,
        use_multiprocessing: bool = False,  # 暂时禁用多进程，简化调试
        num_workers: Optional[int] = None
) -> List[Dict]:
    """批量处理数据库中的视频"""

    conn = sqlite3.connect(analyzer.db_path)
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
    print("综合面部麻痹分析 - 批量处理模式")
    print(f"总视频数: {len(videos)}")
    print(f"输出目录: {output_dir}")
    print("=" * 60 + "\n")

    results = []
    errors = []

    for i, (video_id, exam_id, action_id, file_path,
            start_frame, end_frame, fps, action_name, patient_id) in enumerate(videos, 1):

        if not file_path or not os.path.exists(file_path):
            continue

        print(f"[{i}/{len(videos)}] exam={exam_id} action={action_name}")

        try:
            result = analyzer.analyze_and_visualize(
                video_path=file_path,
                output_dir=output_dir,
                examination_id=exam_id,
                action_name=action_name,
                start_frame=start_frame,
                end_frame=end_frame,
                fps=fps
            )
            results.append({
                "video_id": video_id,
                "examination_id": exam_id,
                "action_name": action_name,
                "patient_id": patient_id,
                **result
            })
        except Exception as e:
            errors.append({
                "video_id": video_id,
                "examination_id": exam_id,
                "action_name": action_name,
                "error": str(e)
            })
            print(f"  ❌ 错误: {e}")

    # 保存汇总
    summary = {
        "success": results,
        "errors": errors,
        "total_tasks": len(videos),
        "success_count": len(results),
        "error_count": len(errors)
    }

    summary_path = os.path.join(output_dir, "z_batch_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print("✅ 批量处理完成!")
    print(f"   成功: {len(results)}/{len(videos)}")
    print(f"   失败: {len(errors)}")
    print(f"{'=' * 60}\n")

    return results


# ==================== Main ====================

def main():
    """示例用法"""
    # 配置路径
    db_path = "/Users/cuijinglei/PycharmProjects/medicalProject/facialPalsy/facialPalsy.db"
    model_path = "/Users/cuijinglei/PycharmProjects/medicalProject/models/face_landmarker.task"
    output_dir = "/Users/cuijinglei/Documents/facialPalsy/HGFA/comprehensive_analysis"

    # 创建分析器
    analyzer = ComprehensiveFacialPalsyAnalyzer(
        db_path=db_path,
        model_path=model_path,
        verbose=True
    )

    # 批量处理
    results = batch_process_database(
        analyzer=analyzer,
        output_dir=output_dir,
        limit=None,
        action_filter=None,
        use_multiprocessing=False
    )

    print("✅ 综合面部麻痹分析完成")


if __name__ == '__main__':
    main()