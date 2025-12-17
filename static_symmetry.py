"""
静态对称性评估模块 (Static Symmetry Assessment)
================================================

用于Sunnybrook表A静态分计算的三个指标：
1. 眼 (Eye/Palpebral Fissure) - 睑裂对称性
2. 颊 (Cheek/Nasolabial Fold) - 鼻唇沟深度
3. 嘴 (Mouth/Oral Commissure) - 口角位置

参考文献:
- 眼部: 使用眼内眦距离作为单位长度归一化
- 鼻唇沟: Wang et al. "Quantitative Evaluation of Nasolabial Fold by 3D Imaging" (2022)
- 口角: Kim et al. "Oral Commissure Lift" (2021)

作者: Rennie
日期: 2025-12
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict
from enum import Enum
import cv2


# ============================================================================
# MediaPipe 478点关键点索引定义
# ============================================================================

class LandmarkIndices:
    """MediaPipe FaceLandmarker 478点索引"""

    # 眼部内外眦点
    EYE_INNER_CANTHUS_LEFT = 362  # 左眼内眦
    EYE_INNER_CANTHUS_RIGHT = 133  # 右眼内眦
    EYE_OUTER_CANTHUS_LEFT = 263  # 左眼外眦
    EYE_OUTER_CANTHUS_RIGHT = 33  # 右眼外眦

    # 眼睑上下缘 (用于眼裂高度)
    EYE_TOP_LEFT = 386  # 左眼上眼睑中点
    EYE_BOTTOM_LEFT = 374  # 左眼下眼睑中点
    EYE_TOP_RIGHT = 159  # 右眼上眼睑中点
    EYE_BOTTOM_RIGHT = 145  # 右眼下眼睑中点

    # 左眼轮廓点 (17点，用于计算眼裂面积)
    # 顺序: 从内眦开始，沿上眼睑到外眦，再沿下眼睑回到内眦
    LEFT_EYE_CONTOUR = [
        362,  # 内眦 (P10)
        398, 384, 385, 386, 387, 388,  # 上眼睑
        263,  # 外眦 (P1)
        249, 390, 373, 374, 380, 381, 382,  # 下眼睑
        362  # 回到内眦 (闭合轮廓)
    ]

    # 右眼轮廓点 (17点)
    RIGHT_EYE_CONTOUR = [
        133,  # 内眦 (P10)
        173, 157, 158, 159, 160, 161,  # 上眼睑
        33,  # 外眦 (P1)
        7, 163, 144, 145, 153, 154, 155,  # 下眼睑
        133  # 回到内眦 (闭合轮廓)
    ]

    # 嘴角
    MOUTH_CORNER_LEFT = 291
    MOUTH_CORNER_RIGHT = 61

    # 上唇
    UPPER_LIP_CENTER = 0
    UPPER_LIP_TOP_LEFT = 37  # 左侧唇峰
    UPPER_LIP_TOP_RIGHT = 267  # 右侧唇峰 (对称点)

    # 下唇
    LOWER_LIP_CENTER = 17

    # 鼻部
    NOSE_TIP = 4
    NOSE_BOTTOM = 2  # 鼻小柱底部
    LEFT_NOSTRIL = 129  # 左鼻翼
    RIGHT_NOSTRIL = 358  # 右鼻翼
    LEFT_ALA = 219  # 左鼻翼外侧
    RIGHT_ALA = 439  # 右鼻翼外侧

    # 鼻唇沟相关点
    # 鼻唇沟起点 (鼻翼旁)
    NLF_START_LEFT = 129
    NLF_START_RIGHT = 358
    # 鼻唇沟终点 (嘴角旁)
    NLF_END_LEFT = 291
    NLF_END_RIGHT = 61

    # 面部参考点
    FOREHEAD = 10
    CHIN = 152


# ============================================================================
# 数据结构定义
# ============================================================================

class EyeStatus(Enum):
    """睑裂状态"""
    NORMAL = 0  # 正常
    NARROWED = 1  # 缩窄
    WIDENED = 2  # 增宽


class NasolabialFoldStatus(Enum):
    """鼻唇沟状态"""
    NORMAL = 0  # 正常
    ABSENT = 1  # 消失/变浅
    LESS_PRONOUNCED = 2  # 不明显
    MORE_PRONOUNCED = 3  # 过于明显/加深


class OralCommissureStatus(Enum):
    """口角状态"""
    NORMAL = 0  # 正常
    DROOPING = 1  # 下垂
    ELEVATED = 2  # 上提


@dataclass
class EyeMetrics:
    """眼部测量结果"""
    # 原始测量值 (像素)
    left_eye_area_raw: float  # 左眼裂面积
    right_eye_area_raw: float  # 右眼裂面积
    left_palpebral_length_raw: float  # 左眼睑裂长度 (内外眦距)
    right_palpebral_length_raw: float  # 右眼睑裂长度
    left_palpebral_height_raw: float  # 左眼睑裂高度
    right_palpebral_height_raw: float  # 右眼睑裂高度
    unit_length: float  # 单位长度 (两眼内眦距)

    # 归一化测量值 (C002L, C002R)
    left_eye_area_norm: float  # 左眼裂面积 / unit_length²
    right_eye_area_norm: float  # 右眼裂面积 / unit_length²

    # 派生指标
    eye_area_ratio: float  # C001: 眼裂面积比 = left/right
    left_eye_openness: float  # C005L: 左眼睁眼度
    right_eye_openness: float  # C005R: 右眼睁眼度
    left_eye_closure: float  # C006L: 左眼闭拢度 = 1 - 睁眼度
    right_eye_closure: float  # C006R: 右眼闭拢度

    # 不对称度评估
    area_asymmetry: float  # 面积不对称度 |1 - ratio|
    status: EyeStatus  # 状态判断
    affected_side: Optional[str]  # 患侧 ('left'/'right'/None)

    def to_dict(self) -> Dict:
        return {
            'left_eye_area_raw': self.left_eye_area_raw,
            'right_eye_area_raw': self.right_eye_area_raw,
            'left_eye_area_norm': self.left_eye_area_norm,
            'right_eye_area_norm': self.right_eye_area_norm,
            'eye_area_ratio': self.eye_area_ratio,
            'left_eye_openness': self.left_eye_openness,
            'right_eye_openness': self.right_eye_openness,
            'area_asymmetry': self.area_asymmetry,
            'status': self.status.name,
            'affected_side': self.affected_side,
        }


@dataclass
class NasolabialFoldMetrics:
    """鼻唇沟测量结果"""
    # 原始测量值 (像素)
    left_nlf_length_raw: float  # 左侧鼻唇沟长度
    right_nlf_length_raw: float  # 右侧鼻唇沟长度
    left_nlf_depth_proxy_raw: float  # 左侧深度代理值
    right_nlf_depth_proxy_raw: float  # 右侧深度代理值
    unit_length: float

    # 归一化测量值
    left_nlf_length_norm: float
    right_nlf_length_norm: float
    left_nlf_depth_proxy_norm: float
    right_nlf_depth_proxy_norm: float

    # 派生指标
    length_ratio: float  # 长度比 left/right
    depth_ratio: float  # 深度代理比
    length_asymmetry: float  # 长度不对称度
    depth_asymmetry: float  # 深度不对称度

    # 状态判断
    status: NasolabialFoldStatus
    affected_side: Optional[str]

    def to_dict(self) -> Dict:
        return {
            'left_nlf_length_norm': self.left_nlf_length_norm,
            'right_nlf_length_norm': self.right_nlf_length_norm,
            'length_ratio': self.length_ratio,
            'depth_ratio': self.depth_ratio,
            'length_asymmetry': self.length_asymmetry,
            'status': self.status.name,
            'affected_side': self.affected_side,
        }


@dataclass
class OralCommissureMetrics:
    """口角测量结果"""
    # 原始测量值
    left_commissure_y_raw: float  # 左嘴角Y坐标
    right_commissure_y_raw: float  # 右嘴角Y坐标
    left_angle_raw: float  # 左口角角度 (度)
    right_angle_raw: float  # 右口角角度 (度)
    mouth_midline_y_raw: float  # 口裂中线Y坐标
    unit_length: float

    # 归一化测量值
    left_height_diff_norm: float  # 左嘴角相对中线高度差 (归一化)
    right_height_diff_norm: float  # 右嘴角相对中线高度差

    # 派生指标
    height_diff: float  # 左右嘴角高度差 (归一化)
    angle_diff: float  # 左右角度差
    asymmetry: float  # 不对称度

    # 状态判断
    left_status: OralCommissureStatus
    right_status: OralCommissureStatus
    affected_side: Optional[str]

    def to_dict(self) -> Dict:
        return {
            'left_angle': self.left_angle_raw,
            'right_angle': self.right_angle_raw,
            'height_diff_norm': self.height_diff,
            'angle_diff': self.angle_diff,
            'asymmetry': self.asymmetry,
            'left_status': self.left_status.name,
            'right_status': self.right_status.name,
            'affected_side': self.affected_side,
        }


@dataclass
class StaticSymmetryResult:
    """静态对称性综合结果 (Sunnybrook表A)"""
    eye_metrics: EyeMetrics
    nlf_metrics: NasolabialFoldMetrics
    oral_metrics: OralCommissureMetrics

    # Sunnybrook静态分项目
    eye_score: int  # A1: 0=正常, 1=缩窄/增宽
    cheek_score: int  # A2: 0=正常, 1=不明显/过深, 2=消失
    mouth_score: int  # A3: 0=正常, 1=下垂/上提

    # 总静态分
    static_total: int  # (A1 + A2 + A3) × 5, 范围 0-20

    def to_dict(self) -> Dict:
        return {
            'eye': self.eye_metrics.to_dict(),
            'nasolabial_fold': self.nlf_metrics.to_dict(),
            'oral_commissure': self.oral_metrics.to_dict(),
            'sunnybrook': {
                'eye_score': self.eye_score,
                'cheek_score': self.cheek_score,
                'mouth_score': self.mouth_score,
                'static_total': self.static_total,
            }
        }


# ============================================================================
# 工具函数
# ============================================================================

def get_point(landmarks, index: int, w: int, h: int) -> Tuple[float, float]:
    """获取单个关键点的像素坐标"""
    return (landmarks[index].x * w, landmarks[index].y * h)


def get_points(landmarks, indices: List[int], w: int, h: int) -> np.ndarray:
    """批量获取关键点坐标"""
    return np.array([get_point(landmarks, idx, w, h) for idx in indices])


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """计算两点欧氏距离"""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def compute_polygon_area(points: np.ndarray) -> float:
    """
    计算多边形面积 (Shoelace公式)

    Args:
        points: (N, 2) 数组，多边形顶点坐标

    Returns:
        面积 (总是正值)
    """
    n = len(points)
    if n < 3:
        return 0.0

    # Shoelace formula
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]

    return abs(area) / 2.0


def compute_angle_with_horizontal(
        point: Tuple[float, float],
        reference: Tuple[float, float],
        horizontal_line: Tuple[Tuple[float, float], Tuple[float, float]]
) -> float:
    """
    计算点相对于参考点与水平线的夹角

    参考 Kim et al. 2021 口角角度测量方法:
    - 负角度表示点低于水平线 (下垂)
    - 正角度表示点高于水平线 (上提)

    Args:
        point: 目标点 (如嘴角)
        reference: 参考点 (如口裂中点)
        horizontal_line: 水平参考线 ((x1,y1), (x2,y2))

    Returns:
        角度 (度), 负=下垂, 正=上提
    """
    # 计算水平线方向向量
    h_vec = (
        horizontal_line[1][0] - horizontal_line[0][0],
        horizontal_line[1][1] - horizontal_line[0][1]
    )

    # 计算参考点到目标点的向量
    p_vec = (
        point[0] - reference[0],
        point[1] - reference[1]
    )

    # 计算叉积确定方向 (Y轴向下，所以符号反转)
    cross = h_vec[0] * p_vec[1] - h_vec[1] * p_vec[0]

    # 计算点积
    dot = h_vec[0] * p_vec[0] + h_vec[1] * p_vec[1]

    # 计算夹角
    h_mag = np.sqrt(h_vec[0] ** 2 + h_vec[1] ** 2)
    p_mag = np.sqrt(p_vec[0] ** 2 + p_vec[1] ** 2)

    if h_mag < 1e-6 or p_mag < 1e-6:
        return 0.0

    cos_theta = np.clip(dot / (h_mag * p_mag), -1.0, 1.0)
    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)

    # 根据叉积确定正负 (Y轴向下，cross > 0表示点在线下方)
    if cross > 0:
        angle_deg = -angle_deg

    return angle_deg


# ============================================================================
# 眼部测量 (睑裂)
# ============================================================================

class EyeAssessor:
    """
    眼部睑裂评估器

    测量指标:
    - C001: 眼裂面积比 = 左眼面积 / 右眼面积
    - C002L/C002R: 归一化眼裂面积 = 眼裂面积 / 单位长度²
    - C005: 睁眼度 = 眼裂面积 / 眼睑裂长度²
    - C006: 闭拢度 = 1 - 睁眼度

    判断标准:
    - 正常: 面积比在 [0.85, 1.15] 范围内
    - 缩窄: 面积比 < 0.85 或 > 1.15 (患侧面积较小)
    - 增宽: 患侧面积较大 (Bell现象或代偿)
    """

    # 阈值设置
    NORMAL_RATIO_MIN = 0.85
    NORMAL_RATIO_MAX = 1.15
    SEVERE_RATIO_MIN = 0.70
    SEVERE_RATIO_MAX = 1.30

    def __init__(self):
        self.idx = LandmarkIndices

    def compute_eye_area(
            self,
            landmarks,
            w: int,
            h: int,
            left: bool = True
    ) -> float:
        """
        计算眼裂面积

        使用眼部轮廓点围成的多边形面积
        """
        contour_indices = (
            self.idx.LEFT_EYE_CONTOUR if left
            else self.idx.RIGHT_EYE_CONTOUR
        )

        points = get_points(landmarks, contour_indices[:-1], w, h)  # 去掉最后重复点
        return compute_polygon_area(points)

    def compute_palpebral_length(
            self,
            landmarks,
            w: int,
            h: int,
            left: bool = True
    ) -> float:
        """
        计算眼睑裂长度 (内外眦距离)
        """
        if left:
            inner = get_point(landmarks, self.idx.EYE_INNER_CANTHUS_LEFT, w, h)
            outer = get_point(landmarks, self.idx.EYE_OUTER_CANTHUS_LEFT, w, h)
        else:
            inner = get_point(landmarks, self.idx.EYE_INNER_CANTHUS_RIGHT, w, h)
            outer = get_point(landmarks, self.idx.EYE_OUTER_CANTHUS_RIGHT, w, h)

        return euclidean_distance(inner, outer)

    def compute_palpebral_height(
            self,
            landmarks,
            w: int,
            h: int,
            left: bool = True
    ) -> float:
        """
        计算眼睑裂高度 (上下眼睑中点距离)
        """
        if left:
            top = get_point(landmarks, self.idx.EYE_TOP_LEFT, w, h)
            bottom = get_point(landmarks, self.idx.EYE_BOTTOM_LEFT, w, h)
        else:
            top = get_point(landmarks, self.idx.EYE_TOP_RIGHT, w, h)
            bottom = get_point(landmarks, self.idx.EYE_BOTTOM_RIGHT, w, h)

        return euclidean_distance(top, bottom)

    def compute_unit_length(self, landmarks, w: int, h: int) -> float:
        """
        计算单位长度 (两眼内眦距离)

        这是所有测量归一化的基准
        """
        left_inner = get_point(landmarks, self.idx.EYE_INNER_CANTHUS_LEFT, w, h)
        right_inner = get_point(landmarks, self.idx.EYE_INNER_CANTHUS_RIGHT, w, h)
        return euclidean_distance(left_inner, right_inner)

    def assess(self, landmarks, w: int, h: int) -> EyeMetrics:
        """
        完整的眼部评估

        Returns:
            EyeMetrics 包含所有测量值和状态判断
        """
        # 计算单位长度
        unit_length = self.compute_unit_length(landmarks, w, h)

        # 计算原始眼裂面积
        left_area_raw = self.compute_eye_area(landmarks, w, h, left=True)
        right_area_raw = self.compute_eye_area(landmarks, w, h, left=False)

        # 计算眼睑裂长度
        left_length_raw = self.compute_palpebral_length(landmarks, w, h, left=True)
        right_length_raw = self.compute_palpebral_length(landmarks, w, h, left=False)

        # 计算眼睑裂高度
        left_height_raw = self.compute_palpebral_height(landmarks, w, h, left=True)
        right_height_raw = self.compute_palpebral_height(landmarks, w, h, left=False)

        # 归一化面积 (C002L, C002R)
        unit_sq = unit_length ** 2
        left_area_norm = left_area_raw / unit_sq if unit_sq > 0 else 0.0
        right_area_norm = right_area_raw / unit_sq if unit_sq > 0 else 0.0

        # 面积比 (C001)
        if right_area_raw > 1e-6:
            eye_area_ratio = left_area_raw / right_area_raw
        else:
            eye_area_ratio = float('inf')

        # 睁眼度 (C005): 眼裂面积 / 眼睑裂长度²
        left_length_sq = left_length_raw ** 2
        right_length_sq = right_length_raw ** 2

        left_openness = left_area_raw / left_length_sq if left_length_sq > 0 else 0.0
        right_openness = right_area_raw / right_length_sq if right_length_sq > 0 else 0.0

        # 闭拢度 (C006): 1 - 睁眼度
        # 注意: 睁眼度不是百分比，需要设定合理范围
        # 这里用相对值表示
        left_closure = 1.0 - min(1.0, left_openness)
        right_closure = 1.0 - min(1.0, right_openness)

        # 不对称度
        area_asymmetry = abs(1.0 - eye_area_ratio)

        # 状态判断
        if self.NORMAL_RATIO_MIN <= eye_area_ratio <= self.NORMAL_RATIO_MAX:
            status = EyeStatus.NORMAL
            affected_side = None
        elif eye_area_ratio < self.NORMAL_RATIO_MIN:
            # 左眼面积小于右眼
            status = EyeStatus.NARROWED
            affected_side = 'left'
        else:  # eye_area_ratio > self.NORMAL_RATIO_MAX
            # 左眼面积大于右眼 (右眼相对缩窄)
            status = EyeStatus.NARROWED
            affected_side = 'right'

        return EyeMetrics(
            left_eye_area_raw=left_area_raw,
            right_eye_area_raw=right_area_raw,
            left_palpebral_length_raw=left_length_raw,
            right_palpebral_length_raw=right_length_raw,
            left_palpebral_height_raw=left_height_raw,
            right_palpebral_height_raw=right_height_raw,
            unit_length=unit_length,
            left_eye_area_norm=left_area_norm,
            right_eye_area_norm=right_area_norm,
            eye_area_ratio=eye_area_ratio,
            left_eye_openness=left_openness,
            right_eye_openness=right_openness,
            left_eye_closure=left_closure,
            right_eye_closure=right_closure,
            area_asymmetry=area_asymmetry,
            status=status,
            affected_side=affected_side,
        )


# ============================================================================
# 鼻唇沟测量
# ============================================================================

class NasolabialFoldAssessor:
    """
    鼻唇沟评估器

    参考: Wang et al. "Quantitative Evaluation of Nasolabial Fold" (2022)

    由于MediaPipe是2D landmarks，无法直接测量深度。
    使用以下替代指标:

    1. 鼻唇沟长度: 从鼻翼点到嘴角的距离
       - 面瘫患侧通常鼻唇沟变浅变短

    2. 深度代理: 使用面部轮廓点与鼻唇沟线的距离
       - 正常侧鼻唇沟凹陷，面颊与线的距离较大
       - 患侧鼻唇沟消失，面颊与线的距离较小

    判断标准:
    - 正常: 长度比在 [0.90, 1.10] 范围内
    - 消失: 患侧长度明显减小 (< 0.85)
    - 不明显: 轻度减小 (0.85-0.90)
    - 过深: 患侧长度明显增大 (> 1.15)
    """

    # 阈值
    NORMAL_RATIO_MIN = 0.90
    NORMAL_RATIO_MAX = 1.10
    LESS_PRONOUNCED_THRESHOLD = 0.85
    ABSENT_THRESHOLD = 0.75
    MORE_PRONOUNCED_THRESHOLD = 1.15

    def __init__(self):
        self.idx = LandmarkIndices

    def compute_nlf_length(
            self,
            landmarks,
            w: int,
            h: int,
            left: bool = True
    ) -> float:
        """
        计算鼻唇沟长度 (鼻翼到嘴角的距离)
        """
        if left:
            start = get_point(landmarks, self.idx.NLF_START_LEFT, w, h)
            end = get_point(landmarks, self.idx.NLF_END_LEFT, w, h)
        else:
            start = get_point(landmarks, self.idx.NLF_START_RIGHT, w, h)
            end = get_point(landmarks, self.idx.NLF_END_RIGHT, w, h)

        return euclidean_distance(start, end)

    def compute_nlf_depth_proxy(
            self,
            landmarks,
            w: int,
            h: int,
            left: bool = True
    ) -> float:
        """
        计算鼻唇沟深度代理值

        使用鼻翼-嘴角连线与面颊中点的垂直距离
        鼻唇沟越深，面颊越凸出，距离越大
        """
        # 面颊点 (在鼻唇沟线旁边)
        # 使用MediaPipe的面颊区域关键点
        if left:
            cheek_indices = [425, 426, 427]  # 左颊点
            start = get_point(landmarks, self.idx.NLF_START_LEFT, w, h)
            end = get_point(landmarks, self.idx.NLF_END_LEFT, w, h)
        else:
            cheek_indices = [205, 206, 207]  # 右颊点
            start = get_point(landmarks, self.idx.NLF_START_RIGHT, w, h)
            end = get_point(landmarks, self.idx.NLF_END_RIGHT, w, h)

        # 计算鼻唇沟线方程
        line_vec = (end[0] - start[0], end[1] - start[1])
        line_length = np.sqrt(line_vec[0] ** 2 + line_vec[1] ** 2)

        if line_length < 1e-6:
            return 0.0

        # 计算面颊点到鼻唇沟线的平均距离
        distances = []
        for idx in cheek_indices:
            try:
                cheek_point = get_point(landmarks, idx, w, h)
                # 点到线的距离
                # d = |cross(line_vec, point_vec)| / |line_vec|
                point_vec = (cheek_point[0] - start[0], cheek_point[1] - start[1])
                cross = abs(line_vec[0] * point_vec[1] - line_vec[1] * point_vec[0])
                distance = cross / line_length
                distances.append(distance)
            except (IndexError, AttributeError):
                continue

        return np.mean(distances) if distances else 0.0

    def assess(self, landmarks, w: int, h: int, unit_length: float) -> NasolabialFoldMetrics:
        """
        完整的鼻唇沟评估
        """
        # 计算长度
        left_length_raw = self.compute_nlf_length(landmarks, w, h, left=True)
        right_length_raw = self.compute_nlf_length(landmarks, w, h, left=False)

        # 计算深度代理
        left_depth_raw = self.compute_nlf_depth_proxy(landmarks, w, h, left=True)
        right_depth_raw = self.compute_nlf_depth_proxy(landmarks, w, h, left=False)

        # 归一化
        left_length_norm = left_length_raw / unit_length if unit_length > 0 else 0.0
        right_length_norm = right_length_raw / unit_length if unit_length > 0 else 0.0
        left_depth_norm = left_depth_raw / unit_length if unit_length > 0 else 0.0
        right_depth_norm = right_depth_raw / unit_length if unit_length > 0 else 0.0

        # 比值
        length_ratio = left_length_raw / right_length_raw if right_length_raw > 1e-6 else 1.0
        depth_ratio = left_depth_raw / right_depth_raw if right_depth_raw > 1e-6 else 1.0

        # 不对称度
        length_asymmetry = abs(1.0 - length_ratio)
        depth_asymmetry = abs(1.0 - depth_ratio)

        # 状态判断 (主要基于长度比)
        if self.NORMAL_RATIO_MIN <= length_ratio <= self.NORMAL_RATIO_MAX:
            status = NasolabialFoldStatus.NORMAL
            affected_side = None
        elif length_ratio < self.ABSENT_THRESHOLD:
            # 左侧鼻唇沟长度明显减小 = 左侧消失
            status = NasolabialFoldStatus.ABSENT
            affected_side = 'left'
        elif length_ratio < self.LESS_PRONOUNCED_THRESHOLD:
            status = NasolabialFoldStatus.LESS_PRONOUNCED
            affected_side = 'left'
        elif length_ratio > 1.0 / self.ABSENT_THRESHOLD:
            # 右侧相对消失
            status = NasolabialFoldStatus.ABSENT
            affected_side = 'right'
        elif length_ratio > 1.0 / self.LESS_PRONOUNCED_THRESHOLD:
            status = NasolabialFoldStatus.LESS_PRONOUNCED
            affected_side = 'right'
        elif length_ratio > self.MORE_PRONOUNCED_THRESHOLD:
            status = NasolabialFoldStatus.MORE_PRONOUNCED
            affected_side = 'left'  # 左侧过深
        else:
            status = NasolabialFoldStatus.NORMAL
            affected_side = None

        return NasolabialFoldMetrics(
            left_nlf_length_raw=left_length_raw,
            right_nlf_length_raw=right_length_raw,
            left_nlf_depth_proxy_raw=left_depth_raw,
            right_nlf_depth_proxy_raw=right_depth_raw,
            unit_length=unit_length,
            left_nlf_length_norm=left_length_norm,
            right_nlf_length_norm=right_length_norm,
            left_nlf_depth_proxy_norm=left_depth_norm,
            right_nlf_depth_proxy_norm=right_depth_norm,
            length_ratio=length_ratio,
            depth_ratio=depth_ratio,
            length_asymmetry=length_asymmetry,
            depth_asymmetry=depth_asymmetry,
            status=status,
            affected_side=affected_side,
        )


# ============================================================================
# 口角测量
# ============================================================================

class OralCommissureAssessor:
    """
    口角评估器

    参考: Kim et al. "Oral Commissure Lift" (2021)

    测量方法:
    1. 定义口裂水平参考线 (通过上唇峰值点的投影)
    2. 计算嘴角相对于水平线的角度
    3. 负角度 = 下垂, 正角度 = 上提

    简化方法 (使用Y坐标):
    - 计算嘴角Y坐标与口裂中线的差值
    - Y值较大 (下方) = 下垂
    - Y值较小 (上方) = 上提

    判断标准:
    - 正常: 高度差 < 0.02 (归一化)
    - 下垂/上提: 高度差 >= 0.02
    """

    # 阈值 (归一化后)
    HEIGHT_DIFF_THRESHOLD = 0.02
    SEVERE_HEIGHT_DIFF_THRESHOLD = 0.05

    def __init__(self):
        self.idx = LandmarkIndices

    def compute_oral_commissure_angle(
            self,
            landmarks,
            w: int,
            h: int,
            left: bool = True
    ) -> Tuple[float, float]:
        """
        计算口角角度 (参考Kim et al. 方法)

        Returns:
            (angle, y_position)
            - angle: 口角角度 (度), 负=下垂, 正=上提
            - y_position: 口角Y坐标
        """
        # 获取关键点
        left_corner = get_point(landmarks, self.idx.MOUTH_CORNER_LEFT, w, h)
        right_corner = get_point(landmarks, self.idx.MOUTH_CORNER_RIGHT, w, h)
        upper_lip_left_peak = get_point(landmarks, self.idx.UPPER_LIP_TOP_LEFT, w, h)
        upper_lip_right_peak = get_point(landmarks, self.idx.UPPER_LIP_TOP_RIGHT, w, h)

        # 计算口裂水平线
        # E点: 左侧唇峰在口裂水平线的投影 (使用嘴角Y坐标)
        # F点: 右侧唇峰在口裂水平线的投影
        midline_y = (left_corner[1] + right_corner[1]) / 2
        midpoint_x = (left_corner[0] + right_corner[0]) / 2

        # 水平参考线通过唇峰投影点
        # 使用嘴角连线作为水平参考
        horizontal_line = (left_corner, right_corner)

        # O点: 水平线中点
        o_point = (midpoint_x, midline_y)

        # 计算角度
        target_corner = left_corner if left else right_corner
        angle = compute_angle_with_horizontal(
            point=target_corner,
            reference=o_point,
            horizontal_line=horizontal_line
        )

        return angle, target_corner[1]

    def assess(self, landmarks, w: int, h: int, unit_length: float) -> OralCommissureMetrics:
        """
        完整的口角评估
        """
        # 获取嘴角位置
        left_corner = get_point(landmarks, self.idx.MOUTH_CORNER_LEFT, w, h)
        right_corner = get_point(landmarks, self.idx.MOUTH_CORNER_RIGHT, w, h)

        # 计算口裂中线Y坐标
        midline_y = (left_corner[1] + right_corner[1]) / 2

        # 计算角度
        left_angle, left_y = self.compute_oral_commissure_angle(
            landmarks, w, h, left=True
        )
        right_angle, right_y = self.compute_oral_commissure_angle(
            landmarks, w, h, left=False
        )

        # 归一化高度差
        left_height_diff_norm = (left_y - midline_y) / unit_length if unit_length > 0 else 0.0
        right_height_diff_norm = (right_y - midline_y) / unit_length if unit_length > 0 else 0.0

        # 左右高度差 (正值表示左侧更低/下垂)
        height_diff = (left_y - right_y) / unit_length if unit_length > 0 else 0.0

        # 角度差
        angle_diff = left_angle - right_angle

        # 不对称度
        asymmetry = abs(height_diff)

        # 状态判断
        def judge_status(height_diff_norm: float) -> OralCommissureStatus:
            if abs(height_diff_norm) < self.HEIGHT_DIFF_THRESHOLD:
                return OralCommissureStatus.NORMAL
            elif height_diff_norm > 0:  # Y值大 = 下方 = 下垂
                return OralCommissureStatus.DROOPING
            else:
                return OralCommissureStatus.ELEVATED

        left_status = judge_status(left_height_diff_norm)
        right_status = judge_status(right_height_diff_norm)

        # 患侧判断
        if left_status == OralCommissureStatus.DROOPING:
            affected_side = 'left'
        elif right_status == OralCommissureStatus.DROOPING:
            affected_side = 'right'
        elif asymmetry >= self.HEIGHT_DIFF_THRESHOLD:
            # 有不对称但没有明显下垂，取较低侧为患侧
            affected_side = 'left' if height_diff > 0 else 'right'
        else:
            affected_side = None

        return OralCommissureMetrics(
            left_commissure_y_raw=left_y,
            right_commissure_y_raw=right_y,
            left_angle_raw=left_angle,
            right_angle_raw=right_angle,
            mouth_midline_y_raw=midline_y,
            unit_length=unit_length,
            left_height_diff_norm=left_height_diff_norm,
            right_height_diff_norm=right_height_diff_norm,
            height_diff=height_diff,
            angle_diff=angle_diff,
            asymmetry=asymmetry,
            left_status=left_status,
            right_status=right_status,
            affected_side=affected_side,
        )


# ============================================================================
# 综合静态对称性评估
# ============================================================================

class StaticSymmetryAssessor:
    """
    静态对称性综合评估器

    整合眼、颊、嘴三个指标，计算Sunnybrook表A静态分

    Sunnybrook表A:
    - 眼 (A1): 0=正常, 1=缩窄或增宽
    - 颊 (A2): 0=正常, 1=不明显或过深, 2=消失
    - 嘴 (A3): 0=正常, 1=下垂或上提

    静态总分 = (A1 + A2 + A3) × 5
    范围: 0-20
    """

    def __init__(self):
        self.eye_assessor = EyeAssessor()
        self.nlf_assessor = NasolabialFoldAssessor()
        self.oral_assessor = OralCommissureAssessor()

    def assess(self, landmarks, w: int, h: int) -> StaticSymmetryResult:
        """
        完整的静态对称性评估

        Args:
            landmarks: MediaPipe face_landmarks对象
            w: 图像宽度
            h: 图像高度

        Returns:
            StaticSymmetryResult 包含所有测量和评分
        """
        # 眼部评估
        eye_metrics = self.eye_assessor.assess(landmarks, w, h)
        unit_length = eye_metrics.unit_length

        # 鼻唇沟评估
        nlf_metrics = self.nlf_assessor.assess(landmarks, w, h, unit_length)

        # 口角评估
        oral_metrics = self.oral_assessor.assess(landmarks, w, h, unit_length)

        # Sunnybrook评分
        # A1: 眼
        if eye_metrics.status == EyeStatus.NORMAL:
            eye_score = 0
        else:
            eye_score = 1

        # A2: 颊 (鼻唇沟)
        if nlf_metrics.status == NasolabialFoldStatus.NORMAL:
            cheek_score = 0
        elif nlf_metrics.status == NasolabialFoldStatus.ABSENT:
            cheek_score = 2
        else:  # LESS_PRONOUNCED or MORE_PRONOUNCED
            cheek_score = 1

        # A3: 嘴
        if oral_metrics.left_status == OralCommissureStatus.NORMAL and \
                oral_metrics.right_status == OralCommissureStatus.NORMAL:
            mouth_score = 0
        else:
            mouth_score = 1

        # 总分
        static_total = (eye_score + cheek_score + mouth_score) * 5

        return StaticSymmetryResult(
            eye_metrics=eye_metrics,
            nlf_metrics=nlf_metrics,
            oral_metrics=oral_metrics,
            eye_score=eye_score,
            cheek_score=cheek_score,
            mouth_score=mouth_score,
            static_total=static_total,
        )


# ============================================================================
# 使用示例和测试
# ============================================================================

def example_usage():
    """示例用法"""
    print("静态对称性评估模块")
    print("=" * 60)
    print()
    print("用法示例:")
    print("""
    from static_symmetry import StaticSymmetryAssessor

    # 初始化评估器
    assessor = StaticSymmetryAssessor()

    # 假设已有MediaPipe landmarks
    # landmarks = face_landmarker.detect(image).face_landmarks[0]
    # w, h = image.shape[1], image.shape[0]

    # 评估
    result = assessor.assess(landmarks, w, h)

    # 输出结果
    print(f"眼部状态: {result.eye_metrics.status.name}")
    print(f"眼裂面积比: {result.eye_metrics.eye_area_ratio:.3f}")
    print(f"鼻唇沟状态: {result.nlf_metrics.status.name}")
    print(f"口角状态: L={result.oral_metrics.left_status.name}, R={result.oral_metrics.right_status.name}")
    print(f"Sunnybrook静态分: {result.static_total}/20")
    """)

    print()
    print("测量指标说明:")
    print("-" * 60)
    print("""
    眼部 (基于C001-C006规范):
    - C001: 眼裂面积比 = 左眼面积/右眼面积
    - C002L/R: 归一化眼裂面积 = 眼裂面积/单位长度²
    - C005: 睁眼度 = 眼裂面积/眼睑裂长度²
    - C006: 闭拢度 = 1 - 睁眼度

    鼻唇沟 (参考Wang et al. 2022):
    - 长度: 鼻翼到嘴角距离
    - 深度代理: 面颊点到鼻唇沟线的距离

    口角 (参考Kim et al. 2021):
    - 角度: 嘴角相对口裂水平线的夹角
    - 高度差: 左右嘴角Y坐标差值
    """)


if __name__ == '__main__':
    example_usage()