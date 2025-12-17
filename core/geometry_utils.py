"""
几何计算核心模块 (Geometry Core)
================================

所有面部测量的几何计算基础

单位长度: ICD (Inner Canthi Distance) = 双眼内眦距离
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass

# 从constants模块导入所有常量
from .constants import LM, Thresholds


# =============================================================================
# 基础函数
# =============================================================================

def pt2d(lm, w: int, h: int) -> Tuple[float, float]:
    """单个landmark → 2D像素坐标"""
    return (lm.x * w, lm.y * h)


def pts2d(landmarks, indices: List[int], w: int, h: int) -> np.ndarray:
    """批量获取2D坐标"""
    return np.array([pt2d(landmarks[i], w, h) for i in indices])


def dist(p1, p2) -> float:
    """2D欧氏距离"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def polygon_area(points: np.ndarray) -> float:
    """多边形面积 (Shoelace公式)"""
    n = len(points)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i, 0] * points[j, 1]
        area -= points[j, 0] * points[i, 1]
    return abs(area) / 2.0


def pt_to_line_dist(pt, line_start, line_end) -> float:
    """点到线段的垂直距离"""
    lv = (line_end[0] - line_start[0], line_end[1] - line_start[1])
    ll = np.sqrt(lv[0]**2 + lv[1]**2)
    if ll < Thresholds.EPSILON:
        return dist(pt, line_start)
    pv = (pt[0] - line_start[0], pt[1] - line_start[1])
    cross = abs(lv[0] * pv[1] - lv[1] * pv[0])
    return cross / ll


# =============================================================================
# ICD计算
# =============================================================================

def compute_icd(landmarks, w: int, h: int) -> float:
    """计算ICD (双眼内眦距离) - 单位长度"""
    l_inner = pt2d(landmarks[LM.EYE_INNER_L], w, h)
    r_inner = pt2d(landmarks[LM.EYE_INNER_R], w, h)
    return dist(l_inner, r_inner)


# =============================================================================
# 眼部测量
# =============================================================================

def compute_eye_area(landmarks, w: int, h: int, left: bool = True) -> Tuple[float, np.ndarray]:
    """计算眼裂面积"""
    contour = LM.EYE_CONTOUR_L if left else LM.EYE_CONTOUR_R
    points = pts2d(landmarks, contour, w, h)
    return polygon_area(points), points


def compute_palpebral_length(landmarks, w: int, h: int, left: bool = True) -> float:
    """计算眼睑裂长度 (内外眦距离)"""
    if left:
        inner = pt2d(landmarks[LM.EYE_INNER_L], w, h)
        outer = pt2d(landmarks[LM.EYE_OUTER_L], w, h)
    else:
        inner = pt2d(landmarks[LM.EYE_INNER_R], w, h)
        outer = pt2d(landmarks[LM.EYE_OUTER_R], w, h)
    return dist(inner, outer)


def compute_palpebral_height(landmarks, w: int, h: int, left: bool = True) -> float:
    """计算眼睑裂高度"""
    if left:
        top = pt2d(landmarks[LM.EYE_TOP_L], w, h)
        bot = pt2d(landmarks[LM.EYE_BOT_L], w, h)
    else:
        top = pt2d(landmarks[LM.EYE_TOP_R], w, h)
        bot = pt2d(landmarks[LM.EYE_BOT_R], w, h)
    return dist(top, bot)


def compute_ear(landmarks, w: int, h: int, left: bool = True) -> float:
    """计算EAR (Eye Aspect Ratio)"""
    pts_idx = LM.EAR_L if left else LM.EAR_R
    p = [pt2d(landmarks[i], w, h) for i in pts_idx]
    # p[0]=外眦, p[1]=上眼睑1, p[2]=上眼睑2, p[3]=内眦, p[4]=下眼睑1, p[5]=下眼睑2
    v1 = dist(p[1], p[5])
    v2 = dist(p[2], p[4])
    h_dist = dist(p[0], p[3])
    if h_dist < Thresholds.EPSILON:
        return 0.0
    return (v1 + v2) / (2.0 * h_dist)


@dataclass
class EyeMeasure:
    """单眼测量结果"""
    area_raw: float           # 原始眼裂面积
    area_norm: float          #  归一化眼裂面积
    palpebral_length: float   # 眼睑裂长度
    palpebral_height: float   # 眼睑裂高度
    openness: float           # 睁眼度 (0-1+)
    closure: float            # 闭拢度 (0-1)
    complete_closure: bool    # 完全闭眼
    ear: float                # EAR值
    contour: np.ndarray       # 轮廓点


def measure_eye(
    landmarks, w: int, h: int, icd: float, left: bool = True,
    baseline_area: float = None,
    baseline_palp_len: float = None
) -> EyeMeasure:
    """
    完整单眼测量

    睁眼度计算方法:
    - 如果有基准(NeutralFace): openness = 当前面积 / (基准面积 * scale²)
    - 如果没有基准: 使用眼裂面积/眼睑裂长度²的归一化方法
    """
    area_raw, contour = compute_eye_area(landmarks, w, h, left)
    palp_len = compute_palpebral_length(landmarks, w, h, left)
    palp_hgt = compute_palpebral_height(landmarks, w, h, left)
    ear = compute_ear(landmarks, w, h, left)

    # 归一化面积
    icd_sq = icd ** 2
    area_norm = area_raw / icd_sq if icd_sq > Thresholds.EPSILON else 0.0

    # 睁眼度
    if baseline_area is not None and baseline_area > Thresholds.EPSILON:
        # 有基准: 计算scale并调整
        if baseline_palp_len is not None and baseline_palp_len > Thresholds.EPSILON:
            scale = palp_len / baseline_palp_len
            scale_sq = scale ** 2
            adjusted_area = area_raw / scale_sq if scale_sq > Thresholds.EPSILON else area_raw
            openness = adjusted_area / baseline_area
        else:
            openness = area_raw / baseline_area
    else:
        # 无基准: 使用面积/长度²
        palp_len_sq = palp_len ** 2
        if palp_len_sq > Thresholds.EPSILON:
            raw_openness = area_raw / palp_len_sq
            # 经验归一化 (典型睁眼时约0.2-0.3)
            openness = min(1.0, raw_openness / 0.25)
        else:
            openness = 0.0

    openness = max(0.0, openness)

    # 闭拢度
    closure = max(0.0, 1.0 - min(1.0, openness))

    # 完全闭眼
    complete_closure = openness <= Thresholds.COMPLETE_CLOSURE

    return EyeMeasure(
        area_raw=area_raw,
        area_norm=area_norm,
        palpebral_length=palp_len,
        palpebral_height=palp_hgt,
        openness=openness,
        closure=closure,
        complete_closure=complete_closure,
        ear=ear,
        contour=contour
    )


@dataclass
class BilateralEyeMeasure:
    """双眼测量结果"""
    left: EyeMeasure
    right: EyeMeasure
    icd: float
    area_ratio: float         # C001: 左/右
    openness_ratio: float
    asymmetry: float


def measure_eyes(
    landmarks, w: int, h: int,
    baseline_l_area: float = None,
    baseline_r_area: float = None,
    baseline_l_palp_len: float = None,
    baseline_r_palp_len: float = None
) -> BilateralEyeMeasure:
    """双眼测量"""
    icd = compute_icd(landmarks, w, h)

    left = measure_eye(landmarks, w, h, icd, left=True,
                       baseline_area=baseline_l_area,
                       baseline_palp_len=baseline_l_palp_len)
    right = measure_eye(landmarks, w, h, icd, left=False,
                        baseline_area=baseline_r_area,
                        baseline_palp_len=baseline_r_palp_len)

    # C001
    area_ratio = left.area_raw / right.area_raw if right.area_raw > Thresholds.EPSILON else float('inf')
    openness_ratio = left.openness / right.openness if right.openness > Thresholds.EPSILON else (
        float('inf') if left.openness > Thresholds.EPSILON else 1.0)
    asymmetry = abs(1.0 - area_ratio) if area_ratio != float('inf') else 1.0

    return BilateralEyeMeasure(
        left=left, right=right, icd=icd,
        area_ratio=area_ratio,
        openness_ratio=openness_ratio,
        asymmetry=asymmetry
    )


# =============================================================================
# 口角测量
# =============================================================================

@dataclass
class OralMeasure:
    """口角测量结果"""
    left_corner: Tuple[float, float]
    right_corner: Tuple[float, float]
    midline_y: float
    left_angle: float          # 左口角角度 (度), 负=下垂
    right_angle: float
    height_diff: float         # 归一化高度差, 正=左侧低
    mouth_width: float
    mouth_height: float


def measure_oral(landmarks, w: int, h: int, icd: float) -> OralMeasure:
    """口角测量"""
    l_corner = pt2d(landmarks[LM.MOUTH_L], w, h)
    r_corner = pt2d(landmarks[LM.MOUTH_R], w, h)

    midline_y = (l_corner[1] + r_corner[1]) / 2
    midpoint_x = (l_corner[0] + r_corner[0]) / 2

    # 计算角度: 正=口角在中线上方, 负=下方
    def angle(corner):
        dy = midline_y - corner[1]  # 正=上方
        dx = abs(corner[0] - midpoint_x)
        if dx < Thresholds.EPSILON:
            return 90.0 if dy > 0 else -90.0 if dy < 0 else 0.0
        return np.degrees(np.arctan2(dy, dx))

    left_angle = angle(l_corner)
    right_angle = angle(r_corner)

    # 高度差 (正=左侧更低=左侧下垂)
    height_diff = (l_corner[1] - r_corner[1]) / icd if icd > Thresholds.EPSILON else 0.0

    # 嘴宽高
    mouth_width = dist(l_corner, r_corner)
    lip_top = pt2d(landmarks[LM.LIP_TOP_CENTER], w, h)
    lip_bot = pt2d(landmarks[LM.LIP_BOT_CENTER], w, h)
    mouth_height = dist(lip_top, lip_bot)

    return OralMeasure(
        left_corner=l_corner,
        right_corner=r_corner,
        midline_y=midline_y,
        left_angle=left_angle,
        right_angle=right_angle,
        height_diff=height_diff,
        mouth_width=mouth_width,
        mouth_height=mouth_height
    )


# =============================================================================
# 鼻唇沟测量
# =============================================================================

@dataclass
class NLFMeasure:
    """鼻唇沟测量结果"""
    left_length: float
    right_length: float
    left_length_norm: float
    right_length_norm: float
    length_ratio: float
    left_depth_proxy: float
    right_depth_proxy: float
    depth_ratio: float


def measure_nlf(landmarks, w: int, h: int, icd: float) -> NLFMeasure:
    """鼻唇沟测量"""
    l_ala = pt2d(landmarks[LM.NOSE_ALA_L], w, h)
    r_ala = pt2d(landmarks[LM.NOSE_ALA_R], w, h)
    l_mouth = pt2d(landmarks[LM.MOUTH_L], w, h)
    r_mouth = pt2d(landmarks[LM.MOUTH_R], w, h)

    l_len = dist(l_ala, l_mouth)
    r_len = dist(r_ala, r_mouth)

    l_len_norm = l_len / icd if icd > Thresholds.EPSILON else 0.0
    r_len_norm = r_len / icd if icd > Thresholds.EPSILON else 0.0
    len_ratio = l_len / r_len if r_len > Thresholds.EPSILON else 1.0

    # 深度代理
    l_cheek = pts2d(landmarks, LM.CHEEK_L, w, h)
    r_cheek = pts2d(landmarks, LM.CHEEK_R, w, h)

    l_dists = [pt_to_line_dist(tuple(p), l_ala, l_mouth) for p in l_cheek]
    r_dists = [pt_to_line_dist(tuple(p), r_ala, r_mouth) for p in r_cheek]

    l_depth = np.mean(l_dists) if l_dists else 0.0
    r_depth = np.mean(r_dists) if r_dists else 0.0
    depth_ratio = l_depth / r_depth if r_depth > Thresholds.EPSILON else 1.0

    return NLFMeasure(
        left_length=l_len, right_length=r_len,
        left_length_norm=l_len_norm, right_length_norm=r_len_norm,
        length_ratio=len_ratio,
        left_depth_proxy=l_depth, right_depth_proxy=r_depth,
        depth_ratio=depth_ratio
    )


# =============================================================================
# 眉毛测量
# =============================================================================

@dataclass
class BrowMeasure:
    """眉毛测量结果"""
    left_height: float         # 左眉眼距
    right_height: float
    left_height_norm: float
    right_height_norm: float
    height_ratio: float
    left_lift: float           # 抬起量(相对基准)
    right_lift: float


def measure_brow(
    landmarks, w: int, h: int, icd: float,
    baseline_l_height: float = None,
    baseline_r_height: float = None
) -> BrowMeasure:
    """眉毛测量"""
    def brow_eye_dist(left: bool):
        if left:
            brow = pt2d(landmarks[LM.BROW_CENTER_L], w, h)
            eye = pt2d(landmarks[LM.EYE_INNER_L], w, h)
        else:
            brow = pt2d(landmarks[LM.BROW_CENTER_R], w, h)
            eye = pt2d(landmarks[LM.EYE_INNER_R], w, h)
        return abs(brow[1] - eye[1])  # Y方向距离

    l_hgt = brow_eye_dist(True)
    r_hgt = brow_eye_dist(False)

    l_hgt_norm = l_hgt / icd if icd > Thresholds.EPSILON else 0.0
    r_hgt_norm = r_hgt / icd if icd > Thresholds.EPSILON else 0.0
    hgt_ratio = l_hgt / r_hgt if r_hgt > Thresholds.EPSILON else 1.0

    l_lift = (l_hgt - baseline_l_height) / icd if baseline_l_height is not None and icd > Thresholds.EPSILON else 0.0
    r_lift = (r_hgt - baseline_r_height) / icd if baseline_r_height is not None and icd > Thresholds.EPSILON else 0.0

    return BrowMeasure(
        left_height=l_hgt, right_height=r_hgt,
        left_height_norm=l_hgt_norm, right_height_norm=r_hgt_norm,
        height_ratio=hgt_ratio,
        left_lift=l_lift, right_lift=r_lift
    )


# =============================================================================
# 关键帧查找
# =============================================================================

def _fill_nan_curve(arr: np.ndarray) -> np.ndarray:
    """将曲线中的 NaN 用前向填充 + 后向填充补齐，避免后续 max/min 产生 NaN。"""
    if arr.size == 0:
        return arr
    if np.isnan(arr).all():
        return np.zeros_like(arr)
    # 前向填充
    for i in range(1, len(arr)):
        if np.isnan(arr[i]):
            arr[i] = arr[i - 1]
    # 后向填充（补齐开头 NaN）
    for i in range(len(arr) - 2, -1, -1):
        if np.isnan(arr[i]):
            arr[i] = arr[i + 1]
    arr[np.isnan(arr)] = 0.0
    return arr


def compute_ear_curve(landmarks_seq, w: int, h: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算左右眼 EAR 曲线（用于眨眼事件检测/可视化）。

    注意：
    - landmarks_seq 中可能存在 None（该帧未检测到人脸），这里会用 NaN 占位，
      再做前/后向填充，保证输出曲线全为有效数值，避免 max()/min() 变成 NaN。
    """
    n = len(landmarks_seq)
    l_curve = np.full(n, np.nan, dtype=np.float32)
    r_curve = np.full(n, np.nan, dtype=np.float32)

    for i, lm in enumerate(landmarks_seq):
        if lm is None:
            continue
        l_curve[i] = compute_ear(lm, w, h, True)
        r_curve[i] = compute_ear(lm, w, h, False)

    l_curve = _fill_nan_curve(l_curve)
    r_curve = _fill_nan_curve(r_curve)
    return l_curve, r_curve

def find_max_ear_frame(landmarks_seq, w: int, h: int) -> Tuple[int, float, float]:
    """找EAR最大帧 (NeutralFace关键帧)"""
    max_ear, max_idx = -1.0, 0
    max_l, max_r = 0.0, 0.0
    for i, lm in enumerate(landmarks_seq):
        if lm is None:
            continue
        l_ear = compute_ear(lm, w, h, True)
        r_ear = compute_ear(lm, w, h, False)
        avg = (l_ear + r_ear) / 2
        if avg > max_ear:
            max_ear, max_idx = avg, i
            max_l, max_r = l_ear, r_ear
    return max_idx, max_l, max_r


def find_min_ear_frame(landmarks_seq, w: int, h: int) -> Tuple[int, float, float]:
    """找EAR最小帧 (闭眼关键帧)"""
    min_ear, min_idx = float('inf'), 0
    min_l, min_r = 0.0, 0.0
    for i, lm in enumerate(landmarks_seq):
        if lm is None:
            continue
        l_ear = compute_ear(lm, w, h, True)
        r_ear = compute_ear(lm, w, h, False)
        avg = (l_ear + r_ear) / 2
        if avg < min_ear:
            min_ear, min_idx = avg, i
            min_l, min_r = l_ear, r_ear
    return min_idx, min_l, min_r


def compute_openness_curve(
    landmarks_seq, w: int, h: int,
    baseline_l_area: float, baseline_r_area: float,
    baseline_l_palp_len: float, baseline_r_palp_len: float
) -> Tuple[np.ndarray, np.ndarray]:
    """计算睁眼度曲线 (用于可视化)"""
    n = len(landmarks_seq)
    l_open = np.zeros(n)
    r_open = np.zeros(n)

    for i, lm in enumerate(landmarks_seq):
        if lm is None:
            continue
        icd = compute_icd(lm, w, h)
        l_m = measure_eye(lm, w, h, icd, True, baseline_l_area, baseline_l_palp_len)
        r_m = measure_eye(lm, w, h, icd, False, baseline_r_area, baseline_r_palp_len)
        l_open[i] = l_m.openness
        r_open[i] = r_m.openness

    return l_open, r_open