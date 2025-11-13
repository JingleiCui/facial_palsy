"""
几何计算工具模块 - 计算原始指标，最后统一归一化
"""
import numpy as np
import cv2
from scipy.spatial import distance as dist
from scipy.spatial import procrustes
from math import acos, degrees, hypot


# MediaPipe关键点索引
EYE_INNER_CANTHUS_LEFT = 362
EYE_INNER_CANTHUS_RIGHT = 133
EYE_OUTER_CANTHUS_LEFT = 263
EYE_OUTER_CANTHUS_RIGHT = 33
EYE_TOP_LEFT = 159
EYE_BOTTOM_LEFT = 145
EYE_TOP_RIGHT = 386
EYE_BOTTOM_RIGHT = 374
PUPIL_LEFT = 473
PUPIL_RIGHT = 468

EYEBROW_LEFT_POINTS = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
EYEBROW_RIGHT_POINTS = [107, 66, 105, 63, 70, 46, 53, 52, 65, 55]

MOUTH_CORNER_LEFT = 291
MOUTH_CORNER_RIGHT = 61
UPPER_LIP_CENTER = 0
UPPER_LIP_BOTTOM_CENTER = 13
LOWER_LIP_TOP_CENTER = 14
LOWER_LIP_BOTTOM_CENTER = 17

UPPER_LIP_LEFT_POINTS = [13, 312, 311, 310, 415, 292, 291, 409, 270, 269, 267, 0, 11, 12]
UPPER_LIP_RIGHT_POINTS = [0, 37, 39, 40, 185, 61, 62, 191, 80, 81, 82, 13, 12, 11]
LOWER_LIP_LEFT_POINTS = [17, 314, 405, 321, 375, 306, 308, 324, 318, 402, 317, 14, 15, 16]
LOWER_LIP_RIGHT_POINTS = [14, 87, 178, 88, 95, 78, 76, 146, 91, 181, 84, 17, 16, 15]

LEFT_EYE_POINTS = [398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381]
RIGHT_EYE_POINTS = [173, 157, 158, 159, 160, 161, 246, 33, 7, 163, 144, 145, 153, 154]

FOREHEAD = 10
CHIN = 152
NOSE_TIP = 4


def get_point(landmarks, index, w, h):
    """获取单个关键点坐标"""
    return (int(landmarks[index].x * w), int(landmarks[index].y * h))


def get_points(landmarks, indices, w, h):
    """批量获取关键点坐标"""
    return [get_point(landmarks, idx, w, h) for idx in indices]


def get_unit_length(landmarks, w, h):
    """计算单位长度（两眼内眦距离）"""
    left_inner = get_point(landmarks, EYE_INNER_CANTHUS_LEFT, w, h)
    right_inner = get_point(landmarks, EYE_INNER_CANTHUS_RIGHT, w, h)
    return dist.euclidean(left_inner, right_inner)


def get_centroid(points):
    """计算质心"""
    if not points:
        return (0, 0)
    x, y = zip(*points)
    return (sum(x) / len(x), sum(y) / len(y))


# 眼部
def get_eye_opening(landmarks, w, h, left=True):
    """眼睛开合度"""
    if left:
        top = get_point(landmarks, EYE_TOP_LEFT, w, h)
        bottom = get_point(landmarks, EYE_BOTTOM_LEFT, w, h)
    else:
        top = get_point(landmarks, EYE_TOP_RIGHT, w, h)
        bottom = get_point(landmarks, EYE_BOTTOM_RIGHT, w, h)
    return dist.euclidean(top, bottom)


def get_eye_length(landmarks, w, h, left=True):
    """眼睛长度"""
    if left:
        inner = get_point(landmarks, EYE_INNER_CANTHUS_LEFT, w, h)
        outer = get_point(landmarks, EYE_OUTER_CANTHUS_LEFT, w, h)
    else:
        inner = get_point(landmarks, EYE_INNER_CANTHUS_RIGHT, w, h)
        outer = get_point(landmarks, EYE_OUTER_CANTHUS_RIGHT, w, h)
    return dist.euclidean(inner, outer)


def get_eye_area(landmarks, w, h, left=True):
    """眼裂面积"""
    points_indices = LEFT_EYE_POINTS if left else RIGHT_EYE_POINTS
    eye_points = get_points(landmarks, points_indices, w, h)
    contour = np.array(eye_points, dtype=np.int32)
    return cv2.contourArea(cv2.convexHull(contour))


# 眉毛
def get_eyebrow_center(landmarks, w, h, left=True):
    """眉毛中心"""
    indices = EYEBROW_LEFT_POINTS if left else EYEBROW_RIGHT_POINTS
    eyebrow_points = get_points(landmarks, indices, w, h)
    return get_centroid(eyebrow_points)


def get_eyebrow_eye_distance(landmarks, w, h, left=True):
    """眉眼距"""
    eyebrow_center = get_eyebrow_center(landmarks, w, h, left=left)
    eye_canthus_idx = EYE_INNER_CANTHUS_LEFT if left else EYE_INNER_CANTHUS_RIGHT
    eye_canthus = get_point(landmarks, eye_canthus_idx, w, h)
    return dist.euclidean(eyebrow_center, eye_canthus)


# 嘴部
def get_mouth_width(landmarks, w, h):
    """嘴角宽度"""
    left_corner = get_point(landmarks, MOUTH_CORNER_LEFT, w, h)
    right_corner = get_point(landmarks, MOUTH_CORNER_RIGHT, w, h)
    return dist.euclidean(left_corner, right_corner)


def get_mouth_height(landmarks, w, h):
    """嘴部高度"""
    upper = get_point(landmarks, UPPER_LIP_BOTTOM_CENTER, w, h)
    lower = get_point(landmarks, LOWER_LIP_TOP_CENTER, w, h)
    return dist.euclidean(upper, lower)


def get_mouth_area(landmarks, w, h):
    """嘴部面积"""
    points = get_points(landmarks, UPPER_LIP_LEFT_POINTS + UPPER_LIP_RIGHT_POINTS, w, h)
    contour = np.array(points, dtype=np.int32)
    return cv2.contourArea(cv2.convexHull(contour))


def get_lip_area(landmarks, w, h, upper=True, left=True):
    """单侧唇部面积"""
    if upper:
        points_indices = UPPER_LIP_LEFT_POINTS if left else UPPER_LIP_RIGHT_POINTS
    else:
        points_indices = LOWER_LIP_LEFT_POINTS if left else LOWER_LIP_RIGHT_POINTS
    lip_points = get_points(landmarks, points_indices, w, h)
    contour = np.array(lip_points, dtype=np.int32)
    return cv2.contourArea(cv2.convexHull(contour))


# 角度
def get_horizontal_line(landmarks, w, h):
    """水平参考线"""
    pupil_left = get_point(landmarks, PUPIL_LEFT, w, h)
    pupil_right = get_point(landmarks, PUPIL_RIGHT, w, h)
    return (pupil_left, pupil_right)


def get_vertical_line(landmarks, w, h):
    """垂直参考线"""
    pupil_left = get_point(landmarks, PUPIL_LEFT, w, h)
    pupil_right = get_point(landmarks, PUPIL_RIGHT, w, h)
    mid_x = (pupil_left[0] + pupil_right[0]) / 2
    mid_y = (pupil_left[1] + pupil_right[1]) / 2
    dx = pupil_right[0] - pupil_left[0]
    dy = pupil_right[1] - pupil_left[1]
    magnitude = (dx**2 + dy**2)**0.5
    perp_dx = dy / magnitude
    perp_dy = -dx / magnitude
    half_length = 500
    x1 = int(mid_x - perp_dx * half_length)
    y1 = int(mid_y - perp_dy * half_length)
    x2 = int(mid_x + perp_dx * half_length)
    y2 = int(mid_y + perp_dy * half_length)
    return ((x1, y1), (x2, y2))


def get_horizontal_angle(h_line, point1, point2):
    """计算与水平线夹角"""
    vector_hline = (h_line[1][0] - h_line[0][0], h_line[1][1] - h_line[0][1])
    vector_points = (point2[0] - point1[0], point2[1] - point1[1])
    dot_product = (vector_points[0] * vector_hline[0] + vector_points[1] * vector_hline[1])
    cross_product = (vector_points[0] * vector_hline[1] - vector_points[1] * vector_hline[0])
    magnitude_points = hypot(vector_points[0], vector_points[1])
    magnitude_hline = hypot(vector_hline[0], vector_hline[1])
    cos_theta = dot_product / (magnitude_points * magnitude_hline + 1e-6)
    cos_theta = min(1, max(-1, cos_theta))
    angle_rad = acos(cos_theta)
    angle_deg = degrees(angle_rad)
    if cross_product < 0:
        angle_deg = -angle_deg
    return angle_deg


# 对称性
def get_shape_disparity(points1, points2):
    """Procrustes形状差异"""
    points1 = np.array(points1)
    points2 = np.array(points2)
    if points1.shape != points2.shape:
        return 0.0
    try:
        _, _, disparity = procrustes(points1, points2)
        return disparity
    except:
        return 0.0


# 时序特征
def compute_motion_range(values):
    """运动范围"""
    if not values:
        return 0.0
    return max(values) - min(values)


def compute_mean_velocity(values, fps):
    """平均速度"""
    if len(values) < 2:
        return 0.0
    diffs = np.abs(np.diff(values))
    return np.mean(diffs) * fps


def compute_max_velocity(values, fps):
    """最大速度"""
    if len(values) < 2:
        return 0.0
    diffs = np.abs(np.diff(values))
    return np.max(diffs) * fps


def compute_smoothness(values):
    """平滑度（jerk）"""
    if len(values) < 4:
        return 0.0
    jerk = np.diff(np.diff(np.diff(values)))
    return np.mean(np.abs(jerk))


# 统一归一化
def normalize_indicators(indicators, unit_length):
    """
    统一归一化所有指标

    Args:
        indicators: dict，原始指标
        unit_length: float，单位长度

    Returns:
        dict: 归一化后的指标
    """
    if unit_length <= 0:
        return {k: 0.0 for k in indicators.keys()}

    normalized = {}
    for key, value in indicators.items():
        if value is None:
            normalized[key] = 0.0
        elif isinstance(value, (int, float)):
            if 'area' in key.lower():
                normalized[key] = value / (unit_length ** 2)
            elif any(x in key.lower() for x in ['angle', 'ratio', 'disparity', '_pct', 'percentage']):
                normalized[key] = value
            else:
                normalized[key] = value / unit_length
        else:
            normalized[key] = value
    return normalized