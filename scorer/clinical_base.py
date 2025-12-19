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
    LIP_TOP = 13  # 上唇中心
    LIP_BOT = 14  # 下唇中心
    LIP_TOP_CENTER = 0
    LIP_BOT_CENTER = 17

    # 嘴唇轮廓
    OUTER_LIP = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
                 375, 321, 405, 314, 17, 84, 181, 91, 146]
    INNER_LIP = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
                 415, 310, 311, 312, 13, 82, 81, 80, 191]

    # ========== 口角角度计算专用 (Oral Commissure Lift论文) ==========
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
    NOSE_ALA_L = 129  # 左鼻翼
    NOSE_ALA_R = 358  # 右鼻翼
    NOSE_TIP_TOP = 6

    # ========== 面颊 ==========
    CHEEK_L = [425, 426, 427, 411, 280]
    CHEEK_R = [205, 206, 207, 187, 50]


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
    """计算眉毛高度 (眉毛到眼睛内眦的距离)"""
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


def compute_mouth_metrics(landmarks, w: int, h: int) -> Dict[str, float]:
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


def compute_nlf_length(landmarks, w: int, h: int, left: bool = True) -> float:
    """计算鼻唇沟长度"""
    if left:
        ala = pt2d(landmarks[LM.NOSE_ALA_L], w, h)
        corner = pt2d(landmarks[LM.MOUTH_L], w, h)
    else:
        ala = pt2d(landmarks[LM.NOSE_ALA_R], w, h)
        corner = pt2d(landmarks[LM.MOUTH_R], w, h)
    return dist(ala, corner)


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
        """从视频提取landmarks序列"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if end_frame is None or end_frame >= total_frames:
            end_frame = total_frames - 1

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        landmarks_seq = []
        frames_seq = []
        current_frame = start_frame

        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            landmarks = self.extract_from_frame(frame)
            landmarks_seq.append(landmarks)
            frames_seq.append(frame)
            current_frame += 1

        cap.release()
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

        return result


def extract_common_indicators(landmarks, w: int, h: int, result: ActionResult) -> None:
    """提取通用指标到ActionResult"""
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

    mouth = compute_mouth_metrics(landmarks, w, h)
    result.mouth_width = mouth["width"]
    result.mouth_height = mouth["height"]

    result.oral_angle = compute_oral_angle(landmarks, w, h)

    result.left_nlf_length = compute_nlf_length(landmarks, w, h, left=True)
    result.right_nlf_length = compute_nlf_length(landmarks, w, h, left=False)
    result.nlf_ratio = result.left_nlf_length / result.right_nlf_length if result.right_nlf_length > 1e-9 else 1.0