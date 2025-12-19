#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
临床分级测试脚本 - Clinical Grading Test Script
=============================================================

修正内容:
1. 口角角度计算: 直接判断点在EF线上方/下方，确保符号正确
   - A点在EF线下方 → AOE为负(下垂)
   - A点在EF线上方 → AOE为正(上提)
2. 静息帧选取: 使用min(left_ear, right_ear)确保两眼都睁开
3. 完整Sunnybrook评分:
   - Resting Symmetry (静态对称性)
   - Symmetry of Voluntary Movement (主动运动对称性)
   - Synkinesis (联动运动)
4. 新增C047/C062嘴部倾斜指标

使用方法:
    在PyCharm中直接运行此脚本
"""

import os
import sys
import sqlite3
import json
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import math

# =============================================================================
# 配置参数 - 根据您的环境修改这些路径
# =============================================================================

DATABASE_PATH = r"facialPalsy.db"
MEDIAPIPE_MODEL_PATH = r"/Users/cuijinglei/PycharmProjects/medicalProject/models/face_landmarker.task"
OUTPUT_DIR = r"/Users/cuijinglei/Documents/facialPalsy/HGFA/clinical_grading_output"
PATIENT_LIMIT = None
TARGET_EXAM_ID = None


# =============================================================================
# Landmark 索引定义 (基于MediaPipe 478点)
# =============================================================================

class LM:
    """MediaPipe FaceLandmarker 478点索引"""

    # ========== 眼部 ==========
    EYE_INNER_L = 362
    EYE_INNER_R = 133
    EYE_OUTER_L = 263
    EYE_OUTER_R = 33

    EYE_TOP_L = 386
    EYE_BOT_L = 374
    EYE_TOP_R = 159
    EYE_BOT_R = 145

    EYE_CONTOUR_L = [263, 466, 388, 387, 386, 385, 384, 398,
                     362, 382, 381, 380, 374, 373, 390, 249]
    EYE_CONTOUR_R = [33, 246, 161, 160, 159, 158, 157, 173,
                     133, 155, 154, 153, 145, 144, 163, 7]

    EAR_L = [263, 386, 387, 362, 373, 374]
    EAR_R = [33, 159, 158, 133, 144, 145]

    # ========== 眉毛 ==========
    BROW_L = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
    BROW_R = [107, 66, 105, 63, 70, 46, 53, 52, 65, 55]
    BROW_CENTER_L = 282
    BROW_CENTER_R = 52
    BROW_INNER_L = 336
    BROW_INNER_R = 107

    # ========== 嘴部 (常规测量用) ==========
    MOUTH_L = 291
    MOUTH_R = 61
    LIP_TOP = 13
    LIP_BOT = 14
    LIP_TOP_CENTER = 0
    LIP_BOT_CENTER = 17

    # ========== 口角角度计算专用 (Oral Commissure Lift论文) ==========
    ORAL_CORNER_R = 78  # A点: 右嘴角
    ORAL_CORNER_L = 308  # B点: 左嘴角
    LIP_PEAK_R = 37  # C点: 右上唇峰
    LIP_PEAK_L = 267  # D点: 左上唇峰
    ORAL_E = 82  # E点: mouth orifice右侧
    ORAL_F = 312  # F点: mouth orifice左侧
    ORAL_O = 13  # O点参考

    # ========== 面中线参考点 ==========
    FOREHEAD = 10
    NOSE_TIP = 4
    CHIN = 152

    # ========== 鼻部 ==========
    NOSE_ALA_L = 129
    NOSE_ALA_R = 358

    # ========== 面颊 ==========
    CHEEK_L = [425, 426, 427, 411, 280]
    CHEEK_R = [205, 206, 207, 187, 50]


# =============================================================================
# 数据库操作函数
# =============================================================================

def db_fetch_examinations(db_path: str, target_exam_id: Optional[str] = None,
                          limit: Optional[int] = None) -> List[Dict[str, Any]]:
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
            "action_name_en": action_en, "action_name_cn": action_cn,
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
# 几何计算函数
# =============================================================================

def pt2d(lm, w: int, h: int) -> Tuple[float, float]:
    return (lm.x * w, lm.y * h)


def pts2d(landmarks, indices: List[int], w: int, h: int) -> np.ndarray:
    return np.array([pt2d(landmarks[i], w, h) for i in indices])


def dist(p1, p2) -> float:
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def polygon_area(points: np.ndarray) -> float:
    n = len(points)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i, 0] * points[j, 1]
        area -= points[j, 0] * points[i, 1]
    return abs(area) / 2.0


def compute_icd(landmarks, w: int, h: int) -> float:
    l_inner = pt2d(landmarks[LM.EYE_INNER_L], w, h)
    r_inner = pt2d(landmarks[LM.EYE_INNER_R], w, h)
    return dist(l_inner, r_inner)


def compute_eye_area(landmarks, w: int, h: int, left: bool = True) -> Tuple[float, np.ndarray]:
    contour = LM.EYE_CONTOUR_L if left else LM.EYE_CONTOUR_R
    points = pts2d(landmarks, contour, w, h)
    return polygon_area(points), points


def compute_ear(landmarks, w: int, h: int, left: bool = True) -> float:
    pts_idx = LM.EAR_L if left else LM.EAR_R
    p = [pt2d(landmarks[i], w, h) for i in pts_idx]
    v1 = dist(p[1], p[5])
    v2 = dist(p[2], p[4])
    h_dist = dist(p[0], p[3])
    if h_dist < 1e-9:
        return 0.0
    return (v1 + v2) / (2.0 * h_dist)


def compute_palpebral_height(landmarks, w: int, h: int, left: bool = True) -> float:
    if left:
        top = pt2d(landmarks[LM.EYE_TOP_L], w, h)
        bot = pt2d(landmarks[LM.EYE_BOT_L], w, h)
    else:
        top = pt2d(landmarks[LM.EYE_TOP_R], w, h)
        bot = pt2d(landmarks[LM.EYE_BOT_R], w, h)
    return dist(top, bot)


def compute_brow_height(landmarks, w: int, h: int, left: bool = True) -> float:
    """计算眉毛高度 (眉毛到眼睛内眦的距离)"""
    if left:
        brow = pt2d(landmarks[LM.BROW_CENTER_L], w, h)
        eye = pt2d(landmarks[LM.EYE_INNER_L], w, h)
    else:
        brow = pt2d(landmarks[LM.BROW_CENTER_R], w, h)
        eye = pt2d(landmarks[LM.EYE_INNER_R], w, h)
    # 主要关注垂直距离
    return abs(brow[1] - eye[1])


def point_to_line_signed_distance(point: Tuple[float, float],
                                  line_p1: Tuple[float, float],
                                  line_p2: Tuple[float, float]) -> float:
    """
    计算点到直线的有符号距离

    在图像坐标系中(Y轴向下):
    - 返回正值: 点在直线下方
    - 返回负值: 点在直线上方

    直线由line_p1和line_p2定义
    """
    # 直线方向向量
    dx = line_p2[0] - line_p1[0]
    dy = line_p2[1] - line_p1[1]

    line_len = math.sqrt(dx * dx + dy * dy)
    if line_len < 1e-9:
        return 0.0

    # 从line_p1到point的向量
    px = point[0] - line_p1[0]
    py = point[1] - line_p1[1]

    # 叉积: dx * py - dy * px
    # 在图像坐标系中，如果叉积为正，点在直线的"右边"（顺时针方向）
    # 对于从左到右的水平线，"右边"就是下方
    cross = dx * py - dy * px

    # 返回有符号距离（正值=下方，负值=上方）
    return cross / line_len


# =============================================================================
# 口角角度计算 - 基于 "Oral Commissure Lift" 论文
# =============================================================================

@dataclass
class OralAngleMeasure:
    """口角角度测量结果

    符号约定:
    - 正值: 口角上提
    - 负值: 口角下垂
    """
    A: Tuple[float, float]  # 右嘴角 (landmark 78)
    B: Tuple[float, float]  # 左嘴角 (landmark 308)
    C: Tuple[float, float]  # 右上唇峰 (landmark 37)
    D: Tuple[float, float]  # 左上唇峰 (landmark 267)
    E: Tuple[float, float]  # mouth orifice右侧 (landmark 82)
    F: Tuple[float, float]  # mouth orifice左侧 (landmark 312)
    O: Tuple[float, float]  # E和F的中点

    AOE_angle: float  # 右口角角度: 负值=下垂, 正值=上提
    BOF_angle: float  # 左口角角度: 负值=下垂, 正值=上提

    angle_diff: float  # 角度差 (左-右)
    angle_asymmetry: float  # 不对称性 |左-右|

    # 调试信息
    A_dist_to_EF: float = 0.0  # A点到EF线的有符号距离
    B_dist_to_EF: float = 0.0  # B点到EF线的有符号距离


def compute_oral_angle(landmarks, w: int, h: int) -> OralAngleMeasure:
    """
    计算口角角度

    方法:
    1. E点(82)和F点(312)定义mouth orientation (水平参考线EF)
    2. O点 = E和F的中点
    3. 计算A点到EF线的有符号距离，转换为角度
    4. 计算B点到EF线的有符号距离，转换为角度

    符号约定 (图像坐标系，Y轴向下):
    - 点在EF线下方: 距离为正 → 角度为负(下垂)
    - 点在EF线上方: 距离为负 → 角度为正(上提)
    """
    A = pt2d(landmarks[LM.ORAL_CORNER_R], w, h)  # 右嘴角
    B = pt2d(landmarks[LM.ORAL_CORNER_L], w, h)  # 左嘴角
    C = pt2d(landmarks[LM.LIP_PEAK_R], w, h)
    D = pt2d(landmarks[LM.LIP_PEAK_L], w, h)
    E = pt2d(landmarks[LM.ORAL_E], w, h)
    F = pt2d(landmarks[LM.ORAL_F], w, h)

    O = ((E[0] + F[0]) / 2.0, (E[1] + F[1]) / 2.0)

    # 计算A点和B点到EF线的有符号距离
    # 注意：E在图像中偏左(被摄者右侧)，F在图像中偏右(被摄者左侧)
    A_dist = point_to_line_signed_distance(A, E, F)
    B_dist = point_to_line_signed_distance(B, E, F)

    # 计算从O到A/B的水平距离作为参考基准
    OA_horiz = abs(A[0] - O[0])
    OB_horiz = abs(B[0] - O[0])

    # 转换为角度: arctan(垂直距离 / 水平距离)
    # 距离为正(点在下方) → 角度为负(下垂)
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
        AOE_angle=AOE_angle,
        BOF_angle=BOF_angle,
        angle_diff=BOF_angle - AOE_angle,
        angle_asymmetry=abs(BOF_angle - AOE_angle),
        A_dist_to_EF=A_dist,
        B_dist_to_EF=B_dist
    )


# =============================================================================
# C047/C062: 嘴部倾斜度
# =============================================================================

@dataclass
class MouthTiltMeasure:
    lip_vertical_tilt: float  # C047
    mouth_horizontal_level: float  # C062
    lip_top_center: Tuple[float, float]
    lip_bot_center: Tuple[float, float]
    midline_top: Tuple[float, float]
    midline_bot: Tuple[float, float]
    mouth_left: Tuple[float, float]
    mouth_right: Tuple[float, float]


def compute_mouth_tilt(landmarks, w: int, h: int) -> MouthTiltMeasure:
    lip_top = pt2d(landmarks[LM.LIP_TOP_CENTER], w, h)
    lip_bot = pt2d(landmarks[LM.LIP_BOT_CENTER], w, h)
    forehead = pt2d(landmarks[LM.FOREHEAD], w, h)
    chin = pt2d(landmarks[LM.CHIN], w, h)
    mouth_l = pt2d(landmarks[LM.MOUTH_L], w, h)
    mouth_r = pt2d(landmarks[LM.MOUTH_R], w, h)

    # C047: 唇部中线与面中线夹角
    lip_midline = (lip_top[0] - lip_bot[0], lip_top[1] - lip_bot[1])
    face_midline = (forehead[0] - chin[0], forehead[1] - chin[1])

    dot = lip_midline[0] * face_midline[0] + lip_midline[1] * face_midline[1]
    mag1 = math.sqrt(lip_midline[0] ** 2 + lip_midline[1] ** 2)
    mag2 = math.sqrt(face_midline[0] ** 2 + face_midline[1] ** 2)

    if mag1 > 1e-9 and mag2 > 1e-9:
        cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        lip_vertical_tilt = math.degrees(math.acos(cos_angle))
        cross = face_midline[0] * lip_midline[1] - face_midline[1] * lip_midline[0]
        if cross < 0:
            lip_vertical_tilt = -lip_vertical_tilt
    else:
        lip_vertical_tilt = 0.0

    # C062: 嘴角连线与水平线夹角
    mouth_line = (mouth_l[0] - mouth_r[0], mouth_l[1] - mouth_r[1])
    face_horizontal = (-face_midline[1], face_midline[0])

    dot2 = mouth_line[0] * face_horizontal[0] + mouth_line[1] * face_horizontal[1]
    mag3 = math.sqrt(mouth_line[0] ** 2 + mouth_line[1] ** 2)
    mag4 = math.sqrt(face_horizontal[0] ** 2 + face_horizontal[1] ** 2)

    if mag3 > 1e-9 and mag4 > 1e-9:
        cos_angle2 = max(-1.0, min(1.0, dot2 / (mag3 * mag4)))
        mouth_horizontal_level = math.degrees(math.acos(cos_angle2))
        if mouth_horizontal_level > 90:
            mouth_horizontal_level = 180 - mouth_horizontal_level
    else:
        mouth_horizontal_level = 0.0

    return MouthTiltMeasure(
        lip_vertical_tilt=lip_vertical_tilt,
        mouth_horizontal_level=mouth_horizontal_level,
        lip_top_center=lip_top, lip_bot_center=lip_bot,
        midline_top=forehead, midline_bot=chin,
        mouth_left=mouth_l, mouth_right=mouth_r
    )


# =============================================================================
# 综合指标提取
# =============================================================================

@dataclass
class FacialIndicators:
    icd: float
    left_eye_area: float
    right_eye_area: float
    eye_area_ratio: float
    left_ear: float
    right_ear: float
    left_palpebral_height: float
    right_palpebral_height: float
    palpebral_height_ratio: float
    left_brow_height: float
    right_brow_height: float
    brow_height_ratio: float
    oral_angle: OralAngleMeasure
    mouth_tilt: MouthTiltMeasure
    mouth_width: float
    mouth_height: float
    left_nlf_length: float
    right_nlf_length: float
    nlf_ratio: float


def extract_indicators(landmarks, w: int, h: int) -> FacialIndicators:
    icd = compute_icd(landmarks, w, h)

    left_eye_area, _ = compute_eye_area(landmarks, w, h, left=True)
    right_eye_area, _ = compute_eye_area(landmarks, w, h, left=False)
    eye_area_ratio = left_eye_area / right_eye_area if right_eye_area > 1e-9 else 1.0

    left_ear = compute_ear(landmarks, w, h, left=True)
    right_ear = compute_ear(landmarks, w, h, left=False)

    left_palp_h = compute_palpebral_height(landmarks, w, h, left=True)
    right_palp_h = compute_palpebral_height(landmarks, w, h, left=False)
    palp_h_ratio = left_palp_h / right_palp_h if right_palp_h > 1e-9 else 1.0

    left_brow_h = compute_brow_height(landmarks, w, h, left=True)
    right_brow_h = compute_brow_height(landmarks, w, h, left=False)
    brow_h_ratio = left_brow_h / right_brow_h if right_brow_h > 1e-9 else 1.0

    oral_angle = compute_oral_angle(landmarks, w, h)
    mouth_tilt = compute_mouth_tilt(landmarks, w, h)

    l_corner = pt2d(landmarks[LM.MOUTH_L], w, h)
    r_corner = pt2d(landmarks[LM.MOUTH_R], w, h)
    mouth_width = dist(l_corner, r_corner)

    lip_top = pt2d(landmarks[LM.LIP_TOP], w, h)
    lip_bot = pt2d(landmarks[LM.LIP_BOT], w, h)
    mouth_height = dist(lip_top, lip_bot)

    l_ala = pt2d(landmarks[LM.NOSE_ALA_L], w, h)
    r_ala = pt2d(landmarks[LM.NOSE_ALA_R], w, h)
    left_nlf = dist(l_ala, l_corner)
    right_nlf = dist(r_ala, r_corner)
    nlf_ratio = left_nlf / right_nlf if right_nlf > 1e-9 else 1.0

    return FacialIndicators(
        icd=icd,
        left_eye_area=left_eye_area, right_eye_area=right_eye_area,
        eye_area_ratio=eye_area_ratio,
        left_ear=left_ear, right_ear=right_ear,
        left_palpebral_height=left_palp_h, right_palpebral_height=right_palp_h,
        palpebral_height_ratio=palp_h_ratio,
        left_brow_height=left_brow_h, right_brow_height=right_brow_h,
        brow_height_ratio=brow_h_ratio,
        oral_angle=oral_angle, mouth_tilt=mouth_tilt,
        mouth_width=mouth_width, mouth_height=mouth_height,
        left_nlf_length=left_nlf, right_nlf_length=right_nlf,
        nlf_ratio=nlf_ratio
    )


# =============================================================================
# Sunnybrook 评分系统 - 完整实现
# =============================================================================

@dataclass
class RestingSymmetry:
    """静态对称性评估 (Sunnybrook Resting Symmetry)"""
    eye_status: str
    eye_score: int  # 0=正常, 1=异常
    eye_detail: str

    cheek_status: str
    cheek_score: int  # 0=正常, 1=轻度, 2=重度
    cheek_detail: str

    mouth_status: str
    mouth_score: int  # 0=正常, 1=异常
    mouth_detail: str

    total_score: int  # 总分 * 5 = Sunnybrook Resting Symmetry Score
    affected_side: str


def compute_resting_symmetry(indicators: FacialIndicators) -> RestingSymmetry:
    """计算静态对称性评估"""
    # ========== 眼 ==========
    eye_ratio = indicators.palpebral_height_ratio

    if 0.85 <= eye_ratio <= 1.15:
        eye_status, eye_score = "正常", 0
        eye_detail = f"睑裂高度比(L/R)={eye_ratio:.3f}，正常范围"
    elif eye_ratio < 0.85:
        eye_status, eye_score = "左侧缩窄", 1
        eye_detail = f"睑裂高度比(L/R)={eye_ratio:.3f}，左眼较小"
    else:
        eye_status, eye_score = "左侧增宽", 1
        eye_detail = f"睑裂高度比(L/R)={eye_ratio:.3f}，左眼较大"

    # ========== 颊 ==========
    nlf_ratio = indicators.nlf_ratio

    if 0.85 <= nlf_ratio <= 1.15:
        cheek_status, cheek_score = "正常", 0
        cheek_detail = f"鼻唇沟比(L/R)={nlf_ratio:.3f}，对称"
    elif 0.75 <= nlf_ratio < 0.85:
        cheek_status, cheek_score = "左侧不明显", 1
        cheek_detail = f"鼻唇沟比(L/R)={nlf_ratio:.3f}，左侧较浅"
    elif nlf_ratio > 1.15 and nlf_ratio <= 1.25:
        cheek_status, cheek_score = "右侧不明显", 1
        cheek_detail = f"鼻唇沟比(L/R)={nlf_ratio:.3f}，右侧较浅"
    elif nlf_ratio < 0.75:
        cheek_status, cheek_score = "左侧消失", 2
        cheek_detail = f"鼻唇沟比(L/R)={nlf_ratio:.3f}，左侧明显减弱"
    else:
        cheek_status, cheek_score = "左侧过深", 2
        cheek_detail = f"鼻唇沟比(L/R)={nlf_ratio:.3f}，右侧明显减弱"

    # ========== 嘴 ==========
    oral = indicators.oral_angle
    aoe, bof = oral.AOE_angle, oral.BOF_angle
    angle_diff = bof - aoe

    THRESHOLD = 5.0
    if abs(angle_diff) <= THRESHOLD:
        mouth_status, mouth_score = "正常", 0
        mouth_detail = f"口角差(L-R)={angle_diff:+.1f}°，正常"
    elif angle_diff < -THRESHOLD:
        mouth_status, mouth_score = "左侧口角下垂", 1
        mouth_detail = f"左口角={bof:+.1f}°, 右口角={aoe:+.1f}°"
    else:
        mouth_status, mouth_score = "右侧口角下垂", 1
        mouth_detail = f"左口角={bof:+.1f}°, 右口角={aoe:+.1f}°"

    # 判断患侧
    left_signs = right_signs = 0
    if "左侧" in eye_status:
        if "缩窄" in eye_status:
            left_signs += 1
        else:
            right_signs += 1
    if "左侧" in cheek_status and ("不明显" in cheek_status or "消失" in cheek_status):
        left_signs += 1
    if "右侧" in cheek_status:
        right_signs += 1
    if "左侧" in mouth_status and "下垂" in mouth_status:
        left_signs += 1
    if "右侧" in mouth_status and "下垂" in mouth_status:
        right_signs += 1

    affected_side = "左侧" if left_signs > right_signs else ("右侧" if right_signs > left_signs else "不确定")
    total_score = eye_score + cheek_score + mouth_score

    return RestingSymmetry(
        eye_status=eye_status, eye_score=eye_score, eye_detail=eye_detail,
        cheek_status=cheek_status, cheek_score=cheek_score, cheek_detail=cheek_detail,
        mouth_status=mouth_status, mouth_score=mouth_score, mouth_detail=mouth_detail,
        total_score=total_score, affected_side=affected_side
    )


@dataclass
class VoluntaryMovement:
    """主动运动对称性评估 (Sunnybrook Symmetry of Voluntary Movement)

    评分: 1-5分
    1 = Unable to initiate movement (无法启动运动)
    2 = Initiates slight movement (轻微启动)
    3 = Initiates movement with mild asymmetry (明显启动但不对称)
    4 = Movement almost complete (运动几乎完整)
    5 = Movement complete (运动完整)
    """
    action_name: str
    action_cn: str

    # 测量值
    left_value: float
    right_value: float
    ratio: float  # left/right

    # 评分
    score: int  # 1-5
    interpretation: str

    # 与基线比较
    baseline_value: Optional[float] = None
    excursion_left: Optional[float] = None
    excursion_right: Optional[float] = None


def compute_voluntary_movement_score(ratio: float, excursion_ratio: Optional[float] = None) -> Tuple[int, str]:
    """
    根据左右比值计算主动运动评分

    Args:
        ratio: 左/右比值
        excursion_ratio: 运动幅度比值(可选)

    Returns:
        (score, interpretation)
    """
    # 使用ratio偏离1.0的程度来评分
    deviation = abs(ratio - 1.0)

    if deviation <= 0.05:
        return 5, "运动完整对称"
    elif deviation <= 0.15:
        return 4, "运动几乎完整"
    elif deviation <= 0.30:
        return 3, "运动启动但明显不对称"
    elif deviation <= 0.50:
        return 2, "仅轻微启动运动"
    else:
        return 1, "无法启动运动"


@dataclass
class Synkinesis:
    """联动运动评估 (Sunnybrook Synkinesis)

    评分: 0-3分 (每个表情)
    0 = None (无联动)
    1 = Mild (轻度联动)
    2 = Moderate (中度联动)
    3 = Severe (重度联动/畸形)
    """
    action_name: str

    # 检测到的联动现象
    eye_synkinesis: int  # 眼部联动
    cheek_synkinesis: int  # 面颊联动
    mouth_synkinesis: int  # 嘴部联动

    total_score: int
    interpretation: str


def detect_synkinesis(neutral_indicators: FacialIndicators,
                      action_indicators: FacialIndicators,
                      action_name: str) -> Synkinesis:
    """
    检测联动运动

    通过比较动作时与静息时的非目标区域变化来检测联动
    """
    eye_syn = cheek_syn = mouth_syn = 0

    # 获取基线和当前值
    neutral_ear_l = neutral_indicators.left_ear
    neutral_ear_r = neutral_indicators.right_ear
    action_ear_l = action_indicators.left_ear
    action_ear_r = action_indicators.right_ear

    neutral_nlf = neutral_indicators.nlf_ratio
    action_nlf = action_indicators.nlf_ratio

    neutral_mouth_w = neutral_indicators.mouth_width
    action_mouth_w = action_indicators.mouth_width

    # 根据动作类型检测非预期的联动
    if action_name in ["Smile", "ShowTeeth", "LipPucker"]:
        # 微笑/露齿/撅嘴时，检测眼部联动
        ear_change_l = abs(action_ear_l - neutral_ear_l)
        ear_change_r = abs(action_ear_r - neutral_ear_r)
        avg_ear_change = (ear_change_l + ear_change_r) / 2

        if avg_ear_change > 0.15:
            eye_syn = 3
        elif avg_ear_change > 0.10:
            eye_syn = 2
        elif avg_ear_change > 0.05:
            eye_syn = 1

    elif action_name in ["CloseEyeSoftly", "CloseEyeHardly", "RaiseEyebrow"]:
        # 闭眼/皱眉时，检测嘴部联动
        mouth_change = abs(action_mouth_w - neutral_mouth_w) / neutral_mouth_w if neutral_mouth_w > 1e-9 else 0

        if mouth_change > 0.20:
            mouth_syn = 3
        elif mouth_change > 0.10:
            mouth_syn = 2
        elif mouth_change > 0.05:
            mouth_syn = 1

    total = eye_syn + cheek_syn + mouth_syn

    if total == 0:
        interp = "无联动"
    elif total <= 2:
        interp = "轻度联动"
    elif total <= 4:
        interp = "中度联动"
    else:
        interp = "重度联动"

    return Synkinesis(
        action_name=action_name,
        eye_synkinesis=eye_syn,
        cheek_synkinesis=cheek_syn,
        mouth_synkinesis=mouth_syn,
        total_score=total,
        interpretation=interp
    )


@dataclass
class SunnybrookScore:
    """完整的Sunnybrook评分"""
    # Resting Symmetry (0-20分, = raw_score * 5)
    resting_symmetry: RestingSymmetry
    resting_score: int  # 0-20

    # Voluntary Movement (20-100分, = sum * 4)
    voluntary_movements: List[VoluntaryMovement]
    voluntary_score: int  # 20-100

    # Synkinesis (0-15分)
    synkinesis_list: List[Synkinesis]
    synkinesis_score: int  # 0-15

    # Composite Score
    composite_score: int  # = voluntary - resting - synkinesis


def compute_sunnybrook_score(resting: RestingSymmetry,
                             voluntary_list: List[VoluntaryMovement],
                             synkinesis_list: List[Synkinesis]) -> SunnybrookScore:
    """计算完整的Sunnybrook评分"""
    # Resting Symmetry Score = raw * 5
    resting_score = resting.total_score * 5

    # Voluntary Movement Score = sum * 4
    vol_sum = sum(v.score for v in voluntary_list) if voluntary_list else 25  # 默认满分
    voluntary_score = vol_sum * 4

    # Synkinesis Score = sum
    syn_sum = sum(s.total_score for s in synkinesis_list) if synkinesis_list else 0
    synkinesis_score = syn_sum

    # Composite Score
    composite_score = voluntary_score - resting_score - synkinesis_score

    return SunnybrookScore(
        resting_symmetry=resting,
        resting_score=resting_score,
        voluntary_movements=voluntary_list,
        voluntary_score=voluntary_score,
        synkinesis_list=synkinesis_list,
        synkinesis_score=synkinesis_score,
        composite_score=composite_score
    )


# =============================================================================
# 可视化函数
# =============================================================================

def visualize_oral_angle(frame: np.ndarray, landmarks, w: int, h: int,
                         oral: OralAngleMeasure) -> np.ndarray:
    """可视化口角角度测量"""
    img = frame.copy()

    def draw_point(pt, color, label, offset=(5, -5), radius=6):
        cv2.circle(img, (int(pt[0]), int(pt[1])), radius, color, -1)
        cv2.putText(img, label, (int(pt[0]) + offset[0], int(pt[1]) + offset[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    COLOR_A = (0, 0, 255)  # 红色 - 右嘴角A
    COLOR_B = (255, 0, 0)  # 蓝色 - 左嘴角B
    COLOR_E = (0, 200, 200)  # 黄绿色 - E点
    COLOR_F = (200, 200, 0)  # 青色 - F点
    COLOR_O = (255, 255, 255)  # 白色 - O点
    COLOR_REF = (0, 255, 0)  # 绿色 - 参考线

    # 绘制关键点
    draw_point(oral.A, COLOR_A, "A(R)", (5, -10))
    draw_point(oral.B, COLOR_B, "B(L)", (-35, -10))
    draw_point(oral.E, COLOR_E, "E", (5, 15))
    draw_point(oral.F, COLOR_F, "F", (-20, 15))
    draw_point(oral.O, COLOR_O, "O", (-5, -15), radius=8)

    # 绘制EF水平参考线
    extend = 80
    dx = oral.F[0] - oral.E[0]
    dy = oral.F[1] - oral.E[1]
    line_len = math.sqrt(dx * dx + dy * dy)
    if line_len > 1e-9:
        ux, uy = dx / line_len, dy / line_len
        p1 = (int(oral.E[0] - extend * ux), int(oral.E[1] - extend * uy))
        p2 = (int(oral.F[0] + extend * ux), int(oral.F[1] + extend * uy))
        cv2.line(img, p1, p2, COLOR_REF, 2, cv2.LINE_AA)

    # 绘制O到A的连线
    cv2.line(img, (int(oral.O[0]), int(oral.O[1])),
             (int(oral.A[0]), int(oral.A[1])), COLOR_A, 2)

    # 绘制O到B的连线
    cv2.line(img, (int(oral.O[0]), int(oral.O[1])),
             (int(oral.B[0]), int(oral.B[1])), COLOR_B, 2)

    # 绘制A点到EF线的垂直距离线（调试用）
    # 计算A在EF线上的投影点
    if line_len > 1e-9:
        # 从E到A的向量在EF方向上的投影长度
        EA = (oral.A[0] - oral.E[0], oral.A[1] - oral.E[1])
        proj_len = (EA[0] * ux + EA[1] * uy)
        A_proj = (oral.E[0] + proj_len * ux, oral.E[1] + proj_len * uy)
        cv2.line(img, (int(oral.A[0]), int(oral.A[1])),
                 (int(A_proj[0]), int(A_proj[1])), (0, 0, 255), 1, cv2.LINE_AA)

        EB = (oral.B[0] - oral.E[0], oral.B[1] - oral.E[1])
        proj_len_b = (EB[0] * ux + EB[1] * uy)
        B_proj = (oral.E[0] + proj_len_b * ux, oral.E[1] + proj_len_b * uy)
        cv2.line(img, (int(oral.B[0]), int(oral.B[1])),
                 (int(B_proj[0]), int(B_proj[1])), (255, 0, 0), 1, cv2.LINE_AA)

    # 文字信息
    y = 30
    cv2.putText(img, "Oral Commissure Angle", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y += 30

    # 右侧角度
    status_r = "DOWN" if oral.AOE_angle < -3 else ("UP" if oral.AOE_angle > 3 else "OK")
    aoe_color = (0, 0, 255) if oral.AOE_angle < -3 else ((0, 255, 255) if oral.AOE_angle > 3 else (0, 255, 0))
    cv2.putText(img, f"AOE (R): {oral.AOE_angle:+.2f} deg [{status_r}]", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, aoe_color, 2)
    y += 22

    # 左侧角度
    status_l = "DOWN" if oral.BOF_angle < -3 else ("UP" if oral.BOF_angle > 3 else "OK")
    bof_color = (255, 0, 0) if oral.BOF_angle < -3 else ((0, 255, 255) if oral.BOF_angle > 3 else (0, 255, 0))
    cv2.putText(img, f"BOF (L): {oral.BOF_angle:+.2f} deg [{status_l}]", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, bof_color, 2)
    y += 22

    # 角度差
    diff_color = (0, 0, 255) if abs(oral.angle_diff) > 5 else (255, 255, 255)
    cv2.putText(img, f"Diff (L-R): {oral.angle_diff:+.2f} deg", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, diff_color, 2)
    y += 22

    # 调试信息
    cv2.putText(img, f"A dist to EF: {oral.A_dist_to_EF:+.1f}px", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    y += 18
    cv2.putText(img, f"B dist to EF: {oral.B_dist_to_EF:+.1f}px", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    y += 22

    # 说明
    cv2.putText(img, "Green line: E-F (mouth orientation)", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    y += 15
    cv2.putText(img, "Positive=UP, Negative=DOWN", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return img


def visualize_mouth_tilt(frame: np.ndarray, landmarks, w: int, h: int,
                         tilt: MouthTiltMeasure) -> np.ndarray:
    img = frame.copy()

    cv2.line(img, (int(tilt.midline_top[0]), int(tilt.midline_top[1])),
             (int(tilt.midline_bot[0]), int(tilt.midline_bot[1])), (255, 255, 255), 2)
    cv2.line(img, (int(tilt.lip_top_center[0]), int(tilt.lip_top_center[1])),
             (int(tilt.lip_bot_center[0]), int(tilt.lip_bot_center[1])), (0, 165, 255), 2)
    cv2.line(img, (int(tilt.mouth_left[0]), int(tilt.mouth_left[1])),
             (int(tilt.mouth_right[0]), int(tilt.mouth_right[1])), (255, 255, 0), 2)

    cv2.circle(img, (int(tilt.lip_top_center[0]), int(tilt.lip_top_center[1])), 5, (0, 165, 255), -1)
    cv2.circle(img, (int(tilt.lip_bot_center[0]), int(tilt.lip_bot_center[1])), 5, (0, 165, 255), -1)
    cv2.circle(img, (int(tilt.mouth_left[0]), int(tilt.mouth_left[1])), 5, (255, 255, 0), -1)
    cv2.circle(img, (int(tilt.mouth_right[0]), int(tilt.mouth_right[1])), 5, (255, 255, 0), -1)

    y = 30
    cv2.putText(img, "Mouth Tilt Analysis", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y += 30

    c047_color = (0, 0, 255) if abs(tilt.lip_vertical_tilt) > 5 else (0, 255, 0)
    cv2.putText(img, f"C047 Lip Vertical Tilt: {tilt.lip_vertical_tilt:+.2f} deg", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, c047_color, 2)
    y += 22

    c062_color = (0, 0, 255) if tilt.mouth_horizontal_level > 5 else (0, 255, 0)
    cv2.putText(img, f"C062 Mouth Horizontal: {tilt.mouth_horizontal_level:.2f} deg", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, c062_color, 2)

    return img


def visualize_resting_symmetry(frame: np.ndarray, landmarks, w: int, h: int,
                               indicators: FacialIndicators,
                               resting: RestingSymmetry) -> np.ndarray:
    img = frame.copy()

    # 绘制眼部轮廓
    for idx_list, color in [(LM.EYE_CONTOUR_L, (255, 0, 0)), (LM.EYE_CONTOUR_R, (0, 165, 255))]:
        pts = pts2d(landmarks, idx_list, w, h).astype(np.int32)
        cv2.polylines(img, [pts], True, color, 2)

    # 绘制鼻唇沟
    l_ala = pt2d(landmarks[LM.NOSE_ALA_L], w, h)
    r_ala = pt2d(landmarks[LM.NOSE_ALA_R], w, h)
    l_mouth = pt2d(landmarks[LM.MOUTH_L], w, h)
    r_mouth = pt2d(landmarks[LM.MOUTH_R], w, h)
    cv2.line(img, (int(l_ala[0]), int(l_ala[1])), (int(l_mouth[0]), int(l_mouth[1])), (255, 0, 0), 2)
    cv2.line(img, (int(r_ala[0]), int(r_ala[1])), (int(r_mouth[0]), int(r_mouth[1])), (0, 165, 255), 2)

    # 绘制口角
    oral = indicators.oral_angle
    cv2.circle(img, (int(oral.A[0]), int(oral.A[1])), 6, (0, 0, 255), -1)
    cv2.circle(img, (int(oral.B[0]), int(oral.B[1])), 6, (255, 0, 0), -1)
    cv2.circle(img, (int(oral.O[0]), int(oral.O[1])), 4, (255, 255, 255), -1)

    # 信息面板
    panel_h = 220
    cv2.rectangle(img, (5, 5), (380, panel_h), (0, 0, 0), -1)
    cv2.rectangle(img, (5, 5), (380, panel_h), (255, 255, 255), 1)

    y = 25
    cv2.putText(img, "RESTING SYMMETRY (Sunnybrook)", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    y += 28

    # 眼
    eye_color = (0, 255, 0) if resting.eye_score == 0 else (0, 0, 255)
    cv2.putText(img, f"Eye: {resting.eye_status} (score={resting.eye_score})", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, eye_color, 1)
    y += 20
    cv2.putText(img, f"  {resting.eye_detail}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    y += 22

    # 颊
    cheek_color = (0, 255, 0) if resting.cheek_score == 0 else (
        (0, 165, 255) if resting.cheek_score == 1 else (0, 0, 255))
    cv2.putText(img, f"Cheek: {resting.cheek_status} (score={resting.cheek_score})", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, cheek_color, 1)
    y += 20
    cv2.putText(img, f"  {resting.cheek_detail}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    y += 22

    # 嘴
    mouth_color = (0, 255, 0) if resting.mouth_score == 0 else (0, 0, 255)
    cv2.putText(img, f"Mouth: {resting.mouth_status} (score={resting.mouth_score})", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, mouth_color, 1)
    y += 20
    cv2.putText(img, f"  {resting.mouth_detail}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    y += 25

    # 分隔线
    cv2.line(img, (15, y - 5), (365, y - 5), (100, 100, 100), 1)
    y += 12

    # 总分
    total_color = (0, 255, 0) if resting.total_score == 0 else (0, 0, 255)
    cv2.putText(img, f"Raw Score: {resting.total_score} | SB Score: {resting.total_score * 5}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, total_color, 2)
    y += 22

    # 患侧
    side_color = (0, 255, 255) if resting.affected_side != "不确定" else (128, 128, 128)
    cv2.putText(img, f"Affected Side: {resting.affected_side}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, side_color, 1)

    return img


def visualize_voluntary_movement(frame: np.ndarray, vol: VoluntaryMovement) -> np.ndarray:
    """可视化主动运动评估"""
    img = frame.copy()

    y = 30
    cv2.putText(img, f"Voluntary Movement: {vol.action_name}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y += 30

    cv2.putText(img, f"Left: {vol.left_value:.3f}  Right: {vol.right_value:.3f}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y += 22

    cv2.putText(img, f"Ratio (L/R): {vol.ratio:.3f}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y += 25

    score_color = (0, 255, 0) if vol.score >= 4 else ((0, 165, 255) if vol.score >= 3 else (0, 0, 255))
    cv2.putText(img, f"Score: {vol.score}/5 - {vol.interpretation}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, score_color, 2)

    return img


def visualize_sunnybrook_summary(frame: np.ndarray, sb_score: SunnybrookScore) -> np.ndarray:
    """可视化Sunnybrook总结评分"""
    img = frame.copy()

    # 半透明背景
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (450, 350), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    y = 40
    cv2.putText(img, "SUNNYBROOK FACIAL GRADING", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    y += 40

    # Resting Symmetry
    cv2.putText(img, f"Resting Symmetry Score: {sb_score.resting_score}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    y += 25

    # Voluntary Movement
    cv2.putText(img, f"Voluntary Movement Score: {sb_score.voluntary_score}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    y += 20
    for vol in sb_score.voluntary_movements[:5]:
        cv2.putText(img, f"  {vol.action_name}: {vol.score}/5", (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y += 18
    y += 10

    # Synkinesis
    cv2.putText(img, f"Synkinesis Score: {sb_score.synkinesis_score}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    y += 35

    # Composite Score
    color = (0, 255, 0) if sb_score.composite_score >= 70 else (
        (0, 165, 255) if sb_score.composite_score >= 50 else (0, 0, 255))
    cv2.putText(img, f"COMPOSITE SCORE: {sb_score.composite_score}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    y += 30

    # 公式
    cv2.putText(img, f"= {sb_score.voluntary_score} - {sb_score.resting_score} - {sb_score.synkinesis_score}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    return img


def visualize_all_indicators(frame: np.ndarray, landmarks, w: int, h: int,
                             indicators: FacialIndicators) -> np.ndarray:
    img = frame.copy()

    for idx_list, color in [(LM.EYE_CONTOUR_L, (255, 0, 0)), (LM.EYE_CONTOUR_R, (0, 165, 255))]:
        pts = pts2d(landmarks, idx_list, w, h).astype(np.int32)
        cv2.polylines(img, [pts], True, color, 1)

    for idx, color in [(LM.EYE_INNER_L, (255, 0, 0)), (LM.EYE_INNER_R, (0, 165, 255)),
                       (LM.EYE_OUTER_L, (255, 0, 0)), (LM.EYE_OUTER_R, (0, 165, 255))]:
        pt = pt2d(landmarks[idx], w, h)
        cv2.circle(img, (int(pt[0]), int(pt[1])), 3, color, -1)

    l_mouth = pt2d(landmarks[LM.MOUTH_L], w, h)
    r_mouth = pt2d(landmarks[LM.MOUTH_R], w, h)
    cv2.circle(img, (int(l_mouth[0]), int(l_mouth[1])), 5, (255, 0, 0), -1)
    cv2.circle(img, (int(r_mouth[0]), int(r_mouth[1])), 5, (0, 165, 255), -1)
    cv2.line(img, (int(l_mouth[0]), int(l_mouth[1])),
             (int(r_mouth[0]), int(r_mouth[1])), (255, 255, 255), 1)

    oral = indicators.oral_angle
    cv2.circle(img, (int(oral.A[0]), int(oral.A[1])), 4, (0, 0, 255), -1)
    cv2.circle(img, (int(oral.B[0]), int(oral.B[1])), 4, (255, 0, 0), -1)
    cv2.circle(img, (int(oral.O[0]), int(oral.O[1])), 4, (255, 255, 255), -1)

    l_ala = pt2d(landmarks[LM.NOSE_ALA_L], w, h)
    r_ala = pt2d(landmarks[LM.NOSE_ALA_R], w, h)
    cv2.line(img, (int(l_ala[0]), int(l_ala[1])), (int(l_mouth[0]), int(l_mouth[1])), (255, 0, 0), 2)
    cv2.line(img, (int(r_ala[0]), int(r_ala[1])), (int(r_mouth[0]), int(r_mouth[1])), (0, 165, 255), 2)

    y = 30
    cv2.putText(img, "Facial Indicators", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    y += 30

    cv2.putText(img, f"ICD: {indicators.icd:.1f}px", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    y += 18
    cv2.putText(img, f"Eye Area L/R: {indicators.eye_area_ratio:.3f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1)
    y += 18
    cv2.putText(img, f"Palpebral H L/R: {indicators.palpebral_height_ratio:.3f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1)
    y += 18
    cv2.putText(img, f"EAR L:{indicators.left_ear:.3f} R:{indicators.right_ear:.3f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1)
    y += 18
    cv2.putText(img, f"AOE(R): {oral.AOE_angle:+.1f}  BOF(L): {oral.BOF_angle:+.1f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1)
    y += 18
    tilt = indicators.mouth_tilt
    cv2.putText(img, f"C047: {tilt.lip_vertical_tilt:+.1f}  C062: {tilt.mouth_horizontal_level:.1f}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    y += 18
    cv2.putText(img, f"NLF L/R: {indicators.nlf_ratio:.3f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255),
                1)

    return img


# =============================================================================
# Landmark提取器
# =============================================================================

class LandmarkExtractor:
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
# 峰值帧查找
# =============================================================================

def find_peak_frame_smile(landmarks_seq: List, frames_seq: List, w: int, h: int) -> int:
    """找微笑峰值帧 (嘴宽最大)"""
    max_width = -1.0
    max_idx = 0
    for i, lm in enumerate(landmarks_seq):
        if lm is None:
            continue
        l_corner = pt2d(lm[LM.MOUTH_L], w, h)
        r_corner = pt2d(lm[LM.MOUTH_R], w, h)
        width = dist(l_corner, r_corner)
        if width > max_width:
            max_width = width
            max_idx = i
    return max_idx


def find_peak_frame_neutral(landmarks_seq: List, frames_seq: List, w: int, h: int) -> int:
    """
    找静息峰值帧

    使用min(left_ear, right_ear)确保两只眼睛都睁开
    """
    max_min_ear = -1.0
    max_idx = 0

    for i, lm in enumerate(landmarks_seq):
        if lm is None:
            continue
        l_ear = compute_ear(lm, w, h, True)
        r_ear = compute_ear(lm, w, h, False)
        # 使用min确保两眼都睁开
        min_ear = min(l_ear, r_ear)
        if min_ear > max_min_ear:
            max_min_ear = min_ear
            max_idx = i

    return max_idx


def find_peak_frame_close_eye(landmarks_seq: List, frames_seq: List, w: int, h: int) -> int:
    """找闭眼峰值帧 (EAR最小)"""
    min_ear = float('inf')
    min_idx = 0

    for i, lm in enumerate(landmarks_seq):
        if lm is None:
            continue
        l_ear = compute_ear(lm, w, h, True)
        r_ear = compute_ear(lm, w, h, False)
        avg_ear = (l_ear + r_ear) / 2
        if avg_ear < min_ear:
            min_ear = avg_ear
            min_idx = i

    return min_idx


def find_peak_frame_raise_brow(landmarks_seq: List, frames_seq: List, w: int, h: int) -> int:
    """找皱额峰值帧 (眉毛最高)"""
    max_brow = -1.0
    max_idx = 0

    for i, lm in enumerate(landmarks_seq):
        if lm is None:
            continue
        l_brow = compute_brow_height(lm, w, h, True)
        r_brow = compute_brow_height(lm, w, h, False)
        avg_brow = (l_brow + r_brow) / 2
        if avg_brow > max_brow:
            max_brow = avg_brow
            max_idx = i

    return max_idx


def find_peak_frame_lip_pucker(landmarks_seq: List, frames_seq: List, w: int, h: int) -> int:
    """找撅嘴峰值帧 (嘴部最窄最高)"""
    max_ratio = -1.0
    max_idx = 0

    for i, lm in enumerate(landmarks_seq):
        if lm is None:
            continue
        lip_top = pt2d(lm[LM.LIP_TOP], w, h)
        lip_bot = pt2d(lm[LM.LIP_BOT], w, h)
        l_corner = pt2d(lm[LM.MOUTH_L], w, h)
        r_corner = pt2d(lm[LM.MOUTH_R], w, h)

        height = dist(lip_top, lip_bot)
        width = dist(l_corner, r_corner)

        if width > 1e-9:
            ratio = height / width
            if ratio > max_ratio:
                max_ratio = ratio
                max_idx = i

    return max_idx


PEAK_FINDERS = {
    "NeutralFace": find_peak_frame_neutral,
    "Smile": find_peak_frame_smile,
    "ShowTeeth": find_peak_frame_smile,
    "CloseEyeSoftly": find_peak_frame_close_eye,
    "CloseEyeHardly": find_peak_frame_close_eye,
    "VoluntaryEyeBlink": find_peak_frame_close_eye,
    "SpontaneousEyeBlink": find_peak_frame_close_eye,
    "RaiseEyebrow": find_peak_frame_raise_brow,
    "LipPucker": find_peak_frame_lip_pucker,
    "BlowCheek": find_peak_frame_lip_pucker,
    "ShrugNose": find_peak_frame_neutral,
}


# =============================================================================
# 主处理函数
# =============================================================================

def process_single_action(
        extractor: LandmarkExtractor,
        video_info: Dict[str, Any],
        output_dir: Path,
        action_name: str,
        neutral_indicators: Optional[FacialIndicators] = None
) -> Optional[Dict[str, Any]]:
    """处理单个动作视频"""
    video_path = video_info["file_path"]
    start_frame = video_info.get("start_frame", 0)
    end_frame = video_info.get("end_frame", None)
    fps = video_info.get("fps", 30.0)

    print(f"  处理动作: {action_name}")
    print(f"    视频: {video_path}")

    if not os.path.exists(video_path):
        print(f"    [!] 视频文件不存在!")
        return None

    landmarks_seq, frames_seq = extractor.extract_sequence(video_path, start_frame, end_frame)

    if not landmarks_seq or not frames_seq:
        print(f"    [!] 无法提取landmarks!")
        return None

    h, w = frames_seq[0].shape[:2]
    print(f"    帧数: {len(frames_seq)}, 尺寸: {w}x{h}")

    peak_finder = PEAK_FINDERS.get(action_name, find_peak_frame_neutral)
    peak_idx = peak_finder(landmarks_seq, frames_seq, w, h)

    peak_landmarks = landmarks_seq[peak_idx]
    peak_frame = frames_seq[peak_idx]

    if peak_landmarks is None:
        print(f"    [!] 峰值帧无landmarks!")
        return None

    print(f"    峰值帧: {peak_idx}")

    indicators = extract_indicators(peak_landmarks, w, h)

    action_dir = output_dir / action_name
    action_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(action_dir / "peak_raw.jpg"), peak_frame)

    vis_frame = visualize_all_indicators(peak_frame, peak_landmarks, w, h, indicators)
    cv2.imwrite(str(action_dir / "peak_indicators.jpg"), vis_frame)

    oral_vis = visualize_oral_angle(peak_frame, peak_landmarks, w, h, indicators.oral_angle)
    cv2.imwrite(str(action_dir / "peak_oral_angle.jpg"), oral_vis)

    tilt_vis = visualize_mouth_tilt(peak_frame, peak_landmarks, w, h, indicators.mouth_tilt)
    cv2.imwrite(str(action_dir / "peak_mouth_tilt.jpg"), tilt_vis)

    # Resting Symmetry (仅NeutralFace)
    resting_dict = None
    if action_name == "NeutralFace":
        resting = compute_resting_symmetry(indicators)
        resting_vis = visualize_resting_symmetry(peak_frame, peak_landmarks, w, h, indicators, resting)
        cv2.imwrite(str(action_dir / "resting_symmetry.jpg"), resting_vis)

        resting_dict = {
            "eye": {"status": resting.eye_status, "score": resting.eye_score, "detail": resting.eye_detail},
            "cheek": {"status": resting.cheek_status, "score": resting.cheek_score, "detail": resting.cheek_detail},
            "mouth": {"status": resting.mouth_status, "score": resting.mouth_score, "detail": resting.mouth_detail},
            "total_score": resting.total_score,
            "sunnybrook_resting_score": resting.total_score * 5,
            "affected_side": resting.affected_side
        }

    # Voluntary Movement
    vol_dict = None
    if action_name != "NeutralFace":
        # 根据动作类型选择比较指标
        if action_name in ["Smile", "ShowTeeth"]:
            left_val = indicators.mouth_width  # 简化：使用嘴宽
            right_val = indicators.mouth_width
            ratio = 1.0  # 需要与baseline比较
            if neutral_indicators:
                baseline_width = neutral_indicators.mouth_width
                excursion = indicators.mouth_width - baseline_width
                ratio = indicators.oral_angle.angle_asymmetry  # 用不对称性
        elif action_name in ["CloseEyeSoftly", "CloseEyeHardly"]:
            left_val = indicators.left_ear
            right_val = indicators.right_ear
            ratio = left_val / right_val if right_val > 1e-9 else 1.0
        elif action_name == "RaiseEyebrow":
            left_val = indicators.left_brow_height
            right_val = indicators.right_brow_height
            ratio = left_val / right_val if right_val > 1e-9 else 1.0
        else:
            left_val = 1.0
            right_val = 1.0
            ratio = 1.0

        score, interp = compute_voluntary_movement_score(ratio)

        vol = VoluntaryMovement(
            action_name=action_name,
            action_cn=video_info.get("action_name_cn", action_name),
            left_value=left_val,
            right_value=right_val,
            ratio=ratio,
            score=score,
            interpretation=interp
        )

        vol_vis = visualize_voluntary_movement(peak_frame, vol)
        cv2.imwrite(str(action_dir / "voluntary_movement.jpg"), vol_vis)

        vol_dict = {
            "action_name": vol.action_name,
            "left_value": vol.left_value,
            "right_value": vol.right_value,
            "ratio": vol.ratio,
            "score": vol.score,
            "interpretation": vol.interpretation
        }

    # Synkinesis
    syn_dict = None
    if neutral_indicators and action_name != "NeutralFace":
        syn = detect_synkinesis(neutral_indicators, indicators, action_name)
        syn_dict = {
            "eye_synkinesis": syn.eye_synkinesis,
            "cheek_synkinesis": syn.cheek_synkinesis,
            "mouth_synkinesis": syn.mouth_synkinesis,
            "total_score": syn.total_score,
            "interpretation": syn.interpretation
        }

    # 保存指标JSON
    oral = indicators.oral_angle
    tilt = indicators.mouth_tilt

    indicators_dict = {
        "action_name": action_name,
        "video_path": video_path,
        "peak_frame_idx": peak_idx,
        "total_frames": len(frames_seq),
        "image_size": {"width": w, "height": h},
        "fps": fps,
        "icd": indicators.icd,
        "eye": {
            "left_area": indicators.left_eye_area,
            "right_area": indicators.right_eye_area,
            "area_ratio": indicators.eye_area_ratio,
            "left_ear": indicators.left_ear,
            "right_ear": indicators.right_ear,
            "left_palpebral_height": indicators.left_palpebral_height,
            "right_palpebral_height": indicators.right_palpebral_height,
            "palpebral_height_ratio": indicators.palpebral_height_ratio,
        },
        "brow": {
            "left_height": indicators.left_brow_height,
            "right_height": indicators.right_brow_height,
            "height_ratio": indicators.brow_height_ratio,
        },
        "oral_angle": {
            "AOE_angle_deg": oral.AOE_angle,
            "BOF_angle_deg": oral.BOF_angle,
            "angle_diff": oral.angle_diff,
            "angle_asymmetry": oral.angle_asymmetry,
            "A_dist_to_EF": oral.A_dist_to_EF,
            "B_dist_to_EF": oral.B_dist_to_EF,
            "points": {
                "A_right_corner": list(oral.A),
                "B_left_corner": list(oral.B),
                "E_mouth_orifice_R": list(oral.E),
                "F_mouth_orifice_L": list(oral.F),
                "O_midpoint": list(oral.O),
            }
        },
        "mouth_tilt": {
            "C047_lip_vertical_tilt_deg": tilt.lip_vertical_tilt,
            "C062_mouth_horizontal_level_deg": tilt.mouth_horizontal_level,
        },
        "mouth": {
            "width": indicators.mouth_width,
            "height": indicators.mouth_height,
        },
        "nlf": {
            "left_length": indicators.left_nlf_length,
            "right_length": indicators.right_nlf_length,
            "ratio": indicators.nlf_ratio,
        }
    }

    if resting_dict:
        indicators_dict["resting_symmetry"] = resting_dict
    if vol_dict:
        indicators_dict["voluntary_movement"] = vol_dict
    if syn_dict:
        indicators_dict["synkinesis"] = syn_dict

    with open(action_dir / "indicators.json", 'w', encoding='utf-8') as f:
        json.dump(indicators_dict, f, indent=2, ensure_ascii=False)

    print(f"    [OK] 口角: AOE(R)={oral.AOE_angle:+.2f}°, BOF(L)={oral.BOF_angle:+.2f}°")
    print(f"    [OK] EAR: L={indicators.left_ear:.3f}, R={indicators.right_ear:.3f}")

    return indicators_dict


def process_examination(examination: Dict[str, Any], db_path: str, output_dir: Path,
                        extractor: LandmarkExtractor) -> Dict[str, Any]:
    """处理单个检查"""
    exam_id = examination["examination_id"]
    patient_id = examination["patient_id"]

    print(f"\n{'=' * 60}")
    print(f"处理检查: {exam_id}")
    print(f"{'=' * 60}")

    videos = db_fetch_videos_for_exam(db_path, exam_id)
    print(f"找到 {len(videos)} 个动作视频")

    labels = db_fetch_labels(db_path, exam_id)
    print(f"医生标注: {labels}")

    exam_output_dir = output_dir / exam_id
    exam_output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "examination_id": exam_id,
        "patient_id": patient_id,
        "capture_datetime": examination["capture_datetime"],
        "ground_truth": labels,
        "actions": {}
    }

    # 先处理NeutralFace获取baseline
    neutral_indicators = None
    if "NeutralFace" in videos:
        neutral_result = process_single_action(extractor, videos["NeutralFace"], exam_output_dir, "NeutralFace")
        if neutral_result:
            results["actions"]["NeutralFace"] = neutral_result
            # 重新提取indicators用于比较
            video_path = videos["NeutralFace"]["file_path"]
            landmarks_seq, frames_seq = extractor.extract_sequence(
                video_path,
                videos["NeutralFace"].get("start_frame", 0),
                videos["NeutralFace"].get("end_frame", None)
            )
            if landmarks_seq and frames_seq:
                h, w = frames_seq[0].shape[:2]
                peak_idx = find_peak_frame_neutral(landmarks_seq, frames_seq, w, h)
                if landmarks_seq[peak_idx]:
                    neutral_indicators = extract_indicators(landmarks_seq[peak_idx], w, h)

    # 处理其他动作
    for action_name, video_info in videos.items():
        if action_name == "NeutralFace":
            continue
        action_result = process_single_action(extractor, video_info, exam_output_dir, action_name, neutral_indicators)
        if action_result:
            results["actions"][action_name] = action_result

    # 计算完整Sunnybrook评分
    if "NeutralFace" in results["actions"] and "resting_symmetry" in results["actions"]["NeutralFace"]:
        resting_data = results["actions"]["NeutralFace"]["resting_symmetry"]
        resting = RestingSymmetry(
            eye_status=resting_data["eye"]["status"],
            eye_score=resting_data["eye"]["score"],
            eye_detail=resting_data["eye"]["detail"],
            cheek_status=resting_data["cheek"]["status"],
            cheek_score=resting_data["cheek"]["score"],
            cheek_detail=resting_data["cheek"]["detail"],
            mouth_status=resting_data["mouth"]["status"],
            mouth_score=resting_data["mouth"]["score"],
            mouth_detail=resting_data["mouth"]["detail"],
            total_score=resting_data["total_score"],
            affected_side=resting_data["affected_side"]
        )

        vol_list = []
        syn_list = []
        for action_name, action_data in results["actions"].items():
            if "voluntary_movement" in action_data:
                vol_data = action_data["voluntary_movement"]
                vol_list.append(VoluntaryMovement(
                    action_name=vol_data["action_name"],
                    action_cn="",
                    left_value=vol_data["left_value"],
                    right_value=vol_data["right_value"],
                    ratio=vol_data["ratio"],
                    score=vol_data["score"],
                    interpretation=vol_data["interpretation"]
                ))
            if "synkinesis" in action_data:
                syn_data = action_data["synkinesis"]
                syn_list.append(Synkinesis(
                    action_name=action_name,
                    eye_synkinesis=syn_data["eye_synkinesis"],
                    cheek_synkinesis=syn_data["cheek_synkinesis"],
                    mouth_synkinesis=syn_data["mouth_synkinesis"],
                    total_score=syn_data["total_score"],
                    interpretation=syn_data["interpretation"]
                ))

        sb_score = compute_sunnybrook_score(resting, vol_list, syn_list)
        results["sunnybrook"] = {
            "resting_score": sb_score.resting_score,
            "voluntary_score": sb_score.voluntary_score,
            "synkinesis_score": sb_score.synkinesis_score,
            "composite_score": sb_score.composite_score
        }

    with open(exam_output_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    generate_html_report(results, exam_output_dir)

    return results


def generate_html_report(results: Dict[str, Any], output_dir: Path):
    """生成HTML报告"""
    exam_id = results["examination_id"]
    patient_id = results["patient_id"]
    ground_truth = results.get("ground_truth", {})
    sunnybrook = results.get("sunnybrook", {})

    resting = None
    if "NeutralFace" in results.get("actions", {}):
        resting = results["actions"]["NeutralFace"].get("resting_symmetry")

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>面部指标分析报告 - {exam_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        h2 {{ color: #666; margin-top: 30px; }}
        .info-box {{ background: #e8f5e9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .ground-truth {{ background: #fff3e0; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .sunnybrook-summary {{ background: #e3f2fd; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .score-box {{ display: inline-block; padding: 10px 20px; margin: 5px; border-radius: 5px; }}
        .resting-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        .resting-table th, .resting-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        .resting-table th {{ background: #1976d2; color: white; }}
        .score-normal {{ color: #4CAF50; font-weight: bold; }}
        .score-mild {{ color: #FF9800; font-weight: bold; }}
        .score-severe {{ color: #f44336; font-weight: bold; }}
        .action-card {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
        .action-title {{ font-weight: bold; font-size: 1.2em; color: #2196F3; }}
        .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 10px; }}
        .metric {{ background: #f9f9f9; padding: 10px; border-radius: 3px; }}
        .metric-label {{ color: #666; font-size: 0.85em; }}
        .metric-value {{ font-size: 1.1em; font-weight: bold; }}
        .images {{ display: flex; gap: 10px; margin-top: 15px; flex-wrap: wrap; }}
        .images img {{ max-width: 300px; border: 1px solid #ddd; border-radius: 5px; }}
        .warning {{ color: #f44336; }}
        .normal {{ color: #4CAF50; }}
    </style>
</head>
<body>
<div class="container">
    <h1>面部指标分析报告</h1>

    <div class="info-box">
        <strong>检查ID:</strong> {exam_id}<br>
        <strong>患者ID:</strong> {patient_id}<br>
        <strong>分析时间:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>

    <div class="ground-truth">
        <h3>医生标注 (Ground Truth)</h3>
        <strong>面瘫:</strong> {'是' if ground_truth.get('has_palsy') else '否'}<br>
        <strong>患侧:</strong> {ground_truth.get('palsy_side', 'N/A')}<br>
        <strong>HB分级:</strong> {ground_truth.get('hb_grade', 'N/A')}<br>
        <strong>Sunnybrook评分:</strong> {ground_truth.get('sunnybrook_score', 'N/A')}
    </div>
"""

    # Sunnybrook Summary
    if sunnybrook:
        composite = sunnybrook.get('composite_score', 0)
        comp_class = "score-normal" if composite >= 70 else ("score-mild" if composite >= 50 else "score-severe")
        html += f"""
    <div class="sunnybrook-summary">
        <h2>Sunnybrook 评分汇总</h2>
        <div class="score-box" style="background: #ffcdd2;">Resting: {sunnybrook.get('resting_score', 0)}</div>
        <div class="score-box" style="background: #c8e6c9;">Voluntary: {sunnybrook.get('voluntary_score', 0)}</div>
        <div class="score-box" style="background: #ffe0b2;">Synkinesis: {sunnybrook.get('synkinesis_score', 0)}</div>
        <div class="score-box {comp_class}" style="background: #e1f5fe; font-size: 1.2em;">
            <strong>Composite: {composite}</strong>
        </div>
        <p style="color: #666; margin-top: 10px;">公式: Composite = Voluntary - Resting - Synkinesis</p>
    </div>
"""

    # Resting Symmetry Table
    if resting:
        eye_class = "score-normal" if resting["eye"]["score"] == 0 else "score-mild"
        cheek_class = "score-normal" if resting["cheek"]["score"] == 0 else (
            "score-mild" if resting["cheek"]["score"] == 1 else "score-severe")
        mouth_class = "score-normal" if resting["mouth"]["score"] == 0 else "score-mild"

        html += f"""
    <h2>Resting Symmetry 静态对称性评估</h2>
    <table class="resting-table">
        <tr><th>部位</th><th>状态</th><th>评分</th><th>详细说明</th></tr>
        <tr>
            <td>眼 (睑裂)</td>
            <td>{resting["eye"]["status"]}</td>
            <td class="{eye_class}">{resting["eye"]["score"]}</td>
            <td>{resting["eye"]["detail"]}</td>
        </tr>
        <tr>
            <td>颊 (鼻唇沟)</td>
            <td>{resting["cheek"]["status"]}</td>
            <td class="{cheek_class}">{resting["cheek"]["score"]}</td>
            <td>{resting["cheek"]["detail"]}</td>
        </tr>
        <tr>
            <td>嘴</td>
            <td>{resting["mouth"]["status"]}</td>
            <td class="{mouth_class}">{resting["mouth"]["score"]}</td>
            <td>{resting["mouth"]["detail"]}</td>
        </tr>
        <tr style="background: #e3f2fd;">
            <td colspan="2"><strong>总分</strong></td>
            <td><strong>{resting["total_score"]} (SB: {resting["sunnybrook_resting_score"]})</strong></td>
            <td><strong>判断患侧: {resting["affected_side"]}</strong></td>
        </tr>
    </table>
    <div class="images" style="margin-top: 15px;">
        <img src="NeutralFace/resting_symmetry.jpg" alt="Resting Symmetry">
    </div>
"""

    html += "<h2>动作分析结果</h2>"

    for action_name, action_data in results.get("actions", {}).items():
        oral = action_data.get("oral_angle", {})
        eye = action_data.get("eye", {})
        nlf = action_data.get("nlf", {})
        tilt = action_data.get("mouth_tilt", {})
        vol = action_data.get("voluntary_movement", {})
        syn = action_data.get("synkinesis", {})

        aoe = oral.get("AOE_angle_deg", 0)
        bof = oral.get("BOF_angle_deg", 0)
        asym = oral.get("angle_asymmetry", 0)
        eye_ratio = eye.get("area_ratio", 1)
        nlf_ratio = nlf.get("ratio", 1)

        oral_class = "warning" if asym > 5 else "normal"
        eye_class = "warning" if abs(eye_ratio - 1) > 0.15 else "normal"
        nlf_class = "warning" if abs(nlf_ratio - 1) > 0.25 else "normal"

        html += f"""
    <div class="action-card">
        <div class="action-title">{action_name}</div>
        <div class="metrics">
            <div class="metric">
                <div class="metric-label">AOE (R)</div>
                <div class="metric-value">{aoe:+.2f}°</div>
            </div>
            <div class="metric">
                <div class="metric-label">BOF (L)</div>
                <div class="metric-value">{bof:+.2f}°</div>
            </div>
            <div class="metric">
                <div class="metric-label">口角不对称</div>
                <div class="metric-value {oral_class}">{asym:.2f}°</div>
            </div>
            <div class="metric">
                <div class="metric-label">EAR L/R</div>
                <div class="metric-value">{eye.get('left_ear', 0):.3f}/{eye.get('right_ear', 0):.3f}</div>
            </div>
"""
        if vol:
            vol_class = "normal" if vol.get("score", 0) >= 4 else "warning"
            html += f"""
            <div class="metric">
                <div class="metric-label">Voluntary Score</div>
                <div class="metric-value {vol_class}">{vol.get('score', 'N/A')}/5</div>
            </div>
"""
        if syn:
            syn_class = "normal" if syn.get("total_score", 0) <= 2 else "warning"
            html += f"""
            <div class="metric">
                <div class="metric-label">Synkinesis</div>
                <div class="metric-value {syn_class}">{syn.get('total_score', 0)}</div>
            </div>
"""
        html += f"""
        </div>
        <div class="images">
            <img src="{action_name}/peak_raw.jpg" alt="原始帧">
            <img src="{action_name}/peak_oral_angle.jpg" alt="口角角度">
"""
        if action_name == "NeutralFace":
            html += f'            <img src="{action_name}/resting_symmetry.jpg" alt="Resting Symmetry">\n'
        html += """        </div>
    </div>
"""

    html += """
</div>
</body>
</html>
"""

    with open(output_dir / "report.html", 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"  [OK] HTML报告: {output_dir / 'report.html'}")


# =============================================================================
# 主函数
# =============================================================================

def main():
    print("=" * 70)
    print("面部临床分级测试脚本 (完整Sunnybrook评分)")
    print("=" * 70)

    print(f"\n配置:")
    print(f"  数据库: {DATABASE_PATH}")
    print(f"  模型: {MEDIAPIPE_MODEL_PATH}")
    print(f"  输出: {OUTPUT_DIR}")

    if not os.path.exists(DATABASE_PATH):
        print(f"\n[ERROR] 数据库不存在: {DATABASE_PATH}")
        return

    if not os.path.exists(MEDIAPIPE_MODEL_PATH):
        print(f"\n[ERROR] MediaPipe模型不存在: {MEDIAPIPE_MODEL_PATH}")
        return

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n获取检查记录...")
    examinations = db_fetch_examinations(DATABASE_PATH, TARGET_EXAM_ID, PATIENT_LIMIT)
    print(f"找到 {len(examinations)} 个检查记录")

    if not examinations:
        print("[ERROR] 没有有效的检查记录")
        return

    print(f"\n初始化MediaPipe...")

    all_results = []

    with LandmarkExtractor(MEDIAPIPE_MODEL_PATH) as extractor:
        for i, exam in enumerate(examinations):
            print(f"\n[{i + 1}/{len(examinations)}]", end="")
            result = process_examination(exam, DATABASE_PATH, output_dir, extractor)
            all_results.append(result)

    print(f"\n\n{'=' * 70}")
    print("处理完成!")
    print(f"{'=' * 70}")
    print(f"处理了 {len(all_results)} 个检查")
    print(f"输出目录: {output_dir}")

    total_actions = sum(len(r.get("actions", {})) for r in all_results)
    print(f"总共分析了 {total_actions} 个动作视频")

    summary = {
        "run_time": datetime.now().isoformat(),
        "total_examinations": len(all_results),
        "total_actions": total_actions,
        "results": all_results
    }

    with open(output_dir / "all_results.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n汇总结果: {output_dir / 'all_results.json'}")


if __name__ == "__main__":
    main()