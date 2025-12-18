#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
临床分级测试脚本 - Clinical Grading Test Script
=============================================================

修正内容:
1. 口角角度计算: 以O为顶点计算∠AOE和∠BOF (而非以E/F为顶点)
2. 新增C047: 唇部竖向倾斜角
3. 新增C062: 双侧嘴角水平度
4. 新增Resting Symmetry评估表

从数据库读取真实患者数据，进行面部指标分析

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
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import math

# =============================================================================
# 配置参数 - 根据您的环境修改这些路径
# =============================================================================

# 数据库路径 (修改为您本地的实际路径)
DATABASE_PATH = r"facialPalsy.db"

# MediaPipe模型路径
MEDIAPIPE_MODEL_PATH = r"/Users/cuijinglei/PycharmProjects/medicalProject/models/face_landmarker.task"

# 输出目录
OUTPUT_DIR = r"/Users/cuijinglei/Documents/facialPalsy/HGFA/clinical_grading_output"

# 处理患者数量限制 (None = 处理所有)
PATIENT_LIMIT = None

# 目标检查ID (None = 处理所有有效检查)
TARGET_EXAM_ID = None


# =============================================================================
# Landmark 索引定义 (基于MediaPipe 478点)
# =============================================================================

class LM:
    """MediaPipe FaceLandmarker 478点索引"""

    # ========== 眼部 ==========
    EYE_INNER_L = 362  # 左眼内眦
    EYE_INNER_R = 133  # 右眼内眦
    EYE_OUTER_L = 263  # 左眼外眦
    EYE_OUTER_R = 33  # 右眼外眦

    EYE_TOP_L = 386  # 左眼上眼睑中点
    EYE_BOT_L = 374  # 左眼下眼睑中点
    EYE_TOP_R = 159  # 右眼上眼睑中点
    EYE_BOT_R = 145  # 右眼下眼睑中点

    # 眼部轮廓 (16点)
    EYE_CONTOUR_L = [263, 466, 388, 387, 386, 385, 384, 398,
                     362, 382, 381, 380, 374, 373, 390, 249]
    EYE_CONTOUR_R = [33, 246, 161, 160, 159, 158, 157, 173,
                     133, 155, 154, 153, 145, 144, 163, 7]

    # EAR计算用点
    EAR_L = [263, 386, 387, 362, 373, 374]
    EAR_R = [33, 159, 158, 133, 144, 145]

    # ========== 眉毛 ==========
    BROW_L = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
    BROW_R = [107, 66, 105, 63, 70, 46, 53, 52, 65, 55]
    BROW_CENTER_L = 282
    BROW_CENTER_R = 52

    # ========== 嘴部 (常规测量用) ==========
    MOUTH_L = 291  # 左嘴角
    MOUTH_R = 61  # 右嘴角
    LIP_TOP = 13  # 上唇顶
    LIP_BOT = 14  # 下唇底
    LIP_TOP_CENTER = 0  # 上唇中心
    LIP_BOT_CENTER = 17  # 下唇中心

    # ========== 口角角度计算专用 (Oral Commissure Lift论文) ==========
    # A点: 右嘴角
    ORAL_CORNER_R = 78  # Right mouth corner (A)
    # B点: 左嘴角
    ORAL_CORNER_L = 308  # Left mouth corner (B)
    # C点: 右上唇峰
    LIP_PEAK_R = 37  # Right upper lip peak (C)
    # D点: 左上唇峰
    LIP_PEAK_L = 267  # Left upper lip peak (D)
    # E点: mouth orifice右侧稳定点 (landmark 82)
    ORAL_E = 82
    # F点: mouth orifice左侧稳定点 (landmark 312)
    ORAL_F = 312
    # O点: 上唇中心 (用于定义水平参考)
    ORAL_O = 13

    # ========== 面中线参考点 ==========
    FOREHEAD = 10  # 额头中心
    NOSE_TIP = 4  # 鼻尖
    CHIN = 152  # 下巴

    # ========== 鼻部 ==========
    NOSE_ALA_L = 129  # 左鼻翼
    NOSE_ALA_R = 358  # 右鼻翼

    # ========== 面颊 ==========
    CHEEK_L = [425, 426, 427, 411, 280]
    CHEEK_R = [205, 206, 207, 187, 50]


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
            FROM examinations
            WHERE examination_id = ?
        """, (target_exam_id,))
    else:
        cursor.execute("""
            SELECT examination_id, patient_id, capture_datetime, has_labels, has_videos
            FROM examinations
            WHERE has_videos = 1 AND is_valid = 1
            ORDER BY capture_datetime DESC
        """)

    rows = cursor.fetchall()
    conn.close()

    exams = []
    for r in rows:
        exams.append({
            "examination_id": r[0],
            "patient_id": r[1],
            "capture_datetime": r[2],
            "has_labels": r[3],
            "has_videos": r[4],
        })

    if limit is not None:
        exams = exams[:int(limit)]
    return exams


def db_fetch_videos_for_exam(db_path: str, examination_id: str) -> Dict[str, Dict[str, Any]]:
    """获取检查对应的视频信息"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            v.video_id, v.action_id, v.file_path, v.start_frame, v.end_frame, 
            v.fps, v.video_file_index,
            at.action_name_en, at.action_name_cn
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
            "video_id": int(video_id),
            "action_id": int(action_id),
            "action_name_en": action_en,
            "action_name_cn": action_cn,
            "file_path": file_path,
            "start_frame": int(start_frame) if start_frame is not None else 0,
            "end_frame": int(end_frame) if end_frame is not None else None,
            "fps": float(fps) if fps is not None else 30.0,
            "video_file_index": int(video_file_index) if video_file_index is not None else 0,
        })

    selected: Dict[str, Dict[str, Any]] = {}
    for action_en, candidates in grouped.items():
        candidates_sorted = sorted(candidates, key=lambda x: x.get("video_file_index", 0))
        selected[action_en] = candidates_sorted[0]

    return selected


def db_fetch_labels(db_path: str, examination_id: str) -> Dict[str, Any]:
    """获取医生标注的标签"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT has_palsy, palsy_side, hb_grade, sunnybrook_score
        FROM examination_labels
        WHERE examination_id = ?
    """, (examination_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return {}
    return {
        "has_palsy": row[0],
        "palsy_side": row[1],
        "hb_grade": row[2],
        "sunnybrook_score": row[3],
    }


# =============================================================================
# 几何计算函数
# =============================================================================

def pt2d(lm, w: int, h: int) -> Tuple[float, float]:
    """单个landmark → 2D像素坐标"""
    return (lm.x * w, lm.y * h)


def pts2d(landmarks, indices: List[int], w: int, h: int) -> np.ndarray:
    """批量获取2D坐标"""
    return np.array([pt2d(landmarks[i], w, h) for i in indices])


def dist(p1, p2) -> float:
    """2D欧氏距离"""
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


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


def compute_icd(landmarks, w: int, h: int) -> float:
    """计算ICD (双眼内眦距离) - 单位长度"""
    l_inner = pt2d(landmarks[LM.EYE_INNER_L], w, h)
    r_inner = pt2d(landmarks[LM.EYE_INNER_R], w, h)
    return dist(l_inner, r_inner)


def compute_eye_area(landmarks, w: int, h: int, left: bool = True) -> Tuple[float, np.ndarray]:
    """计算眼裂面积"""
    contour = LM.EYE_CONTOUR_L if left else LM.EYE_CONTOUR_R
    points = pts2d(landmarks, contour, w, h)
    return polygon_area(points), points


def compute_ear(landmarks, w: int, h: int, left: bool = True) -> float:
    """计算EAR (Eye Aspect Ratio)"""
    pts_idx = LM.EAR_L if left else LM.EAR_R
    p = [pt2d(landmarks[i], w, h) for i in pts_idx]
    v1 = dist(p[1], p[5])
    v2 = dist(p[2], p[4])
    h_dist = dist(p[0], p[3])
    if h_dist < 1e-9:
        return 0.0
    return (v1 + v2) / (2.0 * h_dist)


def compute_palpebral_height(landmarks, w: int, h: int, left: bool = True) -> float:
    """计算眼睑裂高度 (睑裂纵径)"""
    if left:
        top = pt2d(landmarks[LM.EYE_TOP_L], w, h)
        bot = pt2d(landmarks[LM.EYE_BOT_L], w, h)
    else:
        top = pt2d(landmarks[LM.EYE_TOP_R], w, h)
        bot = pt2d(landmarks[LM.EYE_BOT_R], w, h)
    return dist(top, bot)


def angle_between_vectors(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    """计算两个向量之间的夹角(度), 返回0-180度"""
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    if mag1 < 1e-9 or mag2 < 1e-9:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def signed_angle_from_horizontal(point: Tuple[float, float], origin: Tuple[float, float],
                                 ref_direction: Tuple[float, float]) -> float:
    """
    计算从origin到point的向量相对于ref_direction(水平参考方向)的有符号角度

    在图像坐标系中:
    - Y轴向下为正
    - 当point在水平线下方时，返回负角度(表示下垂)
    - 当point在水平线上方时，返回正角度(表示上提)

    Args:
        point: 目标点坐标
        origin: 原点坐标
        ref_direction: 水平参考方向向量(从origin指向参考点)

    Returns:
        有符号角度(度)，负值表示下垂
    """
    # 从origin到point的向量
    op = (point[0] - origin[0], point[1] - origin[1])
    op_len = math.sqrt(op[0] ** 2 + op[1] ** 2)
    ref_len = math.sqrt(ref_direction[0] ** 2 + ref_direction[1] ** 2)

    if op_len < 1e-9 or ref_len < 1e-9:
        return 0.0

    # 单位化
    op_unit = (op[0] / op_len, op[1] / op_len)
    ref_unit = (ref_direction[0] / ref_len, ref_direction[1] / ref_len)

    # 计算点积和叉积
    dot = op_unit[0] * ref_unit[0] + op_unit[1] * ref_unit[1]
    cross = ref_unit[0] * op_unit[1] - ref_unit[1] * op_unit[0]

    # 角度大小
    dot = max(-1.0, min(1.0, dot))
    angle = math.degrees(math.acos(dot))

    # 在图像坐标系中，Y轴向下
    # 如果cross > 0，说明point在ref_direction的顺时针方向（图像中为下方）
    # 下垂应该为负值
    if cross > 0:
        angle = -angle

    return angle


# =============================================================================
# 口角角度计算 - 基于 "Oral Commissure Lift" 论文 (修正版)
# =============================================================================

@dataclass
class OralAngleMeasure:
    """口角角度测量结果 (基于Oral Commissure Lift论文)

    论文方法:
    - E和F点定义mouth orientation(嘴部水平线)
    - O点是E和F的中点
    - ∠AOE: 以O为顶点，OE为水平参考，OA为测量向量的角度
    - ∠BOF: 以O为顶点，OF为水平参考，OB为测量向量的角度
    - 负值表示口角下垂
    """
    # 关键点坐标
    A: Tuple[float, float]  # 右嘴角 (landmark 78)
    B: Tuple[float, float]  # 左嘴角 (landmark 308)
    C: Tuple[float, float]  # 右上唇峰 (landmark 37) - 用于参考
    D: Tuple[float, float]  # 左上唇峰 (landmark 267) - 用于参考
    E: Tuple[float, float]  # mouth orifice右侧点 (landmark 82)
    F: Tuple[float, float]  # mouth orifice左侧点 (landmark 312)
    O: Tuple[float, float]  # E和F的中点 (角度计算的顶点)

    # 角度 (度) - 以O为顶点
    AOE_angle: float  # 右口角角度: ∠AOE，负值表示下垂
    BOF_angle: float  # 左口角角度: ∠BOF，负值表示下垂

    # 派生指标
    angle_diff: float  # 角度差 (左-右)，正值表示左侧相对上提
    angle_asymmetry: float  # 不对称性 |左-右|


def compute_oral_angle(landmarks, w: int, h: int) -> OralAngleMeasure:
    """
    计算口角角度 (基于"Oral Commissure Lift"论文方法) - 修正版

    论文定义:
    - E点(landmark 82)和F点(landmark 312)定义mouth orientation(嘴部水平线)
    - O点 = E和F的中点，作为角度计算的顶点
    - ∠AOE: 顶点在O，从O到E的向量为水平参考，从O到A的向量为测量向量
    - ∠BOF: 顶点在O，从O到F的向量为水平参考，从O到B的向量为测量向量

    符号约定:
    - 在图像坐标系中，Y轴向下为正
    - 当口角在水平线下方时，角度为负(表示下垂)
    - 当口角在水平线上方时，角度为正(表示上提)

    Args:
        landmarks: MediaPipe face landmarks
        w, h: 图像宽高

    Returns:
        OralAngleMeasure: 口角角度测量结果
    """
    # 获取关键点坐标
    A = pt2d(landmarks[LM.ORAL_CORNER_R], w, h)  # 右嘴角 (78)
    B = pt2d(landmarks[LM.ORAL_CORNER_L], w, h)  # 左嘴角 (308)
    C = pt2d(landmarks[LM.LIP_PEAK_R], w, h)  # 右上唇峰 (37)
    D = pt2d(landmarks[LM.LIP_PEAK_L], w, h)  # 左上唇峰 (267)
    E = pt2d(landmarks[LM.ORAL_E], w, h)  # mouth orifice右侧 (82)
    F = pt2d(landmarks[LM.ORAL_F], w, h)  # mouth orifice左侧 (312)

    # O点: E和F的中点 - 这是角度计算的顶点
    O = ((E[0] + F[0]) / 2.0, (E[1] + F[1]) / 2.0)

    # 计算水平参考方向向量
    # OE向量: 从O指向E (向右)
    OE = (E[0] - O[0], E[1] - O[1])
    # OF向量: 从O指向F (向左)
    OF = (F[0] - O[0], F[1] - O[1])

    # 计算∠AOE: 以O为顶点，OE为水平参考
    # OA向量: 从O指向A
    AOE_angle = signed_angle_from_horizontal(A, O, OE)

    # 计算∠BOF: 以O为顶点，OF为水平参考
    # OB向量: 从O指向B
    BOF_angle = signed_angle_from_horizontal(B, O, OF)

    return OralAngleMeasure(
        A=A, B=B, C=C, D=D, E=E, F=F, O=O,
        AOE_angle=AOE_angle,
        BOF_angle=BOF_angle,
        angle_diff=BOF_angle - AOE_angle,
        angle_asymmetry=abs(BOF_angle - AOE_angle)
    )


# =============================================================================
# C047: 唇部竖向倾斜角 / C062: 双侧嘴角水平度
# =============================================================================

@dataclass
class MouthTiltMeasure:
    """嘴部倾斜度测量结果"""
    # C047: 唇部竖向倾斜角
    # 上下唇中点连线与面中线的夹角
    lip_vertical_tilt: float  # 度，正值表示嘴部中线在面中线右侧

    # C062: 双侧嘴角水平度
    # 两侧嘴角连线与面中线垂直线的夹角
    mouth_horizontal_level: float  # 度，表示嘴角连线的倾斜程度

    # 辅助点
    lip_top_center: Tuple[float, float]  # 上唇中点
    lip_bot_center: Tuple[float, float]  # 下唇中点
    midline_top: Tuple[float, float]  # 面中线上点 (额头/鼻根)
    midline_bot: Tuple[float, float]  # 面中线下点 (鼻尖/下巴)
    mouth_left: Tuple[float, float]  # 左嘴角
    mouth_right: Tuple[float, float]  # 右嘴角


def compute_mouth_tilt(landmarks, w: int, h: int) -> MouthTiltMeasure:
    """
    计算嘴部倾斜度指标

    C047 唇部竖向倾斜角:
    - 上下唇中点连线与面中线参考线的夹角
    - 数值越小，唇部越对称
    - 正值表示嘴部中线在面中线右侧(观察者视角左侧)

    C062 双侧嘴角水平度:
    - 两侧嘴角连线与面中线垂直线的夹角
    - 数值越小越好，表示嘴角越水平
    """
    # 获取关键点
    lip_top = pt2d(landmarks[LM.LIP_TOP_CENTER], w, h)  # 上唇中点 (landmark 0)
    lip_bot = pt2d(landmarks[LM.LIP_BOT_CENTER], w, h)  # 下唇中点 (landmark 17)

    # 面中线参考点
    forehead = pt2d(landmarks[LM.FOREHEAD], w, h)  # 额头中心 (landmark 10)
    chin = pt2d(landmarks[LM.CHIN], w, h)  # 下巴 (landmark 152)

    # 嘴角
    mouth_l = pt2d(landmarks[LM.MOUTH_L], w, h)  # 左嘴角 (landmark 291)
    mouth_r = pt2d(landmarks[LM.MOUTH_R], w, h)  # 右嘴角 (landmark 61)

    # === C047: 唇部竖向倾斜角 ===
    # 唇部中线向量 (从下唇中点到上唇中点)
    lip_midline = (lip_top[0] - lip_bot[0], lip_top[1] - lip_bot[1])

    # 面中线向量 (从下巴到额头，即向上)
    face_midline = (forehead[0] - chin[0], forehead[1] - chin[1])

    # 计算夹角
    lip_vertical_tilt = angle_between_vectors(lip_midline, face_midline)

    # 确定符号: 使用叉积判断方向
    # 如果唇部中线在面中线的右侧(顺时针)，为正
    cross = face_midline[0] * lip_midline[1] - face_midline[1] * lip_midline[0]
    if cross < 0:
        lip_vertical_tilt = -lip_vertical_tilt

    # === C062: 双侧嘴角水平度 ===
    # 嘴角连线向量 (从右嘴角到左嘴角)
    mouth_line = (mouth_l[0] - mouth_r[0], mouth_l[1] - mouth_r[1])

    # 面中线的垂直线 (旋转90度)
    face_horizontal = (-face_midline[1], face_midline[0])

    # 计算夹角
    mouth_horizontal_level = angle_between_vectors(mouth_line, face_horizontal)

    # 限制在0-90度范围
    if mouth_horizontal_level > 90:
        mouth_horizontal_level = 180 - mouth_horizontal_level

    return MouthTiltMeasure(
        lip_vertical_tilt=lip_vertical_tilt,
        mouth_horizontal_level=mouth_horizontal_level,
        lip_top_center=lip_top,
        lip_bot_center=lip_bot,
        midline_top=forehead,
        midline_bot=chin,
        mouth_left=mouth_l,
        mouth_right=mouth_r
    )


# =============================================================================
# 综合指标提取
# =============================================================================

@dataclass
class FacialIndicators:
    """面部指标集合"""
    # ICD
    icd: float

    # 眼部
    left_eye_area: float
    right_eye_area: float
    eye_area_ratio: float
    left_ear: float
    right_ear: float
    left_palpebral_height: float  # 左眼睑裂高度
    right_palpebral_height: float  # 右眼睑裂高度
    palpebral_height_ratio: float  # 睑裂高度比

    # 口角 (论文方法)
    oral_angle: OralAngleMeasure

    # 嘴部倾斜度
    mouth_tilt: MouthTiltMeasure

    # 嘴部常规测量
    mouth_width: float
    mouth_height: float

    # 鼻唇沟
    left_nlf_length: float
    right_nlf_length: float
    nlf_ratio: float


def extract_indicators(landmarks, w: int, h: int) -> FacialIndicators:
    """提取所有面部指标"""
    icd = compute_icd(landmarks, w, h)

    # 眼部
    left_eye_area, _ = compute_eye_area(landmarks, w, h, left=True)
    right_eye_area, _ = compute_eye_area(landmarks, w, h, left=False)
    eye_area_ratio = left_eye_area / right_eye_area if right_eye_area > 1e-9 else 1.0

    left_ear = compute_ear(landmarks, w, h, left=True)
    right_ear = compute_ear(landmarks, w, h, left=False)

    # 睑裂高度
    left_palp_h = compute_palpebral_height(landmarks, w, h, left=True)
    right_palp_h = compute_palpebral_height(landmarks, w, h, left=False)
    palp_h_ratio = left_palp_h / right_palp_h if right_palp_h > 1e-9 else 1.0

    # 口角 (论文方法) - 修正版
    oral_angle = compute_oral_angle(landmarks, w, h)

    # 嘴部倾斜度 (C047, C062)
    mouth_tilt = compute_mouth_tilt(landmarks, w, h)

    # 嘴部常规
    l_corner = pt2d(landmarks[LM.MOUTH_L], w, h)
    r_corner = pt2d(landmarks[LM.MOUTH_R], w, h)
    mouth_width = dist(l_corner, r_corner)

    lip_top = pt2d(landmarks[LM.LIP_TOP], w, h)
    lip_bot = pt2d(landmarks[LM.LIP_BOT], w, h)
    mouth_height = dist(lip_top, lip_bot)

    # 鼻唇沟
    l_ala = pt2d(landmarks[LM.NOSE_ALA_L], w, h)
    r_ala = pt2d(landmarks[LM.NOSE_ALA_R], w, h)
    left_nlf = dist(l_ala, l_corner)
    right_nlf = dist(r_ala, r_corner)
    nlf_ratio = left_nlf / right_nlf if right_nlf > 1e-9 else 1.0

    return FacialIndicators(
        icd=icd,
        left_eye_area=left_eye_area,
        right_eye_area=right_eye_area,
        eye_area_ratio=eye_area_ratio,
        left_ear=left_ear,
        right_ear=right_ear,
        left_palpebral_height=left_palp_h,
        right_palpebral_height=right_palp_h,
        palpebral_height_ratio=palp_h_ratio,
        oral_angle=oral_angle,
        mouth_tilt=mouth_tilt,
        mouth_width=mouth_width,
        mouth_height=mouth_height,
        left_nlf_length=left_nlf,
        right_nlf_length=right_nlf,
        nlf_ratio=nlf_ratio
    )


# =============================================================================
# Resting Symmetry 评估 (Sunnybrook静态对称性)
# =============================================================================

@dataclass
class RestingSymmetry:
    """静态对称性评估结果 (Sunnybrook Resting Symmetry)"""
    # 眼 (睑裂)
    eye_status: str  # "正常" / "缩窄" / "增宽"
    eye_score: int  # 0=正常, 1=缩窄/增宽
    eye_detail: str  # 详细说明

    # 颊 (鼻唇沟)
    cheek_status: str  # "正常" / "不明显" / "消失" / "过于明显"
    cheek_score: int  # 0=正常, 1=不明显, 2=消失/过于明显
    cheek_detail: str

    # 嘴
    mouth_status: str  # "正常" / "口角下垂" / "口角上提"
    mouth_score: int  # 0=正常, 1=下垂/上提
    mouth_detail: str

    # 总分
    total_score: int  # Sunnybrook静态对称性总分 (0-4分制或原始计算)

    # 患侧判断
    affected_side: str  # "左侧" / "右侧" / "不确定"


def compute_resting_symmetry(indicators: FacialIndicators) -> RestingSymmetry:
    """
    计算静态对称性评估 (基于NeutralFace)

    Sunnybrook静态对称性评分:
    1. 眼(睑裂): 正常=0, 缩窄/增宽=1
    2. 颊(鼻唇沟): 正常=0, 不明显=1, 消失=2, 过于明显=2
    3. 嘴: 正常=0, 口角下垂=1, 口角上提=1

    阈值定义 (基于临床经验):
    - 眼睑裂比值: 0.85-1.15为正常
    - 鼻唇沟比值: 0.75-1.25为正常
    - 口角角度差: ±3度为正常
    """
    # ========== 眼 (睑裂) ==========
    eye_ratio = indicators.palpebral_height_ratio

    if 0.85 <= eye_ratio <= 1.15:
        eye_status = "正常"
        eye_score = 0
        eye_detail = f"睑裂高度比(L/R)={eye_ratio:.3f}，在正常范围(0.85-1.15)"
    elif eye_ratio < 0.85:
        eye_status = "左侧缩窄"
        eye_score = 1
        eye_detail = f"睑裂高度比(L/R)={eye_ratio:.3f}，左眼睑裂较小"
    else:  # > 1.15
        eye_status = "左侧增宽"
        eye_score = 1
        eye_detail = f"睑裂高度比(L/R)={eye_ratio:.3f}，左眼睑裂较大"

    # ========== 颊 (鼻唇沟) ==========
    nlf_ratio = indicators.nlf_ratio

    if 0.85 <= nlf_ratio <= 1.15:
        cheek_status = "正常"
        cheek_score = 0
        cheek_detail = f"鼻唇沟长度比(L/R)={nlf_ratio:.3f}，基本对称"
    elif 0.75 <= nlf_ratio < 0.85:
        cheek_status = "左侧不明显"
        cheek_score = 1
        cheek_detail = f"鼻唇沟长度比(L/R)={nlf_ratio:.3f}，左侧鼻唇沟较浅"
    elif nlf_ratio > 1.15 and nlf_ratio <= 1.25:
        cheek_status = "右侧不明显"
        cheek_score = 1
        cheek_detail = f"鼻唇沟长度比(L/R)={nlf_ratio:.3f}，右侧鼻唇沟较浅"
    elif nlf_ratio < 0.75:
        cheek_status = "左侧消失"
        cheek_score = 2
        cheek_detail = f"鼻唇沟长度比(L/R)={nlf_ratio:.3f}，左侧鼻唇沟明显减弱"
    else:  # > 1.25
        cheek_status = "左侧过于明显"
        cheek_score = 2
        cheek_detail = f"鼻唇沟长度比(L/R)={nlf_ratio:.3f}，右侧鼻唇沟明显减弱"

    # ========== 嘴 ==========
    oral = indicators.oral_angle
    aoe = oral.AOE_angle  # 右口角
    bof = oral.BOF_angle  # 左口角
    angle_diff = bof - aoe  # 左-右

    # 口角角度阈值: ±5度内为正常
    MOUTH_ANGLE_THRESHOLD = 5.0

    if abs(angle_diff) <= MOUTH_ANGLE_THRESHOLD:
        mouth_status = "正常"
        mouth_score = 0
        mouth_detail = f"口角角度差(L-R)={angle_diff:+.1f}°，在正常范围(±{MOUTH_ANGLE_THRESHOLD}°)"
    elif angle_diff < -MOUTH_ANGLE_THRESHOLD:
        # 左侧角度更负(更下垂)
        mouth_status = "左侧口角下垂"
        mouth_score = 1
        mouth_detail = f"左口角={bof:+.1f}°, 右口角={aoe:+.1f}°, 差值={angle_diff:+.1f}°"
    else:
        # 右侧角度更负(更下垂)
        mouth_status = "右侧口角下垂"
        mouth_score = 1
        mouth_detail = f"左口角={bof:+.1f}°, 右口角={aoe:+.1f}°, 差值={angle_diff:+.1f}°"

    # ========== 患侧判断 ==========
    # 综合眼、颊、嘴的异常情况判断患侧
    left_signs = 0
    right_signs = 0

    if "左侧" in eye_status and ("缩窄" in eye_status or "增宽" in eye_status):
        if "缩窄" in eye_status:
            left_signs += 1  # 左眼缩窄可能是左侧面瘫(Bell's现象补偿)或右侧面瘫上睑下垂
        else:
            right_signs += 1  # 左眼增宽通常是左侧面瘫导致眼睑闭合不全

    if "左侧" in cheek_status:
        if "不明显" in cheek_status or "消失" in cheek_status:
            left_signs += 1  # 左侧鼻唇沟变浅 = 左侧面瘫
    if "右侧" in cheek_status:
        if "不明显" in cheek_status:
            right_signs += 1

    if "左侧" in mouth_status and "下垂" in mouth_status:
        left_signs += 1
    if "右侧" in mouth_status and "下垂" in mouth_status:
        right_signs += 1

    if left_signs > right_signs:
        affected_side = "左侧"
    elif right_signs > left_signs:
        affected_side = "右侧"
    else:
        affected_side = "不确定"

    # 总分
    total_score = eye_score + cheek_score + mouth_score

    return RestingSymmetry(
        eye_status=eye_status,
        eye_score=eye_score,
        eye_detail=eye_detail,
        cheek_status=cheek_status,
        cheek_score=cheek_score,
        cheek_detail=cheek_detail,
        mouth_status=mouth_status,
        mouth_score=mouth_score,
        mouth_detail=mouth_detail,
        total_score=total_score,
        affected_side=affected_side
    )


# =============================================================================
# 可视化函数
# =============================================================================

def visualize_oral_angle(frame: np.ndarray, landmarks, w: int, h: int,
                         oral: OralAngleMeasure) -> np.ndarray:
    """可视化口角角度测量 (修正版 - 以O为顶点)"""
    img = frame.copy()

    # 绘制关键点
    def draw_point(pt, color, label, offset=(5, -5), radius=6):
        cv2.circle(img, (int(pt[0]), int(pt[1])), radius, color, -1)
        cv2.putText(img, label, (int(pt[0]) + offset[0], int(pt[1]) + offset[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 定义颜色
    COLOR_A = (0, 0, 255)  # 红色 - 右嘴角
    COLOR_B = (255, 0, 0)  # 蓝色 - 左嘴角
    COLOR_E = (0, 200, 200)  # 黄绿色 - E点
    COLOR_F = (200, 200, 0)  # 青色 - F点
    COLOR_O = (255, 255, 255)  # 白色 - O点(顶点)
    COLOR_REF = (0, 255, 0)  # 绿色 - 参考线

    # 绘制关键点
    draw_point(oral.A, COLOR_A, "A(R)", (5, -10))
    draw_point(oral.B, COLOR_B, "B(L)", (-35, -10))
    draw_point(oral.E, COLOR_E, "E", (5, 15))
    draw_point(oral.F, COLOR_F, "F", (-20, 15))
    draw_point(oral.O, COLOR_O, "O", (-5, -15), radius=8)

    # 绘制E-F水平参考线 (mouth orientation)
    extend = 60
    dx = oral.F[0] - oral.E[0]
    dy = oral.F[1] - oral.E[1]
    line_len = math.sqrt(dx * dx + dy * dy)
    if line_len > 1e-9:
        ux, uy = dx / line_len, dy / line_len
        p1 = (int(oral.E[0] + extend * ux), int(oral.E[1] + extend * uy))
        p2 = (int(oral.F[0] - extend * ux), int(oral.F[1] - extend * uy))
        cv2.line(img, p1, p2, COLOR_REF, 2, cv2.LINE_AA)

    # 绘制O-E向量 (∠AOE的水平参考)
    cv2.line(img, (int(oral.O[0]), int(oral.O[1])),
             (int(oral.E[0]), int(oral.E[1])), COLOR_REF, 2)

    # 绘制O-F向量 (∠BOF的水平参考)
    cv2.line(img, (int(oral.O[0]), int(oral.O[1])),
             (int(oral.F[0]), int(oral.F[1])), COLOR_REF, 2)

    # 绘制O-A向量 (∠AOE的测量向量)
    cv2.line(img, (int(oral.O[0]), int(oral.O[1])),
             (int(oral.A[0]), int(oral.A[1])), COLOR_A, 2)

    # 绘制O-B向量 (∠BOF的测量向量)
    cv2.line(img, (int(oral.O[0]), int(oral.O[1])),
             (int(oral.B[0]), int(oral.B[1])), COLOR_B, 2)

    # 绘制角度弧线
    def draw_angle_arc(center, start_vec, end_vec, color, angle_val):
        """绘制角度弧线"""
        radius = 30
        # 计算起始和终止角度
        start_angle = math.degrees(math.atan2(start_vec[1], start_vec[0]))
        end_angle = math.degrees(math.atan2(end_vec[1], end_vec[0]))

        # OpenCV的ellipse角度是逆时针的
        if angle_val < 0:
            start_angle, end_angle = end_angle, start_angle

        cv2.ellipse(img, (int(center[0]), int(center[1])), (radius, radius),
                    0, start_angle, end_angle, color, 2)

    # 文字信息面板
    y = 30
    cv2.putText(img, "Oral Commissure Angle (Vertex at O)", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y += 30

    # 右侧角度 ∠AOE
    aoe_color = (0, 0, 255) if oral.AOE_angle < -3 else (0, 255, 0)
    status_r = "DOWN" if oral.AOE_angle < -3 else ("UP" if oral.AOE_angle > 3 else "OK")
    cv2.putText(img, f"AOE (R): {oral.AOE_angle:+.2f} deg [{status_r}]", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, aoe_color, 2)
    y += 25

    # 左侧角度 ∠BOF
    bof_color = (255, 0, 0) if oral.BOF_angle < -3 else (0, 255, 0)
    status_l = "DOWN" if oral.BOF_angle < -3 else ("UP" if oral.BOF_angle > 3 else "OK")
    cv2.putText(img, f"BOF (L): {oral.BOF_angle:+.2f} deg [{status_l}]", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, bof_color, 2)
    y += 25

    # 角度差
    diff_color = (0, 0, 255) if abs(oral.angle_diff) > 5 else (255, 255, 255)
    cv2.putText(img, f"Diff (L-R): {oral.angle_diff:+.2f} deg", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, diff_color, 2)
    y += 25

    # 不对称性
    asym_color = (0, 0, 255) if oral.angle_asymmetry > 5 else (0, 255, 0)
    cv2.putText(img, f"Asymmetry: {oral.angle_asymmetry:.2f} deg", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, asym_color, 2)
    y += 30

    # 说明
    cv2.putText(img, "Green line: E-F (mouth orientation)", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    y += 18
    cv2.putText(img, "Negative angle = drooping", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    return img


def visualize_mouth_tilt(frame: np.ndarray, landmarks, w: int, h: int,
                         tilt: MouthTiltMeasure) -> np.ndarray:
    """可视化嘴部倾斜度 (C047, C062)"""
    img = frame.copy()

    # 面中线 (白色虚线)
    cv2.line(img, (int(tilt.midline_top[0]), int(tilt.midline_top[1])),
             (int(tilt.midline_bot[0]), int(tilt.midline_bot[1])), (255, 255, 255), 2)

    # 唇部中线 (橙色)
    cv2.line(img, (int(tilt.lip_top_center[0]), int(tilt.lip_top_center[1])),
             (int(tilt.lip_bot_center[0]), int(tilt.lip_bot_center[1])), (0, 165, 255), 2)

    # 嘴角连线 (青色)
    cv2.line(img, (int(tilt.mouth_left[0]), int(tilt.mouth_left[1])),
             (int(tilt.mouth_right[0]), int(tilt.mouth_right[1])), (255, 255, 0), 2)

    # 关键点
    cv2.circle(img, (int(tilt.lip_top_center[0]), int(tilt.lip_top_center[1])), 5, (0, 165, 255), -1)
    cv2.circle(img, (int(tilt.lip_bot_center[0]), int(tilt.lip_bot_center[1])), 5, (0, 165, 255), -1)
    cv2.circle(img, (int(tilt.mouth_left[0]), int(tilt.mouth_left[1])), 5, (255, 255, 0), -1)
    cv2.circle(img, (int(tilt.mouth_right[0]), int(tilt.mouth_right[1])), 5, (255, 255, 0), -1)

    # 文字信息
    y = 30
    cv2.putText(img, "Mouth Tilt Analysis", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y += 30

    # C047
    c047_color = (0, 0, 255) if abs(tilt.lip_vertical_tilt) > 5 else (0, 255, 0)
    cv2.putText(img, f"C047 Lip Vertical Tilt: {tilt.lip_vertical_tilt:+.2f} deg", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, c047_color, 2)
    y += 22

    # C062
    c062_color = (0, 0, 255) if tilt.mouth_horizontal_level > 5 else (0, 255, 0)
    cv2.putText(img, f"C062 Mouth Horizontal: {tilt.mouth_horizontal_level:.2f} deg", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, c062_color, 2)
    y += 25

    # 图例
    cv2.putText(img, "White: Face midline", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    y += 15
    cv2.putText(img, "Orange: Lip midline", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
    y += 15
    cv2.putText(img, "Cyan: Mouth corner line", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    return img


def visualize_resting_symmetry(frame: np.ndarray, landmarks, w: int, h: int,
                               indicators: FacialIndicators,
                               resting: RestingSymmetry) -> np.ndarray:
    """可视化Resting Symmetry评估结果"""
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

    # 信息面板背景
    panel_h = 200
    cv2.rectangle(img, (5, 5), (350, panel_h), (0, 0, 0), -1)
    cv2.rectangle(img, (5, 5), (350, panel_h), (255, 255, 255), 1)

    # 标题
    y = 25
    cv2.putText(img, "RESTING SYMMETRY", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    y += 30

    # 眼
    eye_color = (0, 255, 0) if resting.eye_score == 0 else (0, 0, 255)
    cv2.putText(img, f"Eye: {resting.eye_status} (score={resting.eye_score})", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, eye_color, 1)
    y += 22

    # 颊
    cheek_color = (0, 255, 0) if resting.cheek_score == 0 else (
        (0, 165, 255) if resting.cheek_score == 1 else (0, 0, 255))
    cv2.putText(img, f"Cheek: {resting.cheek_status} (score={resting.cheek_score})", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, cheek_color, 1)
    y += 22

    # 嘴
    mouth_color = (0, 255, 0) if resting.mouth_score == 0 else (0, 0, 255)
    cv2.putText(img, f"Mouth: {resting.mouth_status} (score={resting.mouth_score})", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, mouth_color, 1)
    y += 25

    # 分隔线
    cv2.line(img, (15, y - 5), (335, y - 5), (100, 100, 100), 1)
    y += 10

    # 总分
    total_color = (0, 255, 0) if resting.total_score == 0 else (0, 0, 255)
    cv2.putText(img, f"Total Score: {resting.total_score}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, total_color, 2)
    y += 25

    # 患侧
    side_color = (0, 255, 255) if resting.affected_side != "不确定" else (128, 128, 128)
    cv2.putText(img, f"Affected Side: {resting.affected_side}", (15, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, side_color, 1)

    return img


def visualize_all_indicators(frame: np.ndarray, landmarks, w: int, h: int,
                             indicators: FacialIndicators) -> np.ndarray:
    """可视化所有面部指标"""
    img = frame.copy()

    # 绘制眼部轮廓
    for idx_list, color in [(LM.EYE_CONTOUR_L, (255, 0, 0)), (LM.EYE_CONTOUR_R, (0, 165, 255))]:
        pts = pts2d(landmarks, idx_list, w, h).astype(np.int32)
        cv2.polylines(img, [pts], True, color, 1)

    # 绘制眼部中心点
    for idx, color in [(LM.EYE_INNER_L, (255, 0, 0)), (LM.EYE_INNER_R, (0, 165, 255)),
                       (LM.EYE_OUTER_L, (255, 0, 0)), (LM.EYE_OUTER_R, (0, 165, 255))]:
        pt = pt2d(landmarks[idx], w, h)
        cv2.circle(img, (int(pt[0]), int(pt[1])), 3, color, -1)

    # 绘制嘴部
    l_mouth = pt2d(landmarks[LM.MOUTH_L], w, h)
    r_mouth = pt2d(landmarks[LM.MOUTH_R], w, h)
    cv2.circle(img, (int(l_mouth[0]), int(l_mouth[1])), 5, (255, 0, 0), -1)
    cv2.circle(img, (int(r_mouth[0]), int(r_mouth[1])), 5, (0, 165, 255), -1)
    cv2.line(img, (int(l_mouth[0]), int(l_mouth[1])),
             (int(r_mouth[0]), int(r_mouth[1])), (255, 255, 255), 1)

    # 绘制口角分析点 (论文方法)
    oral = indicators.oral_angle
    cv2.circle(img, (int(oral.A[0]), int(oral.A[1])), 4, (0, 0, 255), -1)  # A-红
    cv2.circle(img, (int(oral.B[0]), int(oral.B[1])), 4, (255, 0, 0), -1)  # B-蓝
    cv2.circle(img, (int(oral.O[0]), int(oral.O[1])), 4, (255, 255, 255), -1)  # O-白

    # 绘制鼻唇沟
    l_ala = pt2d(landmarks[LM.NOSE_ALA_L], w, h)
    r_ala = pt2d(landmarks[LM.NOSE_ALA_R], w, h)
    cv2.line(img, (int(l_ala[0]), int(l_ala[1])), (int(l_mouth[0]), int(l_mouth[1])), (255, 0, 0), 2)
    cv2.line(img, (int(r_ala[0]), int(r_ala[1])), (int(r_mouth[0]), int(r_mouth[1])), (0, 165, 255), 2)

    # 文字信息面板
    y = 30
    cv2.putText(img, "Facial Indicators", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    y += 35

    # ICD
    cv2.putText(img, f"ICD: {indicators.icd:.1f}px", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y += 20

    # 眼部
    cv2.putText(img, f"Eye Area L/R: {indicators.eye_area_ratio:.3f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)
    y += 20
    cv2.putText(img, f"Palpebral H L/R: {indicators.palpebral_height_ratio:.3f}", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y += 20

    # 口角
    cv2.putText(img, f"AOE(R): {oral.AOE_angle:+.1f} deg  BOF(L): {oral.BOF_angle:+.1f} deg", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y += 20
    cv2.putText(img, f"Oral Asym: {oral.angle_asymmetry:.1f} deg", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)
    y += 20

    # 嘴部倾斜
    tilt = indicators.mouth_tilt
    cv2.putText(img, f"C047: {tilt.lip_vertical_tilt:+.1f} deg  C062: {tilt.mouth_horizontal_level:.1f} deg", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    y += 20

    # NLF
    cv2.putText(img, f"NLF L/R: {indicators.nlf_ratio:.3f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img


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
    """找静息峰值帧 (EAR最大=眼睛最睁开)"""
    max_ear = -1.0
    max_idx = 0

    for i, lm in enumerate(landmarks_seq):
        if lm is None:
            continue
        l_ear = compute_ear(lm, w, h, True)
        r_ear = compute_ear(lm, w, h, False)
        avg_ear = (l_ear + r_ear) / 2
        if avg_ear > max_ear:
            max_ear = avg_ear
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


# 动作到峰值查找函数的映射
PEAK_FINDERS = {
    "NeutralFace": find_peak_frame_neutral,
    "Smile": find_peak_frame_smile,
    "ShowTeeth": find_peak_frame_smile,
    "CloseEyeSoftly": find_peak_frame_close_eye,
    "CloseEyeHardly": find_peak_frame_close_eye,
    "VoluntaryEyeBlink": find_peak_frame_close_eye,
    "SpontaneousEyeBlink": find_peak_frame_close_eye,
}


# =============================================================================
# 主处理函数
# =============================================================================

def process_single_action(
        extractor: LandmarkExtractor,
        video_info: Dict[str, Any],
        output_dir: Path,
        action_name: str
) -> Optional[Dict[str, Any]]:
    """处理单个动作视频"""
    video_path = video_info["file_path"]
    start_frame = video_info.get("start_frame", 0)
    end_frame = video_info.get("end_frame", None)
    fps = video_info.get("fps", 30.0)

    print(f"  处理动作: {action_name}")
    print(f"    视频: {video_path}")
    print(f"    帧范围: {start_frame} - {end_frame}, FPS: {fps}")

    if not os.path.exists(video_path):
        print(f"    [!] 视频文件不存在!")
        return None

    # 提取landmarks
    landmarks_seq, frames_seq = extractor.extract_sequence(video_path, start_frame, end_frame)

    if not landmarks_seq or not frames_seq:
        print(f"    [!] 无法提取landmarks!")
        return None

    # 获取视频尺寸
    h, w = frames_seq[0].shape[:2]
    print(f"    帧数: {len(frames_seq)}, 尺寸: {w}x{h}")

    # 找峰值帧
    peak_finder = PEAK_FINDERS.get(action_name, find_peak_frame_neutral)
    peak_idx = peak_finder(landmarks_seq, frames_seq, w, h)

    peak_landmarks = landmarks_seq[peak_idx]
    peak_frame = frames_seq[peak_idx]

    if peak_landmarks is None:
        print(f"    [!] 峰值帧无landmarks!")
        return None

    print(f"    峰值帧: {peak_idx}")

    # 提取指标
    indicators = extract_indicators(peak_landmarks, w, h)

    # 生成可视化
    action_dir = output_dir / action_name
    action_dir.mkdir(parents=True, exist_ok=True)

    # 保存原始峰值帧
    cv2.imwrite(str(action_dir / "peak_raw.jpg"), peak_frame)

    # 保存带标注的峰值帧
    vis_frame = visualize_all_indicators(peak_frame, peak_landmarks, w, h, indicators)
    cv2.imwrite(str(action_dir / "peak_indicators.jpg"), vis_frame)

    # 保存口角角度详细可视化
    oral_vis = visualize_oral_angle(peak_frame, peak_landmarks, w, h, indicators.oral_angle)
    cv2.imwrite(str(action_dir / "peak_oral_angle.jpg"), oral_vis)

    # 保存嘴部倾斜度可视化
    tilt_vis = visualize_mouth_tilt(peak_frame, peak_landmarks, w, h, indicators.mouth_tilt)
    cv2.imwrite(str(action_dir / "peak_mouth_tilt.jpg"), tilt_vis)

    # 如果是NeutralFace，生成Resting Symmetry评估
    resting_dict = None
    if action_name == "NeutralFace":
        resting = compute_resting_symmetry(indicators)
        resting_vis = visualize_resting_symmetry(peak_frame, peak_landmarks, w, h, indicators, resting)
        cv2.imwrite(str(action_dir / "resting_symmetry.jpg"), resting_vis)

        resting_dict = {
            "eye": {
                "status": resting.eye_status,
                "score": resting.eye_score,
                "detail": resting.eye_detail
            },
            "cheek": {
                "status": resting.cheek_status,
                "score": resting.cheek_score,
                "detail": resting.cheek_detail
            },
            "mouth": {
                "status": resting.mouth_status,
                "score": resting.mouth_score,
                "detail": resting.mouth_detail
            },
            "total_score": resting.total_score,
            "affected_side": resting.affected_side
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
        "oral_angle": {
            "AOE_angle_deg": oral.AOE_angle,
            "BOF_angle_deg": oral.BOF_angle,
            "angle_diff": oral.angle_diff,
            "angle_asymmetry": oral.angle_asymmetry,
            "points": {
                "A_right_corner": list(oral.A),
                "B_left_corner": list(oral.B),
                "C_right_lip_peak": list(oral.C),
                "D_left_lip_peak": list(oral.D),
                "E_mouth_orifice_R": list(oral.E),
                "F_mouth_orifice_L": list(oral.F),
                "O_midpoint_vertex": list(oral.O),
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

    with open(action_dir / "indicators.json", 'w', encoding='utf-8') as f:
        json.dump(indicators_dict, f, indent=2, ensure_ascii=False)

    print(f"    [OK] 口角角度: AOE(R)={oral.AOE_angle:+.2f} deg, BOF(L)={oral.BOF_angle:+.2f} deg")
    print(
        f"    [OK] C047唇部倾斜: {tilt.lip_vertical_tilt:+.2f} deg, C062嘴角水平: {tilt.mouth_horizontal_level:.2f} deg")
    print(f"    [OK] 眼面积比: {indicators.eye_area_ratio:.3f}")
    print(f"    [OK] NLF比值: {indicators.nlf_ratio:.3f}")

    return indicators_dict


def process_examination(
        examination: Dict[str, Any],
        db_path: str,
        output_dir: Path,
        extractor: LandmarkExtractor
) -> Dict[str, Any]:
    """处理单个检查"""
    exam_id = examination["examination_id"]
    patient_id = examination["patient_id"]
    capture_datetime = examination["capture_datetime"]

    print(f"\n{'=' * 60}")
    print(f"处理检查: {exam_id}")
    print(f"患者ID: {patient_id}")
    print(f"采集时间: {capture_datetime}")
    print(f"{'=' * 60}")

    # 获取视频列表
    videos = db_fetch_videos_for_exam(db_path, exam_id)
    print(f"找到 {len(videos)} 个动作视频")

    # 获取医生标注
    labels = db_fetch_labels(db_path, exam_id)
    print(f"医生标注: {labels}")

    # 创建输出目录
    exam_output_dir = output_dir / exam_id
    exam_output_dir.mkdir(parents=True, exist_ok=True)

    # 处理每个动作
    results = {
        "examination_id": exam_id,
        "patient_id": patient_id,
        "capture_datetime": capture_datetime,
        "ground_truth": labels,
        "actions": {}
    }

    for action_name, video_info in videos.items():
        action_result = process_single_action(extractor, video_info, exam_output_dir, action_name)
        if action_result:
            results["actions"][action_name] = action_result

    # 保存汇总结果
    with open(exam_output_dir / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 生成HTML报告
    generate_html_report(results, exam_output_dir)

    return results


def generate_html_report(results: Dict[str, Any], output_dir: Path):
    """生成HTML报告 (包含Resting Symmetry表)"""
    exam_id = results["examination_id"]
    patient_id = results["patient_id"]
    ground_truth = results.get("ground_truth", {})

    # 获取NeutralFace的Resting Symmetry结果
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
        .resting-symmetry {{ background: #e3f2fd; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .resting-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        .resting-table th, .resting-table td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        .resting-table th {{ background: #1976d2; color: white; }}
        .resting-table tr:nth-child(even) {{ background: #f9f9f9; }}
        .score-normal {{ color: #4CAF50; font-weight: bold; }}
        .score-mild {{ color: #FF9800; font-weight: bold; }}
        .score-severe {{ color: #f44336; font-weight: bold; }}
        .action-card {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
        .action-title {{ font-weight: bold; font-size: 1.2em; color: #2196F3; }}
        .metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 10px; }}
        .metric {{ background: #f9f9f9; padding: 10px; border-radius: 3px; }}
        .metric-label {{ color: #666; font-size: 0.9em; }}
        .metric-value {{ font-size: 1.1em; font-weight: bold; }}
        .images {{ display: flex; gap: 10px; margin-top: 15px; flex-wrap: wrap; }}
        .images img {{ max-width: 320px; border: 1px solid #ddd; border-radius: 5px; }}
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
        <strong>采集时间:</strong> {results.get('capture_datetime', 'N/A')}<br>
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

    # Resting Symmetry 表
    if resting:
        eye_class = "score-normal" if resting["eye"]["score"] == 0 else "score-mild"
        cheek_class = "score-normal" if resting["cheek"]["score"] == 0 else (
            "score-mild" if resting["cheek"]["score"] == 1 else "score-severe")
        mouth_class = "score-normal" if resting["mouth"]["score"] == 0 else "score-mild"
        total_class = "score-normal" if resting["total_score"] == 0 else (
            "score-mild" if resting["total_score"] <= 2 else "score-severe")

        html += f"""
    <div class="resting-symmetry">
        <h2>Resting Symmetry 静态对称性评估 (基于NeutralFace)</h2>
        <table class="resting-table">
            <tr>
                <th>部位</th>
                <th>状态</th>
                <th>评分</th>
                <th>详细说明</th>
            </tr>
            <tr>
                <td><strong>眼 (睑裂)</strong></td>
                <td>{resting["eye"]["status"]}</td>
                <td class="{eye_class}">{resting["eye"]["score"]}</td>
                <td>{resting["eye"]["detail"]}</td>
            </tr>
            <tr>
                <td><strong>颊 (鼻唇沟)</strong></td>
                <td>{resting["cheek"]["status"]}</td>
                <td class="{cheek_class}">{resting["cheek"]["score"]}</td>
                <td>{resting["cheek"]["detail"]}</td>
            </tr>
            <tr>
                <td><strong>嘴</strong></td>
                <td>{resting["mouth"]["status"]}</td>
                <td class="{mouth_class}">{resting["mouth"]["score"]}</td>
                <td>{resting["mouth"]["detail"]}</td>
            </tr>
            <tr style="background: #e3f2fd;">
                <td colspan="2"><strong>总分</strong></td>
                <td class="{total_class}"><strong>{resting["total_score"]}</strong></td>
                <td><strong>判断患侧: {resting["affected_side"]}</strong></td>
            </tr>
        </table>
        <p style="margin-top: 15px; color: #666;">
            <em>评分说明: 眼(0-1分)、颊(0-2分)、嘴(0-1分)。总分越高表示静态不对称越严重。</em>
        </p>
        <div class="images" style="margin-top: 15px;">
            <img src="NeutralFace/resting_symmetry.jpg" alt="Resting Symmetry可视化">
        </div>
    </div>
"""

    html += """
    <h2>动作分析结果</h2>
"""

    for action_name, action_data in results.get("actions", {}).items():
        oral = action_data.get("oral_angle", {})
        eye = action_data.get("eye", {})
        nlf = action_data.get("nlf", {})
        tilt = action_data.get("mouth_tilt", {})

        # 判断异常
        aoe = oral.get("AOE_angle_deg", 0)
        bof = oral.get("BOF_angle_deg", 0)
        asym = oral.get("angle_asymmetry", 0)
        eye_ratio = eye.get("area_ratio", 1)
        nlf_ratio = nlf.get("ratio", 1)
        c047 = tilt.get("C047_lip_vertical_tilt_deg", 0)
        c062 = tilt.get("C062_mouth_horizontal_level_deg", 0)

        oral_class = "warning" if asym > 5 else "normal"
        eye_class = "warning" if abs(eye_ratio - 1) > 0.15 else "normal"
        nlf_class = "warning" if abs(nlf_ratio - 1) > 0.25 else "normal"
        c047_class = "warning" if abs(c047) > 5 else "normal"
        c062_class = "warning" if c062 > 5 else "normal"

        html += f"""
    <div class="action-card">
        <div class="action-title">{action_name}</div>
        <div class="metrics">
            <div class="metric">
                <div class="metric-label">口角角度 (右-AOE)</div>
                <div class="metric-value">{aoe:+.2f} deg</div>
            </div>
            <div class="metric">
                <div class="metric-label">口角角度 (左-BOF)</div>
                <div class="metric-value">{bof:+.2f} deg</div>
            </div>
            <div class="metric">
                <div class="metric-label">口角不对称</div>
                <div class="metric-value {oral_class}">{asym:.2f} deg</div>
            </div>
            <div class="metric">
                <div class="metric-label">眼面积比 (L/R)</div>
                <div class="metric-value {eye_class}">{eye_ratio:.3f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">NLF比值 (L/R)</div>
                <div class="metric-value {nlf_class}">{nlf_ratio:.3f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">C047 唇部倾斜角</div>
                <div class="metric-value {c047_class}">{c047:+.2f} deg</div>
            </div>
            <div class="metric">
                <div class="metric-label">C062 嘴角水平度</div>
                <div class="metric-value {c062_class}">{c062:.2f} deg</div>
            </div>
            <div class="metric">
                <div class="metric-label">ICD</div>
                <div class="metric-value">{action_data.get('icd', 0):.1f}px</div>
            </div>
        </div>
        <div class="images">
            <img src="{action_name}/peak_raw.jpg" alt="原始帧">
            <img src="{action_name}/peak_oral_angle.jpg" alt="口角角度">
            <img src="{action_name}/peak_mouth_tilt.jpg" alt="嘴部倾斜">
        </div>
    </div>
"""

    html += """
</div>
</body>
</html>
"""

    with open(output_dir / "report.html", 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"  [OK] HTML报告已生成: {output_dir / 'report.html'}")


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    print("=" * 70)
    print("面部临床分级测试脚本")
    print("=" * 70)

    # 检查配置
    print(f"\n配置信息:")
    print(f"  数据库: {DATABASE_PATH}")
    print(f"  模型: {MEDIAPIPE_MODEL_PATH}")
    print(f"  输出: {OUTPUT_DIR}")
    print(f"  患者限制: {PATIENT_LIMIT}")
    print(f"  目标检查: {TARGET_EXAM_ID}")

    # 检查文件是否存在
    if not os.path.exists(DATABASE_PATH):
        print(f"\n[ERROR] 数据库不存在: {DATABASE_PATH}")
        return

    if not os.path.exists(MEDIAPIPE_MODEL_PATH):
        print(f"\n[ERROR] MediaPipe模型不存在: {MEDIAPIPE_MODEL_PATH}")
        return

    # 创建输出目录
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取检查列表
    print(f"\n从数据库获取检查记录...")
    examinations = db_fetch_examinations(DATABASE_PATH, TARGET_EXAM_ID, PATIENT_LIMIT)
    print(f"找到 {len(examinations)} 个检查记录")

    if not examinations:
        print("[ERROR] 没有找到有效的检查记录")
        return

    # 初始化landmark提取器
    print(f"\n初始化MediaPipe...")

    all_results = []

    with LandmarkExtractor(MEDIAPIPE_MODEL_PATH) as extractor:
        for i, exam in enumerate(examinations):
            print(f"\n[{i + 1}/{len(examinations)}]", end="")
            result = process_examination(exam, DATABASE_PATH, output_dir, extractor)
            all_results.append(result)

    # 生成汇总报告
    print(f"\n\n{'=' * 70}")
    print("处理完成!")
    print(f"{'=' * 70}")
    print(f"处理了 {len(all_results)} 个检查")
    print(f"输出目录: {output_dir}")

    # 汇总统计
    total_actions = sum(len(r.get("actions", {})) for r in all_results)
    print(f"总共分析了 {total_actions} 个动作视频")

    # 保存汇总
    summary = {
        "run_time": datetime.now().isoformat(),
        "total_examinations": len(all_results),
        "total_actions": total_actions,
        "results": all_results
    }

    with open(output_dir / "all_results.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n汇总结果已保存: {output_dir / 'all_results.json'}")


if __name__ == "__main__":
    main()