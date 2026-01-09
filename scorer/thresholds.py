#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阈值配置模块 (thresholds.py)
============================

集中管理所有动作处理模块的阈值配置，方便统一调整和管理。

"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Thresholds:
    """
    所有模块的阈值配置

    命名规则: {动作名}_{指标名}

    阈值设计原则:
    - 对于"越小越好"的指标(如EAR闭眼): 使用 <= 阈值判断
    - 对于"越大越好"的指标(如眉眼距变化): 使用 >= 阈值判断
    - 对于"接近1越好"的指标(如比值): 使用 abs(ratio-1.0) <= 阈值判断
    """

    # =========================================================================
    # 通用阈值
    # =========================================================================

    # EAR (Eye Aspect Ratio) 闭眼阈值
    EYE_CLOSURE_EAR: float = 0.22  # EAR < 此值认为眼睛闭合

    # =========================================================================
    # 眼睛闭合度相关阈值（基于面积比）
    # =========================================================================

    # 闭合度阈值（1 - current_area/baseline_area）
    EYE_CLOSURE_RATIO_CLOSED: float = 0.70  # 闭合度 > 认为眼睛闭合
    EYE_CLOSURE_RATIO_PARTIAL: float = 0.50  # 闭合度 > 认为部分闭合
    EYE_CLOSURE_RATIO_MINIMAL: float = 0.20  # 闭合度 > 认为有闭眼动作

    # 眼睛对称性阈值（|left - right| / max(left, right)）
    EYE_SYMMETRY_NORMAL: float = 0.08  # 不对称比 < 8% 认为对称
    EYE_SYMMETRY_MILD: float = 0.20  # 不对称比 < 20% 认为轻度不对称
    EYE_SYMMETRY_MODERATE: float = 0.30  # 不对称比 < 30% 认为中度不对称

    # 眼睛同步性阈值
    EYE_SYNC_PEARSON_GOOD: float = 0.85  # Pearson相关 > 0.85 认为同步良好
    EYE_SYNC_PEARSON_FAIR: float = 0.60  # Pearson相关 > 0.60 认为同步一般
    EYE_SYNC_PEAK_DIFF_MAX: int = 5  # 峰值帧差 < 5帧 认为同步

    # =========================================================================
    # 鼓腮 BlowCheek 阈值
    # =========================================================================

    # bulge = (base_rel_z - current_rel_z) / ICD，rel_z = cheek_z - nose_z
    BLOW_CHEEK_BULGE_MIN: float = 0.005  # bulge > 0.5% 认为有鼓腮动作
    BLOW_CHEEK_BULGE_GOOD: float = 0.02  # bulge > 2% 认为鼓腮明显
    BLOW_CHEEK_ASYM_THRESHOLD: float = 0.15  # 左右不对称比 > 15% 判定患侧
    BLOW_CHEEK_BASELINE_FRAMES: int = 10  # 用视频前10帧建立内部baseline

    # 唇封闭距离归一化阈值 (seal_total / ICD)
    MOUTH_SEAL: float = 0.032

    # 嘴部高度归一化阈值 (mouth_height / ICD)
    MOUTH_HEIGHT: float = 0.032

    # 嘴唇内圈面积增幅阈值
    MOUTH_INNER_AREA_INC: float = 3.5
    MOUTH_INNER_AREA_BASE_EPS: float = 1e-4

    # =========================================================================
    # 撅嘴 LipPucker 阈值
    # =========================================================================

    LIP_PUCKER_PROTRUSION_MIN: float = 0.005
    LIP_PUCKER_PROTRUSION_GOOD: float = 0.015
    LIP_PUCKER_WIDTH_RATIO_MAX: float = 0.90
    LIP_PUCKER_BASELINE_FRAMES: int = 10
    LIP_PUCKER_WIDTH_RATIO: float = 0.85
    LIP_PUCKER_Z_DELTA: float = 0.01

    # LipPucker 面瘫检测阈值
    LIP_PUCKER_OFFSET_THRESHOLD: float = 0.020  # 偏移 > 2% 判定有面瘫
    LIP_PUCKER_CORNER_ASYM_TRACE: float = 0.10
    LIP_PUCKER_CORNER_ASYM_MILD: float = 0.20
    LIP_PUCKER_CORNER_ASYM_MODERATE: float = 0.35
    LIP_PUCKER_CORNER_ASYM_SEVERE: float = 0.50

    # =========================================================================
    # 对称性判断通用阈值
    # =========================================================================

    SYMMETRY_NORMAL: float = 0.10  # 偏差 < 10% 认为对称
    SYMMETRY_MILD: float = 0.20  # 偏差 < 20% 认为轻度不对称
    SYMMETRY_MODERATE: float = 0.35  # 偏差 < 35% 认为中度不对称

    # 运动幅度 (excursion) 对称性阈值
    EXCURSION_SYMMETRIC: float = 0.85
    EXCURSION_MILD_ASYM: float = 0.60
    EXCURSION_MODERATE_ASYM: float = 0.30

    # 口角角度不对称阈值 (度)
    ORAL_ANGLE_SYMMETRIC: float = 3.0
    ORAL_ANGLE_MILD: float = 6.0
    ORAL_ANGLE_MODERATE: float = 10.0
    ORAL_ANGLE_SEVERE: float = 15.0

    # =========================================================================
    # 面瘫侧别检测通用阈值
    # =========================================================================

    # 嘴唇中线偏移阈值 (归一化到ICD)
    PALSY_LIP_OFFSET_THRESHOLD: float = 0.025  # 偏移 > 2.5% 判定有面瘫

    # 严重度分级阈值 (基于嘴唇偏移)
    SEVERITY_NORMAL: float = 0.03  # < 3% 正常
    SEVERITY_MILD: float = 0.06  # < 6% 轻度
    SEVERITY_MODERATE: float = 0.10  # < 10% 中度
    SEVERITY_SEVERE: float = 0.15  # < 15% 重度

    # =========================================================================
    # NeutralFace 静息面阈值
    # =========================================================================

    NEUTRAL_MIN_EAR: float = 0.20  # EAR > 此值认为眼睛睁开
    NEUTRAL_MAX_MOUTH_RATIO: float = 0.15  # 嘴高/嘴宽 < 此值认为嘴闭合
    NEUTRAL_PALP_DEVIATION: float = 0.12  # 眼睑裂比偏离 > 12% 认为不对称
    NEUTRAL_NLF_DEVIATION: float = 0.12  # 鼻唇沟比偏离 > 12% 认为不对称
    NEUTRAL_AREA_DEVIATION: float = 0.10  # 眼睛面积比偏离 > 10% 认为不对称

    # =========================================================================
    # CloseEye 闭眼动作阈值
    # =========================================================================

    CLOSE_EYE_EAR_CLOSED: float = 0.2
    CLOSE_EYE_SYMMETRY: float = 0.10

    # 联动检测阈值 (嘴部变化)
    CLOSE_EYE_SYNKINESIS_SEVERE: float = 0.15
    CLOSE_EYE_SYNKINESIS_MODERATE: float = 0.08
    CLOSE_EYE_SYNKINESIS_MILD: float = 0.04

    # =========================================================================
    # EyeBlink 眨眼动作阈值
    # =========================================================================

    BLINK_WORSE_RATIO: float = 0.60  # 某侧闭合不良比例 > 60%
    BLINK_PERSISTENT_RATIO: float = 0.60  # 持续检测到同侧异常的比例
    BLINK_MIN_CLOSURE: float = 0.20  # 最小有效闭合度
    BLINK_ASYMMETRY_MIN: float = 0.01  # 不对称比 < 1% 认为对称
    BLINK_MAX_PEAK_CLOSURE: float = 0.10  # 峰值帧闭合度阈值
    BLINK_SYMMETRY_RATIO_MIN: float = 0.01  # 对称性比阈值

    # =========================================================================
    # RaiseEyebrow 抬眉动作阈值
    # =========================================================================

    RAISE_EYEBROW_CHANGE_MIN: float = 5.0  # 最小变化量(像素)
    RAISE_EYEBROW_SYMMETRY: float = 0.30
    RAISE_EYEBROW_SMOOTH_WIN: int = 5

    # 面瘫检测阈值
    RAISE_EYEBROW_ASYM_NORMAL: float = 0.10  # 不对称 < 10% 正常

    # 严重度分级
    RAISE_EYEBROW_CHANGE_SEVERE: float = 0.20
    RAISE_EYEBROW_CHANGE_MODERATE: float = 0.12
    RAISE_EYEBROW_CHANGE_MILD: float = 0.06

    # 联动检测
    RAISE_EYEBROW_SYNKINESIS_SEVERE: float = 0.15
    RAISE_EYEBROW_SYNKINESIS_MODERATE: float = 0.08
    RAISE_EYEBROW_SYNKINESIS_MILD: float = 0.04

    # =========================================================================
    # Smile 微笑动作阈值
    # =========================================================================

    SMILE_WIDTH_CHANGE_MIN: float = 10.0
    SMILE_SYMMETRY: float = 0.20

    # 面瘫检测阈值
    SMILE_ASYM_SYMMETRIC: float = 0.15  # 不对称 < 15% 对称

    # 严重度分级 (基于嘴角上提不对称)
    SMILE_ASYM_TRACE: float = 0.02  # < 2% 完美
    SMILE_ASYM_NORMAL: float = 0.03  # < 3% 正常
    SMILE_ASYM_MILD: float = 0.08  # < 8% 轻度
    SMILE_ASYM_MODERATE: float = 0.15  # < 15% 中度
    SMILE_ASYM_SEVERE: float = 0.25  # < 25% 重度

    # 口角角度不对称分级
    SMILE_ORAL_ASYM_TRACE: float = 0.08
    SMILE_ORAL_ASYM_MILD: float = 0.18
    SMILE_ORAL_ASYM_MODERATE: float = 0.30
    SMILE_ORAL_ASYM_SEVERE: float = 0.45

    # =========================================================================
    # ShowTeeth 露齿动作阈值
    # =========================================================================

    SHOW_TEETH_WIDTH_CHANGE_MIN: float = 10.0
    SHOW_TEETH_SYMMETRY: float = 0.20
    SHOW_TEETH_OFFSET_THRESHOLD: float = 0.025  # 偏移阈值

    # 联动检测
    SHOW_TEETH_SYNKINESIS_SEVERE: float = 0.15
    SHOW_TEETH_SYNKINESIS_MODERATE: float = 0.08

    # =========================================================================
    # ShrugNose 皱鼻动作阈值
    # =========================================================================

    SHRUG_NOSE_CHANGE_MIN: float = 3.0
    SHRUG_NOSE_SYMMETRY: float = 0.25

    # 面瘫检测阈值
    SHRUG_NOSE_ASYM_NORMAL: float = 0.08  # 不对称 < 8% 正常

    # 严重度分级
    SHRUG_NOSE_CHANGE_SEVERE: float = 0.18
    SHRUG_NOSE_CHANGE_MODERATE: float = 0.10
    SHRUG_NOSE_CHANGE_MILD: float = 0.05

    # 联动检测
    SHRUG_NOSE_SYNKINESIS_SEVERE: float = 0.15
    SHRUG_NOSE_SYNKINESIS_MODERATE: float = 0.08
    SHRUG_NOSE_SYNKINESIS_MILD: float = 0.04

    # =========================================================================
    # Synkinesis 联动运动通用阈值
    # =========================================================================

    SYNKINESIS_MOUTH_MILD: float = 0.05
    SYNKINESIS_MOUTH_MODERATE: float = 0.10
    SYNKINESIS_MOUTH_SEVERE: float = 0.20

    SYNKINESIS_EYE_MILD: float = 0.08
    SYNKINESIS_EYE_MODERATE: float = 0.15
    SYNKINESIS_EYE_SEVERE: float = 0.25

    # =========================================================================
    # Sunnybrook 评分相关阈值
    # =========================================================================

    RESTING_EYE_RATIO: float = 0.20
    RESTING_CHEEK_NLF_RATIO: float = 0.15
    RESTING_CHEEK_NLF_SEVERE: float = 0.30
    RESTING_MOUTH_ANGLE_DIFF: float = 5.0

    # Sunnybrook 眼睑裂比阈值
    SUNNYBROOK_PALP_RATIO_NORMAL: float = 0.85  # 比值 < 0.85 认为异常

    # Sunnybrook NLF比阈值
    SUNNYBROOK_NLF_RATIO_MILD: float = 0.85  # 0.75-0.85 轻度
    SUNNYBROOK_NLF_RATIO_SEVERE: float = 0.75  # < 0.75 重度

    VOLUNTARY_COMPLETE: float = 0.85
    VOLUNTARY_ALMOST: float = 0.70
    VOLUNTARY_INITIATE: float = 0.40
    VOLUNTARY_TRACE: float = 0.15

    # =========================================================================
    # clinical_base.py 中使用的阈值
    # =========================================================================

    EAR_DIFF_SYMMETRIC: float = 0.15  # EAR差异 < 15% 对称
    EAR_DIFF_TRACE: float = 0.05  # EAR差异 < 5% 完美对称
    AREA_RATIO_SYMMETRIC: float = 0.15  # 面积比偏离 < 15% 对称
    CLOSURE_MIN_VALID: float = 0.10  # 最小有效闭合度
    CLOSURE_ASYM_SYMMETRIC: float = 0.15  # 闭合不对称 < 15% 对称
    CHANGE_DIFF_SYMMETRIC: float = 0.15  # 变化差异 < 15% 对称
    REDUCTION_ASYM_SYMMETRIC: float = 0.10  # 缩减不对称 < 10% 对称


# 全局单例
THR = Thresholds()


# =========================================================================
# 便捷函数
# =========================================================================

def is_eye_closed(ear: float) -> bool:
    """判断眼睛是否闭合"""
    return ear < THR.EYE_CLOSURE_EAR


def get_eye_closure_level(closure_ratio: float) -> int:
    """
    获取眼睛闭合程度等级

    Args:
        closure_ratio: 闭合度 (0-1)

    Returns:
        0: 睁开
        1: 轻微闭合
        2: 部分闭合
        3: 完全闭合
    """
    if closure_ratio < THR.EYE_CLOSURE_RATIO_MINIMAL:
        return 0
    elif closure_ratio < THR.EYE_CLOSURE_RATIO_PARTIAL:
        return 1
    elif closure_ratio < THR.EYE_CLOSURE_RATIO_CLOSED:
        return 2
    else:
        return 3


def get_eye_symmetry_level(asymmetry_ratio: float) -> int:
    """
    获取眼睛对称性等级

    Args:
        asymmetry_ratio: 不对称比 (|L-R| / max(L,R))

    Returns:
        0: 对称
        1: 轻度不对称
        2: 中度不对称
        3: 重度不对称
    """
    if asymmetry_ratio < THR.EYE_SYMMETRY_NORMAL:
        return 0
    elif asymmetry_ratio < THR.EYE_SYMMETRY_MILD:
        return 1
    elif asymmetry_ratio < THR.EYE_SYMMETRY_MODERATE:
        return 2
    else:
        return 3


def get_eye_sync_level(pearson_corr: float, peak_diff: int) -> int:
    """获取眼睛同步性等级"""
    if pearson_corr >= THR.EYE_SYNC_PEARSON_GOOD and peak_diff <= THR.EYE_SYNC_PEAK_DIFF_MAX:
        return 0
    elif pearson_corr >= THR.EYE_SYNC_PEARSON_FAIR:
        return 1
    else:
        return 2


def get_severity_level(offset_norm: float) -> int:
    """
    根据偏移量获取严重度等级

    Returns:
        1: 正常
        2: 轻度
        3: 中度
        4: 重度
        5: 完全面瘫
    """
    if offset_norm < THR.SEVERITY_NORMAL:
        return 1
    elif offset_norm < THR.SEVERITY_MILD:
        return 2
    elif offset_norm < THR.SEVERITY_MODERATE:
        return 3
    elif offset_norm < THR.SEVERITY_SEVERE:
        return 4
    else:
        return 5


def get_severity_text(level: int) -> str:
    """获取严重度文字描述"""
    texts = {
        1: "正常",
        2: "轻度异常",
        3: "中度异常",
        4: "重度异常",
        5: "完全面瘫",
    }
    return texts.get(level, "未知")


def is_symmetric(ratio: float, threshold: float = None) -> bool:
    """判断比值是否对称（接近1.0）"""
    if threshold is None:
        threshold = THR.SYMMETRY_NORMAL
    return abs(ratio - 1.0) <= threshold


def get_asymmetry_level(ratio: float) -> int:
    """获取不对称程度等级"""
    deviation = abs(ratio - 1.0)
    if deviation <= THR.SYMMETRY_NORMAL:
        return 0
    elif deviation <= THR.SYMMETRY_MILD:
        return 1
    elif deviation <= THR.SYMMETRY_MODERATE:
        return 2
    else:
        return 3


def get_oral_angle_asymmetry_level(angle_diff: float) -> int:
    """获取口角角度不对称程度等级"""
    if angle_diff < THR.ORAL_ANGLE_SYMMETRIC:
        return 0
    elif angle_diff < THR.ORAL_ANGLE_MILD:
        return 1
    elif angle_diff < THR.ORAL_ANGLE_MODERATE:
        return 2
    else:
        return 3


def get_synkinesis_score(change_ratio: float, region: str = "mouth") -> int:
    """获取联动运动评分"""
    if region == "mouth":
        mild = THR.SYNKINESIS_MOUTH_MILD
        moderate = THR.SYNKINESIS_MOUTH_MODERATE
        severe = THR.SYNKINESIS_MOUTH_SEVERE
    else:  # eye
        mild = THR.SYNKINESIS_EYE_MILD
        moderate = THR.SYNKINESIS_EYE_MODERATE
        severe = THR.SYNKINESIS_EYE_SEVERE

    if change_ratio < mild:
        return 0
    elif change_ratio < moderate:
        return 1
    elif change_ratio < severe:
        return 2
    else:
        return 3


def get_voluntary_score(ratio: float) -> int:
    """
    根据运动比例获取Voluntary Movement评分

    Args:
        ratio: 患侧/健侧运动比例 (0-1)

    Returns:
        5: 完整运动
        4: 几乎完整
        3: 有启动
        2: 轻微启动
        1: 无运动
    """
    if ratio >= THR.VOLUNTARY_COMPLETE:
        return 5
    elif ratio >= THR.VOLUNTARY_ALMOST:
        return 4
    elif ratio >= THR.VOLUNTARY_INITIATE:
        return 3
    elif ratio >= THR.VOLUNTARY_TRACE:
        return 2
    else:
        return 1


if __name__ == "__main__":
    # 打印所有阈值用于检查
    print("=" * 70)
    print("阈值配置一览 ")
    print("=" * 70)

    for field_name in sorted(dir(THR)):
        if not field_name.startswith("_"):
            value = getattr(THR, field_name)
            if not callable(value):
                print(f"{field_name}: {value}")