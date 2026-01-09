#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阈值配置模块 (thresholds.py)
============================

集中管理所有动作处理模块的阈值配置，方便统一调整和管理。

使用方式:
    from thresholds import THR

    if seal_norm <= THR.BLOW_CHEEK_SEAL:
        # 判断闭唇
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
    EYE_CLOSURE_RATIO_CLOSED: float = 0.90  # 闭合度 > 认为眼睛闭合
    EYE_CLOSURE_RATIO_PARTIAL: float = 0.50  # 闭合度 > 认为部分闭合
    EYE_CLOSURE_RATIO_MINIMAL: float = 0.20  # 闭合度 > 认为有闭眼动作

    # 眼睛对称性阈值（|left - right| / max(left, right)）
    EYE_SYMMETRY_NORMAL: float = 0.15  # 不对称比 < 15% 认为对称
    EYE_SYMMETRY_MILD: float = 0.25  # 不对称比 < 25% 认为轻度不对称
    EYE_SYMMETRY_MODERATE: float = 0.40  # 不对称比 < 40% 认为中度不对称

    # 眼睛同步性阈值
    EYE_SYNC_PEARSON_GOOD: float = 0.85  # Pearson相关 > 0.85 认为同步良好
    EYE_SYNC_PEARSON_FAIR: float = 0.60  # Pearson相关 > 0.60 认为同步一般
    EYE_SYNC_PEAK_DIFF_MAX: int = 5  # 峰值帧差 < 5帧 认为同步

    # =========================================================================
    # 鼓腮 BlowCheek 相对鼻尖深度阈值
    # =========================================================================

    # bulge = (base_rel_z - current_rel_z) / ICD，rel_z = cheek_z - nose_z
    BLOW_CHEEK_BULGE_MIN: float = 0.005  # bulge > 0.5% 认为有鼓腮动作
    BLOW_CHEEK_BULGE_GOOD: float = 0.02  # bulge > 2% 认为鼓腮明显
    BLOW_CHEEK_ASYM_THRESHOLD: float = 0.15  # 左右不对称比 > 15% 判定患侧
    BLOW_CHEEK_BASELINE_FRAMES: int = 10  # 用视频前10帧建立内部baseline

    # =========================================================================
    # 撅嘴 LipPucker 相对鼻尖深度阈值
    # =========================================================================

    # protrusion = (base_rel_z - current_rel_z) / ICD，rel_z = lip_z - nose_z
    LIP_PUCKER_PROTRUSION_MIN: float = 0.005  # protrusion > 0.5% 认为有撅嘴动作
    LIP_PUCKER_PROTRUSION_GOOD: float = 0.015  # protrusion > 1.5% 认为撅嘴明显
    LIP_PUCKER_WIDTH_RATIO_MAX: float = 0.90  # 嘴宽比 < 90% 认为有收缩
    LIP_PUCKER_BASELINE_FRAMES: int = 10  # 用视频前10帧建立内部baseline

    # 对称性判断阈值 (ratio 偏离 1.0 的程度)
    SYMMETRY_NORMAL: float = 0.10  # 偏差 < 10% 认为对称
    SYMMETRY_MILD: float = 0.20  # 偏差 < 20% 认为轻度不对称
    SYMMETRY_MODERATE: float = 0.35  # 偏差 < 35% 认为中度不对称

    # 运动幅度 (excursion) 对称性阈值
    EXCURSION_SYMMETRIC: float = 0.85  # 比值 > 0.85 认为运动对称
    EXCURSION_MILD_ASYM: float = 0.60  # 比值 > 0.60 认为轻度不对称
    EXCURSION_MODERATE_ASYM: float = 0.30  # 比值 > 0.30 认为中度不对称

    # 口角角度不对称阈值 (度)
    ORAL_ANGLE_SYMMETRIC: float = 3.0  # < 3° 认为对称
    ORAL_ANGLE_MILD: float = 6.0  # < 6° 认为轻度不对称
    ORAL_ANGLE_MODERATE: float = 10.0  # < 10° 认为中度不对称
    ORAL_ANGLE_SEVERE: float = 15.0  # < 15° 认为重度不对称

    # =========================================================================
    # BlowCheek 鼓腮动作阈值
    # =========================================================================

    # 唇封闭距离归一化阈值 (seal_total / ICD)
    # 值越小表示嘴唇闭合越紧
    MOUTH_SEAL: float = 0.032

    # 嘴部高度归一化阈值 (mouth_height / ICD)
    # 值越小表示嘴巴张开越小
    MOUTH_HEIGHT: float = 0.032

    # 嘴唇内圈面积增幅阈值
    # 计算方式: (current_area / baseline_area) - 1.0
    # 值越大表示嘴张开越多
    MOUTH_INNER_AREA_INC: float = 3.5

    # 嘴唇内圈面积基线最小值 (防止除零)
    MOUTH_INNER_AREA_BASE_EPS: float = 1e-4  # 原1e-5, 增大防止噪声

    # =========================================================================
    # LipPucker 撅嘴动作阈值
    # =========================================================================

    # 嘴宽收缩比例阈值 (越小表示撅嘴越明显)
    LIP_PUCKER_WIDTH_RATIO: float = 0.85  # 当前嘴宽/基线嘴宽 < 此值认为撅嘴

    # 嘴唇 z 轴前移阈值 (归一化到 ICD)
    LIP_PUCKER_Z_DELTA: float = 0.01  # delta_z/ICD > 此值认为前移

    # =========================================================================
    # CloseEye 闭眼动作阈值
    # =========================================================================

    # 闭眼判断 EAR 阈值
    CLOSE_EYE_EAR_CLOSED: float = 0.10  # EAR < 此值认为眼睛闭合

    # 闭眼程度对称性阈值
    CLOSE_EYE_SYMMETRY: float = 0.10  # |L_EAR - R_EAR| / max(L,R) > 此值认为不对称

    # =========================================================================
    # RaiseEyebrow 抬眉动作阈值
    # =========================================================================

    # 眉眼距变化阈值 (像素)
    RAISE_EYEBROW_CHANGE_MIN: float = 5.0  # 变化 > 此值认为有运动

    # 眉眼距变化对称性阈值
    RAISE_EYEBROW_SYMMETRY: float = 0.30  # 左右变化比 < 此值认为不对称

    # 曲线平滑窗口
    RAISE_EYEBROW_SMOOTH_WIN: int = 5

    # =========================================================================
    # ShrugNose 皱鼻动作阈值
    # =========================================================================

    # 鼻翼-内眦距离变化阈值
    SHRUG_NOSE_CHANGE_MIN: float = 3.0  # 变化 > 此值认为有运动

    # 皱鼻对称性阈值
    SHRUG_NOSE_SYMMETRY: float = 0.25

    # =========================================================================
    # Smile 微笑动作阈值
    # =========================================================================

    # 嘴宽变化阈值
    SMILE_WIDTH_CHANGE_MIN: float = 10.0  # 变化 > 此值认为有微笑

    # 微笑对称性阈值
    SMILE_SYMMETRY: float = 0.20

    # =========================================================================
    # ShowTeeth 露齿动作阈值
    # =========================================================================

    # 与 Smile 类似
    SHOW_TEETH_WIDTH_CHANGE_MIN: float = 10.0
    SHOW_TEETH_SYMMETRY: float = 0.20

    # =========================================================================
    # EyeBlink 眨眼动作阈值
    # =========================================================================

    # 眨眼闭合程度阈值
    EYE_BLINK_CLOSURE_RATIO: float = 0.50  # EAR下降比例 > 此值认为有效眨眼

    # 眨眼对称性阈值
    EYE_BLINK_SYMMETRY: float = 0.15

    # =========================================================================
    # Synkinesis 联动运动阈值
    # =========================================================================

    # 嘴部联动阈值 (嘴宽变化比例)
    SYNKINESIS_MOUTH_MILD: float = 0.05  # > 5% 轻度联动
    SYNKINESIS_MOUTH_MODERATE: float = 0.10  # > 10% 中度联动
    SYNKINESIS_MOUTH_SEVERE: float = 0.20  # > 20% 重度联动

    # 眼部联动阈值 (EAR变化比例)
    SYNKINESIS_EYE_MILD: float = 0.08
    SYNKINESIS_EYE_MODERATE: float = 0.15
    SYNKINESIS_EYE_SEVERE: float = 0.25

    # =========================================================================
    # Sunnybrook 评分相关阈值
    # =========================================================================

    # Resting Symmetry 阈值
    RESTING_EYE_RATIO: float = 0.20  # 睑裂高度比偏离 > 此值认为异常
    RESTING_CHEEK_NLF_RATIO: float = 0.15  # NLF比偏离 > 此值认为轻度异常
    RESTING_CHEEK_NLF_SEVERE: float = 0.30  # NLF比偏离 > 此值认为重度异常
    RESTING_MOUTH_ANGLE_DIFF: float = 5.0  # 口角角度差 > 此值认为异常

    # Voluntary Movement 评分阈值
    VOLUNTARY_COMPLETE: float = 0.85  # 运动比 > 85% 认为完整
    VOLUNTARY_ALMOST: float = 0.70  # 运动比 > 70% 认为几乎完整
    VOLUNTARY_INITIATE: float = 0.40  # 运动比 > 40% 认为有启动
    VOLUNTARY_TRACE: float = 0.15  # 运动比 > 15% 认为轻微启动


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
    """
    获取眼睛同步性等级

    Args:
        pearson_corr: Pearson相关系数
        peak_diff: 峰值帧差异

    Returns:
        0: 同步良好
        1: 同步一般
        2: 同步较差
    """
    if pearson_corr >= THR.EYE_SYNC_PEARSON_GOOD and peak_diff <= THR.EYE_SYNC_PEAK_DIFF_MAX:
        return 0
    elif pearson_corr >= THR.EYE_SYNC_PEARSON_FAIR:
        return 1
    else:
        return 2


def get_eye_symmetry_text(level: int) -> str:
    """获取眼睛对称性文字描述"""
    texts = {
        0: "对称",
        1: "轻度不对称",
        2: "中度不对称",
        3: "重度不对称",
    }
    return texts.get(level, "未知")


def get_eye_sync_text(level: int) -> str:
    """获取眼睛同步性文字描述"""
    texts = {
        0: "同步良好",
        1: "同步一般",
        2: "同步较差",
    }
    return texts.get(level, "未知")


def is_symmetric(ratio: float, threshold: float = None) -> bool:
    """判断比值是否对称（接近1.0）"""
    if threshold is None:
        threshold = THR.SYMMETRY_NORMAL
    return abs(ratio - 1.0) <= threshold


def get_asymmetry_level(ratio: float) -> int:
    """
    获取不对称程度等级

    Returns:
        0: 对称
        1: 轻度不对称
        2: 中度不对称
        3: 重度不对称
    """
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
    """
    获取口角角度不对称程度等级

    Args:
        angle_diff: 口角角度差值（度）

    Returns:
        0: 对称
        1: 轻度不对称
        2: 中度不对称
        3: 重度不对称
    """
    if angle_diff < THR.ORAL_ANGLE_SYMMETRIC:
        return 0
    elif angle_diff < THR.ORAL_ANGLE_MILD:
        return 1
    elif angle_diff < THR.ORAL_ANGLE_MODERATE:
        return 2
    else:
        return 3


def get_synkinesis_score(change_ratio: float, region: str = "mouth") -> int:
    """
    获取联动运动评分

    Args:
        change_ratio: 变化比例
        region: "mouth" 或 "eye"

    Returns:
        0: 无联动
        1: 轻度联动
        2: 中度联动
        3: 重度联动
    """
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


if __name__ == "__main__":
    # 打印所有阈值用于检查
    print("=" * 60)
    print("阈值配置一览")
    print("=" * 60)

    for field_name in dir(THR):
        if not field_name.startswith("_"):
            value = getattr(THR, field_name)
            print(f"{field_name}: {value}")