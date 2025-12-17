"""
FacialPalsy - 常量定义模块
=========================

统一管理所有常量、索引和配置项，避免重复定义。

包括:
1. MediaPipe 478点关键点索引 (LM)
2. 面部区域索引 (用于运动分析)
3. 对称区域对定义
4. 阈值和配置常量

使用方法:
    from constants import LM, MotionRegions, Thresholds, Colors
"""

from typing import List, Tuple, Dict


# =============================================================================
# MediaPipe 478点关键点索引 (Landmark Indices)
# =============================================================================

class LM:
    """MediaPipe FaceLandmarker 478点索引"""

    # ========== 眼部 ==========
    # 内外眦点
    EYE_INNER_L = 362      # 左眼内眦
    EYE_INNER_R = 133      # 右眼内眦
    EYE_OUTER_L = 263      # 左眼外眦
    EYE_OUTER_R = 33       # 右眼外眦

    # 眼睑中点
    EYE_TOP_L = 386        # 左眼上眼睑中点
    EYE_BOT_L = 374        # 左眼下眼睑中点
    EYE_TOP_R = 159        # 右眼上眼睑中点
    EYE_BOT_R = 145        # 右眼下眼睑中点

    # 眼部轮廓 (16点)
    EYE_CONTOUR_L = [263, 466, 388, 387, 386, 385, 384, 398,
                     362, 382, 381, 380, 374, 373, 390, 249]
    EYE_CONTOUR_R = [33, 246, 161, 160, 159, 158, 157, 173,
                     133, 155, 154, 153, 145, 144, 163, 7]

    # EAR计算用点
    EAR_L = [263, 386, 387, 362, 373, 374]  # 左眼EAR点
    EAR_R = [33, 159, 158, 133, 144, 145]   # 右眼EAR点

    # ========== 眉毛 ==========
    BROW_L = [276, 283, 282, 295, 300]
    BROW_R = [46, 53, 52, 65, 70]
    BROW_CENTER_L = 282
    BROW_CENTER_R = 52

    # ========== 嘴部 ==========
    MOUTH_L = 291          # 左嘴角
    MOUTH_R = 61           # 右嘴角
    LIP_TOP = 13           # 上唇顶
    LIP_BOT = 14           # 下唇底
    LIP_TOP_CENTER = 0     # 上唇中心
    LIP_BOT_CENTER = 17    # 下唇中心

    # 唇峰 (口角角度计算用)
    LIP_PEAK_L = 37
    LIP_PEAK_R = 267

    # ========== 鼻部 ==========
    NOSE_TIP = 4
    NOSE_ALA_L = 129       # 左鼻翼
    NOSE_ALA_R = 358       # 右鼻翼

    # ========== 面颊 ==========
    CHEEK_L = [425, 426, 427, 411, 280]
    CHEEK_R = [205, 206, 207, 187, 50]

    # ========== 面部轮廓 ==========
    FACE_CONTOUR_L = [234, 93, 132, 58, 172]
    FACE_CONTOUR_R = [454, 323, 361, 288, 397]

    # ========== 其他 ==========
    FOREHEAD = 10
    CHIN = 152


# =============================================================================
# 面部区域索引 (用于运动分析)
# =============================================================================

class MotionRegions:
    """用于运动不对称性计算的面部区域定义"""

    # 左眼区域 (更完整的点集)
    LEFT_EYE = [
        33, 7, 163, 144, 145, 153, 154, 155, 133,
        173, 157, 158, 159, 160, 161, 246
    ]

    # 右眼区域
    RIGHT_EYE = [
        362, 382, 381, 380, 374, 373, 390, 249,
        263, 466, 388, 387, 386, 385, 384, 398
    ]

    # 左眉毛
    LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

    # 右眉毛
    RIGHT_BROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]

    # 左脸颊
    LEFT_CHEEK = [116, 123, 147, 213, 192, 214, 138]

    # 右脸颊
    RIGHT_CHEEK = [345, 352, 376, 433, 416, 434, 367]

    # 所有区域字典
    ALL_REGIONS: Dict[str, List[int]] = {
        'left_eye': LEFT_EYE,
        'right_eye': RIGHT_EYE,
        'left_brow': LEFT_BROW,
        'right_brow': RIGHT_BROW,
        'left_cheek': LEFT_CHEEK,
        'right_cheek': RIGHT_CHEEK,
    }

    # 对称区域对
    SYMMETRIC_PAIRS: List[Tuple[str, str]] = [
        ('left_eye', 'right_eye'),
        ('left_brow', 'right_brow'),
        ('left_cheek', 'right_cheek'),
    ]


# =============================================================================
# 阈值常量
# =============================================================================

class Thresholds:
    """各种阈值和配置常量"""

    # 完全闭眼阈值 (睁眼度 <= 6.25%)
    COMPLETE_CLOSURE = 0.0625

    # 眼睛睁大阈值 (睁眼度 > 110%)
    EYE_WIDE_OPEN = 1.1

    # 眼睛眯起阈值 (睁眼度 < 90%)
    EYE_SQUINT = 0.9

    # 数值精度阈值
    EPSILON = 1e-9
    EPSILON_SMALL = 1e-6

    # 不对称性阈值
    EYE_ASYMMETRY_THRESHOLD = 0.15
    ORAL_HEIGHT_DIFF_THRESHOLD = 0.02
    NLF_ASYMMETRY_THRESHOLD = 0.25


# =============================================================================
# 颜色定义 (用于可视化)
# =============================================================================

class Colors:
    """可视化颜色 (BGR格式)"""

    # 左右颜色
    LEFT_COLOR = (255, 0, 0)      # 蓝色
    RIGHT_COLOR = (0, 165, 255)   # 橙色

    # 状态颜色
    NORMAL_COLOR = (0, 255, 0)    # 绿色
    WARNING_COLOR = (0, 255, 255) # 黄色
    ERROR_COLOR = (0, 0, 255)     # 红色
    NEUTRAL_COLOR = (255, 255, 255)  # 白色
    GRAY_COLOR = (128, 128, 128)  # 灰色


# =============================================================================
# 动作名称映射
# =============================================================================

class ActionNames:
    """11个标准动作的名称和中文名称"""

    NEUTRAL_FACE = "NeutralFace"
    RAISE_EYEBROW = "RaiseEyebrow"
    CLOSE_EYE_SOFTLY = "CloseEyeSoftly"
    CLOSE_EYE_HARDLY = "CloseEyeHardly"
    SMILE = "Smile"
    SHRUG_NOSE = "ShrugNose"
    LIP_PUCKER = "LipPucker"
    SHOW_TEETH = "ShowTeeth"
    BLOW_CHEEK = "BlowCheek"
    VOLUNTARY_EYE_BLINK = "VoluntaryEyeBlink"
    SPONTANEOUS_EYE_BLINK = "SpontaneousEyeBlink"

    # 中文名称映射
    CN_NAMES = {
        NEUTRAL_FACE: "静息",
        RAISE_EYEBROW: "抬眉",
        CLOSE_EYE_SOFTLY: "轻闭眼",
        CLOSE_EYE_HARDLY: "用力闭眼",
        SMILE: "微笑",
        SHRUG_NOSE: "皱鼻",
        LIP_PUCKER: "撅嘴",
        SHOW_TEETH: "露齿",
        BLOW_CHEEK: "鼓腮",
        VOLUNTARY_EYE_BLINK: "自主眨眼",
        SPONTANEOUS_EYE_BLINK: "自发眨眼",
    }

    # 所有动作列表
    ALL_ACTIONS = [
        NEUTRAL_FACE,
        RAISE_EYEBROW,
        CLOSE_EYE_SOFTLY,
        CLOSE_EYE_HARDLY,
        SMILE,
        SHRUG_NOSE,
        LIP_PUCKER,
        SHOW_TEETH,
        BLOW_CHEEK,
        VOLUNTARY_EYE_BLINK,
        SPONTANEOUS_EYE_BLINK,
    ]


# =============================================================================
# 运动特征名称
# =============================================================================

MOTION_FEATURE_NAMES = [
    'mean_displacement',      # 0: 平均位移
    'max_displacement',       # 1: 最大位移
    'std_displacement',       # 2: 位移标准差
    'motion_energy',          # 3: 运动能量
    'motion_asymmetry',       # 4: 运动不对称性
    'temporal_smoothness',    # 5: 时间平滑度
    'spatial_concentration',  # 6: 空间集中度
    'peak_ratio',             # 7: 峰值区域比例
    'motion_center_x',        # 8: 运动重心X
    'motion_center_y',        # 9: 运动重心Y
    'velocity_mean',          # 10: 平均速度
    'acceleration_std',       # 11: 加速度变化
]


# =============================================================================
# 辅助函数
# =============================================================================

def get_cn_action_name(action_name: str) -> str:
    """获取动作的中文名称"""
    return ActionNames.CN_NAMES.get(action_name, action_name)


def get_motion_feature_names() -> List[str]:
    """返回12维运动特征的名称列表"""
    return MOTION_FEATURE_NAMES.copy()


# =============================================================================
# 模块导出
# =============================================================================

__all__ = [
    'LM',
    'MotionRegions',
    'Thresholds',
    'Colors',
    'ActionNames',
    'MOTION_FEATURE_NAMES',
    'get_cn_action_name',
    'get_motion_feature_names',
]