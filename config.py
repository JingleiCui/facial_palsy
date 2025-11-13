"""
配置文件 - 集中管理所有配置参数
"""
import os

# ============================================================
# 路径配置 - 请根据实际情况修改
# ============================================================

# 数据库路径
DB_PATH = 'facialPalsy.db'

# 视频根目录
VIDEOS_BASE_PATH = '/Users/cuijinglei/Documents/facialPalsy/videos'

# MediaPipe模型路径
MEDIAPIPE_MODEL_PATH = '/Users/cuijinglei/PycharmProjects/medicalProject/models/face_landmarker.task'

# 输出根目录
OUTPUT_BASE_DIR = '/Users/cuijinglei/Documents/facialPalsy/processed'

# 子目录
PEAK_FRAMES_DIR = os.path.join(OUTPUT_BASE_DIR, 'peak_frames')
LANDMARKS_DIR = os.path.join(OUTPUT_BASE_DIR, 'landmarks')
FEATURES_DIR = os.path.join(OUTPUT_BASE_DIR, 'features')
VISUALIZATIONS_DIR = os.path.join(OUTPUT_BASE_DIR, 'visualizations')

# 创建输出目录
for dir_path in [OUTPUT_BASE_DIR, PEAK_FRAMES_DIR, LANDMARKS_DIR, FEATURES_DIR, VISUALIZATIONS_DIR]:
    os.makedirs(dir_path, exist_ok=True)


# ============================================================
# MediaPipe关键点索引定义
# ============================================================

# 眼睛关键点
EYE_INNER_CANTHUS_LEFT = 362
EYE_INNER_CANTHUS_RIGHT = 133
EYE_OUTER_CANTHUS_LEFT = 263
EYE_OUTER_CANTHUS_RIGHT = 33

# 眼睛高度（上下眼睑）
EYE_TOP_LEFT = 159
EYE_BOTTOM_LEFT = 145
EYE_TOP_RIGHT = 386
EYE_BOTTOM_RIGHT = 374

# 瞳孔中心
PUPIL_LEFT = 473
PUPIL_RIGHT = 468

# 眉毛关键点
EYEBROW_LEFT_POINTS = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
EYEBROW_RIGHT_POINTS = [107, 66, 105, 63, 70, 46, 53, 52, 65, 55]

# 嘴角
MOUTH_CORNER_LEFT = 291
MOUTH_CORNER_RIGHT = 61

# 嘴唇中点
UPPER_LIP_CENTER = 0
LOWER_LIP_CENTER = 17

# 嘴唇外轮廓
MOUTH_OUTER_CONTOUR = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]

# 鼻子关键点
NOSE_TIP = 4


# ============================================================
# 11个动作配置
# ============================================================

ACTION_CONFIG = {
    'NeutralFace': {
        'action_id': 1,
        'name_cn': '静息',
        'class_name': 'NeutralFaceAction',
    },
    'SpontaneousEyeBlink': {
        'action_id': 2,
        'name_cn': '自然眨眼',
        'class_name': 'SpontaneousEyeBlinkAction',
    },
    'VoluntaryEyeBlink': {
        'action_id': 3,
        'name_cn': '自主眨眼',
        'class_name': 'VoluntaryEyeBlinkAction',
    },
    'CloseEyeSoftly': {
        'action_id': 4,
        'name_cn': '轻轻闭眼',
        'class_name': 'CloseEyeSoftlyAction',
    },
    'CloseEyeHardly': {
        'action_id': 5,
        'name_cn': '用力闭眼',
        'class_name': 'CloseEyeHardlyAction',
    },
    'RaiseEyebrow': {
        'action_id': 6,
        'name_cn': '抬眉',
        'class_name': 'RaiseEyebrowAction',
    },
    'Smile': {
        'action_id': 7,
        'name_cn': '微笑',
        'class_name': 'SmileAction',
    },
    'ShrugNose': {
        'action_id': 8,
        'name_cn': '耸鼻',
        'class_name': 'ShrugNoseAction',
    },
    'ShowTeeth': {
        'action_id': 9,
        'name_cn': '呲牙',
        'class_name': 'ShowTeethAction',
    },
    'BlowCheek': {
        'action_id': 10,
        'name_cn': '鼓腮',
        'class_name': 'BlowCheekAction',
    },
    'LipPucker': {
        'action_id': 11,
        'name_cn': '撅嘴',
        'class_name': 'LipPuckerAction',
    }
}


# ============================================================
# 特征维度配置
# ============================================================

# 静态几何特征维度
STATIC_FEATURE_DIM = 32

# 动态几何特征维度
DYNAMIC_FEATURE_DIM = 16


# ============================================================
# 处理参数
# ============================================================

# 是否保存可视化
SAVE_VISUALIZATION = True

# 是否保存关键点序列
SAVE_LANDMARKS = True

# 是否保存特征到文件
SAVE_FEATURES_TO_FILE = True

# 批量处理参数
BATCH_SIZE = 10  # 每处理多少个视频提交一次数据库

# 最小有效帧比例
MIN_VALID_FRAME_RATIO = 0.5  # 至少50%的帧要有效