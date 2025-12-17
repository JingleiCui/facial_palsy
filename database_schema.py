"""
H-GFA Net 数据库 Schema
===============================

主要更新:
1. 将所有特征统一存储到 video_features 表

2. 数据集划分表 (患者级划分)
3. 数据增强配置表

设计原则:
- 所有特征存储在同一行，避免训练时JOIN查询
- BLOB存储numpy数组，附带维度信息
- 支持增量处理 (通过processed_at和NULL判断)
"""

import sqlite3
import os
from typing import Optional

# =============================================================================
# Schema SQL 定义
# =============================================================================

SCHEMA_SQL = """
-- ============================================================
-- H-GFA Net 数据库模式
-- 核心更新: 所有特征统一存储，支持多模态融合
-- ============================================================

PRAGMA foreign_keys = ON;

-- ============================================================
-- 1. 核心实体表 (保持不变)
-- ============================================================

-- 1.1 患者表 (隐私最小化设计)
CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_patients_created ON patients(created_at);


-- 1.2 检查表 (Examination = Session)
CREATE TABLE IF NOT EXISTS examinations (
    examination_id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    capture_datetime TIMESTAMP NOT NULL,
    json_file_path TEXT,
    video_root_dir TEXT,
    has_videos INTEGER DEFAULT 0,
    has_labels INTEGER DEFAULT 0,
    is_valid INTEGER DEFAULT 1,
    import_version INTEGER DEFAULT 1,
    import_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE INDEX IF NOT EXISTS idx_exam_patient ON examinations(patient_id);
CREATE INDEX IF NOT EXISTS idx_exam_datetime ON examinations(capture_datetime);
CREATE INDEX IF NOT EXISTS idx_exam_has_labels ON examinations(has_labels);


-- 1.3 动作类型映射表 (11个核心动作)
CREATE TABLE IF NOT EXISTS action_types (
    action_id INTEGER PRIMARY KEY AUTOINCREMENT,
    action_name_en TEXT UNIQUE NOT NULL,
    action_name_cn TEXT NOT NULL,
    display_order INTEGER,
    description TEXT
);

INSERT OR IGNORE INTO action_types (action_name_en, action_name_cn, display_order) VALUES
('NeutralFace', '静息', 1),
('SpontaneousEyeBlink', '自然眨眼', 2),
('VoluntaryEyeBlink', '自主眨眼', 3),
('CloseEyeSoftly', '轻轻闭眼', 4),
('CloseEyeHardly', '用力闭眼', 5),
('RaiseEyebrow', '抬眉', 6),
('Smile', '微笑', 7),
('ShrugNose', '耸鼻', 8),
('ShowTeeth', '呲牙', 9),
('BlowCheek', '鼓腮', 10),
('LipPucker', '撅嘴', 11);


-- ============================================================
-- 2. 文件管理表
-- ============================================================

CREATE TABLE IF NOT EXISTS video_files (
    video_id INTEGER PRIMARY KEY AUTOINCREMENT,
    examination_id TEXT NOT NULL,
    action_id INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    relative_path TEXT,
    video_file_index INTEGER,
    start_time_str TEXT,
    end_time_str TEXT,
    start_frame INTEGER,
    end_frame INTEGER,
    duration_frames INTEGER,
    fps REAL DEFAULT 30.0,
    file_exists INTEGER DEFAULT 0,
    file_size_bytes INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (examination_id) REFERENCES examinations(examination_id),
    FOREIGN KEY (action_id) REFERENCES action_types(action_id)
);

CREATE INDEX IF NOT EXISTS idx_video_exam ON video_files(examination_id);
CREATE INDEX IF NOT EXISTS idx_video_action ON video_files(action_id);
CREATE INDEX IF NOT EXISTS idx_video_exists ON video_files(file_exists);


-- ============================================================
-- 3. 标签表
-- ============================================================

-- 3.1 检查级标签 (全局诊断)
CREATE TABLE IF NOT EXISTS examination_labels (
    examination_id TEXT PRIMARY KEY,
    has_palsy INTEGER,                        -- 0=健康, 1=面瘫
    palsy_side INTEGER,                       -- 0=无, 1=患者左侧, 2=患者右侧
    hb_grade INTEGER,                         -- House-Brackmann: 1-6
    sunnybrook_score REAL,                    -- Sunnybrook: 0-100
    label_source TEXT DEFAULT 'doctor',
    labeler_id TEXT,
    label_timestamp TIMESTAMP,
    label_confidence REAL,
    excel_row_number INTEGER,
    excel_session_id TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (examination_id) REFERENCES examinations(examination_id)
);

CREATE INDEX IF NOT EXISTS idx_exam_labels_has_palsy ON examination_labels(has_palsy);
CREATE INDEX IF NOT EXISTS idx_exam_labels_hb ON examination_labels(hb_grade);


-- 3.2 动作级标签
CREATE TABLE IF NOT EXISTS action_labels (
    label_id INTEGER PRIMARY KEY AUTOINCREMENT,
    examination_id TEXT NOT NULL,
    action_id INTEGER NOT NULL,
    severity_score INTEGER NOT NULL,          -- 1-5: 严重程度评分
    label_source TEXT DEFAULT 'doctor',
    excel_column_name TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(examination_id, action_id),
    FOREIGN KEY (examination_id) REFERENCES examinations(examination_id),
    FOREIGN KEY (action_id) REFERENCES action_types(action_id)
);

CREATE INDEX IF NOT EXISTS idx_action_labels_exam ON action_labels(examination_id);
CREATE INDEX IF NOT EXISTS idx_action_labels_severity ON action_labels(severity_score);


-- ============================================================
-- 4. 核心特征表
-- ============================================================
-- 设计原则: 所有特征存储在同一行，训练时无需JOIN

CREATE TABLE IF NOT EXISTS video_features (
    feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL,
    augmentation_type TEXT DEFAULT 'none',      -- 增强类型：none、mirror、rotate_-5等
    aug_palsy_side INTEGER,                    -- 增强后的患侧标签
    
    -- ========== 基础信息 ==========
    peak_frame_idx INTEGER,                   -- 峰值帧索引
    peak_frame_path TEXT,                     -- 峰值帧图像路径
    peak_frame_segmented_path TEXT,           -- 分割后的峰值帧路径
    unit_length REAL,                         -- 归一化单位 (两眼内眦距离)
    
    -- ========== 几何特征 (来自3D Landmarks) ==========
    static_features BLOB,                     -- 静态几何特征
    static_dim INTEGER,                       -- 静态特征维度 (5-11)
    dynamic_features BLOB,                    -- 动态几何特征
    dynamic_dim INTEGER,                      -- 动态特征维度 (0-8)
    
    -- ========== 视觉特征 (来自MobileNetV3) ==========
    visual_features BLOB,                     -- 1280维全局视觉特征
    visual_dim INTEGER DEFAULT 1280,
    
    -- ========== 皱纹特征 ==========
    wrinkle_features BLOB,                    -- 皱纹量化特征
    wrinkle_dim INTEGER DEFAULT 10,           -- 默认10维
    wrinkle_heatmap BLOB,                     -- 皱纹热力图 (压缩后，可选)
    wrinkle_mask BLOB,                        -- 皱纹掩码 (压缩后，可选)
    
    -- ========== 运动热力图特征 ==========
    motion_features BLOB,                     -- 运动统计特征
    motion_dim INTEGER DEFAULT 12,            -- 默认12维
    motion_heatmap BLOB,                      -- 运动热力图 (压缩后，可选)
    motion_landmark_displacements BLOB,       -- 468个关键点位移 (可选详细数据)
    
    -- ========== 融合特征 (中间结果缓存) ==========
    geo_refined_features BLOB,                -- Stage1输出: 几何融合特征 (256维)
    visual_guided_features BLOB,              -- Stage2输出: 视觉引导特征 (256维)
    fused_action_features BLOB,               -- Stage3输出: 多模态融合特征 (512维)
    
    -- ========== 处理状态 ==========
    geometry_processed_at TIMESTAMP,          -- 几何特征提取时间
    visual_processed_at TIMESTAMP,            -- 视觉特征提取时间
    wrinkle_processed_at TIMESTAMP,           -- 皱纹特征提取时间
    motion_processed_at TIMESTAMP,            -- 运动特征提取时间
    fusion_processed_at TIMESTAMP,            -- 融合特征计算时间
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (video_id) REFERENCES video_files(video_id),
    UNIQUE(video_id, augmentation_type)
);

CREATE INDEX IF NOT EXISTS idx_vf_video ON video_features(video_id);
CREATE INDEX IF NOT EXISTS idx_vf_geometry ON video_features(geometry_processed_at);
CREATE INDEX IF NOT EXISTS idx_vf_visual ON video_features(visual_processed_at);
CREATE INDEX IF NOT EXISTS idx_vf_wrinkle ON video_features(wrinkle_processed_at);
CREATE INDEX IF NOT EXISTS idx_vf_motion ON video_features(motion_processed_at);


-- ============================================================
-- 5. 数据集划分表 (患者级划分，避免数据泄露)
-- ============================================================

-- 5.1 划分配置
CREATE TABLE IF NOT EXISTS dataset_splits (
    split_id INTEGER PRIMARY KEY AUTOINCREMENT,
    split_name TEXT UNIQUE NOT NULL,          -- 'train_v1', 'val_v1', 'test_v1'
    split_type TEXT NOT NULL,                 -- 'train', 'val', 'test'
    split_version TEXT DEFAULT 'v1.0',
    split_method TEXT,                        -- 'patient_stratified', 'random'
    split_ratio REAL,
    random_seed INTEGER DEFAULT 42,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- 5.2 划分成员 (患者级)
CREATE TABLE IF NOT EXISTS dataset_members (
    member_id INTEGER PRIMARY KEY AUTOINCREMENT,
    split_id INTEGER NOT NULL,
    patient_id TEXT NOT NULL,                 -- 患者级划分！
    fold_number INTEGER,                      -- K-Fold交叉验证用
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(split_id, patient_id),
    FOREIGN KEY (split_id) REFERENCES dataset_splits(split_id),
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE INDEX IF NOT EXISTS idx_dm_split ON dataset_members(split_id);
CREATE INDEX IF NOT EXISTS idx_dm_patient ON dataset_members(patient_id);


-- ============================================================
-- 6. 模型与预测表
-- ============================================================

-- 6.1 训练记录
CREATE TABLE IF NOT EXISTS training_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_name TEXT NOT NULL,
    model_name TEXT NOT NULL,                 -- 'HGFANet'
    model_version TEXT,
    config_json TEXT,                         -- 完整配置 JSON
    split_id INTEGER,                         -- 使用的数据划分
    
    -- 训练结果
    best_epoch INTEGER,
    best_val_loss REAL,
    best_val_acc REAL,
    final_test_acc REAL,
    
    -- 多任务损失权重
    task_weights_json TEXT,                   -- {"severity": 0.4, "hb": 0.3, ...}
    
    -- 路径
    checkpoint_path TEXT,
    log_path TEXT,
    
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    status TEXT DEFAULT 'running'             -- 'running', 'completed', 'failed'
);


-- 6.2 检查级预测
CREATE TABLE IF NOT EXISTS examination_predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    examination_id TEXT NOT NULL,
    run_id INTEGER,                           -- 关联训练记录
    
    pred_has_palsy INTEGER,
    pred_has_palsy_prob REAL,
    pred_palsy_side INTEGER,
    pred_palsy_side_prob REAL,
    pred_hb_grade INTEGER,
    pred_hb_probs TEXT,                       -- JSON: [p1, p2, p3, p4, p5, p6]
    pred_sunnybrook_score REAL,
    
    inference_time_ms REAL,
    device_type TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (examination_id) REFERENCES examinations(examination_id),
    FOREIGN KEY (run_id) REFERENCES training_runs(run_id)
);


-- 6.3 动作级预测
CREATE TABLE IF NOT EXISTS action_predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL,
    run_id INTEGER,
    
    pred_severity INTEGER,                    -- 1-5
    pred_severity_probs TEXT,                 -- JSON: [p1, p2, p3, p4, p5]
    
    -- 可解释性
    attention_weights TEXT,                   -- JSON: 各模态权重
    important_features TEXT,                  -- JSON: 重要特征排名
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (video_id) REFERENCES video_files(video_id),
    FOREIGN KEY (run_id) REFERENCES training_runs(run_id)
);


-- ============================================================
-- 7. 便捷视图
-- ============================================================

-- 7.1 完整训练数据视图 (单表查询，高效！)
CREATE VIEW IF NOT EXISTS v_training_samples AS
SELECT 
    vf.video_id,
    vfeat.augmentation_type, 
    vf.examination_id,
    vf.action_id,
    at.action_name_en,
    
    -- 患者信息 (用于划分)
    e.patient_id,
    
    -- 所有特征 (直接从video_features取)
    vfeat.static_features,
    vfeat.static_dim,
    vfeat.dynamic_features,
    vfeat.dynamic_dim,
    vfeat.visual_features,
    vfeat.wrinkle_features,
    vfeat.wrinkle_dim,
    vfeat.motion_features,
    vfeat.motion_dim,
    
    -- 融合特征 (如已计算)
    vfeat.geo_refined_features,
    vfeat.visual_guided_features,
    vfeat.fused_action_features,
    
    -- 动作级标签
    al.severity_score,
    
    -- 检查级标签
    el.has_palsy,
    el.palsy_side AS orig_palsy_side,
    COALESCE(vfeat.aug_palsy_side, el.palsy_side) AS palsy_side,  -- 优先使用增强标签
    el.hb_grade,
    el.sunnybrook_score,
    
    -- 特征完整性标志
    CASE WHEN vfeat.static_features IS NOT NULL THEN 1 ELSE 0 END AS has_geometry,
    CASE WHEN vfeat.visual_features IS NOT NULL THEN 1 ELSE 0 END AS has_visual,
    CASE WHEN vfeat.wrinkle_features IS NOT NULL THEN 1 ELSE 0 END AS has_wrinkle,
    CASE WHEN vfeat.motion_features IS NOT NULL THEN 1 ELSE 0 END AS has_motion
    
FROM video_files vf
INNER JOIN action_types at ON vf.action_id = at.action_id
INNER JOIN examinations e ON vf.examination_id = e.examination_id
LEFT JOIN video_features vfeat ON vf.video_id = vfeat.video_id
LEFT JOIN action_labels al ON vf.examination_id = al.examination_id 
    AND vf.action_id = al.action_id
LEFT JOIN examination_labels el ON vf.examination_id = el.examination_id;


-- 7.2 数据集统计视图
CREATE VIEW IF NOT EXISTS v_dataset_stats AS
SELECT
    ds.split_name,
    ds.split_type,
    COUNT(DISTINCT dm.patient_id) AS patient_count,
    COUNT(DISTINCT e.examination_id) AS exam_count,
    COUNT(DISTINCT vf.video_id) AS video_count,
    
    -- HB分级统计
    SUM(CASE WHEN el.hb_grade = 1 THEN 1 ELSE 0 END) AS hb1_count,
    SUM(CASE WHEN el.hb_grade = 2 THEN 1 ELSE 0 END) AS hb2_count,
    SUM(CASE WHEN el.hb_grade = 3 THEN 1 ELSE 0 END) AS hb3_count,
    SUM(CASE WHEN el.hb_grade = 4 THEN 1 ELSE 0 END) AS hb4_count,
    SUM(CASE WHEN el.hb_grade = 5 THEN 1 ELSE 0 END) AS hb5_count,
    SUM(CASE WHEN el.hb_grade = 6 THEN 1 ELSE 0 END) AS hb6_count,
    
    -- 面瘫/健康统计
    SUM(CASE WHEN el.has_palsy = 1 THEN 1 ELSE 0 END) AS palsy_count,
    SUM(CASE WHEN el.has_palsy = 0 THEN 1 ELSE 0 END) AS healthy_count

FROM dataset_splits ds
INNER JOIN dataset_members dm ON ds.split_id = dm.split_id
INNER JOIN examinations e ON dm.patient_id = e.patient_id
INNER JOIN video_files vf ON e.examination_id = vf.examination_id
LEFT JOIN examination_labels el ON e.examination_id = el.examination_id
GROUP BY ds.split_id;


-- 7.3 特征提取进度视图
CREATE VIEW IF NOT EXISTS v_feature_progress AS
SELECT
    at.action_name_en,
    COUNT(*) AS total_videos,
    SUM(CASE WHEN vfeat.static_features IS NOT NULL THEN 1 ELSE 0 END) AS geometry_done,
    SUM(CASE WHEN vfeat.visual_features IS NOT NULL THEN 1 ELSE 0 END) AS visual_done,
    SUM(CASE WHEN vfeat.wrinkle_features IS NOT NULL THEN 1 ELSE 0 END) AS wrinkle_done,
    SUM(CASE WHEN vfeat.motion_features IS NOT NULL THEN 1 ELSE 0 END) AS motion_done,
    SUM(CASE WHEN vfeat.fused_action_features IS NOT NULL THEN 1 ELSE 0 END) AS fusion_done
FROM video_files vf
INNER JOIN action_types at ON vf.action_id = at.action_id
LEFT JOIN video_features vfeat ON vf.video_id = vfeat.video_id
WHERE vf.file_exists = 1
GROUP BY at.action_id
ORDER BY at.display_order;


-- 7.4 简化视频视图 (兼容旧代码)
CREATE VIEW IF NOT EXISTS videos AS
SELECT
    vf.video_id,
    vf.examination_id,
    vf.action_id,
    vf.file_path,
    vf.start_frame,
    vf.end_frame,
    vf.fps,
    vf.file_exists,
    CASE WHEN vfeat.feature_id IS NOT NULL THEN 1 ELSE 0 END AS is_processed,
    vfeat.geometry_processed_at AS processed_at,
    at.action_name_en,
    at.action_name_cn
FROM video_files vf
INNER JOIN action_types at ON vf.action_id = at.action_id
LEFT JOIN video_features vfeat ON vf.video_id = vfeat.video_id;


-- ============================================================
-- 8. 导入日志
-- ============================================================

CREATE TABLE IF NOT EXISTS import_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    import_type TEXT NOT NULL,
    import_source TEXT,
    records_processed INTEGER,
    records_succeeded INTEGER,
    records_failed INTEGER,
    error_messages TEXT,
    import_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_import_type ON import_logs(import_type);
"""


# =============================================================================
# 数据库操作函数
# =============================================================================

def create_database(db_path: str, force_recreate: bool = False):
    """
    创建或升级数据库

    Args:
        db_path: 数据库文件路径
        force_recreate: 是否强制重建 (危险！会删除所有数据)
    """
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    if force_recreate and os.path.exists(db_path):
        os.remove(db_path)
        print(f"[WARN] 已删除旧数据库: {db_path}")

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        print(f"[OK] 数据库初始化完成: {db_path}")
    finally:
        conn.close()


def get_feature_dimensions():
    """
    返回各类特征的标准维度配置
    """
    return {
        # 几何特征 (因动作而异)
        "static_dims": {
            'BlowCheek': 5,
            'CloseEyeHardly': 10,
            'CloseEyeSoftly': 7,
            'LipPucker': 5,
            'NeutralFace': 8,
            'RaiseEyebrow': 10,
            'ShowTeeth': 7,
            'ShrugNose': 5,
            'Smile': 11,
            'SpontaneousEyeBlink': 5,
            'VoluntaryEyeBlink': 5,
        },
        "dynamic_dims": {
            'BlowCheek': 2,
            'CloseEyeHardly': 8,
            'CloseEyeSoftly': 4,
            'LipPucker': 2,
            'NeutralFace': 0,
            'RaiseEyebrow': 3,
            'ShowTeeth': 3,
            'ShrugNose': 2,
            'Smile': 4,
            'SpontaneousEyeBlink': 7,
            'VoluntaryEyeBlink': 4,
        },

        # 固定维度特征
        "visual_dim": 1280,       # MobileNetV3
        "wrinkle_dim": 10,        # 皱纹统计特征
        "motion_dim": 12,         # 运动统计特征

        # 融合特征维度
        "geo_refined_dim": 256,
        "visual_guided_dim": 256,
        "fused_dim": 512,
    }


# =============================================================================
# 主函数
# =============================================================================

if __name__ == '__main__':
        create_database('facialPalsy.db')

        # 打印维度配置
        print("\n特征维度配置:")
        dims = get_feature_dimensions()
        print(f"  视觉特征: {dims['visual_dim']}维")
        print(f"  皱纹特征: {dims['wrinkle_dim']}维")
        print(f"  运动特征: {dims['motion_dim']}维")
        print(f"  融合特征: {dims['fused_dim']}维")