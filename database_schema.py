"""
面瘫评估数据库 Schema 定义
用于存储患者信息、视频元数据、医生标注和训练样本
"""

import sqlite3
import os

SCHEMA_SQL = """
-- ============================================================
-- H-GFA Net 数据库模式 V1.1
-- 修改：简化时间字段、直接使用标准化动作名
-- ============================================================

PRAGMA foreign_keys = ON;

-- ============================================================
-- 1. 核心实体表
-- ============================================================

-- 1.1 患者表 (隐私最小化设计)
CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,              -- 患者ID，如 'XW000157'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT                                -- 可选备注（不含隐私）
);

CREATE INDEX IF NOT EXISTS idx_patients_created ON patients(created_at);


-- 1.2 检查表 (Examination = Session，一次完整的检测)
CREATE TABLE IF NOT EXISTS examinations (
    examination_id TEXT PRIMARY KEY,          -- 检查ID，如 'XW000157_20231016_09-19-14'
    patient_id TEXT NOT NULL,                 -- 外键：患者ID
    
    -- 时间信息 (简化：只保留完整时间戳)
    capture_datetime TIMESTAMP NOT NULL,      -- 检查时间 'YYYY-MM-DD HH:MM:SS'
    
    -- 元数据
    json_file_path TEXT,                      -- JSON元数据文件路径
    video_root_dir TEXT,                      -- 视频根目录（实际文件系统路径）
    
    -- 状态标识
    has_videos INTEGER DEFAULT 0,             -- 是否有视频文件
    has_labels INTEGER DEFAULT 0,             -- 是否有医生标注
    is_valid INTEGER DEFAULT 1,               -- 是否有效（用于标记异常数据）
    
    -- 版本管理
    import_version INTEGER DEFAULT 1,         -- 导入版本号
    import_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (patient_id) REFERENCES patients(patient_id)
);

CREATE INDEX IF NOT EXISTS idx_exam_patient ON examinations(patient_id);
CREATE INDEX IF NOT EXISTS idx_exam_datetime ON examinations(capture_datetime);
CREATE INDEX IF NOT EXISTS idx_exam_has_labels ON examinations(has_labels);
CREATE INDEX IF NOT EXISTS idx_exam_valid ON examinations(is_valid);


-- 1.3 动作类型映射表 (11个核心动作，已标准化)
CREATE TABLE IF NOT EXISTS action_types (
    action_id INTEGER PRIMARY KEY AUTOINCREMENT,
    action_name_en TEXT UNIQUE NOT NULL,      -- 英文标准名称（JSON中已标准化）
    action_name_cn TEXT NOT NULL,             -- 中文名称（对应Excel列名）
    display_order INTEGER,                    -- 显示顺序
    description TEXT                          -- 动作描述
);

-- 插入11个核心动作（名称已标准化，与JSON一致）
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

-- 2.1 视频文件表
CREATE TABLE IF NOT EXISTS video_files (
    video_id INTEGER PRIMARY KEY AUTOINCREMENT,
    examination_id TEXT NOT NULL,
    action_id INTEGER NOT NULL,
    
    -- 文件路径
    file_path TEXT NOT NULL,            -- 完整绝对路径
    relative_path TEXT,                       -- 相对路径（相对于video_root_dir）
    
    -- 视频属性
    video_file_index INTEGER,                 -- JSON中的VideoFileIndex
    start_time_str TEXT,                      -- JSON中的时间格式 '00:00:07:000'
    end_time_str TEXT,
    start_frame INTEGER,                      -- 起始帧号
    end_frame INTEGER,                        -- 结束帧号
    duration_frames INTEGER,                  -- 总帧数
    fps REAL DEFAULT 30.0,                    -- 帧率
    
    -- 状态
    file_exists INTEGER DEFAULT 0,            -- 文件是否存在
    file_size_bytes INTEGER,                  -- 文件大小
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (examination_id) REFERENCES examinations(examination_id),
    FOREIGN KEY (action_id) REFERENCES action_types(action_id)
);

CREATE INDEX IF NOT EXISTS idx_video_exam ON video_files(examination_id);
CREATE INDEX IF NOT EXISTS idx_video_action ON video_files(action_id);
CREATE INDEX IF NOT EXISTS idx_video_exists ON video_files(file_exists);


-- 2.2 帧文件表 (存储峰值帧等关键帧)
CREATE TABLE IF NOT EXISTS frame_files (
    frame_id INTEGER PRIMARY KEY AUTOINCREMENT,
    examination_id TEXT NOT NULL,
    action_id INTEGER NOT NULL,
    video_id INTEGER,                         -- 可选：关联到视频
    
    -- 帧信息
    frame_type TEXT NOT NULL,                 -- 'peak', 'neutral', 'start', 'end'
    frame_number INTEGER,                     -- 帧号
    
    -- 文件路径
    frame_file_path TEXT NOT NULL,            -- 图像文件路径
    
    -- 状态
    file_exists INTEGER DEFAULT 0,
    file_size_bytes INTEGER,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (examination_id) REFERENCES examinations(examination_id),
    FOREIGN KEY (action_id) REFERENCES action_types(action_id),
    FOREIGN KEY (video_id) REFERENCES video_files(video_id)
);

CREATE INDEX IF NOT EXISTS idx_frame_exam_action ON frame_files(examination_id, action_id);
CREATE INDEX IF NOT EXISTS idx_frame_type ON frame_files(frame_type);


-- 2.3 特征文件表 (存储提取的特征)
CREATE TABLE IF NOT EXISTS feature_files (
    feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
    examination_id TEXT NOT NULL,
    action_id INTEGER NOT NULL,
    
    -- 特征类型
    feature_type TEXT NOT NULL,               -- 'static', 'dynamic', 'visual', 'combined'
    feature_version TEXT DEFAULT 'v1.0',      -- 特征提取版本
    
    -- 文件路径
    feature_file_path TEXT NOT NULL,          -- .npy 或 .pkl 文件路径
    
    -- 特征维度
    feature_dim INTEGER,                      -- 特征维度
    
    -- 状态
    file_exists INTEGER DEFAULT 0,
    file_size_bytes INTEGER,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (examination_id) REFERENCES examinations(examination_id),
    FOREIGN KEY (action_id) REFERENCES action_types(action_id)
);

CREATE INDEX IF NOT EXISTS idx_feature_exam_action ON feature_files(examination_id, action_id);
CREATE INDEX IF NOT EXISTS idx_feature_type ON feature_files(feature_type);


-- ============================================================
-- 3. 标签表 (来自 labels.xlsx)
-- ============================================================

-- 3.1 检查级标签 (Examination-Level Labels)
CREATE TABLE IF NOT EXISTS examination_labels (
    examination_id TEXT PRIMARY KEY,
    
    -- ========== 全局诊断标签 ==========
    has_palsy INTEGER,                        -- 0=健康, 1=面瘫
    palsy_side INTEGER,                       -- 0=无, 1=患者左侧(屏右), 2=患者右侧(屏左)
    hb_grade INTEGER,                         -- 1-6
    sunnybrook_score REAL,                    -- 0-100
    
    -- ========== 标签来源与质量 ==========
    label_source TEXT DEFAULT 'doctor',       -- 'doctor', 'auto', 'assumed'
    labeler_id TEXT,                          -- 标注医生ID（可选）
    label_timestamp TIMESTAMP,                -- 标注时间
    label_confidence REAL,                    -- 标注置信度 (可选)
    
    -- ========== Excel原始信息 ==========
    excel_row_number INTEGER,                 -- Excel中的行号
    excel_session_id TEXT,                    -- Excel中的原始编号
    
    notes TEXT,                               -- 备注
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (examination_id) REFERENCES examinations(examination_id)
);

CREATE INDEX IF NOT EXISTS idx_exam_labels_has_palsy ON examination_labels(has_palsy);
CREATE INDEX IF NOT EXISTS idx_exam_labels_hb ON examination_labels(hb_grade);


-- 3.2 动作级标签 (Action-Level Labels)
CREATE TABLE IF NOT EXISTS action_labels (
    label_id INTEGER PRIMARY KEY AUTOINCREMENT,
    examination_id TEXT NOT NULL,
    action_id INTEGER NOT NULL,
    
    -- ========== 动作评分 ==========
    severity_score INTEGER NOT NULL,          -- 1-5: 严重程度评分
    
    -- ========== 标签来源 ==========
    label_source TEXT DEFAULT 'doctor',       -- 'doctor', 'auto', 'assumed'
    
    -- ========== Excel信息 ==========
    excel_column_name TEXT,                   -- Excel中的列名（如'静息'）
    
    notes TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- 唯一约束：每个检查的每个动作只有一个标签
    UNIQUE(examination_id, action_id),
    
    FOREIGN KEY (examination_id) REFERENCES examinations(examination_id),
    FOREIGN KEY (action_id) REFERENCES action_types(action_id)
);

CREATE INDEX IF NOT EXISTS idx_action_labels_exam ON action_labels(examination_id);
CREATE INDEX IF NOT EXISTS idx_action_labels_action ON action_labels(action_id);
CREATE INDEX IF NOT EXISTS idx_action_labels_severity ON action_labels(severity_score);


-- ============================================================
-- 4. 数据集划分表 (用于训练/验证/测试)
-- ============================================================

-- 4.1 数据集划分配置
CREATE TABLE IF NOT EXISTS dataset_splits (
    split_id INTEGER PRIMARY KEY AUTOINCREMENT,
    split_name TEXT UNIQUE NOT NULL,          -- 'train_v1', 'val_v1', 'test_v1'
    split_type TEXT NOT NULL,                 -- 'train', 'val', 'test'
    split_version TEXT DEFAULT 'v1.0',
    
    -- 划分策略
    split_method TEXT,                        -- 'stratified', 'random', 'patient_based'
    split_ratio REAL,                         -- 划分比例
    
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- 4.2 数据集成员关系
CREATE TABLE IF NOT EXISTS dataset_members (
    member_id INTEGER PRIMARY KEY AUTOINCREMENT,
    split_id INTEGER NOT NULL,
    examination_id TEXT NOT NULL,
    
    -- 可选：用于交叉验证
    fold_number INTEGER,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(split_id, examination_id),
    
    FOREIGN KEY (split_id) REFERENCES dataset_splits(split_id),
    FOREIGN KEY (examination_id) REFERENCES examinations(examination_id)
);

CREATE INDEX IF NOT EXISTS idx_members_split ON dataset_members(split_id);
CREATE INDEX IF NOT EXISTS idx_members_exam ON dataset_members(examination_id);


-- ============================================================
-- 5. 模型推理结果表 (用于记录模型预测)
-- ============================================================

-- 5.1 检查级预测结果
CREATE TABLE IF NOT EXISTS examination_predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    examination_id TEXT NOT NULL,
    
    -- 模型信息
    model_name TEXT NOT NULL,                 -- 'HGFANet_v1.0'
    model_version TEXT NOT NULL,
    
    -- 预测结果
    pred_has_palsy INTEGER,
    pred_has_palsy_prob REAL,
    
    pred_palsy_side INTEGER,
    pred_palsy_side_prob REAL,
    
    pred_hb_grade INTEGER,
    pred_hb_grade_prob REAL,
    
    pred_sunnybrook_score REAL,
    
    -- 推理信息
    inference_time_ms REAL,                   -- 推理耗时(毫秒)
    device_type TEXT,                         -- 'cuda', 'cpu', 'mps'
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (examination_id) REFERENCES examinations(examination_id)
);

CREATE INDEX IF NOT EXISTS idx_pred_exam ON examination_predictions(examination_id);
CREATE INDEX IF NOT EXISTS idx_pred_model ON examination_predictions(model_name, model_version);


-- 5.2 动作级预测结果
CREATE TABLE IF NOT EXISTS action_predictions (
    prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    examination_id TEXT NOT NULL,
    action_id INTEGER NOT NULL,
    
    -- 模型信息
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    
    -- 预测结果
    pred_severity_score INTEGER,             -- 1-5
    pred_severity_probs TEXT,                -- JSON: [p1, p2, p3, p4, p5]
    
    -- 可解释性
    attention_map_path TEXT,                 -- 注意力图文件路径
    modal_weights TEXT,                      -- JSON: {geom: 0.3, vis_guided: 0.4, ...}
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (examination_id) REFERENCES examinations(examination_id),
    FOREIGN KEY (action_id) REFERENCES action_types(action_id)
);

CREATE INDEX IF NOT EXISTS idx_action_pred_exam ON action_predictions(examination_id);
CREATE INDEX IF NOT EXISTS idx_action_pred_model ON action_predictions(model_name, model_version);


-- ============================================================
-- 6. 导入日志表
-- ============================================================

CREATE TABLE IF NOT EXISTS import_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    
    import_type TEXT NOT NULL,                -- 'metadata', 'labels', 'features'
    import_source TEXT,                       -- 文件路径或描述
    
    records_processed INTEGER,
    records_succeeded INTEGER,
    records_failed INTEGER,
    
    error_messages TEXT,                      -- JSON格式的错误列表
    
    import_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_import_type ON import_logs(import_type);
CREATE INDEX IF NOT EXISTS idx_import_time ON import_logs(import_timestamp);


-- ============================================================
-- 7. 视图 (便捷查询)
-- ============================================================

-- 7.1 完整检查信息视图
CREATE VIEW IF NOT EXISTS v_examinations_full AS
SELECT 
    e.examination_id,
    e.patient_id,
    e.capture_datetime,
    e.has_videos,
    e.has_labels,
    
    -- 标签
    el.has_palsy,
    el.palsy_side,
    el.hb_grade,
    el.sunnybrook_score,
    
    -- 视频数量
    (SELECT COUNT(*) FROM video_files vf 
     WHERE vf.examination_id = e.examination_id) AS video_count,
    
    -- 标签数量
    (SELECT COUNT(*) FROM action_labels al 
     WHERE al.examination_id = e.examination_id) AS label_count
     
FROM examinations e
LEFT JOIN examination_labels el ON e.examination_id = el.examination_id;


-- 7.2 动作完整信息视图
CREATE VIEW IF NOT EXISTS v_actions_full AS
SELECT 
    vf.video_id,
    vf.examination_id,
    vf.action_id,
    at.action_name_en,
    at.action_name_cn,
    
    vf.video_file_path,
    vf.file_exists AS video_exists,
    
    al.severity_score,
    al.label_source,
    
    ff.frame_file_path AS peak_frame_path
    
FROM video_files vf
INNER JOIN action_types at ON vf.action_id = at.action_id
LEFT JOIN action_labels al ON vf.examination_id = al.examination_id 
    AND vf.action_id = al.action_id
LEFT JOIN frame_files ff ON vf.examination_id = ff.examination_id 
    AND vf.action_id = ff.action_id 
    AND ff.frame_type = 'peak';


-- 7.3 数据集统计视图
CREATE VIEW IF NOT EXISTS v_dataset_stats AS
SELECT
    ds.split_name,
    ds.split_type,
    COUNT(DISTINCT dm.examination_id) AS exam_count,

    -- 按HB分级统计
    SUM(CASE WHEN el.hb_grade = 1 THEN 1 ELSE 0 END) AS hb1_count,
    SUM(CASE WHEN el.hb_grade = 2 THEN 1 ELSE 0 END) AS hb2_count,
    SUM(CASE WHEN el.hb_grade = 3 THEN 1 ELSE 0 END) AS hb3_count,
    SUM(CASE WHEN el.hb_grade = 4 THEN 1 ELSE 0 END) AS hb4_count,
    SUM(CASE WHEN el.hb_grade = 5 THEN 1 ELSE 0 END) AS hb5_count,
    SUM(CASE WHEN el.hb_grade = 6 THEN 1 ELSE 0 END) AS hb6_count,

    -- 按面瘫/健康统计
    SUM(CASE WHEN el.has_palsy = 1 THEN 1 ELSE 0 END) AS palsy_count,
    SUM(CASE WHEN el.has_palsy = 0 THEN 1 ELSE 0 END) AS healthy_count

FROM dataset_splits ds
INNER JOIN dataset_members dm ON ds.split_id = dm.split_id
LEFT JOIN examination_labels el ON dm.examination_id = el.examination_id
GROUP BY ds.split_id;


-- ============================================================
-- 8. 视频处理相关表和视图
-- ============================================================

-- 8.1 视频处理特征表 (存储预处理后的特征)
CREATE TABLE IF NOT EXISTS video_features (
    feature_id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL UNIQUE,             -- 关联到video_files

    -- 峰值帧信息
    peak_frame_idx INTEGER NOT NULL,              -- 峰值帧索引
    peak_frame_path TEXT,                         -- 峰值帧图像路径

    -- 归一化单位长度
    unit_length REAL,                             -- 两眼内眦距离

    -- 几何特征 (存储为二进制BLOB)
    static_features BLOB NOT NULL,                -- 静态特征 (32维)
    dynamic_features BLOB NOT NULL,               -- 动态特征 (16维)

    -- 处理信息
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processing_time_ms REAL,                      -- 处理耗时(毫秒)

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (video_id) REFERENCES video_files(video_id)
);

CREATE INDEX IF NOT EXISTS idx_video_features_video ON video_features(video_id);


-- 8.2 简化视频视图 (用于process_videos.py)
CREATE VIEW IF NOT EXISTS videos AS
SELECT
    vf.video_id,
    vf.examination_id,
    vf.action_id,
    vf.video_file_path AS file_path,
    vf.start_frame,
    vf.end_frame,
    vf.fps,
    vf.file_exists,

    -- 添加is_processed标志
    CASE
        WHEN vfeat.feature_id IS NOT NULL THEN 1
        ELSE 0
    END AS is_processed,

    vfeat.processed_at,

    at.action_name_en,
    at.action_name_cn

FROM video_files vf
INNER JOIN action_types at ON vf.action_id = at.action_id
LEFT JOIN video_features vfeat ON vf.video_id = vfeat.video_id;
"""

def create_database(db_path):
    """
    创建完整的数据库schema

    Args:
        db_path: 数据库文件路径
    """
    # 确保目录存在（如果你想把db放在子目录）
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        print(f"[OK] 初始化数据库: {db_path}")
    finally:
        conn.close()

if __name__ == '__main__':
    # 创建数据库
    create_database('facialPalsy.db')