# 面瘫视频预处理与特征提取

## 概述

本模块实现了H-GFA Net的**阶段1(预处理与峰值帧检测)** + **阶段2(几何特征提取)**的整合流程。

### 核心优化

阶段1和阶段2可以**同时进行**,因为:
- 阶段1提取关键点序列后,可以立即检测峰值帧
- 阶段2的动态特征从整个序列计算
- 阶段2的静态特征从峰值帧计算
- 两个阶段共享同一个关键点序列,无需重复计算

## 项目结构

```
facialPalsy3/
├── config.py                    # 全局配置
├── database_schema.py           # 数据库schema定义
├── import_metadata.py           # 导入视频元数据
├── import_labels.py             # 导入医生标注
├── process_videos.py            # 主处理脚本 ⭐
│
├── core/                        # 核心模块
│   ├── landmark_extractor.py   # MediaPipe关键点提取
│   ├── geometry_utils.py       # 几何计算工具
│   ├── feature_extractor.py    # 特征提取器
│   └── database_manager.py     # 数据库管理
│
└── actions/                     # 11个动作类
    ├── __init__.py             # 动作注册
    ├── base_action.py          # 动作基类
    ├── neutral_face.py         # 1. 静息
    ├── spontaneous_eye_blink.py # 2. 自然眨眼
    ├── voluntary_eye_blink.py  # 3. 自主眨眼
    ├── close_eye_softly.py     # 4. 轻轻闭眼
    ├── close_eye_hardly.py     # 5. 用力闭眼
    ├── raise_eyebrow.py        # 6. 抬眉
    ├── smile.py                # 7. 微笑
    ├── shrug_nose.py           # 8. 耸鼻
    ├── show_teeth.py           # 9. 呲牙
    ├── blow_cheek.py           # 10. 鼓腮
    └── lip_pucker.py           # 11. 撅嘴
```

## 使用流程

### 1. 配置环境

修改 `config.py` 中的路径:

```python
# 数据库路径
DB_PATH = 'facialPalsy.db'

# 视频根目录
VIDEOS_BASE_PATH = '/path/to/videos'

# MediaPipe模型路径
MEDIAPIPE_MODEL_PATH = '/path/to/face_landmarker.task'

# 输出根目录
OUTPUT_BASE_DIR = '/path/to/processed'
```

### 2. 初始化数据库

```bash
cd facialPalsy
python database_schema.py
```

### 3. 导入元数据

```bash
python import_metadata.py
```

这会扫描视频目录,将视频信息导入数据库的 `video_files` 表。

### 4. 导入医生标注

```bash
python import_labels.py
```

这会从Excel读取医生标注,导入到 `examination_labels` 和 `action_labels` 表。

### 5. 处理视频 (阶段1+阶段2)

```bash
python process_videos.py
```

这是**核心脚本**,会:
1. 从数据库读取未处理的视频
2. 对每个视频:
   - 提取关键点序列
   - 检测峰值帧
   - **同时**提取静态和动态几何特征
3. 保存结果到:
   - 数据库 `video_features` 表
   - 峰值帧图像 `processed/peak_frames/`
   - 特征文件 `processed/features/` (可选)

#### 处理参数

在 `process_videos.py` 的 `main()` 函数中可以调整:

```python
processor.process_all_videos(
    limit=10,              # 限制处理数量,None=全部
    skip_processed=True    # 跳过已处理的视频
)
```

### 6. 检查结果

查询数据库:

```sql
-- 查看处理进度
SELECT
    COUNT(*) AS total,
    SUM(is_processed) AS processed,
    COUNT(*) - SUM(is_processed) AS remaining
FROM videos
WHERE file_exists = 1;

-- 查看某个检查的所有动作特征
SELECT
    v.examination_id,
    v.action_name_cn,
    vf.peak_frame_idx,
    vf.unit_length,
    vf.processed_at
FROM videos v
JOIN video_features vf ON v.video_id = vf.video_id
WHERE v.examination_id = 'XW000001_20250905-13-26-45'
ORDER BY v.action_id;
```

## 特征说明

### 静态几何特征 (32维)

从峰值帧提取,描述空间结构:
- 眼睛开合度(左/右)
- 眼睛长度(左/右)
- 眉眼距(左/右)
- 嘴角宽度
- 嘴部高度
- 对称性指标(8维)

### 动态几何特征 (16维)

从整个序列提取,描述时序变化:
- 运动范围(4维): 眼、嘴、眉的运动幅度
- 平均速度(4维): 运动速度
- 最大速度(4维): 峰值速度
- 运动平滑度(4维): 基于jerk的平滑度

## 11个动作的峰值帧检测逻辑

| 动作 | 中文 | 峰值帧检测策略 |
|------|------|----------------|
| NeutralFace | 静息 | 中间段最稳定的帧 |
| SpontaneousEyeBlink | 自然眨眼 | 眼睛开合度最小 |
| VoluntaryEyeBlink | 自主眨眼 | 眼睛开合度最小 |
| CloseEyeSoftly | 轻轻闭眼 | 眼睛开合度最小 |
| CloseEyeHardly | 用力闭眼 | 眼睛开合度最小 |
| RaiseEyebrow | 抬眉 | 眉眼距最大 |
| Smile | 微笑 | 嘴角宽度最大 |
| ShrugNose | 耸鼻 | 上唇到鼻尖距离最小 |
| ShowTeeth | 呲牙 | 嘴角宽度+嘴部高度综合评分最大 |
| BlowCheek | 鼓腮 | 嘴部高度最小(闭合) |
| LipPucker | 撅嘴 | 嘴角宽度最小 |

## 数据库Schema

### video_files 表

存储视频元数据:
- video_id, examination_id, action_id
- video_file_path, start_frame, end_frame
- fps, file_exists

### video_features 表

存储处理结果:
- video_id (外键)
- peak_frame_idx, peak_frame_path
- unit_length
- static_features (BLOB, 32维)
- dynamic_features (BLOB, 16维)
- processed_at

### videos 视图

简化的查询视图:
- 合并 video_files + video_features + action_types
- is_processed 自动计算(是否存在video_features记录)

## 故障排查

### 1. 导入错误

如果遇到 `ModuleNotFoundError`:

```bash
# 确保在项目根目录运行
cd /Users/cuijinglei/PycharmProjects/medicalProject
python facialPalsy/process_videos.py
```

### 2. MediaPipe错误

如果关键点提取失败:
- 检查模型路径是否正确
- 确保视频文件可读
- 检查视频质量(人脸是否清晰)

### 3. 有效帧比例过低

如果提示"有效帧比例过低":
- 默认阈值: 50% (config.py: MIN_VALID_FRAME_RATIO)
- 可能原因: 视频中人脸不清晰或遮挡
- 解决方法: 检查视频质量或降低阈值

### 4. 峰值帧检测不准确

如果峰值帧不是预期的:
- 检查对应动作类的 `detect_peak_frame()` 实现
- 可以添加可视化代码,绘制指标曲线
- 调整检测策略参数

## 下一步

处理完成后,可以进行:

1. **特征可视化**: 绘制特征分布、对称性等
2. **数据集划分**: 创建训练/验证/测试集
3. **模型训练**: 使用提取的特征训练H-GFA Net
4. **模型评估**: 在测试集上评估性能

## 注意事项

1. **单个运行**: 所有脚本都设计为单独运行,不需要pipeline
2. **增量处理**: `skip_processed=True` 会自动跳过已处理的视频
3. **数据完整性**: 使用外键约束,确保数据一致性
4. **特征格式**: 特征存储为二进制BLOB,读取时需要:
   ```python
   import numpy as np
   static_features = np.frombuffer(blob_data, dtype=np.float32)
   ```

## 参考

- 设计文档: `docs/H-GFA_Net_complete_design_doc.md`
- 流程文档: `docs/H-GFA_Net_完整数据处理与训练流程.md`
