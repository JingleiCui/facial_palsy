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
├── video_pipeline.py  
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



## 特征说明

### 静态几何特征

从峰值帧提取,描述空间结构:


### 动态几何特征 

从整个序列提取,描述时序变化:


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
- static_features
- dynamic_features
- processed_at

### videos 视图

简化的查询视图:
- 合并 video_files + video_features + action_types
- is_processed 自动计算(是否存在video_features记录)


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

## 参考

- 设计文档: `docs/H-GFA_Net_complete_design_doc.md`
- 流程文档: `docs/H-GFA_Net_完整数据处理与训练流程.md`
