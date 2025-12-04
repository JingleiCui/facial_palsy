#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
层级面瘫数据集 (V2.0)

按检查(examination)组织数据，每个样本包含一次检查的所有动作
支持:
- 动作级任务 (severity分类)
- 检查级任务 (has_palsy, palsy_side, hb_grade, sunnybrook)
- 缺失动作处理 (通过action_mask)

关键设计:
- 每个样本是一个examination (不是单个视频)
- action_indices记录每个动作对应batch中的哪些examination
- 支持数据增强开关
"""

import sqlite3
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional


class HierarchicalPalsyDataset(Dataset):
    """
    层级面瘫数据集

    每个样本是一次检查(examination)，包含:
    - 最多11个动作的多模态特征
    - 动作级标签 (severity)
    - 检查级标签 (has_palsy, palsy_side, hb_grade, sunnybrook)
    """

    ACTION_NAMES = [
        'NeutralFace', 'Smile', 'RaiseEyebrow', 'CloseEyeHardly',
        'CloseEyeSoftly', 'BlowCheek', 'LipPucker', 'ShowTeeth',
        'ShrugNose', 'SpontaneousEyeBlink', 'VoluntaryEyeBlink'
    ]

    MAX_STATIC_DIM = 11
    MAX_DYNAMIC_DIM = 8
    VISUAL_DIM = 1280
    WRINKLE_DIM = 10
    MOTION_DIM = 12

    def __init__(self, db_path: str, fold: int, split_type: str = 'train',
                 split_version: str = 'v1.0', use_augmentation: bool = True):
        """
        Args:
            db_path: 数据库路径
            fold: 交叉验证折数 (0, 1, 2)
            split_type: 'train' 或 'val'
            split_version: 划分版本
            use_augmentation: 是否使用增强数据 (训练时True, 验证时False)
        """
        self.db_path = db_path
        self.fold = fold
        self.split_type = split_type
        self.split_version = split_version
        self.use_augmentation = use_augmentation

        self.examinations = self._load_examinations()

    def _load_examinations(self) -> List[Dict]:
        """加载该fold的所有检查"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 获取该fold的患者
        cursor.execute("""
            SELECT DISTINCT dm.patient_id
            FROM dataset_members dm
            JOIN dataset_splits ds ON dm.split_id = ds.split_id
            WHERE ds.split_type = ? AND dm.fold_number = ? AND ds.split_version = ?
        """, (self.split_type, self.fold, self.split_version))

        patient_ids = [row[0] for row in cursor.fetchall()]

        if not patient_ids:
            conn.close()
            return []

        # 获取这些患者的所有检查
        placeholders = ','.join(['?'] * len(patient_ids))
        cursor.execute(f"""
            SELECT 
                e.examination_id,
                e.patient_id,
                el.has_palsy,
                el.palsy_side,
                el.hb_grade,
                el.sunnybrook_score
            FROM examinations e
            LEFT JOIN examination_labels el ON e.examination_id = el.examination_id
            WHERE e.patient_id IN ({placeholders})
        """, patient_ids)

        examinations = []
        for row in cursor.fetchall():
            exam_id, patient_id, has_palsy, palsy_side, hb_grade, sunnybrook = row

            # 加载该检查的所有动作特征
            actions = self._load_examination_actions(cursor, exam_id)

            if actions:  # 至少有一个动作有特征
                examinations.append({
                    'examination_id': exam_id,
                    'patient_id': patient_id,
                    'has_palsy': has_palsy if has_palsy is not None else 0,
                    'palsy_side': self._encode_palsy_side(palsy_side),
                    'hb_grade': (hb_grade - 1) if hb_grade is not None else 0,  # 转为0-5索引
                    'sunnybrook': float(sunnybrook) if sunnybrook is not None else 0.0,
                    'actions': actions
                })

        conn.close()
        return examinations

    def _load_examination_actions(self, cursor, exam_id: str) -> Dict:
        """加载一次检查的所有动作特征"""
        # 根据是否使用增强数据选择条件
        if self.use_augmentation:
            aug_condition = ""
        else:
            aug_condition = "AND (vf.augmentation_type = 'original' OR vf.augmentation_type = 'none' OR vf.augmentation_type IS NULL)"

        cursor.execute(f"""
            SELECT 
                at.action_name_en,
                vf.static_features,
                vf.static_dim,
                vf.dynamic_features,
                vf.dynamic_dim,
                vf.visual_features,
                vf.visual_dim,
                vf.wrinkle_features,
                vf.wrinkle_dim,
                vf.motion_features,
                vf.motion_dim,
                al.severity_score
            FROM video_features vf
            JOIN video_files v ON vf.video_id = v.video_id
            JOIN action_types at ON v.action_id = at.action_id
            LEFT JOIN action_labels al ON v.examination_id = al.examination_id 
                AND v.action_id = al.action_id
            WHERE v.examination_id = ?
              AND vf.static_features IS NOT NULL
              {aug_condition}
            ORDER BY at.display_order
        """, (exam_id,))

        actions = {}
        for row in cursor.fetchall():
            (action_name, static_blob, static_dim, dynamic_blob, dynamic_dim,
             visual_blob, visual_dim, wrinkle_blob, wrinkle_dim,
             motion_blob, motion_dim, severity) = row

            # 解码特征
            static_dim = static_dim or self.MAX_STATIC_DIM
            dynamic_dim = dynamic_dim or 0
            visual_dim = visual_dim or self.VISUAL_DIM
            wrinkle_dim = wrinkle_dim or self.WRINKLE_DIM
            motion_dim = motion_dim or self.MOTION_DIM

            static = self._decode_feature(static_blob, static_dim)
            dynamic = self._decode_feature(dynamic_blob, dynamic_dim) if dynamic_dim > 0 else np.zeros(
                self.MAX_DYNAMIC_DIM, dtype=np.float32)
            visual = self._decode_feature(visual_blob, visual_dim) if visual_blob else np.zeros(self.VISUAL_DIM,
                                                                                                dtype=np.float32)
            wrinkle = self._decode_feature(wrinkle_blob, wrinkle_dim) if wrinkle_blob else np.zeros(self.WRINKLE_DIM,
                                                                                                    dtype=np.float32)
            motion = self._decode_feature(motion_blob, motion_dim) if motion_blob else np.zeros(self.MOTION_DIM,
                                                                                                dtype=np.float32)

            actions[action_name] = {
                'static': self._pad_feature(static, self.MAX_STATIC_DIM),
                'dynamic': self._pad_feature(dynamic, self.MAX_DYNAMIC_DIM),
                'visual': visual,
                'wrinkle': wrinkle,
                'motion': motion,
                'severity': (severity - 1) if severity is not None else 0  # 转为0-4索引
            }

        return actions

    def _decode_feature(self, blob: bytes, dim: int) -> np.ndarray:
        """解码float32 BLOB"""
        if blob is None:
            return np.zeros(dim, dtype=np.float32)
        arr = np.frombuffer(blob, dtype=np.float32, count=dim)
        return arr.copy()  # 返回可写的副本

    def _encode_palsy_side(self, side) -> int:
        """编码面瘫侧别"""
        if side is None:
            return 0
        if isinstance(side, int):
            return side
        mapping = {'none': 0, 'left': 1, 'right': 2, 'None': 0, '0': 0, '1': 1, '2': 2}
        return mapping.get(str(side), 0)

    def _pad_feature(self, feature: np.ndarray, max_dim: int) -> np.ndarray:
        """零填充到最大维度"""
        if len(feature) >= max_dim:
            return feature[:max_dim].copy()
        padded = np.zeros(max_dim, dtype=np.float32)
        padded[:len(feature)] = feature
        return padded

    def __len__(self):
        return len(self.examinations)

    def __getitem__(self, idx: int) -> Dict:
        exam = self.examinations[idx]

        # 构建动作特征字典和mask
        actions_tensor = {}
        action_severities = {}
        action_mask = []

        for action_name in self.ACTION_NAMES:
            if action_name in exam['actions']:
                action = exam['actions'][action_name]

                actions_tensor[action_name] = {
                    'static': torch.FloatTensor(action['static']),
                    'dynamic': torch.FloatTensor(action['dynamic']),
                    'visual': torch.FloatTensor(action['visual']),
                    'wrinkle': torch.FloatTensor(action['wrinkle']),  # 改为wrinkle匹配模型
                    'motion': torch.FloatTensor(action['motion']),
                }

                action_severities[action_name] = action['severity']
                action_mask.append(1.0)
            else:
                action_mask.append(0.0)

        return {
            'examination_id': exam['examination_id'],
            'actions': actions_tensor,
            'action_mask': torch.FloatTensor(action_mask),
            'targets': {
                'action_severity': action_severities,
                'has_palsy': exam['has_palsy'],
                'palsy_side': exam['palsy_side'],
                'hb_grade': exam['hb_grade'],
                'sunnybrook': exam['sunnybrook'],
            }
        }


def collate_hierarchical(batch: List[Dict]) -> Dict:
    """
    自定义collate函数

    关键：记录每个动作对应batch中的哪些examination (action_indices)
    """
    batch_size = len(batch)

    # 收集所有动作，并记录每个动作属于batch中的哪个examination
    all_actions = {}
    action_indices = {}

    for action_name in HierarchicalPalsyDataset.ACTION_NAMES:
        action_batch = {
            'static': [],
            'dynamic': [],
            'visual': [],
            'wrinkle': [],  # 改为wrinkle匹配模型
            'motion': [],
        }
        indices = []  # 这个动作对应的examination索引

        for exam_idx, sample in enumerate(batch):
            if action_name in sample['actions']:
                action = sample['actions'][action_name]
                for key in action_batch.keys():
                    action_batch[key].append(action[key])
                indices.append(exam_idx)

        if action_batch['static']:
            all_actions[action_name] = {
                'static': torch.stack(action_batch['static']),
                'dynamic': torch.stack(action_batch['dynamic']),
                'visual': torch.stack(action_batch['visual']),
                'wrinkle': torch.stack(action_batch['wrinkle']),  # 改为wrinkle匹配模型
                'motion': torch.stack(action_batch['motion']),
            }
            action_indices[action_name] = indices

    # 收集动作级标签 - 转换为固定shape的tensor (B, 11)
    # 初始化为-1 (表示缺失的动作,CE loss会ignore)
    severity_matrix = torch.full((batch_size, 11), -1, dtype=torch.long)

    for action_idx, action_name in enumerate(HierarchicalPalsyDataset.ACTION_NAMES):
        for sample_idx, sample in enumerate(batch):
            if action_name in sample['targets']['action_severity']:
                severity = sample['targets']['action_severity'][action_name]
                severity_matrix[sample_idx, action_idx] = severity

    return {
        'actions': all_actions,
        'action_indices': action_indices,
        'action_mask': torch.stack([s['action_mask'] for s in batch]),
        'targets': {
            'action_severity': severity_matrix,  # (B, 11) tensor
            'has_palsy': torch.LongTensor([s['targets']['has_palsy'] for s in batch]),
            'palsy_side': torch.LongTensor([s['targets']['palsy_side'] for s in batch]),
            'hb_grade': torch.LongTensor([s['targets']['hb_grade'] for s in batch]),
            'sunnybrook': torch.FloatTensor([s['targets']['sunnybrook'] for s in batch]),
        }
    }