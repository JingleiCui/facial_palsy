"""
支持层级多任务学习的数据集
按检查(examination)组织，每个样本包含11个动作
"""
import sqlite3
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional
import io
from PIL import Image


class HierarchicalPalsyDataset(Dataset):
    """
    层级面瘫数据集

    每个样本是一次检查(examination)，包含:
    - 11个动作的多模态特征
    - 动作级标签 (severity)
    - 检查级标签 (has_palsy, palsy_side, hb_grade, sunnybrook)
    """

    ACTION_NAMES = [
        'NeutralFace', 'Smile', 'RaiseEyebrow', 'CloseEyeHardly',
        'CloseEyeSoftly', 'BlowCheek', 'LipPucker', 'ShowTeeth',
        'ShrugNose', 'SpontaneousEyeBlink', 'VoluntaryEyeBlink'
    ]

    # 每个动作的特征维度
    ACTION_DIMS = {
        'NeutralFace': {'static': 7, 'dynamic': 0},
        'Smile': {'static': 9, 'dynamic': 6},
        'RaiseEyebrow': {'static': 7, 'dynamic': 4},
        'CloseEyeHardly': {'static': 5, 'dynamic': 8},
        'CloseEyeSoftly': {'static': 5, 'dynamic': 8},
        'BlowCheek': {'static': 7, 'dynamic': 4},
        'LipPucker': {'static': 7, 'dynamic': 4},
        'ShowTeeth': {'static': 11, 'dynamic': 4},
        'ShrugNose': {'static': 9, 'dynamic': 4},
        'SpontaneousEyeBlink': {'static': 5, 'dynamic': 7},
        'VoluntaryEyeBlink': {'static': 5, 'dynamic': 4},
    }

    MAX_STATIC_DIM = 11
    MAX_DYNAMIC_DIM = 8
    VISUAL_DIM = 1280
    WRINKLE_DIM = 10
    MOTION_DIM = 12

    def __init__(self, db_path: str, fold: int, split_type: str = 'train',
                 split_version: str = 'v1.0', use_augmentation: bool = True,
                 use_wrinkle_heatmap: bool = True):
        """
        Args:
            db_path: 数据库路径
            fold: 交叉验证折数 (0, 1, 2)
            split_type: 'train' 或 'val'
            use_augmentation: 是否使用增强数据
            use_wrinkle_heatmap: 是否加载皱纹热力图
        """
        self.db_path = db_path
        self.fold = fold
        self.split_type = split_type
        self.split_version = split_version
        self.use_augmentation = use_augmentation
        self.use_wrinkle_heatmap = use_wrinkle_heatmap

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
            JOIN examination_labels el ON e.examination_id = el.examination_id
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
                    'has_palsy': has_palsy,
                    'palsy_side': self._encode_palsy_side(palsy_side),
                    'hb_grade': hb_grade - 1,  # 转为0-5索引
                    'sunnybrook': sunnybrook or 0,
                    'actions': actions
                })

        conn.close()
        return examinations

    def _load_examination_actions(self, cursor, exam_id: int) -> Dict:
        """加载一次检查的所有动作特征"""
        aug_condition = "" if self.use_augmentation else "AND vf.augmentation_type = 'original'"

        cursor.execute(f"""
            SELECT 
                at.action_name_en,
                vf.static_features,
                vf.static_dim,
                vf.dynamic_features,
                vf.dynamic_dim,
                vf.visual_features,
                vf.wrinkle_features,
                vf.wrinkle_heatmap,
                vf.motion_features,
                al.severity_score
            FROM video_features vf
            JOIN video_files v ON vf.video_id = v.video_id
            JOIN action_types at ON v.action_id = at.action_id
            LEFT JOIN action_labels al ON v.examination_id = al.examination_id 
                AND v.action_id = al.action_id
            WHERE v.examination_id = ?
              AND vf.static_features IS NOT NULL
              {aug_condition}
        """, (exam_id,))

        actions = {}
        for row in cursor.fetchall():
            (action_name, static_blob, static_dim, dynamic_blob, dynamic_dim,
             visual_blob, wrinkle_blob, wrinkle_hm_blob, motion_blob, severity) = row

            # 解码特征
            static = self._decode_feature(static_blob, static_dim)
            dynamic = self._decode_feature(dynamic_blob, dynamic_dim) if dynamic_dim > 0 else np.zeros(0)
            visual = self._decode_feature(visual_blob, self.VISUAL_DIM)
            wrinkle = self._decode_feature(wrinkle_blob, self.WRINKLE_DIM) if wrinkle_blob else np.zeros(
                self.WRINKLE_DIM)
            motion = self._decode_feature(motion_blob, self.MOTION_DIM) if motion_blob else np.zeros(self.MOTION_DIM)

            # 皱纹热力图
            wrinkle_hm = None
            if self.use_wrinkle_heatmap and wrinkle_hm_blob:
                wrinkle_hm = self._decode_heatmap(wrinkle_hm_blob)

            actions[action_name] = {
                'static': static,
                'dynamic': dynamic,
                'visual': visual,
                'wrinkle_scalar': wrinkle,
                'wrinkle_heatmap': wrinkle_hm,
                'motion': motion,
                'severity': severity - 1 if severity else 0  # 转为0-4索引
            }

        return actions

    def _decode_feature(self, blob: bytes, dim: int) -> np.ndarray:
        """解码float32 BLOB"""
        if blob is None:
            return np.zeros(dim, dtype=np.float32)
        return np.frombuffer(blob, dtype=np.float32, count=dim)

    def _decode_heatmap(self, blob: bytes) -> np.ndarray:
        """解码皱纹热力图"""
        try:
            img = Image.open(io.BytesIO(blob))
            arr = np.array(img.convert('L'), dtype=np.float32) / 255.0
            return arr
        except:
            return np.zeros((64, 64), dtype=np.float32)

    def _encode_palsy_side(self, side: str) -> int:
        """编码面瘫侧别"""
        mapping = {'none': 0, 'left': 1, 'right': 2, None: 0}
        return mapping.get(side, 0)

    def _pad_feature(self, feature: np.ndarray, max_dim: int) -> np.ndarray:
        """零填充到最大维度"""
        if len(feature) >= max_dim:
            return feature[:max_dim]
        padded = np.zeros(max_dim, dtype=np.float32)
        padded[:len(feature)] = feature
        return padded

    def __len__(self):
        return len(self.examinations)

    def __getitem__(self, idx: int) -> Dict:
        exam = self.examinations[idx]

        # 构建动作特征字典
        actions_tensor = {}
        action_severities = {}
        action_mask = []

        for action_name in self.ACTION_NAMES:
            if action_name in exam['actions']:
                action = exam['actions'][action_name]

                actions_tensor[action_name] = {
                    'static': torch.FloatTensor(
                        self._pad_feature(action['static'], self.MAX_STATIC_DIM)
                    ),
                    'dynamic': torch.FloatTensor(
                        self._pad_feature(action['dynamic'], self.MAX_DYNAMIC_DIM)
                    ),
                    'visual': torch.FloatTensor(action['visual']),
                    'wrinkle_scalar': torch.FloatTensor(action['wrinkle_scalar']),
                    'motion': torch.FloatTensor(action['motion']),
                }

                if action['wrinkle_heatmap'] is not None:
                    actions_tensor[action_name]['wrinkle_heatmap'] = torch.FloatTensor(
                        action['wrinkle_heatmap']
                    ).unsqueeze(0)  # (1, H, W)

                action_severities[action_name] = action['severity']
                action_mask.append(1)
            else:
                action_mask.append(0)

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
    """自定义collate函数，处理变长动作"""
    batch_size = len(batch)

    # 收集所有动作
    all_actions = {}
    for action_name in HierarchicalPalsyDataset.ACTION_NAMES:
        action_batch = {
            'static': [],
            'dynamic': [],
            'visual': [],
            'wrinkle_scalar': [],
            'motion': [],
            'wrinkle_heatmap': []
        }

        for sample in batch:
            if action_name in sample['actions']:
                action = sample['actions'][action_name]
                for key in ['static', 'dynamic', 'visual', 'wrinkle_scalar', 'motion']:
                    action_batch[key].append(action[key])
                if 'wrinkle_heatmap' in action:
                    action_batch['wrinkle_heatmap'].append(action['wrinkle_heatmap'])

        if action_batch['static']:
            all_actions[action_name] = {
                'static': torch.stack(action_batch['static']),
                'dynamic': torch.stack(action_batch['dynamic']),
                'visual': torch.stack(action_batch['visual']),
                'wrinkle_scalar': torch.stack(action_batch['wrinkle_scalar']),
                'motion': torch.stack(action_batch['motion']),
            }
            if action_batch['wrinkle_heatmap']:
                all_actions[action_name]['wrinkle_heatmap'] = torch.stack(
                    action_batch['wrinkle_heatmap']
                )

    # 收集标签
    action_severities = {}
    for action_name in HierarchicalPalsyDataset.ACTION_NAMES:
        severities = []
        for sample in batch:
            if action_name in sample['targets']['action_severity']:
                severities.append(sample['targets']['action_severity'][action_name])
        if severities:
            action_severities[action_name] = torch.LongTensor(severities)

    return {
        'actions': all_actions,
        'action_mask': torch.stack([s['action_mask'] for s in batch]),
        'targets': {
            'action_severity': action_severities,
            'has_palsy': torch.LongTensor([s['targets']['has_palsy'] for s in batch]),
            'palsy_side': torch.LongTensor([s['targets']['palsy_side'] for s in batch]),
            'hb_grade': torch.LongTensor([s['targets']['hb_grade'] for s in batch]),
            'sunnybrook': torch.FloatTensor([s['targets']['sunnybrook'] for s in batch]),
        }
    }