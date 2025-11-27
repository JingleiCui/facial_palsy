"""
数据增强与数据集划分模块
========================

功能:
1. 患者级数据集划分 (避免数据泄露)
2. 分层抽样 (按HB分级)
3.
划分策略:
- 患者级划分: 同一患者的所有检查只出现在一个集合中
- 分层抽样: 保持各集合中HB分级的比例一致
- 默认比例: Train 80%, Val 10%, Test 10%

用法:
    python data_augmentation.py split [db_path]      # 执行数据划分
    python data_augmentation.py stats [db_path]      # 查看统计信息
"""

import os
import sys
import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold

import torch
import torch.nn as nn


# =============================================================================
# 数据集划分
# =============================================================================

class DatasetSplitter:
    """患者级数据集划分器"""

    def __init__(
            self,
            db_path: str,
            train_ratio: float = 0.80,
            val_ratio: float = 0.10,
            test_ratio: float = 0.10,
            random_seed: int = 42
    ):
        """
        初始化

        Args:
            db_path: 数据库路径
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            random_seed: 随机种子
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "比例之和必须为1"

        self.db_path = db_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed

        np.random.seed(random_seed)

    def get_patient_labels(self) -> Dict[str, int]:
        """
        获取每个患者的主要标签 (用于分层抽样)

        Returns:
            {patient_id: hb_grade}
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 获取每个患者的HB分级 (取最严重的)
        query = """
            SELECT 
                e.patient_id,
                MAX(COALESCE(el.hb_grade, 1)) as max_hb
            FROM examinations e
            LEFT JOIN examination_labels el ON e.examination_id = el.examination_id
            WHERE e.is_valid = 1
            GROUP BY e.patient_id
        """

        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()

        return {pid: hb for pid, hb in results}

    def get_patient_exam_counts(self) -> Dict[str, int]:
        """获取每个患者的检查次数"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
            SELECT patient_id, COUNT(*) as exam_count
            FROM examinations
            WHERE is_valid = 1
            GROUP BY patient_id
        """

        cursor.execute(query)
        results = cursor.fetchall()
        conn.close()

        return {pid: count for pid, count in results}

    def stratified_patient_split(
            self,
            patient_labels: Dict[str, int]
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        分层抽样划分患者

        Args:
            patient_labels: {patient_id: label}

        Returns:
            train_patients, val_patients, test_patients
        """
        patients = list(patient_labels.keys())
        labels = [patient_labels[p] for p in patients]

        # 检查每个标签是否有足够样本
        label_counts = defaultdict(int)
        for label in labels:
            label_counts[label] += 1

        min_count = min(label_counts.values())

        # 如果样本太少，降级为随机划分
        if min_count < 3:
            print(f"[WARN] 某些类别样本太少 (最少={min_count})，使用随机划分")
            return self._random_split(patients)

        # 第一次划分: train vs (val + test)
        train_patients, temp_patients, train_labels, temp_labels = train_test_split(
            patients, labels,
            test_size=(self.val_ratio + self.test_ratio),
            stratify=labels,
            random_state=self.random_seed
        )

        # 第二次划分: val vs test
        val_ratio_adjusted = self.val_ratio / (self.val_ratio + self.test_ratio)

        # 检查temp是否足够分层
        temp_label_counts = defaultdict(int)
        for label in temp_labels:
            temp_label_counts[label] += 1

        if min(temp_label_counts.values()) >= 2:
            val_patients, test_patients = train_test_split(
                temp_patients,
                test_size=(1 - val_ratio_adjusted),
                stratify=temp_labels,
                random_state=self.random_seed
            )
        else:
            # 随机划分
            split_idx = int(len(temp_patients) * val_ratio_adjusted)
            np.random.shuffle(temp_patients)
            val_patients = temp_patients[:split_idx]
            test_patients = temp_patients[split_idx:]

        return train_patients, val_patients, test_patients

    def _random_split(
            self,
            patients: List[str]
    ) -> Tuple[List[str], List[str], List[str]]:
        """随机划分 (备用)"""
        patients = list(patients)
        np.random.shuffle(patients)

        n = len(patients)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)

        return (
            patients[:train_end],
            patients[train_end:val_end],
            patients[val_end:]
        )

    def save_split_to_db(
            self,
            train_patients: List[str],
            val_patients: List[str],
            test_patients: List[str],
            version: str = "v1.0"
    ):
        """
        将划分结果保存到数据库
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 清除旧的划分
        cursor.execute("DELETE FROM dataset_members")
        cursor.execute("DELETE FROM dataset_splits")

        # 创建划分配置
        splits = [
            (f'train_{version}', 'train', self.train_ratio, train_patients),
            (f'val_{version}', 'val', self.val_ratio, val_patients),
            (f'test_{version}', 'test', self.test_ratio, test_patients),
        ]

        for split_name, split_type, ratio, patients in splits:
            cursor.execute("""
                INSERT INTO dataset_splits 
                (split_name, split_type, split_version, split_method, split_ratio, random_seed)
                VALUES (?, ?, ?, 'patient_stratified', ?, ?)
            """, (split_name, split_type, version, ratio, self.random_seed))

            split_id = cursor.lastrowid

            for patient_id in patients:
                cursor.execute("""
                    INSERT INTO dataset_members (split_id, patient_id)
                    VALUES (?, ?)
                """, (split_id, patient_id))

        conn.commit()
        conn.close()

        print(f"[OK] 数据划分已保存 (version={version})")

    def run_split(self, version: str = "v1.0"):
        """执行完整的划分流程"""
        print("=" * 60)
        print("患者级数据集划分")
        print("=" * 60)

        # 1. 获取患者标签
        patient_labels = self.get_patient_labels()
        print(f"总患者数: {len(patient_labels)}")

        # 统计标签分布
        label_dist = defaultdict(int)
        for label in patient_labels.values():
            label_dist[label] += 1
        print("HB分级分布:")
        for hb in sorted(label_dist.keys()):
            print(f"  HB {hb}: {label_dist[hb]} 患者")

        # 2. 执行划分
        train_p, val_p, test_p = self.stratified_patient_split(patient_labels)

        print(f"\n划分结果:")
        print(f"  训练集: {len(train_p)} 患者 ({len(train_p) / len(patient_labels) * 100:.1f}%)")
        print(f"  验证集: {len(val_p)} 患者 ({len(val_p) / len(patient_labels) * 100:.1f}%)")
        print(f"  测试集: {len(test_p)} 患者 ({len(test_p) / len(patient_labels) * 100:.1f}%)")

        # 3. 验证各集合的标签分布
        for name, patients in [('训练', train_p), ('验证', val_p), ('测试', test_p)]:
            dist = defaultdict(int)
            for p in patients:
                dist[patient_labels[p]] += 1
            print(f"\n{name}集HB分布: {dict(sorted(dist.items()))}")

        # 4. 保存到数据库
        self.save_split_to_db(train_p, val_p, test_p, version)

        return train_p, val_p, test_p

    def get_split_video_ids(self, split_type: str = 'train') -> List[int]:
        """
        获取指定划分的所有video_id

        Args:
            split_type: 'train', 'val', 'test'

        Returns:
            video_id列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
            SELECT vf.video_id
            FROM video_files vf
            INNER JOIN examinations e ON vf.examination_id = e.examination_id
            INNER JOIN dataset_members dm ON e.patient_id = dm.patient_id
            INNER JOIN dataset_splits ds ON dm.split_id = ds.split_id
            WHERE ds.split_type = ?
              AND vf.file_exists = 1
        """

        cursor.execute(query, (split_type,))
        results = cursor.fetchall()
        conn.close()

        return [r[0] for r in results]

# =============================================================================
# 数据加载器工厂
# =============================================================================

def load_samples_from_db(
        db_path: str,
        split_type: str = 'train'
) -> List[Dict[str, Any]]:
    """
    从数据库加载指定划分的样本

    Args:
        db_path: 数据库路径
        split_type: 'train', 'val', 'test'

    Returns:
        样本列表
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = """
        SELECT 
            vf.video_id,
            at.action_name_en,
            vfeat.static_features,
            vfeat.static_dim,
            vfeat.dynamic_features,
            vfeat.dynamic_dim,
            vfeat.visual_features,
            vfeat.wrinkle_features,
            vfeat.wrinkle_dim,
            vfeat.motion_features,
            vfeat.motion_dim,
            al.severity_score,
            el.has_palsy,
            el.palsy_side,
            el.hb_grade,
            el.sunnybrook_score
        FROM video_files vf
        INNER JOIN examinations e ON vf.examination_id = e.examination_id
        INNER JOIN action_types at ON vf.action_id = at.action_id
        INNER JOIN dataset_members dm ON e.patient_id = dm.patient_id
        INNER JOIN dataset_splits ds ON dm.split_id = ds.split_id
        LEFT JOIN video_features vfeat ON vf.video_id = vfeat.video_id
        LEFT JOIN action_labels al ON vf.examination_id = al.examination_id 
            AND vf.action_id = al.action_id
        LEFT JOIN examination_labels el ON vf.examination_id = el.examination_id
        WHERE ds.split_type = ?
          AND vf.file_exists = 1
          AND vfeat.static_features IS NOT NULL
    """

    cursor.execute(query, (split_type,))
    rows = cursor.fetchall()
    conn.close()

    samples = []

    for row in rows:
        (video_id, action_name,
         static_blob, static_dim,
         dynamic_blob, dynamic_dim,
         visual_blob,
         wrinkle_blob, wrinkle_dim,
         motion_blob, motion_dim,
         severity, has_palsy, palsy_side, hb_grade, sunnybrook) = row

        sample = {
            'video_id': video_id,
            'action_name': action_name,

            # 几何特征
            'static_features': None,
            'static_dim': static_dim or 0,
            'dynamic_features': None,
            'dynamic_dim': dynamic_dim or 0,

            # 视觉特征
            'visual_features': None,

            # 皱纹特征
            'wrinkle_features': None,
            'wrinkle_dim': wrinkle_dim or 0,

            # 运动特征
            'motion_features': None,
            'motion_dim': motion_dim or 0,

            # 标签
            'severity': severity,
            'has_palsy': has_palsy,
            'palsy_side': palsy_side,
            'hb_grade': hb_grade,
            'sunnybrook': sunnybrook,
        }

        # 解析BLOB
        if static_blob:
            sample['static_features'] = np.frombuffer(static_blob, dtype=np.float32)

        if dynamic_blob:
            sample['dynamic_features'] = np.frombuffer(dynamic_blob, dtype=np.float32)

        if visual_blob:
            sample['visual_features'] = np.frombuffer(visual_blob, dtype=np.float32)

        if wrinkle_blob:
            sample['wrinkle_features'] = np.frombuffer(wrinkle_blob, dtype=np.float32)

        if motion_blob:
            sample['motion_features'] = np.frombuffer(motion_blob, dtype=np.float32)

        samples.append(sample)

    return samples


def print_dataset_stats(db_path: str):
    """打印数据集统计信息"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("\n" + "=" * 60)
    print("数据集统计")
    print("=" * 60)

    # 1. 总体统计
    cursor.execute("SELECT COUNT(*) FROM patients")
    total_patients = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM examinations WHERE is_valid = 1")
    total_exams = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM video_files WHERE file_exists = 1")
    total_videos = cursor.fetchone()[0]

    print(f"\n总体统计:")
    print(f"  患者数: {total_patients}")
    print(f"  检查数: {total_exams}")
    print(f"  视频数: {total_videos}")

    # 2. 划分统计
    cursor.execute("""
        SELECT 
            ds.split_type,
            COUNT(DISTINCT dm.patient_id) as patients,
            COUNT(DISTINCT e.examination_id) as exams,
            COUNT(DISTINCT vf.video_id) as videos
        FROM dataset_splits ds
        LEFT JOIN dataset_members dm ON ds.split_id = dm.split_id
        LEFT JOIN examinations e ON dm.patient_id = e.patient_id
        LEFT JOIN video_files vf ON e.examination_id = vf.examination_id AND vf.file_exists = 1
        GROUP BY ds.split_type
    """)

    results = cursor.fetchall()

    if results:
        print(f"\n划分统计:")
        for split_type, patients, exams, videos in results:
            print(f"  {split_type:6} - 患者:{patients:4}, 检查:{exams:4}, 视频:{videos:5}")
    else:
        print("\n[!] 尚未进行数据划分，请先运行: python data_augmentation.py split")

    # 3. 特征完整性
    cursor.execute("""
        SELECT 
            SUM(CASE WHEN static_features IS NOT NULL THEN 1 ELSE 0 END) as geo,
            SUM(CASE WHEN visual_features IS NOT NULL THEN 1 ELSE 0 END) as vis,
            SUM(CASE WHEN wrinkle_features IS NOT NULL THEN 1 ELSE 0 END) as wrk,
            SUM(CASE WHEN motion_features IS NOT NULL THEN 1 ELSE 0 END) as mot,
            COUNT(*) as total
        FROM video_features
    """)

    geo, vis, wrk, mot, total = cursor.fetchone()

    print(f"\n特征提取进度:")
    print(f"  几何特征: {geo or 0}/{total or 0}")
    print(f"  视觉特征: {vis or 0}/{total or 0}")
    print(f"  皱纹特征: {wrk or 0}/{total or 0}")
    print(f"  运动特征: {mot or 0}/{total or 0}")

    # 4. HB分级分布
    cursor.execute("""
        SELECT hb_grade, COUNT(*) 
        FROM examination_labels 
        WHERE hb_grade IS NOT NULL
        GROUP BY hb_grade
        ORDER BY hb_grade
    """)

    hb_results = cursor.fetchall()

    if hb_results:
        print(f"\nHB分级分布:")
        for hb, count in hb_results:
            print(f"  HB {hb}: {count} 检查")

    conn.close()
    print("=" * 60)


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法:")
        print("  python data_augmentation.py split [db_path]  # 执行数据划分")
        print("  python data_augmentation.py stats [db_path]  # 查看统计信息")
        sys.exit(1)

    command = sys.argv[1]
    db_path = sys.argv[2] if len(sys.argv) > 2 else 'facialPalsy.db'

    if command == 'split':
        splitter = DatasetSplitter(
            db_path=db_path,
            train_ratio=0.80,
            val_ratio=0.10,
            test_ratio=0.10,
            random_seed=42
        )
        splitter.run_split(version="v1.0")

    elif command == 'stats':
        print_dataset_stats(db_path)

    else:
        print(f"未知命令: {command}")
        print("可用命令: split, stats")
        sys.exit(1)


if __name__ == '__main__':
    main()