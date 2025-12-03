#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集划分脚本 (V2.0 - 支持检查级层级多任务学习)

功能：
1. 按患者级别进行分层3折交叉验证 (防止数据泄漏)
2. 按HB分级进行分层采样 (确保各fold分布均衡)
3. 支持检查级(examination)任务的数据划分
4. 将划分结果保存到 dataset_splits 和 dataset_members 表

设计原则:
- 同一患者的所有检查都在同一fold (患者级分离)
- 按HB分级分层 (确保各等级在train/val中分布均衡)
- 支持层级多任务学习 (动作级 + 检查级)

"""

import sqlite3
from collections import defaultdict
import numpy as np
from sklearn.model_selection import StratifiedKFold

# ==================== 配置参数 ====================
DB_PATH = "facialPalsy.db"

# 交叉验证折数
N_FOLDS = 3

# 随机种子
RANDOM_SEED = 42

# 划分版本 (用于区分不同的划分方案)
SPLIT_VERSION = "v1.0"

# 是否强制重新划分
FORCE_RESPLIT = True


def load_patient_info(db_path):
    """
    加载患者信息

    Returns:
        patient_info: {
            patient_id: {
                'examinations': [exam_id1, exam_id2, ...],
                'hb_grades': [grade1, grade2, ...],
                'dominant_hb': 主要HB分级,
                'video_count': 视频数量
            }
        }
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 查询所有有标签的患者及其检查信息
    cursor.execute("""
        SELECT 
            e.patient_id,
            e.examination_id,
            el.hb_grade,
            COUNT(DISTINCT v.video_id) as video_count
        FROM examinations e
        JOIN examination_labels el ON e.examination_id = el.examination_id
        JOIN video_files v ON e.examination_id = v.examination_id
        JOIN video_features vf ON v.video_id = vf.video_id
        WHERE el.hb_grade IS NOT NULL
          AND vf.static_features IS NOT NULL
        GROUP BY e.patient_id, e.examination_id
        ORDER BY e.patient_id
    """)

    patient_info = defaultdict(lambda: {
        'examinations': [],
        'hb_grades': [],
        'dominant_hb': None,
        'video_count': 0
    })

    for row in cursor.fetchall():
        patient_id, exam_id, hb_grade, video_count = row

        patient_info[patient_id]['examinations'].append(exam_id)
        patient_info[patient_id]['hb_grades'].append(hb_grade)
        patient_info[patient_id]['video_count'] += video_count

    conn.close()

    # 确定每个患者的主导HB分级 (用于分层)
    for patient_id, info in patient_info.items():
        if info['hb_grades']:
            # 使用最严重的等级作为主导等级
            info['dominant_hb'] = max(info['hb_grades'])

    return dict(patient_info)


def create_stratified_folds(patient_info, n_folds=3, random_seed=42):
    """
    创建患者级分层K折划分

    Args:
        patient_info: 患者信息字典
        n_folds: 折数
        random_seed: 随机种子

    Returns:
        fold_assignments: {patient_id: fold_number}
    """
    patient_ids = list(patient_info.keys())
    patient_labels = [info['dominant_hb'] for info in patient_info.values()]

    print(f"\n{'='*60}")
    print(f"数据集统计")
    print(f"{'='*60}")
    print(f"患者总数: {len(patient_ids)}")

    # 统计HB分级分布
    hb_dist = defaultdict(int)
    for label in patient_labels:
        hb_dist[label] += 1

    print(f"\nHB分级分布 (患者数):")
    for grade in sorted(hb_dist.keys()):
        print(f"  Grade {grade}: {hb_dist[grade]} 患者")

    # 检查是否有足够的样本进行分层
    min_samples = min(hb_dist.values())
    if min_samples < n_folds:
        print(f"\n⚠️ 警告: 某些类别样本数({min_samples})少于折数({n_folds})")
        print("   将使用非分层划分...")
        use_stratified = False
    else:
        use_stratified = True

    # 执行划分
    fold_assignments = {}

    if use_stratified:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        patient_array = np.array(patient_ids)
        label_array = np.array(patient_labels)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(patient_array, label_array)):
            # val_idx中的患者属于这个fold的验证集
            for idx in val_idx:
                fold_assignments[patient_ids[idx]] = fold_idx
    else:
        # 简单随机划分
        np.random.seed(random_seed)
        indices = np.random.permutation(len(patient_ids))
        fold_size = len(patient_ids) // n_folds

        for fold_idx in range(n_folds):
            start_idx = fold_idx * fold_size
            if fold_idx == n_folds - 1:
                end_idx = len(patient_ids)
            else:
                end_idx = start_idx + fold_size

            for idx in range(start_idx, end_idx):
                fold_assignments[patient_ids[indices[idx]]] = fold_idx

    return fold_assignments


def print_fold_statistics(patient_info, fold_assignments, n_folds):
    """打印各fold的统计信息"""

    for fold_idx in range(n_folds):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx} 统计")
        print(f"{'='*60}")

        # 获取该fold的验证患者和训练患者
        val_patients = [p for p, f in fold_assignments.items() if f == fold_idx]
        train_patients = [p for p, f in fold_assignments.items() if f != fold_idx]

        # 统计
        train_hb = defaultdict(int)
        val_hb = defaultdict(int)
        train_videos = 0
        val_videos = 0
        train_exams = 0
        val_exams = 0

        for p in train_patients:
            info = patient_info[p]
            for hb in info['hb_grades']:
                train_hb[hb] += 1
            train_videos += info['video_count']
            train_exams += len(info['examinations'])

        for p in val_patients:
            info = patient_info[p]
            for hb in info['hb_grades']:
                val_hb[hb] += 1
            val_videos += info['video_count']
            val_exams += len(info['examinations'])

        print(f"\n训练集:")
        print(f"  患者数: {len(train_patients)}")
        print(f"  检查数: {train_exams}")
        print(f"  视频数: {train_videos}")
        print(f"  HB分布: ", end="")
        for grade in sorted(train_hb.keys()):
            print(f"G{grade}:{train_hb[grade]} ", end="")
        print()

        print(f"\n验证集:")
        print(f"  患者数: {len(val_patients)}")
        print(f"  检查数: {val_exams}")
        print(f"  视频数: {val_videos}")
        print(f"  HB分布: ", end="")
        for grade in sorted(val_hb.keys()):
            print(f"G{grade}:{val_hb[grade]} ", end="")
        print()


def save_to_database(db_path, fold_assignments, patient_info, n_folds, split_version, force_resplit=False):
    """
    保存划分结果到数据库

    表结构:
    - dataset_splits: 划分配置 (train_fold_0, val_fold_0, ...)
    - dataset_members: 患者归属 (patient_id -> split_id, fold_number)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 检查是否已存在该版本的划分
    cursor.execute("""
        SELECT COUNT(*) FROM dataset_splits WHERE split_version = ?
    """, (split_version,))
    existing = cursor.fetchone()[0]

    if existing > 0:
        if force_resplit:
            print(f"\n⚠️ 删除旧的划分 (version={split_version})...")
            # 先删除成员，再删除划分
            cursor.execute("""
                DELETE FROM dataset_members WHERE split_id IN (
                    SELECT split_id FROM dataset_splits WHERE split_version = ?
                )
            """, (split_version,))
            cursor.execute("""
                DELETE FROM dataset_splits WHERE split_version = ?
            """, (split_version,))
            conn.commit()
        else:
            print(f"\n✓ 划分已存在 (version={split_version})，跳过...")
            conn.close()
            return

    # 创建划分记录
    print(f"\n保存划分到数据库...")

    for fold_idx in range(n_folds):
        # 创建train split
        cursor.execute("""
            INSERT INTO dataset_splits (split_name, split_type, split_version, split_method, random_seed)
            VALUES (?, 'train', ?, 'patient_stratified', ?)
        """, (f'train_fold_{fold_idx}', split_version, RANDOM_SEED))
        train_split_id = cursor.lastrowid

        # 创建val split
        cursor.execute("""
            INSERT INTO dataset_splits (split_name, split_type, split_version, split_method, random_seed)
            VALUES (?, 'val', ?, 'patient_stratified', ?)
        """, (f'val_fold_{fold_idx}', split_version, RANDOM_SEED))
        val_split_id = cursor.lastrowid

        # 添加成员
        for patient_id, patient_fold in fold_assignments.items():
            if patient_fold == fold_idx:
                # 该患者属于这个fold的验证集
                cursor.execute("""
                    INSERT INTO dataset_members (split_id, patient_id, fold_number)
                    VALUES (?, ?, ?)
                """, (val_split_id, patient_id, fold_idx))
            else:
                # 该患者属于这个fold的训练集
                cursor.execute("""
                    INSERT INTO dataset_members (split_id, patient_id, fold_number)
                    VALUES (?, ?, ?)
                """, (train_split_id, patient_id, fold_idx))

    conn.commit()
    conn.close()
    print(f"✓ 划分保存完成！")


def verify_no_data_leakage(db_path, split_version):
    """验证没有数据泄漏"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print(f"\n{'='*60}")
    print("数据泄漏检查")
    print(f"{'='*60}")

    for fold_idx in range(N_FOLDS):
        # 获取训练集患者
        cursor.execute("""
            SELECT DISTINCT dm.patient_id
            FROM dataset_members dm
            JOIN dataset_splits ds ON dm.split_id = ds.split_id
            WHERE ds.split_type = 'train' AND dm.fold_number = ? AND ds.split_version = ?
        """, (fold_idx, split_version))
        train_patients = set(row[0] for row in cursor.fetchall())

        # 获取验证集患者
        cursor.execute("""
            SELECT DISTINCT dm.patient_id
            FROM dataset_members dm
            JOIN dataset_splits ds ON dm.split_id = ds.split_id
            WHERE ds.split_type = 'val' AND dm.fold_number = ? AND ds.split_version = ?
        """, (fold_idx, split_version))
        val_patients = set(row[0] for row in cursor.fetchall())

        # 检查交集
        overlap = train_patients & val_patients
        if overlap:
            print(f"⚠️ Fold {fold_idx}: 发现数据泄漏！重叠患者: {overlap}")
        else:
            print(f"✓ Fold {fold_idx}: 无数据泄漏 (训练:{len(train_patients)}患者, 验证:{len(val_patients)}患者)")

    conn.close()


def main():
    """主函数"""
    print("=" * 60)
    print("数据集划分器 V2.0 (患者级分层)")
    print("=" * 60)

    # 加载患者信息
    print("\n加载患者信息...")
    patient_info = load_patient_info(DB_PATH)

    if not patient_info:
        print("错误: 未找到有效的患者数据！")
        print("请确保:")
        print("  1. examination_labels表有HB分级数据")
        print("  2. video_features表有几何特征数据")
        return

    # 创建分层划分
    print("\n创建分层K折划分...")
    fold_assignments = create_stratified_folds(
        patient_info,
        n_folds=N_FOLDS,
        random_seed=RANDOM_SEED
    )

    # 打印统计
    print_fold_statistics(patient_info, fold_assignments, N_FOLDS)

    # 保存到数据库
    save_to_database(
        DB_PATH,
        fold_assignments,
        patient_info,
        N_FOLDS,
        SPLIT_VERSION,
        force_resplit=FORCE_RESPLIT
    )

    # 验证无数据泄漏
    verify_no_data_leakage(DB_PATH, SPLIT_VERSION)

    print(f"\n{'='*60}")
    print("划分完成！")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()