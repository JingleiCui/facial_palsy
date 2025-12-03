#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集划分脚本
Patient-level Stratified 3-Fold Cross-Validation

功能：
1. 读取数据库中的所有样本
2. 按患者级别进行分层3折交叉验证
3. 确保同一患者的所有样本都在同一fold中（防止数据泄漏）
4. 按严重程度等级(severity_score)进行分层
5. 将划分结果保存到 dataset_splits 和 dataset_members 表

直接在PyCharm中点击运行即可！可重复运行。
"""

import sqlite3
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold

# ==================== 配置参数 ====================
DB_PATH = "facialPalsy.db"
N_FOLDS = 3
RANDOM_SEED = 42
SPLIT_VERSION = "v1.0"

# 是否强制重新划分（会删除旧的划分）
FORCE_RESPLIT = True


def load_dataset_info(db_path):
    """从数据库加载数据集信息"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 查询所有样本及其标签
    cursor.execute("""
        SELECT 
            vf.feature_id,
            v.video_id,
            e.patient_id,
            e.examination_id,
            al.severity_score,
            at.action_name_en
        FROM video_features vf
        JOIN video_files v ON vf.video_id = v.video_id
        JOIN examinations e ON v.examination_id = e.examination_id
        JOIN action_labels al ON e.examination_id = al.examination_id 
                              AND v.action_id = al.action_id
        JOIN action_types at ON v.action_id = at.action_id
        WHERE vf.fused_action_features IS NOT NULL
          AND al.severity_score IS NOT NULL
        ORDER BY e.patient_id, v.video_id
    """)

    samples = []
    patient_info = defaultdict(lambda: {
        'samples': [],
        'severity_counts': defaultdict(int),
        'dominant_severity': None
    })

    for row in cursor.fetchall():
        feature_id, video_id, patient_id, exam_id, severity, action_name = row

        sample = {
            'feature_id': feature_id,
            'video_id': video_id,
            'patient_id': patient_id,
            'examination_id': exam_id,
            'severity_score': severity,
            'action_name': action_name
        }

        samples.append(sample)
        patient_info[patient_id]['samples'].append(sample)
        patient_info[patient_id]['severity_counts'][severity] += 1

    # 确定每个患者的主导严重程度
    for patient_id, info in patient_info.items():
        severity_counts = info['severity_counts']
        dominant_severity = max(severity_counts.items(), key=lambda x: x[1])[0]
        info['dominant_severity'] = dominant_severity

    conn.close()
    return samples, dict(patient_info)


def create_fold_splits(patient_info, n_folds=3, random_seed=42):
    """创建患者级别的分层交叉验证划分"""
    patient_ids = list(patient_info.keys())
    patient_labels = [info['dominant_severity'] for info in patient_info.values()]

    print(f"\n患者数量: {len(patient_ids)}")
    print(f"严重程度分布:")
    severity_dist = defaultdict(int)
    for label in patient_labels:
        severity_dist[label] += 1
    for severity in sorted(severity_dist.keys()):
        print(f"  Grade {severity}: {severity_dist[severity]} 患者")

    # 分层K折交叉验证
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    fold_splits = []
    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(patient_ids, patient_labels)):
        train_patient_ids = [patient_ids[i] for i in train_indices]
        val_patient_ids = [patient_ids[i] for i in val_indices]

        fold_splits.append({
            'fold': fold_idx,
            'train_patients': train_patient_ids,
            'val_patients': val_patient_ids
        })

        print(f"\nFold {fold_idx}:")
        print(f"  训练患者: {len(train_patient_ids)}")
        print(f"  验证患者: {len(val_patient_ids)}")

        # 统计严重程度分布
        train_severity_dist = defaultdict(int)
        val_severity_dist = defaultdict(int)

        for pid in train_patient_ids:
            train_severity_dist[patient_info[pid]['dominant_severity']] += 1
        for pid in val_patient_ids:
            val_severity_dist[patient_info[pid]['dominant_severity']] += 1

        print(f"  训练集严重程度分布:")
        for severity in sorted(train_severity_dist.keys()):
            print(f"    Grade {severity}: {train_severity_dist[severity]} 患者")

        print(f"  验证集严重程度分布:")
        for severity in sorted(val_severity_dist.keys()):
            print(f"    Grade {severity}: {val_severity_dist[severity]} 患者")

    return fold_splits


def save_splits_to_db(db_path, samples, patient_info, fold_splits, split_version, force_resplit=False):
    """将划分结果保存到数据库"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 检查是否已存在划分
    cursor.execute("SELECT COUNT(*) FROM dataset_splits WHERE split_version = ?", (split_version,))
    existing_count = cursor.fetchone()[0]

    if existing_count > 0:
        if force_resplit:
            print(f"\n正在删除已有的 {split_version} 划分...")
            # 删除旧的划分
            cursor.execute("""
                DELETE FROM dataset_members 
                WHERE split_id IN (
                    SELECT split_id FROM dataset_splits WHERE split_version = ?
                )
            """, (split_version,))
            cursor.execute("DELETE FROM dataset_splits WHERE split_version = ?", (split_version,))
            conn.commit()
            print("✓ 已删除旧划分")
        else:
            print(f"\n✓ 数据集划分 {split_version} 已存在！")
            print("如需重新划分，请设置 FORCE_RESPLIT = True")
            conn.close()
            return

    # 创建新的划分记录
    for fold_split in fold_splits:
        fold = fold_split['fold']

        # 创建 train split
        train_split_name = f"train_fold{fold}_{split_version}"
        cursor.execute("""
            INSERT INTO dataset_splits 
            (split_name, split_type, split_version, split_method, random_seed, description)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (train_split_name, 'train', split_version, 'patient_stratified',
              RANDOM_SEED, f'Training set for fold {fold}'))
        train_split_id = cursor.lastrowid

        # 创建 val split
        val_split_name = f"val_fold{fold}_{split_version}"
        cursor.execute("""
            INSERT INTO dataset_splits 
            (split_name, split_type, split_version, split_method, random_seed, description)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (val_split_name, 'val', split_version, 'patient_stratified',
              RANDOM_SEED, f'Validation set for fold {fold}'))
        val_split_id = cursor.lastrowid

        # 添加患者到 dataset_members
        for patient_id in fold_split['train_patients']:
            cursor.execute("""
                INSERT INTO dataset_members (split_id, patient_id, fold_number)
                VALUES (?, ?, ?)
            """, (train_split_id, patient_id, fold))

        for patient_id in fold_split['val_patients']:
            cursor.execute("""
                INSERT INTO dataset_members (split_id, patient_id, fold_number)
                VALUES (?, ?, ?)
            """, (val_split_id, patient_id, fold))

    conn.commit()

    # 统计每个fold的样本数量
    print("\n" + "=" * 60)
    print("数据集划分统计:")
    print("=" * 60)

    for fold_split in fold_splits:
        fold = fold_split['fold']

        # 统计训练集样本
        train_patients = fold_split['train_patients']
        train_count = sum(len(patient_info[pid]['samples']) for pid in train_patients)

        # 统计验证集样本
        val_patients = fold_split['val_patients']
        val_count = sum(len(patient_info[pid]['samples']) for pid in val_patients)

        print(f"\nFold {fold}:")
        print(f"  训练集: {len(train_patients)} 患者, {train_count} 样本")
        print(f"  验证集: {len(val_patients)} 患者, {val_count} 样本")
        print(f"  总计: {len(train_patients) + len(val_patients)} 患者, {train_count + val_count} 样本")

    conn.close()


def process_dataset_split():
    """执行数据集划分"""
    print("=" * 60)
    print("数据集划分 - Patient-level Stratified 3-Fold CV")
    print("=" * 60)
    print(f"数据库路径: {DB_PATH}")
    print(f"交叉验证折数: {N_FOLDS}")
    print(f"随机种子: {RANDOM_SEED}")
    print(f"版本: {SPLIT_VERSION}")
    print(f"强制重新划分: {FORCE_RESPLIT}")

    # 1. 加载数据
    print("\n正在加载数据...")
    samples, patient_info = load_dataset_info(DB_PATH)

    if len(samples) == 0:
        print("❌ 错误: 数据库中没有找到带有fused_action_features的样本！")
        print("请先运行 stage3_mfa.py")
        return

    print(f"✓ 加载了 {len(samples)} 个样本，来自 {len(patient_info)} 个患者")

    # 2. 创建患者级别的分层划分
    print("\n创建患者级别的分层交叉验证划分...")
    fold_splits = create_fold_splits(
        patient_info=patient_info,
        n_folds=N_FOLDS,
        random_seed=RANDOM_SEED
    )
    print(f"✓ 创建了 {N_FOLDS} 折交叉验证划分")

    # 3. 保存到数据库
    print("\n正在保存划分结果到数据库...")
    save_splits_to_db(DB_PATH, samples, patient_info, fold_splits, SPLIT_VERSION, FORCE_RESPLIT)
    print("✓ 划分结果已保存到 dataset_splits 和 dataset_members 表")

    print("\n" + "=" * 60)
    print("✓ 数据集划分完成！")
    print("=" * 60)
    print("\n下一步: 运行 train_hgfa_net.py 开始训练")
    print("\n提示:")
    print("  - 同一患者的所有样本都在同一fold中")
    print("  - 训练集和验证集的严重程度分布已平衡")
    print("  - 可以开始训练Fold 0, Fold 1, Fold 2")


if __name__ == "__main__":
    process_dataset_split()