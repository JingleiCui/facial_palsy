# visual_feature_extractor_optimized.py
# -*- coding: utf-8 -*-
"""
视觉特征提取器 - 优化版
Visual Feature Extractor - Optimized with True Batch Processing

关键优化:
1. 真正的批处理: 多张图像同时送入 MobileNetV3
2. 预加载数据: 减少数据库查询次数
3. 批量数据库写入: 减少 I/O
4. MPS 优化: 充分利用 Apple Silicon 的 GPU
5. 内存管理: 及时释放不需要的数据
"""

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import sqlite3
from tqdm import tqdm
import time


class VisualFeatureExtractorOptimized:
    """
    优化版视觉特征提取器

    主要改进:
    1. 真正的批处理（batch_size=32）
    2. 预加载所有待处理样本信息
    3. 批量数据库写入
    4. 内存优化
    """

    def __init__(
            self,
            device: Optional[str] = None,
            use_pretrained: bool = True,
            batch_size: int = 32
    ):
        """
        初始化视觉特征提取器

        Args:
            device: 计算设备 ('mps', 'cpu')
            use_pretrained: 是否使用 ImageNet 预训练权重
            batch_size: 批处理大小
        """
        # 设置设备
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        self.batch_size = batch_size
        print(f"[VisualFeatureExtractor-OPT] 使用设备: {self.device}, 批大小: {batch_size}")

        # 加载 MobileNetV3-Large
        if use_pretrained:
            weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
            self.model = mobilenet_v3_large(weights=weights)
        else:
            self.model = mobilenet_v3_large(weights=None)

        # 修改分类头
        original_classifier = self.model.classifier
        self.model.classifier = nn.Sequential(
            original_classifier[0],  # Linear(960, 1280)
            original_classifier[1],  # Hardswish
            original_classifier[2]  # Dropout(p=0.2)
        )

        # 移动到设备
        self.model = self.model.to(self.device)
        self.model.eval()

        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print(f"[VisualFeatureExtractor-OPT] MobileNetV3-Large 加载成功")

    def extract_batch(
            self,
            image_paths: List[Path]
    ) -> List[np.ndarray]:
        """
        批量提取特征（真正的批处理）

        Args:
            image_paths: 图像路径列表

        Returns:
            features_list: 特征列表，每个元素是 1280 维向量
        """
        if not image_paths:
            return []

        # 加载并预处理所有图像
        tensors = []
        valid_indices = []

        for idx, img_path in enumerate(image_paths):
            try:
                pil_image = Image.open(img_path).convert('RGB')
                tensor = self.transform(pil_image)
                tensors.append(tensor)
                valid_indices.append(idx)
            except Exception as e:
                print(f"\n[ERROR] 无法加载图像 {img_path}: {e}")
                continue

        if not tensors:
            return [None] * len(image_paths)

        # 堆叠成批次
        batch_tensor = torch.stack(tensors).to(self.device)

        # 批量推理
        with torch.no_grad():
            features_batch = self.model(batch_tensor)

        # 转换为 numpy
        features_np = features_batch.cpu().numpy()

        # 构建返回列表（保持原始顺序，失败的位置为 None）
        results = [None] * len(image_paths)
        for i, idx in enumerate(valid_indices):
            results[idx] = features_np[i]

        return results

    def batch_extract_from_database_optimized(
            self,
            db_path: str,
            force_reprocess: bool = False,
            db_batch_size: int = 200
    ):
        """
        优化版批量提取

        关键改进:
        1. 预先加载所有待处理样本
        2. 按 batch_size 分批推理
        3. 按 db_batch_size 批量写入数据库

        Args:
            db_path: 数据库路径
            force_reprocess: 是否强制重新处理
            db_batch_size: 数据库批量写入大小
        """
        start_time = time.time()

        conn = sqlite3.connect(db_path)

        # 优化数据库设置
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA cache_size = -64000;")  # 64MB 缓存

        cursor = conn.cursor()

        # ========== 阶段1: 预加载所有待处理样本 ==========
        print("\n[OPT] 阶段1: 加载待处理样本...")

        if force_reprocess:
            where_cond = "vfeat.peak_frame_segmented_path IS NOT NULL"
        else:
            where_cond = """
                vfeat.peak_frame_segmented_path IS NOT NULL
                AND vfeat.visual_features IS NULL
            """

        query = f"""
            SELECT 
                vfeat.feature_id,
                vfeat.video_id,
                vf.examination_id,
                at.action_name_en,
                vfeat.augmentation_type,
                vfeat.peak_frame_segmented_path
            FROM video_features vfeat
            INNER JOIN video_files vf   ON vfeat.video_id = vf.video_id
            INNER JOIN action_types at  ON vf.action_id    = at.action_id
            WHERE {where_cond}
            ORDER BY vfeat.feature_id
        """

        cursor.execute(query)
        samples = cursor.fetchall()

        if not samples:
            print("\n✅ 所有样本的视觉特征已提取完成")
            conn.close()
            return

        total_samples = len(samples)
        print(f"[OPT] 找到 {total_samples} 个待处理样本")

        # ========== 阶段2: 验证图像路径 ==========
        print("\n[OPT] 阶段2: 验证图像路径...")

        valid_samples = []
        for sample in tqdm(samples, desc="验证路径"):
            feature_id, video_id, exam_id, action_name, aug_type, seg_path = sample

            if not seg_path:
                print(f"\n[WARN] feature_id={feature_id} 的 seg_path 为空")
                continue

            image_path = Path(seg_path)
            if not image_path.exists():
                print(f"\n[WARN] 图像不存在: {image_path}")
                continue

            valid_samples.append((feature_id, video_id, image_path))

        valid_count = len(valid_samples)
        print(f"[OPT] 有效样本: {valid_count}/{total_samples}")

        if valid_count == 0:
            conn.close()
            return

        # ========== 阶段3: 批量推理 ==========
        print(f"\n[OPT] 阶段3: 批量特征提取（batch_size={self.batch_size}）...")

        all_features = []

        # 按 batch_size 分批处理
        for i in tqdm(range(0, valid_count, self.batch_size), desc="特征提取"):
            batch_samples = valid_samples[i:i + self.batch_size]

            # 提取这一批的图像路径
            batch_paths = [sample[2] for sample in batch_samples]

            # 批量推理
            batch_features = self.extract_batch(batch_paths)

            # 保存结果（feature_id, features）
            for j, (feature_id, video_id, _) in enumerate(batch_samples):
                if batch_features[j] is not None:
                    all_features.append((feature_id, batch_features[j]))

        print(f"[OPT] 成功提取 {len(all_features)} 个特征")

        # ========== 阶段4: 批量写入数据库 ==========
        print(f"\n[OPT] 阶段4: 批量写入数据库（db_batch_size={db_batch_size}）...")

        saved_count = 0

        # 按 db_batch_size 分批写入
        for i in tqdm(range(0, len(all_features), db_batch_size), desc="写入数据库"):
            batch_to_save = all_features[i:i + db_batch_size]

            # 准备批量插入数据
            batch_data = []
            for feature_id, features in batch_to_save:
                features_blob = features.astype(np.float32).tobytes()
                batch_data.append((features_blob, 1280, feature_id))

            # 批量更新
            cursor.executemany("""
                UPDATE video_features
                SET visual_features     = ?,
                    visual_dim          = ?,
                    visual_processed_at = CURRENT_TIMESTAMP,
                    updated_at          = CURRENT_TIMESTAMP
                WHERE feature_id = ?
            """, batch_data)

            conn.commit()
            saved_count += len(batch_data)

        conn.close()

        # ========== 统计信息 ==========
        elapsed = time.time() - start_time

        print("\n" + "=" * 60)
        print("视觉特征提取完成 (优化版)")
        print("=" * 60)
        print(f"总样本数: {total_samples}")
        print(f"有效样本: {valid_count}")
        print(f"成功提取: {len(all_features)}")
        print(f"已保存: {saved_count}")
        print(f"总耗时: {elapsed:.2f} 秒 ({elapsed / 60:.2f} 分钟)")
        print(f"平均速度: {saved_count / elapsed:.2f} 样本/秒")
        print("=" * 60)


def load_visual_features_from_db(
        db_path: str,
        video_ids: Optional[List[int]] = None,
        augmentation_types: Optional[List[str]] = None
) -> Dict[Tuple[int, str], np.ndarray]:
    """
    从数据库加载视觉特征（优化版）

    改进: 返回 {(video_id, aug_type): features} 字典，支持增强类型

    Args:
        db_path: 数据库路径
        video_ids: 视频ID列表(None表示加载全部)
        augmentation_types: 增强类型列表(None表示全部)

    Returns:
        features_dict: {(video_id, aug_type): features_1280d}
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 构建查询条件
    where_clauses = ["visual_features IS NOT NULL"]
    params = []

    if video_ids is not None:
        placeholders = ','.join('?' * len(video_ids))
        where_clauses.append(f"video_id IN ({placeholders})")
        params.extend(video_ids)

    if augmentation_types is not None:
        placeholders = ','.join('?' * len(augmentation_types))
        where_clauses.append(f"augmentation_type IN ({placeholders})")
        params.extend(augmentation_types)

    where_str = " AND ".join(where_clauses)

    query = f"""
        SELECT video_id, augmentation_type, visual_features
        FROM video_features
        WHERE {where_str}
    """

    cursor.execute(query, params)
    results = cursor.fetchall()
    conn.close()

    features_dict = {}
    for video_id, aug_type, blob in results:
        features = np.frombuffer(blob, dtype=np.float32)
        if features.shape[0] == 1280:
            features_dict[(video_id, aug_type)] = features
        else:
            print(f"[WARN] video_id={video_id}, aug={aug_type} 特征维度错误: {features.shape}")

    return features_dict


def main():
    """
    主函数 - 批量提取视觉特征（优化版）
    """
    db_path = 'facialPalsy.db'

    device = None  # None → 自动检测 mps/cpu
    batch_size = 32  # 推理批大小（根据显存调整，M3 Max 可以用 64）
    db_batch_size = 200  # 数据库批量写入大小
    force_reprocess = False

    print("=" * 60)
    print("视觉特征提取 - 优化版 V3.0")
    print("=" * 60)
    print(f"\n配置:")
    print(f"  数据库: {db_path}")
    print(f"  设备: {device or 'auto'}")
    print(f"  推理批大小: {batch_size}")
    print(f"  数据库批大小: {db_batch_size}")
    print(f"  强制重处理: {force_reprocess}")

    extractor = VisualFeatureExtractorOptimized(
        device=device,
        batch_size=batch_size
    )

    extractor.batch_extract_from_database_optimized(
        db_path=db_path,
        force_reprocess=force_reprocess,
        db_batch_size=db_batch_size
    )

    print("\n✅ 处理完成!")


if __name__ == "__main__":
    main()