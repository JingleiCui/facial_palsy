"""
视觉特征提取器
Visual Feature Extractor - Batch Processing

功能:
1. 批量处理所有视频的峰值帧
2. 使用分割后的人脸图像
3. 提取1280维MobileNetV3特征
4. 保存到数据库video_features表
"""

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import cv2
import sqlite3
from tqdm import tqdm
from datetime import datetime


class VisualFeatureExtractor:
    """
    视觉特征提取器

    使用预训练的MobileNetV3-Large提取峰值帧的视觉特征
    """

    def __init__(
        self,
        device: Optional[str] = None,
        use_pretrained: bool = True
    ):
        """
        初始化视觉特征提取器

        Args:
            device: 计算设备 ('mps', 'cpu')
            use_pretrained: 是否使用ImageNet预训练权重
        """
        # 设置设备
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        print(f"[VisualFeatureExtractor] 使用设备: {self.device}")

        # 加载MobileNetV3-Large
        if use_pretrained:
            weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
            self.model = mobilenet_v3_large(weights=weights)
        else:
            self.model = mobilenet_v3_large(weights=None)

        # 修改分类头: 保留960→1280的投影,去掉最后的1280→1000分类层
        # 原始: Linear(960, 1280) -> Hardswish -> Dropout -> Linear(1280, 1000)
        # 修改: Linear(960, 1280) -> Hardswish -> Dropout (停在这里)
        original_classifier = self.model.classifier
        self.model.classifier = nn.Sequential(
            original_classifier[0],  # Linear(960, 1280)
            original_classifier[1],  # Hardswish
            original_classifier[2]   # Dropout(p=0.2)
        )

        # 移动到设备
        self.model = self.model.to(self.device)
        self.model.eval()

        # 定义图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print(f"[VisualFeatureExtractor] MobileNetV3-Large 加载成功")

    def extract_from_image(
        self,
        image_path: Path
    ) -> np.ndarray:
        """
        从图像文件提取视觉特征

        Args:
            image_path: 图像路径

        Returns:
            features: 1280维特征向量
        """
        # 读取图像
        pil_image = Image.open(image_path).convert('RGB')

        # 预处理
        input_tensor = self.transform(pil_image)
        input_batch = input_tensor.unsqueeze(0).to(self.device)

        # 提取特征
        with torch.no_grad():
            features = self.model(input_batch)

        # 转换为numpy
        features_np = features.cpu().numpy().squeeze()

        return features_np

    def batch_extract_from_database(
        self,
        db_path: str,
        segmented_root: Path,
        batch_size: int = 32,
        force_reprocess: bool = False
    ):
        """
        从数据库批量提取所有视频的视觉特征

        Args:
            db_path: 数据库路径
            segmented_root: 分割图像根目录（现在仅用于日志显示，路径以DB为准）
            batch_size: 批处理大小(用于显示,实际逐个处理)
            force_reprocess: 是否强制重新处理已处理的样本
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 获取需要处理的样本
        if force_reprocess:
            query = """
                SELECT 
                    vf.video_id,
                    vf.examination_id,
                    at.action_name_en,
                    vfeat.peak_frame_segmented_path
                FROM video_files vf
                INNER JOIN action_types at ON vf.action_id = at.action_id
                INNER JOIN video_features vfeat ON vf.video_id = vfeat.video_id
                WHERE vfeat.peak_frame_segmented_path IS NOT NULL
                ORDER BY vf.video_id
            """
        else:
            query = """
                SELECT 
                    vf.video_id,
                    vf.examination_id,
                    at.action_name_en,
                    vfeat.peak_frame_segmented_path
                FROM video_files vf
                INNER JOIN action_types at ON vf.action_id = at.action_id
                INNER JOIN video_features vfeat ON vf.video_id = vfeat.video_id
                WHERE vfeat.peak_frame_segmented_path IS NOT NULL
                  AND vfeat.visual_features IS NULL
                ORDER BY vf.video_id
            """

        cursor.execute(query)
        samples = cursor.fetchall()

        if not samples:
            print("\n✅ 所有样本的视觉特征已提取完成")
            conn.close()
            return

        print(f"\n找到 {len(samples)} 个需要提取视觉特征的样本")
        print(f"分割图像根目录: {segmented_root}")

        # 统计
        success_count = 0
        fail_count = 0
        skip_count = 0

        # 逐个处理
        for video_id, exam_id, action_name, seg_path in tqdm(samples, desc="提取视觉特征"):
            try:
                if not seg_path:
                    print(f"\n[ERROR] video_id={video_id} 的 peak_frame_segmented_path 为空")
                    fail_count += 1
                    continue

                image_path = Path(seg_path)

                if not image_path.exists():
                    print(f"\n[ERROR] 分割图像不存在: {image_path}")
                    fail_count += 1
                    continue

                # 提取特征
                features = self.extract_from_image(image_path)

                # 验证维度
                if features.shape[0] != 1280:
                    print(f"\n[ERROR] 特征维度错误: {features.shape}, 预期(1280,)")
                    fail_count += 1
                    continue

                # 保存到数据库
                self._save_to_database(
                    cursor=cursor,
                    video_id=video_id,
                    features=features
                )

                success_count += 1

                # 定期提交
                if success_count % 10 == 0:
                    conn.commit()

            except Exception as e:
                print(f"\n[ERROR] 处理失败 video_id={video_id}: {e}")
                fail_count += 1

        # 最终提交
        conn.commit()
        conn.close()

        # 统计报告
        print("\n" + "=" * 60)
        print("视觉特征提取完成")
        print("=" * 60)
        print(f"成功: {success_count}")
        print(f"失败: {fail_count}")
        print(f"总计: {len(samples)}")
        print("=" * 60)

    def _save_to_database(
        self,
        cursor,
        video_id: int,
        features: np.ndarray
    ):
        """
        保存特征到数据库

        Args:
            cursor: 数据库游标
            video_id: 视频ID
            features: 视觉特征
        """

        # 更新数据库
        cursor.execute("""
            UPDATE video_features
            SET visual_features = ?,
                visual_dim = ?,
                visual_processed_at = CURRENT_TIMESTAMP
            WHERE video_id = ?
        """, (
            features.astype(np.float32).tobytes(),
            len(features),
            video_id
        ))


def load_visual_features_from_db(
    db_path: str,
    video_ids: Optional[List[int]] = None
) -> dict:
    """
    从数据库加载视觉特征

    Args:
        db_path: 数据库路径
        video_ids: 视频ID列表(None表示加载全部)

    Returns:
        features_dict: {video_id: features_1280d}
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if video_ids is None:
        query = """
            SELECT video_id, visual_features
            FROM video_features
            WHERE visual_features IS NOT NULL
        """
        cursor.execute(query)
    else:
        placeholders = ','.join('?' * len(video_ids))
        query = f"""
            SELECT video_id, visual_features
            FROM video_features
            WHERE video_id IN ({placeholders})
              AND visual_features IS NOT NULL
        """
        cursor.execute(query, video_ids)

    results = cursor.fetchall()
    conn.close()

    # 解析BLOB
    features_dict = {}
    for video_id, blob in results:
        features = np.frombuffer(blob, dtype=np.float32)
        if features.shape[0] == 1280:
            features_dict[video_id] = features
        else:
            print(f"[WARN] video_id={video_id} 特征维度错误: {features.shape}")

    return features_dict


def main():
    """
    主函数 - 批量提取视觉特征
    """
    db_path = 'facialPalsy.db'
    segmented_root = Path('/Users/cuijinglei/Documents/facialPalsy/HGFA/keyframes_segmented')

    # device:
    #   None  → 自动检测（优先 mps, 否则 cpu）
    #   'mps' → 强制走 Apple GPU（如果不可用会报错）
    #   'cpu' → 强制走 CPU
    device = None

    # 是否强制重新处理已提取过视觉特征的样本
    force_reprocess = False

    print("=" * 60)
    print("视觉特征提取 V2.0")
    print("=" * 60)
    print(f"\n配置:")
    print(f"  数据库: {db_path}")
    print(f"  分割图像: {segmented_root}")
    print(f"  设备: {device or 'auto'}")
    print(f"  强制重处理: {force_reprocess}")

    # 初始化提取器
    extractor = VisualFeatureExtractor(device=device)

    # 批量处理
    extractor.batch_extract_from_database(
        db_path=db_path,
        segmented_root=segmented_root,
        force_reprocess=force_reprocess
    )


print("\n 处理完成!")


if __name__ == "__main__":
    main()