"""
人脸分割模块 - Face Segmentation Module
用于从峰值帧图像中分割出人脸区域,去除背景噪声
"""
import os
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import Optional, Tuple
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed

def _worker_segment_one(args):
    img_path, out_path, model_path = args
    seg = FaceSegmenter(model_path=model_path)  # 每个进程一个实例
    try:
        seg.process_peak_frame(Path(img_path), Path(out_path))
        return (img_path, True, "")
    except Exception as e:
        return (img_path, False, str(e))
    finally:
        del seg

def to_segmented_path(peak_path: str) -> str:
    """
    根据原始峰值帧路径构造分割后图像路径

    规则:
    /.../pipeline/keyframes/ActionName/xxx.jpg
      → /.../pipeline/keyframes_segmented/ActionName/xxx_segmented.jpg
    """
    raw = Path(peak_path)
    # /keyframes/ → /keyframes_segmented/
    seg_dir = str(raw).replace("/keyframes/", "/keyframes_segmented/")
    # 文件名加 _segmented.jpg
    seg = Path(seg_dir).with_name(raw.stem + "_segmented.jpg")
    return str(seg)

def update_segmented_paths(db_path: str):
    """
    批量更新 video_features 表中的 peak_frame_segmented_path 字段
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 读取所有已有 peak_frame_path 的记录
    cur.execute("SELECT video_id, peak_frame_path FROM video_features")
    rows = cur.fetchall()

    updated = 0

    for video_id, peak_path in rows:
        # 避免 peak_frame_path 为空
        if not peak_path:
            continue

        seg_path = to_segmented_path(peak_path)
        cur.execute("""
            UPDATE video_features
            SET peak_frame_segmented_path = ?
            WHERE video_id = ?
        """, (seg_path, video_id))
        updated += 1

    conn.commit()
    conn.close()

    print(f"[FaceSegmenter] 已更新 {updated} 条 peak_frame_segmented_path")


class FaceSegmenter:
    """
    人脸分割器 - 使用MediaPipe进行语义分割

    功能:
    1. 从图像中分割出人脸皮肤区域
    2. 去除背景、头发、衣服等无关区域
    3. 提高视觉特征提取的准确性

    使用的模型:
    - MediaPipe Selfie Multiclass Segmentation (256x256)
    - 类别3: face-skin (人脸皮肤)
    """

    def __init__(self, model_path: str):
        """
        初始化人脸分割器

        Args:
            model_path: MediaPipe分割模型路径 (selfie_multiclass_256x256.tflite)
        """
        self.model_path = model_path

        # 配置MediaPipe选项
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            output_category_mask=True  # 必须开启以获取分类mask
        )

        # 创建分割器
        self.segmenter = vision.ImageSegmenter.create_from_options(options)

        print(f"[FaceSegmenter] 初始化成功 - 模型: {model_path}")

    def segment_face(
            self,
            image: np.ndarray,
            background_color: Tuple[int, int, int] = (0, 0, 0)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从图像中分割人脸区域

        Args:
            image: 输入图像 (BGR格式, numpy array)
            background_color: 背景填充颜色 (B, G, R), 默认黑色

        Returns:
            segmented_image: 分割后的图像 (只保留人脸区域)
            face_mask: 二值mask (0/1)
        """
        if image is None or image.size == 0:
            raise ValueError("输入图像为空")

        # 转换为RGB (MediaPipe需要)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 创建MediaPipe Image对象
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # 执行分割
        result = self.segmenter.segment(mp_image)

        # 获取分类mask (H x W, uint8)
        category_mask = result.category_mask.numpy_view()

        # 提取人脸皮肤区域 (类别3)
        face_mask = (category_mask == 3).astype(np.uint8)

        # 应用mask到原图
        segmented_image = self._apply_mask(image, face_mask, background_color)

        return segmented_image, face_mask

    def _apply_mask(
            self,
            image: np.ndarray,
            mask: np.ndarray,
            background_color: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        将mask应用到图像上

        Args:
            image: 原始图像
            mask: 二值mask (0/1)
            background_color: 背景颜色

        Returns:
            处理后的图像
        """
        # 创建背景
        background = np.full_like(image, background_color)

        # 扩展mask到3通道
        mask_3c = np.stack([mask] * 3, axis=-1)

        # 合成: 前景(人脸) + 背景
        result = image * mask_3c + background * (1 - mask_3c)

        return result.astype(np.uint8)

    def process_peak_frame(
            self,
            peak_frame_path: Path,
            output_path: Optional[Path] = None,
            save_mask: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理峰值帧图像

        Args:
            peak_frame_path: 峰值帧图像路径
            output_path: 输出路径 (可选)
            save_mask: 是否保存mask

        Returns:
            segmented_image: 分割后的图像
            face_mask: 人脸mask
        """
        # 读取图像
        image = cv2.imread(str(peak_frame_path))
        if image is None:
            raise ValueError(f"无法读取图像: {peak_frame_path}")

        # 分割
        segmented_image, face_mask = self.segment_face(image)

        # 保存结果
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), segmented_image)
            print(f"[FaceSegmenter] 保存分割结果: {output_path}")

            # 保存mask
            if save_mask:
                mask_path = output_path.parent / f"{output_path.stem}_mask.jpg"
                cv2.imwrite(str(mask_path), face_mask * 255)

        return segmented_image, face_mask

    def batch_process_keyframes(self, keyframes_dir: Path, output_dir: Path, pattern="*.jpg"):
        keyframes_dir = Path(keyframes_dir)
        output_dir = Path(output_dir)

        tasks = []
        for action_dir in keyframes_dir.iterdir():
            if not action_dir.is_dir():
                continue
            action_output_dir = output_dir / action_dir.name
            action_output_dir.mkdir(parents=True, exist_ok=True)

            for img_path in action_dir.glob(pattern):
                out_path = action_output_dir / f"{img_path.stem}_segmented.jpg"
                tasks.append((str(img_path), str(out_path), self.model_path))

        # 6 个进程
        max_workers = 6

        ok = 0
        fail = 0
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_worker_segment_one, t) for t in tasks]
            for fut in as_completed(futures):
                img_path, success, err = fut.result()
                if success:
                    ok += 1
                else:
                    fail += 1
                    print(f"[ERROR] 处理失败 {img_path}: {err}")

        print(f"\n[FaceSegmenter] 批量处理完成 - 成功 {ok} 张 / 失败 {fail} 张 / 总计 {len(tasks)} 张")

    def __del__(self):
        """析构函数 - 释放资源"""
        if hasattr(self, 'segmenter'):
            self.segmenter.close()


def main():
    """
    测试脚本 - 批量处理keyframes + 更新数据库中的 segmented 路径
    """
    # 配置路径
    KEYFRAMES_DIR = Path("/Users/cuijinglei/Documents/facial_palsy/HGFA/keyframes")
    OUTPUT_DIR = Path("/Users/cuijinglei/Documents/facial_palsy/HGFA/keyframes_segmented")
    MODEL_PATH = "/Users/cuijinglei/PycharmProjects/medicalProject/models/selfie_multiclass_256x256.tflite"
    DB_PATH = "facialPalsy.db"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 创建分割器
    segmenter = FaceSegmenter(model_path=MODEL_PATH)

    # 批量处理
    segmenter.batch_process_keyframes(
        keyframes_dir=KEYFRAMES_DIR,
        output_dir=OUTPUT_DIR,
        pattern="*.jpg"
    )
    # 分割完成后，批量更新 video_features 表中的 peak_frame_segmented_path
    update_segmented_paths(DB_PATH)

    print("\n 人脸分割完成!")


if __name__ == "__main__":
    main()