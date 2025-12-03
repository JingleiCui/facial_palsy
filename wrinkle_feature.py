# wrinkle_feature.py
# -*- coding: utf-8 -*-
"""
皱纹特征提取器 - 纯热力图版
=========================================
功能：
1. 读取已分割的人脸图像（黑色背景）
2. 使用 算法 (CLAHE+BlackHat) 检测皱纹
3. 输出纯皱纹热力图 (黑色背景，不叠加原图)
4. 批量写入数据库与本地文件

输出图片：黑色背景，仅保留彩色皱纹线条 (Jet Colormap)
"""

import sqlite3
import time
import os
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from functools import partial

import cv2
import numpy as np
from skimage import morphology, measure
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision


# =============================================================================
# 1. 数据结构 (保持数据库兼容性)
# =============================================================================

@dataclass
class WrinkleMetrics:
    """皱纹量化指标"""
    density: float = 0.0  # 密度
    depth_mean: float = 0.0  # 平均深度
    depth_std: float = 0.0  # 深度标准差
    total_length: float = 0.0  # 总长度
    count: int = 0  # 数量
    area: int = 0  # 皱纹像素面积
    roi_area: int = 0  # 感兴趣区域面积
    severity_score: float = 0.0  # 综合评分
    direction_hist: Optional[np.ndarray] = None  # 方向直方图

    def to_feature_vector(self) -> np.ndarray:
        """转换为10维特征向量"""
        features = np.zeros(10, dtype=np.float32)
        features[0] = min(self.density / 10.0, 1.0)
        features[1] = self.depth_mean / 255.0
        features[2] = self.depth_std / 128.0
        features[3] = min(self.total_length / 10000.0, 1.0)
        features[4] = min(self.count / 50.0, 1.0)
        features[5] = min(self.area / max(self.roi_area, 1), 1.0)
        features[6] = self.severity_score / 100.0

        if self.direction_hist is not None and len(self.direction_hist) > 0:
            hist = self.direction_hist + 1e-10
            hist = hist / hist.sum()
            entropy = -np.sum(hist * np.log2(hist))
            features[7] = entropy / np.log2(len(hist))
            dominant_idx = np.argmax(self.direction_hist)
            dominant_angle = (dominant_idx / len(self.direction_hist)) * np.pi
            features[8] = np.sin(dominant_angle)
            features[9] = np.cos(dominant_angle)

        return features


# =============================================================================
# 2. 核心算法类 (Logic)
# =============================================================================

class WrinkleDetector:
    """基于算法的检测器"""

    def __init__(self, landmarker_path: str):
        self.landmarker = None
        if landmarker_path and Path(landmarker_path).exists():
            try:
                base_options = mp_tasks.BaseOptions(model_asset_path=landmarker_path)
                options = mp_vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                    num_faces=1
                )
                self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            except Exception as e:
                print(f"[WARN] FaceLandmarker 加载失败: {e}")

    def get_analysis_mask(self, image: np.ndarray) -> np.ndarray:
        """获取分析区域 (排除黑色背景和五官)"""
        h, w = image.shape[:2]

        # 1. 基础掩码：非黑色背景区域
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, base_mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        base_mask = cv2.erode(base_mask, kernel_erode, iterations=2)

        # 2. 五官排除
        if self.landmarker:
            try:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                result = self.landmarker.detect(mp_image)

                if result.face_landmarks:
                    landmarks = result.face_landmarks[0]
                    exclusion_mask = np.zeros((h, w), dtype=np.uint8)

                    # 定义需要排除的区域 (眼、嘴、眉)
                    features_indices = [
                        [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],  # Left Eye
                        [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],  # Right Eye
                        [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185],
                        # Lips
                        [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],  # Left Eyebrow
                        [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]  # Right Eyebrow
                    ]

                    for indices in features_indices:
                        points = []
                        for idx in indices:
                            if idx < len(landmarks):
                                points.append((int(landmarks[idx].x * w), int(landmarks[idx].y * h)))
                        if points:
                            hull = cv2.convexHull(np.array(points))
                            cv2.fillPoly(exclusion_mask, [hull], 255)

                    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                    exclusion_mask = cv2.dilate(exclusion_mask, kernel_dilate)
                    return cv2.bitwise_and(base_mask, cv2.bitwise_not(exclusion_mask))
            except Exception:
                pass
        return base_mask

    def detect_wrinkles(self, image: np.ndarray, mask: np.ndarray):
        """核心检测算法"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. 增强 & 滤波
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        filtered = cv2.bilateralFilter(enhanced, d=9, sigmaColor=75, sigmaSpace=75)

        # 2. BlackHat (多尺度)
        kernel_fine = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        bh_fine = cv2.morphologyEx(filtered, cv2.MORPH_BLACKHAT, kernel_fine)
        kernel_wide = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
        bh_wide = cv2.morphologyEx(filtered, cv2.MORPH_BLACKHAT, kernel_wide)
        combined_bh = cv2.addWeighted(bh_fine, 0.6, bh_wide, 0.4, 0)

        # 3. Mask & Threshold
        valid_region = cv2.bitwise_and(combined_bh, combined_bh, mask=mask)
        norm_img = cv2.normalize(valid_region, None, 0, 255, cv2.NORM_MINMAX)
        _, binary = cv2.threshold(norm_img, 25, 255, cv2.THRESH_BINARY)

        # 4. 几何过滤 (Geometry Filter)
        connect_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, connect_kernel)

        contours, _ = cv2.findContours(binary_connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_mask = np.zeros_like(binary)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 15: continue
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(max(w, h)) / (min(w, h) + 1e-5)

            if area > 400 or (area > 20 and aspect_ratio > 2.0):
                cv2.drawContours(clean_mask, [cnt], -1, 255, -1)

        # 5. 生成纯热力图 (Pure Heatmap)
        # 提取保留区域的强度
        intensity_map = cv2.bitwise_and(norm_img, norm_img, mask=clean_mask)

        # 应用伪彩色 (Jet)
        heatmap_color = cv2.applyColorMap(intensity_map.astype(np.uint8), cv2.COLORMAP_JET)

        # 关键：使用 mask 确保非皱纹区域完全为黑色
        heatmap_pure = cv2.bitwise_and(heatmap_color, heatmap_color, mask=clean_mask)

        return heatmap_pure, clean_mask, intensity_map

    def calculate_metrics(self, wrinkle_mask: np.ndarray, intensity_map: np.ndarray,
                          roi_mask: np.ndarray) -> WrinkleMetrics:
        """计算指标"""
        metrics = WrinkleMetrics()
        metrics.roi_area = int(np.count_nonzero(roi_mask))
        metrics.area = int(np.count_nonzero(wrinkle_mask))

        if metrics.roi_area > 0:
            metrics.density = (metrics.area / metrics.roi_area) * 100.0

        if metrics.area > 0:
            wrinkle_pixels = intensity_map[wrinkle_mask > 0]
            metrics.depth_mean = float(np.mean(wrinkle_pixels))
            metrics.depth_std = float(np.std(wrinkle_pixels))

            # Skeleton & Direction
            skeleton = morphology.skeletonize(wrinkle_mask > 0)
            metrics.total_length = float(np.count_nonzero(skeleton))

            sobelx = cv2.Sobel(wrinkle_mask.astype(float), cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(wrinkle_mask.astype(float), cv2.CV_64F, 0, 1, ksize=3)
            angles_deg = np.degrees(np.arctan2(sobely, sobelx)) % 180
            valid_angles = angles_deg[wrinkle_mask > 0]
            hist, _ = np.histogram(valid_angles, bins=18, range=(0, 180))
            metrics.direction_hist = hist

        labeled_mask = measure.label(wrinkle_mask, connectivity=2)
        metrics.count = int(np.max(labeled_mask)) if labeled_mask.max() > 0 else 0

        # Score
        metrics.severity_score = min(metrics.density * 2.5, 40) + \
                                 min((metrics.depth_mean / 255.0) * 50, 30) + \
                                 min(metrics.count / 2, 30)
        return metrics


# =============================================================================
# 3. 进程处理函数 (修改了保存逻辑)
# =============================================================================

def process_single_sample(
        sample_data: Tuple,
        landmarker_path: str,
        wrinkle_root: Path
) -> Optional[Tuple]:
    """单样本处理函数"""
    feature_id, video_id, action_name, peak_path, seg_path = sample_data

    try:
        detector = WrinkleDetector(landmarker_path)

        # 优先使用分割图
        img_path = None
        if seg_path and Path(seg_path).exists():
            img_path = Path(seg_path)
        elif peak_path and Path(peak_path).exists():
            img_path = Path(peak_path)

        if img_path is None: return None
        image = cv2.imread(str(img_path))
        if image is None: return None

        # 检测流程
        roi_mask = detector.get_analysis_mask(image)
        heatmap_pure, wrinkle_mask, intensity_map = detector.detect_wrinkles(image, roi_mask)
        metrics = detector.calculate_metrics(wrinkle_mask, intensity_map, roi_mask)
        features = metrics.to_feature_vector()

        # === 修改点：直接保存纯热力图 ===
        action_dir = wrinkle_root / action_name
        action_dir.mkdir(parents=True, exist_ok=True)

        # 文件名增加 _heatmap 后缀
        save_path = action_dir / f"{img_path.stem}_wrinkle_heatmap.jpg"

        # 直接写入 heatmap_pure (黑色背景+彩色线条)，不叠加原图
        cv2.imwrite(str(save_path), heatmap_pure, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # 准备 Blob
        _, heatmap_encoded = cv2.imencode(".jpg", heatmap_pure, [cv2.IMWRITE_JPEG_QUALITY, 95])
        heatmap_blob = heatmap_encoded.tobytes()

        _, mask_encoded = cv2.imencode(".jpg", wrinkle_mask, [cv2.IMWRITE_JPEG_QUALITY, 95])
        mask_blob = mask_encoded.tobytes()

        features_blob = features.astype(np.float32).tobytes()

        return (feature_id, features_blob, heatmap_blob, mask_blob)

    except Exception as e:
        print(f"[ERROR] ID={feature_id}: {e}")
        return None


# =============================================================================
# 4. 批处理器
# =============================================================================

class WrinkleBatchProcessor:
    def __init__(self, db_path: str, wrinkle_root: Path, landmarker_path: str, num_workers: int = 4):
        self.db_path = db_path
        self.wrinkle_root = wrinkle_root
        self.landmarker_path = landmarker_path
        self.num_workers = num_workers

    def run(self, force_reprocess: bool = False):
        print(f"[INFO] 开始批处理 (纯热力图模式)...")
        start_time = time.time()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cond = "1=1" if force_reprocess else "vfeat.wrinkle_features IS NULL"
        query = f"""
            SELECT vfeat.feature_id, vfeat.video_id, at.action_name_en, 
                   vfeat.peak_frame_path, vfeat.peak_frame_segmented_path
            FROM video_features vfeat
            JOIN video_files vf ON vfeat.video_id = vf.video_id
            JOIN action_types at ON vf.action_id = at.action_id
            WHERE {cond}
              AND (vfeat.peak_frame_segmented_path IS NOT NULL OR vfeat.peak_frame_path IS NOT NULL)
        """

        cursor.execute(query)
        rows = cursor.fetchall()
        total = len(rows)
        conn.close()

        if total == 0:
            print("[INFO] 无任务。")
            return

        print(f"[INFO] 任务数: {total}")

        worker_func = partial(
            process_single_sample,
            landmarker_path=self.landmarker_path,
            wrinkle_root=self.wrinkle_root
        )

        results = []
        with Pool(self.num_workers) as pool:
            for res in tqdm(pool.imap_unordered(worker_func, rows), total=total):
                if res: results.append(res)

        if results:
            self._save_to_db(results)

        print(f"[SUCCESS] 完成! 耗时: {time.time() - start_time:.1f}s")

    def _save_to_db(self, results: List[Tuple]):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        batch_data = [(r[1], 10, r[2], r[3], r[0]) for r in results]  # feats, dim, heatmap, mask, id

        cursor.executemany("""
            UPDATE video_features
            SET wrinkle_features = ?, wrinkle_dim = ?, wrinkle_heatmap = ?, wrinkle_mask = ?,
                wrinkle_processed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
            WHERE feature_id = ?
        """, batch_data)
        conn.commit()
        conn.close()


# =============================================================================
# Main
# =============================================================================

def main():
    # 请根据实际情况修改路径
    DB_PATH = 'facialPalsy.db'
    WRINKLE_OUTPUT_ROOT = Path("/Users/cuijinglei/Documents/facialPalsy/HGFA/keyframes_wrinkles")
    MODEL_PATH = "/Users/cuijinglei/PycharmProjects/medicalProject/models/face_landmarker.task"

    processor = WrinkleBatchProcessor(
        db_path=DB_PATH,
        wrinkle_root=WRINKLE_OUTPUT_ROOT,
        landmarker_path=MODEL_PATH,
        num_workers=max(1, cpu_count() - 2)
    )

    processor.run(force_reprocess=True)


if __name__ == '__main__':
    main()