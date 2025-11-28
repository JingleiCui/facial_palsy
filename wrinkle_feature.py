"""
皱纹特征提取器 (Wrinkle Feature Extractor)
==========================================

功能:
1. 从峰值帧图像提取皱纹特征
2. 量化皱纹的密度、深度、长度、方向分布等
3. 生成10维统计特征向量
4. 批量处理并存入数据库

特征维度 (10维):
    0: density          - 皱纹密度 (%)
    1: depth_mean       - 平均深度 (0-255 归一化)
    2: depth_std        - 深度标准差
    3: total_length     - 总长度 (归一化)
    4: count            - 皱纹数量 (归一化)
    5: area_ratio       - 面积占比
    6: severity_score   - 综合严重程度评分 (0-100)
    7: direction_entropy - 方向熵 (方向多样性)
    8-9: dominant_dirs  - 主要方向 (sin/cos编码)
"""

import os
import sys
import sqlite3
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import cv2
import numpy as np
from scipy import ndimage
from skimage import morphology, measure
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

HAS_MEDIAPIPE = True

# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class WrinkleMetrics:
    """皱纹量化指标"""
    density: float = 0.0  # 密度 (%)
    depth_mean: float = 0.0  # 平均深度
    depth_std: float = 0.0  # 深度标准差
    total_length: float = 0.0  # 总长度 (pixels)
    count: int = 0  # 数量
    area: int = 0  # 皱纹总面积
    roi_area: int = 0  # ROI总面积
    severity_score: float = 0.0  # 综合评分 (0-100)
    direction_hist: Optional[np.ndarray] = None  # 方向直方图
    intensity_map: Optional[np.ndarray] = None  # 强度图

    def to_feature_vector(self) -> np.ndarray:
        """
        转换为10维特征向量
        """
        features = np.zeros(10, dtype=np.float32)

        # 基础统计 (归一化)
        features[0] = min(self.density / 10.0, 1.0)  # 密度归一化到0-1
        features[1] = self.depth_mean / 255.0
        features[2] = self.depth_std / 128.0
        features[3] = min(self.total_length / 10000.0, 1.0)  # 长度归一化
        features[4] = min(self.count / 50.0, 1.0)  # 数量归一化
        features[5] = min(self.area / max(self.roi_area, 1), 1.0)  # 面积比
        features[6] = self.severity_score / 100.0

        # 方向特征
        if self.direction_hist is not None and len(self.direction_hist) > 0:
            # 计算方向熵
            hist = self.direction_hist + 1e-10
            hist = hist / hist.sum()
            entropy = -np.sum(hist * np.log2(hist))
            features[7] = entropy / np.log2(len(hist))  # 归一化熵

            # 主要方向 (sin/cos编码)
            dominant_idx = np.argmax(self.direction_hist)
            dominant_angle = (dominant_idx / len(self.direction_hist)) * np.pi
            features[8] = np.sin(dominant_angle)
            features[9] = np.cos(dominant_angle)

        return features


# =============================================================================
# 皱纹检测核心
# =============================================================================

class WrinkleDetector:
    """皱纹检测与量化"""

    def __init__(self, landmarker_path: Optional[str] = None):
        """
        初始化

        Args:
            landmarker_path: MediaPipe FaceLandmarker模型路径
        """
        self.landmarker = None
        self.segmenter = None

        if HAS_MEDIAPIPE and landmarker_path and Path(landmarker_path).exists():
            try:
                base_options = mp_tasks.BaseOptions(model_asset_path=landmarker_path)
                options = mp_vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                    num_faces=1
                )
                self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
                print(f"[OK] FaceLandmarker已加载: {landmarker_path}")
            except Exception as e:
                print(f"[WARN] FaceLandmarker加载失败: {e}")

    def get_skin_mask(self, image: np.ndarray) -> np.ndarray:
        """
        获取皮肤区域掩码 (排除眼睛、嘴巴等)

        Args:
            image: BGR图像

        Returns:
            皮肤掩码 (0/255)
        """
        h, w = image.shape[:2]

        # 方法1: 使用MediaPipe面部检测 + 颜色过滤
        if self.landmarker is not None:
            try:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
                result = self.landmarker.detect(mp_image)

                if result.face_landmarks:
                    landmarks = result.face_landmarks[0]

                    # 创建面部轮廓掩码
                    face_points = []
                    face_contour_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454,
                                            323, 361, 288, 397, 365, 379, 378, 400, 377,
                                            152, 148, 176, 149, 150, 136, 172, 58, 132,
                                            93, 234, 127, 162, 21, 54, 103, 67, 109]

                    for idx in face_contour_indices:
                        if idx < len(landmarks):
                            lm = landmarks[idx]
                            face_points.append((int(lm.x * w), int(lm.y * h)))

                    if face_points:
                        face_mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.fillPoly(face_mask, [np.array(face_points)], 255)

                        # 排除眼睛、嘴巴区域
                        exclusion_mask = np.zeros((h, w), dtype=np.uint8)

                        # 左眼
                        left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133,
                                            173, 157, 158, 159, 160, 161, 246]
                        left_eye_points = [(int(landmarks[i].x * w), int(landmarks[i].y * h))
                                           for i in left_eye_indices if i < len(landmarks)]
                        if left_eye_points:
                            cv2.fillPoly(exclusion_mask, [np.array(left_eye_points)], 255)

                        # 右眼
                        right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263,
                                             466, 388, 387, 386, 385, 384, 398]
                        right_eye_points = [(int(landmarks[i].x * w), int(landmarks[i].y * h))
                                            for i in right_eye_indices if i < len(landmarks)]
                        if right_eye_points:
                            cv2.fillPoly(exclusion_mask, [np.array(right_eye_points)], 255)

                        # 嘴巴
                        mouth_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                                         291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
                        mouth_points = [(int(landmarks[i].x * w), int(landmarks[i].y * h))
                                        for i in mouth_indices if i < len(landmarks)]
                        if mouth_points:
                            cv2.fillPoly(exclusion_mask, [np.array(mouth_points)], 255)

                        # 膨胀排除区域
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                        exclusion_mask = cv2.dilate(exclusion_mask, kernel)

                        # 最终皮肤掩码
                        skin_mask = cv2.bitwise_and(face_mask, cv2.bitwise_not(exclusion_mask))
                        return skin_mask

            except Exception as e:
                print(f"[WARN] MediaPipe处理失败: {e}")

        # 方法2: 基于颜色的简单皮肤检测
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

        # HSV范围
        lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
        upper_hsv = np.array([20, 255, 255], dtype=np.uint8)
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

        # YCrCb范围
        lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
        upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

        # 组合
        skin_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)

        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

        return skin_mask

    def detect_wrinkles(
            self,
            image: np.ndarray,
            mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        检测皱纹

        Args:
            image: BGR图像 或 灰度图像
            mask: 可选的ROI掩码

        Returns:
            heatmap: 皱纹热力图 (BGR)
            wrinkle_mask: 皱纹二值掩码
            intensity_map: 皱纹强度图
        """
        # 转灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        h, w = gray.shape

        # 1. CLAHE增强
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # 2. 双边滤波 (保边去噪)
        filtered = cv2.bilateralFilter(enhanced, d=9, sigmaColor=100, sigmaSpace=100)

        # 3. Black Hat变换 (检测细小暗线)
        kernel_fine = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        bh_fine = cv2.morphologyEx(filtered, cv2.MORPH_BLACKHAT, kernel_fine)

        kernel_wide = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
        bh_wide = cv2.morphologyEx(filtered, cv2.MORPH_BLACKHAT, kernel_wide)

        # 组合不同尺度
        combined = cv2.addWeighted(bh_fine, 0.6, bh_wide, 0.4, 0)

        # 4. 应用掩码
        if mask is not None:
            combined = cv2.bitwise_and(combined, combined, mask=mask)

        # 5. 归一化
        intensity_map = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX)

        # 6. 自适应阈值
        _, binary = cv2.threshold(intensity_map, 20, 255, cv2.THRESH_BINARY)

        # 7. 几何过滤 (去除孤立点/毛孔)
        connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary_connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, connect_kernel)

        contours, _ = cv2.findContours(
            binary_connected.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        clean_mask = np.zeros_like(binary, dtype=np.uint8)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 10:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            aspect_ratio = float(max(bw, bh)) / (min(bw, bh) + 1e-5)

            # 皱纹判定: 面积足够大 或 细长形状
            is_wrinkle = False
            if area > 300:
                is_wrinkle = True
            elif area > 20 and aspect_ratio > 2.5:
                is_wrinkle = True

            if is_wrinkle:
                cv2.drawContours(clean_mask, [cnt], -1, 255, -1)

        # 最终强度图
        intensity_map = cv2.bitwise_and(
            intensity_map.astype(np.uint8),
            intensity_map.astype(np.uint8),
            mask=clean_mask
        )

        # 热力图
        heatmap = cv2.applyColorMap(intensity_map, cv2.COLORMAP_JET)
        if mask is not None:
            heatmap = cv2.bitwise_and(heatmap, heatmap, mask=mask)

        return heatmap, clean_mask, intensity_map

    def calculate_metrics(
            self,
            wrinkle_mask: np.ndarray,
            intensity_map: np.ndarray,
            roi_mask: Optional[np.ndarray] = None
    ) -> WrinkleMetrics:
        """
        计算皱纹量化指标

        Args:
            wrinkle_mask: 皱纹二值掩码
            intensity_map: 皱纹强度图
            roi_mask: ROI掩码

        Returns:
            WrinkleMetrics对象
        """
        metrics = WrinkleMetrics()

        # ROI面积
        if roi_mask is not None:
            metrics.roi_area = np.count_nonzero(roi_mask)
        else:
            metrics.roi_area = wrinkle_mask.shape[0] * wrinkle_mask.shape[1]

        # 皱纹面积
        metrics.area = np.count_nonzero(wrinkle_mask)

        if metrics.roi_area == 0 or metrics.area == 0:
            metrics.intensity_map = intensity_map
            return metrics

        # 密度
        metrics.density = (metrics.area / metrics.roi_area) * 100

        # 深度统计
        wrinkle_pixels = intensity_map[wrinkle_mask > 0]
        if len(wrinkle_pixels) > 0:
            metrics.depth_mean = np.mean(wrinkle_pixels)
            metrics.depth_std = np.std(wrinkle_pixels)

        # 皱纹数量
        labeled_mask = measure.label(wrinkle_mask, connectivity=2)
        metrics.count = np.max(labeled_mask) if labeled_mask.max() > 0 else 0

        # 总长度 (骨架化)
        skeleton = morphology.skeletonize(wrinkle_mask > 0)
        metrics.total_length = np.count_nonzero(skeleton)

        # 方向分布
        metrics.direction_hist = self._calculate_orientation(wrinkle_mask)

        # 综合评分
        density_score = min(metrics.density * 2, 30)
        depth_score = min((metrics.depth_mean / 255) * 40, 40)
        count_score = min(metrics.count * 2, 30)
        metrics.severity_score = density_score + depth_score + count_score

        metrics.intensity_map = intensity_map

        return metrics

    def _calculate_orientation(self, mask: np.ndarray, num_bins: int = 8) -> np.ndarray:
        """计算皱纹方向分布"""
        if np.count_nonzero(mask) == 0:
            return np.zeros(num_bins)

        # Sobel梯度
        sobelx = cv2.Sobel(mask.astype(float), cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(mask.astype(float), cv2.CV_64F, 0, 1, ksize=3)

        # 角度 (0-180度)
        angles = np.arctan2(sobely, sobelx)
        angles = np.degrees(angles) % 180

        wrinkle_angles = angles[mask > 0]
        if len(wrinkle_angles) == 0:
            return np.zeros(num_bins)

        # 直方图
        hist, _ = np.histogram(wrinkle_angles, bins=num_bins, range=(0, 180))
        hist = hist.astype(float)

        if hist.sum() > 0:
            hist = hist / hist.sum()

        return hist

    def extract_features(self, image: np.ndarray) -> Tuple[np.ndarray, WrinkleMetrics]:
        """
        完整的特征提取流程

        Args:
            image: BGR图像

        Returns:
            features: 10维特征向量
            metrics: 完整的量化指标
        """
        # 1. 获取皮肤掩码
        skin_mask = self.get_skin_mask(image)

        # 2. 检测皱纹
        heatmap, wrinkle_mask, intensity_map = self.detect_wrinkles(image, skin_mask)

        # 3. 计算指标
        metrics = self.calculate_metrics(wrinkle_mask, intensity_map, skin_mask)

        # 4. 转换为特征向量
        features = metrics.to_feature_vector()

        return features, metrics


# =============================================================================
# 数据库批处理
# =============================================================================

class WrinkleFeatureExtractor:
    """皱纹特征批量提取器"""

    def __init__(
            self,
            db_path: str,
            landmarker_path: Optional[str] = None,
            batch_size: int = 32
    ):
        self.db_path = db_path
        self.batch_size = batch_size
        self.detector = WrinkleDetector(landmarker_path)

        # 统计
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'skipped': 0
        }

    def get_pending_samples(self, force_reprocess: bool = False) -> List[Tuple]:
        """获取待处理样本"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if force_reprocess:
            query = """
                SELECT vf.video_id, vfeat.peak_frame_path, vfeat.peak_frame_segmented_path
                FROM video_files vf
                INNER JOIN video_features vfeat ON vf.video_id = vfeat.video_id
                WHERE vfeat.peak_frame_path IS NOT NULL
                   OR vfeat.peak_frame_segmented_path IS NOT NULL
            """
        else:
            query = """
                SELECT vf.video_id, vfeat.peak_frame_path, vfeat.peak_frame_segmented_path
                FROM video_files vf
                INNER JOIN video_features vfeat ON vf.video_id = vfeat.video_id
                WHERE vfeat.wrinkle_features IS NULL
                  AND (vfeat.peak_frame_path IS NOT NULL 
                       OR vfeat.peak_frame_segmented_path IS NOT NULL)
            """

        cursor.execute(query)
        samples = cursor.fetchall()
        conn.close()

        return samples

    def process_batch(self, force_reprocess: bool = False):
        """批量处理"""
        samples = self.get_pending_samples(force_reprocess)
        self.stats['total'] = len(samples)

        if not samples:
            print("[WrinkleExtractor] 没有待处理的样本")
            return

        print(f"[WrinkleExtractor] 开始处理 {len(samples)} 个样本")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for i, (video_id, peak_path, seg_path) in enumerate(samples):
            try:
                # 优先使用分割后的图像
                img_path = seg_path if seg_path and Path(seg_path).exists() else peak_path

                if not img_path or not Path(img_path).exists():
                    self.stats['skipped'] += 1
                    continue

                # 读取图像
                image = cv2.imread(img_path)
                if image is None:
                    self.stats['failed'] += 1
                    continue

                # 提取特征
                features, metrics = self.detector.extract_features(image)

                # 保存到数据库
                cursor.execute("""
                    UPDATE video_features
                    SET wrinkle_features = ?,
                        wrinkle_dim = ?,
                        wrinkle_processed_at = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE video_id = ?
                """, (
                    features.astype(np.float32).tobytes(),
                    len(features),
                    video_id
                ))

                self.stats['success'] += 1

                # 定期提交
                if (i + 1) % self.batch_size == 0:
                    conn.commit()
                    print(f"  [{i + 1}/{len(samples)}] 已处理")

            except Exception as e:
                self.stats['failed'] += 1
                print(f"  [ERROR] video_id={video_id}: {e}")

        conn.commit()
        conn.close()

        self._print_summary()

    def _print_summary(self):
        """打印处理汇总"""
        print("\n" + "=" * 50)
        print("皱纹特征提取完成")
        print("=" * 50)
        print(f"  总数:   {self.stats['total']}")
        print(f"  成功:   {self.stats['success']}")
        print(f"  失败:   {self.stats['failed']}")
        print(f"  跳过:   {self.stats['skipped']}")
        print("=" * 50)


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    db_path = 'facialPalsy.db'

    # MediaPipe模型路径 (根据实际情况修改)
    landmarker_path = "/Users/cuijinglei/PycharmProjects/medicalProject/models/face_landmarker.task"

    if not Path(landmarker_path).exists():
        landmarker_path = None
        print("[INFO] 未找到FaceLandmarker模型，使用简化模式")

    print("=" * 60)
    print("皱纹特征提取器")
    print("=" * 60)
    print(f"数据库: {db_path}")
    print(f"模型: {landmarker_path or '简化模式'}")
    print("=" * 60)

    extractor = WrinkleFeatureExtractor(
        db_path=db_path,
        landmarker_path=landmarker_path,
        batch_size=32
    )

    extractor.process_batch(force_reprocess=False)


if __name__ == '__main__':
    main()