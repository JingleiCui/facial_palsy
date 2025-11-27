"""
运动热力图特征提取器 (Motion Heatmap Feature Extractor)
========================================================

功能:
1. 从视频帧序列提取面部关键点运动
2. 计算累计位移、速度、加速度等运动统计
3. 生成12维运动特征向量
4. 批量处理并存入数据库

特征维度 (12维):
    0: mean_displacement     - 平均位移
    1: max_displacement      - 最大位移
    2: std_displacement      - 位移标准差
    3: motion_energy         - 运动能量 (位移平方和)
    4: motion_asymmetry      - 运动不对称性 (左右差异)
    5: temporal_smoothness   - 时间平滑度
    6: spatial_concentration - 空间集中度 (熵)
    7: peak_ratio            - 峰值区域比例
    8-9: motion_center       - 运动重心 (归一化坐标)
    10: velocity_mean        - 平均速度
    11: acceleration_std     - 加速度变化

用法:
    python motion_feature_extractor.py [db_path]
"""

import os
import sys
import sqlite3
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

import cv2
import numpy as np

# MediaPipe
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision

    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False
    print("[ERROR] MediaPipe未安装，运动特征提取需要MediaPipe")
    sys.exit(1)

# =============================================================================
# 面部区域定义 (用于计算不对称性)
# =============================================================================

# MediaPipe 468点标准区域
FACE_REGIONS = {
    'left_eye': list(range(33, 42)) + list(range(133, 145)) + list(range(153, 160)),
    'right_eye': list(range(362, 374)) + list(range(380, 388)) + list(range(249, 256)),
    'left_brow': list(range(63, 72)) + list(range(105, 111)),
    'right_brow': list(range(293, 302)) + list(range(334, 340)),
    'nose': list(range(1, 20)) + list(range(94, 100)),
    'mouth': list(range(0, 1)) + list(range(37, 45)) + list(range(267, 275)) +
             list(range(84, 92)) + list(range(314, 322)),
    'left_cheek': list(range(117, 126)) + list(range(192, 202)),
    'right_cheek': list(range(346, 355)) + list(range(416, 426)),
}

# 左右对应区域 (用于对称性计算)
SYMMETRIC_PAIRS = [
    ('left_eye', 'right_eye'),
    ('left_brow', 'right_brow'),
    ('left_cheek', 'right_cheek'),
]


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class MotionMetrics:
    """运动量化指标"""
    mean_displacement: float = 0.0  # 平均位移
    max_displacement: float = 0.0  # 最大位移
    std_displacement: float = 0.0  # 位移标准差
    motion_energy: float = 0.0  # 运动能量
    motion_asymmetry: float = 0.0  # 运动不对称性
    temporal_smoothness: float = 0.0  # 时间平滑度
    spatial_concentration: float = 0.0  # 空间集中度
    peak_ratio: float = 0.0  # 峰值比例
    motion_center: Tuple[float, float] = (0.5, 0.5)  # 运动重心
    velocity_mean: float = 0.0  # 平均速度
    acceleration_std: float = 0.0  # 加速度变化

    # 原始数据
    displacement_per_landmark: Optional[np.ndarray] = None

    def to_feature_vector(self) -> np.ndarray:
        """
        转换为12维特征向量
        """
        features = np.zeros(12, dtype=np.float32)

        # 位移统计 (归一化)
        features[0] = min(self.mean_displacement / 50.0, 1.0)
        features[1] = min(self.max_displacement / 200.0, 1.0)
        features[2] = min(self.std_displacement / 30.0, 1.0)
        features[3] = min(self.motion_energy / 10000.0, 1.0)

        # 不对称性 (0-1, 0表示完全对称)
        features[4] = min(self.motion_asymmetry, 1.0)

        # 时间/空间特性
        features[5] = self.temporal_smoothness
        features[6] = self.spatial_concentration
        features[7] = self.peak_ratio

        # 运动重心 (归一化到0-1)
        features[8] = self.motion_center[0]
        features[9] = self.motion_center[1]

        # 速度/加速度
        features[10] = min(self.velocity_mean / 20.0, 1.0)
        features[11] = min(self.acceleration_std / 10.0, 1.0)

        return features


# =============================================================================
# 运动分析核心
# =============================================================================

class MotionAnalyzer:
    """面部运动分析器"""

    def __init__(self, landmarker_path: str):
        """
        初始化

        Args:
            landmarker_path: MediaPipe FaceLandmarker模型路径
        """
        if not Path(landmarker_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {landmarker_path}")

        base_options = mp_tasks.BaseOptions(model_asset_path=landmarker_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        print(f"[OK] FaceLandmarker已加载")

    def detect_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        检测单帧的面部关键点

        Args:
            frame: BGR图像

        Returns:
            landmarks: (468, 2) 或 None
        """
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.landmarker.detect(mp_image)

            if result.face_landmarks:
                landmarks = result.face_landmarks[0]
                h, w = frame.shape[:2]

                coords = np.array(
                    [[lm.x * w, lm.y * h] for lm in landmarks],
                    dtype=np.float32
                )
                return coords

            return None

        except Exception as e:
            print(f"[WARN] 关键点检测失败: {e}")
            return None

    def extract_frames(
            self,
            video_path: str,
            start_frame: int = 0,
            duration_sec: float = 3.0,
            fps: float = 30.0,
            sample_interval: int = 1
    ) -> List[np.ndarray]:
        """
        从视频提取帧

        Args:
            video_path: 视频路径
            start_frame: 起始帧
            duration_sec: 持续时间
            fps: 帧率
            sample_interval: 采样间隔

        Returns:
            帧列表
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        # 设置起始帧
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # 获取实际帧率
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        if actual_fps > 0:
            fps = actual_fps

        total_frames = int(duration_sec * fps)

        frames = []
        frame_count = 0

        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % sample_interval == 0:
                frames.append(frame)

            frame_count += 1

        cap.release()

        return frames

    def calculate_displacement(
            self,
            landmarks_list: List[np.ndarray]
    ) -> np.ndarray:
        """
        计算每个关键点的累计位移

        Args:
            landmarks_list: 关键点序列 [(N, 2), ...]

        Returns:
            displacement: (N,) 累计位移
        """
        if len(landmarks_list) < 2:
            return np.zeros(468, dtype=np.float32)

        n_landmarks = landmarks_list[0].shape[0]
        displacement = np.zeros(n_landmarks, dtype=np.float32)

        for i in range(len(landmarks_list) - 1):
            curr = landmarks_list[i]
            next_lm = landmarks_list[i + 1]

            # 确保维度一致
            n = min(curr.shape[0], next_lm.shape[0], n_landmarks)

            # 计算帧间位移
            diff = np.linalg.norm(next_lm[:n] - curr[:n], axis=1)
            displacement[:n] += diff

        return displacement

    def calculate_velocity(
            self,
            landmarks_list: List[np.ndarray],
            fps: float = 30.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算速度和加速度

        Returns:
            velocities: (frames-1, n_landmarks) 速度
            accelerations: (frames-2, n_landmarks) 加速度
        """
        if len(landmarks_list) < 2:
            return np.array([]), np.array([])

        n_landmarks = landmarks_list[0].shape[0]
        dt = 1.0 / fps

        # 速度
        velocities = []
        for i in range(len(landmarks_list) - 1):
            curr = landmarks_list[i]
            next_lm = landmarks_list[i + 1]
            n = min(curr.shape[0], next_lm.shape[0])

            v = np.linalg.norm(next_lm[:n] - curr[:n], axis=1) / dt
            velocities.append(v)

        velocities = np.array(velocities)

        # 加速度
        if len(velocities) < 2:
            return velocities, np.array([])

        accelerations = np.diff(velocities, axis=0) / dt

        return velocities, accelerations

    def calculate_asymmetry(
            self,
            displacement: np.ndarray
    ) -> float:
        """
        计算运动不对称性 (左右差异)

        Args:
            displacement: (N,) 每个关键点的位移

        Returns:
            asymmetry: 0-1, 0表示完全对称
        """
        asymmetries = []

        for left_region, right_region in SYMMETRIC_PAIRS:
            left_indices = FACE_REGIONS.get(left_region, [])
            right_indices = FACE_REGIONS.get(right_region, [])

            # 确保索引在范围内
            left_indices = [i for i in left_indices if i < len(displacement)]
            right_indices = [i for i in right_indices if i < len(displacement)]

            if not left_indices or not right_indices:
                continue

            left_motion = np.mean(displacement[left_indices])
            right_motion = np.mean(displacement[right_indices])

            # 相对差异
            total = left_motion + right_motion
            if total > 0:
                diff = abs(left_motion - right_motion) / total
                asymmetries.append(diff)

        return np.mean(asymmetries) if asymmetries else 0.0

    def calculate_motion_center(
            self,
            displacement: np.ndarray,
            landmarks: np.ndarray
    ) -> Tuple[float, float]:
        """
        计算运动重心 (位移加权坐标)

        Returns:
            (cx, cy) 归一化坐标
        """
        if len(displacement) == 0 or len(landmarks) == 0:
            return (0.5, 0.5)

        n = min(len(displacement), len(landmarks))

        total_displacement = displacement[:n].sum()
        if total_displacement < 1e-6:
            return (0.5, 0.5)

        weights = displacement[:n]
        coords = landmarks[:n]

        cx = np.sum(weights * coords[:, 0]) / total_displacement
        cy = np.sum(weights * coords[:, 1]) / total_displacement

        # 获取边界框进行归一化
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

        if x_max > x_min:
            cx = (cx - x_min) / (x_max - x_min)
        else:
            cx = 0.5

        if y_max > y_min:
            cy = (cy - y_min) / (y_max - y_min)
        else:
            cy = 0.5

        return (float(np.clip(cx, 0, 1)), float(np.clip(cy, 0, 1)))

    def calculate_spatial_entropy(self, displacement: np.ndarray) -> float:
        """
        计算空间熵 (运动集中度)

        Returns:
            entropy: 0-1, 0表示运动集中在少数点
        """
        if displacement.sum() < 1e-6:
            return 0.0

        # 归一化为概率分布
        probs = displacement / displacement.sum()
        probs = probs[probs > 0]  # 去除零

        # 计算熵
        entropy = -np.sum(probs * np.log2(probs))

        # 归一化 (最大熵为 log2(n))
        max_entropy = np.log2(len(displacement))
        if max_entropy > 0:
            entropy = entropy / max_entropy

        return float(entropy)

    def analyze_video(
            self,
            video_path: str,
            start_frame: int = 0,
            duration_sec: float = 3.0,
            fps: float = 30.0
    ) -> MotionMetrics:
        """
        分析视频的面部运动

        Args:
            video_path: 视频路径
            start_frame: 起始帧
            duration_sec: 分析时长
            fps: 帧率

        Returns:
            MotionMetrics对象
        """
        metrics = MotionMetrics()

        try:
            # 1. 提取帧
            frames = self.extract_frames(
                video_path, start_frame, duration_sec, fps,
                sample_interval=2  # 每2帧采样一次
            )

            if len(frames) < 3:
                print(f"[WARN] 帧数不足: {len(frames)}")
                return metrics

            # 2. 检测关键点
            landmarks_list = []
            for frame in frames:
                lm = self.detect_landmarks(frame)
                if lm is not None:
                    landmarks_list.append(lm)

            if len(landmarks_list) < 3:
                print(f"[WARN] 有效关键点帧数不足: {len(landmarks_list)}")
                return metrics

            # 3. 计算位移
            displacement = self.calculate_displacement(landmarks_list)
            metrics.displacement_per_landmark = displacement

            # 4. 基础统计
            metrics.mean_displacement = float(np.mean(displacement))
            metrics.max_displacement = float(np.max(displacement))
            metrics.std_displacement = float(np.std(displacement))
            metrics.motion_energy = float(np.sum(displacement ** 2))

            # 5. 计算速度和加速度
            velocities, accelerations = self.calculate_velocity(landmarks_list, fps)
            if len(velocities) > 0:
                metrics.velocity_mean = float(np.mean(velocities))
            if len(accelerations) > 0:
                metrics.acceleration_std = float(np.std(accelerations))

            # 6. 时间平滑度 (相邻帧位移的一致性)
            if len(landmarks_list) > 2:
                frame_disps = []
                for i in range(len(landmarks_list) - 1):
                    d = np.linalg.norm(landmarks_list[i + 1] - landmarks_list[i], axis=1).mean()
                    frame_disps.append(d)

                if len(frame_disps) > 1:
                    # 平滑度 = 1 - 变异系数
                    mean_d = np.mean(frame_disps)
                    std_d = np.std(frame_disps)
                    if mean_d > 0:
                        cv = std_d / mean_d
                        metrics.temporal_smoothness = float(1.0 / (1.0 + cv))

            # 7. 空间集中度
            metrics.spatial_concentration = self.calculate_spatial_entropy(displacement)

            # 8. 运动不对称性
            metrics.motion_asymmetry = self.calculate_asymmetry(displacement)

            # 9. 峰值比例 (高运动区域占比)
            threshold = np.percentile(displacement, 75)
            metrics.peak_ratio = float(np.sum(displacement > threshold) / len(displacement))

            # 10. 运动重心
            metrics.motion_center = self.calculate_motion_center(
                displacement, landmarks_list[0]
            )

        except Exception as e:
            print(f"[ERROR] 视频分析失败: {e}")

        return metrics


# =============================================================================
# 数据库批处理
# =============================================================================

class MotionFeatureExtractor:
    """运动特征批量提取器"""

    def __init__(
            self,
            db_path: str,
            landmarker_path: str,
            batch_size: int = 16,
            duration_sec: float = 3.0
    ):
        self.db_path = db_path
        self.batch_size = batch_size
        self.duration_sec = duration_sec
        self.analyzer = MotionAnalyzer(landmarker_path)

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
                SELECT vf.video_id, vf.file_path, vf.start_frame, vf.fps
                FROM video_files vf
                WHERE vf.file_exists = 1
            """
        else:
            query = """
                SELECT vf.video_id, vf.file_path, vf.start_frame, vf.fps
                FROM video_files vf
                LEFT JOIN video_features vfeat ON vf.video_id = vfeat.video_id
                WHERE vf.file_exists = 1
                  AND (vfeat.motion_features IS NULL OR vfeat.video_id IS NULL)
            """

        cursor.execute(query)
        samples = cursor.fetchall()
        conn.close()

        return samples

    def ensure_video_features_row(self, cursor, video_id: int):
        """确保video_features表中存在对应行"""
        cursor.execute(
            "SELECT video_id FROM video_features WHERE video_id = ?",
            (video_id,)
        )
        if cursor.fetchone() is None:
            cursor.execute(
                "INSERT INTO video_features (video_id, peak_frame_idx) VALUES (?, 0)",
                (video_id,)
            )

    def process_batch(self, force_reprocess: bool = False):
        """批量处理"""
        samples = self.get_pending_samples(force_reprocess)
        self.stats['total'] = len(samples)

        if not samples:
            print("[MotionExtractor] 没有待处理的样本")
            return

        print(f"[MotionExtractor] 开始处理 {len(samples)} 个样本")
        print(f"  分析时长: {self.duration_sec}秒")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for i, (video_id, file_path, start_frame, fps) in enumerate(samples):
            try:
                # 检查视频是否存在
                if not file_path or not Path(file_path).exists():
                    self.stats['skipped'] += 1
                    continue

                # 默认值
                start_frame = start_frame or 0
                fps = fps or 30.0

                # 分析视频
                metrics = self.analyzer.analyze_video(
                    file_path,
                    start_frame=start_frame,
                    duration_sec=self.duration_sec,
                    fps=fps
                )

                # 转换为特征向量
                features = metrics.to_feature_vector()

                # 确保行存在
                self.ensure_video_features_row(cursor, video_id)

                # 保存到数据库
                cursor.execute("""
                    UPDATE video_features
                    SET motion_features = ?,
                        motion_dim = ?,
                        motion_processed_at = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE video_id = ?
                """, (
                    features.astype(np.float32).tobytes(),
                    len(features),
                    video_id
                ))

                self.stats['success'] += 1

                # 定期提交和打印进度
                if (i + 1) % self.batch_size == 0:
                    conn.commit()
                    print(f"  [{i + 1}/{len(samples)}] 已处理 "
                          f"(成功:{self.stats['success']}, 失败:{self.stats['failed']})")

            except Exception as e:
                self.stats['failed'] += 1
                print(f"  [ERROR] video_id={video_id}: {e}")

        conn.commit()
        conn.close()

        self._print_summary()

    def _print_summary(self):
        """打印处理汇总"""
        print("\n" + "=" * 50)
        print("运动特征提取完成")
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
        print(f"[ERROR] 模型文件不存在: {landmarker_path}")
        print("请修改 landmarker_path 变量指向正确的模型路径")
        sys.exit(1)

    print("=" * 60)
    print("运动热力图特征提取器")
    print("=" * 60)
    print(f"数据库: {db_path}")
    print(f"模型: {landmarker_path}")
    print("=" * 60)

    extractor = MotionFeatureExtractor(
        db_path=db_path,
        landmarker_path=landmarker_path,
        batch_size=16,
        duration_sec=3.0
    )

    extractor.process_batch(force_reprocess=False)


if __name__ == '__main__':
    main()