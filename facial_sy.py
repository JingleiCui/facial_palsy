"""
面部对称性分析 V2 - M3 Max 优化版
包含：多进程并行加速、时序动态分析、改进的可视化叠加
"""

import os
import cv2
import numpy as np
import sqlite3
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial.distance import mahalanobis
from scipy import stats
import json
import matplotlib.pyplot as plt
import matplotlib
from concurrent.futures import ProcessPoolExecutor
import time

# 适配中文字体 (MacOS/Windows兼容)
font_options = ['SimHei', 'Arial Unicode MS', 'PingFang SC', 'Heiti TC', 'Microsoft YaHei']
for font in font_options:
    try:
        matplotlib.rcParams['font.sans-serif'] = [font]
        break
    except:
        continue
matplotlib.rcParams['axes.unicode_minus'] = False

# ==================== 配置区域 ====================

SYMMETRY_INDEX_CONFIG = [
    {
        "region": "eyebrow",
        "pairs": {
            "left": [336, 296, 334, 293, 300, 276, 283, 282, 295, 285],
            "right": [107, 66, 105, 63, 70, 46, 53, 52, 65, 55],
        }
    },
    {
        "region": "eye",
        "pairs": {
            "left": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            "right": [133, 155, 154, 153, 145, 144, 163, 7, 33, 246, 161, 160, 159, 158, 157, 173],
        }
    },
    {
        "region": "pupil",
        "pairs": {"left": [473], "right": [468]}
    },
    {
        "region": "iris",
        "pairs": {"left": [474, 475, 476, 477], "right": [471, 470, 469, 472]}
    },
    {
        "region": "upper_lip",
        "pairs": {
            "left": [267, 269, 270, 409, 291, 308, 415, 310, 311, 312],
            "right": [37, 39, 40, 185, 61, 78, 191, 80, 81, 82],
        }
    },
    {
        "region": "lower_lip",
        "pairs": {
            "left": [317, 402, 318, 324, 308, 291, 375, 321, 405, 314],
            "right": [87, 178, 88, 95, 78, 61, 146, 91, 181, 84],
        }
    },
    {
        "region": "face_contour",
        "pairs": {
            "left": [338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377],
            "right": [109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148],
        },
    },
]


def build_pairs_and_names(cfg_list: list):
    pairs = []
    names = []
    region_map = {}  # map region name to index list

    current_idx = 0
    for item in cfg_list:
        region = item["region"]
        lr = item["pairs"]
        L = lr["left"]
        R = lr["right"]

        region_indices = []
        for i, (li, ri) in enumerate(zip(L, R), start=1):
            pairs.append((int(li), int(ri)))
            names.append(f"{region}_{i:02d}")
            region_indices.append(current_idx)
            current_idx += 1
        region_map[region] = region_indices

    return pairs, names, region_map


# 全局变量以便多进程调用
FEATURE_PAIRS, FEATURE_NAMES, REGION_MAP = build_pairs_and_names(SYMMETRY_INDEX_CONFIG)


@dataclass
class SymmetryFeatures:
    pearson_coefficients: np.ndarray
    landmark_names: List[str]
    y_coords_left: np.ndarray  # Shape: [frames, features]
    y_coords_right: np.ndarray  # Shape: [frames, features]
    frame_count: int


# ==================== 核心分析类 ====================

class FacialSymmetryAnalyzer:
    def __init__(self, db_path: str, model_path: str):
        self.db_path = db_path
        self.model_path = model_path
        self.feature_pairs = FEATURE_PAIRS
        self.feature_names = FEATURE_NAMES
        self.n_features = len(self.feature_pairs)

    def _create_landmarker(self):
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        return vision.FaceLandmarker.create_from_options(options)

    def extract_landmarks_from_video(self, video_path, start_frame=None, end_frame=None, fps=None):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None, None, None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = 0 if start_frame is None else max(0, int(start_frame))
        end_frame = total_frames if end_frame is None else min(total_frames, int(end_frame))

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        landmarker = self._create_landmarker()
        coords_list = []
        max_asymmetry_frame = None
        max_asymmetry_score = -1.0
        max_asymmetry_landmarks = None

        current_frame = start_frame
        processed_idx = 0

        try:
            while cap.isOpened() and current_frame < end_frame:
                ret, frame = cap.read()
                if not ret: break

                # Timestamp must be monotonically increasing
                timestamp_ms = int(processed_idx * 33.3)  # 默认30fps估算，仅用于MediaPipe内部逻辑
                processed_idx += 1

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.face_landmarks:
                    face_landmarks = result.face_landmarks[0]

                    # 提取坐标
                    curr_coords = []
                    frame_diff_sum = 0

                    for (li, ri) in self.feature_pairs:
                        l = face_landmarks[li]
                        r = face_landmarks[ri]
                        curr_coords.append([[l.x, l.y], [r.x, r.y]])
                        # 简单计算该帧的累积Y轴不对称度，用于找"最不对称帧"作为封面
                        frame_diff_sum += abs(l.y - r.y)

                    coords_list.append(curr_coords)

                    # 更新最不对称帧（用于绘图）
                    if frame_diff_sum > max_asymmetry_score:
                        max_asymmetry_score = frame_diff_sum
                        max_asymmetry_frame = frame.copy()
                        max_asymmetry_landmarks = face_landmarks

                current_frame += 1
        finally:
            cap.release()
            landmarker.close()

        if not coords_list:
            return None, None, None

        coords_array = np.array(coords_list)  # [frames, features, 2(L/R), 2(x,y)]
        left_coords = coords_array[:, :, 0, :]
        right_coords = coords_array[:, :, 1, :]

        debug_info = {
            'frame': max_asymmetry_frame,
            'landmarks': max_asymmetry_landmarks
        }

        return left_coords, right_coords, debug_info

    def calculate_metrics(self, left_coords, right_coords):
        # 1. Pearson Correlation (Linearity)
        y_left = left_coords[:, :, 1]
        y_right = right_coords[:, :, 1]

        L = y_left.astype(np.float32)
        R = y_right.astype(np.float32)

        Lm = L - L.mean(axis=0, keepdims=True)
        Rm = R - R.mean(axis=0, keepdims=True)

        num = np.sum(Lm * Rm, axis=0)
        den = np.sqrt(np.sum(Lm * Lm, axis=0) * np.sum(Rm * Rm, axis=0)) + 1e-8
        pearson_coeffs = (num / den).astype(np.float32)

        return SymmetryFeatures(
            pearson_coefficients=np.array(pearson_coeffs),
            landmark_names=self.feature_names,
            y_coords_left=y_left,
            y_coords_right=y_right,
            frame_count=len(y_left)
        )

    # ==================== 可视化功能 (改进版) ====================

    def plot_region_timeseries(self, features: SymmetryFeatures, title: str, save_path: str):
        """
        绘制改进的时序曲线：按区域展示绝对不对称度 (Abs Diff)
        """
        regions = REGION_MAP.keys()

        plt.figure(figsize=(12, 6))

        # 计算每一帧的绝对差值 |L_y - R_y|
        diffs = np.abs(features.y_coords_left - features.y_coords_right)  # [Frames, Features]

        # 按区域聚合 (Mean)
        for region in regions:
            indices = REGION_MAP[region]
            if not indices: continue
            region_diff = np.mean(diffs[:, indices], axis=1)
            # 平滑曲线 (Rolling mean)
            window = 5
            if len(region_diff) > window:
                region_diff = np.convolve(region_diff, np.ones(window) / window, mode='valid')

            plt.plot(region_diff, label=f"{region} (Avg Diff)", linewidth=1.5)

        plt.title(f"{title}\n动态不对称度 (y轴差值绝对值, 越低越好)", fontsize=12)
        plt.xlabel("帧 (Frame)", fontsize=10)
        plt.ylabel("归一化坐标差值 (|L-R|)", fontsize=10)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    def save_overlay_visualization(self, frame_bgr, face_landmarks, pearson_coeffs, save_path):
        """
        改进的叠加图：骨架连线法
        绿色连线 = 高度对称 (Pearson > 0.8)
        黄色连线 = 中度对称
        红色连线 = 不对称 (Pearson < 0.5)
        """
        if frame_bgr is None or face_landmarks is None:
            return

        h, w = frame_bgr.shape[:2]
        canvas = frame_bgr.copy()

        # 变暗背景，突出连线
        canvas = (canvas * 0.7).astype(np.uint8)

        for i, (li, ri) in enumerate(self.feature_pairs):
            score = pearson_coeffs[i]
            l_pt = face_landmarks[li]
            r_pt = face_landmarks[ri]

            p1 = (int(l_pt.x * w), int(l_pt.y * h))
            p2 = (int(r_pt.x * w), int(r_pt.y * h))

            # 颜色映射: Red (-1.0) -> Yellow -> Green (1.0)
            # 简化逻辑：>0.8 Green, <0.5 Red, else Yellow
            if score >= 0.85:
                color = (0, 255, 0)  # Green
                thickness = 1
            elif score >= 0.5:
                color = (0, 255, 255)  # Yellow
                thickness = 2
            else:
                color = (0, 0, 255)  # Red
                thickness = 3  # 加粗显示问题区域

            cv2.line(canvas, p1, p2, color, thickness, lineType=cv2.LINE_AA)
            cv2.circle(canvas, p1, 2, color, -1)
            cv2.circle(canvas, p2, 2, color, -1)

        # 添加图例
        cv2.putText(canvas, "Green: High Sym (>0.85)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(canvas, "Red: Low Sym (<0.5)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imwrite(save_path, canvas)


# ==================== 多进程 Worker ====================

def process_video_task(args):
    """独立的 Worker 函数，用于多进程池"""
    (row, db_path, model_path, output_dir) = args
    video_id, exam_id, action_id, file_path, start_frame, end_frame, fps, action_name, patient_id = row

    # 每个进程独立实例化 Analyzer
    analyzer = FacialSymmetryAnalyzer(db_path, model_path)

    try:
        # 1. 提取
        left, right, debug_info = analyzer.extract_landmarks_from_video(
            file_path, start_frame, end_frame, fps
        )
        if left is None:
            return None

        # 2. 计算
        features = analyzer.calculate_metrics(left, right)

        # 3. 可视化生成
        os.makedirs(output_dir, exist_ok=True)
        base_name = f"{patient_id}_{action_name}_{video_id}"

        # 图1: 热图 (原有)
        # analyzer.visualize_symmetry_heatmap(...) # (此处省略原有热图代码以精简，您可保留)

        # 图2: 改进的叠加图 (Overlay)
        overlay_path = os.path.join(output_dir, f"{base_name}_overlay.jpg")
        analyzer.save_overlay_visualization(
            debug_info['frame'], debug_info['landmarks'],
            features.pearson_coefficients, overlay_path
        )

        # 图3: 时序曲线 (Time Series)
        ts_path = os.path.join(output_dir, f"{base_name}_timeseries.png")
        analyzer.plot_region_timeseries(
            features, f"{patient_id} - {action_name}", ts_path
        )

        return {
            "patient_id": patient_id,
            "action": action_name,
            "mean_corr": float(np.mean(features.pearson_coefficients)),
            "status": "success"
        }

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# ==================== 主流程控制 ====================

def main_batch_process(db_path, model_path, output_dir, max_workers=8):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # 模拟查询 (请根据实际表结构调整)
    query = """
        SELECT vf.video_id, vf.examination_id, vf.action_id, vf.file_path, 
               vf.start_frame, vf.end_frame, vf.fps, at.action_name_en, e.patient_id
        FROM video_files vf
        JOIN action_types at ON vf.action_id = at.action_id
        JOIN examinations e ON vf.examination_id = e.examination_id
        WHERE vf.file_exists = 1
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    print(f"开始批量处理: 总任务数 {len(rows)} | 进程数 {max_workers}")
    start_time = time.time()

    # 准备任务参数
    tasks = [(row, db_path, model_path, output_dir) for row in rows]

    # 并行执行
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for res in executor.map(process_video_task, tasks):
            if res:
                results.append(res)
                print(f"完成: {res['patient_id']} - {res['action']}")

    total_time = time.time() - start_time
    print(f"\n处理完毕! 耗时: {total_time:.2f}s | 平均: {total_time / len(rows):.2f}s/video")


if __name__ == '__main__':
    DB_PATH = 'facialPalsy.db'
    MODEL_PATH = '/Users/cuijinglei/PycharmProjects/medicalProject/models/face_landmarker.task'
    OUTPUT_DIR = '/Users/cuijinglei/Documents/facialPalsy/HGFA/sy_analysis'

    # M3 Max 通常有 12-16 核，建议设置为 CPU 核心数 - 2
    main_batch_process(DB_PATH, MODEL_PATH, OUTPUT_DIR, max_workers=8)