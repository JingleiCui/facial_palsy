# -*- coding: utf-8 -*-
"""
facial_symmetry_pose_robust.py

面部对称性分析 - 3D 姿态矫正 + 绝对阈值热力图 + 镜像对比 + 区域曲线

核心思路：
1）使用 SolvePnP 精确估计头部姿态 (Roll / Pitch / Yaw)；
2）先用 3D 旋转矩阵把脸“摆正”，再计算左右点对的对称指标；
3）所有不对称度都用「瞳孔间距 IPD」做归一化，给出绝对意义上 2.5%、5% 阈值；
4）输出三类可视化：
   - asymmetry.png：带坐标轴 + 对称连线 + 颜色阈值说明；
   - mirror.png：右脸真实点 vs 左脸镜像点的 2D 示意；
   - charts.png：总不对称度、各区域曲线、姿态鲁棒性检查条形图等。
"""

import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import mediapipe as mp_mediapipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==================== 字体设置 ====================
font_candidates = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'PingFang SC', 'Microsoft YaHei']
for font in font_candidates:
    try:
        matplotlib.rcParams['font.sans-serif'] = [font]
        break
    except Exception:
        continue
matplotlib.rcParams['axes.unicode_minus'] = False

# ==================== 标准 3D 人脸模型 (用于 PnP) ====================
# 选取的6个关键点：鼻尖, 下巴, 左眼左角, 右眼右角, 左嘴角, 右嘴角
# 坐标单位：任意（保持比例即可）
GENERIC_FACE_MODEL_3D = np.array([
    (0.0, 0.0, 0.0),             # Nose Tip -> Index 4
    (0.0, -330.0, -65.0),        # Chin -> Index 152
    (-225.0, 170.0, -135.0),     # Left Eye Left Corner -> Index 33
    (225.0, 170.0, -135.0),      # Right Eye Right Corner -> Index 263
    (-150.0, -150.0, -125.0),    # Left Mouth Corner -> Index 61
    (150.0, -150.0, -125.0)      # Right Mouth Corner -> Index 291
], dtype=np.float64)

# 对应的 MediaPipe 关键点索引
PNP_LANDMARK_INDICES = [4, 152, 33, 263, 61, 291]

# ==================== 对称性点对定义 ====================
SYMMETRY_INDEX_CONFIG = [
    {
        "region": "eyebrow",
        "pairs": {
            "left":  [336, 296, 334, 293, 300, 276, 283, 282, 295, 285],
            "right": [107,  66, 105,  63,  70,  46,  53,  52,  65,  55]
        }
    },
    {
        "region": "eye",
        "pairs": {
            "left":  [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            "right": [133, 155, 154, 153, 145, 144, 163,   7,  33, 246, 161, 160, 159, 158, 157, 173]
        }
    },
    {
        "region": "upper_lip",
        "pairs": {
            "left":  [267, 269, 270, 409, 291, 308, 415, 310, 311, 312],
            "right": [ 37,  39,  40, 185,  61,  78, 191,  80,  81,  82]
        }
    },
    {
        "region": "lower_lip",
        "pairs": {
            "left":  [317, 402, 318, 324, 308, 291, 375, 321, 405, 314],
            "right": [ 87, 178,  88,  95,  78,  61, 146,  91, 181,  84]
        }
    },
    {
        "region": "nose",
        "pairs": {
            "left":  [250, 458, 459, 309, 392, 289, 305, 460, 294, 358, 279, 429],
            "right": [ 20, 238, 239,  79, 166,  59,  75, 240,  64, 129,  49, 209]
        }
    },
    {
        "region": "face_contour",
        "pairs": {
            "left":  [338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377],
            "right": [109,  67, 103,  54,  21, 162, 127, 234,  93, 132,  58, 172, 136, 150, 149, 176, 148]
        }
    },
]

# ==================== 数据结构 ====================

@dataclass
class PoseInfo:
    roll: float
    pitch: float
    yaw: float
    rvec: np.ndarray
    tvec: np.ndarray
    rotation_matrix: np.ndarray


@dataclass
class AsymmetryMetrics:
    """
    所有指标均为：越小越好 (Lower is Better)
    单位：相对于瞳孔间距 IPD 的比例（无量纲）
    """
    mirror_deviation: np.ndarray  # 矫正后，左侧镜像点与右侧点的 3D 距离 / IPD
    y_diff_aligned: np.ndarray    # 矫正后，左右点的垂直高度差 / IPD
    total_score: float            # 当前帧总不对称分：Σ(mirror) / IPD
    ipd_3d: float                 # 3D 空间中的瞳孔间距（未归一化），仅供参考

# ==================== 核心算法 ====================

def estimate_head_pose_pnp(face_landmarks, img_w: int, img_h: int) -> Optional[PoseInfo]:
    """
    使用 PnP 算法估计头部姿态 (Roll / Pitch / Yaw)
    """
    image_points = []
    for idx in PNP_LANDMARK_INDICES:
        lm = face_landmarks[idx]
        image_points.append((lm.x * img_w, lm.y * img_h))
    image_points = np.array(image_points, dtype=np.float64)

    # 简单相机内参
    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]],
        dtype=np.float64
    )
    dist_coeffs = np.zeros((4, 1))  # 无畸变

    success, rvec, tvec = cv2.solvePnP(
        GENERIC_FACE_MODEL_3D,
        image_points,
        camera_matrix,
        dist_coeffs
    )
    if not success:
        return None

    rmat, _ = cv2.Rodrigues(rvec)
    proj_matrix = np.hstack((rmat, tvec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
    # OpenCV 返回 [pitch, yaw, roll]
    pitch, yaw, roll = [float(a) for a in euler_angles]

    # 防止极端值
    pitch = float(np.clip(pitch, -90, 90))
    yaw   = float(np.clip(yaw,   -90, 90))
    roll  = float(np.clip(roll,  -90, 90))

    return PoseInfo(
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        rvec=rvec,
        tvec=tvec,
        rotation_matrix=rmat
    )


def align_landmarks_3d(landmarks_np: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    """
    用逆旋转 R^T 把点云从当前姿态矫正到“正脸”姿态
    landmarks_np : [N, 3]，以鼻尖为原点的坐标
    """
    return landmarks_np @ rotation_matrix.T


def compute_robust_metrics(
        left_pts: np.ndarray,
        right_pts: np.ndarray,
        rotation_matrix: np.ndarray,
        ipd_3d: float
) -> AsymmetryMetrics:
    """
    姿态矫正后的对称性指标计算
    """
    # 1. 矫正姿态
    left_aligned = align_landmarks_3d(left_pts, rotation_matrix)
    right_aligned = align_landmarks_3d(right_pts, rotation_matrix)

    # 2. 镜像：假设对称平面 X=0，把左侧 X 取反映射到右侧
    left_mirrored = left_aligned.copy()
    left_mirrored[:, 0] = -left_mirrored[:, 0]

    # 3. 3D 偏差 + 垂直高度差
    diff_vec = left_mirrored - right_aligned
    mirror_abs = np.linalg.norm(diff_vec, axis=1)
    y_abs = np.abs(left_aligned[:, 1] - right_aligned[:, 1])

    # 4. 归一化（除以 IPD）
    scale = max(ipd_3d, 1e-6)
    mirror_norm = mirror_abs / scale
    y_norm = y_abs / scale
    total = float(mirror_abs.sum() / scale)

    return AsymmetryMetrics(
        mirror_deviation=mirror_norm,
        y_diff_aligned=y_norm,
        total_score=total,
        ipd_3d=ipd_3d
    )

# ==================== 可视化 ====================

def draw_pose_axes(img: np.ndarray, pose: PoseInfo) -> np.ndarray:
    """
    在鼻尖处绘制 3D 坐标轴
    """
    if pose is None:
        return img

    h, w = img.shape[:2]

    # 轴长
    axis_len = 100.0
    axis_points_3d = np.float32([
        [0, 0, 0],
        [axis_len, 0, 0],   # X
        [0, axis_len, 0],   # Y
        [0, 0, axis_len],   # Z
    ]).reshape(-1, 3)

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]],
        dtype=np.float64
    )
    dist_coeffs = np.zeros((4, 1))

    img_points, _ = cv2.projectPoints(
        axis_points_3d,
        pose.rvec,
        pose.tvec,
        camera_matrix,
        dist_coeffs
    )
    img_points = img_points.reshape(-1, 2).astype(int)

    origin = tuple(img_points[0])
    px = tuple(img_points[1])
    py = tuple(img_points[2])
    pz = tuple(img_points[3])

    vis = img.copy()
    cv2.line(vis, origin, px, (0, 0, 255), 3)   # X 红
    cv2.line(vis, origin, py, (0, 255, 0), 3)   # Y 绿
    cv2.line(vis, origin, pz, (255, 0, 0), 3)   # Z 蓝

    cv2.putText(vis, f"Roll:  {pose.roll:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(vis, f"Pitch: {pose.pitch:.1f}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(vis, f"Yaw:   {pose.yaw:.1f}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    return vis


def draw_asymmetry_heatmap(
        img: np.ndarray,
        pairs: List[Tuple[int, int]],
        left_lms_2d: np.ndarray,
        right_lms_2d: np.ndarray,
        metrics: AsymmetryMetrics
) -> np.ndarray:
    """
    基于 mirror_deviation 画连线热力图：
      Green: < 2.5% IPD
      Yellow: 2.5% - 5% IPD
      Red: > 5% IPD
    """
    h, w = img.shape[:2]
    overlay = img.copy()

    TH_LOW = 0.025
    TH_HIGH = 0.05

    for i, (li, ri) in enumerate(pairs):
        lx = int(left_lms_2d[i, 0] * w)
        ly = int(left_lms_2d[i, 1] * h)
        rx = int(right_lms_2d[i, 0] * w)
        ry = int(right_lms_2d[i, 1] * h)

        score = metrics.mirror_deviation[i]

        if score < TH_LOW:
            color = (0, 255, 0)
            thickness = 1
        elif score < TH_HIGH:
            color = (0, 255, 255)
            thickness = 2
        else:
            color = (0, 0, 255)
            thickness = 3

        cv2.line(overlay, (lx, ly), (rx, ry), color, thickness, cv2.LINE_AA)

    cv2.putText(overlay, "Asymmetry vs IPD (Lower is Better)", (10, h - 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(overlay, f"Green: < {TH_LOW * 100:.1f}% (Normal)", (10, h - 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(overlay, f"Yellow: {TH_LOW * 100:.1f}-{TH_HIGH * 100:.1f}% (Mild)", (10, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(overlay, f"Red: > {TH_HIGH * 100:.1f}% (Significant)", (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return overlay


def draw_mirror_comparison(
        img: np.ndarray,
        pairs: List[Tuple[int, int]],
        left_lms_2d: np.ndarray,
        right_lms_2d: np.ndarray,
        nose_x_img: float
) -> np.ndarray:
    """
    2D 镜像对比图（示意）：
    蓝点：真实右侧点
    红点：以鼻尖为轴镜像后的左侧点
    """
    h, w = img.shape[:2]
    vis = img.copy()

    for i, (li, ri) in enumerate(pairs):
        rx = int(right_lms_2d[i, 0] * w)
        ry = int(right_lms_2d[i, 1] * h)
        lx = int(left_lms_2d[i, 0] * w)
        ly = int(left_lms_2d[i, 1] * h)

        mirror_lx = int(nose_x_img + (nose_x_img - lx))
        mirror_ly = ly

        cv2.circle(vis, (rx, ry), 2, (255, 0, 0), -1)      # 蓝：真实右侧
        cv2.circle(vis, (mirror_lx, mirror_ly), 2, (0, 0, 255), -1)  # 红：镜像左侧
        cv2.line(vis, (rx, ry), (mirror_lx, mirror_ly), (255, 255, 255), 1)

    cv2.putText(vis, "Mirror Comparison (2D Projection)", (w - 260, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(vis, "Blue: Real Right", (w - 260, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.putText(vis, "Red: Mirrored Left", (w - 260, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return vis

# ==================== 主分析类 ====================

class FacialSymmetryAnalyzer:
    def __init__(self, db_path: str, model_path: str):
        self.db_path = db_path
        self.model_path = model_path

        self.pairs: List[Tuple[int, int]] = []
        self.pair_names: List[str] = []
        self.region_indices: Dict[str, List[int]] = {}

        idx = 0
        for item in SYMMETRY_INDEX_CONFIG:
            region = item["region"]
            self.region_indices[region] = []
            for li, ri in zip(item["pairs"]["left"], item["pairs"]["right"]):
                self.pairs.append((li, ri))
                self.pair_names.append(f"{region}_{idx}")
                self.region_indices[region].append(idx)
                idx += 1

    def analyze_video(self, video_path: str, output_dir: str, file_prefix: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[WARN] 无法打开视频: {video_path}")
            return None

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1
        )
        landmarker = vision.FaceLandmarker.create_from_options(options)

        frame_metrics: List[Dict] = []
        max_asym_score = -1.0

        best_frame_bgr = None
        best_pose: Optional[PoseInfo] = None
        best_landmarks = None
        best_metrics: Optional[AsymmetryMetrics] = None
        best_left_2d = None
        best_right_2d = None
        best_nose_x = None

        frame_idx = 0
        timestamp_ms = 0

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            h, w = frame_bgr.shape[:2]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp_mediapipe.Image(
                image_format=mp_mediapipe.ImageFormat.SRGB,
                data=frame_rgb
            )

            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            timestamp_ms += 33  # 约 30fps

            if not result.face_landmarks:
                frame_idx += 1
                continue

            landmarks = result.face_landmarks[0]

            # 1. 姿态估计
            pose = estimate_head_pose_pnp(landmarks, w, h)
            if pose is None:
                frame_idx += 1
                continue

            # 2. 以鼻尖为原点构建 3D 坐标
            nose = landmarks[4]
            origin = np.array([nose.x, nose.y, nose.z], dtype=np.float32)

            left_pts_raw = []
            right_pts_raw = []
            left_pts_2d = []
            right_pts_2d = []

            # 瞳孔中心，用来计算 IPD
            l_eye = landmarks[468]  # 左 iris
            r_eye = landmarks[473]  # 右 iris
            ipd_vec_3d = np.array(
                [l_eye.x - r_eye.x, l_eye.y - r_eye.y, l_eye.z - r_eye.z],
                dtype=np.float32
            )
            ipd_3d = float(np.linalg.norm(ipd_vec_3d))

            for li, ri in self.pairs:
                lp = landmarks[li]
                rp = landmarks[ri]

                left_pts_raw.append([lp.x - origin[0], lp.y - origin[1], lp.z - origin[2]])
                right_pts_raw.append([rp.x - origin[0], rp.y - origin[1], rp.z - origin[2]])

                left_pts_2d.append([lp.x, lp.y])
                right_pts_2d.append([rp.x, rp.y])

            left_np = np.array(left_pts_raw, dtype=np.float32)
            right_np = np.array(right_pts_raw, dtype=np.float32)
            left_2d = np.array(left_pts_2d, dtype=np.float32)
            right_2d = np.array(right_pts_2d, dtype=np.float32)

            # 3. 计算姿态鲁棒对称性指标
            metrics = compute_robust_metrics(left_np, right_np, pose.rotation_matrix, ipd_3d)

            frame_metrics.append({
                "frame": frame_idx,
                "roll": pose.roll,
                "pitch": pose.pitch,
                "yaw": pose.yaw,
                "metrics": metrics,
            })

            # 选取“最不对称”的帧用于可视化
            if metrics.total_score > max_asym_score:
                max_asym_score = metrics.total_score
                best_frame_bgr = frame_bgr.copy()
                best_pose = pose
                best_landmarks = landmarks
                best_metrics = metrics
                best_left_2d = left_2d
                best_right_2d = right_2d
                best_nose_x = nose.x * w

            frame_idx += 1

        cap.release()

        if not frame_metrics or best_frame_bgr is None:
            print(f"[INFO] 视频 {video_path} 没有检测到有效人脸")
            return None

        os.makedirs(output_dir, exist_ok=True)

        # ===== 1. 不对称度最大帧的可视化 =====
        img_axes = draw_pose_axes(best_frame_bgr, best_pose)
        img_heat = draw_asymmetry_heatmap(
            img_axes, self.pairs, best_left_2d, best_right_2d, best_metrics
        )
        cv2.imwrite(os.path.join(output_dir, f"{file_prefix}_asymmetry.png"), img_heat)

        img_mirror = draw_mirror_comparison(
            best_frame_bgr, self.pairs, best_left_2d, best_right_2d, best_nose_x
        )
        cv2.imwrite(os.path.join(output_dir, f"{file_prefix}_mirror.png"), img_mirror)

        # ===== 2. 时序曲线 + 区域统计 =====
        self.plot_charts(frame_metrics, os.path.join(output_dir, f"{file_prefix}_charts.png"))

        return frame_metrics

    def plot_charts(self, metrics_data: List[Dict], save_path: str):
        frames = [d["frame"] for d in metrics_data]
        total_scores = [d["metrics"].total_score for d in metrics_data]
        rolls = [d["roll"] for d in metrics_data]

        # 区域平均镜像偏差
        region_scores: Dict[str, List[float]] = {r: [] for r in self.region_indices}
        for d in metrics_data:
            m: AsymmetryMetrics = d["metrics"]
            for region, idxs in self.region_indices.items():
                region_scores[region].append(float(np.mean(m.mirror_deviation[idxs])))

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # A. 总不对称度
        ax = axes[0][0]
        ax.plot(frames, total_scores, "r-", label="Total Asymmetry (Σ mirror/IPD)")
        ax.set_title("总不对称度 (Lower is Better)")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Score")
        ax.grid(alpha=0.3)
        ax.legend()

        # B. 各区域镜像偏差
        ax = axes[0][1]
        for region, scores in region_scores.items():
            ax.plot(frames, scores, label=region)
        ax.set_title("各区域镜像偏差 (Lower is Better)")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Avg mirror/IPD")
        ax.legend(fontsize="small")
        ax.grid(alpha=0.3)

        # C. 姿态鲁棒性检查：Roll vs Aligned |Δy|
        ax = axes[1][0]
        y_diffs = [float(np.mean(d["metrics"].y_diff_aligned)) for d in metrics_data]
        ax.plot(frames, rolls, "b-", label="Roll (deg)")
        ax2 = ax.twinx()
        ax2.plot(frames, y_diffs, "g--", label="Aligned |Δy|/IPD")
        ax.set_title("姿态鲁棒性检查 (Roll vs Aligned |Δy|)")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Roll (deg)")
        ax2.set_ylabel("Aligned |Δy|/IPD")

        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc=0)

        # D. 各区域平均条形图
        ax = axes[1][1]
        regions = list(self.region_indices.keys())
        means = [float(np.mean(region_scores[r])) for r in regions]
        ax.bar(regions, means)
        ax.set_title("各区域平均镜像偏差 (Lower is Better)")
        ax.set_ylabel("Avg mirror/IPD")
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

# ==================== 批量处理入口 ====================

def process_one(task: Dict) -> bool:
    try:
        analyzer = FacialSymmetryAnalyzer(task["db_path"], task["model_path"])
        prefix = f"{task['exam_id']}_{task['action']}"
        analyzer.analyze_video(task["path"], task["out_dir"], prefix)
        return True
    except Exception as e:
        print(f"[ERROR] {task['path']}: {e}")
        return False


def main():
    # ⚠️ 这里按你本地环境改路径
    DB_PATH = "/facial_palsy/facialPalsy.db"
    MODEL_PATH = "/Users/cuijinglei/PycharmProjects/medicalProject/models/face_landmarker.task"
    OUT_DIR = "/Users/cuijinglei/Documents/facial_palsy/HGFA/symmetry_pnp"

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT vf.file_path, vf.examination_id, at.action_name_en
        FROM video_files vf
        JOIN action_types at ON vf.action_id = at.action_id
        WHERE vf.file_exists = 1
        """
    )
    rows = cursor.fetchall()
    conn.close()

    tasks: List[Dict] = []
    for path, eid, act in rows:
        if os.path.exists(path):
            tasks.append({
                "db_path": DB_PATH,
                "model_path": MODEL_PATH,
                "path": path,
                "out_dir": OUT_DIR,
                "exam_id": eid,
                "action": act,
            })

    print(f"开始处理 {len(tasks)} 个视频...")
    if not tasks:
        print("没有可处理的视频。")
        return

    with ProcessPoolExecutor(max_workers=8) as exe:
        futures = [exe.submit(process_one, t) for t in tasks]
        for _ in as_completed(futures):
            pass

    print("全部完成。")


if __name__ == "__main__":
    main()
