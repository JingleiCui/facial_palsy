# -*- coding: utf-8 -*-
"""
facial_symmetry_analysis.py

面部对称性分析 - 参考论文 "Machine Learning Methods to Track Dynamic Facial Function in Facial Palsy"
（实现思路：左右对称点对 -> 时序Pearson相关 + 帧级|Δy|不对称评分 + 可视化 + 马氏距离）

本版本改动（相对你当前的 facial_symmetry_analysis.py）：
1) batch_process_database 支持多进程（ProcessPoolExecutor），提升批量速度
2) 在视频片段内对每一帧累计所有点对的 |Δy|，选择“最不对称帧”作为：
   - overlay 叠加的底图
   - 索引检查图（index_check）的底图
   而不是“第一帧检测到人脸的帧”
3) overlay 热力图默认使用“最不对称帧”的每对点 |Δy|（更直观）
4) rolling Pearson 计算向量化（cumsum 版本），避免 O(T*F*w) 的三重循环

建议（MacBook Pro M3 Max）：
- num_workers 先从 6~10 试起（过多会因为视频解码/IO/内部线程而变慢）
"""

import os
import re
import json
import csv
import sqlite3
import multiprocessing as mp
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import mediapipe as mp_mediapipe
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from scipy import stats
from scipy.spatial.distance import mahalanobis
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")  # 多进程/无GUI更稳
import matplotlib.pyplot as plt

# 中文字体/负号
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'PingFang SC']
matplotlib.rcParams['axes.unicode_minus'] = False


# ==================== MediaPipe 478点中的对称特征点定义 ====================

SYMMETRY_INDEX_CONFIG = [
    {
        "region": "eyebrow",
        "pairs": {
            "left":  [336, 296, 334, 293, 300, 276, 283, 282, 295, 285],
            "right": [107,  66, 105,  63,  70,  46,  53,  52,  65,  55],
        }
    },
    {
        "region": "eye",
        "pairs": {
            "left":  [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            "right": [133, 155, 154, 153, 145, 144, 163,   7,  33, 246, 161, 160, 159, 158, 157, 173],
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
            "left":  [267, 269, 270, 409, 291, 308, 415, 310, 311, 312],
            "right": [ 37,  39,  40, 185,  61,  78, 191,  80,  81,  82],
        }
    },
    {
        "region": "lower_lip",
        "pairs": {
            "left":  [317, 402, 318, 324, 308, 291, 375, 321, 405, 314],
            "right": [ 87, 178,  88,  95,  78,  61, 146,  91, 181,  84],
        }
    },
    {
        "region": "nose",
        "pairs": {
            "left": [250, 458, 459, 309, 392, 289, 305, 460, 294, 358, 279, 429, 420, 456],
            "right": [20, 238, 239,  79, 166,  59,  75, 240,  64, 129,  49, 209, 198, 236],
        }

    },
    {
        "region": "face_contour",
        "pairs": {
            "left":  [338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377],
            "right": [109,  67, 103,  54,  21, 162, 127, 234,  93, 132,  58, 172, 136, 150, 149, 176, 148],
        },
    },
]

EYE_INNER_CANTHUS_LEFT = 362
EYE_INNER_CANTHUS_RIGHT = 133

def build_pairs_and_names(cfg_list: list):
    """
    cfg_list: SYMMETRY_INDEX_CONFIG（list结构）
    return:
      pairs: [(li, ri), ...]
      names: ["eyebrow_01", "eyebrow_02", ..., "eye_01", ...]
      region_feature_indices: {"eyebrow":[0,1,...], "eye":[...], ...}
    """
    pairs = []
    names = []
    region_feature_indices = {}

    for item in cfg_list:
        region = item["region"]
        lr = item["pairs"]
        L = lr["left"]
        R = lr["right"]

        if len(L) != len(R):
            raise ValueError(f"[{region}] left/right 长度不一致: {len(L)} vs {len(R)}")

        region_feature_indices.setdefault(region, [])

        for i, (li, ri) in enumerate(zip(L, R), start=1):
            pairs.append((int(li), int(ri)))
            names.append(f"{region}_{i:02d}")
            region_feature_indices[region].append(len(pairs) - 1)

    return pairs, names, region_feature_indices


@dataclass
class SymmetryFeatures:
    """对称性特征数据结构"""
    pearson_coefficients: np.ndarray              # [F]
    landmark_names: List[str]                     # len=F
    y_coords_left: np.ndarray                     # [T, F]
    y_coords_right: np.ndarray                    # [T, F]
    frame_count: int                              # T


def _safe_name(s: str) -> str:
    s = str(s) if s is not None else ""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", s)[:160]


class FacialSymmetryAnalyzer:
    """面部对称性分析器"""

    def __init__(
        self,
        db_path: str,
        model_path: str = '/Users/cuijinglei/PycharmProjects/medicalProject/models/face_landmarker.task',
        verbose: bool = True,
    ):
        self.db_path = db_path
        self.model_path = model_path
        self.verbose = bool(verbose)

        # 生成点对 + 名称
        self.feature_pairs, self.feature_names, self.region_feature_indices = build_pairs_and_names(
            SYMMETRY_INDEX_CONFIG
        )
        self.n_features = len(self.feature_pairs)

        # debug：最不对称帧
        self._debug_frame: Optional[np.ndarray] = None
        self._debug_landmarks = None
        self._debug_frame_abs_index: Optional[int] = None  # 原视频中的绝对帧号
        self._debug_asym_score: Optional[float] = None     # sum(|Δy|)
        self._debug_pair_absdy: Optional[np.ndarray] = None  # [F] 最不对称帧每对点 |Δy|

        if self.verbose:
            print("✅ 分析器初始化完成")
            print(f"   - 对称点对数: {self.n_features}")
            print(f"   - 数据库: {db_path}")
            print(f"   - Landmarker模型: {model_path}")

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

    def extract_landmarks_from_video(
        self,
        video_path: str,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        fps: Optional[float] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Returns:
            left_coords:  [T, F, 3]
            right_coords: [T, F, 3]
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        start_frame = 0 if start_frame is None else int(start_frame)
        end_frame = total_frames if end_frame is None else int(end_frame)
        if total_frames > 0:
            start_frame = max(0, min(start_frame, total_frames - 1))
            end_frame = max(0, min(end_frame, total_frames))

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # 每个视频创建一个全新的 landmarker（VIDEO模式），避免跨视频timestamp回退
        landmarker = self._create_landmarker()

        coords_list = []

        # reset debug
        self._debug_frame = None
        self._debug_landmarks = None
        self._debug_frame_abs_index = None
        self._debug_asym_score = None
        self._debug_pair_absdy = None

        processed_idx = 0
        last_ts = -1
        frame_abs_idx = start_frame

        try:
            while cap.isOpened() and frame_abs_idx < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp_mediapipe.Image(image_format=mp_mediapipe.ImageFormat.SRGB, data=rgb_frame)

                # 片段内时间戳：从0开始递增，并强制单调递增
                if fps and fps > 0:
                    timestamp_ms = int(processed_idx * 1000.0 / float(fps))
                else:
                    timestamp_ms = processed_idx * 33  # 默认30fps

                if timestamp_ms <= last_ts:
                    timestamp_ms = last_ts + 1
                last_ts = timestamp_ms
                processed_idx += 1

                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                if result.face_landmarks:
                    face_landmarks = result.face_landmarks[0]

                    # 组装坐标
                    coords = []
                    pair_absdy = np.empty((self.n_features,), dtype=np.float32)
                    for j, (left_idx, right_idx) in enumerate(self.feature_pairs):
                        left_lm = face_landmarks[left_idx]
                        right_lm = face_landmarks[right_idx]
                        coords.append([[left_lm.x, left_lm.y, left_lm.z],
                                       [right_lm.x, right_lm.y, right_lm.z]])
                        pair_absdy[j] = abs(float(left_lm.y) - float(right_lm.y))

                    coords_list.append(coords)

                    # ===== 选择“最不对称帧”：score = sum(|Δy|) =====
                    asym_score = float(pair_absdy.sum())
                    if (self._debug_asym_score is None) or (asym_score > self._debug_asym_score):
                        self._debug_asym_score = asym_score
                        self._debug_frame = frame.copy()
                        self._debug_landmarks = face_landmarks
                        self._debug_frame_abs_index = int(frame_abs_idx)
                        self._debug_pair_absdy = pair_absdy.copy()

                frame_abs_idx += 1

        finally:
            cap.release()
            try:
                landmarker.close()
            except Exception:
                pass

        if not coords_list:
            return None, None

        coords_array = np.array(coords_list, dtype=np.float32)  # [T, F, 2, 3]
        left_coords = coords_array[:, :, 0, :]
        right_coords = coords_array[:, :, 1, :]
        return left_coords, right_coords

    def save_index_check_image(self, frame_bgr, face_landmarks, output_path):
        """把点对连线、点索引画在选中的debug帧上（现在是“最不对称帧”），并加一条面中线"""
        if frame_bgr is None or face_landmarks is None:
            return

        h, w = frame_bgr.shape[:2]
        vis = frame_bgr.copy()

        # 1) 先画所有点对 + 编号 + 连线（和之前一样）
        for (li, ri) in self.feature_pairs:
            l = face_landmarks[li]
            r = face_landmarks[ri]

            lx, ly = int(l.x * w), int(l.y * h)
            rx, ry = int(r.x * w), int(r.y * h)

            # 连接左右点
            cv2.line(vis, (lx, ly), (rx, ry), (0, 255, 0), 1)

            # 左点红色 + 索引
            cv2.circle(vis, (lx, ly), 2, (0, 0, 255), -1)
            cv2.putText(
                vis, str(li), (lx + 2, ly - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1
            )

            # 右点蓝色 + 索引
            cv2.circle(vis, (rx, ry), 2, (255, 0, 0), -1)
            cv2.putText(
                vis, str(ri), (rx + 2, ry - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1
            )

        # 2) 画“面中线”：用眼内眦中垂线 + face_contour 的最高/最低点
        try:
            lm_l = face_landmarks[EYE_INNER_CANTHUS_LEFT]
            lm_r = face_landmarks[EYE_INNER_CANTHUS_RIGHT]

            lx, ly = lm_l.x * w, lm_l.y * h
            rx, ry = lm_r.x * w, lm_r.y * h

            # 中点 M（大致在两眼之间）
            mx, my = (lx + rx) / 2.0, (ly + ry) / 2.0

            # 眼内眦连线方向 d = (dx, dy)，面中线方向取法向量 n = (-dy, dx)
            dx, dy = (rx - lx), (ry - ly)
            if abs(dx) + abs(dy) < 1e-6:
                raise ValueError("眼内眦两点太接近，无法计算面中线方向")

            M = np.array([mx, my], dtype=np.float32)
            n = np.array([-dy, dx], dtype=np.float32)  # 中垂线方向
            denom = float(np.dot(n, n)) + 1e-6  # 防止除零

            # -------- 2.2 用 face_contour 中的第一对 / 最后一对作为最高/最低点 --------
            # 约定：feature_pairs 中 face_contour 的第一对是最高点，最后一对是最低点
            top_li, top_ri = self.feature_pairs[0]
            bot_li, bot_ri = self.feature_pairs[-1]

            top_lm_l = face_landmarks[top_li]
            top_lm_r = face_landmarks[top_ri]
            bot_lm_l = face_landmarks[bot_li]
            bot_lm_r = face_landmarks[bot_ri]

            # 顶部/底部“左右中点”，先大致取在脸的中间
            top_mid = np.array([
                (top_lm_l.x * w + top_lm_r.x * w) / 2.0,
                (top_lm_l.y * h + top_lm_r.y * h) / 2.0
            ], dtype=np.float32)
            bot_mid = np.array([
                (bot_lm_l.x * w + bot_lm_r.x * w) / 2.0,
                (bot_lm_l.y * h + bot_lm_r.y * h) / 2.0
            ], dtype=np.float32)

            # -------- 2.3 把 top_mid / bot_mid 投影到“面中线”上 --------
            # 投影公式：P_proj = M + ((P - M)·n / (n·n)) * n
            t_top = float(np.dot(top_mid - M, n)) / denom
            t_bot = float(np.dot(bot_mid - M, n)) / denom

            p_top = M + t_top * n
            p_bot = M + t_bot * n

            pt1 = (int(round(p_top[0])), int(round(p_top[1])))
            pt2 = (int(round(p_bot[0])), int(round(p_bot[1])))

            # -------- 2.4 画面中线 + 端点 --------
            # 用黄色更显眼
            cv2.line(vis, pt1, pt2, (0, 255, 255), 2)
            cv2.circle(vis, pt1, 4, (0, 255, 255), -1)
            cv2.circle(vis, pt2, 4, (0, 255, 255), -1)

        except Exception as e:
            # 出问题时不要影响主流程，只打一个调试信息
            print(f"[WARN] 绘制面中线失败: {e}")

        # 3) 保存图像
        cv2.imwrite(str(output_path), vis)

    def calculate_pearson_coefficients(
        self,
        left_coords: np.ndarray,
        right_coords: np.ndarray,
        use_y_only: bool = True
    ) -> SymmetryFeatures:
        """
        计算左右对称点 y 坐标的 Pearson 相关系数（每个点对一个 r）
        """
        if left_coords is None or right_coords is None:
            raise ValueError("未检测到人脸关键点（left/right coords 为 None）")

        n_frames, n_features, _ = left_coords.shape

        # y坐标（索引1）
        y_left = left_coords[:, :, 1]   # [T, F]
        y_right = right_coords[:, :, 1] # [T, F]

        pearson_coeffs = np.zeros((n_features,), dtype=np.float32)
        for i in range(n_features):
            a = y_left[:, i]
            b = y_right[:, i]
            # 常数序列会让 pearsonr 报错
            if np.std(a) < 1e-8 or np.std(b) < 1e-8:
                pearson_coeffs[i] = np.nan
            else:
                corr, _ = stats.pearsonr(a, b)
                pearson_coeffs[i] = float(corr)

        return SymmetryFeatures(
            pearson_coefficients=pearson_coeffs,
            landmark_names=list(self.feature_names),
            y_coords_left=y_left,
            y_coords_right=y_right,
            frame_count=n_frames
        )

    @staticmethod
    def _rolling_corr_cumsum(y_left: np.ndarray, y_right: np.ndarray, w: int) -> np.ndarray:
        """
        向量化 rolling Pearson：
        y_left/y_right: [T, F]
        return: rolling_corr [T, F]，前 w-1 行为 NaN
        """
        yL = y_left.astype(np.float64, copy=False)
        yR = y_right.astype(np.float64, copy=False)
        T, F = yL.shape
        w = int(max(3, w))
        if T < w:
            return np.full((T, F), np.nan, dtype=np.float32)

        def cumsum_pad(x):
            return np.vstack([np.zeros((1, F), dtype=np.float64), np.cumsum(x, axis=0)])

        cL = cumsum_pad(yL)
        cR = cumsum_pad(yR)
        cLL = cumsum_pad(yL * yL)
        cRR = cumsum_pad(yR * yR)
        cLR = cumsum_pad(yL * yR)

        sumL = cL[w:] - cL[:-w]
        sumR = cR[w:] - cR[:-w]
        sumLL = cLL[w:] - cLL[:-w]
        sumRR = cRR[w:] - cRR[:-w]
        sumLR = cLR[w:] - cLR[:-w]

        meanL = sumL / w
        meanR = sumR / w

        cov = (sumLR / w) - (meanL * meanR)
        varL = (sumLL / w) - (meanL * meanL)
        varR = (sumRR / w) - (meanR * meanR)

        denom = np.sqrt(np.maximum(varL, 0.0) * np.maximum(varR, 0.0))
        corr = np.divide(cov, denom, out=np.full_like(cov, np.nan), where=(denom > 1e-12))

        out = np.full((T, F), np.nan, dtype=np.float32)
        out[w - 1:] = corr.astype(np.float32)
        return out

    def compute_region_timeseries(
        self,
        features: SymmetryFeatures,
        rolling_window: int = 15
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        1) abs_diff: 每帧的 |y_left - y_right|
        2) rolling_corr: 滑动窗口 Pearson（向量化版本）
        """
        y_left = features.y_coords_left  # [T, F]
        y_right = features.y_coords_right
        T, F = y_left.shape

        abs_diff = np.abs(y_left - y_right).astype(np.float32)  # [T, F]
        rolling_corr = self._rolling_corr_cumsum(y_left, y_right, rolling_window)  # [T, F]

        region_abs = {}
        region_corr = {}
        for region, idxs in self.region_feature_indices.items():
            idxs = list(idxs)
            region_abs[region] = abs_diff[:, idxs].mean(axis=1) if idxs else np.full((T,), np.nan, np.float32)
            region_corr[region] = np.nanmean(rolling_corr[:, idxs], axis=1) if idxs else np.full((T,), np.nan, np.float32)

        return {
            "abs_diff": region_abs,
            "rolling_corr": region_corr,
            "abs_diff_raw": abs_diff,
            "rolling_corr_raw": rolling_corr
        }

    def save_region_timeseries_plot(
        self,
        ts: Dict[str, Dict[str, np.ndarray]],
        fps: float,
        title: str,
        save_path_base: str
    ):
        """
        输出两张曲线：
        - abs_diff（越低越对称）
        - rolling_corr（越高越对称）
        """
        region_abs = ts["abs_diff"]
        region_corr = ts["rolling_corr"]

        any_region = next(iter(region_abs.keys()))
        T = len(region_abs[any_region])

        if fps and fps > 0:
            t = np.arange(T) / float(fps)
            xlabel = "Time (s)"
        else:
            t = np.arange(T)
            xlabel = "Frame"

        os.makedirs(os.path.dirname(save_path_base), exist_ok=True)

        # abs_diff
        fig = plt.figure(figsize=(14, 6))
        ax = plt.gca()
        for region, y in region_abs.items():
            ax.plot(t, y, linewidth=1.6, label=region)
        ax.set_title(f"{title}\nRegion Asymmetry (abs(yL-yR), lower=better)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("abs(yL - yR)")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2, fontsize=9)
        out1 = save_path_base.replace(".png", "_region_absdiff.png")
        fig.savefig(out1, dpi=160, bbox_inches="tight")
        plt.close(fig)

        # rolling corr
        fig = plt.figure(figsize=(14, 6))
        ax = plt.gca()
        for region, y in region_corr.items():
            ax.plot(t, y, linewidth=1.6, label=region)
        ax.set_title(f"{title}\nRegion Rolling Pearson (higher=better)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Rolling Pearson r")
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2, fontsize=9)
        out2 = save_path_base.replace(".png", "_region_rollingcorr.png")
        fig.savefig(out2, dpi=160, bbox_inches="tight")
        plt.close(fig)

        if self.verbose:
            print(f"  ✓ Region time-series saved: {out1}")
            print(f"  ✓ Region time-series saved: {out2}")

        return out1, out2

    @staticmethod
    def _fast_splat_heatmap(
        w: int,
        h: int,
        pts_xy: np.ndarray,    # [N,2] in pixel coords
        vals: np.ndarray,      # [N]
        canvas_res: int = 256,
        sigma: float = 8.0
    ) -> np.ndarray:
        """
        比 griddata 快很多：把点“撒”到低分辨率画布 -> 高斯模糊 -> 归一化 -> colormap
        return: heat_bgr resized to (w,h)
        """
        canvas_res = int(max(64, canvas_res))
        sx = canvas_res / float(w)
        sy = canvas_res / float(h)

        acc = np.zeros((canvas_res, canvas_res), dtype=np.float32)
        cnt = np.zeros((canvas_res, canvas_res), dtype=np.float32)

        for (x, y), v in zip(pts_xy, vals):
            cx = int(np.clip(x * sx, 0, canvas_res - 1))
            cy = int(np.clip(y * sy, 0, canvas_res - 1))
            cv2.circle(acc, (cx, cy), 2, float(v), -1)
            cv2.circle(cnt, (cx, cy), 2, 1.0, -1)

        heat = acc / (cnt + 1e-6)
        heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=sigma, sigmaY=sigma)

        vmin = float(np.min(heat))
        vmax = float(np.max(heat))
        heat = (heat - vmin) / (vmax - vmin + 1e-6)
        heat_u8 = np.clip(heat * 255.0, 0, 255).astype(np.uint8)

        heat_bgr_small = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
        heat_bgr = cv2.resize(heat_bgr_small, (w, h), interpolation=cv2.INTER_CUBIC)
        return heat_bgr

    def save_overlay_asymmetry_heatmap(
        self,
        frame_bgr: np.ndarray,
        face_landmarks,
        pair_values: np.ndarray,
        save_path: str,
        alpha: float = 0.45,
        canvas_res: int = 256
    ):
        """
        在原图上叠加热力图：
        - pair_values：每对点的分数（推荐传“最不对称帧”的 |Δy|，越大越不对称）
        """
        if frame_bgr is None or face_landmarks is None:
            if self.verbose:
                print("  ⚠️ overlay skipped: no debug frame/landmarks")
            return

        h, w = frame_bgr.shape[:2]

        pts = []
        vals = []
        for (li, ri), s in zip(self.feature_pairs, pair_values):
            for idx in (li, ri):
                lm = face_landmarks[idx]
                pts.append([lm.x * w, lm.y * h])
                vals.append(float(s))

        pts = np.asarray(pts, dtype=np.float32)
        vals = np.asarray(vals, dtype=np.float32)

        heat_bgr = self._fast_splat_heatmap(w=w, h=h, pts_xy=pts, vals=vals, canvas_res=canvas_res, sigma=8.0)
        overlay = cv2.addWeighted(frame_bgr, 1 - alpha, heat_bgr, alpha, 0)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, overlay)
        if self.verbose:
            print(f"  ✓ Overlay heatmap saved: {save_path}")

    def visualize_symmetry_heatmap(
        self,
        features: SymmetryFeatures,
        title: str,
        save_path: str,
    ):
        """（与你原来一致）Pearson柱状 + 空间分布 + 统计摘要"""
        fig = plt.figure(figsize=(16, 8))

        # 1) Pearson barh
        ax1 = plt.subplot(1, 3, 1)
        coeffs = features.pearson_coefficients
        colors = []
        for coef in coeffs:
            if np.isnan(coef):
                colors.append('gray')
            elif coef > 0.8:
                colors.append('darkblue')
            elif coef > 0.5:
                colors.append('lightblue')
            elif coef > 0:
                colors.append('yellow')
            else:
                colors.append('red')

        bars = ax1.barh(range(self.n_features), coeffs, color=colors)
        ax1.set_yticks(range(self.n_features))
        ax1.set_yticklabels(features.landmark_names, fontsize=9)
        ax1.set_xlabel('Pearson Correlation Coefficient', fontsize=11)
        ax1.set_title('对称性系数\n(蓝色=高对称, 红色=不对称)', fontsize=10)
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        ax1.axvline(x=0.8, color='green', linestyle='--', linewidth=0.8, alpha=0.5)
        ax1.set_xlim(-1, 1)
        ax1.grid(axis='x', alpha=0.3)

        for i, (bar, val) in enumerate(zip(bars, coeffs)):
            if np.isnan(val):
                label = "nan"
                xpos = 0.05
                ha = "left"
            else:
                label = f"{val:.3f}"
                xpos = (val + 0.05) if val > 0 else (val - 0.05)
                ha = "left" if val > 0 else "right"
            ax1.text(xpos, i, label, va='center', ha=ha, fontsize=8)

        # 2) 简化空间分布（按平均y排序）
        ax2 = plt.subplot(1, 3, 2)
        avg_y = (features.y_coords_left.mean(axis=0) + features.y_coords_right.mean(axis=0)) / 2
        y_norm = (avg_y - avg_y.min()) / (avg_y.max() - avg_y.min() + 1e-6)

        x = np.linspace(-1, 1, self.n_features)
        y = y_norm

        scatter = ax2.scatter(x, y, c=np.nan_to_num(coeffs, nan=0.0),
                              s=500, cmap='RdYlBu', vmin=-1, vmax=1,
                              edgecolors='black', linewidth=1.5)
        for i, (xi, yi) in enumerate(zip(x, y)):
            ax2.annotate(f'{i + 1}', (xi, yi), ha='center', va='center',
                         fontsize=9, fontweight='bold')

        ax2.set_xlim(-1.2, 1.2)
        ax2.set_ylim(-0.1, 1.1)
        ax2.set_xlabel('左 ← | → 右', fontsize=11)
        ax2.set_ylabel('上 ← | → 下', fontsize=11)
        ax2.set_title('面部对称性空间分布', fontsize=11)
        ax2.set_aspect('equal')

        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Pearson Coefficient', fontsize=10)

        # 3) 统计摘要
        ax3 = plt.subplot(1, 3, 3)
        ax3.axis('off')

        valid = coeffs[np.isfinite(coeffs)]
        mean_corr = float(np.mean(valid)) if valid.size else float("nan")
        std_corr = float(np.std(valid)) if valid.size else float("nan")
        min_corr = float(np.min(valid)) if valid.size else float("nan")
        max_corr = float(np.max(valid)) if valid.size else float("nan")

        def mean_by_region(region: str) -> float:
            idxs = self.region_feature_indices.get(region, [])
            vals = coeffs[idxs] if idxs else np.array([], dtype=np.float32)
            vals = vals[np.isfinite(vals)]
            return float(np.mean(vals)) if vals.size else float("nan")

        eyebrow_corr = mean_by_region("eyebrow")
        eye_corr = mean_by_region("eye")
        upper_lip_corr = mean_by_region("upper_lip")
        lower_lip_corr = mean_by_region("lower_lip")
        nose_corr = mean_by_region("nose")

        high_sym = int(np.sum((coeffs > 0.8) & np.isfinite(coeffs)))
        medium_sym = int(np.sum((coeffs > 0.5) & (coeffs <= 0.8) & np.isfinite(coeffs)))
        low_sym = int(np.sum((coeffs <= 0.5) & np.isfinite(coeffs)))

        summary_text = f"""
Statistical Summary
{'=' * 30}

Overall Symmetry:
  Mean Correlation: {mean_corr:.3f} ± {std_corr:.3f}
  Range: [{min_corr:.3f}, {max_corr:.3f}]

Regional Analysis:
  Eyebrow Region: {eyebrow_corr:.3f}
  Eye Region: {eye_corr:.3f}
  Upper Lip Region: {upper_lip_corr:.3f}
  Lower Lip Region: {lower_lip_corr:.3f}
  Nose Region: {nose_corr:.3f}

Symmetry Rating:
  High Symmetry (>0.8): {high_sym}/{self.n_features}
  Medium Symmetry (0.5-0.8): {medium_sym}/{self.n_features}
  Low Symmetry (≤0.5): {low_sym}/{self.n_features}

Data Quality:
  Valid Frames: {features.frame_count}

Health Assessment:
  {'Normal Symmetry' if mean_corr > 0.8 else 'Asymmetry Detected' if mean_corr > 0.5 else 'Severe Asymmetry'}
"""
        ax3.text(0.1, 0.95, summary_text, transform=ax3.transAxes,
                 fontsize=10, va='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

        plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        if self.verbose:
            print(f"  ✓ Heatmap saved: {save_path}")

    def calculate_mahalanobis_distance(
        self,
        features: SymmetryFeatures,
        healthy_reference: Optional[Dict] = None
    ) -> float:
        """
        计算Mahalanobis距离，量化偏离健康人群的程度
        注意：如果没有真实健康对照，这里只是占位逻辑。
        """
        if healthy_reference is None:
            healthy_mean = np.full(self.n_features, 0.95, dtype=np.float32)
            healthy_std = np.full(self.n_features, 0.05, dtype=np.float32)
            healthy_cov = np.diag(healthy_std ** 2).astype(np.float32)
        else:
            healthy_mean = np.asarray(healthy_reference['mean'], dtype=np.float32)
            healthy_cov = np.asarray(healthy_reference['cov'], dtype=np.float32)

        x = np.nan_to_num(features.pearson_coefficients, nan=np.nanmean(features.pearson_coefficients))
        try:
            md = mahalanobis(x, healthy_mean, np.linalg.inv(healthy_cov))
        except np.linalg.LinAlgError:
            if self.verbose:
                print("  ⚠️ 协方差矩阵奇异，使用欧氏距离代替")
            md = float(np.linalg.norm(x - healthy_mean))

        return float(md)

    def analyze_single_video(
        self,
        video_path: str,
        action_name: str,
        output_dir: str,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        fps: Optional[float] = None,
        patient_id: Optional[str] = None,
        examination_id: Optional[str] = None,
        video_id: Optional[int] = None,
    ) -> Dict:
        """
        分析单个视频的面部对称性
        """
        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"分析视频: {Path(video_path).name}")
            print(f"动作: {action_name}")
            print(f"{'=' * 60}")

        # 1. 提取特征点（同时会选出“最不对称帧”）
        left_coords, right_coords = self.extract_landmarks_from_video(
            video_path, start_frame, end_frame, fps
        )
        if left_coords is None:
            raise ValueError("未检测到人脸关键点（整段视频无有效检测）")

        # 2. Pearson
        if self.verbose:
            print(f"\n  计算Pearson相关系数...")
        features = self.calculate_pearson_coefficients(left_coords, right_coords)
        if self.verbose:
            print(f"  ✓ 完成")

        # 3. Mahalanobis
        if self.verbose:
            print(f"\n  计算Mahalanobis距离...")
        md = self.calculate_mahalanobis_distance(features)
        if self.verbose:
            print(f"  ✓ 马氏距离: {md:.3f}")

        # 4. 可视化
        if self.verbose:
            print(f"\n  生成可视化...")

        title = f"面部对称性分析: {action_name}"
        if patient_id:
            title += f" | 患者: {patient_id}"
        if examination_id:
            title += f" | 检查: {examination_id}"

        os.makedirs(output_dir, exist_ok=True)

        video_stem = Path(video_path).stem
        prefix = "_".join([p for p in [_safe_name(examination_id), _safe_name(action_name)] if p])

        save_path = os.path.join(output_dir, f"{prefix}_symmetry.png")
        self.visualize_symmetry_heatmap(features, title, save_path)

        # Region时序
        ts = self.compute_region_timeseries(features, rolling_window=15)
        abs_path, corr_path = self.save_region_timeseries_plot(
            ts=ts,
            fps=float(fps) if fps else 0.0,
            title=title,
            save_path_base=save_path
        )

        # Overlay：默认用“最不对称帧”的每对点 |Δy|
        overlay_path = save_path.replace(".png", "_heatmap.png")
        if self._debug_pair_absdy is None:
            # fallback：用 1 - pearson
            pair_values = 1.0 - np.clip(np.nan_to_num(features.pearson_coefficients, nan=0.0), -1.0, 1.0)
        else:
            pair_values = self._debug_pair_absdy

        self.save_overlay_asymmetry_heatmap(
            frame_bgr=self._debug_frame,
            face_landmarks=self._debug_landmarks,
            pair_values=pair_values,
            save_path=overlay_path,
            alpha=0.45,
            canvas_res=256
        )

        # 索引检查图（也用最不对称帧）
        index_img_path = os.path.join(output_dir, f"{prefix}_index_check.png")
        self.save_index_check_image(self._debug_frame, self._debug_landmarks, index_img_path)

        # 5. 保存数据
        result = {
            "video_path": video_path,
            "video_id": video_id,
            "action_name": action_name,
            "patient_id": patient_id,
            "examination_id": examination_id,
            "frame_count": int(features.frame_count),
            "pearson_coefficients": np.nan_to_num(features.pearson_coefficients, nan=float("nan")).tolist(),
            "mahalanobis_distance": float(md),
            "mean_correlation": float(np.nanmean(features.pearson_coefficients)),
            "std_correlation": float(np.nanstd(features.pearson_coefficients)),
            "visualization_path": save_path,
            "overlay_path": overlay_path,
            "index_check_path": index_img_path,
            "region_absdiff_path": abs_path,
            "region_rollingcorr_path": corr_path,
            "worst_frame_abs_index": self._debug_frame_abs_index,
            "worst_frame_asym_score_sum_absdy": self._debug_asym_score,
        }

        json_path = save_path.replace(".png", ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        if self.verbose:
            print(f"  ✓ 结果已保存: {json_path}")
            print(f"\n{'=' * 60}")
            print(f"✅ 分析完成!")
            print(f"   平均对称性: {result['mean_correlation']:.3f}")
            print(f"   马氏距离: {result['mahalanobis_distance']:.3f}")
            print(f"   最不对称帧(abs idx): {result['worst_frame_abs_index']} | score(sum|Δy|): {result['worst_frame_asym_score_sum_absdy']}")
            print(f"{'=' * 60}\n")

        return result

    def batch_process_database(
        self,
        output_dir: str,
        limit: Optional[int] = None,
        action_filter: Optional[List[str]] = None,
        use_multiprocessing: bool = True,
        num_workers: Optional[int] = None,
    ):
        """
        批量处理数据库中的所有视频（可多进程）
        """
        os.makedirs(output_dir, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
            SELECT 
                vf.video_id,
                vf.examination_id,
                vf.action_id,
                vf.file_path,
                vf.start_frame,
                vf.end_frame,
                vf.fps,
                at.action_name_en,
                e.patient_id
            FROM video_files vf
            JOIN action_types at ON vf.action_id = at.action_id
            JOIN examinations e ON vf.examination_id = e.examination_id
            WHERE vf.file_exists = 1
        """

        if action_filter:
            placeholders = ",".join(["?"] * len(action_filter))
            query += f" AND at.action_name_en IN ({placeholders})"
            cursor.execute(query, action_filter)
        else:
            cursor.execute(query)

        videos = cursor.fetchall()
        conn.close()

        if limit:
            videos = videos[: int(limit)]

        print(f"\n{'=' * 60}")
        print("批量处理模式")
        print(f"总视频数: {len(videos)}")
        print(f"输出目录: {output_dir}")
        print(f"多进程: {use_multiprocessing}")
        print(f"{'=' * 60}\n")

        # 过滤不存在的视频（减少无效任务）
        tasks = []
        for (video_id, exam_id, action_id, file_path, start_frame, end_frame, fps, action_name, patient_id) in videos:
            if not file_path or (not os.path.exists(file_path)):
                continue
            tasks.append({
                "db_path": self.db_path,
                "model_path": self.model_path,
                "video_id": int(video_id),
                "video_path": file_path,
                "action_name": action_name,
                "output_dir": output_dir,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "fps": fps,
                "patient_id": patient_id,
                "examination_id": exam_id,
            })

        if not tasks:
            print("⚠️ 没有可处理的视频任务（路径不存在或被过滤）")
            return []

        if num_workers is None:
            cpu = os.cpu_count() or 8
            num_workers = min(8, cpu)  # 默认不要拉满，避免IO/内部线程反噬

        results = []
        errors = []

        # ========== 多进程并行 ==========
        if use_multiprocessing and num_workers > 1:
            # 尽量确保 spawn（macOS 默认就是 spawn）
            try:
                if mp.get_start_method(allow_none=True) is None:
                    mp.set_start_method("spawn", force=True)
            except RuntimeError:
                pass

            print(f"🚀 使用多进程: num_workers={num_workers}, tasks={len(tasks)}")

            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futs = [executor.submit(_worker_analyze_one, task) for task in tasks]
                total = len(futs)
                done = 0

                for fut in as_completed(futs):
                    done += 1
                    try:
                        ok, payload = fut.result()
                        if ok:
                            results.append(payload)
                        else:
                            errors.append(payload)
                    except Exception as e:
                        errors.append({"error": str(e)})

                    if done % 10 == 0 or done == total:
                        print(f"进度: {done}/{total} | 成功: {len(results)} | 失败: {len(errors)}")

        else:
            print(f"🧵 单进程顺序处理: tasks={len(tasks)}")
            old_verbose = self.verbose
            self.verbose = True  # 单进程可以保持输出
            for i, task in enumerate(tasks, 1):
                print(f"[{i}/{len(tasks)}] {task['patient_id']} - {task['action_name']}")
                try:
                    r = self.analyze_single_video(
                        video_path=task["video_path"],
                        action_name=task["action_name"],
                        output_dir=task["output_dir"],
                        start_frame=task["start_frame"],
                        end_frame=task["end_frame"],
                        fps=task["fps"],
                        patient_id=task["patient_id"],
                        examination_id=task["examination_id"],
                        video_id=task["video_id"],
                    )
                    results.append(r)
                except Exception as e:
                    errors.append({"video_id": task["video_id"], "video_path": task["video_path"], "error": str(e)})
                    print(f"  ❌ 错误: {e}")
            self.verbose = old_verbose

        # 保存汇总
        summary = {
            "success": results,
            "errors": errors,
            "total_tasks": len(tasks),
            "success_count": len(results),
            "error_count": len(errors),
        }
        summary_path = os.path.join(output_dir, "batch_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n{'=' * 60}")
        print("✅ 批量处理完成!")
        print(f"   成功处理: {len(results)}/{len(tasks)}")
        print(f"   失败: {len(errors)}")
        print(f"   汇总结果: {summary_path}")
        print(f"{'=' * 60}\n")

        return results

    def compare_actions(
        self,
        video_paths: Dict[str, str],
        patient_id: str,
        output_path: str
    ):
        """比较同一患者不同动作的对称性（保留原有功能）"""
        print(f"\n{'=' * 60}")
        print(f"动作对比分析: {patient_id}")
        print(f"{'=' * 60}")

        features_dict = {}
        for action_name, video_path in video_paths.items():
            print(f"\n处理动作: {action_name}")
            left_coords, right_coords = self.extract_landmarks_from_video(video_path)
            features = self.calculate_pearson_coefficients(left_coords, right_coords)
            features_dict[action_name] = features

        n_actions = len(features_dict)
        fig, axes = plt.subplots(1, n_actions, figsize=(6 * n_actions, 6))
        if n_actions == 1:
            axes = [axes]

        for idx, (action_name, features) in enumerate(features_dict.items()):
            ax = axes[idx]
            coeffs = features.pearson_coefficients
            colors = []
            for coef in coeffs:
                if np.isnan(coef):
                    colors.append('gray')
                elif coef > 0.8:
                    colors.append('darkblue')
                elif coef > 0.5:
                    colors.append('lightblue')
                elif coef > 0:
                    colors.append('yellow')
                else:
                    colors.append('red')

            ax.barh(range(self.n_features), coeffs, color=colors)
            ax.set_yticks(range(self.n_features))
            ax.set_yticklabels(features.landmark_names, fontsize=8)
            ax.set_xlabel('Pearson Coefficient', fontsize=10)
            ax.set_title(f'{action_name}\n平均: {np.nanmean(coeffs):.3f}',
                         fontsize=11, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
            ax.axvline(x=0.8, color='green', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.set_xlim(-1, 1)
            ax.grid(axis='x', alpha=0.3)

        plt.suptitle(f'患者 {patient_id} - 不同动作对称性对比',
                     fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"\n✓ 对比图已保存: {output_path}")


# ==================== multiprocessing worker ====================

def _worker_analyze_one(task: Dict):
    """
    子进程 worker：不要依赖主进程的 analyzer（不可pickle）
    """
    try:
        analyzer = FacialSymmetryAnalyzer(
            db_path=task["db_path"],
            model_path=task["model_path"],
            verbose=False  # 子进程别刷屏
        )
        res = analyzer.analyze_single_video(
            video_path=task["video_path"],
            action_name=task["action_name"],
            output_dir=task["output_dir"],
            start_frame=task["start_frame"],
            end_frame=task["end_frame"],
            fps=task["fps"],
            patient_id=task["patient_id"],
            examination_id=task["examination_id"],
            video_id=task.get("video_id"),
        )
        return True, res
    except Exception as e:
        return False, {
            "video_id": task.get("video_id"),
            "video_path": task.get("video_path"),
            "patient_id": task.get("patient_id"),
            "examination_id": task.get("examination_id"),
            "action_name": task.get("action_name"),
            "error": str(e),
        }


def main():
    analyzer = FacialSymmetryAnalyzer(
        db_path='/Users/cuijinglei/PycharmProjects/medicalProject/facialPalsy/facialPalsy.db'
    )

    analyzer.batch_process_database(
        output_dir='/Users/cuijinglei/Documents/facialPalsy/HGFA/symmetry_analysis',
        limit=None,
        action_filter=None,
        use_multiprocessing=True,
        num_workers=8,
    )


if __name__ == '__main__':
    main()
