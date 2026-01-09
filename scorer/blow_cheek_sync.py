# -*- coding: utf-8 -*-
"""
debug_blowcheek_nose_ref_sync.py

离线调试：指定鼓腮视频 -> 同步显示
1) 左侧：关键点叠加视频
2) 右侧：鼓腮曲线实时同步绘制（左/右/平均）
3) 曲线定义：鼓腮视频内部的相对深度变化
   rel = mean_z(cheek) - z(nose_tip)
   bulge = (base_rel - rel) / base_icd
   （曲线越高 = 脸颊相对鼻尖越靠前）

快捷键：
  Space：暂停/继续
  N：下一个视频
  Q / ESC：退出
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


# -------------------------
# 你需要改这里
# -------------------------
MODEL_PATH = "/Users/cuijinglei/PycharmProjects/medicalProject/models/face_landmarker.task"

VIDEO_JOBS = [
    {
        "name": "XW000085",
        "blowcheek_video": "/Users/cuijinglei/Documents/facialPalsy/videos/XW000085/XW000085_20230904_09-48-59/C7393.MP4",
    },
    {
        "name": "XW000288",
        "blowcheek_video": "/Users/cuijinglei/Documents/facialPalsy/videos/XW000288/XW000288_20240603_08-47/C9228.MP4",
    },
    # 继续添加...
]

# 用视频前多少帧建立“本视频内部基准”
BASELINE_FRAMES = 10

# 曲线面板尺寸（你嫌窄就加大）
PLOT_W = 1100
PLOT_H = 520

# 视频面板显示高度上限（仅影响显示，不影响计算）
MAX_VIEW_H = 720


# -------------------------
# 自动补 sys.path：确保能 import 你项目里的 clinical_base
# -------------------------
def _ensure_project_import():
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "facialPalsy").exists():
            sys.path.insert(0, str(p))
            return
_ensure_project_import()


# -------------------------
# import 你最新版 clinical_base（含 LM / LandmarkExtractor）
# -------------------------
try:
    from clinical_base import (
        LM,
        LandmarkExtractor,
    )
except Exception as e:
    raise RuntimeError(
        "导入失败：请把本文件放到项目内(能找到 facialPalsy/)，或手动把项目根目录加到 PYTHONPATH。\n"
        f"原始错误：{e}"
    )


# -------------------------
# 工具函数
# -------------------------
def lm_norm3d(landmarks, idx: int) -> Optional[np.ndarray]:
    """归一化3D坐标 (x,y,z)，不乘 w/h，避免分辨率差异"""
    if landmarks is None:
        return None
    if idx < 0 or idx >= len(landmarks):
        return None
    lm = landmarks[idx]
    return np.array([lm.x, lm.y, lm.z], dtype=np.float64)


def lms_norm3d(landmarks, indices: List[int]) -> Optional[np.ndarray]:
    pts = []
    for i in indices:
        p = lm_norm3d(landmarks, i)
        if p is None:
            return None
        pts.append(p)
    return np.stack(pts, axis=0) if pts else None


def icd_norm(landmarks) -> float:
    """双眼内眦距离（归一化2D），用于尺度归一化"""
    pL = lm_norm3d(landmarks, LM.EYE_INNER_L)
    pR = lm_norm3d(landmarks, LM.EYE_INNER_R)
    if pL is None or pR is None:
        return 1e-9
    return float(np.linalg.norm(pL[:2] - pR[:2])) + 1e-9


def to_px_xy(landmarks, idx: int, w: int, h: int) -> Optional[Tuple[int, int]]:
    if landmarks is None:
        return None
    if idx < 0 or idx >= len(landmarks):
        return None
    lm = landmarks[idx]
    return int(lm.x * w), int(lm.y * h)


def draw_poly(img, landmarks, indices: List[int], w: int, h: int, color, thickness=2):
    pts = []
    for i in indices:
        p = to_px_xy(landmarks, i, w, h)
        if p is None:
            return
        pts.append(p)
    if len(pts) >= 2:
        cv2.polylines(img, [np.array(pts, dtype=np.int32)], isClosed=True, color=color, thickness=thickness)


def draw_points(img, landmarks, indices: List[int], w: int, h: int, color, r=3):
    for i in indices:
        p = to_px_xy(landmarks, i, w, h)
        if p is None:
            continue
        cv2.circle(img, p, r, color, -1)


def make_plot_canvas(w=PLOT_W, h=PLOT_H) -> np.ndarray:
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    cv2.rectangle(canvas, (50, 30), (w - 30, h - 60), (220, 220, 220), 1)
    return canvas


def draw_curve_panel(
    left_vals: List[float],
    right_vals: List[float],
    mean_vals: List[float],
    cur_idx: int,
    best_idx: Optional[int],
    title: str
) -> np.ndarray:
    canvas = make_plot_canvas(PLOT_W, PLOT_H)
    h, w = canvas.shape[:2]

    cv2.putText(canvas, title, (50, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)

    n = len(mean_vals)
    if n < 2:
        return canvas

    all_y = np.array(mean_vals, dtype=np.float32)
    y_min = float(np.min(all_y))
    y_max = float(np.max(all_y))
    pad = (y_max - y_min) * 0.15 + 1e-6
    y_min -= pad
    y_max += pad

    x0, y0 = 50, 30
    x1, y1 = w - 30, h - 60
    pw = x1 - x0
    ph = y1 - y0

    def x_map(i: int) -> int:
        return int(x0 + (i / max(1, n - 1)) * pw)

    def y_map(v: float) -> int:
        t = (v - y_min) / (y_max - y_min)
        return int(y1 - t * ph)

    # 颜色（清爽点）
    col_left = (40, 140, 210)   # 蓝
    col_right = (40, 160, 80)   # 绿
    col_mean = (180, 80, 190)   # 紫

    def draw_line(vals: List[float], color):
        pts = [(x_map(i), y_map(vals[i])) for i in range(len(vals))]
        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i - 1], pts[i], color, 2)

    draw_line(left_vals, col_left)
    draw_line(right_vals, col_right)
    draw_line(mean_vals, col_mean)

    # 当前帧线
    xi = x_map(cur_idx)
    cv2.line(canvas, (xi, y0), (xi, y1), (160, 160, 160), 1)
    cv2.putText(canvas, f"cur={cur_idx}", (xi + 6, y0 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 100), 1)

    # 峰值帧线（红）
    if best_idx is not None and 0 <= best_idx < n:
        xb = x_map(best_idx)
        cv2.line(canvas, (xb, y0), (xb, y1), (0, 0, 255), 2)
        cv2.putText(canvas, f"peak={best_idx}", (xb + 6, y0 + 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    # legend
    lx, ly = x0 + 10, y1 + 35
    cv2.putText(canvas, "Left bulge", (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col_left, 2)
    cv2.putText(canvas, "Right bulge", (lx + 160, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col_right, 2)
    cv2.putText(canvas, "Mean", (lx + 340, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col_mean, 2)

    return canvas


def compose_view(frame: np.ndarray, plot: np.ndarray) -> np.ndarray:
    fh, fw = frame.shape[:2]
    ph, pw = plot.shape[:2]

    if fh > MAX_VIEW_H:
        scale = MAX_VIEW_H / fh
        frame = cv2.resize(frame, (int(fw * scale), int(fh * scale)), interpolation=cv2.INTER_AREA)
        fh, fw = frame.shape[:2]

    if ph != fh:
        plot = cv2.resize(plot, (pw, fh), interpolation=cv2.INTER_AREA)
        ph, pw = plot.shape[:2]

    canvas = np.zeros((fh, fw + pw, 3), dtype=np.uint8)
    canvas[:, :fw] = frame
    canvas[:, fw:fw + pw] = plot
    return canvas


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] model not found: {MODEL_PATH}")
        return
    if not VIDEO_JOBS:
        print("[ERROR] VIDEO_JOBS is empty.")
        return

    print("====================================================")
    print("BlowCheek Nose-Ref Sync Debugger")
    print("====================================================")
    print("Keys: SPACE pause/resume | N next video | Q/ESC quit")
    print("----------------------------------------------------")

    with LandmarkExtractor(MODEL_PATH) as extractor:
        for job_idx, job in enumerate(VIDEO_JOBS):
            name = job.get("name", f"job{job_idx}")
            video_path = job["blowcheek_video"]

            if not os.path.exists(video_path):
                print(f"[SKIP] video missing: {video_path}")
                continue

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"[SKIP] cannot open video: {video_path}")
                continue

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

            print(f"\n[JOB {job_idx+1}/{len(VIDEO_JOBS)}] {name}")
            print(f"  video : {video_path}")
            print(f"  frames={total}, fps={fps:.2f}")

            # 曲线数据（只存标量，不存帧，内存很小）
            left_vals, right_vals, mean_vals = [], [], []
            paused = False
            cur_idx = -1

            # 本视频内部 baseline（前 BASELINE_FRAMES 帧平均）
            base_ready = False
            base_relL = 0.0
            base_relR = 0.0
            base_icd = 1e-9
            _tmp_relL = []
            _tmp_relR = []
            _tmp_icd = []

            # 峰值：取 mean 曲线最高点
            best_idx = None
            best_score = -1e18

            win = f"BlowCheekSync(NoseRef) - {name}"
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)

            while True:
                if not paused:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    cur_idx += 1

                    h, w = frame.shape[:2]
                    lms = extractor.extract_from_frame(frame)

                    scoreL = scoreR = scoreM = 0.0

                    if lms is not None:
                        # nose tip z
                        nose = lm_norm3d(lms, LM.NOSE_TIP)
                        cheekL = lms_norm3d(lms, LM.BLOW_CHEEK_L)
                        cheekR = lms_norm3d(lms, LM.BLOW_CHEEK_R)
                        icd = icd_norm(lms)

                        if nose is not None and cheekL is not None and cheekR is not None:
                            nose_z = float(nose[2])
                            relL = float(np.mean(cheekL[:, 2]) - nose_z)
                            relR = float(np.mean(cheekR[:, 2]) - nose_z)

                            # 建 baseline（仅用前 N 帧）
                            if not base_ready:
                                _tmp_relL.append(relL)
                                _tmp_relR.append(relR)
                                _tmp_icd.append(icd)
                                if len(_tmp_relL) >= BASELINE_FRAMES:
                                    base_relL = float(np.mean(_tmp_relL))
                                    base_relR = float(np.mean(_tmp_relR))
                                    base_icd = float(np.mean(_tmp_icd)) + 1e-9
                                    base_ready = True

                            # bulge 定义：相对鼻尖的“更靠前”变化（曲线越高越鼓）
                            if base_ready:
                                scoreL = (base_relL - relL) / base_icd
                                scoreR = (base_relR - relR) / base_icd
                                scoreM = 0.5 * (scoreL + scoreR)

                                # 峰值：最高点
                                if scoreM > best_score:
                                    best_score = scoreM
                                    best_idx = cur_idx

                            # 叠加显示：脸颊圈 + 鼻尖点 + 眼内眦
                            draw_poly(frame, lms, LM.BLOW_CHEEK_L, w, h, color=(40, 140, 210), thickness=2)
                            draw_poly(frame, lms, LM.BLOW_CHEEK_R, w, h, color=(40, 160, 80), thickness=2)
                            draw_points(frame, lms, [LM.NOSE_TIP], w, h, color=(0, 0, 255), r=4)
                            draw_points(frame, lms, [LM.EYE_INNER_L, LM.EYE_INNER_R], w, h, color=(0, 0, 255), r=3)

                    left_vals.append(scoreL)
                    right_vals.append(scoreR)
                    mean_vals.append(scoreM)

                    # 文字信息
                    if base_ready:
                        base_txt = f"BASE ready({BASELINE_FRAMES}) icd={base_icd:.4f}"
                    else:
                        base_txt = f"Building BASE... {len(_tmp_relL)}/{BASELINE_FRAMES}"

                    cv2.putText(frame, f"frame {cur_idx}/{max(total-1,0)}  peak={best_idx}  mean={scoreM:.4f}",
                                (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 2)
                    cv2.putText(frame, base_txt,
                                (15, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)
                    cv2.putText(frame, f"L={scoreL:.4f}  R={scoreR:.4f}",
                                (15, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)

                    plot = draw_curve_panel(
                        left_vals, right_vals, mean_vals,
                        cur_idx=cur_idx,
                        best_idx=best_idx,
                        title="BlowCheek bulge curve (nose-ref): bulge=(base_rel - rel)/ICD , rel=mean_z(cheek)-z(nose)  [higher=more bulge]"
                    )

                    view = compose_view(frame, plot)
                    cv2.imshow(win, view)

                key = cv2.waitKey(1 if not paused else 30) & 0xFF
                if key in [27, ord('q'), ord('Q')]:
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                if key == ord(' '):
                    paused = not paused
                if key in [ord('n'), ord('N')]:
                    break

            cap.release()
            cv2.destroyAllWindows()

    print("\n[OK] All done.")


if __name__ == "__main__":
    main()