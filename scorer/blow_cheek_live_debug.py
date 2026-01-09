# -*- coding: utf-8 -*-
import os
import time
from pathlib import Path
from collections import deque

import cv2
import numpy as np

# 兼容：同目录 / 包内导入
try:
    from clinical_base import LandmarkExtractor, draw_landmarks, LM
except Exception:
    from facial_palsy.scorer.clinical_base import LandmarkExtractor, draw_landmarks, LM


# =========================
# 配置区
# =========================
CAM_INDEX = 0
MAX_POINTS = 300                 # 曲线最多保留多少点
AUTO_BASELINE_FRAMES = 20        # 自动 baseline 使用前多少帧
SCREENSHOT_DIRNAME = "screenshots"
MODEL_PATH = "/Users/cuijinglei/PycharmProjects/medicalProject/models/face_landmarker.task"

# 画关键点：鼻尖 + 左右脸颊
DRAW_INDICES = [LM.NOSE_TIP] + LM.BLOW_CHEEK_L + LM.BLOW_CHEEK_R


def _mean_z(landmarks, indices):
    """计算一组 landmarks 的 z 均值（landmarks[i].z）"""
    zs = [landmarks[i].z for i in indices]
    return float(np.mean(zs)) if zs else float("nan")


def _median(values):
    values = [v for v in values if np.isfinite(v)]
    if not values:
        return None
    return float(np.median(values))


def _draw_curve_panel(img, left_vals, right_vals, title="Delta (baseline - current)"):
    """
    用 OpenCV 在图像下方画简单曲线（避免 matplotlib 在不同平台出怪问题）。
    left/right_vals: list[float]
    """
    h, w = img.shape[:2]
    panel_h = 160
    out = np.zeros((h + panel_h, w, 3), dtype=np.uint8)
    out[:h, :w] = img

    # panel background
    y0 = h
    cv2.rectangle(out, (0, y0), (w, h + panel_h), (20, 20, 20), -1)
    cv2.putText(out, title, (10, y0 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

    if len(left_vals) < 2 and len(right_vals) < 2:
        return out

    # 取最近 N 点
    N = min(len(left_vals), len(right_vals), 250)
    lv = np.array(left_vals[-N:], dtype=np.float32)
    rv = np.array(right_vals[-N:], dtype=np.float32)

    # 合并取范围，避免只画一条时缩放离谱
    allv = np.concatenate([lv[np.isfinite(lv)], rv[np.isfinite(rv)]], axis=0)
    if allv.size == 0:
        return out

    vmin = float(np.min(allv))
    vmax = float(np.max(allv))
    if abs(vmax - vmin) < 1e-6:
        vmax = vmin + 1e-6

    # 绘图区域
    px0, py0 = 10, y0 + 40
    px1, py1 = w - 10, h + panel_h - 20
    cv2.rectangle(out, (px0, py0), (px1, py1), (80, 80, 80), 1)

    def to_xy(i, v):
        x = int(px0 + (px1 - px0) * (i / max(1, N - 1)))
        y = int(py1 - (py1 - py0) * ((v - vmin) / (vmax - vmin)))
        return x, y

    # 画 left（黄色）
    pts_l = []
    for i, v in enumerate(lv):
        if np.isfinite(v):
            pts_l.append(to_xy(i, float(v)))
    if len(pts_l) >= 2:
        cv2.polylines(out, [np.array(pts_l, dtype=np.int32)], False, (0, 255, 255), 2)
        cv2.putText(out, "Left", (px0 + 5, py0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # 画 right（青色）
    pts_r = []
    for i, v in enumerate(rv):
        if np.isfinite(v):
            pts_r.append(to_xy(i, float(v)))
    if len(pts_r) >= 2:
        cv2.polylines(out, [np.array(pts_r, dtype=np.int32)], False, (255, 255, 0), 2)
        cv2.putText(out, "Right", (px0 + 80, py0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # 显示当前值
    cur_l = left_vals[-1] if left_vals else float("nan")
    cur_r = right_vals[-1] if right_vals else float("nan")
    txt = f"L={cur_l:+.4f}  R={cur_r:+.4f}  (b:baseline  r:reset)"
    cv2.putText(out, txt, (10, h + panel_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
    return out


class BlowCheekLiveDebugRelativeNose:
    """
    鼓腮 Live Debug:
    - 左右脸颊：LM.BLOW_CHEEK_L / LM.BLOW_CHEEK_R
    - 基准点：LM.NOSE_TIP
    - 相对量：rel = mean_z(cheek) - z(nose)
    - 输出曲线：delta = baseline_rel - current_rel  (鼓起 -> 通常 delta 变大更直观)
    """

    def __init__(self):
        self.extractor = LandmarkExtractor(MODEL_PATH)
        self.left_delta = deque(maxlen=MAX_POINTS)
        self.right_delta = deque(maxlen=MAX_POINTS)

        self._baseline_rel_l = None
        self._baseline_rel_r = None
        self._auto_buf_l = []
        self._auto_buf_r = []

        self._shot_dir = None

    def reset(self):
        self.left_delta.clear()
        self.right_delta.clear()
        self._baseline_rel_l = None
        self._baseline_rel_r = None
        self._auto_buf_l.clear()
        self._auto_buf_r.clear()

    def set_baseline_from_recent(self, recent_rel_l, recent_rel_r):
        b_l = _median(recent_rel_l)
        b_r = _median(recent_rel_r)
        if b_l is not None and b_r is not None:
            self._baseline_rel_l = b_l
            self._baseline_rel_r = b_r
            return True
        return False

    def _ensure_screenshot_dir(self):
        if self._shot_dir is None:
            root = Path.cwd()
            self._shot_dir = root / SCREENSHOT_DIRNAME
            self._shot_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        cap = cv2.VideoCapture(CAM_INDEX)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index={CAM_INDEX}")

        print("=" * 60)
        print("BlowCheek Live Debug Tool (Relative Nose Depth)")
        print("=" * 60)
        print("快捷键:")
        print("  'b' - 设置 baseline（建议放松脸时按一次）")
        print("  'r' - 重置曲线/重新自校准")
        print("  's' - 截图保存")
        print("  'q' - 退出")
        print("=" * 60)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            h, w = frame.shape[:2]
            landmarks = self.extractor.extract(frame)

            if landmarks is None:
                # 没检测到脸就画空曲线
                out = _draw_curve_panel(frame, list(self.left_delta), list(self.right_delta))
                cv2.imshow("BlowCheek Live Debug", out)
            else:
                # --- 计算 nose-referenced 相对深度 ---
                nose_z = float(landmarks[LM.NOSE_TIP].z)
                lz = _mean_z(landmarks, LM.BLOW_CHEEK_L)
                rz = _mean_z(landmarks, LM.BLOW_CHEEK_R)

                rel_l = lz - nose_z
                rel_r = rz - nose_z

                # 自动 baseline：前 AUTO_BASELINE_FRAMES 帧收集
                if self._baseline_rel_l is None or self._baseline_rel_r is None:
                    if np.isfinite(rel_l) and np.isfinite(rel_r):
                        self._auto_buf_l.append(rel_l)
                        self._auto_buf_r.append(rel_r)
                    if len(self._auto_buf_l) >= AUTO_BASELINE_FRAMES:
                        self.set_baseline_from_recent(self._auto_buf_l, self._auto_buf_r)

                # delta：baseline - current（鼓起时一般会更“正向”）
                if self._baseline_rel_l is not None and np.isfinite(rel_l):
                    delta_l = self._baseline_rel_l - rel_l
                else:
                    delta_l = float("nan")

                if self._baseline_rel_r is not None and np.isfinite(rel_r):
                    delta_r = self._baseline_rel_r - rel_r
                else:
                    delta_r = float("nan")

                self.left_delta.append(delta_l)
                self.right_delta.append(delta_r)

                # --- 画关键点（修复点：必须传 indices） ---
                draw_landmarks(frame, landmarks, w, h, DRAW_INDICES, color=(0, 255, 255), radius=2)

                # --- 画曲线面板 ---
                out = _draw_curve_panel(frame, list(self.left_delta), list(self.right_delta))
                cv2.imshow("BlowCheek Live Debug", out)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                self.reset()
            elif key == ord("b"):
                # 用最近 30 个有效 rel 估 baseline（更稳）
                # 注意：这里用 delta 反推不方便，所以直接用 auto buf 逻辑：重新收集一段
                self._baseline_rel_l = None
                self._baseline_rel_r = None
                self._auto_buf_l.clear()
                self._auto_buf_r.clear()
                # 连续收集几帧后会自动设 baseline
            elif key == ord("s"):
                self._ensure_screenshot_dir()
                ts = time.strftime("%Y%m%d_%H%M%S")
                out_path = self._shot_dir / f"blow_cheek_live_{ts}.png"
                # 保存当前窗口画面（带曲线面板）
                if "out" in locals() and isinstance(out, np.ndarray):
                    cv2.imwrite(str(out_path), out)
                else:
                    cv2.imwrite(str(out_path), frame)
                print(f"[OK] saved: {out_path}")

        cap.release()
        cv2.destroyAllWindows()


def main():
    dbg = BlowCheekLiveDebugRelativeNose()
    dbg.run()


if __name__ == "__main__":
    main()