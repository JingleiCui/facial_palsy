# -*- coding: utf-8 -*-
"""
run_actions_interpretability_demo.py

功能
----
1) 读取一组“11动作”视频（两种模式：从数据库读取 / 从文件夹读取）
2) 调用 actions/ 下对应的 11 个动作类做几何分析（不写数据库）
3) 把“可解释性 + 可视化”结果输出到文件夹，方便你人工检查算法是否正确

运行方式
--------
- 直接在 PyCharm 里点击 Run 本文件（不需要命令行参数）
- 先在下面【配置区】把 DB_PATH / MODEL_PATH / OUTPUT_DIR 改成你自己的路径即可

输出结构（示例）
----------------
OUTPUT_DIR/
  <session_id or examination_id>/
    overview.html                  # 一页总览（浏览器打开）
    diagnosis_flow.md              # “诊断流程倒推：需要哪些指标”
    summary.csv                    # 每个动作的关键指标摘要（Excel 直接看）
    _meta.json                     # 本次 session/exam 的元信息
    all_actions_summary.json       # 所有动作的汇总 JSON
    actions/
      NeutralFace/
        peak_raw.jpg
        peak_vis.jpg
        indicators.json
        dynamic_features.json
        interpretability.json
        metrics.md                # 指标“人话解释 + 正负号/比例说明”
        plot_*.png                # 若有曲线则自动生成
      ...

注意
----
- 本脚本只读数据库，不写 video_features / interpretability 等任何列。
- 为了可解释性，会保存曲线与峰值帧，可用于你逐个动作核对。
"""

from __future__ import annotations
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import sys
import json
import csv
import math
import sqlite3
from pathlib import Path
from dataclasses import asdict, is_dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import cv2

# matplotlib 用于曲线图（无 GUI 环境也能保存）
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# 让 “import facialPalsy.xxx” 在 PyCharm 各种 working directory 下都稳
# =============================================================================
_THIS_DIR = Path(__file__).resolve().parent          # .../facialPalsy
_PROJECT_ROOT = _THIS_DIR.parent                    # .../medicalProject
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# =============================================================================
# 配置区（你只需要改这里）
# =============================================================================
# 并行：NeutralFace 之后，其余动作是否多进程并行
PARALLEL_AFTER_NEUTRAL = True

NUM_WORKERS = 8

# 运行模式： "db" 从 SQLite 数据库读取；"folder" 从文件夹读取（适合 phone_videos）
MODE = "db"  # "db" or "folder"

# —— DB 模式配置 —— #
DB_PATH = str(_THIS_DIR / "facialPalsy.db")  # 默认：facialPalsy/facialPalsy.db

# 指定只分析某一个 examination_id（None = 自动按时间倒序取 MAX_EXAMS 个）
TARGET_EXAMINATION_ID: Optional[str] = None

# 最多处理多少个 examination（None=不限制；建议先 5~10 做可解释性检查）
MAX_EXAMS: Optional[int] = None

# —— FOLDER 模式配置 —— #
# INPUT_VIDEO_ROOT 下每个子文件夹视为一个“session”，里面放动作视频：NeutralFace.mp4 等
INPUT_VIDEO_ROOT = "/Users/cuijinglei/Documents/facialPalsy/phone_videos"

# —— 通用配置 —— #
# MediaPipe FaceLandmarker 模型路径
MODEL_PATH = r"/Users/cuijinglei/PycharmProjects/medicalProject/models/face_landmarker.task"

# 输出目录
OUTPUT_DIR = r"/Users/cuijinglei/Documents/facialPalsy/HGFA/actions_interpretability_outputs"

# 是否缩小保存的峰值帧（HTML 打开更快）
SAVE_THUMBNAIL = True
THUMB_MAX_W = 960

# 如果视频特别长，你可以限制最大抽帧数（None=不限制）
MAX_FRAMES_PER_VIDEO: Optional[int] = None  # 例如 300


# =============================================================================
# 导入项目内模块（与 video_pipeline.py 同风格）
# =============================================================================
from facialPalsy.core.landmark_extractor import LandmarkExtractor
from facialPalsy.core.constants import ActionNames

from facialPalsy.actions.neutral_face import NeutralFaceAction
from facialPalsy.actions.spontaneous_eye_blink import SpontaneousEyeBlinkAction
from facialPalsy.actions.voluntary_eye_blink import VoluntaryEyeBlinkAction
from facialPalsy.actions.close_eye_softly import CloseEyeSoftlyAction
from facialPalsy.actions.close_eye_hardly import CloseEyeHardlyAction
from facialPalsy.actions.raise_eyebrow import RaiseEyebrowAction
from facialPalsy.actions.smile import SmileAction
from facialPalsy.actions.shrug_nose import ShrugNoseAction
from facialPalsy.actions.show_teeth import ShowTeethAction
from facialPalsy.actions.blow_cheek import BlowCheekAction
from facialPalsy.actions.lip_pucker import LipPuckerAction


# =============================================================================
# 工具函数
# =============================================================================
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _jsonable(x: Any) -> Any:
    """把 numpy / dataclass / Path 等对象递归转成可 JSON 序列化形式。"""
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return str(x)
        return x
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.float32, np.float64, np.float16)):
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return str(v)
        return v
    if isinstance(x, (np.int32, np.int64, np.int16, np.uint8)):
        return int(x)
    if is_dataclass(x):
        return _jsonable(asdict(x))
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return str(x)


def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(obj), f, ensure_ascii=False, indent=2)


def imwrite(path: Path, bgr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), bgr)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed: {path}")


def resize_keep_aspect(img: np.ndarray, max_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / float(w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def plot_two_curves(
    save_path: Path,
    y1: np.ndarray,
    y2: np.ndarray,
    label1: str,
    label2: str,
    title: str,
    ylabel: str,
    vline_idx: Optional[int] = None,
    spans: Optional[List[Tuple[int, int]]] = None
) -> None:
    x = np.arange(len(y1))
    plt.figure()
    plt.plot(x, y1, label=label1)
    plt.plot(x, y2, label=label2)
    if vline_idx is not None:
        plt.axvline(vline_idx, linestyle="--")
    if spans:
        for (s, e) in spans:
            plt.axvspan(s, e, alpha=0.2)
    plt.title(title)
    plt.xlabel("Frame Index")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=150)
    plt.close()


def plot_one_curve(
    save_path: Path,
    y: np.ndarray,
    label: str,
    title: str,
    ylabel: str,
    vline_idx: Optional[int] = None,
) -> None:
    x = np.arange(len(y))
    plt.figure()
    plt.plot(x, y, label=label)
    if vline_idx is not None:
        plt.axvline(vline_idx, linestyle="--")
    plt.title(title)
    plt.xlabel("Frame Index")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=150)
    plt.close()


def metric_sign_hint(key: str) -> str:
    """
    统一解释“正负号/比例”的读法：
    - *_ratio：>1 左>右，<1 左<右，≈1 对称
    - *_asymmetry：通常 >=0，越接近 0 越对称
    - *_diff（例如 oral_angle_diff = left - right）：可能有正负，正=左>右，负=右>左
    """
    k = key.lower()
    if "ratio" in k:
        return "（比例：>1 左>右，<1 左<右，≈1 对称）"
    if "asym" in k:
        return "（不对称：越接近 0 越对称）"
    if k.endswith("diff") or "angle_diff" in k:
        return "（方向差：正=左>右，负=右>左）"
    return ""


def pick_highlights(action_name: str, indicators: Dict[str, float], dynamic: Dict[str, float], interp: Dict[str, Any]) -> Dict[str, Any]:
    """给 overview.html / summary.csv 用的“关键指标”挑选（尽量少但够用）。"""
    h: Dict[str, Any] = {}
    if "function_pct" in indicators:
        h["function_pct"] = indicators["function_pct"]

    # 通用：左右比例/差异/联动
    for k in [
        "closure_ratio", "eye_asymmetry", "both_complete_closure",
        "left_complete_closure", "right_complete_closure",
        "lift_ratio", "lift_asymmetry",
        "oral_height_diff", "oral_angle_diff", "nlf_change_ratio",
        "cheek_asymmetry", "nostril_asymmetry",
        "mouth_aspect_ratio", "mouth_width_change", "face_width_change",
        "left_eye_synkinesis", "right_eye_synkinesis",
    ]:
        if k in indicators:
            h[k] = indicators[k]

    if "motion_asymmetry" in dynamic:
        h["motion_asymmetry"] = dynamic["motion_asymmetry"]

    # NeutralFace：静态对称 + Sunnybrook 表A示例
    if action_name == ActionNames.NEUTRAL_FACE:
        for k in ["eye_area_ratio", "nlf_length_ratio", "face_symmetry_score"]:
            if k in indicators:
                h[k] = indicators[k]
        if isinstance(interp.get("sunnybrook_static"), dict):
            h["sunnybrook_static"] = interp["sunnybrook_static"]

    # 眨眼：统计和一句话发现
    if isinstance(interp.get("blink_analysis"), dict):
        h["blink_analysis"] = interp["blink_analysis"]
    if isinstance(interp.get("key_findings"), list):
        h["key_findings"] = interp["key_findings"]

    return h


def write_metrics_md(path: Path, indicators: Dict[str, Any], dynamic: Dict[str, Any], interp: Dict[str, Any]) -> None:
    """把指标做成“人能读懂”的列表，方便你肉眼核对。"""
    lines = []
    lines.append("# 指标解释（便于核对）\n")

    if indicators:
        lines.append("## indicators（动作关键几何指标）\n")
        for k in sorted(indicators.keys()):
            v = indicators[k]
            lines.append(f"- `{k}` = {v} {metric_sign_hint(k)}")
        lines.append("")

    if dynamic:
        lines.append("## dynamic_features（动作的运动学特征）\n")
        for k in sorted(dynamic.keys()):
            v = dynamic[k]
            lines.append(f"- `{k}` = {v} {metric_sign_hint(k)}")
        lines.append("")

    if interp:
        lines.append("## interpretability（曲线/事件/关键发现）\n")
        # 只把“可读”的内容列出来，曲线本身放 plot_*.png
        for k in sorted(interp.keys()):
            if isinstance(interp[k], (list, dict, str, int, float, bool)) and k not in [
                "left_ear_curve", "right_ear_curve",
                "left_openness_curve", "right_openness_curve",
                "left_brow_curve", "right_brow_curve",
                "mouth_width_curve",
            ]:
                lines.append(f"- `{k}` = {json.dumps(_jsonable(interp[k]), ensure_ascii=False)}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# =============================================================================
# 11 动作顺序 + 实例
# =============================================================================
ACTION_ORDER = [
    ActionNames.NEUTRAL_FACE,
    ActionNames.SPONTANEOUS_EYE_BLINK,
    ActionNames.VOLUNTARY_EYE_BLINK,
    ActionNames.CLOSE_EYE_SOFTLY,
    ActionNames.CLOSE_EYE_HARDLY,
    ActionNames.RAISE_EYEBROW,
    ActionNames.SMILE,
    ActionNames.SHRUG_NOSE,
    ActionNames.SHOW_TEETH,
    ActionNames.BLOW_CHEEK,
    ActionNames.LIP_PUCKER,
]

ACTION_INSTANCES = {
    ActionNames.NEUTRAL_FACE: NeutralFaceAction(),
    ActionNames.SPONTANEOUS_EYE_BLINK: SpontaneousEyeBlinkAction(),
    ActionNames.VOLUNTARY_EYE_BLINK: VoluntaryEyeBlinkAction(),
    ActionNames.CLOSE_EYE_SOFTLY: CloseEyeSoftlyAction(),
    ActionNames.CLOSE_EYE_HARDLY: CloseEyeHardlyAction(),
    ActionNames.RAISE_EYEBROW: RaiseEyebrowAction(),
    ActionNames.SMILE: SmileAction(),
    ActionNames.SHRUG_NOSE: ShrugNoseAction(),
    ActionNames.SHOW_TEETH: ShowTeethAction(),
    ActionNames.BLOW_CHEEK: BlowCheekAction(),
    ActionNames.LIP_PUCKER: LipPuckerAction(),
}
ACTION_CLASS_MAP = {
    ActionNames.NEUTRAL_FACE: NeutralFaceAction,
    ActionNames.SPONTANEOUS_EYE_BLINK: SpontaneousEyeBlinkAction,
    ActionNames.VOLUNTARY_EYE_BLINK: VoluntaryEyeBlinkAction,
    ActionNames.CLOSE_EYE_SOFTLY: CloseEyeSoftlyAction,
    ActionNames.CLOSE_EYE_HARDLY: CloseEyeHardlyAction,
    ActionNames.RAISE_EYEBROW: RaiseEyebrowAction,
    ActionNames.SMILE: SmileAction,
    ActionNames.SHRUG_NOSE: ShrugNoseAction,
    ActionNames.SHOW_TEETH: ShowTeethAction,
    ActionNames.BLOW_CHEEK: BlowCheekAction,
    ActionNames.LIP_PUCKER: LipPuckerAction,
}


# =============================================================================
# 核心：读视频 → landmarks/frames → 动作分析 → 落盘
# =============================================================================
def extract_landmarks_and_frames(
    extractor: LandmarkExtractor,
    video_path: str,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    max_frames: Optional[int] = None
) -> Tuple[List[Any], List[np.ndarray], Dict[str, Any]]:
    info = extractor.get_video_info(video_path) or {}
    total = info.get("total_frames", None)

    s = int(start_frame or 0)
    if end_frame is None:
        e = (total - 1) if isinstance(total, int) and total > 0 else None
    else:
        e = int(end_frame)

    if e is not None and max_frames is not None and max_frames > 0:
        e = min(e, s + int(max_frames) - 1)

    landmarks_seq, frames_seq = extractor.extract_sequence(video_path, start_frame=s, end_frame=e)
    if landmarks_seq is None or frames_seq is None:
        return [], [], info
    return landmarks_seq, frames_seq, info


def run_one_action(
    action_name: str,
    extractor: LandmarkExtractor,
    video_path: str,
    out_action_dir: Path,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    fps_hint: Optional[float] = None,
    neutral_indicators: Optional[Dict[str, float]] = None
) -> Optional[Dict[str, Any]]:
    action = ACTION_CLASS_MAP[action_name]()  # 每次创建一个实例（无状态，安全）

    landmarks_seq, frames_seq, info = extract_landmarks_and_frames(
        extractor,
        video_path=video_path,
        start_frame=start_frame,
        end_frame=end_frame,
        max_frames=MAX_FRAMES_PER_VIDEO
    )
    if not landmarks_seq or not frames_seq:
        print(f"  ❌ {action_name}: 读取失败或无帧 {video_path}")
        return None

    w = int(info.get("width", frames_seq[0].shape[1]))
    h = int(info.get("height", frames_seq[0].shape[0]))
    fps = float(fps_hint or info.get("fps") or 30.0)

    result = action.process(
        landmarks_seq=landmarks_seq,
        frames_seq=frames_seq,
        w=w,
        h=h,
        fps=fps,
        neutral_indicators=neutral_indicators
    )
    if result is None:
        print(f"  ❌ {action_name}: process() 返回 None（可能关键点全缺失）")
        return None

    peak_idx = int(result.peak_frame_idx)
    peak_lm = landmarks_seq[peak_idx] if 0 <= peak_idx < len(landmarks_seq) else None
    peak_raw = result.peak_frame

    ensure_dir(out_action_dir)

    # 保存峰值帧 raw / vis
    raw_path = out_action_dir / "peak_raw.jpg"
    vis_path = out_action_dir / "peak_vis.jpg"

    raw_img = peak_raw
    if SAVE_THUMBNAIL:
        raw_img = resize_keep_aspect(raw_img, THUMB_MAX_W)
    imwrite(raw_path, raw_img)

    if peak_lm is not None:
        vis_img = action.visualize_peak_frame(
            frame=peak_raw, landmarks=peak_lm, indicators=result.indicators, w=w, h=h
        )
    else:
        vis_img = peak_raw.copy()

    if SAVE_THUMBNAIL:
        vis_img = resize_keep_aspect(vis_img, THUMB_MAX_W)
    imwrite(vis_path, vis_img)

    # 保存 JSON
    write_json(out_action_dir / "indicators.json", result.indicators)
    write_json(out_action_dir / "dynamic_features.json", result.dynamic_features)
    write_json(out_action_dir / "interpretability.json", result.interpretability)

    # 额外：指标“人话解释”
    write_metrics_md(out_action_dir / "metrics.md", result.indicators, result.dynamic_features, result.interpretability)

    # 自动生成曲线图（按 interpretability 里的常见 key）
    interp = result.interpretability or {}

    if "left_ear_curve" in interp and "right_ear_curve" in interp:
        l = np.asarray(interp["left_ear_curve"])
        r = np.asarray(interp["right_ear_curve"])
        spans = None
        if isinstance(interp.get("left_blink_events"), list):
            spans = []
            for e in interp["left_blink_events"]:
                try:
                    spans.append((int(e["start"]), int(e["end"])))
                except Exception:
                    pass
        plot_two_curves(
            save_path=out_action_dir / "plot_ear.png",
            y1=l, y2=r,
            label1="Left EAR", label2="Right EAR",
            title=f"{action_name} - EAR Curve",
            ylabel="EAR",
            vline_idx=peak_idx,
            spans=spans
        )

    if "left_openness_curve" in interp and "right_openness_curve" in interp:
        l = np.asarray(interp["left_openness_curve"])
        r = np.asarray(interp["right_openness_curve"])
        plot_two_curves(
            save_path=out_action_dir / "plot_openness.png",
            y1=l, y2=r,
            label1="Left Openness", label2="Right Openness",
            title=f"{action_name} - Openness Curve (baseline=Neutral)",
            ylabel="Openness (ratio)",
            vline_idx=peak_idx
        )

    if "left_brow_curve" in interp and "right_brow_curve" in interp:
        l = np.asarray(interp["left_brow_curve"])
        r = np.asarray(interp["right_brow_curve"])
        plot_two_curves(
            save_path=out_action_dir / "plot_brow.png",
            y1=l, y2=r,
            label1="Left Brow Height (norm)", label2="Right Brow Height (norm)",
            title=f"{action_name} - Brow Height Curve",
            ylabel="Brow Height (norm)",
            vline_idx=peak_idx
        )

    if "mouth_width_curve" in interp:
        y = np.asarray(interp["mouth_width_curve"])
        plot_one_curve(
            save_path=out_action_dir / "plot_mouth_width.png",
            y=y,
            label="Mouth Width (px)",
            title=f"{action_name} - Mouth Width Curve",
            ylabel="Pixels",
            vline_idx=peak_idx
        )

    # 汇总给 overview / summary.csv
    highlights = pick_highlights(action_name, result.indicators, result.dynamic_features, result.interpretability)
    payload = {
        "action_name": action_name,
        "video_path": video_path,
        "start_frame": int(start_frame or 0),
        "end_frame": int(end_frame) if end_frame is not None else None,
        "fps": fps,
        "w": w,
        "h": h,
        "peak_frame_idx": peak_idx,
        "unit_length_icd": float(result.unit_length),
        "highlights": highlights,
        "files": {
            "peak_raw": "peak_raw.jpg",
            "peak_vis": "peak_vis.jpg",
        }
    }
    write_json(out_action_dir / "summary.json", payload)
    return payload


_WORKER_EXTRACTOR = None

def _worker_init(model_path: str):
    """每个进程启动时初始化一次 MediaPipe Landmarker"""
    global _WORKER_EXTRACTOR
    _WORKER_EXTRACTOR = LandmarkExtractor(model_path)
    _WORKER_EXTRACTOR.__enter__()  # create landmarker

def _worker_run_one_action(task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """子进程执行单个动作（各自读视频、提关键点、分析、落盘）"""
    global _WORKER_EXTRACTOR

    return run_one_action(
        action_name=task["action_name"],
        extractor=_WORKER_EXTRACTOR,
        video_path=task["video_path"],
        out_action_dir=Path(task["out_action_dir"]),
        start_frame=task.get("start_frame", 0),
        end_frame=task.get("end_frame", None),
        fps_hint=task.get("fps_hint", None),
        neutral_indicators=task.get("neutral_indicators", None),
    )


# =============================================================================
# DB 模式：读取 examinations + video_files
# =============================================================================
def db_fetch_examinations(db_path: str, target_exam_id: Optional[str], limit: Optional[int]) -> List[Dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if target_exam_id:
        cursor.execute("""
            SELECT examination_id, patient_id, capture_datetime, has_labels, has_videos
            FROM examinations
            WHERE examination_id = ?
        """, (target_exam_id,))
    else:
        cursor.execute("""
            SELECT examination_id, patient_id, capture_datetime, has_labels, has_videos
            FROM examinations
            WHERE has_videos = 1 AND is_valid = 1
            ORDER BY capture_datetime DESC
        """)

    rows = cursor.fetchall()
    conn.close()

    exams = []
    for r in rows:
        exams.append({
            "examination_id": r[0],
            "patient_id": r[1],
            "capture_datetime": r[2],
            "has_labels": r[3],
            "has_videos": r[4],
        })

    if limit is not None:
        exams = exams[: int(limit)]
    return exams


def db_fetch_videos_for_exam(db_path: str, examination_id: str) -> Dict[str, Dict[str, Any]]:
    """
    返回 {action_name_en: video_info}
    如果一个动作有多个视频：取 video_file_index 最小的一条
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            v.video_id, v.action_id, v.file_path, v.start_frame, v.end_frame, v.fps, v.video_file_index,
            at.action_name_en, at.action_name_cn
        FROM video_files v
        LEFT JOIN action_types at ON v.action_id = at.action_id
        WHERE v.examination_id = ? AND v.file_exists = 1
        ORDER BY at.display_order ASC, v.video_file_index ASC
    """, (examination_id,))
    rows = cursor.fetchall()
    conn.close()

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for (video_id, action_id, file_path, start_frame, end_frame, fps, video_file_index, action_en, action_cn) in rows:
        if not action_en:
            continue
        action_en = str(action_en).strip()
        grouped.setdefault(action_en, []).append({
            "video_id": int(video_id),
            "action_id": int(action_id),
            "action_name_en": action_en,
            "action_name_cn": action_cn,
            "file_path": file_path,
            "start_frame": int(start_frame) if start_frame is not None else 0,
            "end_frame": int(end_frame) if end_frame is not None else None,
            "fps": float(fps) if fps is not None else None,
            "video_file_index": int(video_file_index) if video_file_index is not None else 0,
        })

    selected: Dict[str, Dict[str, Any]] = {}
    for action_en, candidates in grouped.items():
        candidates_sorted = sorted(candidates, key=lambda x: x.get("video_file_index", 0))
        selected[action_en] = candidates_sorted[0]
        selected[action_en]["all_candidates"] = candidates_sorted

    return selected


def db_fetch_labels(db_path: str, examination_id: str) -> Dict[str, Any]:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT has_palsy, palsy_side, hb_grade, sunnybrook_score
        FROM examination_labels
        WHERE examination_id = ?
    """, (examination_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return {}
    return {
        "has_palsy": row[0],
        "palsy_side": row[1],
        "hb_grade": row[2],
        "sunnybrook_score": row[3],
    }


# =============================================================================
# FOLDER 模式：读取 phone_videos/xxx/*.mp4
# =============================================================================
def folder_list_sessions(root_dir: str) -> List[Path]:
    root = Path(root_dir)
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()])


def folder_find_action_video(session_dir: Path, action_name: str) -> Optional[Path]:
    for ext in [".mp4", ".MP4", ".mov", ".MOV"]:
        p = session_dir / f"{action_name}{ext}"
        if p.exists():
            return p
    return None


# =============================================================================
# 报告生成：summary.csv / diagnosis_flow.md / overview.html
# =============================================================================
def write_summary_csv(path: Path, action_summaries: List[Dict[str, Any]]) -> None:
    keys = set()
    for a in action_summaries:
        for k in (a.get("highlights") or {}).keys():
            keys.add(k)
    keys = sorted(keys)

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["action_name"] + keys)
        for a in action_summaries:
            row = [a.get("action_name", "")]
            h = a.get("highlights") or {}
            for k in keys:
                row.append(_jsonable(h.get(k, "")))
            w.writerow(row)


def write_diagnosis_flow_md(path: Path, meta: Dict[str, Any], action_summaries: List[Dict[str, Any]]) -> None:
    by_name = {a["action_name"]: a for a in action_summaries}

    lines: List[str] = []
    lines.append("# 面瘫诊断流程倒推：本次分析用到哪些指标\n")
    lines.append(f"- session/examination_id: `{meta.get('id','')}`")
    if meta.get("patient_id"):
        lines.append(f"- patient_id: `{meta.get('patient_id')}`")
    if meta.get("capture_datetime"):
        lines.append(f"- capture_datetime: `{meta.get('capture_datetime')}`")
    if meta.get("labels"):
        lines.append(f"- labels: `{json.dumps(_jsonable(meta.get('labels')), ensure_ascii=False)}`")
    lines.append("")

    lines.append("## 0) 统一单位（归一化）\n")
    lines.append("- 所有关键距离/面积以 **两眼内眦距离 ICD** 作为单位长度归一化（你的 actions 代码里就是这样做的）。\n")

    lines.append("## 1) 静息 NeutralFace：先确定“基线 + 静态不对称”\n")
    if ActionNames.NEUTRAL_FACE in by_name:
        h = by_name[ActionNames.NEUTRAL_FACE].get("highlights", {})
        lines.append("- 你应该先看：`actions/NeutralFace/peak_vis.jpg` 是否画对区域；再看曲线 `plot_ear.png` 峰值帧是否落在极值处。")
        for k, v in h.items():
            lines.append(f"  - `{k}` = {v} {metric_sign_hint(k)}")
        lines.append("- `sunnybrook_static`：neutral_face.py 里给出表A（静态）阈值化示例（0=正常）。\n")
    else:
        lines.append("- ⚠️ 缺失 NeutralFace：后续变化量/联动的“基线”会不可靠。\n")

    lines.append("## 2) 眼部功能：眨眼 + 轻闭眼 + 用力闭眼\n")
    for an in [ActionNames.SPONTANEOUS_EYE_BLINK, ActionNames.VOLUNTARY_EYE_BLINK,
               ActionNames.CLOSE_EYE_SOFTLY, ActionNames.CLOSE_EYE_HARDLY]:
        if an in by_name:
            h = by_name[an].get("highlights", {})
            lines.append(f"### {an}")
            lines.append("- 建议你按顺序看：`peak_vis.jpg` → `plot_ear.png / plot_openness.png` → `metrics.md`")
            for k, v in h.items():
                lines.append(f"  - `{k}` = {v} {metric_sign_hint(k)}")
            lines.append("")
        else:
            lines.append(f"- ⚠️ 缺失视频：{an}")
    lines.append("")

    lines.append("## 3) 额肌 RaiseEyebrow：抬眉功能 + 眼部联动（synkinesis）\n")
    if ActionNames.RAISE_EYEBROW in by_name:
        h = by_name[ActionNames.RAISE_EYEBROW].get("highlights", {})
        lines.append("- 重点：`left/right_brow_lift`、`lift_ratio`、`function_pct`；联动看 `left/right_eye_synkinesis`（越大越异常）。")
        for k, v in h.items():
            lines.append(f"  - `{k}` = {v} {metric_sign_hint(k)}")
    else:
        lines.append("- ⚠️ 缺失视频：RaiseEyebrow")
    lines.append("")

    lines.append("## 4) 口周功能：Smile / ShowTeeth / LipPucker / BlowCheek / ShrugNose\n")
    for an in [ActionNames.SMILE, ActionNames.SHOW_TEETH, ActionNames.LIP_PUCKER, ActionNames.BLOW_CHEEK, ActionNames.SHRUG_NOSE]:
        if an in by_name:
            h = by_name[an].get("highlights", {})
            lines.append(f"### {an}")
            lines.append("- 重点：`function_pct` + 与该动作相关的 ratio/diff/asymmetry（看 metrics.md 更清晰）。")
            for k, v in h.items():
                lines.append(f"  - `{k}` = {v} {metric_sign_hint(k)}")
            lines.append("")
        else:
            lines.append(f"- ⚠️ 缺失视频：{an}")

    lines.append("\n## 5) 你人工核对算法是否正确：一套“固定检查顺序”\n")
    lines.append("1) 每个动作先看 `peak_vis.jpg`：画的区域对不对（眼轮廓/眉线/口角/鼻翼/颊部）。")
    lines.append("2) 再看 `plot_*.png`：峰值帧是否落在曲线极值（最大/最小）处；左右曲线趋势是否合理。")
    lines.append("3) 最后看 `metrics.md`：ratio/diff 的正负号和方向是否符合你肉眼观察。")

    path.write_text("\n".join(lines), encoding="utf-8")


def write_overview_html(path: Path, meta: Dict[str, Any], action_summaries: List[Dict[str, Any]]) -> None:
    rows_html = []
    for a in action_summaries:
        action_name = a["action_name"]
        rel_dir = f"actions/{action_name}"
        highlights = a.get("highlights") or {}

        li = []
        for k, v in highlights.items():
            li.append(f"<li><code>{k}</code>: {json.dumps(_jsonable(v), ensure_ascii=False)}</li>")
        ul = "<ul>" + "".join(li) + "</ul>" if li else ""

        rows_html.append(f"""
        <tr>
          <td><code>{action_name}</code></td>
          <td>{ul}</td>
          <td>
            <a href="{rel_dir}/peak_vis.jpg" target="_blank">peak_vis</a> |
            <a href="{rel_dir}/plot_ear.png" target="_blank">plot_ear</a> |
            <a href="{rel_dir}/plot_openness.png" target="_blank">plot_open</a> |
            <a href="{rel_dir}/metrics.md" target="_blank">metrics.md</a> |
            <a href="{rel_dir}/summary.json" target="_blank">summary.json</a>
          </td>
        </tr>
        """)

    html = f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8"/>
  <title>11 Actions Interpretability - {meta.get('id','')}</title>
  <style>
    body {{ font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial; margin: 24px; }}
    h1 {{ margin: 0 0 8px 0; }}
    .meta {{ color:#444; margin-bottom:16px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 10px; vertical-align: top; }}
    th {{ background: #f6f6f6; text-align:left; }}
    code {{ background:#f2f2f2; padding:2px 4px; border-radius:4px; }}
    ul {{ margin: 6px 0 0 18px; }}
  </style>
</head>
<body>
  <h1>11动作可解释性总览</h1>
  <div class="meta">
    <div><b>session/exam id</b>: <code>{meta.get('id','')}</code></div>
    {f"<div><b>patient</b>: <code>{meta.get('patient_id')}</code></div>" if meta.get('patient_id') else ""}
    {f"<div><b>capture_datetime</b>: <code>{meta.get('capture_datetime')}</code></div>" if meta.get('capture_datetime') else ""}
    {f"<div><b>labels</b>: <code>{json.dumps(_jsonable(meta.get('labels')), ensure_ascii=False)}</code></div>" if meta.get('labels') else ""}
    <div style="margin-top:8px;">
      <a href="diagnosis_flow.md" target="_blank">diagnosis_flow.md（流程+指标解释）</a> |
      <a href="summary.csv" target="_blank">summary.csv（表格摘要）</a>
    </div>
  </div>

  <table>
    <thead>
      <tr><th style="width:160px;">Action</th><th>Highlights（关键指标）</th><th style="width:360px;">Files</th></tr>
    </thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


# =============================================================================
# 主流程：跑一个 session/exam（先 NeutralFace 再其它动作）
# =============================================================================
def analyze_one_session(
    session_id: str,
    videos: Dict[str, Dict[str, Any]],
    session_out_dir: Path,
    extractor: LandmarkExtractor,
    meta: Dict[str, Any]
) -> None:
    ensure_dir(session_out_dir)
    ensure_dir(session_out_dir / "actions")

    summaries: List[Dict[str, Any]] = []

    # -----------------------
    # 1) NeutralFace 先跑（串行）
    # -----------------------
    neutral_indicators = None
    if ActionNames.NEUTRAL_FACE in videos:
        v = videos[ActionNames.NEUTRAL_FACE]
        video_path = v["file_path"]
        if video_path and os.path.exists(video_path):
            out_action_dir = session_out_dir / "actions" / ActionNames.NEUTRAL_FACE
            print(f"  ▶ NeutralFace (baseline): {Path(video_path).name}")

            t0 = time.time()
            payload = run_one_action(
                action_name=ActionNames.NEUTRAL_FACE,
                extractor=extractor,
                video_path=video_path,
                out_action_dir=out_action_dir,
                start_frame=v.get("start_frame", 0) or 0,
                end_frame=v.get("end_frame", None),
                fps_hint=v.get("fps", None),
                neutral_indicators=None
            )
            print(f"  ✅ NeutralFace done: {time.time()-t0:.2f}s")

            if payload is not None:
                summaries.append(payload)
                neutral_indicators = json.load((out_action_dir / "indicators.json").open("r", encoding="utf-8"))
        else:
            print(f"  ⚠️ NeutralFace 文件不存在：{video_path}")
    else:
        print("  ⚠️ 缺失 NeutralFace：后续动作将没有 baseline（不推荐）")

    # -----------------------
    # 2) 其它动作：并行（多 CPU）
    # -----------------------
    other_actions = [a for a in ACTION_ORDER if a != ActionNames.NEUTRAL_FACE and a in videos]

    tasks = []
    for action_name in other_actions:
        v = videos[action_name]
        video_path = v["file_path"]
        if not video_path or not os.path.exists(video_path):
            print(f"  ⚠️ 文件不存在: {action_name}: {video_path}")
            continue

        out_action_dir = session_out_dir / "actions" / action_name
        tasks.append({
            "action_name": action_name,
            "video_path": video_path,
            "out_action_dir": str(out_action_dir),
            "start_frame": v.get("start_frame", 0) or 0,
            "end_frame": v.get("end_frame", None),
            "fps_hint": v.get("fps", None),
            "neutral_indicators": neutral_indicators
        })

    if PARALLEL_AFTER_NEUTRAL and len(tasks) > 1:
        print(f"  🚀 并行分析其它动作：{len(tasks)} 个任务 | workers={NUM_WORKERS}")
        t_all = time.time()

        with ProcessPoolExecutor(
            max_workers=NUM_WORKERS,
            initializer=_worker_init,
            initargs=(MODEL_PATH,)
        ) as ex:
            futures = [ex.submit(_worker_run_one_action, t) for t in tasks]
            for fu in as_completed(futures):
                res = fu.result()
                if res is not None:
                    summaries.append(res)

        print(f"  ✅ 其它动作并行完成：{time.time()-t_all:.2f}s")

    else:
        # 退回串行
        for t in tasks:
            print(f"  ▶ {t['action_name']}: {Path(t['video_path']).name}")
            res = run_one_action(
                action_name=t["action_name"],
                extractor=extractor,
                video_path=t["video_path"],
                out_action_dir=Path(t["out_action_dir"]),
                start_frame=t["start_frame"],
                end_frame=t["end_frame"],
                fps_hint=t["fps_hint"],
                neutral_indicators=neutral_indicators
            )
            if res is not None:
                summaries.append(res)

    # -----------------------
    # 3) 输出汇总（保持动作顺序）
    # -----------------------
    order_index = {name: i for i, name in enumerate(ACTION_ORDER)}
    summaries.sort(key=lambda x: order_index.get(x.get("action_name", ""), 999))

    meta = dict(meta)
    meta["id"] = session_id
    write_json(session_out_dir / "_meta.json", meta)
    write_json(session_out_dir / "all_actions_summary.json", summaries)
    write_summary_csv(session_out_dir / "summary.csv", summaries)
    write_diagnosis_flow_md(session_out_dir / "diagnosis_flow.md", meta, summaries)
    write_overview_html(session_out_dir / "overview.html", meta, summaries)

# =============================================================================
# 两种入口：DB / Folder
# =============================================================================
def main_db():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"DB_PATH 不存在：{DB_PATH}")

    exams = db_fetch_examinations(DB_PATH, TARGET_EXAMINATION_ID, MAX_EXAMS)
    if not exams:
        print("⚠️ 数据库里没有可处理的 examinations（has_videos=1, is_valid=1）")
        return

    out_root = Path(OUTPUT_DIR)
    ensure_dir(out_root)

    print("=" * 80)
    print(f"MODE=db | DB_PATH={DB_PATH}")
    print(f"MODEL_PATH={MODEL_PATH}")
    print(f"OUTPUT_DIR={out_root}")
    print(f"将处理 examinations: {len(exams)}")
    print("=" * 80)

    with LandmarkExtractor(MODEL_PATH) as extractor:
        for i, e in enumerate(exams, 1):
            exam_id = e["examination_id"]
            print(f"\n[{i}/{len(exams)}] exam_id={exam_id} patient={e.get('patient_id')}")

            videos_all = db_fetch_videos_for_exam(DB_PATH, exam_id)
            vids: Dict[str, Dict[str, Any]] = {an: videos_all[an] for an in ACTION_ORDER if an in videos_all}

            labels = db_fetch_labels(DB_PATH, exam_id)
            meta = {
                "source": "db",
                "db_path": DB_PATH,
                "patient_id": e.get("patient_id"),
                "capture_datetime": e.get("capture_datetime"),
                "labels": labels,
            }

            analyze_one_session(
                session_id=exam_id,
                videos=vids,
                session_out_dir=out_root / exam_id,
                extractor=extractor,
                meta=meta
            )


def main_folder():
    root = Path(INPUT_VIDEO_ROOT)
    if not root.exists():
        raise FileNotFoundError(f"INPUT_VIDEO_ROOT 不存在：{root}")

    sessions = folder_list_sessions(str(root))
    if not sessions:
        print(f"⚠️ INPUT_VIDEO_ROOT 下没有子文件夹：{root}")
        return

    out_root = Path(OUTPUT_DIR)
    ensure_dir(out_root)

    print("=" * 80)
    print(f"MODE=folder | INPUT_VIDEO_ROOT={root}")
    print(f"MODEL_PATH={MODEL_PATH}")
    print(f"OUTPUT_DIR={out_root}")
    print(f"将处理 sessions: {len(sessions)}")
    print("=" * 80)

    with LandmarkExtractor(MODEL_PATH) as extractor:
        for i, sd in enumerate(sessions, 1):
            session_id = sd.name
            print(f"\n[{i}/{len(sessions)}] session={session_id}")

            vids: Dict[str, Dict[str, Any]] = {}
            for an in ACTION_ORDER:
                vp = folder_find_action_video(sd, an)
                if vp is None:
                    continue
                vids[an] = {"file_path": str(vp), "start_frame": 0, "end_frame": None, "fps": None}

            meta = {"source": "folder", "input_dir": str(sd)}
            analyze_one_session(
                session_id=session_id,
                videos=vids,
                session_out_dir=out_root / session_id,
                extractor=extractor,
                meta=meta
            )


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"MODEL_PATH 不存在：{MODEL_PATH}\n"
            f"请把 face_landmarker.task 放到该路径，或修改本文件顶部的 MODEL_PATH。"
        )

    if MODE.lower() == "db":
        main_db()
    elif MODE.lower() == "folder":
        main_folder()
    else:
        raise ValueError("MODE 只能是 'db' 或 'folder'")
