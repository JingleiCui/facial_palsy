# -*- coding: utf-8 -*-
"""
facial_symmetry_analysis_phone_videos.py

你这版需求：
- 递归扫描 VIDEO_ROOT（phone_videos）下面所有子目录里的 mp4（不读取数据库）
- 不使用任何动作别名映射（视频文件名已全部规范）
- 输出固定保存到：phone_symmetry_analysis
- 每个“直接包含 mp4 的目录”视为一个 case
- 对该目录下每个 mp4 调用 facial_symmetry_analysis.py 的 FacialSymmetryAnalyzer.analyze_single_video()
- 生成：
  - <case>/<action>/...（每个视频一个动作目录）
  - <case>/summary_case.csv / summary_case.json
  - 全局 summary.csv / summary.json
  - 可选 compare_actions.png（同一 case 多个动作时）

运行方式：
- 在 PyCharm 直接运行本文件（不需要命令行参数）
"""

import os
import re
import json
import csv
from pathlib import Path
from datetime import datetime

from facial_symmetry_analysis import FacialSymmetryAnalyzer


# ===================== 你只需要改这里 =====================

VIDEO_ROOT = r"/Users/cuijinglei/Documents/facial_palsy/phone_videos"

# 你指定的固定输出目录
OUTPUT_ROOT = r"/Users/cuijinglei/Documents/facial_palsy/HGFA/phone_symmetry_analysis"

# MediaPipe FaceLandmarker 模型路径（如你工程里不同，请改成实际位置）
MODEL_PATH = r"/Users/cuijinglei/PycharmProjects/medicalProject/models/face_landmarker.task"

# 是否生成每个 case 的动作对比图
MAKE_COMPARE_FIG = True

# 根目录本身是否也可能直接放 mp4（一般不用）
INCLUDE_ROOT_ITSELF = False

# =========================================================


VIDEO_EXTS = {".mp4"}  # 如果有 .MP4 也会被 lower() 处理到


def safe_name(s: str) -> str:
    """把动作名/文件夹名变成安全的目录名（保留中文/英文/数字/下划线/横线/点）。"""
    s = "" if s is None else str(s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^0-9a-zA-Z_\-\.\u4e00-\u9fff ]+", "_", s)
    s = s.strip().replace(" ", "_")
    return s or "unnamed"


def infer_action_name_from_filename(video_path: Path) -> str:
    """动作名 = 文件名 stem（你已保证命名正确，所以不做别名映射）。"""
    return video_path.stem.strip()


def is_video_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in VIDEO_EXTS


def list_videos_in_dir(folder: Path):
    videos = [p for p in folder.iterdir() if is_video_file(p)]
    videos.sort(key=lambda x: x.name.lower())
    return videos


def should_skip_dir(dir_name: str) -> bool:
    # 避免把输出目录/隐藏目录扫进去
    if dir_name.startswith("."):
        return True
    if dir_name.startswith("_"):
        return True
    return False


def find_case_dirs(root: Path, include_root: bool):
    """
    找到所有“直接包含 mp4 文件”的目录，作为一个 case。
    例如：
      phone_videos/by_oneplus      -> case
      phone_videos/wzq_iphone      -> case
    """
    case_dirs = []

    if include_root:
        try:
            if any(is_video_file(p) for p in root.iterdir() if p.exists()):
                case_dirs.append(root)
        except Exception:
            pass

    for dirpath, dirnames, filenames in os.walk(root):
        # prune 子目录
        dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]

        has_video = any(str(fn).lower().endswith(".mp4") for fn in filenames)
        if has_video:
            case_dirs.append(Path(dirpath))

    # 去重并按相对路径排序
    uniq = {}
    for d in case_dirs:
        try:
            rel = d.relative_to(root).as_posix()
        except Exception:
            rel = d.as_posix()
        uniq[rel] = d
    return [uniq[k] for k in sorted(uniq.keys())]


def ensure_output_root() -> Path:
    out = Path(OUTPUT_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_summary_csv(rows, csv_path: Path):
    if not rows:
        return

    # 固定字段 + 自动追加 analyzer 返回的其他字段
    base_fields = [
        "case_name",
        "case_dir",
        "action_name",
        "video_path",
        "video_relpath",
        "status",
        "mean_pearson",
        "output_folder",
        "error",
    ]
    extra = []
    for r in rows:
        for k in r.keys():
            if k not in base_fields and k not in extra:
                extra.append(k)
    fieldnames = base_fields + extra

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def relpath_safe(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root))
    except Exception:
        return str(p)


def main():
    video_root = Path(VIDEO_ROOT)
    if not video_root.exists():
        raise FileNotFoundError(f"VIDEO_ROOT 不存在：{video_root}")

    output_root = ensure_output_root()

    case_dirs = find_case_dirs(video_root, INCLUDE_ROOT_ITSELF)
    if not case_dirs:
        print(f"在 {video_root} 下面没有找到任何包含 mp4 的目录。")
        return

    print("=" * 90)
    print("phone_videos 递归批量对称性分析（不读取数据库）")
    print(f"VIDEO_ROOT : {video_root}")
    print(f"输出目录   : {output_root}")
    print(f"case 数量  : {len(case_dirs)}")
    print("=" * 90)

    analyzer = FacialSymmetryAnalyzer(
        db_path=":memory:",      # 占位：不读写数据库
        model_path=MODEL_PATH,
        verbose=True
    )

    global_rows = []
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for case_idx, case_dir in enumerate(case_dirs, 1):
        videos = list_videos_in_dir(case_dir)
        if not videos:
            continue

        rel_case = case_dir.relative_to(video_root).as_posix() if case_dir != video_root else "."
        case_name = safe_name(rel_case.replace("/", "__"))
        case_out = output_root / case_name / run_stamp
        case_out.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 90)
        print(f"[{case_idx}/{len(case_dirs)}] Case: {rel_case}")
        print(f"目录      : {case_dir}")
        print(f"视频数量  : {len(videos)}")
        print(f"输出      : {case_out}")
        print("=" * 90)

        case_rows = []
        action_to_path = {}

        # 用 case_name 当 patient_id 只是为了让输出更清晰（不影响计算）
        patient_id = case_name

        for vid_idx, vp in enumerate(videos, 1):
            action_name = infer_action_name_from_filename(vp)
            action_safe = safe_name(action_name)

            out_dir = case_out / action_safe
            out_dir.mkdir(parents=True, exist_ok=True)

            print("\n" + "#" * 90)
            print(f"[{vid_idx}/{len(videos)}] 分析视频")
            print(f"动作名 : {action_name}")
            print(f"路径   : {vp}")
            print(f"输出   : {out_dir}")
            print("#" * 90)

            try:
                res = analyzer.analyze_single_video(
                    video_path=str(vp),
                    action_name=action_name,
                    output_dir=str(out_dir),
                    patient_id=patient_id,
                    examination_id=None,
                    video_id=None,
                    start_frame=None,
                    end_frame=None,
                    fps=None,
                )

                if not isinstance(res, dict):
                    res = {"status": "failed", "error": "analyze_single_video 返回非 dict（可能 return None）"}

                row = {
                    "case_name": case_name,
                    "case_dir": str(case_dir),
                    "action_name": action_name,
                    "video_path": str(vp),
                    "video_relpath": relpath_safe(vp, video_root),
                    "status": res.get("status", "unknown"),
                    "mean_pearson": res.get("mean_pearson", ""),
                    "output_folder": res.get("output_folder", str(out_dir)),
                    "error": res.get("error", ""),
                }

                # 把 analyzer 可能返回的额外关键字段也保留到行里（存在就写）
                for k in ("left_score", "right_score", "y_diff_mean", "roll", "pitch", "yaw"):
                    if k in res and k not in row:
                        row[k] = res.get(k)

                case_rows.append(row)
                global_rows.append(row)

                if row["status"] == "success":
                    action_to_path[action_name] = str(vp)

            except Exception as e:
                row = {
                    "case_name": case_name,
                    "case_dir": str(case_dir),
                    "action_name": action_name,
                    "video_path": str(vp),
                    "video_relpath": relpath_safe(vp, video_root),
                    "status": "failed",
                    "mean_pearson": "",
                    "output_folder": str(out_dir),
                    "error": str(e),
                }
                case_rows.append(row)
                global_rows.append(row)
                print(f"❌ 该视频分析异常：{e}")

        # 保存 case 汇总
        case_csv = case_out / "summary_case.csv"
        case_json = case_out / "summary_case.json"
        write_summary_csv(case_rows, case_csv)
        with open(case_json, "w", encoding="utf-8") as f:
            json.dump(case_rows, f, ensure_ascii=False, indent=2)

        print("\n✅ Case 完成")
        print(f"Case 汇总 CSV : {case_csv}")
        print(f"Case 汇总 JSON: {case_json}")

        # 可选：生成动作对比图（同一 case 多个动作时）
        if MAKE_COMPARE_FIG and len(action_to_path) >= 2:
            compare_path = case_out / "compare_actions.png"
            try:
                analyzer.compare_actions(
                    video_paths=action_to_path,
                    patient_id=patient_id,
                    output_path=str(compare_path)
                )
                print(f"✅ 动作对比图已生成: {compare_path}")
            except Exception as e:
                print(f"⚠️ 动作对比图生成失败：{e}")

    # 保存全局汇总
    summary_csv = output_root / f"summary_{run_stamp}.csv"
    summary_json = output_root / f"summary_{run_stamp}.json"
    write_summary_csv(global_rows, summary_csv)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(global_rows, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 90)
    print("✅ 全部完成")
    print(f"全局汇总 CSV : {summary_csv}")
    print(f"全局汇总 JSON: {summary_json}")
    print("=" * 90)


if __name__ == "__main__":
    main()
