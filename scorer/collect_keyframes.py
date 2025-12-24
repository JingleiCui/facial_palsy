# -*- coding: utf-8 -*-
"""
collect_peak_raw_by_action.py

批量收集 clinical_grading 下所有动作的 peak_raw 图片，按动作归档到 clinical_grading_debug

运行方式：PyCharm 直接点运行即可
"""

import shutil
from pathlib import Path


SRC_ROOT = Path("/Users/cuijinglei/Documents/facialPalsy/HGFA/clinical_grading")
DST_ROOT = Path("/Users/cuijinglei/Documents/facialPalsy/HGFA/clinical_grading_debug")

# 你项目里的11个动作（用于提前建文件夹；即使不在列表中也会照样复制到对应动作名文件夹）
ACTIONS = [
    "NeutralFace",
    "SpontaneousEyeBlink",
    "VoluntaryEyeBlink",
    "CloseEyeSoftly",
    "CloseEyeHardly",
    "RaiseEyebrow",
    "Smile",
    "ShrugNose",
    "ShowTeeth",
    "BlowCheek",
    "LipPucker",
]


def ensure_dirs():
    DST_ROOT.mkdir(parents=True, exist_ok=True)
    for a in ACTIONS:
        (DST_ROOT / a).mkdir(parents=True, exist_ok=True)


def find_exam_dirs(src_root: Path):
    """
    以“包含 report.html 的目录”为一次检查(exam)的根目录
    """
    for report in src_root.rglob("report.html"):
        yield report.parent


def collect_one_exam(exam_dir: Path):
    """
    exam_dir:
      clinical_grading/<exam_id>/report.html
      clinical_grading/<exam_id>/<ActionName>/peak_raw.jpg
    """
    exam_id = exam_dir.name

    copied = 0
    skipped = 0

    # 遍历 exam_dir 下的子目录（动作名目录）
    for action_dir in exam_dir.iterdir():
        if not action_dir.is_dir():
            continue

        action_name = action_dir.name

        # peak_raw 可能是 jpg/png/jpeg
        peak_files = []
        for ext in ("jpg", "jpeg", "png", "webp"):
            f = action_dir / f"peak_raw.{ext}"
            if f.exists():
                peak_files.append(f)

        if not peak_files:
            continue

        # 理论上只会有一个，取第一个
        src_peak = peak_files[0]
        dst_action_dir = DST_ROOT / action_name
        dst_action_dir.mkdir(parents=True, exist_ok=True)

        dst_name = f"{exam_id}_{action_name}_peak_raw{src_peak.suffix.lower()}"
        dst_path = dst_action_dir / dst_name

        try:
            shutil.copy2(src_peak, dst_path)
            copied += 1
        except Exception as e:
            print(f"[ERROR] copy failed: {src_peak} -> {dst_path} | {e}")
            skipped += 1

    return copied, skipped


def main():
    if not SRC_ROOT.exists():
        raise FileNotFoundError(f"SRC_ROOT 不存在: {SRC_ROOT}")

    ensure_dirs()

    total_exam = 0
    total_copied = 0
    total_skipped = 0

    for exam_dir in find_exam_dirs(SRC_ROOT):
        total_exam += 1
        c, s = collect_one_exam(exam_dir)
        total_copied += c
        total_skipped += s

        if total_exam % 10 == 0:
            print(f"[INFO] processed exams={total_exam}, copied={total_copied}, skipped={total_skipped}")

    print("\n==================== DONE ====================")
    print(f"Exam dirs: {total_exam}")
    print(f"Copied:    {total_copied}")
    print(f"Skipped:   {total_skipped}")
    print(f"Output ->  {DST_ROOT}")


if __name__ == "__main__":
    main()