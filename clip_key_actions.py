# -*- coding: utf-8 -*-
"""
PyCharm 一键运行版（带全局文件名索引与大小写无关匹配）：
- 递归扫描 BASE_DIR 下所有 .json（文件名不限）
- 仅保留 11 个目标动作及其引用的视频；重建 VideoFileList / 重映射 VideoFileIndex
- 真实删除未引用视频（DO_DELETE=True），支持 .mp4/.MP4 等大小写不一致，且 JSON 路径前缀层级错误时，按文件名全局匹配
- 覆盖或写 .filtered.json（可选）
- 输出 CSV 报告

安全策略：
- 若过滤后无动作可保留 -> 跳过该 JSON，不动文件
- 多处同名视频 -> 先用 JSON Path 提取的 token（如 XW000003、20230222 等）打分，仍并列时按 AMBIGUOUS_POLICY 处理
"""

import os
import re
import json
import shutil
from typing import Dict, List, Set, Any, Tuple, DefaultDict
from collections import defaultdict
from datetime import datetime
import cv2
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv

# ========== 配置区 ==========
BASE_DIR = "/Users/cuijinglei/Documents/facialPalsy/videos"  # 数据根目录

START_PATIENT_ID = "XW000427"  # 设置为 None 则不限制
END_PATIENT_ID = "XW000437"    # 设置为 None 则不限制

DO_DELETE = True           # True: 真实删除未引用视频
OVERWRITE_JSON = True     # True: 覆盖原 JSON；False: 写 .filtered.json
MAKE_BACKUP = False         # 覆盖 JSON 前是否创建 .bak 备份

# === 裁剪视频开关 ===
DO_TRIM = True                 # True=会真实裁剪并覆盖原视频
USE_FFMPEG = True              # 推荐 True：保留音频/旋转元数据更稳；OpenCV 会丢音频
FFMPEG_BIN = "ffmpeg"

# 裁剪后是否把 JSON 时间段改成从0开始（强烈建议 True，否则你裁完视频后 JSON 还是旧时间，会二次裁剪出错）
UPDATE_JSON_TIME_RANGE = True

# ========= 性能/鲁棒性配置 =========
MAX_WORKERS = 8                 # 并发裁剪数：M3 Max 建议 4~8，自行调
FFMPEG_TIMEOUT_SEC = 180        # 单个视频裁剪最大允许时间（秒），防止卡死
DEBUG_PRINT_FFMPEG_CMD = False  # True=打印ffmpeg命令，便于排查

# 先尝试“极速不重编码拷贝”，失败再重编码（更准更稳）
FFMPEG_TRY_STREAM_COPY_FIRST = True

# 优先用 Apple 硬件编码（Homebrew ffmpeg 通常支持），失败会自动回退到 libx264
FFMPEG_TRY_VIDEOTOOLBOX = True
VIDEOTOOLBOX_ENCODER = "h264_videotoolbox"

# libx264 回退参数（不追求极致画质，追求速度）
X264_PRESET = "veryfast"
X264_CRF = "20"

REPORT_NAME = "clip_report.csv"  # 汇总报告写在 BASE_DIR

# 允许对“同名不同目录”的情况做全局匹配（很关键，解决 Path 前缀错误问题）
ALLOW_GLOBAL_FILENAME_SEARCH = True

# 多候选同名文件的处理策略： "skip"（默认跳过）、"first"（任取第一个）、"newest"（取最近修改）、"oldest"（最早修改）、"shortest"（路径最短）
# 修改为 "newest" 以便在有多个同名文件时选择最新的进行删除
AMBIGUOUS_POLICY = "newest"

# 仅识别/删除这些扩展名（大小写不敏感）
VIDEO_EXTS = {".mp4"}  # 如有需要可加 ".mov", ".avi"
# ===========================


KEEP_ACTIONS = {
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
}

PREFIXES = ("facialmuscle", "eyemovement")

def _norm(s: str) -> str:
    return re.sub(r"[\s_]+", "", (s or "").lower())

ALIAS_MAP = {
    "neutralface": "NeutralFace",
    "spontaneouseyeblink": "SpontaneousEyeBlink",
    "voluntaryeyeblink": "VoluntaryEyeBlink",
    "closeeyesoftly": "CloseEyeSoftly",
    "closeeyehardly": "CloseEyeHardly",
    "foreheadwrinkles": "RaiseEyebrow",
    "smile": "Smile",
    "shrugnose": "ShrugNose",
    "showteeth": "ShowTeeth",
    "blowcheek": "BlowCheek",
    "lippucker": "LipPucker",
}

def canonicalize_action(raw: str) -> str:
    if not raw:
        return ""
    s = _norm(raw)
    for p in PREFIXES:
        if s.startswith(p):
            s = s[len(p):]
            break
    if s in ALIAS_MAP:
        return ALIAS_MAP[s]
    if raw in KEEP_ACTIONS:
        return raw
    norm_keep = [_norm(x) for x in KEEP_ACTIONS]
    if _norm(raw) in norm_keep:
        idx = norm_keep.index(_norm(raw))
        return list(KEEP_ACTIONS)[idx]
    return ""

def human_bytes(n: int) -> str:
    units = ["B","KB","MB","GB","TB"]
    size = float(n)
    for u in units:
        if size < 1024.0 or u == units[-1]:
            return f"{size:.2f} {u}"
        size /= 1024.0

def is_metadata_json(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    vmeta = obj.get("VideoMetaInfo")
    if not isinstance(vmeta, dict):
        return False
    if not isinstance(vmeta.get("ActionList"), list):
        return False
    if not isinstance(vmeta.get("VideoFileList"), list):
        return False
    return True

def safe_load_json(path: str) -> Any:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            with open(path, "r", encoding=enc) as f:
                return json.load(f)
        except Exception:
            pass
    raise ValueError(f"Failed to load json: {path}")

def safe_write_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def backup_file(path: str) -> str:
    bak = path + ".bak"
    shutil.copy2(path, bak)
    return bak

def resolve_abs_path(base_dir: str, maybe_rel: str) -> str:
    if not isinstance(maybe_rel, str) or not maybe_rel.strip():
        return ""
    p = maybe_rel.strip().replace("\\", "/")
    if os.path.isabs(p):
        return os.path.normpath(p)
    p = p.lstrip("/")
    base_name = os.path.basename(os.path.normpath(base_dir)).lower()
    parts = p.split("/", 1)
    if parts and parts[0].lower() == base_name and len(parts) > 1:
        p = parts[1]  # 去掉重复的 'videos'
    return os.path.normpath(os.path.join(base_dir, p))

def extract_video_path_from_entry(entry: dict) -> str:
    for k in ("Path","FilePath","RelativePath","VideoFilePath","File"):
        v = entry.get(k)
        if isinstance(v, str) and v.strip():
            return v
    v = entry.get("FileName")
    if isinstance(v, str) and v.strip():
        return v
    return ""

def is_video_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in VIDEO_EXTS

# 允许从文件名中抽取关键键（全名、主名、下划线前缀主名、编码如 C9529）
FILENAME_CODE_RE = re.compile(r"[a-z]\d{3,}", re.IGNORECASE)

def derive_filename_keys(filename: str) -> Dict[str, str]:
    """从文件名派生多种匹配键：
    - name: 完整文件名小写（含扩展）
    - stem: 去扩展的小写主名
    - prefix: 主名按 '_' 分段后的第一段（解决 C9529_FacialMuscle_*.MP4 这类命名）
    - code: 匹配如 C9529/P1234 这类字母+数字编码（>=3位数字）
    """
    name_lower = (filename or "").lower()
    stem_lower = os.path.splitext(name_lower)[0]
    prefix_stem_lower = stem_lower.split("_", 1)[0]
    m = FILENAME_CODE_RE.search(stem_lower)
    code = m.group(0).lower() if m else ""
    return {"name": name_lower, "stem": stem_lower, "prefix": prefix_stem_lower, "code": code}

def _take_first_time(v) -> str:
    """StartFrameLocation/EndFrameLocation 在 JSON 里通常是 ['00:00:07:000'] 这种 list。"""
    if isinstance(v, list) and v:
        return str(v[0] or "").strip()
    if isinstance(v, str):
        return v.strip()
    return ""

def time_str_to_seconds(time_str: str) -> float | None:
    """
    '00:00:07:000' (时:分:秒:毫秒) -> seconds(float)
    兼容 'mm:ss:ms' / 'hh:mm:ss:ms'
    """
    if not time_str:
        return None
    parts = time_str.split(":")
    try:
        if len(parts) == 4:
            h, m, s, ms = map(int, parts)
        elif len(parts) == 3:
            h = 0
            m, s, ms = map(int, parts)
        else:
            return None
        return h * 3600 + m * 60 + s + ms / 1000.0
    except Exception:
        return None

def seconds_to_time_str(sec: float) -> str:
    if sec < 0:
        sec = 0
    ms_total = int(round(sec * 1000))
    h = ms_total // 3600000
    ms_total %= 3600000
    m = ms_total // 60000
    ms_total %= 60000
    s = ms_total // 1000
    ms = ms_total % 1000
    return f"{h:02d}:{m:02d}:{s:02d}:{ms:03d}"

def get_video_fps_safe(video_path: str, default: float = 30.0) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return default
    fps = cap.get(cv2.CAP_PROP_FPS) or default
    cap.release()
    # 一些视频会读到 29.97，保留浮点更准；但转帧时取 int 会更稳
    return fps

def time_to_frame(time_sec: float, fps: float) -> int:
    # 和你 import_metadata.py 的 int(total_seconds * fps) 一致思路 :contentReference[oaicite:4]{index=4}
    return int(time_sec * fps)

def _run_ffmpeg(cmd: list[str], timeout_sec: int) -> tuple[bool, str]:
    if DEBUG_PRINT_FFMPEG_CMD:
        print("[FFMPEG]", " ".join(cmd))

    try:
        r = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
        )
        if r.returncode != 0:
            return False, (r.stderr or "").strip()[-1200:]
        return True, "ok"
    except subprocess.TimeoutExpired:
        return False, f"timeout>{timeout_sec}s"
    except Exception as e:
        return False, f"exception: {e}"

def trim_video_inplace_ffmpeg(video_path: str, start_sec: float, end_sec: float, ffmpeg_bin: str) -> tuple[bool, str]:
    """
    覆盖原视频：先输出临时文件，再 os.replace。
    速度优化：-ss 放在 -i 前；优先 stream copy；再 videotoolbox；最后 libx264。
    稳定性：只映射 视频+音频，丢弃字幕/数据流，避免 codec none。
    """
    if end_sec <= start_sec:
        return False, f"invalid range {start_sec:.3f}~{end_sec:.3f}"

    duration = end_sec - start_sec
    tmp_path = video_path + ".tmp_trim.mp4"

    # 通用映射：只保留视频+音频，丢弃字幕/数据流（避免 codec none）
    common = [
        ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y", "-nostdin",
        "-ss", f"{start_sec:.3f}",        # ✅ 快定位：放在 -i 前
        "-i", video_path,
        "-t", f"{duration:.3f}",
        "-map", "0:v:0",
        "-map", "0:a?",
        "-sn", "-dn",
        "-map_metadata", "0",
        "-movflags", "+faststart",
        tmp_path
    ]

    # 1) 极速：不重编码（最快，但少数视频可能失败/不够精准）
    if FFMPEG_TRY_STREAM_COPY_FIRST:
        cmd1 = common.copy()
        # 把输出编码参数插到 tmp_path 前（替换末尾 tmp_path）
        cmd1 = cmd1[:-1] + ["-c:v", "copy", "-c:a", "copy"] + [cmd1[-1]]
        ok, msg = _run_ffmpeg(cmd1, FFMPEG_TIMEOUT_SEC)
        if ok:
            os.replace(tmp_path, video_path)
            return True, "ok(copy)"

        # 清理 tmp
        if os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except: pass

    # 2) 硬件编码：Apple VideoToolbox（通常比 libx264 快很多）
    if FFMPEG_TRY_VIDEOTOOLBOX:
        cmd2 = common.copy()
        cmd2 = cmd2[:-1] + ["-c:v", VIDEOTOOLBOX_ENCODER, "-c:a", "aac", "-b:a", "128k"] + [cmd2[-1]]
        ok, msg = _run_ffmpeg(cmd2, FFMPEG_TIMEOUT_SEC)
        if ok:
            os.replace(tmp_path, video_path)
            return True, f"ok({VIDEOTOOLBOX_ENCODER})"

        if os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except: pass

    # 3) 回退：libx264（最兼容）
    cmd3 = common.copy()
    cmd3 = cmd3[:-1] + ["-c:v", "libx264", "-preset", X264_PRESET, "-crf", str(X264_CRF),
                        "-c:a", "aac", "-b:a", "128k"] + [cmd3[-1]]
    ok, msg = _run_ffmpeg(cmd3, FFMPEG_TIMEOUT_SEC)
    if not ok:
        if os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except: pass
        return False, f"ffmpeg failed: {msg}"

    os.replace(tmp_path, video_path)
    return True, "ok(libx264)"


def trim_video_inplace_opencv(video_path: str, start_sec: float, end_sec: float) -> tuple[bool, str]:
    """
    OpenCV 逐帧写出（会丢音频；旋转元数据也不一定保留）。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "cannot open video"

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    start_f = max(0, time_to_frame(start_sec, fps))
    end_f = min(total, time_to_frame(end_sec, fps))
    if end_f <= start_f:
        cap.release()
        return False, f"invalid frame range {start_f}~{end_f} (total={total})"

    tmp_path = video_path + ".tmp_trim.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        return False, "cannot open writer"

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        cur = start_f
        while cur < end_f:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            cur += 1
        cap.release()
        writer.release()
        os.replace(tmp_path, video_path)
        return True, "ok"
    except Exception as e:
        try:
            cap.release()
        except:
            pass
        try:
            writer.release()
        except:
            pass
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except:
            pass
        return False, f"opencv failed: {e}"


# -------- 全局文件名索引（一次性扫描 BASE_DIR） --------
class FileIndex:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.by_name: DefaultDict[str, List[str]] = defaultdict(list)  # 完整文件名小写 -> [绝对路径...]
        self.by_stem: DefaultDict[str, List[str]] = defaultdict(list)  # 去后缀小写 -> [绝对路径...]
        self.by_prefix: DefaultDict[str, List[str]] = defaultdict(list)   # 下划线前缀主名 -> [绝对路径...]
        self.by_code: DefaultDict[str, List[str]] = defaultdict(list)     # 编码(如 c9529) -> [绝对路径...]

    def build(self):
        for root, _, files in os.walk(self.base_dir):
            for f in files:
                # 只索引视频文件（或全部？此处只索引视频更稳）
                if not is_video_file(f):
                    continue
                f_lower = f.lower()
                path_abs = os.path.join(root, f)
                keys = derive_filename_keys(f)
                self.by_name[keys["name"]].append(path_abs)
                self.by_stem[keys["stem"]].append(path_abs)
                self.by_prefix[keys["prefix"]].append(path_abs)
                if keys["code"]:
                    self.by_code[keys["code"]].append(path_abs)

    def find_by_name(self, filename: str) -> List[str]:
        return self.by_name.get((filename or "").lower(), [])

    def find_by_stem(self, stem: str) -> List[str]:
        return self.by_stem.get((stem or "").lower(), [])

    def find_by_prefix(self, prefix: str) -> List[str]:
        return self.by_prefix.get((prefix or "").lower(), [])

    def find_by_code(self, code: str) -> List[str]:
        return self.by_code.get((code or "").lower(), [])

def tokens_from_hint_path(hint: str) -> List[str]:
    """
    从 JSON 的 Path 中尽可能提取有辨识度的 token（用于多候选打分）：
    - 患者ID：XW 后接数字 (不区分大小写)
    - 日期时间：YYYYMMDD、YYYYMMDD_HHMMSS/HHMM、YYYY-MM-DD-HHMM 等
    - 其他目录名片段
    """
    tokens: List[str] = []
    if not isinstance(hint, str):
        return tokens
    s = hint.replace("\\", "/").strip("/")
    parts = [p for p in s.split("/") if p]

    # 直接加入所有路径段作为弱提示
    tokens.extend(parts)

    # 患者ID
    for p in parts:
        m = re.search(r"(XW\d{3,})", p, flags=re.IGNORECASE)
        if m:
            tokens.append(m.group(1))

    # 日期（纯8位）
    for p in parts:
        if re.match(r"^\d{8}$", p):
            tokens.append(p)

    # 日期_时间
    for p in parts:
        if re.match(r"^\d{8}_[0-9]{4,6}$", p):
            tokens.append(p)

    # 形如 2023-02-22-1004
    for p in parts:
        if re.match(r"^\d{4}-\d{2}-\d{2}-\d{4,6}$", p):
            tokens.append(p)

    # 去重
    uniq = []
    seen = set()
    for t in tokens:
        tl = t.lower()
        if tl not in seen:
            uniq.append(t)
            seen.add(tl)
    return uniq

def score_candidates_by_tokens(cands: List[str], tokens: List[str]) -> List[Tuple[str, int]]:
    """
    对候选路径按 tokens 匹配度打分：出现一个 token 加 1（大小写无关，子串匹配）。
    返回 [(path, score), ...]
    """
    if not tokens:
        return [(p, 0) for p in cands]
    scored = []
    for p in cands:
        pl = p.lower()
        score = 0
        for t in tokens:
            if not t:
                continue
            if t.lower() in pl:
                score += 1
        scored.append((p, score))
    return scored

def choose_on_ambiguous(cands: List[str]) -> str:
    if not cands:
        return ""
    if len(cands) == 1:
        return cands[0]
    policy = AMBIGUOUS_POLICY.lower()
    try:
        if policy == "first":
            return cands[0]
        elif policy == "newest":
            cands.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return cands[0]
        elif policy == "oldest":
            cands.sort(key=lambda x: os.path.getmtime(x))
            return cands[0]
        elif policy == "shortest":
            cands.sort(key=lambda x: len(x))
            return cands[0]
        else:  # "skip"
            return ""
    except Exception:
        # 若取mtime出错，则退化到first或skip
        return cands[0] if policy in ("first","newest","oldest","shortest") else ""

def resolve_by_filename_global(index: FileIndex, base_dir: str, filename: str, hint_path: str) -> str:
    """
    利用全局索引按文件名/主名(stem)在整个 BASE_DIR 下寻找匹配；
    用 JSON Path 提取的 tokens 打分挑选最佳；若并列，按 AMBIGUOUS_POLICY。
    """
    if not filename:
        return ""

    keys = derive_filename_keys(filename)

    candidates: List[str] = []
    # 1) 完整文件名大小写无关匹配
    candidates = index.find_by_name(filename)

    # 2) 主名匹配（无扩展）
    if not candidates:
        candidates = index.find_by_stem(keys["stem"])  # 精确主名

    # 3) 下划线前缀主名（例如 C9529_FacialMuscle_* -> C9529）
    if not candidates and keys["prefix"]:
        candidates = index.find_by_prefix(keys["prefix"])  # 共享前缀

    # 4) 编码匹配（如 C9529/P1234）
    if not candidates and keys["code"]:
        candidates = index.find_by_code(keys["code"])  # 通过编码定位

    if not candidates:
        return ""

    tokens = tokens_from_hint_path(hint_path)
    scored = score_candidates_by_tokens(candidates, tokens)
    max_score = max(s for _, s in scored) if scored else 0
    best = [p for p, s in scored if s == max_score]

    chosen = choose_on_ambiguous(best)
    return chosen

def resolve_actual_video_path(base_dir: str, raw_path: str, file_index: FileIndex) -> str:
    """
    统一的路径解析：
    1) 先按照 JSON 路径（修正分隔符/去重复 'videos' 段）尝试精确路径
    2) 若该目录存在：在该目录下做大小写无关的同名/同stem匹配
    3) 若该目录不存在或仍找不到：启用全局索引在 BASE_DIR 下按文件名/主名匹配
    """
    # 目标文件名（最后一段）是最重要的线索
    filename = os.path.basename(raw_path.replace("\\", "/").strip("/"))

    # 步骤1：直接猜测绝对路径
    abs_guess = resolve_abs_path(base_dir, raw_path)
    if os.path.isfile(abs_guess):
        return abs_guess

    # 步骤2：在“猜测目录”里大小写无关匹配（支持前缀/编码）
    guess_dir = os.path.dirname(abs_guess)
    if os.path.isdir(guess_dir):
        filename_lower = filename.lower()
        stem = os.path.splitext(filename_lower)[0]
        prefix = stem.split("_", 1)[0]
        m = FILENAME_CODE_RE.search(stem)
        code = m.group(0).lower() if m else ""

        # 完整文件名大小写无关
        for f in os.listdir(guess_dir):
            if f.lower() == filename_lower and is_video_file(f):
                return os.path.join(guess_dir, f)

        # 主名等价（同 stem，不同扩展大小写）
        for f in os.listdir(guess_dir):
            stem2, ext2 = os.path.splitext(f)
            if stem2.lower() == stem and ext2.lower() in VIDEO_EXTS:
                return os.path.join(guess_dir, f)

        # 下划线前缀主名（如 C9529_FacialMuscle_*）
        if prefix:
            for f in os.listdir(guess_dir):
                stem2, ext2 = os.path.splitext(f)
                if ext2.lower() in VIDEO_EXTS and stem2.lower().split("_", 1)[0] == prefix:
                    return os.path.join(guess_dir, f)

        # 编码匹配（如 C9529）
        if code:
            for f in os.listdir(guess_dir):
                stem2, ext2 = os.path.splitext(f)
                if ext2.lower() in VIDEO_EXTS and FILENAME_CODE_RE.search(stem2.lower() or ""):
                    if FILENAME_CODE_RE.search(stem2.lower()).group(0) == code:
                        return os.path.join(guess_dir, f)

    # 步骤3：全局索引按文件名匹配
    if ALLOW_GLOBAL_FILENAME_SEARCH and file_index is not None:
        chosen = resolve_by_filename_global(file_index, base_dir, filename, raw_path)
        if chosen:
            return chosen

    return ""  # 仍然找不到


def get_int_by_keys(d: dict, keys, default=None):
    for k in keys:
        v = d.get(k)
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.strip().isdigit():
            return int(v.strip())
    return default

def get_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    cap.release()
    return float(fps) if fps and fps > 1e-6 else 30.0  # 兜底

def process_one_json(
    json_path: str,
    base_dir: str,
    do_delete: bool,
    overwrite_json: bool,
    make_backup: bool,
    file_index: FileIndex
) -> Dict[str, Any]:
    data = safe_load_json(json_path)
    if not is_metadata_json(data):
        return {"json": json_path, "skipped": True, "reason": "not a supported metadata schema"}

    vmeta = data["VideoMetaInfo"]
    vlist = vmeta.get("VideoFileList", [])
    alist = vmeta.get("ActionList", [])

    orig_video_count = len(vlist)
    orig_action_count = len(alist)

    kept_actions: List[dict] = []
    kept_video_indices: Set[int] = set()
    for a in alist:
        canon = canonicalize_action(a.get("Action", ""))
        if canon in KEEP_ACTIONS:
            a2 = dict(a); a2["Action"] = canon
            kept_actions.append(a2)
            vidx = a2.get("VideoFileIndex")
            if isinstance(vidx, int):
                kept_video_indices.add(vidx)

    if not kept_actions:
        return {
            "json": json_path, "skipped": True, "reason": "no actions remain after filtering",
            "orig_action_count": orig_action_count, "orig_video_count": orig_video_count,
        }

    kept_indices_sorted = sorted(list(kept_video_indices))
    old_to_new = {old_i: new_i for new_i, old_i in enumerate(kept_indices_sorted)}
    new_vlist = [vlist[i] for i in kept_indices_sorted if 0 <= i < len(vlist)]

    for a in kept_actions:
        oi = a.get("VideoFileIndex")
        if isinstance(oi, int) and oi in old_to_new:
            a["VideoFileIndex"] = old_to_new[oi]

    removed_indices = [i for i in range(len(vlist)) if i not in kept_video_indices]

    delete_candidates: List[str] = []
    for idx in removed_indices:
        if 0 <= idx < len(vlist):
            raw = extract_video_path_from_entry(vlist[idx])
            if raw:
                real = resolve_actual_video_path(base_dir, raw, file_index)
                if real:
                    delete_candidates.append(real)

    deleted_files: List[str] = []
    freed_bytes_total = 0
    if do_delete:
        # 处理同名多候选仍无法唯一定位的情况（resolve_actual_video_path会返回空）
        # 这里 delete_candidates 都是已解析到唯一真实路径的
        for p in delete_candidates:
            if os.path.isfile(p):
                try:
                    sz = os.path.getsize(p)
                    os.remove(p)
                    deleted_files.append(p)
                    freed_bytes_total += sz
                except Exception as e:
                    print(f"[WARN] 删除失败：{p} -> {e}")

    # === 新增：裁剪 kept_actions 对应的视频（覆盖原视频）===
    trimmed_files: list[str] = []
    trim_failed: list[dict] = []
    processed_real_paths = set()

    def _trim_one(action_obj: dict, real: str, start_str: str, end_str: str, s: float, e: float, json_path: str):
        if USE_FFMPEG:
            ok, msg = trim_video_inplace_ffmpeg(real, s, e, FFMPEG_BIN)
        else:
            ok, msg = trim_video_inplace_opencv(real, s, e)

        return {
            "ok": ok,
            "json": json_path,
            "video": real,
            "start": start_str,
            "end": end_str,
            "start_sec": s,
            "end_sec": e,
            "msg": msg,
            "action_obj": action_obj,  # 回来后再更新 JSON 时间
        }

    if DO_TRIM:
        tasks = []

        for a in kept_actions:
            vidx = a.get("VideoFileIndex")
            if not (isinstance(vidx, int) and 0 <= vidx < len(new_vlist)):
                continue

            raw = extract_video_path_from_entry(new_vlist[vidx])
            if not raw:
                trim_failed.append({"json": json_path, "video": "", "start": "", "end": "",
                                    "msg": f"missing video path for index={vidx}"})
                continue

            real = resolve_actual_video_path(base_dir, raw, file_index)
            if not real or not os.path.isfile(real):
                trim_failed.append(
                    {"json": json_path, "video": raw, "start": "", "end": "", "msg": "cannot resolve real path"})
                continue

            real_norm = os.path.normpath(real)
            if real_norm in processed_real_paths:
                continue
            processed_real_paths.add(real_norm)

            start_str = _take_first_time(a.get("StartFrameLocation"))
            end_str = _take_first_time(a.get("EndFrameLocation"))

            s = time_str_to_seconds(start_str)
            e = time_str_to_seconds(end_str)
            if s is None or e is None or e <= s:
                trim_failed.append(
                    {"json": json_path, "video": real, "start": start_str, "end": end_str, "msg": "invalid time range"})
                continue

            tasks.append((a, real, start_str, end_str, s, e))

        # 并发跑 ffmpeg（线程只是调度子进程，不会被 GIL 卡住）
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            future_map = {
                ex.submit(_trim_one, a, real, start_str, end_str, s, e, json_path): (a, real)
                for (a, real, start_str, end_str, s, e) in tasks
            }

            for fut in as_completed(future_map):
                r = fut.result()
                if r["ok"]:
                    trimmed_files.append(r["video"])

                    # 裁剪后把 JSON 时间段“归零”
                    if UPDATE_JSON_TIME_RANGE:
                        aobj = r["action_obj"]
                        aobj["StartFrameLocation"] = ["00:00:00:000"]
                        aobj["EndFrameLocation"] = [seconds_to_time_str(r["end_sec"] - r["start_sec"])]
                else:
                    trim_failed.append({
                        "json": r["json"],
                        "video": r["video"],
                        "start": r["start"],
                        "end": r["end"],
                        "start_sec": r["start_sec"],
                        "end_sec": r["end_sec"],
                        "msg": r["msg"],
                    })

    new_data = dict(data)
    new_data["VideoMetaInfo"]["VideoFileList"] = new_vlist
    new_data["VideoMetaInfo"]["ActionList"] = kept_actions

    out_path = json_path
    if not overwrite_json:
        root, ext = os.path.splitext(json_path)
        out_path = root + ".filtered" + ext
    else:
        if make_backup and os.path.isfile(json_path):
            try:
                backup_file(json_path)
            except Exception as e:
                print(f"[WARN] 备份失败：{json_path} -> {e}")

    try:
        safe_write_json(out_path, new_data)
    except Exception as e:
        print(f"[ERROR] 写入失败：{out_path} -> {e}")

    return {
        "json": json_path,
        "out_json": out_path,
        "skipped": False,
        "orig_action_count": orig_action_count,
        "kept_action_count": len(kept_actions),
        "orig_video_count": orig_video_count,
        "kept_video_count": len(new_vlist),
        "removed_video_count": len(removed_indices),
        "deleted_files": deleted_files,
        "delete_candidates": delete_candidates,
        "bytes_freed": freed_bytes_total,
        "trimmed_files": trimmed_files,
        "trim_failed": trim_failed,
        "trimmed_count": len(trimmed_files),
        "trim_failed_count": len(trim_failed),

    }

def scan_all_jsons(base_dir: str) -> List[str]:
    jsons = []
    for r, _, files in os.walk(base_dir):
        for fn in files:
            if fn.lower().endswith(".json"):
                jsons.append(os.path.join(r, fn))
    return jsons

def write_report(base_dir: str, rows: list[dict], report_name: str) -> str:
    out = os.path.join(base_dir, report_name)
    headers = [
        "json","out_json","skipped","reason",
        "orig_action_count","kept_action_count",
        "orig_video_count","kept_video_count","removed_video_count",
        "bytes_freed",
        "trimmed_count","trim_failed_count"
    ]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow({h: r.get(h, "") for h in headers})
    return out

def write_trim_failures(base_dir: str, rows: list[dict], name: str = "clip_failures.csv") -> str:
    out = os.path.join(base_dir, name)
    headers = ["json", "video", "start", "end", "start_sec", "end_sec", "msg"]
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            for item in (r.get("trim_failed") or []):
                if isinstance(item, dict):
                    w.writerow({h: item.get(h, "") for h in headers})
                else:
                    # 兼容老格式（字符串）
                    w.writerow({"json": r.get("json",""), "video": "", "msg": str(item)})
    return out

def run():
    base_dir = os.path.abspath(BASE_DIR)
    print("== 开始处理 ==")
    print(f"BaseDir       : {base_dir}")
    print(f"DO_DELETE     : {DO_DELETE}  (True=会真实删除未引用视频)")
    print(f"Overwrite JSON: {OVERWRITE_JSON}  (backup: {MAKE_BACKUP})")
    print(f"Report Name   : {REPORT_NAME}")
    print(f"时间          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    if not os.path.isdir(base_dir):
        print(f"[ERROR] BASE_DIR 非目录：{base_dir}")
        return

    json_files = scan_all_jsons(base_dir)

    # === 过滤 json_files ===
    filtered_jsons = []
    for jp in json_files:
        # 从路径中尝试提取患者ID (假设ID格式为 XW + 数字)
        # 路径示例: /Data/XW000001/20230101/metadata.json
        match = re.search(r"(XW\d+)", jp, re.IGNORECASE)
        if match:
            pid = match.group(1).upper()  # 提取到的ID，如 XW000001

            # 范围判断
            if START_PATIENT_ID and pid < START_PATIENT_ID:
                continue
            if END_PATIENT_ID and pid > END_PATIENT_ID:
                continue

        filtered_jsons.append(jp)

    json_files = filtered_jsons  # 使用过滤后的列表

    if not json_files:
        print("[INFO] 未找到任何 .json 文件。")
        return

    # 建立全局文件名索引（一次）
    file_index = FileIndex(base_dir) if ALLOW_GLOBAL_FILENAME_SEARCH else None
    if file_index is not None:
        print("[INFO] 正在建立全局视频文件索引（按文件名/主名）...")
        file_index.build()
        print("[INFO] 索引建立完成。\n")

    print(f"[INFO] 共发现 {len(json_files)} 个 .json，逐个处理…\n")

    rows: List[Dict[str, Any]] = []
    total_deleted = 0
    total_bytes = 0

    for jp in json_files:
        try:
            stat = process_one_json(
                json_path=jp,
                base_dir=base_dir,
                do_delete=DO_DELETE,
                overwrite_json=OVERWRITE_JSON,
                make_backup=MAKE_BACKUP,
                file_index=file_index
            )
        except Exception as e:
            stat = {"json": jp, "skipped": True, "reason": f"exception: {e}"}

        rows.append(stat)

        if not stat.get("skipped"):
            total_deleted += len(stat.get("deleted_files", []))
            total_bytes += stat.get("bytes_freed", 0) or 0

        # 控制台输出
        if stat.get("skipped"):
            print(f"  - 跳过 {jp}  (原因: {stat.get('reason','')})")
        else:
            print(f"  - 处理 {jp}")
            print(f"      保留动作: {stat.get('kept_action_count',0)}/{stat.get('orig_action_count',0)}")
            print(f"      保留视频: {stat.get('kept_video_count',0)}/{stat.get('orig_video_count',0)}")
            print(f"      移除视频候选: {stat.get('removed_video_count',0)}")
            if DO_DELETE:
                print(f"      实删文件数: {len(stat.get('deleted_files', []))}")
                print(f"      本JSON释放: {human_bytes(stat.get('bytes_freed', 0))}")

        if DO_TRIM:
            print(f"      裁剪成功: {stat.get('trimmed_count', 0)}")
            print(f"      裁剪失败: {stat.get('trim_failed_count', 0)}")
            if stat.get("trim_failed_count", 0) > 0:
                # 只展示前11条，避免刷屏
                for x in stat.get("trim_failed", [])[:11]:
                    print(f"        - {x}")

    report_path = write_report(base_dir, rows, REPORT_NAME)
    fail_path = write_trim_failures(base_dir, rows, "clip_failures.csv")
    print(f"[DONE] report: {report_path}")
    print(f"[DONE] failures: {fail_path}")

    if DO_DELETE:
        print(f"总计删除文件数: {total_deleted}")
        print(f"总计释放空间: {human_bytes(total_bytes)}")
    if DO_TRIM:
        total_trim_ok = sum(r.get("trimmed_count", 0) or 0 for r in rows)
        total_trim_fail = sum(r.get("trim_failed_count", 0) or 0 for r in rows)
        print(f"[SUMMARY] trim_ok={total_trim_ok}, trim_fail={total_trim_fail}")


if __name__ == "__main__":
    run()