# augment_keyframes_optimized.py
# ============================================================
# 优化版本 - 关键改进:
# 1. 多进程并行处理图像增强 (充分利用 M3 Max 的多核)
# 2. 批量数据库操作 (减少 I/O)
# 3. 内存优化 (避免重复加载)
# 4. 预先计算所有路径 (减少文件系统操作)
# ============================================================

import os
import sys
import sqlite3
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import cv2
import multiprocessing
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm
import time

# OpenCV 线程数设置为 1，避免与多进程冲突
cv2.setNumThreads(1)

from facial_palsy.action_feature_integrator import ActionFeatureIntegrator
from facial_palsy.face_segmenter import to_segmented_path
from facial_palsy.core.augmentation_utils import (
    flip_palsy_side,
    adjust_brightness_contrast,
    mirror_indicators_dict,
)


def load_indicators_from_db_row(
        action_name: str,
        static_blob: Optional[bytes],
        static_dim: int,
        dynamic_blob: Optional[bytes],
        dynamic_dim: int,
        afi: ActionFeatureIntegrator
) -> Tuple[Dict[str, float], Dict[str, float], List[str], List[str]]:
    """从 video_features 一行中解析出静态/动态指标字典"""
    key_inds = afi.action_key_indicators.get(action_name)
    if key_inds is None:
        return {}, {}, [], []

    static_names = key_inds["static"]
    dynamic_names = key_inds["dynamic"]

    # 静态
    static_dict: Dict[str, float] = {}
    if static_blob is not None and static_dim > 0 and len(static_names) == static_dim:
        static_arr = np.frombuffer(static_blob, dtype=np.float32, count=static_dim)
        for name, val in zip(static_names, static_arr):
            static_dict[name] = float(val)

    # 动态
    dynamic_dict: Dict[str, float] = {}
    if dynamic_blob is not None and dynamic_dim > 0 and len(dynamic_names) == dynamic_dim:
        dyn_arr = np.frombuffer(dynamic_blob, dtype=np.float32, count=dynamic_dim)
        for name, val in zip(dynamic_names, dyn_arr):
            dynamic_dict[name] = float(val)

    return static_dict, dynamic_dict, static_names, dynamic_names


def _color_jitter_hsv(img: np.ndarray,
                      delta_h: float,
                      scale_s: float,
                      scale_v: float) -> np.ndarray:
    """在 HSV 空间做颜色抖动"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    h = (h + delta_h) % 180.0
    s = np.clip(s * scale_s, 0.0, 255.0)
    v = np.clip(v * scale_v, 0.0, 255.0)

    hsv_j = cv2.merge([h, s, v]).astype(np.uint8)
    return cv2.cvtColor(hsv_j, cv2.COLOR_HSV2BGR)


def process_single_image_augmentation(args: Tuple[str, Path]) -> int:
    """
    处理单张图像的所有增强

    这个函数会被多进程调用，每个进程独立处理一张图像

    Returns:
        增强后生成的图像数量
    """
    base_path, parent = args

    if not os.path.exists(base_path):
        return 0

    img = cv2.imread(base_path)
    if img is None:
        return 0

    h, w = img.shape[:2]
    stem = Path(base_path).stem

    count = 0

    # 旋转增强
    angles = (-7, -4, 4, 7)
    for angle in angles:
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        img_rot = cv2.warpAffine(
            img,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        out_path = parent / f"{stem}_rot{angle:+d}.jpg"
        cv2.imwrite(str(out_path), img_rot, [cv2.IMWRITE_JPEG_QUALITY, 95])
        count += 1

    # 亮度/对比度增强
    bc_configs = [
        ("brightStrong", 1.30, 1.0),
        ("darkStrong", 0.70, 1.0),
        ("contrastHigh", 1.0, 1.35),
        ("contrastLow", 1.0, 0.70),
        ("brightContrast", 1.25, 1.20),
    ]
    for tag, b, c in bc_configs:
        img_bc = adjust_brightness_contrast(img, brightness=b, contrast=c)
        out_path = parent / f"{stem}_{tag}.jpg"
        cv2.imwrite(str(out_path), img_bc, [cv2.IMWRITE_JPEG_QUALITY, 95])
        count += 1

    # 颜色抖动
    cj_configs = [
        ("cj_warm", +12.0, 1.20, 1.00),
        ("cj_cool", -12.0, 0.85, 1.05),
        ("cj_flat", 0.0, 0.75, 1.05),
        ("cj_vivid", 0.0, 1.30, 1.10),
    ]
    for tag, dh, ss, vv in cj_configs:
        img_cj = _color_jitter_hsv(img, delta_h=dh, scale_s=ss, scale_v=vv)
        out_path = parent / f"{stem}_{tag}.jpg"
        cv2.imwrite(str(out_path), img_cj, [cv2.IMWRITE_JPEG_QUALITY, 95])
        count += 1

    # 灰度转换
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    out_path = parent / f"{stem}_gray.jpg"
    cv2.imwrite(str(out_path), gray_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    count += 1

    return count


def batch_save_mirror_features(
        conn: sqlite3.Connection,
        mirror_data_list: List[Tuple]
):
    """
    批量保存镜像特征到数据库

    Args:
        mirror_data_list: [(video_id, seg_mirror_path, static_blob, dynamic_blob, aug_palsy_side), ...]
    """
    if not mirror_data_list:
        return

    cur = conn.cursor()

    # 使用 executemany 批量插入
    cur.executemany(
        """
        INSERT INTO video_features (
            video_id,
            augmentation_type,
            aug_palsy_side,
            peak_frame_idx,
            peak_frame_path,
            peak_frame_segmented_path,
            unit_length,
            static_features,
            static_dim,
            dynamic_features,
            dynamic_dim,
            motion_features,
            motion_dim,
            geometry_processed_at,
            motion_processed_at
        )
        SELECT
            video_id,
            'mirror',
            ?,
            peak_frame_idx,
            peak_frame_path,
            ?,
            unit_length,
            ?,
            static_dim,
            ?,
            dynamic_dim,
            motion_features,
            motion_dim,
            geometry_processed_at,
            motion_processed_at
        FROM video_features
        WHERE video_id = ? AND augmentation_type = 'none'
        """,
        [(aug_palsy_side, seg_mirror_path, static_blob, dynamic_blob, video_id)
         for video_id, seg_mirror_path, static_blob, dynamic_blob, aug_palsy_side in mirror_data_list]
    )


def run_offline_augmentation_optimized(
        db_path: str,
        do_mirror: bool = True,
        do_rotate_and_bc: bool = True,
        num_workers: Optional[int] = None,
        db_batch_size: int = 100,
):
    """
    优化版主入口

    关键优化:
    1. 预先加载所有需要处理的数据（减少数据库查询）
    2. 镜像处理分批提交（减少 I/O）
    3. 图像增强使用多进程并行（充分利用 CPU）
    4. 预先检查已存在的镜像记录（避免重复处理）

    Args:
        num_workers: 进程数，None 表示使用 CPU 核心数
        db_batch_size: 数据库批量提交大小
    """
    if num_workers is None:
        num_workers = cpu_count()

    print(f"[OPT] 使用 {num_workers} 个进程并行处理")

    start_time = time.time()

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")

    cur = conn.cursor()
    afi = ActionFeatureIntegrator()

    # ========== 阶段1: 预先加载所有数据 ==========
    print("[OPT] 阶段1: 加载数据...")
    cur.execute(
        """
        SELECT
            vf.video_id,
            vf.examination_id,
            at.action_name_en,
            feat.peak_frame_idx,
            feat.peak_frame_path,
            feat.peak_frame_segmented_path,
            feat.static_features,
            feat.static_dim,
            feat.dynamic_features,
            feat.dynamic_dim,
            el.palsy_side
        FROM video_features AS feat
        JOIN video_files AS vf ON vf.video_id = feat.video_id
        JOIN action_types AS at ON at.action_id = vf.action_id
        LEFT JOIN examination_labels AS el ON el.examination_id = vf.examination_id
        WHERE feat.augmentation_type = 'none'
        """
    )
    rows = cur.fetchall()
    total = len(rows)
    print(f"[OPT] 找到 {total} 条基础样本")

    if total == 0:
        conn.close()
        return

    # ========== 阶段2: 检查已存在的镜像记录 ==========
    if do_mirror:
        print("[OPT] 阶段2: 检查已存在的镜像记录...")
        video_ids = [row[0] for row in rows]
        placeholders = ','.join('?' * len(video_ids))
        cur.execute(
            f"""
            SELECT video_id 
            FROM video_features 
            WHERE video_id IN ({placeholders}) 
              AND augmentation_type = 'mirror'
            """,
            video_ids
        )
        existing_mirrors = set(row[0] for row in cur.fetchall())
        print(f"[OPT] 已存在 {len(existing_mirrors)} 个镜像记录")
    else:
        existing_mirrors = set()

    # ========== 阶段3: 批量处理镜像增强 ==========
    mirror_count = 0
    mirror_data_batch = []
    image_paths_for_augmentation = []

    if do_mirror:
        print("[OPT] 阶段3: 处理镜像增强...")

        for idx, (
                video_id,
                examination_id,
                action_name,
                peak_frame_idx,
                peak_frame_path,
                seg_path,
                static_blob,
                static_dim,
                dynamic_blob,
                dynamic_dim,
                palsy_side,
        ) in enumerate(tqdm(rows, desc="镜像增强"), start=1):

            # 跳过已存在的镜像
            if video_id in existing_mirrors:
                # 但仍然需要记录路径用于后续的图像增强
                if seg_path:
                    seg_path_real = seg_path
                else:
                    seg_path_real = to_segmented_path(peak_frame_path)

                if seg_path_real and os.path.exists(seg_path_real):
                    seg_mirror_path = str(
                        Path(seg_path_real).with_name(Path(seg_path_real).stem + "_mirror.jpg")
                    )
                    if os.path.exists(seg_mirror_path):
                        image_paths_for_augmentation.append(seg_mirror_path)
                continue

            # 确定分割后峰值帧路径
            if seg_path:
                seg_path_real = seg_path
            else:
                seg_path_real = to_segmented_path(peak_frame_path)

            if not seg_path_real or not os.path.exists(seg_path_real):
                continue

            # 从 DB blob 中恢复原始指标
            static_dict, dynamic_dict, static_names, dynamic_names = load_indicators_from_db_row(
                action_name,
                static_blob,
                static_dim,
                dynamic_blob,
                dynamic_dim,
                afi,
            )

            # 读取并镜像图像
            img_seg = cv2.imread(seg_path_real)
            if img_seg is None:
                continue

            seg_mirror_path = str(
                Path(seg_path_real).with_name(Path(seg_path_real).stem + "_mirror.jpg")
            )
            os.makedirs(Path(seg_mirror_path).parent, exist_ok=True)

            # 图像镜像
            img_seg_mirror = cv2.flip(img_seg, 1)
            cv2.imwrite(seg_mirror_path, img_seg_mirror, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # 指标镜像
            static_mirror = mirror_indicators_dict(static_dict)
            dynamic_mirror = mirror_indicators_dict(dynamic_dict)

            # 打包成 numpy 数组
            if static_dim > 0 and static_names:
                static_arr = np.array(
                    [static_mirror.get(name, 0.0) for name in static_names],
                    dtype=np.float32,
                )
                static_blob_mirror = static_arr.tobytes()
            else:
                static_blob_mirror = None

            if dynamic_dim > 0 and dynamic_names:
                dyn_arr = np.array(
                    [dynamic_mirror.get(name, 0.0) for name in dynamic_names],
                    dtype=np.float32,
                )
                dynamic_blob_mirror = dyn_arr.tobytes()
            else:
                dynamic_blob_mirror = None

            # 患侧翻转
            aug_palsy_side = flip_palsy_side(palsy_side) if palsy_side is not None else None

            # 添加到批次
            mirror_data_batch.append((
                video_id,
                seg_mirror_path,
                static_blob_mirror,
                dynamic_blob_mirror,
                aug_palsy_side
            ))

            # 记录镜像图像路径，用于后续增强
            image_paths_for_augmentation.append(seg_mirror_path)

            mirror_count += 1

            # 批量提交
            if len(mirror_data_batch) >= db_batch_size:
                batch_save_mirror_features(conn, mirror_data_batch)
                conn.commit()
                mirror_data_batch = []

        # 提交剩余的批次
        if mirror_data_batch:
            batch_save_mirror_features(conn, mirror_data_batch)
            conn.commit()

        print(f"[OPT] 镜像增强完成: 新增 {mirror_count} 个镜像样本")

    # ========== 阶段4: 收集所有需要图像增强的路径 ==========
    if do_rotate_and_bc:
        print("[OPT] 阶段4: 收集图像增强路径...")

        # 添加原始分割图路径
        for row in rows:
            seg_path = row[5]  # peak_frame_segmented_path
            peak_frame_path = row[4]  # peak_frame_path

            if seg_path:
                seg_path_real = seg_path
            else:
                seg_path_real = to_segmented_path(peak_frame_path)

            if seg_path_real and os.path.exists(seg_path_real):
                image_paths_for_augmentation.append(seg_path_real)

        # 去重
        image_paths_for_augmentation = list(set(image_paths_for_augmentation))
        print(f"[OPT] 共 {len(image_paths_for_augmentation)} 张图像需要增强")

        # ========== 阶段5: 多进程并行处理图像增强 ==========
        print(f"[OPT] 阶段5: 多进程并行图像增强（{num_workers} 进程）...")

        # 准备参数：(图像路径, 父目录)
        args_list = [(img_path, Path(img_path).parent) for img_path in image_paths_for_augmentation]

        # 使用多进程池
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_image_augmentation, args_list),
                total=len(args_list),
                desc="图像增强"
            ))

        total_augmented = sum(results)
        print(f"[OPT] 图像增强完成: 生成 {total_augmented} 张增强图像")

    conn.close()

    elapsed = time.time() - start_time
    print(f"\n[OPT] 总耗时: {elapsed:.2f} 秒 ({elapsed / 60:.2f} 分钟)")
    print(f"[OPT] 平均速度: {total / elapsed:.2f} 样本/秒")


if __name__ == "__main__":
    DB_PATH = "facialPalsy.db"

    run_offline_augmentation_optimized(
        db_path=DB_PATH,
        do_mirror=True,
        do_rotate_and_bc=True,
        num_workers=6,  # 自动使用所有CPU核心
        db_batch_size=100,  # 每100个样本提交一次数据库
    )