# augment_keyframes.py
# ============================================================
# 关键帧增强:
# 1) 基于分割后的峰值帧做镜像增强(+ 指标镜像)
# 2) 对(原分割图 + 镜像分割图)做小角度旋转/亮度对比度增强(仅图像)
# ============================================================

import os
import sys
import sqlite3
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import cv2
import multiprocessing

# 建议: 使用一半核心给 OpenCV, 避免过度抢占系统
num_cores = multiprocessing.cpu_count()
cv2.setNumThreads(max(1, num_cores // 2))

from facialPalsy.action_feature_integrator import ActionFeatureIntegrator
from facialPalsy.face_segmenter import to_segmented_path
from facialPalsy.core.augmentation_utils import (
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
    """
    从 video_features 一行中解析出静态/动态指标字典

    返回:
        static_dict, dynamic_dict, static_names, dynamic_names
    """
    key_inds = afi.action_key_indicators.get(action_name)
    if key_inds is None:
        # 理论上不会发生
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


def save_mirror_features_into_db(
        conn: sqlite3.Connection,
        video_id: int,
        seg_mirror_path: str,
        static_mirror: Dict[str, float],
        dynamic_mirror: Dict[str, float],
        static_names: List[str],
        dynamic_names: List[str],
        static_dim: int,
        dynamic_dim: int,
        palsy_side: Optional[int]
):
    """
    将镜像后的指标插入 video_features 表，augmentation_type='mirror'
    """
    cur = conn.cursor()

    # 按既定顺序重新打包成 numpy 数组
    if static_dim > 0 and static_names:
        static_arr = np.array(
            [static_mirror.get(name, 0.0) for name in static_names],
            dtype=np.float32,
        )
        static_blob = static_arr.tobytes()
    else:
        static_blob = None

    if dynamic_dim > 0 and dynamic_names:
        dyn_arr = np.array(
            [dynamic_mirror.get(name, 0.0) for name in dynamic_names],
            dtype=np.float32,
        )
        dynamic_blob = dyn_arr.tobytes()
    else:
        dynamic_blob = None

    # 患侧翻转
    aug_palsy_side = flip_palsy_side(palsy_side) if palsy_side is not None else None

    # 直接基于原始那一行复制其他字段, 只改增强相关
    cur.execute(
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
            ?,                      -- augmentation_type
            ?,                      -- aug_palsy_side
            peak_frame_idx,
            peak_frame_path,        -- 原始峰值帧路径不变
            ?,                      -- 镜像后的分割图路径
            unit_length,
            ?,                      -- 静态特征BLOB(镜像)
            static_dim,
            ?,                      -- 动态特征BLOB(镜像)
            dynamic_dim,
            motion_features,
            motion_dim,
            geometry_processed_at,
            motion_processed_at
        FROM video_features
        WHERE video_id = ? AND augmentation_type = 'none'
        """,
        (
            "mirror",
            aug_palsy_side,
            seg_mirror_path,
            static_blob,
            dynamic_blob,
            video_id,
        ),
    )


def _color_jitter_hsv(img: np.ndarray,
                      delta_h: float,
                      scale_s: float,
                      scale_v: float) -> np.ndarray:
    """
    在 HSV 空间做颜色抖动:
    - delta_h: 色相偏移(单位: 0~180), 正值偏暖, 负值偏冷
    - scale_s: 饱和度缩放
    - scale_v: 明度缩放
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = cv2.split(hsv)

    # Hue 在 [0,180) 循环
    h = (h + delta_h) % 180.0
    # Saturation / Value 做缩放并裁剪
    s = np.clip(s * scale_s, 0.0, 255.0)
    v = np.clip(v * scale_v, 0.0, 255.0)

    hsv_j = cv2.merge([h, s, v]).astype(np.uint8)
    return cv2.cvtColor(hsv_j, cv2.COLOR_HSV2BGR)


def augment_images_only(base_path: str):
    """
    对一张(分割后的)关键帧图像做多种增强, 只在图像层面修改, 不改指标:
    1) 多角度轻微旋转(-7°, -4°, +4°, +7°)
    2) 亮度 / 对比度增强(力度更明显)
    3) 颜色抖动(色相、饱和度、明度)
    4) 灰度转换(黑白图)
    """
    if not os.path.exists(base_path):
        print(f"[WARN] 图像不存在, 跳过增强: {base_path}")
        return

    img = cv2.imread(base_path)
    if img is None:
        print(f"[WARN] cv2无法读取图像, 跳过增强: {base_path}")
        return

    h, w = img.shape[:2]
    stem = Path(base_path).stem
    parent = Path(base_path).parent

    # ---------- 1) 旋转增强 ----------
    # 控制在 ±7° 以内, 不会夸张到影响动作语义
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
        cv2.imwrite(str(out_path), img_rot)

    # ---------- 2) 亮度 / 对比度增强 ----------
    # brightness: 亮度系数, 对应你之前写的 brightness 参数
    # contrast:   对比度系数
    bc_configs = [
        ("brightStrong", 1.30, 1.0),    # 明显变亮
        ("darkStrong",   0.70, 1.0),    # 明显变暗
        ("contrastHigh", 1.0,  1.35),   # 对比度明显增强
        ("contrastLow",  1.0,  0.70),   # 对比度降低(偏灰)
        ("brightContrast", 1.25, 1.20), # 又亮又“硬”
    ]
    for tag, b, c in bc_configs:
        img_bc = adjust_brightness_contrast(img, brightness=b, contrast=c)
        out_path = parent / f"{stem}_{tag}.jpg"
        cv2.imwrite(str(out_path), img_bc)

    # ---------- 3) 颜色抖动(Color Jitter) ----------
    # 在 HSV 空间组合几种"风格":
    #  - 偏暖 + 饱和度略增
    #  - 偏冷 + 饱和度略减
    #  - 轻微曝光变化
    cj_configs = [
        ("cj_warm",   +12.0, 1.20, 1.00),  # 色相偏暖, 饱和度增加
        ("cj_cool",   -12.0, 0.85, 1.05),  # 色相偏冷, 饱和度略减, 明度略增
        ("cj_flat",    0.0,  0.75, 1.05),  # 色彩变平, 类似“低饱和滤镜”
        ("cj_vivid",   0.0,  1.30, 1.10),  # 饱和度+明度一起提升, 更鲜艳
    ]
    for tag, dh, ss, vv in cj_configs:
        img_cj = _color_jitter_hsv(img, delta_h=dh, scale_s=ss, scale_v=vv)
        out_path = parent / f"{stem}_{tag}.jpg"
        cv2.imwrite(str(out_path), img_cj)

    # ---------- 4) 灰度转换 ----------
    # 灰度图鼓励模型更多关注形状与纹理, 减少对颜色的依赖
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    out_path = parent / f"{stem}_gray.jpg"
    cv2.imwrite(str(out_path), gray_bgr)


def run_offline_augmentation(
        db_path: str,
        do_mirror: bool = True,
        do_rotate_and_bc: bool = True,
        commit_interval: int = 50,   # 每处理多少条样本提交一次
):
    """
    主入口:
    1) 对 augmentation_type='none' 的样本做镜像增强(含指标镜像 + 镜像分割图)
    2) 对 原分割图 + 镜像分割图 做旋转/亮度对比度/颜色/灰度增强(只在图像层)
    3) 每处理 commit_interval 条样本提交一次事务 -> 既加快速度又方便中途停止+重跑
    """
    conn = sqlite3.connect(db_path)

    # 让 SQLite 写入更快: WAL + synchronous=NORMAL
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")

    cur = conn.cursor()
    afi = ActionFeatureIntegrator()

    # 只取基础样本: augmentation_type='none'
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
    print(f"[AUG] 找到 {total} 条基础样本(augmentation_type='none')用于增强")

    processed = 0

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
    ) in enumerate(rows, start=1):

        # ---------- 0) 检查镜像记录是否已经存在 (支持中途停止+重跑) ----------
        cur.execute(
            "SELECT 1 FROM video_features WHERE video_id = ? AND augmentation_type = 'mirror'",
            (video_id,),
        )
        mirror_exists_in_db = cur.fetchone() is not None

        # 1) 确定分割后峰值帧路径
        if seg_path:
            seg_path_real = seg_path
        else:
            seg_path_real = to_segmented_path(peak_frame_path)

        if not seg_path_real or not os.path.exists(seg_path_real):
            print(f"[WARN] 分割图不存在, 跳过 video_id={video_id}, seg_path={seg_path_real}")
            continue

        # 2) 从 DB blob 中恢复原始指标
        static_dict, dynamic_dict, static_names, dynamic_names = load_indicators_from_db_row(
            action_name,
            static_blob,
            static_dim,
            dynamic_blob,
            dynamic_dim,
            afi,
        )

        seg_mirror_path = None

        # ---------- 3) 镜像增强: 只在 DB 里还没有 mirror 记录时才做 ----------
        if do_mirror and not mirror_exists_in_db:
            img_seg = cv2.imread(seg_path_real)
            if img_seg is None:
                print(f"[WARN] 无法读取分割图, 跳过mirror: {seg_path_real}")
            else:
                seg_mirror_path = str(
                    Path(seg_path_real).with_name(Path(seg_path_real).stem + "_mirror.jpg")
                )
                os.makedirs(Path(seg_mirror_path).parent, exist_ok=True)

                # 图像镜像
                img_seg_mirror = cv2.flip(img_seg, 1)
                cv2.imwrite(seg_mirror_path, img_seg_mirror)

                # 指标镜像
                static_mirror = mirror_indicators_dict(static_dict)
                dynamic_mirror = mirror_indicators_dict(dynamic_dict)

                # 写入 mirror 样本到 DB (不 commit, 由外层统一 commit)
                save_mirror_features_into_db(
                    conn,
                    video_id=video_id,
                    seg_mirror_path=seg_mirror_path,
                    static_mirror=static_mirror,
                    dynamic_mirror=dynamic_mirror,
                    static_names=static_names,
                    dynamic_names=dynamic_names,
                    static_dim=static_dim,
                    dynamic_dim=dynamic_dim,
                    palsy_side=palsy_side,
                )

                print(f"[AUG] video_id={video_id}, action={action_name}: 已生成镜像样本")
        else:
            # DB里已经存在 mirror 行, 推断 mirror 图路径(用于图像增强)
            seg_mirror_path = str(
                Path(seg_path_real).with_name(Path(seg_path_real).stem + "_mirror.jpg")
            )

        # ---------- 4) 图像级别增强(旋转 + 亮度/对比度/颜色/灰度) ----------
        if do_rotate_and_bc:
            # 原分割图增强
            augment_images_only(seg_path_real)

            # 镜像分割图增强(如果存在 mirror 图)
            if seg_mirror_path and os.path.exists(seg_mirror_path):
                augment_images_only(seg_mirror_path)

        processed += 1

        # ---------- 5) 批量提交事务 ----------
        if processed % commit_interval == 0:
            conn.commit()
            print(f"[AUG] 已处理 {processed}/{total} 条样本, 已提交一次事务")

    # 循环结束后再提交一次, 确保最后一批保存
    conn.commit()
    conn.close()
    print(f"[AUG] 离线数据增强完成, 共处理 {processed}/{total} 条样本")


if __name__ == "__main__":
    # TODO: 根据你的实际路径修改
    DB_PATH = "facialPalsy.db"

    # 先确保已经跑过:
    #   1) video_pipeline.py
    #   2) face_segmenter.py 的 batch_process_keyframes + update_segmented_paths

    run_offline_augmentation(
        db_path=DB_PATH,
        do_mirror=True,          # 是否做镜像增强(含指标镜像)
        do_rotate_and_bc=True,   # 是否做旋转 + 亮度/对比度图像增强
    )
