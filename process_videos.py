"""
视频处理主脚本 - 阶段1+阶段2整合

功能:
1. 从数据库读取待处理的视频记录
2. 对每个视频:
   a) 提取关键点序列 (阶段1)
   b) 检测峰值帧 (阶段1)
   c) 提取静态+动态几何特征 (阶段2 - 同时进行)
3. 保存结果到数据库和文件

使用方法:
    python facialPalsy/process_videos.py
"""

import os
import sys
import sqlite3
import cv2
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from core.landmark_extractor import LandmarkExtractor
from core.database_manager import DatabaseManager
from actions import get_action_class


class VideoProcessor:
    """视频处理器 - 整合阶段1和阶段2"""

    def __init__(self, db_path, mediapipe_model_path):
        """
        初始化

        Args:
            db_path: 数据库路径
            mediapipe_model_path: MediaPipe模型路径
        """
        self.db_path = db_path
        self.mediapipe_model_path = mediapipe_model_path
        self.db_manager = DatabaseManager(db_path)

    def process_all_videos(self, limit=None, skip_processed=True):
        """
        处理所有视频

        Args:
            limit: 限制处理数量,None表示全部
            skip_processed: 是否跳过已处理的视频
        """
        # 获取待处理的视频列表
        videos = self._get_videos_to_process(skip_processed, limit)

        if not videos:
            print("没有需要处理的视频")
            return

        print(f"=" * 80)
        print(f"开始处理 {len(videos)} 个视频")
        print(f"=" * 80)

        success_count = 0
        fail_count = 0

        # 使用上下文管理器创建LandmarkExtractor
        with LandmarkExtractor(self.mediapipe_model_path) as extractor:
            for i, video_info in enumerate(videos, 1):
                print(f"\n[{i}/{len(videos)}] 处理视频: {video_info['video_id']}")
                print(f"  路径: {video_info['file_path']}")
                print(f"  动作: {video_info['action_name']}")

                try:
                    result = self._process_single_video(extractor, video_info)

                    if result:
                        self._save_results(video_info, result)
                        success_count += 1
                        print(f"  ✓ 成功")
                    else:
                        fail_count += 1
                        print(f"  ✗ 失败")

                except Exception as e:
                    fail_count += 1
                    print(f"  ✗ 异常: {e}")
                    import traceback
                    traceback.print_exc()

        # 打印汇总
        print(f"\n" + "=" * 80)
        print(f"处理完成!")
        print(f"  成功: {success_count}")
        print(f"  失败: {fail_count}")
        print(f"=" * 80)

    def _get_videos_to_process(self, skip_processed, limit):
        """获取待处理的视频列表"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = """
            SELECT
                v.video_id,
                v.examination_id,
                v.action_id,
                v.file_path,
                v.start_frame,
                v.end_frame,
                v.is_processed,
                v.action_name_en,
                v.action_name_cn,
                v.file_exists
            FROM videos v
        """

        if skip_processed:
            query += " WHERE v.is_processed = 0 AND v.file_exists = 1"
        else:
            query += " WHERE v.file_exists = 1"

        query += " ORDER BY v.examination_id, v.action_id"

        if limit:
            query += f" LIMIT {limit}"

        cursor.execute(query)
        rows = cursor.fetchall()

        videos = []
        for row in rows:
            videos.append({
                'video_id': row[0],
                'examination_id': row[1],
                'action_id': row[2],
                'file_path': row[3],
                'start_frame': row[4],
                'end_frame': row[5],
                'is_processed': row[6],
                'action_name': row[7],
                'action_name_cn': row[8]
            })

        conn.close()
        return videos

    def _process_single_video(self, extractor, video_info):
        """
        处理单个视频 - 阶段1+阶段2整合

        Args:
            extractor: LandmarkExtractor实例
            video_info: 视频信息字典

        Returns:
            dict: 处理结果,包含峰值帧、特征等,失败返回None
        """
        video_path = video_info['file_path']
        action_name = video_info['action_name']
        start_frame = video_info['start_frame']
        end_frame = video_info['end_frame']

        # 检查文件是否存在
        if not os.path.exists(video_path):
            print(f"  错误: 文件不存在")
            return None

        # === 阶段1: 提取关键点序列 ===
        print(f"  提取关键点序列...")
        landmarks_seq, frames_seq = extractor.extract_sequence(
            video_path,
            start_frame=start_frame,
            end_frame=end_frame
        )

        if not landmarks_seq or not frames_seq:
            print(f"  错误: 关键点提取失败")
            return None

        # 检查有效帧比例
        valid_count = sum(1 for lm in landmarks_seq if lm is not None)
        valid_ratio = valid_count / len(landmarks_seq)

        print(f"  有效帧: {valid_count}/{len(landmarks_seq)} ({valid_ratio*100:.1f}%)")

        if valid_ratio < MIN_VALID_FRAME_RATIO:
            print(f"  错误: 有效帧比例过低 (< {MIN_VALID_FRAME_RATIO*100}%)")
            return None

        # 获取视频信息
        video_info_obj = extractor.get_video_info(video_path)
        w = video_info_obj['width']
        h = video_info_obj['height']
        fps = video_info_obj['fps']

        # === 阶段1: 检测峰值帧 + 阶段2: 提取特征 (同时进行) ===
        print(f"  检测峰值帧并提取特征...")

        # 获取动作类
        ActionClass = get_action_class(action_name)
        if not ActionClass:
            print(f"  错误: 未知动作类型 {action_name}")
            return None

        # 从配置获取动作配置
        action_config = ACTION_CONFIG.get(action_name)
        if not action_config:
            print(f"  错误: 未找到动作配置 {action_name}")
            return None

        # 创建动作实例并处理
        action = ActionClass(action_name, action_config)
        result = action.process(landmarks_seq, frames_seq, w, h, fps)

        if not result:
            print(f"  错误: 特征提取失败")
            return None

        print(f"  峰值帧: {result['peak_frame_idx']}")
        print(f"  静态特征: {result['static_features'].shape}")
        print(f"  动态特征: {result['dynamic_features'].shape}")

        return result

    def _save_results(self, video_info, result):
        """
        保存处理结果

        Args:
            video_info: 视频信息
            result: 处理结果
        """
        video_id = video_info['video_id']
        examination_id = video_info['examination_id']
        action_id = video_info['action_id']
        action_name = video_info['action_name']

        # === 1. 保存峰值帧图像 ===
        if SAVE_VISUALIZATION:
            peak_frame_filename = f"{examination_id}_{action_name}_peak.jpg"
            peak_frame_path = os.path.join(PEAK_FRAMES_DIR, peak_frame_filename)
            cv2.imwrite(peak_frame_path, result['peak_frame'])
            print(f"  保存峰值帧: {peak_frame_path}")

        # === 2. 保存特征到数据库 ===
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 保存到 video_features 表
        cursor.execute('''
            INSERT OR REPLACE INTO video_features (
                video_id,
                peak_frame_idx,
                peak_frame_path,
                unit_length,
                static_features,
                dynamic_features
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            video_id,
            result['peak_frame_idx'],
            peak_frame_path if SAVE_VISUALIZATION else None,
            float(result['unit_length']),
            result['static_features'].tobytes(),  # 保存为二进制
            result['dynamic_features'].tobytes()
        ))

        # 注意: is_processed 是通过video_features表的存在性自动计算的,不需要手动更新

        conn.commit()
        conn.close()

        print(f"  保存到数据库: video_id={video_id}")

        # === 3. 可选: 保存特征到numpy文件 ===
        if SAVE_FEATURES_TO_FILE:
            feature_filename = f"{examination_id}_{action_name}_features.npz"
            feature_path = os.path.join(FEATURES_DIR, feature_filename)

            np.savez_compressed(
                feature_path,
                static_features=result['static_features'],
                dynamic_features=result['dynamic_features'],
                peak_frame_idx=result['peak_frame_idx'],
                unit_length=result['unit_length']
            )
            print(f"  保存特征文件: {feature_path}")


def main():
    """主函数"""
    print("=" * 80)
    print("视频预处理与特征提取")
    print("=" * 80)

    # 创建处理器
    processor = VideoProcessor(
        db_path=DB_PATH,
        mediapipe_model_path=MEDIAPIPE_MODEL_PATH
    )

    # 处理所有未处理的视频
    processor.process_all_videos(
        limit=None,  # 处理全部,也可以设置如 limit=10 先测试
        skip_processed=True  # 跳过已处理的
    )


if __name__ == '__main__':
    main()
