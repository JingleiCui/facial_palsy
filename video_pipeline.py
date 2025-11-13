"""
视频处理Pipeline V2 - 利用现有的actions/*.py
保留每个动作的特定特征,不强制统一维度
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
# 这样可以直接运行 video_pipeline.py,同时支持相对导入
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import cv2
import numpy as np
import sqlite3
import json
from datetime import datetime

from facialPalsy.core.landmark_extractor import LandmarkExtractor
from facialPalsy.action_feature_integrator import ActionFeatureIntegrator

# 导入动作类
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


class VideoPipeline:
    """
    视频处理Pipeline

    关键改进:
    1. 使用现有的动作类(actions/*.py)
    2. 保留每个动作的特定特征
    3. 利用与静息帧的对比
    4. 支持动作特异性特征
    """

    def __init__(self, db_path, keyframe_root_dir):
        """
        Args:
            db_path: 数据库路径
            keyframe_root_dir: 关键帧保存根目录
        """
        self.db_path = db_path
        self.keyframe_root_dir = Path(keyframe_root_dir)
        self.keyframe_root_dir.mkdir(parents=True, exist_ok=True)

        # 初始化landmark提取器
        self.landmark_extractor = LandmarkExtractor()

        # 初始化特征整合器
        self.feature_integrator = ActionFeatureIntegrator()

        # 初始化动作检测器
        self.action_detectors = self.feature_integrator.action_detectors

        # 静息帧缓存
        self.neutral_cache = {}

    def process_examination(self, examination_id):
        """
        处理一个完整的examination(11个动作)

        这是推荐的处理方式,因为可以利用静息帧与其他动作对比

        Args:
            examination_id: 检查ID

        Returns:
            dict: 处理结果
        """
        print(f"\n{'=' * 60}")
        print(f"处理检查 ID: {examination_id}")
        print(f"{'=' * 60}")

        start_time = datetime.now()

        # 1. 获取该examination的所有视频
        videos = self._get_examination_videos(examination_id)

        if not videos:
            print(f"[ERROR] 检查 {examination_id} 没有视频")
            return None

        print(f"找到 {len(videos)} 个视频")

        # 2. 首先处理NeutralFace(静息帧)
        neutral_result = None
        neutral_video = next((v for v in videos if v['action_name_en'] == 'NeutralFace'), None)

        if neutral_video:
            print("\n[步骤1] 处理静息帧...")
            neutral_result = self.process_video(
                neutral_video['video_id'],
                neutral_indicators=None  # 静息帧不需要对比
            )

            if neutral_result:
                # 缓存静息帧指标
                self.neutral_cache[examination_id] = {
                    'normalized_indicators': neutral_result['normalized_indicators'],
                    'peak_frame_idx': neutral_result['peak_frame_idx']
                }
                print(f"✓ 静息帧处理完成")

        # 3. 处理其他10个动作
        results = {}
        other_videos = [v for v in videos if v['action_name_en'] != 'NeutralFace']

        for i, video in enumerate(other_videos, 1):
            print(f"\n[步骤{i + 1}] 处理 {video['action_name_cn']}...")

            # 获取静息帧指标(用于对比)
            neutral_indicators = None
            if examination_id in self.neutral_cache:
                neutral_indicators = self.neutral_cache[examination_id]['normalized_indicators']

            result = self.process_video(
                video['video_id'],
                neutral_indicators=neutral_indicators
            )

            if result:
                results[video['action_name_en']] = result
                print(f"✓ {video['action_name_cn']} 处理完成")

        # 4. 添加静息帧结果
        if neutral_result:
            results['NeutralFace'] = neutral_result

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        print(f"\n{'=' * 60}")
        print(f"✓ 检查处理完成! 耗时: {processing_time:.2f}ms")
        print(f"成功处理: {len(results)}/11 个动作")
        print(f"{'=' * 60}")

        return {
            'examination_id': examination_id,
            'results': results,
            'processing_time_ms': processing_time
        }

    def process_video(self, video_id, neutral_indicators=None):
        """
        处理单个视频

        Args:
            video_id: 视频ID
            neutral_indicators: 静息帧的归一化指标(用于对比)

        Returns:
            dict: 处理结果
        """
        # 1. 获取视频信息
        video_info = self._get_video_info(video_id)
        if not video_info:
            print(f"[ERROR] 视频ID {video_id} 不存在")
            return None

        action_name = video_info['action_name_en']
        print(f"  动作: {action_name} ({video_info['action_name_cn']})")

        # 2. 检查文件
        if not os.path.exists(video_info['file_path']):
            print(f"  [ERROR] 文件不存在: {video_info['file_path']}")
            return None

        # 3. 提取landmarks序列和frames
        landmarks_seq, frames_seq = self._extract_sequence(
            video_info['file_path'],
            video_info['start_frame'],
            video_info['end_frame']
        )

        if not landmarks_seq:
            print(f"  [ERROR] 关键点提取失败")
            return None

        # 4. 获取动作检测器
        detector = self.action_detectors.get(action_name)
        if not detector:
            print(f"  [ERROR] 未找到动作检测器: {action_name}")
            return None

        # 5. 使用动作类的process方法
        # 注意: 需要转换neutral_indicators格式
        neutral_raw = self._denormalize_indicators(
            neutral_indicators,
            video_info
        ) if neutral_indicators else None

        result = detector.process(
            landmarks_seq=landmarks_seq,
            frames_seq=frames_seq,
            w=video_info.get('width', 640),
            h=video_info.get('height', 480),
            fps=video_info.get('fps', 30.0),
            neutral_indicators=neutral_raw
        )

        if not result:
            print(f"  [ERROR] 处理失败")
            return None

        # 6. 提取特征向量
        feature_vector = self.feature_integrator.extract_action_features(
            action_name,
            result['normalized_indicators'],
            result['normalized_dynamic_features']
        )

        print(f"  ✓ 特征维度: {feature_vector.shape[0]}")

        # 7. 保存峰值帧
        peak_frame_path = self._save_peak_frame(
            result['peak_frame'],
            video_info['examination_id'],
            action_name
        )

        # 8. 存储到数据库
        self._save_to_database(
            video_id=video_id,
            peak_frame_idx=result['peak_frame_idx'],
            peak_frame_path=str(peak_frame_path),
            unit_length=result['unit_length'],
            feature_vector=feature_vector,
            normalized_indicators=result['normalized_indicators'],
            normalized_dynamic_features=result['normalized_dynamic_features']
        )

        return {
            'video_id': video_id,
            'action_name': action_name,
            'peak_frame_idx': result['peak_frame_idx'],
            'peak_frame_path': str(peak_frame_path),
            'unit_length': result['unit_length'],
            'feature_dim': feature_vector.shape[0],
            'feature_vector': feature_vector,
            'normalized_indicators': result['normalized_indicators'],
            'normalized_dynamic_features': result['normalized_dynamic_features']
        }

    def process_all_examinations(self):
        """批量处理所有未处理的examinations"""
        # 获取所有未完全处理的examinations
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 找出至少有一个视频未处理的examinations
        cursor.execute("""
            SELECT DISTINCT e.examination_id
            FROM examinations e
            INNER JOIN video_files vf ON e.examination_id = vf.examination_id
            LEFT JOIN video_features feat ON vf.video_id = feat.video_id
            WHERE vf.file_exists = 1 AND feat.feature_id IS NULL
            ORDER BY e.examination_id
        """)

        examination_ids = [row[0] for row in cursor.fetchall()]
        conn.close()

        print(f"\n找到 {len(examination_ids)} 个需要处理的检查")

        results = []
        for i, exam_id in enumerate(examination_ids, 1):
            print(f"\n{'#' * 60}")
            print(f"进度: {i}/{len(examination_ids)}")
            print(f"{'#' * 60}")

            try:
                result = self.process_examination(exam_id)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"[ERROR] 处理检查 {exam_id} 时出错: {str(e)}")
                import traceback
                traceback.print_exc()

        print(f"\n{'=' * 60}")
        print(f"批量处理完成!")
        print(f"成功: {len(results)}/{len(examination_ids)}")
        print(f"{'=' * 60}")

        return results

    def _get_video_info(self, video_id):
        """从数据库获取视频信息"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT 
                v.video_id,
                v.examination_id,
                v.action_id,
                v.file_path,
                v.start_frame,
                v.end_frame,
                v.fps,
                at.action_name_en,
                at.action_name_cn
            FROM video_files v
            INNER JOIN action_types at ON v.action_id = at.action_id
            WHERE v.video_id = ?
        """, (video_id,))

        row = cursor.fetchone()
        conn.close()

        return dict(row) if row else None

    def _get_examination_videos(self, examination_id):
        """获取某个examination的所有视频"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("""
            SELECT 
                v.video_id,
                v.examination_id,
                v.file_path,
                v.start_frame,
                v.end_frame,
                v.fps,
                at.action_name_en,
                at.action_name_cn
            FROM video_files v
            INNER JOIN action_types at ON v.action_id = at.action_id
            WHERE v.examination_id = ? AND v.file_exists = 1
            ORDER BY at.action_id
        """, (examination_id,))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def _extract_sequence(self, video_path, start_frame, end_frame):
        """提取视频序列"""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return None, None

        landmarks_seq = []
        frames_seq = []

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_idx = start_frame

        while cap.isOpened() and frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # MediaPipe提取 - 返回2D landmarks对象
            landmarks = self.landmark_extractor.extract(frame)

            landmarks_seq.append(landmarks)
            frames_seq.append(frame)
            frame_idx += 1

        cap.release()

        return landmarks_seq, frames_seq

    def _denormalize_indicators(self, normalized_indicators, video_info):
        """
        将归一化指标转换回原始像素值(用于动作类)

        这是一个近似转换,因为我们没有原始的unit_length
        """
        # TODO: 如果需要精确的原始值,需要从数据库读取NeutralFace的unit_length
        # 这里暂时返回归一化值,大部分动作类可以处理
        return normalized_indicators

    def _save_peak_frame(self, frame, examination_id, action_name):
        """保存峰值帧"""
        action_dir = self.keyframe_root_dir / action_name
        action_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{examination_id}_{action_name}.jpg"
        filepath = action_dir / filename

        cv2.imwrite(str(filepath), frame)

        return filepath

    def _save_to_database(self, video_id, peak_frame_idx, peak_frame_path,
                          unit_length, feature_vector, normalized_indicators,
                          normalized_dynamic_features):
        """保存到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 将特征向量转为BLOB
        feature_blob = feature_vector.tobytes()

        # 将指标转为JSON
        indicators_json = json.dumps(normalized_indicators)
        dynamic_json = json.dumps(normalized_dynamic_features)

        try:
            cursor.execute("""
                INSERT INTO video_features (
                    video_id,
                    peak_frame_idx,
                    peak_frame_path,
                    unit_length,
                    feature_vector,
                    feature_dim,
                    indicators_json,
                    dynamic_features_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                video_id,
                peak_frame_idx,
                peak_frame_path,
                unit_length,
                feature_blob,
                len(feature_vector),
                indicators_json,
                dynamic_json
            ))

            conn.commit()

        except sqlite3.IntegrityError:
            # 更新
            cursor.execute("""
                UPDATE video_features
                SET peak_frame_idx = ?,
                    peak_frame_path = ?,
                    unit_length = ?,
                    feature_vector = ?,
                    feature_dim = ?,
                    indicators_json = ?,
                    dynamic_features_json = ?,
                    processed_at = CURRENT_TIMESTAMP
                WHERE video_id = ?
            """, (
                peak_frame_idx,
                peak_frame_path,
                unit_length,
                feature_blob,
                len(feature_vector),
                indicators_json,
                dynamic_json,
                video_id
            ))

            conn.commit()

        finally:
            conn.close()


def main():
    """主函数：方便在 PyCharm 里一键运行"""

    # 1️⃣ 基本路径配置 —— 按实际路径改，不用命令行了
    db_path = 'facialPalsy.db'
    keyframe_dir = '/Users/cuijinglei/Documents/facialPalsy/pipeline/keyframes'

    # 2️⃣ 选择运行模式（**只需要改这里几行变量**）
    # 说明：
    #   - 如果只想处理某一个 examination：把 examination_id 改成具体 ID（例如 'XW000001_20240101'）
    #   - 如果只想处理某个视频：把 video_id 改成具体 video_id（整数），比如 123
    #   - 如果想批量处理所有未处理的 examinations：保持 run_batch = True 即可
    examination_id = None   # 例如：'XW000001_20240101'，默认 None
    video_id = None         # 例如：123，默认 None
    run_batch = True        # 默认批量处理所有未处理的 examinations

    # 3️⃣ 初始化 Pipeline
    pipeline = VideoPipeline(db_path, keyframe_dir)

    # 4️⃣ 根据上面配置决定怎么跑
    if examination_id is not None:
        # 处理一个完整的检查（推荐，因为会先用 NeutralFace 做对比）
        pipeline.process_examination(examination_id)
    elif video_id is not None:
        # 只处理单个视频（不会自动处理同一个检查下的其他动作）
        pipeline.process_video(video_id)
    elif run_batch:
        # 批量处理所有未处理的 examinations
        pipeline.process_all_examinations()
    else:
        print("当前没有配置任何处理任务，请在 main() 中设置 examination_id / video_id 或 run_batch = True")


if __name__ == '__main__':
    main()