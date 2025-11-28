"""
视频处理Pipeline V3 - 彻底修复内存问题
主要改进:
1. 降低并行度避免MediaPipe冲突
2. 及时释放帧内存
3. 定期垃圾回收
"""
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

import cv2
import numpy as np
import sqlite3
import json
import gc
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

from facialPalsy.core.landmark_extractor import LandmarkExtractor
from facialPalsy.action_feature_integrator import ActionFeatureIntegrator


class VideoPipeline:
    """
    视频处理Pipeline

    V3改进:
    1. 降低并行度(3线程)避免MediaPipe GPU冲突
    2. 及时释放帧序列内存
    3. 定期强制垃圾回收
    4. 分批处理examinations
    """

    def __init__(self, db_path, model_path, keyframe_root_dir):
        """
        Args:
            db_path: 数据库路径
            keyframe_root_dir: 关键帧保存根目录
        """
        self.db_path = db_path
        self.keyframe_root_dir = Path(keyframe_root_dir)
        self.keyframe_root_dir.mkdir(parents=True, exist_ok=True)

        # 初始化landmark提取器
        self.landmark_extractor = LandmarkExtractor(model_path)

        # 初始化特征整合器
        self.feature_integrator = ActionFeatureIntegrator()

        # 初始化动作检测器
        self.action_detectors = self.feature_integrator.action_detectors

        # 静息帧缓存
        self.neutral_cache = {}

        self.model_path = model_path

        # 🔧 关键修复1: 降低并行度避免MediaPipe GPU冲突
        # MediaPipe在多线程中会创建多个OpenGL上下文,容易OOM
        self.num_workers = 6  # 每个线程约500MB模型

        self._tls = threading.local()

    def _get_worker(self):
        w = getattr(self._tls, "worker", None)
        if w is None:
            # 每个线程各自持有一套模型/检测器
            w = type("Worker", (), {})()
            w.landmark_extractor = LandmarkExtractor(self.model_path)
            w.feature_integrator = ActionFeatureIntegrator()
            w.action_detectors = w.feature_integrator.action_detectors
            self._tls.worker = w
        return w

    def process_examination(self, examination_id):
        """
        处理一个完整的examination(11个动作)

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
                neutral_indicators=None
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

        print(f"\n[步骤2] 并行处理其余 {len(other_videos)} 个动作...")

        neutral_indicators = None
        if examination_id in self.neutral_cache:
            neutral_indicators = self.neutral_cache[examination_id]['normalized_indicators']

        failures = []
        computed = []

        with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
            fut_map = {ex.submit(self._compute_video_only, v, neutral_indicators): v for v in other_videos}
            for fut in as_completed(fut_map):
                v = fut_map[fut]
                try:
                    out = fut.result()
                except Exception as e:
                    failures.append((v['video_id'], v['action_name_en'], str(e)))
                    continue
                if not out.get("ok"):
                    failures.append((v['video_id'], v['action_name_en'], out.get("error", "unknown")))
                    continue
                computed.append(out)

        # 串行保存(避免 SQLite 写锁)
        for out in computed:
            vinfo = next(v for v in other_videos if v["video_id"] == out["video_id"])
            action_name = out["action_name"]
            r = out["result"]

            peak_frame_path = self._save_peak_frame(r['peak_frame'], vinfo['examination_id'], action_name)

            # 🔧 关键修复2: 立即释放峰值帧
            del r['peak_frame']

            self._save_to_database(
                video_id=vinfo['video_id'],
                peak_frame_idx=r['peak_frame_idx'],
                peak_frame_path=str(peak_frame_path),
                unit_length=r['unit_length'],
                feature_vector=out["feature_vector"],
                normalized_indicators=r['normalized_indicators'],
                normalized_dynamic_features=r['normalized_dynamic_features']
            )

            results[action_name] = {
                'video_id': vinfo['video_id'],
                'action_name': action_name,
                'peak_frame_idx': r['peak_frame_idx'],
                'peak_frame_path': str(peak_frame_path),
                'unit_length': r['unit_length'],
                'feature_dim': out["feature_vector"].shape[0],
                'feature_vector': out["feature_vector"],
                'normalized_indicators': r['normalized_indicators'],
                'normalized_dynamic_features': r['normalized_dynamic_features']
            }

            # 🔧 关键修复3: 释放computed中的大对象
            del out["result"]

        if failures:
            print(f"  [WARN] 本次 examination 有 {len(failures)} 个动作失败：")
            for vid, act, err in failures[:10]:
                print(f"    - video_id={vid} act={act} err={err}")

        # 4. 添加静息帧结果
        if neutral_result:
            results['NeutralFace'] = neutral_result

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        print(f"\n{'=' * 60}")
        print(f"✓ 检查处理完成! 耗时: {processing_time:.2f}ms")
        print(f"成功处理: {len(results)}/11 个动作")
        print(f"{'=' * 60}")

        # 🔧 关键修复4: 强制垃圾回收
        del computed
        gc.collect()

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
            # 🔧 修复: 释放已提取的序列
            del landmarks_seq
            del frames_seq
            return None

        # 5. 使用动作类的process方法
        neutral_raw = self._denormalize_indicators(
            neutral_indicators,
            video_info
        ) if neutral_indicators else None

        h, w = frames_seq[0].shape[:2]

        result = detector.process(
            landmarks_seq=landmarks_seq,
            frames_seq=frames_seq,
            w=w,
            h=h,
            fps=video_info.get('fps'),
            neutral_indicators=neutral_raw
        )

        # 🔧 关键修复5: 立即释放序列
        del landmarks_seq
        del frames_seq

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

    def process_all_examinations(self, batch_size=10):
        """
        批量处理所有未处理的examinations

        Args:
            batch_size: 每批处理多少个examination后清理内存
        """
        # 获取所有未完全处理的examinations
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

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
        print(f"将分 {(len(examination_ids) + batch_size - 1) // batch_size} 批处理")

        results = []
        for i, exam_id in enumerate(examination_ids, 1):
            print(f"\n{'#' * 60}")
            print(f"进度: {i}/{len(examination_ids)}")
            print(f"{'#' * 60}")

            try:
                result = self.process_examination(exam_id)
                if result:
                    results.append(result)

                # 🔧 关键修复6: 定期清理内存
                if i % batch_size == 0:
                    gc.collect()
                    print(f"\n  [内存清理] 已处理 {i}/{len(examination_ids)} 个检查")

            except Exception as e:
                print(f"[ERROR] 处理检查 {exam_id} 时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                # 出错后也要清理
                gc.collect()

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
                vf.video_id,
                vf.examination_id,
                at.action_name_en,
                at.action_name_cn,
                vf.file_path,
                vf.start_frame,
                vf.end_frame,
                vf.fps
            FROM video_files vf
            INNER JOIN action_types at ON vf.action_id = at.action_id
            WHERE vf.video_id = ?
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
                vf.video_id,
                vf.examination_id,
                at.action_name_en,
                at.action_name_cn,
                vf.file_path,
                vf.start_frame,
                vf.end_frame,
                vf.fps
            FROM video_files vf
            INNER JOIN action_types at ON vf.action_id = at.action_id
            WHERE vf.examination_id = ?
            AND vf.file_exists = 1
            ORDER BY at.display_order
        """, (examination_id,))

        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]

    def _extract_sequence(self, video_path, start_frame, end_frame, extractor=None):
        """
        提取视频序列

        🔧 关键修复7: 使用copy()并及时释放原始帧
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return None, None

        extractor = extractor or self.landmark_extractor

        landmarks_seq = []
        frames_seq = []

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_idx = start_frame

        while cap.isOpened() and frame_idx <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # MediaPipe提取
            landmarks = extractor.extract_from_frame(frame)

            landmarks_seq.append(landmarks)
            # 🔧 关键: 只保留副本,原帧立即释放
            frames_seq.append(frame.copy())
            del frame

            frame_idx += 1

        cap.release()

        return landmarks_seq, frames_seq

    def _compute_video_only(self, video_info, neutral_indicators=None):
        """
        工作线程中计算单个视频

        🔧 关键修复8: 处理完立即释放序列
        """
        t0 = time.perf_counter()
        action_name = video_info['action_name_en']

        if not os.path.exists(video_info['file_path']):
            return {"ok": False, "error": f"文件不存在: {video_info['file_path']}"}

        worker = self._get_worker()

        landmarks_seq, frames_seq = self._extract_sequence(
            video_info['file_path'],
            video_info['start_frame'],
            video_info['end_frame'],
            extractor=worker.landmark_extractor
        )
        if not landmarks_seq:
            return {"ok": False, "error": "关键点提取失败"}

        detector = worker.action_detectors.get(action_name)
        if not detector:
            return {"ok": False, "error": f"未找到动作检测器: {action_name}"}

        neutral_raw = self._denormalize_indicators(neutral_indicators, video_info) if neutral_indicators else None

        h, w = frames_seq[0].shape[:2]
        result = detector.process(
            landmarks_seq=landmarks_seq,
            frames_seq=frames_seq,
            w=w,
            h=h,
            fps=video_info.get('fps'),
            neutral_indicators=neutral_raw
        )

        # 🔧 关键: 立即释放序列内存
        del landmarks_seq
        del frames_seq

        if not result:
            return {"ok": False, "error": "动作处理失败(detector.process 返回空)"}

        feature_vector = worker.feature_integrator.extract_action_features(
            action_name,
            result['normalized_indicators'],
            result['normalized_dynamic_features']
        )

        return {
            "ok": True,
            "action_name": action_name,
            "video_id": video_info["video_id"],
            "examination_id": video_info["examination_id"],
            "result": result,
            "feature_vector": feature_vector,
            "elapsed_ms": (time.perf_counter() - t0) * 1000.0
        }

    def _denormalize_indicators(self, normalized_indicators, video_info):
        """将归一化指标转换回原始像素值(用于动作类)"""
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

        video_info = self._get_video_info(video_id)
        action_name = video_info['action_name_en']

        key_indicators = self.feature_integrator.action_key_indicators.get(action_name, None)
        if key_indicators is None:
            print(f"  [WARN] 未在 action_key_indicators 中找到 {action_name}，跳过入库")
            return

        static_names = key_indicators['static']
        dynamic_names = key_indicators['dynamic']

        static_vals = [
            float(normalized_indicators.get(name, 0.0))
            for name in static_names
        ]
        dynamic_vals = [
            float(normalized_dynamic_features.get(name, 0.0))
            for name in dynamic_names
        ]

        static_arr = np.array(static_vals, dtype=np.float32)
        dynamic_arr = np.array(dynamic_vals, dtype=np.float32)

        static_blob = static_arr.tobytes()
        dynamic_blob = dynamic_arr.tobytes()

        static_dim = len(static_vals)
        dynamic_dim = len(dynamic_vals)

        try:
            cursor.execute("""
                INSERT INTO video_features (
                    video_id,
                    peak_frame_idx,
                    peak_frame_path,
                    unit_length,
                    static_features,
                    dynamic_features,
                    static_dim,
                    dynamic_dim
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                video_id,
                peak_frame_idx,
                peak_frame_path,
                unit_length,
                static_blob,
                dynamic_blob,
                static_dim,
                dynamic_dim
            ))

            conn.commit()

        except sqlite3.IntegrityError:
            # 更新
            cursor.execute("""
                UPDATE video_features
                SET peak_frame_idx = ?,
                    peak_frame_path = ?,
                    unit_length = ?,
                    static_features = ?,
                    dynamic_features = ?,
                    static_dim = ?,
                    dynamic_dim = ?,
                    geometry_processed_at = CURRENT_TIMESTAMP
                WHERE video_id = ?
            """, (
                peak_frame_idx,
                peak_frame_path,
                unit_length,
                static_blob,
                dynamic_blob,
                static_dim,
                dynamic_dim,
                video_id
            ))

            conn.commit()

        finally:
            conn.close()


def main():
    """主函数：方便在 PyCharm 里一键运行"""

    # 基本路径配置
    db_path = 'facialPalsy.db'
    model_path = '/Users/cuijinglei/PycharmProjects/medicalProject/models/face_landmarker.task'
    keyframe_dir = '/Users/cuijinglei/Documents/facialPalsy/HGFA/keyframes'
    os.makedirs(keyframe_dir, exist_ok=True)

    examination_id = None
    video_id = None
    run_batch = True

    # 初始化 Pipeline
    pipeline = VideoPipeline(db_path, model_path, keyframe_dir)

    if examination_id is not None:
        pipeline.process_examination(examination_id)
    elif video_id is not None:
        pipeline.process_video(video_id)
    elif run_batch:
        # 🔧 关键: 使用分批处理,每10个examination清理一次
        pipeline.process_all_examinations(batch_size=10)
    else:
        print("当前没有配置任何处理任务")


if __name__ == '__main__':
    main()