"""
视频处理Pipeline V4 - 集成运动特征提取
========================================

处理流程:
1. 读取视频 → 提取landmarks序列和frames序列
2. 检测峰值帧 → 保存峰值帧图像
3. 提取几何特征 → static_features + dynamic_features
4. 计算运动特征 → motion_features (12维)
5. 存储到数据库

输出特征:
- static_features:  5-11维 (因动作而异)
- dynamic_features: 0-8维  (因动作而异)
- motion_features:  12维   (统一维度)

运动特征说明 (12维):
    0: mean_displacement     - 平均位移
    1: max_displacement      - 最大位移
    2: std_displacement      - 位移标准差
    3: motion_energy         - 运动能量
    4: motion_asymmetry      - 运动不对称性
    5: temporal_smoothness   - 时间平滑度
    6: spatial_concentration - 空间集中度
    7: peak_ratio            - 峰值区域比例
    8-9: motion_center       - 运动重心
    10: velocity_mean        - 平均速度
    11: acceleration_std     - 加速度变化

内存优化:
- 降低并行度(6线程)避免MediaPipe GPU冲突
- 及时释放帧序列内存
- 定期强制垃圾回收
- 分批处理examinations
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
import gc
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

from facialPalsy.core.landmark_extractor import LandmarkExtractor
from facialPalsy.core.motion_utils import compute_motion_features
from facialPalsy.action_feature_integrator import ActionFeatureIntegrator


class VideoPipeline:
    """
    视频处理Pipeline V4

    集成功能:
    1. 几何特征提取 (static + dynamic)
    2. 运动特征提取 (motion) - 复用landmarks_seq
    3. 峰值帧保存
    4. 内存优化
    """

    def __init__(self, db_path, model_path, keyframe_root_dir):
        """
        Args:
            db_path: 数据库路径
            model_path: MediaPipe FaceLandmarker模型路径
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

        # 降低并行度避免MediaPipe GPU冲突
        self.num_workers = 6

        self._tls = threading.local()

    def _get_worker(self):
        """获取线程本地的worker实例"""
        w = getattr(self._tls, "worker", None)
        if w is None:
            w = type("Worker", (), {})()
            w.landmark_extractor = LandmarkExtractor(self.model_path)
            w.feature_integrator = ActionFeatureIntegrator()
            w.action_detectors = w.feature_integrator.action_detectors
            self._tls.worker = w
        return w

    def process_examination(self, examination_id):
        """
        处理一个完整的examination(11个动作)
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

            del r['peak_frame']

            self._save_to_database(
                video_id=vinfo['video_id'],
                peak_frame_idx=r['peak_frame_idx'],
                peak_frame_path=str(peak_frame_path),
                unit_length=r['unit_length'],
                feature_vector=out["feature_vector"],
                normalized_indicators=r['normalized_indicators'],
                normalized_dynamic_features=r['normalized_dynamic_features'],
                motion_features=out.get("motion_features")
            )

            results[action_name] = {
                'video_id': vinfo['video_id'],
                'action_name': action_name,
                'peak_frame_idx': r['peak_frame_idx'],
                'peak_frame_path': str(peak_frame_path),
                'unit_length': r['unit_length'],
                'feature_dim': out["feature_vector"].shape[0],
                'motion_dim': 12,
                'feature_vector': out["feature_vector"],
                'motion_features': out.get("motion_features"),
                'normalized_indicators': r['normalized_indicators'],
                'normalized_dynamic_features': r['normalized_dynamic_features']
            }

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

        del computed
        gc.collect()

        return {
            'examination_id': examination_id,
            'results': results,
            'processing_time_ms': processing_time
        }

    def process_video(self, video_id, neutral_indicators=None):
        """
        处理单个视频 (主线程版本)
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
            del landmarks_seq
            del frames_seq
            return None

        # 5. 使用动作类的process方法
        neutral_raw = self._denormalize_indicators(
            neutral_indicators,
            video_info
        ) if neutral_indicators else None

        h, w = frames_seq[0].shape[:2]
        fps = video_info.get('fps')

        result = detector.process(
            landmarks_seq=landmarks_seq,
            frames_seq=frames_seq,
            w=w,
            h=h,
            fps=fps,
            neutral_indicators=neutral_raw
        )

        # 6. 计算运动特征 (复用landmarks_seq)
        motion_features = compute_motion_features(landmarks_seq, w, h, fps)

        # 释放序列
        del landmarks_seq
        del frames_seq

        if not result:
            print(f"  [ERROR] 处理失败")
            return None

        # 7. 提取特征向量
        feature_vector = self.feature_integrator.extract_action_features(
            action_name,
            result['normalized_indicators'],
            result['normalized_dynamic_features']
        )

        print(f"  ✓ 几何特征: {feature_vector.shape[0]}维, 运动特征: 12维")

        # 8. 保存峰值帧
        peak_frame_path = self._save_peak_frame(
            result['peak_frame'],
            video_info['examination_id'],
            action_name
        )

        # 9. 存储到数据库
        self._save_to_database(
            video_id=video_id,
            peak_frame_idx=result['peak_frame_idx'],
            peak_frame_path=str(peak_frame_path),
            unit_length=result['unit_length'],
            feature_vector=feature_vector,
            normalized_indicators=result['normalized_indicators'],
            normalized_dynamic_features=result['normalized_dynamic_features'],
            motion_features=motion_features
        )

        return {
            'video_id': video_id,
            'action_name': action_name,
            'peak_frame_idx': result['peak_frame_idx'],
            'peak_frame_path': str(peak_frame_path),
            'unit_length': result['unit_length'],
            'feature_dim': feature_vector.shape[0],
            'motion_dim': 12,
            'feature_vector': feature_vector,
            'motion_features': motion_features,
            'normalized_indicators': result['normalized_indicators'],
            'normalized_dynamic_features': result['normalized_dynamic_features']
        }

    def _compute_video_only(self, video_info, neutral_indicators=None):
        """
        工作线程中计算单个视频
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
            del landmarks_seq
            del frames_seq
            return {"ok": False, "error": f"未找到动作检测器: {action_name}"}

        neutral_raw = self._denormalize_indicators(neutral_indicators, video_info) if neutral_indicators else None

        h, w = frames_seq[0].shape[:2]
        fps = video_info.get('fps')

        result = detector.process(
            landmarks_seq=landmarks_seq,
            frames_seq=frames_seq,
            w=w,
            h=h,
            fps=fps,
            neutral_indicators=neutral_raw
        )

        # 计算运动特征 (复用landmarks_seq)
        motion_features = compute_motion_features(landmarks_seq, w, h, fps)

        # 释放序列内存
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
            "motion_features": motion_features,
            "elapsed_ms": (time.perf_counter() - t0) * 1000.0
        }

    def process_all_examinations(self, batch_size=10):
        """批量处理所有未处理的examinations"""
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

                if i % batch_size == 0:
                    gc.collect()
                    print(f"\n  [内存清理] 已处理 {i}/{len(examination_ids)} 个检查")

            except Exception as e:
                print(f"[ERROR] 处理检查 {exam_id} 时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                gc.collect()

        print(f"\n{'=' * 60}")
        print(f"批量处理完成!")
        print(f"成功: {len(results)}/{len(examination_ids)}")
        print(f"{'=' * 60}")

        return results

    def update_motion_features_only(self, batch_size=10):
        """
        仅更新运动特征 (用于已处理过几何特征但缺少运动特征的数据)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 查找已有几何特征但缺少运动特征的记录
        cursor.execute("""
            SELECT vf.video_id, vf.file_path, vf.start_frame, vf.end_frame, vf.fps
            FROM video_files vf
            INNER JOIN video_features vfeat ON vf.video_id = vfeat.video_id
            WHERE vf.file_exists = 1
              AND vfeat.static_features IS NOT NULL
              AND vfeat.motion_features IS NULL
        """)

        pending = cursor.fetchall()
        conn.close()

        if not pending:
            print("[Motion更新] 没有需要更新的记录")
            return

        print(f"[Motion更新] 找到 {len(pending)} 个需要补充运动特征的视频")

        success_count = 0
        fail_count = 0

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for i, (video_id, file_path, start_frame, end_frame, fps) in enumerate(pending, 1):
            try:
                if not file_path or not os.path.exists(file_path):
                    fail_count += 1
                    continue

                fps = fps

                # 提取landmarks
                landmarks_seq, frames_seq = self._extract_sequence(
                    file_path, start_frame, end_frame
                )

                if not landmarks_seq or not frames_seq:
                    fail_count += 1
                    continue

                h, w = frames_seq[0].shape[:2]

                # 计算运动特征
                motion_features = compute_motion_features(landmarks_seq, w, h, fps)

                del landmarks_seq
                del frames_seq

                # 更新数据库
                cursor.execute("""
                    UPDATE video_features
                    SET motion_features = ?,
                        motion_dim = 12,
                        motion_processed_at = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE video_id = ?
                """, (motion_features.tobytes(), video_id))

                success_count += 1

                if i % batch_size == 0:
                    conn.commit()
                    gc.collect()
                    print(f"  [{i}/{len(pending)}] 成功:{success_count} 失败:{fail_count}")

            except Exception as e:
                fail_count += 1
                print(f"  [ERROR] video_id={video_id}: {e}")

        conn.commit()
        conn.close()

        print(f"\n[Motion更新完成] 成功:{success_count} 失败:{fail_count}")

    # =========================================================================
    # 辅助方法
    # =========================================================================

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
        """提取视频序列"""
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

            landmarks = extractor.extract_from_frame(frame)

            landmarks_seq.append(landmarks)
            frames_seq.append(frame.copy())
            del frame

            frame_idx += 1

        cap.release()

        return landmarks_seq, frames_seq

    def _denormalize_indicators(self, normalized_indicators, video_info):
        """将归一化指标转换回原始像素值"""
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
                          normalized_dynamic_features, motion_features=None):
        """
        保存到数据库 (包含motion_features)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        video_info = self._get_video_info(video_id)
        action_name = video_info['action_name_en']

        key_indicators = self.feature_integrator.action_key_indicators.get(action_name, None)
        if key_indicators is None:
            print(f"  [WARN] 未在 action_key_indicators 中找到 {action_name}，跳过入库")
            conn.close()
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

        # 运动特征
        motion_blob = None
        motion_dim = 0
        if motion_features is not None:
            motion_blob = motion_features.astype(np.float32).tobytes()
            motion_dim = 12

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
                    dynamic_dim,
                    motion_features,
                    motion_dim,
                    geometry_processed_at,
                    motion_processed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """, (
                video_id,
                peak_frame_idx,
                peak_frame_path,
                unit_length,
                static_blob,
                dynamic_blob,
                static_dim,
                dynamic_dim,
                motion_blob,
                motion_dim
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
                    motion_features = ?,
                    motion_dim = ?,
                    geometry_processed_at = CURRENT_TIMESTAMP,
                    motion_processed_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE video_id = ?
            """, (
                peak_frame_idx,
                peak_frame_path,
                unit_length,
                static_blob,
                dynamic_blob,
                static_dim,
                dynamic_dim,
                motion_blob,
                motion_dim,
                video_id
            ))

            conn.commit()

        finally:
            conn.close()


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""

    # 基本路径配置
    db_path = 'facialPalsy.db'
    model_path = '/Users/cuijinglei/PycharmProjects/medicalProject/models/face_landmarker.task'
    keyframe_dir = '/Users/cuijinglei/Documents/facialPalsy/HGFA/keyframes'
    os.makedirs(keyframe_dir, exist_ok=True)

    # 运行模式
    examination_id = None       # 指定单个检查ID
    video_id = None             # 指定单个视频ID
    run_batch = True            # 批量处理所有
    update_motion_only = False  # 仅更新运动特征

    # 初始化 Pipeline
    pipeline = VideoPipeline(db_path, model_path, keyframe_dir)

    if update_motion_only:
        # 仅补充运动特征 (针对已处理几何特征的数据)
        pipeline.update_motion_features_only(batch_size=10)
    elif examination_id is not None:
        pipeline.process_examination(examination_id)
    elif video_id is not None:
        pipeline.process_video(video_id)
    elif run_batch:
        pipeline.process_all_examinations(batch_size=10)
    else:
        print("当前没有配置任何处理任务")


if __name__ == '__main__':
    main()