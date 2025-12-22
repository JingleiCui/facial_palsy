"""
视频处理Pipeline - 适配V2 Actions架构
========================================

本模块已更新以适配最新的 actions/ 模块结构:
1. 使用 BaseAction 基类的统一 process() 接口
2. 支持 NeutralBaseline dataclass 和 ActionResult dataclass
3. 动作类实例化和注册机制
4. 与 ActionFeatureIntegrator 的无缝集成

处理流程:
1. 读取视频 → 提取landmarks序列和frames序列
2. 检测峰值帧 → 保存峰值帧图像
3. 提取几何特征 → static_features + dynamic_features
4. 计算运动特征 → motion_features (12维)
5. 存储到数据库

输出特征:
- static_features:  维度因动作而异
- dynamic_features: 维度因动作而异
- motion_features:  12维 (统一维度)

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
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import sqlite3
import gc
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import dataclasses
from typing import Dict, List, Optional, Any, Tuple

# =============================================================================
# 导入核心模块
# =============================================================================

# 根据实际项目结构调整导入路径

from facial_palsy.core.landmark_extractor import LandmarkExtractor
from facial_palsy.core.motion_utils import compute_motion_features
from facial_palsy.core.constants import ActionNames
from facial_palsy.action_feature_integrator import ActionFeatureIntegrator

# 导入所有动作类
from facial_palsy.actions.base_action import BaseAction, ActionResult, NeutralBaseline
from facial_palsy.actions.neutral_face import NeutralFaceAction
from facial_palsy.actions.raise_eyebrow import RaiseEyebrowAction
from facial_palsy.actions.close_eye_softly import CloseEyeSoftlyAction
from facial_palsy.actions.close_eye_hardly import CloseEyeHardlyAction
from facial_palsy.actions.smile import SmileAction
from facial_palsy.actions.shrug_nose import ShrugNoseAction
from facial_palsy.actions.lip_pucker import LipPuckerAction
from facial_palsy.actions.show_teeth import ShowTeethAction
from facial_palsy.actions.blow_cheek import BlowCheekAction
from facial_palsy.actions.voluntary_eye_blink import VoluntaryEyeBlinkAction
from facial_palsy.actions.spontaneous_eye_blink import SpontaneousEyeBlinkAction



# =============================================================================
# 动作类注册表
# =============================================================================

# 动作名称到动作类的映射
ACTION_CLASSES: Dict[str, type] = {
    'NeutralFace': NeutralFaceAction,
    'RaiseEyebrow': RaiseEyebrowAction,
    'CloseEyeSoftly': CloseEyeSoftlyAction,
    'CloseEyeHardly': CloseEyeHardlyAction,
    'Smile': SmileAction,
    'ShrugNose': ShrugNoseAction,
    'LipPucker': LipPuckerAction,
    'ShowTeeth': ShowTeethAction,
    'BlowCheek': BlowCheekAction,
    'VoluntaryEyeBlink': VoluntaryEyeBlinkAction,
    'SpontaneousEyeBlink': SpontaneousEyeBlinkAction,
}


def create_action_detectors() -> Dict[str, BaseAction]:
    """
    创建所有动作检测器实例

    Returns:
        字典，键为动作名称，值为动作类实例
    """
    detectors = {}
    for name, cls in ACTION_CLASSES.items():
        try:
            detectors[name] = cls()
        except Exception as e:
            print(f"[WARN] 创建动作检测器 {name} 失败: {e}")
    return detectors


# =============================================================================
# VideoPipeline 类
# =============================================================================

class VideoPipeline:
    """
    视频处理Pipeline

    适配最新的 actions 架构:
    1. 使用 BaseAction.process() 统一接口
    2. 支持 NeutralBaseline 和 ActionResult dataclass
    3. 集成 ActionFeatureIntegrator
    """

    def __init__(self, db_path: str, model_path: str, keyframe_root_dir: str):
        """
        初始化 Pipeline

        Args:
            db_path: 数据库路径
            model_path: MediaPipe FaceLandmarker模型路径
            keyframe_root_dir: 关键帧保存根目录
        """
        self.db_path = db_path
        self.model_path = model_path
        self.keyframe_root_dir = Path(keyframe_root_dir)
        self.keyframe_root_dir.mkdir(parents=True, exist_ok=True)

        # 初始化 landmark 提取器
        self.landmark_extractor = LandmarkExtractor(model_path)

        # 初始化特征整合器
        self.feature_integrator = ActionFeatureIntegrator()

        # 创建动作检测器实例
        self.action_detectors = create_action_detectors()

        # 静息帧缓存 (examination_id -> NeutralBaseline dict)
        self.neutral_cache: Dict[int, Dict] = {}

        # 并行处理配置
        self.num_workers = 8

        # 线程本地存储
        self._tls = threading.local()

    def _get_worker(self):
        """
        获取线程本地的 worker 实例

        为每个工作线程创建独立的:
        - LandmarkExtractor
        - ActionFeatureIntegrator
        - action_detectors
        """
        w = getattr(self._tls, "worker", None)
        if w is None:
            w = type("Worker", (), {})()
            w.landmark_extractor = LandmarkExtractor(self.model_path)
            w.feature_integrator = ActionFeatureIntegrator()
            w.action_detectors = create_action_detectors()
            self._tls.worker = w
        return w

    # =========================================================================
    # NeutralBaseline 处理
    # =========================================================================

    def _build_neutral_baseline(self, neutral_indicators: Optional[Dict]) -> Optional[NeutralBaseline]:
        """
        从指标字典构建 NeutralBaseline 对象

        Args:
            neutral_indicators: 静息帧指标字典

        Returns:
            NeutralBaseline 对象，或 None
        """
        if neutral_indicators is None:
            return None

        try:
            return NeutralBaseline.from_dict(neutral_indicators)
        except Exception:
            # 如果 from_dict 不存在，尝试直接构造
            try:
                # 获取 dataclass 字段
                if dataclasses.is_dataclass(NeutralBaseline):
                    fields = [f.name for f in dataclasses.fields(NeutralBaseline)]
                    kwargs = {k: neutral_indicators.get(k, 0) for k in fields}
                    return NeutralBaseline(**kwargs)
            except Exception:
                pass

        return None

    def _neutral_baseline_to_dict(self, baseline: Optional[NeutralBaseline]) -> Optional[Dict]:
        """
        将 NeutralBaseline 转换为字典
        """
        if baseline is None:
            return None

        if dataclasses.is_dataclass(baseline):
            return dataclasses.asdict(baseline)

        if hasattr(baseline, 'to_dict'):
            return baseline.to_dict()

        if hasattr(baseline, '__dict__'):
            return dict(baseline.__dict__)

        return None

    # =========================================================================
    # ActionResult 处理
    # =========================================================================

    def _action_result_to_dict(self, result: Any) -> Optional[Dict]:
        """
        将 ActionResult 或其他结果类型转换为字典

        支持:
        - dict: 直接返回
        - dataclass: 使用 asdict
        - 普通对象: 取 __dict__
        """
        if result is None:
            return None

        if isinstance(result, dict):
            return result

        if dataclasses.is_dataclass(result):
            d = dataclasses.asdict(result)
            # 特殊处理 numpy 数组
            for k, v in d.items():
                if isinstance(v, np.ndarray):
                    d[k] = v  # 保持 numpy 数组
            return d

        if hasattr(result, '__dict__'):
            return dict(result.__dict__)

        return None

    # =========================================================================
    # 动作处理调用
    # =========================================================================

    def _call_action_process(
        self,
        detector: BaseAction,
        landmarks_seq: List,
        frames_seq: List[np.ndarray],
        w: int,
        h: int,
        fps: float,
        neutral_indicators: Optional[Dict]
    ) -> Optional[Dict]:
        """
        调用动作检测器的 process 方法

        统一处理新旧两种接口:
        - 新接口: neutral_indicators 作为字典传入，内部转换为 NeutralBaseline
        - 旧接口: 直接传递 neutral_indicators 字典

        Args:
            detector: 动作检测器实例
            landmarks_seq: 关键点序列
            frames_seq: 帧序列
            w, h: 图像尺寸
            fps: 帧率
            neutral_indicators: 静息帧指标字典

        Returns:
            处理结果字典，或 None
        """
        try:
            # BaseAction.process() 接口
            result = detector.process(
                landmarks_seq=landmarks_seq,
                frames_seq=frames_seq,
                w=w,
                h=h,
                fps=fps,
                neutral_indicators=neutral_indicators
            )

            return self._action_result_to_dict(result)

        except TypeError as e:
            # 可能是参数名不匹配，尝试其他方式
            print(f"[WARN] 动作处理调用失败: {e}")
            return None
        except Exception as e:
            print(f"[ERROR] 动作处理异常: {e}")
            return None

    # =========================================================================
    # 主处理流程
    # =========================================================================

    def process_examination(self, examination_id: int) -> Optional[Dict]:
        """
        处理一个完整的 examination (11个动作)

        流程:
        1. 获取该 examination 的所有视频
        2. 首先处理 NeutralFace 获取基准
        3. 并行处理其余 10 个动作
        4. 保存结果到数据库

        Args:
            examination_id: 检查 ID

        Returns:
            处理结果字典
        """
        print(f"\n{'=' * 60}")
        print(f"处理检查 ID: {examination_id}")
        print(f"{'=' * 60}")

        start_time = datetime.now()

        # 1. 获取该 examination 的所有视频
        videos = self._get_examination_videos(examination_id)

        if not videos:
            print(f"[ERROR] 检查 {examination_id} 没有视频")
            return None

        print(f"找到 {len(videos)} 个视频")

        # 2. 首先处理 NeutralFace (静息帧)
        neutral_result = None
        neutral_video = next(
            (v for v in videos if v['action_name_en'] == 'NeutralFace'),
            None
        )

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

        # 3. 获取静息帧指标用于其他动作
        neutral_indicators = None
        if examination_id in self.neutral_cache:
            neutral_indicators = self.neutral_cache[examination_id]['normalized_indicators']

        # 4. 并行处理其他动作
        results = {}
        other_videos = [v for v in videos if v['action_name_en'] != 'NeutralFace']

        print(f"\n[步骤2] 并行处理其余 {len(other_videos)} 个动作...")

        failures = []
        computed = []

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_map = {
                executor.submit(
                    self._compute_video_only,
                    v,
                    neutral_indicators
                ): v
                for v in other_videos
            }

            for future in as_completed(future_map):
                v = future_map[future]
                try:
                    out = future.result()
                except Exception as e:
                    failures.append((v['video_id'], v['action_name_en'], str(e)))
                    continue

                if not out.get("ok"):
                    failures.append((
                        v['video_id'],
                        v['action_name_en'],
                        out.get("error", "unknown")
                    ))
                    continue

                computed.append(out)

        # 5. 串行保存到数据库 (避免 SQLite 写锁冲突)
        for out in computed:
            vinfo = next(v for v in other_videos if v["video_id"] == out["video_id"])
            action_name = out["action_name"]
            r = out["result"]

            # 保存峰值帧
            peak_frame_path = self._save_peak_frame(
                r['peak_frame'],
                vinfo['examination_id'],
                action_name
            )

            # 删除帧数据避免序列化
            del r['peak_frame']

            # 保存到数据库
            self._save_to_database(
                video_id=vinfo['video_id'],
                peak_frame_idx=r['peak_frame_idx'],
                peak_frame_path=str(peak_frame_path),
                unit_length=r['unit_length'],
                feature_vector=out["feature_vector"],
                normalized_indicators=r['normalized_indicators'],
                normalized_dynamic_features=r['normalized_dynamic_features'],
                motion_features=out.get("motion_features"),
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

        # 6. 报告失败
        if failures:
            print(f"  [WARN] 本次 examination 有 {len(failures)} 个动作失败：")
            for vid, act, err in failures[:10]:
                print(f"    - video_id={vid} act={act} err={err}")

        # 7. 添加静息帧结果
        if neutral_result:
            results['NeutralFace'] = neutral_result

        # 8. 计算处理时间
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        print(f"\n{'=' * 60}")
        print(f"✓ 检查处理完成! 耗时: {processing_time:.2f}ms")
        print(f"成功处理: {len(results)}/11 个动作")
        print(f"{'=' * 60}")

        # 清理内存
        del computed
        gc.collect()

        return {
            'examination_id': examination_id,
            'results': results,
            'processing_time_ms': processing_time
        }

    def process_video(
        self,
        video_id: int,
        neutral_indicators: Optional[Dict] = None
    ) -> Optional[Dict]:
        """
        处理单个视频 (主线程版本)

        Args:
            video_id: 视频 ID
            neutral_indicators: 静息帧指标字典

        Returns:
            处理结果字典
        """
        # 1. 获取视频信息
        video_info = self._get_video_info(video_id)
        if not video_info:
            print(f"[ERROR] 视频ID {video_id} 不存在")
            return None

        action_name = video_info['action_name_en']
        print(f"  动作: {action_name} ({video_info['action_name_cn']})")

        # 2. 检查文件存在
        if not os.path.exists(video_info['file_path']):
            print(f"  [ERROR] 文件不存在: {video_info['file_path']}")
            return None

        # 3. 提取 landmarks 序列和 frames
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

        # 5. 获取视频尺寸和帧率
        h, w = frames_seq[0].shape[:2]
        fps = video_info.get('fps', 30.0)
        if fps is None:
            fps = 30.0
        fps = float(fps)

        # 6. 调用动作处理
        result = self._call_action_process(
            detector,
            landmarks_seq=landmarks_seq,
            frames_seq=frames_seq,
            w=w,
            h=h,
            fps=fps,
            neutral_indicators=neutral_indicators
        )

        # 7. 计算运动特征 (复用 landmarks_seq)
        motion_features = None
        try:
            motion_features = compute_motion_features(landmarks_seq, w, h, fps)
        except Exception as e:
            print(f"  [WARN] 运动特征计算异常: {e}")

        # 释放序列内存
        del landmarks_seq
        del frames_seq

        if not result:
            print(f"  [ERROR] 处理失败")
            return None

        # 8. 提取特征向量
        feature_vector = self.feature_integrator.extract_action_features(
            action_name,
            result.get('normalized_indicators', {}),
            result.get('normalized_dynamic_features', {})
        )

        print(f"  ✓ 几何特征: {feature_vector.shape[0]}维, 运动特征: 12维")

        # 9. 保存峰值帧
        peak_frame_path = self._save_peak_frame(
            result['peak_frame'],
            video_info['examination_id'],
            action_name
        )

        # 10. 保存到数据库
        self._save_to_database(
            video_id=video_id,
            peak_frame_idx=result['peak_frame_idx'],
            peak_frame_path=str(peak_frame_path),
            unit_length=result['unit_length'],
            feature_vector=feature_vector,
            normalized_indicators=result.get('normalized_indicators', {}),
            normalized_dynamic_features=result.get('normalized_dynamic_features', {}),
            motion_features=motion_features,
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
            'normalized_indicators': result.get('normalized_indicators', {}),
            'normalized_dynamic_features': result.get('normalized_dynamic_features', {})
        }

    def _compute_video_only(
        self,
        video_info: Dict,
        neutral_indicators: Optional[Dict] = None
    ) -> Dict:
        """
        工作线程中计算单个视频

        Args:
            video_info: 视频信息字典
            neutral_indicators: 静息帧指标字典

        Returns:
            处理结果字典
        """
        t0 = time.perf_counter()
        action_name = video_info['action_name_en']

        # 检查文件
        if not os.path.exists(video_info['file_path']):
            return {"ok": False, "error": f"文件不存在: {video_info['file_path']}"}

        # 获取线程本地 worker
        worker = self._get_worker()

        # 提取序列
        landmarks_seq, frames_seq = self._extract_sequence(
            video_info['file_path'],
            video_info['start_frame'],
            video_info['end_frame'],
            extractor=worker.landmark_extractor
        )

        if not landmarks_seq:
            return {"ok": False, "error": "关键点提取失败"}

        # 获取动作检测器
        detector = worker.action_detectors.get(action_name)
        if not detector:
            del landmarks_seq
            del frames_seq
            return {"ok": False, "error": f"未找到动作检测器: {action_name}"}

        # 获取视频尺寸和帧率
        h, w = frames_seq[0].shape[:2]
        fps = video_info.get('fps', 30.0)
        if fps is None:
            fps = 30.0
        elif isinstance(fps, (list, tuple)):
            fps = float(fps[0]) if fps else 30.0
        else:
            fps = float(fps)

        # 调用动作处理
        result = self._call_action_process(
            detector,
            landmarks_seq=landmarks_seq,
            frames_seq=frames_seq,
            w=w,
            h=h,
            fps=fps,
            neutral_indicators=neutral_indicators
        )

        # 计算运动特征
        motion_features = None
        try:
            motion_features = compute_motion_features(landmarks_seq, w, h, fps)
        except Exception as e:
            print(f"  [WARN] 运动特征计算异常: {e}")

        # 释放内存
        del landmarks_seq
        del frames_seq

        if not result:
            return {"ok": False, "error": "动作处理失败(detector.process 返回空)"}

        # 提取特征向量
        feature_vector = worker.feature_integrator.extract_action_features(
            action_name,
            result.get('normalized_indicators', {}),
            result.get('normalized_dynamic_features', {})
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

    def process_all_examinations(self, batch_size: int = 10, force_recompute: bool = False) -> List[Dict]:
        """
        批量处理 examinations

        模式说明:
        - force_recompute=False: 仅处理“未处理”或 motion_features 为空的检查（默认）
        - force_recompute=True : 强制重算所有检查（忽略 video_features 是否已有数据）

        Args:
            batch_size: 每批处理数量
            force_recompute: 是否强制重算全部

        Returns:
            处理结果列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if force_recompute:
            # 强制重算：只要 video_files 中标记 file_exists=1，就纳入处理
            cursor.execute("""
                SELECT DISTINCT e.examination_id
                FROM examinations e
                INNER JOIN video_files vf ON e.examination_id = vf.examination_id
                WHERE vf.file_exists = 1
                ORDER BY e.examination_id
            """)
        else:
            # 默认：仅处理“未处理”或 motion_features 为空的检查
            cursor.execute("""
                SELECT DISTINCT e.examination_id
                FROM examinations e
                INNER JOIN video_files vf ON e.examination_id = vf.examination_id
                LEFT JOIN video_features feat ON vf.video_id = feat.video_id
                WHERE vf.file_exists = 1 
                  AND (feat.feature_id IS NULL OR feat.motion_features IS NULL)
                ORDER BY e.examination_id
            """)

        examination_ids = [row[0] for row in cursor.fetchall()]
        conn.close()

        mode = "强制重算全部" if force_recompute else "仅处理缺失/未处理"
        print(f"\n[{mode}] 找到 {len(examination_ids)} 个需要处理的检查")
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

    def update_motion_features_only(self, batch_size: int = 10):
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

                # 提取 landmarks
                landmarks_seq, frames_seq = self._extract_sequence(
                    file_path, start_frame, end_frame
                )

                if not landmarks_seq or not frames_seq:
                    fail_count += 1
                    continue

                h, w = frames_seq[0].shape[:2]

                # 计算运动特征
                motion_features = compute_motion_features(landmarks_seq, w, h, fps or 30.0)

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
    # 数据库操作
    # =========================================================================

    def _get_video_info(self, video_id: int) -> Optional[Dict]:
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

    def _get_examination_videos(self, examination_id: int) -> List[Dict]:
        """获取某个 examination 的所有视频"""
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

    def _extract_sequence(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
        extractor: Optional[LandmarkExtractor] = None
    ) -> Tuple[Optional[List], Optional[List[np.ndarray]]]:
        """
        提取视频序列

        Args:
            video_path: 视频文件路径
            start_frame: 起始帧
            end_frame: 结束帧
            extractor: landmark 提取器

        Returns:
            (landmarks_seq, frames_seq)
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

            landmarks = extractor.extract_from_frame(frame)

            landmarks_seq.append(landmarks)
            frames_seq.append(frame.copy())
            del frame

            frame_idx += 1

        cap.release()

        return landmarks_seq, frames_seq

    def _save_peak_frame(
        self,
        frame: np.ndarray,
        examination_id: int,
        action_name: str,
        augmentation_type: str = 'none'
    ) -> Path:
        """保存峰值帧"""
        action_dir = self.keyframe_root_dir / action_name
        action_dir.mkdir(parents=True, exist_ok=True)

        aug = '' if augmentation_type in (None, '', 'none') else f"_{augmentation_type}"
        filename = f"{examination_id}_{action_name}{aug}.jpg"
        filepath = action_dir / filename

        cv2.imwrite(str(filepath), frame)

        return filepath

    def _save_to_database(
        self,
        video_id: int,
        peak_frame_idx: int,
        peak_frame_path: str,
        unit_length: float,
        feature_vector: np.ndarray,
        normalized_indicators: Dict,
        normalized_dynamic_features: Dict,
        motion_features: Optional[np.ndarray] = None,
        augmentation_type: str = 'none',
        aug_palsy_side: Optional[str] = None
    ):
        """
        保存到数据库 (包含 motion_features)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        video_info = self._get_video_info(video_id)
        action_name = video_info['action_name_en']

        # 获取该动作的关键指标定义
        key_indicators = self.feature_integrator.action_key_indicators.get(action_name, None)
        if key_indicators is None:
            print(f"  [WARN] 未在 action_key_indicators 中找到 {action_name}，跳过入库")
            conn.close()
            return

        static_names = key_indicators['static']
        dynamic_names = key_indicators['dynamic']

        # 提取静态特征
        static_vals = [
            float(normalized_indicators.get(name, 0.0) or 0.0)
            for name in static_names
        ]

        # 提取动态特征
        dynamic_vals = [
            float(normalized_dynamic_features.get(name, 0.0) or 0.0)
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

        # 可解释性数据 (JSON)
        interpretability_json = json.dumps({
            'normalized_indicators': normalized_indicators,
            'normalized_dynamic_features': normalized_dynamic_features
        }, ensure_ascii=False, default=str)

        try:
            cursor.execute("""
                INSERT INTO video_features (
                    video_id,
                    augmentation_type,
                    aug_palsy_side,
                    peak_frame_idx,
                    peak_frame_path,
                    unit_length,
                    static_features,
                    dynamic_features,
                    static_dim,
                    dynamic_dim,
                    motion_features,
                    motion_dim,
                    interpretability,
                    geometry_processed_at,
                    motion_processed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """, (
                video_id,
                augmentation_type,
                aug_palsy_side,
                peak_frame_idx,
                peak_frame_path,
                unit_length,
                static_blob,
                dynamic_blob,
                static_dim,
                dynamic_dim,
                motion_blob,
                motion_dim,
                interpretability_json
            ))

            conn.commit()

        except sqlite3.IntegrityError:
            # 已存在则更新
            cursor.execute("""
                UPDATE video_features
                SET augmentation_type = ?,
                    aug_palsy_side = ?,
                    peak_frame_idx = ?,
                    peak_frame_path = ?,
                    unit_length = ?,
                    static_features = ?,
                    dynamic_features = ?,
                    static_dim = ?,
                    dynamic_dim = ?,
                    motion_features = ?,
                    motion_dim = ?,
                    interpretability = ?,
                    geometry_processed_at = CURRENT_TIMESTAMP,
                    motion_processed_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE video_id = ?
            """, (
                augmentation_type,
                aug_palsy_side,
                peak_frame_idx,
                peak_frame_path,
                unit_length,
                static_blob,
                dynamic_blob,
                static_dim,
                dynamic_dim,
                motion_blob,
                motion_dim,
                interpretability_json,
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

    # 基本路径配置 - 根据实际环境修改
    db_path = 'facialPalsy.db'
    model_path = '/Users/cuijinglei/PycharmProjects/medicalProject/models/face_landmarker.task'
    keyframe_dir = '/Users/cuijinglei/Documents/facialPalsy/HGFA/keyframes_new'
    os.makedirs(keyframe_dir, exist_ok=True)

    # 运行模式
    examination_id = None       # 指定单个检查ID
    video_id = None             # 指定单个视频ID
    run_batch = True            # 批量处理
    force_recompute = True     # True=重算全部(算法更新后用)，False=只补缺失/续跑
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
        pipeline.process_all_examinations(batch_size=10, force_recompute=force_recompute)
    else:
        print("当前没有配置任何处理任务")


if __name__ == '__main__':
    main()