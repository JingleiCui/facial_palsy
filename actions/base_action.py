"""
动作基类 (Base Action)
======================

所有11个动作的基类，定义统一接口和通用功能

核心设计:
1. 每个动作都有 process() 方法处理视频序列
2. 每个动作都有 find_peak_frame() 方法找关键帧
3. 每个动作返回统一格式的 ActionResult
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import cv2


@dataclass
class NeutralBaseline:
    """NeutralFace基准数据"""
    icd: float                     # 单位长度
    left_eye_area: float           # 左眼裂面积
    right_eye_area: float          # 右眼裂面积
    left_palpebral_length: float   # 左眼睑裂长度
    right_palpebral_length: float  # 右眼睑裂长度
    left_brow_height: float        # 左眉眼距
    right_brow_height: float       # 右眉眼距
    mouth_width: float             # 嘴宽
    mouth_height: float            # 嘴高
    left_nlf_length: float         # 左鼻唇沟长度
    right_nlf_length: float        # 右鼻唇沟长度

    @classmethod
    def from_dict(cls, d: Dict) -> 'NeutralBaseline':
        """从字典创建"""
        return cls(
            icd=d.get('icd', 0),
            left_eye_area=d.get('left_eye_area', 0),
            right_eye_area=d.get('right_eye_area', 0),
            left_palpebral_length=d.get('left_palpebral_length', 0),
            right_palpebral_length=d.get('right_palpebral_length', 0),
            left_brow_height=d.get('left_brow_height', 0),
            right_brow_height=d.get('right_brow_height', 0),
            mouth_width=d.get('mouth_width', 0),
            mouth_height=d.get('mouth_height', 0),
            left_nlf_length=d.get('left_nlf_length', 0),
            right_nlf_length=d.get('right_nlf_length', 0),
        )

    def to_dict(self) -> Dict:
        """转为字典"""
        return {
            'icd': self.icd,
            'left_eye_area': self.left_eye_area,
            'right_eye_area': self.right_eye_area,
            'left_palpebral_length': self.left_palpebral_length,
            'right_palpebral_length': self.right_palpebral_length,
            'left_brow_height': self.left_brow_height,
            'right_brow_height': self.right_brow_height,
            'mouth_width': self.mouth_width,
            'mouth_height': self.mouth_height,
            'left_nlf_length': self.left_nlf_length,
            'right_nlf_length': self.right_nlf_length,
        }


@dataclass
class ActionResult:
    """动作处理结果"""
    action_name: str                        # 动作名称
    peak_frame_idx: int                     # 峰值帧索引
    peak_frame: np.ndarray                  # 峰值帧图像
    unit_length: float                      # 单位长度 (ICD)

    # 核心指标
    indicators: Dict[str, float] = field(default_factory=dict)

    # 归一化指标 (用于模型训练)
    normalized_indicators: Dict[str, float] = field(default_factory=dict)

    # 动态特征 (时序信息)
    dynamic_features: Dict[str, float] = field(default_factory=dict)
    normalized_dynamic_features: Dict[str, float] = field(default_factory=dict)

    # 可解释性数据 (用于可视化)
    interpretability: Dict[str, Any] = field(default_factory=dict)

    # 睁眼度曲线 (用于眼部相关动作)
    left_openness_curve: Optional[np.ndarray] = None
    right_openness_curve: Optional[np.ndarray] = None


class BaseAction(ABC):
    """动作基类"""

    # 子类必须定义
    ACTION_NAME: str = "BaseAction"
    ACTION_NAME_CN: str = "基础动作"

    def __init__(self):
        pass

    @abstractmethod
    def find_peak_frame(
        self,
        landmarks_seq: List,
        w: int, h: int,
        **kwargs
    ) -> int:
        """
        找到峰值帧索引

        不同动作有不同的峰值定义:
        - NeutralFace: 最大EAR帧
        - CloseEye: 最小EAR帧
        - Smile: 最大嘴宽帧
        - RaiseEyebrow: 最大眉眼距帧
        """
        pass

    @abstractmethod
    def extract_indicators(
        self,
        landmarks,
        w: int, h: int,
        neutral_baseline: Optional[NeutralBaseline] = None
    ) -> Dict[str, float]:
        """
        从单帧提取指标

        每个动作定义自己关注的指标
        """
        pass

    def extract_dynamic_features(
        self,
        landmarks_seq: List,
        w: int, h: int,
        fps: float = 30.0,
        neutral_baseline: Optional[NeutralBaseline] = None
    ) -> Dict[str, float]:
        """
        提取动态特征 (默认实现，子类可覆盖)

        返回:
            motion_range_left/right: 运动范围
            velocity_left/right: 平均速度
            smoothness_left/right: 平滑度
        """
        # 默认返回空
        return {}

    def process(
        self,
        landmarks_seq: List,
        frames_seq: List[np.ndarray],
        w: int, h: int,
        fps: float = 30.0,
        neutral_indicators: Optional[Dict] = None
    ) -> Optional[ActionResult]:
        """
        处理视频序列

        Args:
            landmarks_seq: 关键点序列
            frames_seq: 帧序列
            w, h: 图像尺寸
            fps: 帧率
            neutral_indicators: 静息帧指标 (用于其他动作对比)

        Returns:
            ActionResult 包含所有结果
        """
        if not landmarks_seq or not frames_seq:
            return None

        # 1. 转换neutral_indicators为NeutralBaseline
        neutral_baseline = None
        if neutral_indicators is not None:
            neutral_baseline = NeutralBaseline.from_dict(neutral_indicators)

        # 2. 找峰值帧
        peak_idx = self.find_peak_frame(
            landmarks_seq, w, h,
            neutral_baseline=neutral_baseline
        )
        peak_idx = max(0, min(peak_idx, len(landmarks_seq) - 1))

        peak_landmarks = landmarks_seq[peak_idx]
        peak_frame = frames_seq[peak_idx].copy()

        if peak_landmarks is None:
            return None

        # 3. 提取峰值帧指标
        indicators = self.extract_indicators(
            peak_landmarks, w, h, neutral_baseline
        )

        # 4. 提取动态特征
        dynamic_features = self.extract_dynamic_features(
            landmarks_seq, w, h, fps, neutral_baseline
        )

        # 5. 获取单位长度
        from .geometry import compute_icd
        icd = compute_icd(peak_landmarks, w, h)

        # 6. 归一化指标 (已经是归一化的，直接复制)
        normalized_indicators = indicators.copy()
        normalized_dynamic_features = dynamic_features.copy()

        # 7. 构建可解释性数据
        interpretability = self._build_interpretability(
            landmarks_seq, w, h,
            peak_idx, indicators,
            neutral_baseline
        )

        return ActionResult(
            action_name=self.ACTION_NAME,
            peak_frame_idx=peak_idx,
            peak_frame=peak_frame,
            unit_length=icd,
            indicators=indicators,
            normalized_indicators=normalized_indicators,
            dynamic_features=dynamic_features,
            normalized_dynamic_features=normalized_dynamic_features,
            interpretability=interpretability,
            left_openness_curve=interpretability.get('left_openness_curve'),
            right_openness_curve=interpretability.get('right_openness_curve'),
        )

    def _build_interpretability(
        self,
        landmarks_seq: List,
        w: int, h: int,
        peak_idx: int,
        indicators: Dict,
        neutral_baseline: Optional[NeutralBaseline]
    ) -> Dict[str, Any]:
        """
        构建可解释性数据 (子类可覆盖扩展)
        """
        return {
            'peak_frame_idx': peak_idx,
            'total_frames': len(landmarks_seq),
        }

    def visualize_peak_frame(
        self,
        frame: np.ndarray,
        landmarks,
        indicators: Dict,
        w: int, h: int
    ) -> np.ndarray:
        """
        在峰值帧上绘制可视化标注

        子类应该覆盖此方法添加动作特定的可视化
        """
        img = frame.copy()

        # 绘制动作名称
        cv2.putText(img, f"{self.ACTION_NAME}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return img