"""
SpontaneousEyeBlink (自发眨眼): 自然眨眼


这两个动作与CloseEye的区别:
- 眨眼是快速的闭-开循环
- 关注眨眼速度、频率、对称性

关键帧选择: 最小EAR帧 (眼睛闭得最紧的瞬间)

核心指标:
- 基础闭眼指标 (继承CloseEyeBase)
- 眨眼速度: 从睁到闭的时间
- 眨眼周期: 完整眨眼所需帧数
- 眨眼对称性: 左右眼到达最小EAR的时间差
"""

import numpy as np
from typing import Dict, List, Optional, Any
import cv2

# ========== 第1步：导入常量 ==========
# 从constants模块导入所有需要的常量
from ..core.constants import LM, Colors, Thresholds

# ========== 第2步：导入几何工具 ==========
# 根据动作需要选择性导入
from ..core.geometry_utils import (
    compute_icd,           # ICD计算
    compute_ear,           # EAR计算
    measure_eyes,          # 眼部测量
    measure_oral,          # 口角测量
    measure_nlf,           # 鼻唇沟测量
    measure_brow,          # 眉毛测量
    find_max_ear_frame,    # 找最大EAR帧
    find_min_ear_frame,    # 找最小EAR帧
    compute_openness_curve,  # 计算睁眼度曲线
    compute_ear_curve,      # 计算EAR曲线
    pts2d,                 # 批量获取2D坐标
    pt2d,                  # 单点2D坐标
    dist,                  # 距离计算
)

# ========== 第3步：导入基类 ==========
from .base_action import BaseAction, ActionResult, NeutralBaseline

def detect_blink_events(ear_curve: np.ndarray, threshold_ratio: float = 0.5) -> List[Dict]:
    """
    检测眨眼事件

    Args:
        ear_curve: EAR曲线
        threshold_ratio: 阈值比例 (相对于最大EAR)

    Returns:
        眨眼事件列表, 每个事件包含:
        - start_idx: 开始帧
        - min_idx: 最小EAR帧
        - end_idx: 结束帧
        - duration: 持续帧数
        - min_ear: 最小EAR值
    """
    if len(ear_curve) < 3:
        return []

    max_ear = ear_curve.max()
    threshold = max_ear * threshold_ratio

    events = []
    in_blink = False
    start_idx = 0
    min_idx = 0
    min_ear = float('inf')

    for i, ear in enumerate(ear_curve):
        if not in_blink and ear < threshold:
            # 眨眼开始
            in_blink = True
            start_idx = i
            min_ear = ear
            min_idx = i
        elif in_blink:
            if ear < min_ear:
                min_ear = ear
                min_idx = i
            if ear >= threshold:
                # 眨眼结束
                in_blink = False
                events.append({
                    'start_idx': start_idx,
                    'min_idx': min_idx,
                    'end_idx': i,
                    'duration': i - start_idx,
                    'min_ear': min_ear,
                })

    # 处理末尾未结束的眨眼
    if in_blink:
        events.append({
            'start_idx': start_idx,
            'min_idx': min_idx,
            'end_idx': len(ear_curve) - 1,
            'duration': len(ear_curve) - 1 - start_idx,
            'min_ear': min_ear,
        })

    return events


def compute_blink_speed(ear_curve: np.ndarray, fps: float, blink_event: Dict) -> Dict[str, float]:
    """
    计算眨眼速度指标

    Returns:
        closing_speed: 闭眼速度 (EAR变化/秒)
        opening_speed: 睁眼速度
        closing_frames: 闭眼帧数
        opening_frames: 睁眼帧数
    """
    start_idx = blink_event['start_idx']
    min_idx = blink_event['min_idx']
    end_idx = blink_event['end_idx']

    dt = 1.0 / fps if fps > 0 else 1.0 / 30.0

    # 闭眼阶段
    closing_frames = min_idx - start_idx
    if closing_frames > 0:
        ear_drop = ear_curve[start_idx] - ear_curve[min_idx]
        closing_speed = ear_drop / (closing_frames * dt)
    else:
        closing_speed = 0.0

    # 睁眼阶段
    opening_frames = end_idx - min_idx
    if opening_frames > 0:
        ear_rise = ear_curve[end_idx] - ear_curve[min_idx]
        opening_speed = ear_rise / (opening_frames * dt)
    else:
        opening_speed = 0.0

    return {
        'closing_speed': closing_speed,
        'opening_speed': opening_speed,
        'closing_frames': closing_frames,
        'opening_frames': opening_frames,
    }


class BlinkBase(BaseAction):
    """眨眼动作基类"""

    ACTION_NAME = "BlinkBase"
    ACTION_NAME_CN = "眨眼基类"

    # 动态特征名称 (比CloseEye多了眨眼速度相关)
    DYNAMIC_FEATURE_NAMES = [
        'left_motion_range', 'right_motion_range',
        'left_mean_velocity', 'right_mean_velocity',
        'left_smoothness', 'right_smoothness',
        'motion_asymmetry',
        # 眨眼特定
        'left_closing_speed', 'right_closing_speed',
        'left_opening_speed', 'right_opening_speed',
        'blink_duration', 'blink_symmetry',
    ]

    def find_peak_frame(
        self,
        landmarks_seq: List,
        w: int, h: int,
        **kwargs
    ) -> int:
        """找最小EAR帧 (眼睛闭得最紧)"""
        peak_idx, _, _ = find_min_ear_frame(landmarks_seq, w, h)
        return peak_idx

    def extract_indicators(
        self,
        landmarks,
        w: int, h: int,
        neutral_baseline: Optional[NeutralBaseline] = None
    ) -> Dict[str, float]:
        """提取眨眼指标 (与CloseEye类似)"""
        icd = compute_icd(landmarks, w, h)

        # 获取基准数据
        baseline_l_area = None
        baseline_r_area = None
        baseline_l_palp_len = None
        baseline_r_palp_len = None

        if neutral_baseline is not None:
            baseline_l_area = neutral_baseline.left_eye_area
            baseline_r_area = neutral_baseline.right_eye_area
            baseline_l_palp_len = neutral_baseline.left_palpebral_length
            baseline_r_palp_len = neutral_baseline.right_palpebral_length

        # 眼部测量 (使用基准)
        eyes = measure_eyes(
            landmarks, w, h,
            baseline_l_area, baseline_r_area,
            baseline_l_palp_len, baseline_r_palp_len
        )

        # 功能百分比计算
        l_closure = eyes.left.closure
        r_closure = eyes.right.closure
        max_closure = max(l_closure, r_closure)
        min_closure = min(l_closure, r_closure)
        function_pct = min_closure / max_closure if max_closure > 0 else 1.0

        # 闭拢比
        closure_ratio = l_closure / r_closure if r_closure > 0 else 1.0

        return {
            'icd': icd,

            # 左眼
            'left_eye_area': eyes.left.area_raw,
            'left_eye_area_norm': eyes.left.area_norm,
            'left_eye_openness': eyes.left.openness,
            'left_eye_closure': eyes.left.closure,
            'left_complete_closure': 1.0 if eyes.left.complete_closure else 0.0,
            'left_eye_ear': eyes.left.ear,

            # 右眼
            'right_eye_area': eyes.right.area_raw,
            'right_eye_area_norm': eyes.right.area_norm,
            'right_eye_openness': eyes.right.openness,
            'right_eye_closure': eyes.right.closure,
            'right_complete_closure': 1.0 if eyes.right.complete_closure else 0.0,
            'right_eye_ear': eyes.right.ear,

            # 对称性
            'eye_area_ratio': eyes.area_ratio,
            'closure_ratio': closure_ratio,
            'eye_asymmetry': eyes.asymmetry,

            # 功能评估
            'function_pct': function_pct,
            'both_complete_closure': 1.0 if (eyes.left.complete_closure and eyes.right.complete_closure) else 0.0,
        }

    def extract_dynamic_features(
        self,
        landmarks_seq: List,
        w: int, h: int,
        fps: float,
        neutral_baseline: Optional[NeutralBaseline] = None
    ) -> Dict[str, float]:
        """提取动态特征 (包含眨眼速度)"""
        if len(landmarks_seq) < 3:
            return {}

        # 计算EAR曲线
        left_ear_curve, right_ear_curve = compute_ear_curve(landmarks_seq, w, h)

        dt = 1.0 / fps if fps > 0 else 1.0 / 30.0

        # 基础运动范围
        l_range = left_ear_curve.max() - left_ear_curve.min()
        r_range = right_ear_curve.max() - right_ear_curve.min()

        # 速度
        l_velocity = np.abs(np.diff(left_ear_curve)) / dt
        r_velocity = np.abs(np.diff(right_ear_curve)) / dt

        l_mean_vel = float(np.mean(l_velocity)) if len(l_velocity) > 0 else 0.0
        r_mean_vel = float(np.mean(r_velocity)) if len(r_velocity) > 0 else 0.0

        # 平滑度
        l_smoothness = 1.0 / (1.0 + np.std(l_velocity)) if len(l_velocity) > 0 else 0.0
        r_smoothness = 1.0 / (1.0 + np.std(r_velocity)) if len(r_velocity) > 0 else 0.0

        # 运动不对称性
        max_range = max(l_range, r_range)
        motion_asymmetry = abs(l_range - r_range) / max_range if max_range > 0 else 0.0

        # 眨眼特定指标
        # 检测眨眼事件
        l_blink_events = detect_blink_events(left_ear_curve)
        r_blink_events = detect_blink_events(right_ear_curve)

        # 眨眼速度 (取第一个眨眼事件)
        l_closing_speed = 0.0
        l_opening_speed = 0.0
        r_closing_speed = 0.0
        r_opening_speed = 0.0
        blink_duration = 0.0
        blink_symmetry = 1.0

        if l_blink_events:
            l_speeds = compute_blink_speed(left_ear_curve, fps, l_blink_events[0])
            l_closing_speed = l_speeds['closing_speed']
            l_opening_speed = l_speeds['opening_speed']
            blink_duration = l_blink_events[0]['duration']

        if r_blink_events:
            r_speeds = compute_blink_speed(right_ear_curve, fps, r_blink_events[0])
            r_closing_speed = r_speeds['closing_speed']
            r_opening_speed = r_speeds['opening_speed']
            if blink_duration > 0:
                blink_duration = (blink_duration + r_blink_events[0]['duration']) / 2.0
            else:
                blink_duration = r_blink_events[0]['duration']

        # 眨眼对称性: 左右眼到达最小EAR的时间差
        if l_blink_events and r_blink_events:
            l_min_idx = l_blink_events[0]['min_idx']
            r_min_idx = r_blink_events[0]['min_idx']
            time_diff = abs(l_min_idx - r_min_idx)
            # 归一化到0-1, 差异越小越对称
            blink_symmetry = 1.0 / (1.0 + time_diff)

        return {
            'left_motion_range': float(l_range),
            'right_motion_range': float(r_range),
            'left_mean_velocity': l_mean_vel,
            'right_mean_velocity': r_mean_vel,
            'left_smoothness': float(l_smoothness),
            'right_smoothness': float(r_smoothness),
            'motion_asymmetry': float(motion_asymmetry),
            # 眨眼特定
            'left_closing_speed': l_closing_speed,
            'right_closing_speed': r_closing_speed,
            'left_opening_speed': l_opening_speed,
            'right_opening_speed': r_opening_speed,
            'blink_duration': blink_duration,
            'blink_symmetry': blink_symmetry,
        }

    def _build_interpretability(
        self,
        landmarks_seq: List,
        w: int, h: int,
        peak_idx: int,
        indicators: Dict,
        neutral_baseline: Optional[NeutralBaseline]
    ) -> Dict[str, Any]:
        """构建可解释性数据"""
        # 获取基准数据
        baseline_l_area = neutral_baseline.left_eye_area if neutral_baseline else None
        baseline_r_area = neutral_baseline.right_eye_area if neutral_baseline else None
        baseline_l_palp_len = neutral_baseline.left_palpebral_length if neutral_baseline else None
        baseline_r_palp_len = neutral_baseline.right_palpebral_length if neutral_baseline else None

        # 睁眼度曲线
        if baseline_l_area and baseline_r_area:
            left_openness_curve, right_openness_curve = compute_openness_curve(
                landmarks_seq, w, h,
                baseline_l_area, baseline_r_area,
                baseline_l_palp_len, baseline_r_palp_len
            )
        else:
            left_openness_curve = np.zeros(len(landmarks_seq))
            right_openness_curve = np.zeros(len(landmarks_seq))

        # EAR曲线
        left_ear_curve, right_ear_curve = compute_ear_curve(landmarks_seq, w, h)

        # 检测眨眼事件
        l_blink_events = detect_blink_events(left_ear_curve)
        r_blink_events = detect_blink_events(right_ear_curve)

        return {
            'peak_frame_idx': peak_idx,
            'total_frames': len(landmarks_seq),
            'peak_reason': 'min_ear (眼闭最紧瞬间)',
            'left_openness_curve': left_openness_curve,
            'right_openness_curve': right_openness_curve,
            'left_ear_curve': left_ear_curve,
            'right_ear_curve': right_ear_curve,
            'left_blink_events': l_blink_events,
            'right_blink_events': r_blink_events,
            'blink_analysis': self._analyze_blinks(l_blink_events, r_blink_events),
            'key_findings': self._generate_key_findings(indicators, l_blink_events, r_blink_events),
        }

    def _analyze_blinks(self, l_events: List, r_events: List) -> Dict:
        """分析眨眼事件"""
        return {
            'left_blink_count': len(l_events),
            'right_blink_count': len(r_events),
            'left_avg_duration': np.mean([e['duration'] for e in l_events]) if l_events else 0,
            'right_avg_duration': np.mean([e['duration'] for e in r_events]) if r_events else 0,
        }

    def _generate_key_findings(
        self,
        indicators: Dict,
        l_events: List,
        r_events: List
    ) -> List[str]:
        """生成关键发现"""
        findings = []

        l_closure = indicators.get('left_eye_closure', 0)
        r_closure = indicators.get('right_eye_closure', 0)
        l_complete = indicators.get('left_complete_closure', 0) > 0.5
        r_complete = indicators.get('right_complete_closure', 0) > 0.5

        # 闭眼情况
        if not l_complete and not r_complete:
            findings.append("眨眼时双眼均未完全闭合")
        elif not l_complete:
            findings.append(f"眨眼时左眼未完全闭合 (闭拢度: {l_closure*100:.1f}%)")
        elif not r_complete:
            findings.append(f"眨眼时右眼未完全闭合 (闭拢度: {r_closure*100:.1f}%)")
        else:
            findings.append(f"眨眼时双眼可完全闭合 (L: {l_closure*100:.1f}%, R: {r_closure*100:.1f}%)")

        # 眨眼次数
        l_count = len(l_events)
        r_count = len(r_events)
        findings.append(f"检测到眨眼次数: 左眼 {l_count}, 右眼 {r_count}")

        # 眨眼对称性
        if l_events and r_events:
            l_duration = np.mean([e['duration'] for e in l_events])
            r_duration = np.mean([e['duration'] for e in r_events])
            if abs(l_duration - r_duration) > 2:
                faster = "左眼" if l_duration < r_duration else "右眼"
                findings.append(f"眨眼速度不对称: {faster}更快")

        # 闭拢比
        closure_ratio = indicators.get('closure_ratio', 1.0)
        if abs(closure_ratio - 1.0) > 0.2:
            if closure_ratio < 1.0:
                findings.append(f"左眼闭拢能力较弱 (比值: {closure_ratio:.2f})")
            else:
                findings.append(f"右眼闭拢能力较弱 (比值: {closure_ratio:.2f})")

        return findings

    def visualize_peak_frame(
        self,
        frame: np.ndarray,
        landmarks,
        indicators: Dict,
        w: int, h: int
    ) -> np.ndarray:
        """可视化峰值帧"""
        img = frame.copy()

        # 判断闭眼状态
        l_complete = indicators.get('left_complete_closure', 0) > 0.5
        r_complete = indicators.get('right_complete_closure', 0) > 0.5

        # 绘制眼部轮廓 (颜色根据闭合状态)
        l_eye_pts = pts2d(landmarks, LM.EYE_CONTOUR_L, w, h).astype(np.int32)
        r_eye_pts = pts2d(landmarks, LM.EYE_CONTOUR_R, w, h).astype(np.int32)

        l_color = (0, 255, 0) if l_complete else (0, 0, 255)  # 绿=完全闭眼, 红=未闭
        r_color = (0, 255, 0) if r_complete else (0, 0, 255)

        cv2.polylines(img, [l_eye_pts], True, l_color, 2)
        cv2.polylines(img, [r_eye_pts], True, r_color, 2)

        # 绘制ICD线
        l_inner = tuple(map(int, pt2d(landmarks[LM.EYE_INNER_L], w, h)))
        r_inner = tuple(map(int, pt2d(landmarks[LM.EYE_INNER_R], w, h)))
        cv2.line(img, l_inner, r_inner, (128, 128, 128), 1)

        # 文字标注
        y = 30
        cv2.putText(img, f"{self.ACTION_NAME} ({self.ACTION_NAME_CN})", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        y += 30
        cv2.putText(img, f"ICD: {indicators['icd']:.1f}px", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 左眼
        y += 30
        l_closure = indicators.get('left_eye_closure', 0)
        l_status = "闭合" if l_complete else "未闭"
        cv2.putText(img, f"L Eye: {l_closure*100:.1f}% ({l_status})", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, l_color, 1)

        # 右眼
        y += 25
        r_closure = indicators.get('right_eye_closure', 0)
        r_status = "闭合" if r_complete else "未闭"
        cv2.putText(img, f"R Eye: {r_closure*100:.1f}% ({r_status})", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, r_color, 1)

        # EAR值
        y += 30
        l_ear = indicators.get('left_eye_ear', 0)
        r_ear = indicators.get('right_eye_ear', 0)
        cv2.putText(img, f"EAR: L={l_ear:.3f}, R={r_ear:.3f}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 功能百分比
        y += 25
        func_pct = indicators.get('function_pct', 0)
        cv2.putText(img, f"Function: {func_pct*100:.1f}%", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        return img


class SpontaneousEyeBlinkAction(BlinkBase):
    """自发眨眼动作"""
    ACTION_NAME = "SpontaneousEyeBlink"
    ACTION_NAME_CN = "自发眨眼"

    def _generate_key_findings(
        self,
        indicators: Dict,
        l_events: List,
        r_events: List
    ) -> List[str]:
        """自发眨眼的关键发现"""
        findings = super()._generate_key_findings(indicators, l_events, r_events)

        # 自发眨眼特定说明
        findings.insert(0, "【自发眨眼】自然状态下的眨眼反射")

        return findings