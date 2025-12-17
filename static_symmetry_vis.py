"""
静态对称性可视化工具
===================

用于验证眼/颊/嘴三个静态指标的测量正确性
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass


class StaticSymmetryVisualizer:
    """静态对称性可视化器"""

    # 颜色定义 (BGR)
    COLORS = {
        'normal': (0, 255, 0),  # 绿色 - 正常
        'mild': (0, 255, 255),  # 黄色 - 轻度异常
        'severe': (0, 0, 255),  # 红色 - 严重异常
        'left': (255, 0, 0),  # 蓝色 - 左侧
        'right': (0, 165, 255),  # 橙色 - 右侧
        'reference': (128, 128, 128),  # 灰色 - 参考线
        'text': (255, 255, 255),  # 白色 - 文字
    }

    # MediaPipe关键点索引
    LEFT_EYE_CONTOUR = [362, 398, 384, 385, 386, 387, 388, 263, 249, 390, 373, 374, 380, 381, 382]
    RIGHT_EYE_CONTOUR = [133, 173, 157, 158, 159, 160, 161, 33, 7, 163, 144, 145, 153, 154, 155]

    EYE_INNER_LEFT = 362
    EYE_INNER_RIGHT = 133
    EYE_OUTER_LEFT = 263
    EYE_OUTER_RIGHT = 33

    MOUTH_LEFT = 291
    MOUTH_RIGHT = 61
    NOSE_LEFT = 129
    NOSE_RIGHT = 358

    LEFT_CHEEK = [425, 426, 427]
    RIGHT_CHEEK = [205, 206, 207]

    def __init__(self):
        pass

    def get_point(self, landmarks, idx: int, w: int, h: int) -> Tuple[int, int]:
        """获取关键点像素坐标"""
        return (int(landmarks[idx].x * w), int(landmarks[idx].y * h))

    def get_points(self, landmarks, indices: List[int], w: int, h: int) -> List[Tuple[int, int]]:
        """批量获取关键点"""
        return [self.get_point(landmarks, idx, w, h) for idx in indices]

    def draw_eye_contours(
            self,
            image: np.ndarray,
            landmarks,
            eye_result: dict,
            show_measurements: bool = True
    ) -> np.ndarray:
        """
        绘制眼部轮廓和测量

        Args:
            image: 输入图像
            landmarks: MediaPipe landmarks
            eye_result: 眼部评估结果
            show_measurements: 是否显示测量值
        """
        img = image.copy()
        h, w = img.shape[:2]

        # 绘制左眼轮廓
        left_points = self.get_points(landmarks, self.LEFT_EYE_CONTOUR, w, h)
        left_pts = np.array(left_points, dtype=np.int32)
        cv2.polylines(img, [left_pts], True, self.COLORS['left'], 2)

        # 绘制右眼轮廓
        right_points = self.get_points(landmarks, self.RIGHT_EYE_CONTOUR, w, h)
        right_pts = np.array(right_points, dtype=np.int32)
        cv2.polylines(img, [right_pts], True, self.COLORS['right'], 2)

        # 绘制内眦连线 (单位长度)
        left_inner = self.get_point(landmarks, self.EYE_INNER_LEFT, w, h)
        right_inner = self.get_point(landmarks, self.EYE_INNER_RIGHT, w, h)
        cv2.line(img, left_inner, right_inner, self.COLORS['reference'], 1)

        # 绘制眼睑裂长度线
        left_outer = self.get_point(landmarks, self.EYE_OUTER_LEFT, w, h)
        right_outer = self.get_point(landmarks, self.EYE_OUTER_RIGHT, w, h)
        cv2.line(img, left_inner, left_outer, self.COLORS['left'], 1)
        cv2.line(img, right_inner, right_outer, self.COLORS['right'], 1)

        if show_measurements:
            # 显示测量值
            ratio = eye_result.get('eye_area_ratio', 1.0)
            status = eye_result.get('status', 'NORMAL')

            # 根据状态选择颜色
            if status == 'NORMAL':
                color = self.COLORS['normal']
            elif abs(1.0 - ratio) < 0.30:
                color = self.COLORS['mild']
            else:
                color = self.COLORS['severe']

            # 标注
            cv2.putText(img, f"Eye Ratio: {ratio:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(img, f"Status: {status}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 标注面积
            left_area = eye_result.get('left_eye_area_norm', 0)
            right_area = eye_result.get('right_eye_area_norm', 0)

            left_center = np.mean(left_pts, axis=0).astype(int)
            right_center = np.mean(right_pts, axis=0).astype(int)

            cv2.putText(img, f"L:{left_area:.3f}", tuple(left_center),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['left'], 1)
            cv2.putText(img, f"R:{right_area:.3f}", tuple(right_center),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['right'], 1)

        return img

    def draw_nasolabial_fold(
            self,
            image: np.ndarray,
            landmarks,
            nlf_result: dict,
            show_measurements: bool = True
    ) -> np.ndarray:
        """
        绘制鼻唇沟线和测量
        """
        img = image.copy()
        h, w = img.shape[:2]

        # 获取关键点
        left_nose = self.get_point(landmarks, self.NOSE_LEFT, w, h)
        right_nose = self.get_point(landmarks, self.NOSE_RIGHT, w, h)
        left_mouth = self.get_point(landmarks, self.MOUTH_LEFT, w, h)
        right_mouth = self.get_point(landmarks, self.MOUTH_RIGHT, w, h)

        # 绘制鼻唇沟线
        cv2.line(img, left_nose, left_mouth, self.COLORS['left'], 2)
        cv2.line(img, right_nose, right_mouth, self.COLORS['right'], 2)

        # 绘制面颊参考点
        left_cheek_pts = self.get_points(landmarks, self.LEFT_CHEEK, w, h)
        right_cheek_pts = self.get_points(landmarks, self.RIGHT_CHEEK, w, h)

        for pt in left_cheek_pts:
            cv2.circle(img, pt, 3, self.COLORS['left'], -1)
        for pt in right_cheek_pts:
            cv2.circle(img, pt, 3, self.COLORS['right'], -1)

        if show_measurements:
            ratio = nlf_result.get('length_ratio', 1.0)
            status = nlf_result.get('status', 'NORMAL')

            # 状态颜色
            if status == 'NORMAL':
                color = self.COLORS['normal']
            elif status == 'ABSENT':
                color = self.COLORS['severe']
            else:
                color = self.COLORS['mild']

            # 标注
            cv2.putText(img, f"NLF Ratio: {ratio:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(img, f"NLF Status: {status}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 显示长度
            left_len = nlf_result.get('left_nlf_length_norm', 0)
            right_len = nlf_result.get('right_nlf_length_norm', 0)

            mid_left = ((left_nose[0] + left_mouth[0]) // 2, (left_nose[1] + left_mouth[1]) // 2)
            mid_right = ((right_nose[0] + right_mouth[0]) // 2, (right_nose[1] + right_mouth[1]) // 2)

            cv2.putText(img, f"{left_len:.2f}", mid_left,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['left'], 1)
            cv2.putText(img, f"{right_len:.2f}", mid_right,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['right'], 1)

        return img

    def draw_oral_commissure(
            self,
            image: np.ndarray,
            landmarks,
            oral_result: dict,
            show_measurements: bool = True
    ) -> np.ndarray:
        """
        绘制口角位置和角度
        """
        img = image.copy()
        h, w = img.shape[:2]

        # 获取嘴角
        left_corner = self.get_point(landmarks, self.MOUTH_LEFT, w, h)
        right_corner = self.get_point(landmarks, self.MOUTH_RIGHT, w, h)

        # 计算口裂中线
        midline_y = (left_corner[1] + right_corner[1]) // 2
        midpoint_x = (left_corner[0] + right_corner[0]) // 2

        # 绘制口裂水平参考线
        cv2.line(img, (left_corner[0] - 50, midline_y),
                 (right_corner[0] + 50, midline_y),
                 self.COLORS['reference'], 1, cv2.LINE_AA)

        # 绘制嘴角
        # 根据高度差选择颜色
        height_diff = oral_result.get('height_diff', 0)

        # 左嘴角
        if height_diff > 0.02:  # 左侧下垂
            left_color = self.COLORS['severe']
        elif height_diff < -0.02:
            left_color = self.COLORS['normal']
        else:
            left_color = self.COLORS['normal']

        # 右嘴角
        if height_diff < -0.02:  # 右侧下垂
            right_color = self.COLORS['severe']
        elif height_diff > 0.02:
            right_color = self.COLORS['normal']
        else:
            right_color = self.COLORS['normal']

        cv2.circle(img, left_corner, 8, left_color, -1)
        cv2.circle(img, right_corner, 8, right_color, -1)

        # 绘制从中线到嘴角的连线
        cv2.line(img, (left_corner[0], midline_y), left_corner, self.COLORS['left'], 2)
        cv2.line(img, (right_corner[0], midline_y), right_corner, self.COLORS['right'], 2)

        if show_measurements:
            # 角度
            left_angle = oral_result.get('left_angle', 0)
            right_angle = oral_result.get('right_angle', 0)

            # 状态
            left_status = oral_result.get('left_status', 'NORMAL')
            right_status = oral_result.get('right_status', 'NORMAL')

            # 标注
            if left_status == 'DROOPING' or right_status == 'DROOPING':
                color = self.COLORS['severe']
            elif abs(height_diff) > 0.01:
                color = self.COLORS['mild']
            else:
                color = self.COLORS['normal']

            cv2.putText(img, f"Height Diff: {height_diff:.3f}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(img, f"L Angle: {left_angle:.1f}°", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['left'], 2)
            cv2.putText(img, f"R Angle: {right_angle:.1f}°", (10, 205),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['right'], 2)

            # 在嘴角旁标注状态
            cv2.putText(img, left_status, (left_corner[0] - 30, left_corner[1] + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, left_color, 1)
            cv2.putText(img, right_status, (right_corner[0] - 30, right_corner[1] + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, right_color, 1)

        return img

    def draw_comprehensive(
            self,
            image: np.ndarray,
            landmarks,
            result: dict
    ) -> np.ndarray:
        """
        绘制综合可视化
        """
        img = image.copy()

        # 绘制眼部
        img = self.draw_eye_contours(img, landmarks, result.get('eye', {}), False)

        # 绘制鼻唇沟
        img = self.draw_nasolabial_fold(img, landmarks, result.get('nlf', {}), False)

        # 绘制口角
        img = self.draw_oral_commissure(img, landmarks, result.get('oral', {}), False)

        # 绘制Sunnybrook评分
        h, w = img.shape[:2]
        sunnybrook = result.get('sunnybrook', {})

        # 评分板
        start_y = h - 120
        cv2.rectangle(img, (0, start_y), (250, h), (0, 0, 0), -1)
        cv2.rectangle(img, (0, start_y), (250, h), (255, 255, 255), 1)

        cv2.putText(img, "Sunnybrook Static Score", (10, start_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['text'], 1)

        eye_score = sunnybrook.get('eye_score', 0)
        cheek_score = sunnybrook.get('cheek_score', 0)
        mouth_score = sunnybrook.get('mouth_score', 0)
        total = sunnybrook.get('static_total', 0)

        cv2.putText(img, f"A1 Eye: {eye_score}", (10, start_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['text'], 1)
        cv2.putText(img, f"A2 Cheek: {cheek_score}", (10, start_y + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['text'], 1)
        cv2.putText(img, f"A3 Mouth: {mouth_score}", (10, start_y + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS['text'], 1)

        # 总分
        total_color = self.COLORS['normal'] if total == 0 else self.COLORS['severe']
        cv2.putText(img, f"Total: {total}/20", (10, start_y + 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, total_color, 2)

        return img

    def create_comparison_view(
            self,
            image: np.ndarray,
            landmarks,
            result: dict
    ) -> np.ndarray:
        """
        创建三格对比视图
        """
        h, w = image.shape[:2]

        # 创建三个子图
        eye_img = self.draw_eye_contours(image, landmarks, result.get('eye', {}))
        nlf_img = self.draw_nasolabial_fold(image, landmarks, result.get('nlf', {}))
        oral_img = self.draw_oral_commissure(image, landmarks, result.get('oral', {}))

        # 缩小每个图像
        scale = 0.5
        new_w, new_h = int(w * scale), int(h * scale)

        eye_small = cv2.resize(eye_img, (new_w, new_h))
        nlf_small = cv2.resize(nlf_img, (new_w, new_h))
        oral_small = cv2.resize(oral_img, (new_w, new_h))

        # 水平拼接
        combined = np.hstack([eye_small, nlf_small, oral_small])

        # 添加标题
        cv2.putText(combined, "Eye (A1)", (new_w // 2 - 30, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['text'], 2)
        cv2.putText(combined, "NLF (A2)", (new_w + new_w // 2 - 30, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['text'], 2)
        cv2.putText(combined, "Oral (A3)", (2 * new_w + new_w // 2 - 30, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLORS['text'], 2)

        return combined


def demo_visualization():
    """演示用法"""
    print("""
静态对称性可视化工具
====================

用法:
    from visualization import StaticSymmetryVisualizer

    # 初始化
    visualizer = StaticSymmetryVisualizer()

    # 绘制单独区域
    eye_vis = visualizer.draw_eye_contours(image, landmarks, eye_result)
    nlf_vis = visualizer.draw_nasolabial_fold(image, landmarks, nlf_result)
    oral_vis = visualizer.draw_oral_commissure(image, landmarks, oral_result)

    # 绘制综合视图
    comprehensive = visualizer.draw_comprehensive(image, landmarks, result)

    # 创建对比视图
    comparison = visualizer.create_comparison_view(image, landmarks, result)

    # 保存
    cv2.imwrite('static_symmetry_analysis.jpg', comprehensive)
""")


if __name__ == '__main__':
    demo_visualization()