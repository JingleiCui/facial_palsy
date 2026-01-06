#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断可视化辅助模块 - Diagnosis Visualization Helpers
====================================================

提供在peak_indicators.jpg上标注患侧和严重度信息的通用函数。
"""

import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional


# =============================================================================
# 颜色定义
# =============================================================================

class DiagnosisColors:
    """诊断结果颜色编码 (BGR格式)"""

    # 患侧状态
    SYMMETRIC = (0, 255, 0)  # 绿色 - 对称/正常
    LEFT_PALSY = (255, 0, 0)  # 蓝色 - 左侧面瘫（患者左侧）
    RIGHT_PALSY = (0, 0, 255)  # 红色 - 右侧面瘫（患者右侧）
    UNCERTAIN = (0, 165, 255)  # 橙色 - 不确定

    # 严重度等级
    SEVERITY_1 = (0, 255, 0)  # 绿色 - 正常
    SEVERITY_2 = (0, 255, 255)  # 黄色 - 轻度
    SEVERITY_3 = (0, 165, 255)  # 橙色 - 中度
    SEVERITY_4 = (0, 100, 255)  # 深橙色 - 中重度
    SEVERITY_5 = (0, 0, 255)  # 红色 - 重度

    # 背景
    PANEL_BG = (40, 40, 40)  # 深灰色面板背景
    PANEL_BORDER = (100, 100, 100)  # 面板边框

    @classmethod
    def get_palsy_color(cls, palsy_side: int) -> Tuple[int, int, int]:
        """根据患侧获取颜色"""
        if palsy_side == 0:
            return cls.SYMMETRIC
        elif palsy_side == 1:
            return cls.LEFT_PALSY
        elif palsy_side == 2:
            return cls.RIGHT_PALSY
        else:
            return cls.UNCERTAIN

    @classmethod
    def get_severity_color(cls, severity: int) -> Tuple[int, int, int]:
        """根据严重度获取颜色"""
        colors = {
            1: cls.SEVERITY_1,
            2: cls.SEVERITY_2,
            3: cls.SEVERITY_3,
            4: cls.SEVERITY_4,
            5: cls.SEVERITY_5,
        }
        return colors.get(severity, cls.UNCERTAIN)


# =============================================================================
# 标注绘制函数
# =============================================================================

def draw_diagnosis_badge(
        img: np.ndarray,
        palsy_side: int,
        palsy_confidence: float,
        severity_score: int,
        voluntary_score: int,
        position: str = "top_right",
        show_confidence: bool = True
) -> np.ndarray:
    """
    在图像上绘制诊断徽章（患侧+严重度）

    Args:
        img: 输入图像
        palsy_side: 患侧 (0=无, 1=左, 2=右)
        palsy_confidence: 患侧判断置信度 (0-1)
        severity_score: 严重度 (1-5)
        voluntary_score: Voluntary Movement Score (1-5)
        position: 位置 ("top_right", "top_left", "bottom_right", "bottom_left")
        show_confidence: 是否显示置信度

    Returns:
        标注后的图像
    """
    h, w = img.shape[:2]
    result = img.copy()

    # 徽章尺寸
    badge_w, badge_h = 220, 90
    padding = 15

    # 确定位置
    if position == "top_right":
        x = w - badge_w - padding
        y = padding
    elif position == "top_left":
        x = padding
        y = padding
    elif position == "bottom_right":
        x = w - badge_w - padding
        y = h - badge_h - padding
    else:  # bottom_left
        x = padding
        y = h - badge_h - padding

    # 绘制背景
    overlay = result.copy()
    cv2.rectangle(overlay, (x, y), (x + badge_w, y + badge_h),
                  DiagnosisColors.PANEL_BG, -1)
    cv2.rectangle(overlay, (x, y), (x + badge_w, y + badge_h),
                  DiagnosisColors.PANEL_BORDER, 2)

    # 混合（半透明效果）
    alpha = 0.85
    result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)

    # 获取颜色
    palsy_color = DiagnosisColors.get_palsy_color(palsy_side)
    severity_color = DiagnosisColors.get_severity_color(severity_score)

    # 患侧文本
    palsy_texts = {0: "Symmetric", 1: "LEFT PALSY", 2: "RIGHT PALSY"}
    palsy_text = palsy_texts.get(palsy_side, "UNCERTAIN")

    # 绘制患侧标签
    text_x = x + 10
    text_y = y + 28
    cv2.putText(result, palsy_text, (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, palsy_color, 2)

    # 绘制置信度条
    if show_confidence and palsy_side != 0:
        bar_x = text_x
        bar_y = text_y + 8
        bar_w = int(180 * palsy_confidence)
        cv2.rectangle(result, (bar_x, bar_y), (bar_x + 180, bar_y + 6),
                      (80, 80, 80), -1)
        cv2.rectangle(result, (bar_x, bar_y), (bar_x + bar_w, bar_y + 6),
                      palsy_color, -1)
        cv2.putText(result, f"{palsy_confidence:.0%}", (bar_x + 185, bar_y + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    # 绘制严重度和Voluntary Score
    sev_y = text_y + 35
    severity_texts = {1: "Normal", 2: "Mild", 3: "Moderate", 4: "Mod-Severe", 5: "Severe"}
    sev_text = severity_texts.get(severity_score, "?")

    cv2.putText(result, f"Severity: {severity_score}/5 ({sev_text})",
                (text_x, sev_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, severity_color, 1)

    vol_y = sev_y + 20
    cv2.putText(result, f"Voluntary: {voluntary_score}/5",
                (text_x, vol_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return result


def draw_diagnosis_panel(
        img: np.ndarray,
        action_name: str,
        action_name_cn: str,
        palsy_side: int,
        palsy_confidence: float,
        severity_score: int,
        voluntary_score: int,
        metrics: Dict[str, Any] = None,
        panel_position: Tuple[int, int] = (10, 100),
        panel_size: Tuple[int, int] = (350, 280)
) -> np.ndarray:
    """
    绘制完整的诊断信息面板

    Args:
        img: 输入图像
        action_name: 动作英文名
        action_name_cn: 动作中文名
        palsy_side: 患侧
        palsy_confidence: 置信度
        severity_score: 严重度
        voluntary_score: Voluntary Score
        metrics: 额外指标字典
        panel_position: 面板位置 (x, y)
        panel_size: 面板尺寸 (w, h)

    Returns:
        标注后的图像
    """
    result = img.copy()
    x, y = panel_position
    w, h = panel_size

    # 绘制面板背景
    overlay = result.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), DiagnosisColors.PANEL_BG, -1)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), DiagnosisColors.PANEL_BORDER, 2)
    alpha = 0.9
    result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)

    # 字体设置
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    # 标题
    title_y = y + 35
    cv2.putText(result, f"{action_name} - {action_name_cn}", (x + 15, title_y),
                FONT, 0.8, (0, 255, 0), 2)

    # 分隔线
    cv2.line(result, (x + 10, title_y + 10), (x + w - 10, title_y + 10),
             (100, 100, 100), 1)

    line_y = title_y + 35
    line_h = 25

    # 显示额外指标
    if metrics:
        for key, value in metrics.items():
            if isinstance(value, float):
                text = f"{key}: {value:.3f}"
            else:
                text = f"{key}: {value}"
            cv2.putText(result, text, (x + 15, line_y), FONT, 0.5, (255, 255, 255), 1)
            line_y += line_h

    # 分隔线
    cv2.line(result, (x + 10, line_y), (x + w - 10, line_y), (100, 100, 100), 1)
    line_y += 15

    # === 诊断结果区域 ===
    cv2.putText(result, "=== Diagnosis ===", (x + 15, line_y), FONT, 0.6, (0, 255, 255), 1)
    line_y += 30

    # 患侧
    palsy_color = DiagnosisColors.get_palsy_color(palsy_side)
    palsy_texts = {0: "Symmetric", 1: "Left Palsy", 2: "Right Palsy"}
    palsy_text = palsy_texts.get(palsy_side, "Uncertain")
    cv2.putText(result, f"Palsy: {palsy_text}", (x + 15, line_y),
                FONT, 0.65, palsy_color, 2)
    line_y += line_h

    # 置信度条
    if palsy_side != 0:
        bar_x = x + 15
        bar_w = int(200 * palsy_confidence)
        cv2.rectangle(result, (bar_x, line_y - 5), (bar_x + 200, line_y + 5),
                      (80, 80, 80), -1)
        cv2.rectangle(result, (bar_x, line_y - 5), (bar_x + bar_w, line_y + 5),
                      palsy_color, -1)
        cv2.putText(result, f"Conf: {palsy_confidence:.0%}", (bar_x + 210, line_y + 5),
                    FONT, 0.45, (200, 200, 200), 1)
        line_y += line_h

    # 严重度
    severity_color = DiagnosisColors.get_severity_color(severity_score)
    severity_texts = {1: "Normal", 2: "Mild", 3: "Moderate", 4: "Mod-Severe", 5: "Severe"}
    sev_text = severity_texts.get(severity_score, "?")
    cv2.putText(result, f"Severity: {severity_score}/5 ({sev_text})",
                (x + 15, line_y), FONT, 0.6, severity_color, 2)
    line_y += line_h

    # Voluntary Score
    cv2.putText(result, f"Voluntary Score: {voluntary_score}/5",
                (x + 15, line_y), FONT, 0.6, (0, 255, 255), 2)

    return result


def draw_palsy_indicator_on_face(
        img: np.ndarray,
        palsy_side: int,
        landmarks=None,
        w: int = None,
        h: int = None
) -> np.ndarray:
    """
    在面部图像上绘制患侧指示器（箭头指向患侧）

    Args:
        img: 输入图像
        palsy_side: 患侧 (1=左, 2=右)
        landmarks: 面部关键点（可选，用于定位）
        w, h: 图像尺寸

    Returns:
        标注后的图像
    """
    if palsy_side == 0:
        return img

    result = img.copy()
    img_h, img_w = result.shape[:2]

    # 确定箭头位置
    if palsy_side == 1:  # 患者左侧 = 图像右侧
        arrow_x = int(img_w * 0.85)
        text = "LEFT"
        color = DiagnosisColors.LEFT_PALSY
    else:  # 患者右侧 = 图像左侧
        arrow_x = int(img_w * 0.15)
        text = "RIGHT"
        color = DiagnosisColors.RIGHT_PALSY

    arrow_y = int(img_h * 0.5)

    # 绘制箭头
    arrow_len = 60
    arrow_thickness = 3

    if palsy_side == 1:  # 指向右边（图像右侧=患者左侧）
        cv2.arrowedLine(result, (arrow_x - arrow_len, arrow_y), (arrow_x, arrow_y),
                        color, arrow_thickness, tipLength=0.3)
    else:  # 指向左边
        cv2.arrowedLine(result, (arrow_x + arrow_len, arrow_y), (arrow_x, arrow_y),
                        color, arrow_thickness, tipLength=0.3)

    # 绘制文本
    text_x = arrow_x - 30 if palsy_side == 2 else arrow_x - 50
    cv2.putText(result, f"{text} PALSY", (text_x, arrow_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return result


def add_diagnosis_overlay(
        img: np.ndarray,
        palsy_side: int,
        palsy_confidence: float,
        severity_score: int,
        voluntary_score: int,
        show_badge: bool = True,
        show_face_indicator: bool = True,
        badge_position: str = "top_right"
) -> np.ndarray:
    """
    添加完整的诊断覆盖层

    整合徽章和面部指示器

    Args:
        img: 输入图像
        palsy_side: 患侧
        palsy_confidence: 置信度
        severity_score: 严重度
        voluntary_score: Voluntary Score
        show_badge: 是否显示徽章
        show_face_indicator: 是否显示面部指示器
        badge_position: 徽章位置

    Returns:
        标注后的图像
    """
    result = img.copy()

    if show_badge:
        result = draw_diagnosis_badge(
            result, palsy_side, palsy_confidence,
            severity_score, voluntary_score,
            position=badge_position
        )

    if show_face_indicator and palsy_side != 0:
        result = draw_palsy_indicator_on_face(result, palsy_side)

    return result


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    # 创建测试图像
    test_img = np.zeros((600, 800, 3), dtype=np.uint8)
    test_img[:] = (50, 50, 50)

    # 测试徽章
    result = draw_diagnosis_badge(
        test_img,
        palsy_side=1,
        palsy_confidence=0.85,
        severity_score=3,
        voluntary_score=3
    )

    # 测试面部指示器
    result = draw_palsy_indicator_on_face(result, palsy_side=1)

    # 保存测试结果
    cv2.imwrite("test_diagnosis_overlay.jpg", result)
    print("Test image saved to test_diagnosis_overlay.jpg")