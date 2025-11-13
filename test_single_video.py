"""
单视频测试脚本 - 用于测试处理流程

使用方法:
    python facialPalsy/test_single_video.py <video_path> <action_name>

示例:
    python facialPalsy/test_single_video.py /path/to/video.mov Smile
"""

import sys
import os
import cv2
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from core.landmark_extractor import LandmarkExtractor
from actions import get_action_class


def test_single_video(video_path, action_name):
    """
    测试单个视频的处理流程

    Args:
        video_path: 视频文件路径
        action_name: 动作名称(英文),如 'Smile'
    """
    print("=" * 80)
    print(f"测试视频处理流程")
    print(f"  视频: {video_path}")
    print(f"  动作: {action_name}")
    print("=" * 80)

    # 检查文件是否存在
    if not os.path.exists(video_path):
        print(f"错误: 文件不存在: {video_path}")
        return False

    # 检查动作是否有效
    ActionClass = get_action_class(action_name)
    if not ActionClass:
        print(f"错误: 未知动作类型: {action_name}")
        print(f"可用动作: {list(ACTION_CONFIG.keys())}")
        return False

    # 获取动作配置
    action_config = ACTION_CONFIG.get(action_name)
    if not action_config:
        print(f"错误: 未找到动作配置: {action_name}")
        return False

    print(f"\n动作配置: {action_config}")

    # 创建LandmarkExtractor
    print(f"\n初始化MediaPipe...")
    print(f"  模型路径: {MEDIAPIPE_MODEL_PATH}")

    try:
        with LandmarkExtractor(MEDIAPIPE_MODEL_PATH) as extractor:
            # 获取视频信息
            video_info = extractor.get_video_info(video_path)
            print(f"\n视频信息:")
            print(f"  分辨率: {video_info['width']} x {video_info['height']}")
            print(f"  帧率: {video_info['fps']:.2f}")
            print(f"  总帧数: {video_info['total_frames']}")

            # 提取关键点序列
            print(f"\n提取关键点序列...")
            landmarks_seq, frames_seq = extractor.extract_sequence(video_path)

            if not landmarks_seq or not frames_seq:
                print(f"错误: 关键点提取失败")
                return False

            # 统计有效帧
            valid_count = sum(1 for lm in landmarks_seq if lm is not None)
            valid_ratio = valid_count / len(landmarks_seq)

            print(f"  总帧数: {len(landmarks_seq)}")
            print(f"  有效帧: {valid_count} ({valid_ratio*100:.1f}%)")

            if valid_ratio < MIN_VALID_FRAME_RATIO:
                print(f"  警告: 有效帧比例过低 (< {MIN_VALID_FRAME_RATIO*100}%)")

            # 创建动作实例
            action = ActionClass(action_name, action_config)

            # 处理视频
            print(f"\n处理视频...")
            result = action.process(
                landmarks_seq,
                frames_seq,
                video_info['width'],
                video_info['height'],
                video_info['fps']
            )

            if not result:
                print(f"错误: 处理失败")
                return False

            # 打印结果
            print(f"\n处理结果:")
            print(f"  峰值帧索引: {result['peak_frame_idx']} / {len(landmarks_seq)}")
            print(f"  单位长度: {result['unit_length']:.2f} 像素")
            print(f"  静态特征: {result['static_features'].shape} - {result['static_features'].dtype}")
            print(f"  动态特征: {result['dynamic_features'].shape} - {result['dynamic_features'].dtype}")

            # 打印部分特征值
            print(f"\n静态特征前8维:")
            print(f"  {result['static_features'][:8]}")

            print(f"\n动态特征前8维:")
            print(f"  {result['dynamic_features'][:8]}")

            # 保存峰值帧
            peak_frame_path = f"test_peak_frame_{action_name}.jpg"
            cv2.imwrite(peak_frame_path, result['peak_frame'])
            print(f"\n保存峰值帧: {peak_frame_path}")

            # 保存特征
            feature_path = f"test_features_{action_name}.npz"
            np.savez_compressed(
                feature_path,
                static_features=result['static_features'],
                dynamic_features=result['dynamic_features'],
                peak_frame_idx=result['peak_frame_idx'],
                unit_length=result['unit_length']
            )
            print(f"保存特征文件: {feature_path}")

            print(f"\n" + "=" * 80)
            print(f"测试成功!")
            print(f"=" * 80)

            return True

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    if len(sys.argv) < 3:
        print("使用方法:")
        print(f"  python {sys.argv[0]} <video_path> <action_name>")
        print(f"\n示例:")
        print(f"  python {sys.argv[0]} /path/to/video.mov Smile")
        print(f"\n可用动作:")
        for action_name, config in ACTION_CONFIG.items():
            print(f"  {action_name:20s} - {config['name_cn']}")
        sys.exit(1)

    video_path = sys.argv[1]
    action_name = sys.argv[2]

    success = test_single_video(video_path, action_name)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
