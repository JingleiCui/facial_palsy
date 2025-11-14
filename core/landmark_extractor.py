"""
MediaPipe关键点提取器
"""
import cv2
import numpy as np
import mediapipe as mp


class LandmarkExtractor:
    """MediaPipe面部关键点提取器"""

    def __init__(self, model_path):
        """
        初始化

        Args:
            model_path: MediaPipe模型路径
        """
        self.model_path = model_path

        BaseOptions = mp.tasks.BaseOptions
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1
        )

        self.landmarker = None

    def __enter__(self):
        """上下文管理器 - 进入"""
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        self.landmarker = FaceLandmarker.create_from_options(self.options)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器 - 退出"""
        if self.landmarker:
            self.landmarker.close()

    def _ensure_landmarker(self):
        """确保 self.landmarker 已经初始化"""
        if self.landmarker is None:
            FaceLandmarker = mp.tasks.vision.FaceLandmarker
            self.landmarker = FaceLandmarker.create_from_options(self.options)

    def extract_from_frame(self, frame):
        """
        从单帧图像提取关键点

        Args:
            frame: BGR图像 (OpenCV格式)

        Returns:
            face_landmarks: MediaPipe landmarks对象，如果未检测到返回None
        """
        if self.landmarker is None:
            self._ensure_landmarker()

        # 转换为RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 创建MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # 检测
        results = self.landmarker.detect(mp_image)

        if results and results.face_landmarks and len(results.face_landmarks) > 0:
            return results.face_landmarks[0]

        return None

    def extract_sequence(self, video_path, start_frame=0, end_frame=None):
        """
        从视频中提取关键点序列

        Args:
            video_path: 视频文件路径
            start_frame: 起始帧（包含）
            end_frame: 结束帧（包含），None表示到视频结尾

        Returns:
            tuple: (landmarks_sequence, frames_sequence)
                landmarks_sequence: list of face_landmarks对象
                frames_sequence: list of frames
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return None, None

        # 获取视频信息
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if end_frame is None or end_frame >= total_frames:
            end_frame = total_frames - 1

        # 跳转到起始帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        landmarks_sequence = []
        frames_sequence = []
        current_frame = start_frame

        while current_frame <= end_frame:
            ret, frame = cap.read()

            if not ret:
                break

            # 提取关键点
            landmarks = self.extract_from_frame(frame)

            landmarks_sequence.append(landmarks)
            frames_sequence.append(frame)

            current_frame += 1

        cap.release()

        return landmarks_sequence, frames_sequence

    @staticmethod
    def get_video_info(video_path):
        """
        获取视频信息

        Args:
            video_path: 视频文件路径

        Returns:
            dict: {
                'width': int,
                'height': int,
                'fps': float,
                'total_frames': int
            }
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return None

        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }

        cap.release()

        return info