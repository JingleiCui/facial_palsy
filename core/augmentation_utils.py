"""
augmentation_utils
===================

This module provides helper functions for performing data augmentation on
facial landmark sequences and video frames. It includes utilities for
horizontal flipping, rotating landmarks, adjusting brightness/contrast of
frames, and flipping palsy-side labels.

Functions:
    flip_palsy_side(side: int) -> int:
        Given a palsy-side label (1 for left, 2 for right, 0 for none), return
        the label corresponding to a horizontally flipped sample.

    horizontal_flip_landmarks_seq(lm_seq: list, img_width: int, normalized: bool = False) -> list:
        Return a new landmarks sequence where x-coordinates are mirrored. If
        normalized is True, x coordinates are assumed to be in [0,1], otherwise
        pixel coordinates in [0, img_width-1] are mirrored.

    rotate_landmarks_seq(lm_seq: list, angle_deg: float, center: tuple, normalized: bool = False) -> list:
        Rotate a sequence of landmarks around a given center by angle_deg
        degrees. If normalized is True, center is in normalized coordinates.

    adjust_brightness_contrast(frame: ndarray, brightness: float = 1.0, contrast: float = 1.0) -> ndarray:
        Adjust the brightness and contrast of a frame. Brightness values less
        than 1.0 darken the image, values greater than 1.0 brighten it.
        Contrast values <1.0 reduce contrast, >1.0 increase contrast.

These functions should be imported and used by the video processing pipeline
to generate augmented versions of the extracted landmark sequences and peak
frames.
"""

from typing import List, Tuple
import numpy as np
import cv2

def flip_palsy_side(side: int) -> int:
    """Horizontally flip the palsy-side label.

    Args:
        side: Integer label. 0 means no palsy, 1 means left-side palsy,
            2 means right-side palsy.

    Returns:
        Flipped palsy-side label: 1 becomes 2, 2 becomes 1, 0 unchanged.
    """
    if side == 1:
        return 2
    elif side == 2:
        return 1
    return side


def horizontal_flip_landmarks_seq(lm_seq: List[np.ndarray], img_width: int, normalized: bool = False) -> List[np.ndarray]:
    """Flip a sequence of landmarks horizontally.

    Args:
        lm_seq: A list of landmark arrays for each frame. Each array has
            shape (N, 3) or (N, 2), where the first column is x-coordinates.
        img_width: Width of the image in pixels. Used when normalized=False.
        normalized: If True, x-values are assumed to be normalized in [0,1].

    Returns:
        A new list of landmark arrays with x-coordinates mirrored.
    """
    flipped_seq = []
    for lm in lm_seq:
        if lm is None:
            flipped_seq.append(None)
            continue
        lm = np.asarray(lm)
        lm_flipped = lm.copy()
        if normalized:
            # x in [0,1] -> 1 - x
            lm_flipped[:, 0] = 1.0 - lm_flipped[:, 0]
        else:
            # x in [0, img_width-1] -> (img_width-1) - x
            lm_flipped[:, 0] = img_width - 1 - lm_flipped[:, 0]
        flipped_seq.append(lm_flipped)
    return flipped_seq

def mirror_indicators_dict(ind: dict) -> dict:
    out = {}

    for k, v in ind.items():
        # 1) 左右互换
        if "left" in k:
            out[k.replace("left", "right")] = v
        elif "right" in k:
            out[k.replace("right", "left")] = v

        # 2) 比值: left/right -> right/left
        elif "ratio" in k:
            # 比如 eye_opening_ratio = left/right
            # 镜像后应该是 right/left = 1/(left/right)
            out[k] = 1.0 / (v + 1e-6)

        # 3) 差值: left - right -> right - left
        # elif "diff" in k:
        #     out[k] = -v

        # 4) 与左右无关的全局量：直接保留
        else:
            out[k] = v

    return out


def rotate_landmarks_seq(lm_seq: List[np.ndarray], angle_deg: float, center: Tuple[float, float], normalized: bool = False) -> List[np.ndarray]:
    """Rotate a sequence of landmarks around a given center.

    Args:
        lm_seq: List of landmark arrays, each with shape (N, 2) or (N, 3).
        angle_deg: Rotation angle in degrees, counter-clockwise.
        center: The rotation center (cx, cy). If normalized=True,
            coordinates should be in normalized units.
        normalized: If True, coordinates are normalized (0-1) instead of pixel.

    Returns:
        A new list of landmark arrays rotated by the specified angle.
    """
    rad = np.deg2rad(angle_deg)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)
    rotated_seq = []
    cx, cy = center
    for lm in lm_seq:
        if lm is None:
            rotated_seq.append(None)
            continue
        lm = np.asarray(lm)
        lm_rot = lm.copy()
        xs = lm_rot[:, 0] - cx
        ys = lm_rot[:, 1] - cy
        # Apply rotation
        x_new = cos_a * xs - sin_a * ys + cx
        y_new = sin_a * xs + cos_a * ys + cy
        lm_rot[:, 0] = x_new
        lm_rot[:, 1] = y_new
        rotated_seq.append(lm_rot)
    return rotated_seq


def adjust_brightness_contrast(frame: np.ndarray, brightness: float = 1.0, contrast: float = 1.0) -> np.ndarray:
    """Adjust brightness and contrast of an image.

    Args:
        frame: Input image (BGR or grayscale) as a NumPy array.
        brightness: Brightness factor. 1.0 means no change. Values less than 1
            darken the image, values greater than 1 brighten it.
        contrast: Contrast factor. 1.0 means no change. Values less than 1
            decrease contrast, values greater than 1 increase contrast.

    Returns:
        Adjusted image as a NumPy array.
    """
    # Beta is shift added to all channels; scale factor alpha adjusts contrast.
    beta = (brightness - 1.0) * 128.0
    return cv2.convertScaleAbs(frame, alpha=contrast, beta=beta)