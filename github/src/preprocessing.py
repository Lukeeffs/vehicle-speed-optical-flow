"""
preprocessing.py
================
Implements photometric and geometric preprocessing for monocular dashcam frames:
1. Grayscale luminance conversion.
2. Contrast-Limited Adaptive Histogram Equalization (CLAHE) for shadow invariance.
3. Gaussian low-pass spatial smoothing for stable spatio-temporal derivatives.
4. Perspective Region-of-Interest (ROI) cropping & trapezoidal road-plane masking
   (removing non-rigid sky motion and static ego-vehicle hood glare).
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def build_trapezoid_road_mask(
    height: int,
    width: int,
    vertices_norm: dict,
) -> np.ndarray:
    """
    Construct a binary mask (uint8, values 0 or 1) corresponding to the perspective
    trapezoid of the drivable road plane.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(
        [
            [int(vertices_norm["top_left"][0] * width), int(vertices_norm["top_left"][1] * height)],
            [int(vertices_norm["top_right"][0] * width), int(vertices_norm["top_right"][1] * height)],
            [int(vertices_norm["bottom_right"][0] * width), int(vertices_norm["bottom_right"][1] * height)],
            [int(vertices_norm["bottom_left"][0] * width), int(vertices_norm["bottom_left"][1] * height)],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [pts], 1)
    return mask


def get_roi_bounds(height: int, width: int, preproc_cfg: dict) -> Tuple[int, int, int, int]:
    """Return pixel bounds (y_min, y_max, x_min, x_max) for the rectangular road ROI."""
    y_min = int(preproc_cfg["roi_y_min_frac"] * height)
    y_max = int(preproc_cfg["roi_y_max_frac"] * height)
    x_min = int(preproc_cfg["roi_x_min_frac"] * width)
    x_max = int(preproc_cfg["roi_x_max_frac"] * width)
    return y_min, y_max, x_min, x_max


def preprocess_frame(
    frame_bgr: np.ndarray,
    preproc_cfg: dict,
    use_clahe: bool = True,
    crop_roi: bool = True,
) -> np.ndarray:
    """
    Convert BGR frame to grayscale, optionally apply CLAHE and Gaussian blur,
    and crop to the perspective road ROI.
    """
    if frame_bgr.ndim == 3:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame_bgr.copy()

    if use_clahe:
        clip_lim = float(preproc_cfg.get("clahe_clip_limit", 2.0))
        grid_sz = tuple(preproc_cfg.get("clahe_grid_size", [8, 8]))
        clahe = cv2.createCLAHE(clipLimit=clip_lim, tileGridSize=grid_sz)
        gray = clahe.apply(gray)

    ksize = tuple(preproc_cfg.get("gaussian_kernel_size", [5, 5]))
    sigma = float(preproc_cfg.get("gaussian_sigma", 1.2))
    smoothed = cv2.GaussianBlur(gray, ksize, sigma)

    if crop_roi:
        h, w = smoothed.shape
        y_min, y_max, x_min, x_max = get_roi_bounds(h, w, preproc_cfg)
        return smoothed[y_min:y_max, x_min:x_max]

    return smoothed
