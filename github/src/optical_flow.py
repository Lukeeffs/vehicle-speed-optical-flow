"""
optical_flow.py
===============
Dense and Sparse Optical Flow extractors for monocular vehicle speed estimation:
1. Dense Gunnar Farnebäck polynomial expansion optical flow.
2. Sparse Pyramidal Lucas-Kanade optical flow over Shi-Tomasi corners with
   forward-backward consistency verification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


@dataclass
class SparseFlowResult:
    """Container for sparse Lucas-Kanade tracked keypoints and displacement vectors."""
    prev_pts: np.ndarray       # Shape: (K, 2)
    next_pts: np.ndarray       # Shape: (K, 2)
    flow_vectors: np.ndarray   # Shape: (K, 2) -> (u, v)
    magnitudes: np.ndarray     # Shape: (K,)
    valid_count: int


def compute_dense_farneback_flow(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    fb_cfg: dict,
) -> np.ndarray:
    """
    Compute dense 2-channel optical flow field (u, v) in pixels/frame using
    Farnebäck's quadratic polynomial expansion algorithm.

    Returns
    -------
    flow : np.ndarray of shape (H, W, 2), dtype float32
        flow[..., 0] = horizontal displacement u(x, y)
        flow[..., 1] = vertical displacement v(x, y)
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev=prev_gray,
        next=curr_gray,
        flow=None,
        pyr_scale=float(fb_cfg.get("pyr_scale", 0.5)),
        levels=int(fb_cfg.get("levels", 3)),
        winsize=int(fb_cfg.get("winsize", 15)),
        iterations=int(fb_cfg.get("iterations", 3)),
        poly_n=int(fb_cfg.get("poly_n", 5)),
        poly_sigma=float(fb_cfg.get("poly_sigma", 1.2)),
        flags=int(fb_cfg.get("flags", 0)),
    )
    return flow


def compute_sparse_lk_flow(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    lk_cfg: dict,
    mask: Optional[np.ndarray] = None,
    fb_threshold: float = 1.5,
) -> SparseFlowResult:
    """
    Detect Shi-Tomasi corner keypoints in `prev_gray` and track them into `curr_gray`
    using pyramidal Lucas-Kanade optical flow with forward-backward consistency check.
    """
    pts0 = cv2.goodFeaturesToTrack(
        image=prev_gray,
        maxCorners=int(lk_cfg.get("max_corners", 200)),
        qualityLevel=float(lk_cfg.get("quality_level", 0.01)),
        minDistance=float(lk_cfg.get("min_distance", 7)),
        mask=mask,
        blockSize=int(lk_cfg.get("block_size", 7)),
    )

    if pts0 is None or len(pts0) == 0:
        empty = np.zeros((0, 2), dtype=np.float32)
        return SparseFlowResult(
            prev_pts=empty,
            next_pts=empty,
            flow_vectors=empty,
            magnitudes=np.zeros((0,), dtype=np.float32),
            valid_count=0,
        )

    lk_params = dict(
        winSize=tuple(lk_cfg.get("win_size", [15, 15])),
        maxLevel=int(lk_cfg.get("max_level", 2)),
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
    )

    pts1, st_fwd, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, pts0, None, **lk_params)
    pts0_back, st_bwd, _ = cv2.calcOpticalFlowPyrLK(curr_gray, prev_gray, pts1, None, **lk_params)

    fb_err = np.linalg.norm(pts0 - pts0_back, axis=2).reshape(-1)
    valid = (st_fwd.reshape(-1) == 1) & (st_bwd.reshape(-1) == 1) & (fb_err < fb_threshold)

    if not np.any(valid):
        empty = np.zeros((0, 2), dtype=np.float32)
        return SparseFlowResult(
            prev_pts=empty,
            next_pts=empty,
            flow_vectors=empty,
            magnitudes=np.zeros((0,), dtype=np.float32),
            valid_count=0,
        )

    p0 = pts0[valid].reshape(-1, 2)
    p1 = pts1[valid].reshape(-1, 2)
    vecs = p1 - p0
    mags = np.linalg.norm(vecs, axis=1)

    return SparseFlowResult(
        prev_pts=p0,
        next_pts=p1,
        flow_vectors=vecs,
        magnitudes=mags,
        valid_count=len(mags),
    )


def extract_sparse_summary_stats(
    lk_res: SparseFlowResult,
    clip_min: float = 0.15,
    clip_max: float = 65.0,
) -> Dict[str, float]:
    """
    Compute robust summary statistics from Sparse Lucas-Kanade optical flow vectors
    for direct comparison against Dense Farnebäck features.
    """
    if lk_res.valid_count == 0:
        return {
            "lk_mean_mag": 0.0,
            "lk_median_mag": 0.0,
            "lk_iqr_mag": 0.0,
            "lk_p75_mag": 0.0,
            "lk_p90_mag": 0.0,
            "lk_mean_vy": 0.0,
            "lk_median_vy": 0.0,
            "lk_abs_vx": 0.0,
            "lk_valid_ratio": 0.0,
        }

    valid_mags = lk_res.magnitudes[
        (lk_res.magnitudes >= clip_min) & (lk_res.magnitudes <= clip_max)
    ]
    if len(valid_mags) == 0:
        valid_mags = lk_res.magnitudes

    vy = lk_res.flow_vectors[:, 1]
    vx = lk_res.flow_vectors[:, 0]
    q25, q50, q75, q90 = np.percentile(valid_mags, [25, 50, 75, 90])

    return {
        "lk_mean_mag": float(np.mean(valid_mags)),
        "lk_median_mag": float(q50),
        "lk_iqr_mag": float(q75 - q25),
        "lk_p75_mag": float(q75),
        "lk_p90_mag": float(q90),
        "lk_mean_vy": float(np.mean(vy)),
        "lk_median_vy": float(np.median(vy)),
        "lk_abs_vx": float(np.mean(np.abs(vx))),
        "lk_valid_ratio": float(lk_res.valid_count / 200.0),
    }
