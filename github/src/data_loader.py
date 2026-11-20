"""
data_loader.py
==============
Handles video frame loading, ground-truth speed label alignment, contiguous
temporal splitting (preventing temporal data leakage across adjacent frames),
and a physics-based pinhole-camera dashcam sequence generator for reproducible
end-to-end evaluation when raw comma.ai video files are not locally mounted.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import yaml


@dataclass
class DashcamDataset:
    """Container for aligned video frames, ground-truth speeds, and metadata."""
    frames: np.ndarray          # Shape: (N, H, W, 3) uint8 BGR frames
    speeds: np.ndarray          # Shape: (N,) float64 speed in m/s
    timestamps: np.ndarray      # Shape: (N,) float64 time in seconds
    regimes: np.ndarray         # Shape: (N,) str driving regime label
    lighting: np.ndarray        # Shape: (N,) str illumination condition
    pitch_jitter: np.ndarray    # Shape: (N,) float64 camera pitch perturbation (px)
    fps: float


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_velocity_profile(
    num_frames: int = 1200,
    fps: float = 20.0,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthesize a realistic, continuous, piecewise-smooth ego-vehicle velocity
    trajectory v(t) in m/s (0 to 30 m/s, ~0 to 67 mph) covering urban stop-and-go,
    arterial cruising, highway transit, and deceleration phases.

    Returns
    -------
    timestamps : np.ndarray of shape (N,)
    speeds : np.ndarray of shape (N,)
    regimes : np.ndarray of shape (N,)
    lighting : np.ndarray of shape (N,)
    """
    rng = np.random.default_rng(seed)
    t = np.arange(num_frames, dtype=np.float64) / fps

    # Construct realistic acceleration profile a(t) in m/s^2 bounded in [-2.2, 2.2]
    # Ensure Train, Val, and Test splits all span urban, arterial, and highway speeds
    raw_target = (
        15.5
        + 9.5 * np.sin(2.0 * np.pi * t / 19.0)
        + 4.2 * np.cos(2.0 * np.pi * t / 11.5)
        + 2.0 * np.sin(2.0 * np.pi * t / 6.5)
    )
    raw_target[: int(0.04 * num_frames)] = 1.2  # Initial low-speed urban start

    # Smooth via Hanning low-pass filter to respect physical vehicle inertia
    kernel_len = 31
    kernel = np.hanning(kernel_len)
    kernel /= kernel.sum()
    speeds = np.convolve(
        np.pad(raw_target, (kernel_len // 2, kernel_len // 2), mode="edge"),
        kernel,
        mode="valid",
    )[:num_frames]
    speeds = np.clip(speeds, 0.5, 30.0)

    # Assign semantic driving regime per frame
    regimes = np.where(
        speeds < 8.0,
        "low_speed_urban",
        np.where(speeds < 19.0, "medium_arterial", "high_speed_highway"),
    )

    # Assign illumination condition (normal vs shadow/glare)
    lighting = np.full(num_frames, "normal_daylight", dtype="<U20")
    shadow_idx = (
        ((t >= 11.0) & (t <= 16.5))
        | ((t >= 31.0) & (t <= 36.0))
        | ((t >= 49.0) & (t <= 54.0))
    )
    lighting[shadow_idx] = "shadow_or_glare"

    return t, speeds, regimes, lighting


def generate_synthetic_dashcam_dataset(
    width: int = 640,
    height: int = 480,
    num_frames: int = 1200,
    fps: float = 20.0,
    seed: int = 42,
) -> DashcamDataset:
    """
    Render a physically grounded synthetic dashcam sequence using a pinhole camera
    model over a textured 3D ground plane with sky clutter, static vehicle hood,
    illumination changes, and camera pitch vibration.

    Pinhole Ground-Plane Projection:
    For camera height H_c and focal length f, a ground point at image row y > y_0
    corresponds to longitudinal distance Z(y) = (f * H_c) / (y - y_0).
    Under forward ego-motion V(t), the world coordinate Z decreases at rate V(t),
    inducing optical flow dy/dt = ((y - y_0)^2 / (f * H_c)) * V(t).
    """
    rng = np.random.default_rng(seed)
    timestamps, speeds, regimes, lighting = generate_velocity_profile(
        num_frames=num_frames, fps=fps, seed=seed
    )

    dt = 1.0 / fps
    # Cumulative longitudinal distance traveled s(t) = \int_0^t v(\tau) d\tau
    cum_dist = np.cumsum(speeds * dt)

    # Pre-generate high-resolution tileable 2D world road texture map (longitudinal Z vs lateral X)
    tex_h, tex_w = 2048, 512
    # Multi-scale spatial frequencies so both Farneback and Shi-Tomasi corners lock onto physical road texture
    base_noise = rng.integers(75, 145, size=(tex_h, tex_w), dtype=np.uint8)
    fine_tex = cv2.GaussianBlur(base_noise, (3, 3), 0.8).astype(np.float32)
    med_tex = cv2.GaussianBlur(
        rng.integers(40, 180, size=(tex_h, tex_w), dtype=np.uint8), (9, 9), 2.2
    ).astype(np.float32)
    road_tex = np.clip(0.55 * fine_tex + 0.45 * med_tex, 30, 220)

    # Add high-contrast dashed lane markers and road edge rumble strips in world coordinates
    # Center-left and center-right lane boundaries
    for z_start in range(0, tex_h, 64):
        road_tex[z_start : z_start + 34, int(0.32 * tex_w) : int(0.34 * tex_w)] = 235.0
        road_tex[z_start : z_start + 34, int(0.66 * tex_w) : int(0.68 * tex_w)] = 235.0
    # Solid shoulder lines and roadside gravel texture
    road_tex[:, int(0.12 * tex_w) : int(0.135 * tex_w)] = 220.0
    road_tex[:, int(0.865 * tex_w) : int(0.88 * tex_w)] = 220.0

    # Scatter distinct asphalt patches / pebbles for reliable Shi-Tomasi feature tracking
    num_patches = 1800
    py_idx = rng.integers(0, tex_h - 4, size=num_patches)
    px_idx = rng.integers(int(0.14 * tex_w), int(0.86 * tex_w), size=num_patches)
    patch_vals = rng.choice([35.0, 210.0], size=num_patches)
    for r_i, c_i, val in zip(py_idx, px_idx, patch_vals):
        road_tex[r_i : r_i + 3, c_i : c_i + 3] = val

    # Pre-generate drifting cloud/sky texture (non-ego motion noise above horizon)
    sky_tex = cv2.GaussianBlur(
        rng.integers(140, 245, size=(height // 2, width * 2), dtype=np.uint8),
        (21, 21),
        6.0,
    )

    y_horizon_base = int(0.38 * height)
    y_hood_start = int(0.86 * height)
    f_hc = 420.0  # Effective focal-height product controlling perspective scale

    # Camera pitch jitter (simulated road bumpiness proportional to speed + road roughness)
    pitch_jitter = rng.normal(0.0, 1.35, size=num_frames) * (
        0.45 + 0.55 * (speeds / 25.0)
    )

    frames = np.empty((num_frames, height, width, 3), dtype=np.uint8)

    # Precompute pixel grid for vectorized perspective warping
    x_Screen = np.arange(width, dtype=np.float32)[None, :] - (width / 2.0)

    for i in range(num_frames):
        y_h = y_horizon_base + pitch_jitter[i]
        frame_gray = np.zeros((height, width), dtype=np.uint8)

        # 1. Sky region (0 .. y_horizon_base): fast non-ego cloud/tree canopy motion
        cloud_shift = int((i * 11 + int(18 * np.sin(i * 0.17))) % width)
        frame_gray[:y_horizon_base, :] = sky_tex[
            :y_horizon_base, cloud_shift : cloud_shift + width
        ]

        # 2. Road surface region (y_horizon_base .. y_hood_start): true perspective projection
        y_rows = np.arange(y_horizon_base, y_hood_start, dtype=np.float32)[:, None]
        dy = np.maximum(y_rows - y_h, 4.0)

        # World longitudinal coordinate Z(y) = f_hc / dy; shifts by cumulative distance * tex_scale
        # At 20 fps, tex_scale_z = 3.2 yields 1..22 px/frame optical flow across 0..30 m/s
        tex_scale_z = 3.2
        z_world = (f_hc * 18.0 / dy) - cum_dist[i] * tex_scale_z
        # World lateral coordinate X(x, y) = x_Screen * Z(y) / f
        x_world = (x_Screen / dy) * 155.0 + (tex_w / 2.0)

        map_x = np.clip(x_world, 0, tex_w - 1).astype(np.float32)
        map_y = np.broadcast_to(np.mod(z_world, tex_h), map_x.shape).astype(np.float32)

        road_patch = cv2.remap(
            road_tex,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_WRAP,
        )
        frame_gray[y_horizon_base:y_hood_start, :] = road_patch.astype(np.uint8)

        # 3. Static ego-vehicle hood region (y_hood_start .. height) with moving glare artifact
        hood = np.full((height - y_hood_start, width), 42, dtype=np.uint8)
        glare_x = int((width // 2) + 140 * np.sin(i * 0.09))
        cv2.circle(hood, (glare_x, 25), 38, 135, -1)
        hood = cv2.GaussianBlur(hood, (15, 15), 4.0)
        frame_gray[y_hood_start:, :] = hood

        # 4. Illumination variations (tree shadows / overpass darkening / sensor noise)
        if lighting[i] == "shadow_or_glare":
            shadow_band = int((i * 14) % (y_hood_start - y_horizon_base)) + y_horizon_base
            b_end = min(shadow_band + 55, y_hood_start)
            frame_gray[shadow_band:b_end, :] = (
                frame_gray[shadow_band:b_end, :].astype(np.float32) * 0.62
            ).astype(np.uint8)

        sensor_noise = rng.normal(0.0, 2.2, size=(height, width)).astype(np.float32)
        frame_gray = np.clip(frame_gray.astype(np.float32) + sensor_noise, 0, 255).astype(
            np.uint8
        )

        frames[i] = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

    return DashcamDataset(
        frames=frames,
        speeds=speeds,
        timestamps=timestamps,
        regimes=regimes,
        lighting=lighting,
        pitch_jitter=pitch_jitter,
        fps=fps,
    )


def load_or_generate_dataset(config: dict, root_dir: str = ".") -> DashcamDataset:
    """
    Load real video + speed labels if available at config['data']['video_path'],
    otherwise synthesize the physics-grounded dashcam sequence.
    """
    video_path = os.path.join(root_dir, config["data"]["video_path"])
    labels_path = os.path.join(root_dir, config["data"]["labels_path"])
    fps = float(config["project"]["fps"])

    if os.path.exists(video_path) and os.path.exists(labels_path):
        speeds = np.loadtxt(labels_path, dtype=np.float64)
        cap = cv2.VideoCapture(video_path)
        frames_list: List[np.ndarray] = []
        target_w = int(config["data"]["frame_width"])
        target_h = int(config["data"]["frame_height"])
        while cap.isOpened() and len(frames_list) < len(speeds):
            ret, frame = cap.read()
            if not ret:
                break
            if frame.shape[1] != target_w or frame.shape[0] != target_h:
                frame = cv2.resize(frame, (target_w, target_h))
            frames_list.append(frame)
        cap.release()
        n = min(len(frames_list), len(speeds))
        frames = np.stack(frames_list[:n], axis=0)
        speeds = speeds[:n]
        timestamps = np.arange(n, dtype=np.float64) / fps
        regimes = np.where(
            speeds < 8.0,
            "low_speed_urban",
            np.where(speeds < 19.0, "medium_arterial", "high_speed_highway"),
        )
        lighting = np.full(n, "normal_daylight", dtype="<U20")
        pitch_jitter = np.zeros(n, dtype=np.float64)
        return DashcamDataset(
            frames=frames,
            speeds=speeds,
            timestamps=timestamps,
            regimes=regimes,
            lighting=lighting,
            pitch_jitter=pitch_jitter,
            fps=fps,
        )

    return generate_synthetic_dashcam_dataset(
        width=int(config["data"]["frame_width"]),
        height=int(config["data"]["frame_height"]),
        num_frames=int(config["data"]["num_synthetic_frames"]),
        fps=fps,
        seed=int(config["project"]["seed"]),
    )


def contiguous_temporal_split(
    X: pd.DataFrame,
    y: np.ndarray,
    timestamps: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    embargo_frames: int = 5,
) -> Dict[str, Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Perform strict contiguous chronological splitting into Train, Validation, and Test sets
    with a temporal embargo gap between splits to prevent autocorrelation leakage from
    rolling window features.

    Note: Random K-fold splitting on video frames leaks adjacent frames (|t_i - t_j| = 50ms)
    across train and test sets, artificially inflating R^2. Contiguous splitting mirrors
    causal deployment on unseen driving sequences.
    """
    n = len(X)
    n_train = int(n * train_ratio)
    n_val = int(n * (train_ratio + val_ratio))

    idx = np.arange(n)
    train_idx = idx[: n_train - embargo_frames]
    val_idx = idx[n_train : n_val - embargo_frames]
    test_idx = idx[n_val:]

    return {
        "train": (X.iloc[train_idx].copy(), y[train_idx], timestamps[train_idx], train_idx),
        "val": (X.iloc[val_idx].copy(), y[val_idx], timestamps[val_idx], val_idx),
        "test": (X.iloc[test_idx].copy(), y[test_idx], timestamps[test_idx], test_idx),
    }
