"""
feature_engineering.py
======================
Extracts physics-informed optical flow feature representations for vehicle speed
estimation:
1. Full-frame baseline statistics (unmasked).
2. Perspective ROI global statistics (mean, median, IQR, percentiles, directional).
3. Focus-of-Expansion (FOE) singularity masking & radial unit-vector projection
   v_r(x, y) = u(x, y) * cos(theta) + v(x, y) * sin(theta).
4. Multi-zone 3x3 perspective grid statistics (Near/Mid/Far x Left/Center/Right)
   capturing the quadratic inverse-depth scaling dy/dt ~ (y - y_0)^2 * V_ego.
5. Causal multi-scale temporal lag and rolling-window kinematic features.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.optical_flow import (
    compute_dense_farneback_flow,
    compute_sparse_lk_flow,
    extract_sparse_summary_stats,
)
from src.preprocessing import (
    build_trapezoid_road_mask,
    get_roi_bounds,
    preprocess_frame,
)


def _compute_zone_stats(
    mag: np.ndarray,
    vy: np.ndarray,
    vx: np.ndarray,
    vr: np.ndarray,
    valid_mask: np.ndarray,
    prefix: str,
) -> Dict[str, float]:
    """Compute robust order statistics and directional moments within a spatial mask."""
    if not np.any(valid_mask):
        return {
            f"{prefix}_mean_mag": 0.0,
            f"{prefix}_median_mag": 0.0,
            f"{prefix}_iqr_mag": 0.0,
            f"{prefix}_p75_mag": 0.0,
            f"{prefix}_p90_mag": 0.0,
            f"{prefix}_mean_vy": 0.0,
            f"{prefix}_mean_vr": 0.0,
            f"{prefix}_abs_vx": 0.0,
        }

    m = mag[valid_mask]
    v_y = vy[valid_mask]
    v_x = vx[valid_mask]
    v_r = vr[valid_mask]

    q25, q50, q75, q90 = np.percentile(m, [25, 50, 75, 90])
    return {
        f"{prefix}_mean_mag": float(np.mean(m)),
        f"{prefix}_median_mag": float(q50),
        f"{prefix}_iqr_mag": float(q75 - q25),
        f"{prefix}_p75_mag": float(q75),
        f"{prefix}_p90_mag": float(q90),
        f"{prefix}_mean_vy": float(np.mean(v_y)),
        f"{prefix}_mean_vr": float(np.mean(v_r)),
        f"{prefix}_abs_vx": float(np.mean(np.abs(v_x))),
    }


def extract_pair_features(
    prev_bgr: np.ndarray,
    curr_bgr: np.ndarray,
    config: dict,
) -> Dict[str, float]:
    """
    Extract complete feature dictionary from a consecutive pair of BGR frames
    (t-1, t), supporting all ablation feature subsets.
    """
    preproc_cfg = config["preprocessing"]
    fb_cfg = config["optical_flow"]["farneback"]
    lk_cfg = config["optical_flow"]["lucas_kanade"]
    fe_cfg = config["feature_engineering"]

    h, w = prev_bgr.shape[:2]
    clip_min = float(fe_cfg.get("magnitude_clip_min", 0.15))
    clip_max = float(fe_cfg.get("magnitude_clip_max", 65.0))

    # 1. Full-frame preprocess (with CLAHE + Gaussian smoothing, uncropped)
    prev_full = preprocess_frame(prev_bgr, preproc_cfg, use_clahe=True, crop_roi=False)
    curr_full = preprocess_frame(curr_bgr, preproc_cfg, use_clahe=True, crop_roi=False)

    flow_full = compute_dense_farneback_flow(prev_full, curr_full, fb_cfg)
    u_full = flow_full[..., 0]
    v_full = flow_full[..., 1]
    mag_full = np.hypot(u_full, v_full)

    # Coordinate grid relative to Focus of Expansion (FOE)
    foe_x = float(fe_cfg.get("foe_x_frac", 0.50)) * w
    foe_y = float(fe_cfg.get("foe_y_frac", 0.38)) * h
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx_foe = xx - foe_x
    dy_foe = yy - foe_y
    dist_foe = np.hypot(dx_foe, dy_foe)
    inv_dist = 1.0 / np.maximum(dist_foe, 1.0)

    # Radial flow projection away from FOE: v_r = (u * dx + v * dy) / ||(dx, dy)||
    vr_full = (u_full * dx_foe + v_full * dy_foe) * inv_dist

    # Inverse-perspective planar depth-normalized vertical flow:
    # Since dy/dt ~ (y - y_0)^2 * V_ego, dividing by ((y - y_0)/H)^2 linearizes depth response
    norm_dy_sq = np.square(np.maximum(dy_foe / float(h), 0.06))
    ipm_vy_full = v_full / norm_dy_sq

    feats: Dict[str, float] = {}

    # --- Group A: Full-Frame Unmasked Baseline Statistics ---
    full_valid = (mag_full >= clip_min) & (mag_full <= clip_max)
    feats.update(_compute_zone_stats(mag_full, v_full, u_full, vr_full, full_valid, "full"))

    # --- Group B: Rectangular ROI Cropped Global Statistics ---
    y_min, y_max, x_min, x_max = get_roi_bounds(h, w, preproc_cfg)
    roi_rect_mask = np.zeros((h, w), dtype=bool)
    roi_rect_mask[y_min:y_max, x_min:x_max] = True
    roi_valid = roi_rect_mask & full_valid
    feats.update(_compute_zone_stats(mag_full, v_full, u_full, vr_full, roi_valid, "roi"))

    # --- Group C: Trapezoidal Road Mask + FOE Singularity Exclusion ---
    trap_mask = build_trapezoid_road_mask(h, w, preproc_cfg["trapezoid_vertices"]).astype(bool)
    foe_radius = float(fe_cfg.get("foe_exclusion_radius_px", 28.0))
    foe_mask = dist_foe >= foe_radius
    road_foe_valid = trap_mask & foe_mask & full_valid
    feats.update(
        _compute_zone_stats(mag_full, v_full, u_full, vr_full, road_foe_valid, "foe_clean")
    )
    if np.any(road_foe_valid):
        feats["foe_clean_ipm_vy_mean"] = float(np.mean(ipm_vy_full[road_foe_valid]))
        feats["foe_clean_ipm_vy_median"] = float(np.median(ipm_vy_full[road_foe_valid]))
        # Pitch jitter proxy: upper-ROI vertical flow near horizon
        horizon_band = trap_mask & (yy >= y_min) & (yy < y_min + 28)
        feats["horizon_pitch_proxy"] = (
            float(np.median(v_full[horizon_band])) if np.any(horizon_band) else 0.0
        )
    else:
        feats["foe_clean_ipm_vy_mean"] = 0.0
        feats["foe_clean_ipm_vy_median"] = 0.0
        feats["horizon_pitch_proxy"] = 0.0

    # --- Group D: Multi-Zone 3x3 Perspective Grid Statistics ---
    n_rows = int(fe_cfg.get("grid_rows", 3))
    n_cols = int(fe_cfg.get("grid_cols", 3))
    row_edges = np.linspace(y_min, y_max, n_rows + 1, dtype=int)
    col_edges = np.linspace(x_min, x_max, n_cols + 1, dtype=int)

    row_names = ["far", "mid", "near"] if n_rows == 3 else [f"r{r}" for r in range(n_rows)]
    col_names = ["left", "center", "right"] if n_cols == 3 else [f"c{c}" for c in range(n_cols)]

    for r_idx in range(n_rows):
        for c_idx in range(n_cols):
            r0, r1 = row_edges[r_idx], row_edges[r_idx + 1]
            c0, c1 = col_edges[c_idx], col_edges[c_idx + 1]
            zone_mask = np.zeros((h, w), dtype=bool)
            zone_mask[r0:r1, c0:c1] = True
            z_valid = zone_mask & foe_mask & full_valid
            z_name = f"zone_{row_names[r_idx]}_{col_names[c_idx]}"
            z_stats = _compute_zone_stats(mag_full, v_full, u_full, vr_full, z_valid, z_name)
            feats.update(z_stats)

    # Inter-zone perspective gradient ratios (Near-to-Mid and Mid-to-Far flow ratios)
    feats["near_to_mid_vr_ratio"] = feats["zone_near_center_mean_vr"] / (
        feats["zone_mid_center_mean_vr"] + 0.25
    )
    feats["lateral_asymmetry_vx"] = (
        feats["zone_near_right_mean_vr"] - feats["zone_near_left_mean_vr"]
    )

    # --- Group E: Sparse Pyramidal Lucas-Kanade Features (within Trapezoid Road Mask) ---
    lk_mask = (trap_mask & foe_mask).astype(np.uint8) * 255
    lk_res = compute_sparse_lk_flow(prev_full, curr_full, lk_cfg, mask=lk_mask)
    feats.update(extract_sparse_summary_stats(lk_res, clip_min=clip_min, clip_max=clip_max))

    return feats


def append_temporal_kinematic_features(
    df: pd.DataFrame,
    fe_cfg: dict,
) -> pd.DataFrame:
    """
    Append causal multi-frame autoregressive lag features, rolling window means,
    rolling standard deviations, and first-order finite differences (flow acceleration).
    Strictly causal (only uses frames <= t) so real-time inference has zero lookahead.
    """
    out = df.copy()
    core_cols = [
        "foe_clean_mean_vr",
        "foe_clean_median_mag",
        "foe_clean_ipm_vy_mean",
        "zone_near_center_mean_vr",
        "zone_mid_center_mean_vr",
        "zone_far_center_mean_vr",
        "lk_median_mag",
    ]
    core_cols = [c for c in core_cols if c in out.columns]

    lags = fe_cfg.get("temporal_lags", [1, 2, 3, 5])
    windows = fe_cfg.get("rolling_windows", [5, 10, 20])

    new_cols = {}
    for col in core_cols:
        s = out[col]
        # Causal lags
        for lag in lags:
            new_cols[f"{col}_lag{lag}"] = s.shift(lag).bfill()
        # First-order finite difference (instantaneous flow derivative)
        new_cols[f"{col}_diff1"] = s.diff(1).fillna(0.0)
        # Causal rolling statistics
        for w in windows:
            new_cols[f"{col}_roll{w}_mean"] = s.rolling(window=w, min_periods=1).mean()
            new_cols[f"{col}_roll{w}_std"] = (
                s.rolling(window=w, min_periods=1).std().fillna(0.0)
            )

    return pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)


def extract_dataset_features(
    frames: np.ndarray,
    config: dict,
) -> pd.DataFrame:
    """
    Extract frame-pair optical flow features across the entire sequence of N frames.
    Frame 0 repeats Frame 1's pair features so output length matches N.
    """
    n = len(frames)
    records: List[Dict[str, float]] = []

    for i in range(1, n):
        feat = extract_pair_features(frames[i - 1], frames[i], config)
        records.append(feat)

    # Prepend first pair for index 0 alignment
    records.insert(0, records[0].copy())
    df = pd.DataFrame(records)
    df = append_temporal_kinematic_features(df, config["feature_engineering"])
    return df


def get_feature_subsets(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Partition engineered columns into structured feature sets for systematic
    ablation experiments.
    """
    all_cols = list(df.columns)

    s1_full_global = [c for c in all_cols if c.startswith("full_")]
    s2_roi_global = [c for c in all_cols if c.startswith("roi_")]
    s3_sparse_lk = [c for c in all_cols if c.startswith("lk_") and "_lag" not in c and "_roll" not in c and "_diff" not in c]
    s4_foe_clean = [
        c
        for c in all_cols
        if c.startswith("foe_clean_")
        and "_lag" not in c
        and "_roll" not in c
        and "_diff" not in c
    ] + ["horizon_pitch_proxy"]
    s5_grid_zones = [
        c
        for c in all_cols
        if (c.startswith("zone_") or c in ("near_to_mid_vr_ratio", "lateral_asymmetry_vx"))
        and "_lag" not in c
        and "_roll" not in c
        and "_diff" not in c
    ]
    s6_spatial_combined = s2_roi_global + s4_foe_clean + s5_grid_zones
    s7_temporal_kinematic = [
        c for c in all_cols if ("_lag" in c or "_roll" in c or "_diff" in c)
    ]
    s8_full_engineered = s6_spatial_combined + s3_sparse_lk + s7_temporal_kinematic

    return {
        "1_FullFrame_Global_Dense": s1_full_global,
        "2_ROI_Cropped_Global_Dense": s2_roi_global,
        "3_ROI_Sparse_LucasKanade": s3_sparse_lk,
        "4_ROI_FOE_Radial_IPM": s4_foe_clean,
        "5_MultiZone_3x3_Grid": s4_foe_clean + s5_grid_zones,
        "6_Spatial_Hybrid_Dense_LK": s6_spatial_combined + s3_sparse_lk,
        "7_Full_Spatial_Temporal_Kinematic": s8_full_engineered,
    }
