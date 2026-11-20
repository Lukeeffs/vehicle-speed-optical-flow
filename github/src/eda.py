"""
eda.py
======
Exploratory Data Analysis (EDA) for monocular dashcam speed estimation:
1. Summary statistics of ground-truth velocity and acceleration across splits.
2. Autocorrelation and stationarity analysis justifying contiguous splitting.
3. Spatial correlation analysis between zone-level optical flow features and
   ground-truth vehicle speed.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from src.data_loader import DashcamDataset


def compute_eda_summary(
    dataset: DashcamDataset,
    features_df: pd.DataFrame,
    splits: dict,
) -> Dict[str, pd.DataFrame]:
    """
    Generate tabular EDA summaries:
    - Split-level speed distribution statistics (mean, std, min, median, IQR, max).
    - Top feature Pearson and Spearman correlations with ground-truth speed.
    - Regime distribution across frames.
    """
    dt = 1.0 / dataset.fps
    accel = np.gradient(dataset.speeds, dt)

    split_rows = []
    for split_name, (X_s, y_s, t_s, idx_s) in splits.items():
        q25, q50, q75 = np.percentile(y_s, [25, 50, 75])
        split_rows.append(
            {
                "split": split_name,
                "num_frames": len(y_s),
                "duration_sec": len(y_s) * dt,
                "mean_speed_ms": float(np.mean(y_s)),
                "std_speed_ms": float(np.std(y_s)),
                "min_speed_ms": float(np.min(y_s)),
                "median_speed_ms": float(q50),
                "iqr_speed_ms": float(q75 - q25),
                "max_speed_ms": float(np.max(y_s)),
                "mean_abs_accel_ms2": float(np.mean(np.abs(accel[idx_s]))),
            }
        )
    split_summary_df = pd.DataFrame(split_rows)

    # Feature correlations with ground-truth speed on training split
    X_train, y_train, _, _ = splits["train"]
    corr_rows = []
    y_series = pd.Series(y_train, index=X_train.index)
    for col in X_train.columns:
        pearson_r = float(X_train[col].corr(y_series, method="pearson"))
        spearman_r = float(X_train[col].corr(y_series, method="spearman"))
        corr_rows.append(
            {
                "feature": col,
                "pearson_r": pearson_r,
                "abs_pearson_r": abs(pearson_r),
                "spearman_r": spearman_r,
            }
        )
    corr_df = (
        pd.DataFrame(corr_rows)
        .sort_values("abs_pearson_r", ascending=False)
        .reset_index(drop=True)
    )

    # Regime summary
    regime_df = (
        pd.DataFrame(
            {
                "regime": dataset.regimes,
                "lighting": dataset.lighting,
                "speed": dataset.speeds,
                "abs_pitch_jitter": np.abs(dataset.pitch_jitter),
            }
        )
        .groupby(["regime", "lighting"], as_index=False)
        .agg(
            frame_count=("speed", "count"),
            mean_speed=("speed", "mean"),
            std_speed=("speed", "std"),
            mean_pitch_px=("abs_pitch_jitter", "mean"),
        )
    )

    return {
        "split_summary": split_summary_df,
        "feature_correlations": corr_df,
        "regime_summary": regime_df,
    }
