"""
evaluate.py
===========
Comprehensive evaluation suite for Vehicle Speed Estimation:
1. Metrics: MSE (official comma.ai metric), RMSE, MAE, R^2, P95 error.
2. Full model comparison table (Raw vs. SMA vs. EMA vs. 1D Kalman Filter).
3. Systematic 9-stage feature engineering & post-processing ablation study.
4. Stratified error analysis across driving regimes, illumination conditions,
   camera pitch jitter, and longitudinal acceleration phases.
5. Empirical demonstration of temporal data leakage under random K-fold CV vs.
   strict contiguous chronological splitting.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from src.data_loader import DashcamDataset
from src.feature_engineering import get_feature_subsets
from src.models import (
    TrainedModelArtifact,
    train_all_models,
    train_ols,
    train_ridge,
)
from src.temporal_filter import (
    causal_exponential_moving_average,
    causal_moving_average,
    kalman_kinematic_filter,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MSE, RMSE, MAE, R^2, and 95th percentile absolute error."""
    errors = y_true - y_pred
    abs_err = np.abs(errors)
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    p95 = float(np.percentile(abs_err, 95))
    bias = float(np.mean(errors))
    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "P95_AbsErr": p95,
        "Mean_Residual": bias,
    }


def apply_temporal_filters(
    raw_preds: np.ndarray,
    tf_cfg: dict,
) -> Dict[str, np.ndarray]:
    """Apply Raw, Causal SMA, Causal EMA, and 1D Kinematic Kalman filters."""
    win_sz = int(tf_cfg["moving_average"]["window_size"])
    ema_alpha = float(tf_cfg["moving_average"]["ema_alpha"])
    k_cfg = tf_cfg["kalman"]

    sma_preds = causal_moving_average(raw_preds, window_size=win_sz)
    ema_preds = causal_exponential_moving_average(raw_preds, alpha=ema_alpha)
    kf_preds, _ = kalman_kinematic_filter(
        raw_preds,
        dt=float(k_cfg["dt"]),
        process_noise_pos=float(k_cfg["process_noise_pos"]),
        process_noise_vel=float(k_cfg["process_noise_vel"]),
        measurement_noise=float(k_cfg["measurement_noise"]),
    )
    return {
        "Raw": np.clip(raw_preds, 0.0, None),
        "Moving_Average": sma_preds,
        "EMA": ema_preds,
        "Kalman_Filter": kf_preds,
    }


def evaluate_all_models(
    artifacts: Dict[str, TrainedModelArtifact],
    splits: dict,
    config: dict,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
    """
    Evaluate all trained models on Train, Validation, and Test sets, both raw
    and with temporal post-processing filters.
    """
    tf_cfg = config["temporal_filter"]
    rows = []
    test_predictions: Dict[str, Dict[str, np.ndarray]] = {}

    X_train, y_train, _, _ = splits["train"]
    X_val, y_val, _, _ = splits["val"]
    X_test, y_test, _, _ = splits["test"]

    for key, art in artifacts.items():
        cols = art.feature_names
        train_pred = np.clip(art.pipeline.predict(X_train[cols]), 0.0, None)
        val_pred = np.clip(art.pipeline.predict(X_val[cols]), 0.0, None)
        test_raw = np.clip(art.pipeline.predict(X_test[cols]), 0.0, None)

        train_m = compute_metrics(y_train, train_pred)
        val_m = compute_metrics(y_val, val_pred)

        filtered_dict = apply_temporal_filters(test_raw, tf_cfg)
        test_predictions[key] = filtered_dict

        for f_name, f_pred in filtered_dict.items():
            test_m = compute_metrics(y_test, f_pred)
            rows.append(
                {
                    "model_key": key,
                    "model_name": art.name,
                    "post_filter": f_name,
                    "num_features": len(cols),
                    "best_hyperparam": art.best_hyperparam,
                    "train_MSE": train_m["MSE"],
                    "val_MSE": val_m["MSE"],
                    "test_MSE": test_m["MSE"],
                    "test_RMSE": test_m["RMSE"],
                    "test_MAE": test_m["MAE"],
                    "test_R2": test_m["R2"],
                    "test_P95": test_m["P95_AbsErr"],
                }
            )

    return pd.DataFrame(rows), test_predictions


def run_ablation_study(
    features_df: pd.DataFrame,
    splits: dict,
    config: dict,
) -> pd.DataFrame:
    """
    Run systematic ablation across the 7 feature engineering stages + 2 temporal
    filtering stages using Ridge Regression (and comparing against unregularized OLS).
    """
    subsets = get_feature_subsets(features_df)
    X_train, y_train, _, _ = splits["train"]
    X_val, y_val, _, _ = splits["val"]
    X_test, y_test, _, _ = splits["test"]

    alphas = config["models"]["ridge"]["alphas"]
    tf_cfg = config["temporal_filter"]

    ablation_rows = []
    last_ridge_test_raw = None

    for stage_name, cols in subsets.items():
        ols_art = train_ols(X_train[cols], y_train, model_name=f"OLS_{stage_name}")
        ridge_art = train_ridge(
            X_train[cols], y_train, X_val[cols], y_val, alphas=alphas, model_name=f"Ridge_{stage_name}"
        )

        ols_pred = np.clip(ols_art.pipeline.predict(X_test[cols]), 0.0, None)
        ridge_pred = np.clip(ridge_art.pipeline.predict(X_test[cols]), 0.0, None)
        last_ridge_test_raw = ridge_pred

        ols_m = compute_metrics(y_test, ols_pred)
        ridge_m = compute_metrics(y_test, ridge_pred)
        val_m = compute_metrics(
            y_val, np.clip(ridge_art.pipeline.predict(X_val[cols]), 0.0, None)
        )

        ablation_rows.append(
            {
                "Stage": stage_name,
                "Num_Features": len(cols),
                "Condition_Number": ridge_art.condition_number,
                "OLS_Test_MSE": ols_m["MSE"],
                "Ridge_Val_MSE": val_m["MSE"],
                "Ridge_Test_MSE": ridge_m["MSE"],
                "Ridge_Test_RMSE": ridge_m["RMSE"],
                "Ridge_Test_MAE": ridge_m["MAE"],
                "Ridge_Test_R2": ridge_m["R2"],
            }
        )

    # Append Stage 8 (Causal Moving Average) and Stage 9 (1D Kinematic Kalman Filter)
    if last_ridge_test_raw is not None:
        filtered = apply_temporal_filters(last_ridge_test_raw, tf_cfg)
        sma_m = compute_metrics(y_test, filtered["Moving_Average"])
        kf_m = compute_metrics(y_test, filtered["Kalman_Filter"])

        ablation_rows.append(
            {
                "Stage": "8_Full_Features_+_MovingAverage",
                "Num_Features": ablation_rows[-1]["Num_Features"],
                "Condition_Number": ablation_rows[-1]["Condition_Number"],
                "OLS_Test_MSE": ablation_rows[-1]["OLS_Test_MSE"],
                "Ridge_Val_MSE": ablation_rows[-1]["Ridge_Val_MSE"],
                "Ridge_Test_MSE": sma_m["MSE"],
                "Ridge_Test_RMSE": sma_m["RMSE"],
                "Ridge_Test_MAE": sma_m["MAE"],
                "Ridge_Test_R2": sma_m["R2"],
            }
        )
        ablation_rows.append(
            {
                "Stage": "9_Full_Features_+_KalmanFilter",
                "Num_Features": ablation_rows[-1]["Num_Features"],
                "Condition_Number": ablation_rows[-1]["Condition_Number"],
                "OLS_Test_MSE": ablation_rows[-1]["OLS_Test_MSE"],
                "Ridge_Val_MSE": ablation_rows[-1]["Ridge_Val_MSE"],
                "Ridge_Test_MSE": kf_m["MSE"],
                "Ridge_Test_RMSE": kf_m["RMSE"],
                "Ridge_Test_MAE": kf_m["MAE"],
                "Ridge_Test_R2": kf_m["R2"],
            }
        )

    return pd.DataFrame(ablation_rows)


def run_error_analysis_slices(
    dataset: DashcamDataset,
    splits: dict,
    y_pred_raw: np.ndarray,
    y_pred_kf: np.ndarray,
) -> Dict[str, pd.DataFrame]:
    """
    Slice test-set prediction errors across four physical dimensions:
    1. Speed Regime (Low Urban <8 m/s, Medium Arterial 8-19 m/s, High Highway >19 m/s).
    2. Illumination Condition (Normal Daylight vs. Shadow/Glare).
    3. Camera Pitch Jitter (Low Bumpiness vs. High Pitch Vibration).
    4. Kinematic Acceleration State (Steady Cruise vs. Rapid Accel/Decel).
    """
    _, y_test, _, test_idx = splits["test"]
    dt = 1.0 / dataset.fps
    full_accel = np.gradient(dataset.speeds, dt)

    test_regimes = dataset.regimes[test_idx]
    test_lighting = dataset.lighting[test_idx]
    test_pitch = np.abs(dataset.pitch_jitter[test_idx])
    test_accel = np.abs(full_accel[test_idx])

    pitch_med = float(np.median(test_pitch))
    pitch_slice = np.where(test_pitch <= pitch_med, "low_pitch_jitter", "high_pitch_jitter")
    accel_slice = np.where(test_accel < 0.8, "steady_cruise", "rapid_accel_decel")

    df_err = pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred_raw": y_pred_raw,
            "y_pred_kf": y_pred_kf,
            "sq_err_raw": (y_test - y_pred_raw) ** 2,
            "sq_err_kf": (y_test - y_pred_kf) ** 2,
            "abs_err_raw": np.abs(y_test - y_pred_raw),
            "abs_err_kf": np.abs(y_test - y_pred_kf),
            "residual_kf": y_pred_kf - y_test,
            "regime": test_regimes,
            "lighting": test_lighting,
            "pitch_condition": pitch_slice,
            "kinematic_state": accel_slice,
        }
    )

    slices_out: Dict[str, pd.DataFrame] = {}
    for col in ["regime", "lighting", "pitch_condition", "kinematic_state"]:
        grouped = (
            df_err.groupby(col, as_index=False)
            .agg(
                frames=("y_true", "count"),
                mean_true_speed=("y_true", "mean"),
                raw_MSE=("sq_err_raw", "mean"),
                kf_MSE=("sq_err_kf", "mean"),
                raw_MAE=("abs_err_raw", "mean"),
                kf_MAE=("abs_err_kf", "mean"),
                mean_signed_bias=("residual_kf", "mean"),
            )
        )
        grouped["raw_RMSE"] = np.sqrt(grouped["raw_MSE"])
        grouped["kf_RMSE"] = np.sqrt(grouped["kf_MSE"])
        grouped["mse_reduction_pct"] = (
            100.0 * (grouped["raw_MSE"] - grouped["kf_MSE"]) / np.maximum(grouped["raw_MSE"], 1e-8)
        )
        slices_out[col] = grouped

    return slices_out


def evaluate_leakage_experiment(
    features_df: pd.DataFrame,
    speeds: np.ndarray,
    contiguous_test_mse: float,
    contiguous_test_r2: float,
) -> pd.DataFrame:
    """
    Quantify the severity of temporal data leakage when using random 5-fold CV
    on video frames compared against strict contiguous chronological splitting.
    """
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rand_mses = []
    rand_r2s = []

    for train_i, test_i in kf.split(features_df):
        art = train_ridge(
            features_df.iloc[train_i],
            speeds[train_i],
            features_df.iloc[test_i],
            speeds[test_i],
            alphas=[1.0, 10.0, 50.0],
        )
        preds = np.clip(art.pipeline.predict(features_df.iloc[test_i]), 0.0, None)
        rand_mses.append(mean_squared_error(speeds[test_i], preds))
        rand_r2s.append(r2_score(speeds[test_i], preds))

    return pd.DataFrame(
        [
            {
                "Split_Protocol": "Random 5-Fold Shuffle (Temporal Leakage)",
                "Eval_MSE": float(np.mean(rand_mses)),
                "Eval_RMSE": float(np.sqrt(np.mean(rand_mses))),
                "Eval_R2": float(np.mean(rand_r2s)),
            },
            {
                "Split_Protocol": "Contiguous Temporal Split (Leakage-Free)",
                "Eval_MSE": float(contiguous_test_mse),
                "Eval_RMSE": float(np.sqrt(contiguous_test_mse)),
                "Eval_R2": float(contiguous_test_r2),
            },
        ]
    )
