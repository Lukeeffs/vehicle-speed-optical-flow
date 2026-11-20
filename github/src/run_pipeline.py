"""
run_pipeline.py
===============
Master execution script for the CS229 Vehicle Speed Estimation pipeline.
Executes data loading/synthesis, preprocessing, Dense Farnebäck and Sparse
Lucas-Kanade optical flow extraction, multi-zone perspective feature engineering,
model training, 9-stage ablation study, error analysis slicing, and 300-DPI
figure generation.
"""

from __future__ import annotations

import json
import os
import sys

# Ensure project root is on sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_loader import (
    contiguous_temporal_split,
    load_config,
    load_or_generate_dataset,
)
from src.eda import compute_eda_summary
from src.evaluate import (
    evaluate_all_models,
    evaluate_leakage_experiment,
    run_ablation_study,
    run_error_analysis_slices,
)
from src.feature_engineering import extract_dataset_features
from src.models import extract_linear_coefficients, train_all_models
from src.visualize import (
    plot_fig1_roi_and_optical_flow,
    plot_fig2_eda_and_correlations,
    plot_fig3_model_comparison_and_trajectories,
    plot_fig4_ablation_and_regularization,
    plot_fig5_error_analysis_and_residuals,
)


def main() -> None:
    cfg_path = os.path.join(ROOT_DIR, "configs", "config.yaml")
    cfg = load_config(cfg_path)

    print("[1/6] Loading / synthesizing pinhole-camera dashcam dataset...")
    ds = load_or_generate_dataset(cfg, ROOT_DIR)

    print("[2/6] Extracting Dense Farnebäck & Sparse Lucas-Kanade optical flow features...")
    df_feat = extract_dataset_features(ds.frames, cfg)
    splits = contiguous_temporal_split(
        df_feat,
        ds.speeds,
        ds.timestamps,
        train_ratio=float(cfg["data"]["train_ratio"]),
        val_ratio=float(cfg["data"]["val_ratio"]),
    )

    print("[3/6] Computing Exploratory Data Analysis (EDA) summaries...")
    eda_res = compute_eda_summary(ds, df_feat, splits)

    print("[4/6] Training baseline, regularized linear, and non-linear regression models...")
    X_tr, y_tr, _, _ = splits["train"]
    X_va, y_va, _, _ = splits["val"]
    artifacts = train_all_models(X_tr, y_tr, X_va, y_va, cfg)

    print("[5/6] Running model evaluation, 9-stage ablation study, and error slicing...")
    eval_df, test_preds = evaluate_all_models(artifacts, splits, cfg)
    ablation_df = run_ablation_study(df_feat, splits, cfg)

    ridge_raw = test_preds["Ridge_L2"]["Raw"]
    ridge_kf = test_preds["Ridge_L2"]["Kalman_Filter"]
    err_slices = run_error_analysis_slices(ds, splits, ridge_raw, ridge_kf)

    ridge_raw_row = eval_df[
        (eval_df["model_key"] == "Ridge_L2") & (eval_df["post_filter"] == "Raw")
    ].iloc[0]
    leak_df = evaluate_leakage_experiment(
        df_feat, ds.speeds, float(ridge_raw_row["test_MSE"]), float(ridge_raw_row["test_R2"])
    )

    print("[6/6] Rendering 300-DPI publication figures into figures/...")
    fig_dir = os.path.join(ROOT_DIR, "figures")
    plot_fig1_roi_and_optical_flow(ds, cfg, fig_dir)
    plot_fig2_eda_and_correlations(ds, splits, eda_res, fig_dir)
    plot_fig3_model_comparison_and_trajectories(splits, eval_df, test_preds, fig_dir)
    plot_fig4_ablation_and_regularization(
        ablation_df, artifacts["Ridge_L2"], artifacts["Lasso_L1"], fig_dir
    )
    plot_fig5_error_analysis_and_residuals(splits, ridge_raw, ridge_kf, err_slices, fig_dir)

    print("\n=== SPLIT SUMMARY ===")
    print(eda_res["split_summary"].to_string(index=False))
    print("\n=== TOP 10 FEATURE CORRELATIONS ===")
    print(eda_res["feature_correlations"].head(10).to_string(index=False))
    print("\n=== MODEL BENCHMARK EVALUATION ===")
    print(
        eval_df[
            [
                "model_key",
                "post_filter",
                "num_features",
                "best_hyperparam",
                "val_MSE",
                "test_MSE",
                "test_RMSE",
                "test_MAE",
                "test_R2",
                "test_P95",
            ]
        ].to_string(index=False)
    )
    print("\n=== 9-STAGE ABLATION STUDY ===")
    print(ablation_df.to_string(index=False))
    print("\n=== ERROR SLICES (SPEED REGIME) ===")
    print(err_slices["regime"].to_string(index=False))
    print("\n=== ERROR SLICES (LIGHTING) ===")
    print(err_slices["lighting"].to_string(index=False))
    print("\n=== ERROR SLICES (PITCH JITTER) ===")
    print(err_slices["pitch_condition"].to_string(index=False))
    print("\n=== ERROR SLICES (KINEMATIC ACCEL) ===")
    print(err_slices["kinematic_state"].to_string(index=False))
    print("\n=== TEMPORAL LEAKAGE EXPERIMENT ===")
    print(leak_df.to_string(index=False))
    print("\n=== TOP 10 RIDGE COEFFICIENTS ===")
    print(extract_linear_coefficients(artifacts["Ridge_L2"]).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
