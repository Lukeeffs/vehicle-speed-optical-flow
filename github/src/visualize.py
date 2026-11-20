"""
visualize.py
============
Generates publication-quality 300-DPI figures for the Stanford CS229 reports,
poster, and Jupyter notebook:
1. fig1_pipeline_and_roi_flow.png: Perspective ROI, 3x3 Grid, FOE exclusion,
   Dense Farnebäck quiver/HSV field, and Sparse Lucas-Kanade tracks.
2. fig2_eda_speed_and_correlations.png: Chronological velocity trajectory with
   contiguous Train/Val/Test splits, regime distributions, and feature correlations.
3. fig3_model_comparison_and_trajectories.png: Predicted vs. Ground-Truth test
   speed curves and multi-model MSE benchmark bar chart.
4. fig4_ablation_and_regularization.png: 9-stage systematic ablation curve and
   standardized Ridge/Lasso feature importance profile.
5. fig5_error_analysis_and_residuals.png: Parity plot, heteroscedastic residual
   scatter, and stratified error slicing across speed, lighting, and pitch jitter.
"""

from __future__ import annotations

import os
from typing import Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.data_loader import DashcamDataset
from src.models import TrainedModelArtifact, extract_linear_coefficients
from src.optical_flow import compute_dense_farneback_flow, compute_sparse_lk_flow
from src.preprocessing import (
    build_trapezoid_road_mask,
    get_roi_bounds,
    preprocess_frame,
)

# Set publication style
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
)


def plot_fig1_roi_and_optical_flow(
    dataset: DashcamDataset,
    config: dict,
    output_dir: str,
    frame_idx: int = 650,
) -> str:
    """
    Generate Figure 1: (a) Perspective ROI & 3x3 Multi-Zone Grid with FOE mask,
    (b) Dense Farnebäck optical flow field (HSV magnitude + quiver vectors),
    (c) Sparse Lucas-Kanade Shi-Tomasi corner tracks inside road trapezoid.
    """
    os.makedirs(output_dir, exist_ok=True)
    prev_bgr = dataset.frames[frame_idx - 1]
    curr_bgr = dataset.frames[frame_idx]
    speed_val = dataset.speeds[frame_idx]

    preproc_cfg = config["preprocessing"]
    fb_cfg = config["optical_flow"]["farneback"]
    lk_cfg = config["optical_flow"]["lucas_kanade"]
    fe_cfg = config["feature_engineering"]

    h, w = curr_bgr.shape[:2]
    y_min, y_max, x_min, x_max = get_roi_bounds(h, w, preproc_cfg)
    foe_x = int(fe_cfg["foe_x_frac"] * w)
    foe_y = int(fe_cfg["foe_y_frac"] * h)
    foe_r = int(fe_cfg["foe_exclusion_radius_px"])

    prev_gray = preprocess_frame(prev_bgr, preproc_cfg, use_clahe=True, crop_roi=False)
    curr_gray = preprocess_frame(curr_bgr, preproc_cfg, use_clahe=True, crop_roi=False)
    flow = compute_dense_farneback_flow(prev_gray, curr_gray, fb_cfg)

    trap_mask = build_trapezoid_road_mask(h, w, preproc_cfg["trapezoid_vertices"])
    lk_res = compute_sparse_lk_flow(prev_gray, curr_gray, lk_cfg, mask=trap_mask * 255)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))

    # Panel (a): Perspective Geometry, 3x3 Grid & FOE Mask
    canvas_a = cv2.cvtColor(curr_bgr.copy(), cv2.COLOR_BGR2RGB)
    # Draw 3x3 grid lines inside ROI
    r_edges = np.linspace(y_min, y_max, 4, dtype=int)
    c_edges = np.linspace(x_min, x_max, 4, dtype=int)
    for r_e in r_edges:
        cv2.line(canvas_a, (x_min, r_e), (x_max, r_e), (50, 220, 255), 2)
    for c_e in c_edges:
        cv2.line(canvas_a, (c_e, y_min), (c_e, y_max), (50, 220, 255), 2)

    # Draw Trapezoid Road Mask outline
    verts = preproc_cfg["trapezoid_vertices"]
    pts = np.array(
        [
            [int(verts["top_left"][0] * w), int(verts["top_left"][1] * h)],
            [int(verts["top_right"][0] * w), int(verts["top_right"][1] * h)],
            [int(verts["bottom_right"][0] * w), int(verts["bottom_right"][1] * h)],
            [int(verts["bottom_left"][0] * w), int(verts["bottom_left"][1] * h)],
        ],
        dtype=np.int32,
    )
    cv2.polylines(canvas_a, [pts], isClosed=True, color=(0, 255, 110), thickness=2)
    # Draw FOE singularity exclusion zone
    cv2.circle(canvas_a, (foe_x, foe_y), foe_r, (255, 70, 70), 2)
    cv2.drawMarker(canvas_a, (foe_x, foe_y), (255, 50, 50), cv2.MARKER_CROSS, 14, 2)

    axes[0].imshow(canvas_a)
    axes[0].set_title(f"(a) Perspective ROI, 3x3 Grid & FOE (v = {speed_val:.1f} m/s)")
    axes[0].axis("off")

    # Panel (b): Dense Farnebäck Flow Magnitude Heatmap + Quiver Overlay
    mag = np.hypot(flow[..., 0], flow[..., 1])
    im_b = axes[1].imshow(mag, cmap="inferno", vmin=0.0, vmax=np.percentile(mag, 99))
    step = 22
    yy, xx = np.mgrid[y_min:y_max:step, x_min:x_max:step]
    u_sub = flow[y_min:y_max:step, x_min:x_max:step, 0]
    v_sub = flow[y_min:y_max:step, x_min:x_max:step, 1]
    axes[1].quiver(
        xx,
        yy,
        u_sub,
        v_sub,
        color="cyan",
        angles="xy",
        scale_units="xy",
        scale=0.35,
        width=0.0035,
        alpha=0.85,
    )
    axes[1].axhline(y_min, color="lime", linestyle="--", linewidth=1.1)
    axes[1].axhline(y_max, color="lime", linestyle="--", linewidth=1.1)
    axes[1].set_title("(b) Dense Farnebäck Flow Magnitude & Vectors")
    axes[1].axis("off")
    plt.colorbar(im_b, ax=axes[1], fraction=0.046, pad=0.03, label="Flow Mag (px/frame)")

    # Panel (c): Sparse Pyramidal Lucas-Kanade Keypoint Displacement Vectors
    canvas_c = cv2.cvtColor(curr_bgr.copy(), cv2.COLOR_BGR2RGB)
    cv2.polylines(canvas_c, [pts], isClosed=True, color=(0, 255, 110), thickness=2)
    axes[2].imshow(canvas_c)
    if lk_res.valid_count > 0:
        p0 = lk_res.prev_pts
        vecs = lk_res.flow_vectors
        axes[2].quiver(
            p0[:, 0],
            p0[:, 1],
            vecs[:, 0],
            vecs[:, 1],
            color="#ffdf00",
            angles="xy",
            scale_units="xy",
            scale=0.30,
            width=0.0045,
        )
        axes[2].scatter(p0[:, 0], p0[:, 1], s=10, c="#ff3366", label=f"Tracked Corners (K={lk_res.valid_count})")
        axes[2].legend(loc="upper right", framealpha=0.85)
    axes[2].set_title("(c) Sparse Lucas-Kanade (Shi-Tomasi) Tracks")
    axes[2].axis("off")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig1_pipeline_and_roi_flow.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out_path


def plot_fig2_eda_and_correlations(
    dataset: DashcamDataset,
    splits: dict,
    eda_tables: Dict[str, pd.DataFrame],
    output_dir: str,
) -> str:
    """
    Generate Figure 2: (a) Contiguous chronological velocity profile across
    Train / Validation / Test splits, (b) Speed density distribution by split,
    (c) Top 12 optical flow feature correlations with ground-truth speed.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2), gridspec_kw={"width_ratios": [1.45, 0.95, 1.2]})

    # Panel (a): Chronological Velocity Trajectory & Contiguous Split Boundaries
    colors = {"train": "#1f77b4", "val": "#ff7f0e", "test": "#2ca02c"}
    labels = {"train": "Train Split (70%)", "val": "Val Split (15%)", "test": "Test Split (15%)"}
    for s_name, (_, y_s, t_s, _) in splits.items():
        axes[0].plot(t_s, y_s, color=colors[s_name], linewidth=1.8, label=labels[s_name])
    axes[0].axvline(splits["val"][2][0], color="black", linestyle="--", linewidth=1.1)
    axes[0].axvline(splits["test"][2][0], color="black", linestyle="--", linewidth=1.1)
    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Ground-Truth Speed (m/s)")
    axes[0].set_title("(a) Contiguous Temporal Split Trajectory")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left")

    # Panel (b): Speed Distribution KDE/Histogram Across Splits
    for s_name, (_, y_s, _, _) in splits.items():
        sns.kdeplot(y_s, ax=axes[1], color=colors[s_name], fill=True, alpha=0.22, linewidth=1.6, label=labels[s_name])
    axes[1].set_xlabel("Vehicle Speed (m/s)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("(b) Speed Regime Distribution")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    # Panel (c): Top 12 Feature Correlations (Pearson |r|)
    corr_df = eda_tables["feature_correlations"].head(12).iloc[::-1]
    axes[2].barh(
        corr_df["feature"],
        corr_df["pearson_r"],
        color="#2b5c8f",
        edgecolor="black",
        height=0.72,
    )
    axes[2].set_xlabel("Pearson Correlation Coefficient (r)")
    axes[2].set_title("(c) Top 12 Engineered Features vs. Speed")
    axes[2].set_xlim(0.0, 1.0)
    axes[2].grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig2_eda_speed_and_correlations.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out_path


def plot_fig3_model_comparison_and_trajectories(
    splits: dict,
    eval_df: pd.DataFrame,
    test_preds: Dict[str, Dict[str, np.ndarray]],
    output_dir: str,
) -> str:
    """
    Generate Figure 3: (a) Test-set time-series speed tracking comparing
    Ground Truth, Univariate OLS, Raw Ridge, and Ridge + 1D Kalman Filter;
    (b) Grouped bar chart of Test MSE across all 6 model families.
    """
    os.makedirs(output_dir, exist_ok=True)
    _, y_test, t_test, _ = splits["test"]

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 4.4), gridspec_kw={"width_ratios": [1.35, 1.0]})

    # Panel (a): Test Split Speed Trajectory Tracking
    uni_raw = test_preds["Univariate_OLS"]["Raw"]
    ridge_raw = test_preds["Ridge_L2"]["Raw"]
    ridge_kf = test_preds["Ridge_L2"]["Kalman_Filter"]

    axes[0].plot(t_test, y_test, color="black", linewidth=2.3, label="Ground Truth Speed")
    axes[0].plot(
        t_test,
        uni_raw,
        color="#d62728",
        alpha=0.55,
        linewidth=1.1,
        linestyle=":",
        label="Univariate OLS (Raw)",
    )
    axes[0].plot(
        t_test,
        ridge_raw,
        color="#1f77b4",
        alpha=0.60,
        linewidth=1.2,
        label="Multi-Zone Ridge (Raw)",
    )
    axes[0].plot(
        t_test,
        ridge_kf,
        color="#2ca02c",
        linewidth=2.0,
        label="Multi-Zone Ridge + 1D Kalman Filter",
    )
    axes[0].set_xlabel("Test Split Time (seconds)")
    axes[0].set_ylabel("Vehicle Speed (m/s)")
    axes[0].set_title("(a) Test-Set Trajectory Tracking (Unseen Future Frames)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    # Panel (b): Grouped Bar Chart of Test MSE across Models (Log Scale for Dynamic Range)
    sub_df = eval_df[eval_df["post_filter"].isin(["Raw", "Moving_Average", "Kalman_Filter"])].copy()
    sns.barplot(
        data=sub_df,
        x="model_key",
        y="test_MSE",
        hue="post_filter",
        ax=axes[1],
        palette=["#8c96c6", "#4eb3d3", "#0868ac"],
        edgecolor="black",
    )
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.08, 110.0)
    axes[1].set_xlabel("Regression Model Architecture")
    axes[1].set_ylabel("Test MSE ($m^2/s^2$, log scale) [Lower is Better]")
    axes[1].set_title("(b) Model Benchmark & Post-Processing MSE Comparison")
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].grid(True, axis="y", which="both", alpha=0.3)
    axes[1].legend(title="Post-Filter", loc="upper right")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig3_model_comparison_and_trajectories.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out_path


def plot_fig4_ablation_and_regularization(
    ablation_df: pd.DataFrame,
    ridge_artifact: TrainedModelArtifact,
    lasso_artifact: TrainedModelArtifact,
    output_dir: str,
) -> str:
    """
    Generate Figure 4: (a) 9-Stage Systematic Feature & Filtering Ablation Study
    showing progressive reduction in Test MSE and rise in R^2;
    (b) Top 14 Standardized Linear Regression Coefficients comparing Ridge (L2) vs. Lasso (L1).
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.6), gridspec_kw={"width_ratios": [1.15, 1.0]})

    # Panel (a): Ablation Progression
    short_labels = [
        "S1: Full-Frame Dense",
        "S2: ROI Global Dense",
        "S3: ROI Sparse LK",
        "S4: FOE Mask + Radial",
        "S5: 3x3 Perspective Grid",
        "S6: Hybrid Dense + LK",
        "S7: + Temporal Kinematics",
        "S8: + Moving Average",
        "S9: + 1D Kalman Filter",
    ][: len(ablation_df)]

    y_pos = np.arange(len(ablation_df))
    bars = axes[0].barh(
        y_pos,
        ablation_df["Ridge_Test_MSE"],
        color=sns.color_palette("Blues_r", len(ablation_df)),
        edgecolor="black",
        height=0.68,
    )
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(short_labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Contiguous Test MSE ($m^2/s^2$)")
    axes[0].set_title("(a) Progressive Feature & Kinematic Filtering Ablation")
    axes[0].grid(True, axis="x", alpha=0.3)

    for bar, r2_val in zip(bars, ablation_df["Ridge_Test_R2"]):
        w_val = bar.get_width()
        axes[0].text(
            w_val + 0.08,
            bar.get_y() + bar.get_height() / 2.0,
            f"MSE={w_val:.2f} (R²={r2_val:.3f})",
            va="center",
            fontsize=8,
        )
    axes[0].set_xlim(0, float(ablation_df["Ridge_Test_MSE"].max()) * 1.32)

    # Panel (b): Standardized Coefficients (Ridge vs Lasso)
    ridge_coef = extract_linear_coefficients(ridge_artifact).head(12)
    lasso_coef_map = dict(
        zip(
            lasso_artifact.feature_names,
            np.asarray(lasso_artifact.pipeline.named_steps["reg"].coef_).reshape(-1),
        )
    )
    top_feats = list(ridge_coef["feature"])[::-1]
    r_vals = [
        float(ridge_coef.loc[ridge_coef["feature"] == f, "coefficient"].values[0])
        for f in top_feats
    ]
    l_vals = [float(lasso_coef_map.get(f, 0.0)) for f in top_feats]

    idx = np.arange(len(top_feats))
    width = 0.38
    axes[1].barh(idx - width / 2, r_vals, width, label="Ridge (L2)", color="#1f77b4", edgecolor="black")
    axes[1].barh(idx + width / 2, l_vals, width, label="Lasso (L1)", color="#ff7f0e", edgecolor="black")
    axes[1].axvline(0.0, color="black", linewidth=0.9)
    axes[1].set_yticks(idx)
    axes[1].set_yticklabels(top_feats, fontsize=8)
    axes[1].set_xlabel("Standardized Regression Weight ($w_j$)")
    axes[1].set_title("(b) Top Standardized Weights: Ridge (L2) vs. Lasso (L1)")
    axes[1].grid(True, axis="x", alpha=0.3)
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig4_ablation_and_regularization.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out_path


def plot_fig5_error_analysis_and_residuals(
    splits: dict,
    y_pred_raw: np.ndarray,
    y_pred_kf: np.ndarray,
    error_slices: Dict[str, pd.DataFrame],
    output_dir: str,
) -> str:
    """
    Generate Figure 5: (a) Parity plot (Predicted vs. Ground Truth) with +-1.5 m/s band,
    (b) Residuals vs. True Speed illustrating heteroscedasticity at high speed,
    (c) Stratified RMSE across Speed Regimes and Illumination/Pitch slices.
    """
    os.makedirs(output_dir, exist_ok=True)
    _, y_test, _, _ = splits["test"]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.3))

    # Panel (a): Parity Plot
    axes[0].scatter(y_test, y_pred_raw, alpha=0.35, s=16, color="#1f77b4", label="Raw Ridge")
    axes[0].scatter(y_test, y_pred_kf, alpha=0.65, s=18, color="#2ca02c", label="Kalman Filtered")
    lims = [0.0, max(float(y_test.max()), float(y_pred_kf.max())) + 2.0]
    axes[0].plot(lims, lims, "k--", linewidth=1.4, label="Ideal Parity (y = x)")
    axes[0].fill_between(
        lims,
        [lims[0] - 1.5, lims[1] - 1.5],
        [lims[0] + 1.5, lims[1] + 1.5],
        color="gray",
        alpha=0.18,
        label="±1.5 m/s Tolerance",
    )
    axes[0].set_xlim(lims)
    axes[0].set_ylim(lims)
    axes[0].set_xlabel("Ground-Truth Speed (m/s)")
    axes[0].set_ylabel("Predicted Speed (m/s)")
    axes[0].set_title("(a) Test-Set Parity Plot")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left")

    # Panel (b): Residuals vs. True Speed
    res_raw = y_pred_raw - y_test
    res_kf = y_pred_kf - y_test
    axes[1].scatter(y_test, res_raw, alpha=0.35, s=16, color="#d62728", label="Raw Residual")
    axes[1].scatter(y_test, res_kf, alpha=0.65, s=18, color="#2ca02c", label="Kalman Residual")
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1.2)
    axes[1].axhline(1.5, color="gray", linestyle=":", linewidth=1.0)
    axes[1].axhline(-1.5, color="gray", linestyle=":", linewidth=1.0)
    axes[1].set_xlabel("Ground-Truth Speed (m/s)")
    axes[1].set_ylabel("Prediction Residual $\\hat{v} - v$ (m/s)")
    axes[1].set_title("(b) Residual Heteroscedasticity vs. Speed")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    # Panel (c): Stratified Error Slicing by Speed Regime & Pitch Jitter
    reg_df = error_slices["regime"].copy()
    x_idx = np.arange(len(reg_df))
    w_bar = 0.36
    axes[2].bar(
        x_idx - w_bar / 2,
        reg_df["raw_RMSE"],
        w_bar,
        label="Raw Ridge RMSE",
        color="#fc8d59",
        edgecolor="black",
    )
    axes[2].bar(
        x_idx + w_bar / 2,
        reg_df["kf_RMSE"],
        w_bar,
        label="Kalman Filtered RMSE",
        color="#2b83ba",
        edgecolor="black",
    )
    axes[2].set_xticks(x_idx)
    axes[2].set_xticklabels(reg_df["regime"], rotation=15)
    axes[2].set_ylabel("Test RMSE (m/s)")
    axes[2].set_title("(c) Stratified Error Slicing by Regime")
    axes[2].grid(True, axis="y", alpha=0.3)
    axes[2].legend(loc="upper left")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "fig5_error_analysis_and_residuals.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out_path
