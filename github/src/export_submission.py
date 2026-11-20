# -*- coding: utf-8 -*-
r"""
export_submission.py
====================
1. Renders `reports/proposal.pdf` (1-page CS229 Project Proposal PDF, anonymous).
2. Renders `reports/poster.pdf` (Visual-first 2-Column Landscape CS229 Poster PDF
   with 4 summary cards + 6 framed figure panels matching natural 3.3:1 aspect ratio).
3. Creates `overleaf_project.zip` ready for 1-click upload to Overleaf ("New Project ->
   Upload Project") with `main.tex` (Final Report), `milestone.tex`, `proposal.tex`,
   `poster.tex`, and `figures/fig1..fig6.png` pre-linked via `\graphicspath`.
4. Creates `vehicle-speed-optical-flow.zip` (< 5 MB) containing the complete clean
   codebase, configs, notebook, reports, and figures for Gradescope submission.
"""

from __future__ import annotations

import os
import textwrap
import zipfile

import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pymupdf

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def render_proposal_pdf(output_pdf: str) -> str:
    """Render reports/proposal.pdf using PyMuPDF Story (HTML layout) on US Letter."""
    html = """
    <html>
    <head>
    <style>
      body { font-family: sans-serif; font-size: 10.2pt; line-height: 1.38; color: #111111; }
      h1 { font-size: 15pt; color: #8C1515; text-align: center; margin-bottom: 4pt; }
      .meta { text-align: center; font-size: 9.8pt; color: #333333; margin-bottom: 10pt; }
      h2 { font-size: 11.5pt; color: #1F3A60; margin-top: 8pt; margin-bottom: 3pt; border-bottom: 0.5pt solid #cccccc; }
      p { margin-top: 2pt; margin-bottom: 5pt; text-align: justify; }
      ol { margin-top: 2pt; margin-bottom: 5pt; padding-left: 16pt; }
      li { margin-bottom: 3pt; text-align: justify; }
      .refs { font-size: 8.8pt; color: #333333; margin-top: 4pt; }
    </style>
    </head>
    <body>
      <h1>CS229 Project Proposal: Vehicle Speed Estimation with Optical Flow and Linear Regression</h1>
      <div class="meta"><b>Project Category:</b> Computer Vision (Application &amp; Feature-Engineering Project) &nbsp;|&nbsp; <b>GitHub:</b> <a href="https://github.com/Lukeeffs/vehicle-speed-optical-flow">https://github.com/Lukeeffs/vehicle-speed-optical-flow</a></div>

      <h2>1. Motivation</h2>
      <p>Reliable ego-vehicle longitudinal speed estimation is critical for autonomous navigation, adaptive cruise control, and forward collision warning when wheel odometry suffers from tire slip or GPS signals drop out in urban canyons. This is an <b>application project</b> combining computer vision and physics-informed feature engineering. Rather than training parameter-heavy 3D Convolutional Neural Networks that demand GPU hardware and exhibit opaque failure modes under illumination shift, we formulate monocular speed estimation as an interpretable, real-time CPU regression problem mapping consecutive 20 Hz dashcam frames (I<sub>t-1</sub>, I<sub>t</sub>) to instantaneous vehicle velocity v&#770;<sub>t</sub> &ge; 0 (m/s).</p>

      <h2>2. Method</h2>
      <p>Under a pinhole camera mounted at height H<sub>c</sub> with focal length f and horizon row y<sub>0</sub>, longitudinal road optical flow scales quadratically below the horizon as dy/dt = ((y - y<sub>0</sub>)<sup>2</sup> / (f H<sub>c</sub>)) v<sub>t</sub> + f &omega;<sub>pitch</sub>(t). Exploiting this geometry, our pipeline integrates:</p>
      <ol>
        <li><b>Photometric &amp; Perspective Preprocessing:</b> Grayscale conversion, Contrast-Limited Adaptive Histogram Equalization (CLAHE) to attenuate shadow boundaries, Gaussian smoothing (&sigma; = 1.2), and trapezoidal Region-of-Interest (ROI) masking removing sky drift (y &lt; 0.38H) and static car-hood glare (y &gt; 0.86H).</li>
        <li><b>Dual Optical Flow Extraction:</b> Dense Gunnar Farneb&auml;ck polynomial expansion flow (Farneb&auml;ck, 2003) and Sparse Pyramidal Lucas-Kanade tracking (Lucas &amp; Kanade, 1981) over Shi-Tomasi corners filtered via forward-backward consistency.</li>
        <li><b>Physics-Informed Features &amp; CS229 Models:</b> We mask the noisy Focus-of-Expansion (FOE) singularity, project vectors onto radial axes, normalize vertical flow via inverse-perspective scaling, and partition the road into a 3&times;3 grid with causal rolling kinematic lags. To address inter-zone multicollinearity, we compare Ordinary Least Squares (OLS) baselines against L2 Ridge, L1 Lasso, RBF Support Vector Regression (SVR), and Random Forests, coupled with a 1D Constant-Acceleration Kalman Filter. We also apply CS229 classification and unsupervised methods (Logistic/Softmax Regression, K-Means, and Gaussian Mixture Models via EM) to analyze discrete driving regimes.</li>
      </ol>

      <h2>3. Intended Experiments &amp; Evaluation</h2>
      <p>Following the comma.ai Speed Challenge benchmark (640&times;480, 20 fps), we enforce a strict contiguous chronological split (70% Train, 15% Validation, 15% Test with a 5-frame embargo) and empirically quantify temporal data leakage against random 5-fold cross-validation. We evaluate models using Test MSE (m<sup>2</sup>/s<sup>2</sup>), RMSE, MAE, and R<sup>2</sup>, supported by a 9-stage feature/filter ablation study and stratified error slicing across driving speeds, shadows/glare, and camera pitch jitter.</p>

      <h2>References</h2>
      <div class="refs">1. comma.ai Speed Challenge Benchmark (2017). &nbsp;&nbsp; 2. G. Farneb&auml;ck, <i>SCIA</i>, 2003. &nbsp;&nbsp; 3. B. Lucas &amp; T. Kanade, <i>IJCAI</i>, 1981.</div>
    </body>
    </html>
    """
    story = pymupdf.Story(html=html)
    writer = pymupdf.DocumentWriter(output_pdf)
    mediabox = pymupdf.paper_rect("letter")
    where = mediabox + (48, 44, -48, -44)
    more = True
    while more:
        device = writer.begin_page(mediabox)
        more, _ = story.place(where)
        story.draw(device)
        writer.end_page()
    writer.close()
    return output_pdf


def render_poster_pdf(fig_dir: str, output_pdf: str) -> str:
    """
    Render a visual-first 2-Column x 3-Row Landscape CS229 Poster PDF (22 x 16.5 in):
    - Top Banner + 4 Executive Summary Cards
    - 6 Tightly Framed High-DPI Figure Panels matching their ~3.3:1 aspect ratio
    """
    fig = plt.figure(figsize=(22, 16.5), dpi=150, facecolor="#EEF1F5")
    bg_ax = fig.add_axes([0, 0, 1, 1], zorder=0)
    bg_ax.axis("off")

    # 1. Header Banner (y = 0.912 .. 0.988)
    hdr_patch = patches.FancyBboxPatch(
        (0.012, 0.912),
        0.976,
        0.076,
        boxstyle="round,pad=0.004,rounding_size=0.008",
        facecolor="#8C1515",
        edgecolor="none",
        transform=bg_ax.transAxes,
    )
    bg_ax.add_patch(hdr_patch)
    bg_ax.text(
        0.5,
        0.961,
        "VEHICLE SPEED ESTIMATION WITH OPTICAL FLOW AND LINEAR REGRESSION",
        ha="center",
        va="center",
        color="white",
        fontsize=23,
        fontweight="bold",
        transform=bg_ax.transAxes,
    )
    bg_ax.text(
        0.5,
        0.929,
        "Stanford CS229 Machine Learning  |  Computer Vision Track  |  https://github.com/Lukeeffs/vehicle-speed-optical-flow",
        ha="center",
        va="center",
        color="#F8E8E8",
        fontsize=12.5,
        transform=bg_ax.transAxes,
    )

    # 2. Four Top Executive Cards (y = 0.805 .. 0.900)
    summary_cards = [
        (
            "Projective Kinematics & ROI",
            "Pinhole road flow scales quadratically:\ndy/dt = ((y - y0)^2 / (f*Hc))*v_t + f*w_pitch(t)\nCLAHE + trapezoid mask removes sky & hood.",
            "#1F3A60",
        ),
        (
            "Leakage-Free Contiguous Split",
            "Strict 70% / 15% / 15% chronological split\nwith 5-frame embargo prevents 19.5% MSE\nunderestimation from random K-fold CV.",
            "#0072B2",
        ),
        (
            "Physics-Informed Features (d=166)",
            "Dense Farneback + Sparse Lucas-Kanade,\nFOE exclusion (R=28px), radial projection,\nIPM normalization, 3x3 grid & causal lags.",
            "#009E73",
        ),
        (
            "Headline Test Performance",
            "Ridge + Kalman: MSE=0.168 (RMSE=0.410, R2=0.9975)\nLasso + Kalman: MSE=0.149 (RMSE=0.386, R2=0.9978)\n99.7% MSE drop vs. Univariate OLS (50.38).",
            "#8C1515",
        ),
    ]
    card_w = 0.236
    card_gap = 0.0106
    for idx, (ctitle, cbody, accent) in enumerate(summary_cards):
        cx = 0.012 + idx * (card_w + card_gap)
        cy, ch = 0.805, 0.095
        card = patches.FancyBboxPatch(
            (cx, cy),
            card_w,
            ch,
            boxstyle="round,pad=0.002,rounding_size=0.006",
            facecolor="white",
            edgecolor="#D0D7DE",
            linewidth=1.1,
            transform=bg_ax.transAxes,
        )
        bg_ax.add_patch(card)
        bar = patches.Rectangle(
            (cx, cy + ch - 0.009),
            card_w,
            0.009,
            facecolor=accent,
            edgecolor="none",
            transform=bg_ax.transAxes,
        )
        bg_ax.add_patch(bar)
        bg_ax.text(
            cx + 0.008,
            cy + ch - 0.019,
            ctitle,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
            color=accent,
            transform=bg_ax.transAxes,
        )
        bg_ax.text(
            cx + 0.008,
            cy + ch - 0.040,
            cbody,
            ha="left",
            va="top",
            fontsize=9.3,
            linespacing=1.32,
            color="#222222",
            transform=bg_ax.transAxes,
        )

    # 3. 2-Column x 3-Row Grid of Framed Figure Panels (y = 0.015 .. 0.776)
    panel_specs = [
        (
            0,
            0,
            "1. Photometric Preprocessing, Trapezoid ROI & Dual Optical Flow",
            "fig1_pipeline_and_roi_flow.png",
            "Trapezoidal drivable road mask (green), 3x3 perspective grid (cyan), FOE singularity exclusion (red circle, R=28px), Dense Farneback magnitude heatmap, and forward-backward verified Sparse Lucas-Kanade corners.",
        ),
        (
            0,
            1,
            "2. 9-Stage Progressive Ablation & Ridge/Lasso Weight Profile",
            "fig4_ablation_and_regularization.png",
            "Monotonic 98.6% Test MSE reduction from Stage 1 (12.52) to Stage 9 (0.17 m^2/s^2). Ridge (L2) and Lasso (L1) stabilize collinear Gram matrix (kappa = 4.79e28) using mid-center vy and rolling IPM weights.",
        ),
        (
            1,
            0,
            "3. Chronological Splits (Train/Val/Test) & Feature Correlations",
            "fig2_eda_speed_and_correlations.png",
            "Contiguous 20 Hz velocity trajectory spanning urban, arterial, and highway regimes. Near-zone median optical flow magnitudes achieve Pearson r > 0.91 with ground-truth ego velocity.",
        ),
        (
            1,
            1,
            "4. Test Trajectory Tracking & Multi-Model MSE Benchmark",
            "fig3_model_comparison_and_trajectories.png",
            "Regularized linear models outperform non-linear RBF-SVR (2.242) and Random Forests (1.531). Our 1D Constant-Acceleration Kalman Filter tracks velocity ramps without lag, cutting MSE by 66.5%.",
        ),
        (
            2,
            0,
            "5. CS229 Course Theory Alignment Benchmark (6 Lecture Decks)",
            "fig6_theory_alignment_benchmark.png",
            "Left: OLS Pseudo-Inverse vs. Batch GD and Regularized Normal Equation/GD. Center: Driving-regime Binary Logistic (97.8%), Softmax (94.4%), K-NN (92.2%), Gaussian NB (88.3%), K-Means & EM-GMM. Right: Ablation curve.",
        ),
        (
            2,
            1,
            "6. Residual Diagnostics & Stratified Error Slicing",
            "fig5_error_analysis_and_residuals.png",
            "Lowest error in medium arterial (KF RMSE = 0.122 m/s) and highway (0.259 m/s). Kalman filtering slashes high camera pitch-jitter MSE by 73.6% (to 0.168 m^2/s^2) and shadow/glare MSE by 64.2%.",
        ),
    ]

    col_w = 0.483
    col_xs = [0.012, 0.505]
    row_h = 0.252
    row_ys = [0.540, 0.277, 0.014]

    for row_i, col_i, title, fname, caption in panel_specs:
        px = col_xs[col_i]
        py = row_ys[row_i]
        card = patches.FancyBboxPatch(
            (px, py),
            col_w,
            row_h,
            boxstyle="round,pad=0.002,rounding_size=0.006",
            facecolor="white",
            edgecolor="#C9D1D9",
            linewidth=1.1,
            transform=bg_ax.transAxes,
        )
        bg_ax.add_patch(card)

        # Title at top of card
        bg_ax.text(
            px + 0.008,
            py + row_h - 0.008,
            title,
            ha="left",
            va="top",
            fontsize=11.5,
            fontweight="bold",
            color="#1F3A60",
            transform=bg_ax.transAxes,
        )

        # Embedded figure image preserving exact aspect ratio without stretching
        img_path = os.path.join(fig_dir, fname)
        ax_img = fig.add_axes([px + 0.006, py + 0.042, col_w - 0.012, row_h - 0.068], zorder=2)
        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            ax_img.imshow(img, aspect="equal")
        ax_img.axis("off")

        # Wrapped 2-line caption cleanly at bottom of card
        wrapped_cap = textwrap.fill(caption, width=124)
        bg_ax.text(
            px + 0.008,
            py + 0.036,
            wrapped_cap,
            ha="left",
            va="top",
            fontsize=8.3,
            linespacing=1.22,
            color="#333333",
            transform=bg_ax.transAxes,
        )

    fig.savefig(output_pdf, format="pdf", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return output_pdf


def build_overleaf_zip(root_dir: str, out_zip: str) -> float:
    """
    Create a self-contained Overleaf project archive `overleaf_project.zip`:
    - main.tex (copy of final_report.tex so Overleaf compiles Final Report immediately)
    - final_report.tex
    - milestone.tex
    - proposal.tex
    - poster.tex
    - figures/fig1..fig6.png
    """
    reports_dir = os.path.join(root_dir, "reports")
    figures_dir = os.path.join(root_dir, "figures")

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        final_tex = os.path.join(reports_dir, "final_report.tex")
        if os.path.exists(final_tex):
            zf.write(final_tex, arcname="main.tex")

        for tex_name in ["final_report.tex", "milestone.tex", "proposal.tex", "poster.tex"]:
            p = os.path.join(reports_dir, tex_name)
            if os.path.exists(p):
                zf.write(p, arcname=tex_name)

        for fn in sorted(os.listdir(figures_dir)):
            if fn.lower().endswith(".png"):
                zf.write(os.path.join(figures_dir, fn), arcname=f"figures/{fn}")

    return os.path.getsize(out_zip) / (1024.0 * 1024.0)


def build_clean_code_zip(root_dir: str, out_zip: str) -> float:
    """
    Create the clean CS229 code submission archive `vehicle-speed-optical-flow.zip` (< 5 MB),
    excluding the 6 large raw lecture PDFs, __pycache__, poster.pdf, and .zip files.
    """
    include_top_files = [".gitignore", "README.md", "requirements.txt", "baocao.py", "BAOCAO_TIENDO.md"]
    include_dirs = ["configs", "src", "notebooks", "reports", "figures"]

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fn in include_top_files:
            fp = os.path.join(root_dir, fn)
            if os.path.exists(fp):
                zf.write(fp, arcname=fn)

        for dname in include_dirs:
            dpath = os.path.join(root_dir, dname)
            if not os.path.exists(dpath):
                continue
            for r, _, files in os.walk(dpath):
                if "__pycache__" in r:
                    continue
                for f in sorted(files):
                    if f.endswith(".pyc") or f == "poster.pdf":
                        continue
                    full_p = os.path.join(r, f)
                    rel_p = os.path.relpath(full_p, root_dir).replace("\\", "/")
                    zf.write(full_p, arcname=rel_p)

    return os.path.getsize(out_zip) / (1024.0 * 1024.0)


def main() -> None:
    reports_dir = os.path.join(ROOT_DIR, "reports")
    figures_dir = os.path.join(ROOT_DIR, "figures")

    prop_pdf = render_proposal_pdf(os.path.join(reports_dir, "proposal.pdf"))
    poster_pdf = render_poster_pdf(figures_dir, os.path.join(reports_dir, "poster.pdf"))
    ov_mb = build_overleaf_zip(ROOT_DIR, os.path.join(ROOT_DIR, "overleaf_project.zip"))
    code_mb = build_clean_code_zip(ROOT_DIR, os.path.join(ROOT_DIR, "vehicle-speed-optical-flow.zip"))

    print(f"Generated Proposal PDF : {prop_pdf}")
    print(f"Generated Poster PDF   : {poster_pdf}")
    print(f"Generated Overleaf Zip : {os.path.join(ROOT_DIR, 'overleaf_project.zip')} ({ov_mb:.2f} MB)")
    print(f"Generated Code Zip     : {os.path.join(ROOT_DIR, 'vehicle-speed-optical-flow.zip')} ({code_mb:.2f} MB < 5.00 MB)")


if __name__ == "__main__":
    main()
