# Vehicle Speed Estimation with Optical Flow and Linear Regression

**Course Track:** Stanford CS229 Machine Learning — Term Project  
**Category:** Computer Vision (Application & Feature-Engineering Project)  
**GitHub Repository:** https://github.com/Lukeeffs/vehicle-speed-optical-flow

---

## 1. Overview & Motivation
This repository implements an end-to-end, reproducible machine learning pipeline for estimating instantaneous ego-vehicle longitudinal velocity $\hat{v}_t \in \mathbb{R}_{\ge 0}$ ($\text{m/s}$) from a monocular forward-facing dashcam video stream ($640 \times 480$ at $20\text{ Hz}$). Rather than relying on parameter-heavy 3D Convolutional Neural Networks, we combine **projective ground-plane geometry**, **Dense Farnebäck & Sparse Lucas-Kanade Optical Flow**, **Multi-Zone $3 \times 3$ Perspective Grid Feature Engineering**, **Regularized Linear Regression ($\ell_2$ Ridge & $\ell_1$ Lasso)**, and **1D Kinematic Kalman Filtering**.

### Key Methodological Contributions
1. **Perspective ROI & Focus-of-Expansion (FOE) Masking:** Eliminates non-rigid sky motion ($y < 0.38H$), static car-hood reflections ($y > 0.86H$), and vanishing-point SNR collapse within radius $R_{\text{FOE}} = 28\text{ px}$ of $(x_0, y_0)$.
2. **Physics-Informed Multi-Zone $3 \times 3$ Grid Decomposition:** Under pinhole ground-plane projection, optical flow scales quadratically with vertical offset below the horizon:
   $$\dot{y}(x, y, t) = \frac{(y - y_0)^2}{f H_c} v_t + f \dot{\theta}_{\text{pitch}}(t)$$
   Partitioning the drivable trapezoid into Near/Mid/Far $\times$ Left/Center/Right zones enables linear models to learn depth-specific weights and compensate for camera pitch jitter.
3. **Contiguous Chronological Evaluation Protocol:** Prevents the severe temporal autocorrelation leakage ($\Delta t = 50\text{ ms}$) inherent to random $K$-fold cross-validation on video streams by enforcing a strict $70\%/15\%/15\%$ contiguous Train/Validation/Test split with a $5$-frame embargo gap.
4. **Causal Kinematic Post-Processing:** Couples frame-level regression outputs with a 1D Constant-Acceleration Kalman Filter tracking state $\mathbf{s}_t = [v_t, a_t]^\top$ under physical vehicle inertia bounds.

---

## 2. Repository Structure
```text
BTL/
├── README.md                  # Setup, dataset instructions, reproduction commands
├── requirements.txt           # Python dependencies
├── configs/
│   └── config.yaml            # ROI coordinates, Optical Flow params, split ratios, model hyperparameters
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Video/frame loading, speed label alignment, contiguous temporal splitting
│   ├── eda.py                 # Data exploration & distribution/temporal correlation summaries
│   ├── preprocessing.py       # Grayscale, CLAHE, Gaussian blur, perspective ROI & trapezoid masking
│   ├── optical_flow.py        # Dense Farnebäck Optical Flow & Sparse Lucas-Kanade (Shi-Tomasi) extractors
│   ├── feature_engineering.py # Multi-zone 3x3 perspective grid, FOE radial projection, causal temporal lags
│   ├── models.py              # Baselines (OLS) & regularized/non-linear models (Ridge, Lasso, SVR, RF)
│   ├── temporal_filter.py     # Post-processing smoothing (Causal Moving Average, EMA & 1D Kalman Filter)
│   ├── evaluate.py            # MSE, RMSE, MAE, R², 9-stage ablation runner, and error analysis slicing
│   ├── visualize.py           # 300-DPI publication plots (flow fields, speed curves, ablation, residuals)
│   ├── run_pipeline.py        # End-to-end experiment runner
│   └── export_submission.py   # Renders proposal.pdf, poster.pdf, overleaf_project.zip, & clean code zip
├── notebooks/
│   └── full_experiment_pipeline.ipynb
├── reports/
│   ├── proposal.tex           # LaTeX Project Proposal (strict 300–500 words body)
│   ├── proposal.pdf           # Rendered 1-page Project Proposal PDF
│   ├── milestone.tex          # Max 3 pages (excluding references), CS229 LaTeX format
│   ├── poster.tex             # LaTeX visual-first poster embedding fig1..fig6
│   ├── poster.pdf             # Rendered visual-first 2-column landscape poster PDF
│   └── final_report.tex       # Max 5 pages (including figures & appendices) + References
├── figures/                   # High-DPI (300 DPI) publication-ready plots (fig1..fig6)
├── overleaf_project.zip       # Self-contained Overleaf upload (TeX + figures/ pre-linked)
└── vehicle-speed-optical-flow.zip  # Clean submission archive (< 5 MB, excl. raw PDFs)
```

---

## 3. Environment Setup & Dataset Instructions

### 3.1 Install Dependencies
```bash
pip install -r requirements.txt
```

### 3.2 Dataset Configuration (comma.ai Speed Challenge or Built-In Physics Synthesizer)
- **Option A — Real comma.ai Speed Challenge Dataset:** Place `train.mp4` ($640 \times 480$, $20\text{ fps}$) and `train.txt` (one speed label in $\text{m/s}$ per line) inside `data/train.mp4` and `data/train.txt`. `src/data_loader.py` automatically detects and loads them.
- **Option B — Reproducible Pinhole-Camera Dashcam Synthesizer (Default Fallback):** When `data/train.mp4` is not present (keeping the repository lightweight under $5\text{ MB}$), `load_or_generate_dataset()` automatically synthesizes a $1{,}200$-frame ($60\text{ s}$ at $20\text{ fps}$) $640 \times 480$ sequence rendered via exact pinhole ground-plane projection with realistic sky clutter, car hood reflections, tree shadows, and road-bump camera pitch jitter.

---

## 4. Reproducing All Experiments & Figures

### Run End-to-End Pipeline & Submission Exporter
```bash
python -m src.run_pipeline
python baocao.py
python -m src.export_submission
```


### Compile LaTeX Reports Locally
```bash
cd reports
pdflatex milestone.tex
pdflatex final_report.tex
```
