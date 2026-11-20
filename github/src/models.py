"""
models.py
=========
Implements linear baselines, regularized linear regressors, and non-linear benchmark
models for vehicle speed estimation from optical flow features:
1. Univariate OLS Baseline (single global mean flow magnitude).
2. Multivariate Ordinary Least Squares (OLS) Linear Regression.
3. Ridge Regression (L2-regularized linear model, tuned on validation split).
4. Lasso Regression (L1-regularized sparse linear model, tuned on validation split).
5. Support Vector Regression (RBF kernel, epsilon-insensitive loss).
6. Random Forest Regressor (non-parametric tree ensemble).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


@dataclass
class TrainedModelArtifact:
    """Container for a fitted sklearn pipeline, selected hyperparameters, and metadata."""
    name: str
    pipeline: Pipeline
    feature_names: List[str]
    best_hyperparam: Optional[float] = None
    condition_number: Optional[float] = None


def compute_design_matrix_condition_number(X: pd.DataFrame) -> float:
    """Compute condition number kappa(X_scaled^T X_scaled) to quantify multicollinearity."""
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X.values)
    s = np.linalg.svd(X_std, compute_uv=False)
    s_min = max(float(s[-1]), 1e-12)
    return float((s[0] / s_min) ** 2)


def train_ols(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    model_name: str = "OLS Linear Regression",
) -> TrainedModelArtifact:
    """Fit unregularized Ordinary Least Squares with standardized features."""
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("reg", LinearRegression()),
        ]
    )
    pipe.fit(X_train, y_train)
    cond = compute_design_matrix_condition_number(X_train)
    return TrainedModelArtifact(
        name=model_name,
        pipeline=pipe,
        feature_names=list(X_train.columns),
        best_hyperparam=0.0,
        condition_number=cond,
    )


def train_ridge(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    alphas: List[float],
    model_name: str = "Ridge Regression (L2)",
) -> TrainedModelArtifact:
    """Fit L2-regularized Ridge Regression, selecting alpha via contiguous validation MSE."""
    best_alpha = alphas[0]
    best_val_mse = float("inf")
    best_pipe = None

    for alpha in alphas:
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=float(alpha), random_state=42)),
            ]
        )
        pipe.fit(X_train, y_train)
        val_pred = pipe.predict(X_val)
        val_mse = float(mean_squared_error(y_val, val_pred))
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_alpha = float(alpha)
            best_pipe = pipe

    cond = compute_design_matrix_condition_number(X_train)
    return TrainedModelArtifact(
        name=model_name,
        pipeline=best_pipe,
        feature_names=list(X_train.columns),
        best_hyperparam=best_alpha,
        condition_number=cond,
    )


def train_lasso(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    alphas: List[float],
    max_iter: int = 10000,
    model_name: str = "Lasso Regression (L1)",
) -> TrainedModelArtifact:
    """Fit L1-regularized Lasso Regression, selecting alpha via contiguous validation MSE."""
    best_alpha = alphas[0]
    best_val_mse = float("inf")
    best_pipe = None

    for alpha in alphas:
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("reg", Lasso(alpha=float(alpha), max_iter=max_iter, random_state=42)),
            ]
        )
        pipe.fit(X_train, y_train)
        val_pred = pipe.predict(X_val)
        val_mse = float(mean_squared_error(y_val, val_pred))
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_alpha = float(alpha)
            best_pipe = pipe

    cond = compute_design_matrix_condition_number(X_train)
    return TrainedModelArtifact(
        name=model_name,
        pipeline=best_pipe,
        feature_names=list(X_train.columns),
        best_hyperparam=best_alpha,
        condition_number=cond,
    )


def train_svr(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    C: float = 10.0,
    epsilon: float = 0.2,
    gamma: str = "scale",
    model_name: str = "RBF-SVR",
) -> TrainedModelArtifact:
    """Fit non-linear Support Vector Regressor with Radial Basis Function kernel."""
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("reg", SVR(kernel="rbf", C=float(C), epsilon=float(epsilon), gamma=gamma)),
        ]
    )
    pipe.fit(X_train, y_train)
    return TrainedModelArtifact(
        name=model_name,
        pipeline=pipe,
        feature_names=list(X_train.columns),
        best_hyperparam=float(C),
    )


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    n_estimators: int = 200,
    max_depth: int = 10,
    min_samples_leaf: int = 4,
    model_name: str = "Random Forest",
) -> TrainedModelArtifact:
    """Fit non-parametric Random Forest Regressor."""
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "reg",
                RandomForestRegressor(
                    n_estimators=int(n_estimators),
                    max_depth=int(max_depth),
                    min_samples_leaf=int(min_samples_leaf),
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    return TrainedModelArtifact(
        name=model_name,
        pipeline=pipe,
        feature_names=list(X_train.columns),
        best_hyperparam=float(max_depth),
    )


def train_all_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    config: dict,
) -> Dict[str, TrainedModelArtifact]:
    """Train the complete suite of baseline, regularized linear, and non-linear models."""
    m_cfg = config["models"]
    artifacts: Dict[str, TrainedModelArtifact] = {}

    # 0. Naive Univariate OLS Baseline (single ROI mean flow magnitude)
    uni_col = ["roi_mean_mag"] if "roi_mean_mag" in X_train.columns else [X_train.columns[0]]
    artifacts["Univariate_OLS"] = train_ols(
        X_train[uni_col], y_train, model_name="Univariate OLS (Global Mean Flow)"
    )

    # 1. Multivariate OLS Linear Regression
    artifacts["Multivariate_OLS"] = train_ols(
        X_train, y_train, model_name="Multivariate OLS"
    )

    # 2. Ridge Regression (L2)
    artifacts["Ridge_L2"] = train_ridge(
        X_train,
        y_train,
        X_val,
        y_val,
        alphas=m_cfg["ridge"]["alphas"],
        model_name="Ridge Regression (L2)",
    )

    # 3. Lasso Regression (L1)
    artifacts["Lasso_L1"] = train_lasso(
        X_train,
        y_train,
        X_val,
        y_val,
        alphas=m_cfg["lasso"]["alphas"],
        max_iter=int(m_cfg["lasso"]["max_iter"]),
        model_name="Lasso Regression (L1)",
    )

    # 4. Support Vector Regression (RBF)
    artifacts["SVR_RBF"] = train_svr(
        X_train,
        y_train,
        C=float(m_cfg["svr"]["C"]),
        epsilon=float(m_cfg["svr"]["epsilon"]),
        gamma=m_cfg["svr"]["gamma"],
        model_name="SVR (RBF Kernel)",
    )

    # 5. Random Forest
    artifacts["Random_Forest"] = train_random_forest(
        X_train,
        y_train,
        n_estimators=int(m_cfg["random_forest"]["n_estimators"]),
        max_depth=int(m_cfg["random_forest"]["max_depth"]),
        min_samples_leaf=int(m_cfg["random_forest"]["min_samples_leaf"]),
        model_name="Random Forest",
    )

    return artifacts


def extract_linear_coefficients(artifact: TrainedModelArtifact) -> pd.DataFrame:
    """Extract standardized regression coefficients w_j for interpretability."""
    reg = artifact.pipeline.named_steps["reg"]
    if not hasattr(reg, "coef_"):
        raise ValueError(f"Model {artifact.name} does not expose linear coefficients.")
    coefs = np.asarray(reg.coef_).reshape(-1)
    df_coef = pd.DataFrame(
        {
            "feature": artifact.feature_names,
            "coefficient": coefs,
            "abs_coefficient": np.abs(coefs),
        }
    )
    return df_coef.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
