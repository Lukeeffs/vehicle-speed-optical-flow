"""
temporal_filter.py
==================
Causal post-processing filters enforcing longitudinal vehicle kinematics on
raw frame-level speed predictions:
1. Causal Simple Moving Average (SMA).
2. Causal Exponential Moving Average (EMA).
3. 1D Linear Kinematic Kalman Filter tracking state s_t = [v_t, a_t]^T
   (speed and longitudinal acceleration) under physical inertia constraints.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def causal_moving_average(preds: np.ndarray, window_size: int = 10) -> np.ndarray:
    r"""
    Apply a strictly causal backward moving average filter of length `window_size`:
        \tilde{v}_t = (1 / min(t+1, W)) \sum_{k=0}^{min(t, W-1)} \hat{v}_{t-k}
    """
    n = len(preds)
    smoothed = np.empty(n, dtype=np.float64)
    for t in range(n):
        start = max(0, t - window_size + 1)
        smoothed[t] = float(np.mean(preds[start : t + 1]))
    return np.clip(smoothed, 0.0, None)


def causal_exponential_moving_average(preds: np.ndarray, alpha: float = 0.22) -> np.ndarray:
    r"""
    Apply a first-order causal Infinite Impulse Response (IIR) exponential filter:
        \tilde{v}_t = \alpha \hat{v}_t + (1 - \alpha) \tilde{v}_{t-1}
    """
    n = len(preds)
    smoothed = np.empty(n, dtype=np.float64)
    smoothed[0] = float(preds[0])
    for t in range(1, n):
        smoothed[t] = alpha * float(preds[t]) + (1.0 - alpha) * smoothed[t - 1]
    return np.clip(smoothed, 0.0, None)


def kalman_kinematic_filter(
    preds: np.ndarray,
    dt: float = 0.05,
    process_noise_pos: float = 0.02,
    process_noise_vel: float = 0.15,
    measurement_noise: float = 2.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    1D Constant-Acceleration Linear Kalman Filter for vehicle speed estimation.

    State vector:
        s_t = [v_t, a_t]^T  (ego speed in m/s, longitudinal acceleration in m/s^2)
    State transition:
        s_{t|t-1} = F s_{t-1|t-1},   F = [[1, dt], [0, 1]]
    Measurement equation:
        z_t = H s_t + nu_t,          H = [[1, 0]],  nu_t ~ N(0, R)

    Returns
    -------
    filtered_speed : np.ndarray of shape (N,)
    estimated_accel : np.ndarray of shape (N,)
    """
    n = len(preds)
    F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
    H = np.array([[1.0, 0.0]], dtype=np.float64)
    Q = np.array(
        [
            [process_noise_pos * dt, 0.0],
            [0.0, process_noise_vel * dt],
        ],
        dtype=np.float64,
    )
    R = np.array([[measurement_noise]], dtype=np.float64)
    I = np.eye(2, dtype=np.float64)

    x = np.array([[float(preds[0])], [0.0]], dtype=np.float64)
    P = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    filtered_speed = np.empty(n, dtype=np.float64)
    estimated_accel = np.empty(n, dtype=np.float64)

    for t in range(n):
        if t > 0:
            # Predict step (Constant-Acceleration state transition)
            x = F @ x
            P = F @ P @ F.T + Q

        # Innovation / Update step
        z = np.array([[float(preds[t])]], dtype=np.float64)
        y_tilde = z - (H @ x)
        S = H @ P @ H.T + R
        K = (P @ H.T) / S[0, 0]

        x = x + K @ y_tilde
        P = (I - K @ H) @ P

        # Enforce non-negative speed and physical acceleration bounds [-6, 6] m/s^2
        x[0, 0] = max(0.0, x[0, 0])
        x[1, 0] = np.clip(x[1, 0], -6.0, 6.0)

        filtered_speed[t] = x[0, 0]
        estimated_accel[t] = x[1, 0]

    return filtered_speed, estimated_accel
