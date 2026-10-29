"""
Trajectory decoding analysis.

Two analyses:
1. Ball endpoint decoding — from population state at each time bin,
   decode the ball's y-position at paddle interception.

2. Full trajectory decoding — from population state at 250ms,
   decode the ball's position at multiple future points along its trajectory.
   Trials aligned to endpoints (paddle interception = 0ms).
   Decode 29 time bins x 50ms = 1450ms back from interception
   (duration of shortest trial).

Both use nested CV: 5-fold outer, 4-fold inner for hyperparameter selection.
Ridge regression for decoding.
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from typing import Dict


RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]


def endpoint_decoding(states: np.ndarray, endpoint_y: np.ndarray,
                      n_outer_folds: int = 5,
                      n_inner_folds: int = 4) -> Dict[str, np.ndarray]:
    """
    Decode ball endpoint from population state at each time bin.

    Args:
        states: (N_units, 79, T) population states
        endpoint_y: (79,) ball y-position at paddle interception
        n_outer_folds: outer CV folds (default 5)
        n_inner_folds: inner CV folds for hyperparameter selection (default 4)

    Returns:
        dict with:
            'correlation': (T,) Pearson r between predicted and actual endpoint
            'predictions': (79, T) cross-validated predicted endpoints
    """
    N, C, T = states.shape

    correlations = np.zeros(T)
    all_predictions = np.full((C, T), np.nan)

    for t in range(T):
        X = states[:, :, t].T  # (79, N)
        y = endpoint_y

        valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
        if valid.sum() < 10:
            correlations[t] = np.nan
            continue

        X_v, y_v = X[valid], y[valid]
        valid_idx = np.where(valid)[0]

        y_pred = _nested_cv_ridge(X_v, y_v, n_outer_folds, n_inner_folds)

        valid_pred = np.isfinite(y_pred)
        if valid_pred.sum() > 2:
            r, _ = pearsonr(y_v[valid_pred], y_pred[valid_pred])
            correlations[t] = r
            all_predictions[valid_idx[valid_pred], t] = y_pred[valid_pred]
        else:
            correlations[t] = np.nan

    return {
        "correlation": correlations,
        "predictions": all_predictions,
    }


def full_trajectory_decoding(states: np.ndarray,
                             ball_positions_y: np.ndarray,
                             trial_durations: np.ndarray,
                             decode_from_time: int = 5,
                             n_outer_folds: int = 5,
                             n_inner_folds: int = 4,
                             n_decode_bins: int = 29) -> Dict[str, np.ndarray]:
    """
    From a single early population state, decode the ball's y-position
    at multiple future time points along its trajectory.

    Args:
        states: (N_units, 79, T) population states
        ball_positions_y: (79, T_full) ball y-positions over full trial
                          (visible + occluded epochs concatenated)
        trial_durations: (79,) number of valid time bins per condition
        decode_from_time: time bin to decode from (5 = 250ms)
        n_outer_folds: outer CV folds (default 5)
        n_inner_folds: inner CV folds (default 4)
        n_decode_bins: number of bins to decode back from endpoint (default 29)

    Returns:
        dict with:
            'correlation_vs_time': (n_decode_bins,) Pearson r at each time point
                                   relative to interception
            'time_relative_ms': (n_decode_bins,) time in ms relative to interception
            'decode_from_time': int
    """
    N, C, T = states.shape

    # Get the state at the decode time
    X = states[:, :, decode_from_time].T  # (79, N)

    # Align ball positions to endpoint: for each condition, take the last
    # n_decode_bins positions before trial end
    aligned_positions = np.full((C, n_decode_bins), np.nan)
    for i in range(C):
        dur = int(trial_durations[i])
        if dur < n_decode_bins:
            continue
        # Take last n_decode_bins time points (aligned to endpoint)
        t_start = dur - n_decode_bins
        t_end = dur
        if t_end <= ball_positions_y.shape[1]:
            aligned_positions[i, :] = ball_positions_y[i, t_start:t_end]

    correlations = np.zeros(n_decode_bins)

    for t_rel in range(n_decode_bins):
        y = aligned_positions[:, t_rel]

        valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
        if valid.sum() < 10:
            correlations[t_rel] = np.nan
            continue

        X_v, y_v = X[valid], y[valid]
        y_pred = _nested_cv_ridge(X_v, y_v, n_outer_folds, n_inner_folds)

        valid_pred = np.isfinite(y_pred)
        if valid_pred.sum() > 2:
            r, _ = pearsonr(y_v[valid_pred], y_pred[valid_pred])
            correlations[t_rel] = r
        else:
            correlations[t_rel] = np.nan

    # Time relative to interception (negative = before interception)
    time_relative_ms = (np.arange(n_decode_bins) - n_decode_bins + 1) * 50

    return {
        "correlation_vs_time": correlations,
        "time_relative_ms": time_relative_ms,
        "decode_from_time": decode_from_time,
    }


def _nested_cv_ridge(X: np.ndarray, y: np.ndarray,
                     n_outer_folds: int = 5,
                     n_inner_folds: int = 4) -> np.ndarray:
    """
    Nested cross-validated ridge regression.

    Args:
        X: (n_samples, n_features)
        y: (n_samples,)
        n_outer_folds: outer CV folds
        n_inner_folds: inner CV folds for alpha selection

    Returns:
        y_pred: (n_samples,) cross-validated predictions
    """
    y_pred = np.full(len(y), np.nan)

    outer_cv = KFold(n_splits=min(n_outer_folds, len(y)),
                     shuffle=True, random_state=42)

    for train_idx, test_idx in outer_cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardize
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Inner CV to select alpha
        best_alpha = 1.0
        best_inner_score = -np.inf

        if len(X_train) >= n_inner_folds:
            inner_cv = KFold(n_splits=min(n_inner_folds, len(X_train)),
                             shuffle=True, random_state=43)

            for alpha in RIDGE_ALPHAS:
                inner_scores = []
                for inner_train, inner_test in inner_cv.split(X_train_s):
                    model = Ridge(alpha=alpha)
                    model.fit(X_train_s[inner_train], y_train[inner_train])
                    pred = model.predict(X_train_s[inner_test])
                    valid = np.isfinite(pred) & np.isfinite(y_train[inner_test])
                    if valid.sum() > 1:
                        r, _ = pearsonr(y_train[inner_test][valid], pred[valid])
                        inner_scores.append(r)

                if len(inner_scores) > 0:
                    mean_score = np.mean(inner_scores)
                    if mean_score > best_inner_score:
                        best_inner_score = mean_score
                        best_alpha = alpha

        # Fit on full training set with best alpha
        model = Ridge(alpha=best_alpha)
        model.fit(X_train_s, y_train)
        y_pred[test_idx] = model.predict(X_test_s)

    return y_pred
