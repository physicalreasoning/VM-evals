"""
Behavioral comparison: model predictions vs monkey interception performance.

Compares each system's decoded endpoint predictions against the monkey's
actual paddle placement, measuring whether models make human-like errors.

Metrics:
  - Model-monkey error correlation: do models err on the same conditions?
  - Model vs monkey accuracy (r with true endpoint)
  - Signed error comparison (systematic biases for bounce vs no-bounce)
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, spearmanr
from typing import Dict


RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def behavioral_comparison(states: np.ndarray,
                          true_endpoint_y: np.ndarray,
                          monkey_paddle_y: np.ndarray,
                          n_bounces: np.ndarray,
                          n_folds: int = 5) -> Dict:
    """
    Compare model endpoint predictions against monkey interception behavior.

    For each time bin, decode endpoint from model states via cross-validated
    ridge regression, then compare error patterns to the monkey's errors.

    Args:
        states: (N_units, 79, T) population states
        true_endpoint_y: (79,) true ball y at interception
        monkey_paddle_y: (79,) monkey's paddle y at interception
        n_bounces: (79,) bounce count per condition
        n_folds: CV folds for decoding

    Returns:
        dict with:
            'model_monkey_error_corr': (T,) correlation of signed errors
            'model_accuracy': (T,) model r with true endpoint
            'monkey_accuracy': float, monkey r with true endpoint
            'best_error_corr': float, peak model-monkey error correlation
            'best_error_corr_t': int, time bin of peak
            'bounce_bias_model': (T,) mean signed error on bounce conditions
            'bounce_bias_monkey': float, monkey mean signed error on bounce
    """
    N, C, T = states.shape

    monkey_error = monkey_paddle_y - true_endpoint_y
    r_monkey, _ = pearsonr(monkey_paddle_y, true_endpoint_y)

    bounce = n_bounces > 0
    monkey_bounce_bias = float(np.mean(monkey_error[bounce]))
    monkey_nobounce_bias = float(np.mean(monkey_error[~bounce]))

    model_monkey_corr = np.full(T, np.nan)
    model_accuracy = np.full(T, np.nan)
    model_bounce_bias = np.full(T, np.nan)
    model_nobounce_bias = np.full(T, np.nan)

    for t in range(T):
        X = states[:, :, t].T  # (79, N)
        y = true_endpoint_y

        valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
        if valid.sum() < 10:
            continue

        X_v, y_v = X[valid], y[valid]
        monkey_err_v = monkey_error[valid]
        bounce_v = bounce[valid]

        # Cross-validated predictions
        y_pred = np.full(len(y_v), np.nan)
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        for train_idx, test_idx in kf.split(X_v):
            # Inner CV for alpha selection
            best_alpha = _select_alpha(X_v[train_idx], y_v[train_idx])
            ridge = Ridge(alpha=best_alpha)
            ridge.fit(X_v[train_idx], y_v[train_idx])
            y_pred[test_idx] = ridge.predict(X_v[test_idx])

        pred_valid = np.isfinite(y_pred)
        if pred_valid.sum() < 5:
            continue

        # Model accuracy
        r_model, _ = pearsonr(y_v[pred_valid], y_pred[pred_valid])
        model_accuracy[t] = r_model

        # Model errors
        model_error = y_pred - y_v

        # Error correlation with monkey
        r_err, _ = pearsonr(model_error[pred_valid], monkey_err_v[pred_valid])
        model_monkey_corr[t] = r_err

        # Bounce bias
        if bounce_v[pred_valid].sum() > 2:
            model_bounce_bias[t] = float(np.mean(model_error[pred_valid & bounce_v]))
        if (~bounce_v[pred_valid]).sum() > 2:
            model_nobounce_bias[t] = float(np.mean(model_error[pred_valid & ~bounce_v]))

    best_t = np.nanargmax(model_monkey_corr)

    return {
        "model_monkey_error_corr": model_monkey_corr,
        "model_accuracy": model_accuracy,
        "monkey_accuracy": float(r_monkey),
        "best_error_corr": float(np.nanmax(model_monkey_corr)),
        "best_error_corr_t": int(best_t),
        "bounce_bias_monkey": monkey_bounce_bias,
        "nobounce_bias_monkey": monkey_nobounce_bias,
        "bounce_bias_model": model_bounce_bias,
        "nobounce_bias_model": model_nobounce_bias,
    }


def _select_alpha(X_train, y_train, n_inner=4):
    """Select best ridge alpha via inner CV."""
    if len(X_train) < n_inner:
        return 1.0

    best_alpha = 1.0
    best_score = -np.inf

    kf = KFold(n_splits=n_inner, shuffle=True, random_state=0)
    for alpha in RIDGE_ALPHAS:
        scores = []
        for tr, val in kf.split(X_train):
            ridge = Ridge(alpha=alpha)
            ridge.fit(X_train[tr], y_train[tr])
            pred = ridge.predict(X_train[val])
            if np.isfinite(pred).all() and len(pred) > 2:
                r, _ = pearsonr(y_train[val], pred)
                scores.append(r)
        if scores and np.mean(scores) > best_score:
            best_score = np.mean(scores)
            best_alpha = alpha

    return best_alpha
