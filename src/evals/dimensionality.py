"""
Dimensionality analysis.

1. Covariance eigenspectrum — % variance explained per principal component
2. Time-resolved effective dimensionality via participation ratio (PR)
"""

import numpy as np
from sklearn.decomposition import PCA
from typing import Dict


def eigenspectrum(states: np.ndarray, n_components: int = 30,
                  n_shuffles: int = 20) -> Dict[str, np.ndarray]:
    """
    Compute the covariance eigenspectrum.

    Args:
        states: (N_units, 79, T) — neural or model hidden states
        n_components: number of PCs to report
        n_shuffles: number of shuffles for baseline

    Returns:
        dict with:
            'variance_explained': (n_components,) % variance per PC
            'shuffle_mean': (n_components,) mean shuffled baseline
            'shuffle_std': (n_components,) std of shuffled baseline
    """
    N, C, T = states.shape

    # Flatten to (C*T, N) — each row is one condition at one time bin
    X = states.reshape(N, C * T).T  # (C*T, N)

    # Remove rows with NaN
    valid = np.isfinite(X).all(axis=1)
    X_valid = X[valid]

    n_comp = min(n_components, X_valid.shape[0], X_valid.shape[1])
    pca = PCA(n_components=n_comp, svd_solver="full")
    pca.fit(X_valid)
    var_exp = pca.explained_variance_ratio_ * 100

    # Shuffled baseline: global element-wise permutation
    shuffle_vars = []
    for _ in range(n_shuffles):
        X_shuf = X_valid.copy()
        flat = X_shuf.flatten()
        np.random.shuffle(flat)
        X_shuf = flat.reshape(X_shuf.shape)

        pca_shuf = PCA(n_components=n_comp, svd_solver="full")
        pca_shuf.fit(X_shuf)
        shuffle_vars.append(pca_shuf.explained_variance_ratio_ * 100)

    shuffle_vars = np.array(shuffle_vars)

    return {
        "variance_explained": var_exp,
        "shuffle_mean": shuffle_vars.mean(axis=0),
        "shuffle_std": shuffle_vars.std(axis=0),
    }


def participation_ratio(states: np.ndarray, bin_size_ms: int = 50,
                        n_top_components: int = 100,
                        min_active_frac: float = 0.9) -> Dict[str, np.ndarray]:
    """
    Time-resolved effective dimensionality via participation ratio.

    Args:
        states: (N_units, 79, T)
        bin_size_ms: time bin size for labeling
        n_top_components: number of PCs for global PCA (default 100)
        min_active_frac: minimum fraction of conditions active at a time bin
                         to include it in the global PCA (default 0.9)

    Returns:
        dict with:
            'pr': (T,) participation ratio at each time bin
            'time_ms': (T,) time in ms from trial start
    """
    N, C, T = states.shape

    # Determine which time bins have >= 90% of conditions active
    active_per_bin = np.zeros(T)
    for t in range(T):
        X_t = states[:, :, t].T  # (79, N)
        active_per_bin[t] = np.isfinite(X_t).all(axis=1).sum() / C

    # Cut off at last bin where >= min_active_frac conditions are active
    valid_bins = np.where(active_per_bin >= min_active_frac)[0]
    if len(valid_bins) == 0:
        return {"pr": np.full(T, np.nan), "time_ms": np.arange(T) * bin_size_ms}

    t_cutoff = valid_bins[-1] + 1

    # Build global input matrix: (79 * t_cutoff, N)
    X_global = states[:, :, :t_cutoff].reshape(N, C * t_cutoff).T
    valid_global = np.isfinite(X_global).all(axis=1)
    X_clean = X_global[valid_global]

    n_comp = min(n_top_components, X_clean.shape[0], X_clean.shape[1])
    pca = PCA(n_components=n_comp, svd_solver="full")
    pca.fit(X_clean)

    # At each time bin, project 79 conditions and compute per-PC variance
    pr_values = np.zeros(T)
    for t in range(T):
        X_t = states[:, :, t].T  # (79, N)
        valid_t = np.isfinite(X_t).all(axis=1)
        if valid_t.sum() < 3:
            pr_values[t] = np.nan
            continue

        # Project onto global PCs: (n_valid_conditions, n_comp)
        X_proj = pca.transform(X_t[valid_t])

        # lambda_i(t) = variance of PC_i across conditions at time t
        lambdas = np.var(X_proj, axis=0)  # (n_comp,)

        # PR(t) = (sum lambda_i)^2 / sum(lambda_i^2)
        sum_lambda = np.sum(lambdas)
        sum_lambda_sq = np.sum(lambdas ** 2)

        if sum_lambda_sq < 1e-20:
            pr_values[t] = np.nan
        else:
            pr_values[t] = sum_lambda ** 2 / sum_lambda_sq

    time_ms = np.arange(T) * bin_size_ms

    return {
        "pr": pr_values,
        "time_ms": time_ms,
    }
