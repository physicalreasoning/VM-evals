"""
Representational Similarity Analysis (RSA).

Compares the full pairwise similarity structure of population states
across all 79 conditions and time bins between two systems
(e.g., DMFC neural data vs. video model hidden states).

Also supports partial correlation RSA: test whether one model retains
explanatory power after removing variance explained by another.
"""

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import euclidean_distances
from typing import Dict, Optional


def compute_rdm(states: np.ndarray) -> np.ndarray:
    """
    Compute the Representational Dissimilarity Matrix (RDM).

    Args:
        states: (N_units, 79, T) population states

    Returns:
        rdm: (C*T, C*T) pairwise Euclidean distance matrix
             with upper triangle set to NaN
    """
    N, C, T = states.shape

    # Flatten to (C*T, N)
    X = states.reshape(N, C * T).T

    # Handle NaNs: compute distances only for valid pairs
    valid = np.isfinite(X).all(axis=1)
    X_valid = X[valid]

    rdm_full = np.full((C * T, C * T), np.nan)
    valid_idx = np.where(valid)[0]

    rdm_sub = euclidean_distances(X_valid)
    for i, vi in enumerate(valid_idx):
        for j, vj in enumerate(valid_idx):
            rdm_full[vi, vj] = rdm_sub[i, j]

    # Set upper triangle to NaN (avoid double-counting)
    rdm_full[np.triu_indices(rdm_full.shape[0])] = np.nan

    return rdm_full


def rsa_correlation(rdm1: np.ndarray, rdm2: np.ndarray) -> Dict[str, float]:
    """
    Compute Spearman correlation between two RDMs.

    Args:
        rdm1, rdm2: (C*T, C*T) RDMs from compute_rdm()

    Returns:
        dict with 'rho' (Spearman correlation) and 'p_value'
    """
    # Extract lower triangle values
    v1 = rdm1.flatten()
    v2 = rdm2.flatten()

    valid = np.isfinite(v1) & np.isfinite(v2)
    if valid.sum() < 3:
        return {"rho": np.nan, "p_value": np.nan}

    rho, p = spearmanr(v1[valid], v2[valid])

    return {"rho": rho, "p_value": p}


def rsa_compare(neural_states: np.ndarray,
                model_states_dict: Dict[str, np.ndarray]) -> Dict[str, Dict]:
    """
    Compare multiple models against neural data via RSA.

    Args:
        neural_states: (N_neurons, 79, T) DMFC neural data
        model_states_dict: {model_name: (N_units, 79, T)} model hidden states

    Returns:
        {model_name: {'rho': float, 'p_value': float}}
    """
    rdm_neural = compute_rdm(neural_states)

    results = {}
    for name, model_states in model_states_dict.items():
        rdm_model = compute_rdm(model_states)
        results[name] = rsa_correlation(rdm_neural, rdm_model)

    return results


def partial_rsa(rdm_target: np.ndarray, rdm_model: np.ndarray,
                rdm_control: np.ndarray) -> Dict[str, float]:
    """
    Partial correlation RSA: correlation between target and model
    after removing variance explained by control.

    Tests whether a model captures unique structure beyond what
    the control model explains.

    Args:
        rdm_target: (C*T, C*T) neural RDM
        rdm_model: (C*T, C*T) model to test
        rdm_control: (C*T, C*T) model to control for

    Returns:
        dict with 'rho_partial' and 'rho_full'
    """
    v_target = rdm_target.flatten()
    v_model = rdm_model.flatten()
    v_control = rdm_control.flatten()

    valid = (np.isfinite(v_target) & np.isfinite(v_model) &
             np.isfinite(v_control))

    if valid.sum() < 3:
        return {"rho_partial": np.nan, "rho_full": np.nan}

    t, m, c = v_target[valid], v_model[valid], v_control[valid]

    # Full correlation
    rho_full, _ = spearmanr(t, m)

    # Residualize: remove control from both target and model
    # Using OLS regression
    c_design = np.column_stack([c, np.ones_like(c)])

    # Residuals of target after removing control
    beta_t = np.linalg.lstsq(c_design, t, rcond=None)[0]
    resid_t = t - c_design @ beta_t

    # Residuals of model after removing control
    beta_m = np.linalg.lstsq(c_design, m, rcond=None)[0]
    resid_m = m - c_design @ beta_m

    rho_partial, _ = spearmanr(resid_t, resid_m)

    return {
        "rho_partial": rho_partial,
        "rho_full": rho_full,
    }
