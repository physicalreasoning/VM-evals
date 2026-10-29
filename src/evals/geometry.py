"""
Geometry analysis.

Tests whether the population's variance structure factorizes the task's
causal variables (bounce vs. no-bounce, launch direction) onto orthogonal axes.

Method:
  1. Define 4 classes: no-bounce upward, no-bounce downward,
     top-wall bounce, bottom-wall bounce.
  2. PCA on early epoch (0-300ms = first 6 bins at 50ms), retain top 15 PCs.
  3. Project all conditions at all time bins onto these early PCs.
  4. For every pair of PCs and every 50ms bin (up to 500ms, before
     earliest bounce at 550ms):
     a. Compute two discriminant axes:
        - v1: separates upward-launched from downward-launched conditions
        - v2: separates upward-arriving from downward-arriving conditions
              (encodes the bounce contrast: a bounce reverses arrival direction
               relative to launch direction, creating XOR structure)
        Both normalized to unit length.
     b. Score each condition as correct if its projections onto v1 and v2
        have the signs of its class.
     c. Report the cosine angle theta between v1 and v2 (measured over
        [0, 180], with 90 = orthogonal) and 4-way accuracy.
  5. Select the PC pair and time bin with highest 4-way accuracy.
"""

import numpy as np
from sklearn.decomposition import PCA
from typing import Dict, Tuple


def classify_conditions(n_bounces: np.ndarray,
                        vel_y0: np.ndarray) -> np.ndarray:
    """
    Classify 79 conditions into 4 groups.

    Args:
        n_bounces: (79,) number of bounces per condition
        vel_y0: (79,) initial y-velocity per condition

    Returns:
        labels: (79,) class labels 0-3
          0: no-bounce, upward launch (+vy)
          1: no-bounce, downward launch (-vy)
          2: top-wall bounce (launched upward)
          3: bottom-wall bounce (launched downward)
    """
    labels = np.zeros(len(n_bounces), dtype=int)
    for i in range(len(n_bounces)):
        if n_bounces[i] == 0:
            labels[i] = 0 if vel_y0[i] >= 0 else 1
        else:
            labels[i] = 2 if vel_y0[i] >= 0 else 3
    return labels


def geometry_analysis(states: np.ndarray, labels: np.ndarray,
                      n_early_bins: int = 6, n_pcs: int = 15,
                      max_time_bin: int = 10) -> Dict:
    """
    Run the PCA/LDA geometry analysis.

    Args:
        states: (N_units, 79, T) population states
        labels: (79,) class labels from classify_conditions
        n_early_bins: number of early time bins for PCA fitting (6 = 300ms)
        n_pcs: number of PCs to retain (default 15)
        max_time_bin: analyze up to this time bin (10 = 500ms, before
                      earliest bounce at 550ms)

    Returns:
        dict with:
            'best_angle': float — angle between discriminant axes [0-180], 90=orthogonal
            'best_accuracy': float — 4-way classification accuracy at best bin/PC pair
            'best_time_bin': int
            'best_pc_pair': tuple of (int, int)
            'angles_over_time': (max_time_bin,) best angle at each time bin
            'accuracies_over_time': (max_time_bin,) best accuracy at each time bin
    """
    N, C, T = states.shape
    T_use = min(T, max_time_bin)

    # Step 1: PCA on early epoch
    early_data = states[:, :, :n_early_bins]  # (N, 79, n_early_bins)
    X_early = early_data.reshape(N, C * n_early_bins).T  # (79*n_early_bins, N)

    valid = np.isfinite(X_early).all(axis=1)
    X_early_clean = X_early[valid]

    n_comp = min(n_pcs, X_early_clean.shape[0], X_early_clean.shape[1])
    pca = PCA(n_components=n_comp, svd_solver="full")
    pca.fit(X_early_clean)

    # Step 2: Sweep PC pairs and time bins
    best_angle = 0.0
    best_accuracy = 0.0
    best_time_bin = 0
    best_pc_pair = (0, 1)
    angles_over_time = np.full(T_use, np.nan)
    accuracies_over_time = np.full(T_use, np.nan)

    for t in range(T_use):
        X_t = states[:, :, t].T  # (79, N)
        valid_t = np.isfinite(X_t).all(axis=1)
        if valid_t.sum() < 10:
            continue

        X_proj = pca.transform(X_t[valid_t])  # (n_valid, n_comp)
        labels_t = labels[valid_t]

        # Check we have all 4 classes
        if len(np.unique(labels_t)) < 4:
            continue

        best_t_acc = 0.0
        best_t_angle = 0.0
        best_t_pair = (0, 1)

        for i in range(n_comp):
            for j in range(i + 1, n_comp):
                X_pair = X_proj[:, [i, j]]

                angle, acc = _compute_discriminant_geometry(X_pair, labels_t)

                if acc > best_t_acc:
                    best_t_acc = acc
                    best_t_angle = angle
                    best_t_pair = (i, j)

        angles_over_time[t] = best_t_angle
        accuracies_over_time[t] = best_t_acc

        if best_t_acc > best_accuracy:
            best_accuracy = best_t_acc
            best_angle = best_t_angle
            best_time_bin = t
            best_pc_pair = best_t_pair

    return {
        "best_angle": best_angle,
        "best_accuracy": best_accuracy,
        "best_time_bin": best_time_bin,
        "best_pc_pair": best_pc_pair,
        "angles_over_time": angles_over_time,
        "accuracies_over_time": accuracies_over_time,
    }


def _compute_discriminant_geometry(X: np.ndarray, labels: np.ndarray
                                   ) -> Tuple[float, float]:
    """
    Compute the discriminant axes and evaluate geometry.

    Classes:
      0: no-bounce up, 1: no-bounce down, 2: top-wall bounce, 3: bottom-wall bounce

    v1 (launch direction): up (0, 2) vs down (1, 3)
    v2 (arrival direction): upward-arriving (0, 3) vs downward-arriving (1, 2)

    Args:
        X: (n_samples, 2) projected data
        labels: (n_samples,) class labels 0-3

    Returns:
        angle: float in degrees [0, 180], 90 = orthogonal
        accuracy: float, 4-way accuracy based on sign matching
    """
    # Compute class means
    means = {}
    for lab in [0, 1, 2, 3]:
        mask = labels == lab
        if mask.sum() == 0:
            return 0.0, 0.25
        means[lab] = X[mask].mean(axis=0)

    # v1: launch direction axis
    # upward (classes 0, 2) vs downward (classes 1, 3)
    up_mean = (means[0] + means[2]) / 2
    down_mean = (means[1] + means[3]) / 2
    v1 = up_mean - down_mean  # positive direction = upward
    v1_norm = np.linalg.norm(v1)
    if v1_norm < 1e-10:
        return 0.0, 0.25
    v1 = v1 / v1_norm

    # v2: arrival direction axis (the "bounce contrast")
    # v2 separates upward- from downward-arriving conditions
    # A bounce reverses vertical velocity, so:
    #   class 0 (no-bounce, launched up)    → arrives going UP
    #   class 1 (no-bounce, launched down)  → arrives going DOWN
    #   class 2 (top-wall bounce, up)       → bounces off top → arrives going DOWN
    #   class 3 (bottom-wall bounce, down)  → bounces off bottom → arrives going UP
    # Upward-arriving: classes 0, 3.  Downward-arriving: classes 1, 2.
    up_arriving_mean = (means[0] + means[3]) / 2
    down_arriving_mean = (means[1] + means[2]) / 2
    v2 = up_arriving_mean - down_arriving_mean  # positive = upward-arriving
    v2_norm = np.linalg.norm(v2)
    if v2_norm < 1e-10:
        return 0.0, 0.25
    v2 = v2 / v2_norm

    # Angle between v1 and v2, measured over [0, 180] without folding
    # "the sign of each axis is fixed by its definition (upward minus downward)"
    cos_angle = np.clip(np.dot(v1, v2), -1, 1)
    angle = np.degrees(np.arccos(cos_angle))

    # 4-way classification by sign matching
    # For each condition, project onto v1 and v2.
    # Expected signs (v1 = launch direction, v2 = arrival direction):
    #   class 0 (no-bounce up):       v1 > 0 (up launch),  v2 > 0 (up arrival)
    #   class 1 (no-bounce down):     v1 < 0 (down launch), v2 < 0 (down arrival)
    #   class 2 (top-wall bounce):    v1 > 0 (up launch),  v2 < 0 (down arrival)
    #   class 3 (bottom-wall bounce): v1 < 0 (down launch), v2 > 0 (up arrival)
    expected_v1_sign = {0: 1, 1: -1, 2: 1, 3: -1}
    expected_v2_sign = {0: 1, 1: -1, 2: -1, 3: 1}

    proj_v1 = X @ v1  # (n_samples,)
    proj_v2 = X @ v2  # (n_samples,)

    correct = 0
    total = 0
    for idx in range(len(labels)):
        lab = labels[idx]
        sign_v1_correct = (np.sign(proj_v1[idx]) == expected_v1_sign[lab])
        sign_v2_correct = (np.sign(proj_v2[idx]) == expected_v2_sign[lab])
        if sign_v1_correct and sign_v2_correct:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.25

    return angle, accuracy
