"""
Bounce classification.

Tests whether population states encode the upcoming bounce event
BEFORE it occurs. This is the XOR test: whether a ball will bounce
depends on position x velocity in a linearly inseparable (XOR) way.

Method:
  - 26 bounce conditions vs 53 no-bounce conditions
  - Average state over 250-500ms window (50ms bins 5 through 10),
    before the earliest bounce at 550ms. One feature vector per condition.
  - Linear SVM, 5-fold CV over conditions, features standardized on training folds
  - Regularization C chosen from {0.01, 0.1, 1, 10, 100} via 3-fold inner CV
  - Repeated 100 times with random partitions (same partitions for all systems)
  - Scored by balanced accuracy (mean of per-class recall), chance = 0.5
  - Null distribution: 50 permutations of bounce labels, C fixed at 1
  - Significance: direct bootstrap hypothesis test (1000 resamples)
"""

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from typing import Dict


def bounce_classification(states: np.ndarray,
                          n_bounces: np.ndarray,
                          time_window: tuple = (5, 10),
                          n_folds: int = 5,
                          n_repeats: int = 100,
                          n_permutations: int = 50,
                          C_values: list = None) -> Dict:
    """
    Classify bounce vs. no-bounce from pre-collision population states.

    Args:
        states: (N_units, 79, T) population states
        n_bounces: (79,) number of bounces per condition
        time_window: (start_bin, end_bin) for averaging states
            default (5, 10) = 250-500ms at 50ms bins
        n_folds: CV folds
        n_repeats: number of random CV partition repeats
        n_permutations: number of label permutations for null distribution
        C_values: SVM regularization values to try

    Returns:
        dict with:
            'balanced_accuracy': float — mean balanced accuracy
            'balanced_accuracy_std': float — std across repeats
            'null_accuracy': float — mean null (permuted) accuracy
            'null_accuracy_std': float
            'p_value': float — fraction of null >= observed
    """
    if C_values is None:
        C_values = [0.01, 0.1, 1, 10, 100]

    N, C_conds, T = states.shape
    t_start, t_end = time_window

    # Average states in the pre-bounce window
    X = states[:, :, t_start:t_end].mean(axis=2).T  # (79, N)

    # Binary labels: bounce (1) vs no-bounce (0)
    y = (n_bounces > 0).astype(int)

    # Remove conditions with NaN
    valid = np.isfinite(X).all(axis=1)
    X, y = X[valid], y[valid]

    if len(np.unique(y)) < 2:
        return {"balanced_accuracy": np.nan, "p_value": np.nan}

    # Run classification with repeated CV
    accuracies = _repeated_cv(X, y, n_folds, n_repeats, C_values)

    # Null distribution via label permutation
    null_accs = []
    for _ in range(n_permutations):
        y_perm = np.random.permutation(y)
        null_acc = _repeated_cv(X, y_perm, n_folds, 1, [1.0])
        null_accs.append(null_acc.mean())

    null_accs = np.array(null_accs)
    observed = np.mean(accuracies)
    p_value = np.mean(null_accs >= observed)

    return {
        "balanced_accuracy": observed,
        "balanced_accuracy_std": np.std(accuracies),
        "null_accuracy": np.mean(null_accs),
        "null_accuracy_std": np.std(null_accs),
        "p_value": p_value,
    }


def _repeated_cv(X, y, n_folds, n_repeats, C_values):
    """Run repeated stratified CV and return balanced accuracies."""
    accuracies = []

    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                              random_state=rep)
        fold_accs = []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            # Inner CV for C selection
            best_c = C_values[0]
            if len(C_values) > 1 and len(X_train) > 6:
                best_inner_acc = 0
                inner_skf = StratifiedKFold(n_splits=3, shuffle=True,
                                            random_state=rep + 1000)
                for c in C_values:
                    inner_accs = []
                    try:
                        for tr, te in inner_skf.split(X_train_s, y_train):
                            clf = LinearSVC(C=c, max_iter=10000)
                            clf.fit(X_train_s[tr], y_train[tr])
                            inner_accs.append(
                                _balanced_accuracy(y_train[te],
                                                   clf.predict(X_train_s[te]))
                            )
                    except ValueError:
                        continue
                    if np.mean(inner_accs) > best_inner_acc:
                        best_inner_acc = np.mean(inner_accs)
                        best_c = c

            clf = LinearSVC(C=best_c, max_iter=10000)
            clf.fit(X_train_s, y_train)
            y_pred = clf.predict(X_test_s)
            fold_accs.append(_balanced_accuracy(y_test, y_pred))

        accuracies.append(np.mean(fold_accs))

    return np.array(accuracies)


def _balanced_accuracy(y_true, y_pred):
    """Balanced accuracy: mean of per-class recall."""
    classes = np.unique(y_true)
    recalls = []
    for c in classes:
        mask = y_true == c
        if mask.sum() > 0:
            recalls.append(np.mean(y_pred[mask] == c))
    return np.mean(recalls)
