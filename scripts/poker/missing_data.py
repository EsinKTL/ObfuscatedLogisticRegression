"""
Missing data generation schemes for binary classification experiments.

- MCAR  : Missing Completely At Random — P(S=1 | X, Y) = c
- MAR1  : Missing At Random (single variable) — P(S=1 | X) depends on one feature
- MAR2  : Missing At Random (all variables) — P(S=1 | X) depends on all features
- MNAR  : Missing Not At Random — P(S=1 | X, Y) depends on both X and Y

All functions follow the same interface:
    Input  : X (np.ndarray or pd.DataFrame), y (np.ndarray or pd.Series)
    Output : y_obs (np.ndarray) where missing labels are replaced with -1
"""

import numpy as np
import pandas as pd
from typing import Union

ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]


def _to_numpy(arr: ArrayLike) -> np.ndarray:
    if isinstance(arr, (pd.DataFrame, pd.Series)):
        return arr.to_numpy()
    return np.asarray(arr)


def generate_mcar(
    X: ArrayLike,
    y: ArrayLike,
    c: float = 0.2,
    random_state: int = None,
) -> np.ndarray:
    """
    Generate missing labels under the MCAR mechanism.

    Each label is independently masked with probability `c`, regardless
    of the feature values or the true label.

    Parameters:
    X : array-like of shape (n_samples, n_features)
        Feature matrix (not used in masking, included for interface consistency).
    y : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    c : float, default=0.2
        Probability that a label is missing, i.e. P(S=1) = c.
        Must be in the interval [0, 1).
    random_state : int or None, default=None
        Seed for the random number generator.

    Returns:
    y_obs : np.ndarray of shape (n_samples,)
        Observed labels: true label if S=0, -1 if S=1 (missing).
    """
    if not 0.0 <= c < 1.0:
        raise ValueError(f"`c` must be in [0, 1), got {c}.")

    rng = np.random.default_rng(random_state)
    y = _to_numpy(y).copy()
    n = len(y)

    s = rng.random(n) < c          # S=1 where True; label is missing
    y_obs = y.copy().astype(float)
    y_obs[s] = -1
    return y_obs


def generate_mar1(
    X: ArrayLike,
    y: ArrayLike,
    feature_idx: int = 0,
    c: float = 0.2,
    random_state: int = None,
) -> np.ndarray:
    """
    Generate missing labels under the MAR mechanism using a single feature.

    The probability of missingness is modeled via a logistic function applied
    to one selected feature:

        P(S=1 | X) = sigmoid(alpha * (X_j - median(X_j)))

    The scalar `alpha` is chosen so that the expected proportion of missing
    labels equals `c`.

    Parameters:
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    feature_idx : int, default=0
        Index of the feature used to drive missingness.
    c : float, default=0.2
        Target expected proportion of missing labels. Must be in [0, 1).
    random_state : int or None, default=None
        Seed for the random number generator.

    Returns:
    y_obs : np.ndarray of shape (n_samples,)
        Observed labels: true label if S=0, -1 if S=1 (missing).

    """
    if not 0.0 <= c < 1.0:
        raise ValueError(f"`c` must be in [0, 1), got {c}.")

    rng = np.random.default_rng(random_state)
    X = _to_numpy(X)
    y = _to_numpy(y).copy()

    x_j = X[:, feature_idx]
    x_centered = x_j - np.median(x_j)

    # Normalize by std so alpha search works regardless of feature scale
    std = x_centered.std()
    if std > 0:
        x_centered = x_centered / std

    # Binary-search for alpha such that mean(sigmoid(alpha * x_centered)) ≈ c
    def _mean_prob(alpha):
        return np.mean(1.0 / (1.0 + np.exp(-alpha * x_centered)))

    lo, hi = -20.0, 20.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if _mean_prob(mid) < c:
            lo = mid
        else:
            hi = mid
    alpha = (lo + hi) / 2.0

    prob_missing = 1.0 / (1.0 + np.exp(-alpha * x_centered))
    s = rng.random(len(y)) < prob_missing
    y_obs = y.copy().astype(float)
    y_obs[s] = -1
    return y_obs


def generate_mar2(
        X: ArrayLike,
        y: ArrayLike,
        c: float = 0.2,
        random_state: int = None,
) -> np.ndarray:
    """
    Generate missing labels under the MAR mechanism using all features.

    The probability of missingness is modeled via a logistic function applied
    to a random linear combination of all features:

        z_i = w^T X_i,   w N(0, I)
        P(S=1 | X_i) = sigmoid(alpha * (z_i - median(z)))

    Parameters:
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    c : float, default=0.2
        Target expected proportion of missing labels. Must be in [0, 1).
    random_state : int or None, default=None
        Seed for the random number generator.

    Returns:
    y_obs : np.ndarray of shape (n_samples,)
        Observed labels: true label if S=0, -1 if S=1 (missing).

    """
    if not 0.0 <= c < 1.0:
        raise ValueError(f"`c` must be in [0, 1), got {c}.")

    rng = np.random.default_rng(random_state)
    X = _to_numpy(X)
    y = _to_numpy(y).copy()

    # Find features whose min_prob (at alpha=-20) is below c.
    # These are the only features that can drive missingness to the target level.
    # We then take a random weighted combination of those features.
    candidates = []
    for j in range(X.shape[1]):
        col = X[:, j].astype(float)
        std_j = col.std()
        if std_j == 0:
            continue
        xc = (col - np.median(col)) / std_j
        min_prob = np.mean(1.0 / (1.0 + np.exp(20.0 * xc)))  # alpha=-20
        if min_prob <= c + 0.05:
            candidates.append(j)

    if len(candidates) == 0:
        raise ValueError("No feature can drive missingness to the target level c. "
            "Try a higher value of c or use a different dataset.")

    # Random weighted combination of candidate features
    w = rng.standard_normal(len(candidates))
    w /= np.linalg.norm(w)

    X_cand = np.column_stack([
        (X[:, j] - np.median(X[:, j])) / (X[:, j].std() + 1e-8)
        for j in candidates
    ])
    z = X_cand @ w
    z_centered = z - np.median(z)
    std = z_centered.std()
    if std > 0:
        z_centered = z_centered / std

    def _mean_prob(alpha):
        return np.mean(1.0 / (1.0 + np.exp(-alpha * z_centered)))

    lo, hi = -20.0, 20.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if _mean_prob(mid) < c:
            lo = mid
        else:
            hi = mid
    alpha = (lo + hi) / 2.0

    prob_missing = 1.0 / (1.0 + np.exp(-alpha * z_centered))
    s = rng.random(len(y)) < prob_missing
    y_obs = y.copy().astype(float)
    y_obs[s] = -1
    return y_obs

def generate_mnar(
    X: ArrayLike,
    y: ArrayLike,
    feature_idx: int = 0,
    c: float = 0.2,
    y_weight: float = 2.0,
    random_state: int = None,
) -> np.ndarray:
    """
    Generate missing labels under the MNAR mechanism.

    The probability of missingness depends on both the feature values and the
    true (unobserved) label:

        z_i = x_{i, feature_idx} + y_weight * y_i
        P(S=1 | X_i, y_i) = sigmoid(alpha * (z_i - median(z)))

    Because the missingness depends on the true label `y`, this mechanism
    cannot be corrected solely from the observed data — making MNAR the most
    challenging setting.

    Parameters:
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    feature_idx : int, default=0
        Index of the feature combined with the label to drive missingness.
    c : float, default=0.2
        Target expected proportion of missing labels. Must be in [0, 1).
    y_weight : float, default=2.0
        Weight controlling how strongly the true label influences missingness.
        Higher values increase label-dependent bias.
    random_state : int or None, default=None
        Seed for the random number generator.

    Returns:
    y_obs : np.ndarray of shape (n_samples,)
        Observed labels: true label if S=0, -1 if S=1 (missing).
    """
    if not 0.0 <= c < 1.0:
        raise ValueError(f"`c` must be in [0, 1), got {c}.")

    rng = np.random.default_rng(random_state)
    X = _to_numpy(X)
    y = _to_numpy(y).copy()

    x_j = X[:, feature_idx]
    # Linear combination of feature and true label
    z = x_j + y_weight * y
    z_centered = z - np.median(z)
    std = z_centered.std()
    if std > 0:
        z_centered = z_centered / std

    def _mean_prob(alpha):
        return np.mean(1.0 / (1.0 + np.exp(-alpha * z_centered)))

    lo, hi = -20.0, 20.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if _mean_prob(mid) < c:
            lo = mid
        else:
            hi = mid
    alpha = (lo + hi) / 2.0

    prob_missing = 1.0 / (1.0 + np.exp(-alpha * z_centered))
    s = rng.random(len(y)) < prob_missing
    y_obs = y.copy().astype(float)
    y_obs[s] = -1
    return y_obs


def generate_missing(
    X: ArrayLike,
    y: ArrayLike,
    scheme: str,
    c: float = 0.2,
    random_state: int = None,
    **kwargs,
) -> np.ndarray:
    scheme = scheme.lower()
    dispatch = {
        "mcar": generate_mcar,
        "mar1": generate_mar1,
        "mar2": generate_mar2,
        "mnar": generate_mnar,
    }
    if scheme not in dispatch:
        raise ValueError(
            f"Unknown scheme '{scheme}'. Choose from {list(dispatch.keys())}."
        )
    return dispatch[scheme](X, y, c=c, random_state=random_state, **kwargs)