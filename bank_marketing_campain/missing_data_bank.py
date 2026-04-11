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


def _calibrate_intercept(score: np.ndarray, c: float) -> float:
    """Find intercept b such that mean(sigmoid(score + b)) == c.

    Uses binary search on b. This is more robust than searching for alpha
    because it works regardless of the score's variance or distribution shape.

    Parameters:
    score : np.ndarray — linear score for each observation
    c     : float      — target mean probability

    Returns:
    b : float — intercept that achieves mean sigmoid output == c
    """
    lo, hi = -500.0, 500.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        mean_p = np.mean(1.0 / (1.0 + np.exp(-np.clip(score + mid, -500, 500))))
        if mean_p < c:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def generate_mcar(
        X: ArrayLike,
        y: ArrayLike,
        c: float = 0.2,
        random_state: int = None,
        **kwargs,
) -> np.ndarray:
    """
    Generate missing labels under the MCAR mechanism.

    Each label is independently masked with probability c, regardless
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
    
    s = rng.random(len(y)) < c
    y_obs = y.copy().astype(float)
    y_obs[s] = -1
    return y_obs


def generate_mar1(
        X: ArrayLike,
        y: ArrayLike,
        feature_idx: int = 0,
        c: float = 0.2,
        random_state: int = None,
        **kwargs,
) -> np.ndarray:
    """
    Generate missing labels under the MAR mechanism using a single feature.

    The probability of missingness is modeled via a logistic function applied
    to one selected feature:

        P(S=1 | X) = sigmoid(score + b),   score = X_j - median(X_j)

    The intercept b is calibrated so that mean(P(S=1)) == c exactly.

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
    
    x_j = X[:, feature_idx].astype(float)
    # Standardize so intercept search range is scale-invariant
    std = x_j.std()
    score = (x_j - np.median(x_j)) / (std if std > 0 else 1.0)
    
    b = _calibrate_intercept(score, c)
    prob_missing = 1.0 / (1.0 + np.exp(-np.clip(score + b, -500, 500)))
    
    s = rng.random(len(y)) < prob_missing
    y_obs = y.copy().astype(float)
    y_obs[s] = -1
    return y_obs


def generate_mar2(
        X: ArrayLike,
        y: ArrayLike,
        c: float = 0.2,
        random_state: int = None,
        **kwargs,
) -> np.ndarray:
    """
    Generate missing labels under the MAR mechanism using all features.

    The probability of missingness is modeled via a logistic function applied
    to a random linear combination of all features:

        z_i = w^T X_i,   w ~ N(0, I) normalized
        P(S=1 | X_i) = sigmoid(z_i + b)

    The intercept b is calibrated so that mean(P(S=1)) == c exactly.

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
    X = _to_numpy(X).astype(float)
    y = _to_numpy(y).copy()
    
    # Standardize each column
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1.0
    X_std = (X - means) / stds
    
    # Random unit-norm weight vector
    w = rng.standard_normal(X_std.shape[1])
    w /= np.linalg.norm(w)
    
    score = X_std @ w
    b = _calibrate_intercept(score, c)
    prob_missing = 1.0 / (1.0 + np.exp(-np.clip(score + b, -500, 500)))
    
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
        **kwargs,
) -> np.ndarray:
    """
    Generate missing labels under the MNAR mechanism.

    The probability of missingness depends on both the feature values and the
    true (unobserved) label:

        z_i = x_{i, feature_idx} + y_weight * y_i
        P(S=1 | X_i, y_i) = sigmoid(z_i + b)

    The intercept b is calibrated so that mean(P(S=1)) == c exactly.
    Because the missingness depends on the true label y, this mechanism
    cannot be corrected solely from the observed data.

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
    X = _to_numpy(X).astype(float)
    y = _to_numpy(y).copy()
    
    x_j = X[:, feature_idx]
    std = x_j.std()
    x_j_std = (x_j - np.median(x_j)) / (std if std > 0 else 1.0)
    
    # Score depends on both X and Y (the defining property of MNAR)
    score = x_j_std + y_weight * y.astype(float)
    
    b = _calibrate_intercept(score, c)
    prob_missing = 1.0 / (1.0 + np.exp(-np.clip(score + b, -500, 500)))
    
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
    """
    Unified entry point for all missing-label schemes.

    Parameters:
    X            : array-like — feature matrix
    y            : array-like — true binary labels (0/1)
    scheme       : str        — one of 'mcar', 'mar1', 'mar2', 'mnar'
    c            : float      — target missingness rate, default 0.2
    random_state : int        — for reproducibility
    **kwargs     : passed to the scheme function
                   mar1 / mnar accept: feature_idx (int)
                   mnar also accepts:  y_weight (float)

    Returns:
    y_obs : np.ndarray — values in {-1, 0, 1}, -1 means label not observed
    """
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