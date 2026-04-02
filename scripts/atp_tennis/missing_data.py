"""
missing_data.py
───────────────
Reusable missing-data generation utilities for the Advanced ML Project 1.

Implements four label-missingness schemes that take a fully labelled dataset
(X, Y) and return (X, Y_obs) where Y_obs = -1 for observations whose label
has been hidden.

Schemes
-------
  • MCAR  – Missing Completely At Random:  P(S=1|X,Y) = c
  • MAR1  – Missing At Random (single X):  P(S=1|X,Y) = P(S=1|X_j)
  • MAR2  – Missing At Random (all X):     P(S=1|X,Y) = P(S=1|X)
  • MNAR  – Missing Not At Random:         P(S=1|X,Y) depends on Y (and X)

Convention
----------
  • S = 1  →  label is *missing*  →  Y_obs = -1
  • S = 0  →  label is *observed* →  Y_obs = Y

All public functions share the same signature and return semantics so they
can be used interchangeably in experiment loops.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────
#  Internal helpers
# ──────────────────────────────────────────────


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z)),
    )


def _apply_mask(
    y: np.ndarray,
    probs: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Given per-observation missingness probabilities, draw S and return Y_obs.

    Parameters
    ----------
    y : array of shape (n,)
        True labels (0 or 1).
    probs : array of shape (n,)
        P(S=1 | ·) for each observation.
    rng : numpy Generator

    Returns
    -------
    y_obs : array of shape (n,)
        Observed labels: original value when S=0, -1 when S=1.
    """
    s = rng.binomial(1, probs)
    y_obs = np.where(s == 1, -1, y)
    return y_obs


def _calibrate_intercept(
    z_raw: np.ndarray,
    target_rate: float,
    tol: float = 1e-4,
    max_iter: int = 200,
) -> float:
    """Find intercept *b* such that mean(sigmoid(z_raw + b)) ≈ target_rate.

    Uses simple bisection.
    """
    lo, hi = -30.0, 30.0
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        rate = _sigmoid(z_raw + mid).mean()
        if rate < target_rate - tol:
            lo = mid
        elif rate > target_rate + tol:
            hi = mid
        else:
            return mid
    return (lo + hi) / 2.0


# ──────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────


def generate_mcar(
    X: pd.DataFrame,
    y: np.ndarray,
    missing_rate: float = 0.3,
    random_state: int = 121553223,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """MCAR: every observation has the same constant probability of its
    label being missing.

        P(S=1 | X, Y) = c = *missing_rate*

    Parameters
    ----------
    X : DataFrame of shape (n, p)
        Feature matrix.
    y : array of shape (n,)
        True labels (0 or 1).
    missing_rate : float
        Fraction of labels to hide (constant probability c).
    random_state : int

    Returns
    -------
    (X, y_obs, probs)
        X is returned unchanged.
        y_obs has -1 where the label was hidden.
        probs is the vector of missingness probabilities (constant here).
    """
    rng = np.random.default_rng(random_state)
    n = len(y)
    probs = np.full(n, missing_rate)
    y_obs = _apply_mask(np.asarray(y), probs, rng)
    return X, y_obs, probs


def generate_mar1(
    X: pd.DataFrame,
    y: np.ndarray,
    missing_rate: float = 0.3,
    feature_col: Optional[str] = None,
    random_state: int = 121553223,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """MAR1: the missingness probability depends on a *single* feature.

        P(S=1 | X, Y) = P(S=1 | X_j) = σ(α · X_j + b)

    where j is either specified via *feature_col* or automatically chosen
    as the feature with the highest absolute correlation with Y, and *b* is
    calibrated so that the overall missing rate is approximately *missing_rate*.

    Parameters
    ----------
    X : DataFrame of shape (n, p)
    y : array of shape (n,)
    missing_rate : float
    feature_col : str, optional
        Which feature drives missingness.  If ``None``, the feature with the
        highest |corr(X_j, Y)| is picked automatically.
    random_state : int

    Returns
    -------
    (X, y_obs, probs)
    """
    rng = np.random.default_rng(random_state)
    y_arr = np.asarray(y, dtype=float)

    # Pick driving feature
    if feature_col is None:
        correlations = X.corrwith(pd.Series(y_arr, index=X.index)).abs()
        feature_col = correlations.idxmax()

    x_j = X[feature_col].values.astype(float)

    # Standardize for numerical stability
    mu, sigma = x_j.mean(), x_j.std()
    if sigma < 1e-10:
        sigma = 1.0
    x_j_std = (x_j - mu) / sigma

    # Coefficient: sign chosen so higher values of x_j → more missingness
    alpha = 1.0

    z_raw = alpha * x_j_std
    intercept = _calibrate_intercept(z_raw, missing_rate)
    probs = _sigmoid(z_raw + intercept)

    y_obs = _apply_mask(y_arr.astype(int), probs, rng)
    return X, y_obs, probs


def generate_mar2(
    X: pd.DataFrame,
    y: np.ndarray,
    missing_rate: float = 0.3,
    random_state: int = 121553223,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """MAR2: the missingness probability depends on *all* features.

        P(S=1 | X, Y) = P(S=1 | X) = σ(X·w + b)

    where w is a random weight vector (drawn once from the provided RNG)
    and *b* is calibrated to achieve the desired overall missing rate.

    Parameters
    ----------
    X : DataFrame of shape (n, p)
    y : array of shape (n,)
    missing_rate : float
    random_state : int

    Returns
    -------
    (X, y_obs, probs)
    """
    rng = np.random.default_rng(random_state)
    y_arr = np.asarray(y, dtype=int)

    # Standardize all features for numerical stability
    X_std = (X - X.mean()) / X.std().replace(0, 1)
    X_vals = X_std.values.astype(float)

    # Random weight vector (unit-norm so scale is controlled)
    w = rng.standard_normal(X_vals.shape[1])
    w = w / (np.linalg.norm(w) + 1e-10)

    z_raw = X_vals @ w
    intercept = _calibrate_intercept(z_raw, missing_rate)
    probs = _sigmoid(z_raw + intercept)

    y_obs = _apply_mask(y_arr, probs, rng)
    return X, y_obs, probs


def generate_mnar(
    X: pd.DataFrame,
    y: np.ndarray,
    missing_rate: float = 0.3,
    y_weight: float = 2.0,
    random_state: int = 121553223,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """MNAR: the missingness probability depends on Y (and on X).

        P(S=1 | X, Y) = σ(X·w + γ·Y + b)

    where γ = *y_weight* controls how strongly Y influences missingness
    (positive γ means positive-class labels are *more likely* to be missing).

    Parameters
    ----------
    X : DataFrame of shape (n, p)
    y : array of shape (n,)
    missing_rate : float
    y_weight : float
        Coefficient for Y in the linear predictor.  Larger absolute values
        make the dependence on Y more pronounced.
    random_state : int

    Returns
    -------
    (X, y_obs, probs)
    """
    rng = np.random.default_rng(random_state)
    y_arr = np.asarray(y, dtype=float)

    # Standardize features
    X_std = (X - X.mean()) / X.std().replace(0, 1)
    X_vals = X_std.values.astype(float)

    # Random weight vector for X (unit-norm)
    w = rng.standard_normal(X_vals.shape[1])
    w = w / (np.linalg.norm(w) + 1e-10)

    z_raw = X_vals @ w + y_weight * y_arr
    intercept = _calibrate_intercept(z_raw, missing_rate)
    probs = _sigmoid(z_raw + intercept)

    y_obs = _apply_mask(y_arr.astype(int), probs, rng)
    return X, y_obs, probs


# ──────────────────────────────────────────────
#  Convenience wrapper
# ──────────────────────────────────────────────

SCHEME_REGISTRY = {
    "MCAR": generate_mcar,
    "MAR1": generate_mar1,
    "MAR2": generate_mar2,
    "MNAR": generate_mnar,
}


def generate_missing(
    scheme: str,
    X: pd.DataFrame,
    y: np.ndarray,
    missing_rate: float = 0.3,
    random_state: int = 121553223,
    **kwargs,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Dispatch to the requested missingness scheme.

    Parameters
    ----------
    scheme : {"MCAR", "MAR1", "MAR2", "MNAR"}
    X, y, missing_rate, random_state :
        Forwarded to the scheme function.
    **kwargs :
        Additional keyword arguments forwarded to the scheme function
        (e.g. ``feature_col`` for MAR1, ``y_weight`` for MNAR).

    Returns
    -------
    (X, y_obs, probs)
    """
    if scheme not in SCHEME_REGISTRY:
        raise ValueError(
            f"Unknown scheme '{scheme}'. Choose from {list(SCHEME_REGISTRY.keys())}"
        )
    fn = SCHEME_REGISTRY[scheme]
    return fn(X, y, missing_rate=missing_rate, random_state=random_state, **kwargs)


# ──────────────────────────────────────────────
#  Summary / diagnostic helpers
# ──────────────────────────────────────────────


def missingness_summary(
    y_true: np.ndarray,
    y_obs: np.ndarray,
) -> dict:
    """Return a quick diagnostic dict about the generated missingness.

    Parameters
    ----------
    y_true : array
        Ground-truth labels.
    y_obs : array
        Observed labels (-1 = missing).

    Returns
    -------
    dict with keys:
        n_total, n_missing, n_observed, missing_rate,
        missing_rate_class0, missing_rate_class1
    """
    y_true = np.asarray(y_true)
    y_obs = np.asarray(y_obs)
    s = (y_obs == -1).astype(int)

    n = len(y_true)
    n_missing = int(s.sum())
    n_observed = n - n_missing

    # Per-class missing rates
    mask0 = y_true == 0
    mask1 = y_true == 1
    mr0 = s[mask0].mean() if mask0.any() else float("nan")
    mr1 = s[mask1].mean() if mask1.any() else float("nan")

    return {
        "n_total": n,
        "n_missing": n_missing,
        "n_observed": n_observed,
        "missing_rate": n_missing / n,
        "missing_rate_class0": mr0,
        "missing_rate_class1": mr1,
    }
