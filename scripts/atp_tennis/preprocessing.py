"""
preprocessing.py
────────────────
Reusable data-preprocessing utilities for the Advanced ML Project 1.

Provides functions for:
  • missing-value imputation  (median for numeric, mode for categorical)
  • collinear-feature removal (VIF-based and correlation-threshold-based)
  • generic DataFrame cleaning helpers

All functions are pure (no side-effects on the input DataFrame unless
explicitly noted) and return new DataFrames / arrays.
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ──────────────────────────────────────────────
#  Missing-value helpers
# ──────────────────────────────────────────────


def impute_missing(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_strategy: str = "mode",
    exclude_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Fill missing values in *df*.

    Parameters
    ----------
    df : DataFrame
        Input data (not modified in-place).
    numeric_strategy : {"median", "mean"}
        How to fill numeric columns.
    categorical_strategy : {"mode", "missing"}
        How to fill categorical columns.  ``"missing"`` inserts a literal
        ``"_MISSING_"`` category.
    exclude_cols : list[str], optional
        Columns to skip entirely.

    Returns
    -------
    DataFrame
        A copy of *df* with missing values filled.
    """
    df = df.copy()
    exclude_cols = set(exclude_cols or [])

    for col in df.columns:
        if col in exclude_cols or df[col].isnull().sum() == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            if numeric_strategy == "median":
                fill_value = df[col].median()
            elif numeric_strategy == "mean":
                fill_value = df[col].mean()
            else:
                raise ValueError(f"Unknown numeric strategy: {numeric_strategy}")
            df[col] = df[col].fillna(fill_value)

        else:  # categorical / object
            if categorical_strategy == "mode":
                mode_vals = df[col].mode()
                fill_value = mode_vals.iloc[0] if len(mode_vals) > 0 else "_MISSING_"
            elif categorical_strategy == "missing":
                fill_value = "_MISSING_"
            else:
                raise ValueError(
                    f"Unknown categorical strategy: {categorical_strategy}"
                )
            df[col] = df[col].fillna(fill_value)

    return df


# ──────────────────────────────────────────────
#  Collinearity removal
# ──────────────────────────────────────────────


def remove_collinear_by_correlation(
    df: pd.DataFrame,
    feature_cols: List[str],
    threshold: float = 0.90,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Drop features whose pairwise Pearson |r| exceeds *threshold*.

    When two features are correlated above the threshold the one with the
    higher *mean* absolute correlation with all other features is dropped
    (i.e. we keep the less "redundant" one).

    Parameters
    ----------
    df : DataFrame
    feature_cols : list[str]
        Numeric columns to check.
    threshold : float
        Absolute correlation ceiling.
    verbose : bool
        Print information about dropped columns.

    Returns
    -------
    (DataFrame, dropped_cols)
        DataFrame with collinear columns removed and a list of the dropped
        column names.
    """
    df = df.copy()
    corr_matrix = df[feature_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1))

    to_drop: set[str] = set()
    for col in upper.columns:
        correlated = upper.index[upper[col] > threshold].tolist()
        for other in correlated:
            if other in to_drop:
                continue
            # drop whichever has higher mean correlation with the rest
            mean_corr_col = corr_matrix[col].drop([col, other]).mean()
            mean_corr_other = corr_matrix[other].drop([col, other]).mean()
            drop_candidate = col if mean_corr_col > mean_corr_other else other
            to_drop.add(drop_candidate)

    to_drop_list = sorted(to_drop)
    if verbose and to_drop_list:
        print(
            f"[correlation] Dropping {len(to_drop_list)} collinear features "
            f"(|r| > {threshold}): {to_drop_list}"
        )

    df = df.drop(columns=to_drop_list)
    remaining = [c for c in feature_cols if c not in to_drop]
    return df, remaining


def remove_collinear_by_vif(
    df: pd.DataFrame,
    feature_cols: List[str],
    threshold: float = 10.0,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Iteratively drop the feature with the highest VIF until all are
    below *threshold*.

    Parameters
    ----------
    df : DataFrame
    feature_cols : list[str]
        Numeric columns to check.
    threshold : float
        Maximum allowable VIF (common choices: 5 or 10).
    verbose : bool

    Returns
    -------
    (DataFrame, remaining_feature_cols)
    """
    df = df.copy()
    remaining = list(feature_cols)
    dropped: list[str] = []

    while True:
        if len(remaining) < 2:
            break

        X = df[remaining].values.astype(np.float64)

        # Guard against constant / near-constant columns
        std = X.std(axis=0)
        const_mask = std < 1e-10
        if const_mask.any():
            const_cols = [remaining[i] for i, m in enumerate(const_mask) if m]
            if verbose:
                print(f"[VIF] Dropping near-constant columns: {const_cols}")
            dropped.extend(const_cols)
            remaining = [c for c in remaining if c not in const_cols]
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vifs = np.array(
                [variance_inflation_factor(X, i) for i in range(X.shape[1])]
            )

        max_vif_idx = int(np.argmax(vifs))
        max_vif = vifs[max_vif_idx]

        if max_vif <= threshold:
            break

        drop_col = remaining[max_vif_idx]
        if verbose:
            print(f"[VIF] Dropping '{drop_col}' (VIF = {max_vif:.1f})")
        dropped.append(drop_col)
        remaining.pop(max_vif_idx)

    df = df.drop(columns=dropped)
    if verbose and dropped:
        print(f"[VIF] Total dropped: {len(dropped)} columns")
    return df, remaining


# ──────────────────────────────────────────────
#  General cleaning helpers
# ──────────────────────────────────────────────


def drop_low_variance(
    df: pd.DataFrame,
    feature_cols: List[str],
    threshold: float = 1e-8,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Remove numeric columns whose variance is below *threshold*."""
    df = df.copy()
    variances = df[feature_cols].var()
    to_drop = variances[variances < threshold].index.tolist()
    if verbose and to_drop:
        print(f"[variance] Dropping {len(to_drop)} low-variance features: {to_drop}")
    remaining = [c for c in feature_cols if c not in to_drop]
    df = df.drop(columns=to_drop)
    return df, remaining


def standardize(
    df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Z-score standardize *feature_cols* in place (returns means & stds)."""
    df = df.copy()
    means = df[feature_cols].mean()
    stds = df[feature_cols].std().replace(0, 1)
    df[feature_cols] = (df[feature_cols] - means) / stds
    return df, means, stds


def full_preprocessing_pipeline(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    impute: bool = True,
    corr_threshold: float = 0.90,
    vif_threshold: float = 10.0,
    do_standardize: bool = False,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Run the full preprocessing pipeline and return the cleaned DataFrame
    together with the final list of feature columns.

    Steps (in order):
      1. Impute missing values (median / mode).
      2. Drop low-variance features.
      3. Remove collinear features (correlation-based).
      4. Remove collinear features (VIF-based).
      5. (Optional) Standardize features.

    Parameters
    ----------
    df : DataFrame
    feature_cols : list[str]
    target_col : str
    impute : bool
    corr_threshold : float
    vif_threshold : float
    do_standardize : bool
    verbose : bool

    Returns
    -------
    (DataFrame, remaining_feature_cols)
    """
    out = df.copy()

    if verbose:
        print(
            f"Starting preprocessing: {len(out)} rows, "
            f"{len(feature_cols)} features, target='{target_col}'"
        )
        print(f"  Missing values: {out[feature_cols].isnull().sum().sum()}")

    # 1. Impute
    if impute:
        out = impute_missing(out, exclude_cols=[target_col])
        if verbose:
            remaining_nulls = out[feature_cols].isnull().sum().sum()
            print(f"  After imputation: {remaining_nulls} missing values remain")

    # 2. Low-variance
    out, feature_cols = drop_low_variance(out, feature_cols, verbose=verbose)

    # 3. Correlation
    out, feature_cols = remove_collinear_by_correlation(
        out,
        feature_cols,
        threshold=corr_threshold,
        verbose=verbose,
    )

    # 4. VIF
    out, feature_cols = remove_collinear_by_vif(
        out,
        feature_cols,
        threshold=vif_threshold,
        verbose=verbose,
    )

    # 5. Standardize
    if do_standardize:
        out, _, _ = standardize(out, feature_cols)

    if verbose:
        print(
            f"Preprocessing complete: {len(out)} rows, "
            f"{len(feature_cols)} features remaining"
        )

    return out, feature_cols
