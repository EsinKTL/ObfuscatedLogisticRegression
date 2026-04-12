"""
Task 2 – FISTA for ATP Tennis Dataset
======================================
Trains a custom L1-regularised logistic regression (FISTA) on the ATP tennis
upset-prediction task.  Compares with sklearn's L1 LogisticRegression using
the same effective regularisation strength.

Run from the *atp_tennis/* directory:
    python fista_run_atp.py

Or from the project root:
    python atp_tennis/fista_run_atp.py
"""

import os
import sys

# Make sure the project root is on the path so `common` can be imported.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common.FISTASelector import FISTASelector
from common.metrics import Metric, print_evaluation

# ── Configuration ──────────────────────────────────────────────────────────────
PROCESSED_FILE = Path(_HERE) / "processed" / "atp_upset.csv"
MEASURE        = Metric.AUC_ROC
RANDOM_STATE   = 42
N_LAMBDAS      = 30
LAMBDA_MIN     = 1e-4
LAMBDA_MAX     = 10.0

# ── Load preprocessed data ─────────────────────────────────────────────────────
df = pd.read_csv(PROCESSED_FILE)
feature_names = [c for c in df.columns if c != "y"]
X = df[feature_names].values.astype(np.float64)
y = df["y"].values.astype(np.float64)

print(f"Dataset: {X.shape[0]:,} samples, {X.shape[1]} features")
print(f"Positive class ratio: {y.mean():.3f}\n")

# ── 60 / 20 / 20 split ─────────────────────────────────────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, stratify=y, random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
)

print(f"Train : {X_train.shape[0]:,} rows")
print(f"Val   : {X_val.shape[0]:,} rows")
print(f"Test  : {X_test.shape[0]:,} rows")
print(
    f"Positive ratio — train: {y_train.mean():.3f}  "
    f"val: {y_val.mean():.3f}  test: {y_test.mean():.3f}\n"
)

# ── Standardise features ───────────────────────────────────────────────────────
# FISTA is a gradient-based method whose step size depends on the Lipschitz
# constant of the log-loss gradient (∝ ‖X‖_F² / n).  Raw ATP features span
# very different scales (rank diffs ~100s, percentage diffs ~0.01–0.1), so
# standardisation is essential for numerical stability and convergence.
# The scaler is fitted on the training set only to prevent data leakage.
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)
print("Features standardised (zero mean, unit variance).\n")

# ── Lambda selection via FISTASelector ────────────────────────────────────────
# Searches a log-spaced grid of 30 lambda values; picks the one maximising
# AUC-ROC on the validation set.
print(f"Selecting best lambda over {N_LAMBDAS} values "
      f"[{LAMBDA_MIN:.0e}, {LAMBDA_MAX:.0e}] by {MEASURE.name} …")

selector = FISTASelector(
    lambdas=np.logspace(np.log10(LAMBDA_MIN), np.log10(LAMBDA_MAX), N_LAMBDAS),
    max_iter=1000,
    tol=1e-4,
)
selector.fit(X_train, y_train, X_val, y_val, measure=MEASURE)
print(f"Best lambda (FISTA): {selector.best_lambda:.6f}\n")

# ── Evaluate custom FISTA on test set ─────────────────────────────────────────
fista_proba = selector.predict_proba(X_test)
print_evaluation("Custom Logistic Regression (FISTA, L1)", y_test, fista_proba)

# ── sklearn L1 LogReg with equivalent regularisation ──────────────────────────
# sklearn's C parameter equals 1/(n_train * lambda).
C_sklearn = 1.0 / (X_train.shape[0] * selector.best_lambda)
sklearn_model = LogisticRegression(
    penalty="l1",
    solver="saga",
    C=C_sklearn,
    max_iter=3000,
    random_state=RANDOM_STATE,
)
sklearn_model.fit(X_train, y_train)
sk_proba = sklearn_model.predict_proba(X_test)[:, 1]
print_evaluation("Sklearn L1 LogisticRegression", y_test, sk_proba)

# ── Coefficient comparison ─────────────────────────────────────────────────────
fista_coef   = selector.best_model.w[1:]   # w[0] is bias
sklearn_coef = sklearn_model.coef_[0]

print("=== Coefficient Comparison ===")
print(f"{'Feature':30s}  {'FISTA':>10s}  {'sklearn':>10s}")
print("─" * 57)
for fname, w_f, w_sk in zip(feature_names, fista_coef, sklearn_coef):
    marker = " *" if (abs(w_f) < 1e-6) != (abs(w_sk) < 1e-6) else ""
    print(f"{fname:30s}  {w_f:10.4f}  {w_sk:10.4f}{marker}")

n_zero_fista   = int((np.abs(fista_coef)   < 1e-6).sum())
n_zero_sklearn = int((np.abs(sklearn_coef) < 1e-6).sum())
print(
    f"\nZero coefficients — FISTA: {n_zero_fista}/{len(feature_names)}  "
    f"sklearn: {n_zero_sklearn}/{len(feature_names)}"
)
print(f"Bias (FISTA): {selector.best_model.w[0]:.4f}\n")

# ── Regularisation-path plots ─────────────────────────────────────────────────
# Plot 1: validation AUC-ROC vs lambda (log scale)
selector.plot(measure=MEASURE)

# Plot 2: coefficient values vs lambda (regularisation path)
selector.plot_coefficients(feature_names=feature_names)
