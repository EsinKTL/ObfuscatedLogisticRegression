"""
Task 3 Experiments – ATP Tennis Dataset
========================================
Runs two experiments comparing Naive / UnlabeledLogReg (EM & Label Prop) /
Oracle on the ATP tennis upset-prediction task:

  Experiment 1 – Four missing-label schemes at c = 0.3
      MCAR, MAR1, MAR2, MNAR  ×  {Naive, EM, Oracle}
      MCAR, MAR1, MAR2, MNAR  ×  {Naive, Label Prop, Oracle}

  Experiment 2 – MCAR sensitivity
      c ∈ {0.1, 0.2, 0.3, 0.4, 0.5}  ×  {Naive, EM, Oracle}
      c ∈ {0.1, 0.2, 0.3, 0.4, 0.5}  ×  {Naive, Label Prop, Oracle}

Evaluation metrics (test set only): accuracy, balanced accuracy, F1, ROC AUC.
Missing data affects only the training set labels; the test set is always fully
labeled.

Run from the *atp_tennis/* directory:
    python run_exp_atp.py

Or from the project root:
    python atp_tennis/run_exp_atp.py
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))  # project root → common
sys.path.insert(0, _HERE)                       # atp_tennis/  → scripts, local modules

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from common.FISTASelector import FISTASelector
from common.metrics import Metric
from unlabeled_logreg_atp import UnlabeledLogReg

# ── Configuration ──────────────────────────────────────────────────────────────
PROCESSED_FILE = Path(_HERE) / "processed" / "atp_upset.csv"
RESULTS_DIR    = Path(_HERE)          # save CSVs and PNGs alongside this script
RANDOM_STATE   = 42
MISSING_RATE   = 0.3                  # Experiment 1 missingness rate
C_VALUES       = [0.1, 0.2, 0.3, 0.4, 0.5]

METRICS = ["Accuracy", "Balanced Accuracy", "F1", "ROC AUC"]

# ── Colour palette (consistent across all plots) ───────────────────────────────
_PALETTE = {
    "Naive":      "#4C72B0",   # blue
    "EM":         "#DD8452",   # orange
    "Label Prop": "#55A868",   # green
    "Oracle":     "#C44E52",   # red
}


# ── Plotting helpers ───────────────────────────────────────────────────────────

def _bar_color(method: str) -> str:
    return _PALETTE.get(method, "#888888")


def plot_schemes(
    df: pd.DataFrame,
    method_label: str,
    title: str,
    save_path: Path,
) -> None:
    """Grouped bar chart: 4 metrics × 4 schemes, bars grouped by method.

    Parameters
    ----------
    df           : DataFrame indexed by (Scheme, Method)
    method_label : label used for the ULR method ("EM" or "Label Prop")
    title        : figure suptitle
    save_path    : where to write the PNG
    """
    schemes = ["MCAR", "MAR1", "MAR2", "MNAR"]
    # Preserve canonical order: Naive → ULR → Oracle
    methods_present = [m for m in ["Naive", method_label, "Oracle"]
                       if m in df.index.get_level_values("Method").unique()]

    x      = np.arange(len(schemes))
    width  = 0.25
    n_bars = len(methods_present)
    offsets = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * width

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
    axes_flat = axes.flatten()

    for ax, metric in zip(axes_flat, METRICS):
        for offset, mname in zip(offsets, methods_present):
            vals = []
            for scheme in schemes:
                try:
                    vals.append(float(df.loc[(scheme, mname), metric]))
                except KeyError:
                    vals.append(0.0)
            bars = ax.bar(
                x + offset, vals, width,
                label=mname,
                color=_bar_color(mname),
                alpha=0.85,
                edgecolor="white",
                linewidth=0.6,
            )
            # Annotate bar tops
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.004,
                    f"{val:.3f}",
                    ha="center", va="bottom", fontsize=6.5, rotation=90,
                )

        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(schemes, fontsize=10)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Score", fontsize=9)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(axis="y", alpha=0.35, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved → {save_path.name}")


def plot_mcar_sensitivity(
    df: pd.DataFrame,
    method_label: str,
    title: str,
    save_path: Path,
) -> None:
    """Line plot: 4 metrics vs missingness rate c, one line per method.

    Parameters
    ----------
    df           : DataFrame indexed by (c, Method)
    method_label : "EM" or "Label Prop"
    title        : figure suptitle
    save_path    : where to write the PNG
    """
    c_vals  = sorted(df.index.get_level_values("c").unique())
    methods = [m for m in ["Naive", method_label, "Oracle"]
               if m in df.index.get_level_values("Method").unique()]

    markers = {"Naive": "o", "EM": "s", "Label Prop": "D", "Oracle": "^"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    axes_flat = axes.flatten()

    for ax, metric in zip(axes_flat, METRICS):
        for mname in methods:
            vals = []
            for c in c_vals:
                try:
                    vals.append(float(df.loc[(c, mname), metric]))
                except KeyError:
                    vals.append(np.nan)
            ax.plot(
                c_vals, vals,
                marker=markers.get(mname, "x"),
                label=mname,
                color=_bar_color(mname),
                linewidth=2.0,
                markersize=6,
            )

        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.set_xlabel("Missingness rate  c", fontsize=9)
        ax.set_ylabel("Score", fontsize=9)
        ax.set_xticks(c_vals)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.35, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved → {save_path.name}")


def print_table(df: pd.DataFrame, header: str) -> None:
    """Pretty-print a results DataFrame."""
    print(f"\n{'─' * 70}")
    print(f"  {header}")
    print(f"{'─' * 70}")
    print(df.to_string(float_format="{:.4f}".format))
    print()


# ── Data loading & splitting ───────────────────────────────────────────────────

print("=" * 70)
print("ATP Tennis – Task 3 Experiments")
print("=" * 70)

df_data = pd.read_csv(PROCESSED_FILE)
feature_names = [c for c in df_data.columns if c != "y"]
X = df_data[feature_names].values.astype(np.float64)
y = df_data["y"].values.astype(np.float64)

# 60 / 20 / 20  split (test set is never touched during training)
X_tv, X_test, y_tv, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
X_train, X_val, y_train, y_val = train_test_split(
    X_tv, y_tv, test_size=0.25, stratify=y_tv, random_state=RANDOM_STATE
)

print(f"\nDataset : {X.shape[0]:,} samples, {X.shape[1]} features")
print(f"Train   : {X_train.shape[0]:,}   Val: {X_val.shape[0]:,}   Test: {X_test.shape[0]:,}")
print(
    f"Class balance — train: {y_train.mean():.3f}  "
    f"val: {y_val.mean():.3f}  test: {y_test.mean():.3f}\n"
)

# ── Standardise features (fitted on train only — no leakage) ──────────────────
# FISTA's step size is O(1 / ‖X‖_F²).  The raw ATP features span very
# different magnitudes, so standardisation is mandatory for convergence.
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)
print("Features standardised (zero mean, unit variance).\n")

# ── Lambda selection (on oracle/fully-labeled data) ───────────────────────────
# We pick one shared lambda for all UnlabeledLogReg models so that comparisons
# are fair.  Tuning is done on the clean training split.

print("Selecting lambda via FISTASelector (AUC-ROC on validation set) …")
sel = FISTASelector(
    lambdas=np.logspace(-4, 1, 20),
    max_iter=1000,
    tol=1e-4,
)
sel.fit(X_train, y_train, X_val, y_val, measure=Metric.AUC_ROC)
BEST_LAMBDA = sel.best_lambda
print(f"Best lambda: {BEST_LAMBDA:.6f}\n")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 – Four missing-label schemes at c = 0.3
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print(f"EXPERIMENT 1 – Four schemes  (c = {MISSING_RATE})")
print("=" * 70)

exp1_results: dict[str, pd.DataFrame] = {}

for method in ("em", "label_propagation"):
    method_label = "EM" if method == "em" else "Label Prop"
    print(f"\n>>> Method: {method_label}")

    model = UnlabeledLogReg(
        method=method,
        lambda_val=BEST_LAMBDA,
        max_iter=20,
        tol=1e-4,
        fista_max_iter=1000,
        n_neighbors=10,
        random_state=RANDOM_STATE,
    )

    df_schemes = model.run_schemes(
        X_train, y_train, X_test, y_test,
        feature_names=feature_names,
        c=MISSING_RATE,
        verbose=True,
    )

    # Save CSV
    csv_path = RESULTS_DIR / f"results_schemes_{method}.csv"
    df_schemes.to_csv(csv_path)
    print(f"  Saved → {csv_path.name}")

    # Plot
    plot_schemes(
        df_schemes,
        method_label=method_label,
        title=(
            f"ATP Tennis – Task 3 | {method_label}\n"
            f"Four missing-label schemes  (c = {MISSING_RATE})"
        ),
        save_path=RESULTS_DIR / f"plot_schemes_{method}.png",
    )

    exp1_results[method] = df_schemes
    print_table(df_schemes, f"Scheme comparison – {method_label}")


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 – MCAR sensitivity vs missingness rate c
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("EXPERIMENT 2 – MCAR sensitivity vs  c")
print("=" * 70)

exp2_results: dict[str, pd.DataFrame] = {}

for method in ("em", "label_propagation"):
    method_label = "EM" if method == "em" else "Label Prop"
    print(f"\n>>> Method: {method_label}")

    model = UnlabeledLogReg(
        method=method,
        lambda_val=BEST_LAMBDA,
        max_iter=20,
        tol=1e-4,
        fista_max_iter=1000,
        n_neighbors=10,
        random_state=RANDOM_STATE,
    )

    df_sens = model.run_mcar_sensitivity(
        X_train, y_train, X_test, y_test,
        feature_names=feature_names,
        c_values=C_VALUES,
        verbose=True,
    )

    # Save CSV
    csv_path = RESULTS_DIR / f"results_mcar_sensitivity_{method}.csv"
    df_sens.to_csv(csv_path)
    print(f"  Saved → {csv_path.name}")

    # Plot
    plot_mcar_sensitivity(
        df_sens,
        method_label=method_label,
        title=(
            f"ATP Tennis – Task 3 | {method_label}\n"
            f"MCAR sensitivity: performance vs missingness rate  c"
        ),
        save_path=RESULTS_DIR / f"plot_mcar_sensitivity_{method}.png",
    )

    exp2_results[method] = df_sens
    print_table(df_sens, f"MCAR sensitivity – {method_label}")


# ══════════════════════════════════════════════════════════════════════════════
# Combined summary across methods
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("SUMMARY – Oracle (upper bound) vs best semi-supervised at c=0.3")
print("=" * 70)

for metric in METRICS:
    print(f"\n  {metric}")
    print(f"    {'Scheme':8s}  {'Naive':>8s}  {'EM':>8s}  {'Label Prop':>10s}  {'Oracle':>8s}")
    print("    " + "─" * 50)
    for scheme in ["MCAR", "MAR1", "MAR2", "MNAR"]:
        try:
            naive  = exp1_results["em"].loc[(scheme, "Naive"),      metric]
            em_v   = exp1_results["em"].loc[(scheme, "EM"),          metric]
            lp_v   = exp1_results["label_propagation"].loc[(scheme, "Label Prop"), metric]
            oracle = exp1_results["em"].loc[(scheme, "Oracle"),      metric]
            print(f"    {scheme:8s}  {naive:8.4f}  {em_v:8.4f}  {lp_v:10.4f}  {oracle:8.4f}")
        except KeyError:
            print(f"    {scheme:8s}  (data not available)")

print("\nAll experiments completed successfully.")
print(f"Results saved to: {RESULTS_DIR.resolve()}")
