"""
Task 3 – UnlabeledLogReg for Heart Disease Dataset
====================================================
Semi-supervised logistic regression that exploits both labeled and
unlabeled observations (Y_obs = -1) to train a FISTA model.

Two Y-completion algorithms:
  1. EM  — iteratively re-estimates soft pseudo-labels for unlabeled rows.
  2. Label Propagation — Gaussian k-NN graph diffusion (closed-form sparse solve).

Three comparison methods:
  - Naive  : train FISTA only on labeled rows (S=0 ⇒ Y_obs ≠ -1)
  - EM / LP: semi-supervised approaches
  - Oracle : train FISTA on all rows with true labels (upper bound)
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))  # project root → common
sys.path.insert(0, _HERE)                       # heart_disease/ → local modules

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

from common.LogisticRegressionFISTA import LogisticRegressionFISTA
from missing_data import generate_missing

MISSING = -1


class UnlabeledLogReg:
    """Semi-supervised logistic regression exploiting unlabeled observations.

    Parameters
    ----------
    method : {"em", "label_propagation"}
    lambda_val : float
    max_iter : int
    tol : float
    fista_max_iter : int
    n_neighbors : int
    random_state : int
    """

    def __init__(
        self,
        method: str = "em",
        lambda_val: float = 1e-3,
        max_iter: int = 20,
        tol: float = 1e-4,
        fista_max_iter: int = 1000,
        n_neighbors: int = 10,
        random_state: int = 42,
    ) -> None:
        if method not in ("em", "label_propagation"):
            raise ValueError("method must be 'em' or 'label_propagation'")
        self.method         = method
        self.lambda_val     = lambda_val
        self.max_iter       = max_iter
        self.tol            = tol
        self.fista_max_iter = fista_max_iter
        self.n_neighbors    = n_neighbors
        self.random_state   = random_state

        self.model_        = None
        self.model_naive_  = None
        self.model_oracle_ = None
        self.y_completed_  = None

    def _fista(self) -> LogisticRegressionFISTA:
        return LogisticRegressionFISTA(
            lambda_val=self.lambda_val,
            max_iter=self.fista_max_iter,
            tol=self.tol,
        )

    @staticmethod
    def _metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
        y_pred = (y_proba >= 0.5).astype(int)
        return {
            "Accuracy":          round(float(accuracy_score(y_true, y_pred)),            4),
            "Balanced Accuracy": round(float(balanced_accuracy_score(y_true, y_pred)),   4),
            "F1":                round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "ROC AUC":           round(float(roc_auc_score(y_true, y_proba)),             4),
        }

    # ── Y-completion: EM ──────────────────────────────────────────────────────

    def _em(self, X: np.ndarray, y_obs: np.ndarray) -> np.ndarray:
        labeled   = y_obs != MISSING
        unlabeled = ~labeled

        if labeled.sum() == 0:
            raise ValueError("EM requires at least one labeled observation.")

        init_model = self._fista().fit(X[labeled], y_obs[labeled])
        y_soft = y_obs.copy()
        y_soft[unlabeled] = init_model.predict_proba(X[unlabeled])
        prev = y_soft[unlabeled].copy()

        for i in range(self.max_iter):
            model = self._fista().fit(X, y_soft)
            new_p = model.predict_proba(X[unlabeled])
            y_soft[unlabeled] = new_p
            delta = float(np.max(np.abs(new_p - prev)))
            prev  = new_p.copy()
            if delta < self.tol:
                print(f"    [EM] converged at iteration {i + 1}  (Δ={delta:.2e})")
                break
        else:
            print(f"    [EM] max_iter={self.max_iter} reached  (Δ={delta:.2e})")

        y_completed = y_obs.copy()
        y_completed[unlabeled] = (new_p >= 0.5).astype(float)
        return y_completed

    # ── Y-completion: Label Propagation ───────────────────────────────────────

    def _label_propagation(self, X: np.ndarray, y_obs: np.ndarray) -> np.ndarray:
        labeled_mask   = y_obs != MISSING
        unlabeled_mask = ~labeled_mask
        labeled_idx    = np.where(labeled_mask)[0]
        unlabeled_idx  = np.where(unlabeled_mask)[0]
        n_labeled      = len(labeled_idx)
        n_unlabeled    = len(unlabeled_idx)

        if n_labeled == 0:
            raise ValueError("Label Propagation requires at least one labeled observation.")
        if n_unlabeled == 0:
            return y_obs.copy()

        k_eff = min(self.n_neighbors, n_labeled - 1)
        knn = kneighbors_graph(X, n_neighbors=k_eff, mode="distance",
                               include_self=False, n_jobs=-1)
        knn = (knn + knn.T) / 2.0

        sigma = float(np.median(knn.data)) if len(knn.data) > 0 else 1.0
        knn.data = np.exp(-(knn.data ** 2) / (sigma ** 2 + 1e-10))

        row_sums = np.asarray(knn.sum(axis=1)).ravel()
        row_sums[row_sums == 0.0] = 1.0
        W = sp.diags(1.0 / row_sums, format="csr") @ knn

        order = np.concatenate([labeled_idx, unlabeled_idx])
        W_ord = W[order][:, order].tocsr()
        W_ul  = W_ord[n_labeled:, :n_labeled]
        W_uu  = W_ord[n_labeled:, n_labeled:]
        f_l   = y_obs[labeled_idx].astype(np.float64)

        A   = sp.eye(n_unlabeled, format="csc") - W_uu.tocsc()
        b   = np.asarray(W_ul @ f_l).ravel()
        f_u = np.clip(spsolve(A, b), 0.0, 1.0)

        y_completed = y_obs.copy()
        y_completed[unlabeled_idx] = (f_u >= 0.5).astype(float)
        return y_completed

    # ── Public: fit methods ───────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y_obs: np.ndarray) -> "UnlabeledLogReg":
        X     = np.asarray(X,     dtype=np.float64)
        y_obs = np.asarray(y_obs, dtype=np.float64)
        y_completed = self._em(X, y_obs) if self.method == "em" else self._label_propagation(X, y_obs)
        self.y_completed_ = y_completed
        self.model_       = self._fista().fit(X, y_completed)
        return self

    def naive_fit(self, X: np.ndarray, y_obs: np.ndarray) -> "UnlabeledLogReg":
        X     = np.asarray(X,     dtype=np.float64)
        y_obs = np.asarray(y_obs, dtype=np.float64)
        labeled = y_obs != MISSING
        self.model_naive_ = self._fista().fit(X[labeled], y_obs[labeled])
        return self

    def oracle_fit(self, X: np.ndarray, y_true: np.ndarray) -> "UnlabeledLogReg":
        X      = np.asarray(X,      dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.float64)
        self.model_oracle_ = self._fista().fit(X, y_true)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Call fit() first.")
        return self.model_.predict_proba(np.asarray(X, dtype=np.float64))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    # ── Public: evaluation ────────────────────────────────────────────────────

    def compare(self, X_test: np.ndarray, y_test: np.ndarray, verbose: bool = True) -> dict:
        X_test = np.asarray(X_test, dtype=np.float64)
        y_test = np.asarray(y_test, dtype=np.float64)
        results = {}
        for name, mdl in [
            (f"UnlabeledLogReg ({self.method})", self.model_),
            ("Naive",                             self.model_naive_),
            ("Oracle",                            self.model_oracle_),
        ]:
            if mdl is None:
                continue
            scores = self._metrics(y_test, mdl.predict_proba(X_test))
            results[name] = scores
            if verbose:
                print(f"  {name:40s}: {scores}")
        return results

    # ── Public: experiments ───────────────────────────────────────────────────

    def run_schemes(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        c: float = 0.3,
        feature_idx: int = 0,
        y_weight: float = 2.0,
        verbose: bool = True,
    ) -> pd.DataFrame:
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64)
        X_test  = np.asarray(X_test,  dtype=np.float64)
        y_test  = np.asarray(y_test,  dtype=np.float64)

        schemes = {
            "MCAR": {"scheme": "mcar"},
            "MAR1": {"scheme": "mar1", "feature_idx": feature_idx},
            "MAR2": {"scheme": "mar2"},
            "MNAR": {"scheme": "mnar", "feature_idx": feature_idx, "y_weight": y_weight},
        }

        self.oracle_fit(X_train, y_train)

        rows = []
        for scheme_name, kwargs in schemes.items():
            if verbose:
                print("=" * 60)
                print(f"Scheme: {scheme_name}  (c={c})")
                print("=" * 60)

            y_obs = generate_missing(X_train, y_train, c=c,
                                     random_state=self.random_state, **kwargs)
            n_miss = int((y_obs == MISSING).sum())
            if verbose:
                print(f"  Missing: {n_miss}/{len(y_obs)} ({n_miss / len(y_obs):.1%})\n")

            self.naive_fit(X_train, y_obs)
            naive_scores = self._metrics(y_test, self.model_naive_.predict_proba(X_test))
            if verbose:
                print(f"  Naive  : {naive_scores}")

            self.fit(X_train, y_obs)
            ulr_scores = self._metrics(y_test, self.predict_proba(X_test))
            method_label = "EM" if self.method == "em" else "Label Prop"
            if verbose:
                print(f"  {method_label:11s}: {ulr_scores}")

            oracle_scores = self._metrics(y_test, self.model_oracle_.predict_proba(X_test))
            if verbose:
                print(f"  Oracle : {oracle_scores}\n")

            for mname, scores in [
                ("Naive",      naive_scores),
                (method_label, ulr_scores),
                ("Oracle",     oracle_scores),
            ]:
                rows.append({"Scheme": scheme_name, "Method": mname, **scores})

        df_out = pd.DataFrame(rows).set_index(["Scheme", "Method"])
        if verbose:
            print("\n=== FULL RESULTS ===")
            print(df_out.to_string())
        return df_out

    def run_mcar_sensitivity(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        c_values: list[float] | None = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        if c_values is None:
            c_values = [0.1, 0.2, 0.3, 0.4, 0.5]

        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64)
        X_test  = np.asarray(X_test,  dtype=np.float64)
        y_test  = np.asarray(y_test,  dtype=np.float64)

        method_label = "EM" if self.method == "em" else "Label Prop"

        self.oracle_fit(X_train, y_train)
        oracle_scores = self._metrics(y_test, self.model_oracle_.predict_proba(X_test))

        rows = []
        for c in c_values:
            if verbose:
                print(f"MCAR  c={c:.1f}")

            y_obs = generate_missing(X_train, y_train, scheme="mcar", c=c,
                                     random_state=self.random_state)

            self.naive_fit(X_train, y_obs)
            naive_scores = self._metrics(y_test, self.model_naive_.predict_proba(X_test))

            self.fit(X_train, y_obs)
            ulr_scores = self._metrics(y_test, self.predict_proba(X_test))

            if verbose:
                print(f"  Naive      : {naive_scores}")
                print(f"  {method_label:11s}: {ulr_scores}")
                print(f"  Oracle     : {oracle_scores}\n")

            for mname, scores in [
                ("Naive",      naive_scores),
                (method_label, ulr_scores),
                ("Oracle",     oracle_scores),
            ]:
                rows.append({"c": c, "Method": mname, **scores})

        df_out = pd.DataFrame(rows).set_index(["c", "Method"])
        if verbose:
            print("\n=== MCAR SENSITIVITY ===")
            print(df_out.to_string())
        return df_out
