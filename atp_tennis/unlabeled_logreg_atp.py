"""
Task 3 – UnlabeledLogReg for ATP Tennis Dataset
=================================================
Implements a semi-supervised logistic regression class that uses both labeled
and unlabeled observations (Y_obs = -1) to train a FISTA model.

Two Y-completion algorithms are provided:

  1. **EM** (Expectation–Maximisation)
     Iteratively re-estimates soft pseudo-labels for unlabeled observations
     using the current FISTA model, then re-trains on the full dataset.
     Converges when the maximum change in predicted probabilities is < tol.

  2. **Label Propagation**
     Constructs a Gaussian-weighted k-NN graph over all training points.
     The label information "diffuses" from labeled to unlabeled nodes via the
     closed-form solution  f_u = (I - W_uu)^{-1} W_ul f_l,  solved
     efficiently with a sparse direct solver.

Three methods are available for comparison:
  - Naive  : train FISTA only on labeled rows (S = 0 ⇒ Y_obs ≠ -1).
  - EM / LP: the two semi-supervised approaches.
  - Oracle : train FISTA on all rows with *true* labels (upper bound).

The class mirrors the bank-marketing implementation so the two can be compared
side-by-side in the project report.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))  # project root  → common
sys.path.insert(0, _HERE)                       # atp_tennis/   → scripts

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
from scripts.missing_data import generate_missing as _generate_missing_df

MISSING = -1  # sentinel value for unlabeled observations


# ── Private helper ─────────────────────────────────────────────────────────────

def _apply_missing_scheme(
    X_arr: np.ndarray,
    y_arr: np.ndarray,
    feature_names: list[str],
    scheme: str,
    missing_rate: float,
    random_state: int,
    **kwargs,
) -> np.ndarray:
    """Wrap the ATP missing_data API (which expects DataFrames) for numpy arrays.

    Parameters
    ----------
    X_arr        : (n, p) float64 array
    y_arr        : (n,) int/float array of true labels {0, 1}
    feature_names: column names corresponding to X_arr columns
    scheme       : one of {"MCAR", "MAR1", "MAR2", "MNAR"}
    missing_rate : fraction of labels to hide
    random_state : RNG seed
    **kwargs     : forwarded to the scheme function (e.g. ``feature_col``)

    Returns
    -------
    y_obs : (n,) array — original label where observed, -1 where hidden
    """
    X_df = pd.DataFrame(X_arr, columns=feature_names)
    _, y_obs, _ = _generate_missing_df(
        scheme=scheme,
        X=X_df,
        y=y_arr,
        missing_rate=missing_rate,
        random_state=random_state,
        **kwargs,
    )
    return y_obs.astype(np.float64)


# ── Main class ─────────────────────────────────────────────────────────────────

class UnlabeledLogReg:
    """Semi-supervised logistic regression exploiting unlabeled observations.

    Parameters
    ----------
    method : {"em", "label_propagation"}
        Algorithm used to complete missing labels before fitting FISTA.
    lambda_val : float
        L1 regularisation coefficient for FISTA.
    max_iter : int
        Maximum number of outer EM iterations (ignored for Label Propagation).
    tol : float
        EM convergence criterion (max change in predicted probabilities) and
        FISTA inner stopping tolerance.
    fista_max_iter : int
        Maximum FISTA iterations per internal call.
    n_neighbors : int
        Number of nearest neighbours for the Label-Propagation k-NN graph.
    random_state : int
        RNG seed for reproducibility.
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

        # Fitted models (populated by the respective fit_* methods)
        self.model_        = None   # EM or LP model
        self.model_naive_  = None   # Naive (labeled-only) model
        self.model_oracle_ = None   # Oracle (all true labels) model
        self.y_completed_  = None   # Final completed label vector

    # ── Private: FISTA factory ────────────────────────────────────────────────

    def _fista(self) -> LogisticRegressionFISTA:
        """Return a fresh FISTA instance with this object's hyperparameters."""
        return LogisticRegressionFISTA(
            lambda_val=self.lambda_val,
            max_iter=self.fista_max_iter,
            tol=self.tol,
        )

    # ── Private: metrics ──────────────────────────────────────────────────────

    @staticmethod
    def _metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
        """Compute the four Task-3 evaluation metrics at threshold 0.5."""
        y_pred = (y_proba >= 0.5).astype(int)
        return {
            "Accuracy":          round(float(accuracy_score(y_true, y_pred)),           4),
            "Balanced Accuracy": round(float(balanced_accuracy_score(y_true, y_pred)),  4),
            "F1":                round(float(f1_score(y_true, y_pred, zero_division=0)),4),
            "ROC AUC":           round(float(roc_auc_score(y_true, y_proba)),            4),
        }

    # ── Y-completion: EM ──────────────────────────────────────────────────────

    def _em(self, X: np.ndarray, y_obs: np.ndarray) -> np.ndarray:
        """Expectation–Maximisation label completion.

        E-step : predict P(Y=1 | X, w) for unlabeled observations.
        M-step : re-fit FISTA on the full dataset using soft labels for unlabeled.

        The algorithm terminates early when max|Δp| < tol.

        Parameters
        ----------
        X     : (n, p) training features
        y_obs : (n,) observed labels; -1 for unlabeled

        Returns
        -------
        y_completed : (n,) hard labels (0/1) with unlabeled positions filled in
        """
        labeled   = y_obs != MISSING
        unlabeled = ~labeled

        if labeled.sum() == 0:
            raise ValueError("EM requires at least one labeled observation.")

        # Initialise with model trained on labeled data only
        init_model = self._fista().fit(X[labeled], y_obs[labeled])
        y_soft = y_obs.copy()
        y_soft[unlabeled] = init_model.predict_proba(X[unlabeled])
        prev = y_soft[unlabeled].copy()

        for i in range(self.max_iter):
            # M-step: fit on soft labels
            model = self._fista().fit(X, y_soft)
            # E-step: update soft labels for unlabeled
            new_p = model.predict_proba(X[unlabeled])
            y_soft[unlabeled] = new_p
            delta = float(np.max(np.abs(new_p - prev)))
            prev  = new_p.copy()
            if delta < self.tol:
                print(f"    [EM] converged at iteration {i + 1}  (Δ={delta:.2e})")
                break
        else:
            print(f"    [EM] max_iter={self.max_iter} reached  (Δ={delta:.2e})")

        # Threshold to hard labels for the final FISTA fit
        y_completed = y_obs.copy()
        y_completed[unlabeled] = (new_p >= 0.5).astype(float)
        return y_completed

    # ── Y-completion: Label Propagation ───────────────────────────────────────

    def _label_propagation(self, X: np.ndarray, y_obs: np.ndarray) -> np.ndarray:
        """Graph-based label propagation with Gaussian k-NN weighting.

        Constructs a symmetric k-NN graph with Gaussian edge weights:

            w_ij = exp( -d_ij^2 / sigma^2 )

        then row-normalises to get a transition matrix W.  The labels of
        unlabeled nodes are computed in closed form:

            f_u = (I − W_uu)^{-1} W_ul f_l

        where the linear system is solved efficiently with a sparse direct
        solver (scipy.sparse.linalg.spsolve).

        Parameters
        ----------
        X     : (n, p) training features
        y_obs : (n,) observed labels; -1 for unlabeled

        Returns
        -------
        y_completed : (n,) hard labels (0/1)
        """
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

        # Build Gaussian-weighted k-NN graph (sparse, all n points)
        k_eff = min(self.n_neighbors, n_labeled - 1)
        knn = kneighbors_graph(
            X,
            n_neighbors=k_eff,
            mode="distance",
            include_self=False,
            n_jobs=-1,
        )
        knn = (knn + knn.T) / 2.0  # symmetrise

        # Gaussian kernel: w = exp(-d^2 / sigma^2), sigma = median non-zero distance
        sigma = float(np.median(knn.data)) if len(knn.data) > 0 else 1.0
        knn.data = np.exp(-(knn.data ** 2) / (sigma ** 2 + 1e-10))

        # Row-normalise → transition matrix W
        row_sums = np.asarray(knn.sum(axis=1)).ravel()
        row_sums[row_sums == 0.0] = 1.0
        W = sp.diags(1.0 / row_sums, format="csr") @ knn

        # Reorder rows/cols: labeled nodes first, unlabeled nodes last
        order   = np.concatenate([labeled_idx, unlabeled_idx])
        W_ord   = W[order][:, order].tocsr()
        W_ul    = W_ord[n_labeled:, :n_labeled]   # unlabeled → labeled block
        W_uu    = W_ord[n_labeled:, n_labeled:]   # unlabeled → unlabeled block

        f_l = y_obs[labeled_idx].astype(np.float64)

        # Solve (I − W_uu) f_u = W_ul f_l   (sparse, much cheaper than dense solve)
        A   = sp.eye(n_unlabeled, format="csc") - W_uu.tocsc()
        b   = np.asarray(W_ul @ f_l).ravel()
        f_u = np.clip(spsolve(A, b), 0.0, 1.0)

        y_completed = y_obs.copy()
        y_completed[unlabeled_idx] = (f_u >= 0.5).astype(float)
        return y_completed

    # ── Public: fit methods ───────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y_obs: np.ndarray) -> "UnlabeledLogReg":
        """Complete missing labels with the chosen algorithm, then fit FISTA.

        Parameters
        ----------
        X     : (n_samples, n_features)
        y_obs : (n_samples,) — observed labels in {-1, 0, 1}; -1 = missing

        Returns
        -------
        self
        """
        X     = np.asarray(X,     dtype=np.float64)
        y_obs = np.asarray(y_obs, dtype=np.float64)

        if self.method == "em":
            y_completed = self._em(X, y_obs)
        else:
            y_completed = self._label_propagation(X, y_obs)

        self.y_completed_ = y_completed
        self.model_       = self._fista().fit(X, y_completed)
        return self

    def naive_fit(self, X: np.ndarray, y_obs: np.ndarray) -> "UnlabeledLogReg":
        """Train FISTA only on labeled observations (S = 0 ⇒ Y_obs ≠ −1).

        This is the Naive baseline — no use of unlabeled data.

        Parameters
        ----------
        X     : (n_samples, n_features)
        y_obs : (n_samples,) — observed labels; -1 where missing

        Returns
        -------
        self
        """
        X     = np.asarray(X,     dtype=np.float64)
        y_obs = np.asarray(y_obs, dtype=np.float64)
        labeled = y_obs != MISSING
        self.model_naive_ = self._fista().fit(X[labeled], y_obs[labeled])
        return self

    def oracle_fit(self, X: np.ndarray, y_true: np.ndarray) -> "UnlabeledLogReg":
        """Train FISTA on all observations with their *true* labels.

        Serves as an upper-bound benchmark: it would not be achievable in
        practice because it uses the hidden labels.

        Parameters
        ----------
        X      : (n_samples, n_features)
        y_true : (n_samples,) — true binary labels {0, 1}, no missing values

        Returns
        -------
        self
        """
        X      = np.asarray(X,      dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.float64)
        self.model_oracle_ = self._fista().fit(X, y_true)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Positive-class probabilities from the fitted UnlabeledLogReg model."""
        if self.model_ is None:
            raise RuntimeError("Call fit() first.")
        return self.model_.predict_proba(np.asarray(X, dtype=np.float64))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary predictions from the fitted UnlabeledLogReg model."""
        return (self.predict_proba(X) >= threshold).astype(int)

    # ── Public: evaluation ────────────────────────────────────────────────────

    def compare(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        verbose: bool = True,
    ) -> dict:
        """Evaluate all fitted models on the held-out test set.

        Reports accuracy, balanced accuracy, F1, and ROC AUC.

        Parameters
        ----------
        X_test  : (n_test, n_features)
        y_test  : (n_test,) — true binary labels
        verbose : print a summary table

        Returns
        -------
        dict mapping method name → metrics dict
        """
        X_test = np.asarray(X_test, dtype=np.float64)
        y_test = np.asarray(y_test, dtype=np.float64)

        results = {}
        model_map = [
            (f"UnlabeledLogReg ({self.method})", self.model_),
            ("Naive",                             self.model_naive_),
            ("Oracle",                            self.model_oracle_),
        ]
        for name, mdl in model_map:
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
        feature_names: list[str],
        c: float = 0.3,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Compare Naive / UnlabeledLogReg / Oracle under four missing-label
        schemes (MCAR, MAR1, MAR2, MNAR) at a fixed missingness rate *c*.

        Parameters
        ----------
        X_train, y_train : fully labeled training data (numpy arrays)
        X_test,  y_test  : held-out test data (never modified)
        feature_names    : column names matching X_train columns (needed for the
                           ATP missing_data API which uses DataFrames)
        c                : missingness rate applied to the training labels
        verbose          : print per-scheme progress

        Returns
        -------
        pd.DataFrame indexed by (Scheme, Method) with one column per metric
        """
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64)
        X_test  = np.asarray(X_test,  dtype=np.float64)
        y_test  = np.asarray(y_test,  dtype=np.float64)

        schemes = ["MCAR", "MAR1", "MAR2", "MNAR"]

        # Oracle is the same for all schemes (trained on full y_train)
        self.oracle_fit(X_train, y_train)

        rows = []
        for scheme in schemes:
            if verbose:
                print("=" * 60)
                print(f"Scheme: {scheme}  (c={c})")
                print("=" * 60)

            y_obs = _apply_missing_scheme(
                X_train, y_train, feature_names,
                scheme=scheme, missing_rate=c, random_state=self.random_state,
            )
            n_miss = int((y_obs == MISSING).sum())
            if verbose:
                print(f"  Missing: {n_miss}/{len(y_obs)} ({n_miss / len(y_obs):.1%})\n")

            # ── Naive ──────────────────────────────────────────────────────────
            self.naive_fit(X_train, y_obs)
            naive_scores = self._metrics(y_test, self.model_naive_.predict_proba(X_test))
            if verbose:
                print(f"  Naive  : {naive_scores}")

            # ── UnlabeledLogReg (EM or LP) ─────────────────────────────────────
            self.fit(X_train, y_obs)
            ulr_scores = self._metrics(y_test, self.predict_proba(X_test))
            method_label = "EM" if self.method == "em" else "Label Prop"
            if verbose:
                print(f"  {method_label:11s}: {ulr_scores}")

            # ── Oracle ─────────────────────────────────────────────────────────
            oracle_scores = self._metrics(y_test, self.model_oracle_.predict_proba(X_test))
            if verbose:
                print(f"  Oracle : {oracle_scores}\n")

            for mname, scores in [
                ("Naive",        naive_scores),
                (method_label,   ulr_scores),
                ("Oracle",       oracle_scores),
            ]:
                rows.append({"Scheme": scheme, "Method": mname, **scores})

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
        feature_names: list[str],
        c_values: list[float] | None = None,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Analyse performance vs missingness rate *c* in the MCAR setting.

        Parameters
        ----------
        X_train, y_train : fully labeled training data
        X_test,  y_test  : held-out test data
        feature_names    : column names matching X_train columns
        c_values         : missingness rates to test (default [0.1, 0.2, 0.3, 0.4, 0.5])
        verbose          : print per-c progress

        Returns
        -------
        pd.DataFrame indexed by (c, Method)
        """
        if c_values is None:
            c_values = [0.1, 0.2, 0.3, 0.4, 0.5]

        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64)
        X_test  = np.asarray(X_test,  dtype=np.float64)
        y_test  = np.asarray(y_test,  dtype=np.float64)

        method_label = "EM" if self.method == "em" else "Label Prop"

        # Oracle is constant across c values (always trained on full y_train)
        self.oracle_fit(X_train, y_train)
        oracle_scores = self._metrics(y_test, self.model_oracle_.predict_proba(X_test))

        rows = []
        for c in c_values:
            if verbose:
                print(f"MCAR  c={c:.1f}")

            y_obs = _apply_missing_scheme(
                X_train, y_train, feature_names,
                scheme="MCAR", missing_rate=c, random_state=self.random_state,
            )

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
