"""
Implements all Task 3 requirements inside a single class:
  - Two Y-completion algorithms: EM and Label Propagation
  - Naive method  (train only on labeled rows where S=0)
  - Oracle method (train on full (X, Y) as benchmark)
  - compare()     (evaluate all methods on test set)
  - run_schemes() (compare under MCAR / MAR1 / MAR2 / MNAR)
  - run_mcar_sensitivity() (analyse performance vs c in MCAR setting)
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
)
import scipy.sparse as sp

from fista import LogisticRegressionFISTA, FISTASelector
from missing_data import generate_missing

MISSING = -1


class UnlabeledLogReg:
    """
    Logistic regression that exploits unlabeled observations (Y = -1).

    Parameters
    ----------
    method : str
        'em' or 'label_propagation' — algorithm used to complete missing labels.
    lambda_val : float
        L1 penalty for FISTA.
    max_iter : int
        Max iterations for the completion algorithm.
    tol : float
        Convergence tolerance.
    fista_max_iter : int
        Max iterations for each internal FISTA call.
    n_neighbors : int
        Number of neighbors for Label Propagation k-NN graph.
    random_state : int
    """

    def __init__(
        self,
        method: str = "em",
        lambda_val: float = 1e-3,
        max_iter: int = 20,
        tol: float = 1e-4,
        fista_max_iter: int = 1000,
        n_neighbors: int = 15,
        random_state: int = 42,
    ):
        if method not in ("em", "label_propagation"):
            raise ValueError("method must be 'em' or 'label_propagation'")
        self.method = method
        self.lambda_val = lambda_val
        self.max_iter = max_iter
        self.tol = tol
        self.fista_max_iter = fista_max_iter
        self.n_neighbors = n_neighbors
        self.random_state = random_state

        self.model_         = None   # UnlabeledLogReg (EM or LP) final model
        self.model_naive_   = None   # Naive model
        self.model_oracle_  = None   # Oracle model
        self.y_completed_   = None   # completed labels after algorithm

    # Internal FISTA factory 

    def _fista(self):
        return LogisticRegressionFISTA(
            lambda_val=self.lambda_val,
            max_iter=self.fista_max_iter,
            tol=self.tol,
        )

    # Public fit methods

    def fit(self, X, y_obs):
        """
        Complete missing labels then train FISTA on the full dataset.

        Parameters
        ----------
        X     : array (n_samples, n_features)
        y_obs : array (n_samples,) — labels in {-1, 0, 1}, -1 = missing

        Returns
        -------
        self
        """
        X     = np.asarray(X, dtype=np.float64)
        y_obs = np.asarray(y_obs, dtype=np.float64)

        if self.method == "em":
            y_completed = self._em(X, y_obs)
        else:
            y_completed = self._label_propagation(X, y_obs)

        self.y_completed_ = y_completed
        self.model_ = self._fista().fit(X, y_completed)
        return self

    def naive_fit(self, X, y_obs):
        """
        Naive method: train FISTA only on labeled rows (S = 0).

        Parameters
        ----------
        X     : array (n_samples, n_features)
        y_obs : array (n_samples,) — labels in {-1, 0, 1}

        Returns
        -------
        self
        """
        X     = np.asarray(X, dtype=np.float64)
        y_obs = np.asarray(y_obs, dtype=np.float64)

        labeled = y_obs != MISSING
        self.model_naive_ = self._fista().fit(X[labeled], y_obs[labeled])
        return self

    def oracle_fit(self, X, y_true):
        """
        Oracle method: train FISTA on full (X, Y) — upper-bound benchmark.

        Parameters
        ----------
        X      : array (n_samples, n_features)
        y_true : array (n_samples,) — true binary labels (0/1), no missing

        Returns
        -------
        self
        """
        X      = np.asarray(X, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.float64)

        self.model_oracle_ = self._fista().fit(X, y_true)
        return self

    def predict_proba(self, X):
        """Positive class probability from the UnlabeledLogReg model."""
        if self.model_ is None:
            raise RuntimeError("Call fit() first.")
        return self.model_.predict_proba(X)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    # Evaluation 

    @staticmethod
    def _metrics(y_true, y_proba):
        """Compute accuracy, balanced accuracy, F1, ROC AUC."""
        y_pred = (y_proba >= 0.5).astype(int)
        return {
            "Accuracy":          round(float(accuracy_score(y_true, y_pred)),          4),
            "Balanced Accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
            "F1":                round(float(f1_score(y_true, y_pred, zero_division=0)),4),
            "ROC AUC":           round(float(roc_auc_score(y_true, y_proba)),           4),
        }

    def compare(self, X_test, y_test, verbose=True):
        """
        Evaluate all fitted models on the test set.

        Reports: Accuracy, Balanced Accuracy, F1, ROC AUC.
        Models included depend on which fit methods were called.

        Parameters
        ----------
        X_test  : array (n_test, n_features)
        y_test  : array (n_test,) — true binary labels
        verbose : bool — print results

        Returns
        -------
        dict mapping method name to metrics dict
        """
        X_test = np.asarray(X_test, dtype=np.float64)
        y_test = np.asarray(y_test, dtype=np.float64)

        results = {}
        model_map = [
            ("UnlabeledLogReg (" + self.method + ")", self.model_),
            ("Naive",                                  self.model_naive_),
            ("Oracle",                                 self.model_oracle_),
        ]
        for name, model in model_map:
            if model is None:
                continue
            scores = self._metrics(y_test, model.predict_proba(X_test))
            results[name] = scores
            if verbose:
                print(f"  {name:35s}: {scores}")

        return results

    # Experiment: all four missing schemes

    def run_schemes(
        self,
        X_train, y_train,
        X_test,  y_test,
        c: float = 0.3,
        feature_idx: int = 0,
        y_weight: float = 5.0,
        verbose: bool = True,
    ):
        """
        Compare Naive / UnlabeledLogReg / Oracle on the test set under
        MCAR, MAR1, MAR2, and MNAR missing label schemes.

        Parameters
        ----------
        X_train, y_train : training data (fully labeled)
        X_test,  y_test  : test data (never modified)
        c                : missingness rate applied to y_train
        feature_idx      : feature index used by MAR1 and MNAR
        y_weight         : label weight for MNAR
        verbose          : print results

        Returns
        -------
        pd.DataFrame with columns [Scheme, Method, Accuracy, Balanced Accuracy, F1, ROC AUC]
        """
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

        # Oracle is the same for all schemes (trained on full y_train)
        self.oracle_fit(X_train, y_train)

        rows = []
        for scheme_name, kwargs in schemes.items():
            if verbose:
                print("=" * 60)
                print(f"Scheme: {scheme_name}  (c={c})")
                print("=" * 60)

            y_obs = generate_missing(
                X_train, y_train, c=c, random_state=self.random_state, **kwargs
            )
            n_miss = (y_obs == MISSING).sum()
            if verbose:
                print(f"  Missing: {n_miss} / {len(y_obs)} ({n_miss/len(y_obs):.1%})\n")

            # Naive
            self.naive_fit(X_train, y_obs)
            naive_scores = self._metrics(y_test, self.model_naive_.predict_proba(X_test))
            if verbose:
                print(f"  Naive  : {naive_scores}")

            # UnlabeledLogReg (EM or LP)
            self.fit(X_train, y_obs)
            ulr_scores = self._metrics(y_test, self.predict_proba(X_test))
            if verbose:
                print(f"  {self.method.upper():6s}  : {ulr_scores}")

            # Oracle
            oracle_scores = self._metrics(y_test, self.model_oracle_.predict_proba(X_test))
            if verbose:
                print(f"  Oracle : {oracle_scores}\n")

            for method_name, scores in [
                ("Naive",  naive_scores),
                (self.method.upper(), ulr_scores),
                ("Oracle", oracle_scores),
            ]:
                rows.append({"Scheme": scheme_name, "Method": method_name, **scores})

        df = pd.DataFrame(rows).set_index(["Scheme", "Method"])
        if verbose:
            print("\n=== FULL RESULTS ===")
            print(df.to_string())
        return df

    # Experiment: MCAR sensitivity vs c 

    def run_mcar_sensitivity(
        self,
        X_train, y_train,
        X_test,  y_test,
        c_values=None,
        verbose: bool = True,
    ):
        """
        Analyse performance vs missingness rate c in the MCAR setting.

        Parameters
        ----------
        X_train, y_train : training data (fully labeled)
        X_test,  y_test  : test data
        c_values         : list of c values to test (default: [0.1, 0.2, 0.3, 0.4, 0.5])
        verbose          : print results

        Returns
        -------
        pd.DataFrame with columns [c, Method, Accuracy, Balanced Accuracy, F1, ROC AUC]
        """
        if c_values is None:
            c_values = [0.1, 0.2, 0.3, 0.4, 0.5]

        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64)
        X_test  = np.asarray(X_test,  dtype=np.float64)
        y_test  = np.asarray(y_test,  dtype=np.float64)

        self.oracle_fit(X_train, y_train)
        oracle_scores = self._metrics(y_test, self.model_oracle_.predict_proba(X_test))

        rows = []
        for c in c_values:
            if verbose:
                print(f"MCAR c={c:.1f}")

            y_obs = generate_missing(
                X_train, y_train, scheme="mcar", c=c, random_state=self.random_state
            )

            self.naive_fit(X_train, y_obs)
            naive_scores = self._metrics(y_test, self.model_naive_.predict_proba(X_test))

            self.fit(X_train, y_obs)
            ulr_scores = self._metrics(y_test, self.predict_proba(X_test))

            if verbose:
                print(f"  Naive  : {naive_scores}")
                print(f"  {self.method.upper():6s}  : {ulr_scores}")
                print(f"  Oracle : {oracle_scores}\n")

            for method_name, scores in [
                ("Naive",  naive_scores),
                (self.method.upper(), ulr_scores),
                ("Oracle", oracle_scores),
            ]:
                rows.append({"c": c, "Method": method_name, **scores})

        df = pd.DataFrame(rows).set_index(["c", "Method"])
        if verbose:
            print("\n=== MCAR SENSITIVITY ===")
            print(df.to_string())
        return df

    # EM
    def _em(self, X, y_obs):
        labeled   = y_obs != MISSING
        unlabeled = ~labeled

        init = self._fista().fit(X[labeled], y_obs[labeled])
        y_soft = y_obs.copy()
        y_soft[unlabeled] = init.predict_proba(X[unlabeled])
        prev = y_soft[unlabeled].copy()

        for i in range(self.max_iter):
            model = self._fista().fit(X, y_soft)
            new_p = model.predict_proba(X[unlabeled])
            y_soft[unlabeled] = new_p
            delta = np.max(np.abs(new_p - prev))
            prev  = new_p.copy()
            if delta < self.tol:
                print(f"  [EM] converged at iteration {i+1}  (delta={delta:.6f})")
                break
        else:
            print(f"  [EM] reached max_iter={self.max_iter}  (delta={delta:.6f})")

        y_completed = y_obs.copy()
        y_completed[unlabeled] = (new_p >= 0.5).astype(float)
        return y_completed

    def _label_propagation(self, X, y_obs):
        n = X.shape[0]
        labeled_mask   = y_obs != MISSING
        unlabeled_mask = ~labeled_mask
        labeled_idx    = np.where(labeled_mask)[0]
        unlabeled_idx  = np.where(unlabeled_mask)[0]
        n_labeled      = len(labeled_idx)
        n_unlabeled    = len(unlabeled_idx)

        knn = kneighbors_graph(
            X,
            n_neighbors=min(self.n_neighbors, n_labeled - 1),
            mode="distance",
            include_self=False,
        )
        knn = (knn + knn.T) / 2.0
        sigma = np.median(knn.data) if len(knn.data) > 0 else 1.0
        knn.data = np.exp(-(knn.data ** 2) / (sigma ** 2 + 1e-10))

        row_sums = np.array(knn.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0
        W = sp.diags(1.0 / row_sums) @ knn

        order   = np.concatenate([labeled_idx, unlabeled_idx])
        W_ord   = W[order][:, order]
        W_ul    = W_ord[n_labeled:, :n_labeled]
        W_uu    = W_ord[n_labeled:, n_labeled:]
        f_l     = y_obs[labeled_idx].astype(float)

        I = sp.eye(n_unlabeled, format="csc")
        A = (I - W_uu).toarray()
        b = np.array(W_ul @ f_l).flatten()
        f_u = np.clip(np.linalg.solve(A, b), 0.0, 1.0)

        y_completed = y_obs.copy()
        y_completed[unlabeled_idx] = (f_u >= 0.5).astype(float)
        return y_completed
