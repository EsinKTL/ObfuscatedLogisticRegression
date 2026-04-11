"""
unlabeled_logreg.py
===================
UnlabeledLogReg: uses both labeled (Y=0/1) and unlabeled (Y=-1) observations.

Two completion algorithms:
  - 'em'                : Expectation-Maximization with soft labels
  - 'label_propagation' : Graph-based label propagation via k-NN

After completing Y, trains LogisticRegressionFISTA on the full dataset.
"""

import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.linear_model import LogisticRegression
import scipy.sparse as sp

from fista_bank import LogisticRegressionFISTA, FISTASelector


MISSING = -1


class UnlabeledLogReg:
	"""
	Logistic regression that exploits unlabeled observations (Y = -1).

	Parameters
	----------
	method : str
		'em' or 'label_propagation'
	lambda_val : float or None
		L1 penalty for FISTA. If None, FISTASelector tunes it on labeled data.
	max_iter : int
		Max iterations for the completion algorithm (EM rounds / LP iterations).
	tol : float
		Convergence tolerance for the completion algorithm.
	fista_max_iter : int
		Max iterations passed to each FISTA fit call.
	n_neighbors : int
		Number of neighbors for label propagation k-NN graph.
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
		
		self.model_ = None          # final LogisticRegressionFISTA
		self.y_completed_ = None    # completed label vector after algorithm
	
	# ── Public API ─────────────────────────────────────────────────────────────
	
	def fit(self, X, y_obs):
		"""
		Complete missing labels then train FISTA on the full dataset.

		Parameters
		----------
		X     : array (n_samples, n_features)  — preprocessed features
		y_obs : array (n_samples,)             — labels in {-1, 0, 1}
												 -1 means label is missing
		Returns
		-------
		self
		"""
		X = np.asarray(X, dtype=np.float64)
		y_obs = np.asarray(y_obs, dtype=np.float64)
		
		if self.method == "em":
			y_completed = self._em(X, y_obs)
		else:
			y_completed = self._label_propagation(X, y_obs)
		
		self.y_completed_ = y_completed
		
		# Final FISTA on all observations with completed labels
		self.model_ = LogisticRegressionFISTA(
			lambda_val=self.lambda_val,
			max_iter=self.fista_max_iter,
			tol=self.tol,
		)
		self.model_.fit(X, y_completed)
		return self
	
	def predict_proba(self, X):
		"""Positive class probability from the final FISTA model."""
		if self.model_ is None:
			raise RuntimeError("Call fit() first.")
		return self.model_.predict_proba(X)
	
	def predict(self, X, threshold=0.5):
		"""Binary prediction at given threshold."""
		return (self.predict_proba(X) >= threshold).astype(int)
	
	# ── EM ─────────────────────────────────────────────────────────────────────
	
	def _em(self, X, y_obs):
		"""
		EM algorithm for missing labels.

		E-step: estimate P(Y=1 | X_i) for unlabeled observations using
				the current FISTA model → soft labels
		M-step: re-fit FISTA using soft labels for unlabeled + true labels
				for labeled observations
		Repeat until label assignments stop changing.

		Initialization: train on labeled observations only (naive start).
		"""
		labeled   = y_obs != MISSING
		unlabeled = ~labeled
		
		# ── Init: fit on labeled data only ────────────────────────────────────
		init_model = LogisticRegressionFISTA(
			lambda_val=self.lambda_val,
			max_iter=self.fista_max_iter,
			tol=self.tol,
		)
		init_model.fit(X[labeled], y_obs[labeled])
		
		# Soft labels: start with model predictions for unlabeled
		y_soft = y_obs.copy()
		y_soft[unlabeled] = init_model.predict_proba(X[unlabeled])
		
		prev_unlabeled_proba = y_soft[unlabeled].copy()
		
		for iteration in range(self.max_iter):
			
			# M-step: fit FISTA on all data using current soft labels
			model = LogisticRegressionFISTA(
				lambda_val=self.lambda_val,
				max_iter=self.fista_max_iter,
				tol=self.tol,
			)
			model.fit(X, y_soft)
			
			# E-step: update soft labels for unlabeled observations
			new_proba = model.predict_proba(X[unlabeled])
			y_soft[unlabeled] = new_proba
			
			# Convergence: max change in unlabeled probabilities
			delta = np.max(np.abs(new_proba - prev_unlabeled_proba))
			prev_unlabeled_proba = new_proba.copy()
			
			if delta < self.tol:
				print(f"  [EM] converged at iteration {iteration + 1}  (delta={delta:.6f})")
				break
		else:
			print(f"  [EM] reached max_iter={self.max_iter}  (delta={delta:.6f})")
		
		# Hard labels from final soft labels (threshold = 0.5)
		y_completed = y_obs.copy()
		y_completed[unlabeled] = (new_proba >= 0.5).astype(float)
		return y_completed
	
	# ── Label Propagation ──────────────────────────────────────────────────────
	
	def _label_propagation(self, X, y_obs):
		"""
		Graph-based label propagation via k-NN.

		Algorithm (harmonic function / Zhu & Ghahramani 2002):
		  1. Build k-NN graph on all observations (labeled + unlabeled)
		  2. Compute row-normalized weight matrix W
		  3. Solve:  f_u = (I - W_uu)^{-1} W_ul f_l
			 where f_l = known labels, f_u = unknown labels to be inferred
		  4. Threshold f_u at 0.5 → hard labels

		This finds the harmonic function on the graph that exactly
		interpolates the labeled nodes — the smoothest possible label
		assignment consistent with what we know.
		"""
		n = X.shape[0]
		labeled_mask   = y_obs != MISSING
		unlabeled_mask = ~labeled_mask
		
		labeled_idx   = np.where(labeled_mask)[0]
		unlabeled_idx = np.where(unlabeled_mask)[0]
		
		n_labeled   = len(labeled_idx)
		n_unlabeled = len(unlabeled_idx)
		
		# ── Build k-NN graph ──────────────────────────────────────────────────
		# connectivity='distance' → edge weight = Euclidean distance
		# We convert to RBF weight: w_ij = exp(-d_ij^2 / sigma^2)
		knn = kneighbors_graph(
			X,
			n_neighbors=min(self.n_neighbors, n_labeled - 1),
			mode="distance",
			include_self=False,
		)
		# Symmetrize
		knn = (knn + knn.T) / 2.0
		
		# RBF kernel: sigma = median of non-zero distances
		data = knn.data
		sigma = np.median(data) if len(data) > 0 else 1.0
		knn.data = np.exp(-(data ** 2) / (sigma ** 2 + 1e-10))
		
		# Row-normalize → transition matrix
		row_sums = np.array(knn.sum(axis=1)).flatten()
		row_sums[row_sums == 0] = 1.0
		D_inv = sp.diags(1.0 / row_sums)
		W = D_inv @ knn                 # row-stochastic
		
		# ── Harmonic solution ─────────────────────────────────────────────────
		# Reorder: labeled first, unlabeled second
		order = np.concatenate([labeled_idx, unlabeled_idx])
		W_reordered = W[order][:, order]
		
		W_ul = W_reordered[n_labeled:, :n_labeled]    # unlabeled → labeled
		W_uu = W_reordered[n_labeled:, n_labeled:]    # unlabeled → unlabeled
		
		f_l = y_obs[labeled_idx].astype(float)         # known labels
		
		# f_u = (I - W_uu)^{-1} W_ul f_l
		I = sp.eye(n_unlabeled, format="csc")
		A = (I - W_uu).toarray()
		b = (W_ul @ f_l)
		if sp.issparse(b):
			b = b.toarray().flatten()
		else:
			b = np.array(b).flatten()
		
		f_u = np.linalg.solve(A, b)
		f_u = np.clip(f_u, 0.0, 1.0)
		
		# Hard labels
		hard_labels = (f_u >= 0.5).astype(float)
		
		y_completed = y_obs.copy()
		y_completed[unlabeled_idx] = hard_labels
		return y_completed