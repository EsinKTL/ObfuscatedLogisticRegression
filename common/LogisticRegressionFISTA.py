import numpy as np
import matplotlib.pyplot as plt
from .metrics import evaluate


class LogisticRegressionFISTA:
	"""
	Logistic regression with L1 regularization using FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)

	Parameters
	----------
	lambda_val : float 
        L1 penalty coefficient. Larger value - sparser model.
	max_iter : int
        Maximum number of iterations.
	tol : float
		Convergence tolerance. The change between two iterations stops when it falls below this.
	"""
	
	def __init__(self, lambda_val=1.0, max_iter=1000, tol=1e-4):
		self.lambda_val = lambda_val
		self.max_iter = max_iter
		self.tol = tol
		self.w = None
		self.L = None
	
	def _sigmoid(self, z):
		return np.where(
			z >= 0,
			1.0 / (1.0 + np.exp(-z)),
			np.exp(z) / (1.0 + np.exp(z))
		)
	
	def _soft_thresholding(self, w, threshold):
		w_thresh = np.sign(w) * np.maximum(np.abs(w) - threshold, 0.0)
		w_thresh[0] = w[0] # Don't touch bias
		return w_thresh
	
	def _compute_lipschitz(self, X_design):
		n = X_design.shape[0]
		return np.linalg.norm(X_design, 'fro') ** 2 / (4 * n)
	
	def _gradient(self, X_design, y, w):
		n = X_design.shape[0]
		p = self._sigmoid(X_design @ w)
		return (X_design.T @ (p - y)) / n
	
	def fit(self, X, y):
		X = np.asarray(X, dtype=np.float64)
		y = np.asarray(y, dtype=np.float64)
		
		n_samples, n_features = X.shape
		X_design = np.c_[np.ones(n_samples), X]
		
		self.L = self._compute_lipschitz(X_design)
		step = 1.0 / self.L
		
		w_k = np.zeros(n_features + 1)
		z_k = w_k.copy()
		t_k = 1.0
		
		for i in range(self.max_iter):
			grad = self._gradient(X_design, y, z_k)
			v = z_k - step * grad
			w_next = self._soft_thresholding(v, self.lambda_val * step)
			
			if np.linalg.norm(w_next - w_k) < self.tol:
				break
			
			t_next = (1.0 + np.sqrt(1.0 + 4.0 * t_k ** 2)) / 2.0
			z_k = w_next + (t_k - 1.0) / t_next * (w_next - w_k)
			w_k = w_next
			t_k = t_next
		
		self.w = w_k
		return self
	
	def predict_proba(self, X):
		"""Returns aprobaility of class 1 for each observation"""
		if self.w is None:
			raise RuntimeError("fit() method must be called first")
		X = np.asarray(X, dtype=np.float64)
		X_design = np.c_[np.ones(X.shape[0]), X]
		return self._sigmoid(X_design @ self.w)
	
	def predict(self, X, threshold=0.5):
		# TODO: threshold should be calculated per metric per dataset???
		return (self.predict_proba(X) >= threshold).astype(int)
	
	def validate(self, X_valid, y_valid, measure):
		y_prob = self.predict_proba(X_valid)
		return evaluate(y_valid, y_prob, measure)

