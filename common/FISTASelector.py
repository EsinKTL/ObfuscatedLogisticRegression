import numpy as np
import matplotlib.pyplot as plt
from .metrics import evaluate, Metric
from .LogisticRegressionFISTA import LogisticRegressionFISTA

class FISTASelector:
	"""
	Helper class to run FISTA for different lambda values,selects and visualizes the best lambda in the validation set.
	"""
	
	def __init__(self, lambdas=None, max_iter=1000, tol=1e-4):
		if lambdas is None:
			self.lambdas = np.logspace(-4, 1, 30)
		else:
			self.lambdas = np.array(lambdas)
		
		self.max_iter = max_iter
		self.tol = tol
		self.models = {}
		self.scores = {}
		self.best_lambda = None
		self.best_model = None
	
	def fit(self, X_train, y_train, X_valid, y_valid, measure=Metric.AUC_ROC):
		best_score = -np.inf
		
		for lam in self.lambdas:
			model = LogisticRegressionFISTA(
				lambda_val=lam,
				max_iter=self.max_iter,
				tol=self.tol
			)
			model.fit(X_train, y_train)
			score = model.validate(X_valid, y_valid, measure)
			
			self.models[lam] = model
			self.scores[lam] = score
			
			if score > best_score:
				best_score = score
				self.best_lambda = lam
				self.best_model = model
		
		return self
	
	def predict_proba(self, X):
		return self.best_model.predict_proba(X)
	
	def predict(self, X, threshold=0.5):
		return (self.predict_proba(X) >= threshold).astype(int)
	
	def plot(self, measure=Metric.AUC_ROC):
		if not self.scores:
			raise RuntimeError("fit() method must be called first")
		
		lambdas = self.lambdas
		scores = [self.scores[lam] for lam in lambdas]
		
		fig, ax = plt.subplots(figsize=(8, 4))
		ax.semilogx(lambdas, scores, 'o-', color='#378ADD', linewidth=2, markersize=5)
		ax.axvline(self.best_lambda, color='#E24B4A', linestyle='--',
		           label=f'best lambda = {self.best_lambda:.4f}')
		ax.set_xlabel('lambda (log scale)')
		ax.set_ylabel(measure)
		ax.set_title(f'Validation {measure} vs lambda')
		ax.legend()
		ax.grid(True, alpha=0.3)
		plt.tight_layout()
		plt.show()
	
	def plot_coefficients(self, feature_names=None):
		if not self.models:
			raise RuntimeError("fit() method must be called first")
		
		coef_matrix = np.array([
			self.models[lam].w[1:]
			for lam in self.lambdas
		])
		
		fig, ax = plt.subplots(figsize=(9, 5))
		for j in range(coef_matrix.shape[1]):
			label = feature_names[j] if feature_names is not None else None
			ax.semilogx(self.lambdas, coef_matrix[:, j],
			            linewidth=1.5, alpha=0.8, label=label)
		
		ax.axvline(self.best_lambda, color='#E24B4A', linestyle='--',
		           label=f'best lambda = {self.best_lambda:.4f}')
		ax.axhline(0, color='gray', linewidth=0.8, linestyle=':')
		ax.set_xlabel('lambda (log scale)')
		ax.set_ylabel('coefficient value')
		ax.set_title('Regularization Path')
		ax.grid(True, alpha=0.3)
		plt.tight_layout()
		plt.show()