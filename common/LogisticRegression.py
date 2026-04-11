import numpy as np
import matplotlib.pyplot as plt
from .fista import sigmoid, fista
from .metrics import evaluate



class LogisticRegression:
    """
    Implementation of Logistic regression based on FISTA from scratch

    Usage
    -----
    model = LogisticRegression(l=0.01)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    """

    def __init__(self, l=0.01, max_iter=500, tol=1e-6):
        self.l = l
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
        self.history = None

    def fit(self, X, y):
        self.w, self.history = fista(X, y, self.l, self.max_iter, self.tol)

    def validate(self, X_valid, y_valid, measure):
        y_prob = self.predict_proba(X_valid)
        return evaluate(y_valid, y_prob, measure)

    def predict_proba(self, X):
        return sigmoid(X @ self.w)

    @staticmethod
    def plot(measure, X_train, y_train, X_valid, y_valid, lambdas=None, max_iter=500, tol=1e-6):
        if lambdas is None:
            lambdas = np.logspace(-4, 1, 30)

        scores = []
        for l in lambdas:
            model = LogisticRegression(l=l, max_iter=max_iter, tol=tol)
            model.fit(X_train, y_train)
            scores.append(model.validate(X_valid, y_valid, measure))

        plt.figure()
        plt.plot(lambdas, scores)
        plt.xscale("log")
        plt.xlabel("lambda")
        plt.ylabel(measure.value)
        plt.title(f"{measure.value} vs lambda")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_coefficients(X_train, y_train, lambdas=None, max_iter=500, tol=1e-6, feature_names=None):
        if lambdas is None:
            lambdas = np.logspace(-4, 1, 30)

        weights = []
        for l in lambdas:
            model = LogisticRegression(l=l, max_iter=max_iter, tol=tol)
            model.fit(X_train, y_train)
            weights.append(model.w)

        weights = np.array(weights)  # (n_lambdas, d)

        plt.figure()
        for j in range(weights.shape[1]):
            label = feature_names[j] if feature_names is not None else None
            plt.plot(lambdas, weights[:, j], label=label)

        plt.xscale("log")
        plt.xlabel("lambda")
        plt.ylabel("coefficient value")
        plt.title("Regularization path")
        plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
        if feature_names is not None:
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
        plt.tight_layout()
        plt.show()
