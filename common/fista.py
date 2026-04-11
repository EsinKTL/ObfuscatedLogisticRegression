import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def logistic_loss(X, y, w, l):
    """
    Logistic loss + L1 penalty.

    Parameters
    ----------
    X : (n, d) feature matrix
    y : (n,) binary labels in {0, 1}
    w : (d,) weight vector
    l : float  L1 regularisation strength

    Returns
    -------
    float  — scalar objective value
    """
    scores = X @ w
    loss = np.mean(np.log(1 + np.exp(-scores)) + (1 - y) * scores)
    penalty = l * np.sum(np.abs(w))
    return loss + penalty


def gradient(X, y, w):
    """
    Gradient of the logistic loss (already computed and simplified but may need some additional checking).

    Parameters
    ----------
    X : (n, d)
    y : (n,)
    w : (d,)

    Returns
    -------
    grad : (d,)
    """
    residuals = sigmoid(X @ w) - y      # prediction error  (n,)
    return X.T @ residuals / len(y)     # averaged gradient (d,)


def _soft_threshold(v, threshold):
    """
    Proximal operator for the L1 norm.

    Elementwise: sign(vi) * max(|vi| - threshold, 0)
    Values inside [-threshold, threshold] are mapped to exactly zero.

    Parameters
    ----------
    v         : (d,) array
    threshold : float  equals lambda / L

    Returns
    -------
    (d,) array
    """
    return np.sign(v) * np.maximum(np.abs(v) - threshold, 0)


def _lipschitz_constant(X):
    """
    Upper bound on the Lipschitz constant of gradient of f for logistic regression.

    This guarantees the gradient step 1/L does not overshoot.
    """
    n = X.shape[0]
    sigma_max = np.linalg.norm(X, ord=2)    # spectral norm = largest singular value
    return sigma_max ** 2 / (4 * n)

def fista(X, y, l, max_iter=500, tol=1e-6):
    """
    FISTA: Fast Iterative Shrinkage-Thresholding Algorithm
    for L1-regularised logistic regression.

    Convergence rate: O(1/k^2) — optimal for first-order methods on this
    problem class.

    Parameters
    ----------
    X        : (n, d) float array   feature matrix (should be standardised)
    y        : (n,)   int array     binary labels in {0, 1}
    l      : float                L1 regularisation strength (lambda >= 0)
    max_iter : int                  maximum number of iterations
    tol      : float                stop when ||w_new − w|| < tol

    Returns
    -------
    w        : (d,)  sparse weight vector
    history  : list of float  objective value F(w) at each iteration
    """
    n, d = X.shape

    #step size (fixed, based on Lipschitz constant) 
    lipschitz_constant = _lipschitz_constant(X)
    step = 1.0 / lipschitz_constant
    threshold = l * step

    # initialise
    w = np.zeros(d)
    w_prev = np.zeros(d)
    t = 1.0
    history = []

    for k in range(max_iter):

        # 1. momentum point  y_k = w + ((t-1)/t_new) * (w - w_prev)
        #    Note: we use the *current* t here; t_new is computed after.
        #    On the first iteration t=1 so momentum term is 0 (pure gradient step).
        momentum_coeff = (t - 1) / t
        y_k = w + momentum_coeff * (w - w_prev)

        # 2. gradient of smooth loss at the momentum point
        grad = gradient(X, y, y_k)

        # 3. gradient step
        v = y_k - step * grad

        # 4. proximal step (soft-threshold applies L1 penalty)
        w_new = _soft_threshold(v, threshold)

        # 5. Nesterov momentum scalar update
        t_new = (1 + np.sqrt(1 + 4 * t ** 2)) / 2

        # 6. track objective
        history.append(logistic_loss(X, y, w_new, l))

        # 7. convergence check
        if np.linalg.norm(w_new - w) < tol:
            break

        # 8. advance state
        w_prev = w
        w = w_new
        t = t_new

    return w, history