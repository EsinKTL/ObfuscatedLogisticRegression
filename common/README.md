# FISTA for Logistic Regression

## The Problem

In machine learning, training means finding weights `w` that minimize a **loss function**. For logistic regression with L1 regularization:

```
min  (1/n) Σ log(1 + exp(-yᵢ * xᵢᵀw))   +   λ‖w‖₁
 w   └─────────────────────────────────┘       └────┘
              fit the data                    keep w sparse
```

The tension: you want a model that fits well *and* uses few features. The L1 term `λ‖w‖₁` enforces sparsity by driving weights to exactly zero — automatic feature selection.

The *difficulty*: `‖w‖₁` is not differentiable at zero. Classical gradient descent requires a gradient everywhere, so it breaks here.

---

## Convexity — What It Means and Why It Matters

### Geometric Definition

A function is **convex** if the line segment between any two points on the graph lies *above or on* the graph:

```
f(αx + (1-α)y)  ≤  αf(x) + (1-α)f(y)    for all α ∈ [0,1]
```

Visually — a "bowl shape". No local minima that aren't global.

```
convex:           non-convex:
   \   /             /\_  _/\
    \ /             /   \/   \
     V              
  one minimum     many local minima
```

### Why Convexity Matters

- **Guarantees**: any local minimum is the global minimum
- **Convergence proofs**: you can formally bound how fast algorithms converge
- **Reliability**: gradient-based methods won't get stuck

Logistic loss is convex. The L1 norm is convex. A sum of convex functions is convex. So the full objective is convex — we're guaranteed to find the global optimum.

### Strongly Convex vs Convex

| Type | Intuition | Convergence |
|---|---|---|
| Convex | bowl, can be flat at bottom | O(1/k) |
| Strongly convex | strictly curved bowl | O(ρᵏ), exponential |
| Non-convex | hills and valleys | no guarantee |

L2 regularization (`‖w‖₂²`) makes logistic regression *strongly* convex. L1 does not — the bowl can have a flat ridge.

---

## Optimization Approaches

### 1. Gradient Descent (baseline)
```
w ← w - α * ∇f(w)
```
**Works when**: f is smooth everywhere.  
**Fails here**: L1 has no gradient at 0.

### 2. Subgradient Descent
Replace the gradient with a **subgradient** — any value that "fits" at non-differentiable points (for `|w|` at 0, any value in `[-1, 1]`).

```
w ← w - α * g    where g is a subgradient of F at w
```

**Problem**: converges at O(1/√k) — very slow. Step sizes must decay. Never reaches machine precision.

### 3. Newton's Method
Use second-order information (Hessian):
```
w ← w - [∇²f(w)]⁻¹ ∇f(w)
```
**Problem**: Hessian is `d×d` — inverting it is O(d³). Infeasible for high-dimensional data. Also can't handle L1 directly.

### 4. L-BFGS (Limited-memory BFGS)
Approximates the Hessian using recent gradient history. Standard tool for smooth logistic regression (no L1).  
**Problem**: still can't handle the non-smooth L1 term without modifications.

### 5. Coordinate Descent
Optimize one weight at a time, cycling through all coordinates. For L1 logistic regression, each 1D subproblem has a closed-form solution.  
**Good for**: sparse problems, implemented in `sklearn`'s `liblinear`.  
**Problem**: parallelization is hard; convergence can be slow if features are correlated.

### 6. ISTA (Iterative Shrinkage-Thresholding Algorithm)
Proximal gradient descent — the direct predecessor of FISTA.  
Convergence: **O(1/k)**

### 7. FISTA
ISTA + Nesterov momentum.  
Convergence: **O(1/k²)**  
Same cost per iteration as ISTA. Optimal for this problem class.

---

## Proximal Operators — The Key Tool

### Motivation

When `g(w)` is non-smooth (like L1), you can't take its gradient. But you can define an operator that "applies" the penalty implicitly.

### Definition

The proximal operator of `g` with step size `t` is:

```
prox_{tg}(v) = argmin_w { g(w) + (1/2t)‖w - v‖² }
                          └────┘   └────────────┘
                          penalty  stay close to v
```

**Intuition**: find the point `w` that minimizes the penalty, but don't move too far from `v`. The `1/2t` term controls how much you trust the current point vs. the penalty.

### Proximal Operator of the L1 Norm

For `g(w) = λ‖w‖₁`, the solution is **soft-thresholding**, applied elementwise:

```
prox_{tλ‖·‖₁}(v)ᵢ = sign(vᵢ) * max(|vᵢ| - tλ, 0)
```

Geometric meaning:

```
|vᵢ| > tλ  →  shrink toward zero by tλ      (keep, but smaller)
|vᵢ| ≤ tλ  →  set to exactly zero           (eliminate feature)
```

```
output
  |          /
  |         /
tλ|        /
  |-------·
  |      /
  +------·-----------·---- input
        tλ           
```

No gradient needed — this is a closed-form formula.

### Other Common Proximal Operators

| Penalty `g(w)` | `prox_{tg}(v)` |
|---|---|
| L1: `λ‖w‖₁` | soft-threshold: `sign(v)·max(\|v\|-tλ, 0)` |
| L2²: `λ‖w‖₂²` | scaling: `v / (1 + 2tλ)` |
| Box constraint `w ∈ [a,b]` | projection: `clip(v, a, b)` |
| Indicator of convex set C | projection onto C |

The last row reveals something deep: **projection is a special case of a proximal operator**. This unifies a huge family of constrained and regularized problems.

---

## The FISTA Algorithm

### Core Idea: Proximal Gradient

Split `F(w) = f(w) + g(w)` into:
- `f(w)` — smooth part (logistic loss) → use gradient
- `g(w)` — non-smooth part (L1) → use proximal operator

Each step does two things:

**1. Gradient step** on the smooth loss:
```
v = w - (1/L) * ∇f(w)
```

**2. Proximal step** applying the L1 penalty:
```
w_new = prox_{λ/L}(v) = sign(v) * max(|v| - λ/L, 0)
```

This is soft-thresholding — it shrinks weights toward zero, setting small ones exactly to zero.

### The "Fast" Part: Nesterov Momentum

Vanilla proximal gradient (ISTA) converges at **O(1/k)**. FISTA adds a momentum term that achieves **O(1/k²)**:

```
t_new = (1 + sqrt(1 + 4t²)) / 2
y = w + ((t - 1) / t_new) * (w - w_prev)   ← momentum point
```

The next gradient step is taken from `y`, not `w`. This "looks ahead" using the history of iterates.

### Full Algorithm

```
Initialize: w₀ = 0, y₁ = w₀, t₁ = 1

For k = 1, 2, ...:
    1. Compute gradient:  g = ∇f(yₖ)           # logistic loss gradient at momentum point
    2. Gradient step:     v = yₖ - (1/L) * g
    3. Proximal step:     wₖ = soft_threshold(v, λ/L)
    4. Update momentum:   tₖ₊₁ = (1 + sqrt(1 + 4tₖ²)) / 2
    5. Update y:          yₖ₊₁ = wₖ + ((tₖ - 1)/tₖ₊₁) * (wₖ - wₖ₋₁)
```

For logistic regression:
- `L` = Lipschitz constant of `∇f` = `‖X‖²_spectral / (4n)` (or estimated via backtracking line search)
- `∇f(w) = Xᵀ(σ(Xw) - y) / n` where `σ` is the sigmoid function

### Key Properties

| Property | Value |
|---|---|
| Convergence rate | O(1/k²) vs ISTA's O(1/k) |
| Memory | Only needs last 2 iterates |
| Handles L1 | Yes, via soft-thresholding |
| Output | Sparse weights (automatic feature selection) |
| Optimality | Proven optimal for first-order methods on this problem class |

---

## How It All Connects

```
Problem:  min f(w) + g(w)
              └──┘   └──┘
            smooth  non-smooth
         (logistic loss)  (L1)

Key insight: split the problem
  - Handle f with gradient steps  (exploits smoothness)
  - Handle g with proximal steps  (avoids needing a gradient)

ISTA:  gradient step → proximal step             O(1/k)
FISTA: gradient step from momentum point → prox  O(1/k²)
                        └──────────────┘
                    Nesterov's acceleration trick
                    (look ahead using iterate history)
```

FISTA is considered optimal because it has been proven that **no first-order method** (using only gradient and function values, not Hessians) can converge faster than O(1/k²) for this problem class. FISTA achieves that bound.
