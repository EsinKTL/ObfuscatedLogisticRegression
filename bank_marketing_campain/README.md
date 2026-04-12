# AML Project 1 — Bank Marketing Dataset
### Missing Label Learning with Logistic Lasso Regression (FISTA)
**Advanced Machine Learning · 2026**

---

## Overview

This project studies logistic regression under **missing label conditions** using the [Bank Marketing dataset](https://www.kaggle.com/datasets/ruthgn/bank-marketing-data-set) (UCI). The binary classification task is to predict whether a client will subscribe to a term deposit.

Three tasks are covered:

| Task | Description |
|------|-------------|
| **Task 1** | Data preprocessing + four missing label generation schemes (MCAR / MAR1 / MAR2 / MNAR) |
| **Task 2** | Custom FISTA implementation for L1-regularised logistic regression |
| **Task 3** | `UnlabeledLogReg` — exploits unlabeled observations via EM and Label Propagation |

---

## Project Structure

```
.
├── processing.py      # Task 1 — full preprocessing pipeline
├── missing_data_bank.py            # Task 1 — MCAR / MAR1 / MAR2 / MNAR schemes
├── generate_dataset.py       # Task 1 — applies missing schemes, saves CSVs
├── fista_bank.py                   # Task 2 — FISTA + FISTASelector classes
├── fista_run_bank.py               # Task 2 — lambda selection + sklearn comparison
├── unlabeled_logreg_bank.py        # Task 3 — UnlabeledLogReg (EM + Label Propagation)
├── run_exp_bank.py         # Task 3 — full experiment runner
├── bank_marketing_project.ipynb  # Full walkthrough notebook
└── README.md
```

## Installation

```bash
pip install numpy pandas scikit-learn scipy matplotlib
```

Python 3.9+ required. No other dependencies.

---

## Quick Start

### Step 1 — Preprocess the dataset

```python
from bank_preprocessing import run_pipeline

df = run_pipeline("bank-additional-full.csv", "bank_preprocessed.csv")
```

Outputs `bank_preprocessed.csv` with 48 features (9 numeric + 39 one-hot encoded) and a binary target `y`.

### Step 2 — Generate missing-label datasets

```bash
python generate_datasets.py
```

Produces four CSVs (`bank_mcar.csv`, `bank_mar1.csv`, `bank_mar2.csv`, `bank_mnar.csv`), each containing features + `y_obs` column where `-1` means the label is missing.

Or generate programmatically:

```python
from missing_data import generate_missing

y_obs = generate_missing(X_train, y_train, scheme="mcar", c=0.3)
y_obs = generate_missing(X_train, y_train, scheme="mar1", c=0.3, feature_idx=0)
y_obs = generate_missing(X_train, y_train, scheme="mar2", c=0.3)
y_obs = generate_missing(X_train, y_train, scheme="mnar", c=0.3, feature_idx=0, y_weight=5.0)
```

### Step 3 — Train with FISTA

```python
from fista_bank import LogisticRegressionFISTA, FISTASelector
import numpy as np

# Single lambda
model = LogisticRegressionFISTA(lambda_val=1e-3, max_iter=1000)
model.fit(X_train, y_train)
proba = model.predict_proba(X_test)

# Automatic lambda selection
selector = FISTASelector(lambdas=np.logspace(-4, 1, 30))
selector.fit(X_train, y_train, X_val, y_val, measure="roc_auc")
selector.plot(measure="roc_auc")
selector.plot_coefficients()
```

### Step 4 — Unlabeled Logistic Regression (Task 3)

```python
from unlabeled_logreg_bank import UnlabeledLogReg

model = UnlabeledLogReg(method="em", lambda_val=1e-3)

model.fit(X_train, y_obs)           # EM or Label Propagation on labeled + unlabeled
model.naive_fit(X_train, y_obs)     # Naive — labeled rows only
model.oracle_fit(X_train, y_true)   # Oracle — all true labels (benchmark)

model.compare(X_test, y_test)       # Print accuracy / bal. acc. / F1 / ROC AUC
```

Run all experiments at once:

```bash
python run_exp_bank.py
```

---

## Module Reference

### `processing.py`

| Function | Description |
|----------|-------------|
| `load_data(filepath)` | Load raw CSV, select relevant columns |
| `binarize_labels(df)` | Map `yes/no` → `1/0` |
| `engineer_features(df)` | Add `campaign_ratio` |
| `encode_categoricals(df)` | One-hot encode with `drop_first=True` |
| `remove_collinear_features(df, threshold)` | Drop columns with \|r\| > threshold |
| `scale_and_transform(df)` | Log1p + MinMax scaling |
| `run_pipeline(input, output)` | End-to-end pipeline |

---

### `missing_data_bank.py`

| Function | Scheme | Mechanism |
|----------|--------|-----------|
| `generate_mcar(X, y, c)` | MCAR | `P(S=1) = c` — uniform random |
| `generate_mar1(X, y, c, feature_idx)` | MAR1 | Sigmoid on one feature |
| `generate_mar2(X, y, c)` | MAR2 | Sigmoid on random linear combination of all features |
| `generate_mnar(X, y, c, feature_idx, y_weight)` | MNAR | Sigmoid on feature + true label |
| `generate_missing(X, y, scheme, c, ...)` | Unified | Dispatch to any of the above |

All schemes calibrate the sigmoid intercept via binary search so the actual missing rate equals `c` exactly.

---

### `fista_bank.py`

#### `LogisticRegressionFISTA`

```python
model = LogisticRegressionFISTA(lambda_val=1e-3, max_iter=1000, tol=1e-4)
model.fit(X_train, y_train)
model.predict_proba(X_test)
model.predict(X_test, threshold=0.5)
model.validate(X_val, y_val, measure)  # measure ∈ {recall, precision, f1, balanced_accuracy, roc_auc, pr_auc}
```

#### `FISTASelector`

```python
selector = FISTASelector(lambdas=np.logspace(-4, 1, 30))
selector.fit(X_train, y_train, X_val, y_val, measure="roc_auc")
selector.plot(measure="roc_auc")          # validation metric vs lambda
selector.plot_coefficients()              # regularisation path
selector.best_lambda                      # best lambda value
selector.best_model                       # best LogisticRegressionFISTA instance
```

---

### `unlabeled_logreg_bank.py`

#### `UnlabeledLogReg`

```python
model = UnlabeledLogReg(
    method="em",          # "em" or "label_propagation"
    lambda_val=1e-3,
    max_iter=20,
    tol=1e-4,
    fista_max_iter=1000,
    n_neighbors=15,       # for label propagation
    random_state=42,
)
```

| Method | Description |
|--------|-------------|
| `fit(X, y_obs)` | Complete missing labels (EM or LP), then fit FISTA on all data |
| `naive_fit(X, y_obs)` | Fit FISTA on labeled rows only (S = 0) |
| `oracle_fit(X, y_true)` | Fit FISTA on full true labels (upper-bound benchmark) |
| `predict_proba(X)` | Predict from the `fit()` model |
| `predict(X, threshold)` | Binary prediction |
| `compare(X_test, y_test)` | Evaluate all fitted models — prints accuracy / balanced accuracy / F1 / ROC AUC |
| `run_schemes(X_train, y_train, X_test, y_test, c)` | Compare Naive / UnlabeledLogReg / Oracle under MCAR, MAR1, MAR2, MNAR |
| `run_mcar_sensitivity(X_train, y_train, X_test, y_test, c_values)` | Performance vs missing rate in MCAR setting |

---

## Results Summary

### Task 2 — FISTA vs sklearn (test set, λ selected by ROC AUC)

| Metric | FISTA (custom) | sklearn L1 |
|--------|---------------|------------|
| Recall | 0.2101 | 0.2080 |
| Precision | 0.7143 | 0.7148 |
| F1 | 0.3247 | 0.3222 |
| Balanced Accuracy | 0.5997 | 0.5987 |
| **ROC AUC** | **0.8012** | **0.8012** |
| PR AUC | 0.4647 | 0.4649 |

Custom FISTA matches sklearn exactly on ROC AUC. Low recall is expected due to class imbalance (~11% positive).

### Task 3 — Method comparison (c = 0.3)

| Scheme | Method | Balanced Acc. | F1 | ROC AUC |
|--------|--------|:---:|:---:|:---:|
| MCAR | Naive | 0.6005 | 0.3257 | 0.7988 |
| MCAR | EM | 0.5925 | 0.3060 | 0.7987 |
| MCAR | Label Propagation | 0.5722 | 0.2511 | 0.7981 |
| MCAR | Oracle | 0.6021 | 0.3298 | 0.8011 |
| MAR1 | Naive | 0.5958 | 0.3151 | 0.8006 |
| MAR1 | EM | 0.5863 | 0.2896 | 0.7925 |
| MAR1 | Label Propagation | 0.5520 | 0.1901 | 0.7792 |
| MAR2 | Naive | 0.5978 | 0.3195 | 0.7992 |
| MAR2 | EM | 0.5942 | 0.3108 | 0.7901 |
| MAR2 | Label Propagation | 0.5461 | 0.1709 | 0.7826 |
| **MNAR** | **All methods** | **0.500** | **0.000** | **≤0.754** |

**Key finding:** Under MNAR, all methods collapse (F1 = 0) because subscribed clients (Y = 1) are systematically more likely to be missing, making the observed label distribution heavily skewed toward Y = 0.

---

## Notebook

`bank_marketing_project.ipynb` contains the full walkthrough with all outputs, plots, and analysis. Run it with:

```bash
jupyter notebook bank_marketing_project.ipynb
```

---

## Dataset

| Property | Value |
|----------|-------|
| Source | UCI Machine Learning Repository |
| File | `bank-additional-full.csv` (semicolon-separated) |
| Observations | 41,188 |
| Raw features | 20 |
| Features after preprocessing | 48 |
| Target | `y` — term deposit subscription (binary) |
| Class balance | ~11% positive |
| Split | 60% train / 20% val / 20% test (stratified) |
