import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as SklearnLR
from common.FISTASelector import FISTASelector
from common.metrics import Metric, print_evaluation


PREPROCESSED_FILE = Path(__file__).parent / "poker_data_preprocessed.csv"
MEASURE = Metric.F_MEASURE
RANDOM_STATE = 42

df = pd.read_csv(PREPROCESSED_FILE)
feature_names = [c for c in df.columns if c != "result"]

X = df[feature_names].values.astype(np.float64)
y = df["result"].values.astype(np.float64)

print(f"Dataset: {X.shape[0]} rows, {X.shape[1]} features")
print(f"Positive class ratio: {y.mean():.2f}\n")

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=RANDOM_STATE
)

print(f"Train : {X_train.shape[0]} rows")
print(f"Val   : {X_val.shape[0]} rows")
print(f"Test  : {X_test.shape[0]} rows")
print(f"Positive ratio — train: {y_train.mean():.3f}  val: {y_val.mean():.3f}  test: {y_test.mean():.3f}\n")


print(f"Lambda selection: Based on the {MEASURE} metric")
selector = FISTASelector(
    lambdas=np.logspace(-4, 1, 30),
    max_iter=1000,
    tol=1e-5
)
selector.fit(X_train, y_train, X_val, y_val, measure=MEASURE)
print(f"The best lambda: {selector.best_lambda:.6f}\n")

# Custom Logistic Regression (FISTA)
fista_proba = selector.predict_proba(X_test)
print_evaluation("Custom Logistic Regression (FISTA)", y_test, fista_proba)

w = selector.best_model.w[1:]  # exclude bias
print(f"  Zero coefficients : {(w == 0).sum()} / {len(w)}\n")

# Sklearn comparison — same lambda, penalty='l1', solver='saga'
C_sklearn = 1.0 / (X_train.shape[0] * selector.best_lambda)
sklearn_model = SklearnLR(
    solver="saga", C=C_sklearn, l1_ratio=1.0,
    max_iter=2000, random_state=RANDOM_STATE
)
sklearn_model.fit(X_train, y_train)
sk_proba = sklearn_model.predict_proba(X_test)[:, 1]
print_evaluation("Sklearn L1 LogReg", y_test, sk_proba)

sk_w = sklearn_model.coef_[0]
print(f"  Zero coefficients : {(sk_w == 0).sum()} / {len(sk_w)}\n")

print("=== Coefficient Comparison ===")
print(f"{'Feature':15s}  {'FISTA':>10s}  {'sklearn':>10s}")
print("-" * 40)
for fname, w_fista, w_sk in zip(feature_names, selector.best_model.w[1:], sklearn_model.coef_[0]):
    print(f"{fname:15s}  {w_fista:10.4f}  {w_sk:10.4f}")
print()

selector.plot(measure=MEASURE)
selector.plot_coefficients(feature_names=feature_names)
