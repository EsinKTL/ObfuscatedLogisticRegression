import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from common.FISTASelector import FISTASelector
from common.metrics import Metric, print_evaluation


PREPROCESSED_FILE = Path(__file__).parent / "data" / "heart_preprocessed.csv"
MEASURE = Metric.AUC_ROC
RANDOM_STATE = 42

df = pd.read_csv(PREPROCESSED_FILE)
feature_names = [c for c in df.columns if c != "target"]

X = df[feature_names].values.astype(np.float64)
y = df["target"].values.astype(np.float64)

X_train, X_temp, y_train, y_temp = train_test_split(
	X, y, test_size=0.40, stratify=y, random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
	X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
)

print(f"Train : {X_train.shape[0]} rows")
print(f"Val   : {X_val.shape[0]} rows")
print(f"Test  : {X_test.shape[0]} rows")
print(f"Positive ratio — train: {y_train.mean():.3f}  val: {y_val.mean():.3f}  test: {y_test.mean():.3f} \n")


print(f"Lambda selection: Based on the {MEASURE} metric")
selector = FISTASelector(
	lambdas=np.logspace(-4, 1, 30),
	max_iter=1000,
	tol=1e-4
)
selector.fit(X_train, y_train, X_val, y_val, measure=MEASURE)
print(f"The best lambda: {selector.best_lambda:.6f} \n")

# Custom Logistic regression (FISTA)
fista_proba = selector.predict_proba(X_test)
print_evaluation("Custom Logistic regression (FISTA)", y_test, fista_proba)

# Sklearn comparison - same lambda, penalty='l1', solver='saga'
C_sklearn = 1.0 / (X_train.shape[0] * selector.best_lambda)
sklearn_model = LogisticRegression(
	solver="saga", C=C_sklearn, l1_ratio=1.0,
	max_iter=2000, random_state=RANDOM_STATE
)
sklearn_model.fit(X_train, y_train)
sk_proba = sklearn_model.predict_proba(X_test)[:, 1]
print_evaluation("Sklearn L1 LogReg", y_test, sk_proba)

print("=== Coefficient Comparison ===")
print(f"{'Feature':15s}  {'FISTA':>10s}  {'sklearn':>10s}")
print("-" * 40)
for fname, w_fista, w_sk in zip(feature_names, selector.best_model.w[1:], sklearn_model.coef_[0]):
    print(f"{fname:15s}  {w_fista:10.4f}  {w_sk:10.4f}")
print()

selector.plot(measure=MEASURE)
selector.plot_coefficients(feature_names=feature_names)