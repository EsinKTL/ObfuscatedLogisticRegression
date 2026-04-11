from pathlib import Path
import sys
import os
import numpy as np
import pandas as pd
from common.fista import sigmoid, fista, logistic_loss, gradient
from common.metrics import evaluate, Metric
from .. import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
 



CSV_PATH = PROJECT_ROOT / "bank_marketing_campain" / "data" / "cleaned_bank-direct-marketing-campaigns.csv"



def load_data():
    df = pd.read_csv(CSV_PATH, index_col=0)
    X = df.drop(columns=["y"]).values.astype(float)
    y = df["y"].values.astype(int)
    # simple train/valid split (80/20)
    split = int(0.8 * len(y))
    return X[:split], y[:split], X[split:], y[split:]


def test_sigmoid():
    print("  sigmoid(0)      :", sigmoid(np.array([0.0])))
    print("  sigmoid(large)  :", sigmoid(np.array([1000.0])))
    print("  sigmoid(-large) :", sigmoid(np.array([-1000.0])))
    print("  [OK] sigmoid")


def test_fista_runs():
    X_train, y_train, _, _ = load_data()
    w, history = fista(X_train, y_train, l=0.01, max_iter=50)
    print(f"  w shape    : {w.shape}")
    print(f"  history len: {len(history)}")
    print(f"  final loss : {history[-1]:.6f}")
    print("  [OK] fista")


def test_logistic_regression_fit_predict():
    X_train, y_train, X_valid, y_valid = load_data()
    model = LogisticRegression(l=0.01, max_iter=100)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_valid)
    print(f"  probs shape : {probs.shape}")
    print(f"  probs range : [{probs.min():.4f}, {probs.max():.4f}]")
    print("  [OK] fit + predict_proba")


def test_all_metrics():
    X_train, y_train, X_valid, y_valid = load_data()
    model = LogisticRegression(l=0.01, max_iter=100)
    model.fit(X_train, y_train)

    for metric in Metric:
        score = model.validate(X_valid, y_valid, metric)
        print(f"  {metric.value:<22}: {score:.4f}")
    print("  [OK] all metrics")


def test_logistic_loss():
    X_train, y_train, _, _ = load_data()
    w = np.zeros(X_train.shape[1])
    loss = logistic_loss(X_train, y_train, w, l=0.01)
    print(f"  loss at zero weights: {loss:.6f}")
    print("  [OK] logistic_loss")


def test_gradient():
    X_train, y_train, _, _ = load_data()
    w = np.zeros(X_train.shape[1])
    grad = gradient(X_train, y_train, w)
    print(f"  grad shape : {grad.shape}")
    print(f"  grad norm  : {np.linalg.norm(grad):.6f}")
    print("  [OK] gradient")


if __name__ == "__main__":
    tests = [
        ("sigmoid",                   test_sigmoid),
        ("logistic_loss",             test_logistic_loss),
        ("gradient",                  test_gradient),
        ("fista runs",                test_fista_runs),
        ("fit + predict_proba",       test_logistic_regression_fit_predict),
        ("all metrics",               test_all_metrics),
    ]

    failed = []
    for name, fn in tests:
        print(f"\n[TEST] {name}")
        try:
            fn()
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed.append(name)

    print("\n" + "="*40)
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    else:
        print(f"All {len(tests)} tests passed.")