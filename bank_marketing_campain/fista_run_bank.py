"""
run_fista.py
============
Bank Marketing dataseti için FISTA çalıştırma scripti.

Adımlar:
  1. bank_preprocessed.csv yükle
  2. Train / Val / Test split (60 / 20 / 20, stratified)
  3. FISTASelector ile lambda seçimi
  4. Test setinde değerlendirme
  5. Sklearn LogisticRegression (penalty='l1') ile karşılaştırma
  6. Grafik: validation metriği vs lambda + regularization path
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	recall_score, precision_score, f1_score,
	balanced_accuracy_score, roc_auc_score, average_precision_score
)

from fista_bank import FISTASelector

# ── Konfigürasyon ──────────────────────────────────────────────────────────────

PREPROCESSED_FILE = "bank_preprocessed.csv"
MEASURE = "roc_auc"        # lambda seçimi için kullanılacak metrik
RANDOM_STATE = 42

# ── Veri yükleme ve split ──────────────────────────────────────────────────────

df = pd.read_csv(PREPROCESSED_FILE)
feature_names = [c for c in df.columns if c != "y"]

X = df[feature_names].values.astype(np.float64)
y = df["y"].values.astype(np.float64)

X_train, X_temp, y_train, y_temp = train_test_split(
	X, y, test_size=0.40, stratify=y, random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
	X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
)

print(f"Train : {X_train.shape[0]} rows")
print(f"Val   : {X_val.shape[0]} rows")
print(f"Test  : {X_test.shape[0]} rows")
print(f"Positive ratio — train: {y_train.mean():.3f}  val: {y_val.mean():.3f}  test: {y_test.mean():.3f}")
print()

# ── FISTA lambda seçimi ────────────────────────────────────────────────────────

print(f"Lambda seçimi: {MEASURE} metriğine göre...")
selector = FISTASelector(
	lambdas=np.logspace(-4, 1, 30),
	max_iter=1000,
	tol=1e-4
)
selector.fit(X_train, y_train, X_val, y_val, measure=MEASURE)
print(f"  → En iyi lambda: {selector.best_lambda:.6f}")
print()

# ── Test değerlendirmesi ───────────────────────────────────────────────────────

def evaluate(name, y_true, y_pred, y_proba):
	print(f"[{name}]")
	print(f"  Recall            : {recall_score(y_true, y_pred, zero_division=0):.4f}")
	print(f"  Precision         : {precision_score(y_true, y_pred, zero_division=0):.4f}")
	print(f"  F1                : {f1_score(y_true, y_pred, zero_division=0):.4f}")
	print(f"  Balanced Accuracy : {balanced_accuracy_score(y_true, y_pred):.4f}")
	print(f"  ROC AUC           : {roc_auc_score(y_true, y_proba):.4f}")
	print(f"  PR AUC            : {average_precision_score(y_true, y_proba):.4f}")
	print()

# FISTA
fista_proba = selector.predict_proba(X_test)
fista_pred  = (fista_proba >= 0.5).astype(int)
evaluate("FISTA (own)", y_test, fista_pred, fista_proba)

# Sklearn karşılaştırma — aynı lambda, penalty='l1', solver='saga'
# C = 1 / (n_samples * lambda)  dönüşümü sklearn konvansiyonu
C_sklearn = 1.0 / (X_train.shape[0] * selector.best_lambda)
sklearn_model = LogisticRegression(
	solver="saga", C=C_sklearn, l1_ratio=1.0,
	max_iter=2000, random_state=RANDOM_STATE
)
sklearn_model.fit(X_train, y_train)
sk_proba = sklearn_model.predict_proba(X_test)[:, 1]
sk_pred  = sklearn_model.predict(X_test)
evaluate("Sklearn L1 LogReg", y_test, sk_pred, sk_proba)

# ── Grafikler ──────────────────────────────────────────────────────────────────

selector.plot(measure=MEASURE)
selector.plot_coefficients(feature_names=feature_names)