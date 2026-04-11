"""
run_experiments.py
==================
Task 3 deneyleri:
  - Naive vs UnlabeledLogReg (algo1, algo2) vs Oracle
  - 4 missing data şeması: MCAR, MAR1, MAR2, MNAR
  - MCAR için farklı c değerleri analizi

Kullanım:
    python run_experiments.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
	accuracy_score, balanced_accuracy_score,
	f1_score, roc_auc_score
)
from fista import FISTASelector
from missing_data import generate_missing
from unlabeled_logreg import UnlabeledLogReg

# -------------------------------------------------------------------
# 0. Sabitler
# -------------------------------------------------------------------

SEED        = 42
C_DEFAULT   = 0.3
LAMBDAS     = np.logspace(-4, 1, 15)
MEASURE     = 'f1'
METRICS     = ['accuracy', 'balanced_accuracy', 'f1', 'roc_auc']
SCHEMES     = ['mcar', 'mar1', 'mar2', 'mnar']
C_VALUES    = [0.1, 0.2, 0.3, 0.4, 0.5]   # MCAR analizi için

FEATURE_IDX = 11   # pot_growth — MAR1/MNAR için

# -------------------------------------------------------------------
# 1. Veriyi yükle ve böl
# -------------------------------------------------------------------

df = pd.read_csv("poker_data_preprocessed.csv")
X  = df.drop(columns=["result"]).to_numpy()
y  = df["result"].to_numpy()
feature_names = df.drop(columns=["result"]).columns.tolist()

print(f"Dataset: {X.shape[0]} satır, {X.shape[1]} feature\n")

X_trainval, X_test, y_trainval, y_test = train_test_split(
	X, y, test_size=0.2, random_state=SEED, stratify=y
)
X_train, X_valid, y_train, y_valid = train_test_split(
	X_trainval, y_trainval, test_size=0.25, random_state=SEED, stratify=y_trainval
)

print(f"Train: {len(y_train)}  Valid: {len(y_valid)}  Test: {len(y_test)}\n")

# -------------------------------------------------------------------
# 2. Yardımcı fonksiyonlar
# -------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_proba):
	"""4 metriği dict olarak döndür."""
	return {
		'accuracy':          accuracy_score(y_true, y_pred),
		'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
		'f1':                f1_score(y_true, y_pred, zero_division=0),
		'roc_auc':           roc_auc_score(y_true, y_proba),
	}


def run_naive(X_tr, y_obs, X_val, y_val, X_te, y_te):
	"""Sadece labeled gözlemlerle FISTA."""
	mask = (y_obs != -1)
	sel  = FISTASelector(lambdas=LAMBDAS, max_iter=500)
	sel.fit(X_tr[mask], y_obs[mask], X_val, y_val, measure=MEASURE)
	proba = sel.predict_proba(X_te)
	pred  = (proba >= 0.5).astype(int)
	return compute_metrics(y_te, pred, proba)


def run_oracle(X_tr, y_tr, X_val, y_val, X_te, y_te):
	"""Tüm gerçek etiketlerle FISTA."""
	sel = FISTASelector(lambdas=LAMBDAS, max_iter=500)
	sel.fit(X_tr, y_tr, X_val, y_val, measure=MEASURE)
	proba = sel.predict_proba(X_te)
	pred  = (proba >= 0.5).astype(int)
	return compute_metrics(y_te, pred, proba)


def run_unlabeled(X_tr, y_obs, X_val, y_val, X_te, y_te, method):
	"""UnlabeledLogReg — label_propagation veya em."""
	model = UnlabeledLogReg(
		method=method,
		lambdas=LAMBDAS,
		measure=MEASURE,
		max_iter_em=10,
		max_iter_fista=500,
		random_state=SEED,
	)
	model.fit(X_tr, y_obs, X_val, y_val)
	proba = model.predict_proba(X_te)
	pred  = (proba >= 0.5).astype(int)
	return compute_metrics(y_te, pred, proba)


def print_results(title, results):
	"""Sonuçları tabloya yazdır."""
	print(f"\n{'='*55}")
	print(f"  {title}")
	print(f"{'='*55}")
	print(f"{'Yöntem':22s}  {'Acc':>6}  {'BalAcc':>6}  {'F1':>6}  {'AUC':>6}")
	print(f"{'-'*55}")
	for name, m in results.items():
		print(
			f"{name:22s}  "
			f"{m['accuracy']:.4f}  "
			f"{m['balanced_accuracy']:.4f}  "
			f"{m['f1']:.4f}  "
			f"{m['roc_auc']:.4f}"
		)


# -------------------------------------------------------------------
# 3. Ana deney: 4 şema × 4 yöntem
# -------------------------------------------------------------------

print("=" * 55)
print("  DENEY 1: 4 Şema Karşılaştırması")
print("=" * 55)

scheme_kwargs = {
	'mcar': {'scheme': 'mcar'},
	'mar1': {'scheme': 'mar1', 'feature_idx': FEATURE_IDX},
	'mar2': {'scheme': 'mar2'},
	'mnar': {'scheme': 'mnar', 'feature_idx': FEATURE_IDX, 'y_weight': 5.0},
}

all_results = {}

for scheme in SCHEMES:
	print(f"\n--- {scheme.upper()} (c={C_DEFAULT}) ---")
	
	y_obs = generate_missing(
		X_train, y_train, c=C_DEFAULT,
		random_state=SEED, **scheme_kwargs[scheme]
	)
	n_missing = (y_obs == -1).sum()
	print(f"Eksik etiket: {n_missing} ({n_missing/len(y_obs)*100:.1f}%)")
	
	results = {}
	results['Naive']             = run_naive(X_train, y_obs, X_valid, y_valid, X_test, y_test)
	results['LabelProp']         = run_unlabeled(X_train, y_obs, X_valid, y_valid, X_test, y_test, 'label_propagation')
	results['EM']                = run_unlabeled(X_train, y_obs, X_valid, y_valid, X_test, y_test, 'em')
	results['Oracle']            = run_oracle(X_train, y_train, X_valid, y_valid, X_test, y_test)
	
	print_results(f"Şema: {scheme.upper()}", results)
	all_results[scheme] = results

# -------------------------------------------------------------------
# 4. MCAR analizi: farklı c değerleri
# -------------------------------------------------------------------

print("\n\n" + "=" * 55)
print("  DENEY 2: MCAR — Farklı c Değerleri")
print("=" * 55)

mcar_results = {m: {method: [] for method in ['Naive', 'LabelProp', 'EM', 'Oracle']}
                for m in METRICS}

for c in C_VALUES:
	print(f"\n  c = {c}")
	y_obs = generate_missing(X_train, y_train, scheme='mcar', c=c, random_state=SEED)
	
	r = {
		'Naive':     run_naive(X_train, y_obs, X_valid, y_valid, X_test, y_test),
		'LabelProp': run_unlabeled(X_train, y_obs, X_valid, y_valid, X_test, y_test, 'label_propagation'),
		'EM':        run_unlabeled(X_train, y_obs, X_valid, y_valid, X_test, y_test, 'em'),
		'Oracle':    run_oracle(X_train, y_train, X_valid, y_valid, X_test, y_test),
	}
	for metric in METRICS:
		for method in r:
			mcar_results[metric][method].append(r[method][metric])

# -------------------------------------------------------------------
# 5. Grafikler
# -------------------------------------------------------------------

# Grafik 1: 4 şema × 4 metrik — grouped bar chart
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()
methods  = ['Naive', 'LabelProp', 'EM', 'Oracle']
colors   = ['#E24B4A', '#378ADD', '#1D9E75', '#888780']
x        = np.arange(len(SCHEMES))
width    = 0.18

for idx, metric in enumerate(METRICS):
	ax = axes[idx]
	for i, (method, color) in enumerate(zip(methods, colors)):
		vals = [all_results[s][method][metric] for s in SCHEMES]
		ax.bar(x + i * width - 1.5 * width, vals, width,
		       label=method, color=color, alpha=0.85)
	ax.set_xticks(x)
	ax.set_xticklabels([s.upper() for s in SCHEMES])
	ax.set_title(metric)
	ax.set_ylim(0, 1.05)
	ax.legend(fontsize=8)
	ax.grid(axis='y', alpha=0.3)
	ax.set_ylabel('skor')

fig.suptitle('Yöntem Karşılaştırması — 4 Missing Data Şeması', fontsize=14)
plt.tight_layout()
plt.savefig('results_schemes.png', dpi=150, bbox_inches='tight')
plt.show()

# Grafik 2: MCAR — c değerine göre performans
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, metric in enumerate(METRICS):
	ax = axes[idx]
	for method, color in zip(methods, colors):
		ax.plot(C_VALUES, mcar_results[metric][method],
		        'o-', label=method, color=color, linewidth=2, markersize=6)
	ax.set_xlabel('c (missing oranı)')
	ax.set_ylabel('skor')
	ax.set_title(metric)
	ax.set_ylim(0, 1.05)
	ax.legend(fontsize=8)
	ax.grid(alpha=0.3)

fig.suptitle('MCAR — Farklı c Değerleri', fontsize=14)
plt.tight_layout()
plt.savefig('results_mcar_c.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nDeney tamamlandı.")
print("Grafikler: results_schemes.png, results_mcar_c.png")

