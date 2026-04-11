
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
	accuracy_score, balanced_accuracy_score,
	f1_score, roc_auc_score
)

from fista_bank import LogisticRegressionFISTA, FISTASelector
from missing_data_bank import generate_missing
from unlabeled_logreg_bank import UnlabeledLogReg

PREPROCESSED_FILE = "bank_preprocessed.csv"
C_MISSING = 0.3
RANDOM_STATE = 42
LAMBDA_VAL = 1e-3        # For EM and LP constant  lambda (from run_fista)
N_NEIGHBORS = 15         # label propagation k-NN

SCHEMES = {
	"mcar": {"scheme": "mcar"},
	"mar1": {"scheme": "mar1", "feature_idx": 0},   # age = index 0
	"mar2": {"scheme": "mar2"},
	"mnar": {"scheme": "mnar", "feature_idx": 0, "y_weight": 5.0},
}


df = pd.read_csv(PREPROCESSED_FILE)
feature_names = [c for c in df.columns if c != "y"]
X_all = df[feature_names].values.astype(np.float64)
y_all = df["y"].values.astype(np.float64)

# Train(60) / Val(20) / Test(20) — stratified
X_tv, X_test, y_tv, y_test = train_test_split(
	X_all, y_all, test_size=0.20, stratify=y_all, random_state=RANDOM_STATE
)
X_train, X_val, y_train, y_val = train_test_split(
	X_tv, y_tv, test_size=0.25, stratify=y_tv, random_state=RANDOM_STATE
)

print(f"Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")
print(f"Positive ratio — train: {y_train.mean():.3f}  test: {y_test.mean():.3f}\n")

# Lambda (oracle train, val)
print("Selecting best lambda on oracle data...")
selector = FISTASelector(lambdas=np.logspace(-4, 1, 20), max_iter=1000, tol=1e-4)
selector.fit(X_train, y_train, X_val, y_val, measure="roc_auc")
BEST_LAMBDA = selector.best_lambda
print(f"Best lambda: {BEST_LAMBDA:.6f}\n")


def evaluate(y_true, y_proba):
	y_pred = (y_proba >= 0.5).astype(int)
	return {
		"Accuracy":          round(accuracy_score(y_true, y_pred), 4),
		"Balanced Accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
		"F1":                round(f1_score(y_true, y_pred, zero_division=0), 4),
		"ROC AUC":           round(roc_auc_score(y_true, y_proba), 4),
	}

# Oracle baseline

oracle_model = LogisticRegressionFISTA(
	lambda_val=BEST_LAMBDA, max_iter=1000, tol=1e-4
)
oracle_model.fit(X_train, y_train)
oracle_scores = evaluate(y_test, oracle_model.predict_proba(X_test))

# main experiament

all_results = []

for scheme_name, scheme_kwargs in SCHEMES.items():
	print("=" * 60)
	print(f"Scheme: {scheme_name.upper()}  (c={C_MISSING})")
	print("=" * 60)
	
	y_obs = generate_missing(
		X_train, y_train,
		c=C_MISSING,
		random_state=RANDOM_STATE,
		**scheme_kwargs,
	)
	n_missing = (y_obs == -1).sum()
	print(f"  Missing labels: {n_missing} / {len(y_obs)} ({n_missing/len(y_obs):.1%})\n")
	
	labeled_mask = y_obs != -1
	naive_model = LogisticRegressionFISTA(
		lambda_val=BEST_LAMBDA, max_iter=1000, tol=1e-4
	)
	naive_model.fit(X_train[labeled_mask], y_obs[labeled_mask])
	naive_scores = evaluate(y_test, naive_model.predict_proba(X_test))
	print(f"  Naive  : {naive_scores}")
	
	em_model = UnlabeledLogReg(
		method="em",
		lambda_val=BEST_LAMBDA,
		max_iter=20,
		tol=1e-4,
		fista_max_iter=1000,
		random_state=RANDOM_STATE,
	)
	em_model.fit(X_train, y_obs)
	em_scores = evaluate(y_test, em_model.predict_proba(X_test))
	print(f"  EM     : {em_scores}")
	
	# Label Propagation
	lp_model = UnlabeledLogReg(
		method="label_propagation",
		lambda_val=BEST_LAMBDA,
		max_iter=20,
		tol=1e-4,
		fista_max_iter=1000,
		n_neighbors=N_NEIGHBORS,
		random_state=RANDOM_STATE,
	)
	lp_model.fit(X_train, y_obs)
	lp_scores = evaluate(y_test, lp_model.predict_proba(X_test))
	print(f"  LP     : {lp_scores}")
	
	# Oracle
	print(f"  Oracle : {oracle_scores}\n")
	
	# Save
	for method_name, scores in [
		("Naive", naive_scores),
		("EM", em_scores),
		("Label Propagation", lp_scores),
		("Oracle", oracle_scores),
	]:
		row = {"Scheme": scheme_name.upper(), "Method": method_name}
		row.update(scores)
		all_results.append(row)

#reasults

results_df = pd.DataFrame(all_results)
results_df = results_df.set_index(["Scheme", "Method"])

print("\n" + "=" * 60)
print("FULL RESULTS TABLE")
print("=" * 60)
print(results_df.to_string())

results_df.to_csv("experiment_results.csv")
print("\nSaved to experiment_results.csv")