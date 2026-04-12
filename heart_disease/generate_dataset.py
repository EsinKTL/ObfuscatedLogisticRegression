"""
generate_datasets.py
--------------------
Applies the four missing-label schemes to the Heart Disease training split
and saves one CSV per scheme.

Output files
------------
    bank_mcar.csv
    bank_mar1.csv
    bank_mar2.csv
    bank_mnar.csv
"""

import pandas as pd
from missing_data import generate_missing

df = pd.read_csv("data/heart_preprocessed.csv")

X = df.drop(columns=["target"])
y = df["target"]

print(f"Dataset loaded: {X.shape[0]} rows, {X.shape[1]} features")
print(f"Positive class ratio: {y.mean():.2f}\n")

C = 0.3
SEED = 42

# age is column index 0 after preprocessing — used for MAR1 and MNAR
target_feature = "age"
FEATURE_IDX = X.columns.get_loc(target_feature)

schemes = {
	"mcar": {"scheme": "mcar"},
	"mar1": {"scheme": "mar1", "feature_idx": FEATURE_IDX},
	"mar2": {"scheme": "mar2"},
	"mnar": {"scheme": "mnar", "feature_idx": FEATURE_IDX, "y_weight": 5.0},
}

for name, kwargs in schemes.items():
	y_obs = generate_missing(X, y, c=C, random_state=SEED, **kwargs)
	
	df_out = X.copy()
	df_out["y_obs"] = y_obs
	
	n_missing = (y_obs == -1).sum()
	pct = n_missing / len(y_obs) * 100
	print(f"{name.upper()}: {n_missing} labels hidden ({pct:.1f}%)")
	
	filename = f"data/heart_{name}.csv"
	df_out.to_csv(filename, index=False)
	print(f"  → {filename} saved\n")

print("All datasets have been created.")