"""
    poker_mcar.csv
    poker_mar1.csv
    poker_mar2.csv
    poker_mnar.csv
"""

import pandas as pd
from missing_data import generate_missing

df = pd.read_csv("poker_data_preprocessed.csv")

X = df.drop(columns=["result"])
y = df["result"]

print(f"Dataset loaded: {X.shape[0]} rows, {X.shape[1]} features")
print(f"Positive class ratio: {y.mean():.2f}\n")


C = 0.3
SEED = 42
FEATURE_IDX = 13


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

    filename = f"poker_{name}.csv"
    df_out.to_csv(filename, index=False)
    print(f"  → {filename} saved\n")

print("All datasets have been created.")