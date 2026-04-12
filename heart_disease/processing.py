import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


NUMERIC_FEATURES = [
	"age",
	"trestbps",
	"chol",
	"thalach",
	"oldpeak",
	# binary categorical features
	"sex",
	"fbs",
	"exang",

]

CATEGORICAL_FEATURES = [
	"cp",
	"restecg",
	"slope",
	"ca",
	"thal",
]

TARGET = "target"
LOG_FEATURES = ["oldpeak"]
COLLINEARITY_THRESHOLD = 0.85


def load_data(filepath: str) -> pd.DataFrame:
	try:
		df_raw = pd.read_csv(filepath)
	except FileNotFoundError:
		raise FileNotFoundError(f"File not found: {filepath}")
	
	needed = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
	missing = [c for c in needed if c not in df_raw.columns]
	if missing:
		raise KeyError(f"Expected columns not found: {missing}")
	
	return df_raw[needed].copy()



def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
	"""One-hot encode categorical features with drop_first=True.

	drop_first=True produces k-1 columns per k-category feature,
	eliminating the dummy-variable trap (perfect multicollinearity).
	'unknown' values get their own dummy column — they may carry signal.

	Parameters
	----------
	df : pd.DataFrame

	Returns
	-------
	pd.DataFrame
		Categorical columns replaced by binary dummy columns.
	"""
	df = df.copy()
	df = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=True)
	# Convert bool dummy columns to int (True/False → 1/0)
	bool_cols = df.select_dtypes(include="bool").columns
	df[bool_cols] = df[bool_cols].astype(int)
	return df


def remove_collinear_features(df: pd.DataFrame,
                              threshold: float = COLLINEARITY_THRESHOLD) -> pd.DataFrame:
	df = df.copy()
	
	numeric_cols = [c for c in df.columns
	                if c != TARGET
	                and pd.api.types.is_numeric_dtype(df[c])
	                and not pd.api.types.is_bool_dtype(df[c])]
	
	corr = df[numeric_cols].corr().abs()
	upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
	to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
	
	if to_drop:
		print(f"  [collinearity] Dropping: {to_drop}  (|corr| > {threshold})")
	df = df.drop(columns=to_drop, errors="ignore")
	return df


def scale_and_transform(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	
	for col in LOG_FEATURES:
		if col in df.columns:
			df[col] = np.log1p(df[col])
	
	numeric_to_scale = [c for c in df.columns
	                    if c != TARGET
	                    and pd.api.types.is_numeric_dtype(df[c])
	                    and not pd.api.types.is_bool_dtype(df[c])]
	
	scaler = MinMaxScaler()
	df[numeric_to_scale] = scaler.fit_transform(df[numeric_to_scale])
	
	return df


def run_pipeline(input_path: str, output_path: str) -> pd.DataFrame:

	print("[1/5] Loading data...")
	df = load_data(input_path)
	print(f"      Shape: {df.shape}")
	
	print("[2/5] Checking target distribution...")
	dist = df[TARGET].value_counts().to_dict()
	print(f"      Class distribution — 0: {dist[0]}  1: {dist[1]}")
	
	print("[3/5] Encoding categoricals...")
	df = encode_categoricals(df)
	print(f"      Shape after encoding: {df.shape}")
	
	print("[4/5] Removing collinear features...")
	df = remove_collinear_features(df)
	print(f"      Shape after collinearity removal: {df.shape}")
	
	print("[5/5] Scaling and transforming...")
	df = scale_and_transform(df)
	
	df.to_csv(output_path, index=False)
	print(f"\nDone. Saved to: {output_path}")
	print(f"Final shape:   {df.shape}")
	print(f"Final columns: {list(df.columns)}")
	return df


if __name__ == "__main__":
	import os
	_DIR = os.path.dirname(__file__)
	INPUT_FILE = os.path.join(_DIR, "data/heart.csv")
	OUTPUT_FILE = os.path.join(_DIR, "data/heart_preprocessed.csv")
	run_pipeline(INPUT_FILE, OUTPUT_FILE)