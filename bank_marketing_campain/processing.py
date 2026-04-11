import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


# ── Feature lists ──────────────────────────────────────────────────────────────

NUMERIC_FEATURES = [
	"age",             # client age
	"campaign",        # contacts during this campaign  (right-skewed → log1p)
	"previous",        # contacts before this campaign  (right-skewed → log1p)
	"emp.var.rate",    # employment variation rate
	"cons.price.idx",  # consumer price index
	"cons.conf.idx",   # consumer confidence index
	"euribor3m",       # euribor 3-month rate
	"nr.employed",     # number of employees
]

# Excluded numeric:
#   pdays  — 96 % of rows are 999 ("never contacted"), almost no signal

CATEGORICAL_FEATURES = [
	"job",          # 11 categories
	"marital",      # 3 categories
	"education",    # 7 categories
	"contact",      # 2 categories
	"month",        # 12 categories
	"day_of_week",  # 5 categories
	"poutcome",     # 3 categories
]

# Excluded categorical:
#   default  — ~0.2 % "yes", effectively constant
#   housing  — redundant loan-status info; keeping feature count minimal
#   loan     — same reason as housing

TARGET = "y"
LOG_FEATURES = ["campaign", "previous"]
COLLINEARITY_THRESHOLD = 0.85


def load_data(filepath: str) -> pd.DataFrame:
	"""Load raw Bank Marketing CSV and keep only the columns we need.

	Parameters
	----------
	filepath : str
		Path to bank-additional-full.csv (semicolon-separated).

	Returns
	-------
	pd.DataFrame
		Raw subset with selected columns and target.
	"""
	try:
		df_raw = pd.read_csv(filepath)
	except FileNotFoundError:
		raise FileNotFoundError(f"File not found: {filepath}")
	
	needed = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
	missing = [c for c in needed if c not in df_raw.columns]
	if missing:
		raise KeyError(f"Expected columns not found: {missing}")
	
	return df_raw[needed].copy()


def binarize_labels(df: pd.DataFrame) -> pd.DataFrame:
	"""Convert target column 'y' from yes/no strings to 1/0 integers.

	Parameters
	----------
	df : pd.DataFrame

	Returns
	-------
	pd.DataFrame
		Same dataframe with 'y' as int (1 = subscribed, 0 = not subscribed).
	"""
	df = df.copy()
	df[TARGET] = df[TARGET].map({"yes": 1, "no": 0})
	if df[TARGET].isna().any():
		raise ValueError("Unexpected values in target column 'y'.")
	return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
	"""Add campaign_ratio: contacts this campaign relative to prior history.

	High ratio → client contacted heavily for the first time.
	Low ratio  → long prior history relative to current contacts.
	Captures a different dimension than campaign and previous individually.

	Parameters
	----------
	df : pd.DataFrame
		Must contain 'campaign' and 'previous' (before log transform).

	Returns
	-------
	pd.DataFrame
		DataFrame with campaign_ratio column appended.
	"""
	df = df.copy()
	epsilon = 1e-6
	df["campaign_ratio"] = df["campaign"] / (df["previous"] + epsilon)
	return df


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
	"""Drop one column from each highly-correlated numeric pair.

	emp.var.rate, euribor3m, nr.employed are correlated ~0.9+ in this dataset.
	Keeping all three inflates variance and complicates FISTA coefficients.

	Parameters
	----------
	df : pd.DataFrame
	threshold : float
		Absolute Pearson correlation above which one column is dropped.

	Returns
	-------
	pd.DataFrame
		DataFrame with collinear columns removed.
	"""
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
	"""Apply log1p to skewed counts then MinMax-scale all numeric features.

	Log1p compresses the right tail of campaign and previous before scaling
	(same approach as poker's stack/pot columns).
	Bool dummy columns are excluded — they are already 0/1.

	Parameters
	----------
	df : pd.DataFrame

	Returns
	-------
	pd.DataFrame
		Fully scaled DataFrame ready for logistic regression / FISTA.
	"""
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
	"""End-to-end preprocessing pipeline for the Bank Marketing dataset.

	Steps
	-----
	1. Load and subset columns
	2. Binarize target  (yes/no → 1/0)
	3. Engineer features  (campaign_ratio)
	4. One-hot encode categoricals
	5. Remove collinear numeric features
	6. Log1p + MinMax scale

	Parameters
	----------
	input_path : str
		Path to raw bank-additional-full.csv (semicolon-separated).
	output_path : str
		Path where the preprocessed CSV will be saved.

	Returns
	-------
	pd.DataFrame
		Preprocessed DataFrame (also saved to output_path).
	"""
	print("[1/6] Loading data...")
	df = load_data(input_path)
	print(f"      Shape: {df.shape}")
	
	print("[2/6] Binarizing labels...")
	df = binarize_labels(df)
	dist = df[TARGET].value_counts().to_dict()
	print(f"      Class distribution — 0: {dist[0]}  1: {dist[1]}")
	
	print("[3/6] Engineering features...")
	df = engineer_features(df)
	
	print("[4/6] Encoding categoricals...")
	df = encode_categoricals(df)
	print(f"      Shape after encoding: {df.shape}")
	
	print("[5/6] Removing collinear features...")
	df = remove_collinear_features(df)
	print(f"      Shape after collinearity removal: {df.shape}")
	
	print("[6/6] Scaling and transforming...")
	df = scale_and_transform(df)
	
	df.to_csv(output_path, index=False)
	print(f"\nDone. Saved to: {output_path}")
	print(f"Final shape:   {df.shape}")
	print(f"Final columns: {list(df.columns)}")
	return df


if __name__ == "__main__":
	INPUT_FILE = "bank-direct-marketing-campaigns.csv"
	OUTPUT_FILE = "bank_preprocessed.csv"
	run_pipeline(INPUT_FILE, OUTPUT_FILE)