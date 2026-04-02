# Project 1 — Logistic Regression with Missing Labels

**Course:** Advanced Machine Learning, Politechnika Warszawska  
**Objective:** Analyze logistic regression in a binary classification setting where training data contains observations with missing labels, under four distinct missingness mechanisms.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Theoretical Background](#3-theoretical-background)
4. [Scripts API Reference](#4-scripts-api-reference)
   - 4.1 [`preprocessing.py`](#41-preprocessingpy)
   - 4.2 [`missing_data.py`](#42-missing_datapy)
5. [Dataset 1: ATP Tennis Matches](#5-dataset-1-atp-tennis-matches)
   - 5.1 [Data Source](#51-data-source)
   - 5.2 [Target Variable Design](#52-target-variable-design)
   - 5.3 [The Symmetrization Problem and Solution](#53-the-symmetrization-problem-and-solution)
   - 5.4 [Feature Engineering](#54-feature-engineering)
   - 5.5 [Preprocessing Pipeline Results](#55-preprocessing-pipeline-results)
   - 5.6 [Missing-Label Generation Results](#56-missing-label-generation-results)
   - 5.7 [Baseline Model Performance](#57-baseline-model-performance)
6. [Output Files](#6-output-files)
7. [Reproducing Results](#7-reproducing-results)
8. [Design Decisions and Lessons Learned](#8-design-decisions-and-lessons-learned)

---

## 1. Project Overview

We consider a binary classification problem where Y ∈ {0, 1} is the response variable and X is the feature vector. A missingness indicator S governs label observability:

| S | Label Status | Convention |
|---|---|---|
| S = 0 | Label **observed** | Y_obs = Y (0 or 1) |
| S = 1 | Label **missing** | Y_obs = −1 |

The project requires:

1. **Four real-world datasets** prepared for logistic regression (missing values filled, collinear variables removed, all features numerical).
2. **Four missingness schemes** implemented as reusable functions that take fully-labelled (X, Y) data and return (X, Y_obs) data.

**Current status:** One dataset (ATP Tennis) is fully implemented end-to-end. Three more datasets are pending.

---

## 2. Repository Structure

```
Project 1/
├── README.md                  ← This file
├── datasets/                  ← Raw data (ATP match CSVs, 1968–2024)
│   ├── atp_matches_2015.csv
│   ├── atp_matches_2016.csv
│   ├── ...
│   └── atp_matches_2024.csv
├── scripts/                   ← Reusable Python utilities (dataset-agnostic)
│   ├── __init__.py
│   ├── preprocessing.py       ← Imputation, collinearity removal, standardization
│   └── missing_data.py        ← MCAR, MAR1, MAR2, MNAR label-hiding generators
├── notebooks/                 ← Dataset-specific logic (Jupyter notebooks)
│   └── 01_atp_dataset.ipynb   ← ATP dataset: loading, feature engineering, analysis
└── processed/                 ← Cleaned, ready-to-use datasets
    ├── atp_upset.csv           ← Fully labelled ground truth
    ├── atp_upset_MCAR.csv      ← Labels hidden under MCAR
    ├── atp_upset_MAR1.csv      ← Labels hidden under MAR1
    ├── atp_upset_MAR2.csv      ← Labels hidden under MAR2
    └── atp_upset_MNAR.csv      ← Labels hidden under MNAR
```

### Separation of Concerns

| Layer | Location | Purpose | Reusability |
|---|---|---|---|
| **General utilities** | `scripts/` | Preprocessing, missing-data generation | Shared across all 4 datasets |
| **Dataset-specific logic** | `notebooks/` | Loading raw files, domain-specific feature engineering, visualizations | One notebook per dataset |
| **Outputs** | `processed/` | Clean CSVs ready for downstream experiments | Consumed by later tasks |

This separation means that when adding a new dataset (e.g., Dataset 2), you only need a new notebook in `notebooks/` — the scripts are imported and reused as-is.

---

## 3. Theoretical Background

### Missingness Mechanisms

The four implemented schemes correspond to the standard taxonomy from Rubin (1976):

| Scheme | Formal Definition | Interpretation |
|---|---|---|
| **MCAR** | P(S=1 \| X, Y) = c | Missingness is a coin flip with constant probability c. Completely independent of both features and the label. |
| **MAR1** | P(S=1 \| X, Y) = P(S=1 \| X_j) | Missingness depends on a **single** observed feature X_j. Given X_j, it is independent of Y. |
| **MAR2** | P(S=1 \| X, Y) = P(S=1 \| X) | Missingness depends on **all** observed features jointly. Given X, it is independent of Y. |
| **MNAR** | P(S=1 \| X, Y) depends on Y | Missingness depends on the **unobserved label itself** (and possibly on X). This is the most adversarial setting. |

### Implementation Approach

All non-MCAR schemes model the missingness probability through a **sigmoid link function**:

```
P(S=1 | ·) = σ(z)  where  z = linear_predictor + intercept
```

The intercept is **calibrated** via bisection search so that the average missingness probability across the dataset matches the desired `missing_rate` (default: 0.3). This ensures all schemes produce approximately the same overall fraction of missing labels, making them directly comparable.

---

## 4. Scripts API Reference

### 4.1 `preprocessing.py`

**Module purpose:** Dataset-agnostic preprocessing utilities. All functions are **pure** — they return new DataFrames without modifying the input.

#### `impute_missing(df, numeric_strategy, categorical_strategy, exclude_cols) → DataFrame`

Fills missing values column-by-column.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `df` | DataFrame | — | Input data |
| `numeric_strategy` | `"median"` \| `"mean"` | `"median"` | Fill strategy for numeric columns |
| `categorical_strategy` | `"mode"` \| `"missing"` | `"mode"` | Fill strategy for categorical columns. `"missing"` inserts a literal `"_MISSING_"` string. |
| `exclude_cols` | list[str] \| None | `None` | Columns to skip (e.g., the target column) |

**Returns:** A copy of `df` with NaN values replaced.

**Behavior details:**
- Iterates over every column in `df`.
- Skips columns that have zero missing values (short-circuit).
- For numeric columns: fills with the column median (or mean).
- For categorical/object columns: fills with the mode (most frequent value), or with `"_MISSING_"` if requested. Falls back to `"_MISSING_"` if the mode is empty.

---

#### `remove_collinear_by_correlation(df, feature_cols, threshold, verbose) → (DataFrame, list[str])`

Drops features whose pairwise Pearson |r| exceeds a threshold.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `df` | DataFrame | — | Input data |
| `feature_cols` | list[str] | — | Numeric columns to check |
| `threshold` | float | `0.90` | Absolute correlation ceiling |
| `verbose` | bool | `True` | Print dropped columns |

**Returns:** `(cleaned_df, remaining_feature_cols)` — the DataFrame with collinear columns removed, and the updated list of feature column names.

**Tie-breaking strategy:** When two features are correlated above the threshold, the one with the **higher mean absolute correlation** with all other features is dropped. This keeps the feature that is more "independent" of the rest of the feature set.

**Algorithm:**
1. Compute the full absolute correlation matrix.
2. Extract the upper triangle (to avoid double-counting pairs).
3. For each pair exceeding the threshold, compute each feature's mean |r| with all remaining features.
4. Drop the more redundant one.

---

#### `remove_collinear_by_vif(df, feature_cols, threshold, verbose) → (DataFrame, list[str])`

Iteratively drops the feature with the highest Variance Inflation Factor until all VIFs fall below the threshold.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `df` | DataFrame | — | Input data |
| `feature_cols` | list[str] | — | Numeric columns to check |
| `threshold` | float | `10.0` | Maximum allowable VIF |
| `verbose` | bool | `True` | Print each drop decision |

**Returns:** `(cleaned_df, remaining_feature_cols)`

**Algorithm:**
1. Compute VIF for every remaining feature using `statsmodels.stats.outliers_influence.variance_inflation_factor`.
2. If the maximum VIF ≤ threshold → stop.
3. Otherwise, drop the feature with the highest VIF and repeat.

**Safety guards:**
- Near-constant columns (std < 1e-10) are dropped preemptively to avoid numerical instability in VIF computation.
- Stops if fewer than 2 features remain.

---

#### `drop_low_variance(df, feature_cols, threshold, verbose) → (DataFrame, list[str])`

Removes numeric columns whose variance falls below a threshold (default: 1e-8).

---

#### `standardize(df, feature_cols) → (DataFrame, means, stds)`

Z-score standardization: `(x - mean) / std`. Returns the transformed DataFrame along with the fitted means and standard deviations for potential inverse transformation.

---

#### `full_preprocessing_pipeline(df, feature_cols, target_col, ...) → (DataFrame, list[str])`

End-to-end preprocessing in a single call. Executes the following steps in order:

| Step | Function Called | Default Config |
|---|---|---|
| 1. Impute missing values | `impute_missing()` | median/mode, target excluded |
| 2. Drop low-variance features | `drop_low_variance()` | threshold = 1e-8 |
| 3. Remove correlation-based collinearity | `remove_collinear_by_correlation()` | \|r\| > 0.90 |
| 4. Remove VIF-based collinearity | `remove_collinear_by_vif()` | VIF > 10.0 |
| 5. (Optional) Standardize | `standardize()` | Off by default |

| Parameter | Type | Default | Description |
|---|---|---|---|
| `df` | DataFrame | — | Features + target in a single DataFrame |
| `feature_cols` | list[str] | — | Which columns are features |
| `target_col` | str | — | Name of the target column (excluded from imputation) |
| `impute` | bool | `True` | Whether to run step 1 |
| `corr_threshold` | float | `0.90` | For step 3 |
| `vif_threshold` | float | `10.0` | For step 4 |
| `do_standardize` | bool | `False` | Whether to run step 5 |
| `verbose` | bool | `True` | Print progress and decisions |

**Returns:** `(cleaned_df, final_feature_cols)` — the cleaned DataFrame (including the target column) and the surviving feature column names.

---

### 4.2 `missing_data.py`

**Module purpose:** Generate label-missingness according to four theoretical schemes. All functions share a uniform signature and return type, so they can be used interchangeably in experiment loops.

#### Common Signature

Every generator function has this form:

```
generate_<scheme>(X, y, missing_rate=0.3, random_state=121553223, **kwargs) → (X, y_obs, probs)
```

| Return Value | Type | Description |
|---|---|---|
| `X` | DataFrame | Feature matrix (returned unchanged) |
| `y_obs` | ndarray | Observed labels: original Y where S=0, **−1** where S=1 |
| `probs` | ndarray | Per-observation missingness probabilities P(S=1 \| ·) |

The `probs` vector is returned for diagnostic/visualization purposes — it lets you verify that the missingness mechanism has the intended shape.

---

#### `generate_mcar(X, y, missing_rate, random_state) → (X, y_obs, probs)`

**MCAR — Missing Completely At Random**

```
P(S=1 | X, Y) = c = missing_rate
```

Every observation has the **same constant** probability of having its label hidden. The missingness is independent of both features and the true label.

| Parameter | Default | Description |
|---|---|---|
| `missing_rate` | `0.3` | Constant probability c |
| `random_state` | `121553223` | Seed for the NumPy Generator |

**Implementation:** Creates a constant probability vector `probs = [c, c, ..., c]` and draws Bernoulli samples.

---

#### `generate_mar1(X, y, missing_rate, feature_col, random_state) → (X, y_obs, probs)`

**MAR1 — Missing At Random, single-feature dependence**

```
P(S=1 | X, Y) = P(S=1 | X_j) = σ(α · standardize(X_j) + b)
```

Missingness depends on exactly **one** explanatory variable X_j.

| Parameter | Default | Description |
|---|---|---|
| `missing_rate` | `0.3` | Target overall missing rate |
| `feature_col` | `None` | Which feature drives missingness. If `None`, automatically selects the feature with the **highest absolute correlation with Y**. |
| `random_state` | `121553223` | Seed |

**Implementation details:**
1. The driving feature X_j is standardized (zero mean, unit variance) for numerical stability.
2. The slope coefficient α is fixed at 1.0 (higher X_j → higher missingness probability).
3. The intercept b is calibrated via **bisection** so that `mean(σ(α·X_j_std + b)) ≈ missing_rate`.
4. Bernoulli draws are made from the resulting per-observation probabilities.

**Automatic feature selection:** When `feature_col=None`, the function computes `|corr(X_j, Y)|` for every feature and picks the one with the strongest correlation. This ensures the MAR1 mechanism creates the most interesting/pronounced differential missingness across classes.

---

#### `generate_mar2(X, y, missing_rate, random_state) → (X, y_obs, probs)`

**MAR2 — Missing At Random, all-feature dependence**

```
P(S=1 | X, Y) = P(S=1 | X) = σ(X_std · w + b)
```

Missingness depends on **all** features simultaneously through a linear combination.

| Parameter | Default | Description |
|---|---|---|
| `missing_rate` | `0.3` | Target overall missing rate |
| `random_state` | `121553223` | Seed (also used to draw the weight vector) |

**Implementation details:**
1. All features are standardized independently.
2. A random weight vector **w** is drawn from N(0, 1) and then **unit-normalized** (`w / ||w||`), ensuring the linear predictor's scale is controlled regardless of the number of features.
3. The intercept b is calibrated via bisection.
4. Bernoulli draws from the resulting probabilities.

**Note:** The weight vector w is drawn from the **same RNG** that later generates the Bernoulli mask. This means the first `p` random draws (where p = number of features) are consumed for w, and subsequent draws produce the mask. The `random_state` parameter makes this entirely deterministic.

---

#### `generate_mnar(X, y, missing_rate, y_weight, random_state) → (X, y_obs, probs)`

**MNAR — Missing Not At Random**

```
P(S=1 | X, Y) = σ(X_std · w + γ · Y + b)
```

Missingness depends on both the features **and the true label Y itself**. This is the most adversarial mechanism.

| Parameter | Default | Description |
|---|---|---|
| `missing_rate` | `0.3` | Target overall missing rate |
| `y_weight` | `2.0` | Coefficient γ for Y in the linear predictor. Positive γ means **positive-class labels (Y=1) are more likely to be hidden**. |
| `random_state` | `121553223` | Seed |

**Implementation details:**
1. Features are standardized, and a unit-norm weight vector w is drawn (same as MAR2).
2. The term `γ · Y` is added to the linear predictor before applying the sigmoid. Since Y ∈ {0, 1}, this shifts the log-odds of missingness **upward** for positive-class observations.
3. The intercept b is calibrated via bisection to maintain the target overall rate.

**Effect of γ = 2.0 (default):** In practice this creates a substantial asymmetry — positive-class observations are roughly 2.5× more likely to have their labels hidden than negative-class observations (exact ratio depends on the data).

---

#### `generate_missing(scheme, X, y, missing_rate, random_state, **kwargs) → (X, y_obs, probs)`

**Convenience dispatcher.** Routes to the appropriate generator based on the `scheme` string.

| `scheme` value | Function called |
|---|---|
| `"MCAR"` | `generate_mcar()` |
| `"MAR1"` | `generate_mar1()` |
| `"MAR2"` | `generate_mar2()` |
| `"MNAR"` | `generate_mnar()` |

Any additional keyword arguments (e.g., `feature_col` for MAR1, `y_weight` for MNAR) are forwarded to the underlying function.

Raises `ValueError` if `scheme` is not one of the four recognized strings.

---

#### `missingness_summary(y_true, y_obs) → dict`

Diagnostic helper that computes summary statistics about the generated missingness.

**Returns a dict with:**

| Key | Type | Description |
|---|---|---|
| `n_total` | int | Total number of observations |
| `n_missing` | int | Count where y_obs = −1 |
| `n_observed` | int | Count where y_obs ≠ −1 |
| `missing_rate` | float | n_missing / n_total |
| `missing_rate_class0` | float | Fraction of true Y=0 observations that were hidden |
| `missing_rate_class1` | float | Fraction of true Y=1 observations that were hidden |

The per-class rates are the key diagnostic: for MCAR they should be approximately equal; for MNAR they should be substantially different.

---

#### Internal Helpers

These are prefixed with `_` and not part of the public API, but are documented here for completeness:

| Function | Purpose |
|---|---|
| `_sigmoid(z)` | Numerically stable sigmoid using the split formula: `1/(1+exp(-z))` for z ≥ 0, `exp(z)/(1+exp(z))` for z < 0. Avoids overflow. |
| `_apply_mask(y, probs, rng)` | Draws Bernoulli(probs) for each observation and replaces y with −1 where S=1. |
| `_calibrate_intercept(z_raw, target_rate, tol, max_iter)` | Bisection search over [−30, +30] to find intercept b such that `mean(σ(z_raw + b)) ≈ target_rate` within tolerance 1e-4, up to 200 iterations. |

---

## 5. Dataset 1: ATP Tennis Matches

### 5.1 Data Source

**Source:** [Jeff Sackmann's ATP match dataset](https://github.com/JeffSackmann/tennis_atp) — comprehensive match-level data for professional men's tennis.

**Files used:** `atp_matches_2015.csv` through `atp_matches_2024.csv` (10 years of main-tour singles matches, excluding doubles, futures, and qualifying/challenger events).

**Raw data shape:** 27,672 rows × 49 columns.

**Each row represents one match**, with columns for:
- Tournament metadata (name, surface, draw size, level, date, round)
- Winner attributes (ID, name, hand, height, age, nationality, rank, rank points)
- Loser attributes (same set of fields)
- Match score (as a string)
- In-match statistics: aces, double faults, service points, 1st serve in, 1st serve won, 2nd serve won, service games, break points saved/faced — for **both** winner and loser
- Match duration in minutes

### 5.2 Target Variable Design

**Target:** Y = 1 if Player A won the match, Y = 0 if Player A lost.

Where "Player A" is a **randomly assigned** role (see §5.3 below). By construction, the target is **balanced**: Y=0 appears 12,274 times (49.9%), Y=1 appears 12,342 times (50.1%).

### 5.3 The Symmetrization Problem and Solution

#### The Problem

The raw data is structured as `winner_*` / `loser_*` columns. If we naïvely compute features as `winner_stat − loser_stat`, the resulting differentials **perfectly encode the outcome**: the "winner" always has better break-point conversion, the rank differential always has a predictable sign distribution, etc. A logistic regression trained on such features achieves **99.98% accuracy** — a clear sign of target leakage.

#### The Solution: Random Role Assignment

For each match, we flip a **fair coin** (Bernoulli(0.5)):

| Coin Result | Player A becomes | Player B becomes | Target Y |
|---|---|---|---|
| Heads (flip=1) | Winner | Loser | 1 (A won) |
| Tails (flip=0) | Loser | Winner | 0 (A lost) |

All player-level columns (rank, age, height, service stats) are swapped accordingly. Match-level columns (minutes, best_of) remain unchanged.

After symmetrization, features like `rank_diff = A_rank − B_rank` are roughly symmetric around zero: sometimes positive (A is worse-ranked), sometimes negative (A is better-ranked). The logistic regression must learn from the **magnitude and direction** of these differentials, yielding a realistic 94.3% accuracy.

#### Seed Separation

A critical implementation detail: the symmetrization coin flip uses `random_state=0`, while the missing-data generators use `random_state=121553223`. **Using the same seed for both would create correlated random sequences**, causing pathological behavior — e.g., MCAR producing 0% missingness in class 0 and 60% in class 1, violating the MCAR definition entirely. The seed separation was validated empirically.

### 5.4 Feature Engineering

Starting from the symmetrized data, we engineer **23 numerical features** organized into six categories:

#### Category 1: Rank & Rating Differentials (3 features)

| Feature | Formula | Interpretation |
|---|---|---|
| `rank_diff` | A_rank − B_rank | Raw ranking differential. Positive = A is worse-ranked. |
| `rank_points_diff` | A_rank_points − B_rank_points | Rating points differential. |
| `log_rank_ratio` | log(1 + A_rank) − log(1 + B_rank) | Log-scale rank ratio. Compresses extreme differences (e.g., rank 500 vs rank 1). |

#### Category 2: Player Attribute Differentials (2 features)

| Feature | Formula | Interpretation |
|---|---|---|
| `age_diff` | A_age − B_age | Age difference in years. |
| `height_diff` | A_height − B_height | Height difference in cm. Missing heights are imputed with the column median before computing the differential. |

#### Category 3: Service Statistic Differentials — Rates (7 features)

Raw service counts are first converted to rates, then differenced (A − B):

| Feature | Formula | Interpretation |
|---|---|---|
| `ace_rate_diff` | (A_aces/A_svpt) − (B_aces/B_svpt) | Aces per service point differential |
| `df_rate_diff` | (A_df/A_svpt) − (B_df/B_svpt) | Double-fault rate differential |
| `first_in_pct_diff` | (A_1stIn/A_svpt) − (B_1stIn/B_svpt) | 1st serve percentage differential |
| `first_won_pct_diff` | (A_1stWon/A_1stIn) − (B_1stWon/B_1stIn) | 1st serve points won % differential |
| `second_won_pct_diff` | (A_2ndWon/(A_svpt−A_1stIn)) − (…) | 2nd serve points won % differential |
| `bp_save_pct_diff` | (A_bpSaved/A_bpFaced) − (…) | Break point save % differential |
| `hold_pressure_diff` | (A_bpFaced/A_SvGms) − (…) | Break points faced per service game differential |

**Division safety:** All divisions use ε = 1e-6 added to the denominator, and second-serve denominators are clipped to ε from below to handle edge cases (e.g., 100% first-serve percentage).

#### Category 4: Normalized Count Differentials (4 features)

| Feature | Formula | Interpretation |
|---|---|---|
| `ace_count_diff` | (A_aces − B_aces) / total_svpt | Ace count difference normalized by total service points |
| `df_count_diff` | (A_df − B_df) / total_svpt | Double-fault count difference normalized |
| `svpt_diff` | (A_svpt − B_svpt) / total_svpt | Service point share differential (proxy for who served more) |
| `bp_faced_diff` | (A_bpFaced − B_bpFaced) / total_games | Break points faced difference per total games |

#### Category 5: Match-Level Features (4 features)

| Feature | Formula | Interpretation |
|---|---|---|
| `minutes` | match duration | Longer matches may indicate closer contests |
| `best_of` | 3 or 5 | Best-of-3 (most tournaments) vs best-of-5 (Grand Slams) |
| `total_svpt` | A_svpt + B_svpt | Total service points in the match |
| `total_games` | A_SvGms + B_SvGms | Total service games played |

#### Category 6: Absolute Averages (3 features)

| Feature | Formula | Interpretation |
|---|---|---|
| `avg_rank` | (A_rank + B_rank) / 2 | Average ranking of the two players (match "quality level") |
| `avg_age` | (A_age + B_age) / 2 | Average age |
| `avg_height` | (A_height + B_height) / 2 | Average height |

These are symmetric (invariant to the A/B assignment) and capture the **absolute context** of the match.

### 5.5 Preprocessing Pipeline Results

Starting with 24,616 rows and 23 features, the pipeline produced:

#### Step 1: Imputation
- **0 missing values** in the feature matrix (height was already imputed during feature engineering).

#### Step 2: Low-Variance Removal
- **0 features dropped** (all had sufficient variance).

#### Step 3: Correlation-Based Collinearity Removal (|r| > 0.90)
- **4 features dropped:**
  - `ace_rate_diff` — highly correlated with `ace_count_diff`
  - `df_count_diff` — highly correlated with `df_rate_diff`
  - `hold_pressure_diff` — highly correlated with `bp_faced_diff`
  - `total_svpt` — highly correlated with `minutes` and `total_games`

#### Step 4: VIF-Based Collinearity Removal (VIF > 10)
- **3 features dropped** (iteratively, highest VIF first):
  - `avg_height` (VIF = 85.0)
  - `total_games` (VIF = 63.5)
  - `best_of` (VIF = 22.6)

#### Final Feature Set: 16 Features

| # | Feature | Category |
|---|---|---|
| 1 | `rank_diff` | Rank differential |
| 2 | `rank_points_diff` | Rank differential |
| 3 | `log_rank_ratio` | Rank differential |
| 4 | `age_diff` | Player attribute |
| 5 | `height_diff` | Player attribute |
| 6 | `df_rate_diff` | Service rate |
| 7 | `first_in_pct_diff` | Service rate |
| 8 | `first_won_pct_diff` | Service rate |
| 9 | `second_won_pct_diff` | Service rate |
| 10 | `bp_save_pct_diff` | Service rate |
| 11 | `ace_count_diff` | Normalized count |
| 12 | `svpt_diff` | Normalized count |
| 13 | `bp_faced_diff` | Normalized count |
| 14 | `minutes` | Match-level |
| 15 | `avg_rank` | Absolute average |
| 16 | `avg_age` | Absolute average |

**Final dataset dimensions:** 24,616 rows × 16 features + 1 target = 17 columns total.  
**Missing values:** 0.  
**Target balance:** Y=0: 12,274 (49.9%) | Y=1: 12,342 (50.1%).

### 5.6 Missing-Label Generation Results

All schemes configured with `missing_rate=0.3`, `random_state=121553223`.

| Scheme | Overall Missing Rate | Missing Rate (Y=0) | Missing Rate (Y=1) | Ratio (Y=1 / Y=0) | Labels Hidden |
|---|---|---|---|---|---|
| **MCAR** | 30.1% | 29.6% | 30.6% | 1.03 | 7,412 |
| **MAR1** | 30.2% | 17.4% | 43.0% | 2.47 | 7,443 |
| **MAR2** | 30.3% | 34.6% | 26.1% | 0.75 | 7,464 |
| **MNAR** | 29.9% | 16.7% | 43.1% | 2.58 | 7,360 |

#### Interpretation

- **MCAR:** The per-class missing rates are nearly identical (29.6% vs 30.6%). The small difference is purely due to random sampling — exactly what MCAR guarantees. The ratio is ≈1.0.

- **MAR1:** Substantial asymmetry. The automatically selected driving feature has a strong correlation with Y, so observations with Y=1 are 2.47× more likely to have their label hidden. Importantly, this dependence is mediated entirely through X_j — given X_j, missingness is independent of Y.

- **MAR2:** Moderate asymmetry in the opposite direction (Y=0 observations are slightly more likely to be missing). The random weight vector happened to produce a linear combination that correlates slightly negatively with Y. The ratio of 0.75 indicates a milder effect than MAR1.

- **MNAR:** The most extreme asymmetry, with a ratio of 2.58. The γ=2.0 coefficient on Y directly pushes positive-class observations toward missingness. This is the hardest setting for any method that assumes MAR.

### 5.7 Baseline Model Performance

Logistic regression with StandardScaler, 5-fold cross-validation on the **fully labelled** dataset:

| Metric | Mean | Std |
|---|---|---|
| **Accuracy** | 0.9434 | ±0.0030 |
| **ROC-AUC** | 0.9881 | ±0.0013 |

This confirms:
1. The features are **highly informative** — logistic regression can predict match outcomes with 94% accuracy from the player/match statistics.
2. There is **no target leakage** — performance is strong but not perfect (compare with the 99.98% accuracy before symmetrization).
3. The dataset is **well-suited for logistic regression** — the high AUC indicates good class separability, giving us room to observe degradation under different missingness schemes.

---

## 6. Output Files

### `processed/atp_upset.csv`

The fully-labelled ground truth dataset.

| Property | Value |
|---|---|
| Rows | 24,616 |
| Columns | 17 (16 features + `y`) |
| Target column | `y` (0 or 1) |
| Missing values | None |

### `processed/atp_upset_{SCHEME}.csv`

One file per missingness scheme (MCAR, MAR1, MAR2, MNAR).

| Property | Value |
|---|---|
| Rows | 24,616 |
| Columns | 18 (16 features + `y_obs` + `y_true`) |
| `y_obs` | Observed label: 0, 1, or **−1** (missing) |
| `y_true` | Ground-truth label: 0 or 1 (for evaluation only — not available during training) |
| Missing values | None (−1 is an explicit sentinel, not NaN) |

---

## 7. Reproducing Results

### Prerequisites

```
Python ≥ 3.9
numpy
pandas
scikit-learn
statsmodels
matplotlib
seaborn
```

### Running the ATP Dataset Pipeline

1. Open `notebooks/01_atp_dataset.ipynb` in Jupyter.
2. Run all cells in order.
3. Outputs are written to `processed/`.

The notebook imports from `scripts/` via a `sys.path` insert pointing to `../scripts` relative to the notebook's working directory. No package installation is needed.

### Adding a New Dataset

1. Place raw data in `datasets/`.
2. Create a new notebook `notebooks/02_<dataset_name>.ipynb`.
3. Import the shared utilities:
   ```python
   from preprocessing import full_preprocessing_pipeline
   from missing_data import generate_missing, missingness_summary
   ```
4. Implement dataset-specific loading and feature engineering.
5. Call `full_preprocessing_pipeline()` for cleaning.
6. Call `generate_missing()` for each scheme.
7. Save to `processed/`.

---

## 8. Design Decisions and Lessons Learned

### Why differentials instead of separate A/B features?

Using `A_stat − B_stat` differentials rather than separate `A_stat` and `B_stat` columns:
- Halves the feature count (fewer collinearity issues).
- Directly encodes the **competitive relationship** that logistic regression should learn.
- Naturally produces features centered around zero, which is well-suited for logistic regression.

### Why bisection for intercept calibration?

The intercept calibration problem (`find b such that mean(σ(z + b)) = target`) is a scalar root-finding problem. Bisection was chosen over Newton's method because:
- It is guaranteed to converge (the function is monotonically increasing in b).
- It needs no derivatives.
- 200 iterations in [−30, +30] achieve tolerance 1e-4, which is more than sufficient.

### Why unit-norm weight vectors in MAR2 and MNAR?

Without normalization, the linear predictor `X·w` would have variance proportional to `p` (the number of features) when w is drawn from N(0,1). Normalizing w to unit norm ensures the linear predictor's scale is O(1) regardless of p, making the intercept calibration stable and the sigmoid produce meaningful probabilities (not saturated at 0 or 1).

### Why separate seeds for symmetrization and missingness?

NumPy's `default_rng` produces a deterministic sequence for a given seed. If both the symmetrization coin flip and the MCAR Bernoulli mask use seed 42, they produce **identical** sequences of random numbers. Since the coin flip determines Y, this creates a perfect correlation between Y and S — turning MCAR into a pathological MNAR where one class has 0% missingness and the other has ~60%. Using distinct seeds (0 and 121553223) eliminates this artifact entirely.

### Why is accuracy 94% and not lower?

The 94% accuracy might seem surprisingly high for "predicting who wins a tennis match." However, the features include **in-match statistics** (aces, break points, etc.), not just pre-match attributes. A player who served more aces, saved more break points, and won more first-serve points is very likely the winner. The features are informative precisely because they describe what happened during the match. For the project's purpose (studying missing-label effects on logistic regression), this is ideal: a strong baseline makes degradation under missingness more measurable.
