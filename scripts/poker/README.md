# Poker Dataset Preprocessing & Missing Data Simulation

This repository provides scripts to preprocess a Spin & Go poker dataset and generate missing labels under different missing data mechanisms for binary classification experiments.

## 1. Preprocessing (`preprocessing.py`)

This script performs the following steps:

1. **Load raw dataset** ('one_dollar_spin_and_go.csv')
2. **Select relevant columns**: level, stack, pot_pre, pot_flop, pot_turn, pot_river, blinds, bet_pre, bet_flop, bet_turn, bet_river, result
3.  **Convert 'result' column to binary**:
- 'gave up' → 0
- 'lost' → 0
- 'won' → 1
- 'took chips' → 1

4. **Feature Engineering**:
- stack_to_pot = stack / pot_pre
- total_bet = bet_pre + bet_flop + bet_turn + bet_river
- pot_growth = pot_river / pot_pre
- stack_depth = stack / blinds
- total_pot = pot_pre + pot_flop + pot_turn + pot_river

5. **Log transformation** ('log1p') of selected features
6. **Min-Max scaling** of all numeric features
7. **Save final preprocessed dataset** ('poker_data_preprocessed.csv')

## 2. Missing Data Generation ('missing_data.py')

This module provides functions to simulate missing labels in a binary classification setting:

- **MCAR**: Missing Completely At Random
- **MAR1**: Missing At Random (single feature)
- **MAR2**: Missing At Random (all features)
- **MNAR**: Missing Not At Random (depends on feature and label)

All functions follow the same interface:

y_obs = generate_missing(X, y, scheme="mcar", c=0.3, random_state=42)

- X → features
- y → true labels
- scheme → missingness type (mcar, mar1, mar2, mnar)
- c → proportion of missing labels
-random_state → seed for reproducibility

Missing labels are represented as -1 in the output.

## 3. Generate Dataset (generate_dataset.py)

This script applies the missingness schemes to the preprocessed dataset and saves the resulting datasets:

Loads poker_data_preprocessed.csv
Defines missingness schemes with parameters
Generates missing labels for each scheme
Saves the resulting datasets:
poker_mcar.csv
poker_mar1.csv
poker_mar2.csv
poker_mnar.csv

Example usage:

python generate_dataset.py

Output:

MCAR: 900 labels hidden (30.0%)
MAR1: 900 labels hidden (30.0%)
MAR2: 900 labels hidden (30.0%)
MNAR: 900 labels hidden (30.0%)

## 4. Requirements
   
Python 3.8+
pandas
numpy
scikit-learn

Install dependencies:

# pip install pandas numpy scikit-learn
