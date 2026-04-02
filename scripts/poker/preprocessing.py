import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

#Create clean dataset from raw
df = pd.read_csv("one_dollar_spin_and_go.csv")

selected_columns = [
    "level",
    "stack",
    "pot_pre",
    "pot_flop",
    "pot_turn",
    "pot_river",
    "blinds",
    "bet_pre",
    "bet_flop",
    "bet_turn",
    "bet_river",
    "result"
]

df = df[selected_columns]

# Convert result column to binary
mapping = {
    'gave up': 0,
    'lost': 0,
    'won': 1,
    'took chips': 1
}

df['result'] = df['result'].map(mapping)

df.to_csv("poker_clean.csv", index=False)

df = pd.read_csv("poker_clean.csv")


#Feature Engineering

# Avoid division by zero
epsilon = 1e-6

# stack_to_pot = stack / pot_pre
df['stack_to_pot'] = df['stack'] / (df['pot_pre'] + epsilon)

# total_bet = bet_pre + bet_flop + bet_turn + bet_river
df['total_bet'] = df['bet_pre'] + df['bet_flop'] + df['bet_turn'] + df['bet_river']

# pot_growth = pot_river / pot_pre
df['pot_growth'] = df['pot_river'] / (df['pot_pre'] + epsilon)

# stack_depth = stack / blinds
df['stack_depth'] = df['stack'] / (df['blinds'] + 1)

# total_pot = sum of all pots
df['total_pot'] = df['pot_pre'] + df['pot_flop'] + df['pot_turn'] + df['pot_river']

# Save final dataset
df.to_csv("poker_with_new_features.csv", index=False)

df = pd.read_csv("poker_with_new_features.csv")


log_features = [
    'stack', 'pot_pre', 'pot_flop', 'pot_turn', 'pot_river',
    'bet_pre', 'bet_flop', 'bet_turn', 'bet_river',
    'total_bet', 'stack_depth', 'total_pot'
]

# Log1p (log(1+x))
for col in log_features:
    df[col] = np.log1p(df[col])

scaler = MinMaxScaler()

# Min-Max
minmax_features = [
    'stack', 'pot_pre', 'pot_flop', 'pot_turn', 'pot_river',
    'bet_pre', 'bet_flop', 'bet_turn', 'bet_river',
    'total_bet', 'stack_depth', 'total_pot',
    'stack_to_pot', 'pot_growth', 'blinds'
]

df[minmax_features] = scaler.fit_transform(df[minmax_features])

df.to_csv("poker_data_preprocessed.csv", index=False)