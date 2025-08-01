import pandas as pd
import numpy as np

q2l = pd.read_csv("q2l_results.csv")
q2p = pd.read_csv("q2p_results.csv")

# combine the two DataFrames
combined_df = pd.merge(q2l, q2p, on=['linear_fix_coeffecient', 'penalty_multi_head_fix_coeffecient'])
# print(combined_df.head())

for i, row in combined_df.iterrows():
    if np.isnan(row['Q2L']) or np.isnan(row['Q2P']):
        continue
    if row['Q2L'] < 1.0 or row['Q2L'] > 1.6:
        continue
    if row['Q2P'] < 1.5 or row['Q2P'] > 2.2:
        continue
    print(f"linear_fix_coeffecient: {row['linear_fix_coeffecient']}, penalty_multi_head_fix_coeffecient: {row['penalty_multi_head_fix_coeffecient']}, Q2L: {row['Q2L']}, Q2P: {row['Q2P']}")
