import numpy as np
import pandas as pd
from itertools import combinations

MONO_KEYS = ["Time (h)", "Global Conc (M)", "Acid (equiv.)", "Amine (equiv.)", "Activator (equiv.)", "Base (equiv.)"]

def nearly_matched(df, i, j, tol=1e-8):
    # same categorical chemistry context: exact SMILES match for acid/amine/activator/base/solvent
    cols = ["Acid (SMILES)", "Amine (SMILES)", "Activator (SMILES)", "Base (SMILES)", "Solvent (SMILES)"]
    return all(df.loc[i,c] == df.loc[j,c] for c in cols)

def generate_pairs(X_num, X_lat, y, original_df, mono_cols_idx, max_pairs_per_rxn=20):
    n = len(y)
    pairs = []
    for i in range(n):
        cands = [j for j in range(n) if j != i and nearly_matched(original_df, i, j)]
        # limit number of pairs per anchor to avoid explosion
        cands = cands[:max_pairs_per_rxn]
        for j in cands:
            # build a pair only if monotone features differ (gives supervision)
            xi_num, xj_num = X_num[i], X_num[j]
            if not np.any(np.abs(xi_num[mono_cols_idx] - xj_num[mono_cols_idx]) > 1e-12):
                continue
            # label: 1 if y_i >= y_j, else 0
            y_ij = 1 if y[i] >= y[j] else 0
            pairs.append((i, j, y_ij))
    # arrays of indices and labels
    if not pairs:
        return np.array([]), np.array([])
    ij = np.array([(i,j) for (i,j,_) in pairs], dtype=int)
    lab = np.array([yij for (*_,yij) in pairs], dtype=np.float32)
    return ij, lab
