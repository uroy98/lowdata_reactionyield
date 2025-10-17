# -*- coding: utf-8 -*-
"""
Chemically-constrained XGBoost to beat RL+GB (R2≈0.70) on 199 examples.

- Uses compact physchem descriptors from SMILES (acid/amine/activator/base/solvent)
- Builds one combined-reaction Morgan fingerprint, PCA-compressed (64D)
- Concatenates process vars: [Acid eq, Amine eq, Activator eq, Base eq, Global Conc, Temp, Time]
- Enforces monotonicity: non-decreasing w.r.t. (Acid eq, Amine eq, Activator eq, Base eq, Global Conc, Time)
- Leaves Temp unconstrained (can be non-monotone in practice)
"""

import os, json, warnings
import numpy as np
import pandas as pd

# --- Deterministic + quiet ---
os.environ["PYTHONHASHSEED"] = "42"
os.environ["OMP_NUM_THREADS"] = "1"
np.random.seed(42)
warnings.filterwarnings("ignore")

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
from rdkit.DataStructs import ConvertToNumpyArray

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor

# ---------- Paths ----------
TRAIN_XLSX = "data/chem/raw/SMILES_TrainingDataset_July2_172_Plus_16LY_Fixed.xlsx"
TEST_XLSX  = "data/chem/raw/SMILES_TestingDataset_July3_Final.xlsx"
OUT_DIR    = "results/monotone_xgb"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Columns ----------
SMI_COLS = [
    "Acid (SMILES)", "Amine (SMILES)", "Activator (SMILES)",
    "Base (SMILES)", "Solvent (SMILES)"
]
NUMERIC_COLS = [
    "Acid (equiv.)", "Amine (equiv.)", "Activator (equiv.)", "Base (equiv.)",
    "Global Conc (M)", "Temp (C)", "Time (h)"
]
TARGET_COL = "Reaction_Yield"

# ---------- SMILES cleaning patches (same spirit as what you did) ----------
def clean_known_issues(df: pd.DataFrame, is_test: bool=False):
    # You can extend these if you discover more issues.
    if not is_test:
        # Training fixes
        if 96 in df.index:
            df.loc[96, 'Solvent (SMILES)'] = 'C1CCOC1.OO'           # THF·H2O
        if 148 in df.index:
            df.loc[148, 'Acid (SMILES)'] = 'O=C(O)C(F)(F)F'         # TFA
        if 29 in df.index:
            df.loc[29, 'Solvent (SMILES)'] = 'ClCCl.CN(C)C=O'       # DCM·DMF
        if 73 in df.index:
            df.loc[73, SMI_COLS] = [np.nan]*5
    else:
        # Test fixes
        if 4 in df.index:
            df.loc[4, 'Solvent (SMILES)'] = 'C1CCOC1.OO'
    return df

# ---------- RDKit helpers ----------
def smiles_to_mol(s):
    if not isinstance(s, str): return None
    s = s.strip().rstrip('.')
    if len(s)==0 or s.lower()=='nil': return None
    m = Chem.MolFromSmiles(s)
    if m is None:
        return None
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return None
    return m

def morgan_fp_arr(mol, nbits=1024, radius=2):
    if mol is None:
        return np.zeros((nbits,), dtype=np.int8)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits)
    arr = np.zeros((nbits,), dtype=np.int8)
    ConvertToNumpyArray(fp, arr)
    return arr

def physchem_desc(mol):
    """Compact physchem set less likely to overfit on N=199."""
    if mol is None:
        return dict(
            MolWt=0.0, TPSA=0.0, LogP=0.0, HBD=0, HBA=0,
            Ring=0, NRot=0, Arom=0, FC=0, Fsp3=0.0
        )
    return dict(
        MolWt=Descriptors.MolWt(mol),
        TPSA=Descriptors.TPSA(mol),
        LogP=Descriptors.MolLogP(mol),
        HBD=Descriptors.NumHDonors(mol),
        HBA=Descriptors.NumHAcceptors(mol),
        Ring=Descriptors.RingCount(mol),
        NRot=Descriptors.NumRotatableBonds(mol),
        Arom=rdMolDescriptors.CalcNumAromaticRings(mol),
        FC=Chem.GetFormalCharge(mol),
        Fsp3=rdMolDescriptors.CalcFractionCSP3(mol),
    )

def featurize_row(row, nbits=1024):
    # 1) Parse molecules
    mols = {name: smiles_to_mol(row[name]) for name in SMI_COLS}
    # 2) Per-role physchem with prefixes
    feats = {}
    for role, m in mols.items():
        desc = physchem_desc(m)
        for k, v in desc.items():
            feats[f"{role.split()[0].lower()}_{k}"] = v
    # 3) Combined reaction fingerprint (join available mols)
    valid_mols = [m for m in mols.values() if m is not None]
    if len(valid_mols)==0:
        combo = None
    else:
        combo = valid_mols[0]
        for m in valid_mols[1:]:
            combo = Chem.CombineMols(combo, m)
        try:
            Chem.SanitizeMol(combo)
        except Exception:
            combo = None
    fp = morgan_fp_arr(combo, nbits=nbits)
    return feats, fp

# ---------- Load data ----------
train_df = pd.read_excel(TRAIN_XLSX)
test_df  = pd.read_excel(TEST_XLSX)

train_df = clean_known_issues(train_df, is_test=False)
test_df  = clean_known_issues(test_df,  is_test=True)

# Basic drops (names & refs not used for training)
drop_cols = ['Acid (name)','Amine (name)','Activator (name)','Base (name)','Solvent','References']
for c in drop_cols:
    if c in train_df.columns: train_df.drop(columns=c, inplace=True)
    if c in test_df.columns:  test_df.drop(columns=c,  inplace=True)

# ---------- Build features ----------
NBITS = 1024
train_desc_rows, train_fps = [], []
for idx, row in train_df.iterrows():
    desc, fp = featurize_row(row, nbits=NBITS)
    train_desc_rows.append(desc)
    train_fps.append(fp)
test_desc_rows, test_fps = [], []
for idx, row in test_df.iterrows():
    desc, fp = featurize_row(row, nbits=NBITS)
    test_desc_rows.append(desc)
    test_fps.append(fp)

desc_train = pd.DataFrame(train_desc_rows).fillna(0.0)
desc_test  = pd.DataFrame(test_desc_rows).fillna(0.0)

fp_train = np.vstack(train_fps).astype(np.float32)
fp_test  = np.vstack(test_fps).astype(np.float32)

# Process variables
Xnum_train = train_df[NUMERIC_COLS].astype(np.float32).to_numpy()
Xnum_test  = test_df[NUMERIC_COLS].astype(np.float32).to_numpy()

y_train = train_df[TARGET_COL].astype(np.float32).to_numpy()
y_test  = test_df[TARGET_COL].astype(np.float32).to_numpy()

# ---------- Scale numeric & descriptors; leave binary fp un-centered ----------
sc_num = StandardScaler(with_mean=True)
sc_desc = StandardScaler(with_mean=True)
Xnum_train_s = sc_num.fit_transform(Xnum_train)
Xnum_test_s  = sc_num.transform(Xnum_test)
Xdesc_train_s = sc_desc.fit_transform(desc_train.to_numpy().astype(np.float32))
Xdesc_test_s  = sc_desc.transform(desc_test.to_numpy().astype(np.float32))

# ---------- PCA on fingerprint (low-data friendly) ----------
pca = PCA(n_components=64, random_state=42)
Xfp_train_pca = pca.fit_transform(fp_train)
Xfp_test_pca  = pca.transform(fp_test)

# ---------- Final feature assembly ----------
# Order: [NUMERIC(7)] + [DESC(...)] + [FP_PCA(64)]
X_train = np.hstack([Xnum_train_s, Xdesc_train_s, Xfp_train_pca]).astype(np.float32)
X_test  = np.hstack([Xnum_test_s,  Xdesc_test_s,  Xfp_test_pca ]).astype(np.float32)

# ---------- Monotone constraints for XGBoost ----------
#   We constrain the 7 numeric features only (first 7 positions in X_*):
#     +1: Acid eq, Amine eq, Activator eq, Base eq, Global Conc, Time
#      0: Temp (leave unconstrained)
mono_vec = [+1, +1, +1, +1, +1, 0, +1]  # length 7
# For the rest (descriptors + PCA), use 0 (no constraint)
mono_vec += [0] * (X_train.shape[1] - len(mono_vec))
assert len(mono_vec) == X_train.shape[1]

mono_str = "(" + ",".join(str(v) for v in mono_vec) + ")"

# ---------- Train XGBoost (small model; strong regularization) ----------
model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=800,            # small but enough with early_stopping
    learning_rate=0.03,
    max_depth=4,                 # shallow trees to avoid overfit
    subsample=0.8,
    colsample_bytree=0.6,
    reg_alpha=1.0,               # L1
    reg_lambda=2.0,              # L2
    random_state=42,
    monotone_constraints=mono_str,
    n_jobs=1
)

# Use a small validation slice from training for early stopping
from sklearn.model_selection import train_test_split
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    eval_metric="rmse",
    verbose=False,
    early_stopping_rounds=60
)

# ---------- Evaluate ----------
pred = model.predict(X_test)
rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
mae  = float(mean_absolute_error(y_test, pred))
r2   = float(r2_score(y_test, pred))

print(f"[Monotone-XGB] RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.3f}")

# ---------- Save ----------
pd.DataFrame({"y_true": y_test, "y_pred": pred}).to_csv(os.path.join(OUT_DIR, "preds.csv"), index=False)
with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
    json.dump({"RMSE": rmse, "MAE": mae, "R2": r2}, f, indent=2)

# Save feature importances for your appendix
imp = pd.Series(model.feature_importances_)
imp.to_csv(os.path.join(OUT_DIR, "feature_importances.csv"), header=["importance"], index=False)
print("✅ Results saved in", OUT_DIR)
