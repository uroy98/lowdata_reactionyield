# contains your provided code block (SMILES fixes + featurization)

import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray


def featurize_reaction(row):
    """Return a 1024-bit ECFP4 fingerprint combining all reaction components."""
    raw = {
        'acid': row['Acid (SMILES)'],
        'amine': row['Amine (SMILES)'],
        'activator': row['Activator (SMILES)'],
        'base': row['Base (SMILES)'],
        'solvent': row['Solvent (SMILES)']
    }
    mols = []
    for name, smi in raw.items():
        if not isinstance(smi, str):
            continue
        s = smi.strip().rstrip('.')
        if s.lower() in ('', 'nil'):
            continue
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        try:
            Chem.SanitizeMol(m)
            mols.append(m)
        except Exception:
            continue

    if not mols:
        return np.zeros((1024,), dtype=np.int8)

    combo = mols[0]
    for m in mols[1:]:
        combo = Chem.CombineMols(combo, m)

    try:
        Chem.SanitizeMol(combo)
    except Exception:
        return np.zeros((1024,), dtype=np.int8)

    fp = AllChem.GetMorganFingerprintAsBitVect(combo, radius=2, nBits=1024)
    arr = np.zeros((1024,), dtype=np.int8)
    ConvertToNumpyArray(fp, arr)
    return arr


def preprocess_data(train_path, test_path, out_dir="data/chem/processed"):
    """Load Excel datasets, fix SMILES, featurize, and save NumPy arrays."""
    pd.set_option('display.max_rows', None)

    # Load Excel files
    df = pd.read_excel(train_path)
    df_test = pd.read_excel(test_path)

    drop_cols = ['Acid (name)', 'Amine (name)', 'Activator (name)',
                 'Base (name)', 'Solvent', 'References']
    df.drop(columns=drop_cols, inplace=True)
    df_test.drop(columns=drop_cols, inplace=True)

    # Fix special SMILES
    df.loc[96, 'Solvent (SMILES)'] = 'C1CCOC1.OO'
    df.loc[148, 'Acid (SMILES)'] = 'O=C(O)C(F)(F)F'
    df.loc[29, 'Solvent (SMILES)'] = 'ClCCl.CN(C)C=O'
    df.loc[73, ['Acid (SMILES)', 'Amine (SMILES)', 'Activator (SMILES)',
                'Base (SMILES)', 'Solvent (SMILES)']] = [np.nan]*5
    df_test.loc[4, 'Solvent (SMILES)'] = 'C1CCOC1.OO'

    # Featurize
    print("Featurizing training and testing data ...")
    fp_train = np.vstack(df.apply(featurize_reaction, axis=1).values)
    fp_test = np.vstack(df_test.apply(featurize_reaction, axis=1).values)

    numeric_cols = ['Acid (equiv.)', 'Amine (equiv.)',
                    'Activator (equiv.)', 'Base (equiv.)',
                    'Global Conc (M)', 'Temp (C)', 'Time (h)']

    X_train_num = df[numeric_cols].to_numpy()
    X_test_num = df_test[numeric_cols].to_numpy()

    X_train = np.hstack([X_train_num, fp_train])
    X_test = np.hstack([X_test_num, fp_test])
    y_train = df['Reaction_Yield'].to_numpy()
    y_test = df_test['Reaction_Yield'].to_numpy()

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(out_dir, "X_test.npy"), X_test)
    np.save(os.path.join(out_dir, "y_test.npy"), y_test)

    print(f"✅ Saved processed arrays to {out_dir}/")
