import os, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from src.models.monorank import build_monorank
from src.experiments.build_pairs import generate_pairs

OUT_DIR = "results/monorank_pfn"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Load processed arrays (your pipeline) ----
X_train = np.load("data/chem/processed/X_train.npy").astype(np.float32)
X_test  = np.load("data/chem/processed/X_test.npy").astype(np.float32)
y_train = np.load("data/chem/processed/y_train.npy").astype(np.float32)
y_test  = np.load("data/chem/processed/y_test.npy").astype(np.float32)

# Optional: load yield-aware latent (from your fine-tuned encoder). If not available, fallback to RL latent.
try:
    X_train_lat = np.load("data/chem/processed/X_train_lat_finetuned.npy").astype(np.float32)
    X_test_lat  = np.load("data/chem/processed/X_test_lat_finetuned.npy").astype(np.float32)
except:
    X_train_lat = np.load("data/chem/processed/X_train_encoded.npy").astype(np.float32)
    X_test_lat  = np.load("data/chem/processed/X_test_encoded.npy").astype(np.float32)

NUM_NUMERIC = 7
Xtr_num, Xte_num = X_train[:, :NUM_NUMERIC], X_test[:, :NUM_NUMERIC]

# Standardize numeric + latent independently (critical!)
sc_num = StandardScaler()
sc_lat = StandardScaler()
Xtr_num_s = sc_num.fit_transform(Xtr_num).astype(np.float32)
Xte_num_s = sc_num.transform(Xte_num).astype(np.float32)
Xtr_lat_s = sc_lat.fit_transform(X_train_lat).astype(np.float32)
Xte_lat_s = sc_lat.transform(X_test_lat).astype(np.float32)

# ---- Build pairwise supervision from original table (needed to check “nearly matched”) ----
# If you don't have the joined original df here, you can pass a serialized minimal frame with the five SMILES and process cols.
orig_df = pd.read_excel("data/chem/raw/SMILES_TrainingDataset_July2_172_Plus_16LY_Fixed.xlsx")
mono_idx = [6, 4, 0, 1, 2, 3]  # Time(h)=6, GlobalConc=4, then 4 equiv columns [0..3] in your numeric order
ij, y_ij = generate_pairs(Xtr_num_s, Xtr_lat_s, y_train, orig_df, mono_idx, max_pairs_per_rxn=15)

# ---- Build model ----
model = build_monorank(
    input_num_dim=NUM_NUMERIC,
    input_lat_dim=Xtr_lat_s.shape[1],
    monotone_idx_num=mono_idx,
    lattice_sizes=6
)

# Losses
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def pairwise_loss(y_true, s_i, s_j):
    # logistic Bradley–Terry on score differences
    return bce(y_true, s_i - s_j)

def pinball(y, yhat, tau):
    e = y - yhat
    return tf.reduce_mean(tf.maximum(tau*e, (tau-1)*e))

# Optimizer
opt = Adam(3e-3)

# ---- Training loop (simple) ----
BATCH = 32
EPOCHS = 250

# Convert to per-sample tensors
xnum = [Xtr_num_s[:,k:k+1] for k in range(NUM_NUMERIC)]
xlat = Xtr_lat_s

@tf.function
def train_step(batch_pairs):
    with tf.GradientTape() as tape:
        # Ranking part
        i,j,lab = batch_pairs
        inputs_i = [tf.gather(x, i) for x in xnum] + [tf.gather(xlat, i)]
        inputs_j = [tf.gather(x, j) for x in xnum] + [tf.gather(xlat, j)]
        s_i, m_i, q10_i, q90_i = model(inputs_i, training=True)
        s_j, m_j, q10_j, q90_j = model(inputs_j, training=True)
        loss_rank = pairwise_loss(tf.expand_dims(lab,1), s_i, s_j)

        # Regression/quantile part (on i only, sampled)
        yi = tf.gather(y_train, i)
        loss_med = tf.reduce_mean(tf.square(yi - m_i))
        loss_qlo = pinball(tf.expand_dims(yi,1), q10_i, 0.1)
        loss_qhi = pinball(tf.expand_dims(yi,1), q90_i, 0.9)

        loss = loss_rank + 0.5*loss_med + 0.2*(loss_qlo + loss_qhi)

    grads = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# mini-batching over pairs
num_pairs = len(ij)
if num_pairs == 0:
    raise RuntimeError("No pairwise supervision could be constructed; relax matching or increase pairs per rxn.")

rng = np.random.default_rng(42)
for epoch in range(EPOCHS):
    # sample batches of pair indices
    idx = rng.permutation(num_pairs)
    total = 0.0
    for k in range(0, num_pairs, BATCH):
        sel = idx[k:k+BATCH]
        bi = tf.constant(ij[sel,0], dtype=tf.int32)
        bj = tf.constant(ij[sel,1], dtype=tf.int32)
        bl = tf.constant(y_ij[sel], dtype=tf.float32)
        loss = train_step((bi,bj,bl))
        total += float(loss.numpy()) * len(sel)

    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1}: loss={total/num_pairs:.4f}")

# ---- Evaluate on test ----
# Build test inputs once
test_inputs = [Xte_num_s[:,k:k+1] for k in range(NUM_NUMERIC)] + [Xte_lat_s]
s_te, m_te, q10_te, q90_te = model(test_inputs, training=False)
m_te = m_te.numpy().reshape(-1)
q10_te = q10_te.numpy().reshape(-1)
q90_te = q90_te.numpy().reshape(-1)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
rmse = float(np.sqrt(mean_squared_error(y_test, m_te)))
mae  = float(mean_absolute_error(y_test, m_te))
r2   = float(r2_score(y_test, m_te))
print(f"[MonoRank] RMSE={rmse:.3f} MAE={mae:.3f} R2={r2:.3f}")

pd.DataFrame({"y_true": y_test, "y_pred": m_te, "q10": q10_te, "q90": q90_te}).to_csv(
    os.path.join(OUT_DIR, "preds.csv"), index=False)
with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
    json.dump({"RMSE": rmse, "MAE": mae, "R2": r2}, f, indent=2)
