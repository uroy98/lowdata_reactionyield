# run_hybrid_model.py
import os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.utils.set_deterministic import set_deterministic
set_deterministic(42, use_cpu=True)

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from src.models.hybrid import build_chem_iml, gaussian_nll_multihead, split_heads

# --------------------------
# Paths & constants
# --------------------------
DATA_DIR   = "data/chem/processed"
OUT_DIR    = "results/hybrid"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_NUMERIC = 7            # [Acid eq, Amine eq, Activator eq, Base eq, Global Conc, Temp, Time]
HEADS       = 5
LR          = 1e-3
BATCH       = 32
EPOCHS      = 400
VAL_SIZE    = 0.2

# --------------------------
# Load data
# --------------------------
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy")).astype(np.float32)
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy")).astype(np.float32)
X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy")).astype(np.float32)
y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy")).astype(np.float32)

# Split into numeric + fp (your preprocessing put numeric first, then 1024 fp)
fp_dim = X_train.shape[1] - NUM_NUMERIC
Xtr_num, Xtr_fp = X_train[:, :NUM_NUMERIC], X_train[:, NUM_NUMERIC:]
Xte_num, Xte_fp = X_test[:,  :NUM_NUMERIC], X_test[:,  NUM_NUMERIC:]

# Keep Time(h) for trend calibration
time_train = Xtr_num[:, 6]
time_test  = Xte_num[:, 6]

# --------------------------
# Scale features
#   - numeric: with_mean=True
#   - fp: with_mean=False (binary-like)
# --------------------------
sc_num = StandardScaler(with_mean=True)
sc_fp  = StandardScaler(with_mean=False)

Xtr_num_s = sc_num.fit_transform(Xtr_num)
Xte_num_s = sc_num.transform(Xte_num)

Xtr_fp_s  = sc_fp.fit_transform(Xtr_fp)
Xte_fp_s  = sc_fp.transform(Xte_fp)

# Train/val split on TRAIN only
Xn_tr, Xn_va, Xf_tr, Xf_va, y_tr, y_va, t_tr, t_va = train_test_split(
    Xtr_num_s, Xtr_fp_s, y_train, time_train,
    test_size=VAL_SIZE, random_state=42
)

# --------------------------
# Build & train model
# --------------------------
model = build_chem_iml(
    num_numeric=NUM_NUMERIC,
    fp_dim=fp_dim,
    # hidden_num=64,
    hidden_num=128,     # more capacity
    # hidden_fp=128,
    hidden_fp=256,      # more capacity
    # fusion=128,
    fusion=256,
    # heads=HEADS,
    heads=5,
    # dropout_num=0.1,
    dropout_num=0.05,   # less dropout
    # dropout_fp=0.1,
    dropout_fp=0.05,
    # l2=1e-5
    l2=1e-6             # weaker regularization
)

# model.compile(optimizer=Adam(learning_rate=LR),
#               loss=gaussian_nll_multihead(HEADS))

model.compile(optimizer=Adam(learning_rate=5e-3),
              loss=gaussian_nll_multihead(HEADS))

# callbacks = [
#     EarlyStopping(monitor="val_loss", patience=40, restore_best_weights=True, verbose=1),
#     ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=15, min_lr=1e-5, verbose=1)
# ]

callbacks = [
    EarlyStopping(monitor="val_loss", patience=80, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.7, patience=25, min_lr=1e-5, verbose=1)
]

history = model.fit(
    x={"x_num": Xn_tr, "x_fp": Xf_tr},
    y=y_tr,
    validation_data=({"x_num": Xn_va, "x_fp": Xf_va}, y_va),
    epochs=EPOCHS,
    batch_size=BATCH,
    verbose=1,
    callbacks=callbacks
)

# --------------------------
# Predict (mean & sigma)
# --------------------------
pred_val = model.predict({"x_num": Xn_va, "x_fp": Xf_va}, verbose=0)
pred_te  = model.predict({"x_num": Xte_num_s, "x_fp": Xte_fp_s}, verbose=0)

# Split heads and average
mus_va, sigmas_va = split_heads(tf.constant(pred_val), HEADS)
mus_te, sigmas_te = split_heads(tf.constant(pred_te),  HEADS)

mu_va = np.array(tf.reduce_mean(mus_va, axis=-1))
sd_va = np.array(tf.reduce_mean(sigmas_va, axis=-1))
mu_te = np.array(tf.reduce_mean(mus_te, axis=-1))
sd_te = np.array(tf.reduce_mean(sigmas_te, axis=-1))

# --------------------------
# Metrics (raw)
# --------------------------
rmse  = float(np.sqrt(mean_squared_error(y_test, mu_te)))
mae   = float(mean_absolute_error(y_test, mu_te))
r2    = float(r2_score(y_test, mu_te))

# Gaussian NLL (using mean sigma on test)
sigma_te = sd_te
nll = float(np.mean(0.5*np.log(2*np.pi*(sigma_te**2) + 1e-9) +
                    0.5*((y_test - mu_te)**2) / (sigma_te**2 + 1e-9)))

# Save raw predictions
pd.DataFrame({
    "y_true": y_test,
    "mu": mu_te,
    "sigma": sigma_te,
    "time_h": time_test
}).to_csv(os.path.join(OUT_DIR, "test_preds_raw.csv"), index=False)

with open(os.path.join(OUT_DIR, "metrics_raw.json"), "w") as f:
    json.dump({"RMSE": rmse, "MAE": mae, "R2": r2, "NLL": nll}, f, indent=2)

print(f"[HYBRID raw] RMSE={rmse:.3f} MAE={mae:.3f} R2={r2:.3f} NLL={nll:.3f}")

# --------------------------
# Post-hoc isotonic calibration vs Time(h)
#  Ensures non-decreasing yield w.r.t. time
#  Fit on (train+val) predictions to avoid test leakage
# --------------------------
# Fit isotonic on validation (or train+val if you prefer)
iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
iso.fit(t_va, mu_va)  # map time -> calibrated mean

mu_te_iso = iso.predict(time_test)

rmse_iso = float(np.sqrt(mean_squared_error(y_test, mu_te_iso)))
mae_iso  = float(mean_absolute_error(y_test, mu_te_iso))
r2_iso   = float(r2_score(y_test, mu_te_iso))

pd.DataFrame({
    "y_true": y_test,
    "mu_iso": mu_te_iso,
    "mu_raw": mu_te,
    "sigma_raw": sigma_te,
    "time_h": time_test
}).to_csv(os.path.join(OUT_DIR, "test_preds_iso.csv"), index=False)

with open(os.path.join(OUT_DIR, "metrics_iso.json"), "w") as f:
    json.dump({"RMSE": rmse_iso, "MAE": mae_iso, "R2": r2_iso}, f, indent=2)

print(f"[HYBRID iso] RMSE={rmse_iso:.3f} MAE={mae_iso:.3f} R2={r2_iso:.3f}")
