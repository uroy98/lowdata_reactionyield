import os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.models.hybrid_v2 import build_chemlatent_hybrid
from src.utils.set_deterministic import set_deterministic
set_deterministic(42)

DATA_DIR = "data/chem/processed"
OUT_DIR  = "results/hybrid_v2"
os.makedirs(OUT_DIR, exist_ok=True)

# Load data
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy")).astype(np.float32)
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy")).astype(np.float32)
X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy")).astype(np.float32)
y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy")).astype(np.float32)
# X_train_lat = np.load(os.path.join(DATA_DIR, "X_train_encoded.npy")).astype(np.float32)
# X_test_lat  = np.load(os.path.join(DATA_DIR, "X_test_encoded.npy")).astype(np.float32)
X_train_lat = np.load(os.path.join(DATA_DIR, "X_train_lat_finetuned.npy")).astype(np.float32)
X_test_lat  = np.load(os.path.join(DATA_DIR, "X_test_lat_finetuned.npy")).astype(np.float32)


NUM_NUMERIC = 7
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[:, :NUM_NUMERIC])
X_test_num  = scaler.transform(X_test[:, :NUM_NUMERIC])

Xn_tr, Xn_va, Xl_tr, Xl_va, y_tr, y_va = train_test_split(
    X_train_num, X_train_lat, y_train, test_size=0.2, random_state=42)

model = build_chemlatent_hybrid(num_process=NUM_NUMERIC, latent_dim=X_train_lat.shape[1])
model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")

callbacks = [
    EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=15, min_lr=1e-5)
]

history = model.fit(
    [Xn_tr, Xl_tr], y_tr,
    validation_data=([Xn_va, Xl_va], y_va),
    epochs=400, batch_size=16, verbose=1, callbacks=callbacks
)

y_pred = model.predict([X_test_num, X_test_lat]).flatten()
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
mae  = float(mean_absolute_error(y_test, y_pred))
r2   = float(r2_score(y_test, y_pred))

print(f"[Hybrid-v2] RMSE={rmse:.3f} MAE={mae:.3f} R2={r2:.3f}")

pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(os.path.join(OUT_DIR, "preds.csv"), index=False)
with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
    json.dump({"RMSE": rmse, "MAE": mae, "R2": r2}, f, indent=2)
