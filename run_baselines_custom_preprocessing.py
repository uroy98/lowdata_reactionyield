# Baseline results with ridge, lasso, rf, xgb

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.makedirs("results/baselines_custom", exist_ok=True)

# Load preprocessed data
X_train = np.load("data/chem/processed/X_train.npy")
y_train = np.load("data/chem/processed/y_train.npy")
X_test = np.load("data/chem/processed/X_test.npy")
y_test = np.load("data/chem/processed/y_test.npy")

print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# Standardize
scaler = StandardScaler(with_mean=False)
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Models
models = {
    "ridge": Ridge(alpha=1.0),
    "lasso": Lasso(alpha=0.001, max_iter=5000),
    "rf": RandomForestRegressor(n_estimators=400, random_state=0),
    "xgb": XGBRegressor(
        n_estimators=600, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0, random_state=0
    ),
}

results = []

for name, model in models.items():
    print(f"\nTraining {name} ...")
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    results.append({"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2})

    out = pd.DataFrame({"y_true": y_test, "y_pred": preds})
    out.to_csv(f"results/baselines_custom/{name}_preds.csv", index=False)
    print(f"{name}: RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.3f}")

pd.DataFrame(results).to_csv("results/baselines_custom/baseline_summary.csv", index=False)
print("\n✅ Baseline results saved to results/baselines_custom/")
