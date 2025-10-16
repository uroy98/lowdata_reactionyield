# ==== Deterministic configuration ====
from src.utils.set_deterministic import set_deterministic
set_deterministic(42)

from src.representation_learning.rl_gradient_boosting import run_rl_gradient_boosting
from src.representation_learning.rl_xgboost import run_rl_xgboost
from src.representation_learning.rl_random_forest import run_rl_random_forest
from src.representation_learning.rl_svr import run_rl_svr
import pandas as pd
import os

os.makedirs("results/representation_baselines", exist_ok=True)

if __name__ == "__main__":
    results = []
    print("=== Running RL + Gradient Boosting ===")
    results.append(run_rl_gradient_boosting())

    print("=== Running RL + XGBoost ===")
    results.append(run_rl_xgboost())

    print("=== Running RL + Random Forest ===")
    results.append(run_rl_random_forest())

    print("=== Running RL + SVR ===")
    results.append(run_rl_svr())

    df = pd.DataFrame(results)
    df.to_csv("results/representation_baselines/representation_summary.csv", index=False)
    print("\n✅ All RL baselines complete. Summary saved.")
