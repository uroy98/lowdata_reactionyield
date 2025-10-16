# ==== Deterministic configuration ====
from src.utils.set_deterministic import set_deterministic
set_deterministic(42)

# Import necessary libraries
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import tensorflow as tf
import numpy as np
import random

def run_rl_xgboost():
    # your full code

    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)

    # Assuming X_train and X_test are your preprocessed datasets
    # Make sure to load/prepare X_train, X_test, y_train, y_test appropriately

    # Load preprocessed data
    X_train = np.load("data/chem/processed/X_train.npy")
    y_train = np.load("data/chem/processed/y_train.npy")
    X_test_xgb = np.load("data/chem/processed/X_test.npy")
    y_test_xgb = np.load("data/chem/processed/y_test.npy")

    # Autoencoder Model for Representation Learning
    input_dim = X_train.shape[1]  # Input dimension
    encoding_dim = 10  # Latent representation dimension

    # Define Autoencoder architecture
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)

    # Build Autoencoder and Encoder models
    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=encoded)

    # Compile Autoencoder with Adam optimizer
    autoencoder.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

    # Use K-Fold Cross Validation to train the Autoencoder
    kf = KFold(n_splits=5)
    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        
        # Train Autoencoder on each fold
        autoencoder.fit(X_train_fold, X_train_fold, epochs=60, batch_size=16, shuffle=False, validation_data=(X_val_fold, X_val_fold))

    # Extract learned representations (encoded data) from the Encoder
    X_train_encoded = encoder.predict(X_train)
    X_test_encoded = encoder.predict(X_test_xgb)

    # Use XGBoost for regression on the encoded data

    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train_encoded, label=y_train)
    dtest = xgb.DMatrix(X_test_encoded, label=y_test_xgb)

    # XGBoost parameters (fine-tuned)
    params = {
        'objective': 'reg:squarederror',  # Regression task
    #     'max_depth': 8,  # Increased depth for better feature learning
    #     'eta': 0.006,  # Lower learning rate for gradual learning
    #     'min_child_weight': 3,  # Prevents overfitting
    #     #'subsample': 0.8,  # Randomly sample 80% of data for each tree
    #     #'colsample_bytree': 0.8,  # Randomly sample 80% of features for each tree
    #     'nthread': 4,  # Parallel threads for faster processing
    #     'eval_metric': 'rmse'  # Use RMSE as evaluation metric
    }

    # Train the XGBoost model using early stopping to prevent overfitting
    num_boost_round = 10  # Max number of boosting iterations
    early_stopping_rounds = 50  # Stop if no improvement in 50 rounds
    bst = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=[(dtest, 'eval')], early_stopping_rounds=early_stopping_rounds)

    # Predict on the test set
    y_pred = bst.predict(dtest)

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test_xgb, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_xgb, y_pred)

    # Print evaluation results
    # print(f"Mean Squared Error (MSE): {mse}")
    # print(f"Root Mean Squared Error (RMSE): {rmse}")
    # print(f"R-squared (R²): {r2}")

    return {"Model": "RL + XGBoost", "RMSE": rmse, "R2": r2}