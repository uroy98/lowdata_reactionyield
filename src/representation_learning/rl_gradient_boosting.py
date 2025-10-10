import optuna
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
import numpy as np
import random
import tensorflow as tf

def run_rl_gradient_boosting():
    # your full code

    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)

    # Load preprocessed data
    X_train = np.load("data/chem/processed/X_train.npy")
    y_train = np.load("data/chem/processed/y_train.npy")
    X_test = np.load("data/chem/processed/X_test.npy")
    y_test = np.load("data/chem/processed/y_test.npy")

    # 1. Autoencoder Model (Representation Learning)
    input_dim = X_train.shape[1]
    encoding_dim = 10  # Increased the dimension of the latent representation

    # Define a deeper Autoencoder architecture
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128, activation='relu')(input_layer)  # Increased the number of neurons
    encoded = Dropout(0.3)(encoded)  # Increased dropout rate
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dropout(0.2)(encoded)
    encoded = Dense(encoding_dim, activation='relu')(encoded)

    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    # Autoencoder model
    autoencoder = Model(inputs=input_layer, outputs=decoded)

    # Encoder model (for extracting learned representations)
    encoder = Model(inputs=input_layer, outputs=encoded)

    # Compile the autoencoder with a smaller learning rate
    autoencoder.compile(optimizer=Adam(learning_rate=0.0004), loss='mse')

    # 2. Train the autoencoder with cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        
        # Train the autoencoder on the current fold with more epochs #batch_size=32
        autoencoder.fit(X_train_fold, X_train_fold, epochs=300, batch_size=32, shuffle=False, validation_data=(X_val_fold, X_val_fold), verbose=0)

    # 3. Extract learned representations from the encoder
    X_train_encoded = encoder.predict(X_train)
    X_test_encoded = encoder.predict(X_test)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
            #'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.2),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.0005, 0.05),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
            'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None]),
            'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0)
            #'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0)
        }
        model = GradientBoostingRegressor(**params, random_state=42)
        model.fit(X_train_encoded, y_train)
        y_pred = model.predict(X_test_encoded)
        mse = mean_squared_error(y_test, y_pred)
        return mse

    #study = optuna.create_study(direction='minimize')
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))  # Add seed to the sampler
    study.optimize(objective, n_trials=200)

    best_params = study.best_params
    best_model = GradientBoostingRegressor(**best_params, random_state=42)
    best_model.fit(X_train_encoded, y_train)
    y_pred = best_model.predict(X_test_encoded)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    # print(f"Optuna Optimized Model RMSE: {rmse}")
    # print(f"R-squared (R²): {r2}")

    return {"Model": "RL + GradientBoosting", "RMSE": rmse, "R2": r2}
