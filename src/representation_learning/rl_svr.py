from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
import numpy as np
import random
import tensorflow as tf

def run_rl_svr():
    # your full code

    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)

    # Load preprocessed data
    X_train = np.load("data/chem/processed/X_train.npy")
    y_train = np.load("data/chem/processed/y_train.npy")
    X_test_svr = np.load("data/chem/processed/X_test.npy")
    y_test_svr = np.load("data/chem/processed/y_test.npy")    

    # 1. Autoencoder Model (Representation Learning)
    input_dim = X_train.shape[1]
    encoding_dim = 10  # Dimension of the latent representation

    # Define a deeper Autoencoder architecture
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)  # Increased the number of neurons
    encoded = Dropout(0.3)(encoded)  # Increased dropout rate
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dropout(0.2)(encoded)
    encoded = Dense(encoding_dim, activation='relu')(encoded)

    decoded = Dense(32, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)

    # Autoencoder model
    autoencoder = Model(inputs=input_layer, outputs=decoded)

    # Encoder model (for extracting learned representations)
    encoder = Model(inputs=input_layer, outputs=encoded)

    # Compile the autoencoder with a smaller learning rate
    autoencoder.compile(optimizer=Adam(learning_rate=0.0004), loss='mse')

    # 2. Train the autoencoder with cross-validation
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        
        # Train the autoencoder on the current fold with more epochs
        autoencoder.fit(X_train_fold, X_train_fold, epochs=100, batch_size=32, shuffle=False, validation_data=(X_val_fold, X_val_fold), verbose=0)

    # 3. Extract learned representations from the encoder
    X_train_encoded = encoder.predict(X_train)
    X_test_encoded = encoder.predict(X_test_svr)

    # 3. Initialize the SVR model
    svr = SVR(kernel='rbf', C=100, gamma=0.01, epsilon=0.1)

    # Train the SVR on the encoded data
    svr.fit(X_train_encoded, y_train)

    # Predict on the test set using the encoded test data
    y_pred = svr.predict(X_test_encoded)

    # 4. Evaluate the model performance
    mse = mean_squared_error(y_test_svr, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test_svr, y_pred)

    # print(f"Mean Squared Error (MSE): {mse}")
    # print(f"Root Mean Squared Error (RMSE): {rmse}")
    # print(f"R-squared (R²): {r2}")


    return {"Model": "RL + SVR", "RMSE": rmse, "R2": r2}
