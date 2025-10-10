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

def run_rl_random_forest():
    # your full code

    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)

    # Load preprocessed data
    X_train = np.load("data/chem/processed/X_train.npy")
    y_train = np.load("data/chem/processed/y_train.npy")
    X_test_rf = np.load("data/chem/processed/X_test.npy")
    y_test_rf = np.load("data/chem/processed/y_test.npy")    

    # 1. Autoencoder Model (Representation Learning)
    input_dim = X_train.shape[1]
    encoding_dim = 10  # Dimension of the latent representation

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
        
        # Train the autoencoder on the current fold with more epochs
        autoencoder.fit(X_train_fold, X_train_fold, epochs=300, batch_size=32, shuffle=False, validation_data=(X_val_fold, X_val_fold), verbose=0)

    # 3. Extract learned representations from the encoder
    X_train_encoded = encoder.predict(X_train)
    X_test_encoded = encoder.predict(X_test_rf)

    '''
    # Define fixed hyperparameters for the RandomForestRegressor
    params = {
        'n_estimators': 300,        # Number of trees=500
        'max_depth': 20,            # Maximum depth of the tree
        'min_samples_split': 2,     # Minimum number of samples required to split an internal node
        'min_samples_leaf': 1,      # Minimum number of samples required to be at a leaf node
        'max_features': 'auto'      # The number of features to consider when looking for the best split
    }

    # Initialize and train the RandomForestRegressor with fixed hyperparameters
    model = RandomForestRegressor(**params, random_state=42)
    model.fit(X_train_encoded, y_train)
    '''
    # Initialize a RandomForestRegressor
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)

    # Train the model
    rf.fit(X_train_encoded, y_train)

    # Predict on the test set
    y_pred = rf.predict(X_test_encoded)

    # Calculate performance metrics
    mse = mean_squared_error(y_test_rf, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test_rf, y_pred)

    # print(f"Random Forest Model RMSE: {rmse}")
    # print(f"R-squared (R²): {r2}")

    return {"Model": "RL + RandomForest", "RMSE": rmse, "R2": r2}
