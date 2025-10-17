"""
Fine-tune the pretrained autoencoder encoder using reaction yields
to obtain yield-aware latent representations.
"""

import os
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from src.utils.set_deterministic import set_deterministic

set_deterministic(42)

DATA_DIR = "data/chem/processed"
MODEL_DIR = "src/models"

# 1. Load data
X_train = np.load(os.path.join(DATA_DIR, "X_train.npy")).astype(np.float32)
y_train = np.load(os.path.join(DATA_DIR, "y_train.npy")).astype(np.float32)
X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy")).astype(np.float32)
y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy")).astype(np.float32)

# Optional: scale features for stability
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled  = sc.transform(X_test)

# 2. Load pretrained encoder (from your RL baseline)
encoder = load_model(os.path.join(MODEL_DIR, "autoencoder_encoder.h5"))
print("Loaded encoder:", encoder.summary())

# 3. Freeze all but last layer for transfer learning
for layer in encoder.layers[:-1]:
    layer.trainable = False

# 4. Add a small supervised head for yield prediction
# x = encoder.output
# x = Dense(32, activation="relu")(x)
# out = Dense(1, activation="linear")(x)
# fine_tune_model = Model(inputs=encoder.input, outputs=out)

x = encoder.output
x = Dense(32, activation="relu", name="ft_dense")(x)
out = Dense(1, activation="linear", name="ft_output")(x)
fine_tune_model = Model(inputs=encoder.input, outputs=out, name="fine_tune_model")


fine_tune_model.compile(optimizer=Adam(1e-3), loss="mse")
fine_tune_model.summary()

# 5. Fine-tune on your labeled data
callbacks = [EarlyStopping(monitor="val_loss", patience=40, restore_best_weights=True)]
fine_tune_model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=300, batch_size=8, verbose=1,
    callbacks=callbacks
)

# 6. Extract yield-aware latent features
encoder_finetuned = Model(inputs=encoder.input, outputs=encoder.layers[-2].output)
X_train_lat = encoder_finetuned.predict(X_train_scaled)
X_test_lat  = encoder_finetuned.predict(X_test_scaled)

# 7. Save for hybrid training
np.save(os.path.join(DATA_DIR, "X_train_lat_finetuned.npy"), X_train_lat)
np.save(os.path.join(DATA_DIR, "X_test_lat_finetuned.npy"),  X_test_lat)
encoder_finetuned.save(os.path.join(MODEL_DIR, "encoder_finetuned.h5"))

print("✅ Saved fine-tuned latent representations.")
