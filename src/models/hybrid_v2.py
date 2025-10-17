import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers

def build_chemlatent_hybrid(num_process: int, latent_dim: int = 10,
                            hidden_proc: int = 64, fusion_dim: int = 128, l2=1e-5, dropout=0.1):
    # Inputs
    x_proc = layers.Input(shape=(num_process,), name="x_proc")
    x_lat  = layers.Input(shape=(latent_dim,), name="x_lat")

    # Process encoder
    h_proc = layers.Dense(hidden_proc, activation="relu",
                          kernel_regularizer=regularizers.l2(l2))(x_proc)
    h_proc = layers.Dropout(dropout)(h_proc)
    h_proc = layers.Dense(hidden_proc, activation="relu",
                          kernel_regularizer=regularizers.l2(l2))(h_proc)

    # Fusion
    h = layers.Concatenate()([h_proc, x_lat])
    h = layers.Dense(fusion_dim, activation="relu")(h)
    h = layers.Dense(fusion_dim//2, activation="relu")(h)
    out = layers.Dense(1, activation="linear", name="yield")(h)

    model = Model(inputs=[x_proc, x_lat], outputs=out, name="ChemHybrid_v2")
    return model
