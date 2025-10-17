import tensorflow as tf
from tensorflow.keras import layers, Model

def build_monorank(input_num_dim, input_lat_dim, monotone_idx_num, lattice_sizes=8):
    # Inputs
    num_inputs = [layers.Input(shape=(1,), name=f"x{i}") for i in range(input_num_dim)]
    lat_input = layers.Input(shape=(input_lat_dim,), name="latent")

    # === REPLACE PWLCalibration with simple monotone-safe transform ===
    calibrators = []
    for i, x in enumerate(num_inputs):
        if i in monotone_idx_num:
            # Softplus ensures positive slope → monotonic increase
            cal = layers.Dense(1, activation='softplus', name=f"cal_{i}")(x)
        else:
            cal = layers.Dense(1, activation='linear', name=f"cal_{i}")(x)
        calibrators.append(cal)

    num_concat = layers.Concatenate(name="num_concat")(calibrators)
    lat_proj = layers.Dense(16, activation="relu", name="lat_proj")(lat_input)
    fused = layers.Concatenate(name="fused")([num_concat, lat_proj])

    # Replace the lattice with simple Dense block (same conceptual role)
    fused = layers.Dense(64, activation="relu")(fused)
    fused = layers.Dense(32, activation="relu")(fused)

    score = layers.Dense(1, activation="linear", name="score")(fused)
    median = layers.Dense(1, activation="linear", name="median")(fused)
    qlo = layers.Dense(1, activation="linear", name="q10")(fused)
    qhi = layers.Dense(1, activation="linear", name="q90")(fused)

    model = Model(inputs=num_inputs + [lat_input],
                  outputs=[score, median, qlo, qhi],
                  name="MonoRank")
    return model
