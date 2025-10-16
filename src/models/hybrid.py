# low-data hybrid model with uncertainty (ensemble neural net)
# Here’s a small placeholder (I will expand this later in November)

'''
import torch
import torch.nn as nn

class HybridModel(nn.Module):
    """Numeric + fingerprint fusion model for low-data yield prediction."""
    def __init__(self, num_numeric, fp_dim=1024, hidden=256):
        super().__init__()
        self.numeric = nn.Sequential(
            nn.Linear(num_numeric, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        self.fp = nn.Sequential(
            nn.Linear(fp_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden)
        )
        self.out = nn.Linear(2 * hidden, 1)

    def forward(self, x_num, x_fp):
        n = self.numeric(x_num)
        f = self.fp(x_fp)
        x = torch.cat([n, f], dim=1)
        return self.out(x)
'''

# Trying out this code -- 10/16

# src/models/hybrid.py
import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers
import tensorflow.keras.backend as K

def _mlp_block(units, dropout=0.1, l2=1e-5, name="mlp"):
    return tf.keras.Sequential([
        layers.Dense(units, activation="relu",
                     kernel_regularizer=regularizers.l2(l2),
                     name=f"{name}_dense1"),
        layers.Dropout(dropout, name=f"{name}_drop1"),
        layers.Dense(units, activation="relu",
                     kernel_regularizer=regularizers.l2(l2),
                     name=f"{name}_dense2"),
        layers.Dropout(dropout, name=f"{name}_drop2"),
    ], name=name)

def build_chem_iml(num_numeric: int,
                   fp_dim: int,
                   hidden_num: int = 64,
                   hidden_fp: int = 128,
                   fusion: int = 128,
                   heads: int = 5,
                   dropout_num: float = 0.1,
                   dropout_fp: float = 0.1,
                   l2: float = 1e-5):
    """
    Returns a Keras model that outputs concatenated [mu_1..mu_H, log_var_1..log_var_H]
    of shape (batch, 2*heads).
    """
    # Inputs
    x_num = layers.Input(shape=(num_numeric,), name="x_num")
    x_fp  = layers.Input(shape=(fp_dim,), name="x_fp")

    # Encoders
    enc_num = _mlp_block(hidden_num, dropout=dropout_num, l2=l2, name="enc_num")(x_num)
    enc_fp  = _mlp_block(hidden_fp, dropout=dropout_fp,  l2=l2, name="enc_fp")(x_fp)

    # Fusion
    h = layers.Concatenate(name="fuse")([enc_num, enc_fp])
    h = layers.Dense(fusion, activation="relu",
                     kernel_regularizer=regularizers.l2(l2),
                     name="fuse_dense")(h)
    h = layers.Dropout(0.1, name="fuse_drop")(h)

    # Multi-head outputs
    mus = []
    logvars = []
    for k in range(heads):
        head_h = layers.Dense(fusion//2, activation="relu",
                              kernel_regularizer=regularizers.l2(l2),
                              name=f"head{k}_h")(h)
        mu_k = layers.Dense(1, activation="linear", name=f"head{k}_mu")(head_h)
        lv_k = layers.Dense(1, activation="linear", name=f"head{k}_logvar")(head_h)
        mus.append(mu_k)
        logvars.append(lv_k)

    mus   = layers.Concatenate(name="mus")(mus)         # (B, H)
    lvars = layers.Concatenate(name="logvars")(logvars)  # (B, H)
    out   = layers.Concatenate(name="out_concat")([mus, lvars])  # (B, 2H)

    model = Model(inputs=[x_num, x_fp], outputs=out, name="ChemIML")
    return model

def gaussian_nll_multihead(heads: int):
    """
    Keras loss: y_true (B,), y_pred (B, 2H) -> mean NLL over heads and batch.
    """
    def loss_fn(y_true, y_pred):
        y_true = K.expand_dims(y_true, axis=-1)  # (B, 1)
        mus    = y_pred[:, :heads]               # (B, H)
        lvars  = y_pred[:, heads:]               # (B, H)
        # Softplus for numeric stability on variance
        sigma2 = K.softplus(lvars) + K.epsilon()  # (B, H)
        nll = 0.5 * (K.log(2.0 * K.constant(3.1415926535)) + K.log(sigma2)
                     + K.square(mus - y_true) / sigma2)  # (B, H)
        return K.mean(nll)  # average over heads and batch
    return loss_fn

def split_heads(y_pred, heads: int):
    """
    Utility to split concatenated output into mus, sigmas.
    y_pred: (B, 2H)
    returns mus (B, H), sigmas (B, H)
    """
    mus   = y_pred[:, :heads]
    lvars = y_pred[:, heads:]
    sigmas = K.sqrt(K.softplus(lvars) + K.epsilon())
    return mus, sigmas
