"""
keras_transformer.py — Transformer and GRU-CNN hybrid models for time series forecasting.

Reference implementation from the StockLTSMTransformerQuantum project demonstrating:
  - Multi-head self-attention with residual connections and layer normalisation
  - Feed-forward sub-layer within the Transformer block
  - Slicing the last time step for sequence-to-one prediction
  - A GRU-CNN hybrid that combines Conv1D feature extraction with GRU memory

Usage:
    python keras_transformer.py

Requirements:
    pip install tensorflow numpy
"""

import numpy as np

# ---------------------------------------------------------------------------
# TensorFlow / Keras imports
# ---------------------------------------------------------------------------
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
    Add,
    Lambda,
    Conv1D,
    MaxPooling1D,
    GRU,
)
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import MeanAbsoluteError


# ============================================================================
# 1. Transformer Model for Time Series
# ============================================================================

def build_transformer_model(
    timesteps: int,
    input_dim: int,
    num_heads: int = 4,
    key_dim: int = 64,
    ff_dim: int = 128,
    head_units: tuple[int, ...] = (100, 50),
    dropout_rate: float = 0.1,
) -> Model:
    """Build a single-block Transformer encoder for time-series prediction.

    Architecture overview:
        Input (timesteps, input_dim)
          -> MultiHeadAttention (self-attention)
          -> Residual Add + LayerNorm
          -> Feed-Forward (Dense 128 relu -> Dense input_dim)
          -> Residual Add + LayerNorm
          -> Slice last timestep  x[:, -1, :]
          -> Dense(100, relu) -> Dropout -> Dense(50, relu) -> Dense(1)

    Args:
        timesteps: Sequence length (look-back window).
        input_dim: Number of features per time step.
        num_heads: Attention heads.
        key_dim: Dimensionality of query/key projections.
        ff_dim: Hidden units in the feed-forward sub-layer.
        head_units: Tuple of dense-layer sizes after the Transformer block.
        dropout_rate: Dropout probability.

    Returns:
        Compiled Keras Model.
    """
    inputs = Input(shape=(timesteps, input_dim), name="ts_input")

    # ------------------------------------------------------------------
    # Multi-Head Self-Attention
    # ------------------------------------------------------------------
    # query = key = value = inputs  (self-attention)
    attn_output = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        name="self_attention",
    )(query=inputs, key=inputs, value=inputs)

    # Residual connection followed by layer normalisation
    attn_add = Add(name="attn_residual")([inputs, attn_output])
    attn_norm = LayerNormalization(epsilon=1e-6, name="attn_layernorm")(attn_add)

    # ------------------------------------------------------------------
    # Position-wise Feed-Forward Network
    # ------------------------------------------------------------------
    ff = Dense(ff_dim, activation="relu", name="ff_dense_1")(attn_norm)
    ff = Dense(input_dim, name="ff_dense_2")(ff)  # project back to input_dim

    # Second residual + layer norm
    ff_add = Add(name="ff_residual")([attn_norm, ff])
    ff_norm = LayerNormalization(epsilon=1e-6, name="ff_layernorm")(ff_add)

    # ------------------------------------------------------------------
    # Slice the *last* time step: x[:, -1, :]
    # This converts (batch, timesteps, features) -> (batch, features)
    # ------------------------------------------------------------------
    last_step = Lambda(
        lambda x: x[:, -1, :],
        name="slice_last_timestep",
    )(ff_norm)

    # ------------------------------------------------------------------
    # Classification / Regression Head
    # ------------------------------------------------------------------
    x = Dense(head_units[0], activation="relu", name="head_dense_1")(last_step)
    x = Dropout(dropout_rate, name="head_dropout")(x)
    x = Dense(head_units[1], activation="relu", name="head_dense_2")(x)
    outputs = Dense(1, name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="TimeSeriesTransformer")
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=[MeanAbsoluteError(name="mae")],
    )
    return model


# ============================================================================
# 2. GRU-CNN Hybrid Model
# ============================================================================

def build_gru_cnn_model(timesteps: int, input_dim: int) -> Model:
    """Build a hybrid Conv1D + GRU model for time-series prediction.

    Architecture:
        Input (timesteps, input_dim)
          -> Conv1D(32, kernel_size=3, relu)
          -> MaxPooling1D(pool_size=2)
          -> GRU(64, return_sequences=True)
          -> GRU(32)
          -> Dense(1)

    The 1-D convolution extracts local patterns (short-term features),
    while the stacked GRU layers capture longer temporal dependencies.

    Args:
        timesteps: Sequence length.
        input_dim: Number of features per time step.

    Returns:
        Compiled Keras Model.
    """
    inputs = Input(shape=(timesteps, input_dim), name="cnn_gru_input")

    # --- Convolutional feature extraction ---
    x = Conv1D(
        filters=32,
        kernel_size=3,
        activation="relu",
        padding="same",
        name="conv1d",
    )(inputs)
    x = MaxPooling1D(pool_size=2, name="maxpool")(x)

    # --- Recurrent layers ---
    x = GRU(64, return_sequences=True, name="gru_1")(x)
    x = GRU(32, name="gru_2")(x)

    # --- Output ---
    outputs = Dense(1, name="output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="GRU_CNN_Hybrid")
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=[MeanAbsoluteError(name="mae")],
    )
    return model


# ============================================================================
# 3. Data helper — synthetic time-series sequences
# ============================================================================

def make_demo_data(
    n_samples: int = 1500,
    timesteps: int = 60,
    n_features: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create synthetic windowed data for quick demonstration.

    Returns:
        X_train, y_train, X_val, y_val
    """
    rng = np.random.default_rng(seed)

    # Simulate a noisy sine wave as the 'close' price
    t = np.linspace(0, 8 * np.pi, n_samples + timesteps)
    close = 50 + 10 * np.sin(t) + rng.normal(0, 0.5, len(t))

    # Build multi-feature matrix (close + correlated noise columns)
    features = np.column_stack(
        [close] + [close + rng.normal(0, 0.3, len(t)) for _ in range(n_features - 1)]
    )

    # Normalise to [0, 1]
    f_min, f_max = features.min(axis=0), features.max(axis=0)
    features = (features - f_min) / (f_max - f_min + 1e-8)

    # Sliding-window sequences
    X, y = [], []
    for i in range(timesteps, len(features)):
        X.append(features[i - timesteps: i])
        y.append(features[i, 0])  # predict the first column ('close')
    X = np.array(X)
    y = np.array(y)

    # Time-ordered split (80/20)
    split = int(len(X) * 0.8)
    return X[:split], y[:split], X[split:], y[split:]


# ============================================================================
# 4. Main — train both models and compare
# ============================================================================

def main():
    TIMESTEPS = 60
    N_FEATURES = 5
    EPOCHS = 10  # short for demo purposes
    BATCH_SIZE = 32

    print("=" * 70)
    print("Transformer & GRU-CNN Hybrid — Time Series Demo")
    print("=" * 70)

    # --- Data ---
    X_train, y_train, X_val, y_val = make_demo_data(
        timesteps=TIMESTEPS, n_features=N_FEATURES,
    )
    print(f"\nTrain: X={X_train.shape}  y={y_train.shape}")
    print(f"Val  : X={X_val.shape}  y={y_val.shape}")

    # ------------------------------------------------------------------
    # Transformer
    # ------------------------------------------------------------------
    print("\n--- Building Transformer model ---")
    transformer = build_transformer_model(
        timesteps=TIMESTEPS,
        input_dim=N_FEATURES,
        num_heads=4,
        key_dim=64,
        ff_dim=128,
        head_units=(100, 50),
        dropout_rate=0.1,
    )
    transformer.summary()

    print("\n--- Training Transformer ---")
    transformer.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    # ------------------------------------------------------------------
    # GRU-CNN Hybrid
    # ------------------------------------------------------------------
    print("\n--- Building GRU-CNN Hybrid model ---")
    gru_cnn = build_gru_cnn_model(timesteps=TIMESTEPS, input_dim=N_FEATURES)
    gru_cnn.summary()

    print("\n--- Training GRU-CNN Hybrid ---")
    gru_cnn.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    # ------------------------------------------------------------------
    # Compare final validation MAE
    # ------------------------------------------------------------------
    t_eval = transformer.evaluate(X_val, y_val, verbose=0)
    g_eval = gru_cnn.evaluate(X_val, y_val, verbose=0)

    print("\n--- Validation Results ---")
    print(f"  Transformer  — loss: {t_eval[0]:.5f}  mae: {t_eval[1]:.5f}")
    print(f"  GRU-CNN      — loss: {g_eval[0]:.5f}  mae: {g_eval[1]:.5f}")


if __name__ == "__main__":
    main()
