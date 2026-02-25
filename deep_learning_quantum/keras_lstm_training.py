"""
keras_lstm_training.py — LSTM model with Keras Functional API for stock price prediction.

Reference implementation from the StockLTSMTransformerQuantum project demonstrating:
  - LSTM architecture with dropout regularization
  - Custom callback for tracking financial metrics (MAE, MAPE, R-squared, Sharpe ratio)
  - EarlyStopping and ModelCheckpoint callbacks
  - Train/validation split preserving time-series order
  - Model persistence in the .keras format

Usage:
    python keras_lstm_training.py

Requirements:
    pip install tensorflow numpy scikit-learn
"""

import os
import numpy as np
from sklearn.metrics import r2_score

# ---------------------------------------------------------------------------
# TensorFlow / Keras imports
# ---------------------------------------------------------------------------
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.metrics import MeanAbsoluteError, MeanAbsolutePercentageError
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback


# ============================================================================
# 1. Custom Callback — tracks MAE, MAPE, R-squared, and Sharpe ratio per epoch
# ============================================================================

class FinancialMetricsCallback(Callback):
    """Track financial-quality metrics at the end of every epoch.

    Attributes:
        X_val (np.ndarray): Validation features of shape (samples, timesteps, features).
        y_val (np.ndarray): Validation targets of shape (samples,) or (samples, 1).
        history (dict): Accumulated per-epoch metric values.
    """

    def __init__(self, X_val: np.ndarray, y_val: np.ndarray):
        super().__init__()
        self.X_val = X_val
        # Flatten targets to 1-D for metric calculations
        self.y_val = y_val.ravel()
        self.history: dict[str, list[float]] = {
            "val_mae": [],
            "val_mape": [],
            "val_r2": [],
            "val_sharpe": [],
        }

    # ------------------------------------------------------------------
    # Helper: annualised Sharpe ratio from prediction vs actual returns
    # ------------------------------------------------------------------
    @staticmethod
    def _sharpe_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute a Sharpe-like ratio from the *returns* implied by
        predictions vs actuals.

        Uses simple differenced returns and assumes 252 trading days per year.
        """
        # Simple returns (day-over-day changes)
        actual_returns = np.diff(y_true)
        pred_returns = np.diff(y_pred)

        # Excess return of the prediction over the actual
        excess = pred_returns - actual_returns

        std = np.std(excess)
        if std < 1e-9:
            return 0.0
        # Annualise: multiply by sqrt(252)
        return float(np.mean(excess) / std * np.sqrt(252))

    # ------------------------------------------------------------------
    # Epoch-end hook
    # ------------------------------------------------------------------
    def on_epoch_end(self, epoch: int, logs: dict | None = None):
        """Evaluate and log custom metrics after each epoch."""
        predictions = self.model.predict(self.X_val, verbose=0).ravel()

        mae = float(np.mean(np.abs(self.y_val - predictions)))
        # Guard against zero actuals when computing MAPE
        nonzero_mask = np.abs(self.y_val) > 1e-9
        if nonzero_mask.any():
            mape = float(
                np.mean(
                    np.abs((self.y_val[nonzero_mask] - predictions[nonzero_mask])
                           / self.y_val[nonzero_mask])
                ) * 100.0
            )
        else:
            mape = 0.0

        r2 = float(r2_score(self.y_val, predictions))
        sharpe = self._sharpe_ratio(self.y_val, predictions)

        self.history["val_mae"].append(mae)
        self.history["val_mape"].append(mape)
        self.history["val_r2"].append(r2)
        self.history["val_sharpe"].append(sharpe)

        print(
            f"  [FinancialMetrics] epoch {epoch + 1}: "
            f"MAE={mae:.4f}  MAPE={mape:.2f}%  "
            f"R²={r2:.4f}  Sharpe={sharpe:.4f}"
        )


# ============================================================================
# 2. Build LSTM model — Keras Functional API
# ============================================================================

def build_lstm_model(timesteps: int, n_features: int) -> Model:
    """Construct a two-layer LSTM with dropout for stock price prediction.

    Architecture:
        Input -> LSTM(50, return_sequences) -> Dropout(0.2)
              -> LSTM(50) -> Dropout(0.2)
              -> Dense(1)

    Args:
        timesteps: Number of look-back time steps.
        n_features: Number of input features per time step.

    Returns:
        Compiled Keras Model.
    """
    # --- Functional API ---
    inputs = Input(shape=(timesteps, n_features), name="price_input")

    # First LSTM layer — return full sequence for stacking
    x = LSTM(50, return_sequences=True, name="lstm_1")(inputs)
    x = Dropout(0.2, name="dropout_1")(x)

    # Second LSTM layer — return only final hidden state
    x = LSTM(50, name="lstm_2")(x)
    x = Dropout(0.2, name="dropout_2")(x)

    # Output layer — single price prediction
    outputs = Dense(1, name="price_output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="StockLSTM")

    # --- Compile with Adam optimiser and MSE loss ---
    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=[
            MeanAbsoluteError(name="mae"),
            MeanAbsolutePercentageError(name="mape"),
        ],
    )

    return model


# ============================================================================
# 3. Data helpers — synthetic data and sequence creation
# ============================================================================

def generate_synthetic_stock_data(
    n_samples: int = 2000,
    n_features: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic stock-like data for demonstration.

    Returns:
        features: (n_samples, n_features) array simulating OHLCV-style data.
        target: (n_samples,) array — the 'close' price we want to predict.
    """
    rng = np.random.default_rng(seed)
    # Simulate a random-walk close price
    close = 100.0 + np.cumsum(rng.normal(0, 1, n_samples))
    # Build correlated features around close
    features = np.column_stack([
        close + rng.normal(0, 0.5, n_samples),   # open
        close + np.abs(rng.normal(0, 1, n_samples)),  # high
        close - np.abs(rng.normal(0, 1, n_samples)),  # low
        close,                                     # close
        rng.uniform(1e6, 5e6, n_samples),          # volume
    ])
    # If more features requested, pad with noise
    if n_features > 5:
        extra = rng.normal(0, 1, (n_samples, n_features - 5))
        features = np.hstack([features, extra])

    return features[:, :n_features], close


def create_sequences(
    features: np.ndarray,
    target: np.ndarray,
    look_back: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window of *look_back* steps over the data to create
    (X, y) pairs suitable for LSTM training.

    Args:
        features: (n_samples, n_features)
        target: (n_samples,)
        look_back: Number of past time steps per sample.

    Returns:
        X: (n_samples - look_back, look_back, n_features)
        y: (n_samples - look_back,)
    """
    X, y = [], []
    for i in range(look_back, len(features)):
        X.append(features[i - look_back: i])
        y.append(target[i])
    return np.array(X), np.array(y)


# ============================================================================
# 4. Training pipeline
# ============================================================================

def train_model(
    model: Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    checkpoint_path: str = "best_lstm_model.keras",
):
    """Run the training loop with callbacks.

    Callbacks:
        - EarlyStopping: stop if val_loss doesn't improve for 5 epochs;
          restore the best weights automatically.
        - ModelCheckpoint: persist the best model to disk in .keras format.
        - FinancialMetricsCallback: log MAE, MAPE, R², Sharpe every epoch.

    Returns:
        (keras History object, FinancialMetricsCallback instance)
    """
    # --- Callbacks ---
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    financial_cb = FinancialMetricsCallback(X_val, y_val)

    # --- Fit ---
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, checkpoint, financial_cb],
        verbose=1,
    )

    return history, financial_cb


# ============================================================================
# 5. Model save / load helpers
# ============================================================================

def save_model(model: Model, path: str = "stock_lstm.keras") -> None:
    """Persist a trained model in the modern .keras format."""
    model.save(path)
    print(f"Model saved to {path}")


def load_saved_model(path: str = "stock_lstm.keras") -> Model:
    """Load a previously saved .keras model."""
    model = load_model(path)
    print(f"Model loaded from {path}")
    return model


# ============================================================================
# 6. Main — end-to-end demo
# ============================================================================

def main():
    # --- Hyper-parameters ---
    LOOK_BACK = 60
    N_FEATURES = 5
    EPOCHS = 20          # keep short for demo
    BATCH_SIZE = 32
    TRAIN_RATIO = 0.8    # 80 / 20 time-ordered split
    MODEL_PATH = "best_stock_lstm.keras"

    print("=" * 70)
    print("LSTM Stock Price Prediction — Keras Functional API Demo")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Generate synthetic data (replace with real OHLCV in practice)
    # ------------------------------------------------------------------
    features, target = generate_synthetic_stock_data(
        n_samples=2000, n_features=N_FEATURES,
    )
    print(f"\nRaw data shape: features={features.shape}, target={target.shape}")

    # ------------------------------------------------------------------
    # Step 2: Simple MinMax normalisation (per-feature)
    # ------------------------------------------------------------------
    f_min, f_max = features.min(axis=0), features.max(axis=0)
    features_norm = (features - f_min) / (f_max - f_min + 1e-8)

    t_min, t_max = target.min(), target.max()
    target_norm = (target - t_min) / (t_max - t_min + 1e-8)

    # ------------------------------------------------------------------
    # Step 3: Create sequences
    # ------------------------------------------------------------------
    X, y = create_sequences(features_norm, target_norm, look_back=LOOK_BACK)
    print(f"Sequence data: X={X.shape}, y={y.shape}")

    # ------------------------------------------------------------------
    # Step 4: Time-ordered train / validation split (NO shuffle)
    # ------------------------------------------------------------------
    split_idx = int(len(X) * TRAIN_RATIO)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    print(f"Train: {X_train.shape[0]} samples | Val: {X_val.shape[0]} samples")

    # ------------------------------------------------------------------
    # Step 5: Build model
    # ------------------------------------------------------------------
    model = build_lstm_model(timesteps=LOOK_BACK, n_features=N_FEATURES)
    model.summary()

    # ------------------------------------------------------------------
    # Step 6: Train
    # ------------------------------------------------------------------
    history, fin_cb = train_model(
        model, X_train, y_train, X_val, y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        checkpoint_path=MODEL_PATH,
    )

    # ------------------------------------------------------------------
    # Step 7: Save and reload
    # ------------------------------------------------------------------
    save_model(model, MODEL_PATH)
    reloaded = load_saved_model(MODEL_PATH)

    # Quick sanity check — predictions from both should match
    preds_original = model.predict(X_val[:5], verbose=0)
    preds_reloaded = reloaded.predict(X_val[:5], verbose=0)
    print("\nSanity check (original vs reloaded predictions):")
    print(f"  Original : {preds_original.ravel()}")
    print(f"  Reloaded : {preds_reloaded.ravel()}")

    # ------------------------------------------------------------------
    # Step 8: Print final financial metrics
    # ------------------------------------------------------------------
    print("\n--- Final Epoch Financial Metrics ---")
    for key, values in fin_cb.history.items():
        print(f"  {key}: {values[-1]:.4f}")

    # Clean up checkpoint file
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
        print(f"\nCleaned up {MODEL_PATH}")


if __name__ == "__main__":
    main()
