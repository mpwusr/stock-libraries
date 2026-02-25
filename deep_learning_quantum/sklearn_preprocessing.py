"""
sklearn_preprocessing.py — scikit-learn utilities for financial data preprocessing.

Reference implementation from the StockLTSMTransformerQuantum project demonstrating:
  - MinMaxScaler for OHLCV + technical indicator normalisation
  - inverse_transform for converting predictions back to original scale
  - Sliding-window sequence creation for time-series models
  - Time-ordered train/test split (no shuffling)
  - Evaluation metrics: MAE, MAPE, R-squared
  - Scaler persistence with joblib

Usage:
    python sklearn_preprocessing.py

Requirements:
    pip install scikit-learn numpy joblib
"""

import os
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
import joblib


# ============================================================================
# 1. Synthetic financial data generator
# ============================================================================

def generate_ohlcv_data(
    n_days: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Create synthetic OHLCV + technical indicator data.

    Columns:
        0: Open, 1: High, 2: Low, 3: Close, 4: Volume,
        5: SMA_20 (20-day simple moving average of close),
        6: RSI_14  (simplified 14-day RSI approximation)

    Args:
        n_days: Number of trading days to simulate.
        seed: Random seed.

    Returns:
        Array of shape (n_days, 7).
    """
    rng = np.random.default_rng(seed)

    # Simulate close price as a random walk
    close = 100.0 + np.cumsum(rng.normal(0, 1.2, n_days))

    open_price = close + rng.normal(0, 0.5, n_days)
    high = close + np.abs(rng.normal(0, 1.5, n_days))
    low = close - np.abs(rng.normal(0, 1.5, n_days))
    volume = rng.uniform(1e6, 1e7, n_days)

    # Simple Moving Average (20-day)
    sma_20 = np.convolve(close, np.ones(20) / 20, mode="same")

    # Simplified RSI proxy — percentage of up-days in a 14-day window
    daily_change = np.diff(close, prepend=close[0])
    rsi_14 = np.array([
        np.sum(daily_change[max(0, i - 13): i + 1] > 0) / 14 * 100
        for i in range(n_days)
    ])

    data = np.column_stack([open_price, high, low, close, volume, sma_20, rsi_14])
    return data


# ============================================================================
# 2. MinMaxScaler — fit, transform, inverse_transform
# ============================================================================

def fit_scaler(data: np.ndarray) -> tuple[np.ndarray, MinMaxScaler]:
    """Fit a MinMaxScaler on the data and return the transformed result.

    Scales every feature independently to [0, 1].

    Args:
        data: Raw feature array, shape (n_samples, n_features).

    Returns:
        (scaled_data, fitted_scaler)
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)
    return scaled, scaler


def inverse_transform_predictions(
    predictions: np.ndarray,
    scaler: MinMaxScaler,
    target_col_idx: int = 3,
) -> np.ndarray:
    """Convert scaled predictions back to the original price domain.

    Because the scaler was fitted on *all* features, we need to pad the
    prediction column with zeros for the other features, run
    ``inverse_transform``, and then extract the column of interest.

    Args:
        predictions: 1-D array of scaled predicted values.
        scaler: The fitted MinMaxScaler.
        target_col_idx: Index of the target column within the original data
                        (default 3 = Close).

    Returns:
        1-D array of predictions in the original scale.
    """
    n_features = scaler.n_features_in_

    # Build a dummy array with all zeros, then fill the target column
    dummy = np.zeros((len(predictions), n_features))
    dummy[:, target_col_idx] = predictions.ravel()

    # Inverse transform and extract the target column
    inv = scaler.inverse_transform(dummy)
    return inv[:, target_col_idx]


# ============================================================================
# 3. Sliding-window sequence creation
# ============================================================================

def create_sequences(
    data: np.ndarray,
    target: np.ndarray,
    look_back: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Create input/output pairs using a sliding window.

    For each index *i* in [look_back, len(data)), the input is
    ``data[i-look_back : i]`` and the output is ``target[i]``.

    Args:
        data: Scaled feature array, shape (n_samples, n_features).
        target: Scaled target array, shape (n_samples,).
        look_back: Window size.

    Returns:
        X: (n_sequences, look_back, n_features)
        y: (n_sequences,)
    """
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back: i])
        y.append(target[i])
    return np.array(X), np.array(y)


# ============================================================================
# 4. Time-ordered train/test split
# ============================================================================

def time_series_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split sequences into train and test sets *without* shuffling.

    For time-series data, shuffling would leak future information into the
    training set.  We use ``train_test_split`` with ``shuffle=False`` to
    preserve chronological order.

    Args:
        X: Feature sequences.
        y: Targets.
        test_size: Fraction allocated to the test set.

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        shuffle=False,  # CRITICAL for time-series data
    )
    return X_train, X_test, y_train, y_test


# ============================================================================
# 5. Evaluation metrics
# ============================================================================

def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute MAE, MAPE, and R-squared for a set of predictions.

    Args:
        y_true: Ground-truth values.
        y_pred: Model predictions.

    Returns:
        Dictionary with keys 'mae', 'mape', 'r2'.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {"mae": mae, "mape": mape, "r2": r2}


# ============================================================================
# 6. Scaler persistence with joblib
# ============================================================================

def save_scaler(scaler: MinMaxScaler, path: str = "scaler.joblib") -> None:
    """Save a fitted scaler to disk using joblib.

    Joblib is preferred over pickle for scikit-learn estimators because it
    handles large NumPy arrays more efficiently.

    Args:
        scaler: Fitted MinMaxScaler.
        path: Destination file path.
    """
    joblib.dump(scaler, path)
    print(f"Scaler saved to {path}")


def load_scaler(path: str = "scaler.joblib") -> MinMaxScaler:
    """Load a previously saved scaler from disk.

    Args:
        path: File path to the saved scaler.

    Returns:
        Fitted MinMaxScaler.
    """
    scaler = joblib.load(path)
    print(f"Scaler loaded from {path}")
    return scaler


# ============================================================================
# 7. Main — end-to-end preprocessing demo
# ============================================================================

def main():
    LOOK_BACK = 60
    TEST_SIZE = 0.2
    TARGET_COL = 3  # Close price
    SCALER_PATH = "demo_scaler.joblib"

    print("=" * 70)
    print("scikit-learn Preprocessing for Financial Time Series")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Generate synthetic OHLCV + indicators
    # ------------------------------------------------------------------
    raw_data = generate_ohlcv_data(n_days=1000)
    col_names = ["Open", "High", "Low", "Close", "Volume", "SMA_20", "RSI_14"]
    print(f"\nRaw data shape: {raw_data.shape}")
    print(f"Columns: {col_names}")
    print(f"Close price range: [{raw_data[:, TARGET_COL].min():.2f}, "
          f"{raw_data[:, TARGET_COL].max():.2f}]")

    # ------------------------------------------------------------------
    # Step 2: Fit MinMaxScaler
    # ------------------------------------------------------------------
    scaled_data, scaler = fit_scaler(raw_data)
    print(f"\nScaled data range: [{scaled_data.min():.4f}, {scaled_data.max():.4f}]")

    # ------------------------------------------------------------------
    # Step 3: Create sequences
    # ------------------------------------------------------------------
    target_scaled = scaled_data[:, TARGET_COL]
    X, y = create_sequences(scaled_data, target_scaled, look_back=LOOK_BACK)
    print(f"\nSequence shapes: X={X.shape}, y={y.shape}")

    # ------------------------------------------------------------------
    # Step 4: Time-ordered split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = time_series_split(X, y, test_size=TEST_SIZE)
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # ------------------------------------------------------------------
    # Step 5: Simulate predictions (use train mean as a naive baseline)
    # ------------------------------------------------------------------
    naive_preds = np.full_like(y_test, fill_value=y_train.mean())

    # Evaluate in scaled space
    metrics_scaled = evaluate_predictions(y_test, naive_preds)
    print("\n--- Metrics (scaled space) ---")
    for k, v in metrics_scaled.items():
        print(f"  {k}: {v:.6f}")

    # ------------------------------------------------------------------
    # Step 6: Inverse-transform back to original prices
    # ------------------------------------------------------------------
    y_test_original = inverse_transform_predictions(y_test, scaler, TARGET_COL)
    preds_original = inverse_transform_predictions(naive_preds, scaler, TARGET_COL)

    metrics_original = evaluate_predictions(y_test_original, preds_original)
    print("\n--- Metrics (original price scale) ---")
    for k, v in metrics_original.items():
        print(f"  {k}: {v:.6f}")

    print(f"\nSample actual prices : {y_test_original[:5]}")
    print(f"Sample predictions   : {preds_original[:5]}")

    # ------------------------------------------------------------------
    # Step 7: Save and reload scaler
    # ------------------------------------------------------------------
    save_scaler(scaler, SCALER_PATH)
    reloaded_scaler = load_scaler(SCALER_PATH)

    # Verify the reloaded scaler produces identical results
    check_inv = inverse_transform_predictions(y_test[:3], reloaded_scaler, TARGET_COL)
    print(f"\nReloaded scaler check: {check_inv}")
    assert np.allclose(y_test_original[:3], check_inv), "Scaler mismatch!"
    print("Scaler persistence verified.")

    # Clean up
    if os.path.exists(SCALER_PATH):
        os.remove(SCALER_PATH)
        print(f"Cleaned up {SCALER_PATH}")


if __name__ == "__main__":
    main()
