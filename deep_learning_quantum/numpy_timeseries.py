"""
numpy_timeseries.py — NumPy utilities for time series operations.

Reference implementation from the StockLTSMTransformerQuantum project demonstrating:
  - Sliding-window sequence creation
  - Moving average via np.convolve
  - np.roll for autoregressive prediction loops
  - Feature normalisation to [-pi, pi] for quantum angle encoding
  - Zero-padding for inverse_transform compatibility
  - np.diff for daily return calculations

Usage:
    python numpy_timeseries.py

Requirements:
    pip install numpy
"""

import numpy as np


# ============================================================================
# 1. Sliding-window sequence creation
# ============================================================================

def create_sequences(
    data: np.ndarray,
    look_back: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Create input/output pairs using a sliding window over 1-D data.

    For each index *i* in [look_back, len(data)), the input is the previous
    ``look_back`` values and the target is the value at index *i*.

    Args:
        data: 1-D array of sequential observations (e.g. closing prices).
        look_back: Number of past time steps in each input window.

    Returns:
        X: Array of shape (n_samples, look_back).
        y: Array of shape (n_samples,).
    """
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i - look_back: i])
        y.append(data[i])
    return np.array(X), np.array(y)


# ============================================================================
# 2. Moving average via np.convolve
# ============================================================================

def moving_average(data: np.ndarray, window: int = 3) -> np.ndarray:
    """Compute a simple moving average using np.convolve.

    The ``'valid'`` mode is used so the output length is
    ``len(data) - window + 1`` — no padding artefacts at the edges.

    Args:
        data: 1-D array of values.
        window: Width of the averaging kernel.

    Returns:
        1-D array of averaged values.
    """
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


# ============================================================================
# 3. Autoregressive prediction loop with np.roll
# ============================================================================

def autoregressive_predict(
    model_fn,
    seed_window: np.ndarray,
    n_steps: int = 10,
) -> np.ndarray:
    """Generate future values by feeding each prediction back as input.

    At each step:
        1. Pass the current window to ``model_fn`` to get one prediction.
        2. Use ``np.roll`` to shift the window left by one position.
        3. Place the new prediction at the end of the shifted window.

    Args:
        model_fn: Callable that accepts a 1-D window and returns a scalar
                  prediction (e.g. a trained model's predict method).
        seed_window: Initial input window, shape (look_back,).
        n_steps: Number of future steps to predict.

    Returns:
        1-D array of shape (n_steps,) with the predicted values.
    """
    current = seed_window.copy()
    predictions = np.empty(n_steps)

    for step in range(n_steps):
        pred = model_fn(current)
        predictions[step] = pred

        # Shift window left: element at index 0 is discarded,
        # the last position is filled with the new prediction.
        current = np.roll(current, -1)
        current[-1] = pred

    return predictions


# ============================================================================
# 4. Feature normalisation to [-pi, pi]
# ============================================================================

def normalize_to_angle_range(
    data: np.ndarray,
    data_min: np.ndarray | None = None,
    data_max: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalise features to [-pi, pi] for quantum angle encoding.

    Formula:
        x_norm = (x - min) / (max - min + 1e-6) * 2*pi - pi

    Args:
        data: Array of shape (n_samples,) or (n_samples, n_features).
        data_min: Pre-computed minimum(s). Computed from *data* if None.
        data_max: Pre-computed maximum(s). Computed from *data* if None.

    Returns:
        (normalised_data, data_min, data_max)
    """
    if data_min is None:
        data_min = data.min(axis=0)
    if data_max is None:
        data_max = data.max(axis=0)

    eps = 1e-6
    normalised = (data - data_min) / (data_max - data_min + eps) * (2 * np.pi) - np.pi
    return normalised, data_min, data_max


def inverse_normalize_from_angle_range(
    normalised: np.ndarray,
    data_min: np.ndarray,
    data_max: np.ndarray,
) -> np.ndarray:
    """Reverse the [-pi, pi] normalisation back to the original scale.

    Args:
        normalised: Data in [-pi, pi].
        data_min: Original minimum(s).
        data_max: Original maximum(s).

    Returns:
        Data in the original scale.
    """
    eps = 1e-6
    ratio = (normalised + np.pi) / (2 * np.pi)
    return ratio * (data_max - data_min + eps) + data_min


# ============================================================================
# 5. Padding for inverse_transform compatibility
# ============================================================================

def pad_for_inverse_transform(
    predictions: np.ndarray,
    n_features: int,
    target_col: int = 0,
) -> np.ndarray:
    """Pad a 1-D prediction array with zero columns so it can be passed
    through a scaler's ``inverse_transform`` method.

    Many scalers (e.g. ``MinMaxScaler``) expect the same number of columns
    they were fitted on.  This helper inserts the predictions into the
    correct column of an otherwise-zero array.

    Example:
        If the scaler was fitted on 7 features and predictions correspond to
        column 3 (Close), the output is shape (n, 7) with column 3 populated
        and all other columns zero.

    Args:
        predictions: 1-D array of length n.
        n_features: Total number of features the scaler expects.
        target_col: Column index for the predictions.

    Returns:
        2-D array of shape (n, n_features).
    """
    n = len(predictions)
    # Create zero-filled array
    padded = np.zeros((n, n_features))
    # Insert predictions into the target column
    padded[:, target_col] = predictions

    # Equivalent explicit form using np.concatenate:
    #   left  = np.zeros((n, target_col))
    #   right = np.zeros((n, n_features - target_col - 1))
    #   padded = np.concatenate([left, predictions.reshape(-1, 1), right], axis=1)

    return padded


# ============================================================================
# 6. Returns calculation with np.diff
# ============================================================================

def compute_returns(prices: np.ndarray, method: str = "simple") -> np.ndarray:
    """Compute daily returns from a price series.

    Args:
        prices: 1-D array of prices (e.g. closing prices).
        method: ``'simple'`` for arithmetic returns (p[t]/p[t-1] - 1),
                ``'log'`` for logarithmic returns (ln(p[t]) - ln(p[t-1])).

    Returns:
        1-D array of returns with length ``len(prices) - 1``.
    """
    if method == "simple":
        # np.diff gives p[t] - p[t-1]; divide by p[t-1] for percentage
        return np.diff(prices) / prices[:-1]
    elif method == "log":
        # Log returns: ln(p[t]) - ln(p[t-1]) = diff(ln(prices))
        return np.diff(np.log(prices))
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'simple' or 'log'.")


# ============================================================================
# 7. Main — demonstrate all utilities
# ============================================================================

def main():
    LOOK_BACK = 60
    MA_WINDOW = 20
    N_DAYS = 500

    print("=" * 70)
    print("NumPy Time Series Utilities — Demo")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Synthetic price data (random walk)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(42)
    prices = 100.0 + np.cumsum(rng.normal(0, 1, N_DAYS))
    print(f"\nPrice series: {N_DAYS} days, "
          f"range [{prices.min():.2f}, {prices.max():.2f}]")

    # ------------------------------------------------------------------
    # 1. Sliding-window sequences
    # ------------------------------------------------------------------
    X, y = create_sequences(prices, look_back=LOOK_BACK)
    print(f"\nSliding windows (look_back={LOOK_BACK}):")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    # Verify: last element of each X row should equal the prior y
    assert np.isclose(X[1, -1], y[0]), "Sequence alignment check failed"

    # ------------------------------------------------------------------
    # 2. Moving average
    # ------------------------------------------------------------------
    ma = moving_average(prices, window=MA_WINDOW)
    print(f"\nMoving average (window={MA_WINDOW}):")
    print(f"  Output length: {len(ma)}  (input was {N_DAYS})")
    print(f"  First 5 values: {ma[:5]}")

    # ------------------------------------------------------------------
    # 3. Autoregressive prediction with np.roll
    # ------------------------------------------------------------------
    # Use a trivial "model": predict the mean of the window
    def naive_model(window):
        return np.mean(window)

    seed = prices[-LOOK_BACK:]
    future = autoregressive_predict(naive_model, seed, n_steps=5)
    print(f"\nAutoregressive prediction (5 steps, naive mean model):")
    print(f"  Last known price: {prices[-1]:.2f}")
    print(f"  Predictions: {future}")

    # ------------------------------------------------------------------
    # 4. Normalisation to [-pi, pi]
    # ------------------------------------------------------------------
    normed, p_min, p_max = normalize_to_angle_range(prices)
    print(f"\nAngle-range normalisation:")
    print(f"  Original range: [{p_min:.2f}, {p_max:.2f}]")
    print(f"  Normalised range: [{normed.min():.4f}, {normed.max():.4f}]")

    # Round-trip check
    recovered = inverse_normalize_from_angle_range(normed, p_min, p_max)
    roundtrip_err = np.max(np.abs(prices - recovered))
    print(f"  Round-trip max error: {roundtrip_err:.2e}")

    # ------------------------------------------------------------------
    # 5. Padding for inverse_transform
    # ------------------------------------------------------------------
    dummy_preds = np.array([0.3, 0.5, 0.7, 0.9])
    padded = pad_for_inverse_transform(dummy_preds, n_features=7, target_col=3)
    print(f"\nPadding for inverse_transform:")
    print(f"  Input shape: {dummy_preds.shape}")
    print(f"  Padded shape: {padded.shape}")
    print(f"  Target column values: {padded[:, 3]}")
    print(f"  Other columns sum: {padded[:, [0,1,2,4,5,6]].sum()}")

    # ------------------------------------------------------------------
    # 6. Returns with np.diff
    # ------------------------------------------------------------------
    simple_ret = compute_returns(prices, method="simple")
    log_ret = compute_returns(prices, method="log")
    print(f"\nDaily returns:")
    print(f"  Simple returns — mean: {simple_ret.mean():.6f}, "
          f"std: {simple_ret.std():.6f}")
    print(f"  Log returns    — mean: {log_ret.mean():.6f}, "
          f"std: {log_ret.std():.6f}")
    print(f"  Length: {len(simple_ret)} (= {N_DAYS} - 1)")

    print("\nDone.")


if __name__ == "__main__":
    main()
