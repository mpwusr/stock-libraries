"""
pennylane_quantum.py — Quantum Machine Learning with PennyLane for stock prediction.

Reference implementation from the StockLTSMTransformerQuantum project demonstrating:
  - Quantum device setup (default.qubit simulator, 4 wires)
  - Variational quantum circuit with RY/RZ rotation gates and CNOT entanglement
  - 4-layer parameterised ansatz
  - Feature normalisation to [-pi, pi]
  - Gradient-based optimisation with qml.AdamOptimizer
  - Autoregressive quantum prediction with inverse scaling

Usage:
    python pennylane_quantum.py

Requirements:
    pip install pennylane numpy
"""

import numpy as np

try:
    import pennylane as qml
except ImportError:
    raise ImportError(
        "PennyLane is required.  Install with:  pip install pennylane"
    )


# ============================================================================
# 1. Device and circuit configuration
# ============================================================================

# Number of qubits — matches the feature dimension after encoding
N_QUBITS = 4
# Number of variational layers in the ansatz
N_LAYERS = 4

# Simulator device — swap for a hardware backend when available
dev = qml.device("default.qubit", wires=N_QUBITS)


# ============================================================================
# 2. Quantum circuit (QNode)
# ============================================================================

@qml.qnode(dev)
def quantum_circuit(inputs: np.ndarray, weights: np.ndarray) -> float:
    """Parameterised quantum circuit for regression.

    Encoding strategy:
        Each input feature is encoded as an RY rotation on its respective qubit.

    Variational ansatz (repeated *N_LAYERS* times):
        1. RY(weight) and RZ(weight) on every qubit  (single-qubit rotations)
        2. Ring of CNOT gates for entanglement (wire i -> wire (i+1) % n)

    Measurement:
        Expectation value of PauliZ on wire 0 — returned as a scalar in [-1, 1].

    Args:
        inputs: Feature vector of length N_QUBITS, pre-normalised to [-pi, pi].
        weights: Trainable parameters, shape (N_LAYERS, N_QUBITS, 2).

    Returns:
        Expectation <Z_0> as a float.
    """
    # --- Feature encoding layer ---
    for i in range(N_QUBITS):
        qml.RY(inputs[i], wires=i)

    # --- Variational layers ---
    for layer in range(N_LAYERS):
        # Single-qubit rotations with trainable angles
        for qubit in range(N_QUBITS):
            qml.RY(weights[layer, qubit, 0], wires=qubit)
            qml.RZ(weights[layer, qubit, 1], wires=qubit)

        # Entangling CNOT ring
        for qubit in range(N_QUBITS):
            qml.CNOT(wires=[qubit, (qubit + 1) % N_QUBITS])

    # --- Measurement ---
    return qml.expval(qml.PauliZ(0))


# ============================================================================
# 3. Feature normalisation
# ============================================================================

def normalize_input(
    features: np.ndarray,
    feature_min: np.ndarray | None = None,
    feature_max: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map raw features to the range [-pi, pi] for angle encoding.

    The mapping is:  x_norm = (x - min) / (max - min + eps) * 2*pi - pi

    Args:
        features: Array of shape (n_samples, n_features).
        feature_min: Pre-computed per-feature minimums (optional).
        feature_max: Pre-computed per-feature maximums (optional).

    Returns:
        (normalised_features, feature_min, feature_max)
    """
    if feature_min is None:
        feature_min = features.min(axis=0)
    if feature_max is None:
        feature_max = features.max(axis=0)

    eps = 1e-6
    normalised = (features - feature_min) / (feature_max - feature_min + eps)
    # Scale from [0, 1] to [-pi, pi]
    normalised = normalised * 2.0 * np.pi - np.pi

    return normalised, feature_min, feature_max


# ============================================================================
# 4. Cost (loss) function — MSE between quantum predictions and targets
# ============================================================================

def cost_function(
    weights: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
) -> float:
    """Mean Squared Error between quantum circuit predictions and targets.

    Args:
        weights: Shape (N_LAYERS, N_QUBITS, 2).
        X: Normalised input features, shape (n_samples, N_QUBITS).
        y: Target values, shape (n_samples,).

    Returns:
        Scalar MSE loss.
    """
    predictions = np.array([quantum_circuit(x, weights) for x in X])
    mse = np.mean((predictions - y) ** 2)
    return mse


# ============================================================================
# 5. Optimisation loop
# ============================================================================

def optimize_quantum_weights(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_epochs: int = 30,
    learning_rate: float = 0.05,
    seed: int = 42,
) -> np.ndarray:
    """Train variational weights using PennyLane's AdamOptimizer.

    Args:
        X_train: Normalised training features, shape (n_samples, N_QUBITS).
        y_train: Training targets, shape (n_samples,).
        n_epochs: Number of optimisation steps.
        learning_rate: Adam learning rate.
        seed: Random seed for weight initialisation.

    Returns:
        Optimised weight array, shape (N_LAYERS, N_QUBITS, 2).
    """
    rng = np.random.default_rng(seed)

    # Initialise weights uniformly in [0, 2*pi)
    weights = rng.uniform(0, 2 * np.pi, size=(N_LAYERS, N_QUBITS, 2))
    # PennyLane requires numpy arrays (not JAX/Torch tensors for default.qubit)
    weights = np.array(weights, requires_grad=True)

    optimizer = qml.AdamOptimizer(stepsize=learning_rate)

    print(f"{'Epoch':>6}  {'Loss':>10}")
    print("-" * 20)

    for epoch in range(1, n_epochs + 1):
        # AdamOptimizer.step expects (cost_fn, *args) and returns updated args
        weights, loss_val = optimizer.step_and_cost(
            lambda w: cost_function(w, X_train, y_train),
            weights,
        )

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>6}  {loss_val:>10.6f}")

    return weights


# ============================================================================
# 6. Autoregressive prediction with inverse scaling
# ============================================================================

def quantum_predict_future(
    weights: np.ndarray,
    last_window: np.ndarray,
    n_steps: int = 10,
    target_min: float = 0.0,
    target_max: float = 1.0,
) -> np.ndarray:
    """Generate future predictions autoregressively.

    At each step the circuit produces one normalised prediction which is
    appended to the input window (shifting by one) for the next step.
    After all steps, predictions are inverse-scaled back to the original
    price domain.

    Args:
        weights: Trained variational weights, shape (N_LAYERS, N_QUBITS, 2).
        last_window: Most recent normalised input, shape (N_QUBITS,).
        n_steps: How many future steps to predict.
        target_min: Original target minimum (for inverse scaling).
        target_max: Original target maximum (for inverse scaling).

    Returns:
        Array of shape (n_steps,) with predictions in the original scale.
    """
    current_input = last_window.copy()
    raw_preds = []

    for _ in range(n_steps):
        # Quantum circuit returns a value in [-1, 1]; treat as normalised pred
        pred_norm = quantum_circuit(current_input, weights)
        raw_preds.append(float(pred_norm))

        # Shift the input window: drop oldest, append new prediction
        current_input = np.roll(current_input, -1)
        current_input[-1] = pred_norm

    raw_preds = np.array(raw_preds)

    # --- Inverse scaling ---
    # Circuit output in [-1, 1] -> map to [0, 1] -> map to [target_min, target_max]
    normalised_01 = (raw_preds + 1.0) / 2.0  # [-1,1] -> [0,1]
    predictions = normalised_01 * (target_max - target_min) + target_min

    return predictions


# ============================================================================
# 7. Main — end-to-end demo
# ============================================================================

def main():
    N_SAMPLES = 80   # small dataset for quantum simulation speed
    N_EPOCHS = 20
    LR = 0.05

    print("=" * 70)
    print("Quantum ML with PennyLane — Stock Prediction Demo")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Synthetic data — 4 features (to match 4 qubits)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(0)
    t = np.linspace(0, 4 * np.pi, N_SAMPLES)
    close = 100 + 10 * np.sin(t) + rng.normal(0, 0.5, N_SAMPLES)

    features_raw = np.column_stack([
        close,
        close + rng.normal(0, 0.3, N_SAMPLES),
        close - rng.normal(0, 0.3, N_SAMPLES),
        rng.uniform(1e6, 2e6, N_SAMPLES),
    ])

    # Normalise features to [-pi, pi]
    X_norm, f_min, f_max = normalize_input(features_raw)
    print(f"\nNormalised features shape: {X_norm.shape}")
    print(f"Value range: [{X_norm.min():.3f}, {X_norm.max():.3f}]")

    # Target: normalise close price to [-1, 1] to match PauliZ expectation
    t_min, t_max = close.min(), close.max()
    y_norm = 2.0 * (close - t_min) / (t_max - t_min + 1e-6) - 1.0

    # Simple train / test split (time-ordered)
    split = int(0.75 * N_SAMPLES)
    X_train, y_train = X_norm[:split], y_norm[:split]
    X_test, y_test = X_norm[split:], y_norm[split:]
    print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # ------------------------------------------------------------------
    # Optimise
    # ------------------------------------------------------------------
    print("\n--- Optimising quantum weights ---")
    weights = optimize_quantum_weights(
        X_train, y_train,
        n_epochs=N_EPOCHS,
        learning_rate=LR,
    )

    # ------------------------------------------------------------------
    # Evaluate on test set
    # ------------------------------------------------------------------
    test_preds = np.array([quantum_circuit(x, weights) for x in X_test])
    test_mse = np.mean((test_preds - y_test) ** 2)
    print(f"\nTest MSE (normalised): {test_mse:.6f}")

    # ------------------------------------------------------------------
    # Autoregressive future prediction
    # ------------------------------------------------------------------
    print("\n--- Autoregressive future prediction (5 steps) ---")
    future = quantum_predict_future(
        weights,
        last_window=X_test[-1],
        n_steps=5,
        target_min=t_min,
        target_max=t_max,
    )
    print(f"Predicted future prices: {future}")

    print("\nDone.")


if __name__ == "__main__":
    main()
