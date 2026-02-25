"""
matplotlib_finance.py — matplotlib for Stock Charts
====================================================

Reference implementation showing matplotlib patterns for financial
data visualization: multi-subplot layouts, price charts with SMA
overlays, dual y-axis volume plots, training loss curves, forecast
charts with confidence bands, and proper date formatting.

Patterns from: StockLTSMTransformerQuantum project

Dependencies:
    pip install matplotlib pandas numpy yfinance ta

Usage:
    python matplotlib_finance.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator


# ---------------------------------------------------------------------------
# 1. Multi-Subplot Stock Chart (Price + RSI + Volume)
# ---------------------------------------------------------------------------
def plot_stock_chart(
    df: pd.DataFrame,
    ticker: str = "AAPL",
    save_path: str | None = None,
) -> None:
    """
    Create a 3-row subplot figure: price with SMA overlays, RSI, and volume.

    Layout
    ------
    Row 1 (60%): Closing price + SMA_20 + SMA_50
    Row 2 (20%): RSI with overbought/oversold zones
    Row 3 (20%): Volume bars with dual y-axis

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame with DatetimeIndex. SMA and RSI columns
        will be computed if not present.
    ticker : str
        Ticker symbol for the chart title.
    save_path : str or None
        If provided, save the figure to this path.
    """
    # Ensure we have the required indicator columns
    if "SMA_20" not in df.columns:
        sma20 = SMAIndicator(close=df["Close"], window=20)
        df["SMA_20"] = sma20.sma_indicator()
    if "SMA_50" not in df.columns:
        sma50 = SMAIndicator(close=df["Close"], window=50)
        df["SMA_50"] = sma50.sma_indicator()
    if "RSI" not in df.columns:
        rsi = RSIIndicator(close=df["Close"], window=14)
        df["RSI"] = rsi.rsi()

    # --- Create the figure with 3 subplots sharing the x-axis ---
    # gridspec_kw controls the relative heights of each subplot
    fig, axes = plt.subplots(
        3, 1,
        figsize=(14, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1, 1]},
    )

    ax_price, ax_rsi, ax_volume = axes

    # ---- Row 1: Price Chart with SMA Overlays ----
    ax_price.plot(df.index, df["Close"], label="Close", color="#2196F3",
                  linewidth=1.2, alpha=0.9)
    ax_price.plot(df.index, df["SMA_20"], label="SMA 20", color="#FF9800",
                  linewidth=1.0, linestyle="--", alpha=0.8)
    ax_price.plot(df.index, df["SMA_50"], label="SMA 50", color="#E91E63",
                  linewidth=1.0, linestyle="--", alpha=0.8)

    ax_price.set_title(f"{ticker} — Price, RSI & Volume", fontsize=14,
                       fontweight="bold")
    ax_price.set_ylabel("Price ($)", fontsize=11)
    ax_price.legend(loc="upper left", fontsize=9)
    ax_price.grid(True, alpha=0.3)

    # Fill between SMA_20 and SMA_50 to highlight crossover regions
    ax_price.fill_between(
        df.index,
        df["SMA_20"], df["SMA_50"],
        where=(df["SMA_20"] > df["SMA_50"]),
        alpha=0.1, color="green", label="_bullish",
    )
    ax_price.fill_between(
        df.index,
        df["SMA_20"], df["SMA_50"],
        where=(df["SMA_20"] < df["SMA_50"]),
        alpha=0.1, color="red", label="_bearish",
    )

    # ---- Row 2: RSI ----
    ax_rsi.plot(df.index, df["RSI"], color="#9C27B0", linewidth=1.0)
    ax_rsi.axhline(y=70, color="red", linestyle="--", alpha=0.5,
                   label="Overbought (70)")
    ax_rsi.axhline(y=30, color="green", linestyle="--", alpha=0.5,
                   label="Oversold (30)")
    ax_rsi.fill_between(df.index, 70, 100, alpha=0.05, color="red")
    ax_rsi.fill_between(df.index, 0, 30, alpha=0.05, color="green")
    ax_rsi.set_ylabel("RSI", fontsize=11)
    ax_rsi.set_ylim(0, 100)
    ax_rsi.legend(loc="upper left", fontsize=8)
    ax_rsi.grid(True, alpha=0.3)

    # ---- Row 3: Volume Bars with Dual Y-Axis ----
    # Color volume bars green/red based on price direction
    colors = [
        "#4CAF50" if c >= o else "#F44336"
        for c, o in zip(df["Close"], df["Open"])
    ]
    ax_volume.bar(df.index, df["Volume"], color=colors, alpha=0.6, width=1.0)
    ax_volume.set_ylabel("Volume", fontsize=11)
    ax_volume.grid(True, alpha=0.3)

    # --- Dual y-axis: overlay closing price on the volume chart ---
    ax_price_overlay = ax_volume.twinx()
    ax_price_overlay.plot(df.index, df["Close"], color="#2196F3",
                          linewidth=0.8, alpha=0.5)
    ax_price_overlay.set_ylabel("Close ($)", fontsize=9, color="#2196F3")
    ax_price_overlay.tick_params(axis="y", labelcolor="#2196F3")

    # ---- Date Formatting ----
    # Use matplotlib.dates for clean x-axis labels
    ax_volume.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax_volume.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax_volume.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # ---- Finalize ----
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved stock chart to: {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# 2. Training Loss Curves (Train vs Validation)
# ---------------------------------------------------------------------------
def plot_training_curves(
    log: pd.DataFrame,
    save_path: str | None = None,
) -> None:
    """
    Plot training and validation loss curves from an epoch-level log.

    This is the standard visualization for monitoring model convergence
    and detecting overfitting in the LSTM/Transformer training loop.

    Parameters
    ----------
    log : pd.DataFrame
        Must contain 'epoch', 'train_loss', 'val_loss' columns.
        Optionally 'train_mae', 'val_mae', 'learning_rate'.
    save_path : str or None
        If provided, save the figure to this path.
    """
    fig, axes = plt.subplots(
        3, 1,
        figsize=(14, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 2, 1]},
    )

    ax_loss, ax_mae, ax_lr = axes

    epochs = log["epoch"]

    # ---- Row 1: Loss Curves ----
    ax_loss.plot(epochs, log["train_loss"], label="Train Loss",
                 color="#2196F3", linewidth=1.5, marker="o", markersize=3)
    ax_loss.plot(epochs, log["val_loss"], label="Val Loss",
                 color="#F44336", linewidth=1.5, marker="s", markersize=3)

    # Mark the best epoch
    if "is_best" in log.columns:
        best_row = log[log["is_best"]]
        if not best_row.empty:
            ax_loss.axvline(x=best_row["epoch"].iloc[0], color="green",
                            linestyle=":", alpha=0.7, label="Best Epoch")
            ax_loss.scatter(
                best_row["epoch"].iloc[0], best_row["val_loss"].iloc[0],
                color="green", s=100, zorder=5, marker="*",
            )

    # Fill the gap between train and val loss to highlight overfitting
    ax_loss.fill_between(
        epochs, log["train_loss"], log["val_loss"],
        alpha=0.1, color="orange", label="Generalization Gap",
    )

    ax_loss.set_title("Training Progress", fontsize=14, fontweight="bold")
    ax_loss.set_ylabel("Loss (MSE)", fontsize=11)
    ax_loss.legend(loc="upper right", fontsize=9)
    ax_loss.grid(True, alpha=0.3)

    # ---- Row 2: MAE Curves ----
    if "train_mae" in log.columns:
        ax_mae.plot(epochs, log["train_mae"], label="Train MAE",
                    color="#2196F3", linewidth=1.2)
        ax_mae.plot(epochs, log["val_mae"], label="Val MAE",
                    color="#F44336", linewidth=1.2)
        ax_mae.set_ylabel("MAE", fontsize=11)
        ax_mae.legend(loc="upper right", fontsize=9)
        ax_mae.grid(True, alpha=0.3)

    # ---- Row 3: Learning Rate Schedule ----
    if "learning_rate" in log.columns:
        ax_lr.plot(epochs, log["learning_rate"], color="#4CAF50",
                   linewidth=1.5, drawstyle="steps-post")
        ax_lr.set_ylabel("Learning Rate", fontsize=11)
        ax_lr.set_xlabel("Epoch", fontsize=11)
        ax_lr.set_yscale("log")
        ax_lr.grid(True, alpha=0.3)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved training curves to: {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# 3. Forecast Chart with Prediction + Confidence Bands
# ---------------------------------------------------------------------------
def plot_forecast(
    actual: pd.Series,
    predicted: pd.Series,
    forecast_dates: pd.DatetimeIndex | None = None,
    forecast_values: np.ndarray | None = None,
    confidence_level: float = 0.95,
    ticker: str = "AAPL",
    save_path: str | None = None,
) -> None:
    """
    Plot actual vs predicted prices with a forward-looking forecast
    and confidence bands.

    Parameters
    ----------
    actual : pd.Series
        Actual closing prices with DatetimeIndex.
    predicted : pd.Series
        Model predictions aligned with actual dates.
    forecast_dates : pd.DatetimeIndex or None
        Future dates for the forward forecast.
    forecast_values : np.ndarray or None
        Point forecast values for future dates.
    confidence_level : float
        Confidence level for the prediction interval (default 0.95).
    ticker : str
        Ticker symbol for the chart title.
    save_path : str or None
        If provided, save the figure to this path.
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    # ---- Actual prices ----
    ax.plot(actual.index, actual.values, label="Actual",
            color="#2196F3", linewidth=1.5, alpha=0.9)

    # ---- In-sample predictions ----
    ax.plot(predicted.index, predicted.values, label="Predicted",
            color="#FF9800", linewidth=1.2, linestyle="--", alpha=0.8)

    # ---- Forward forecast with confidence bands ----
    if forecast_dates is not None and forecast_values is not None:
        ax.plot(forecast_dates, forecast_values, label="Forecast",
                color="#4CAF50", linewidth=2.0, marker="o", markersize=4)

        # Compute expanding confidence bands
        # Width grows with sqrt(time) — a simplified random walk assumption
        n_forecast = len(forecast_values)
        residuals = (actual.values[-len(predicted):] - predicted.values).std()
        z_score = 1.96 if confidence_level == 0.95 else 1.645

        band_width = z_score * residuals * np.sqrt(np.arange(1, n_forecast + 1))

        upper = forecast_values + band_width
        lower = forecast_values - band_width

        ax.fill_between(
            forecast_dates, lower, upper,
            alpha=0.15, color="#4CAF50",
            label=f"{confidence_level:.0%} Confidence Band",
        )

        # Vertical line separating historical data from forecast
        ax.axvline(
            x=actual.index[-1], color="gray", linestyle=":",
            alpha=0.5, label="Forecast Start",
        )

    # ---- Formatting ----
    ax.set_title(f"{ticker} — Price Forecast", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Price ($)", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Date formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved forecast chart to: {save_path}")

    plt.show()


# ---------------------------------------------------------------------------
# Helper: Generate Simulated Training Log
# ---------------------------------------------------------------------------
def _generate_training_log(n_epochs: int = 50) -> pd.DataFrame:
    """Generate a realistic-looking training log for demo purposes."""
    np.random.seed(42)
    epochs = list(range(1, n_epochs + 1))

    base_train = 0.5 * np.exp(-0.05 * np.array(epochs)) + 0.02
    train_loss = base_train + np.random.normal(0, 0.005, n_epochs)

    base_val = 0.55 * np.exp(-0.04 * np.array(epochs)) + 0.03
    val_loss = base_val + np.random.normal(0, 0.008, n_epochs)

    lr = [0.001 * (0.5 ** (e // 15)) for e in range(n_epochs)]

    log = pd.DataFrame({
        "epoch": epochs,
        "train_loss": np.clip(train_loss, 0.01, None),
        "val_loss": np.clip(val_loss, 0.01, None),
        "train_mae": np.abs(train_loss * 1.2 + np.random.normal(0, 0.003, n_epochs)),
        "val_mae": np.abs(val_loss * 1.3 + np.random.normal(0, 0.005, n_epochs)),
        "learning_rate": lr,
    })

    best_idx = log["val_loss"].idxmin()
    log["is_best"] = False
    log.loc[best_idx, "is_best"] = True

    return log


# ---------------------------------------------------------------------------
# Helper: Generate Simulated Forecast Data
# ---------------------------------------------------------------------------
def _generate_forecast_data(
    actual: pd.Series,
    n_forecast_days: int = 30,
) -> tuple[pd.Series, pd.DatetimeIndex, np.ndarray]:
    """Generate simulated predictions and a forward forecast."""
    np.random.seed(123)

    # In-sample "predictions" (actual + noise to simulate model output)
    noise = np.random.normal(0, actual.std() * 0.03, len(actual))
    predicted = pd.Series(actual.values + noise, index=actual.index)

    # Forward forecast dates (business days only)
    last_date = actual.index[-1]
    forecast_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1),
        periods=n_forecast_days,
    )

    # Forecast values: simple drift + noise
    last_price = actual.iloc[-1]
    daily_drift = actual.pct_change().mean()
    forecast_values = np.zeros(n_forecast_days)
    forecast_values[0] = last_price * (1 + daily_drift)
    for i in range(1, n_forecast_days):
        forecast_values[i] = forecast_values[i - 1] * (
            1 + daily_drift + np.random.normal(0, 0.01)
        )

    return predicted, forecast_dates, forecast_values


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Run all matplotlib visualization examples."""
    print("=" * 70)
    print("matplotlib Financial Visualization Examples")
    print("=" * 70)

    # Download data for demonstrations
    print("\nDownloading AAPL data...")
    df = yf.download("AAPL", period="1y", auto_adjust=False)

    # Flatten MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 1. Multi-subplot stock chart
    print("\n1. Plotting stock chart (Price + RSI + Volume)...")
    plot_stock_chart(df.copy(), ticker="AAPL", save_path="stock_chart.png")

    # 2. Training loss curves
    print("\n2. Plotting training curves...")
    training_log = _generate_training_log(n_epochs=50)
    plot_training_curves(training_log, save_path="training_curves.png")

    # 3. Forecast chart
    print("\n3. Plotting forecast chart...")
    actual = df["Close"].copy()
    predicted, forecast_dates, forecast_values = _generate_forecast_data(actual)
    plot_forecast(
        actual=actual,
        predicted=predicted,
        forecast_dates=forecast_dates,
        forecast_values=forecast_values,
        ticker="AAPL",
        save_path="forecast_chart.png",
    )

    print("\n" + "=" * 70)
    print("All charts generated. Check .png files in the current directory.")
    print("=" * 70)


if __name__ == "__main__":
    main()
