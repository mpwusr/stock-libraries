"""
pandas_finance.py — pandas for Financial Data Processing
=========================================================

Reference implementation showing pandas patterns commonly used in
financial data pipelines: DataFrame construction, groupby aggregation,
time resampling, rolling windows, returns calculation, CSV export
with timestamps, and training log DataFrames for ML experiments.

Patterns from: StockLTSMTransformerQuantum project

Dependencies:
    pip install pandas yfinance

Usage:
    python pandas_finance.py
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime


# ---------------------------------------------------------------------------
# 1. DataFrame Construction from yfinance Data
# ---------------------------------------------------------------------------
def construct_dataframe(
    tickers: list[str] | None = None,
    period: str = "2y",
) -> dict[str, pd.DataFrame]:
    """
    Download data for multiple tickers and construct individual DataFrames.

    Parameters
    ----------
    tickers : list[str]
        List of Yahoo Finance symbols.
    period : str
        Download period.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of ticker → OHLCV DataFrame.
    """
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOGL"]

    print("--- DataFrame Construction ---")

    # Download all tickers at once (multi-ticker mode)
    raw = yf.download(tickers, period=period, auto_adjust=False)

    ticker_frames: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        # Extract single-ticker data from the MultiIndex columns
        # raw["Close"]["AAPL"], raw["Volume"]["AAPL"], etc.
        df = pd.DataFrame({
            "Open": raw["Open"][ticker],
            "High": raw["High"][ticker],
            "Low": raw["Low"][ticker],
            "Close": raw["Close"][ticker],
            "Volume": raw["Volume"][ticker],
        })

        # Add derived columns
        df["Ticker"] = ticker
        df["Daily_Range"] = df["High"] - df["Low"]
        df["Mid_Price"] = (df["High"] + df["Low"]) / 2

        ticker_frames[ticker] = df
        print(f"  {ticker}: {len(df)} rows, "
              f"{df.index.min().date()} → {df.index.max().date()}")

    return ticker_frames


# ---------------------------------------------------------------------------
# 2. .groupby() and .agg() — Summary Statistics
# ---------------------------------------------------------------------------
def groupby_summary(ticker_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Combine multiple ticker DataFrames and compute summary statistics
    using groupby + agg.

    Parameters
    ----------
    ticker_frames : dict[str, pd.DataFrame]
        Output from construct_dataframe().

    Returns
    -------
    pd.DataFrame
        Per-ticker summary with mean close, total volume, etc.
    """
    # Stack all tickers into one long DataFrame
    combined = pd.concat(ticker_frames.values(), axis=0)

    print("\n--- GroupBy Summary Statistics ---")
    print(f"  Combined shape: {combined.shape}")

    # Compute multiple aggregations in one pass
    summary = combined.groupby("Ticker").agg(
        mean_close=("Close", "mean"),
        std_close=("Close", "std"),
        min_close=("Close", "min"),
        max_close=("Close", "max"),
        total_volume=("Volume", "sum"),
        avg_daily_range=("Daily_Range", "mean"),
        trading_days=("Close", "count"),
    )

    # Round for readability
    summary = summary.round(2)

    print(summary)
    return summary


# ---------------------------------------------------------------------------
# 3. .resample() — Monthly Aggregation
# ---------------------------------------------------------------------------
def resample_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily data to monthly frequency using .resample().

    Common aggregation patterns:
      - 'M'  → calendar month end
      - 'MS' → calendar month start
      - 'W'  → weekly
      - 'Q'  → quarterly

    Parameters
    ----------
    df : pd.DataFrame
        Daily OHLCV DataFrame with a DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Monthly aggregated data.
    """
    print("\n--- Monthly Resampling ---")

    # Multiple aggregation methods shown
    monthly = df.resample("M").agg({
        "Open": "first",       # First trading day's open
        "High": "max",         # Monthly high
        "Low": "min",          # Monthly low
        "Close": "last",       # Last trading day's close
        "Volume": "sum",       # Total monthly volume
    })

    # Also compute monthly mean close for reference
    monthly["Mean_Close"] = df["Close"].resample("M").mean()

    print(f"  Daily rows    : {len(df)}")
    print(f"  Monthly rows  : {len(monthly)}")
    print(monthly.tail(6))

    return monthly


# ---------------------------------------------------------------------------
# 4. .rolling() — Moving Averages
# ---------------------------------------------------------------------------
def rolling_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling window calculations to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Daily OHLCV DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with rolling average columns added.
    """
    print("\n--- Rolling Window Calculations ---")

    # Simple Moving Average (SMA) using .rolling().mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()

    # Rolling standard deviation (volatility proxy)
    df["Volatility_20"] = df["Close"].rolling(window=20).std()

    # Rolling min/max (e.g., for Donchian channels)
    df["Rolling_High_20"] = df["High"].rolling(window=20).max()
    df["Rolling_Low_20"] = df["Low"].rolling(window=20).min()

    # Rolling volume average (useful for volume spike detection)
    df["Avg_Volume_20"] = df["Volume"].rolling(window=20).mean()

    print(f"  Added columns: SMA_20, SMA_50, Volatility_20, "
          f"Rolling_High_20, Rolling_Low_20, Avg_Volume_20")
    print(df[["Close", "SMA_20", "SMA_50", "Volatility_20"]].tail(5))

    return df


# ---------------------------------------------------------------------------
# 5. .pct_change() — Returns Calculation
# ---------------------------------------------------------------------------
def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute various return metrics using .pct_change().

    Parameters
    ----------
    df : pd.DataFrame
        Daily OHLCV DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with return columns added.
    """
    print("\n--- Returns Calculation ---")

    # Daily percentage returns
    df["Daily_Return"] = df["Close"].pct_change()

    # Cumulative returns (growth of $1 invested)
    df["Cumulative_Return"] = (1 + df["Daily_Return"]).cumprod() - 1

    # 5-day rolling return
    df["Return_5d"] = df["Close"].pct_change(periods=5)

    # 20-day rolling return
    df["Return_20d"] = df["Close"].pct_change(periods=20)

    # Log returns (preferred for statistical modeling)
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    # Summary statistics
    daily_ret = df["Daily_Return"].dropna()
    print(f"  Mean daily return    : {daily_ret.mean():.6f}")
    print(f"  Std daily return     : {daily_ret.std():.6f}")
    print(f"  Annualized return    : {daily_ret.mean() * 252:.4f}")
    print(f"  Annualized volatility: {daily_ret.std() * np.sqrt(252):.4f}")
    print(f"  Sharpe ratio (est.)  : "
          f"{(daily_ret.mean() * 252) / (daily_ret.std() * np.sqrt(252)):.4f}")
    print(f"  Cumulative return    : {df['Cumulative_Return'].iloc[-1]:.4f}")

    return df


# ---------------------------------------------------------------------------
# 6. CSV Export with Timestamps
# ---------------------------------------------------------------------------
def export_to_csv(
    df: pd.DataFrame,
    ticker: str = "AAPL",
    export_dir: str = "exports",
) -> str:
    """
    Export DataFrame to CSV with a timestamp in the filename.

    File naming pattern: exports/{ticker}_{YYYYMMDD_HHMMSS}.csv
    This prevents overwriting previous exports and creates a clear
    audit trail of data snapshots.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to export.
    ticker : str
        Ticker symbol (used in filename).
    export_dir : str
        Output directory (created if it does not exist).

    Returns
    -------
    str
        Path to the exported CSV file.
    """
    print("\n--- CSV Export ---")

    # Ensure the export directory exists
    os.makedirs(export_dir, exist_ok=True)

    # Build filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ticker}_{timestamp}.csv"
    filepath = os.path.join(export_dir, filename)

    # Export — include the index (dates) and use ISO format
    df.to_csv(filepath, index=True, date_format="%Y-%m-%d")

    file_size_kb = os.path.getsize(filepath) / 1024
    print(f"  Exported to : {filepath}")
    print(f"  File size   : {file_size_kb:.1f} KB")
    print(f"  Rows        : {len(df)}")
    print(f"  Columns     : {list(df.columns)}")

    return filepath


# ---------------------------------------------------------------------------
# 7. Training Log DataFrame — tracking ML experiment metrics
# ---------------------------------------------------------------------------
def create_training_log(n_epochs: int = 50) -> pd.DataFrame:
    """
    Create a simulated training log DataFrame.

    In the StockLTSM project, each training run produces a DataFrame
    of per-epoch metrics that is used for loss curve visualization
    and early stopping analysis.

    Parameters
    ----------
    n_epochs : int
        Number of training epochs to simulate.

    Returns
    -------
    pd.DataFrame
        Training log with epoch-level metrics.
    """
    print("\n--- Training Log DataFrame ---")

    # Simulate typical training metrics (loss decreases, then plateaus)
    np.random.seed(42)
    epochs = list(range(1, n_epochs + 1))

    # Exponential decay + noise for realistic loss curves
    base_train_loss = 0.5 * np.exp(-0.05 * np.array(epochs)) + 0.02
    train_loss = base_train_loss + np.random.normal(0, 0.005, n_epochs)
    train_loss = np.clip(train_loss, 0.01, None)  # Floor at 0.01

    # Validation loss: similar but slightly higher and more noisy
    base_val_loss = 0.55 * np.exp(-0.04 * np.array(epochs)) + 0.03
    val_loss = base_val_loss + np.random.normal(0, 0.008, n_epochs)
    val_loss = np.clip(val_loss, 0.01, None)

    # Learning rate schedule (step decay every 15 epochs)
    lr = [0.001 * (0.5 ** (e // 15)) for e in range(n_epochs)]

    # Additional metrics
    train_mae = train_loss * 1.2 + np.random.normal(0, 0.003, n_epochs)
    val_mae = val_loss * 1.3 + np.random.normal(0, 0.005, n_epochs)

    # Build the training log DataFrame
    log = pd.DataFrame({
        "epoch": epochs,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_mae": np.abs(train_mae),
        "val_mae": np.abs(val_mae),
        "learning_rate": lr,
        "duration_sec": np.random.uniform(12, 18, n_epochs),
    })

    # Round for readability
    log = log.round(6)

    # Add cumulative training time
    log["cumulative_time_min"] = (log["duration_sec"].cumsum() / 60).round(2)

    # Best epoch tracking
    best_epoch = log["val_loss"].idxmin()
    log["is_best"] = False
    log.loc[best_epoch, "is_best"] = True

    print(f"  Epochs       : {n_epochs}")
    print(f"  Best epoch   : {best_epoch + 1} "
          f"(val_loss = {log.loc[best_epoch, 'val_loss']:.6f})")
    print(f"  Final lr     : {log['learning_rate'].iloc[-1]}")
    print(f"  Total time   : {log['cumulative_time_min'].iloc[-1]:.1f} min")
    print(log.head(10))

    return log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Run all pandas financial data examples."""
    print("=" * 70)
    print("pandas Financial Data Processing Examples")
    print("=" * 70)

    # 1. Construct DataFrames from yfinance
    ticker_frames = construct_dataframe(["AAPL", "MSFT", "GOOGL"], period="2y")

    # 2. GroupBy summary statistics
    summary = groupby_summary(ticker_frames)

    # Use AAPL for the remaining single-ticker demonstrations
    aapl = ticker_frames["AAPL"].copy()

    # 3. Monthly resampling
    monthly = resample_monthly(aapl)

    # 4. Rolling averages
    aapl = rolling_averages(aapl)

    # 5. Returns calculation
    aapl = calculate_returns(aapl)

    # 6. CSV export with timestamp
    export_path = export_to_csv(aapl, ticker="AAPL")

    # 7. Training log DataFrame
    training_log = create_training_log(n_epochs=50)

    # Also export the training log
    log_path = export_to_csv(training_log, ticker="training_log")

    print("\n" + "=" * 70)
    print("All pandas examples completed successfully.")
    print(f"Exported files: {export_path}, {log_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
