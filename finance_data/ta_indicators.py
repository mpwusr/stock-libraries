"""
ta_indicators.py — Technical Analysis Indicators
=================================================

Reference implementation showing how to compute common technical
indicators using the `ta` (Technical Analysis) library and combine
them into actionable buy/sell signals.

Patterns from: StockLTSMTransformerQuantum project

Dependencies:
    pip install ta yfinance pandas

Usage:
    python ta_indicators.py
"""

import pandas as pd
import yfinance as yf
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands


# ---------------------------------------------------------------------------
# 1. Simple Moving Averages (SMA) — short-term and long-term
# ---------------------------------------------------------------------------
def add_sma(df: pd.DataFrame, close_col: str = "Close") -> pd.DataFrame:
    """
    Add 20-day and 50-day Simple Moving Averages.

    SMA is the arithmetic mean of the closing price over the last N days.
    A "golden cross" (SMA_20 crossing above SMA_50) is a classic bullish
    signal; a "death cross" (below) is bearish.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``close_col`` column.
    close_col : str
        Name of the closing-price column.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with SMA_20 and SMA_50 columns added.
    """
    close = df[close_col]

    # 20-day short-term SMA
    sma_20 = SMAIndicator(close=close, window=20)
    df["SMA_20"] = sma_20.sma_indicator()

    # 50-day longer-term SMA
    sma_50 = SMAIndicator(close=close, window=50)
    df["SMA_50"] = sma_50.sma_indicator()

    print("--- SMA Indicators ---")
    print(f"  SMA_20 NaN count: {df['SMA_20'].isna().sum()} (first 19 rows)")
    print(f"  SMA_50 NaN count: {df['SMA_50'].isna().sum()} (first 49 rows)")
    print(df[["Close", "SMA_20", "SMA_50"]].tail(5))

    return df


# ---------------------------------------------------------------------------
# 2. Relative Strength Index (RSI) — momentum oscillator
# ---------------------------------------------------------------------------
def add_rsi(df: pd.DataFrame, close_col: str = "Close", window: int = 14) -> pd.DataFrame:
    """
    Add RSI (Relative Strength Index) with a 14-day window.

    RSI oscillates between 0 and 100:
      - RSI > 70 → overbought (potential sell)
      - RSI < 30 → oversold  (potential buy)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``close_col`` column.
    close_col : str
        Name of the closing-price column.
    window : int
        RSI look-back window (default 14).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with RSI column added.
    """
    close = df[close_col]

    rsi = RSIIndicator(close=close, window=window)
    df["RSI"] = rsi.rsi()

    print("\n--- RSI Indicator ---")
    print(f"  Window        : {window}")
    print(f"  Current RSI   : {df['RSI'].iloc[-1]:.2f}")
    print(f"  RSI > 70 days : {(df['RSI'] > 70).sum()}")
    print(f"  RSI < 30 days : {(df['RSI'] < 30).sum()}")

    return df


# ---------------------------------------------------------------------------
# 3. MACD — Moving Average Convergence Divergence
# ---------------------------------------------------------------------------
def add_macd(df: pd.DataFrame, close_col: str = "Close") -> pd.DataFrame:
    """
    Add MACD line, signal line, and histogram (diff).

    MACD = EMA_12 - EMA_26
    Signal = EMA_9(MACD)
    Histogram = MACD - Signal

    Buy when MACD crosses above signal; sell when it crosses below.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``close_col`` column.
    close_col : str
        Name of the closing-price column.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with MACD, MACD_Signal, MACD_Hist columns.
    """
    close = df[close_col]

    macd = MACD(close=close)

    # The three MACD components
    df["MACD"] = macd.macd()                # MACD line
    df["MACD_Signal"] = macd.macd_signal()  # Signal line
    df["MACD_Hist"] = macd.macd_diff()      # Histogram (MACD - Signal)

    print("\n--- MACD Indicator ---")
    print(df[["Close", "MACD", "MACD_Signal", "MACD_Hist"]].tail(5))

    return df


# ---------------------------------------------------------------------------
# 4. Bollinger Bands — volatility envelope around SMA
# ---------------------------------------------------------------------------
def add_bollinger_bands(
    df: pd.DataFrame,
    close_col: str = "Close",
    window: int = 20,
    window_dev: int = 2,
) -> pd.DataFrame:
    """
    Add Bollinger Bands (upper, middle, lower) and bandwidth.

    Bands widen during high volatility and narrow during low volatility.
    Price touching the upper band may indicate overbought; lower band
    may indicate oversold.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``close_col`` column.
    close_col : str
        Name of the closing-price column.
    window : int
        SMA window for the middle band (default 20).
    window_dev : int
        Number of standard deviations for upper/lower bands (default 2).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with BB_Upper, BB_Middle, BB_Lower, BB_Width.
    """
    close = df[close_col]

    bb = BollingerBands(close=close, window=window, window_dev=window_dev)

    df["BB_Upper"] = bb.bollinger_hband()       # Upper band
    df["BB_Middle"] = bb.bollinger_mavg()        # Middle band (SMA)
    df["BB_Lower"] = bb.bollinger_lband()        # Lower band
    df["BB_Width"] = bb.bollinger_wband()        # Bandwidth (normalized)

    print("\n--- Bollinger Bands ---")
    print(f"  Window     : {window}")
    print(f"  Std Dev    : {window_dev}")
    print(df[["Close", "BB_Upper", "BB_Middle", "BB_Lower"]].tail(5))

    return df


# ---------------------------------------------------------------------------
# 5. Composite Buy / Sell Signal Generation
# ---------------------------------------------------------------------------
def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine multiple indicators into a single buy/sell signal column.

    Signal Logic
    -------------
    BUY  (+1) when ALL of:
      - SMA_20 > SMA_50          (short-term trend is up)
      - RSI < 40                  (not overbought, room to run)
      - MACD_Hist > 0             (MACD above signal line)
      - Close < BB_Middle         (price near lower band region)

    SELL (-1) when ALL of:
      - SMA_20 < SMA_50          (short-term trend is down)
      - RSI > 60                  (approaching overbought)
      - MACD_Hist < 0             (MACD below signal line)
      - Close > BB_Middle         (price near upper band region)

    HOLD (0) otherwise.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with all indicators already added.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with a ``Signal`` column (-1, 0, +1).
    """
    # Initialize signal column to HOLD (0)
    df["Signal"] = 0

    # --- Buy conditions ---
    buy_mask = (
        (df["SMA_20"] > df["SMA_50"])
        & (df["RSI"] < 40)
        & (df["MACD_Hist"] > 0)
        & (df["Close"] < df["BB_Middle"])
    )
    df.loc[buy_mask, "Signal"] = 1

    # --- Sell conditions ---
    sell_mask = (
        (df["SMA_20"] < df["SMA_50"])
        & (df["RSI"] > 60)
        & (df["MACD_Hist"] < 0)
        & (df["Close"] > df["BB_Middle"])
    )
    df.loc[sell_mask, "Signal"] = -1

    # --- Summary ---
    buy_count = (df["Signal"] == 1).sum()
    sell_count = (df["Signal"] == -1).sum()
    hold_count = (df["Signal"] == 0).sum()

    print("\n--- Signal Generation ---")
    print(f"  BUY  signals : {buy_count}")
    print(f"  SELL signals : {sell_count}")
    print(f"  HOLD signals : {hold_count}")

    # Show the most recent signal rows
    recent_signals = df[df["Signal"] != 0].tail(5)
    if not recent_signals.empty:
        print("\n  Most recent non-HOLD signals:")
        print(recent_signals[["Close", "SMA_20", "RSI", "MACD_Hist", "Signal"]])

    return df


# ---------------------------------------------------------------------------
# 6. Full Indicator Pipeline — download data and add all indicators
# ---------------------------------------------------------------------------
def build_indicator_dataframe(
    ticker: str = "AAPL",
    period: str = "2y",
) -> pd.DataFrame:
    """
    End-to-end pipeline: download data → add all indicators → generate signals.

    This mirrors the preprocessing step in the StockLTSM training pipeline
    where raw OHLCV data is enriched with technical features before being
    fed to the model.

    Parameters
    ----------
    ticker : str
        Yahoo Finance symbol.
    period : str
        Download period (e.g. "1y", "2y").

    Returns
    -------
    pd.DataFrame
        Fully enriched DataFrame with indicators and signals.
    """
    print(f"\n{'=' * 60}")
    print(f"Building indicator DataFrame for {ticker} ({period})")
    print(f"{'=' * 60}")

    # Download raw data
    df = yf.download(ticker, period=period, auto_adjust=False)

    # Flatten MultiIndex columns if present (yfinance >= 0.2.31 returns
    # MultiIndex even for single tickers).
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Sequentially add each indicator group
    df = add_sma(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)

    # Drop rows where indicators are NaN (warm-up period)
    initial_len = len(df)
    df = df.dropna()
    print(f"\nDropped {initial_len - len(df)} warm-up rows (NaN from indicators)")
    print(f"Final DataFrame shape: {df.shape}")

    # Generate composite signals
    df = generate_signals(df)

    # Show all columns in the final DataFrame
    print(f"\nFinal columns: {list(df.columns)}")

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """Run the full indicator pipeline as a demonstration."""
    df = build_indicator_dataframe(ticker="AAPL", period="2y")

    print("\n" + "=" * 60)
    print("Sample output (last 10 rows, selected columns):")
    print("=" * 60)
    display_cols = [
        "Close", "SMA_20", "SMA_50", "RSI",
        "MACD", "BB_Upper", "BB_Lower", "Signal",
    ]
    available = [c for c in display_cols if c in df.columns]
    print(df[available].tail(10).to_string())


if __name__ == "__main__":
    main()
