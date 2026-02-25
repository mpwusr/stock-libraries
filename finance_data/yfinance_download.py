"""
yfinance_download.py — Yahoo Finance Data Downloading
=====================================================

Reference implementation showing how to download stock market data
using the yfinance library. Covers single-ticker downloads, multi-ticker
batch downloads, ticker metadata retrieval, and DataFrame operations
for column selection and date range handling.

Patterns from: StockLTSMTransformerQuantum project

Dependencies:
    pip install yfinance pandas

Usage:
    python yfinance_download.py
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


# ---------------------------------------------------------------------------
# 1. Single Ticker Download — explicit date range, raw OHLCV + Adj Close
# ---------------------------------------------------------------------------
def download_single_ticker(
    ticker: str = "AAPL",
    start: str = "2020-01-01",
    end: str = "2025-01-01",
) -> pd.DataFrame:
    """
    Download historical daily bars for one ticker.

    Parameters
    ----------
    ticker : str
        Yahoo Finance symbol (e.g. "AAPL", "TSLA").
    start : str
        Start date in YYYY-MM-DD format.
    end : str
        End date in YYYY-MM-DD format (exclusive).

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame indexed by Date.
    """
    # auto_adjust=False keeps both 'Close' and 'Adj Close' columns,
    # which is useful when you need to compare raw vs. split/dividend-
    # adjusted prices.
    data = yf.download(ticker, start=start, end=end, auto_adjust=False)

    print(f"\n--- Single Ticker: {ticker} ---")
    print(f"Date range : {data.index.min().date()} → {data.index.max().date()}")
    print(f"Shape      : {data.shape}")
    print(f"Columns    : {list(data.columns)}")
    print(data.head())

    return data


# ---------------------------------------------------------------------------
# 2. Multi-Ticker Download — batch fetch with a shared date axis
# ---------------------------------------------------------------------------
def download_multi_ticker(
    tickers: list[str] | None = None,
    period: str = "1y",
) -> pd.DataFrame:
    """
    Download data for multiple tickers at once.

    yfinance returns a DataFrame with a MultiIndex on the columns:
    level-0 = price field (Open, High, …), level-1 = ticker symbol.

    Parameters
    ----------
    tickers : list[str]
        List of Yahoo Finance symbols.
    period : str
        Valid yfinance period string ("1d", "5d", "1mo", "3mo",
        "6mo", "1y", "2y", "5y", "10y", "ytd", "max").

    Returns
    -------
    pd.DataFrame
        MultiIndex-columned OHLCV DataFrame.
    """
    if tickers is None:
        tickers = ["AAPL", "MSFT", "GOOGL"]

    # Passing a list triggers multi-ticker mode automatically.
    data = yf.download(tickers, period=period)

    print(f"\n--- Multi-Ticker: {tickers} ---")
    print(f"Period     : {period}")
    print(f"Shape      : {data.shape}")
    # Show the two-level column structure
    print(f"Col levels : {data.columns.names}")
    print(data.head())

    # Accessing a single field across all tickers:
    close_df = data["Close"]  # DataFrame with one column per ticker
    print(f"\nClose prices shape: {close_df.shape}")
    print(close_df.tail(3))

    return data


# ---------------------------------------------------------------------------
# 3. Ticker Info — metadata, sector, market cap, etc.
# ---------------------------------------------------------------------------
def get_ticker_info(symbol: str = "AAPL") -> dict:
    """
    Retrieve company metadata from the Ticker.info property.

    The .info dict contains ~150 keys including shortName, sector,
    industry, marketCap, trailingPE, dividendYield, and more.

    Parameters
    ----------
    symbol : str
        Yahoo Finance ticker symbol.

    Returns
    -------
    dict
        Selected metadata fields.
    """
    ticker = yf.Ticker(symbol)
    info = ticker.info

    # Pick the fields most relevant to the StockLTSM project
    fields_of_interest = [
        "shortName",
        "sector",
        "industry",
        "marketCap",
        "trailingPE",
        "forwardPE",
        "dividendYield",
        "fiftyTwoWeekHigh",
        "fiftyTwoWeekLow",
    ]

    selected = {}
    print(f"\n--- Ticker Info: {symbol} ---")
    for key in fields_of_interest:
        value = info.get(key, "N/A")
        selected[key] = value
        print(f"  {key:24s}: {value}")

    return selected


# ---------------------------------------------------------------------------
# 4. Historical Data via Ticker Object — alternative to yf.download()
# ---------------------------------------------------------------------------
def get_ticker_history(symbol: str = "AAPL", period: str = "6mo") -> pd.DataFrame:
    """
    Fetch history through the Ticker object instead of yf.download().

    Ticker.history() always returns adjusted data and includes
    Dividends and Stock Splits columns by default.

    Parameters
    ----------
    symbol : str
        Yahoo Finance ticker symbol.
    period : str
        Look-back period (e.g. "1mo", "6mo", "1y").

    Returns
    -------
    pd.DataFrame
        Adjusted OHLCV plus Dividends and Stock Splits.
    """
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=period)

    print(f"\n--- Ticker History: {symbol} ({period}) ---")
    print(f"Shape   : {hist.shape}")
    print(f"Columns : {list(hist.columns)}")
    print(hist.head())

    return hist


# ---------------------------------------------------------------------------
# 5. Column Selection & Date Range Handling
# ---------------------------------------------------------------------------
def select_columns_and_filter(
    data: pd.DataFrame,
    columns: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """
    Select specific columns and filter by date range.

    This is a common preprocessing step before feeding data into
    the LSTM / Transformer pipeline.

    Parameters
    ----------
    data : pd.DataFrame
        Raw OHLCV DataFrame from yf.download().
    columns : list[str]
        Columns to keep (default: Close, Volume, High, Low, Open).
    start_date : str or None
        Optional start date filter (inclusive).
    end_date : str or None
        Optional end date filter (inclusive).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with selected columns.
    """
    if columns is None:
        columns = ["Close", "Volume", "High", "Low", "Open"]

    # --- Column selection ---
    # Only keep columns that actually exist in the DataFrame.
    available = [c for c in columns if c in data.columns]
    subset = data[available].copy()

    # --- Date range filtering ---
    if start_date is not None:
        subset = subset.loc[start_date:]
    if end_date is not None:
        subset = subset.loc[:end_date]

    print(f"\n--- Column Selection & Date Filter ---")
    print(f"Requested columns : {columns}")
    print(f"Available columns : {available}")
    print(f"Date filter       : {start_date} → {end_date}")
    print(f"Result shape      : {subset.shape}")
    print(subset.head())

    return subset


# ---------------------------------------------------------------------------
# 6. Date Range Utility — generate dynamic windows
# ---------------------------------------------------------------------------
def dynamic_date_range(lookback_days: int = 365) -> tuple[str, str]:
    """
    Compute a (start, end) date pair relative to today.

    Useful for scripts that always need "the last N days" without
    hard-coding dates.

    Parameters
    ----------
    lookback_days : int
        Number of calendar days to look back from today.

    Returns
    -------
    tuple[str, str]
        (start_date, end_date) in YYYY-MM-DD format.
    """
    end = datetime.now()
    start = end - timedelta(days=lookback_days)

    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    print(f"\n--- Dynamic Date Range ---")
    print(f"Lookback  : {lookback_days} days")
    print(f"Start     : {start_str}")
    print(f"End       : {end_str}")

    return start_str, end_str


# ---------------------------------------------------------------------------
# Main — run all examples
# ---------------------------------------------------------------------------
def main():
    """Execute all download examples sequentially."""
    print("=" * 70)
    print("yfinance Download Examples")
    print("=" * 70)

    # 1. Single ticker with explicit date range
    aapl = download_single_ticker("AAPL", "2020-01-01", "2025-01-01")

    # 2. Multi-ticker batch download
    multi = download_multi_ticker(["AAPL", "MSFT", "GOOGL"], period="1y")

    # 3. Company metadata
    info = get_ticker_info("AAPL")

    # 4. History via Ticker object
    hist = get_ticker_history("AAPL", period="6mo")

    # 5. Column selection and date filtering
    selected = select_columns_and_filter(
        aapl,
        columns=["Close", "Volume", "High", "Low", "Open"],
        start_date="2023-01-01",
        end_date="2024-06-30",
    )

    # 6. Dynamic date range
    start, end = dynamic_date_range(lookback_days=730)

    print("\n" + "=" * 70)
    print("All examples completed successfully.")
    print("=" * 70)


if __name__ == "__main__":
    main()
