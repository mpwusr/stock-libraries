"""
alphavantage_api.py — Alpha Vantage & Polygon.io REST APIs
==========================================================

Reference implementation showing how to fetch financial data from
REST APIs (Alpha Vantage and Polygon.io), parse JSON responses into
pandas DataFrames, and handle rate limits and errors gracefully.

Patterns from: StockLTSMTransformerQuantum project

Dependencies:
    pip install requests pandas

Usage:
    # Set your API key(s) as environment variables first:
    export ALPHAVANTAGE_API_KEY="your_key_here"
    export POLYGON_API_KEY="your_key_here"

    python alphavantage_api.py
"""

import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime


# ---------------------------------------------------------------------------
# Alpha Vantage Configuration
# ---------------------------------------------------------------------------
ALPHAVANTAGE_BASE_URL = "https://www.alphavantage.co/query"

# Free tier: 5 API calls per minute, 500 per day
ALPHAVANTAGE_RATE_LIMIT_CALLS = 5
ALPHAVANTAGE_RATE_LIMIT_WINDOW = 60  # seconds


# ---------------------------------------------------------------------------
# 1. Alpha Vantage — TIME_SERIES_DAILY
# ---------------------------------------------------------------------------
def fetch_alphavantage_daily(
    symbol: str = "AAPL",
    outputsize: str = "compact",
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV data from Alpha Vantage TIME_SERIES_DAILY endpoint.

    Parameters
    ----------
    symbol : str
        Stock ticker symbol (e.g. "AAPL", "MSFT").
    outputsize : str
        "compact" → last 100 data points.
        "full"    → up to 20 years of data.
    api_key : str or None
        Alpha Vantage API key. If None, reads from the
        ALPHAVANTAGE_API_KEY environment variable.

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame indexed by date, sorted chronologically.

    Raises
    ------
    ValueError
        If the API key is missing or the API returns an error.
    """
    # --- Resolve API key ---
    if api_key is None:
        api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise ValueError(
            "Alpha Vantage API key not found. Set the ALPHAVANTAGE_API_KEY "
            "environment variable or pass api_key directly."
        )

    # --- Build request parameters ---
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": outputsize,
        "apikey": api_key,
    }

    print(f"\n--- Alpha Vantage: TIME_SERIES_DAILY ({symbol}) ---")
    print(f"  URL        : {ALPHAVANTAGE_BASE_URL}")
    print(f"  Output size: {outputsize}")

    # --- Make the API call ---
    response = requests.get(ALPHAVANTAGE_BASE_URL, params=params, timeout=30)
    response.raise_for_status()  # Raise on HTTP errors (4xx, 5xx)

    data = response.json()

    # --- Error handling ---
    # Alpha Vantage returns errors as JSON with specific keys
    if "Error Message" in data:
        raise ValueError(
            f"Alpha Vantage error for '{symbol}': {data['Error Message']}"
        )

    if "Note" in data:
        # This typically means we've hit the rate limit
        print(f"  WARNING — Rate limit note: {data['Note']}")
        raise ValueError(f"Rate limit hit: {data['Note']}")

    if "Information" in data:
        # Informational message (e.g., premium endpoint on free tier)
        print(f"  INFO: {data['Information']}")

    # --- Parse JSON to DataFrame ---
    time_series_key = "Time Series (Daily)"
    if time_series_key not in data:
        raise ValueError(
            f"Unexpected response structure. Keys: {list(data.keys())}"
        )

    ts = data[time_series_key]

    # Convert nested dict → DataFrame
    # Each key is a date string; each value is a dict of OHLCV fields
    df = pd.DataFrame.from_dict(ts, orient="index")

    # Rename columns from "1. open" → "Open", etc.
    column_map = {
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close",
        "5. volume": "Volume",
    }
    df = df.rename(columns=column_map)

    # Convert string values to numeric types
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Set up proper datetime index
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    df = df.sort_index()  # Chronological order

    print(f"  Records    : {len(df)}")
    print(f"  Date range : {df.index.min().date()} → {df.index.max().date()}")
    print(df.tail(3))

    return df


# ---------------------------------------------------------------------------
# 2. Rate Limit Handler
# ---------------------------------------------------------------------------
class RateLimiter:
    """
    Simple rate limiter for API calls.

    Tracks timestamps of recent calls and sleeps if necessary
    before allowing the next call to proceed.

    Parameters
    ----------
    max_calls : int
        Maximum number of calls allowed in the time window.
    window_seconds : int
        Duration of the rate-limit window in seconds.
    """

    def __init__(self, max_calls: int = 5, window_seconds: int = 60):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self.call_timestamps: list[float] = []

    def wait_if_needed(self) -> None:
        """Block until it is safe to make the next API call."""
        now = time.time()

        # Remove timestamps outside the current window
        self.call_timestamps = [
            ts for ts in self.call_timestamps
            if now - ts < self.window_seconds
        ]

        if len(self.call_timestamps) >= self.max_calls:
            # Calculate how long we need to wait
            oldest = self.call_timestamps[0]
            sleep_time = self.window_seconds - (now - oldest) + 0.5  # 0.5s buffer
            if sleep_time > 0:
                print(f"  [Rate limiter] Sleeping {sleep_time:.1f}s ...")
                time.sleep(sleep_time)

        # Record this call
        self.call_timestamps.append(time.time())


# ---------------------------------------------------------------------------
# 3. Batch Download with Rate Limiting
# ---------------------------------------------------------------------------
def fetch_multiple_tickers_av(
    symbols: list[str],
    api_key: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Download daily data for multiple tickers with rate limiting.

    Parameters
    ----------
    symbols : list[str]
        List of ticker symbols.
    api_key : str or None
        Alpha Vantage API key.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of symbol → OHLCV DataFrame.
    """
    limiter = RateLimiter(
        max_calls=ALPHAVANTAGE_RATE_LIMIT_CALLS,
        window_seconds=ALPHAVANTAGE_RATE_LIMIT_WINDOW,
    )

    results: dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        try:
            limiter.wait_if_needed()
            df = fetch_alphavantage_daily(symbol, api_key=api_key)
            results[symbol] = df
        except ValueError as e:
            print(f"  ERROR fetching {symbol}: {e}")
            continue

    print(f"\n  Successfully fetched: {list(results.keys())}")
    return results


# ---------------------------------------------------------------------------
# 4. Error Handling for Invalid Tickers
# ---------------------------------------------------------------------------
def safe_fetch(symbol: str, api_key: str | None = None) -> pd.DataFrame | None:
    """
    Fetch data with comprehensive error handling.

    Catches network errors, invalid tickers, rate limits, and
    unexpected response formats.

    Parameters
    ----------
    symbol : str
        Ticker symbol to fetch.
    api_key : str or None
        Alpha Vantage API key.

    Returns
    -------
    pd.DataFrame or None
        OHLCV DataFrame, or None if the fetch failed.
    """
    try:
        return fetch_alphavantage_daily(symbol, api_key=api_key)

    except requests.exceptions.Timeout:
        print(f"  TIMEOUT: Request for {symbol} timed out after 30s")
        return None

    except requests.exceptions.ConnectionError:
        print(f"  CONNECTION ERROR: Could not reach Alpha Vantage API")
        return None

    except requests.exceptions.HTTPError as e:
        print(f"  HTTP ERROR for {symbol}: {e}")
        return None

    except ValueError as e:
        # Covers: missing API key, invalid ticker, rate limit, bad response
        print(f"  VALUE ERROR for {symbol}: {e}")
        return None

    except Exception as e:
        print(f"  UNEXPECTED ERROR for {symbol}: {type(e).__name__}: {e}")
        return None


# ---------------------------------------------------------------------------
# 5. Polygon.io API — similar REST pattern
# ---------------------------------------------------------------------------
POLYGON_BASE_URL = "https://api.polygon.io"


def fetch_polygon_daily(
    symbol: str = "AAPL",
    from_date: str = "2024-01-01",
    to_date: str = "2024-12-31",
    api_key: str | None = None,
) -> pd.DataFrame:
    """
    Fetch daily aggregate bars from the Polygon.io REST API.

    The Polygon.io pattern is very similar to Alpha Vantage — a GET
    request with query parameters, returning JSON that we parse into
    a DataFrame.

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    from_date : str
        Start date (YYYY-MM-DD).
    to_date : str
        End date (YYYY-MM-DD).
    api_key : str or None
        Polygon.io API key. If None, reads from POLYGON_API_KEY env var.

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame indexed by date.
    """
    if api_key is None:
        api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        raise ValueError(
            "Polygon API key not found. Set the POLYGON_API_KEY "
            "environment variable or pass api_key directly."
        )

    # Polygon.io endpoint for daily aggregates
    url = (
        f"{POLYGON_BASE_URL}/v2/aggs/ticker/{symbol}/range"
        f"/1/day/{from_date}/{to_date}"
    )
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 5000,
        "apiKey": api_key,
    }

    print(f"\n--- Polygon.io: Daily Aggregates ({symbol}) ---")
    print(f"  Date range: {from_date} → {to_date}")

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    # Check for errors
    if data.get("status") == "ERROR":
        raise ValueError(f"Polygon error: {data.get('error', 'Unknown')}")

    if "results" not in data or not data["results"]:
        raise ValueError(f"No data returned for {symbol}")

    # Parse results array into DataFrame
    # Polygon fields: o=open, h=high, l=low, c=close, v=volume, t=timestamp
    records = data["results"]
    df = pd.DataFrame(records)

    # Rename to standard OHLCV column names
    column_map = {
        "o": "Open",
        "h": "High",
        "l": "Low",
        "c": "Close",
        "v": "Volume",
        "t": "Timestamp",
    }
    df = df.rename(columns=column_map)

    # Convert millisecond timestamps to datetime
    df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms")
    df = df.set_index("Date")

    # Keep only standard OHLCV columns
    keep_cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in keep_cols if c in df.columns]]

    print(f"  Records: {len(df)}")
    print(df.tail(3))

    return df


# ---------------------------------------------------------------------------
# 6. API Response Comparison Helper
# ---------------------------------------------------------------------------
def compare_api_responses(
    av_df: pd.DataFrame | None,
    poly_df: pd.DataFrame | None,
    symbol: str = "AAPL",
) -> None:
    """
    Compare DataFrames from Alpha Vantage and Polygon for the same ticker.

    This is useful for validating data consistency across providers.
    """
    print(f"\n--- API Response Comparison: {symbol} ---")

    if av_df is not None:
        print(f"  Alpha Vantage : {len(av_df)} rows, "
              f"{av_df.index.min().date()} → {av_df.index.max().date()}")
    else:
        print("  Alpha Vantage : no data")

    if poly_df is not None:
        print(f"  Polygon.io    : {len(poly_df)} rows, "
              f"{poly_df.index.min().date()} → {poly_df.index.max().date()}")
    else:
        print("  Polygon.io    : no data")

    if av_df is not None and poly_df is not None:
        # Find overlapping dates
        common_dates = av_df.index.intersection(poly_df.index)
        print(f"  Overlapping   : {len(common_dates)} trading days")

        if len(common_dates) > 0:
            # Compare closing prices on overlapping dates
            av_close = av_df.loc[common_dates, "Close"]
            poly_close = poly_df.loc[common_dates, "Close"]
            diff = (av_close - poly_close).abs()
            print(f"  Max close diff: ${diff.max():.4f}")
            print(f"  Avg close diff: ${diff.mean():.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    """
    Run API examples.

    Note: Requires valid API keys set as environment variables.
    If keys are not set, the demo will show the error handling.
    """
    print("=" * 70)
    print("Alpha Vantage & Polygon.io API Examples")
    print("=" * 70)

    # --- Alpha Vantage ---
    av_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if av_key:
        print("\nAlpha Vantage API key found. Fetching data...")

        # Single ticker fetch
        aapl_av = safe_fetch("AAPL", api_key=av_key)

        # Invalid ticker (demonstrates error handling)
        invalid = safe_fetch("XYZINVALID123", api_key=av_key)

        # Multi-ticker with rate limiting
        # (Commented out to avoid burning API quota in demo)
        # multi = fetch_multiple_tickers_av(["AAPL", "MSFT"], api_key=av_key)
    else:
        print("\nNo ALPHAVANTAGE_API_KEY set — skipping Alpha Vantage examples.")
        print("  Set it with: export ALPHAVANTAGE_API_KEY='your_key_here'")
        aapl_av = None

    # --- Polygon.io ---
    poly_key = os.environ.get("POLYGON_API_KEY")
    if poly_key:
        print("\nPolygon.io API key found. Fetching data...")
        try:
            aapl_poly = fetch_polygon_daily(
                "AAPL", "2024-01-01", "2024-12-31", api_key=poly_key
            )
        except (ValueError, requests.RequestException) as e:
            print(f"  Polygon fetch failed: {e}")
            aapl_poly = None
    else:
        print("\nNo POLYGON_API_KEY set — skipping Polygon.io examples.")
        print("  Set it with: export POLYGON_API_KEY='your_key_here'")
        aapl_poly = None

    # --- Compare ---
    compare_api_responses(aapl_av, aapl_poly, "AAPL")

    print("\n" + "=" * 70)
    print("API examples completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
