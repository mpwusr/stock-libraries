"""
HTTP Requests for Financial APIs with Retry Logic
==================================================
Reference implementation: robust HTTP calls to Polygon.io and
Alpha Vantage using requests + tenacity.

Libraries:
    requests  >= 2.31.0   — HTTP client
    tenacity  >= 8.2.0    — Retry / back-off decorator

Docs:
    https://docs.python-requests.org/
    https://tenacity.readthedocs.io/
"""

import logging
import time
from typing import Any

import requests
from requests.exceptions import ConnectionError, HTTPError, Timeout
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUEST_TIMEOUT = 30  # seconds
RATE_LIMIT_PAUSE = 0.25  # seconds between consecutive calls


# ---------------------------------------------------------------------------
# Generic Retry-Enabled GET
# ---------------------------------------------------------------------------

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((ConnectionError, Timeout, HTTPError)),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def resilient_get(url: str, params: dict | None = None) -> dict[str, Any]:
    """Perform an HTTP GET with exponential-backoff retries.

    Parameters
    ----------
    url : str
        Fully-qualified URL.
    params : dict or None
        Optional query-string parameters.

    Returns
    -------
    dict
        Parsed JSON response body.

    Raises
    ------
    requests.exceptions.HTTPError
        After 5 failed attempts or on a non-retryable status code.
    ValueError
        If the response body is not valid JSON.
    """
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)

    # Raise HTTPError for 4xx / 5xx status codes so tenacity can retry
    response.raise_for_status()

    # Parse JSON with explicit error handling
    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError as exc:
        raise ValueError(f"Non-JSON response from {url}: {response.text[:200]}") from exc

    return data


# ---------------------------------------------------------------------------
# Polygon.io — Aggregate Bars
# ---------------------------------------------------------------------------

def fetch_polygon_bars(
    ticker: str,
    start: str,
    end: str,
    api_key: str,
    multiplier: int = 1,
    timespan: str = "day",
) -> list[dict[str, Any]]:
    """Fetch OHLCV bars from the Polygon.io Aggregates endpoint.

    Parameters
    ----------
    ticker : str
        Stock symbol, e.g. ``"AAPL"``.
    start, end : str
        Date range in ``YYYY-MM-DD`` format.
    api_key : str
        Polygon.io API key.
    multiplier : int
        Size of the timespan multiplier.
    timespan : str
        ``"minute"``, ``"hour"``, ``"day"``, ``"week"``, etc.

    Returns
    -------
    list[dict]
        List of bar dictionaries with keys ``o, h, l, c, v, t, ...``
    """
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}"
        f"/range/{multiplier}/{timespan}/{start}/{end}"
        f"?apiKey={api_key}"
    )

    logger.info("Polygon request: %s -> %s  (%s)", ticker, timespan, f"{start}/{end}")
    data = resilient_get(url)

    if data.get("resultsCount", 0) == 0:
        logger.warning("Polygon returned 0 results for %s", ticker)
        return []

    return data.get("results", [])


# ---------------------------------------------------------------------------
# Alpha Vantage — Daily Time Series
# ---------------------------------------------------------------------------

def fetch_alphavantage_daily(
    ticker: str,
    api_key: str,
    outputsize: str = "compact",
) -> dict[str, dict[str, str]]:
    """Fetch daily OHLCV data from Alpha Vantage.

    Parameters
    ----------
    ticker : str
        Stock symbol, e.g. ``"MSFT"``.
    api_key : str
        Alpha Vantage API key.
    outputsize : str
        ``"compact"`` (last 100 days) or ``"full"`` (20+ years).

    Returns
    -------
    dict
        Mapping of date strings to OHLCV dictionaries.
    """
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_DAILY"
        f"&symbol={ticker}"
        f"&outputsize={outputsize}"
        f"&apikey={api_key}"
    )

    logger.info("Alpha Vantage request: %s (outputsize=%s)", ticker, outputsize)
    data = resilient_get(url)

    # Alpha Vantage returns errors as {"Error Message": "..."} or
    # rate-limit notes as {"Note": "..."}.
    if "Error Message" in data:
        raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
    if "Note" in data:
        logger.warning("Alpha Vantage rate-limit note: %s", data["Note"])

    return data.get("Time Series (Daily)", {})


# ---------------------------------------------------------------------------
# Rate-Limited Batch Helper
# ---------------------------------------------------------------------------

def fetch_multiple_tickers(
    tickers: list[str],
    api_key: str,
    source: str = "alphavantage",
    **kwargs,
) -> dict[str, Any]:
    """Sequentially fetch data for multiple tickers with rate-limit pauses.

    Parameters
    ----------
    tickers : list[str]
        List of stock symbols.
    api_key : str
        API key for the selected provider.
    source : str
        ``"alphavantage"`` or ``"polygon"``.
    **kwargs
        Additional keyword arguments forwarded to the underlying fetcher.

    Returns
    -------
    dict
        ``{ticker: data, ...}``
    """
    results: dict[str, Any] = {}

    for i, ticker in enumerate(tickers):
        logger.info("Fetching %d/%d: %s", i + 1, len(tickers), ticker)

        if source == "polygon":
            results[ticker] = fetch_polygon_bars(ticker, api_key=api_key, **kwargs)
        else:
            results[ticker] = fetch_alphavantage_daily(ticker, api_key=api_key, **kwargs)

        # Respect rate limits between consecutive requests
        if i < len(tickers) - 1:
            time.sleep(RATE_LIMIT_PAUSE)

    return results


# ---------------------------------------------------------------------------
# Usage Example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    polygon_key = os.getenv("POLYGON_API_KEY", "")
    av_key = os.getenv("ALPHAVANTAGE_API_KEY", "")

    # --- Polygon.io example ------------------------------------------------
    if polygon_key:
        bars = fetch_polygon_bars(
            ticker="AAPL",
            start="2024-01-01",
            end="2024-01-31",
            api_key=polygon_key,
        )
        print(f"Polygon: received {len(bars)} daily bars for AAPL")

    # --- Alpha Vantage example ---------------------------------------------
    if av_key:
        daily = fetch_alphavantage_daily(ticker="MSFT", api_key=av_key)
        print(f"Alpha Vantage: received {len(daily)} daily records for MSFT")

    # --- Batch fetch -------------------------------------------------------
    if av_key:
        tickers = ["AAPL", "MSFT", "GOOGL"]
        batch = fetch_multiple_tickers(tickers, api_key=av_key, source="alphavantage")
        for sym, records in batch.items():
            print(f"  {sym}: {len(records)} records")
