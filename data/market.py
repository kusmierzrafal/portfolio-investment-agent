"""Current Dow 30 catalogue and daily price downloads."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import yfinance as yf

# Current constituents only. This is not historical index membership.
DOW_30: dict[str, str] = {
    "AAPL": "Apple Inc.",
    "AMGN": "Amgen Inc.",
    "AMZN": "Amazon.com Inc.",
    "AXP": "American Express Company",
    "BA": "Boeing Company",
    "CAT": "Caterpillar Inc.",
    "CRM": "Salesforce, Inc.",
    "CSCO": "Cisco Systems, Inc.",
    "CVX": "Chevron Corporation",
    "DIS": "The Walt Disney Company",
    "GOOGL": "Alphabet Inc.",
    "GS": "The Goldman Sachs Group, Inc.",
    "HD": "The Home Depot, Inc.",
    "HON": "Honeywell International Inc.",
    "IBM": "International Business Machines Corporation",
    "JNJ": "Johnson & Johnson",
    "JPM": "JPMorgan Chase & Co.",
    "KO": "The Coca-Cola Company",
    "MCD": "McDonald's Corporation",
    "MMM": "3M Company",
    "MRK": "Merck & Co., Inc.",
    "MSFT": "Microsoft Corporation",
    "NKE": "NIKE, Inc.",
    "NVDA": "NVIDIA Corporation",
    "PG": "The Procter & Gamble Company",
    "SHW": "The Sherwin-Williams Company",
    "TRV": "The Travelers Companies, Inc.",
    "UNH": "UnitedHealth Group Incorporated",
    "V": "Visa Inc.",
    "WMT": "Walmart Inc.",
}

_PRICE_COLUMNS: tuple[str, ...] = (
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Dividends",
    "Stock Splits",
)


class MarketDataError(Exception):
    """Daily prices could not be retrieved."""


def download_daily_history(ticker: str, start: date, end: date) -> pd.DataFrame:
    """Download daily prices for one ticker from ``start`` through ``end``.

    ``end`` is inclusive. yfinance treats its end parameter as exclusive, so
    the request uses the following calendar day.

    The frame is indexed by date and keeps the price columns Yahoo returns:
    open, high, low, close, adjusted close, volume, dividends, and stock splits.
    Prices stay unadjusted (``auto_adjust=False``) and corporate actions are
    included (``actions=True``).

    Raises:
        ValueError: The ticker or the date range is invalid.
        MarketDataError: The download failed or returned no prices.
    """
    symbol = ticker.strip().upper()
    if not symbol:
        raise ValueError("Choose a ticker.")
    if not isinstance(start, date) or not isinstance(end, date):
        raise ValueError("Choose a start date and an end date.")
    if end < start:
        raise ValueError("The end date must be on or after the start date.")

    request_end = end + timedelta(days=1)
    try:
        frame = yf.Ticker(symbol).history(
            start=start,
            end=request_end,
            interval="1d",
            auto_adjust=False,
            actions=True,
        )
    except Exception as exc:
        raise MarketDataError(f"Could not download prices for {symbol}.") from exc

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise MarketDataError(_unavailable(symbol, start, end))

    if isinstance(frame.columns, pd.MultiIndex):
        frame = frame.copy()
        frame.columns = frame.columns.get_level_values(0)

    columns = [column for column in _PRICE_COLUMNS if column in frame.columns]
    if any(column not in columns for column in ("Open", "High", "Low", "Close")):
        raise MarketDataError(_unavailable(symbol, start, end))

    history = frame.loc[:, columns].copy()
    history.index = pd.DatetimeIndex(pd.to_datetime(history.index))
    if history.index.tz is not None:
        history.index = history.index.tz_localize(None)
    history.index.name = "Date"
    history = history.sort_index()
    history = history.loc[history["Close"].notna()]
    if history.empty:
        raise MarketDataError(_unavailable(symbol, start, end))
    return history


def download_closes_and_dividends(
    symbols: list[str],
    start: date,
    end: date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download split-adjusted closes and cash dividends for several tickers.

    Close is the split-adjusted price. Dividends stay in their own column, so
    a later valuation does not count a dividend twice.
    """
    closes: dict[str, pd.Series] = {}
    dividends: dict[str, pd.Series] = {}
    for symbol in symbols:
        history = download_daily_history(symbol, start, end)
        closes[symbol] = history["Close"]
        if "Dividends" in history.columns:
            dividends[symbol] = history["Dividends"]
        else:
            dividends[symbol] = pd.Series(0.0, index=history.index)
    return pd.DataFrame(closes), pd.DataFrame(dividends)


def _unavailable(symbol: str, start: date, end: date) -> str:
    return (
        f"No price data is available for {symbol} "
        f"from {start.isoformat()} to {end.isoformat()}."
    )
