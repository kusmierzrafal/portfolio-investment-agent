"""Equal-weight buy-and-hold backtest.

Prices are the split-adjusted closes from ``market_data``. Dividends are the
split-adjusted cash amounts from the same download and are not reinvested.
Adjusted close is not used, so a dividend is not counted twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from market_data import DOW_30, download_daily_history

FEE_RATE = 0.001
BENCHMARKS: tuple[str, ...] = ("SPY", "DIA")


class BacktestError(Exception):
    """The backtest inputs cannot be simulated."""


@dataclass(frozen=True)
class Position:
    symbol: str
    allocation: float
    price: float
    shares: float
    fee: float


@dataclass(frozen=True)
class BacktestResult:
    purchase_date: pd.Timestamp
    initial_capital: float
    positions: tuple[Position, ...]
    benchmarks: tuple[Position, ...]
    values: pd.DataFrame


def total_return(final_value: float, initial_capital: float) -> float:
    """Return final value / initial capital - 1, including purchase fees."""
    if initial_capital <= 0:
        raise ValueError("Initial capital must be a positive amount.")
    return final_value / initial_capital - 1


def maximum_drawdown(values: pd.Series) -> float:
    """Return the lowest value / running peak - 1.

    The result is zero when value never declines and negative otherwise.
    """
    if values.empty:
        raise BacktestError("Cannot compute drawdown without values.")
    peak = values.cummax()
    if (peak <= 0).any():
        raise BacktestError("Cannot compute drawdown when the running peak is not positive.")
    return float((values / peak - 1).min())


def run_equal_weight_backtest(
    symbols: list[str],
    start: date,
    end: date,
    initial_capital: float,
) -> BacktestResult:
    """Download prices and simulate an equal-weight buy-and-hold portfolio.

    ``end`` is inclusive. SPY and DIA are simulated with the same capital,
    purchase session, and fee, and are not portfolio holdings.
    """
    selected = _validate_request(symbols, start, end, initial_capital)
    needed = list(selected) + [symbol for symbol in BENCHMARKS if symbol not in selected]
    prices, dividends = _load_market_data(needed, start, end)
    return simulate_equal_weight(
        prices,
        dividends,
        selected,
        initial_capital,
        start,
        end,
    )


def simulate_equal_weight(
    prices: pd.DataFrame,
    dividends: pd.DataFrame,
    symbols: tuple[str, ...] | list[str],
    initial_capital: float,
    start: date,
    end: date,
) -> BacktestResult:
    """Simulate the strategy from split-adjusted closes and cash dividends.

    The purchase is the first session in ``[start, end]`` on which every
    selected symbol and both benchmarks have a positive close. Each selected
    symbol receives an equal share of capital. The fee is deducted from that
    allocation. Share counts then stay constant. Dividend cash is added once,
    on ex-dates after the purchase, and is not reinvested.
    """
    if not isinstance(start, date) or not isinstance(end, date):
        raise ValueError("Choose a start date and an end date.")
    if end < start:
        raise ValueError("The end date must be on or after the start date.")
    _validate_capital(initial_capital)
    selected = _clean_symbols(symbols)
    if any(symbol in BENCHMARKS for symbol in selected):
        raise ValueError("SPY and DIA are benchmarks, not portfolio stocks.")

    closes = _normalize_frame(prices, "Price data")
    dividend_frame = _normalize_frame(dividends, "Dividend data")
    required = list(selected) + [symbol for symbol in BENCHMARKS if symbol not in selected]
    missing = [symbol for symbol in required if symbol not in closes.columns]
    if missing:
        raise BacktestError(f"Missing price data for {', '.join(missing)}.")

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    closes = closes.loc[(closes.index >= start_ts) & (closes.index <= end_ts)]
    dividend_frame = dividend_frame.loc[
        (dividend_frame.index >= start_ts) & (dividend_frame.index <= end_ts)
    ]
    if closes.empty:
        raise BacktestError("No closing prices in the selected date range.")

    empty = [symbol for symbol in required if closes[symbol].dropna().empty]
    if empty:
        raise BacktestError(f"No closing prices in range for {', '.join(empty)}.")

    available = (closes[required].notna() & (closes[required] > 0)).all(axis=1)
    valuation_dates = closes.index[available]
    if len(valuation_dates) == 0:
        raise BacktestError(
            "No common session in the selected range with a closing price "
            "for every selected stock and both benchmarks."
        )

    purchase_date = valuation_dates[0]
    purchase_prices = closes.loc[purchase_date, required]
    positions = tuple(
        _open_position(symbol, initial_capital / len(selected), float(purchase_prices[symbol]))
        for symbol in selected
    )
    benchmarks = tuple(
        _open_position(symbol, initial_capital, float(purchase_prices[symbol]))
        for symbol in BENCHMARKS
    )

    values = pd.DataFrame(index=valuation_dates)
    values.index.name = "Date"
    portfolio_parts = [
        _position_value(
            closes[position.symbol],
            _dividend_series(dividend_frame, position.symbol),
            position.shares,
            purchase_date,
            valuation_dates,
        )
        for position in positions
    ]
    values["Portfolio"] = pd.concat(portfolio_parts, axis=1).sum(axis=1)
    for position in benchmarks:
        values[position.symbol] = _position_value(
            closes[position.symbol],
            _dividend_series(dividend_frame, position.symbol),
            position.shares,
            purchase_date,
            valuation_dates,
        )
    return BacktestResult(
        purchase_date=purchase_date,
        initial_capital=float(initial_capital),
        positions=positions,
        benchmarks=benchmarks,
        values=values,
    )


def _validate_request(
    symbols: list[str] | tuple[str, ...],
    start: date,
    end: date,
    initial_capital: float,
) -> tuple[str, ...]:
    if not isinstance(start, date) or not isinstance(end, date):
        raise ValueError("Choose a start date and an end date.")
    if end < start:
        raise ValueError("The end date must be on or after the start date.")
    _validate_capital(initial_capital)
    selected = _clean_symbols(symbols)
    if any(symbol in BENCHMARKS for symbol in selected):
        raise ValueError("SPY and DIA are benchmarks, not portfolio stocks.")
    unknown = [symbol for symbol in selected if symbol not in DOW_30]
    if unknown:
        raise ValueError(f"Unknown Dow 30 ticker: {', '.join(unknown)}.")
    return selected


def _validate_capital(initial_capital: float) -> None:
    if isinstance(initial_capital, bool) or not isinstance(initial_capital, (int, float)):
        raise ValueError("Initial capital must be a positive amount.")
    if pd.isna(initial_capital) or initial_capital <= 0:
        raise ValueError("Initial capital must be a positive amount.")


def _clean_symbols(symbols: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(symbols, str):
        raise ValueError("Select at least one stock.")
    cleaned: list[str] = []
    for symbol in symbols:
        text = str(symbol).strip().upper()
        if not text:
            raise ValueError("Select at least one stock.")
        cleaned.append(text)
    if not cleaned:
        raise ValueError("Select at least one stock.")
    if len(cleaned) != len(set(cleaned)):
        raise ValueError("Select each stock only once.")
    return tuple(cleaned)


def _open_position(symbol: str, allocation: float, price: float) -> Position:
    fee = allocation * FEE_RATE
    return Position(
        symbol=symbol,
        allocation=allocation,
        price=price,
        shares=(allocation - fee) / price,
        fee=fee,
    )


def _position_value(
    close: pd.Series,
    dividend: pd.Series,
    shares: float,
    purchase_date: pd.Timestamp,
    dates: pd.DatetimeIndex,
) -> pd.Series:
    held = close.reindex(dates)
    events = dividend.fillna(0.0)
    events = events.loc[(events.index > purchase_date) & (events.index <= dates[-1])]
    events = events.loc[events != 0.0]
    if events.empty:
        cash = pd.Series(0.0, index=dates)
    else:
        timeline = dates.union(events.index).sort_values()
        cash = (events.reindex(timeline).fillna(0.0) * shares).cumsum().reindex(dates)
    return shares * held + cash


def _dividend_series(dividends: pd.DataFrame, symbol: str) -> pd.Series:
    if symbol not in dividends.columns:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    return dividends[symbol]


def _normalize_frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise BacktestError(f"{label} must be a table of daily values.")
    normalized = frame.copy()
    normalized.index = pd.DatetimeIndex(pd.to_datetime(normalized.index))
    if normalized.index.tz is not None:
        normalized.index = normalized.index.tz_localize(None)
    normalized.index = normalized.index.normalize()
    normalized = normalized.sort_index()
    if normalized.index.has_duplicates:
        raise BacktestError(f"{label} contains duplicate dates.")
    return normalized


def _load_market_data(
    symbols: list[str],
    start: date,
    end: date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
