"""Shared portfolio simulation.

Strategies choose target weights. This module applies those weights: it buys
and sells, charges fees, collects dividends, and values the portfolio and the
SPY and DIA benchmarks. It does not decide which stocks a strategy wants.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from data.market import DOW_30, download_closes_and_dividends

FEE_RATE = 0.001
BENCHMARKS: tuple[str, ...] = ("SPY", "DIA")
CASH = "CASH"
WEIGHT_TOLERANCE = 1e-8
_DUST = 1e-6


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
class Transaction:
    date: pd.Timestamp
    symbol: str
    side: str
    shares: float
    amount: float
    fee: float


@dataclass(frozen=True)
class Holding:
    symbol: str
    shares: float
    price: float
    market_value: float


@dataclass(frozen=True)
class BacktestResult:
    purchase_date: pd.Timestamp
    initial_capital: float
    positions: tuple[Position, ...]
    benchmarks: tuple[Position, ...]
    values: pd.DataFrame
    transactions: tuple[Transaction, ...] = ()
    holdings: tuple[Holding, ...] = ()
    cash: float = 0.0


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


def load_backtest_inputs(
    symbols: list[str],
    start: date,
    end: date,
    initial_capital: float,
    history_start: date | None = None,
) -> tuple[tuple[str, ...], pd.DataFrame, pd.DataFrame]:
    """Validate one request and download its closes and dividends once.

    Several strategies can then be simulated from the returned frames.
    ``history_start`` downloads earlier sessions when a signal needs them.
    Portfolio valuation still begins at ``start``.
    """
    selected = _validate_request(symbols, start, end, initial_capital)
    download_start = start if history_start is None else history_start
    if history_start is not None and (
        not isinstance(history_start, date) or history_start > start
    ):
        raise ValueError("The history start must be on or before the start date.")
    needed = list(selected) + [symbol for symbol in BENCHMARKS if symbol not in selected]
    prices, dividends = download_closes_and_dividends(needed, download_start, end)
    return selected, prices, dividends


def simulate(
    prices: pd.DataFrame,
    dividends: pd.DataFrame,
    symbols: tuple[str, ...] | list[str],
    initial_capital: float,
    start: date,
    end: date,
    decide,
) -> BacktestResult:
    """Simulate one strategy from split-adjusted closes and cash dividends.

    ``decide(as_of, history, symbols, is_opening)`` returns target weights,
    or None when that date should not trade. ``history`` is the close table
    for the selected symbols on sessions strictly before ``as_of``. Weights
    are nonnegative, use the selected symbols plus optional ``CASH``, and
    sum to 1. The engine asks for a decision at the close of the first
    common session and at the close of the first common session of each
    later month. Trades use that session's close. A dividend on a decision
    session belongs to the shares held before the trade.
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
    window = closes.loc[(closes.index >= start_ts) & (closes.index <= end_ts)]
    if window.empty:
        raise BacktestError("No closing prices in the selected date range.")
    empty = [symbol for symbol in required if window[symbol].dropna().empty]
    if empty:
        raise BacktestError(f"No closing prices in range for {', '.join(empty)}.")

    available = (window[required].notna() & (window[required] > 0)).all(axis=1)
    valuation_dates = window.index[available]
    if len(valuation_dates) == 0:
        raise BacktestError(
            "No common session in the selected range with a closing price "
            "for every selected stock and both benchmarks."
        )

    purchase_date = valuation_dates[0]
    purchase_prices = closes.loc[purchase_date, required]
    benchmarks = tuple(
        _open_position(symbol, float(initial_capital), float(purchase_prices[symbol]))
        for symbol in BENCHMARKS
    )

    shares = {symbol: 0.0 for symbol in selected}
    cash = float(initial_capital)
    transactions: list[Transaction] = []
    portfolio: list[float] = []
    opening_weights: dict[str, float] | None = None
    previous = None
    for current in valuation_dates:
        if previous is not None:
            for symbol in selected:
                cash += shares[symbol] * _cash_dividends(
                    _dividend_series(dividend_frame, symbol),
                    previous,
                    current,
                )
        if current == purchase_date or _is_rebalance_date(current, purchase_date, valuation_dates):
            history = closes.loc[closes.index < current, list(selected)].copy()
            weights = decide(current, history, selected, current == purchase_date)
            if weights is not None:
                trade_prices = {
                    symbol: float(closes.loc[current, symbol]) for symbol in selected
                }
                targets, trade_order = _target_weights(weights, selected)
                if current == purchase_date:
                    opening_weights = targets
                cash, trades = _rebalance_to_weights(
                    shares,
                    trade_prices,
                    cash,
                    targets,
                    selected,
                    trade_order,
                    current,
                )
                transactions.extend(trades)
        equity = sum(
            shares[symbol] * float(closes.loc[current, symbol]) for symbol in selected
        )
        portfolio.append(equity + cash)
        previous = current

    values = pd.DataFrame(index=valuation_dates)
    values.index.name = "Date"
    values["Portfolio"] = pd.Series(portfolio, index=valuation_dates)
    for position in benchmarks:
        values[position.symbol] = _position_value(
            closes[position.symbol],
            _dividend_series(dividend_frame, position.symbol),
            position.shares,
            purchase_date,
            valuation_dates,
        )

    last = valuation_dates[-1]
    holdings = tuple(
        Holding(
            symbol=symbol,
            shares=shares[symbol],
            price=float(closes.loc[last, symbol]),
            market_value=shares[symbol] * float(closes.loc[last, symbol]),
        )
        for symbol in selected
        if shares[symbol] > 0.0
    )
    if opening_weights is None:
        positions: tuple[Position, ...] = ()
    else:
        positions = tuple(
            Position(
                symbol=trade.symbol,
                allocation=float(initial_capital) * opening_weights[trade.symbol],
                price=float(closes.loc[purchase_date, trade.symbol]),
                shares=trade.shares,
                fee=trade.fee,
            )
            for trade in transactions
            if trade.date == purchase_date and trade.side == "BUY"
        )
    return BacktestResult(
        purchase_date=purchase_date,
        initial_capital=float(initial_capital),
        positions=positions,
        benchmarks=benchmarks,
        values=values,
        transactions=tuple(transactions),
        holdings=holdings,
        cash=float(cash),
    )


def _rebalance_to_weights(
    shares: dict[str, float],
    prices: dict[str, float],
    cash: float,
    weights: dict[str, float],
    symbols: tuple[str, ...],
    trade_order: tuple[str, ...],
    when: pd.Timestamp,
) -> tuple[float, list[Transaction]]:
    """Trade the portfolio to ``weights``, charging fees on traded amounts.

    A buy deducts the fee from the cash spent. A sale deducts the fee from
    the proceeds. ``CASH`` is left uninvested. Names leaving the portfolio
    are sold first, then the remaining sells, then the buys.
    """
    scale = _portfolio_scale(shares, prices, cash, weights, symbols)
    market_values = {symbol: shares[symbol] * prices[symbol] for symbol in symbols}
    trades: list[Transaction] = []
    retained = 1.0 - FEE_RATE
    exits = tuple(symbol for symbol in symbols if weights[symbol] == 0.0)
    sells = exits + tuple(symbol for symbol in trade_order if symbol not in exits)
    for symbol in sells:
        target = weights[symbol] * scale
        sell_value = market_values[symbol] - target
        if sell_value <= _DUST:
            continue
        sell_shares = min(shares[symbol], sell_value / prices[symbol])
        sold_value = sell_shares * prices[symbol]
        fee = sold_value * FEE_RATE
        shares[symbol] -= sell_shares
        cash += sold_value - fee
        trades.append(Transaction(when, symbol, "SELL", sell_shares, sold_value, fee))
    for symbol in trade_order:
        target = weights[symbol] * scale
        buy_value = target - market_values[symbol]
        if buy_value <= _DUST:
            continue
        gross = buy_value / retained
        if gross > cash:
            gross = cash
        if gross <= _DUST:
            continue
        fee = gross * FEE_RATE
        bought_value = gross - fee
        bought_shares = bought_value / prices[symbol]
        shares[symbol] += bought_shares
        cash -= gross
        trades.append(Transaction(when, symbol, "BUY", bought_shares, bought_value, fee))
    return cash, trades


def _portfolio_scale(
    shares: dict[str, float],
    prices: dict[str, float],
    cash: float,
    weights: dict[str, float],
    symbols: tuple[str, ...],
) -> float:
    """Return the post-trade portfolio value that meets the target weights.

    Each stock's post-trade market value is its weight times this scale, and
    ending cash is the cash weight times the same scale. Fees make the scale
    smaller than the pre-trade wealth, so it is found by bisection.
    """
    wealth = sum(shares[symbol] * prices[symbol] for symbol in symbols) + cash
    low = 0.0
    high = wealth
    for _ in range(64):
        mid = (low + high) / 2.0
        if _cash_gap(shares, prices, cash, weights, symbols, mid) > 0.0:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def _cash_gap(
    shares: dict[str, float],
    prices: dict[str, float],
    cash: float,
    weights: dict[str, float],
    symbols: tuple[str, ...],
    scale: float,
) -> float:
    """Return ending cash minus the cash required by the cash weight."""
    retained = 1.0 - FEE_RATE
    leftover = cash
    for symbol in symbols:
        market_value = shares[symbol] * prices[symbol]
        traded = market_value - weights[symbol] * scale
        if traded > 0.0:
            leftover += traded * retained
        elif traded < 0.0:
            leftover -= -traded / retained
    return leftover - weights[CASH] * scale


def _target_weights(
    weights: dict,
    symbols: tuple[str, ...],
) -> tuple[dict[str, float], tuple[str, ...]]:
    if not isinstance(weights, dict) or not weights:
        raise BacktestError("A decision must return target weights.")
    allowed = set(symbols)
    cleaned: dict[str, float] = {}
    order: list[str] = []
    for key, value in weights.items():
        symbol = str(key).strip().upper()
        if isinstance(value, bool) or not isinstance(value, (int, float)) or pd.isna(value):
            raise BacktestError(f"Weight for {symbol} must be a nonnegative number.")
        if value < 0.0:
            raise BacktestError(f"Weight for {symbol} must be nonnegative.")
        if symbol != CASH and symbol not in allowed:
            raise BacktestError(f"Unknown portfolio symbol: {symbol}.")
        cleaned[symbol] = float(value)
        if symbol != CASH and float(value) > 0.0 and symbol not in order:
            order.append(symbol)
    if abs(sum(cleaned.values()) - 1.0) > WEIGHT_TOLERANCE:
        raise BacktestError("Target weights must sum to 1.")
    targets = {symbol: cleaned.get(symbol, 0.0) for symbol in symbols}
    targets[CASH] = cleaned.get(CASH, 0.0)
    return targets, tuple(order)


def _is_rebalance_date(
    current: pd.Timestamp,
    purchase_date: pd.Timestamp,
    dates: pd.DatetimeIndex,
) -> bool:
    """Return whether ``current`` is the first common session of a later month."""
    if (current.year, current.month) <= (purchase_date.year, purchase_date.month):
        return False
    month = dates[(dates.year == current.year) & (dates.month == current.month)]
    return len(month) > 0 and current == month[0]


def _cash_dividends(
    dividends: pd.Series,
    after: pd.Timestamp,
    through: pd.Timestamp,
) -> float:
    paid = dividends.fillna(0.0)
    paid = paid.loc[(paid.index > after) & (paid.index <= through)]
    if paid.empty:
        return 0.0
    return float(paid.sum())


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
