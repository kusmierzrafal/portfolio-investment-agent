"""Equal-weight and momentum portfolio backtests.

Prices are the split-adjusted closes from ``market_data``. Dividends are the
split-adjusted cash amounts from the same download. Adjusted close is not
used, so a dividend is not counted twice.

Buy and hold invests once. Monthly rebalancing invests the same way, then
returns to equal weights on the first common session of each later month.
Momentum Monthly holds the two stocks with the strongest 60-session return,
using only sessions completed before each execution date.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from market_data import DOW_30, download_daily_history

FEE_RATE = 0.001
BENCHMARKS: tuple[str, ...] = ("SPY", "DIA")
MOMENTUM_LOOKBACK = 60
MOMENTUM_WARMUP_DAYS = 180


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
    prices, dividends = _load_market_data(needed, download_start, end)
    return selected, prices, dividends


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
    selected, prices, dividends = load_backtest_inputs(
        symbols, start, end, initial_capital
    )
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


def run_equal_weight_monthly_backtest(
    symbols: list[str],
    start: date,
    end: date,
    initial_capital: float,
) -> BacktestResult:
    """Download prices and simulate equal-weight monthly rebalancing.

    The opening trade matches buy and hold. On the first common session of
    each later calendar month, holdings are traded back to equal weights.
    SPY and DIA stay buy-and-hold benchmarks.
    """
    selected, prices, dividends = load_backtest_inputs(
        symbols, start, end, initial_capital
    )
    return simulate_equal_weight_monthly(
        prices,
        dividends,
        selected,
        initial_capital,
        start,
        end,
    )


def simulate_equal_weight_monthly(
    prices: pd.DataFrame,
    dividends: pd.DataFrame,
    symbols: tuple[str, ...] | list[str],
    initial_capital: float,
    start: date,
    end: date,
) -> BacktestResult:
    """Simulate equal weights with a rebalance on each later month's first session.

    The opening purchase is the buy-and-hold purchase: equal allocations,
    fractional shares, and the purchase fee taken from each allocation.
    That session is not a rebalance. Each later rebalance trades at the close
    so every selected stock has the same market value. Dividend cash already
    received is part of that value. Fees are charged on the shares actually
    bought or sold.
    """
    baseline = simulate_equal_weight(
        prices,
        dividends,
        symbols,
        initial_capital,
        start,
        end,
    )
    closes = _normalize_frame(prices, "Price data")
    dividend_frame = _normalize_frame(dividends, "Dividend data")
    dates = baseline.values.index
    selected = [position.symbol for position in baseline.positions]
    shares = {position.symbol: position.shares for position in baseline.positions}
    cash = 0.0
    transactions = [
        Transaction(
            date=baseline.purchase_date,
            symbol=position.symbol,
            side="BUY",
            shares=position.shares,
            amount=position.shares * position.price,
            fee=position.fee,
        )
        for position in baseline.positions
    ]

    portfolio: list[float] = []
    previous = None
    for current in dates:
        if previous is not None:
            for symbol in selected:
                cash += shares[symbol] * _cash_dividends(
                    _dividend_series(dividend_frame, symbol),
                    previous,
                    current,
                )
        if _is_rebalance_date(current, baseline.purchase_date, dates):
            trade_prices = {
                symbol: float(closes.loc[current, symbol]) for symbol in selected
            }
            cash, trades = _rebalance_to_equal_weight(
                shares,
                trade_prices,
                cash,
                selected,
                current,
            )
            transactions.extend(trades)
        equity = sum(
            shares[symbol] * float(closes.loc[current, symbol]) for symbol in selected
        )
        portfolio.append(equity + cash)
        previous = current

    values = baseline.values.copy()
    values["Portfolio"] = pd.Series(portfolio, index=dates)
    last = dates[-1]
    holdings = tuple(
        Holding(
            symbol=symbol,
            shares=shares[symbol],
            price=float(closes.loc[last, symbol]),
            market_value=shares[symbol] * float(closes.loc[last, symbol]),
        )
        for symbol in selected
    )
    return BacktestResult(
        purchase_date=baseline.purchase_date,
        initial_capital=baseline.initial_capital,
        positions=baseline.positions,
        benchmarks=baseline.benchmarks,
        values=values,
        transactions=tuple(transactions),
        holdings=holdings,
        cash=float(cash),
    )


def simulate_momentum_monthly(
    prices: pd.DataFrame,
    dividends: pd.DataFrame,
    symbols: tuple[str, ...] | list[str],
    initial_capital: float,
    start: date,
    end: date,
) -> BacktestResult:
    """Hold the two strongest stocks and reset that pair each month.

    The opening session is the same common session used by buy and hold.
    On that close, and on the first common close of each later month, the
    portfolio sells anything outside the new top two and puts half of its
    value in each winner. Momentum uses only positive closes strictly before
    the execution session. The portfolio stays invested when those returns are
    negative.
    """
    baseline = simulate_equal_weight(
        prices,
        dividends,
        symbols,
        initial_capital,
        start,
        end,
    )
    selected = [position.symbol for position in baseline.positions]
    if len(selected) < 3:
        raise ValueError("Momentum Monthly needs at least 3 stocks.")

    closes = _normalize_frame(prices, "Price data")
    dividend_frame = _normalize_frame(dividends, "Dividend data")
    dates = baseline.values.index
    purchase = baseline.purchase_date
    shares = {symbol: 0.0 for symbol in selected}
    cash = float(baseline.initial_capital)
    transactions: list[Transaction] = []
    portfolio: list[float] = []
    previous = None
    for current in dates:
        if previous is not None:
            for symbol in selected:
                cash += shares[symbol] * _cash_dividends(
                    _dividend_series(dividend_frame, symbol),
                    previous,
                    current,
                )
        if current == purchase or _is_rebalance_date(current, purchase, dates):
            winners = _select_momentum(closes, selected, current)
            trade_prices = {
                symbol: float(closes.loc[current, symbol]) for symbol in selected
            }
            cash, trades = _invest_in_winners(
                shares,
                trade_prices,
                cash,
                winners,
                current,
            )
            transactions.extend(trades)
        equity = sum(
            shares[symbol] * float(closes.loc[current, symbol]) for symbol in selected
        )
        portfolio.append(equity + cash)
        previous = current

    values = baseline.values.copy()
    values["Portfolio"] = pd.Series(portfolio, index=dates)
    last = dates[-1]
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
    opening = tuple(
        Position(
            symbol=trade.symbol,
            allocation=baseline.initial_capital / 2,
            price=float(closes.loc[purchase, trade.symbol]),
            shares=trade.shares,
            fee=trade.fee,
        )
        for trade in transactions
        if trade.date == purchase and trade.side == "BUY"
    )
    return BacktestResult(
        purchase_date=purchase,
        initial_capital=baseline.initial_capital,
        positions=opening,
        benchmarks=baseline.benchmarks,
        values=values,
        transactions=tuple(transactions),
        holdings=holdings,
        cash=float(cash),
    )


def _select_momentum(
    closes: pd.DataFrame,
    symbols: list[str],
    as_of: pd.Timestamp,
) -> list[str]:
    """Return the top two symbols by completed 60-session return.

    Equal returns are ordered by ticker. The execution session is excluded.
    """
    ranked: list[tuple[float, str]] = []
    when = pd.Timestamp(as_of).date().isoformat()
    for symbol in symbols:
        history = closes.loc[closes.index < as_of, symbol].dropna()
        history = history[history > 0]
        if len(history) <= MOMENTUM_LOOKBACK:
            raise BacktestError(
                f"Not enough completed sessions before {when} to compute "
                f"{MOMENTUM_LOOKBACK}-session momentum for {symbol}. "
                "The lookback was not shortened."
            )
        momentum = float(history.iloc[-1] / history.iloc[-1 - MOMENTUM_LOOKBACK] - 1)
        ranked.append((momentum, symbol))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [symbol for _, symbol in ranked[:2]]


def _invest_in_winners(
    shares: dict[str, float],
    prices: dict[str, float],
    cash: float,
    winners: list[str],
    when: pd.Timestamp,
) -> tuple[float, list[Transaction]]:
    """Sell stocks that left the top two, then split the portfolio across it."""
    trades: list[Transaction] = []
    chosen = set(winners)
    for symbol in list(shares):
        if symbol in chosen or shares[symbol] <= 0.0:
            continue
        sold_value = shares[symbol] * prices[symbol]
        sold_shares = shares[symbol]
        shares[symbol] = 0.0
        if sold_value <= 1e-6:
            continue
        fee = sold_value * FEE_RATE
        cash += sold_value - fee
        trades.append(Transaction(when, symbol, "SELL", sold_shares, sold_value, fee))
    cash, rebalance = _rebalance_to_equal_weight(
        shares,
        prices,
        cash,
        winners,
        when,
    )
    trades.extend(rebalance)
    return cash, trades


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


def _equal_market_value(market_values: list[float], cash: float) -> float:
    """Return the equal post-trade market value that spends the available cash.

    A buy spends gross cash and receives shares worth gross cash times
    ``1 - fee``. A sale pays the fee out of the sale proceeds. The target is
    the value at which those cash flows leave no uninvested cash.
    """
    wealth = sum(market_values) + cash
    low = 0.0
    high = wealth
    for _ in range(64):
        mid = (low + high) / 2.0
        if _cash_after_equal_weight(market_values, cash, mid) > 0.0:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def _cash_after_equal_weight(
    market_values: list[float],
    cash: float,
    target: float,
) -> float:
    retained = 1.0 - FEE_RATE
    leftover = cash
    for market_value in market_values:
        traded = market_value - target
        if traded > 0.0:
            leftover += traded * retained
        elif traded < 0.0:
            leftover -= -traded / retained
    return leftover


def _rebalance_to_equal_weight(
    shares: dict[str, float],
    prices: dict[str, float],
    cash: float,
    symbols: list[str],
    when: pd.Timestamp,
) -> tuple[float, list[Transaction]]:
    market_values = [shares[symbol] * prices[symbol] for symbol in symbols]
    target = _equal_market_value(market_values, cash)
    retained = 1.0 - FEE_RATE
    trades: list[Transaction] = []
    for symbol, market_value in zip(symbols, market_values):
        sell_value = market_value - target
        if sell_value <= 1e-6:
            continue
        sell_shares = min(shares[symbol], sell_value / prices[symbol])
        sold_value = sell_shares * prices[symbol]
        fee = sold_value * FEE_RATE
        shares[symbol] -= sell_shares
        cash += sold_value - fee
        trades.append(Transaction(when, symbol, "SELL", sell_shares, sold_value, fee))
    for symbol, market_value in zip(symbols, market_values):
        buy_value = target - market_value
        if buy_value <= 1e-6:
            continue
        gross = buy_value / retained
        if gross > cash:
            gross = cash
        if gross <= 1e-6:
            continue
        fee = gross * FEE_RATE
        bought_value = gross - fee
        bought_shares = bought_value / prices[symbol]
        shares[symbol] += bought_shares
        cash -= gross
        trades.append(Transaction(when, symbol, "BUY", bought_shares, bought_value, fee))
    return cash, trades


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
