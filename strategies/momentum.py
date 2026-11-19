"""Monthly momentum allocation decisions."""

from __future__ import annotations

import pandas as pd

from backtesting.engine import BacktestError

MOMENTUM_LOOKBACK = 60
MOMENTUM_WARMUP_DAYS = 180


def momentum_decision(
    as_of: pd.Timestamp,
    history: pd.DataFrame,
    symbols: tuple[str, ...],
    is_opening: bool,
) -> dict[str, float] | None:
    """Put half the portfolio in each of the two strongest prior returns.

    ``history`` contains only sessions completed before ``as_of``. The two
    highest 60-session returns are selected even when both are negative.
    Equal returns are ordered alphabetically by ticker.
    """
    del is_opening
    if len(symbols) < 3:
        raise ValueError("Momentum Monthly needs at least 3 stocks.")

    ranked: list[tuple[float, str]] = []
    when = pd.Timestamp(as_of).date().isoformat()
    for symbol in symbols:
        series = history[symbol].dropna()
        series = series[series > 0]
        if len(series) <= MOMENTUM_LOOKBACK:
            raise BacktestError(
                f"Not enough completed sessions before {when} to compute "
                f"{MOMENTUM_LOOKBACK}-session momentum for {symbol}. "
                "The lookback was not shortened."
            )
        momentum = float(series.iloc[-1] / series.iloc[-1 - MOMENTUM_LOOKBACK] - 1)
        ranked.append((momentum, symbol))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    first, second = (symbol for _, symbol in ranked[:2])
    return {first: 0.5, second: 0.5}
