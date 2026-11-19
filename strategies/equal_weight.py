"""Equal-weight allocation decisions.

Buy and hold uses this allocation on the opening session only. Monthly
rebalancing uses it again on the first common session of each later month.
"""

from __future__ import annotations

import pandas as pd


def equal_weights(symbols: tuple[str, ...]) -> dict[str, float]:
    """Return an equal invested weight for every symbol and no cash weight."""
    weight = 1.0 / len(symbols)
    return {symbol: weight for symbol in symbols}


def buy_and_hold_decision(
    as_of: pd.Timestamp,
    history: pd.DataFrame,
    symbols: tuple[str, ...],
    is_opening: bool,
) -> dict[str, float] | None:
    """Buy equal weights once. Later decision dates are left unchanged."""
    del as_of, history
    if not is_opening:
        return None
    return equal_weights(symbols)


def monthly_rebalance_decision(
    as_of: pd.Timestamp,
    history: pd.DataFrame,
    symbols: tuple[str, ...],
    is_opening: bool,
) -> dict[str, float] | None:
    """Restore equal weights on every decision date, including the opening."""
    del as_of, history, is_opening
    return equal_weights(symbols)
