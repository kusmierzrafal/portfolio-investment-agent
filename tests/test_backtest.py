"""Deterministic tests for the equal-weight buy-and-hold backtest."""

from __future__ import annotations

import unittest
from datetime import date

import pandas as pd

from backtest import FEE_RATE, simulate_equal_weight

START = date(2024, 1, 2)
END = date(2024, 1, 5)


def _dates(*iso_dates: str) -> pd.DatetimeIndex:
    return pd.to_datetime(list(iso_dates))


def _prices(rows: dict[str, list[float]], index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(rows, index=index)


def _flat_dividends(prices: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)


class EqualWeightBacktestTests(unittest.TestCase):
    def test_allocation_shares_and_fees(self) -> None:
        index = _dates("2024-01-02", "2024-01-03")
        prices = _prices(
            {
                "AAA": [100.0, 110.0],
                "BBB": [25.0, 20.0],
                "SPY": [400.0, 400.0],
                "DIA": [300.0, 300.0],
            },
            index,
        )
        result = simulate_equal_weight(
            prices, _flat_dividends(prices), ["AAA", "BBB"], 10_000, START, END
        )

        self.assertEqual([position.allocation for position in result.positions], [5_000, 5_000])
        self.assertAlmostEqual(result.positions[0].shares, 4_995 / 100)
        self.assertAlmostEqual(result.positions[1].shares, 4_995 / 25)
        self.assertAlmostEqual(result.positions[0].fee, 5)
        self.assertAlmostEqual(result.positions[1].fee, 5)
        # A later price does not change the shares bought at the first close.
        self.assertAlmostEqual(result.positions[0].price, 100)
        self.assertAlmostEqual(
            result.values.iloc[1]["Portfolio"],
            result.positions[0].shares * 110 + result.positions[1].shares * 20,
        )

    def test_daily_value_counts_each_dividend_once(self) -> None:
        index = _dates("2024-01-02", "2024-01-03", "2024-01-04")
        prices = _prices(
            {
                "AAA": [100.0, 98.0, 98.0],
                "BBB": [50.0, 50.0, 55.0],
                "SPY": [400.0, 400.0, 400.0],
                "DIA": [300.0, 300.0, 300.0],
            },
            index,
        )
        dividends = _flat_dividends(prices)
        dividends.loc[index[0], "AAA"] = 5.0
        dividends.loc[index[1], "AAA"] = 2.0
        dividends.loc[index[2], "BBB"] = 1.0
        result = simulate_equal_weight(
            prices, dividends, ["AAA", "BBB"], 2_000, START, END
        )
        aaa_shares = result.positions[0].shares
        bbb_shares = result.positions[1].shares

        self.assertAlmostEqual(result.values.iloc[0]["Portfolio"], aaa_shares * 100 + bbb_shares * 50)
        # AAA's close falls by 2 and a cash dividend of 2 is added once.
        self.assertAlmostEqual(
            result.values.iloc[1]["Portfolio"],
            aaa_shares * 98 + bbb_shares * 50 + aaa_shares * 2,
        )
        self.assertAlmostEqual(
            result.values.iloc[2]["Portfolio"],
            aaa_shares * 98 + bbb_shares * 55 + aaa_shares * 2 + bbb_shares * 1,
        )

    def test_results_stay_inside_the_selected_dates(self) -> None:
        index = _dates("2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04")
        prices = _prices(
            {
                "AAA": [1.0, 10.0, 10.0, 1_000.0],
                "SPY": [10.0, 10.0, 10.0, 10.0],
                "DIA": [10.0, 10.0, 10.0, 10.0],
            },
            index,
        )
        dividends = _flat_dividends(prices)
        dividends.loc[index[0], "AAA"] = 100.0
        dividends.loc[index[3], "AAA"] = 50.0
        result = simulate_equal_weight(
            prices,
            dividends,
            ["AAA"],
            1_000,
            date(2024, 1, 2),
            date(2024, 1, 3),
        )

        self.assertEqual(list(result.values.index), list(_dates("2024-01-02", "2024-01-03")))
        self.assertAlmostEqual(result.positions[0].shares, (1_000 * (1 - FEE_RATE)) / 10)
        self.assertAlmostEqual(result.values.iloc[-1]["Portfolio"], 1_000 * (1 - FEE_RATE))


if __name__ == "__main__":
    unittest.main()
