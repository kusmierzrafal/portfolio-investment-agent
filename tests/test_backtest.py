"""Deterministic tests for the equal-weight backtests."""

from __future__ import annotations

import unittest
from datetime import date

import pandas as pd

from backtest import FEE_RATE, simulate_equal_weight, simulate_equal_weight_monthly

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


class MonthlyRebalanceTests(unittest.TestCase):
    def test_rebalance_timing_and_equal_weights(self) -> None:
        index = _dates("2024-01-02", "2024-01-31", "2024-02-05", "2024-02-15", "2024-03-04")
        prices = _prices(
            {
                "AAA": [100.0, 200.0, 200.0, 100.0, 200.0],
                "BBB": [100.0, 100.0, 100.0, 100.0, 300.0],
                "SPY": [400.0, 400.0, 400.0, 400.0, 400.0],
                "DIA": [300.0, 300.0, 300.0, 300.0, 300.0],
            },
            index,
        )
        result = simulate_equal_weight_monthly(
            prices,
            _flat_dividends(prices),
            ["AAA", "BBB"],
            10_000,
            date(2024, 1, 2),
            date(2024, 3, 4),
        )

        self.assertEqual(
            [trade.date for trade in result.transactions],
            [index[0], index[0], index[2], index[2], index[4], index[4]],
        )
        self.assertEqual([trade.side for trade in result.transactions[:2]], ["BUY", "BUY"])
        self.assertEqual(
            [(trade.symbol, trade.side) for trade in result.transactions[2:4]],
            [("AAA", "SELL"), ("BBB", "BUY")],
        )
        self.assertEqual(
            [(trade.symbol, trade.side) for trade in result.transactions[4:]],
            [("BBB", "SELL"), ("AAA", "BUY")],
        )

        # January keeps the opening share counts while prices drift.
        opening = result.positions[0].shares
        self.assertAlmostEqual(result.positions[1].shares, opening)
        self.assertAlmostEqual(
            result.values.loc[index[1], "Portfolio"],
            opening * 200 + opening * 100,
        )

        aaa_shares = opening
        bbb_shares = opening
        for trade in result.transactions[2:4]:
            if trade.side == "SELL":
                aaa_shares -= trade.shares
            else:
                bbb_shares += trade.shares
        self.assertAlmostEqual(aaa_shares * 200, bbb_shares * 100)
        # February 15 changes the price and does not trade again.
        self.assertAlmostEqual(
            result.values.loc[index[3], "Portfolio"],
            aaa_shares * 100 + bbb_shares * 100,
        )
        self.assertNotAlmostEqual(aaa_shares * 100, bbb_shares * 100)

        # March equalizes the drifted holdings at that session's closes.
        self.assertAlmostEqual(result.holdings[0].market_value, result.holdings[1].market_value)
        self.assertAlmostEqual(result.holdings[0].price, 200)
        self.assertAlmostEqual(result.holdings[1].price, 300)

        same_month = simulate_equal_weight_monthly(
            prices,
            _flat_dividends(prices),
            ["AAA", "BBB"],
            10_000,
            date(2024, 1, 2),
            date(2024, 1, 31),
        )
        self.assertEqual([trade.side for trade in same_month.transactions], ["BUY", "BUY"])
        self.assertEqual([trade.date for trade in same_month.transactions], [index[0], index[0]])

    def test_fees_cash_and_portfolio_accounting(self) -> None:
        index = _dates("2024-01-02", "2024-01-16", "2024-02-01")
        prices = _prices(
            {
                "AAA": [10.0, 10.0, 20.0],
                "BBB": [10.0, 10.0, 10.0],
                "SPY": [10.0, 10.0, 10.0],
                "DIA": [10.0, 10.0, 10.0],
            },
            index,
        )
        dividends = _flat_dividends(prices)
        dividends.loc[index[0], "AAA"] = 100.0
        dividends.loc[index[1], "AAA"] = 2.0
        monthly = simulate_equal_weight_monthly(
            prices, dividends, ["AAA", "BBB"], 2_000, date(2024, 1, 2), date(2024, 2, 1)
        )
        held = simulate_equal_weight(
            prices, dividends, ["AAA", "BBB"], 2_000, date(2024, 1, 2), date(2024, 2, 1)
        )

        self.assertAlmostEqual(monthly.positions[0].fee, held.positions[0].fee)
        self.assertAlmostEqual(monthly.positions[0].shares, held.positions[0].shares)
        self.assertAlmostEqual(monthly.values.iloc[0]["Portfolio"], held.values.iloc[0]["Portfolio"])
        self.assertAlmostEqual(monthly.values.iloc[1]["Portfolio"], held.values.iloc[1]["Portfolio"])
        self.assertTrue(monthly.values["SPY"].equals(held.values["SPY"]))
        self.assertTrue(monthly.values["DIA"].equals(held.values["DIA"]))

        keep = 1 - FEE_RATE
        aaa_shares = held.positions[0].shares
        aaa_value = aaa_shares * 20
        bbb_value = held.positions[1].shares * 10
        dividend_cash = aaa_shares * 2
        target = (aaa_value * keep**2 + dividend_cash * keep + bbb_value) / (1 + keep**2)
        sell_amount = aaa_value - target
        buy_amount = target - bbb_value
        buy_gross = buy_amount / keep
        sell_fee = sell_amount * FEE_RATE
        buy_fee = buy_gross * FEE_RATE

        sell = next(trade for trade in monthly.transactions if trade.side == "SELL")
        buy = next(trade for trade in monthly.transactions if trade.date == index[2] and trade.side == "BUY")
        self.assertEqual(sell.symbol, "AAA")
        self.assertEqual(sell.date, index[2])
        self.assertEqual(buy.symbol, "BBB")
        self.assertAlmostEqual(sell.amount, sell_amount)
        self.assertAlmostEqual(sell.fee, sell_fee)
        self.assertAlmostEqual(sell.shares, sell_amount / 20)
        self.assertLess(sell.amount, aaa_value)
        self.assertAlmostEqual(buy.amount, buy_amount)
        self.assertAlmostEqual(buy.fee, buy_fee)
        self.assertAlmostEqual(buy.shares, buy_amount / 10)
        self.assertAlmostEqual(buy_gross, sell_amount * keep + dividend_cash)

        pre_trade = float(held.values.iloc[2]["Portfolio"])
        self.assertAlmostEqual(monthly.values.iloc[2]["Portfolio"], pre_trade - sell_fee - buy_fee)
        self.assertAlmostEqual(monthly.values.iloc[2]["Portfolio"], target * 2)
        self.assertAlmostEqual(monthly.cash, 0.0)
        self.assertGreater(monthly.holdings[0].shares, 0)
        self.assertGreater(monthly.holdings[1].shares, 0)
        self.assertAlmostEqual(
            sum(holding.market_value for holding in monthly.holdings) + monthly.cash,
            monthly.values.iloc[2]["Portfolio"],
        )

    def test_shared_prices_keep_each_strategy_unchanged(self) -> None:
        index = _dates("2024-01-02", "2024-01-16", "2024-02-01")
        prices = _prices(
            {
                "AAA": [10.0, 10.0, 20.0],
                "BBB": [10.0, 10.0, 10.0],
                "SPY": [10.0, 10.0, 10.0],
                "DIA": [10.0, 10.0, 10.0],
            },
            index,
        )
        dividends = _flat_dividends(prices)
        dividends.loc[index[1], "AAA"] = 2.0
        symbols = ["AAA", "BBB"]
        start, end, capital = date(2024, 1, 2), date(2024, 2, 1), 2_000

        held = simulate_equal_weight(prices, dividends, symbols, capital, start, end)
        monthly = simulate_equal_weight_monthly(prices, dividends, symbols, capital, start, end)
        held_after = simulate_equal_weight(prices, dividends, symbols, capital, start, end)
        monthly_after = simulate_equal_weight_monthly(
            prices, dividends, symbols, capital, start, end
        )

        self.assertTrue(held.values["Portfolio"].equals(held_after.values["Portfolio"]))
        self.assertTrue(monthly.values["Portfolio"].equals(monthly_after.values["Portfolio"]))
        self.assertEqual(held.purchase_date, monthly.purchase_date)
        self.assertTrue(held.values.index.equals(monthly.values.index))
        self.assertTrue(held.values["SPY"].equals(monthly.values["SPY"]))
        self.assertTrue(held.values["DIA"].equals(monthly.values["DIA"]))
        self.assertFalse(held.values["Portfolio"].equals(monthly.values["Portfolio"]))


if __name__ == "__main__":
    unittest.main()
