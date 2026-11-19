"""Deterministic tests for the equal-weight and momentum backtests."""

from __future__ import annotations

import unittest
from datetime import date

import pandas as pd

from backtesting.engine import FEE_RATE, BacktestError, simulate
from strategies.equal_weight import buy_and_hold_decision, monthly_rebalance_decision
from strategies.momentum import momentum_decision


def simulate_equal_weight(prices, dividends, symbols, initial_capital, start, end):
    return simulate(
        prices, dividends, symbols, initial_capital, start, end, buy_and_hold_decision
    )


def simulate_equal_weight_monthly(prices, dividends, symbols, initial_capital, start, end):
    return simulate(
        prices, dividends, symbols, initial_capital, start, end, monthly_rebalance_decision
    )


def simulate_momentum_monthly(prices, dividends, symbols, initial_capital, start, end):
    return simulate(
        prices, dividends, symbols, initial_capital, start, end, momentum_decision
    )

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


class MomentumTests(unittest.TestCase):
    def test_ranking_excludes_the_execution_close(self) -> None:
        prior = pd.bdate_range(end="2024-01-31", periods=70)
        feb1 = pd.Timestamp("2024-02-01")
        feb15 = pd.Timestamp("2024-02-15")
        mar1 = pd.Timestamp("2024-03-01")
        index = prior.append(pd.DatetimeIndex([feb1, feb15, mar1]))
        symbols = ["AAA", "BBB", "CCC"]
        prices = pd.DataFrame(100.0, index=index, columns=[*symbols, "SPY", "DIA"])
        feb_history = index[index < feb1]
        mar_history = index[index < mar1]
        feb_last, feb_lag, feb_off = feb_history[-1], feb_history[-61], feb_history[-60]
        mar_last, mar_lag, mar_off = mar_history[-1], mar_history[-61], mar_history[-60]
        self.assertEqual(len({feb_last, feb_lag, feb_off, mar_last, mar_lag, mar_off}), 6)

        # February signal is negative for every stock. BBB leads, then AAA.
        # The session one step inside the window and the execution close would
        # select different winners.
        prices.loc[feb_last, "AAA"] = 90
        prices.loc[feb_last, "BBB"] = 95
        prices.loc[feb_last, "CCC"] = 80
        prices.loc[feb_off, "BBB"] = 1_000
        prices.loc[feb1, "AAA"] = 10
        prices.loc[feb1, "BBB"] = 10
        prices.loc[feb1, "CCC"] = 1_000
        # March: AAA leads and BBB ties CCC. Alphabetical order keeps BBB.
        prices.loc[mar_last, "AAA"] = 150
        prices.loc[mar_last, "BBB"] = 140
        prices.loc[mar_last, "CCC"] = 140
        prices.loc[mar_off, "BBB"] = 1_000
        prices.loc[mar1, "AAA"] = 10
        prices.loc[mar1, "BBB"] = 50
        prices.loc[mar1, "CCC"] = 10_000
        dividends = pd.DataFrame(0.0, index=index, columns=prices.columns)
        dividends.loc[feb_last, "BBB"] = 1_000
        start, end, capital = date(2024, 2, 1), date(2024, 3, 1), 2_000

        result = simulate_momentum_monthly(prices, dividends, symbols, capital, start, end)
        early = simulate_momentum_monthly(
            prices, dividends, symbols, capital, start, date(2024, 2, 15)
        )

        def bought(backtest, when: pd.Timestamp) -> list[str]:
            return sorted(
                trade.symbol
                for trade in backtest.transactions
                if trade.date == when and trade.side == "BUY"
            )

        self.assertEqual(bought(result, feb1), ["AAA", "BBB"])
        self.assertEqual(bought(early, feb1), ["AAA", "BBB"])
        self.assertLess(prices.loc[feb_last, "BBB"] / prices.loc[feb_lag, "BBB"] - 1, 0)
        self.assertLess(prices.loc[feb_last, "AAA"] / prices.loc[feb_lag, "AAA"] - 1, 0)
        self.assertEqual({holding.symbol for holding in result.holdings}, {"AAA", "BBB"})
        self.assertEqual(
            {trade.date for trade in result.transactions},
            {feb1, mar1},
        )
        self.assertTrue(result.values.index.equals(pd.DatetimeIndex([feb1, feb15, mar1])))
        self.assertAlmostEqual(result.values.loc[feb1, "Portfolio"], capital * (1 - FEE_RATE))
        self.assertAlmostEqual(result.cash, 0.0)

        window = prices.loc[index >= feb1]
        held_long = simulate_equal_weight(prices, dividends, symbols, capital, start, end)
        held_window = simulate_equal_weight(
            window, dividends.loc[window.index], symbols, capital, start, end
        )
        monthly_long = simulate_equal_weight_monthly(
            prices, dividends, symbols, capital, start, end
        )
        monthly_window = simulate_equal_weight_monthly(
            window, dividends.loc[window.index], symbols, capital, start, end
        )
        self.assertTrue(held_long.values.equals(held_window.values))
        self.assertTrue(monthly_long.values.equals(monthly_window.values))
        self.assertTrue(result.values["SPY"].equals(held_long.values["SPY"]))
        self.assertTrue(result.values["DIA"].equals(held_long.values["DIA"]))
        self.assertEqual(result.purchase_date, held_long.purchase_date)

        short_prior = feb_history[-60:]
        short_index = short_prior.append(index[index >= feb1])
        with self.assertRaises(BacktestError) as missing:
            simulate_momentum_monthly(
                prices.loc[short_index],
                dividends.loc[short_index],
                symbols,
                capital,
                start,
                end,
            )
        self.assertIn("not shortened", str(missing.exception))
        self.assertIn("AAA", str(missing.exception))

        tiny = pd.DataFrame(
            {"AAA": [10.0], "BBB": [10.0], "SPY": [10.0], "DIA": [10.0]},
            index=pd.to_datetime(["2024-02-01"]),
        )
        with self.assertRaisesRegex(ValueError, "at least 3"):
            simulate_momentum_monthly(
                tiny,
                pd.DataFrame(0.0, index=tiny.index, columns=tiny.columns),
                ["AAA", "BBB"],
                capital,
                start,
                start,
            )

    def test_holding_change_preserves_cash_accounting(self) -> None:
        prior = pd.bdate_range(end="2024-01-31", periods=61)
        feb1 = pd.Timestamp("2024-02-01")
        feb15 = pd.Timestamp("2024-02-15")
        mar1 = pd.Timestamp("2024-03-01")
        index = prior.append(pd.DatetimeIndex([feb1, feb15, mar1]))
        symbols = ["AAA", "BBB", "CCC"]
        prices = pd.DataFrame(10.0, index=index, columns=[*symbols, "SPY", "DIA"])
        feb_history = index[index < feb1]
        mar_history = index[index < mar1]
        prices.loc[feb_history[-1], "BBB"] = 20
        prices.loc[feb_history[-1], "CCC"] = 15
        prices.loc[mar_history[-61], "AAA"] = 5
        prices.loc[mar_history[-61], "BBB"] = 8
        prices.loc[mar_history[-61], "CCC"] = 20
        prices.loc[mar1, "BBB"] = 30
        dividends = pd.DataFrame(0.0, index=index, columns=prices.columns)
        dividends.loc[feb1, "BBB"] = 50
        dividends.loc[feb15, "BBB"] = 2
        copied_prices = prices.copy(deep=True)
        copied_dividends = dividends.copy(deep=True)
        start, end, capital = date(2024, 2, 1), date(2024, 3, 1), 2_000

        held = simulate_equal_weight(prices, dividends, symbols, capital, start, end)
        result = simulate_momentum_monthly(prices, dividends, symbols, capital, start, end)
        again = simulate_momentum_monthly(prices, dividends, symbols, capital, start, end)
        held_again = simulate_equal_weight(prices, dividends, symbols, capital, start, end)

        self.assertTrue(prices.equals(copied_prices))
        self.assertTrue(dividends.equals(copied_dividends))
        self.assertTrue(result.values["Portfolio"].equals(again.values["Portfolio"]))
        self.assertTrue(held.values.equals(held_again.values))
        self.assertTrue(result.values["SPY"].equals(held.values["SPY"]))
        self.assertEqual(result.purchase_date, held.purchase_date)
        self.assertTrue(result.values.index.equals(held.values.index))

        shares = capital / 2 * (1 - FEE_RATE) / 10
        self.assertAlmostEqual(result.values.loc[feb1, "Portfolio"], capital * (1 - FEE_RATE))
        opening = [trade for trade in result.transactions if trade.date == feb1]
        self.assertEqual(sorted(trade.symbol for trade in opening), ["BBB", "CCC"])
        for trade in opening:
            self.assertEqual(trade.side, "BUY")
            self.assertAlmostEqual(trade.shares, shares)
            self.assertAlmostEqual(trade.amount, shares * 10)
            self.assertAlmostEqual(trade.fee, capital / 2 * FEE_RATE)
        self.assertAlmostEqual(
            result.values.loc[feb15, "Portfolio"],
            shares * 10 + shares * 10 + shares * 2,
        )

        keep = 1 - FEE_RATE
        ccc_value = shares * 10
        bbb_value = shares * 30
        cash_after_sale = shares * 2 + ccc_value * keep
        target = (cash_after_sale + bbb_value * keep) * keep / (1 + keep**2)
        self.assertLess(target, bbb_value)
        march = [trade for trade in result.transactions if trade.date == mar1]
        sold_ccc = next(trade for trade in march if trade.symbol == "CCC")
        sold_bbb = next(trade for trade in march if trade.symbol == "BBB")
        bought_aaa = next(trade for trade in march if trade.symbol == "AAA")
        self.assertEqual(sold_ccc.side, "SELL")
        self.assertAlmostEqual(sold_ccc.shares, shares)
        self.assertAlmostEqual(sold_ccc.amount, ccc_value)
        self.assertAlmostEqual(sold_ccc.fee, ccc_value * FEE_RATE)
        self.assertEqual(sold_bbb.side, "SELL")
        self.assertAlmostEqual(sold_bbb.amount, bbb_value - target)
        self.assertAlmostEqual(sold_bbb.fee, (bbb_value - target) * FEE_RATE)
        self.assertLess(sold_bbb.shares, shares)
        self.assertEqual(bought_aaa.side, "BUY")
        self.assertAlmostEqual(bought_aaa.amount, target)
        self.assertAlmostEqual(bought_aaa.fee, target / keep * FEE_RATE)
        self.assertAlmostEqual(result.values.loc[mar1, "Portfolio"], target * 2)
        self.assertAlmostEqual(result.cash, 0.0)
        self.assertEqual([holding.symbol for holding in result.holdings], ["AAA", "BBB"])
        self.assertAlmostEqual(result.holdings[0].market_value, target)
        self.assertAlmostEqual(result.holdings[1].market_value, target)
        self.assertAlmostEqual(
            sum(holding.market_value for holding in result.holdings) + result.cash,
            result.values.loc[mar1, "Portfolio"],
        )

        position = {"AAA": 0.0, "BBB": 0.0, "CCC": 0.0}
        for trade in result.transactions:
            sign = 1 if trade.side == "BUY" else -1
            position[trade.symbol] += sign * trade.shares
            self.assertGreaterEqual(position[trade.symbol], -1e-9)
        self.assertAlmostEqual(position["CCC"], 0.0)
        self.assertGreater(position["AAA"], 0.0)
        self.assertGreater(position["BBB"], 0.0)


class EngineWeightTests(unittest.TestCase):
    def test_arbitrary_weights_leave_uninvested_cash(self) -> None:
        index = _dates("2024-01-02", "2024-02-03")
        prices = _prices(
            {
                "AAA": [10.0, 10.0],
                "BBB": [20.0, 20.0],
                "SPY": [10.0, 10.0],
                "DIA": [10.0, 10.0],
            },
            index,
        )
        capital = 1_000.0

        def decide(as_of, history, symbols, is_opening):
            del as_of, history, symbols
            if is_opening:
                return {"AAA": 0.4, "BBB": 0.3, "CASH": 0.3}
            return {"AAA": 0.5, "CASH": 0.5}

        result = simulate(
            prices,
            _flat_dividends(prices),
            ["AAA", "BBB"],
            capital,
            date(2024, 1, 2),
            date(2024, 2, 3),
            decide,
        )

        keep = 1 - FEE_RATE
        opening = capital / (0.7 / keep + 0.3)
        self.assertAlmostEqual(result.values.iloc[0]["Portfolio"], opening)
        self.assertAlmostEqual(result.positions[0].shares * 10, 0.4 * opening)
        self.assertAlmostEqual(result.positions[1].shares * 20, 0.3 * opening)
        self.assertEqual([trade.side for trade in result.transactions[:2]], ["BUY", "BUY"])
        self.assertNotIn("SELL", [trade.side for trade in result.transactions[:2]])

        aaa_before = 0.4 * opening
        bbb_before = 0.3 * opening
        cash_before = 0.3 * opening
        scale = (cash_before + bbb_before * keep + aaa_before / keep) / (0.5 * (1 / keep + 1))
        self.assertGreater(0.5 * scale, aaa_before)
        sold = next(
            trade
            for trade in result.transactions
            if trade.symbol == "BBB" and trade.side == "SELL"
        )
        bought = next(
            trade for trade in result.transactions if trade.date == index[1] and trade.symbol == "AAA"
        )
        self.assertEqual(sold.side, "SELL")
        self.assertAlmostEqual(sold.amount, bbb_before)
        self.assertAlmostEqual(sold.fee, bbb_before * FEE_RATE)
        self.assertEqual(bought.side, "BUY")
        self.assertAlmostEqual(bought.amount, 0.5 * scale - aaa_before)
        self.assertAlmostEqual(bought.fee, (0.5 * scale - aaa_before) / keep * FEE_RATE)
        self.assertAlmostEqual(result.values.iloc[1]["Portfolio"], scale)
        self.assertAlmostEqual(result.cash, 0.5 * scale)
        self.assertEqual([holding.symbol for holding in result.holdings], ["AAA"])
        self.assertAlmostEqual(result.holdings[0].market_value, 0.5 * scale)
        self.assertAlmostEqual(
            result.holdings[0].market_value + result.cash,
            result.values.iloc[1]["Portfolio"],
        )
        self.assertAlmostEqual(result.values.iloc[1]["SPY"], capital * keep)

        def overweight(as_of, history, symbols, is_opening):
            del as_of, history, symbols, is_opening
            return {"AAA": 0.8, "BBB": 0.8}

        with self.assertRaisesRegex(BacktestError, "sum to 1"):
            simulate(
                prices,
                _flat_dividends(prices),
                ["AAA", "BBB"],
                capital,
                date(2024, 1, 2),
                date(2024, 1, 2),
                overweight,
            )

        def short(as_of, history, symbols, is_opening):
            del as_of, history, symbols, is_opening
            return {"AAA": -0.2, "BBB": 1.2}

        with self.assertRaisesRegex(BacktestError, "nonnegative"):
            simulate(
                prices,
                _flat_dividends(prices),
                ["AAA", "BBB"],
                capital,
                date(2024, 1, 2),
                date(2024, 1, 2),
                short,
            )


if __name__ == "__main__":
    unittest.main()
