"""Mocked tests for LLM allocation requests and validation."""

from __future__ import annotations

import json
import unittest

import pandas as pd

from datetime import date

from backtesting.engine import BacktestError, CASH, simulate
from data.market import DOW_30
from strategies.llm import (
    build_allocation_request,
    llm_decision,
    translate_weights,
    validate_allocation,
)
from ui.backtesting import expected_llm_calls, llm_decision_dates, track_llm_decision


def _history() -> pd.DataFrame:
    """61 completed closes, then one execution close that must stay out of the request."""
    index = pd.bdate_range("2024-01-02", periods=62)
    return pd.DataFrame(
        {
            "AAPL": [20.0 + 10.0 * i for i in range(61)] + [9_999.0],
            "MSFT": [10.0 + 5.0 * i for i in range(61)] + [9_999.0],
        },
        index=index,
    )


def _scaled_series() -> str:
    return ", ".join(f"{100.0 + 50.0 * i:.6f}" for i in range(61))


class AllocationRequestTests(unittest.TestCase):
    def test_identity_modes_share_series_and_hide_or_show_names(self) -> None:
        history = _history()
        as_of = history.index[-1]
        symbols = ("MSFT", "AAPL")
        identified = build_allocation_request(
            history, symbols, as_of, identified=True, allow_cash=False
        )
        anonymized = build_allocation_request(
            history, symbols, as_of, identified=False, allow_cash=False
        )

        self.assertEqual(identified.series, anonymized.series)
        self.assertEqual(identified.schema, anonymized.schema)
        self.assertEqual(identified.series, (_scaled_series(),) * 2)
        self.assertEqual(len(identified.series[0].split(", ")), 61)
        self.assertNotIn("9999", anonymized.series[0])
        self.assertEqual(identified.id_to_symbol, {"A01": "MSFT", "A02": "AAPL"})

        identified_text = json.dumps(identified.messages)
        anonymized_request = json.dumps(
            {"messages": anonymized.messages, "schema": anonymized.schema}
        )
        self.assertIn("MSFT", identified_text)
        self.assertIn(DOW_30["MSFT"], identified_text)
        self.assertIn("AAPL", identified_text)
        self.assertIn(DOW_30["AAPL"], identified_text)
        for secret in ("MSFT", "AAPL", "Apple", "Microsoft", "2024-01"):
            self.assertNotIn(secret, anonymized_request)
        self.assertNotIn(CASH, anonymized.schema["properties"])
        self.assertNotIn(CASH, anonymized.schema["required"])

    def test_cash_setting_changes_only_the_permitted_assets(self) -> None:
        history = _history()
        as_of = history.index[-1]
        invested = build_allocation_request(
            history, ("AAPL",), as_of, identified=False, allow_cash=False
        )
        cash = build_allocation_request(
            history, ("AAPL",), as_of, identified=False, allow_cash=True
        )
        self.assertEqual(invested.series, cash.series)
        self.assertNotIn(CASH, invested.permitted)
        self.assertEqual(cash.permitted[-1], CASH)
        self.assertIn(CASH, cash.schema["properties"])

    def test_short_history_is_rejected_without_shortening(self) -> None:
        history = _history().iloc[:3]
        with self.assertRaises(BacktestError) as caught:
            build_allocation_request(
                history,
                ("AAPL",),
                _history().index[3],
                identified=False,
                allow_cash=False,
            )
        self.assertIn("not shortened", str(caught.exception))
        self.assertIn("AAPL", str(caught.exception))

    def test_missing_observations_share_one_window(self) -> None:
        index = pd.bdate_range("2024-01-02", periods=64)
        aapl = [100.0 + i for i in range(63)] + [9_999.0]
        msft = [200.0 + i for i in range(63)] + [9_999.0]
        msft[10] = float("nan")
        history = pd.DataFrame({"AAPL": aapl, "MSFT": msft}, index=index)
        request = build_allocation_request(
            history,
            ("MSFT", "AAPL"),
            index[-1],
            identified=False,
            allow_cash=False,
        )
        window_index = [i for i in range(1, 63) if i != 10]
        msft_values = pd.Series([200.0 + i for i in window_index], dtype=float)
        aapl_values = pd.Series([100.0 + i for i in window_index], dtype=float)
        expected_msft = ", ".join(f"{value:.6f}" for value in 100.0 * msft_values / msft_values.iloc[0])
        expected_aapl = ", ".join(f"{value:.6f}" for value in 100.0 * aapl_values / aapl_values.iloc[0])
        own_aapl = pd.Series([100.0 + i for i in range(2, 63)], dtype=float)
        unaligned_aapl = ", ".join(f"{value:.6f}" for value in 100.0 * own_aapl / own_aapl.iloc[0])

        self.assertEqual(len(window_index), 61)
        self.assertEqual(request.series, (expected_msft, expected_aapl))
        self.assertNotEqual(request.series[1], unaligned_aapl)
        self.assertNotIn("9999", "".join(request.series))
        self.assertEqual(request.id_to_symbol, {"A01": "MSFT", "A02": "AAPL"})


class AllocationValidationTests(unittest.TestCase):
    def test_valid_weights_map_back_to_tickers(self) -> None:
        permitted = ("A01", "A02")
        weights = validate_allocation({"A01": 0.25, "A02": 0.75}, permitted, allow_cash=False)
        self.assertEqual(
            translate_weights(weights, {"A01": "MSFT", "A02": "AAPL"}),
            {"MSFT": 0.25, "AAPL": 0.75},
        )

    def test_cash_allocation_is_preserved_only_when_enabled(self) -> None:
        weights = validate_allocation(
            {"A01": 0.0, CASH: 1.0},
            ("A01", CASH),
            allow_cash=True,
        )
        self.assertEqual(translate_weights(weights, {"A01": "AAPL"}), {"AAPL": 0.0, CASH: 1.0})
        with self.assertRaisesRegex(BacktestError, "CASH"):
            validate_allocation({"A01": 0.5, CASH: 0.5}, ("A01",), allow_cash=False)

    def test_invalid_weights_are_rejected(self) -> None:
        permitted = ("A01", "A02")
        cases = [
            {"A01": 0.4, "A02": 0.4},
            {"A01": -0.1, "A02": 1.1},
            {"A01": float("nan"), "A02": 1.0},
            {"A01": True, "A02": 0.0},
            {"A01": 1.0, "A99": 0.0},
            {"A01": 1.0},
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(BacktestError):
                    validate_allocation(payload, permitted, allow_cash=False)


class DecisionTests(unittest.TestCase):
    def test_decision_uses_the_mock_and_returns_tickers(self) -> None:
        seen = {}

        def complete(model, api_key, messages, schema):
            seen["model"] = model
            seen["api_key"] = api_key
            seen["messages"] = messages
            seen["schema"] = schema
            return {"A01": 0.2, "A02": 0.8}

        decide = llm_decision(
            identified=True,
            allow_cash=False,
            model="gpt-test",
            api_key="test-key",
            complete=complete,
        )
        index = pd.bdate_range("2024-01-02", periods=62)
        history = pd.DataFrame(
            {"MSFT": range(1, 63), "AAPL": range(2, 64)},
            index=index,
            dtype=float,
        )
        history.iloc[-1] = 9_999.0
        weights = decide(index[-1], history, ("MSFT", "AAPL"), True)
        self.assertEqual(weights, {"MSFT": 0.2, "AAPL": 0.8})
        self.assertEqual(seen["model"], "gpt-test")
        self.assertEqual(seen["api_key"], "test-key")
        self.assertIn(DOW_30["MSFT"], json.dumps(seen["messages"]))
        self.assertNotIn("9999", json.dumps(seen["messages"]))
        self.assertNotIn(CASH, seen["schema"]["properties"])


class ProgressCountTests(unittest.TestCase):
    def test_completed_decisions_follow_the_common_session_calendar(self) -> None:
        index = pd.to_datetime(
            [
                "2023-12-29",
                "2024-01-02",
                "2024-01-03",
                "2024-02-01",
                "2024-02-05",
                "2024-03-01",
            ]
        )
        prices = pd.DataFrame({"AAA": 100.0, "SPY": 50.0, "DIA": 40.0}, index=index)
        prices.loc[pd.Timestamp("2024-02-01"), "AAA"] = 0.0
        dividends = pd.DataFrame(0.0, index=index, columns=prices.columns)
        start = date(2024, 1, 1)
        end = date(2024, 3, 31)
        dates = llm_decision_dates(prices, ("AAA",), start, end)
        self.assertEqual(
            [stamp.date().isoformat() for stamp in dates],
            ["2024-01-02", "2024-02-05", "2024-03-01"],
        )

        calls = []

        def decide(as_of, history, symbols, is_opening):
            del history, symbols, is_opening
            calls.append(pd.Timestamp(as_of))
            return {"AAA": 1.0}

        simulate(prices, dividends, ["AAA"], 10_000.0, start, end, decide)
        self.assertEqual(calls, dates)
        self.assertEqual(expected_llm_calls(prices, ("AAA",), start, end, 4), 12)

        completed = []

        def fail(as_of, history, symbols, is_opening):
            del as_of, history, symbols, is_opening
            raise BacktestError("The OpenAI request failed.")

        wrapped = track_llm_decision(
            fail,
            on_waiting=lambda as_of: None,
            on_completed=completed.append,
        )
        with self.assertRaises(BacktestError):
            wrapped(dates[0], prices.iloc[:1], ("AAA",), True)
        self.assertEqual(completed, [])


if __name__ == "__main__":
    unittest.main()
