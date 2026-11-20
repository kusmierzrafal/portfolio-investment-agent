"""Langfuse tracing stays off without credentials and records decisions without the API key."""

from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from backtesting.engine import BacktestError
from data.market import DOW_30
from strategies.llm import ANONYMIZED_INVESTED, _call_usage, _usage_and_cost, llm_decision
from strategies.tracing import backtest_trace, strategy_span


def _history() -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", periods=62)
    history = pd.DataFrame(
        {"MSFT": range(1, 63), "AAPL": range(2, 64)},
        index=index,
        dtype=float,
    )
    history.iloc[-1] = 9_999.0
    return history


class _Observation:
    def __init__(self, client: "_Client", kind: str, name: str, parent: str | None, initial: dict) -> None:
        self.kind = kind
        self.name = name
        self.parent = parent
        self.initial = initial
        self.updates: list[dict] = []
        self.client = client

    def update(self, **fields) -> None:
        self.updates.append(fields)

    def __enter__(self):
        self.client.stack.append(self)
        return self

    def __exit__(self, exc_type, exc, tb):
        self.client.stack.pop()
        return False


class _Client:
    def __init__(self) -> None:
        self.stack: list[_Observation] = []
        self.created: list[_Observation] = []
        self.flushed = False

    def start_as_current_observation(self, *, name: str, as_type: str = "span", **fields):
        parent = self.stack[-1].name if self.stack else None
        observation = _Observation(self, as_type, name, parent, fields)
        self.created.append(observation)
        return observation

    def flush(self) -> None:
        self.flushed = True


class TracingTests(unittest.TestCase):
    def test_missing_credentials_do_not_create_a_client(self) -> None:
        history = _history()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LANGFUSE_PUBLIC_KEY", None)
            os.environ.pop("LANGFUSE_SECRET_KEY", None)
            with patch("strategies.tracing._client", side_effect=AssertionError("client")):
                decide = llm_decision(
                    identified=False,
                    allow_cash=False,
                    model="gpt-test",
                    api_key="sk-test-secret",
                    complete=lambda model, api_key, messages, schema: {"A01": 0.25, "A02": 0.75},
                )
                weights = decide(history.index[-1], history, ("MSFT", "AAPL"), True)
        self.assertEqual(weights, {"MSFT": 0.25, "AAPL": 0.75})

    def test_trace_groups_decisions_without_the_api_key_or_anonymized_names(self) -> None:
        history = _history()
        client = _Client()
        secret = "sk-test-secret"

        answers = [
            {"A01": 0.25, "A02": 0.75},
            {"A01": 0.4, "A02": 0.4},
        ]

        def complete(model, api_key, messages, schema):
            self.assertEqual(api_key, secret)
            self.assertNotIn("MSFT", json.dumps(messages))
            self.assertNotIn(DOW_30["MSFT"], json.dumps(messages))
            payload = answers.pop(0)
            if payload["A01"] + payload["A02"] == 1:
                _call_usage.set(({"input": 11, "output": 3}, {"input": 0.01}))
            return payload

        decide = llm_decision(
            identified=False,
            allow_cash=False,
            model="gpt-test",
            api_key=secret,
            complete=complete,
        )
        with (
            patch("strategies.tracing.tracing_enabled", return_value=True),
            patch("strategies.tracing._client", return_value=client),
        ):
            with backtest_trace("gpt-test", [ANONYMIZED_INVESTED]):
                with strategy_span(ANONYMIZED_INVESTED):
                    weights = decide(history.index[-1], history, ("MSFT", "AAPL"), True)
                    with self.assertRaises(BacktestError):
                        decide(history.index[-1], history, ("MSFT", "AAPL"), False)

        self.assertEqual(weights, {"MSFT": 0.25, "AAPL": 0.75})
        self.assertTrue(client.flushed)
        self.assertEqual(
            [(item.kind, item.name, item.parent) for item in client.created],
            [
                ("span", "llm-backtest", None),
                ("span", ANONYMIZED_INVESTED, "llm-backtest"),
                ("generation", str(history.index[-1].date()), ANONYMIZED_INVESTED),
                ("generation", str(history.index[-1].date()), ANONYMIZED_INVESTED),
            ],
        )
        success = client.created[2]
        failure = client.created[3]
        prompt = json.dumps(success.updates[0]["input"])
        self.assertNotIn("MSFT", prompt)
        self.assertNotIn("AAPL", prompt)
        self.assertNotIn("Apple", prompt)
        self.assertNotIn("Microsoft", prompt)
        self.assertEqual(
            success.updates[0]["metadata"]["ticker_mapping"],
            {"A01": "MSFT", "A02": "AAPL"},
        )
        self.assertEqual(success.updates[1]["usage_details"], {"input": 11, "output": 3})
        self.assertEqual(success.updates[1]["cost_details"], {"input": 0.01})
        self.assertEqual(
            success.updates[1]["output"]["allocations"],
            {"MSFT": 0.25, "AAPL": 0.75},
        )
        self.assertEqual(success.updates[1]["output"]["validated_weights"], {"A01": 0.25, "A02": 0.75})
        self.assertEqual(failure.updates[-1]["level"], "ERROR")
        self.assertIn("sum to 1", failure.updates[-1]["status_message"])
        self.assertEqual(failure.updates[-1]["output"]["response"], {"A01": 0.4, "A02": 0.4})
        recorded = json.dumps(
            [
                {"initial": item.initial, "updates": item.updates}
                for item in client.created
            ]
        )
        self.assertNotIn(secret, recorded)

    def test_usage_comes_from_the_model_response(self) -> None:
        response = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=4,
                input_tokens_details=SimpleNamespace(cached_tokens=2),
            )
        )
        usage, cost = _usage_and_cost(response)
        self.assertEqual(usage, {"input": 10, "output": 4, "cache_read_input_tokens": 2})
        self.assertIsNone(cost)


if __name__ == "__main__":
    unittest.main()
