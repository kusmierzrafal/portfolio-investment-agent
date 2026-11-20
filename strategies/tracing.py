"""Optional Langfuse traces for LLM backtests.

Tracing stays off unless both Langfuse keys are set. Nothing here is sent to
the allocation model, and the OpenAI key is never included.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

from langfuse import get_client

from backtesting.engine import BacktestError


def tracing_enabled() -> bool:
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
    return bool(public_key and secret_key)


def _client():
    return get_client()


def _public_error(exc: BaseException) -> str:
    if isinstance(exc, BacktestError):
        return str(exc)
    return "The LLM decision failed."


def _safe_update(observation, **fields) -> None:
    try:
        observation.update(**fields)
    except Exception:
        return


def _flush(client) -> None:
    try:
        client.flush()
    except Exception:
        return


class _Silent:
    def started(self, request) -> None:
        del request

    def finished(self, raw, weights, allocations, usage, cost) -> None:
        del raw, weights, allocations, usage, cost

    def failed(self, request, raw, exc, usage, cost) -> None:
        del request, raw, exc, usage, cost


class _Decision:
    def __init__(self, observation, strategy: str, when: str) -> None:
        self.observation = observation
        self.strategy = strategy
        self.when = when

    def started(self, request) -> None:
        _safe_update(
            self.observation,
            input={"messages": list(request.messages), "schema": request.schema},
            metadata={
                "decision_date": self.when,
                "strategy": self.strategy,
                "synthetic_ids": list(request.permitted),
                "ticker_mapping": dict(request.id_to_symbol),
            },
        )

    def finished(self, raw, weights, allocations, usage, cost) -> None:
        fields = {
            "output": {
                "response": raw,
                "validated_weights": weights,
                "allocations": allocations,
            }
        }
        if usage:
            fields["usage_details"] = usage
        if cost:
            fields["cost_details"] = cost
        _safe_update(self.observation, **fields)

    def failed(self, request, raw, exc, usage, cost) -> None:
        output = {"response": raw}
        if request is not None:
            output["synthetic_ids"] = list(request.permitted)
            output["ticker_mapping"] = dict(request.id_to_symbol)
        fields = {
            "level": "ERROR",
            "status_message": _public_error(exc),
            "output": output,
        }
        if usage:
            fields["usage_details"] = usage
        if cost:
            fields["cost_details"] = cost
        _safe_update(self.observation, **fields)


@contextmanager
def backtest_trace(model: str, strategies: list[str]):
    """One trace for a backtest. Children are the strategy spans."""
    if not tracing_enabled():
        yield
        return
    try:
        client = _client()
        context = client.start_as_current_observation(
            as_type="span",
            name="llm-backtest",
            input={"model": model, "strategies": list(strategies)},
        )
    except Exception:
        yield
        return
    with context as span:
        try:
            yield span
        except Exception as exc:
            _safe_update(span, level="ERROR", status_message=_public_error(exc))
            raise
    _flush(client)


@contextmanager
def strategy_span(name: str):
    """Group one strategy's monthly decisions under the current backtest trace."""
    if not tracing_enabled():
        yield
        return
    try:
        context = _client().start_as_current_observation(as_type="span", name=name)
    except Exception:
        yield
        return
    with context:
        yield


@contextmanager
def traced_decision(strategy: str, model: str, when: str):
    """One generation for a single decision date."""
    if not tracing_enabled():
        yield _Silent()
        return
    try:
        context = _client().start_as_current_observation(
            as_type="generation",
            name=when,
            model=model,
            metadata={"strategy": strategy, "decision_date": when},
        )
    except Exception:
        yield _Silent()
        return
    with context as generation:
        yield _Decision(generation, strategy, when)
