"""LLM allocation decisions.

One implementation covers identified and anonymized inputs, with or without
cash. The model sees scaled closes and returns weights. Python checks those
weights before they reach the portfolio engine.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import pandas as pd
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    NotFoundError,
    OpenAI,
    RateLimitError,
)

from backtesting.engine import WEIGHT_TOLERANCE, BacktestError, CASH
from data.market import DOW_30

IDENTIFIED_INVESTED = "LLM Identified — Fully Invested"
IDENTIFIED_CASH = "LLM Identified — CASH Allowed"
ANONYMIZED_INVESTED = "LLM Anonymized — Fully Invested"
ANONYMIZED_CASH = "LLM Anonymized — CASH Allowed"
LLM_STRATEGIES: tuple[str, ...] = (
    IDENTIFIED_INVESTED,
    IDENTIFIED_CASH,
    ANONYMIZED_INVESTED,
    ANONYMIZED_CASH,
)
LLM_SETTINGS: dict[str, tuple[bool, bool]] = {
    IDENTIFIED_INVESTED: (True, False),
    IDENTIFIED_CASH: (True, True),
    ANONYMIZED_INVESTED: (False, False),
    ANONYMIZED_CASH: (False, True),
}

LLM_LOOKBACK = 60
LLM_HISTORY_DAYS = (LLM_LOOKBACK + 1) * 3
DEFAULT_MODEL = "gpt-6-luna"

_SYSTEM_PROMPT = (
    "Allocate the portfolio across the listed asset IDs. "
    "Return one weight for every listed ID. "
    "Each weight is nonnegative and the weights sum to 1."
)


@dataclass(frozen=True)
class AllocationRequest:
    messages: tuple[dict[str, str], ...]
    schema: dict
    permitted: tuple[str, ...]
    id_to_symbol: dict[str, str]
    series: tuple[str, ...]


def llm_decision(
    *,
    identified: bool,
    allow_cash: bool,
    model: str,
    api_key: str,
    complete=None,
):
    """Return a decide(as_of, history, symbols, is_opening) function."""
    caller = complete_openai if complete is None else complete

    def decide(as_of, history, symbols, is_opening):
        del is_opening
        request = build_allocation_request(
            history,
            symbols,
            as_of,
            identified=identified,
            allow_cash=allow_cash,
        )
        raw = caller(model, api_key, request.messages, request.schema)
        weights = validate_allocation(raw, request.permitted, allow_cash=allow_cash)
        return translate_weights(weights, request.id_to_symbol)

    return decide


def build_allocation_request(
    history: pd.DataFrame,
    symbols: tuple[str, ...],
    as_of,
    *,
    identified: bool,
    allow_cash: bool,
) -> AllocationRequest:
    """Build the model request from closes strictly before ``as_of``."""
    prior = history.loc[history.index < pd.Timestamp(as_of)]
    window = _shared_window(prior, symbols, as_of)
    labels = [(f"A{index:02d}", symbol) for index, symbol in enumerate(symbols, start=1)]
    series: list[str] = []
    lines = ["Scaled closes are listed from oldest to newest."]
    for asset_id, symbol in labels:
        text = ", ".join(f"{value:.6f}" for value in window[symbol])
        series.append(text)
        if identified:
            lines.append(f"{asset_id} {symbol} {DOW_30[symbol]}")
        else:
            lines.append(asset_id)
        lines.append(text)
    permitted = [asset_id for asset_id, _ in labels]
    if allow_cash:
        permitted.append(CASH)
        lines.append(CASH)
    return AllocationRequest(
        messages=(
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(lines)},
        ),
        schema=_allocation_schema(permitted),
        permitted=tuple(permitted),
        id_to_symbol={asset_id: symbol for asset_id, symbol in labels},
        series=tuple(series),
    )


def validate_allocation(payload, permitted: tuple[str, ...], *, allow_cash: bool) -> dict[str, float]:
    """Accept only finite nonnegative weights for the permitted IDs."""
    if not isinstance(payload, dict):
        raise BacktestError("The allocation must be an object of asset weights.")
    if not allow_cash and (CASH in payload or any(str(key).strip().upper() == CASH for key in payload)):
        raise BacktestError("CASH is not allowed for a fully invested allocation.")
    unknown = [str(key) for key in payload if key not in permitted]
    if unknown:
        raise BacktestError(f"The allocation uses unknown assets: {', '.join(unknown)}.")
    missing = [asset for asset in permitted if asset not in payload]
    if missing:
        raise BacktestError(f"The allocation is missing assets: {', '.join(missing)}.")
    weights: dict[str, float] = {}
    for asset in permitted:
        value = payload[asset]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise BacktestError(f"The weight for {asset} must be a finite number.")
        if value < 0.0:
            raise BacktestError(f"The weight for {asset} must be nonnegative.")
        weights[asset] = float(value)
    if abs(sum(weights.values()) - 1.0) > WEIGHT_TOLERANCE:
        raise BacktestError("Allocation weights must sum to 1.")
    return weights


def translate_weights(weights: dict[str, float], id_to_symbol: dict[str, str]) -> dict[str, float]:
    """Replace synthetic IDs with tickers. CASH stays CASH."""
    translated: dict[str, float] = {}
    for asset, weight in weights.items():
        translated[CASH if asset == CASH else id_to_symbol[asset]] = weight
    return translated


def complete_openai(model: str, api_key: str, messages, schema: dict) -> dict:
    """Request one structured allocation. The API key is not included in errors."""
    client = OpenAI(api_key=api_key)
    try:
        response = client.responses.create(
            model=model,
            input=list(messages),
            text={
                "format": {
                    "type": "json_schema",
                    "name": "portfolio_allocation",
                    "strict": True,
                    "schema": schema,
                }
            },
        )
    except AuthenticationError as exc:
        raise BacktestError("The OpenAI API key was rejected.") from exc
    except NotFoundError as exc:
        raise BacktestError("The model ID is not available.") from exc
    except (APIConnectionError, APITimeoutError, RateLimitError, APIError) as exc:
        raise BacktestError("The OpenAI request failed.") from exc
    if getattr(response, "status", None) == "incomplete":
        raise BacktestError("The model returned an incomplete allocation.")
    message = next(
        (item for item in response.output if getattr(item, "type", None) == "message"),
        None,
    )
    content = message.content[0] if message is not None and message.content else None
    if content is None:
        raise BacktestError("The model returned no allocation.")
    if content.type == "refusal":
        raise BacktestError("The model refused to return an allocation.")
    if content.type != "output_text" or not content.text:
        raise BacktestError("The model returned no allocation.")
    try:
        parsed = json.loads(content.text)
    except json.JSONDecodeError as exc:
        raise BacktestError("The model returned an allocation that was not valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise BacktestError("The model returned an allocation that was not an object.")
    return parsed


def _shared_window(history: pd.DataFrame, symbols: tuple[str, ...], as_of) -> pd.DataFrame:
    """Last 61 dates on which every stock has a positive close, each scaled to 100."""
    needed = LLM_LOOKBACK + 1
    values = history.loc[:, list(symbols)].sort_index()
    valid = values.notna() & (values > 0) & (values < math.inf)
    common = values.loc[valid.all(axis=1)]
    if len(common) < needed:
        when = pd.Timestamp(as_of).date().isoformat()
        raise BacktestError(
            f"Not enough common completed sessions before {when} to build a "
            f"{LLM_LOOKBACK}-session history for {', '.join(symbols)}. "
            "The lookback was not shortened."
        )
    window = common.iloc[-needed:]
    return 100.0 * window / window.iloc[0]


def _allocation_schema(permitted: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {asset: {"type": "number"} for asset in permitted},
        "required": list(permitted),
        "additionalProperties": False,
    }
