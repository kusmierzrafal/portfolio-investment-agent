"""Backtesting page: configuration, run, and results."""

import os
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtesting.engine import (
    BENCHMARKS,
    BacktestError,
    BacktestResult,
    Holding,
    load_backtest_inputs,
    maximum_drawdown,
    simulate,
    total_return,
)
from data.market import DOW_30, MarketDataError
from strategies.equal_weight import buy_and_hold_decision, monthly_rebalance_decision
from strategies.llm import (
    DEFAULT_MODEL,
    LLM_HISTORY_DAYS,
    LLM_SETTINGS,
    LLM_STRATEGIES,
    llm_decision,
)
from strategies.momentum import MOMENTUM_WARMUP_DAYS, momentum_decision
from ui.methodology import (
    BUY_AND_HOLD,
    MOMENTUM,
    MONTHLY_REBALANCE,
    show_strategy_methodology,
)

DEFAULT_START = date(2024, 1, 1)
DEFAULT_END = date(2024, 12, 31)
DEFAULT_CAPITAL = 10_000.0
ORIGINAL_STRATEGIES = [BUY_AND_HOLD, MONTHLY_REBALANCE, MOMENTUM]
STRATEGIES = ORIGINAL_STRATEGIES + list(LLM_STRATEGIES)


def show_backtest() -> None:
    st.header("Backtesting")
    chosen = st.multiselect(
        "Strategies",
        options=STRATEGIES,
        default=ORIGINAL_STRATEGIES,
        key="backtest_strategies",
    )
    strategies = [name for name in STRATEGIES if name in chosen]
    st.caption("Compare the selected Dow 30 strategies with SPY and DIA.")
    tickers = list(DOW_30)
    with st.form("backtest"):
        selected = st.multiselect(
            "Stocks",
            options=tickers,
            format_func=lambda symbol: f"{symbol} — {DOW_30[symbol]}",
            placeholder="Select Dow 30 stocks",
            key="backtest_symbols",
        )
        start_column, end_column = st.columns(2)
        with start_column:
            start = st.date_input("Start date", value=DEFAULT_START, key="backtest_start")
        with end_column:
            end = st.date_input("End date", value=DEFAULT_END, key="backtest_end")
        capital = st.number_input(
            "Initial capital",
            min_value=0.0,
            value=DEFAULT_CAPITAL,
            step=100.0,
            key="backtest_capital",
        )
        if any(name in LLM_STRATEGIES for name in strategies):
            st.subheader("LLM settings")
            api_key = st.text_input(
                "OpenAI API key",
                type="password",
                key="backtest_openai_api_key",
            )
            model = st.text_input("Model ID", value=DEFAULT_MODEL, key="backtest_openai_model")
        else:
            api_key, model = "", DEFAULT_MODEL
        submitted = st.form_submit_button("Run backtest", type="primary")

    with st.expander("How does this strategy work?", expanded=False):
        show_strategy_methodology(strategies)

    if not submitted and st.session_state.get("backtest_shown_strategies") != tuple(strategies):
        st.session_state.pop("backtest_results", None)
        st.session_state.pop("backtest_shown_strategies", None)

    guidance = st.empty()
    if submitted:
        if not strategies:
            st.session_state.pop("backtest_results", None)
            st.session_state.pop("backtest_shown_strategies", None)
            st.error("Select at least one strategy.")
            return
        guidance.empty()
        try:
            if any(name in LLM_STRATEGIES for name in strategies):
                results = _run_llm_backtest(
                    selected,
                    start,
                    end,
                    capital,
                    strategies,
                    model=model,
                    api_key=api_key,
                )
            else:
                with st.spinner("Downloading prices and running the backtest..."):
                    results = _run_selected(
                        selected,
                        start,
                        end,
                        capital,
                        strategies,
                        model=model,
                        api_key=api_key,
                    )
        except ValueError as exc:
            st.session_state.pop("backtest_results", None)
            st.session_state.pop("backtest_shown_strategies", None)
            st.error(str(exc))
            return
        except (MarketDataError, BacktestError) as exc:
            st.session_state.pop("backtest_results", None)
            st.session_state.pop("backtest_shown_strategies", None)
            st.warning(str(exc))
            return
        st.session_state["backtest_results"] = results
        st.session_state["backtest_shown_strategies"] = tuple(strategies)

    results = st.session_state.get("backtest_results")
    if not results:
        guidance.info("Select at least one stock and one strategy, then run the backtest.")
        return
    _show_backtest_result(results)


def _run_selected(
    symbols: list[str],
    start: date,
    end: date,
    capital: float,
    strategies: list[str],
    *,
    model: str,
    api_key: str,
    progress: "_LlmProgress | None" = None,
) -> list[tuple[str, BacktestResult]]:
    history_start = None
    if MOMENTUM in strategies:
        history_start = start - timedelta(days=MOMENTUM_WARMUP_DAYS)
    llm_names = [name for name in strategies if name in LLM_STRATEGIES]
    resolved_key = ""
    if llm_names:
        resolved_key = _require_llm_credentials(model, api_key)
        llm_start = start - timedelta(days=LLM_HISTORY_DAYS)
        history_start = llm_start if history_start is None else min(history_start, llm_start)
    if progress is not None:
        progress.downloading()
    selected, prices, dividends = load_backtest_inputs(
        symbols,
        start,
        end,
        capital,
        history_start=history_start,
    )
    if progress is not None:
        progress.ready(prices, selected, start, end, len(llm_names))
    decisions = {
        BUY_AND_HOLD: buy_and_hold_decision,
        MONTHLY_REBALANCE: monthly_rebalance_decision,
        MOMENTUM: momentum_decision,
    }
    for name in llm_names:
        identified, allow_cash = LLM_SETTINGS[name]
        decide = llm_decision(
            identified=identified,
            allow_cash=allow_cash,
            model=model.strip(),
            api_key=resolved_key,
        )
        decisions[name] = decide if progress is None else progress.wrap(name, decide)
    results = [
        (
            name,
            simulate(prices, dividends, selected, capital, start, end, decisions[name]),
        )
        for name in strategies
    ]
    _require_shared_calendar(results)
    return results


def _run_llm_backtest(
    symbols: list[str],
    start: date,
    end: date,
    capital: float,
    strategies: list[str],
    *,
    model: str,
    api_key: str,
) -> list[tuple[str, BacktestResult]]:
    with st.status("Running the backtest.", expanded=True) as status:
        progress = _LlmProgress(status)
        try:
            results = _run_selected(
                symbols,
                start,
                end,
                capital,
                strategies,
                model=model,
                api_key=api_key,
                progress=progress,
            )
        except Exception:
            progress.fail()
            raise
        progress.finish()
        return results


def llm_decision_dates(
    prices: pd.DataFrame,
    symbols: tuple[str, ...] | list[str],
    start: date,
    end: date,
) -> list[pd.Timestamp]:
    """Sessions where the engine asks for a decision.

    That is the first common session in the range, then the first common
    session of each later month. A session counts only when every selected
    stock and both benchmarks have a positive close.
    """
    required = list(symbols) + [symbol for symbol in BENCHMARKS if symbol not in symbols]
    closes = prices.copy()
    closes.index = pd.DatetimeIndex(pd.to_datetime(closes.index))
    if closes.index.tz is not None:
        closes.index = closes.index.tz_localize(None)
    closes.index = closes.index.normalize()
    closes = closes.sort_index()
    window = closes.loc[
        (closes.index >= pd.Timestamp(start)) & (closes.index <= pd.Timestamp(end)),
        required,
    ]
    available = (window.notna() & (window > 0)).all(axis=1)
    dates = window.index[available]
    if len(dates) == 0:
        return []
    purchase = dates[0]
    decisions = [purchase]
    for current in dates[1:]:
        if (current.year, current.month) <= (purchase.year, purchase.month):
            continue
        month = dates[(dates.year == current.year) & (dates.month == current.month)]
        if len(month) > 0 and current == month[0]:
            decisions.append(current)
    return decisions


def expected_llm_calls(
    prices: pd.DataFrame,
    symbols: tuple[str, ...] | list[str],
    start: date,
    end: date,
    strategy_count: int,
) -> int:
    """Monthly decision dates times the number of selected LLM strategies."""
    return len(llm_decision_dates(prices, symbols, start, end)) * strategy_count


def track_llm_decision(decide, on_waiting, on_completed):
    """Call the model, then record the decision only if it returns."""

    def wrapped(as_of, history, symbols, is_opening):
        on_waiting(as_of)
        weights = decide(as_of, history, symbols, is_opening)
        on_completed(as_of)
        return weights

    return wrapped


class _LlmProgress:
    def __init__(self, status) -> None:
        self.status = status
        self.completed = 0
        self.total = 0
        self.bar = None

    def downloading(self) -> None:
        self.status.update(label="Downloading historical market data.", state="running")

    def ready(self, prices, symbols, start: date, end: date, strategy_count: int) -> None:
        self.total = expected_llm_calls(prices, symbols, start, end, strategy_count)
        self.bar = self.status.progress(0.0, text=self._text())
        self.status.update(label=self._text(), state="running")

    def wrap(self, name: str, decide):
        def on_waiting(as_of) -> None:
            when = pd.Timestamp(as_of).strftime("%Y-%m-%d")
            self.status.update(
                label=f"{name} — {when}. Waiting for OpenAI response.",
                state="running",
            )

        def on_completed(as_of) -> None:
            del as_of
            self.completed += 1
            fraction = 0.0 if self.total == 0 else self.completed / self.total
            self.bar.progress(fraction, text=self._text())
            self.status.update(label=self._text(), state="running")

        return track_llm_decision(decide, on_waiting, on_completed)

    def finish(self) -> None:
        self.status.update(label="Completed", state="complete", expanded=False)

    def fail(self) -> None:
        self.status.update(label="Failed", state="error", expanded=True)

    def _text(self) -> str:
        return f"{self.completed} / {self.total} LLM decisions"


def _require_llm_credentials(model: str, api_key: str) -> str:
    resolved_key = _api_key(api_key)
    if not resolved_key:
        raise ValueError(
            "Add an OpenAI API key, or set OPENAI_API_KEY, to run an LLM strategy."
        )
    if not model.strip():
        raise ValueError("Enter a model ID.")
    return resolved_key


def _api_key(typed: str) -> str:
    cleaned = typed.strip()
    if cleaned:
        return cleaned
    return os.environ.get("OPENAI_API_KEY", "").strip()


def _require_shared_calendar(results: list[tuple[str, BacktestResult]]) -> None:
    first = results[0][1]
    for _, result in results[1:]:
        shared = (
            result.purchase_date == first.purchase_date
            and result.values.index.equals(first.values.index)
            and result.values["SPY"].equals(first.values["SPY"])
            and result.values["DIA"].equals(first.values["DIA"])
        )
        if not shared:
            raise BacktestError(
                "The strategies do not share a purchase date, trading calendar, and benchmarks."
            )


def _show_backtest_result(results: list[tuple[str, BacktestResult]]) -> None:
    first = results[0][1]
    names = [name for name, _ in results]
    purchased = first.purchase_date.date().isoformat()
    if names == [MONTHLY_REBALANCE]:
        st.caption(
            f"Bought at the close on {purchased}. "
            "Rebalanced at the first common session of each later month."
        )
    elif names == [BUY_AND_HOLD]:
        st.caption(f"Bought at the close on {purchased}. No later trades are made.")
    elif names == [MOMENTUM]:
        st.caption(
            f"Bought at the close on {purchased}. "
            "That session and the first common session of each later month "
            "hold the two stocks with the strongest 60-session momentum."
        )
    elif names == [BUY_AND_HOLD, MONTHLY_REBALANCE]:
        st.caption(
            f"Both strategies buy at the close on {purchased}. "
            "Monthly rebalancing trades again on the first common session of each later month."
        )
    elif any(name in LLM_STRATEGIES for name in names):
        st.caption(
            f"All selected strategies are invested from the close on {purchased}. "
            "LLM strategies choose weights on that session and on the first common "
            "session of each later month."
        )
    else:
        st.caption(
            f"All selected strategies are invested from the close on {purchased}. "
            "Momentum Monthly selects two stocks on that session and on the "
            "first common session of each later month."
        )

    chart = pd.DataFrame(index=first.values.index)
    rows = []
    for name, result in results:
        series = result.values["Portfolio"]
        chart[name] = series
        rows.append(_summary_row(name, series, result.initial_capital))
    for benchmark in BENCHMARKS:
        series = first.values[benchmark]
        chart[benchmark] = series
        rows.append(_summary_row(benchmark, series, first.initial_capital))

    figure = go.Figure()
    for name in chart.columns:
        figure.add_trace(
            go.Scatter(x=chart.index, y=chart[name], mode="lines", name=name)
        )
    figure.update_layout(
        title="Portfolio value",
        xaxis_title="Date",
        yaxis_title="Value (USD)",
        hovermode="x unified",
    )
    st.plotly_chart(figure)
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        column_config={
            "Final value": st.column_config.NumberColumn(format="dollar"),
            "Total return": st.column_config.NumberColumn(format="percent"),
            "Maximum drawdown": st.column_config.NumberColumn(format="percent"),
        },
    )

    if len(names) == 1:
        detail_name = names[0]
    else:
        if st.session_state.get("backtest_detail_strategy") not in names:
            st.session_state["backtest_detail_strategy"] = names[0]
        detail_name = st.selectbox("Details", names, key="backtest_detail_strategy")
    _show_strategy_details(detail_name, dict(results)[detail_name])


def _summary_row(name: str, series: pd.Series, initial_capital: float) -> dict:
    final_value = float(series.iloc[-1])
    return {
        "Name": name,
        "Final value": final_value,
        "Total return": total_return(final_value, initial_capital),
        "Maximum drawdown": maximum_drawdown(series),
    }


def _show_strategy_details(strategy: str, result: BacktestResult) -> None:
    if strategy != BUY_AND_HOLD:
        st.subheader("Holdings")
        st.caption("Shares and prices at the last reported session.")
        st.dataframe(
            _holdings_table(result.holdings),
            hide_index=True,
            column_config={
                "Shares": st.column_config.NumberColumn(format="%.4f"),
                "Price": st.column_config.NumberColumn(format="dollar"),
                "Market value": st.column_config.NumberColumn(format="dollar"),
            },
        )
        st.caption(f"Cash: ${result.cash:,.2f}")
    else:
        st.subheader("Initial allocations")
        st.dataframe(
            _positions_table(result.positions),
            hide_index=True,
            column_config=_position_column_config(),
        )
    st.subheader("Transactions")
    st.dataframe(
        _transactions_table(result),
        hide_index=True,
        column_config={
            "Date": st.column_config.DateColumn("Date"),
            "Shares": st.column_config.NumberColumn(format="%.4f"),
            "Amount": st.column_config.NumberColumn(format="dollar"),
            "Fee": st.column_config.NumberColumn(format="dollar"),
        },
    )
    st.subheader("Benchmarks")
    st.caption("SPY and DIA are not portfolio holdings. Each is bought with the full initial capital.")
    st.dataframe(
        _positions_table(result.benchmarks),
        hide_index=True,
        column_config=_position_column_config(),
    )

    st.subheader("Daily portfolio value")
    daily = result.values.rename(columns={"Portfolio": strategy}).reset_index()
    value_format = {
        column: st.column_config.NumberColumn(format="dollar")
        for column in daily.columns
        if column != "Date"
    }
    st.dataframe(
        daily,
        hide_index=True,
        column_config={"Date": st.column_config.DateColumn("Date"), **value_format},
    )


def _positions_table(positions) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Symbol": position.symbol,
                "Allocation": position.allocation,
                "Price": position.price,
                "Shares": position.shares,
                "Fee": position.fee,
            }
            for position in positions
        ]
    )


def _position_column_config() -> dict:
    money = st.column_config.NumberColumn(format="dollar")
    return {
        "Allocation": money,
        "Price": money,
        "Shares": st.column_config.NumberColumn(format="%.4f"),
        "Fee": money,
    }


def _holdings_table(holdings: tuple[Holding, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Symbol": holding.symbol,
                "Shares": holding.shares,
                "Price": holding.price,
                "Market value": holding.market_value,
            }
            for holding in holdings
        ]
    )


def _transactions_table(result: BacktestResult) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Date": trade.date,
                "Ticker": trade.symbol,
                "Side": trade.side,
                "Shares": trade.shares,
                "Amount": trade.amount,
                "Fee": trade.fee,
            }
            for trade in result.transactions
        ]
    )
