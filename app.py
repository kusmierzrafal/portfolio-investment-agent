"""Portfolio Investment Agent."""

import textwrap
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtest import (
    BENCHMARKS,
    FEE_RATE,
    BacktestError,
    BacktestResult,
    Holding,
    Transaction,
    load_backtest_inputs,
    maximum_drawdown,
    simulate_equal_weight,
    simulate_equal_weight_monthly,
    total_return,
)
from market_data import DOW_30, MarketDataError, download_daily_history

DEFAULT_TICKER = "NVDA"
DEFAULT_START = date(2024, 1, 1)
DEFAULT_END = date(2024, 12, 31)
DEFAULT_CAPITAL = 10_000.0
BUY_AND_HOLD = "Equal Weight Buy & Hold"
MONTHLY_REBALANCE = "Equal Weight Monthly Rebalancing"
STRATEGIES = [BUY_AND_HOLD, MONTHLY_REBALANCE]


def _show_history(ticker: str, history) -> None:
    latest_close = float(history["Close"].iloc[-1])
    return_column = "Adj Close" if "Adj Close" in history.columns else "Close"
    prices = history[return_column].dropna()
    period_return = None
    if not prices.empty and float(prices.iloc[0]) != 0:
        period_return = float(prices.iloc[-1] / prices.iloc[0] - 1)

    latest_column, return_column_ui, sessions_column = st.columns(3)
    latest_column.metric("Latest close", f"${latest_close:,.2f}")
    return_column_ui.metric(
        "Period return",
        "—" if period_return is None else f"{period_return:+.2%}",
        help="Change in adjusted close from the first session to the last.",
    )
    sessions_column.metric("Trading sessions", f"{len(history):,}")
    st.caption(
        "Prices are split-adjusted. Period return uses dividend-adjusted closing prices."
    )

    price_chart = go.Figure(
        data=[
            go.Candlestick(
                x=history.index,
                open=history["Open"],
                high=history["High"],
                low=history["Low"],
                close=history["Close"],
                name=ticker,
            )
        ]
    )
    price_chart.update_layout(
        title=f"{ticker} — {DOW_30[ticker]}",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )
    st.plotly_chart(price_chart)

    if "Volume" in history.columns:
        volume_chart = go.Figure(
            data=[go.Bar(x=history.index, y=history["Volume"], name="Volume")]
        )
        volume_chart.update_layout(
            title="Trading volume",
            xaxis_title="Date",
            yaxis_title="Shares",
            hovermode="x unified",
        )
        st.plotly_chart(volume_chart)

    table = history.reset_index()
    column_config: dict = {
        "Date": st.column_config.DateColumn("Date"),
    }
    for column in ("Open", "High", "Low", "Close", "Adj Close"):
        if column in table.columns:
            column_config[column] = st.column_config.NumberColumn(format="%.2f")
    if "Volume" in table.columns:
        column_config["Volume"] = st.column_config.NumberColumn(format="%d")
    for column in ("Dividends", "Stock Splits"):
        if column in table.columns:
            column_config[column] = st.column_config.NumberColumn(format="%.4f")

    st.subheader("Historical data")
    st.dataframe(
        table,
        hide_index=True,
        column_config=column_config,
    )


def _show_market_explorer() -> None:
    st.header("Market Explorer")
    st.caption(
        "Daily prices for the current Dow Jones Industrial Average constituents."
    )

    tickers = list(DOW_30)
    with st.form("market_explorer"):
        ticker = st.selectbox(
            "Ticker",
            options=tickers,
            index=tickers.index(DEFAULT_TICKER),
            format_func=lambda symbol: f"{symbol} — {DOW_30[symbol]}",
            key="explorer_ticker",
        )
        start_column, end_column = st.columns(2)
        with start_column:
            start = st.date_input("Start date", value=DEFAULT_START, key="explorer_start")
        with end_column:
            end = st.date_input("End date", value=DEFAULT_END, key="explorer_end")
        submitted = st.form_submit_button("Download", type="primary")

    if not submitted:
        st.info("Choose a ticker and a date range, then download daily prices.")
        return

    try:
        history = download_daily_history(ticker, start, end)
    except ValueError as exc:
        st.error(str(exc))
    except MarketDataError as exc:
        st.warning(str(exc))
    else:
        _show_history(ticker, history)


def _show_backtest() -> None:
    st.header("Backtesting")
    chosen = st.multiselect(
        "Strategies",
        options=STRATEGIES,
        default=STRATEGIES,
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
        submitted = st.form_submit_button("Run backtest", type="primary")

    with st.expander("How does this strategy work?", expanded=False):
        _show_strategy_methodology(strategies)

    if not submitted and st.session_state.get("backtest_shown_strategies") != tuple(strategies):
        st.session_state.pop("backtest_results", None)
        st.session_state.pop("backtest_shown_strategies", None)

    if submitted:
        if not strategies:
            st.session_state.pop("backtest_results", None)
            st.session_state.pop("backtest_shown_strategies", None)
            st.error("Select at least one strategy.")
            return
        try:
            with st.spinner("Downloading prices and running the backtest..."):
                results = _run_selected(selected, start, end, capital, strategies)
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
        st.info("Select at least one stock and one strategy, then run the backtest.")
        return
    _show_backtest_result(results)


def _run_selected(
    symbols: list[str],
    start: date,
    end: date,
    capital: float,
    strategies: list[str],
) -> list[tuple[str, BacktestResult]]:
    selected, prices, dividends = load_backtest_inputs(symbols, start, end, capital)
    results: list[tuple[str, BacktestResult]] = []
    if BUY_AND_HOLD in strategies:
        results.append(
            (
                BUY_AND_HOLD,
                simulate_equal_weight(prices, dividends, selected, capital, start, end),
            )
        )
    if MONTHLY_REBALANCE in strategies:
        results.append(
            (
                MONTHLY_REBALANCE,
                simulate_equal_weight_monthly(prices, dividends, selected, capital, start, end),
            )
        )
    _require_shared_calendar(results)
    return results


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
    else:
        st.caption(
            f"Both strategies buy at the close on {purchased}. "
            "Monthly rebalancing trades again on the first common session of each later month."
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
    if strategy == MONTHLY_REBALANCE:
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
    trades = result.transactions or tuple(
        Transaction(
            date=result.purchase_date,
            symbol=position.symbol,
            side="BUY",
            shares=position.shares,
            amount=position.shares * position.price,
            fee=position.fee,
        )
        for position in result.positions
    )
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
            for trade in trades
        ]
    )


def _show_strategy_methodology(strategies: list[str]) -> None:
    if not strategies:
        st.caption("Select a strategy.")
        return
    if BUY_AND_HOLD in strategies:
        if len(strategies) > 1:
            st.markdown(f"**{BUY_AND_HOLD}**")
        _show_buy_and_hold_methodology()
    if MONTHLY_REBALANCE in strategies:
        if len(strategies) > 1:
            st.markdown(f"**{MONTHLY_REBALANCE}**")
        _show_monthly_methodology()


def _show_buy_and_hold_methodology() -> None:
    fee = f"{FEE_RATE:.1%}"
    benchmarks = " and ".join(BENCHMARKS)
    st.markdown(
        textwrap.dedent(
            f"""
The portfolio buys the selected Dow 30 stocks once and holds them through the selected end date. The catalogue is the current static membership, not the historical index.

**Allocation.** Initial capital is split equally. Each selected stock is allocated initial capital divided by the number of stocks.

**Execution.** Shares are bought at the closing price on the first session in the requested range when every selected stock and both benchmarks have a positive close. Nothing is bought or sold after that session. Fractional shares are allowed, and the share count does not change. There is no rebalancing.

**Fees.** Each purchase pays a {fee} fee, deducted from the cash allocated to that purchase before the share count is calculated. {benchmarks} use the same fee. Each benchmark is bought with the full initial capital, so its fee is {fee} of initial capital. The portfolio's fees add up to the same amount.

**Prices.** Positions are valued with the split-adjusted close from the market-data download (`Close`, requested with `auto_adjust=False`). Share counts are not changed again on split dates, because those closes are already split-adjusted. Dividend-adjusted close is not used.

**Dividends.** Cash dividends come from the `Dividends` column of the same download, in split-adjusted dollars per share. They are added as cash and are not reinvested. If that ex-date is not a reported session, the cash is included from the next reported session. A dividend on the purchase session is not received, because the order is filled at that session's close. Dividends are not also embedded in the price, so each dividend is counted once.

**Benchmarks.** {benchmarks} are not holdings. Each is a separate buy-and-hold of the same initial capital, purchased on the same session, with the same fee and dividend treatment.

**Reported sessions.** A session appears in the results only when every selected stock and both benchmarks have a positive close. Sessions with missing prices are omitted rather than filled forward.

**Statistics.** Total return is final value divided by initial capital, minus one, so the purchase fee reduces it. Maximum drawdown is the lowest value of value / running peak minus one. It is zero when value never declines. The running peak starts at the post-fee purchase value.

**Limitations.** The simulation has no taxes, slippage, borrowing, or interest on dividend cash. It does not reinvest dividends, rebalance, short, or optimize weights. Results depend on Yahoo Finance for the requested range.
"""
        ).strip()
    )


def _show_monthly_methodology() -> None:
    fee = f"{FEE_RATE:.1%}"
    benchmarks = " and ".join(BENCHMARKS)
    st.markdown(
        textwrap.dedent(
            f"""
The portfolio opens like equal-weight buy and hold, then returns to equal weights at the first common session of each later month. The catalogue is the current static Dow 30 membership, not the historical index.

**Opening trade.** Initial capital is split equally. Shares are bought at the close on the first session in the requested range when every selected stock and both benchmarks have a positive close. A {fee} fee is deducted from each allocation before the share count is calculated. Fractional shares are allowed. That session is not a rebalance.

**Rebalance.** At the close of the first common session of each later calendar month, every selected stock is set to the same market value. The target uses the whole portfolio, including dividend cash. Overweight stocks are sold and underweight stocks are bought. A month with no common session is skipped. Nothing is traded after the selected end date.

**Fees.** A {fee} fee is charged on the amount actually traded, on both buys and sells. A buy uses the opening-trade rule: the fee comes out of the cash spent, and the shares are the remainder divided by the close. A sale pays the fee out of the proceeds. A holding that is not traded pays no fee. Share quantities are solved so the fees can be paid from available cash, without spending more cash than is on hand and without selling more shares than are held.

**Prices and dividends.** Positions are valued with the split-adjusted close (`Close`, requested with `auto_adjust=False`). Dividend cash comes from the `Dividends` column of the same download and is not also embedded in the close. Cash is held until the next rebalance, then included in the target. A dividend on the opening session is not received. A dividend on a rebalance session belongs to the shares held before that trade. If the ex-date is not a reported session, the cash is included on the next reported session.

**Benchmarks.** {benchmarks} are not holdings and are not rebalanced. Each is a separate buy-and-hold of the full initial capital, purchased on the opening session, with the same fee and dividend treatment.

**Reported sessions.** A session appears only when every selected stock and both benchmarks have a positive close. Sessions with missing prices are omitted rather than filled forward.

**Statistics.** Total return is final value divided by initial capital, minus one. Maximum drawdown is the lowest value of value / running peak minus one. It is zero when value never declines. The running peak starts at the post-fee opening value.

**Limitations.** The simulation has no taxes, slippage, borrowing, or interest on cash. It does not rebalance daily or yearly, short, or optimize weights. Results depend on Yahoo Finance for the requested range.
"""
        ).strip()
    )


st.set_page_config(page_title="Portfolio Investment Agent", layout="wide")
st.title("Portfolio Investment Agent")

explorer_tab, backtest_tab = st.tabs(["Market Explorer", "Backtesting"])
with explorer_tab:
    _show_market_explorer()
with backtest_tab:
    _show_backtest()
