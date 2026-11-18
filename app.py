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
    maximum_drawdown,
    run_equal_weight_backtest,
    total_return,
)
from market_data import DOW_30, MarketDataError, download_daily_history

DEFAULT_TICKER = "NVDA"
DEFAULT_START = date(2024, 1, 1)
DEFAULT_END = date(2024, 12, 31)
DEFAULT_CAPITAL = 10_000.0


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
    st.caption(
        "Equal-weight buy and hold for selected Dow 30 stocks, compared with SPY and DIA."
    )
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
        _show_strategy_methodology()

    if not submitted:
        st.info("Select at least one stock, then run the backtest.")
        return

    try:
        with st.spinner("Downloading prices and running the backtest..."):
            result = run_equal_weight_backtest(selected, start, end, capital)
    except ValueError as exc:
        st.error(str(exc))
    except (MarketDataError, BacktestError) as exc:
        st.warning(str(exc))
    else:
        _show_backtest_result(result)


def _show_backtest_result(result: BacktestResult) -> None:
    portfolio = result.values["Portfolio"]
    final_value = float(portfolio.iloc[-1])
    st.caption(
        f"Bought at the close on {result.purchase_date.date().isoformat()}. "
        "No later trades are made."
    )
    return_column, drawdown_column, value_column = st.columns(3)
    return_column.metric(
        "Total return",
        f"{total_return(final_value, result.initial_capital):+.2%}",
    )
    drawdown_column.metric("Maximum drawdown", f"{maximum_drawdown(portfolio):.2%}")
    value_column.metric("Final portfolio value", f"${final_value:,.2f}")

    figure = go.Figure()
    for name in result.values.columns:
        figure.add_trace(
            go.Scatter(
                x=result.values.index,
                y=result.values[name],
                mode="lines",
                name=name,
            )
        )
    figure.update_layout(
        title="Portfolio value",
        xaxis_title="Date",
        yaxis_title="Value (USD)",
        hovermode="x unified",
    )
    st.plotly_chart(figure)

    comparison = pd.DataFrame(
        [
            {
                "Series": name,
                "Total return": total_return(float(series.iloc[-1]), result.initial_capital),
                "Maximum drawdown": maximum_drawdown(series),
                "Final value": float(series.iloc[-1]),
            }
            for name, series in result.values.items()
        ]
    )
    st.dataframe(
        comparison,
        hide_index=True,
        column_config={
            "Total return": st.column_config.NumberColumn(format="percent"),
            "Maximum drawdown": st.column_config.NumberColumn(format="percent"),
            "Final value": st.column_config.NumberColumn(format="dollar"),
        },
    )

    st.subheader("Initial allocations")
    st.dataframe(
        _positions_table(result.positions),
        hide_index=True,
        column_config=_position_column_config(),
    )
    st.subheader("Benchmarks")
    st.caption("SPY and DIA are not portfolio holdings. Each is bought with the full initial capital.")
    st.dataframe(
        _positions_table(result.benchmarks),
        hide_index=True,
        column_config=_position_column_config(),
    )

    st.subheader("Daily portfolio value")
    daily = result.values.reset_index()
    value_format = {
        column: st.column_config.NumberColumn(format="dollar")
        for column in result.values.columns
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


def _show_strategy_methodology() -> None:
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


st.set_page_config(page_title="Portfolio Investment Agent", layout="wide")
st.title("Portfolio Investment Agent")

explorer_tab, backtest_tab = st.tabs(["Market Explorer", "Backtesting"])
with explorer_tab:
    _show_market_explorer()
with backtest_tab:
    _show_backtest()
