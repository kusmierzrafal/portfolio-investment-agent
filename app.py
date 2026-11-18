"""Portfolio Investment Agent — Market Explorer."""

from datetime import date

import plotly.graph_objects as go
import streamlit as st

from market_data import DOW_30, MarketDataError, download_daily_history

DEFAULT_TICKER = "NVDA"
DEFAULT_START = date(2024, 1, 1)
DEFAULT_END = date(2024, 12, 31)


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


st.set_page_config(page_title="Portfolio Investment Agent", layout="wide")
st.title("Portfolio Investment Agent")
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
    )
    start_column, end_column = st.columns(2)
    with start_column:
        start = st.date_input("Start date", value=DEFAULT_START)
    with end_column:
        end = st.date_input("End date", value=DEFAULT_END)
    submitted = st.form_submit_button("Download", type="primary")

if not submitted:
    st.info("Choose a ticker and a date range, then download daily prices.")
else:
    try:
        history = download_daily_history(ticker, start, end)
    except ValueError as exc:
        st.error(str(exc))
    except MarketDataError as exc:
        st.warning(str(exc))
    else:
        _show_history(ticker, history)
