"""Methodology text for the strategies shown in the Backtesting tab."""

import textwrap

import streamlit as st

from backtesting.engine import BENCHMARKS, FEE_RATE
from strategies.llm import LLM_STRATEGIES
from strategies.momentum import MOMENTUM_LOOKBACK

BUY_AND_HOLD = "Equal Weight Buy & Hold"
MONTHLY_REBALANCE = "Equal Weight Monthly Rebalancing"
MOMENTUM = "Momentum Monthly"


def show_strategy_methodology(strategies: list[str]) -> None:
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
    if MOMENTUM in strategies:
        if len(strategies) > 1:
            st.markdown(f"**{MOMENTUM}**")
        _show_momentum_methodology()
    if any(name in LLM_STRATEGIES for name in strategies):
        if len(strategies) > 1:
            st.markdown("**LLM strategies**")
        _show_llm_methodology()


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


def _show_momentum_methodology() -> None:
    fee = f"{FEE_RATE:.1%}"
    benchmarks = " and ".join(BENCHMARKS)
    st.markdown(
        textwrap.dedent(
            f"""
The portfolio holds the two strongest stocks in the selected universe and replaces that pair at the first common session of each later month. At least three stocks are required. The catalogue is the current static Dow 30 membership, not the historical index.

**Signal.** A stock's momentum is its last completed close divided by the close {MOMENTUM_LOOKBACK} trading sessions earlier, minus one. Only sessions strictly before the execution date are used. The execution close is not part of the ranking. The two highest returns are selected, even when both are negative. Equal returns are ordered alphabetically by ticker.

**Execution.** The first trade is at the close of the same opening session as the other strategies. Momentum for that trade comes from sessions before the opening, including history downloaded before the selected start date. Later trades are at the close of the first common session of each new month. A stock that leaves the top two is sold. A stock that stays is traded back to half of the portfolio. Nothing is traded after the selected end date.

**Allocation and fees.** Each winner receives half of the portfolio, including dividend cash already received. Fractional shares are allowed. Buys and sells pay a {fee} fee on the amount traded. A buy deducts the fee from the cash spent. A sale deducts the fee from the proceeds. The trade cannot spend more cash than is available or sell more shares than are held.

**Prices and dividends.** Positions use the split-adjusted close (`Close`, requested with `auto_adjust=False`). Dividend cash comes from the `Dividends` column and is not also embedded in the close. Cash is held until the next rebalance. A dividend on the opening session is not received. A dividend on a rebalance session belongs to the shares held before that trade. Sessions downloaded only for the lookback are excluded from valuation, dividends, benchmarks, and reported performance.

**Benchmarks.** {benchmarks} are not holdings and are not ranked. Each is a separate buy-and-hold of the full initial capital, purchased on the opening session, with the same fee and dividend treatment as the other strategies.

**Reported sessions.** A session appears only when every selected stock and both benchmarks have a positive close. Sessions with missing prices are omitted rather than filled forward. If a stock does not have {MOMENTUM_LOOKBACK} completed sessions before an execution date, the backtest stops instead of shortening the lookback.

**Statistics.** Total return is final value divided by initial capital, minus one. Maximum drawdown is the lowest value of value / running peak minus one. It is zero when value never declines. The running peak starts at the post-fee opening value.

**Limitations.** The simulation has no taxes, slippage, borrowing, interest on cash, stop-loss, or cash held back because momentum is negative. It always holds two stocks, does not search for a better lookback, and does not rebalance daily or yearly. Weights can drift after the last rebalance. Results depend on Yahoo Finance for the requested range and the preceding lookback.
"""
        ).strip()
    )


def _show_llm_methodology() -> None:
    st.markdown(
        textwrap.dedent(
            """
The model chooses weights at the opening close and at the first common close of each later month. It sees one scaled close series per stock: the last 60 completed sessions, plus the close that starts that window. Each series begins at 100. The request contains no dates and no raw prices.

Identified strategies also send the ticker and company name. Anonymized strategies send only a synthetic ID, such as A01, and the scaled series. Fully invested strategies cannot hold cash. CASH Allowed strategies may put any share of the portfolio in cash, including all of it.

Python rejects weights that use an unknown ID, are negative or not finite, or do not sum to 1. It also rejects cash when the strategy is fully invested. A rejected response stops the backtest. Accepted weights then use the same fees, dividends, and benchmarks as the other strategies.
"""
        ).strip()
    )
