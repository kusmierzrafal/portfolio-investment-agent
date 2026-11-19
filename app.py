"""Portfolio Investment Agent."""

import streamlit as st

from ui.backtesting import show_backtest
from ui.market_explorer import show_market_explorer

st.set_page_config(page_title="Portfolio Investment Agent", layout="wide")
st.title("Portfolio Investment Agent")

explorer_tab, backtest_tab = st.tabs(["Market Explorer", "Backtesting"])
with explorer_tab:
    show_market_explorer()
with backtest_tab:
    show_backtest()
