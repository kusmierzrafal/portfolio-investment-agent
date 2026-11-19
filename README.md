# Portfolio Investment Agent

Research app for the current Dow 30. It downloads daily prices and compares three portfolio strategies with SPY and DIA.

## Run

```bash
uv sync
uv run streamlit run app.py
uv run python -m unittest tests.test_backtest
```

## Layout

- `data/` defines the current Dow 30 catalogue in code and downloads split-adjusted closes and cash dividends.
- `strategies/` decide target portfolio weights. They do not trade or account for fees.
- `backtesting/` simulates those weights: trades, the 0.1% fee, dividends, cash, and benchmarks.
- `ui/` is the Streamlit interface. `app.py` starts it.

Buy and hold allocates once. Equal-weight monthly rebalancing and momentum are asked for new weights on the opening session and on the first common session of each later month.
