# Portfolio Investment Agent

Research app for the current Dow 30. It downloads daily prices and compares portfolio strategies with SPY and DIA.

## Run

```bash
uv sync
uv run streamlit run app.py
uv run python -m unittest discover tests
```

## LLM strategies

Select an LLM strategy in Backtesting. Set the model ID. Paste an OpenAI API key into the password field, or export `OPENAI_API_KEY` and leave the field empty. The key stays in the Streamlit session and is not saved.

The model receives the last 61 scaled closing prices for each stock. Identified strategies also receive the ticker and company name. Anonymized strategies do not. Fully invested strategies allocate only to the selected stocks. CASH Allowed strategies may also allocate to cash.

Tracing is optional. Export these before starting the app to send one Langfuse trace per LLM backtest:

```bash
export LANGFUSE_PUBLIC_KEY="pk-lf-..."
export LANGFUSE_SECRET_KEY="sk-lf-..."
export LANGFUSE_BASE_URL="https://cloud.langfuse.com"
```

Leave them unset to run without tracing. The OpenAI key is not sent to Langfuse. Anonymized prompts still omit tickers; the ticker mapping is stored only on the trace. Token usage is recorded when the model response includes it, and Langfuse can price known models from that usage.

## Layout

- `data/` defines the current Dow 30 catalogue in code and downloads split-adjusted closes and cash dividends.
- `strategies/` decide target portfolio weights. They do not trade or account for fees.
- `backtesting/` simulates those weights: trades, the 0.1% fee, dividends, cash, and benchmarks.
- `ui/` is the Streamlit interface. `app.py` starts it.

Buy and hold allocates once. Equal-weight monthly rebalancing and momentum are asked for new weights on the opening session and on the first common session of each later month.
