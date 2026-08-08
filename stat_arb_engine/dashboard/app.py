"""Minimal Streamlit dashboard for exploring screened pairs and backtests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.backtest.engine import configs_from_yaml, run_pairs_backtest
from src.pipeline.ingest import load_price_panel
from src.screening.cointegration import load_pairs
from src.utils.config import PROJECT_ROOT, load_config

st.set_page_config(page_title="Stat Arb Engine", layout="wide")
st.title("Statistical Arbitrage Research Engine")
st.caption("Research demo — past performance ≠ future results.")

cfg = load_config(PROJECT_ROOT / "configs" / "default.yaml")
db_path = Path(cfg["data"]["db_path"])

if not db_path.exists():
    st.warning("No database found. Run `stat-arb ingest-data` then `stat-arb screen-pairs`.")
    st.stop()

pairs = load_pairs(db_path)
if pairs.empty:
    st.warning("No screened pairs. Run `stat-arb screen-pairs`.")
    st.stop()

pair_id = st.selectbox("Pair", pairs["pair_id"].tolist())
row = pairs.loc[pairs["pair_id"] == pair_id].iloc[0]
st.write(
    {
        "hedge_ratio": row["hedge_ratio"],
        "coint_pvalue": row["cointegration_pvalue"],
        "correlation": row["correlation"],
    }
)

panel = load_price_panel(
    [row["ticker_a"], row["ticker_b"]], Path(cfg["data"]["cache_dir"]), join="inner"
)
signal_cfg, bt_cfg = configs_from_yaml(cfg)
detail, report = run_pairs_backtest(
    panel[row["ticker_a"]],
    panel[row["ticker_b"]],
    hedge_ratio=float(row["hedge_ratio"]),
    signal_cfg=signal_cfg,
    bt_cfg=bt_cfg,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Sharpe", f"{report.sharpe:.2f}")
c2.metric("Max DD", f"{report.max_drawdown:.1%}")
c3.metric("Total return", f"{report.total_return:.1%}")
c4.metric("Trades", f"{report.n_trades}")

st.subheader("Equity curve")
st.line_chart(detail["equity"])

st.subheader("Z-score")
st.line_chart(detail["zscore"])
