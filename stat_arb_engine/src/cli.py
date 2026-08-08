"""
Command-line interface for the statistical arbitrage research engine.

Examples
--------
  stat-arb ingest-data
  stat-arb screen-pairs
  stat-arb run-backtest --pair-id JPM_BAC
  stat-arb run-walkforward --pair-id JPM_BAC
  stat-arb run-significance-test --pair-id JPM_BAC
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click
import pandas as pd

from src.backtest.engine import configs_from_yaml, run_pairs_backtest, save_backtest_result
from src.pipeline.ingest import ingest_universe, load_price_panel
from src.screening.cointegration import load_pairs, run_screen
from src.utils.config import load_config, setup_logging
from src.validation.walkforward import (
    monte_carlo_permutation_test,
    walk_forward_pair,
    wf_config_from_yaml,
)

logger = logging.getLogger(__name__)


def _pair_prices(cfg: dict, pair_id: str) -> tuple[pd.Series, pd.Series, float]:
    db_path = Path(cfg["data"]["db_path"])
    pairs = load_pairs(db_path)
    if pairs.empty:
        raise click.ClickException("No pairs in DB. Run screen-pairs first.")
    row = pairs.loc[pairs["pair_id"] == pair_id]
    if row.empty:
        raise click.ClickException(f"pair_id '{pair_id}' not found. Try screen-pairs.")
    row = row.iloc[0]
    panel = load_price_panel(
        [row["ticker_a"], row["ticker_b"]],
        Path(cfg["data"]["cache_dir"]),
        join="inner",
    )
    return panel[row["ticker_a"]], panel[row["ticker_b"]], float(row["hedge_ratio"])


@click.group()
@click.option("--config", "config_path", default="configs/default.yaml", show_default=True)
@click.option("--log-level", default="INFO", show_default=True)
@click.pass_context
def cli(ctx: click.Context, config_path: str, log_level: str) -> None:
    """Statistical Arbitrage Research & Backtesting Engine."""
    setup_logging(log_level)
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(config_path)
    ctx.obj["config_path"] = config_path


@cli.command("ingest-data")
@click.option("--force", is_flag=True, help="Re-download even if Parquet cache exists.")
@click.pass_context
def ingest_data(ctx: click.Context, force: bool) -> None:
    """Download / cache OHLCV for the configured universe."""
    frames = ingest_universe(ctx.obj["config"], force=force)
    click.echo(f"Ingested {len(frames)} tickers.")


@cli.command("screen-pairs")
@click.pass_context
def screen_pairs_cmd(ctx: click.Context) -> None:
    """Run Engle-Granger cointegration screen and store pairs in SQLite."""
    results = run_screen(ctx.obj["config"])
    if results.empty:
        click.echo("No cointegrated pairs found under current thresholds.")
        return
    click.echo(results.to_string(index=False))
    click.echo(f"\nSaved {len(results)} pairs to {ctx.obj['config']['data']['db_path']}")


@cli.command("list-pairs")
@click.option("--max-pairs", default=20, show_default=True)
@click.pass_context
def list_pairs(ctx: click.Context, max_pairs: int) -> None:
    """Show screened pairs ranked by cointegration p-value."""
    pairs = load_pairs(Path(ctx.obj["config"]["data"]["db_path"]), max_pairs=max_pairs)
    if pairs.empty:
        click.echo("No pairs found. Run screen-pairs first.")
        return
    cols = [
        "pair_id",
        "ticker_a",
        "ticker_b",
        "hedge_ratio",
        "cointegration_pvalue",
        "correlation",
    ]
    click.echo(pairs[cols].to_string(index=False))


@cli.command("run-backtest")
@click.option("--pair-id", required=True, help="e.g. BAC_JPM")
@click.option("--save/--no-save", default=True, show_default=True)
@click.pass_context
def run_backtest_cmd(ctx: click.Context, pair_id: str, save: bool) -> None:
    """Backtest one cointegrated pair with costs/slippage."""
    cfg = ctx.obj["config"]
    price_a, price_b, beta = _pair_prices(cfg, pair_id)
    signal_cfg, bt_cfg = configs_from_yaml(cfg)
    detail, report = run_pairs_backtest(
        price_a, price_b, hedge_ratio=beta, signal_cfg=signal_cfg, bt_cfg=bt_cfg
    )
    click.echo(json.dumps(report.to_dict(), indent=2))
    if save:
        run_id = save_backtest_result(
            Path(cfg["data"]["db_path"]),
            pair_id,
            report,
            params={"signal": signal_cfg.__dict__, "backtest": bt_cfg.__dict__},
        )
        out = Path("results") / f"{pair_id}_backtest.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        detail.to_csv(out)
        click.echo(f"Saved run_id={run_id} and detail -> {out}")


@cli.command("run-walkforward")
@click.option("--pair-id", required=True)
@click.pass_context
def run_walkforward_cmd(ctx: click.Context, pair_id: str) -> None:
    """Walk-forward out-of-sample validation for one pair."""
    cfg = ctx.obj["config"]
    price_a, price_b, _beta = _pair_prices(cfg, pair_id)
    signal_cfg, bt_cfg = configs_from_yaml(cfg)
    wf_cfg = wf_config_from_yaml(cfg)
    result = walk_forward_pair(price_a, price_b, signal_cfg, bt_cfg, wf_cfg)
    click.echo("Per-window summary:")
    click.echo(pd.DataFrame(result.window_reports).to_string(index=False))
    click.echo("\nAggregate OOS performance:")
    click.echo(json.dumps(result.aggregate.to_dict(), indent=2))
    out = Path("results") / f"{pair_id}_walkforward.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.oos_detail.to_csv(out)
    click.echo(f"Wrote {out}")


@cli.command("run-significance-test")
@click.option("--pair-id", required=True)
@click.option(
    "--source",
    type=click.Choice(["backtest", "walkforward"]),
    default="walkforward",
    show_default=True,
)
@click.pass_context
def run_significance_test_cmd(ctx: click.Context, pair_id: str, source: str) -> None:
    """Monte Carlo timing test: shuffle positions vs fixed spread returns."""
    cfg = ctx.obj["config"]
    price_a, price_b, beta = _pair_prices(cfg, pair_id)
    signal_cfg, bt_cfg = configs_from_yaml(cfg)
    v = cfg.get("validation", {})

    if source == "walkforward":
        wf_cfg = wf_config_from_yaml(cfg)
        result = walk_forward_pair(price_a, price_b, signal_cfg, bt_cfg, wf_cfg)
        detail = result.oos_detail
    else:
        detail, _ = run_pairs_backtest(
            price_a, price_b, hedge_ratio=beta, signal_cfg=signal_cfg, bt_cfg=bt_cfg
        )

    # Reconstruct approximate pre-cost spread returns from stored columns.
    # ret_gross ≈ pos * spread_ret * position_size  ⇒  spread_ret ≈ ret_gross / (pos * size)
    pos = detail["pos"].fillna(0.0)
    ret_gross = detail["ret_gross"].fillna(0.0)
    size = bt_cfg.position_size
    spread_ret = pd.Series(0.0, index=detail.index)
    mask = pos.abs() > 0
    spread_ret.loc[mask] = ret_gross.loc[mask] / (pos.loc[mask] * size)
    # On flat days, use 0 (timing null only scores when a shuffled pos is nonzero).

    stats = monte_carlo_permutation_test(
        position=pos,
        spread_returns=spread_ret,
        n_permutations=int(v.get("n_permutations", 1000)),
        position_size=size,
        transaction_cost_bps=bt_cfg.transaction_cost_bps,
        slippage_bps=bt_cfg.slippage_bps,
        risk_free_rate=bt_cfg.risk_free_rate,
        periods_per_year=bt_cfg.trading_days_per_year,
        random_seed=int(v.get("random_seed", 42)),
    )
    click.echo(json.dumps(stats, indent=2))


@cli.command("launch-dashboard")
def launch_dashboard() -> None:
    """Launch the optional Streamlit dashboard (if installed)."""
    try:
        import streamlit.web.cli as stcli
    except ImportError as exc:
        raise click.ClickException(
            "Streamlit not installed. pip install -e '.[dashboard]'"
        ) from exc
    import sys

    dash = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")
    sys.argv = ["streamlit", "run", dash]
    stcli.main()


if __name__ == "__main__":
    cli()
