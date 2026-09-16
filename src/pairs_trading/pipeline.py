"""End-to-end orchestration of the five stages. Importable so notebooks and tests can call it."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import pandas as pd

from pairs_trading.backtest import PairResult, aggregate_portfolio, backtest_pair
from pairs_trading.cointegration import screen_pairs, select_cointegrated
from pairs_trading.config import PipelineConfig
from pairs_trading.data import clean_prices, load_prices, split_periods
from pairs_trading.metrics import summarize_portfolio
from pairs_trading.report import build_results_table, build_summary, write_reports
from pairs_trading.signals import generate_target_positions, zscore_for_trading_period
from pairs_trading.universe import get_universe

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    cfg: PipelineConfig
    universe: dict[str, str]
    dropped_tickers: list[str]
    close: pd.DataFrame
    open: pd.DataFrame
    screened: pd.DataFrame
    selected: pd.DataFrame
    results: list[PairResult]
    zscores: dict[str, pd.Series]
    portfolio_pnl: pd.Series
    portfolio_metrics: dict[str, float]
    table: pd.DataFrame
    summary: dict[str, object]
    files: dict[str, Path] = field(default_factory=dict)


def backtest_selected_pairs(
    selected: pd.DataFrame,
    close: pd.DataFrame,
    open_: pd.DataFrame,
    trading_index: pd.Index,
    cfg: PipelineConfig,
) -> tuple[list[PairResult], dict[str, pd.Series]]:
    """Stages 3 + 4 for every selected pair.

    ``close`` spans formation + trading (the rolling z-score warms up on the formation tail);
    positions are only generated and executed on ``trading_index``.
    """
    results: list[PairResult] = []
    zscores: dict[str, pd.Series] = {}
    for row in selected.itertuples(index=False):
        a, b, beta = row.ticker_a, row.ticker_b, float(row.hedge_ratio)
        z = zscore_for_trading_period(close[a], close[b], beta, cfg.zscore_window, trading_index)
        target = generate_target_positions(z, cfg.entry_z, cfg.exit_z, cfg.stop_z)
        result = backtest_pair(
            close_a=close[a],
            close_b=close[b],
            open_a=open_[a],
            open_b=open_[b],
            target_positions=target,
            hedge_ratio=beta,
            cfg=cfg,
            ticker_a=a,
            ticker_b=b,
        )
        results.append(result)
        zscores[result.name] = z
    return results, zscores


def run_pipeline(
    cfg: PipelineConfig, refresh_data: bool = False, write: bool = True
) -> PipelineResult:
    t0 = perf_counter()

    # Stage 1 - data
    universe = get_universe(cfg.sectors)
    tickers = sorted(universe)
    close_raw, open_raw = load_prices(
        tickers, cfg.start, cfg.end, cfg.data_dir, refresh=refresh_data
    )
    close, open_, dropped = clean_prices(close_raw, open_raw, cfg.max_missing_frac, cfg.ffill_limit)
    close_formation, close_trading = split_periods(close, cfg.formation_end)
    _, open_trading = split_periods(open_, cfg.formation_end)
    log.info(
        "data ready: %d tickers, %d formation days, %d trading days (%.1fs)",
        close.shape[1],
        len(close_formation),
        len(close_trading),
        perf_counter() - t0,
    )

    # Stage 2 - cointegration screen on the formation period only
    t1 = perf_counter()
    screened = screen_pairs(close_formation, universe, n_jobs=cfg.n_jobs)
    selected = select_cointegrated(screened, cfg.coint_pvalue, cfg.require_positive_hedge_ratio)
    log.info(
        "screened %d pairs: %d cointegrated at p < %.2f, %d selected for backtesting (%.1fs)",
        len(screened),
        int((screened["pvalue"] < cfg.coint_pvalue).sum()),
        cfg.coint_pvalue,
        len(selected),
        perf_counter() - t1,
    )
    if selected.empty:
        raise RuntimeError("no cointegrated pairs passed the screen; nothing to backtest")

    # Stages 3 + 4 - signals and backtest on the out-of-sample trading period
    t2 = perf_counter()
    results, zscores = backtest_selected_pairs(selected, close, open_, close_trading.index, cfg)
    portfolio_pnl = aggregate_portfolio(results)
    portfolio_metrics = summarize_portfolio(portfolio_pnl, results, cfg)
    log.info(
        "backtested %d pairs, %d trades, portfolio Sharpe %.2f (%.1fs)",
        len(results),
        portfolio_metrics["n_trades"],
        portfolio_metrics["sharpe"],
        perf_counter() - t2,
    )

    # Stage 5 - report
    table = build_results_table(selected, results, cfg)
    summary = build_summary(
        cfg=cfg,
        n_tickers=close.shape[1],
        dropped_tickers=dropped,
        screened=screened,
        selected=selected,
        table=table,
        trading_index=close_trading.index,
        portfolio_metrics=portfolio_metrics,
    )
    files: dict[str, Path] = {}
    if write:
        files = write_reports(
            screened=screened,
            table=table,
            results=results,
            zscores=zscores,
            portfolio_pnl=portfolio_pnl,
            summary=summary,
            cfg=cfg,
            out_dir=cfg.reports_dir,
        )
    log.info("pipeline finished in %.1fs", perf_counter() - t0)

    return PipelineResult(
        cfg=cfg,
        universe=universe,
        dropped_tickers=dropped,
        close=close,
        open=open_,
        screened=screened,
        selected=selected,
        results=results,
        zscores=zscores,
        portfolio_pnl=portfolio_pnl,
        portfolio_metrics=portfolio_metrics,
        table=table,
        summary=summary,
        files=files,
    )
