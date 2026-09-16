import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import bdays
from pairs_trading.backtest import PairResult, aggregate_portfolio, backtest_pair
from pairs_trading.cointegration import PAIR_COLUMNS, select_cointegrated
from pairs_trading.config import PipelineConfig
from pairs_trading.metrics import summarize_portfolio
from pairs_trading.report import (
    build_results_table,
    build_summary,
    render_summary_markdown,
    write_reports,
)


def _result(cfg: PipelineConfig, ticker_a: str, ticker_b: str, target: list[int]) -> PairResult:
    """Six-day pair on the hand-computed fixture: A rallies from 10 to 11 on day 2."""
    idx = bdays(6)

    def s(values):
        return pd.Series(values, index=idx, dtype=float)

    return backtest_pair(
        close_a=s([10, 10, 11, 11, 11, 11]),
        close_b=s([10] * 6),
        open_a=s([10, 10, 10, 10, 11, 11]),
        open_b=s([10] * 6),
        target_positions=pd.Series(target, index=idx),
        hedge_ratio=1.0,
        cfg=cfg,
        ticker_a=ticker_a,
        ticker_b=ticker_b,
    )


def _screened(rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    """``(ticker_a, ticker_b, pvalue, hedge_ratio)`` rows in the screen's column layout."""
    return pd.DataFrame(
        [
            {
                "ticker_a": a,
                "ticker_b": b,
                "sector": "s",
                "pvalue": p,
                "hedge_ratio": h,
                "intercept": 0.0,
                "half_life": 5.0,
                "n_obs": 100,
            }
            for a, b, p, h in rows
        ],
        columns=PAIR_COLUMNS,
    )


def _summary(cfg, screened, selected, results):
    table = build_results_table(selected, results, cfg)
    portfolio_pnl = aggregate_portfolio(results)
    summary = build_summary(
        cfg=cfg,
        n_tickers=2 * len(selected),
        dropped_tickers=[],
        screened=screened,
        selected=selected,
        table=table,
        trading_index=results[0].daily_pnl.index,
        portfolio_metrics=summarize_portfolio(portfolio_pnl, results, cfg),
    )
    return summary, table, portfolio_pnl


def test_best_and_worst_by_sharpe_ignore_pairs_without_a_sharpe(cfg):
    screened = _screened([("A", "B", 0.01, 1.0), ("C", "D", 0.02, 1.0)])
    traded = _result(cfg, "A", "B", [0, 1, 1, 0, 0, 0])  # finite Sharpe
    idle = _result(cfg, "C", "D", [0] * 6)  # never trades -> zero volatility -> NaN Sharpe

    summary, table, _ = _summary(cfg, screened, screened, [traded, idle])
    assert summary["pairs"]["best_by_sharpe"]["pair"] == "A/B"
    assert summary["pairs"]["worst_by_sharpe"]["pair"] == "A/B"
    assert summary["pairs"]["n_with_trades"] == 1

    # Every pair NaN: nothing is best or worst, and the markdown does not pretend otherwise.
    idle2 = _result(cfg, "A", "B", [0] * 6)
    summary, table, _ = _summary(cfg, screened, screened, [idle2, idle])
    assert summary["pairs"]["best_by_sharpe"] is None
    assert summary["pairs"]["worst_by_sharpe"] is None
    markdown = render_summary_markdown(summary, table)
    assert "Best pair by Sharpe" not in markdown
    assert "Worst pair by Sharpe" not in markdown


def test_summary_separates_pvalue_count_from_hedge_ratio_selection(cfg):
    screened = _screened([("A", "B", 0.01, 1.0), ("C", "D", 0.02, -0.5), ("E", "F", 0.2, 1.0)])
    selected = select_cointegrated(screened, cfg.coint_pvalue, cfg.require_positive_hedge_ratio)
    assert selected["ticker_a"].tolist() == ["A"]

    summary, table, _ = _summary(
        cfg, screened, selected, [_result(cfg, "A", "B", [0, 1, 1, 0, 0, 0])]
    )
    scr = summary["screening"]
    assert scr["n_pairs_tested"] == 3
    assert scr["n_cointegrated"] == 2  # p-value only, comparable with the chance expectation
    assert scr["n_selected"] == 1  # after the positive-hedge-ratio filter
    assert scr["require_positive_hedge_ratio"] is True
    markdown = render_summary_markdown(summary, table)
    assert "Cointegrated at p < 0.05: **2**" in markdown
    assert "Selected for backtesting: 1 (positive hedge ratio required)." in markdown


def test_portfolio_block_is_on_the_same_basis_as_pair_rows(cfg):
    screened = _screened([("A", "B", 0.01, 1.0), ("C", "D", 0.02, 1.0)])
    results = [_result(cfg, "A", "B", [0, 1, 1, 0, 0, 0]), _result(cfg, "C", "D", [0] * 6)]
    summary, table, _ = _summary(cfg, screened, screened, results)
    port = summary["portfolio"]
    assert port["net_pnl"] == pytest.approx(port["total_return"])
    assert port["total_return"] == pytest.approx(table["total_return"].mean())
    assert (table["net_pnl"] - table["total_return"]).abs().max() < 1e-12


def test_write_reports_writes_every_artefact_and_respects_n_plot_pairs(cfg, tmp_path):
    cfg = cfg.replace(n_plot_pairs=1)
    screened = _screened([("A", "B", 0.01, 1.0), ("C", "D", 0.02, 1.0)])
    results = [
        _result(cfg, "A", "B", [0, 1, 1, 0, 0, 0]),  # long spread into the rally: wins
        _result(cfg, "C", "D", [0, -1, -1, 0, 0, 0]),  # short spread into the rally: loses
    ]
    summary, table, portfolio_pnl = _summary(cfg, screened, screened, results)
    zscores = {r.name: pd.Series(np.linspace(-3.0, 3.0, 6), index=bdays(6)) for r in results}

    files = write_reports(
        screened=screened,
        table=table,
        results=results,
        zscores=zscores,
        portfolio_pnl=portfolio_pnl,
        summary=summary,
        cfg=cfg,
        out_dir=tmp_path,
    )

    assert all(path.exists() and path.stat().st_size > 0 for path in files.values())
    assert sorted(p.name for p in tmp_path.glob("*.png")) == ["A_B.png", "portfolio_equity.png"]
    assert (tmp_path / "A_B.png").read_bytes()[:4] == b"\x89PNG"
    reloaded = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert reloaded["pairs"]["best_by_sharpe"]["pair"] == "A/B"
    assert len(pd.read_csv(tmp_path / "trades.csv")) == 2


def test_importing_report_leaves_the_matplotlib_backend_alone():
    """A notebook that imports the pipeline must keep its own (inline) backend."""
    code = (
        "import matplotlib; matplotlib.use('svg'); import matplotlib.pyplot; "
        "import pairs_trading.report; print(matplotlib.get_backend())"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "svg"
