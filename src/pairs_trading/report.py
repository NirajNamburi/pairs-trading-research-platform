"""Stage 5 - reporting: ranked results table, trade log, JSON/markdown summary and charts."""

import json
import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure

from pairs_trading.backtest import PairResult, trades_to_frame
from pairs_trading.config import PipelineConfig
from pairs_trading.metrics import summarize

log = logging.getLogger(__name__)

RESULT_COLUMNS = [
    "ticker_a",
    "ticker_b",
    "sector",
    "pvalue",
    "test_stat",
    "hedge_ratio",
    "orientation",
    "half_life",
    "sharpe",
    "annualized_return",
    "annualized_volatility",
    "total_return",
    "max_drawdown",
    "n_trades",
    "win_rate",
    "avg_holding_days",
    "gross_pnl",
    "total_costs",
    "net_pnl",
]


def build_results_table(
    selected: pd.DataFrame, results: Sequence[PairResult], cfg: PipelineConfig
) -> pd.DataFrame:
    """One row per backtested pair: formation statistics + out-of-sample performance."""
    stats = selected.set_index(["ticker_a", "ticker_b"])
    rows = []
    for r in results:
        s = stats.loc[(r.ticker_a, r.ticker_b)]
        rows.append(
            {
                "ticker_a": r.ticker_a,
                "ticker_b": r.ticker_b,
                "sector": s["sector"],
                "pvalue": float(s["pvalue"]),
                "test_stat": float(s["test_stat"]),
                "hedge_ratio": r.hedge_ratio,
                "orientation": s["orientation"],
                "half_life": float(s["half_life"]),
                **summarize(r.daily_pnl, r.trades, cfg),
            }
        )
    table = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    return table.sort_values("sharpe", ascending=False, na_position="last").reset_index(drop=True)


def build_summary(
    *,
    cfg: PipelineConfig,
    n_tickers: int,
    dropped_tickers: Sequence[str],
    screened: pd.DataFrame,
    selected: pd.DataFrame,
    table: pd.DataFrame,
    trading_index: pd.Index,
    portfolio_metrics: Mapping[str, float],
    pretest: pd.DataFrame | None = None,
) -> dict[str, object]:
    profitable = table.loc[table["net_pnl"] > 0]
    traded = table.loc[table["n_trades"] > 0]
    # Rank best and worst on the same frame: a pair without a Sharpe (never traded, or too few
    # days) is neither the best nor the worst by Sharpe, even though it sorts first when every
    # pair is NaN.
    ranked = table.dropna(subset=["sharpe"])
    best = ranked.iloc[0] if not ranked.empty else None
    worst = ranked.iloc[-1] if not ranked.empty else None

    def pair_blurb(row: pd.Series | None) -> dict[str, object] | None:
        if row is None:
            return None
        return {
            "pair": f"{row['ticker_a']}/{row['ticker_b']}",
            "sharpe": _clean(row["sharpe"]),
            "total_return": _clean(row["total_return"]),
            "max_drawdown": _clean(row["max_drawdown"]),
            "n_trades": int(row["n_trades"]),
        }

    return {
        "config": cfg.to_dict(),
        "universe": {"n_tickers": n_tickers, "dropped_tickers": list(dropped_tickers)},
        "formation_period": {"start": cfg.start, "end": cfg.formation_end},
        "trading_period": {
            "start": str(trading_index[0].date()),
            "end": str(trading_index[-1].date()),
            "n_days": int(len(trading_index)),
        },
        "screening": {
            "n_pairs_tested": int(len(screened)),
            # p-value count only, so it is comparable with the chance expectation below; the
            # hedge-ratio filter is reported separately as n_selected.
            "n_cointegrated": int((screened["pvalue"] < cfg.coint_pvalue).sum()),
            "pvalue_threshold": cfg.coint_pvalue,
            "expected_false_positives_at_threshold": float(len(screened) * cfg.coint_pvalue),
            "require_positive_hedge_ratio": cfg.require_positive_hedge_ratio,
            "n_selected": int(len(selected)),
            "unit_root_pretest": {
                "enabled": cfg.unit_root_pretest,
                "pvalue_threshold": cfg.unit_root_pvalue,
                "n_tested": int(len(pretest)) if pretest is not None else 0,
                "dropped": (
                    pretest.loc[~pretest["is_i1"], "ticker"].tolist() if pretest is not None else []
                ),
            },
            "both_orderings": cfg.test_both_orderings,
        },
        "pairs": {
            "n_backtested": int(len(table)),
            "n_with_trades": int(len(traded)),
            "n_profitable_after_costs": int(len(profitable)),
            "best_by_sharpe": pair_blurb(best),
            "worst_by_sharpe": pair_blurb(worst),
        },
        "portfolio": {k: _clean(v) for k, v in portfolio_metrics.items()},
    }


def _clean(value: object) -> object:
    """Make NaN/inf JSON-safe."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):  # numpy scalar
        return _clean(value.item())
    return value


def _fmt(value: object, pct: bool = False, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    if pct:
        return f"{100 * float(value):.{digits}f}%"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown_table(
    frame: pd.DataFrame, columns: Sequence[str], formats: Mapping[str, dict]
) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = [header, sep]
    for _, row in frame.iterrows():
        cells = [_fmt(row[c], **formats.get(c, {})) for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_summary_markdown(
    summary: Mapping[str, object], table: pd.DataFrame, top_n: int = 10
) -> str:
    cfg = summary["config"]
    scr = summary["screening"]
    pairs = summary["pairs"]
    port = summary["portfolio"]
    fp = summary["formation_period"]
    tp = summary["trading_period"]
    uni = summary["universe"]
    best, worst = pairs["best_by_sharpe"], pairs["worst_by_sharpe"]
    pre = scr.get("unit_root_pretest", {"enabled": False})
    if pre["enabled"]:
        dropped = pre["dropped"]
        pretest_line = (
            f"- Unit-root pretest: {pre['n_tested']} tickers tested; {len(dropped)} dropped as "
            f"stationary on their own (levels ADF p <= {pre['pvalue_threshold']:g})"
            + (": " + ", ".join(dropped) if dropped else "")
            + "."
        )
    else:
        pretest_line = "- Unit-root pretest: off (naive screen)."
    orientation_note = (
        " Both orientations tested per pair."
        if scr.get("both_orderings", False)
        else " Alphabetical orientation only."
    )

    lines = [
        "# Pairs trading backtest summary",
        "",
        f"Formation (screening) period: **{fp['start']} to {fp['end']}**. "
        f"Out-of-sample trading period: **{tp['start']} to {tp['end']}** "
        f"({tp['n_days']} trading days).",
        "",
        "## Headline findings",
        "",
        f"- Universe: {uni['n_tickers']} tickers across {', '.join(cfg['sectors'])}"
        + (
            f" ({len(uni['dropped_tickers'])} dropped for missing data: "
            f"{', '.join(uni['dropped_tickers'])})"
            if uni["dropped_tickers"]
            else ""
        )
        + ".",
        pretest_line,
        f"- Pairs tested: {scr['n_pairs_tested']}. Cointegrated at p < {scr['pvalue_threshold']}: "
        f"**{scr['n_cointegrated']}** (about {scr['expected_false_positives_at_threshold']:.0f} "
        f"would be expected by chance). Selected for backtesting: {scr['n_selected']}"
        + (" (positive hedge ratio required)" if scr["require_positive_hedge_ratio"] else "")
        + "."
        + orientation_note,
        f"- Pairs profitable after costs out of sample: "
        f"**{pairs['n_profitable_after_costs']} of {pairs['n_backtested']}** "
        f"({pairs['n_with_trades']} traded at least once).",
        f"- Equal-weight portfolio across all {pairs['n_backtested']} pairs: "
        f"Sharpe **{_fmt(port['sharpe'])}**, "
        f"annualised return {_fmt(port['annualized_return'], pct=True)}, "
        f"max drawdown {_fmt(port['max_drawdown'], pct=True)}, "
        f"{port['n_trades']} trades, win rate {_fmt(port['win_rate'], pct=True, digits=1)}, "
        f"average hold {_fmt(port['avg_holding_days'], digits=1)} days.",
    ]
    if best:
        lines.append(
            f"- Best pair by Sharpe: **{best['pair']}** (Sharpe {_fmt(best['sharpe'])}, "
            f"return {_fmt(best['total_return'], pct=True)}, {best['n_trades']} trades)."
        )
    if worst:
        lines.append(
            f"- Worst pair by Sharpe: {worst['pair']} (Sharpe {_fmt(worst['sharpe'])}, "
            f"return {_fmt(worst['total_return'], pct=True)})."
        )
    lines += [
        "",
        "## Assumptions",
        "",
        "- Signal at close, execution at the next day's open. "
        f"Rolling z-score window {cfg['zscore_window']} days; "
        f"entry |z| > {cfg['entry_z']}, exit |z| < {cfg['exit_z']}"
        + (f", stop |z| > {cfg['stop_z']}" if cfg["stop_z"] is not None else "")
        + ".",
        f"- Slippage {cfg['slippage_bps']:g} bps and commission "
        f"{cfg['transaction_cost_bps']:g} bps of notional, each leg, on entry and exit.",
        "- $1 gross notional per pair per trade, fixed hedge ratio from the formation regression, "
        f"risk-free rate {cfg['risk_free_rate']:g}, {cfg['trading_days_per_year']} trading days "
        "per year.",
        "- Open positions on the last day are force-closed at the final close.",
        "",
        f"## Top {min(top_n, len(table))} pairs by out-of-sample Sharpe",
        "",
        _markdown_table(
            table.head(top_n),
            [
                "ticker_a",
                "ticker_b",
                "sector",
                "pvalue",
                "hedge_ratio",
                "half_life",
                "sharpe",
                "total_return",
                "max_drawdown",
                "n_trades",
                "win_rate",
            ],
            {
                "pvalue": {"digits": 4},
                "half_life": {"digits": 1},
                "total_return": {"pct": True},
                "max_drawdown": {"pct": True},
                "win_rate": {"pct": True, "digits": 0},
            },
        ),
        "",
    ]
    return "\n".join(lines)


def plot_pair(result: PairResult, zscore: pd.Series, cfg: PipelineConfig, path: Path) -> Path:
    """Two-panel chart: z-score with bands and entry/exit markers, and the pair's equity curve.

    Figures are built without ``pyplot`` so writing a PNG never touches the global matplotlib
    backend of the importing process (a notebook keeps its inline backend).
    """
    fig = Figure(figsize=(12, 8))
    ax_z, ax_eq = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 2]})

    ax_z.plot(zscore.index, zscore.to_numpy(), color="#1f77b4", linewidth=1.0, label="z-score")
    ax_z.axhline(0.0, color="black", linewidth=0.8)
    for level, style, colour, label in (
        (cfg.entry_z, "--", "#d62728", f"entry ±{cfg.entry_z:g}"),
        (cfg.exit_z, ":", "#7f7f7f", f"exit ±{cfg.exit_z:g}"),
    ):
        ax_z.axhline(level, linestyle=style, color=colour, linewidth=0.9, label=label)
        ax_z.axhline(-level, linestyle=style, color=colour, linewidth=0.9)
    if cfg.stop_z is not None:
        ax_z.axhline(
            cfg.stop_z,
            linestyle="-.",
            color="#9467bd",
            linewidth=0.9,
            label=f"stop ±{cfg.stop_z:g}",
        )
        ax_z.axhline(-cfg.stop_z, linestyle="-.", color="#9467bd", linewidth=0.9)

    long_entries = [t.signal_date for t in result.trades if t.direction == 1]
    short_entries = [t.signal_date for t in result.trades if t.direction == -1]
    exits = []
    for t in result.trades:
        if t.exit_reason == "signal":
            exits.append(zscore.index[zscore.index.get_loc(t.exit_date) - 1])  # decision close
        else:
            exits.append(t.exit_date)
    if long_entries:
        ax_z.scatter(
            long_entries,
            zscore.loc[long_entries],
            marker="^",
            color="green",
            s=45,
            zorder=3,
            label="long spread",
        )
    if short_entries:
        ax_z.scatter(
            short_entries,
            zscore.loc[short_entries],
            marker="v",
            color="red",
            s=45,
            zorder=3,
            label="short spread",
        )
    if exits:
        ax_z.scatter(
            exits, zscore.loc[exits], marker="x", color="black", s=40, zorder=3, label="exit"
        )
    ax_z.set_ylabel("spread z-score")
    ax_z.set_title(
        f"{result.name}  (hedge ratio {result.hedge_ratio:.3f}) - signal at close, fill next open"
    )
    ax_z.legend(loc="upper left", ncol=3, fontsize=8)
    ax_z.grid(alpha=0.3)

    ax_eq.plot(result.equity.index, result.equity.to_numpy(), color="#2ca02c", linewidth=1.2)
    ax_eq.axhline(1.0, color="black", linewidth=0.8)
    ax_eq.set_ylabel("equity ($1 capital)")
    ax_eq.set_xlabel("date")
    ax_eq.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return path


def plot_portfolio(equity: pd.Series, path: Path, title: str) -> Path:
    fig = Figure(figsize=(12, 5))
    ax = fig.subplots()
    ax.plot(equity.index, equity.to_numpy(), color="#2ca02c", linewidth=1.3)
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel("equity ($1 capital)")
    ax.set_xlabel("date")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    return path


def write_reports(
    *,
    screened: pd.DataFrame,
    table: pd.DataFrame,
    results: Sequence[PairResult],
    zscores: Mapping[str, pd.Series],
    portfolio_pnl: pd.Series,
    summary: Mapping[str, object],
    cfg: PipelineConfig,
    out_dir: Path,
    pretest: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Write every report artefact to ``out_dir`` and return ``{name: path}``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}

    if pretest is not None:
        files["unit_root_pretest"] = out_dir / "unit_root_pretest.csv"
        pretest.to_csv(files["unit_root_pretest"], index=False)

    files["screened_pairs"] = out_dir / "screened_pairs.csv"
    screened.to_csv(files["screened_pairs"], index=False)

    files["pair_results"] = out_dir / "pair_results.csv"
    table.to_csv(files["pair_results"], index=False)

    files["trades"] = out_dir / "trades.csv"
    trades_to_frame(list(results)).to_csv(files["trades"], index=False)

    portfolio_equity = (1.0 + portfolio_pnl.cumsum()).rename("equity")
    files["portfolio_equity_csv"] = out_dir / "portfolio_equity.csv"
    pd.concat([portfolio_pnl, portfolio_equity], axis=1).to_csv(files["portfolio_equity_csv"])

    files["summary_json"] = out_dir / "summary.json"
    files["summary_json"].write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    files["summary_md"] = out_dir / "summary.md"
    files["summary_md"].write_text(render_summary_markdown(summary, table), encoding="utf-8")

    n_pairs = len(results)
    files["portfolio_equity_png"] = plot_portfolio(
        portfolio_equity,
        out_dir / "portfolio_equity.png",
        f"Equal-weight portfolio of {n_pairs} cointegrated pairs - out-of-sample, net of costs",
    )

    by_name = {r.name: r for r in results}
    for _, row in table.head(cfg.n_plot_pairs).iterrows():
        name = f"{row['ticker_a']}/{row['ticker_b']}"
        result = by_name[name]
        stem = f"{row['ticker_a']}_{row['ticker_b']}"
        files[f"pair_{stem}"] = plot_pair(result, zscores[name], cfg, out_dir / f"{stem}.png")

    log.info("wrote %d report files to %s", len(files), out_dir)
    return files
