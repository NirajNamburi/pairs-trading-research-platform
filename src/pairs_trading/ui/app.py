"""Streamlit research viewer for a completed pipeline run.

The viewer reads the screen (selected pairs, hedge ratios, configuration) from a ``reports/``
directory written by ``python -m pairs_trading run`` and re-runs the *signal and backtest*
stages in memory for the parameters in the sidebar. The cointegration screen itself is never
re-run here (it is the slow part and depends only on the universe and formation dates), so
changing the z-score window, thresholds or costs updates every chart in well under a second.

    python -m pairs_trading.ui                 # opens reports/
    python -m pairs_trading.ui --reports other_dir
"""

import argparse
import io
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pairs_trading import data as data_stage
from pairs_trading.backtest import PairResult, aggregate_portfolio, trades_to_frame
from pairs_trading.cointegration import select_cointegrated
from pairs_trading.config import PipelineConfig
from pairs_trading.metrics import summarize_portfolio
from pairs_trading.pipeline import backtest_selected_pairs
from pairs_trading.report import build_results_table
from pairs_trading.universe import get_universe

DEFAULT_REPORTS = "reports"


# --- loading ----------------------------------------------------------------------------------


RUN_FILES = ("summary.json", "screened_pairs.csv", "unit_root_pretest.csv")


def _cli_reports_dir() -> str:
    argv = sys.argv[1:]
    # streamlit forwards a user-typed "--" verbatim; argparse would treat it as end-of-options
    if argv[:1] == ["--"]:
        argv = argv[1:]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--reports", default=DEFAULT_REPORTS)
    args, _ = parser.parse_known_args(argv)
    return args.reports


def run_stamp(reports_dir: str) -> tuple[tuple[str, int, int], ...]:
    """Identity of the run files (name, mtime, size), so a re-run pipeline is picked up."""
    root = Path(reports_dir)
    stamp = []
    for name in RUN_FILES:
        path = root / name
        if path.exists():
            info = path.stat()
            stamp.append((name, info.st_mtime_ns, info.st_size))
    return tuple(stamp)


@st.cache_data(show_spinner=False)
def load_run(reports_dir: str, stamp: tuple[tuple[str, int, int], ...]) -> dict:
    """The committed run: summary.json, the screened pairs and the pretest table (if any).

    ``stamp`` is :func:`run_stamp` of the directory; it is part of the cache key so that the
    viewer re-reads the files when ``python -m pairs_trading run`` overwrites them.
    """
    root = Path(reports_dir)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    screened = pd.read_csv(root / "screened_pairs.csv")
    pretest_path = root / "unit_root_pretest.csv"
    pretest = pd.read_csv(pretest_path) if pretest_path.exists() else None
    return {"summary": summary, "screened": screened, "pretest": pretest}


def config_from_summary(summary: dict) -> PipelineConfig:
    cfg = summary["config"]
    fields = {
        k: v
        for k, v in cfg.items()
        if k in PipelineConfig.__dataclass_fields__ and k not in ("data_dir", "reports_dir")
    }
    fields["sectors"] = tuple(cfg["sectors"])
    return PipelineConfig(
        **fields, data_dir=Path(cfg["data_dir"]), reports_dir=Path(cfg["reports_dir"])
    )


@st.cache_data(show_spinner="Loading prices from the local cache...")
def load_prices_for(cfg_key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = PipelineConfig(**json.loads(cfg_key))
    tickers = sorted(get_universe(cfg.sectors))
    close_raw, open_raw = data_stage.load_prices(tickers, cfg.start, cfg.end, cfg.data_dir)
    close, open_, _ = data_stage.clean_prices(
        close_raw, open_raw, cfg.max_missing_frac, cfg.ffill_limit
    )
    return close, open_


def _cfg_key(cfg: PipelineConfig) -> str:
    d = cfg.to_dict()
    return json.dumps(d, sort_keys=True, default=str)


@st.cache_data(show_spinner="Running signals and backtest...")
def run_backtest(cfg_key: str, screened_json: str) -> dict:
    """Signals + backtest for ``cfg`` over the pairs selected from ``screened``."""
    cfg = PipelineConfig(**json.loads(cfg_key))
    screened = pd.read_json(io.StringIO(screened_json), orient="split")
    close, open_ = load_prices_for(cfg_key)
    _, close_trading = data_stage.split_periods(close, cfg.formation_end)
    selected = select_cointegrated(screened, cfg.coint_pvalue, cfg.require_positive_hedge_ratio)
    results, zscores = backtest_selected_pairs(selected, close, open_, close_trading.index, cfg)
    portfolio_pnl = aggregate_portfolio(results)
    metrics = summarize_portfolio(portfolio_pnl, results, cfg)
    table = build_results_table(selected, results, cfg)
    return {
        "results": results,
        "zscores": zscores,
        "portfolio_pnl": portfolio_pnl,
        "metrics": metrics,
        "table": table,
        "trades": trades_to_frame(results),
        "close": close,
    }


# --- charts -----------------------------------------------------------------------------------


def equity_chart(equity: pd.Series, title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity.index, y=equity.to_numpy(), mode="lines", name="equity"))
    fig.add_hline(y=1.0, line_color="black", line_width=1)
    fig.update_layout(title=title, yaxis_title="equity ($1 capital)", height=380, margin=dict(t=50))
    return fig


def zscore_chart(result: PairResult, z: pd.Series, cfg: PipelineConfig) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=z.index, y=z.to_numpy(), mode="lines", name="z-score"))
    for level, dash in ((cfg.entry_z, "dash"), (cfg.exit_z, "dot")):
        fig.add_hline(y=level, line_dash=dash, line_color="grey", line_width=1)
        fig.add_hline(y=-level, line_dash=dash, line_color="grey", line_width=1)
    if cfg.stop_z is not None:
        fig.add_hline(y=cfg.stop_z, line_dash="dashdot", line_color="purple", line_width=1)
        fig.add_hline(y=-cfg.stop_z, line_dash="dashdot", line_color="purple", line_width=1)
    fig.add_hline(y=0.0, line_color="black", line_width=1)

    longs = [t.signal_date for t in result.trades if t.direction == 1]
    shorts = [t.signal_date for t in result.trades if t.direction == -1]
    exits = []
    for t in result.trades:
        if t.exit_reason == "signal":
            exits.append(z.index[z.index.get_loc(t.exit_date) - 1])
        else:
            exits.append(t.exit_date)
    for dates, name, symbol, colour in (
        (longs, "long spread (buy A, sell B)", "triangle-up", "green"),
        (shorts, "short spread (sell A, buy B)", "triangle-down", "red"),
        (exits, "exit", "x", "black"),
    ):
        if dates:
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=z.loc[dates].to_numpy(),
                    mode="markers",
                    name=name,
                    marker=dict(symbol=symbol, size=11, color=colour),
                )
            )
    fig.update_layout(
        title=f"{result.name}: spread z-score (signal at close, fill next open)",
        yaxis_title="z-score",
        height=420,
        margin=dict(t=50),
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def prices_chart(close: pd.DataFrame, result: PairResult, trading_index: pd.Index) -> go.Figure:
    fig = go.Figure()
    for ticker in (result.ticker_a, result.ticker_b):
        s = close[ticker]
        fig.add_trace(
            go.Scatter(x=s.index, y=(s / s.iloc[0]).to_numpy(), mode="lines", name=ticker)
        )
    fig.add_vline(x=trading_index[0], line_dash="dash", line_color="grey")
    fig.add_annotation(
        x=trading_index[0], y=1.0, text="formation | trading", showarrow=False, yshift=-10
    )
    fig.update_layout(
        title=f"{result.ticker_a} and {result.ticker_b}: prices rebased to 1 at the start",
        yaxis_title="rebased price",
        height=380,
        margin=dict(t=50),
    )
    return fig


# --- page -------------------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Pairs Trading Research", layout="wide")
    st.title("Pairs trading research viewer")

    reports_dir = st.sidebar.text_input("Reports directory", value=_cli_reports_dir())
    if not (Path(reports_dir) / "summary.json").exists():
        st.error(
            f"No run found in `{reports_dir}`. Run `python -m pairs_trading run` first, or point "
            "the sidebar at a directory that contains summary.json and screened_pairs.csv."
        )
        return

    reports_dir = str(Path(reports_dir).resolve())
    run = load_run(reports_dir, run_stamp(reports_dir))
    base_cfg = config_from_summary(run["summary"])

    with st.sidebar.form("params"):
        st.markdown("**Signal**")
        window = st.number_input("z-score window (days)", 2, 250, base_cfg.zscore_window)
        entry_z = st.number_input("entry |z|", 0.5, 5.0, float(base_cfg.entry_z), 0.1)
        exit_z = st.number_input("exit |z|", 0.0, 4.9, float(base_cfg.exit_z), 0.1)
        stop_z = st.number_input(
            "stop |z| (0 = off)", 0.0, 10.0, float(base_cfg.stop_z or 0.0), 0.5
        )
        st.markdown("**Costs (bps per leg, per execution)**")
        slippage = st.number_input("slippage", 0.0, 100.0, float(base_cfg.slippage_bps), 1.0)
        commission = st.number_input(
            "commission", 0.0, 100.0, float(base_cfg.transaction_cost_bps), 1.0
        )
        st.markdown("**Screen**")
        pvalue = st.number_input(
            "cointegration p-value threshold",
            0.001,
            0.5,
            float(base_cfg.coint_pvalue),
            0.005,
            format="%.3f",  # the default "%0.2f" would show 0.001 as 0.00 and 0.025 as 0.03
        )
        submitted = st.form_submit_button("Re-run signals and backtest")

    try:
        cfg = replace(
            base_cfg,
            zscore_window=int(window),
            entry_z=float(entry_z),
            exit_z=float(exit_z),
            stop_z=None if stop_z == 0 else float(stop_z),
            slippage_bps=float(slippage),
            transaction_cost_bps=float(commission),
            coint_pvalue=float(pvalue),
        )
    except ValueError as exc:
        st.sidebar.error(str(exc))
        return
    changed = cfg != base_cfg
    st.sidebar.caption(
        "Parameters differ from the committed run; charts below are recomputed live."
        if changed
        else "Showing the committed run's parameters."
    )
    if submitted and not changed:
        st.sidebar.info("Nothing changed.")

    selected = select_cointegrated(
        run["screened"], cfg.coint_pvalue, cfg.require_positive_hedge_ratio
    )
    if selected.empty:
        st.warning(
            f"No pair passes p < {cfg.coint_pvalue:g}"
            + (" with a positive hedge ratio" if cfg.require_positive_hedge_ratio else "")
            + f" (the smallest screened p-value is {run['screened']['pvalue'].min():.4f}); "
            "raise the threshold."
        )
        return

    out = run_backtest(_cfg_key(cfg), run["screened"].to_json(orient="split", date_format="iso"))
    results: list[PairResult] = out["results"]
    table: pd.DataFrame = out["table"]
    m = out["metrics"]
    equity = 1.0 + out["portfolio_pnl"].cumsum()
    trading_index = out["portfolio_pnl"].index

    scr = run["summary"]["screening"]
    st.caption(
        f"Universe **{', '.join(run['summary']['config']['sectors'])}** · formation "
        f"{cfg.start} to {cfg.formation_end} · trading {trading_index[0].date()} to "
        f"{trading_index[-1].date()} · {scr['n_pairs_tested']} pairs tested, "
        f"{len(results)} backtested"
    )

    c = st.columns(6)
    c[0].metric("Sharpe (net)", f"{m['sharpe']:.2f}")
    c[1].metric("Annualised return", f"{100 * m['annualized_return']:.2f}%")
    c[2].metric("Max drawdown", f"{100 * m['max_drawdown']:.1f}%")
    c[3].metric("Trades", f"{m['n_trades']:,}")
    c[4].metric("Win rate", f"{100 * m['win_rate']:.1f}%")
    c[5].metric("Pairs profitable", f"{int((table['net_pnl'] > 0).sum())} / {len(table)}")

    tab_portfolio, tab_pairs, tab_screen, tab_trades = st.tabs(
        ["Portfolio", "Pairs", "Screen", "Trades"]
    )

    with tab_portfolio:
        st.plotly_chart(
            equity_chart(equity, f"Equal-weight portfolio of {len(results)} pairs, net of costs"),
            width="stretch",
        )
        yearly = out["portfolio_pnl"].groupby(out["portfolio_pnl"].index.year).sum()
        st.dataframe(
            yearly.rename("net P&L").map(lambda v: f"{100 * v:+.2f}%").to_frame().T,
            width="stretch",
        )

    with tab_pairs:
        show = table[
            [
                "ticker_a",
                "ticker_b",
                "pvalue",
                "hedge_ratio",
                "half_life",
                "sharpe",
                "total_return",
                "max_drawdown",
                "n_trades",
                "win_rate",
            ]
        ]
        st.dataframe(
            show.style.format(
                {
                    "pvalue": "{:.4f}",
                    "hedge_ratio": "{:.3f}",
                    "half_life": "{:.1f}",
                    "sharpe": "{:.2f}",
                    "total_return": "{:+.1%}",
                    "max_drawdown": "{:.1%}",
                    "win_rate": "{:.0%}",
                }
            ),
            width="stretch",
            height=300,
        )
        names = [f"{r.ticker_a}/{r.ticker_b}" for r in table.itertuples()]
        choice = st.selectbox("Pair", names, index=0)
        result = next(r for r in results if r.name == choice)
        row = table.loc[
            (table.ticker_a == result.ticker_a) & (table.ticker_b == result.ticker_b)
        ].iloc[0]
        st.markdown(
            f"**{result.name}** · hedge ratio {result.hedge_ratio:.3f} "
            f"(spread = {result.ticker_a} - {result.hedge_ratio:.3f} x {result.ticker_b}) · "
            f"cointegration p = {row['pvalue']:.4f} · half-life {row['half_life']:.1f} days · "
            f"{len(result.trades)} trades · Sharpe {row['sharpe']:.2f}"
        )
        st.plotly_chart(zscore_chart(result, out["zscores"][result.name], cfg), width="stretch")
        left, right = st.columns(2)
        left.plotly_chart(prices_chart(out["close"], result, trading_index), width="stretch")
        right.plotly_chart(equity_chart(result.equity, f"{result.name}: equity"), width="stretch")
        st.dataframe(
            trades_to_frame([result])[
                [
                    "direction",
                    "signal_date",
                    "entry_date",
                    "exit_date",
                    "holding_days",
                    "exit_reason",
                    "gross_pnl",
                    "slippage_cost",
                    "commission_cost",
                    "net_pnl",
                ]
            ],
            width="stretch",
            height=260,
        )

    with tab_screen:
        if run["pretest"] is not None:
            dropped = run["pretest"].loc[~run["pretest"]["is_i1"]]
            names = ", ".join(dropped["ticker"])
            verdict = (
                f"{len(dropped)} dropped as stationary on their own ({names})"
                if len(dropped)
                else "none dropped"
            )
            st.markdown(
                f"**Step 1, unit-root pretest:** {len(run['pretest'])} tickers tested, {verdict}."
            )
            st.dataframe(run["pretest"], width="stretch", height=240)
        st.markdown(
            f"**Steps 2 to 4, Engle-Granger on every pair:** {scr['n_pairs_tested']} tested, "
            f"{scr['n_cointegrated']} at p < {scr['pvalue_threshold']} "
            f"(about {scr['expected_false_positives_at_threshold']:.0f} expected by chance), "
            f"{scr['n_selected']} selected."
        )
        st.dataframe(run["screened"], width="stretch", height=360)

    with tab_trades:
        st.dataframe(out["trades"], width="stretch", height=500)
        st.download_button(
            "Download trades.csv",
            out["trades"].to_csv(index=False).encode("utf-8"),
            file_name="trades.csv",
            mime="text/csv",
        )


main()
