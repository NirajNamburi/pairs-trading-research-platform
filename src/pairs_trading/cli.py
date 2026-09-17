"""Command-line entry point: ``python -m pairs_trading run [options]``."""

import argparse
import logging
import sys
from pathlib import Path

from pairs_trading.config import PipelineConfig
from pairs_trading.pipeline import run_pipeline
from pairs_trading.report import render_summary_markdown
from pairs_trading.universe import available_sectors

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    defaults = PipelineConfig()
    parser = argparse.ArgumentParser(
        prog="pairs-trading",
        description=(
            "Statistical arbitrage research pipeline: cointegration screen, "
            "z-score signal, cost-aware backtest."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the full pipeline and write reports")

    g = run.add_argument_group("data")
    g.add_argument(
        "--start", default=defaults.start, help="first date of the formation period (inclusive)"
    )
    g.add_argument(
        "--formation-end",
        default=defaults.formation_end,
        help="last date of the formation period (inclusive)",
    )
    g.add_argument(
        "--end", default=defaults.end, help="last date of the trading period (inclusive)"
    )
    g.add_argument(
        "--sectors",
        nargs="+",
        default=list(defaults.sectors),
        choices=available_sectors(),
        metavar="SECTOR",
        help=(
            f"sectors to screen within; choices: {', '.join(available_sectors())}. "
            "Overlapping universes (e.g. utilities_top50 with utilities) cannot be combined"
        ),
    )
    g.add_argument(
        "--refresh-data", action="store_true", help="ignore the Parquet cache and re-download"
    )
    g.add_argument("--data-dir", type=Path, default=defaults.data_dir)

    g = run.add_argument_group("strategy")
    g.add_argument(
        "--pvalue",
        type=float,
        default=defaults.coint_pvalue,
        help="cointegration p-value threshold",
    )
    g.add_argument(
        "--no-unit-root-pretest",
        action="store_true",
        help=(
            "skip step 1 (ADF on each stock's price levels); together with --single-ordering "
            "this reproduces the naive screen"
        ),
    )
    g.add_argument(
        "--unit-root-pvalue",
        type=float,
        default=defaults.unit_root_pvalue,
        help="a stock is I(1)-eligible when its levels ADF p-value is above this",
    )
    g.add_argument(
        "--single-ordering",
        action="store_true",
        help="test only the alphabetical orientation of each pair instead of both",
    )
    g.add_argument("--zscore-window", type=int, default=defaults.zscore_window)
    g.add_argument("--entry-z", type=float, default=defaults.entry_z)
    g.add_argument("--exit-z", type=float, default=defaults.exit_z)
    g.add_argument(
        "--stop-z", type=float, default=None, help="optional blow-out stop; disabled by default"
    )
    g.add_argument("--slippage-bps", type=float, default=defaults.slippage_bps)
    g.add_argument("--commission-bps", type=float, default=defaults.transaction_cost_bps)
    g.add_argument(
        "--risk-free-rate",
        type=float,
        default=defaults.risk_free_rate,
        help="annual, used in the Sharpe ratio",
    )

    g = run.add_argument_group("output")
    g.add_argument("--reports-dir", type=Path, default=defaults.reports_dir)
    g.add_argument("--n-plot-pairs", type=int, default=defaults.n_plot_pairs)
    g.add_argument(
        "--n-jobs", type=int, default=defaults.n_jobs, help="processes for the cointegration screen"
    )
    g.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        start=args.start,
        formation_end=args.formation_end,
        end=args.end,
        sectors=tuple(args.sectors),
        coint_pvalue=args.pvalue,
        unit_root_pretest=not args.no_unit_root_pretest,
        unit_root_pvalue=args.unit_root_pvalue,
        test_both_orderings=not args.single_ordering,
        zscore_window=args.zscore_window,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        stop_z=args.stop_z,
        slippage_bps=args.slippage_bps,
        transaction_cost_bps=args.commission_bps,
        risk_free_rate=args.risk_free_rate,
        n_jobs=args.n_jobs,
        data_dir=args.data_dir,
        reports_dir=args.reports_dir,
        n_plot_pairs=args.n_plot_pairs,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        cfg = config_from_args(args)
    except ValueError as exc:
        log.error("invalid configuration: %s", exc)
        return 2

    try:
        result = run_pipeline(cfg, refresh_data=args.refresh_data)
    except (RuntimeError, ValueError) as exc:
        # RuntimeError: the pipeline gave up (no data, no pairs passed). ValueError: an input
        # the config validator cannot see is unusable (e.g. a trading window with no trading
        # days, no candidate pairs). Either way a message beats a traceback; --log-level DEBUG
        # keeps the traceback for genuine bugs.
        log.error("%s", exc, exc_info=log.isEnabledFor(logging.DEBUG))
        return 1

    sys.stdout.write("\n" + render_summary_markdown(result.summary, result.table) + "\n")
    sys.stdout.write("Report files:\n" + "\n".join(f"  {p}" for p in result.files.values()) + "\n")
    return 0
