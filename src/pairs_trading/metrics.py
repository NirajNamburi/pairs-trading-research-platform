"""Performance metrics on daily P&L series and trade lists.

All P&L series are in dollars on $1 of capital, so they can be read directly as returns.
Annualisation uses ``trading_days_per_year`` (252). The Sharpe ratio subtracts the (annual)
risk-free rate divided by the number of periods; the default rate is 0, which is stated in the
report rather than hidden.
"""

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

from pairs_trading.backtest import PairResult, Trade
from pairs_trading.config import PipelineConfig


def total_return(daily_pnl: pd.Series) -> float:
    return float(daily_pnl.sum())


def annualized_return(daily_pnl: pd.Series, periods_per_year: int = 252) -> float:
    """Arithmetic annualised return (mean daily P&L times periods per year)."""
    if len(daily_pnl) == 0:
        return math.nan
    return float(daily_pnl.mean() * periods_per_year)


def annualized_volatility(daily_pnl: pd.Series, periods_per_year: int = 252) -> float:
    if len(daily_pnl) < 2:
        return math.nan
    return float(daily_pnl.std(ddof=1) * math.sqrt(periods_per_year))


def sharpe_ratio(
    daily_pnl: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Annualised Sharpe ratio. NaN if fewer than two observations or zero volatility."""
    if len(daily_pnl) < 2:
        return math.nan
    excess = daily_pnl - risk_free_rate / periods_per_year
    vol = excess.std(ddof=1)
    if not np.isfinite(vol) or vol == 0.0:
        return math.nan
    return float(excess.mean() / vol * math.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline as a (non-positive) fraction of the running peak."""
    if len(equity) == 0:
        return math.nan
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def win_rate(trades: Sequence[Trade]) -> float:
    if not trades:
        return math.nan
    wins = sum(1 for t in trades if t.net_pnl > 0)
    return wins / len(trades)


def avg_holding_days(trades: Sequence[Trade]) -> float:
    if not trades:
        return math.nan
    return float(np.mean([t.holding_days for t in trades]))


def summarize(
    daily_pnl: pd.Series, trades: Sequence[Trade], cfg: PipelineConfig
) -> dict[str, float]:
    """Standard metric bundle for one pair or for the portfolio."""
    equity = 1.0 + daily_pnl.cumsum()
    return {
        "total_return": total_return(daily_pnl),
        "annualized_return": annualized_return(daily_pnl, cfg.trading_days_per_year),
        "annualized_volatility": annualized_volatility(daily_pnl, cfg.trading_days_per_year),
        "sharpe": sharpe_ratio(daily_pnl, cfg.risk_free_rate, cfg.trading_days_per_year),
        "max_drawdown": max_drawdown(equity),
        "n_trades": len(trades),
        "win_rate": win_rate(trades),
        "avg_holding_days": avg_holding_days(trades),
        "gross_pnl": float(sum(t.gross_pnl for t in trades)),
        "total_costs": float(sum(t.slippage_cost + t.commission_cost for t in trades)),
        "net_pnl": float(sum(t.net_pnl for t in trades)),
    }


def summarize_portfolio(
    portfolio_pnl: pd.Series, results: Sequence[PairResult], cfg: PipelineConfig
) -> dict[str, float]:
    """Metric bundle for the equal-weight portfolio, every key per $1 of *total* capital.

    ``portfolio_pnl`` (:func:`~pairs_trading.backtest.aggregate_portfolio`) is already the mean
    across pairs, but each trade was booked on its own pair's $1 book, which is only
    ``1 / n_pairs`` of the portfolio's capital. The trade-level dollar sums are rescaled by that
    weight so ``net_pnl == total_return`` holds for the portfolio exactly as it does per pair.
    """
    if not results:
        raise ValueError("no pair results to summarize")
    trades = [t for r in results for t in r.trades]
    metrics = summarize(portfolio_pnl, trades, cfg)
    for key in ("gross_pnl", "total_costs", "net_pnl"):
        metrics[key] /= len(results)
    return metrics
