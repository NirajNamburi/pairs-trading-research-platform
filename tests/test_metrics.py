import math

import numpy as np
import pandas as pd
import pytest

from conftest import bdays
from pairs_trading.backtest import PairResult, Trade, aggregate_portfolio
from pairs_trading.metrics import (
    annualized_return,
    annualized_volatility,
    avg_holding_days,
    max_drawdown,
    sharpe_ratio,
    summarize,
    summarize_portfolio,
    total_return,
    win_rate,
)


def _trade(net_pnl: float, holding_days: int = 5) -> Trade:
    d = pd.Timestamp("2022-01-03")
    return Trade(
        ticker_a="A",
        ticker_b="B",
        direction=1,
        signal_date=d,
        entry_date=d,
        exit_date=d,
        holding_days=holding_days,
        exit_reason="signal",
        entry_price_a=1.0,
        entry_price_b=1.0,
        exit_price_a=1.0,
        exit_price_b=1.0,
        qty_a=0.5,
        qty_b=0.5,
        gross_pnl=net_pnl + 0.002,
        slippage_cost=0.001,
        commission_cost=0.001,
        net_pnl=net_pnl,
    )


def test_sharpe_is_zero_for_zero_mean_and_nan_for_zero_volatility():
    idx = bdays(4)
    assert sharpe_ratio(pd.Series([0.01, -0.01, 0.01, -0.01], index=idx)) == pytest.approx(0.0)
    assert math.isnan(sharpe_ratio(pd.Series([0.01] * 4, index=idx)))
    assert math.isnan(sharpe_ratio(pd.Series([0.01], index=idx[:1])))


def test_sharpe_matches_manual_formula_and_is_scale_invariant():
    idx = bdays(5)
    pnl = pd.Series([0.01, 0.02, -0.005, 0.015, 0.0], index=idx)
    manual = pnl.mean() / pnl.std(ddof=1) * math.sqrt(252)
    assert sharpe_ratio(pnl) == pytest.approx(manual)
    assert sharpe_ratio(pnl * 10) == pytest.approx(manual)


def test_sharpe_subtracts_risk_free_rate():
    idx = bdays(3)
    pnl = pd.Series([0.0002, 0.0000, 0.0001], index=idx)  # mean 0.0001 per day
    assert sharpe_ratio(pnl, risk_free_rate=0.0001 * 252) == pytest.approx(0.0)


def test_max_drawdown_is_relative_to_running_peak():
    idx = bdays(4)
    assert max_drawdown(pd.Series([1.0, 1.2, 0.9, 1.0], index=idx)) == pytest.approx(-0.25)
    assert max_drawdown(pd.Series([1.0, 1.1, 1.2, 1.3], index=idx)) == 0.0
    assert math.isnan(max_drawdown(pd.Series([], dtype=float)))


def test_returns_and_volatility():
    idx = bdays(4)
    pnl = pd.Series([0.01, 0.02, 0.03, 0.04], index=idx)
    assert total_return(pnl) == pytest.approx(0.10)
    assert annualized_return(pnl) == pytest.approx(0.025 * 252)
    assert annualized_volatility(pnl) == pytest.approx(
        np.std([0.01, 0.02, 0.03, 0.04], ddof=1) * math.sqrt(252)
    )


def test_trade_statistics():
    trades = [_trade(0.01, 3), _trade(-0.02, 7), _trade(0.005, 2)]
    assert win_rate(trades) == pytest.approx(2 / 3)
    assert avg_holding_days(trades) == pytest.approx(4.0)
    assert math.isnan(win_rate([]))
    assert math.isnan(avg_holding_days([]))


def test_summarize_bundle(cfg):
    idx = bdays(3)
    pnl = pd.Series([0.01, -0.005, 0.02], index=idx)
    trades = [_trade(0.01), _trade(0.015)]
    s = summarize(pnl, trades, cfg)
    assert s["total_return"] == pytest.approx(0.025)
    assert s["n_trades"] == 2
    assert s["win_rate"] == 1.0
    assert s["net_pnl"] == pytest.approx(0.025)
    assert s["gross_pnl"] == pytest.approx(0.029)
    assert s["total_costs"] == pytest.approx(0.004)
    assert s["max_drawdown"] == pytest.approx(-0.005 / 1.01)
    assert s["net_pnl"] == pytest.approx(s["total_return"])  # one $1 book: same basis
    assert set(s) == {
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "max_drawdown",
        "n_trades",
        "win_rate",
        "avg_holding_days",
        "gross_pnl",
        "total_costs",
        "net_pnl",
    }


def test_summarize_portfolio_puts_trade_sums_on_the_total_capital_basis(cfg):
    idx = bdays(3)
    r1 = PairResult("A", "B", 1.0, [_trade(0.01)], pd.Series([0.01, 0.0, 0.0], index=idx))
    r2 = PairResult(
        "C", "D", 1.0, [_trade(0.04), _trade(0.02)], pd.Series([0.03, 0.02, 0.01], index=idx)
    )
    portfolio_pnl = aggregate_portfolio([r1, r2])
    s = summarize_portfolio(portfolio_pnl, [r1, r2], cfg)

    assert s["total_return"] == pytest.approx(0.035)  # mean of the two $1 books
    assert s["net_pnl"] == pytest.approx(s["total_return"])  # not the raw sum 0.07
    assert s["gross_pnl"] == pytest.approx((0.07 + 3 * 0.002) / 2)
    assert s["total_costs"] == pytest.approx(3 * 0.002 / 2)
    assert s["n_trades"] == 3  # counts are not rescaled
    assert s["win_rate"] == 1.0
    with pytest.raises(ValueError):
        summarize_portfolio(portfolio_pnl, [], cfg)
