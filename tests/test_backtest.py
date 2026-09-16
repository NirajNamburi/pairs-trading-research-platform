import numpy as np
import pandas as pd
import pytest

from conftest import bdays
from pairs_trading.backtest import PairResult, aggregate_portfolio, backtest_pair, trades_to_frame
from pairs_trading.config import PipelineConfig
from pairs_trading.signals import generate_target_positions, rolling_zscore


def _series(values, idx) -> pd.Series:
    return pd.Series(values, index=idx, dtype=float)


def _hand_example(cfg: PipelineConfig):
    """Six-day toy example with a single long-spread round trip (hedge ratio 1).

    Signal at close of day 1 -> entry at open of day 2 -> exit signal at close of day 3 ->
    exit at open of day 4. A rallies from 10 to 11 between day 2's open and close.
    """
    idx = bdays(6)
    close_a = _series([10, 10, 11, 11, 11, 11], idx)
    open_a = _series([10, 10, 10, 10, 11, 11], idx)
    close_b = _series([10] * 6, idx)
    open_b = _series([10] * 6, idx)
    target = pd.Series([0, 1, 1, 0, 0, 0], index=idx)
    return backtest_pair(
        close_a=close_a,
        close_b=close_b,
        open_a=open_a,
        open_b=open_b,
        target_positions=target,
        hedge_ratio=1.0,
        cfg=cfg,
        ticker_a="A",
        ticker_b="B",
    )


def test_hand_computed_round_trip(cfg):
    result = _hand_example(cfg)  # 5 bps slippage, 5 bps commission
    assert len(result.trades) == 1
    t = result.trades[0]

    assert t.direction == 1
    assert t.signal_date == result.daily_pnl.index[1]
    assert t.entry_date == result.daily_pnl.index[2]
    assert t.exit_date == result.daily_pnl.index[4]
    assert t.holding_days == 2
    assert t.exit_reason == "signal"

    # $1 gross notional: qty = 1 / (10 + 1 * 10)
    assert t.qty_a == pytest.approx(0.05)
    assert t.qty_b == pytest.approx(0.05)
    # adverse fills: buy A above the open, sell B below it; the reverse on exit
    assert t.entry_price_a == pytest.approx(10.005)
    assert t.entry_price_b == pytest.approx(9.995)
    assert t.exit_price_a == pytest.approx(10.9945)
    assert t.exit_price_b == pytest.approx(10.005)

    assert t.gross_pnl == pytest.approx(0.05)
    assert t.slippage_cost == pytest.approx(0.0005 * (1.0 + 1.05))
    assert t.commission_cost == pytest.approx(0.0005 * (1.0 + 1.05))
    assert t.net_pnl == pytest.approx(0.05 - 0.00205)

    expected_daily = [0.0, 0.0, 0.049, 0.0, -0.00105, 0.0]
    assert result.daily_pnl.tolist() == pytest.approx(expected_daily, abs=1e-12)
    assert result.equity.iloc[-1] == pytest.approx(1.0 + t.net_pnl)


def test_zero_costs_make_net_equal_gross(cfg):
    result = _hand_example(cfg.replace(slippage_bps=0.0, transaction_cost_bps=0.0))
    t = result.trades[0]
    assert t.slippage_cost == 0.0
    assert t.commission_cost == 0.0
    assert t.net_pnl == pytest.approx(t.gross_pnl) == pytest.approx(0.05)


def test_execution_happens_at_next_open_not_signal_close(cfg):
    """Changing the close on the signal day must not change that day's P&L."""
    idx = bdays(6)
    close_b = open_b = _series([10] * 6, idx)
    open_a = _series([10] * 6, idx)
    target = pd.Series([0, 1, 1, 1, 1, 1], index=idx)

    base = backtest_pair(
        close_a=_series([10, 10, 10, 10, 10, 10], idx),
        close_b=close_b,
        open_a=open_a,
        open_b=open_b,
        target_positions=target,
        hedge_ratio=1.0,
        cfg=cfg,
    )
    shocked = backtest_pair(
        close_a=_series([10, 15, 10, 10, 10, 10], idx),  # close jumps on the signal day only
        close_b=close_b,
        open_a=open_a,
        open_b=open_b,
        target_positions=target,
        hedge_ratio=1.0,
        cfg=cfg,
    )
    assert base.daily_pnl.iloc[1] == 0.0
    assert shocked.daily_pnl.iloc[1] == 0.0
    assert shocked.trades[0].entry_date == idx[2]
    assert shocked.trades[0].entry_price_a == pytest.approx(10.005)


def test_daily_pnl_sums_to_trade_net_pnl_on_random_data(cfg):
    idx = bdays(400)
    rng = np.random.default_rng(7)
    close_b = _series(100.0 + np.cumsum(rng.normal(size=400)), idx)
    close_a = _series(
        1.5 * close_b + np.cumsum(rng.normal(scale=0.5, size=400)) * 0.1 + rng.normal(size=400), idx
    )
    open_a = close_a.shift(1).fillna(close_a.iloc[0]) * (1 + rng.normal(scale=0.002, size=400))
    open_b = close_b.shift(1).fillna(close_b.iloc[0]) * (1 + rng.normal(scale=0.002, size=400))
    z = rolling_zscore(close_a - 1.5 * close_b, 20)
    target = generate_target_positions(z, 2.0, 0.5)
    assert target.abs().sum() > 0  # the synthetic path must actually trade

    result = backtest_pair(
        close_a=close_a,
        close_b=close_b,
        open_a=open_a,
        open_b=open_b,
        target_positions=target,
        hedge_ratio=1.5,
        cfg=cfg,
    )
    assert len(result.trades) >= 2
    assert result.daily_pnl.sum() == pytest.approx(sum(t.net_pnl for t in result.trades))
    for t in result.trades:
        assert t.entry_date > t.signal_date
        # An entry at the last open is force-closed at that day's close, so equality is legal.
        assert t.exit_date >= t.entry_date
        if t.exit_reason == "signal":
            assert t.exit_date > t.entry_date
        assert t.slippage_cost >= 0 and t.commission_cost >= 0
        assert t.net_pnl == pytest.approx(t.gross_pnl - t.slippage_cost - t.commission_cost)
        assert t.qty_a * t.entry_price_a / (
            1 + cfg.slippage_bps / 1e4 * t.direction
        ) + t.qty_b * t.entry_price_b / (1 - cfg.slippage_bps / 1e4 * t.direction) == pytest.approx(
            1.0
        )


def test_no_signal_means_no_trades_and_flat_equity(cfg):
    idx = bdays(10)
    ones = _series([10] * 10, idx)
    result = backtest_pair(
        close_a=ones,
        close_b=ones,
        open_a=ones,
        open_b=ones,
        target_positions=pd.Series([0] * 10, index=idx),
        hedge_ratio=1.0,
        cfg=cfg,
    )
    assert result.trades == []
    assert (result.daily_pnl == 0).all()
    assert (result.equity == 1.0).all()


def test_open_position_is_force_closed_at_end_of_period(cfg):
    idx = bdays(5)
    ones = _series([10] * 5, idx)
    result = backtest_pair(
        close_a=ones,
        close_b=ones,
        open_a=ones,
        open_b=ones,
        target_positions=pd.Series([0, 0, 1, 1, 1], index=idx),
        hedge_ratio=1.0,
        cfg=cfg,
    )
    assert len(result.trades) == 1
    t = result.trades[0]
    assert t.exit_reason == "end_of_period"
    assert t.exit_date == idx[-1]
    assert t.gross_pnl == pytest.approx(0.0)
    assert t.net_pnl == pytest.approx(-(t.slippage_cost + t.commission_cost))
    assert result.daily_pnl.sum() == pytest.approx(t.net_pnl)


def test_entry_on_the_last_day_is_force_closed_at_that_days_close(cfg):
    """A signal at the penultimate close fills at the last open; holding_days is legitimately 0."""
    idx = bdays(5)
    tens = _series([10] * 5, idx)
    result = backtest_pair(
        close_a=_series([10, 10, 10, 10, 11], idx),  # A rallies between the last open and close
        close_b=tens,
        open_a=tens,
        open_b=tens,
        target_positions=pd.Series([0, 0, 0, 1, 0], index=idx),
        hedge_ratio=1.0,
        cfg=cfg,
    )
    assert len(result.trades) == 1
    t = result.trades[0]
    assert t.exit_reason == "end_of_period"
    assert t.signal_date == idx[3]
    assert t.entry_date == t.exit_date == idx[4]
    assert t.holding_days == 0
    assert t.gross_pnl == pytest.approx(0.05)  # qty 1 / (10 + 10) on a $1 move in A
    assert t.net_pnl == pytest.approx(0.05 - 0.00205)  # same costs as the hand-computed round trip
    assert result.daily_pnl.iloc[:4].tolist() == [0.0] * 4
    assert result.daily_pnl.sum() == pytest.approx(t.net_pnl)


def test_direct_flip_closes_then_opens_on_the_same_open(cfg):
    idx = bdays(5)
    ones = _series([10] * 5, idx)
    result = backtest_pair(
        close_a=ones,
        close_b=ones,
        open_a=ones,
        open_b=ones,
        target_positions=pd.Series([0, 1, -1, 0, 0], index=idx),
        hedge_ratio=1.0,
        cfg=cfg,
    )
    assert [t.direction for t in result.trades] == [1, -1]
    assert result.trades[0].exit_date == result.trades[1].entry_date == idx[3]
    assert result.trades[0].exit_reason == "signal"


def test_invalid_inputs_raise(cfg):
    idx = bdays(3)
    ones = _series([10] * 3, idx)
    with_nan = _series([10, np.nan, 10], idx)
    target = pd.Series([0, 1, 0], index=idx)
    with pytest.raises(ValueError):
        backtest_pair(
            close_a=with_nan,
            close_b=ones,
            open_a=ones,
            open_b=ones,
            target_positions=target,
            hedge_ratio=1.0,
            cfg=cfg,
        )
    with pytest.raises(ValueError):
        backtest_pair(
            close_a=ones,
            close_b=ones,
            open_a=ones,
            open_b=ones,
            target_positions=target,
            hedge_ratio=-1.0,
            cfg=cfg,
        )
    with pytest.raises(ValueError):
        backtest_pair(
            close_a=ones,
            close_b=ones,
            open_a=ones,
            open_b=ones,
            target_positions=target.iloc[:0],
            hedge_ratio=1.0,
            cfg=cfg,
        )
    with pytest.raises(ValueError):
        backtest_pair(
            close_a=ones,
            close_b=ones,
            open_a=ones,
            open_b=ones,
            target_positions=pd.Series([0, 2, 0], index=idx),
            hedge_ratio=1.0,
            cfg=cfg,
        )


def test_aggregate_portfolio_is_equal_weight_mean():
    idx = bdays(3)
    r1 = PairResult(
        "A", "B", 1.0, [], _series([0.01, 0.00, -0.01], idx), _series([1.01, 1.01, 1.00], idx)
    )
    r2 = PairResult(
        "C", "D", 1.0, [], _series([0.03, 0.02, 0.01], idx), _series([1.03, 1.05, 1.06], idx)
    )
    portfolio = aggregate_portfolio([r1, r2])
    assert portfolio.tolist() == pytest.approx([0.02, 0.01, 0.0])
    with pytest.raises(ValueError):
        aggregate_portfolio([])


def test_trades_to_frame_round_trips_columns(cfg):
    result = _hand_example(cfg)
    frame = trades_to_frame([result])
    assert len(frame) == 1
    assert frame.iloc[0]["net_pnl"] == pytest.approx(result.trades[0].net_pnl)
    assert trades_to_frame([]).empty
