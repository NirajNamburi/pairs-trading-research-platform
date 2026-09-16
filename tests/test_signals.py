import numpy as np
import pandas as pd
import pytest

from conftest import bdays
from pairs_trading.signals import (
    compute_spread,
    generate_target_positions,
    rolling_zscore,
    zscore_for_trading_period,
)


def _positions(z_values, entry=2.0, exit_=0.5, stop=None) -> list[int]:
    z = pd.Series(z_values, index=bdays(len(z_values)), dtype=float)
    return generate_target_positions(z, entry, exit_, stop).tolist()


def test_compute_spread_uses_hedge_ratio():
    idx = bdays(3)
    a = pd.Series([10.0, 11.0, 12.0], index=idx)
    b = pd.Series([4.0, 4.0, 5.0], index=idx)
    spread = compute_spread(a, b, hedge_ratio=2.0)
    assert spread.tolist() == [2.0, 3.0, 2.0]


def test_rolling_zscore_matches_manual_calculation():
    idx = bdays(6)
    spread = pd.Series([1.0, 2.0, 3.0, 6.0, 5.0, 4.0], index=idx)
    z = rolling_zscore(spread, window=3)
    assert z.iloc[:2].isna().all()
    window = spread.iloc[1:4]
    expected = (spread.iloc[3] - window.mean()) / window.std(ddof=1)
    assert z.iloc[3] == pytest.approx(expected)


def test_state_machine_enters_and_exits_on_thresholds():
    z = [np.nan, 0.0, 2.5, 1.0, 0.4, -2.5, -1.0, 0.6, 0.3]
    assert _positions(z) == [0, 0, -1, -1, 0, 1, 1, 1, 0]


def test_entry_requires_strict_threshold_breach():
    assert _positions([2.0, -2.0, 0.0]) == [0, 0, 0]
    assert _positions([2.01]) == [-1]


def test_stop_closes_and_blocks_reentry_until_back_inside_entry_band():
    z = [2.5, 3.5, 2.5, 0.2, 2.5]
    assert _positions(z, stop=3.0) == [-1, 0, 0, 0, -1]


def test_nan_zscore_holds_current_position():
    assert _positions([2.5, np.nan, 0.1]) == [-1, -1, 0]


def test_invalid_thresholds_raise():
    z = pd.Series([0.0], index=bdays(1))
    with pytest.raises(ValueError):
        generate_target_positions(z, entry_z=1.0, exit_z=1.5)
    with pytest.raises(ValueError):
        generate_target_positions(z, entry_z=2.0, exit_z=0.5, stop_z=1.0)


def test_trading_period_zscore_is_warmed_up_on_prior_closes():
    idx = bdays(100)
    rng = np.random.default_rng(0)
    close_b = pd.Series(100.0 + np.cumsum(rng.normal(size=100)), index=idx)
    close_a = pd.Series(2.0 * close_b + rng.normal(size=100), index=idx)
    trading = idx[50:]
    z = zscore_for_trading_period(
        close_a, close_b, hedge_ratio=2.0, window=10, trading_index=trading
    )

    assert z.index.equals(trading)
    assert not z.isna().any()  # warm-up came from the 10 closes before the trading period
    full = rolling_zscore(compute_spread(close_a, close_b, 2.0), 10)
    pd.testing.assert_series_equal(z, full.loc[trading], check_names=False)


def test_zscore_has_no_look_ahead():
    idx = bdays(60)
    rng = np.random.default_rng(1)
    close_b = pd.Series(100.0 + np.cumsum(rng.normal(size=60)), index=idx)
    close_a = pd.Series(close_b + rng.normal(size=60), index=idx)
    base = rolling_zscore(compute_spread(close_a, close_b, 1.0), 10)

    shocked_a = close_a.copy()
    shocked_a.iloc[40] += 50.0  # a huge move on day 40
    shocked = rolling_zscore(compute_spread(shocked_a, close_b, 1.0), 10)

    pd.testing.assert_series_equal(base.iloc[:40], shocked.iloc[:40])
    assert base.iloc[40] != shocked.iloc[40]
