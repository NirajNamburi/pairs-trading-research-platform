"""Stage 3 - signal generation.

``spread_t = A_t - hedge_ratio * B_t`` (hedge ratio fixed from the formation regression)
``z_t = (spread_t - rolling_mean) / rolling_std`` over a trailing window.

A rolling window is used rather than the full-history mean/std because the level of the spread
drifts over time; a 20-30 day window tracks that drift.

Rules (classic Gatev, Goetzmann & Rouwenhorst style thresholds):

* flat  and z >  +entry -> short the spread (short A, long B)
* flat  and z <  -entry -> long the spread (long A, short B)
* in a position and |z| < exit -> close
* optional: in a position and |z| > stop -> close (blow-out stop) and stay out until |z| falls
  back below the entry threshold, so the stop is not immediately re-entered.

The target position is *decided at the close* of day ``t``. It is the backtest's job to execute
it at the *open* of day ``t + 1`` (see :mod:`pairs_trading.backtest`); nothing here looks ahead.
"""

import numpy as np
import pandas as pd

LONG_SPREAD = 1
SHORT_SPREAD = -1
FLAT = 0


def compute_spread(price_a: pd.Series, price_b: pd.Series, hedge_ratio: float) -> pd.Series:
    """``price_a - hedge_ratio * price_b`` on the intersection of the two indexes."""
    a, b = price_a.align(price_b, join="inner")
    spread = a - hedge_ratio * b
    spread.name = "spread"
    return spread


def rolling_zscore(spread: pd.Series, window: int) -> pd.Series:
    """Trailing z-score; NaN until ``window`` observations are available."""
    if window < 2:
        raise ValueError("window must be at least 2")
    mean = spread.rolling(window, min_periods=window).mean()
    std = spread.rolling(window, min_periods=window).std(ddof=1)
    z = (spread - mean) / std
    z.name = "zscore"
    return z


def generate_target_positions(
    zscore: pd.Series,
    entry_z: float,
    exit_z: float,
    stop_z: float | None = None,
) -> pd.Series:
    """Map a z-score path to a target position in ``{-1, 0, +1}`` decided at each close.

    NaN z-scores (warm-up) hold the current position (which is flat during warm-up).
    """
    if not 0.0 < exit_z < entry_z:
        raise ValueError("expected 0 < exit_z < entry_z")
    if stop_z is not None and stop_z <= entry_z:
        raise ValueError("stop_z must exceed entry_z")

    z = zscore.to_numpy(dtype=float)
    out = np.zeros(len(z), dtype=np.int8)
    position = FLAT
    blocked_after_stop = False

    for i, value in enumerate(z):
        if np.isnan(value):
            out[i] = position
            continue
        magnitude = abs(value)
        if position == FLAT:
            if blocked_after_stop:
                if magnitude < entry_z:
                    blocked_after_stop = False
            elif value > entry_z:
                position = SHORT_SPREAD
            elif value < -entry_z:
                position = LONG_SPREAD
        else:
            if magnitude < exit_z:
                position = FLAT
            elif stop_z is not None and magnitude > stop_z:
                position = FLAT
                blocked_after_stop = True
        out[i] = position

    return pd.Series(out, index=zscore.index, name="target_position")


def zscore_for_trading_period(
    close_a: pd.Series,
    close_b: pd.Series,
    hedge_ratio: float,
    window: int,
    trading_index: pd.Index,
) -> pd.Series:
    """Z-score restricted to ``trading_index`` but warmed up on the days before it.

    ``close_a`` / ``close_b`` should span formation + trading so the rolling statistics are
    already defined on the first trading day. Only the *window* of prior closes influences the
    signal; no formation-period statistics beyond that leak in.
    """
    z = rolling_zscore(compute_spread(close_a, close_b, hedge_ratio), window)
    return z.reindex(trading_index)
