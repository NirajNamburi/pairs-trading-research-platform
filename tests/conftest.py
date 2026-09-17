"""Synthetic fixtures. No test touches the network."""

import numpy as np
import pandas as pd
import pytest

from pairs_trading.config import PipelineConfig


@pytest.fixture
def cfg() -> PipelineConfig:
    """Defaults, but with costs pinned: the hand-computed backtest examples assume 5 + 5 bps."""
    return PipelineConfig(slippage_bps=5.0, transaction_cost_bps=5.0)


def make_cointegrated_pair(
    n: int = 750,
    beta: float = 2.0,
    alpha: float = 10.0,
    phi: float = 0.9,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """``b`` is a random walk; ``a = alpha + beta * b + AR(1) spread`` with decay ``phi``."""
    rng = np.random.default_rng(seed)
    b = 200.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    eps = rng.normal(0.0, 1.0, n)
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = phi * spread[i - 1] + eps[i]
    a = alpha + beta * b + spread
    return a, b


def make_random_walks(n: int = 750, seed: int = 1) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a = 100.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    b = 100.0 + np.cumsum(rng.normal(0.0, 1.0, n))
    return a, b


def bdays(n: int, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n, name="date")


def make_synthetic_market(seed: int = 4) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A six-ticker market over 2019-2025: two cointegrated pairs, a lone random walk and one
    stationary ticker (``S``) that the unit-root pretest must drop.

    ``A = 20 + 2 * B + AR(1)``, ``C = 5 + 0.5 * D + AR(1)``; ``B``, ``D``, ``E`` are random walks;
    ``S`` is an AR(1) around 50. Opens are the previous close plus a small gap.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", "2025-12-31", name="date")
    n = len(idx)

    def walk(start: float, scale: float) -> np.ndarray:
        return start + np.cumsum(rng.normal(0.0, scale, n))

    def ar1(phi: float, scale: float) -> np.ndarray:
        out = np.zeros(n)
        eps = rng.normal(0.0, scale, n)
        for i in range(1, n):
            out[i] = phi * out[i - 1] + eps[i]
        return out

    b = walk(200.0, 1.0)
    d = walk(300.0, 1.5)
    close = pd.DataFrame(
        {
            "A": 20.0 + 2.0 * b + ar1(0.9, 1.5),
            "B": b,
            "C": 5.0 + 0.5 * d + ar1(0.9, 1.0),
            "D": d,
            "E": walk(150.0, 1.0),
            "S": 50.0 + ar1(0.95, 0.5),
        },
        index=idx,
    )
    gaps = 1.0 + rng.normal(0.0, 0.001, size=close.shape)
    open_ = close.shift(1).fillna(close.iloc[0]) * gaps
    assert (close > 0).all().all() and (open_ > 0).all().all()
    return close, open_
