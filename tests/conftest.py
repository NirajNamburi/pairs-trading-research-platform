"""Synthetic fixtures. No test touches the network."""

import numpy as np
import pandas as pd
import pytest

from pairs_trading.config import PipelineConfig


@pytest.fixture
def cfg() -> PipelineConfig:
    return PipelineConfig()


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
