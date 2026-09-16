import math

import numpy as np
import pandas as pd
import pytest

from conftest import bdays, make_cointegrated_pair, make_random_walks
from pairs_trading.cointegration import (
    PAIR_COLUMNS,
    candidate_pairs,
    engle_granger_test,
    estimate_half_life,
    fit_hedge_ratio,
    screen_pairs,
    select_cointegrated,
)


def test_cointegrated_pair_is_detected_with_correct_hedge_ratio():
    a, b = make_cointegrated_pair(n=750, beta=2.0, phi=0.9, seed=0)
    stats = engle_granger_test(a, b)
    assert stats["pvalue"] < 0.05
    assert stats["hedge_ratio"] == pytest.approx(2.0, rel=0.10)  # OLS on levels is noisy
    assert 3.0 < stats["half_life"] < 15.0  # true half-life ln2 / 0.1 ~= 6.9 days
    assert stats["n_obs"] == 750


def test_independent_random_walks_are_not_cointegrated():
    a, b = make_random_walks(n=750, seed=1)
    stats = engle_granger_test(a, b)
    assert stats["pvalue"] > 0.05


def test_fit_hedge_ratio_recovers_exact_linear_relation():
    b = np.linspace(10.0, 20.0, 50)
    a = 3.0 + 1.5 * b
    beta, alpha = fit_hedge_ratio(a, b)
    assert beta == pytest.approx(1.5)
    assert alpha == pytest.approx(3.0)


def test_half_life_recovers_ar1_decay():
    rng = np.random.default_rng(42)
    phi = 0.9
    s = np.zeros(5000)
    for i in range(1, len(s)):
        s[i] = phi * s[i - 1] + rng.normal()
    expected = -math.log(2.0) / math.log(phi)  # ~6.58 days
    assert estimate_half_life(s) == pytest.approx(expected, rel=0.2)


def test_half_life_is_infinite_without_mean_reversion():
    assert estimate_half_life(np.arange(100, dtype=float)) == math.inf
    assert estimate_half_life(np.array([1.0, 2.0])) == math.inf
    assert estimate_half_life(np.ones(100)) == math.inf  # constant spread: nothing to fit


def test_constant_series_are_rejected_with_value_error():
    a, b = make_random_walks(100)
    constant = np.full(100, 50.0)
    with pytest.raises(ValueError):
        fit_hedge_ratio(a, constant)
    with pytest.raises(ValueError):
        engle_granger_test(a, constant)
    with pytest.raises(ValueError):
        engle_granger_test(constant, b)


def test_engle_granger_test_rejects_nan_or_mismatched_lengths():
    a, b = make_random_walks(100)
    with pytest.raises(ValueError):
        engle_granger_test(a[:-1], b)
    a[3] = np.nan
    with pytest.raises(ValueError):
        engle_granger_test(a, b)


def test_candidate_pairs_are_within_sector_and_alphabetical():
    universe = {
        "XOM": "energy",
        "CVX": "energy",
        "COP": "energy",
        "JPM": "financials",
        "BAC": "financials",
        "NEE": "utilities",
    }
    pairs = candidate_pairs(universe, tickers=["XOM", "CVX", "COP", "JPM", "BAC", "NEE", "ZZZ"])
    assert pairs == [
        ("COP", "CVX", "energy"),
        ("COP", "XOM", "energy"),
        ("CVX", "XOM", "energy"),
        ("BAC", "JPM", "financials"),
    ]
    assert all(a < b for a, b, _ in pairs)


def _screen_frame() -> tuple[pd.DataFrame, dict[str, str]]:
    a, b = make_cointegrated_pair(n=400, seed=3)
    c, d = make_random_walks(n=400, seed=4)
    e = 50.0 + np.cumsum(np.random.default_rng(5).normal(size=400))
    close = pd.DataFrame({"A": a, "B": b, "C": c, "D": d, "E": e}, index=bdays(400))
    universe = {"A": "energy", "B": "energy", "C": "energy", "D": "energy", "E": "utilities"}
    return close, universe


def test_screen_pairs_returns_every_tested_pair_sorted_by_pvalue():
    close, universe = _screen_frame()
    screened = screen_pairs(close, universe, n_jobs=1)
    assert list(screened.columns) == PAIR_COLUMNS
    assert len(screened) == 6  # C(4, 2) energy pairs; the lone utility has no partner
    assert screened["pvalue"].is_monotonic_increasing
    assert screened.iloc[0][["ticker_a", "ticker_b"]].tolist() == ["A", "B"]


def test_screen_pairs_parallel_matches_serial():
    close, universe = _screen_frame()
    serial = screen_pairs(close, universe, n_jobs=1)
    parallel = screen_pairs(close, universe, n_jobs=2)
    pd.testing.assert_frame_equal(serial, parallel)


def test_screen_pairs_requires_overlap():
    close, _ = _screen_frame()
    with pytest.raises(ValueError):
        screen_pairs(close, {"ZZZ": "energy"})


def test_select_cointegrated_filters_on_pvalue_and_hedge_sign():
    screened = pd.DataFrame(
        {
            "ticker_a": ["A", "C", "E"],
            "ticker_b": ["B", "D", "F"],
            "sector": ["s"] * 3,
            "pvalue": [0.01, 0.02, 0.20],
            "hedge_ratio": [1.5, -0.7, 1.0],
            "intercept": [0.0] * 3,
            "half_life": [5.0] * 3,
            "n_obs": [100] * 3,
        }
    )
    selected = select_cointegrated(screened, pvalue_threshold=0.05)
    assert selected["ticker_a"].tolist() == ["A"]
    selected = select_cointegrated(
        screened, pvalue_threshold=0.05, require_positive_hedge_ratio=False
    )
    assert selected["ticker_a"].tolist() == ["A", "C"]
