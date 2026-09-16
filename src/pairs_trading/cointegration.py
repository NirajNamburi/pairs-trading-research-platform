"""Stage 2 - cointegration screening (Engle-Granger two-step).

For every within-sector pair ``(A, B)``:

1. Regress ``A = alpha + beta * B`` by OLS. ``beta`` is the hedge ratio.
2. The residual ``A - alpha - beta * B`` is the spread.
3. Test the spread for stationarity with an Augmented Dickey-Fuller test.

``statsmodels.tsa.stattools.coint`` implements exactly this and returns a p-value using the
MacKinnon critical values appropriate for residual-based tests (a plain ADF p-value on the
residuals would be too optimistic because the residuals were *chosen* to look stationary).

Why cointegration and not correlation: two correlated assets can drift apart permanently.
Cointegration tests the property the strategy actually needs - that the spread mean-reverts.

Known simplification: Engle-Granger is not symmetric in ``(A, B)``. Pairs are ordered
alphabetically and only that ordering is tested.
"""

import itertools
import logging
import math
from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

log = logging.getLogger(__name__)

PAIR_COLUMNS = [
    "ticker_a",
    "ticker_b",
    "sector",
    "pvalue",
    "hedge_ratio",
    "intercept",
    "half_life",
    "n_obs",
]


def fit_hedge_ratio(a: np.ndarray | pd.Series, b: np.ndarray | pd.Series) -> tuple[float, float]:
    """OLS ``a = intercept + hedge_ratio * b``. Returns ``(hedge_ratio, intercept)``.

    Raises ``ValueError`` if ``b`` is constant: the slope is then unidentified, and
    ``sm.add_constant`` (default ``has_constant="skip"``) would silently drop the intercept
    column, so the fit could not be read back by position.
    """
    y = np.asarray(a, dtype=float)
    x = np.asarray(b, dtype=float)
    if np.ptp(x) == 0:
        raise ValueError("cannot fit a hedge ratio on a constant series")
    params = sm.OLS(y, sm.add_constant(x)).fit().params
    return float(params[1]), float(params[0])


def estimate_half_life(spread: np.ndarray | pd.Series) -> float:
    """Mean-reversion half-life in trading days from an AR(1) fit of ``d s_t = lam * s_{t-1}``.

    Returns ``inf`` when the estimated ``lam`` is non-negative (no mean reversion) or when the
    spread is constant (there are no dynamics to fit).
    """
    s = np.asarray(spread, dtype=float)
    if len(s) < 3:
        return math.inf
    lagged = s[:-1]
    delta = np.diff(s)
    if np.ptp(lagged) == 0:
        return math.inf
    lam = sm.OLS(delta, sm.add_constant(lagged)).fit().params[1]
    if lam >= -1e-10:  # tolerance: a pure trend gives lam ~ -1e-19 from floating-point noise
        return math.inf
    return float(-math.log(2.0) / lam)


def engle_granger_test(a: np.ndarray | pd.Series, b: np.ndarray | pd.Series) -> dict[str, float]:
    """Engle-Granger statistics for one ordered pair."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError("series must have equal length")
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("series must not contain NaN")
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        raise ValueError("series must not be constant")
    _, pvalue, _ = coint(a, b, trend="c", autolag="aic")
    hedge_ratio, intercept = fit_hedge_ratio(a, b)
    residual = a - intercept - hedge_ratio * b
    return {
        "pvalue": float(pvalue),
        "hedge_ratio": hedge_ratio,
        "intercept": intercept,
        "half_life": estimate_half_life(residual),
        "n_obs": int(len(a)),
    }


def candidate_pairs(
    universe: Mapping[str, str], tickers: Iterable[str]
) -> list[tuple[str, str, str]]:
    """All within-sector, alphabetically ordered ``(ticker_a, ticker_b, sector)`` combinations."""
    available = set(tickers)
    by_sector: dict[str, list[str]] = {}
    for ticker, sector in universe.items():
        if ticker in available:
            by_sector.setdefault(sector, []).append(ticker)
    pairs: list[tuple[str, str, str]] = []
    for sector in sorted(by_sector):
        for a, b in itertools.combinations(sorted(by_sector[sector]), 2):
            pairs.append((a, b, sector))
    return pairs


def _screen_one(task: tuple[str, str, str, np.ndarray, np.ndarray]) -> dict[str, object]:
    ticker_a, ticker_b, sector, a, b = task
    stats = engle_granger_test(a, b)
    return {"ticker_a": ticker_a, "ticker_b": ticker_b, "sector": sector, **stats}


def screen_pairs(close: pd.DataFrame, universe: Mapping[str, str], n_jobs: int = 1) -> pd.DataFrame:
    """Run the Engle-Granger test on every candidate pair using formation-period closes.

    Returns one row per *tested* pair (not just the ones that pass), sorted by p-value, with the
    columns in :data:`PAIR_COLUMNS`. Filtering is a separate step (:func:`select_cointegrated`)
    so the report can state how many pairs were tested.

    The screen is O(n^2) in tickers; ``n_jobs > 1`` spreads the tests across processes.
    """
    pairs = candidate_pairs(universe, close.columns)
    if not pairs:
        raise ValueError("no candidate pairs: universe and price columns do not overlap")
    log.info("screening %d within-sector pairs on %d formation days", len(pairs), len(close))
    tasks = [
        (a, b, sector, close[a].to_numpy(float), close[b].to_numpy(float)) for a, b, sector in pairs
    ]
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            rows = list(pool.map(_screen_one, tasks, chunksize=16))
    else:
        rows = [_screen_one(task) for task in tasks]
    return pd.DataFrame(rows, columns=PAIR_COLUMNS).sort_values("pvalue").reset_index(drop=True)


def select_cointegrated(
    screened: pd.DataFrame,
    pvalue_threshold: float = 0.05,
    require_positive_hedge_ratio: bool = True,
) -> pd.DataFrame:
    """Keep pairs with ``pvalue < threshold`` (and, by default, a positive hedge ratio).

    A negative hedge ratio means the "pair" is really a bet that two same-sector stocks move in
    opposite directions, which has no economic rationale here.
    """
    mask = screened["pvalue"] < pvalue_threshold
    if require_positive_hedge_ratio:
        mask &= screened["hedge_ratio"] > 0
    return screened.loc[mask].sort_values("pvalue").reset_index(drop=True)
