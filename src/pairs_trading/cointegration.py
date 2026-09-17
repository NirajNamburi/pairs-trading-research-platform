"""Stage 2 - cointegration screening (Engle-Granger, all four steps).

For a universe of price series the screen runs, in order:

1. **Unit-root pretest on each stock.** An Augmented Dickey-Fuller test on the price levels must
   *fail* to reject a unit root (p-value above ``unit_root_pvalue``). A stock whose price is
   already stationary on its own is dropped: as the regressand, paired with anything, its
   residual would inherit that stationarity and the pair test would "pass" without any
   relationship between the two stocks.
   The first differences are also tested and reported (a random walk's differences are
   stationary) but not gated on, because daily stock prices essentially always satisfy it.
2. **Cointegrating regression.** OLS ``A = alpha + beta * B``; ``beta`` is the hedge ratio. Both
   orientations (``A`` on ``B`` and ``B`` on ``A``) are fitted, because the Engle-Granger test is
   not symmetric, and the orientation with the more negative test statistic is kept. The chosen
   regressand is reported as ``ticker_a`` so the traded spread is always ``A - beta * B``.
3. **Residual.** ``A - alpha - beta * B``, the spread.
4. **Stationarity test on the residual** with the Engle-Granger / MacKinnon critical values that
   account for ``beta`` having been chosen to make the residual look stationary (plain ADF
   critical values would be too lenient). ``statsmodels.tsa.stattools.coint`` implements steps
   2 to 4 for one orientation and returns that p-value.

A pair is selected when the step-4 p-value is below the threshold and the hedge ratio is
positive. The mean-reversion half-life of the residual (AR(1) fit) is reported as a diagnostic;
it is not used to select or rank pairs (the screen sorts by p-value, the results table by
out-of-sample Sharpe), and a half-life filter is listed as future work.

Why cointegration and not correlation: two correlated assets can drift apart permanently.
Cointegration tests the property the strategy actually needs - that the spread mean-reverts.
"""

import itertools
import logging
import math
from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

log = logging.getLogger(__name__)

PAIR_COLUMNS = [
    "ticker_a",
    "ticker_b",
    "sector",
    "pvalue",
    "test_stat",
    "hedge_ratio",
    "intercept",
    "half_life",
    "n_obs",
    "orientation",
]

PRETEST_COLUMNS = [
    "ticker",
    "adf_stat_levels",
    "adf_p_levels",
    "adf_p_diff",
    "is_i1",
    "reason",
]


# --- step 1: unit-root pretest ---------------------------------------------------------------


def unit_root_test(series: np.ndarray | pd.Series) -> dict[str, float]:
    """ADF statistics for one price series: levels and first differences.

    ``adf_p_levels`` is the p-value of the null "the levels have a unit root". A *large* value
    means the series behaves like a random walk, which is what cointegration needs. A small value
    means the series is stationary on its own. ``adf_p_diff`` is the same test on the first
    differences; for an I(1) series it is small.
    """
    x = np.asarray(series, dtype=float)
    if len(x) < 20:
        raise ValueError("need at least 20 observations for a unit-root test")
    if np.isnan(x).any():
        raise ValueError("series must not contain NaN")
    if np.ptp(x) == 0:
        raise ValueError("series must not be constant")
    levels = adfuller(x, regression="c", autolag="AIC", result_object=True)
    diffs = adfuller(np.diff(x), regression="c", autolag="AIC", result_object=True)
    return {
        "adf_stat_levels": float(levels.statistic),
        "adf_p_levels": float(levels.pvalue),
        "adf_p_diff": float(diffs.pvalue),
    }


def pretest_universe(close: pd.DataFrame, pvalue_threshold: float = 0.05) -> pd.DataFrame:
    """Run :func:`unit_root_test` on every column; flag the ones eligible for pairing.

    A ticker is I(1)-eligible when its levels do **not** reject a unit root at
    ``pvalue_threshold``. Returns one row per ticker with the columns in
    :data:`PRETEST_COLUMNS`, sorted by ``adf_p_levels`` so the closest calls are at the top.
    """
    if not 0.0 < pvalue_threshold < 1.0:
        raise ValueError("pvalue_threshold must be in (0, 1)")
    rows = []
    for ticker in close.columns:
        stats = unit_root_test(close[ticker].to_numpy(float))
        is_i1 = stats["adf_p_levels"] > pvalue_threshold
        reason = (
            ""
            if is_i1
            else (
                f"levels reject a unit root (ADF p = {stats['adf_p_levels']:.3f} <= "
                f"{pvalue_threshold:g}): stationary on its own, not eligible for pairing"
            )
        )
        rows.append({"ticker": ticker, **stats, "is_i1": bool(is_i1), "reason": reason})
    frame = pd.DataFrame(rows, columns=PRETEST_COLUMNS)
    return frame.sort_values("adf_p_levels").reset_index(drop=True)


# --- steps 2-4: Engle-Granger on one pair ----------------------------------------------------


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


def _engle_granger_one_way(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Steps 2-4 for the orientation ``a`` on ``b``: ``(test_stat, pvalue)``."""
    test_stat, pvalue, _ = coint(a, b, trend="c", autolag="aic")
    return float(test_stat), float(pvalue)


def engle_granger_test(
    a: np.ndarray | pd.Series,
    b: np.ndarray | pd.Series,
    both_orderings: bool = True,
) -> dict[str, float | bool]:
    """Engle-Granger statistics for a pair, optionally trying both orientations.

    Returns ``pvalue``, ``test_stat``, ``hedge_ratio``, ``intercept``, ``half_life``, ``n_obs``
    and ``swapped``. When ``swapped`` is True the winning regression was ``b`` on ``a``: the
    caller should treat ``b`` as the regressand (``ticker_a``) so the spread is
    ``regressand - hedge_ratio * regressor``.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError("series must have equal length")
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("series must not contain NaN")
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        raise ValueError("series must not be constant")

    stat_ab, p_ab = _engle_granger_one_way(a, b)
    swapped = False
    test_stat, pvalue = stat_ab, p_ab
    if both_orderings:
        stat_ba, p_ba = _engle_granger_one_way(b, a)
        if stat_ba < stat_ab:  # more negative = stronger rejection of "no cointegration"
            swapped, test_stat, pvalue = True, stat_ba, p_ba

    regressand, regressor = (b, a) if swapped else (a, b)
    hedge_ratio, intercept = fit_hedge_ratio(regressand, regressor)
    residual = regressand - intercept - hedge_ratio * regressor
    return {
        "pvalue": pvalue,
        "test_stat": test_stat,
        "hedge_ratio": hedge_ratio,
        "intercept": intercept,
        "half_life": estimate_half_life(residual),
        "n_obs": int(len(a)),
        "swapped": swapped,
    }


# --- the screen over a universe ---------------------------------------------------------------


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


def _screen_one(task: tuple[str, str, str, np.ndarray, np.ndarray, bool]) -> dict[str, object]:
    ticker_a, ticker_b, sector, a, b, both = task
    stats = engle_granger_test(a, b, both_orderings=both)
    swapped = bool(stats.pop("swapped"))
    regressand, regressor = (ticker_b, ticker_a) if swapped else (ticker_a, ticker_b)
    return {
        "ticker_a": regressand,
        "ticker_b": regressor,
        "sector": sector,
        **stats,
        "orientation": f"{regressand} on {regressor}",
    }


def screen_pairs(
    close: pd.DataFrame,
    universe: Mapping[str, str],
    n_jobs: int = 1,
    both_orderings: bool = True,
) -> pd.DataFrame:
    """Steps 2-4 on every candidate pair using formation-period closes.

    ``close`` should already exclude tickers that failed :func:`pretest_universe`. Returns one
    row per *tested* pair (not just the ones that pass), sorted by p-value, with the columns in
    :data:`PAIR_COLUMNS`; ``ticker_a`` is the regressand of the winning orientation. Filtering is
    a separate step (:func:`select_cointegrated`) so the report can state how many were tested.

    The screen is O(n^2) in tickers; ``n_jobs > 1`` spreads the tests across processes.
    """
    pairs = candidate_pairs(universe, close.columns)
    if not pairs:
        raise ValueError("no candidate pairs: universe and price columns do not overlap")
    log.info(
        "screening %d within-sector pairs on %d formation days (%s)",
        len(pairs),
        len(close),
        "both orientations" if both_orderings else "alphabetical orientation only",
    )
    tasks = [
        (a, b, sector, close[a].to_numpy(float), close[b].to_numpy(float), both_orderings)
        for a, b, sector in pairs
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
