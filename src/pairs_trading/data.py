"""Stage 1 - data collection and cleaning.

Downloads adjusted daily OHLCV data with yfinance, caches it as Parquet, and produces two aligned
price matrices (rows = trading days, columns = tickers): closes for signal generation and opens
for simulated execution.

Only :func:`download_prices` touches the network; everything else is a pure function of
DataFrames so it can be unit-tested on synthetic data.
"""

import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

CLOSE_FILE = "close.parquet"
OPEN_FILE = "open.parquet"


def _cache_dir(data_dir: Path, tickers: Sequence[str], start: str, end: str) -> Path:
    key = hashlib.sha1(",".join(sorted(tickers)).encode()).hexdigest()[:8]
    return Path(data_dir) / f"prices_{start}_{end}_{key}"


def download_prices(
    tickers: Sequence[str], start: str, end: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download adjusted close and open prices for ``tickers`` (``end`` inclusive).

    Returns ``(close, open)`` DataFrames indexed by date with one column per requested ticker.
    Tickers that yfinance could not return are present as all-NaN columns so the cleaning step
    can report them.
    """
    import yfinance as yf  # imported lazily so tests never need network-capable deps loaded

    tickers = list(dict.fromkeys(tickers))  # de-duplicate, keep order
    # yfinance treats ``end`` as exclusive; shift by one day to make our config inclusive.
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    log.info("downloading %d tickers from yfinance (%s .. %s)", len(tickers), start, end)
    raw = yf.download(
        tickers,
        start=start,
        end=end_exclusive,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data; check tickers, dates or network access")

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
        open_ = raw["Open"].copy()
    else:  # single ticker -> flat columns
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
        open_ = raw[["Open"]].rename(columns={"Open": tickers[0]})

    close = close.reindex(columns=tickers)
    open_ = open_.reindex(columns=tickers)
    close.index = pd.to_datetime(close.index).tz_localize(None)
    open_.index = pd.to_datetime(open_.index).tz_localize(None)
    close.index.name = open_.index.name = "date"
    missing = [t for t in tickers if close[t].isna().all()]
    if len(missing) == len(tickers):
        raise RuntimeError("yfinance returned no usable data for any ticker; check network access")
    if missing:
        log.warning("no data returned for %d tickers: %s", len(missing), missing)
    return close.astype(float), open_.astype(float)


def load_prices(
    tickers: Sequence[str],
    start: str,
    end: str,
    data_dir: Path,
    refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return raw ``(close, open)`` matrices, downloading and caching them if needed.

    Only tickers that actually came back are written to the cache. yfinance swallows per-ticker
    failures (rate limits, timeouts) and returns an all-NaN column instead, and persisting that
    would silently truncate the universe on every later run. On a cache hit the frames are
    reindexed to ``tickers``, so a ticker absent from the cache still comes back as an all-NaN
    column for the cleaning step to report; it is logged here and re-attempted only when
    ``refresh=True`` (``--refresh-data``) re-downloads everything.
    """
    tickers = list(dict.fromkeys(tickers))
    cache = _cache_dir(data_dir, tickers, start, end)
    close_path, open_path = cache / CLOSE_FILE, cache / OPEN_FILE
    if not refresh and close_path.exists() and open_path.exists():
        log.info("loading cached prices from %s", cache)
        close = pd.read_parquet(close_path).reindex(columns=tickers)
        open_ = pd.read_parquet(open_path).reindex(columns=tickers)
        missing = [t for t in tickers if close[t].isna().all()]
        if missing:
            log.warning(
                "%d tickers have no data in the cache (their download failed when it was "
                "built): %s; pass --refresh-data (refresh=True) to retry them",
                len(missing),
                missing,
            )
        return close, open_

    close, open_ = download_prices(tickers, start, end)
    received = [t for t in tickers if not close[t].isna().all()]
    cache.mkdir(parents=True, exist_ok=True)
    close[received].to_parquet(close_path)
    open_[received].to_parquet(open_path)
    log.info("cached prices for %d of %d tickers to %s", len(received), len(tickers), cache)
    return close, open_


def clean_prices(
    close: pd.DataFrame,
    open_: pd.DataFrame,
    max_missing_frac: float = 0.02,
    ffill_limit: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Clean and align the close/open matrices.

    1. Drop days on which nothing traded (all-NaN rows, e.g. exchange holidays).
    2. Drop tickers whose *raw* close series is missing more than ``max_missing_frac`` of days
       (late IPOs, delistings, symbols yfinance could not return).
    3. Forward-fill remaining gaps of at most ``ffill_limit`` days (isolated bad prints).
    4. Drop any remaining day with a NaN in either matrix so both are dense and aligned.

    Returns ``(close, open, dropped_tickers)``.
    """
    if not close.columns.equals(open_.columns):
        open_ = open_.reindex(columns=close.columns)
    close = close.sort_index()
    open_ = open_.reindex(close.index)

    close = close.dropna(how="all")
    open_ = open_.loc[close.index]

    missing_frac = close.isna().mean()
    keep = missing_frac[missing_frac <= max_missing_frac].index.tolist()
    dropped = sorted(set(close.columns) - set(keep))
    if dropped:
        log.warning(
            "dropping %d tickers with > %.0f%% missing data: %s",
            len(dropped),
            100 * max_missing_frac,
            dropped,
        )

    close = close[keep].ffill(limit=ffill_limit)
    open_ = open_[keep].ffill(limit=ffill_limit)

    dense = close.notna().all(axis=1) & open_.notna().all(axis=1)
    n_dropped_rows = int((~dense).sum())
    if n_dropped_rows:
        log.warning("dropping %d days with unfillable gaps", n_dropped_rows)
    close, open_ = close.loc[dense], open_.loc[dense]

    if (close <= 0).any().any() or (open_ <= 0).any().any():
        raise ValueError("non-positive prices found after cleaning")
    return close, open_, dropped


def split_periods(df: pd.DataFrame, formation_end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a date-indexed frame into ``(formation, trading)`` at ``formation_end`` (inclusive)."""
    cutoff = pd.Timestamp(formation_end)
    formation = df.loc[df.index <= cutoff]
    trading = df.loc[df.index > cutoff]
    if formation.empty or trading.empty:
        raise ValueError(
            f"formation_end={formation_end} leaves an empty formation or trading period"
        )
    return formation, trading
