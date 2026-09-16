import logging

import numpy as np
import pandas as pd
import pytest

from conftest import bdays
from pairs_trading.data import clean_prices, load_prices, split_periods


def _prices(n: int = 100, tickers=("A", "B", "C")) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = bdays(n)
    base = np.linspace(50.0, 60.0, n)
    close = pd.DataFrame({t: base + i for i, t in enumerate(tickers)}, index=idx)
    open_ = close - 0.1
    return close, open_


def test_clean_drops_ticker_with_too_many_gaps_and_fills_small_gaps():
    close, open_ = _prices()
    close.iloc[10:12, close.columns.get_loc("B")] = np.nan  # 2-day gap -> forward-filled
    close.iloc[[5, 20, 40, 60, 80], close.columns.get_loc("C")] = np.nan  # 5% missing -> dropped

    clean_close, clean_open, dropped = clean_prices(
        close, open_, max_missing_frac=0.02, ffill_limit=3
    )

    assert dropped == ["C"]
    assert list(clean_close.columns) == ["A", "B"]
    assert list(clean_open.columns) == ["A", "B"]
    assert len(clean_close) == 100
    assert clean_close.index.equals(clean_open.index)
    fill_value = close.iloc[9]["B"]
    assert clean_close.iloc[10]["B"] == fill_value
    assert clean_close.iloc[11]["B"] == fill_value
    assert not clean_close.isna().any().any()
    assert not clean_open.isna().any().any()


def test_clean_drops_all_nan_days():
    close, open_ = _prices()
    close.iloc[3] = np.nan
    open_.iloc[3] = np.nan
    clean_close, clean_open, dropped = clean_prices(close, open_)
    assert dropped == []
    assert len(clean_close) == 99
    assert close.index[3] not in clean_close.index
    assert clean_close.index.equals(clean_open.index)


def test_clean_drops_days_with_unfillable_gaps():
    close, open_ = _prices()
    close.iloc[30:35, close.columns.get_loc("A")] = (
        np.nan
    )  # 5-day gap, limit 3 -> 2 days unfillable
    clean_close, clean_open, dropped = clean_prices(
        close, open_, max_missing_frac=0.10, ffill_limit=3
    )
    assert dropped == []
    assert len(clean_close) == 98
    assert len(clean_open) == 98


def test_clean_missing_fraction_uses_raw_data_not_filled_data():
    close, open_ = _prices()
    close.iloc[10:13, close.columns.get_loc("A")] = np.nan  # 3% raw, fillable in full
    _, _, dropped = clean_prices(close, open_, max_missing_frac=0.02, ffill_limit=3)
    assert dropped == ["A"]


def test_clean_rejects_non_positive_prices():
    close, open_ = _prices()
    close.iloc[0, 0] = 0.0
    with pytest.raises(ValueError):
        clean_prices(close, open_)


def _never_download(*args, **kwargs):
    pytest.fail("a cache hit must not call download_prices")


def test_load_prices_round_trips_a_complete_download(tmp_path, monkeypatch):
    close, open_ = _prices(20)
    monkeypatch.setattr("pairs_trading.data.download_prices", lambda t, s, e: (close, open_))
    args = (["A", "B", "C"], "2020-01-01", "2020-06-30", tmp_path)
    load_prices(*args)

    monkeypatch.setattr("pairs_trading.data.download_prices", _never_download)
    cached_close, cached_open = load_prices(*args)
    pd.testing.assert_frame_equal(cached_close, close, check_freq=False)
    pd.testing.assert_frame_equal(cached_open, open_, check_freq=False)


def test_load_prices_never_caches_failed_tickers_and_reports_them_on_reload(
    tmp_path, monkeypatch, caplog
):
    close, open_ = _prices(20)
    close["C"] = np.nan  # what yfinance hands back for a rate-limited / timed-out ticker
    open_["C"] = np.nan
    calls: list[list[str]] = []

    def fake_download(tickers, start, end):
        calls.append(list(tickers))
        return close.copy(), open_.copy()

    monkeypatch.setattr("pairs_trading.data.download_prices", fake_download)
    args = (["A", "B", "C"], "2020-01-01", "2020-06-30", tmp_path)

    first_close, first_open = load_prices(*args)
    assert list(first_close.columns) == ["A", "B", "C"]
    assert first_close["C"].isna().all()
    cache_dir = next(tmp_path.glob("prices_*"))
    assert list(pd.read_parquet(cache_dir / "close.parquet").columns) == ["A", "B"]
    assert list(pd.read_parquet(cache_dir / "open.parquet").columns) == ["A", "B"]

    # Cache hit: no network, the failed ticker is still an all-NaN column, and it is reported.
    monkeypatch.setattr("pairs_trading.data.download_prices", _never_download)
    with caplog.at_level(logging.WARNING, logger="pairs_trading.data"):
        cached_close, cached_open = load_prices(*args)
    assert list(cached_close.columns) == ["A", "B", "C"]
    assert cached_close["C"].isna().all() and cached_open["C"].isna().all()
    pd.testing.assert_frame_equal(cached_close[["A", "B"]], close[["A", "B"]], check_freq=False)
    assert "'C'" in caplog.text and "--refresh-data" in caplog.text

    # refresh=True is the way to retry it.
    monkeypatch.setattr("pairs_trading.data.download_prices", fake_download)
    load_prices(*args, refresh=True)
    assert len(calls) == 2


def test_split_periods_boundary_is_inclusive():
    close, _ = _prices(10)
    cutoff = close.index[3]
    formation, trading = split_periods(close, cutoff.strftime("%Y-%m-%d"))
    assert len(formation) == 4
    assert len(trading) == 6
    assert formation.index[-1] == cutoff
    assert trading.index[0] > cutoff


def test_split_periods_rejects_empty_side():
    close, _ = _prices(10)
    with pytest.raises(ValueError):
        split_periods(close, "2030-01-01")
