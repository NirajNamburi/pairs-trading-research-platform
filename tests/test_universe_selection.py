"""Pure-function tests for the universe selection; nothing here touches the network."""

import json

import numpy as np
import pandas as pd
import pytest

from pairs_trading import universe
from pairs_trading.universe_selection import (
    build_spec,
    company_key,
    dedupe_share_classes,
    normalize_symbol,
    rank_top_n,
    write_spec,
)


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["NEE", "DUK", "SO", "TLN", "CWEN", "CWEN-A", "XYZ"],
            "name": [
                "NextEra Energy",
                "Duke Energy",
                "Southern Company",
                "Talen Energy",
                "Clearway Energy, Inc. (Class C)",
                "Clearway Energy, Inc. (Class A)",
                "Xyz Power",
            ],
            "index": ["S&P 500"] * 3 + ["S&P 400"] * 3 + ["S&P 600"],
            "sub_industry": ["Electric Utilities"] * 7,
            "market_cap_usd": [160e9, 90e9, 100e9, 20e9, 6e9, 6e9, np.nan],
            "first_price_date": ["2019-01-02"] * 3 + ["2023-07-10"] + ["2019-01-02"] * 3,
            "eligible": [True, True, True, False, True, True, False],
            "reason": [
                "",
                "",
                "",
                "31.0% of trading days missing",
                "",
                "",
                "no market cap available",
            ],
        }
    )


def test_normalize_symbol_converts_share_class_dots():
    assert normalize_symbol(" cwen.a ") == "CWEN-A"
    assert normalize_symbol("NEE") == "NEE"


def test_company_key_strips_share_class_suffix():
    assert company_key("Clearway Energy, Inc. (Class C)") == "clearway energy, inc."
    assert company_key("Berkshire Hathaway (Class B)") == "berkshire hathaway"
    assert company_key("Duke Energy") == "duke energy"


def test_dedupe_share_classes_keeps_the_plain_listing():
    deduped = dedupe_share_classes(_candidates())
    assert "CWEN" in deduped["ticker"].tolist()
    assert "CWEN-A" not in deduped["ticker"].tolist()
    assert len(deduped) == 6
    assert deduped["ticker"].is_monotonic_increasing


def test_rank_top_n_orders_by_market_cap_and_skips_ineligible():
    ranked = rank_top_n(_candidates(), n=3)
    assert ranked["ticker"].tolist() == ["NEE", "SO", "DUK"]
    assert ranked["rank"].tolist() == [1, 2, 3]

    everything = rank_top_n(_candidates(), n=50)
    assert "TLN" not in everything["ticker"].tolist()  # ineligible: short history
    assert "XYZ" not in everything["ticker"].tolist()  # no market cap
    with pytest.raises(ValueError):
        rank_top_n(_candidates(), n=0)


def test_build_spec_records_members_and_exclusions_with_reasons(tmp_path):
    spec = build_spec(
        name="test_universe",
        sector="Utilities",
        candidates=dedupe_share_classes(_candidates()),
        n=3,
        start="2019-01-01",
        end="2025-12-31",
        max_missing_frac=0.02,
        selected_at="2026-09-16",
    )
    assert spec["tickers"] == ["NEE", "SO", "DUK"]
    assert spec["n_candidates"] == 6 and spec["n_selected"] == 3
    assert [m["rank"] for m in spec["members"]] == [1, 2, 3]
    excluded = {e["ticker"]: e["reason"] for e in spec["excluded"]}
    assert excluded["TLN"].startswith("31.0%")
    assert excluded["XYZ"] == "no market cap available"
    assert excluded["CWEN"] == "eligible, but ranked below the top 3 by market cap"

    path = write_spec(spec, tmp_path / "u.json")
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded == spec  # JSON-safe: no NaN, no numpy scalars


def test_json_universes_are_loaded_as_sectors(tmp_path, monkeypatch):
    spec = {"name": "toy_universe", "tickers": ["AAA", "BBB", "AAA"]}
    (tmp_path / "toy_universe.json").write_text(json.dumps(spec), encoding="utf-8")
    monkeypatch.setattr(universe, "UNIVERSE_DIR", tmp_path)
    universe.json_universes.cache_clear()
    try:
        assert "toy_universe" in universe.available_sectors()
        assert universe.get_universe(["toy_universe"]) == {
            "AAA": "toy_universe",
            "BBB": "toy_universe",
        }
        assert universe.universe_spec("toy_universe")["name"] == "toy_universe"
        assert universe.universe_spec("energy") is None
    finally:
        universe.json_universes.cache_clear()


def test_shipped_top50_universe_matches_its_own_criteria():
    spec = universe.universe_spec("utilities_top50")
    assert spec is not None, "utilities_top50.json is committed package data and the default"
    tickers = universe.get_universe(["utilities_top50"])
    assert len(tickers) == spec["n_selected"] == len(spec["tickers"]) <= spec["criteria"]["n"]
    caps = [m["market_cap_usd"] for m in spec["members"]]
    assert caps == sorted(caps, reverse=True)
    assert all(t == t.upper() and "." not in t for t in tickers)
    assert all(m["first_price_date"] is not None for m in spec["members"])


def test_overlapping_sectors_are_rejected_in_either_order(tmp_path, monkeypatch):
    # ``toy`` shares XOM with the hardcoded ``energy`` list. Pairs are screened within a label,
    # so combining the two would silently relabel XOM and drop its pairs against one side.
    spec = {"name": "toy", "tickers": ["XOM", "AAA"]}
    (tmp_path / "toy.json").write_text(json.dumps(spec), encoding="utf-8")
    monkeypatch.setattr(universe, "UNIVERSE_DIR", tmp_path)
    universe.json_universes.cache_clear()
    try:
        for order in (["toy", "energy"], ["energy", "toy"]):
            with pytest.raises(ValueError, match="XOM is in both"):
                universe.get_universe(order)
        assert universe.get_universe(["toy", "toy"]) == {"XOM": "toy", "AAA": "toy"}
        assert set(universe.get_universe(["toy", "utilities"])) == {"XOM", "AAA"} | set(
            universe.SECTORS["utilities"]
        )
    finally:
        universe.json_universes.cache_clear()


def test_shipped_top50_cannot_be_combined_with_the_sp500_utilities_list():
    # Every hardcoded S&P 500 utilities ticker is inside utilities_top50, so the two overlap
    # completely; the CLI accepts both names, and this is what stops a silent 609-pair screen.
    assert set(universe.SECTORS["utilities"]) <= set(universe.get_universe(["utilities_top50"]))
    with pytest.raises(ValueError, match="overlapping universes cannot be combined"):
        universe.get_universe(["utilities_top50", "utilities"])
    with pytest.raises(ValueError, match="overlapping universes cannot be combined"):
        universe.get_universe(["utilities", "utilities_top50"])


def test_check_history_labels_an_all_nan_ticker_as_no_data(monkeypatch):
    from pairs_trading import universe_selection

    idx = pd.bdate_range("2019-01-02", periods=100, name="date")
    close = pd.DataFrame(
        {
            "NEE": np.linspace(50.0, 60.0, 100),
            "LATE": [np.nan] * 40 + list(np.linspace(20.0, 25.0, 60)),
            "XXX": np.nan,  # yfinance returned nothing: an all-NaN column, not a missing one
        },
        index=idx,
    )
    monkeypatch.setattr(universe_selection, "download_prices", lambda *a, **k: (close, close))

    history = universe_selection.check_history(
        ["NEE", "LATE", "XXX"], "2019-01-02", "2019-05-31", max_missing_frac=0.02
    ).set_index("ticker")

    assert history.loc["NEE", "eligible"] and history.loc["NEE", "reason"] == ""
    assert history.loc["NEE", "first_price_date"] == "2019-01-02"
    assert not history.loc["LATE", "eligible"]
    assert history.loc["LATE", "reason"].startswith("40.0% of trading days missing (first price ")
    assert not history.loc["XXX", "eligible"]
    assert pd.isna(history.loc["XXX", "first_price_date"])  # None -> NaN in a string column
    assert history.loc["XXX", "reason"].startswith("no price data returned by yfinance")
    assert "NaT" not in history.loc["XXX", "reason"]
