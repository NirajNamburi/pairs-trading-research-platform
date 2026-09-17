"""End-to-end run on synthetic data: every stage, no network, both screen variants."""

import json

import pandas as pd
import pytest

from conftest import make_synthetic_market
from pairs_trading.config import PipelineConfig
from pairs_trading.pipeline import run_pipeline


@pytest.fixture
def synthetic_pipeline(monkeypatch):
    """Patch the two network-facing seams so ``run_pipeline`` sees the synthetic market."""
    close, open_ = make_synthetic_market()

    def fake_load_prices(tickers, start, end, data_dir, refresh=False):
        return close[list(tickers)].copy(), open_[list(tickers)].copy()

    monkeypatch.setattr("pairs_trading.pipeline.load_prices", fake_load_prices)
    monkeypatch.setattr(
        "pairs_trading.pipeline.get_universe",
        lambda sectors: {t: "synthetic" for t in close.columns},
    )
    return close, open_


def _cfg(tmp_path, **overrides) -> PipelineConfig:
    base = dict(
        sectors=("synthetic",),
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        slippage_bps=5.0,
        transaction_cost_bps=5.0,
        n_plot_pairs=1,
    )
    base.update(overrides)
    return PipelineConfig(**base)


def test_pretest_drops_the_stationary_ticker_and_the_screen_finds_the_real_pairs(
    tmp_path, synthetic_pipeline
):
    result = run_pipeline(_cfg(tmp_path))

    pretest = result.pretest
    assert pretest is not None and len(pretest) == 6
    assert pretest.loc[pretest["ticker"] == "S", "is_i1"].item() is False
    assert pretest.loc[pretest["ticker"] != "S", "is_i1"].all()
    assert "stationary on its own" in pretest.loc[pretest["ticker"] == "S", "reason"].item()

    screened = result.screened
    assert "S" not in set(screened["ticker_a"]) | set(screened["ticker_b"])
    assert len(screened) == 10  # C(5, 2) among the I(1) survivors
    assert list(screened.columns[:5]) == ["ticker_a", "ticker_b", "sector", "pvalue", "test_stat"]

    selected_pairs = {
        frozenset(p)
        for p in zip(result.selected["ticker_a"], result.selected["ticker_b"], strict=True)
    }
    assert selected_pairs == {frozenset("AB"), frozenset("CD")}  # exactly the two true pairs
    assert (screened["pvalue"] < 0.05).sum() == 2
    assert (result.selected["hedge_ratio"] > 0).all()
    # the traded spread is regressand - beta * regressor, with the regressand reported as ticker_a
    for r in result.results:
        assert r.name in {
            f"{a}/{b}"
            for a, b in zip(result.selected.ticker_a, result.selected.ticker_b, strict=True)
        }

    summary = json.loads((tmp_path / "reports" / "summary.json").read_text(encoding="utf-8"))
    assert summary["screening"]["unit_root_pretest"]["dropped"] == ["S"]
    assert summary["screening"]["both_orderings"] is True
    assert (tmp_path / "reports" / "unit_root_pretest.csv").exists()
    written = pd.read_csv(tmp_path / "reports" / "unit_root_pretest.csv")
    assert set(written["ticker"]) == set("ABCDES")
    assert "Unit-root pretest: 6 tickers tested; 1 dropped" in (
        tmp_path / "reports" / "summary.md"
    ).read_text(encoding="utf-8")


def test_naive_screen_keeps_the_stationary_ticker_and_alphabetical_orientation(
    tmp_path, synthetic_pipeline
):
    naive = run_pipeline(
        _cfg(
            tmp_path,
            unit_root_pretest=False,
            test_both_orderings=False,
            reports_dir=tmp_path / "naive",
        )
    )
    assert naive.pretest is None
    assert len(naive.screened) == 15  # C(6, 2): nothing dropped
    assert "S" in set(naive.screened["ticker_a"]) | set(naive.screened["ticker_b"])
    assert (naive.screened["ticker_a"] < naive.screened["ticker_b"]).all()
    assert (
        naive.screened["orientation"]
        == naive.screened["ticker_a"] + " on " + naive.screened["ticker_b"]
    ).all()
    summary = json.loads((tmp_path / "naive" / "summary.json").read_text(encoding="utf-8"))
    assert summary["screening"]["unit_root_pretest"]["enabled"] is False
    assert "Unit-root pretest: off" in (tmp_path / "naive" / "summary.md").read_text(
        encoding="utf-8"
    )
    assert not (tmp_path / "naive" / "unit_root_pretest.csv").exists()


def test_stationary_ticker_passes_as_regressand_without_the_pretest(tmp_path, synthetic_pipeline):
    """The artefact the pretest exists to remove: a stationary *regressand* 'cointegrates' with
    anything, because the residual is essentially the regressand itself. ``S`` sorts after
    ``A``-``E``, so under the alphabetical orientation it is always the regressor and the
    residual stays a random walk; the artefact only appears through the both-orientation swap
    that makes ``S`` the regressand. (In the committed naive run on real data the stationary
    tickers sort early enough to be the regressand under the alphabetical orientation.)"""

    def s_rows(result):
        screened = result.screened
        return screened[(screened["ticker_a"] == "S") | (screened["ticker_b"] == "S")]

    both = run_pipeline(
        _cfg(
            tmp_path,
            unit_root_pretest=False,
            test_both_orderings=True,
            reports_dir=tmp_path / "naive",
        ),
        write=False,
    )
    with_s = s_rows(both)
    assert len(with_s) == 5
    passing = with_s[with_s["pvalue"] < 0.05]
    assert len(passing) > 0
    assert (passing["ticker_a"] == "S").all()  # every pass has S as the regressand

    single = run_pipeline(
        _cfg(
            tmp_path,
            unit_root_pretest=False,
            test_both_orderings=False,
            reports_dir=tmp_path / "naive",
        ),
        write=False,
    )
    with_s = s_rows(single)
    assert (with_s["ticker_b"] == "S").all()  # alphabetical: S is always the regressor
    assert not (with_s["pvalue"] < 0.05).any()
