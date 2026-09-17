"""Smoke test for the Streamlit viewer on a synthetic run. Skipped when streamlit is absent."""

import sys
from pathlib import Path

import pandas as pd
import pytest

from conftest import make_synthetic_market
from pairs_trading.config import PipelineConfig
from pairs_trading.pipeline import run_pipeline

streamlit = pytest.importorskip("streamlit")
pytest.importorskip("plotly")
from streamlit.testing.v1 import AppTest  # noqa: E402

from pairs_trading.ui.__main__ import build_command  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "src" / "pairs_trading" / "ui" / "app.py"


@pytest.fixture
def synthetic_run(tmp_path, monkeypatch):
    """A committed run in ``tmp_path/reports`` plus the seams the viewer needs patched."""
    streamlit.cache_data.clear()  # st.cache_data is process-global: never share runs across tests
    close, open_ = make_synthetic_market()

    def fake_load_prices(tickers, start, end, data_dir, refresh=False):
        return close[list(tickers)].copy(), open_[list(tickers)].copy()

    universe = {t: "synthetic" for t in close.columns}
    monkeypatch.setattr("pairs_trading.pipeline.load_prices", fake_load_prices)
    monkeypatch.setattr("pairs_trading.pipeline.get_universe", lambda sectors: universe)
    monkeypatch.setattr("pairs_trading.data.load_prices", fake_load_prices)
    monkeypatch.setattr("pairs_trading.universe.get_universe", lambda sectors: universe)
    cfg = PipelineConfig(
        sectors=("synthetic",),
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        slippage_bps=5.0,
        transaction_cost_bps=5.0,
        n_plot_pairs=0,
    )
    run_pipeline(cfg)
    monkeypatch.chdir(tmp_path)  # the viewer's default "reports" path now resolves to this run
    return tmp_path / "reports"


def test_viewer_renders_the_run_and_recomputes_on_new_parameters(synthetic_run):
    at = AppTest.from_file(str(APP), default_timeout=180)
    at.run()
    assert not at.exception, at.exception
    assert not at.error
    assert at.title[0].value.startswith("Pairs trading research viewer")
    labels = [m.label for m in at.metric]
    assert "Sharpe (net)" in labels and "Pairs profitable" in labels
    assert "committed run" in " ".join(c.value for c in at.sidebar.caption)
    assert "10 pairs tested, 2 backtested" in at.main.caption[0].value
    # the p-value box must render the value that is actually used (0.001 is not "0.00")
    pvalue_input = next(n for n in at.sidebar.number_input if "p-value" in n.label)
    assert pvalue_input.proto.format == "%.3f"

    # Adjust the entry threshold and re-run: the viewer recomputes and says so.
    entry_input = next(n for n in at.sidebar.number_input if n.label == "entry |z|")
    entry_input.set_value(1.5)
    at.sidebar.button[0].click().run()
    assert not at.exception, at.exception
    assert "recomputed live" in " ".join(c.value for c in at.sidebar.caption)


def test_viewer_rereads_a_rewritten_run_and_warns_when_no_pair_passes(synthetic_run):
    at = AppTest.from_file(str(APP), default_timeout=180).run()
    assert not at.exception, at.exception
    assert "2 backtested" in at.main.caption[0].value

    # The pipeline is re-run into the same directory (here: every p-value pushed above any
    # threshold). A fresh page load must pick the new files up, not the cached first run, and
    # an empty selection must produce a warning rather than a traceback.
    screened = pd.read_csv(synthetic_run / "screened_pairs.csv")
    screened["pvalue"] = screened["pvalue"] + 1.0
    screened.to_csv(synthetic_run / "screened_pairs.csv", index=False)
    at = AppTest.from_file(str(APP), default_timeout=180).run()
    assert not at.exception, at.exception
    assert not at.error
    assert at.warning and "No pair passes p < 0.05" in at.warning[0].value
    assert not at.metric


@pytest.mark.parametrize("prefix", [[], ["--"]])
def test_viewer_honours_reports_flag_from_the_command_line(
    synthetic_run, tmp_path, monkeypatch, prefix
):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)  # the default "reports" would not resolve from here
    # streamlit sets sys.argv to [script, *forwarded args]; a user-typed "--" is forwarded too
    monkeypatch.setattr(sys, "argv", [str(APP), *prefix, "--reports", str(synthetic_run)])
    at = AppTest.from_file(str(APP), default_timeout=180).run()
    assert not at.exception, at.exception
    assert not at.error
    assert at.sidebar.text_input[0].value == str(synthetic_run)
    assert "2 backtested" in at.main.caption[0].value


def test_launcher_forwards_arguments_after_a_single_separator():
    cmd = build_command(["--reports", "reports/naive_screen"])
    assert cmd[-3:] == ["--", "--reports", "reports/naive_screen"]
    assert cmd.count("--") == 1
    assert cmd[1:4] == ["-m", "streamlit", "run"] and cmd[4].endswith("app.py")
    # the documented-by-mistake form `python -m pairs_trading.ui -- --reports DIR` also works
    assert build_command(["--", "--reports", "reports/naive_screen"]) == cmd
    assert build_command([])[-1] == "--"


def test_viewer_reports_a_missing_run_without_crashing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no reports/ here
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception
    assert at.error and "No run found" in at.error[0].value
    at.sidebar.text_input[0].set_value(str(tmp_path / "nowhere")).run()
    assert not at.exception
    assert at.error and "No run found" in at.error[0].value
