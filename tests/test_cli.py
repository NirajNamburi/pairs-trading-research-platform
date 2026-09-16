"""The CLI is tested with ``run_pipeline`` stubbed out; nothing here touches data or network."""

import logging

import pytest

from pairs_trading import cli


def test_invalid_configuration_is_logged_and_returns_2(caplog):
    with caplog.at_level(logging.ERROR):
        assert cli.main(["run", "--n-plot-pairs", "-1"]) == 2
        assert cli.main(["run", "--formation-end", "2021-13-45"]) == 2
    assert "n_plot_pairs" in caplog.text
    assert "ISO" in caplog.text


@pytest.mark.parametrize(
    "exc",
    [
        RuntimeError("no cointegrated pairs passed the screen; nothing to backtest"),
        ValueError("formation_end=2021-12-31 leaves an empty formation or trading period"),
    ],
)
def test_pipeline_errors_are_logged_not_raised(monkeypatch, caplog, exc):
    def fail(cfg, refresh_data=False):
        raise exc

    monkeypatch.setattr(cli, "run_pipeline", fail)
    with caplog.at_level(logging.ERROR):
        assert cli.main(["run"]) == 1
    assert str(exc) in caplog.text


def test_unexpected_exceptions_still_propagate(monkeypatch):
    def fail(cfg, refresh_data=False):
        raise KeyError("programming error")

    monkeypatch.setattr(cli, "run_pipeline", fail)
    with pytest.raises(KeyError):
        cli.main(["run"])
