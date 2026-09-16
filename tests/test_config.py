import pytest

from pairs_trading.config import PipelineConfig


def test_defaults_are_valid_and_dates_are_ordered():
    cfg = PipelineConfig()
    assert cfg.start < cfg.formation_end < cfg.end
    assert cfg.n_plot_pairs >= 0


@pytest.mark.parametrize(
    "changes",
    [
        {"formation_end": "2018-01-01"},  # out of order
        {"formation_end": "2021-13-45"},  # sorts between start and end as a string, not a date
        {"end": "not-a-date"},
        {"exit_z": 2.5},
        {"stop_z": 1.0},
        {"zscore_window": 1},
        {"coint_pvalue": 1.0},
        {"slippage_bps": -1.0},
        {"n_jobs": 0},
        {"sectors": ()},
        {"max_missing_frac": 1.5},
        {"ffill_limit": 0},
        {"trading_days_per_year": 0},
        {"n_plot_pairs": -1},
    ],
)
def test_invalid_values_are_rejected(changes):
    with pytest.raises(ValueError):
        PipelineConfig(**changes)


def test_replace_re_runs_validation():
    assert PipelineConfig().replace(n_plot_pairs=0).n_plot_pairs == 0
    with pytest.raises(ValueError):
        PipelineConfig().replace(n_plot_pairs=-1)
