"""Pipeline configuration.

Every modelling assumption lives in :class:`PipelineConfig` so it is easy to find, cite and change.
The cost constants below are deliberately module-level so they are greppable.
"""

import dataclasses
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# --- Explicit cost assumptions -------------------------------------------------------------
# Both are expressed in basis points of the traded notional and are applied on EVERY execution
# (entry and exit, each leg). 5 bps + 5 bps is a simplified, conservative estimate for liquid
# large-cap US equities. A more advanced model would make slippage a function of trade size
# relative to average daily volume; that is explicitly out of scope.
SLIPPAGE_BPS: float = 5.0
TRANSACTION_COST_BPS: float = 5.0


@dataclass(frozen=True)
class PipelineConfig:
    """All tunable parameters of the research pipeline.

    Dates are inclusive ISO strings. The *formation* period (``start`` .. ``formation_end``) is
    used only to screen pairs and fit hedge ratios; the *trading* period (the day after
    ``formation_end`` .. ``end``) is the out-of-sample backtest window.
    """

    # Data
    start: str = "2019-01-01"
    formation_end: str = "2021-12-31"
    end: str = "2024-12-31"
    sectors: tuple[str, ...] = ("energy", "financials", "utilities")
    max_missing_frac: float = 0.02  # drop a ticker if more than this fraction of closes is missing
    ffill_limit: int = 3  # forward-fill gaps of at most this many trading days

    # Cointegration screening (formation period only)
    coint_pvalue: float = 0.05
    require_positive_hedge_ratio: bool = True

    # Signal
    zscore_window: int = 30  # rolling window (trading days) for spread mean / std
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float | None = None  # optional spread blow-out stop; disabled by default

    # Costs (basis points of traded notional, per execution)
    slippage_bps: float = SLIPPAGE_BPS
    transaction_cost_bps: float = TRANSACTION_COST_BPS

    # Performance metrics
    risk_free_rate: float = 0.0  # annual; 0 keeps the Sharpe ratio simple and is stated as such
    trading_days_per_year: int = 252

    # Execution / IO
    n_jobs: int = 1  # >1 parallelises the pairwise cointegration tests across processes
    data_dir: Path = Path("data")
    reports_dir: Path = Path("reports")
    n_plot_pairs: int = 2  # how many top pairs get z-score / equity charts

    def __post_init__(self) -> None:
        try:
            start, formation_end, end = (
                date.fromisoformat(d) for d in (self.start, self.formation_end, self.end)
            )
        except ValueError as exc:
            raise ValueError(f"dates must be ISO strings (YYYY-MM-DD): {exc}") from exc
        if not start < formation_end < end:
            raise ValueError("expected start < formation_end < end")
        if not 0.0 <= self.max_missing_frac <= 1.0:
            raise ValueError("max_missing_frac must be in [0, 1]")
        if self.ffill_limit < 1:
            raise ValueError("ffill_limit must be >= 1")
        if not 0.0 < self.exit_z < self.entry_z:
            raise ValueError("expected 0 < exit_z < entry_z")
        if self.stop_z is not None and self.stop_z <= self.entry_z:
            raise ValueError("stop_z must be greater than entry_z")
        if self.zscore_window < 2:
            raise ValueError("zscore_window must be at least 2")
        if not 0.0 < self.coint_pvalue < 1.0:
            raise ValueError("coint_pvalue must be in (0, 1)")
        if self.slippage_bps < 0 or self.transaction_cost_bps < 0:
            raise ValueError("costs cannot be negative")
        if self.trading_days_per_year < 1:
            raise ValueError("trading_days_per_year must be >= 1")
        if self.n_jobs < 1:
            raise ValueError("n_jobs must be >= 1")
        if self.n_plot_pairs < 0:
            raise ValueError("n_plot_pairs must be >= 0")
        if not self.sectors:
            raise ValueError("at least one sector is required")

    def replace(self, **changes: object) -> "PipelineConfig":
        """Return a copy with the given fields changed (re-runs validation)."""
        return dataclasses.replace(self, **changes)

    def to_dict(self) -> dict[str, object]:
        d = dataclasses.asdict(self)
        d["data_dir"] = str(self.data_dir)
        d["reports_dir"] = str(self.reports_dir)
        d["sectors"] = list(self.sectors)
        return d
