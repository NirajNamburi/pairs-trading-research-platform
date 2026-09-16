"""Stage 4 - event-driven backtest with realistic costs.

Execution model
---------------
* The target position for day ``t`` is decided at the close of ``t`` (signals module).
* It is executed at the **open of ``t + 1``**. Internally ``held = target.shift(1)``: a change in
  ``held`` on day ``i`` is filled at ``open[i]``. This is what removes look-ahead bias - you
  cannot trade at a closing price you have only just observed.
* Position sizing is dollar-neutral-ish and normalised: each trade puts exactly **$1 of gross
  notional** on (``qty_a * open_a + qty_b * open_b = 1`` with ``qty_b = hedge_ratio * qty_a``),
  so P&L is directly a return on $1 of capital per pair. Quantities are fixed for the life of
  the trade.
* **Slippage** (``slippage_bps``) moves every fill against the trader: buys fill above the open,
  sells fill below it. Applied on both legs, on entry and exit.
* **Commission** (``transaction_cost_bps``) is charged on the gross notional traded, on entry
  and exit.
* Daily P&L is marked to market on closes while in a position; costs are booked on the day they
  are incurred. ``equity = 1 + cumsum(daily_pnl)`` (capital is not compounded; a $1 book per
  pair is the stated assumption).
* A position still open on the last day is force-closed at the final close (with costs) so
  every trade has a realised P&L. Such trades are tagged ``exit_reason="end_of_period"``.
"""

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pairs_trading.config import PipelineConfig

TRADE_COLUMNS = [
    "ticker_a",
    "ticker_b",
    "direction",
    "signal_date",
    "entry_date",
    "exit_date",
    "holding_days",
    "exit_reason",
    "entry_price_a",
    "entry_price_b",
    "exit_price_a",
    "exit_price_b",
    "qty_a",
    "qty_b",
    "gross_pnl",
    "slippage_cost",
    "commission_cost",
    "net_pnl",
]


@dataclass(frozen=True)
class Trade:
    """One round trip. Prices are the *filled* prices (after slippage). All P&L is in dollars on
    the trade's $1 gross entry notional."""

    ticker_a: str
    ticker_b: str
    direction: int  # +1 long spread (long A / short B), -1 short spread (short A / long B)
    signal_date: pd.Timestamp  # close at which the decision was made
    entry_date: pd.Timestamp  # open at which it was executed (next trading day)
    exit_date: pd.Timestamp
    holding_days: int  # trading days between entry and exit
    exit_reason: str  # "signal" | "end_of_period"
    entry_price_a: float
    entry_price_b: float
    exit_price_a: float
    exit_price_b: float
    qty_a: float
    qty_b: float
    gross_pnl: float  # P&L at the theoretical (unslipped) prices, before commissions
    slippage_cost: float  # >= 0
    commission_cost: float  # >= 0
    net_pnl: float  # gross - slippage - commission


@dataclass
class PairResult:
    ticker_a: str
    ticker_b: str
    hedge_ratio: float
    trades: list[Trade] = field(default_factory=list)
    daily_pnl: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    @property
    def name(self) -> str:
        return f"{self.ticker_a}/{self.ticker_b}"


def _fill_prices(px_a: float, px_b: float, direction: int, slip: float) -> tuple[float, float]:
    """Fill prices after adverse slippage for trading ``direction`` on the spread.

    ``direction=+1`` buys A and sells B; ``direction=-1`` sells A and buys B.
    """
    if direction == 1:
        return px_a * (1.0 + slip), px_b * (1.0 - slip)
    return px_a * (1.0 - slip), px_b * (1.0 + slip)


@dataclass
class _OpenPosition:
    direction: int
    signal_idx: int
    entry_idx: int
    raw_a: float
    raw_b: float
    fill_a: float
    fill_b: float
    qty_a: float
    qty_b: float
    entry_slippage: float
    entry_commission: float


def _close_trade(
    pos: _OpenPosition,
    exit_idx: int,
    raw_a: float,
    raw_b: float,
    fill_a: float,
    fill_b: float,
    exit_commission: float,
    exit_reason: str,
    dates: pd.Index,
    ticker_a: str,
    ticker_b: str,
) -> Trade:
    d = pos.direction
    gross = d * (pos.qty_a * (raw_a - pos.raw_a) - pos.qty_b * (raw_b - pos.raw_b))
    exit_slippage = pos.qty_a * abs(fill_a - raw_a) + pos.qty_b * abs(fill_b - raw_b)
    slippage = pos.entry_slippage + exit_slippage
    commission = pos.entry_commission + exit_commission
    return Trade(
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        direction=d,
        signal_date=dates[pos.signal_idx],
        entry_date=dates[pos.entry_idx],
        exit_date=dates[exit_idx],
        holding_days=exit_idx - pos.entry_idx,
        exit_reason=exit_reason,
        entry_price_a=pos.fill_a,
        entry_price_b=pos.fill_b,
        exit_price_a=fill_a,
        exit_price_b=fill_b,
        qty_a=pos.qty_a,
        qty_b=pos.qty_b,
        gross_pnl=gross,
        slippage_cost=slippage,
        commission_cost=commission,
        net_pnl=gross - slippage - commission,
    )


def backtest_pair(
    *,
    close_a: pd.Series,
    close_b: pd.Series,
    open_a: pd.Series,
    open_b: pd.Series,
    target_positions: pd.Series,
    hedge_ratio: float,
    cfg: PipelineConfig,
    ticker_a: str = "A",
    ticker_b: str = "B",
) -> PairResult:
    """Simulate one pair day by day over ``target_positions.index``.

    ``target_positions[t]`` is the position decided at the close of ``t``; it is held from the
    open of the next day in the index. Price series must cover every date in that index.
    """
    dates = target_positions.index
    if len(dates) == 0:
        raise ValueError("target_positions is empty")
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("target_positions index must be sorted and unique")

    ca = close_a.reindex(dates).to_numpy(dtype=float)
    cb = close_b.reindex(dates).to_numpy(dtype=float)
    oa = open_a.reindex(dates).to_numpy(dtype=float)
    ob = open_b.reindex(dates).to_numpy(dtype=float)
    for name, arr in (("close_a", ca), ("close_b", cb), ("open_a", oa), ("open_b", ob)):
        if np.isnan(arr).any():
            raise ValueError(f"{name} is missing prices for some target dates")
    if hedge_ratio <= 0 or not math.isfinite(hedge_ratio):
        raise ValueError("hedge_ratio must be a positive finite number")

    target = target_positions.to_numpy(dtype=int)
    if not np.isin(target, (-1, 0, 1)).all():
        raise ValueError("target positions must be in {-1, 0, 1}")
    held = np.concatenate(([0], target[:-1]))  # decided at close i-1, held from open i

    slip = cfg.slippage_bps / 1e4
    comm = cfg.transaction_cost_bps / 1e4

    n = len(dates)
    daily = np.zeros(n, dtype=float)
    trades: list[Trade] = []
    pos: _OpenPosition | None = None
    prev_ca = prev_cb = math.nan

    for i in range(n):
        desired = int(held[i])
        current = pos.direction if pos is not None else 0
        pnl = 0.0

        if desired != current:
            if pos is not None:
                # Exit at today's open: trade in the opposite direction of the position.
                fill_a, fill_b = _fill_prices(oa[i], ob[i], -pos.direction, slip)
                pnl += pos.direction * (
                    pos.qty_a * (fill_a - prev_ca) - pos.qty_b * (fill_b - prev_cb)
                )
                exit_commission = comm * (pos.qty_a * oa[i] + pos.qty_b * ob[i])
                pnl -= exit_commission
                trades.append(
                    _close_trade(
                        pos,
                        i,
                        oa[i],
                        ob[i],
                        fill_a,
                        fill_b,
                        exit_commission,
                        "signal",
                        dates,
                        ticker_a,
                        ticker_b,
                    )
                )
                pos = None
            if desired != 0:
                gross_notional = oa[i] + hedge_ratio * ob[i]
                qty_a = 1.0 / gross_notional
                qty_b = hedge_ratio * qty_a
                fill_a, fill_b = _fill_prices(oa[i], ob[i], desired, slip)
                entry_commission = comm * 1.0  # entry notional is exactly $1 by construction
                entry_slippage = qty_a * abs(fill_a - oa[i]) + qty_b * abs(fill_b - ob[i])
                pnl -= entry_commission
                pnl += desired * (qty_a * (ca[i] - fill_a) - qty_b * (cb[i] - fill_b))
                pos = _OpenPosition(
                    direction=desired,
                    signal_idx=i - 1,
                    entry_idx=i,
                    raw_a=oa[i],
                    raw_b=ob[i],
                    fill_a=fill_a,
                    fill_b=fill_b,
                    qty_a=qty_a,
                    qty_b=qty_b,
                    entry_slippage=entry_slippage,
                    entry_commission=entry_commission,
                )
        elif pos is not None:
            pnl += pos.direction * (pos.qty_a * (ca[i] - prev_ca) - pos.qty_b * (cb[i] - prev_cb))

        daily[i] = pnl
        prev_ca, prev_cb = ca[i], cb[i]

    if pos is not None:
        # Force-close at the final close so the trade is realised. Daily P&L is already marked
        # to that close; only the slippage and commission of the exit remain to be booked.
        last = n - 1
        fill_a, fill_b = _fill_prices(ca[last], cb[last], -pos.direction, slip)
        slippage_adj = pos.direction * (
            pos.qty_a * (fill_a - ca[last]) - pos.qty_b * (fill_b - cb[last])
        )
        exit_commission = comm * (pos.qty_a * ca[last] + pos.qty_b * cb[last])
        daily[last] += slippage_adj - exit_commission
        trades.append(
            _close_trade(
                pos,
                last,
                ca[last],
                cb[last],
                fill_a,
                fill_b,
                exit_commission,
                "end_of_period",
                dates,
                ticker_a,
                ticker_b,
            )
        )

    daily_pnl = pd.Series(daily, index=dates, name=f"{ticker_a}/{ticker_b}")
    equity = (1.0 + daily_pnl.cumsum()).rename("equity")
    return PairResult(ticker_a, ticker_b, hedge_ratio, trades, daily_pnl, equity)


def trades_to_frame(results: list[PairResult]) -> pd.DataFrame:
    rows = [vars(t) for r in results for t in r.trades]
    frame = pd.DataFrame(rows, columns=TRADE_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(["entry_date", "ticker_a", "ticker_b"]).reset_index(drop=True)
    return frame


def aggregate_portfolio(results: list[PairResult]) -> pd.Series:
    """Equal-weight portfolio daily P&L per $1 of total capital.

    Capital is split evenly across *all* backtested pairs whether or not they trade on a given
    day, so this is the honest headline number - it includes pairs that never fire.
    """
    if not results:
        raise ValueError("no pair results to aggregate")
    frame = pd.concat([r.daily_pnl.rename(r.name) for r in results], axis=1).fillna(0.0)
    return frame.mean(axis=1).rename("portfolio_pnl")
