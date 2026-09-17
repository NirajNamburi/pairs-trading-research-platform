"""Build the top-N US utilities universe by market capitalisation, reproducibly.

Candidates are every company classified in the GICS "Utilities" sector across the S&P 500,
S&P MidCap 400 and S&P SmallCap 600 (together the S&P 1500), read from Wikipedia's constituent
tables. Market capitalisations come from yfinance at selection time. A candidate is eligible
only if it has a continuous daily price history over the pipeline's date range, so the universe
never contains a stock the backtest would have to drop.

    python -m pairs_trading.universe_selection --n 50 --start 2019-01-01 --end 2025-12-31

writes ``src/pairs_trading/universes/utilities_top50.json`` listing every candidate, its market
cap, and whether it was selected or why it was excluded. The file is committed, so the claim
"top 50 by market cap" is auditable and re-runnable.

Known limitation: market caps and index membership are as of the selection date, not as of the
start of the backtest (survivorship bias; point-in-time constituents need paid data).
"""

import argparse
import io
import json
import logging
import re
import sys
from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from pairs_trading.data import clean_prices, download_prices

log = logging.getLogger(__name__)

WIKIPEDIA_TABLES: dict[str, str] = {
    "S&P 500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "S&P 400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
    "S&P 600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
}
USER_AGENT = "pairs-trading-research-pipeline/0.1 (universe selection script)"
DEFAULT_OUTPUT = Path(__file__).parent / "universes" / "utilities_top50.json"

CANDIDATE_COLUMNS = ["ticker", "name", "index", "sub_industry"]
_CLASS_SUFFIX = re.compile(r"\s*\((?:class|cl\.?)\s*[a-z]\)\s*$", re.IGNORECASE)


# --- pure functions (unit-tested) ------------------------------------------------------------


def normalize_symbol(symbol: str) -> str:
    """Wikipedia writes share classes as ``BRK.B``; yfinance wants ``BRK-B``."""
    return symbol.strip().upper().replace(".", "-")


def company_key(name: str) -> str:
    """``"Clearway Energy, Inc. (Class C)"`` -> ``"clearway energy, inc."``."""
    return _CLASS_SUFFIX.sub("", str(name)).strip().lower()


def dedupe_share_classes(candidates: pd.DataFrame) -> pd.DataFrame:
    """Keep one listing per company (the plain symbol over a hyphenated share class).

    Two share classes of the same company are the same economic exposure; pairing them with
    each other would be a trivially cointegrated, untradeable "pair".
    """
    ordered = candidates.assign(
        _key=candidates["name"].map(company_key),
        _has_class=candidates["ticker"].str.contains("-"),
    ).sort_values(["_key", "_has_class", "ticker"])
    deduped = ordered.drop_duplicates(subset="_key", keep="first")
    return deduped.drop(columns=["_key", "_has_class"]).sort_values("ticker").reset_index(drop=True)


def rank_top_n(candidates: pd.DataFrame, n: int) -> pd.DataFrame:
    """Top ``n`` eligible candidates by ``market_cap_usd`` (descending), with a 1-based rank."""
    if n < 1:
        raise ValueError("n must be >= 1")
    eligible = candidates.loc[candidates["eligible"] & candidates["market_cap_usd"].notna()]
    ranked = eligible.sort_values(["market_cap_usd", "ticker"], ascending=[False, True]).head(n)
    return ranked.assign(rank=range(1, len(ranked) + 1)).reset_index(drop=True)


def build_spec(
    *,
    name: str,
    sector: str,
    candidates: pd.DataFrame,
    n: int,
    start: str,
    end: str,
    max_missing_frac: float,
    selected_at: str,
) -> dict[str, object]:
    """Assemble the JSON document: selection criteria, chosen members, excluded candidates."""
    selected = rank_top_n(candidates, n)
    chosen = set(selected["ticker"])
    excluded = candidates.loc[~candidates["ticker"].isin(chosen)].copy()
    below_cut = excluded["eligible"] & (excluded["reason"] == "")
    excluded.loc[below_cut, "reason"] = f"eligible, but ranked below the top {n} by market cap"
    excluded = excluded.sort_values("market_cap_usd", ascending=False)

    def cap(value: object) -> float | None:
        return None if pd.isna(value) else float(value)

    return {
        "name": name,
        "sector": sector,
        "selected_at": selected_at,
        "criteria": {
            "n": n,
            "candidate_pool": "GICS Utilities constituents of the S&P 500, 400 and 600",
            "ranking": "market capitalisation from yfinance on selected_at, descending",
            "eligibility": (
                f"continuous daily price history {start} to {end}: at most "
                f"{max_missing_frac:.0%} of trading days missing"
            ),
        },
        "n_candidates": int(len(candidates)),
        "n_selected": int(len(selected)),
        "tickers": selected["ticker"].tolist(),
        "members": [
            {
                "rank": int(r.rank),
                "ticker": r.ticker,
                "name": r.name,
                "index": r.index,
                "sub_industry": r.sub_industry,
                "market_cap_usd": cap(r.market_cap_usd),
                "first_price_date": r.first_price_date,
            }
            for r in selected.itertuples(index=False)
        ],
        "excluded": [
            {
                "ticker": r.ticker,
                "name": r.name,
                "index": r.index,
                "market_cap_usd": cap(r.market_cap_usd),
                "reason": r.reason,
            }
            for r in excluded.itertuples(index=False)
        ],
    }


def write_spec(spec: dict[str, object], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return path


# --- network functions ----------------------------------------------------------------------


def fetch_candidates(sector: str = "Utilities") -> pd.DataFrame:
    """All ``sector`` constituents of the S&P 1500 from Wikipedia, one row per company."""
    import requests

    frames = []
    for index_name, url in WIKIPEDIA_TABLES.items():
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        response.raise_for_status()
        tables = pd.read_html(io.StringIO(response.text))
        table = next(t for t in tables if "GICS Sector" in t.columns and "Symbol" in t.columns)
        rows = table.loc[table["GICS Sector"] == sector]
        frames.append(
            pd.DataFrame(
                {
                    "ticker": rows["Symbol"].map(normalize_symbol),
                    "name": rows["Security"].astype(str).str.strip(),
                    "index": index_name,
                    "sub_industry": rows["GICS Sub-Industry"].astype(str).str.strip(),
                }
            )
        )
        log.info("%s: %d %s constituents", index_name, len(rows), sector)
    candidates = pd.concat(frames, ignore_index=True)[CANDIDATE_COLUMNS]
    candidates = candidates.drop_duplicates(subset="ticker")
    return dedupe_share_classes(candidates)


def fetch_market_caps(tickers: Iterable[str]) -> dict[str, float | None]:
    """Current market cap in USD per ticker (``None`` when yfinance has no figure)."""
    import yfinance as yf

    caps: dict[str, float | None] = {}
    for ticker in tickers:
        value = None
        try:
            value = yf.Ticker(ticker).fast_info["marketCap"]
        except Exception as exc:  # noqa: BLE001 - yfinance raises assorted types
            log.warning("no market cap for %s: %s", ticker, exc)
        caps[ticker] = float(value) if value is not None and value > 0 else None
    return caps


def check_history(
    tickers: Sequence[str], start: str, end: str, max_missing_frac: float
) -> pd.DataFrame:
    """Per-ticker eligibility: does a continuous price history exist over ``start``..``end``?

    Uses the pipeline's own download and cleaning rules so "eligible" here means exactly
    "survives :func:`pairs_trading.data.clean_prices`".
    """
    close, open_ = download_prices(list(tickers), start, end)
    close = close.dropna(how="all")
    missing_frac = close.isna().mean()
    first_dates = close.apply(lambda s: s.first_valid_index())
    _, _, dropped = clean_prices(close, open_.loc[close.index], max_missing_frac, ffill_limit=3)
    dropped_set = set(dropped)
    rows = []
    for ticker in tickers:
        frac = float(missing_frac.get(ticker, 1.0))
        first = first_dates.get(ticker)
        if pd.isna(first):
            # An all-NaN column (yfinance returned nothing for the symbol) yields NaT, not None,
            # because ``first_dates`` is a datetime Series.
            first = None
        eligible = ticker not in dropped_set
        if eligible:
            reason = ""
        elif first is None:
            reason = "no price data returned by yfinance (download failed or unknown symbol)"
        else:
            reason = f"{frac:.1%} of trading days missing (first price {first.date()})"
        rows.append(
            {
                "ticker": ticker,
                "first_price_date": None if first is None else str(first.date()),
                "missing_frac": frac,
                "eligible": eligible,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def build_universe(
    *,
    n: int,
    start: str,
    end: str,
    max_missing_frac: float = 0.02,
    sector: str = "Utilities",
    name: str = "utilities_top50",
    selected_at: str | None = None,
) -> dict[str, object]:
    candidates = fetch_candidates(sector)
    log.info("%d %s candidates after de-duplicating share classes", len(candidates), sector)
    caps = fetch_market_caps(candidates["ticker"])
    candidates["market_cap_usd"] = candidates["ticker"].map(caps)
    history = check_history(candidates["ticker"].tolist(), start, end, max_missing_frac)
    candidates = candidates.merge(history, on="ticker", how="left")
    no_cap = candidates["market_cap_usd"].isna()
    candidates.loc[no_cap, "eligible"] = False
    candidates.loc[no_cap & (candidates["reason"] == ""), "reason"] = "no market cap available"
    return build_spec(
        name=name,
        sector=sector,
        candidates=candidates,
        n=n,
        start=start,
        end=end,
        max_missing_frac=max_missing_frac,
        selected_at=selected_at or date.today().isoformat(),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--max-missing-frac", type=float, default=0.02)
    parser.add_argument("--sector", default="Utilities")
    parser.add_argument("--name", default="utilities_top50")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")

    spec = build_universe(
        n=args.n,
        start=args.start,
        end=args.end,
        max_missing_frac=args.max_missing_frac,
        sector=args.sector,
        name=args.name,
    )
    path = write_spec(spec, args.output)
    members = spec["members"]
    sys.stdout.write(
        f"selected {spec['n_selected']} of {spec['n_candidates']} {args.sector} candidates "
        f"-> {path}\n"
    )
    for m in members:
        cap = m["market_cap_usd"]
        sys.stdout.write(
            f"{m['rank']:3d}  {m['ticker']:6s} {m['name'][:34]:34s} {m['index']:8s} "
            f"${cap / 1e9:7.1f}B\n"
        )
    if spec["excluded"]:
        sys.stdout.write("excluded:\n")
        for e in spec["excluded"]:
            sys.stdout.write(f"     {e['ticker']:6s} {e['name'][:34]:34s} {e['reason']}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
