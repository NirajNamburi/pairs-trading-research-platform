"""Stock universe: S&P 500 constituents grouped by GICS sector.

Pairs are only screened *within* a sector. Same-sector pairs have an economic reason to be
cointegrated (shared factor exposure) and restricting to them shrinks the number of hypothesis
tests, which matters because Engle-Granger screening at p < 0.05 across thousands of unrelated
pairs would produce hundreds of false positives by chance alone.

Known limitation (survivorship bias): these are constituents as of 2024. Companies that were in
the index in 2019 but were later delisted, acquired or removed are not represented. Free data
sources cannot easily fix this, so it is documented rather than solved.

Symbols that Yahoo Finance no longer serves under their 2024 ticker (BK, CTRA, HES and MMC as of
September 2026, renamed or acquired) are left out so the default run is clean. Any symbol that
breaks in future is dropped by the cleaning step with a warning rather than crashing the run.
"""

from collections.abc import Iterable

SECTORS: dict[str, tuple[str, ...]] = {
    "energy": (
        "APA",
        "BKR",
        "COP",
        "CVX",
        "DVN",
        "EOG",
        "EQT",
        "FANG",
        "HAL",
        "KMI",
        "MPC",
        "OKE",
        "OXY",
        "PSX",
        "SLB",
        "TRGP",
        "VLO",
        "WMB",
        "XOM",
    ),
    "financials": (
        # Banks
        "BAC",
        "C",
        "CFG",
        "COF",
        "FITB",
        "GS",
        "HBAN",
        "JPM",
        "KEY",
        "MS",
        "MTB",
        "NTRS",
        "PNC",
        "RF",
        "STT",
        "TFC",
        "USB",
        "WFC",
        "ZION",
        # Insurance
        "AFL",
        "AIG",
        "ALL",
        "CB",
        "MET",
        "PGR",
        "PRU",
        "TRV",
        # Asset management, exchanges, data and payments
        "AJG",
        "AMP",
        "AON",
        "AXP",
        "BLK",
        "CME",
        "ICE",
        "MA",
        "MCO",
        "MSCI",
        "NDAQ",
        "SCHW",
        "SPGI",
        "TROW",
        "V",
    ),
    "utilities": (
        "AEE",
        "AEP",
        "AES",
        "ATO",
        "AWK",
        "CMS",
        "CNP",
        "D",
        "DTE",
        "DUK",
        "ED",
        "EIX",
        "ES",
        "ETR",
        "EVRG",
        "EXC",
        "FE",
        "LNT",
        "NEE",
        "NI",
        "NRG",
        "PEG",
        "PNW",
        "PPL",
        "SO",
        "SRE",
        "WEC",
        "XEL",
    ),
}


def available_sectors() -> list[str]:
    return sorted(SECTORS)


def get_universe(sectors: Iterable[str]) -> dict[str, str]:
    """Return ``{ticker: sector}`` for the requested sectors.

    Raises ``KeyError`` naming the unknown sector and the available ones.
    """
    universe: dict[str, str] = {}
    for sector in sectors:
        key = sector.lower()
        if key not in SECTORS:
            raise KeyError(f"unknown sector {sector!r}; available: {available_sectors()}")
        for ticker in SECTORS[key]:
            universe[ticker] = key
    return universe
