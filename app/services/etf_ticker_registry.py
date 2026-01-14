"""
MARKET DATA — ETF TICKER REGISTRY

Maps internal ETF symbols → market tickers.
This is NOT strategy logic.
This file is safe to evolve if tickers change.
"""

ETF_TICKER_MAP = {
    # Equity ETFs
    "NIFTYBEES": "NIFTYBEES.NS",
    "JUNIORBEES": "JUNIORBEES.NS",
    "LOWVOLIETF": "LOWVOLIETF.NS",
    "MIDCAPETF": "MIDCAPETF.NS",

    # Bond ETF
    "BHARATBOND": "BHARATBOND.NS",

    # Commodity ETF
    "GOLDBEES": "GOLDBEES.NS",
}


def get_ticker(symbol: str) -> str | None:
    return ETF_TICKER_MAP.get(symbol)
