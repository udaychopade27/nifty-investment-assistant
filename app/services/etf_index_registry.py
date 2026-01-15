"""
ETF → INDEX REGISTRY
Single source of truth for ETF → Index mapping
"""

ETF_INDEX_MAP = {
    "NIFTYBEES": "NIFTY_50",
    "JUNIORBEES": "NIFTY_NEXT_50",
    "LOWVOLIETF": "NIFTY_LOW_VOL_30",
    "MIDCAPETF": "NIFTY_MIDCAP_150",
    "GOLDBEES": "GOLD_SPOT",
    "BHARATBOND": "BHARAT_BOND",
}


def get_index_for_etf(etf_symbol: str) -> str:
    if not etf_symbol:
        raise ValueError("ETF symbol required")

    symbol = etf_symbol.upper()
    if symbol not in ETF_INDEX_MAP:
        raise ValueError(f"No index mapping found for ETF {symbol}")

    return ETF_INDEX_MAP[symbol]
