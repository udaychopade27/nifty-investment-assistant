from dataclasses import dataclass


@dataclass(frozen=True)
class ETF:
    """
    Represents an allowed ETF instrument in the strategy universe.
    """
    symbol: str
    asset_class: str
    risk_role: str
