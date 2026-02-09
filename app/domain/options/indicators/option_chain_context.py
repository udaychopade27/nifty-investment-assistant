"""Option chain context: PCR, OI walls (ATM only)."""
from typing import Optional


def pcr(oi_ce: Optional[float], oi_pe: Optional[float]) -> Optional[float]:
    if oi_ce is None or oi_pe is None or oi_ce == 0:
        return None
    return oi_pe / oi_ce
