"""Typed structures for options trading domain.

Keep these as simple, serializable structures. Do not embed strategy logic here.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Literal

OptionSide = Literal["CE", "PE"]
SignalAction = Literal["BUY"]


@dataclass(frozen=True)
class OptionSymbol:
    underlying: str
    expiry: str
    strike: float
    side: OptionSide
    exchange: str = "NSE"


@dataclass(frozen=True)
class OptionSignal:
    symbol: OptionSymbol
    action: SignalAction
    price: float
    timestamp: datetime
    reason: str
    confidence: Optional[float] = None
