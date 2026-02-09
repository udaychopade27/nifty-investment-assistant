"""In-memory or Redis-backed market state for options domain."""
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class MarketState:
    spot: Dict[str, float] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)
    indicators: Dict[str, Any] = field(default_factory=dict)
    current_candles: Dict[str, Any] = field(default_factory=dict)
    signals: Dict[str, Any] = field(default_factory=dict)
    risk: Dict[str, Any] = field(default_factory=dict)
    paper: Dict[str, Any] = field(default_factory=dict)
