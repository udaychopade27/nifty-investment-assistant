"""Base interface for stateless options strategies."""
from typing import Protocol, Optional
from app.domain.options.models.types import OptionSignal


class Strategy(Protocol):
    def evaluate(self, market_state: dict) -> Optional[OptionSignal]:
        ...
