"""Non-bypassable risk checks for options signals."""
from dataclasses import dataclass
from typing import Optional
from app.domain.options.models.types import OptionSignal


@dataclass
class RiskDecision:
    allowed: bool
    reason: str


class RiskManager:
    def check(self, signal: OptionSignal, capital_state: dict) -> RiskDecision:
        # Placeholder: always block until configured
        return RiskDecision(allowed=False, reason="Risk rules not configured")
