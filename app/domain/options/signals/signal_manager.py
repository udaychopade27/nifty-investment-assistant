"""Validate, deduplicate, and throttle signals."""
from typing import Optional
from app.domain.options.models.types import OptionSignal


class SignalManager:
    def accept(self, signal: OptionSignal) -> Optional[OptionSignal]:
        # Placeholder: pass-through
        return signal
