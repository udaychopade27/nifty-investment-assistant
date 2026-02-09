"""Execution stub for options orders."""
from typing import Dict
from app.domain.options.models.types import OptionSignal


class OptionsExecutor:
    def execute(self, signal: OptionSignal) -> Dict[str, str]:
        # Placeholder: no-op
        return {"status": "skipped", "reason": "Execution not implemented"}
