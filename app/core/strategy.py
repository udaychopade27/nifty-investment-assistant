"""
Central strategy registry.

This module exposes the active strategy version and metadata
in a read-only manner. No strategy rules live here.
"""

from app.domain.strategy.governance import get_strategy_version
from app.domain.models.strategy import StrategyMetadata


def load_strategy_metadata() -> StrategyMetadata:
    """
    Load current strategy metadata.
    """
    version = get_strategy_version()
    description = "Deterministic, rule-based long-term ETF investing strategy"

    return StrategyMetadata(
        version=version,
        description=description,
    )
