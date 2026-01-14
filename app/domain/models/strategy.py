from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyMetadata:
    """
    Immutable metadata describing the active strategy version.
    """
    version: str
    description: str
