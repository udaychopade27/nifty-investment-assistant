"""
STRATEGY GOVERNANCE & VERSIONING — MODULE S11

This module defines the explicit strategy version identifier.
It exists to guarantee long-term auditability, reproducibility,
and protection against silent strategy changes.

Every persisted decision and record must reference this version.
"""

# -------------------------------------------------------------------
# Strategy Version
# -------------------------------------------------------------------

STRATEGY_VERSION = "1.0.0"

# -------------------------------------------------------------------
# Helper Function
# -------------------------------------------------------------------

def get_strategy_version() -> str:
    """
    Return the current strategy version.

    This function is intentionally simple and side-effect free to ensure
    deterministic access to the strategy version across the system.
    """
    return STRATEGY_VERSION
