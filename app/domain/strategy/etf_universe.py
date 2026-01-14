"""
ETF UNIVERSE STRATEGY — MODULE S1

This file defines the fixed universe of ETFs allowed by the strategy and
their strategic roles. It specifies WHAT instruments are permitted,
not how much to allocate or when to invest.

Rules:
- Read-only strategy definition
- No database imports
- No API imports
- No allocation or execution logic
- Deterministic and version-stable
"""

# -------------------------------------------------------------------
# Asset Class Constants
# -------------------------------------------------------------------

ASSET_CLASS_EQUITY = "equity"
ASSET_CLASS_BOND = "bond"
ASSET_CLASS_COMMODITY = "commodity"

ASSET_CLASSES = (
    ASSET_CLASS_EQUITY,
    ASSET_CLASS_BOND,
    ASSET_CLASS_COMMODITY,
)

# -------------------------------------------------------------------
# Risk Role Constants
# -------------------------------------------------------------------

RISK_ROLE_CORE = "core"
RISK_ROLE_GROWTH = "growth"
RISK_ROLE_DEFENSIVE = "defensive"
RISK_ROLE_CAPITAL_PROTECTION = "capital_protection"
RISK_ROLE_CRISIS_HEDGE = "crisis_hedge"
RISK_ROLE_HIGH_RISK_CAPPED = "high_risk_growth_capped"

RISK_ROLES = (
    RISK_ROLE_CORE,
    RISK_ROLE_GROWTH,
    RISK_ROLE_DEFENSIVE,
    RISK_ROLE_CAPITAL_PROTECTION,
    RISK_ROLE_CRISIS_HEDGE,
    RISK_ROLE_HIGH_RISK_CAPPED,
)

# -------------------------------------------------------------------
# ETF Registry
# -------------------------------------------------------------------

ETF_UNIVERSE = {
    "NIFTYBEES": {
        "asset_class": ASSET_CLASS_EQUITY,
        "risk_role": RISK_ROLE_CORE,
    },
    "JUNIORBEES": {
        "asset_class": ASSET_CLASS_EQUITY,
        "risk_role": RISK_ROLE_GROWTH,
    },
    "LOWVOLIETF": {
        "asset_class": ASSET_CLASS_EQUITY,
        "risk_role": RISK_ROLE_DEFENSIVE,
    },
    "BHARATBOND": {
        "asset_class": ASSET_CLASS_BOND,
        "risk_role": RISK_ROLE_CAPITAL_PROTECTION,
    },
    "GOLDBEES": {
        "asset_class": ASSET_CLASS_COMMODITY,
        "risk_role": RISK_ROLE_CRISIS_HEDGE,
    },
    "MIDCAPETF": {
        "asset_class": ASSET_CLASS_EQUITY,
        "risk_role": RISK_ROLE_HIGH_RISK_CAPPED,
    },
}

# -------------------------------------------------------------------
# Helper Validation
# -------------------------------------------------------------------

def is_valid_etf(symbol: str) -> bool:
    """
    Check whether a given ETF symbol is part of the allowed strategy universe.
    """
    if not isinstance(symbol, str):
        return False
    return symbol in ETF_UNIVERSE
