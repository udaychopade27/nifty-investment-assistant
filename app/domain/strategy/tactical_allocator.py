"""
STRATEGY — TACTICAL ETF ALLOCATOR (S4)

Determines WHICH ETFs are suitable for tactical deployment
based on market regime and dip severity.

This module:
• Is deterministic
• Uses NO prediction
• Does NOT execute trades
• Returns guidance only (human-in-the-loop)
"""

from typing import List


def determine_tactical_etfs(
    drawdown_pct: float,
    deploy_pct: float,
    vix_value: float | None = None,
    is_bear_market: bool = False,
) -> dict:
    """
    Determine preferred ETFs for tactical deployment.

    Args:
        drawdown_pct: Market drawdown from recent high (%)
        deploy_pct: Tactical deploy percentage (0–1)
        vix_value: Optional volatility index value
        is_bear_market: Structural bear flag

    Returns:
        dict with:
        - primary_etfs
        - secondary_etfs
        - rationale
    """

    # Safety: no deployment, no ETF guidance
    if deploy_pct <= 0:
        return {
            "primary_etfs": [],
            "secondary_etfs": [],
            "rationale": "No tactical deployment today.",
        }

    # Defensive regime
    if is_bear_market:
        return {
            "primary_etfs": ["LOWVOLIETF"],
            "secondary_etfs": ["GOLDBEES"],
            "rationale": "Bear market regime detected. Defensive ETFs preferred.",
        }

    # High volatility override
    if vix_value and vix_value >= 20:
        return {
            "primary_etfs": ["LOWVOLIETF"],
            "secondary_etfs": ["NIFTYBEES"],
            "rationale": "High volatility environment. Low-volatility bias applied.",
        }

    # Deep dip → allow risk
    if drawdown_pct >= 5:
        return {
            "primary_etfs": ["NIFTYBEES"],
            "secondary_etfs": ["MIDCAPETF"],
            "rationale": "Deep market drawdown. Large-cap anchor with mid-cap exposure.",
        }

    # Medium dip
    if drawdown_pct >= 3:
        return {
            "primary_etfs": ["NIFTYBEES"],
            "secondary_etfs": ["JUNIORBEES"],
            "rationale": "Moderate dip. Broad-market exposure preferred.",
        }

    # Shallow dip
    if drawdown_pct >= 1:
        return {
            "primary_etfs": ["NIFTYBEES"],
            "secondary_etfs": ["LOWVOLIETF"],
            "rationale": "Mild dip. Conservative large-cap bias.",
        }

    # Default fallback
    return {
        "primary_etfs": ["NIFTYBEES"],
        "secondary_etfs": [],
        "rationale": "No significant dip. Tactical exposure limited to large caps.",
    }
