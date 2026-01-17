from datetime import datetime

STRATEGY_REGISTRY = {
    "BASE_CORE_V1": {
        "type": "LONG_TERM",
        "risk": "MEDIUM",
        "rebalance": "ANNUAL",
        "dip_enabled": True,
    },
    "AGGRESSIVE_GROWTH_V1": {
        "type": "LONG_TERM",
        "risk": "HIGH",
        "rebalance": "ANNUAL",
        "dip_enabled": True,
    },
}


def get_strategy(strategy_id: str) -> dict:
    if strategy_id not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {strategy_id}")
    return STRATEGY_REGISTRY[strategy_id]