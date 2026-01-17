from datetime import datetime

STRATEGIES = {
    "AGGRESSIVE_GROWTH_V1": {
        "strategy_id": "AGGRESSIVE_GROWTH_V1",
        "strategy_type": "LONG_TERM",
        "risk_class": "HIGH",
        "rebalance_frequency": "ANNUAL",
        "dip_deployment_enabled": True,
        "created_at": datetime.utcnow(),
    }
}


def get_strategy(strategy_id: str) -> dict:
    if strategy_id not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_id}")
    return STRATEGIES[strategy_id]
