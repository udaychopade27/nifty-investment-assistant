import logging

logger = logging.getLogger(__name__)


class AggressiveRebalanceService:
    @staticmethod
    def evaluate(
        current_weights: dict,
        target_weights: dict,
    ) -> dict:

        actions = []

        for etf, target in target_weights.items():
            current = current_weights.get(etf, 0.0)
            drift = current - target

            if abs(drift) > 0.08:
                actions.append(
                    {
                        "etf": etf,
                        "action": "SELL_OR_REBALANCE",
                        "drift": round(drift, 3),
                    }
                )

        return {
            "rebalance_required": bool(actions),
            "actions": actions,
        }
