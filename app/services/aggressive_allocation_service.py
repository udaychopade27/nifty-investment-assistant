import logging
from app.domain.strategy.aggressive_etf_universe import get_target_weights

logger = logging.getLogger(__name__)


class AggressiveAllocationService:
    @staticmethod
    def allocate(monthly_capital: float) -> dict:
        weights = get_target_weights()
        allocation = {}

        for etf, w in weights.items():
            allocation[etf] = round(monthly_capital * w, 2)

        logger.info(
            "AGGRESSIVE_GROWTH_V1 allocation: %s (capital=%s)",
            allocation,
            monthly_capital,
        )

        return allocation
