import logging
from app.services.aggressive_allocation_service import (
    AggressiveAllocationService,
)
from app.domain.strategy.aggressive_dip_strategy import (
    evaluate_aggressive_dip,
)

logger = logging.getLogger(__name__)


class AggressiveStrategyService:
    STRATEGY_ID = "AGGRESSIVE_GROWTH_V1"

    @staticmethod
    def run_monthly_sip(monthly_capital: float) -> dict:
        allocation = AggressiveAllocationService.allocate(monthly_capital)

        return {
            "strategy": AggressiveStrategyService.STRATEGY_ID,
            "type": "MONTHLY_SIP",
            "allocation": allocation,
        }

    @staticmethod
    def run_dip(
        nifty_drawdown_52w: float,
        midcap_underperformance: float,
        vix_value: float | None,
        dip_capital: float,
    ) -> dict:

        decision = evaluate_aggressive_dip(
            nifty_drawdown_52w,
            midcap_underperformance,
            vix_value,
        )

        if not decision["deploy"]:
            return decision

        allocation = {
            etf: round(dip_capital * w, 2)
            for etf, w in decision["allocation"].items()
        }

        return {
            "strategy": AggressiveStrategyService.STRATEGY_ID,
            "type": "DIP",
            "reason": decision["reason"],
            "allocation": allocation,
        }
