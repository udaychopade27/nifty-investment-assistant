import math
import logging

from app.services.market_data_service import MarketDataService
from app.db.models import DailyDecisionETF
from app.services.etf_index_registry import get_index_for_etf

logger = logging.getLogger(__name__)


class DailyETFAllocationService:
    """
    Converts tactical capital into ETF-wise unit plans.
    India-safe:
    • Whole units only
    • No forced buying
    • Capital-safe
    """

    @staticmethod
    def allocate(
        db,
        decision_date,
        etfs: list[str],
        available_capital: float,
    ) -> float:
        """
        Returns remaining capital after allocation.
        """

        if available_capital <= 0 or not etfs:
            return available_capital

        prices = MarketDataService.get_current_prices(etfs)

        remaining = available_capital

        for etf in etfs:
            price = prices.get(etf)
            if not price:
                logger.warning("No price for %s, skipping", etf)
                continue

            effective_price = price * 1.02  # slippage buffer
            units = math.floor(remaining / effective_price)

            if units < 1:
                db.add(
                    DailyDecisionETF(
                        decision_date=decision_date,
                        etf_symbol=etf,
                        index_name=get_index_for_etf(etf),
                        units_planned=0,
                        planned_amount=0.0,
                        status="SKIPPED",
                    )
                )
                continue

            amount = units * price
            remaining -= amount

            db.add(
                DailyDecisionETF(
                    decision_date=decision_date,
                    etf_symbol=etf,
                    index_name=get_index_for_etf(etf),
                    units_planned=units,
                    planned_amount=amount,
                    status="PLANNED",
                )
            )

            logger.info(
                "Planned %s | units=%s | amount=₹%.2f",
                etf,
                units,
                amount,
            )

        db.commit()
        return remaining
