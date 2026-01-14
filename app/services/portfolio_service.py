# app/services/portfolio_service.py

import logging
from collections import defaultdict

from app.db.models import ExecutedInvestment
from app.services.market_data_service import MarketDataService

logger = logging.getLogger(__name__)


class PortfolioService:
    @staticmethod
    def build_snapshot(db) -> dict:
        logger.info("🔍 Building portfolio snapshot")

        investments = db.query(ExecutedInvestment).all()

        # ------------------------------------------------------------
        # No investments case
        # ------------------------------------------------------------
        if not investments:
            logger.info("ℹ️ No executed investments found")
            return {
                "positions": [],
                "summary": {
                    "base_invested": 0.0,
                    "tactical_invested": 0.0,
                    "total_invested": 0.0,
                    "current_value": 0.0,
                    "pnl": 0.0,
                    "pnl_pct": 0.0,
                },
            }

        # ------------------------------------------------------------
        # Aggregate executions
        # ------------------------------------------------------------
        grouped = defaultdict(lambda: {"units": 0.0, "invested": 0.0})
        base_invested = 0.0
        tactical_invested = 0.0

        for inv in investments:
            grouped[inv.etf_symbol]["units"] += inv.units
            grouped[inv.etf_symbol]["invested"] += inv.invested_amount

            if inv.capital_type == "BASE":
                base_invested += inv.invested_amount
            elif inv.capital_type == "TACTICAL":
                tactical_invested += inv.invested_amount

        # ------------------------------------------------------------
        # Fetch live prices (may be partial or empty)
        # ------------------------------------------------------------
        prices = MarketDataService.get_current_prices(list(grouped.keys()))

        positions = []
        total_invested = 0.0
        current_value = 0.0

        # ------------------------------------------------------------
        # Build ETF-wise positions
        # ------------------------------------------------------------
        for etf, data in grouped.items():
            invested = data["invested"]
            units = data["units"]
            avg_price = invested / units if units > 0 else 0.0

            total_invested += invested

            # ---------- Price unavailable ----------
            if etf not in prices:
                logger.warning("Live price missing for %s", etf)

                positions.append(
                    {
                        "etf_symbol": etf,
                        "units": round(units, 4),
                        "avg_buy_price": round(avg_price, 2),
                        "current_price": None,
                        "invested_amount": round(invested, 2),
                        "current_value": None,
                        "pnl": None,
                        "pnl_pct": None,
                        "price_status": "UNAVAILABLE",
                    }
                )
                continue

            # ---------- Price available ----------
            curr_price = prices[etf]
            value = units * curr_price
            pnl = value - invested

            current_value += value

            positions.append(
                {
                    "etf_symbol": etf,
                    "units": round(units, 4),
                    "avg_buy_price": round(avg_price, 2),
                    "current_price": round(curr_price, 2),
                    "invested_amount": round(invested, 2),
                    "current_value": round(value, 2),
                    "pnl": round(pnl, 2),
                    "pnl_pct": round((pnl / invested) * 100, 2)
                    if invested > 0
                    else 0.0,
                    "price_status": "LIVE",
                }
            )

        total_pnl = current_value - total_invested

        logger.info(
            "✅ Portfolio snapshot ready | invested=%.2f value=%.2f pnl=%.2f",
            total_invested,
            current_value,
            total_pnl,
        )

        # ------------------------------------------------------------
        # Final response
        # ------------------------------------------------------------
        return {
            "positions": positions,
            "summary": {
                "base_invested": round(base_invested, 2),
                "tactical_invested": round(tactical_invested, 2),
                "total_invested": round(total_invested, 2),
                "current_value": round(current_value, 2),
                "pnl": round(total_pnl, 2),
                "pnl_pct": round((total_pnl / total_invested) * 100, 2)
                if total_invested > 0
                else 0.0,
            },
        }
