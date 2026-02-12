"""
Decision Service
High-level service for decision generation and execution

✅ FIXED: Fetches CapitalState via CapitalEngine before calling DecisionEngine
✅ FIXED: Uses monthly_config.id (not month.month)
✅ FIXED: Sets ETFDecision.daily_decision_id correctly
"""

from datetime import date
from decimal import Decimal
from typing import Tuple, List, Optional
import logging

from app.domain.services.decision_engine import DecisionEngine
from app.domain.services.market_context_engine import MarketContextEngine
from app.domain.services.capital_engine import CapitalEngine
from app.infrastructure.market_data.types import MarketDataProvider
from app.infrastructure.calendar.nse_calendar import NSECalendar
from app.infrastructure.db.repositories.monthly_config_repository import MonthlyConfigRepository
from app.infrastructure.db.repositories.decision_repository import (
    DailyDecisionRepository,
    ETFDecisionRepository
)
from app.domain.models import DailyDecision, ETFDecision, CapitalState

logger = logging.getLogger(__name__)


class DecisionService:
    """
    Decision Service
    Orchestrates decision generation from market data to persistence
    
    ✅ FIXED: Now receives CapitalEngine and uses it to fetch state
    """
    
    def __init__(
        self,
        decision_engine: DecisionEngine,
        market_context_engine: MarketContextEngine,
        capital_engine: CapitalEngine,  # ✅ ADDED: CapitalEngine
        market_data_provider: MarketDataProvider,
        nse_calendar: NSECalendar,
        monthly_config_repo: MonthlyConfigRepository,
        daily_decision_repo: DailyDecisionRepository,
        etf_decision_repo: ETFDecisionRepository,
        etf_symbols: List[str],
        etf_index_map: Optional[dict[str, str]] = None
    ):
        """Initialize decision service with all dependencies"""
        self.decision_engine = decision_engine
        self.market_context_engine = market_context_engine
        self.capital_engine = capital_engine  # ✅ Store it
        self.market_data_provider = market_data_provider
        self.nse_calendar = nse_calendar
        self.monthly_config_repo = monthly_config_repo
        self.daily_decision_repo = daily_decision_repo
        self.etf_decision_repo = etf_decision_repo
        self.etf_symbols = etf_symbols
        self.etf_index_map = etf_index_map or {}
    
    async def generate_today_decision(self) -> Tuple[DailyDecision, List[ETFDecision]]:
        """
        Generate decision for today
        
        Returns:
            Tuple of (DailyDecision, List of ETFDecisions)
        
        Raises:
            ValueError: If not a trading day or config missing
        """
        return await self.generate_decision_for_date(date.today())
    
    async def generate_decision_for_date(
        self,
        decision_date: date
    ) -> Tuple[DailyDecision, List[ETFDecision]]:
        """
        Generate decision for a specific date
        
        Args:
            decision_date: Date to generate decision for
        
        Returns:
            Tuple of (DailyDecision, List of ETFDecisions)
        
        Raises:
            ValueError: If not a trading day or config missing
        """
        # Validate trading day
        if not self.nse_calendar.is_trading_day(decision_date):
            raise ValueError(f"{decision_date} is not a trading day")
        
        # Check if decision already exists
        existing = await self.daily_decision_repo.get_for_date(decision_date)
        if existing:
            logger.info(f"✅ Decision already exists for {decision_date}")
            # Get ETF decisions
            etf_decisions = await self.etf_decision_repo.get_for_daily_decision(existing.id)
            return existing, etf_decisions
        
        logger.info(f"📊 Generating decision for {decision_date}")
        
        # Step 1: Get monthly config
        month = date(decision_date.year, decision_date.month, 1)
        monthly_config = await self.monthly_config_repo.get_for_month(month)
        
        if not monthly_config:
            raise ValueError(f"No monthly config found for {month}")
        
        logger.info(f"💰 Monthly config: ₹{monthly_config.monthly_capital:,.2f}")
        
        # Step 2: ✅ Get capital state via CapitalEngine (this is where we use it)
        capital_state = await self.capital_engine.get_capital_state(monthly_config.month)
        
        logger.info(f"📊 Capital state - Base: ₹{capital_state.base_remaining:,.2f}, Tactical: ₹{capital_state.tactical_remaining:,.2f}")
        
        # Step 3: Fetch market data
        logger.info("📈 Fetching NIFTY data...")
        nifty_data = await self.market_data_provider.get_nifty_data(decision_date)
        
        if not nifty_data:
            raise ValueError(f"Could not fetch NIFTY data for {decision_date}")
        
        # Get previous 3-day closes
        logger.info("📊 Fetching historical NIFTY data...")
        last_3_closes = await self.market_data_provider.get_last_n_closes(
            'NIFTY50',
            n=3,
            end_date=self.nse_calendar.get_previous_trading_day(decision_date)
        )
        
        # Get VIX
        vix = await self.market_data_provider.get_india_vix(decision_date)
        
        # Step 4: Calculate market context
        logger.info("🧮 Calculating market context...")
        market_context = self.market_context_engine.calculate_context(
            calc_date=decision_date,
            nifty_close=nifty_data['close'],
            nifty_previous_close=nifty_data['previous_close'],
            last_3_day_closes=last_3_closes,
            india_vix=vix
        )
        
        logger.info(f"📉 Market: {market_context.stress_level}, Change: {market_context.daily_change_pct}%")
        
        # Step 5: Fetch current ETF prices
        logger.info("💹 Fetching ETF prices...")
        current_prices = await self.market_data_provider.get_prices_for_date(
            self.etf_symbols,
            decision_date
        )
        
        logger.info(f"✅ Fetched prices for {len(current_prices)} ETFs")
        
        # Step 6: Fetch underlying index changes (for tactical dip logic)
        index_changes_by_etf: dict[str, Decimal] = {}
        index_metrics_by_etf: dict[str, dict[str, Decimal]] = {}
        for symbol, index_name in self.etf_index_map.items():
            if not index_name:
                continue
            change = await self.market_data_provider.get_index_daily_change(index_name, decision_date)
            if change is not None:
                index_changes_by_etf[symbol] = change
                index_metrics_by_etf[symbol] = {
                    "daily_change_pct": change,
                    "data_quality": Decimal("1"),
                }
            else:
                index_metrics_by_etf[symbol] = {"data_quality": Decimal("0")}

            # Optional enrichments: 3-day trend and 20D distance from average.
            try:
                closes_20 = await self.market_data_provider.get_last_n_closes(
                    index_name,
                    n=20,
                    end_date=decision_date,
                )
                if closes_20:
                    latest = closes_20[-1]
                    avg_20 = (sum(closes_20) / Decimal(len(closes_20)))
                    if avg_20 > Decimal("0"):
                        ma_dist = ((latest - avg_20) / avg_20 * Decimal("100")).quantize(Decimal("0.01"))
                        index_metrics_by_etf[symbol]["ma20_distance_pct"] = ma_dist
                if len(closes_20) >= 3:
                    start_3 = closes_20[-3]
                    end_3 = closes_20[-1]
                    if start_3 > Decimal("0"):
                        three_day = ((end_3 - start_3) / start_3 * Decimal("100")).quantize(Decimal("0.01"))
                        index_metrics_by_etf[symbol]["three_day_change_pct"] = three_day
                if len(closes_20) >= 5:
                    start_5 = closes_20[-5]
                    end_5 = closes_20[-1]
                    if start_5 > Decimal("0"):
                        five_day = ((end_5 - start_5) / start_5 * Decimal("100")).quantize(Decimal("0.01"))
                        index_metrics_by_etf[symbol]["five_day_change_pct"] = five_day
            except Exception:
                # Non-fatal: provider may not support this index symbol path consistently.
                pass

        # Step 7: ✅ Generate decision (passing capital_state)
        logger.info("🎯 Generating decision...")
        daily_decision, etf_decisions = self.decision_engine.generate_decision(
            decision_date=decision_date,
            market_context=market_context,
            monthly_config=monthly_config,
            capital_state=capital_state,  # ✅ Passed in
            current_prices=current_prices,
            index_changes_by_etf=index_changes_by_etf,
            index_metrics_by_etf=index_metrics_by_etf,
            deploy_base_daily=False
        )
        
        logger.info(f"✅ Decision: {daily_decision.decision_type.value}, Amount: ₹{daily_decision.actual_investable_amount:,.2f}")
        
        # Step 7: ✅ Persist decision with CORRECT monthly_config_id
        logger.info("💾 Persisting decision to database...")
        daily_decision_id = await self.daily_decision_repo.create(
            daily_decision,
            monthly_config_id=monthly_config.id  # ✅ FIXED: Use .id, not .month.month
        )
        
        logger.info(f"✅ Daily decision persisted with ID: {daily_decision_id}")
        
        # Step 8: ✅ Persist ETF decisions with correct daily_decision_id
        if etf_decisions:
            await self.etf_decision_repo.create_batch(
                etf_decisions,
                daily_decision_id  # ✅ Use the returned ID
            )
            logger.info(f"✅ Persisted {len(etf_decisions)} ETF decisions")
        
        # Step 9: Update IDs in returned objects
        daily_decision.id = daily_decision_id
        for etf_decision in etf_decisions:
            etf_decision.daily_decision_id = daily_decision_id
        
        logger.info(f"✅ Decision generation complete for {decision_date}")
        
        return daily_decision, etf_decisions
    
    async def get_today_decision(self) -> Optional[Tuple[DailyDecision, List[ETFDecision]]]:
        """
        Get today's decision if it exists
        
        Returns:
            Tuple of (DailyDecision, ETFDecisions) or None
        """
        daily_decision = await self.daily_decision_repo.get_today()
        
        if not daily_decision:
            return None
        
        # Get ETF decisions
        etf_decisions = await self.etf_decision_repo.get_for_daily_decision(daily_decision.id)
        
        return daily_decision, etf_decisions
