"""
Decision Service
High-level service for decision generation and execution
"""

from datetime import date
from decimal import Decimal
from typing import Tuple, List, Optional
import logging

from app.domain.services.decision_engine import DecisionEngine
from app.domain.services.market_context_engine import MarketContextEngine
from app.infrastructure.market_data.yfinance_provider import YFinanceProvider
from app.infrastructure.calendar.nse_calendar import NSECalendar
from app.infrastructure.db.repositories.monthly_config_repository import MonthlyConfigRepository
from app.infrastructure.db.repositories.decision_repository import (
    DailyDecisionRepository,
    ETFDecisionRepository
)
from app.domain.models import DailyDecision, ETFDecision

logger = logging.getLogger(__name__)


class DecisionService:
    """
    Decision Service
    Orchestrates decision generation from market data to persistence
    """
    
    def __init__(
        self,
        decision_engine: DecisionEngine,
        market_context_engine: MarketContextEngine,
        market_data_provider: YFinanceProvider,
        nse_calendar: NSECalendar,
        monthly_config_repo: MonthlyConfigRepository,
        daily_decision_repo: DailyDecisionRepository,
        etf_decision_repo: ETFDecisionRepository,
        etf_symbols: List[str]
    ):
        """Initialize decision service with all dependencies"""
        self.decision_engine = decision_engine
        self.market_context_engine = market_context_engine
        self.market_data_provider = market_data_provider
        self.nse_calendar = nse_calendar
        self.monthly_config_repo = monthly_config_repo
        self.daily_decision_repo = daily_decision_repo
        self.etf_decision_repo = etf_decision_repo
        self.etf_symbols = etf_symbols
    
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
            logger.info(f"Decision already exists for {decision_date}")
            # Get ETF decisions
            etf_decisions = await self.etf_decision_repo.get_for_daily_decision(existing.market_context_id)
            return existing, etf_decisions
        
        logger.info(f"Generating decision for {decision_date}")
        
        # Step 1: Get monthly config
        month = date(decision_date.year, decision_date.month, 1)
        monthly_config = await self.monthly_config_repo.get_for_month(month)
        
        if not monthly_config:
            raise ValueError(f"No monthly config found for {month}")
        
        # Step 2: Fetch market data
        logger.info("Fetching NIFTY data...")
        nifty_data = await self.market_data_provider.get_nifty_data(decision_date)
        
        if not nifty_data:
            raise ValueError(f"Could not fetch NIFTY data for {decision_date}")
        
        # Get previous 3-day closes
        logger.info("Fetching historical NIFTY data...")
        last_3_closes = await self.market_data_provider.get_last_n_closes(
            'NIFTY50',
            n=3,
            end_date=self.nse_calendar.get_previous_trading_day(decision_date)
        )
        
        # Get VIX
        vix = await self.market_data_provider.get_india_vix(decision_date)
        
        # Step 3: Calculate market context
        logger.info("Calculating market context...")
        market_context = self.market_context_engine.calculate_context(
            calc_date=decision_date,
            nifty_close=nifty_data['close'],
            nifty_previous_close=nifty_data['previous_close'],
            last_3_day_closes=last_3_closes,
            india_vix=vix
        )
        
        logger.info(f"Market context: {market_context.stress_level}, Change: {market_context.daily_change_pct}%")
        
        # Step 4: Fetch current ETF prices
        logger.info("Fetching ETF prices...")
        current_prices = await self.market_data_provider.get_prices_for_date(
            self.etf_symbols,
            decision_date
        )
        
        logger.info(f"Fetched prices for {len(current_prices)} ETFs")
        
        # Step 5: Generate decision
        logger.info("Generating decision...")
        daily_decision, etf_decisions = self.decision_engine.generate_decision(
            decision_date=decision_date,
            market_context=market_context,
            monthly_config=monthly_config,
            current_prices=current_prices
        )
        
        logger.info(f"Decision: {daily_decision.decision_type}, Amount: ₹{daily_decision.actual_investable_amount}")
        
        # Step 6: Persist decision
        logger.info("Persisting decision to database...")
        daily_decision_id = await self.daily_decision_repo.create(
            daily_decision,
            monthly_config_id=monthly_config.month.month  # Simplified - should fetch actual ID
        )
        
        # Persist ETF decisions
        if etf_decisions:
            await self.etf_decision_repo.create_batch(
                etf_decisions,
                daily_decision_id
            )
        
        logger.info(f"Decision persisted with ID: {daily_decision_id}")
        
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
        
        # Get ETF decisions (simplified - need proper ID handling)
        etf_decisions = []
        
        return daily_decision, etf_decisions
