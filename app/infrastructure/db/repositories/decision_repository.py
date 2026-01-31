"""
Decision Repositories
CRUD operations for daily decisions and ETF decisions
"""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from datetime import date
from typing import Optional, List

from app.infrastructure.db.models import (
    DailyDecisionModel,
    ETFDecisionModel,
    DecisionTypeEnum,
    ETFStatusEnum
)
from app.domain.models import DailyDecision, ETFDecision, DecisionType, ETFStatus


class DailyDecisionRepository:
    """Repository for DailyDecision"""
    
    def __init__(self, session: AsyncSession):
        """Initialize with database session"""
        self.session = session
    
    async def create(
        self,
        daily_decision: DailyDecision,
        monthly_config_id: int
    ) -> int:
        """
        Create new daily decision
        
        Args:
            daily_decision: DailyDecision domain object
            monthly_config_id: ID of monthly config
        
        Returns:
            ID of created decision
        """
        model = DailyDecisionModel(
            date=daily_decision.date,
            monthly_config_id=monthly_config_id,
            decision_type=DecisionTypeEnum(daily_decision.decision_type.value),
            nifty_change_pct=daily_decision.nifty_change_pct,
            suggested_total_amount=daily_decision.suggested_total_amount,
            actual_investable_amount=daily_decision.actual_investable_amount,
            unused_amount=daily_decision.unused_amount,
            remaining_base_capital=daily_decision.remaining_base_capital,
            remaining_tactical_capital=daily_decision.remaining_tactical_capital,
            explanation=daily_decision.explanation,
            strategy_version=daily_decision.strategy_version,
            created_at=daily_decision.created_at
        )
        
        self.session.add(model)
        await self.session.flush()
        
        return model.id
    
    async def get_for_date(self, decision_date: date) -> Optional[DailyDecision]:
        """
        Get decision for a specific date
        
        Args:
            decision_date: Date to fetch
        
        Returns:
            DailyDecision or None
        """
        result = await self.session.execute(
            select(DailyDecisionModel)
            .where(DailyDecisionModel.date == decision_date)
            .options(selectinload(DailyDecisionModel.etf_decisions))
        )
        model = result.scalar_one_or_none()
        
        return self._to_domain(model) if model else None
    
    async def get_today(self) -> Optional[DailyDecision]:
        """
        Get today's decision
        
        Returns:
            DailyDecision for today or None
        """
        return await self.get_for_date(date.today())
    
    async def get_recent(self, limit: int = 30) -> List[DailyDecision]:
        """
        Get recent decisions
        
        Args:
            limit: Number of decisions to fetch
        
        Returns:
            List of DailyDecisions
        """
        result = await self.session.execute(
            select(DailyDecisionModel)
            .order_by(DailyDecisionModel.date.desc())
            .limit(limit)
        )
        models = result.scalars().all()
        
        return [self._to_domain(m) for m in models]
    
    @staticmethod
    def _to_domain(model: Optional[DailyDecisionModel]) -> Optional[DailyDecision]:
        """Convert database model to domain entity"""
        if model is None:
            return None
        
        return DailyDecision(
            date=model.date,
            decision_type=DecisionType(model.decision_type.value),
            nifty_change_pct=model.nifty_change_pct,
            suggested_total_amount=model.suggested_total_amount,
            actual_investable_amount=model.actual_investable_amount,
            unused_amount=model.unused_amount,
            remaining_base_capital=model.remaining_base_capital,
            remaining_tactical_capital=model.remaining_tactical_capital,
            explanation=model.explanation,
            strategy_version=model.strategy_version,
            created_at=model.created_at
        )


class ETFDecisionRepository:
    """Repository for ETFDecision"""
    
    def __init__(self, session: AsyncSession):
        """Initialize with database session"""
        self.session = session
    
    async def create_batch(
        self,
        etf_decisions: List[ETFDecision],
        daily_decision_id: int
    ) -> List[int]:
        """
        Create multiple ETF decisions
        
        Args:
            etf_decisions: List of ETFDecision domain objects
            daily_decision_id: ID of parent daily decision
        
        Returns:
            List of created IDs
        """
        ids = []
        
        for etf_decision in etf_decisions:
            model = ETFDecisionModel(
                daily_decision_id=daily_decision_id,
                etf_symbol=etf_decision.etf_symbol,
                ltp=etf_decision.ltp,
                effective_price=etf_decision.effective_price,
                units=etf_decision.units,
                actual_amount=etf_decision.actual_amount,
                status=ETFStatusEnum(etf_decision.status.value),
                reason=etf_decision.reason,
                created_at=etf_decision.created_at
            )
            
            self.session.add(model)
            await self.session.flush()
            ids.append(model.id)
        
        return ids
    
    async def get_for_daily_decision(
        self,
        daily_decision_id: int
    ) -> List[ETFDecision]:
        """
        Get all ETF decisions for a daily decision
        
        Args:
            daily_decision_id: ID of daily decision
        
        Returns:
            List of ETFDecisions
        """
        result = await self.session.execute(
            select(ETFDecisionModel)
            .where(ETFDecisionModel.daily_decision_id == daily_decision_id)
        )
        models = result.scalars().all()
        
        return [self._to_domain(m) for m in models]
    
    async def get_by_id(self, etf_decision_id: int) -> Optional[ETFDecision]:
        """
        Get ETF decision by ID
        
        Args:
            etf_decision_id: ETF decision ID
        
        Returns:
            ETFDecision or None
        """
        result = await self.session.execute(
            select(ETFDecisionModel)
            .where(ETFDecisionModel.id == etf_decision_id)
        )
        model = result.scalar_one_or_none()
        
        return self._to_domain(model) if model else None
    
    @staticmethod
    def _to_domain(model: Optional[ETFDecisionModel]) -> Optional[ETFDecision]:
        """Convert database model to domain entity"""
        if model is None:
            return None
        
        return ETFDecision(
            daily_decision_id=model.daily_decision_id,
            etf_symbol=model.etf_symbol,
            ltp=model.ltp,
            effective_price=model.effective_price,
            units=model.units,
            actual_amount=model.actual_amount,
            status=ETFStatus(model.status.value),
            reason=model.reason,
            created_at=model.created_at
        )
