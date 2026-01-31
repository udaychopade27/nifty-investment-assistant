"""
Monthly Config Repository
CRUD operations for monthly capital configuration
"""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from datetime import date
from decimal import Decimal
from typing import Optional

from app.infrastructure.db.models import MonthlyConfigModel
from app.domain.models import MonthlyConfig


class MonthlyConfigRepository:
    """Repository for MonthlyConfig data access"""
    
    def __init__(self, session: AsyncSession):
        """Initialize with database session"""
        self.session = session
    
    async def create(
        self,
        month: date,
        monthly_capital: Decimal,
        base_capital: Decimal,
        tactical_capital: Decimal,
        trading_days: int,
        daily_tranche: Decimal,
        strategy_version: str
    ) -> MonthlyConfig:
        """
        Create new monthly configuration
        
        Args:
            month: First day of month
            monthly_capital: Total monthly capital
            base_capital: Base capital (60%)
            tactical_capital: Tactical capital (40%)
            trading_days: Number of trading days
            daily_tranche: Daily investment amount
            strategy_version: Strategy version
        
        Returns:
            Created MonthlyConfig
        """
        model = MonthlyConfigModel(
            month=month,
            monthly_capital=monthly_capital,
            base_capital=base_capital,
            tactical_capital=tactical_capital,
            trading_days=trading_days,
            daily_tranche=daily_tranche,
            strategy_version=strategy_version
        )
        
        self.session.add(model)
        await self.session.flush()
        
        return self._to_domain(model)
    
    async def get_for_month(self, month: date) -> Optional[MonthlyConfig]:
        """
        Get monthly config for a specific month
        
        Args:
            month: First day of month
        
        Returns:
            MonthlyConfig or None
        """
        result = await self.session.execute(
            select(MonthlyConfigModel).where(MonthlyConfigModel.month == month)
        )
        model = result.scalar_one_or_none()
        
        return self._to_domain(model) if model else None
    
    async def get_current(self) -> Optional[MonthlyConfig]:
        """
        Get current month's configuration
        
        Returns:
            MonthlyConfig for current month or None
        """
        today = date.today()
        current_month = date(today.year, today.month, 1)
        return await self.get_for_month(current_month)
    
    async def get_latest(self) -> Optional[MonthlyConfig]:
        """
        Get the most recent monthly config
        
        Returns:
            Latest MonthlyConfig or None
        """
        result = await self.session.execute(
            select(MonthlyConfigModel).order_by(MonthlyConfigModel.month.desc())
        )
        model = result.first()
        
        return self._to_domain(model[0]) if model else None
    
    @staticmethod
    def _to_domain(model: Optional[MonthlyConfigModel]) -> Optional[MonthlyConfig]:
        """Convert database model to domain entity"""
        if model is None:
            return None
        
        return MonthlyConfig(
            month=model.month,
            monthly_capital=model.monthly_capital,
            base_capital=model.base_capital,
            tactical_capital=model.tactical_capital,
            trading_days=model.trading_days,
            daily_tranche=model.daily_tranche,
            strategy_version=model.strategy_version,
            created_at=model.created_at
        )
