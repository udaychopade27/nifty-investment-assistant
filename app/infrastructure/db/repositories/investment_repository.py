"""
Executed Investment Repository
CRUD operations for executed investments (audit records)
"""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from datetime import date
from decimal import Decimal
from typing import Optional, List

from app.infrastructure.db.models import ExecutedInvestmentModel
from app.domain.models import ExecutedInvestment


class ExecutedInvestmentRepository:
    """Repository for ExecutedInvestment"""
    
    def __init__(self, session: AsyncSession):
        """Initialize with database session"""
        self.session = session
    
    async def create(
        self,
        executed_investment: ExecutedInvestment,
        etf_decision_id: int,
        capital_bucket: str
    ) -> int:
        """
        Create new executed investment record
        
        Args:
            executed_investment: ExecutedInvestment domain object
            etf_decision_id: ID of ETF decision
            capital_bucket: Capital bucket (base/tactical/extra)
        
        Returns:
            ID of created record
        """
        model = ExecutedInvestmentModel(
            etf_decision_id=etf_decision_id,
            etf_symbol=executed_investment.etf_symbol,
            units=executed_investment.units,
            executed_price=executed_investment.executed_price,
            total_amount=executed_investment.total_amount,
            slippage_pct=executed_investment.slippage_pct,
            capital_bucket=capital_bucket,
            executed_at=executed_investment.executed_at,
            execution_notes=executed_investment.execution_notes
        )
        
        self.session.add(model)
        await self.session.flush()
        
        return model.id
    
    async def get_by_etf_decision(
        self,
        etf_decision_id: int
    ) -> Optional[ExecutedInvestment]:
        """
        Get execution for an ETF decision
        
        Args:
            etf_decision_id: ETF decision ID
        
        Returns:
            ExecutedInvestment or None
        """
        result = await self.session.execute(
            select(ExecutedInvestmentModel)
            .where(ExecutedInvestmentModel.etf_decision_id == etf_decision_id)
        )
        model = result.scalar_one_or_none()
        
        return self._to_domain(model) if model else None
    
    async def get_total_base_deployed(self, month: date) -> Decimal:
        """
        Get total base capital deployed in a month
        
        Args:
            month: First day of month
        
        Returns:
            Total deployed amount
        """
        # Get last day of month
        if month.month == 12:
            next_month = date(month.year + 1, 1, 1)
        else:
            next_month = date(month.year, month.month + 1, 1)
        
        result = await self.session.execute(
            select(func.coalesce(func.sum(ExecutedInvestmentModel.total_amount), 0))
            .where(
                ExecutedInvestmentModel.capital_bucket == 'base',
                ExecutedInvestmentModel.executed_at >= month,
                ExecutedInvestmentModel.executed_at < next_month
            )
        )
        
        total = result.scalar()
        return Decimal(str(total)) if total else Decimal('0')
    
    async def get_total_tactical_deployed(self, month: date) -> Decimal:
        """
        Get total tactical capital deployed in a month
        
        Args:
            month: First day of month
        
        Returns:
            Total deployed amount
        """
        # Get last day of month
        if month.month == 12:
            next_month = date(month.year + 1, 1, 1)
        else:
            next_month = date(month.year, month.month + 1, 1)
        
        result = await self.session.execute(
            select(func.coalesce(func.sum(ExecutedInvestmentModel.total_amount), 0))
            .where(
                ExecutedInvestmentModel.capital_bucket == 'tactical',
                ExecutedInvestmentModel.executed_at >= month,
                ExecutedInvestmentModel.executed_at < next_month
            )
        )
        
        total = result.scalar()
        return Decimal(str(total)) if total else Decimal('0')
    
    async def get_total_extra_deployed(self, month: date) -> Decimal:
        """
        Get total extra capital deployed in a month
        
        Args:
            month: First day of month
        
        Returns:
            Total deployed amount
        """
        # Get last day of month
        if month.month == 12:
            next_month = date(month.year + 1, 1, 1)
        else:
            next_month = date(month.year, month.month + 1, 1)
        
        result = await self.session.execute(
            select(func.coalesce(func.sum(ExecutedInvestmentModel.total_amount), 0))
            .where(
                ExecutedInvestmentModel.capital_bucket == 'extra',
                ExecutedInvestmentModel.executed_at >= month,
                ExecutedInvestmentModel.executed_at < next_month
            )
        )
        
        total = result.scalar()
        return Decimal(str(total)) if total else Decimal('0')
    
    async def get_all_for_month(self, month: date) -> List[ExecutedInvestment]:
        """
        Get all executions for a month
        
        Args:
            month: First day of month
        
        Returns:
            List of ExecutedInvestments
        """
        # Get last day of month
        if month.month == 12:
            next_month = date(month.year + 1, 1, 1)
        else:
            next_month = date(month.year, month.month + 1, 1)
        
        result = await self.session.execute(
            select(ExecutedInvestmentModel)
            .where(
                ExecutedInvestmentModel.executed_at >= month,
                ExecutedInvestmentModel.executed_at < next_month
            )
            .order_by(ExecutedInvestmentModel.executed_at)
        )
        models = result.scalars().all()
        
        return [self._to_domain(m) for m in models]
    
    async def get_holdings_summary(self) -> List[dict]:
        """
        Get holdings summary (ETF-wise total units and invested amount)
        
        Returns:
            List of holding summaries
        """
        result = await self.session.execute(
            select(
                ExecutedInvestmentModel.etf_symbol,
                func.sum(ExecutedInvestmentModel.units).label('total_units'),
                func.sum(ExecutedInvestmentModel.total_amount).label('total_invested'),
                func.avg(ExecutedInvestmentModel.executed_price).label('avg_price')
            )
            .group_by(ExecutedInvestmentModel.etf_symbol)
        )
        
        holdings = []
        for row in result:
            holdings.append({
                'etf_symbol': row.etf_symbol,
                'total_units': int(row.total_units),
                'total_invested': Decimal(str(row.total_invested)),
                'average_price': Decimal(str(row.avg_price)).quantize(Decimal('0.01'))
            })
        
        return holdings
    
    @staticmethod
    def _to_domain(model: Optional[ExecutedInvestmentModel]) -> Optional[ExecutedInvestment]:
        """Convert database model to domain entity"""
        if model is None:
            return None
        
        return ExecutedInvestment(
            etf_decision_id=model.etf_decision_id,
            etf_symbol=model.etf_symbol,
            units=model.units,
            executed_price=model.executed_price,
            total_amount=model.total_amount,
            slippage_pct=model.slippage_pct,
            executed_at=model.executed_at,
            execution_notes=model.execution_notes
        )


class ExtraCapitalRepository:
    """Repository for extra capital injections"""
    
    def __init__(self, session: AsyncSession):
        """Initialize with database session"""
        self.session = session
    
    async def get_total_for_month(self, month: date) -> Decimal:
        """
        Get total extra capital injected in a month
        
        Args:
            month: First day of month
        
        Returns:
            Total injected amount
        """
        from app.infrastructure.db.models import ExtraCapitalInjectionModel
        
        # Get last day of month
        if month.month == 12:
            next_month = date(month.year + 1, 1, 1)
        else:
            next_month = date(month.year, month.month + 1, 1)
        
        result = await self.session.execute(
            select(func.coalesce(func.sum(ExtraCapitalInjectionModel.amount), 0))
            .where(
                ExtraCapitalInjectionModel.month == month
            )
        )
        
        total = result.scalar()
        return Decimal(str(total)) if total else Decimal('0')
