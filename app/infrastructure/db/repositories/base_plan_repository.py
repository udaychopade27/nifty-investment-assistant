"""
Repository for Base Investment Plans
"""

from datetime import date
from decimal import Decimal
from typing import Optional, Dict, Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.db.models import BaseInvestmentPlanModel


class BaseInvestmentPlanRepository:
    """CRUD for base investment plans"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_for_month(self, month: date) -> Optional[BaseInvestmentPlanModel]:
        result = await self.session.execute(
            select(BaseInvestmentPlanModel).where(BaseInvestmentPlanModel.month == month)
        )
        return result.scalar_one_or_none()

    async def create(
        self,
        month: date,
        base_capital: Decimal,
        strategy_version: str,
        plan_json: Dict[str, Any],
    ) -> BaseInvestmentPlanModel:
        model = BaseInvestmentPlanModel(
            month=month,
            base_capital=base_capital,
            strategy_version=strategy_version,
            plan_json=plan_json,
        )
        self.session.add(model)
        await self.session.commit()
        await self.session.refresh(model)
        return model
