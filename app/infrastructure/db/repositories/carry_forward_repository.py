"""
Carry Forward Log Repository
CRUD operations for carry-forward audit logs
"""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import date
from decimal import Decimal
from typing import Optional

from app.infrastructure.db.models import CarryForwardLogModel


class CarryForwardLogRepository:
    """Repository for CarryForwardLog"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        month: date,
        previous_month: date,
        base_inflow: Decimal,
        tactical_inflow: Decimal,
        total_inflow: Decimal,
        base_carried_forward: Decimal,
        tactical_carried_forward: Decimal,
        total_monthly_capital: Decimal
    ) -> int:
        existing = await self.get_for_month(month)
        if existing:
            return existing.id
        model = CarryForwardLogModel(
            month=month,
            previous_month=previous_month,
            base_inflow=base_inflow,
            tactical_inflow=tactical_inflow,
            total_inflow=total_inflow,
            base_carried_forward=base_carried_forward,
            tactical_carried_forward=tactical_carried_forward,
            total_monthly_capital=total_monthly_capital
        )
        self.session.add(model)
        await self.session.flush()
        return model.id

    async def get_for_month(self, month: date) -> Optional[CarryForwardLogModel]:
        result = await self.session.execute(
            select(CarryForwardLogModel).where(CarryForwardLogModel.month == month)
        )
        return result.scalar_one_or_none()
