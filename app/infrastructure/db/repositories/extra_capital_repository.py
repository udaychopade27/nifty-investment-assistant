"""
Extra Capital Repository

Handles additional / injected capital beyond monthly allocation
(e.g. manual top-ups, bonuses, corrections)
"""

from ast import stmt
from calendar import month
from decimal import Decimal
from typing import Optional
from unittest import result
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.db.models import ExtraCapitalInjectionModel


class ExtraCapitalRepository:
    """
    Repository for extra capital injections
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_total_extra_capital_for_month(self, month) -> Decimal:
        """
        Get total extra capital injected for a given month

        Args:
            month: date object (YYYY-MM-01)

        Returns:
            Decimal total extra capital
        """
        result = await self.session.execute(
            select(func.coalesce(func.sum(ExtraCapitalInjectionModel.amount), 0))
            .where(ExtraCapitalInjectionModel.month == month)
        )

        return Decimal(result.scalar() or 0)

    async def create(
        self,
        *,
        month,
        amount: Decimal,
        reason: Optional[str] = None,
    ) -> ExtraCapitalInjectionModel:
        """
        Record extra capital injection
        """
        record = ExtraCapitalInjectionModel(
            month=month,
            amount=amount,
            reason=reason,
        )

        self.session.add(record)
        await self.session.commit()
        await self.session.refresh(record)

        return record
    async def get_total_for_month(self, month: str) -> Decimal:
        stmt = (
        select(func.coalesce(func.sum(ExtraCapitalInjectionModel.amount), 0))
        .where(ExtraCapitalInjectionModel.month == month)
        )
        result = await self.session.execute(stmt)
        return Decimal(result.scalar_one())

