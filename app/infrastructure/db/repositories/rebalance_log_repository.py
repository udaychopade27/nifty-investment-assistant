"""
Annual Rebalance Log Repository
Audit log for yearly rebalance execution
"""

from datetime import date
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.db.models import RebalanceLogModel


class RebalanceLogRepository:
    """Repository for rebalance audit logs"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_fiscal_year(self, fiscal_year: str) -> Optional[RebalanceLogModel]:
        result = await self.session.execute(
            select(RebalanceLogModel).where(RebalanceLogModel.fiscal_year == fiscal_year)
        )
        return result.scalars().first()

    async def create(self, fiscal_year: str, rebalance_date: date, payload: dict) -> int:
        model = RebalanceLogModel(
            fiscal_year=fiscal_year,
            rebalance_date=rebalance_date,
            payload=payload,
        )
        self.session.add(model)
        await self.session.flush()
        return model.id
