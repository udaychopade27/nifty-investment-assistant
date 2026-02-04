"""
Executed Sell Repository
CRUD operations for executed sells (audit records)
"""

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from decimal import Decimal
from typing import Optional, List
from datetime import date

from app.infrastructure.db.models import ExecutedSellModel


class ExecutedSellRepository:
    """Repository for executed sells"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        etf_symbol: str,
        units: int,
        sell_price: Decimal,
        total_amount: Decimal,
        realized_pnl: Decimal,
        capital_bucket: str,
        sold_at,
        sell_notes: Optional[str]
    ) -> int:
        model = ExecutedSellModel(
            etf_symbol=etf_symbol,
            units=units,
            sell_price=sell_price,
            total_amount=total_amount,
            realized_pnl=realized_pnl,
            capital_bucket=capital_bucket,
            sold_at=sold_at,
            sell_notes=sell_notes
        )
        self.session.add(model)
        await self.session.flush()
        return model.id

    async def get_sell_summary(self) -> List[dict]:
        """
        Get sell summary by ETF symbol.
        """
        result = await self.session.execute(
            select(
                ExecutedSellModel.etf_symbol,
                func.sum(ExecutedSellModel.units).label("total_units"),
                func.sum(ExecutedSellModel.total_amount).label("total_proceeds"),
                func.sum(ExecutedSellModel.realized_pnl).label("total_realized_pnl"),
            )
            .group_by(ExecutedSellModel.etf_symbol)
        )
        summary = []
        for row in result:
            summary.append({
                "etf_symbol": row.etf_symbol,
                "total_units": int(row.total_units or 0),
                "total_proceeds": Decimal(str(row.total_proceeds or 0)),
                "total_realized_pnl": Decimal(str(row.total_realized_pnl or 0)),
            })
        return summary

    async def get_total_realized_pnl(self) -> Decimal:
        result = await self.session.execute(
            select(func.coalesce(func.sum(ExecutedSellModel.realized_pnl), 0))
        )
        total = result.scalar()
        return Decimal(str(total)) if total else Decimal("0")

    async def get_recent(self, limit: int = 50) -> List[ExecutedSellModel]:
        result = await self.session.execute(
            select(ExecutedSellModel)
            .order_by(ExecutedSellModel.sold_at.desc())
            .limit(limit)
        )
        return result.scalars().all()

    async def get_total_units_sold(self, etf_symbol: str) -> int:
        result = await self.session.execute(
            select(func.coalesce(func.sum(ExecutedSellModel.units), 0))
            .where(ExecutedSellModel.etf_symbol == etf_symbol)
        )
        total = result.scalar()
        return int(total or 0)
