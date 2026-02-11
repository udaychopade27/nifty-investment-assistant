"""Persistence for options monthly capital and audit events."""
from datetime import date as date_type
from decimal import Decimal
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.db.models import OptionsCapitalMonthModel, OptionsCapitalEventModel
from app.utils.time import now_ist_naive, to_ist_iso_db


class OptionsCapitalRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def upsert_month(
        self,
        month: date_type,
        monthly_capital: float,
        initialized: bool,
    ) -> int:
        result = await self.session.execute(
            select(OptionsCapitalMonthModel).where(OptionsCapitalMonthModel.month == month)
        )
        model = result.scalar_one_or_none()
        if model is None:
            model = OptionsCapitalMonthModel(
                month=month,
                monthly_capital=Decimal(str(monthly_capital)),
                initialized=bool(initialized),
                created_at=now_ist_naive(),
                updated_at=now_ist_naive(),
            )
            self.session.add(model)
            await self.session.flush()
            return model.id

        model.monthly_capital = Decimal(str(monthly_capital))
        model.initialized = bool(initialized)
        model.updated_at = now_ist_naive()
        await self.session.flush()
        return model.id

    async def add_event(
        self,
        month: date_type,
        event_type: str,
        amount: float,
        rollover_applied: float,
        previous_capital: Optional[float],
        new_capital: float,
        payload: Optional[Dict[str, Any]] = None,
    ) -> int:
        model = OptionsCapitalEventModel(
            month=month,
            event_type=str(event_type),
            amount=Decimal(str(amount)),
            rollover_applied=Decimal(str(rollover_applied)),
            previous_capital=Decimal(str(previous_capital)) if previous_capital is not None else None,
            new_capital=Decimal(str(new_capital)),
            payload=payload,
            created_at=now_ist_naive(),
        )
        self.session.add(model)
        await self.session.flush()
        return model.id

    async def get_month(self, month: date_type) -> Optional[Dict[str, Any]]:
        result = await self.session.execute(
            select(OptionsCapitalMonthModel).where(OptionsCapitalMonthModel.month == month)
        )
        model = result.scalar_one_or_none()
        if model is None:
            return None
        return {
            "id": model.id,
            "month": model.month.isoformat(),
            "monthly_capital": float(model.monthly_capital),
            "initialized": bool(model.initialized),
            "created_at": to_ist_iso_db(model.created_at) if model.created_at else None,
            "updated_at": to_ist_iso_db(model.updated_at) if model.updated_at else None,
        }

    async def get_events(self, month: Optional[date_type] = None, limit: int = 200) -> List[Dict[str, Any]]:
        q = select(OptionsCapitalEventModel)
        if month is not None:
            q = q.where(OptionsCapitalEventModel.month == month)
        q = q.order_by(OptionsCapitalEventModel.created_at.desc()).limit(limit)
        result = await self.session.execute(q)
        rows = result.scalars().all()
        return [
            {
                "id": r.id,
                "month": r.month.isoformat(),
                "event_type": r.event_type,
                "amount": float(r.amount),
                "rollover_applied": float(r.rollover_applied),
                "previous_capital": float(r.previous_capital) if r.previous_capital is not None else None,
                "new_capital": float(r.new_capital),
                "payload": r.payload,
                "created_at": to_ist_iso_db(r.created_at) if r.created_at else None,
            }
            for r in rows
        ]
