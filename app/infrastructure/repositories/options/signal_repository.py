"""Persistence for options signals and executions."""
from datetime import datetime, date as date_type
from decimal import Decimal
from typing import Dict, Any, List

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.infrastructure.db.models import OptionsSignalModel
from app.utils.time import now_ist_naive


class OptionsSignalRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def save_signal(self, signal: Dict[str, Any]) -> int:
        ts_raw = signal.get("ts")
        try:
            signal_ts = datetime.fromisoformat(ts_raw) if ts_raw else now_ist_naive()
        except Exception:
            signal_ts = now_ist_naive()
        if getattr(signal_ts, "tzinfo", None) is not None:
            # Normalize to IST naive for DB storage
            from app.utils.time import IST
            signal_ts = signal_ts.astimezone(IST).replace(tzinfo=None)
        entry = Decimal(str(signal.get("entry", 0)))
        stop_loss = Decimal(str(signal.get("stop_loss", 0)))
        target = Decimal(str(signal.get("target", 0)))
        rr = Decimal(str(signal.get("rr", 0)))
        est_profit = Decimal(str(signal.get("estimated_profit_per_unit", 0)))

        model = OptionsSignalModel(
            date=signal_ts.date(),
            signal_ts=signal_ts,
            underlying=str(signal.get("symbol")),
            signal=str(signal.get("signal")),
            entry=entry,
            stop_loss=stop_loss,
            target=target,
            rr=rr,
            estimated_profit=est_profit,
            entry_source=str(signal.get("entry_source", "spot")),
            blocked=bool(signal.get("blocked", False)),
            reason=signal.get("reason"),
            payload=signal,
            created_at=now_ist_naive(),
        )
        self.session.add(model)
        await self.session.flush()
        return model.id

    async def get_by_date(self, signal_date: date_type, limit: int = 200) -> List[Dict[str, Any]]:
        result = await self.session.execute(
            select(OptionsSignalModel)
            .where(OptionsSignalModel.date == signal_date)
            .order_by(OptionsSignalModel.signal_ts.desc())
            .limit(limit)
        )
        rows = result.scalars().all()
        return [
            {
                "id": r.id,
                "date": r.date.isoformat(),
                "signal_ts": r.signal_ts.isoformat(),
                "underlying": r.underlying,
                "signal": r.signal,
                "entry": float(r.entry),
                "stop_loss": float(r.stop_loss),
                "target": float(r.target),
                "rr": float(r.rr),
                "estimated_profit": float(r.estimated_profit),
                "entry_source": r.entry_source,
                "blocked": r.blocked,
                "reason": r.reason,
                "payload": r.payload,
            }
            for r in rows
        ]
