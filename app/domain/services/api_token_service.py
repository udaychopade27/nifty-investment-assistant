"""
API Token Service
Encapsulates token storage/retrieval + freshness logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, Callable, Awaitable

from app.infrastructure.db.repositories.api_token_repository import ApiTokenRepository
from app.utils.time import now_ist_naive, to_ist_iso_db
from app.infrastructure.db.database import async_session_factory


@dataclass(frozen=True)
class TokenStatus:
    provider: str
    has_token: bool
    last_updated: Optional[str]
    needs_refresh: bool
    masked_token: Optional[str]


def _mask_token(token: str) -> str:
    if not token:
        return ""
    if len(token) <= 8:
        return token[:2] + "..." + token[-2:]
    return token[:4] + "..." + token[-4:]


class ApiTokenService:
    """Token service using async DB sessions"""

    def __init__(self, provider: str):
        self.provider = provider

    async def get_token(self) -> Optional[str]:
        async with async_session_factory() as session:
            repo = ApiTokenRepository(session)
            record = await repo.get_by_provider(self.provider)
            return record.token if record else None

    async def set_token(self, token: str, updated_by: Optional[str] = None) -> TokenStatus:
        async with async_session_factory() as session:
            repo = ApiTokenRepository(session)
            record = await repo.upsert(self.provider, token, updated_by=updated_by)
            await session.commit()
            return self._status_from_record(record)

    async def get_status(self) -> TokenStatus:
        async with async_session_factory() as session:
            repo = ApiTokenRepository(session)
            record = await repo.get_by_provider(self.provider)
            return self._status_from_record(record)

    def _status_from_record(self, record) -> TokenStatus:
        if not record:
            return TokenStatus(
                provider=self.provider,
                has_token=False,
                last_updated=None,
                needs_refresh=True,
                masked_token=None,
            )

        last_updated = record.updated_at.date()
        today = now_ist_naive().date()
        needs_refresh = last_updated != today
        return TokenStatus(
            provider=self.provider,
            has_token=True,
            last_updated=to_ist_iso_db(record.updated_at),
            needs_refresh=needs_refresh,
            masked_token=_mask_token(record.token),
        )
