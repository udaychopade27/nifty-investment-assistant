"""
API Token Repository
Simple upsert + fetch for provider tokens.
"""

from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.db.models import ApiTokenModel
from app.utils.time import now_ist_naive


class ApiTokenRepository:
    """Repository for API tokens"""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_provider(self, provider: str) -> Optional[ApiTokenModel]:
        result = await self.session.execute(
            select(ApiTokenModel).where(ApiTokenModel.provider == provider)
        )
        return result.scalar_one_or_none()

    async def upsert(
        self,
        provider: str,
        token: str,
        updated_by: Optional[str] = None
    ) -> ApiTokenModel:
        existing = await self.get_by_provider(provider)
        if existing:
            existing.token = token
            existing.updated_by = updated_by
            existing.updated_at = now_ist_naive()
            return existing

        record = ApiTokenModel(
            provider=provider,
            token=token,
            updated_by=updated_by,
            updated_at=now_ist_naive(),
        )
        self.session.add(record)
        return record
