from datetime import date
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.db.session import get_db_session
from app.services.execution_service import ExecutionService

logger = logging.getLogger(__name__)

router = APIRouter()


class ExecutionRequest(BaseModel):
    execution_date: date = Field(
        ...,
        example="2026-01-14",
        description="Date on which the ETF was manually executed (YYYY-MM-DD)",
    )
    etf_symbol: str = Field(
        ...,
        example="NIFTYBEES",
        description="ETF symbol exactly as per supported ETF universe",
    )
    invested_amount: float = Field(
        ...,
        example=5000.0,
        description="Total amount invested in INR for this execution",
    )
    execution_price: float = Field(
        ...,
        example=245.75,
        description="Actual execution price per unit",
    )
    capital_type: str = Field(
        ...,
        example="BASE",
        description="Capital type: BASE or TACTICAL",
    )


@router.post(
    "/confirm",
    summary="Confirm a manually executed ETF investment",
)
def confirm_execution(payload: ExecutionRequest, db=Depends(get_db_session)):
    logger.info(
        "📥 Execution request received | %s | %s | ₹%.2f | %s",
        payload.execution_date,
        payload.etf_symbol,
        payload.invested_amount,
        payload.capital_type,
    )

    try:
        result = ExecutionService.confirm_execution(
            db=db,
            execution_date=payload.execution_date,
            etf_symbol=payload.etf_symbol,
            invested_amount=payload.invested_amount,
            execution_price=payload.execution_price,
            capital_type=payload.capital_type,
        )

        logger.info(
            "✅ Execution recorded | %s | %s | units=%.4f",
            payload.capital_type,
            payload.etf_symbol,
            result["units"],
        )
        return result

    except ValueError as e:
        # Strategy / validation blocks (expected)
        logger.warning("⚠️ Execution blocked: %s", str(e))
        raise HTTPException(status_code=409, detail=str(e))

    except Exception:
        # System failure (unexpected)
        logger.exception("❌ Execution failed due to system error")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while confirming execution",
        )
