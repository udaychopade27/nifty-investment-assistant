import logging
from datetime import date

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import MonthlyConfig, DailyDecision
from app.market.nifty_service import NiftyService

logger = logging.getLogger(__name__)
settings = get_settings()


def decide_investment_for_today(db: Session) -> dict:
    today = date.today()
    month_start = date(today.year, today.month, 1)

    config = (
        db.query(MonthlyConfig)
        .filter(MonthlyConfig.month == month_start)
        .first()
    )

    if not config:
        raise RuntimeError("Monthly capital not set for current month")

    invested_so_far = (
        db.query(DailyDecision)
        .filter(DailyDecision.month == month_start)
        .with_entities(DailyDecision.suggested_amount)
        .all()
    )
    invested_so_far = sum(x[0] for x in invested_so_far)
    remaining_capital = config.monthly_capital - invested_so_far

    nifty_data = NiftyService().get_today_close()
    nifty_change = nifty_data["change_percent"]

    invest_amount = 0
    reason = (
    "Market stable — capital preserved for future dips. "
    "Investments trigger only when NIFTY falls ≥ 1%."
    )

    if remaining_capital <= 0:
        return {
            "action": "NO_ACTION",
            "nifty_change": nifty_change,
            "suggested_amount": 0,
            "decision_reason": "Monthly capital fully deployed",
            "remaining_capital": 0,
        }

    if nifty_change <= -5:
        invest_amount = remaining_capital
        reason = "Deep market fall (>=5%)"
    elif nifty_change <= -3:
        invest_amount = min(settings.DIP_3_PERCENT_AMOUNT, remaining_capital)
        reason = "Market fall >=3%"
    elif nifty_change <= -2:
        invest_amount = min(settings.DIP_2_PERCENT_AMOUNT, remaining_capital)
        reason = "Market fall >=2%"
    elif nifty_change <= -1:
        invest_amount = min(settings.DIP_1_PERCENT_AMOUNT, remaining_capital)
        reason = "Market fall >=1%"

    if invest_amount == 0:
        return {
            "action": "NO_ACTION",
            "nifty_change": nifty_change,
            "suggested_amount": 0,
            "decision_reason": reason,
            "remaining_capital": remaining_capital,
        }

    decision = DailyDecision(
        decision_date=today,
        month=month_start,
        nifty_change=nifty_change,
        suggested_amount=invest_amount,
        decision_reason=reason,
        remaining_capital=remaining_capital - invest_amount,
    )

    db.add(decision)
    db.commit()

    return {
        "action": "INVEST",
        "nifty_change": nifty_change,
        "suggested_amount": invest_amount,
        "decision_reason": reason,
        "remaining_capital": remaining_capital - invest_amount,
    }

