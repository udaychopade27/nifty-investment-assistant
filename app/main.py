import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app.core.constants import IST
from app.core.logging import setup_logging

from app.db.init_db import init_db
from app.db.db import SessionLocal

from app.utils.market_calendar import fetch_and_store_trading_holidays
from app.market.nifty_service import NiftyService

from app.scheduler.daily_job import run_daily_decision_job
from app.scheduler.monthly_job import run_monthly_closure_job

from app.api.capital import router as capital_router
from app.api.decision import router as decision_router
from app.api.telegram_debug import router as telegram_debug_router

from app.bot.telegram_bot import telegram_bot

# -------------------------------------------------
# Logging
# -------------------------------------------------
setup_logging()
logger = logging.getLogger(__name__)

# -------------------------------------------------
# Scheduler
# -------------------------------------------------
scheduler = BackgroundScheduler(timezone=IST)

# -------------------------------------------------
# Lifespan (CORRECT WAY)
# -------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Application startup initiated")

    # -------------------------------
    # DB init
    # -------------------------------
    init_db()
    db = SessionLocal()
    try:
        fetch_and_store_trading_holidays(db)
    finally:
        db.close()

    # -------------------------------
    # Scheduler jobs
    # -------------------------------
    scheduler.add_job(
        run_daily_decision_job,
        CronTrigger(hour=15, minute=16),
        id="daily_decision_job",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )

    scheduler.add_job(
        run_monthly_closure_job,
        CronTrigger(hour=15, minute=16),
        id="monthly_closure_job",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )

    scheduler.start()
    logger.info("⏱ Scheduler started (3:16 PM IST daily)")

    # -------------------------------
    # Telegram bot (BACKGROUND TASK)
    # -------------------------------
    if telegram_bot.token:
        asyncio.create_task(telegram_bot.start())
        logger.info("🤖 Telegram bot startup task created")

    logger.info("✅ Application startup complete")
    yield

    # -------------------------------
    # Shutdown
    # -------------------------------
    logger.info("🛑 Shutting down application")
    scheduler.shutdown(wait=False)
    await telegram_bot.stop()
    logger.info("🤖 Telegram bot stopped")

# -------------------------------------------------
# App
# -------------------------------------------------
app = FastAPI(
    title="Nifty Investment Assistant",
    lifespan=lifespan,
)

# -------------------------------------------------
# Routers
# -------------------------------------------------
app.include_router(capital_router)
app.include_router(decision_router)
app.include_router(telegram_debug_router)

# -------------------------------------------------
# APIs
# -------------------------------------------------
@app.get("/market/nifty")
def get_nifty_data():
    service = NiftyService()
    return service.get_today_close()


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "telegram_bot": telegram_bot.is_running,
    }
