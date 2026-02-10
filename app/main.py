"""
FastAPI Main Application with Scheduler and Telegram Bot
Complete orchestration of all services in one application
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path
import logging
import asyncio
import os
import time
from typing import AsyncGenerator
from decimal import Decimal
from sqlalchemy import text

from app.config import settings
from app.infrastructure.db.database import init_db, close_db
from app.domain.services.config_engine import ConfigEngine
from app.domain.services.market_context_engine import MarketContextEngine
from app.domain.services.allocation_engine import AllocationEngine
from app.domain.services.unit_calculation_engine import UnitCalculationEngine
from app.infrastructure.market_data.provider_factory import get_market_data_provider
from app.infrastructure.calendar.nse_calendar import NSECalendar
from app.utils.logging_redaction import install_redaction_filter
from app.realtime.runtime import RealtimeRuntime
from app.domain.options.runtime import OptionsRuntime

# Import scheduler and telegram
from app.scheduler.main import ETFScheduler
from app.telegram.bot import ETFTelegramBot

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
install_redaction_filter()

# Reduce noisy loggers in production
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Set process timezone to IST for logging
os.environ["TZ"] = "Asia/Kolkata"
if hasattr(time, "tzset"):
    time.tzset()


# Global instances
config_engine: ConfigEngine | None = None
market_context_engine: MarketContextEngine | None = None
allocation_engine: AllocationEngine | None = None
unit_calculation_engine: UnitCalculationEngine | None = None
market_data_provider = None
nse_calendar: NSECalendar | None = None

# Service instances
scheduler: ETFScheduler | None = None
telegram_bot: ETFTelegramBot | None = None
telegram_task: asyncio.Task | None = None
realtime_runtime: RealtimeRuntime | None = None
options_runtime: OptionsRuntime | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Application lifespan manager
    Handles startup and shutdown of all services
    """
    global config_engine, market_context_engine
    global allocation_engine, unit_calculation_engine
    global market_data_provider, nse_calendar
    global scheduler, telegram_bot, telegram_task
    global realtime_runtime
    global options_runtime
    
    # ===================
    # STARTUP
    # ===================
    logger.info("="*60)
    logger.info("🚀 Starting ETF Assistant - All Services")
    logger.info("="*60)
    
    # 1. Initialize database
    logger.info("\n📊 Step 1/5: Initializing database...")
    await init_db()
    logger.info("✅ Database initialized")
    
    # 2. Load configuration
    logger.info("\n⚙️  Step 2/5: Loading configuration...")
    config_dir = Path(__file__).parent.parent / "config"
    config_engine = ConfigEngine(config_dir)
    config_engine.load_all()
    logger.info("✅ Configuration loaded successfully")
    logger.info(f"   📊 ETF Universe: {len(config_engine.etf_universe.etfs)} ETFs")
    logger.info(f"   📈 Strategy Version: {config_engine.strategy_version}")
    
    # 3. Initialize infrastructure
    logger.info("\n🏗️  Step 3/5: Initializing infrastructure...")
    market_data_provider = get_market_data_provider(config_engine)
    nse_calendar = NSECalendar()
    logger.info("✅ Infrastructure initialized")
    
    # 4. Initialize domain engines
    logger.info("\n🔧 Step 4/5: Initializing domain engines...")
    market_context_engine = MarketContextEngine()
    
    etf_dict = {etf.symbol: etf for etf in config_engine.etf_universe.etfs}
    
    allocation_engine = AllocationEngine(
        risk_constraints=config_engine.risk_constraints,
        etf_universe=etf_dict
    )
    
    unit_calculation_engine = UnitCalculationEngine(
        price_buffer_pct=Decimal(str(settings.PRICE_BUFFER_PERCENT)),
        min_unit_value=Decimal(str(settings.MIN_INVESTMENT_AMOUNT))
    )
    
    logger.info("✅ Domain engines initialized")
    
    # 5. Start background services
    logger.info("\n🚀 Step 5/5: Starting background services...")
    
    # Start scheduler
    if settings.SCHEDULER_ENABLED:
        try:
            logger.info("📅 Starting scheduler...")
            scheduler = ETFScheduler()
            scheduler.start()
            logger.info("✅ Scheduler started successfully")
        except Exception as e:
            logger.error(f"❌ Failed to start scheduler: {e}")
    else:
        logger.info("⏰ Scheduler disabled")

    # Start realtime runtime (optional)
    try:
        realtime_runtime = RealtimeRuntime(config_engine)
        await realtime_runtime.start()
        app.state.realtime_runtime = realtime_runtime
        logger.info("⚡ Realtime runtime initialized")
    except Exception as e:
        logger.error(f"❌ Failed to start realtime runtime: {e}")

    # Start options runtime if enabled for this service
    if settings.OPTIONS_RUNTIME_ENABLED:
        try:
            options_runtime = OptionsRuntime(config_engine, realtime_runtime)
            await options_runtime.start()
            app.state.options_runtime = options_runtime
            logger.info("📈 Options runtime initialized")
        except Exception as e:
            logger.error(f"❌ Failed to start options runtime: {e}")
    else:
        logger.info("📉 Options runtime disabled for this service")
    
    # Start Telegram bot (async, no threads!)
    if settings.TELEGRAM_ENABLED and settings.TELEGRAM_BOT_TOKEN:
        try:
            logger.info("🤖 Starting Telegram bot...")
            telegram_bot = ETFTelegramBot()
            
            # Create async task (runs in FastAPI's event loop)
            telegram_task = asyncio.create_task(telegram_bot.start_async())
            
            logger.info("✅ Telegram bot task created successfully")
        except Exception as e:
            logger.error(f"❌ Failed to start Telegram bot: {e}")
    else:
        logger.info("📱 Telegram bot disabled")
    
    logger.info("\n" + "="*60)
    logger.info("🎯 All Services Running:")
    logger.info(f"   ✅ API Server: http://{settings.API_HOST}:{settings.API_PORT}")
    logger.info(f"   ✅ API Docs: http://{settings.API_HOST}:{settings.API_PORT}/docs")
    logger.info(f"   ✅ Scheduler: {'Enabled' if settings.SCHEDULER_ENABLED else 'Disabled'}")
    logger.info(f"   ✅ Telegram: {'Enabled' if settings.TELEGRAM_ENABLED else 'Disabled'}")
    logger.info("   ✅ Capital Model: Base + Tactical (Strict Separation)")
    logger.info("="*60 + "\n")
    
    yield
    
    # ===================
    # SHUTDOWN
    # ===================
    logger.info("\n" + "="*60)
    logger.info("🛑 Shutting down ETF Assistant...")
    logger.info("="*60)
    
    # Stop scheduler
    if scheduler:
        logger.info("⏰ Stopping scheduler...")
        scheduler.stop()
        logger.info("✅ Scheduler stopped")

    if realtime_runtime:
        logger.info("⚡ Stopping realtime runtime...")
        await realtime_runtime.stop()
        logger.info("✅ Realtime runtime stopped")

    if options_runtime:
        await options_runtime.stop()
    
    # Stop Telegram bot
    if telegram_task and not telegram_task.done():
        logger.info("🤖 Stopping Telegram bot...")
        telegram_task.cancel()
        try:
            await telegram_task
        except asyncio.CancelledError:
            logger.info("✅ Telegram bot stopped gracefully")
    
    # Close database
    logger.info("📊 Closing database connections...")
    await close_db()
    logger.info("✅ Database connections closed")
    
    logger.info("\n" + "="*60)
    logger.info("👋 ETF Assistant shutdown complete")
    logger.info("="*60 + "\n")


# Create FastAPI app
app = FastAPI(
    title="Indian ETF Investing Assistant - Base + Tactical",
    description="Disciplined ETF investing with strict capital separation",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# Health check endpoint
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    # Database health (real check)
    db_status = "disconnected"
    db_error = None
    try:
        from app.infrastructure.db.database import engine
        if engine is None:
            db_status = "not_initialized"
        else:
            async with engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            db_status = "connected"
    except Exception as exc:
        db_status = "error"
        db_error = str(exc)

    # Scheduler health
    scheduler_status = "disabled"
    if scheduler:
        try:
            scheduler_running = getattr(scheduler.scheduler, "running", False)
            scheduler_status = "running" if scheduler_running else "stopped"
        except Exception:
            scheduler_status = "error"

    # Telegram health
    telegram_status = "disabled"
    if telegram_task:
        if telegram_task.done():
            telegram_status = "stopped"
        else:
            telegram_status = "running"
    if telegram_bot and getattr(telegram_bot, "application", None):
        try:
            if getattr(telegram_bot.application, "running", False):
                telegram_status = "running"
        except Exception:
            telegram_status = "error"

    return {
        "status": "healthy",
        "service": "ETF Assistant",
        "version": "1.0.0",
        "strategy": config_engine.strategy_version if config_engine else "Not loaded",
        "services": {
            "api": "running",
            "scheduler": scheduler_status,
            "telegram": telegram_status,
            "database": db_status
        },
        "database_error": db_error,
        "capital_model": "Base + Tactical (Strict Separation)"
    }


@app.get("/health/market")
async def market_health():
    """Realtime market data health."""
    runtime = getattr(app.state, "realtime_runtime", None)
    if runtime:
        return await runtime.get_realtime_status()
    return {"enabled": False, "connected": False, "ts": None}


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "🇮🇳 Indian ETF Investing Assistant",
        "version": "1.0.0",
        "capital_model": {
            "base": "Systematic, any day",
            "tactical": "Signal-driven only"
        },
        "docs": "/docs"
    }


# Import and include routers
from app.api.routes import decision, portfolio, config as config_routes, capital, invest, market_data
from app.api.routes.options import router as options_router

app.include_router(capital.router, prefix="/api/v1/capital", tags=["Capital"])
app.include_router(decision.router, prefix="/api/v1/decision", tags=["Tactical Signals"])
app.include_router(invest.router, prefix="/api/v1/invest", tags=["Base & Tactical Execution"])
app.include_router(portfolio.router, prefix="/api/v1/portfolio", tags=["Portfolio"])
app.include_router(config_routes.router, prefix="/api/v1/config", tags=["Config"])
app.include_router(market_data.router, prefix="/api/v1/market-data", tags=["Market Data"])
app.include_router(options_router.router, prefix="/api/v1/options", tags=["Options"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=settings.DEBUG)
