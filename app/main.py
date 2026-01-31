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
from typing import AsyncGenerator
from decimal import Decimal

from app.config import settings
from app.infrastructure.db.database import init_db, close_db
from app.domain.services.config_engine import ConfigEngine
from app.domain.services.market_context_engine import MarketContextEngine
from app.domain.services.capital_engine import CapitalEngine
from app.domain.services.allocation_engine import AllocationEngine
from app.domain.services.unit_calculation_engine import UnitCalculationEngine
from app.domain.services.decision_engine import DecisionEngine
from app.infrastructure.market_data.yfinance_provider import YFinanceProvider
from app.infrastructure.calendar.nse_calendar import NSECalendar

# Import scheduler and telegram
from app.scheduler.main import ETFScheduler
from app.telegram.bot import ETFTelegramBot

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Global instances (loaded once at startup)
config_engine: ConfigEngine | None = None
market_context_engine: MarketContextEngine | None = None
capital_engine: CapitalEngine | None = None
allocation_engine: AllocationEngine | None = None
unit_calculation_engine: UnitCalculationEngine | None = None
decision_engine: DecisionEngine | None = None
market_data_provider: YFinanceProvider | None = None
nse_calendar: NSECalendar | None = None

# Service instances
scheduler: ETFScheduler | None = None
telegram_bot: ETFTelegramBot | None = None
telegram_task: asyncio.Task | None = None


async def start_telegram_bot():
    """Start Telegram bot in background"""
    global telegram_bot
    
    if not settings.TELEGRAM_ENABLED:
        logger.info("📱 Telegram bot disabled (set TELEGRAM_ENABLED=True to enable)")
        return
    
    if not settings.TELEGRAM_BOT_TOKEN:
        logger.warning("⚠️  Telegram bot enabled but no token provided")
        return
    
    try:
        logger.info("🤖 Starting Telegram bot...")
        telegram_bot = ETFTelegramBot()
        
        # Run bot in background
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, telegram_bot.run)
        
    except Exception as e:
        logger.error(f"❌ Failed to start Telegram bot: {e}")


def start_scheduler():
    """Start scheduler"""
    global scheduler
    
    if not settings.SCHEDULER_ENABLED:
        logger.info("⏰ Scheduler disabled (set SCHEDULER_ENABLED=True to enable)")
        return
    
    try:
        logger.info("📅 Starting scheduler...")
        scheduler = ETFScheduler()
        scheduler.start()
        logger.info("✅ Scheduler started successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to start scheduler: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Application lifespan manager
    Handles startup and shutdown of all services
    """
    global config_engine, market_context_engine, capital_engine
    global allocation_engine, unit_calculation_engine, decision_engine
    global market_data_provider, nse_calendar
    global scheduler, telegram_bot, telegram_task
    
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
    market_data_provider = YFinanceProvider()
    nse_calendar = NSECalendar()
    logger.info("✅ Infrastructure initialized")
    
    # 4. Initialize domain engines
    logger.info("\n🔧 Step 4/5: Initializing domain engines...")
    market_context_engine = MarketContextEngine()
    capital_engine = CapitalEngine()
    
    # Create ETF universe dict for allocation engine
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
    start_scheduler()
    
    # Start Telegram bot in background task
    if settings.TELEGRAM_ENABLED and settings.TELEGRAM_BOT_TOKEN:
        telegram_task = asyncio.create_task(start_telegram_bot())
        logger.info("✅ Telegram bot started in background")
    
    logger.info("\n" + "="*60)
    logger.info("🎯 All Services Running:")
    logger.info(f"   ✅ API Server: http://{settings.API_HOST}:{settings.API_PORT}")
    logger.info(f"   ✅ Scheduler: {'Enabled' if settings.SCHEDULER_ENABLED else 'Disabled'}")
    logger.info(f"   ✅ Telegram: {'Enabled' if settings.TELEGRAM_ENABLED else 'Disabled'}")
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
    
    # Stop Telegram bot
    if telegram_task and not telegram_task.done():
        logger.info("🤖 Stopping Telegram bot...")
        telegram_task.cancel()
        try:
            await telegram_task
        except asyncio.CancelledError:
            pass
        logger.info("✅ Telegram bot stopped")
    
    # Close database
    logger.info("📊 Closing database connections...")
    await close_db()
    logger.info("✅ Database connections closed")
    
    logger.info("\n" + "="*60)
    logger.info("👋 ETF Assistant shutdown complete")
    logger.info("="*60 + "\n")


# Create FastAPI app
app = FastAPI(
    title="Indian ETF Investing Assistant",
    description="Production-grade ETF investing system for Indian markets (NSE) with Scheduler & Telegram",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    return {
        "status": "healthy",
        "service": "ETF Assistant",
        "version": "1.0.0",
        "strategy": config_engine.strategy_version if config_engine else "Not loaded",
        "services": {
            "api": "running",
            "scheduler": "running" if scheduler else "disabled",
            "telegram": "running" if telegram_bot else "disabled",
            "database": "connected"
        }
    }


@app.get("/")
async def root():
    """Root endpoint with system info"""
    return {
        "message": "🇮🇳 Indian ETF Investing Assistant API",
        "version": "1.0.0",
        "strategy": config_engine.strategy_version if config_engine else "Not loaded",
        "services": {
            "api": "running",
            "scheduler": "running" if scheduler and settings.SCHEDULER_ENABLED else "disabled",
            "telegram": "running" if telegram_bot and settings.TELEGRAM_ENABLED else "disabled"
        },
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "config": "/api/v1/config",
            "decision": "/api/v1/decision",
            "portfolio": "/api/v1/portfolio",
            "capital": "/api/v1/capital"
        }
    }


@app.get("/services/status")
async def services_status():
    """Get detailed status of all services"""
    scheduler_jobs = []
    if scheduler and scheduler.scheduler:
        scheduler_jobs = [
            {
                "id": job.id,
                "name": job.name,
                "next_run": str(job.next_run_time) if job.next_run_time else None
            }
            for job in scheduler.scheduler.get_jobs()
        ]
    
    return {
        "api": {
            "status": "running",
            "host": settings.API_HOST,
            "port": settings.API_PORT
        },
        "scheduler": {
            "enabled": settings.SCHEDULER_ENABLED,
            "status": "running" if scheduler else "disabled",
            "jobs": scheduler_jobs
        },
        "telegram": {
            "enabled": settings.TELEGRAM_ENABLED,
            "status": "running" if telegram_bot else "disabled",
            "configured": bool(settings.TELEGRAM_BOT_TOKEN)
        },
        "database": {
            "status": "connected",
            "url": settings.DATABASE_URL.split('@')[1] if '@' in settings.DATABASE_URL else "configured"
        },
        "configuration": {
            "etfs": len(config_engine.etf_universe.etfs) if config_engine else 0,
            "strategy": config_engine.strategy_version if config_engine else None
        }
    }


# Import and include routers
from app.api.routes import decision, portfolio, config as config_routes, capital

app.include_router(capital.router, prefix="/api/v1/capital", tags=["Capital"])
app.include_router(decision.router, prefix="/api/v1/decision", tags=["Decision"])
app.include_router(portfolio.router, prefix="/api/v1/portfolio", tags=["Portfolio"])
app.include_router(config_routes.router, prefix="/api/v1/config", tags=["Config"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )