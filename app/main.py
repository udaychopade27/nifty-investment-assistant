"""
FastAPI Main Application
Composition Root - Dependencies assembled here only
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path
import logging
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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Application lifespan manager
    Handles startup and shutdown
    """
    global config_engine, market_context_engine, capital_engine
    global allocation_engine, unit_calculation_engine, decision_engine
    global market_data_provider, nse_calendar
    
    # Startup
    logger.info("🚀 Starting ETF Assistant...")
    
    # Initialize database
    await init_db()
    logger.info("✅ Database initialized")
    
    # Load configuration
    config_dir = Path(__file__).parent.parent / "config"
    config_engine = ConfigEngine(config_dir)
    config_engine.load_all()
    logger.info("✅ Configuration loaded successfully")
    logger.info(f"📊 ETF Universe: {len(config_engine.etf_universe.etfs)} ETFs")
    logger.info(f"📈 Strategy Version: {config_engine.strategy_version}")
    
    # Initialize infrastructure
    market_data_provider = YFinanceProvider()
    nse_calendar = NSECalendar()
    logger.info("✅ Infrastructure initialized")
    
    # Initialize domain engines
    market_context_engine = MarketContextEngine()
    
    # Note: Capital engine needs repository instances
    # These would be created per-request with database sessions
    # For now, we'll initialize engines that don't need DB
    
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
    logger.info("🎯 System ready for operation")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down ETF Assistant...")
    await close_db()
    logger.info("✅ Database connections closed")


# Create FastAPI app
app = FastAPI(
    title="Indian ETF Investing Assistant",
    description="Production-grade ETF investing system for Indian markets (NSE)",
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
    """Basic health check"""
    return {
        "status": "healthy",
        "service": "ETF Assistant",
        "version": "1.0.0",
        "strategy": config_engine.strategy_version if config_engine else "Not loaded"
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "🇮🇳 Indian ETF Investing Assistant API",
        "version": "1.0.0",
        "strategy": config_engine.strategy_version if config_engine else "Not loaded",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "config": "/api/v1/config",
            "decision": "/api/v1/decision",
            "portfolio": "/api/v1/portfolio"
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
