"""
Application Settings
Load from environment variables
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application settings from environment"""

    # ======================
    # Database
    # ======================
    DATABASE_URL: str = "postgresql://etf_user:etf_password@localhost:5432/etf_assistant"
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "etf_assistant"
    DB_USER: str = "etf_user"
    DB_PASSWORD: str = "etf_password"

    # ======================
    # Application
    # ======================
    APP_ENV: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    SECRET_KEY: str = "change-me-in-production"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # ======================
    # Telegram
    # ======================
    TELEGRAM_BOT_TOKEN: Optional[str] = None
    TELEGRAM_CHAT_ID: Optional[str] = None
    TELEGRAM_ENABLED: bool = False

    # ======================
    # Market Data
    # ======================
    MARKET_DATA_PROVIDER: str = "yfinance"
    MARKET_DATA_API_KEY: Optional[str] = None

    # Market Timings (🔥 MISSING BEFORE)
    MARKET_OPEN_HOUR: int = 9
    MARKET_OPEN_MINUTE: int = 15
    MARKET_CLOSE_HOUR: int = 15
    MARKET_CLOSE_MINUTE: int = 30

    # ======================
    # Scheduler
    # ======================
    SCHEDULER_ENABLED: bool = False
    DAILY_DECISION_TIME: str = "10:00"
    MONTHLY_SUMMARY_TIME: str = "18:00"

    # ======================
    # Risk & Safety
    # ======================
    PRICE_BUFFER_PERCENT: float = 2.0
    MAX_SLIPPAGE_PERCENT: float = 3.0
    MIN_INVESTMENT_AMOUNT: float = 100.0
    MAX_SINGLE_INVESTMENT: float = 100000.0

    # ======================
    # Strategy
    # ======================
    STRATEGY_VERSION: str = "2025-Q1"

    # ======================
    # Timezone
    # ======================
    TIMEZONE: str = "Asia/Kolkata"

    # ======================
    # Logging (🔥 MISSING BEFORE)
    # ======================
    LOG_FILE: str = "logs/app.log"
    LOG_MAX_BYTES: int = 10_485_760
    LOG_BACKUP_COUNT: int = 5

    # ======================
    # Redis (🔥 MISSING BEFORE)
    # ======================
    REDIS_ENABLED: bool = False
    REDIS_URL: str = "redis://localhost:6379/0"

    # ======================
    # Sentry (🔥 MISSING BEFORE)
    # ======================
    SENTRY_DSN: str = ""
    SENTRY_ENABLED: bool = False

    # ======================
    # Flags (🔥 MISSING BEFORE)
    # ======================
    TEST_MODE: bool = False
    MOCK_MARKET_DATA: bool = False

    # ======================
    # Pydantic v2 config
    # ======================
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="forbid",   # keep strict (GOOD)
    )


settings = Settings()
