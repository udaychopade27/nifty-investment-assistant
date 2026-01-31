"""
Application Configuration
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://etf_user:etf_password@localhost:5432/etf_assistant"
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    # Timezone
    TIMEZONE: str = "Asia/Kolkata"
    
    # Investment settings
    PRICE_BUFFER_PERCENT: float = 2.0
    MIN_INVESTMENT_AMOUNT: float = 100.0
    
    # Telegram Bot
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_ENABLED: bool = True
    TELEGRAM_CHAT_ID: str = ""  # Optional
    
    # Scheduler
    SCHEDULER_ENABLED: bool = True
    DAILY_DECISION_TIME: str = "10:00"
    MONTHLY_SUMMARY_TIME: str = "18:00"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()