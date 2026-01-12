from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Central application configuration (Pydantic v2 compatible).
    """

    # =========================
    # Application
    # =========================
    APP_NAME: str = "Nifty Investment Assistant"
    ENVIRONMENT: str = Field(default="development")
    TIMEZONE: str = Field(default="Asia/Kolkata")

    # =========================
    # Database (PostgreSQL)
    # =========================
    DB_HOST: str
    DB_PORT: int = 5432
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str

    DATABASE_URL: Optional[str] = None

    # =========================
    # Telegram
    # =========================
    TELEGRAM_BOT_TOKEN: str
    TELEGRAM_CHAT_ID: Optional[str] = None

    # =========================
    # Scheduler
    # =========================
    MARKET_JOB_HOUR: int = 15
    MARKET_JOB_MINUTE: int = 16

    # =========================
    # Investment Rules
    # =========================
    MANDATORY_DEPLOYMENT_RATIO: float = 0.7

    DIP_1_PERCENT_AMOUNT: int = 500
    DIP_2_PERCENT_AMOUNT: int = 1000
    DIP_3_PERCENT_AMOUNT: int = 2000

    # =========================
    # Logging
    # =========================
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = True

    def build_database_url(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL

        return (
            f"postgresql+psycopg2://"
            f"{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.DATABASE_URL = settings.build_database_url()
    return settings
