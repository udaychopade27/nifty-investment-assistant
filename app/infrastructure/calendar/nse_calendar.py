"""
NSE Trading Calendar
Handle NSE trading days, holidays, and market hours
Dynamically fetch holidays from NSE website
"""

from datetime import date, datetime, time, timedelta
from typing import Dict, List, Set, Optional, Tuple
import calendar
import requests
from bs4 import BeautifulSoup
import logging
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from app.config import settings
from app.infrastructure.db.models import TradingHolidayModel

logger = logging.getLogger(__name__)


class NSECalendar:
    """
    NSE (National Stock Exchange of India) Trading Calendar
    Fetches holidays dynamically from NSE website
    """
    
    def __init__(self, use_cache: bool = True):
        """Initialize NSE calendar"""
        # NSE market hours (IST)
        self.market_open = time(9, 15)   # 9:15 AM
        self.market_close = time(15, 30)  # 3:30 PM
        
        # Cache for fetched holidays
        self._holidays_cache: Set[date] = set()
        self._cache_loaded = False
        self.use_cache = use_cache
        self._last_refresh_month: Optional[Tuple[int, int]] = None

        # Lazy DB session (sync) for holiday caching
        self._db_engine = None
        self._db_session_factory = None
        
        # Fallback holidays (if fetch fails)
        self.fallback_holidays_2025 = {
            date(2025, 1, 26),  # Republic Day
            date(2025, 3, 14),  # Mahashivratri
            date(2025, 3, 31),  # Holi
            date(2025, 4, 10),  # Mahavir Jayanti
            date(2025, 4, 14),  # Dr. Ambedkar Jayanti
            date(2025, 4, 18),  # Good Friday
            date(2025, 5, 1),   # Maharashtra Day
            date(2025, 8, 15),  # Independence Day
            date(2025, 8, 27),  # Ganesh Chaturthi
            date(2025, 10, 2),  # Gandhi Jayanti
            date(2025, 10, 20), # Dussehra
            date(2025, 11, 5),  # Diwali - Laxmi Pujan
            date(2025, 11, 6),  # Diwali - Balipratipada
            date(2025, 11, 24), # Gurunanak Jayanti
            date(2025, 12, 25), # Christmas
        }
        
        # Fallback holidays 2026
        self.fallback_holidays_2026 = {
            date(2026, 1, 26),  # Republic Day
            date(2026, 3, 3),   # Mahashivratri
            date(2026, 3, 25),  # Holi
            date(2026, 12, 25), # Christmas
        }
        
        # All fallback holidays
        self.fallback_holidays = self.fallback_holidays_2025 | self.fallback_holidays_2026
    
    def _normalize_sync_db_url(self, url: str) -> str:
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        if "+asyncpg" in url:
            url = url.replace("+asyncpg", "+psycopg2")
        return url

    def _get_db_session(self):
        if self._db_session_factory is None:
            try:
                sync_url = self._normalize_sync_db_url(settings.DATABASE_URL)
                self._db_engine = create_engine(sync_url, pool_pre_ping=True)
                self._db_session_factory = sessionmaker(bind=self._db_engine)
            except Exception as exc:
                logger.warning(f"⚠️  Holiday DB connection disabled: {exc}")
                self._db_engine = None
                self._db_session_factory = None
        if self._db_session_factory is None:
            return None
        return self._db_session_factory()

    def _load_holidays_from_db(self, years: List[int]) -> Dict[date, str]:
        session = self._get_db_session()
        if session is None:
            return {}
        try:
            result = session.execute(
                select(TradingHolidayModel).where(TradingHolidayModel.year.in_(years))
            )
            rows = result.scalars().all()
            holidays = {row.date: (row.description or "Holiday") for row in rows}
            if holidays:
                logger.info(f"✅ Loaded {len(holidays)} holidays from DB for {years}")
            return holidays
        except SQLAlchemyError as exc:
            logger.warning(f"⚠️  Failed to load holidays from DB: {exc}")
            return {}
        finally:
            session.close()

    def _save_holidays_to_db(self, holidays_by_date: Dict[date, str]) -> None:
        if not holidays_by_date:
            return
        session = self._get_db_session()
        if session is None:
            return
        try:
            dates = list(holidays_by_date.keys())
            existing = session.execute(
                select(TradingHolidayModel.date).where(TradingHolidayModel.date.in_(dates))
            ).scalars().all()
            existing_dates = set(existing)

            for holiday_date, description in holidays_by_date.items():
                if holiday_date in existing_dates:
                    continue
                session.add(
                    TradingHolidayModel(
                        date=holiday_date,
                        description=description or "Holiday",
                        year=holiday_date.year
                    )
                )
            session.commit()
        except SQLAlchemyError as exc:
            session.rollback()
            logger.warning(f"⚠️  Failed to save holidays to DB: {exc}")
        finally:
            session.close()

    def fetch_nse_holidays(self, year: int) -> Dict[date, str]:
        """
        Fetch holidays from NSE website for a given year
        
        Args:
            year: Year to fetch holidays for
        
        Returns:
            Dict of holiday date -> description
        """
        try:
            # NSE holiday calendar URL
            url = f"https://www.nseindia.com/api/holiday-master?type=trading"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
                'Accept-Language': 'en-US,en;q=0.9',
            }
            
            logger.info(f"Fetching NSE holidays for year {year}...")
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            holidays: Dict[date, str] = {}
            
            # Parse CM (Capital Market) holidays
            if 'CM' in data:
                for holiday in data['CM']:
                    holiday_date_str = holiday.get('tradingDate')
                    if holiday_date_str:
                        # Parse date (format: DD-MMM-YYYY)
                        holiday_date = datetime.strptime(holiday_date_str, '%d-%b-%Y').date()
                        if holiday_date.year == year:
                            description = holiday.get('description', 'Holiday')
                            holidays[holiday_date] = description
                            logger.debug(f"  • {holiday_date}: {description}")
            
            logger.info(f"✅ Fetched {len(holidays)} holidays from NSE for {year}")
            return holidays
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to fetch NSE holidays: {e}")
            logger.info("Using fallback holiday list")
            return {h: "Holiday" for h in self.fallback_holidays if h.year == year}
    
    def _ensure_monthly_cache(self, month_date: date) -> None:
        if not self.use_cache:
            return

        month_key = (month_date.year, month_date.month)
        if self._cache_loaded and self._last_refresh_month == month_key:
            return

        years = [month_date.year, month_date.year + 1]
        holidays_by_date: Dict[date, str] = {}

        # Load from DB first
        holidays_by_date.update(self._load_holidays_from_db(years))

        # Fetch missing years (monthly refresh)
        missing_years = [y for y in years if not any(d.year == y for d in holidays_by_date)]
        for year in missing_years:
            fetched = self.fetch_nse_holidays(year)
            holidays_by_date.update(fetched)
            self._save_holidays_to_db(fetched)

        self._holidays_cache = set(holidays_by_date.keys())
        self._cache_loaded = True
        self._last_refresh_month = month_key
        logger.info(f"✅ Loaded {len(self._holidays_cache)} holidays total")

    def load_holidays(self, years: Optional[List[int]] = None):
        """
        Load holidays for specified years
        
        Args:
            years: List of years to load (default: current and next year)
        """
        if years is None:
            current_year = date.today().year
            years = [current_year, current_year + 1]

        holidays_by_date: Dict[date, str] = {}
        holidays_by_date.update(self._load_holidays_from_db(years))

        missing_years = [y for y in years if not any(d.year == y for d in holidays_by_date)]
        for year in missing_years:
            fetched = self.fetch_nse_holidays(year)
            holidays_by_date.update(fetched)
            self._save_holidays_to_db(fetched)

        self._holidays_cache = set(holidays_by_date.keys())
        self._cache_loaded = True
        self._last_refresh_month = (date.today().year, date.today().month)
        logger.info(f"✅ Loaded {len(self._holidays_cache)} holidays total")
    
    def get_holidays(self) -> Set[date]:
        """
        Get all cached holidays
        
        Returns:
            Set of all holidays
        """
        if self.use_cache:
            self._ensure_monthly_cache(date.today())
        
        return self._holidays_cache if self.use_cache else self.fallback_holidays
    
    def is_trading_day(self, check_date: date) -> bool:
        """
        Check if a date is a trading day
        
        Args:
            check_date: Date to check
        
        Returns:
            True if trading day, False otherwise
        """
        # Check if weekend
        if check_date.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        
        # Get holidays (monthly cached)
        if self.use_cache:
            self._ensure_monthly_cache(check_date)
        # Get holidays
        all_holidays = self.get_holidays()
        
        # Check if holiday
        if check_date in all_holidays:
            return False
        
        return True
    
    def get_next_trading_day(self, from_date: date) -> date:
        """
        Get next trading day from a given date
        
        Args:
            from_date: Starting date
        
        Returns:
            Next trading day
        """
        next_day = from_date + timedelta(days=1)
        
        while not self.is_trading_day(next_day):
            next_day += timedelta(days=1)
            
            # Safety check (max 30 days ahead)
            if (next_day - from_date).days > 30:
                raise ValueError("Could not find trading day within 30 days")
        
        return next_day
    
    def get_previous_trading_day(self, from_date: date) -> date:
        """
        Get previous trading day from a given date
        
        Args:
            from_date: Starting date
        
        Returns:
            Previous trading day
        """
        prev_day = from_date - timedelta(days=1)
        
        while not self.is_trading_day(prev_day):
            prev_day -= timedelta(days=1)
            
            # Safety check (max 30 days back)
            if (from_date - prev_day).days > 30:
                raise ValueError("Could not find trading day within 30 days")
        
        return prev_day
    
    def get_trading_days_in_month(self, month: date) -> int:
        """
        Get number of trading days in a month
        
        Args:
            month: First day of month
        
        Returns:
            Number of trading days
        """
        # Get last day of month
        last_day = calendar.monthrange(month.year, month.month)[1]
        
        trading_days = 0
        for day in range(1, last_day + 1):
            check_date = date(month.year, month.month, day)
            if self.is_trading_day(check_date):
                trading_days += 1
        
        return trading_days
    
    def get_trading_days_list(
        self,
        start_date: date,
        end_date: date
    ) -> List[date]:
        """
        Get list of all trading days between two dates
        
        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
        
        Returns:
            List of trading days
        """
        trading_days = []
        current = start_date
        
        while current <= end_date:
            if self.is_trading_day(current):
                trading_days.append(current)
            current += timedelta(days=1)
        
        return trading_days
    
    def is_market_open(self, check_time: datetime) -> bool:
        """
        Check if market is currently open
        
        Args:
            check_time: Time to check
        
        Returns:
            True if market is open
        """
        # Check if trading day
        if not self.is_trading_day(check_time.date()):
            return False
        
        # Check if within market hours
        current_time = check_time.time()
        
        return self.market_open <= current_time <= self.market_close
    
    def add_holiday(self, holiday_date: date, description: str = ""):
        """
        Add a custom holiday
        
        Args:
            holiday_date: Date of holiday
            description: Holiday description
        """
        self._holidays_cache.add(holiday_date)
    
    def get_holidays_in_month(self, month: date) -> List[date]:
        """
        Get all holidays in a specific month
        
        Args:
            month: First day of month
        
        Returns:
            List of holiday dates
        """
        all_holidays = self.get_holidays()
        holidays = []
        for holiday in sorted(all_holidays):
            if holiday.year == month.year and holiday.month == month.month:
                holidays.append(holiday)
        
        return holidays
    
    def get_remaining_trading_days_in_month(
        self,
        from_date: date
    ) -> int:
        """
        Get number of trading days remaining in month from a date
        
        Args:
            from_date: Starting date
        
        Returns:
            Number of remaining trading days
        """
        # Get last day of month
        last_day = calendar.monthrange(from_date.year, from_date.month)[1]
        end_date = date(from_date.year, from_date.month, last_day)
        
        trading_days = self.get_trading_days_list(from_date, end_date)
        return len(trading_days)
    
    def is_month_end_approaching(
        self,
        check_date: date,
        days_threshold: int = 3
    ) -> bool:
        """
        Check if month end is approaching
        
        Args:
            check_date: Date to check
            days_threshold: Number of trading days threshold
        
        Returns:
            True if within threshold days of month end
        """
        remaining = self.get_remaining_trading_days_in_month(check_date)
        return remaining <= days_threshold
