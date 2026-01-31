"""
MARKET CONTEXT ENGINE (ENGINE-1)
Compute market stress context (NOT decisions)

RESPONSIBILITIES:
- Calculate daily market change
- Calculate multi-day trends
- Determine stress level
- NO PREDICTIONS, NO SIGNALS, NO DECISIONS

RULES:
❌ No predictions
❌ No signals
❌ No investing decisions
✅ Pure calculation
✅ Deterministic output
"""

from datetime import date
from decimal import Decimal
from typing import Optional

from app.domain.models import MarketContext, StressLevel


class MarketContextEngine:
    """
    Market Context Engine
    Calculates market environment, does NOT make decisions
    """
    
    def __init__(self):
        """Initialize engine"""
        pass
    
    def calculate_context(
        self,
        calc_date: date,
        nifty_close: Decimal,
        nifty_previous_close: Decimal,
        last_3_day_closes: list[Decimal],
        india_vix: Optional[Decimal] = None
    ) -> MarketContext:
        """
        Calculate market context for a given date
        
        Args:
            calc_date: Date of calculation
            nifty_close: Today's NIFTY closing price
            nifty_previous_close: Previous trading day's close
            last_3_day_closes: Last 3 trading day closes (oldest first)
            india_vix: India VIX value (optional)
        
        Returns:
            MarketContext object
        """
        # Calculate daily change
        daily_change_pct = self._calculate_percentage_change(
            old_value=nifty_previous_close,
            new_value=nifty_close
        )
        
        # Calculate 3-day cumulative change
        cumulative_3day_pct = self._calculate_cumulative_change(
            closes=last_3_day_closes,
            current_close=nifty_close
        )
        
        # Determine stress level
        stress_level = self._determine_stress_level(
            daily_change=daily_change_pct,
            cumulative_3day=cumulative_3day_pct,
            vix=india_vix
        )
        
        return MarketContext(
            date=calc_date,
            nifty_close=nifty_close,
            nifty_previous_close=nifty_previous_close,
            daily_change_pct=daily_change_pct,
            cumulative_3day_pct=cumulative_3day_pct,
            india_vix=india_vix,
            stress_level=stress_level
        )
    
    @staticmethod
    def _calculate_percentage_change(
        old_value: Decimal,
        new_value: Decimal
    ) -> Decimal:
        """
        Calculate percentage change
        
        Formula: ((new - old) / old) * 100
        """
        if old_value <= Decimal('0'):
            raise ValueError("Old value must be positive")
        
        change = ((new_value - old_value) / old_value) * Decimal('100')
        return change.quantize(Decimal('0.01'))
    
    @staticmethod
    def _calculate_cumulative_change(
        closes: list[Decimal],
        current_close: Decimal
    ) -> Decimal:
        """
        Calculate cumulative change over multiple days
        
        Args:
            closes: List of closing prices (oldest first)
            current_close: Today's close
        
        Returns:
            Cumulative percentage change
        """
        if not closes:
            return Decimal('0')
        
        oldest_close = closes[0]
        if oldest_close <= Decimal('0'):
            raise ValueError("Oldest close must be positive")
        
        cumulative_change = (
            (current_close - oldest_close) / oldest_close
        ) * Decimal('100')
        
        return cumulative_change.quantize(Decimal('0.01'))
    
    @staticmethod
    def _determine_stress_level(
        daily_change: Decimal,
        cumulative_3day: Decimal,
        vix: Optional[Decimal]
    ) -> StressLevel:
        """
        Determine market stress level based on multiple factors
        
        Logic:
        - HIGH: Daily < -3% OR 3-day < -5% OR VIX >= 25
        - MEDIUM: Daily < -2% OR 3-day < -3% OR VIX >= 20
        - LOW: Daily < -1% OR 3-day < -1.5%
        - NONE: Otherwise
        """
        # Check HIGH stress
        if daily_change < Decimal('-3.0'):
            return StressLevel.HIGH
        if cumulative_3day < Decimal('-5.0'):
            return StressLevel.HIGH
        if vix and vix >= Decimal('25'):
            return StressLevel.HIGH
        
        # Check MEDIUM stress
        if daily_change < Decimal('-2.0'):
            return StressLevel.MEDIUM
        if cumulative_3day < Decimal('-3.0'):
            return StressLevel.MEDIUM
        if vix and vix >= Decimal('20'):
            return StressLevel.MEDIUM
        
        # Check LOW stress
        if daily_change < Decimal('-1.0'):
            return StressLevel.LOW
        if cumulative_3day < Decimal('-1.5'):
            return StressLevel.LOW
        
        # No stress
        return StressLevel.NONE
    
    def is_dip_day(
        self,
        context: MarketContext,
        threshold: Decimal = Decimal('-1.0')
    ) -> bool:
        """
        Check if today is a dip day (market down beyond threshold)
        
        Args:
            context: Market context
            threshold: Dip threshold (default -1%)
        
        Returns:
            True if market fell below threshold
        """
        return context.daily_change_pct < threshold
    
    def get_dip_magnitude(
        self,
        context: MarketContext
    ) -> Decimal:
        """
        Get magnitude of dip (absolute value of negative change)
        Returns 0 if market is up
        
        Args:
            context: Market context
        
        Returns:
            Absolute value of negative change, or 0
        """
        if context.daily_change_pct < Decimal('0'):
            return abs(context.daily_change_pct)
        return Decimal('0')
