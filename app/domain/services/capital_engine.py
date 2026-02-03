"""
CAPITAL ENGINE (ENGINE-2) - ASYNC VERSION
Single source of truth for capital state

✅ FIXED: All repository methods properly awaited
✅ FIXED: All methods that call repositories are now async

RESPONSIBILITIES:
- Track monthly capital configuration
- Calculate remaining capital per bucket
- Handle carry-forward logic
- NO INVESTING LOGIC, NO PRICE DATA

RULES:
❌ No investing logic
❌ No price data
❌ No rounding shortcuts
✅ Capital buckets isolated
✅ Carry-forward preserved
✅ Pure state calculation
✅ Async/await properly used
"""

from datetime import date
from decimal import Decimal
from typing import Protocol, Optional

from app.domain.models import CapitalState, MonthlyConfig


class MonthlyConfigRepository(Protocol):
    """Protocol for monthly config data access - ASYNC"""
    
    async def get_for_month(self, month: date) -> Optional[MonthlyConfig]:
        """Get monthly config for a specific month"""
        ...
    
    async def get_current(self) -> Optional[MonthlyConfig]:
        """Get current month's config"""
        ...


class ExecutedInvestmentRepository(Protocol):
    """Protocol for executed investment data access - ASYNC"""
    
    async def get_total_base_deployed(self, month: date) -> Decimal:
        """Get total base capital deployed this month"""
        ...
    
    async def get_total_tactical_deployed(self, month: date) -> Decimal:
        """Get total tactical capital deployed this month"""
        ...
    
    async def get_total_extra_deployed(self, month: date) -> Decimal:
        """Get total extra capital deployed this month"""
        ...


class ExtraCapitalRepository(Protocol):
    """Protocol for extra capital injection data access - ASYNC"""
    
    async def get_total_for_month(self, month: date) -> Decimal:
        """Get total extra capital injected this month"""
        ...


class CapitalEngine:
    """
    Capital Engine - ASYNC VERSION
    Maintains truthful capital state across buckets
    
    ✅ CRITICAL FIX: All methods that access repositories are now async
    """
    
    def __init__(
        self,
        monthly_config_repo: MonthlyConfigRepository,
        executed_investment_repo: ExecutedInvestmentRepository,
        extra_capital_repo: ExtraCapitalRepository
    ):
        """Initialize with repository dependencies"""
        self.monthly_config_repo = monthly_config_repo
        self.executed_investment_repo = executed_investment_repo
        self.extra_capital_repo = extra_capital_repo
    
    async def get_capital_state(self, month: date) -> CapitalState:
        """
        Get current capital state for a month
        
        ✅ FIXED: Now properly awaits all repository calls
        
        Args:
            month: Month to get state for (first day of month)
        
        Returns:
            CapitalState object
        
        Raises:
            ValueError: If MonthlyConfig doesn't exist
        """
        # ✅ FIXED: Await the async call
        config = await self.monthly_config_repo.get_for_month(month)
        if config is None:
            raise ValueError(f"No MonthlyConfig found for {month}")
        
        # ✅ FIXED: Await all async repository calls
        base_deployed = await self.executed_investment_repo.get_total_base_deployed(month)
        tactical_deployed = await self.executed_investment_repo.get_total_tactical_deployed(month)
        extra_deployed = await self.executed_investment_repo.get_total_extra_deployed(month)
        
        # ✅ FIXED: Await the async call
        extra_injected = await self.extra_capital_repo.get_total_for_month(month)
        
        # Calculate remaining
        base_remaining = config.base_capital - base_deployed
        tactical_remaining = config.tactical_capital - tactical_deployed
        extra_remaining = extra_injected - extra_deployed
        
        # Ensure non-negative (safety check)
        base_remaining = max(base_remaining, Decimal('0'))
        tactical_remaining = max(tactical_remaining, Decimal('0'))
        extra_remaining = max(extra_remaining, Decimal('0'))
        
        return CapitalState(
            month=month,
            base_total=config.base_capital,
            base_remaining=base_remaining,
            tactical_total=config.tactical_capital,
            tactical_remaining=tactical_remaining,
            extra_total=extra_injected,
            extra_remaining=extra_remaining
        )
    
    async def get_current_capital_state(self) -> CapitalState:
        """
        Get capital state for current month
        
        ✅ FIXED: Now async and awaits repository call
        
        Returns:
            CapitalState object
        
        Raises:
            ValueError: If no current MonthlyConfig exists
        """
        # ✅ FIXED: Await the async call
        config = await self.monthly_config_repo.get_current()
        if config is None:
            raise ValueError("No MonthlyConfig found for current month")
        
        # ✅ FIXED: Await the async call
        return await self.get_capital_state(config.month)
    
    def calculate_daily_tranche(self, config: MonthlyConfig) -> Decimal:
        """
        Calculate daily base capital tranche
        
        This method remains sync as it only does calculations
        
        Args:
            config: Monthly configuration
        
        Returns:
            Amount to invest daily from base capital
        """
        if config.trading_days <= 0:
            raise ValueError("Trading days must be positive")
        
        return config.base_capital / Decimal(str(config.trading_days))
    
    def can_deploy_tactical(
        self,
        capital_state: CapitalState,
        amount: Decimal
    ) -> tuple[bool, str]:
        """
        Check if tactical capital deployment is possible
        
        This method remains sync as it only validates state
        
        Args:
            capital_state: Current capital state
            amount: Amount to deploy
        
        Returns:
            (can_deploy, reason)
        """
        if amount <= Decimal('0'):
            return False, "Deployment amount must be positive"
        
        if capital_state.tactical_remaining < amount:
            return False, f"Insufficient tactical capital: ₹{capital_state.tactical_remaining}"
        
        return True, "OK"
    
    def can_deploy_extra(
        self,
        capital_state: CapitalState,
        amount: Decimal
    ) -> tuple[bool, str]:
        """
        Check if extra capital deployment is possible
        
        This method remains sync as it only validates state
        
        Args:
            capital_state: Current capital state
            amount: Amount to deploy
        
        Returns:
            (can_deploy, reason)
        """
        if amount <= Decimal('0'):
            return False, "Deployment amount must be positive"
        
        if capital_state.extra_remaining < amount:
            return False, f"Insufficient extra capital: ₹{capital_state.extra_remaining}"
        
        return True, "OK"
    
    def calculate_tactical_carry_forward(
        self,
        previous_month_state: CapitalState,
        new_monthly_capital: Decimal,
        carry_forward_cap_multiplier: Decimal = Decimal('1.5')
    ) -> Decimal:
        """
        Calculate how much tactical capital to carry forward to next month
        
        This method remains sync as it only does calculations
        
        Args:
            previous_month_state: Previous month's final state
            new_monthly_capital: New month's total capital
            carry_forward_cap_multiplier: Maximum carry-forward as multiple of monthly capital
        
        Returns:
            Amount to carry forward (capped)
        """
        unused_tactical = previous_month_state.tactical_remaining
        
        # Calculate cap (40% of new monthly capital * multiplier)
        tactical_portion = new_monthly_capital * Decimal('0.4')
        carry_forward_cap = tactical_portion * carry_forward_cap_multiplier
        
        # Return minimum of unused and cap
        return min(unused_tactical, carry_forward_cap)
    
    def validate_capital_integrity(
        self,
        capital_state: CapitalState
    ) -> tuple[bool, list[str]]:
        """
        Validate capital state integrity
        
        This method remains sync as it only validates state
        
        Args:
            capital_state: Capital state to validate
        
        Returns:
            (is_valid, list of issues)
        """
        issues = []
        
        # Check non-negative
        if capital_state.base_remaining < Decimal('0'):
            issues.append("Base capital is negative")
        
        if capital_state.tactical_remaining < Decimal('0'):
            issues.append("Tactical capital is negative")
        
        if capital_state.extra_remaining < Decimal('0'):
            issues.append("Extra capital is negative")
        
        # Check remaining <= total
        if capital_state.base_remaining > capital_state.base_total:
            issues.append("Base remaining exceeds total")
        
        if capital_state.tactical_remaining > capital_state.tactical_total:
            issues.append("Tactical remaining exceeds total")
        
        if capital_state.extra_remaining > capital_state.extra_total:
            issues.append("Extra remaining exceeds total")
        
        return len(issues) == 0, issues