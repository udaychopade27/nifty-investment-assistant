"""
UNIT CALCULATION ENGINE (ENGINE-4)
Convert ₹ amounts → whole ETF units (INDIA-CRITICAL)

RESPONSIBILITIES:
- Convert allocation amounts to executable units
- Apply price buffers
- Floor() for whole units only
- Preserve unused capital

RULES (LOCKED):
❌ No fractional units
❌ No redistribution to force buys
❌ No price guessing
✅ Use floor() ALWAYS
✅ Preserve unused capital
✅ 2% price buffer
"""

import math
from decimal import Decimal
from typing import List, Dict, Tuple

from app.domain.models import (
    ETFAllocation,
    ETFUnitPlan,
    ETFStatus
)


class UnitCalculationEngine:
    """
    Unit Calculation Engine
    India's #1 critical rule: Whole units only, always floor()
    """
    
    def __init__(
        self,
        price_buffer_pct: Decimal = Decimal('2.0'),
        min_unit_value: Decimal = Decimal('100.0')
    ):
        """
        Initialize unit calculation engine
        
        Args:
            price_buffer_pct: Price buffer percentage (default 2%)
            min_unit_value: Minimum value for 1 unit (default ₹100)
        """
        self.price_buffer_pct = price_buffer_pct
        self.min_unit_value = min_unit_value
    
    def calculate_units(
        self,
        allocations: List[ETFAllocation],
        current_prices: Dict[str, Decimal]
    ) -> Tuple[List[ETFUnitPlan], Decimal]:
        """
        Convert allocations to unit-based plans
        
        Args:
            allocations: ETF allocations with amounts
            current_prices: Current LTP for each ETF (symbol -> price)
        
        Returns:
            Tuple of (unit plans list, total unused amount)
        """
        plans = []
        total_unused = Decimal('0')
        
        for allocation in allocations:
            plan, unused = self._calculate_single_etf(
                allocation,
                current_prices
            )
            plans.append(plan)
            total_unused += unused
        
        return plans, total_unused
    
    def _calculate_single_etf(
        self,
        allocation: ETFAllocation,
        current_prices: Dict[str, Decimal]
    ) -> Tuple[ETFUnitPlan, Decimal]:
        """
        Calculate units for a single ETF
        
        Args:
            allocation: ETF allocation
            current_prices: Price dictionary
        
        Returns:
            Tuple of (unit plan, unused amount)
        """
        symbol = allocation.etf_symbol
        allocated_amount = allocation.allocated_amount
        
        # Get current price
        ltp = current_prices.get(symbol)
        
        # Price not available
        if ltp is None or ltp <= Decimal('0'):
            return ETFUnitPlan(
                etf_symbol=symbol,
                ltp=Decimal('0'),
                effective_price=Decimal('0'),
                units=0,
                actual_amount=Decimal('0'),
                unused_amount=allocated_amount,
                status=ETFStatus.SKIPPED,
                reason="Price not available"
            ), allocated_amount
        
        # Calculate effective price with buffer
        effective_price = self._calculate_effective_price(ltp)
        
        # Calculate units using floor() - CRITICAL
        units = self._calculate_floor_units(allocated_amount, effective_price)
        
        # Not enough for even 1 unit
        if units < 1:
            return ETFUnitPlan(
                etf_symbol=symbol,
                ltp=ltp,
                effective_price=effective_price,
                units=0,
                actual_amount=Decimal('0'),
                unused_amount=allocated_amount,
                status=ETFStatus.SKIPPED,
                reason=f"Insufficient amount for 1 unit (need ≥₹{effective_price})"
            ), allocated_amount
        
        # Too expensive per unit
        if effective_price < self.min_unit_value:
            # This is actually fine - low price stocks
            pass
        
        # Calculate actual investment amount
        actual_amount = effective_price * Decimal(str(units))
        unused_amount = allocated_amount - actual_amount
        
        # Ensure non-negative unused (should always be true)
        if unused_amount < Decimal('0'):
            unused_amount = Decimal('0')
        
        return ETFUnitPlan(
            etf_symbol=symbol,
            ltp=ltp,
            effective_price=effective_price,
            units=units,
            actual_amount=actual_amount,
            unused_amount=unused_amount,
            status=ETFStatus.PLANNED,
            reason=None
        ), unused_amount
    
    def _calculate_effective_price(self, ltp: Decimal) -> Decimal:
        """
        Calculate effective price with buffer
        
        Args:
            ltp: Last traded price
        
        Returns:
            Effective price with buffer added
        """
        buffer_multiplier = Decimal('1') + (self.price_buffer_pct / Decimal('100'))
        effective = ltp * buffer_multiplier
        return effective.quantize(Decimal('0.01'))
    
    @staticmethod
    def _calculate_floor_units(amount: Decimal, price: Decimal) -> int:
        """
        Calculate units using floor() - ALWAYS
        
        This is the India-critical function.
        NEVER use ceiling, NEVER round, ALWAYS floor.
        
        Args:
            amount: Allocated amount
            price: Effective price per unit
        
        Returns:
            Number of whole units (floored)
        """
        if price <= Decimal('0'):
            return 0
        
        # Calculate exact units
        exact_units = float(amount / price)
        
        # ALWAYS floor - this is the law in Indian markets
        units = int(math.floor(exact_units))
        
        # Never negative
        return max(0, units)
    
    def calculate_summary(
        self,
        plans: List[ETFUnitPlan]
    ) -> Dict:
        """
        Calculate summary statistics for unit plans
        
        Args:
            plans: List of unit plans
        
        Returns:
            Summary dictionary
        """
        total_planned = sum(p.actual_amount for p in plans if p.status == ETFStatus.PLANNED)
        total_unused = sum(p.unused_amount for p in plans)
        total_units = sum(p.units for p in plans if p.status == ETFStatus.PLANNED)
        
        planned_count = sum(1 for p in plans if p.status == ETFStatus.PLANNED)
        skipped_count = sum(1 for p in plans if p.status == ETFStatus.SKIPPED)
        
        return {
            'total_planned_amount': total_planned,
            'total_unused_amount': total_unused,
            'total_units': total_units,
            'etfs_planned': planned_count,
            'etfs_skipped': skipped_count,
            'plans': [
                {
                    'etf': p.etf_symbol,
                    'units': p.units,
                    'amount': float(p.actual_amount),
                    'status': p.status.value,
                    'reason': p.reason
                }
                for p in plans
            ]
        }
    
    def validate_execution_price(
        self,
        plan: ETFUnitPlan,
        executed_price: Decimal,
        max_slippage_pct: Decimal = Decimal('3.0')
    ) -> Tuple[bool, str]:
        """
        Validate if execution price is within acceptable range
        
        Args:
            plan: Original unit plan
            executed_price: Actual execution price
            max_slippage_pct: Maximum acceptable slippage
        
        Returns:
            Tuple of (is_valid, reason)
        """
        if executed_price <= Decimal('0'):
            return False, "Executed price must be positive"
        
        # Calculate slippage
        slippage = abs(executed_price - plan.ltp) / plan.ltp * Decimal('100')
        
        if slippage > max_slippage_pct:
            return False, f"Slippage {slippage}% exceeds max {max_slippage_pct}%"
        
        # Check if price is within buffered range
        max_acceptable = plan.effective_price
        if executed_price > max_acceptable:
            excess = executed_price - max_acceptable
            return False, f"Price ₹{executed_price} exceeds buffered price ₹{max_acceptable} by ₹{excess}"
        
        return True, "Price acceptable"
    
    def recalculate_with_new_prices(
        self,
        plans: List[ETFUnitPlan],
        new_prices: Dict[str, Decimal]
    ) -> List[ETFUnitPlan]:
        """
        Recalculate plans with updated prices (for refresh scenarios)
        
        Args:
            plans: Original unit plans
            new_prices: Updated price dictionary
        
        Returns:
            New list of unit plans
        """
        # Convert plans back to allocations
        allocations = [
            ETFAllocation(
                etf_symbol=p.etf_symbol,
                allocated_amount=p.actual_amount + p.unused_amount,
                allocation_pct=Decimal('0')  # Not needed for recalc
            )
            for p in plans
        ]
        
        # Recalculate
        new_plans, _ = self.calculate_units(allocations, new_prices)
        return new_plans
