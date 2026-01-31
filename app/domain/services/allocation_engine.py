"""
ALLOCATION ENGINE (ENGINE-3)
Convert deployable capital → ETF-wise amounts

RESPONSIBILITIES:
- Distribute capital across ETFs per blueprint
- Apply risk constraints
- Validate allocations
- NO UNIT CALCULATION, NO PRICES

RULES:
❌ No unit calculation
❌ No prices
❌ No execution assumptions
✅ Respect caps (max equity 75%, max single ETF 45%, etc.)
✅ Pure allocation logic
✅ Deterministic output
"""

from decimal import Decimal
from typing import List, Dict, Tuple

from app.domain.models import (
    ETFAllocation,
    AllocationBlueprint,
    RiskConstraints,
    AssetClass,
    ETF
)


class AllocationEngine:
    """
    Allocation Engine
    Distributes capital across ETFs based on allocation blueprint
    """
    
    def __init__(
        self,
        risk_constraints: RiskConstraints,
        etf_universe: Dict[str, ETF]
    ):
        """
        Initialize allocation engine
        
        Args:
            risk_constraints: Investment risk constraints
            etf_universe: Dictionary of symbol -> ETF
        """
        self.risk_constraints = risk_constraints
        self.etf_universe = etf_universe
    
    def allocate(
        self,
        deployable_amount: Decimal,
        blueprint: AllocationBlueprint,
        enforce_constraints: bool = True
    ) -> Tuple[List[ETFAllocation], List[str]]:
        """
        Allocate amount across ETFs per blueprint
        
        Args:
            deployable_amount: Total amount to allocate
            blueprint: Allocation percentages blueprint
            enforce_constraints: Whether to enforce risk constraints
        
        Returns:
            Tuple of (allocations list, warnings list)
        """
        if deployable_amount <= Decimal('0'):
            return [], ["Deployable amount must be positive"]
        
        allocations = []
        warnings = []
        
        # Calculate allocations
        for etf_symbol, pct in blueprint.allocations.items():
            if pct == Decimal('0'):
                continue
            
            # Calculate allocated amount
            allocated = self._calculate_allocation(
                deployable_amount,
                pct
            )
            
            if allocated > Decimal('0'):
                allocations.append(ETFAllocation(
                    etf_symbol=etf_symbol,
                    allocated_amount=allocated,
                    allocation_pct=pct
                ))
        
        # Validate constraints if required
        if enforce_constraints:
            is_valid, constraint_warnings = self._validate_constraints(allocations)
            if not is_valid:
                warnings.extend(constraint_warnings)
        
        # Verify total allocation
        total_allocated = sum(a.allocated_amount for a in allocations)
        if abs(total_allocated - deployable_amount) > Decimal('0.01'):
            diff = deployable_amount - total_allocated
            warnings.append(f"Allocation discrepancy: ₹{diff}")
        
        return allocations, warnings
    
    @staticmethod
    def _calculate_allocation(
        total_amount: Decimal,
        percentage: Decimal
    ) -> Decimal:
        """
        Calculate allocation amount for a percentage
        
        Args:
            total_amount: Total deployable amount
            percentage: Allocation percentage (0-100)
        
        Returns:
            Allocated amount (rounded to 2 decimals)
        """
        allocated = (total_amount * percentage / Decimal('100'))
        return allocated.quantize(Decimal('0.01'))
    
    def _validate_constraints(
        self,
        allocations: List[ETFAllocation]
    ) -> Tuple[bool, List[str]]:
        """
        Validate allocations against risk constraints
        
        Args:
            allocations: List of ETF allocations
        
        Returns:
            Tuple of (is_valid, list of warnings)
        """
        warnings = []
        
        # Calculate totals by asset class
        equity_total = Decimal('0')
        debt_total = Decimal('0')
        gold_total = Decimal('0')
        total_allocated = Decimal('0')
        
        for allocation in allocations:
            etf = self.etf_universe.get(allocation.etf_symbol)
            if etf is None:
                warnings.append(f"Unknown ETF: {allocation.etf_symbol}")
                continue
            
            total_allocated += allocation.allocated_amount
            
            # Sum by asset class
            if etf.asset_class == AssetClass.EQUITY:
                equity_total += allocation.allocated_amount
            elif etf.asset_class == AssetClass.DEBT:
                debt_total += allocation.allocated_amount
            elif etf.asset_class == AssetClass.GOLD:
                gold_total += allocation.allocated_amount
        
        if total_allocated == Decimal('0'):
            return True, warnings
        
        # Calculate percentages
        equity_pct = (equity_total / total_allocated * Decimal('100')).quantize(Decimal('0.01'))
        debt_pct = (debt_total / total_allocated * Decimal('100')).quantize(Decimal('0.01'))
        gold_pct = (gold_total / total_allocated * Decimal('100')).quantize(Decimal('0.01'))
        
        # Check equity constraint
        if equity_pct > self.risk_constraints.max_equity_allocation:
            warnings.append(
                f"Equity allocation {equity_pct}% exceeds max {self.risk_constraints.max_equity_allocation}%"
            )
        
        # Check debt constraint
        if debt_pct < self.risk_constraints.min_debt:
            warnings.append(
                f"Debt allocation {debt_pct}% below min {self.risk_constraints.min_debt}%"
            )
        
        # Check gold constraint
        if gold_pct > self.risk_constraints.max_gold:
            warnings.append(
                f"Gold allocation {gold_pct}% exceeds max {self.risk_constraints.max_gold}%"
            )
        
        # Check single ETF constraint
        for allocation in allocations:
            etf_pct = (allocation.allocated_amount / total_allocated * Decimal('100')).quantize(Decimal('0.01'))
            
            if etf_pct > self.risk_constraints.max_single_etf:
                warnings.append(
                    f"{allocation.etf_symbol} allocation {etf_pct}% exceeds max single ETF {self.risk_constraints.max_single_etf}%"
                )
            
            # Check mid-cap specific constraint
            etf = self.etf_universe.get(allocation.etf_symbol)
            if etf and etf.category == 'growth' and 'MIDCAP' in etf.symbol.upper():
                if etf_pct > self.risk_constraints.max_midcap:
                    warnings.append(
                        f"Mid-cap allocation {etf_pct}% exceeds max {self.risk_constraints.max_midcap}%"
                    )
        
        is_valid = len(warnings) == 0
        return is_valid, warnings
    
    def get_allocation_breakdown(
        self,
        allocations: List[ETFAllocation]
    ) -> Dict[str, Dict[str, Decimal]]:
        """
        Get detailed breakdown of allocations
        
        Args:
            allocations: List of ETF allocations
        
        Returns:
            Dictionary with breakdown by asset class and ETF
        """
        total = sum(a.allocated_amount for a in allocations)
        
        breakdown = {
            'total': total,
            'by_asset_class': {},
            'by_etf': {}
        }
        
        # By asset class
        for allocation in allocations:
            etf = self.etf_universe.get(allocation.etf_symbol)
            if etf:
                asset_class = etf.asset_class.value
                if asset_class not in breakdown['by_asset_class']:
                    breakdown['by_asset_class'][asset_class] = Decimal('0')
                breakdown['by_asset_class'][asset_class] += allocation.allocated_amount
        
        # By ETF
        for allocation in allocations:
            breakdown['by_etf'][allocation.etf_symbol] = {
                'amount': allocation.allocated_amount,
                'percentage': allocation.allocation_pct
            }
        
        return breakdown
