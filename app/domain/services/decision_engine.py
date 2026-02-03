"""
DECISION ENGINE (ENGINE-5) - CORE ORCHESTRATOR
Daily investment decision generation

RESPONSIBILITIES:
- Orchestrate all engines
- Determine decision type (NONE/SMALL/MEDIUM/FULL)
- Generate complete daily decision

RULES:
❌ No execution
❌ No state mutation
❌ No retries
❌ No DB access (CRITICAL FIX)
✅ Idempotent
✅ Persist even if no investment
✅ Always explain
"""

from datetime import date, datetime
from decimal import Decimal
from typing import List, Tuple, Optional

from app.domain.models import (
    DailyDecision,
    ETFDecision,
    DecisionType,
    MarketContext,
    CapitalState,
    ETFAllocation,
    ETFUnitPlan,
    MonthlyConfig
)
from app.domain.services.market_context_engine import MarketContextEngine
from app.domain.services.allocation_engine import AllocationEngine
from app.domain.services.unit_calculation_engine import UnitCalculationEngine
from app.utils.time import now_ist_naive
from app.domain.models import AllocationBlueprint


class DecisionEngine:
    """
    Decision Engine - The Brain
    Orchestrates all engines to produce daily decision
    
    ✅ FIXED: No longer accesses DB via CapitalEngine
    ✅ FIXED: CapitalState is passed in as parameter
    """
    
    def __init__(
        self,
        market_context_engine: MarketContextEngine,
        allocation_engine: AllocationEngine,
        unit_calculation_engine: UnitCalculationEngine,
        base_allocation: AllocationBlueprint,
        tactical_allocation: AllocationBlueprint,
        strategy_version: str,
        dip_thresholds: dict
    ):
        """
        Initialize decision engine with all dependencies
        
        ✅ REMOVED: capital_engine (violates pure function rules)
        """
        self.market_context_engine = market_context_engine
        # ✅ capital_engine removed - capital state will be passed in
        self.allocation_engine = allocation_engine
        self.unit_calculation_engine = unit_calculation_engine
        self.base_allocation = base_allocation
        self.tactical_allocation = tactical_allocation
        self.strategy_version = strategy_version
        self.dip_thresholds = dip_thresholds
    
    def generate_decision(
        self,
        decision_date: date,
        market_context: MarketContext,
        monthly_config: MonthlyConfig,
        capital_state: CapitalState,  # ✅ ADDED: Capital state passed in
        current_prices: dict[str, Decimal]
    ) -> Tuple[DailyDecision, List[ETFDecision]]:
        """
        Generate complete daily decision
        
        Args:
            decision_date: Date of decision
            market_context: Market environment
            monthly_config: Monthly capital configuration
            capital_state: Current capital state (✅ PASSED IN, not fetched)
            current_prices: Current ETF prices
        
        Returns:
            Tuple of (DailyDecision, List of ETFDecisions)
        """
        # Step 1: ✅ Capital state is now passed in (no DB access)
        # capital_state = self.capital_engine.get_capital_state(monthly_config.month)  # ❌ REMOVED
        
        # Step 2: Determine decision type
        decision_type = self._determine_decision_type(market_context)
        
        # Step 3: Calculate deployable amounts
        base_amount, tactical_amount = self._calculate_deployable_amounts(
            decision_type,
            capital_state,
            monthly_config
        )
        
        suggested_total = base_amount + tactical_amount
        
        # Step 4: If no deployment, create NONE decision
        if suggested_total <= Decimal('0'):
            return self._create_none_decision(
                decision_date,
                market_context,
                capital_state
            ), []
        
        # Step 5: Allocate to ETFs
        allocations = self._allocate_capital(
            base_amount,
            tactical_amount
        )
        
        # Step 6: Calculate units
        unit_plans, total_unused = self.unit_calculation_engine.calculate_units(
            allocations,
            current_prices
        )
        
        # Step 7: Calculate actual investable amount
        actual_investable = sum(
            p.actual_amount for p in unit_plans if p.units > 0
        )
        
        # Step 8: Generate explanation
        explanation = self._generate_explanation(
            decision_type,
            market_context,
            base_amount,
            tactical_amount,
            actual_investable,
            total_unused
        )
        
        # Step 9: Create DailyDecision
        daily_decision = DailyDecision(
            date=decision_date,
            decision_type=decision_type,
            nifty_change_pct=market_context.daily_change_pct,
            suggested_total_amount=suggested_total,
            actual_investable_amount=actual_investable,
            unused_amount=total_unused,
            remaining_base_capital=capital_state.base_remaining - base_amount,
            remaining_tactical_capital=capital_state.tactical_remaining - tactical_amount,
            explanation=explanation,
            strategy_version=self.strategy_version,
            created_at=now_ist_naive()
        )
        
        # Step 10: Create ETFDecisions
        etf_decisions = self._create_etf_decisions(unit_plans)
        
        return daily_decision, etf_decisions
    
    def _determine_decision_type(
        self,
        market_context: MarketContext
    ) -> DecisionType:
        """
        Determine decision type based on market context
        
        Args:
            market_context: Market environment
        
        Returns:
            Decision type (NONE/SMALL/MEDIUM/FULL)
        """
        daily_change = market_context.daily_change_pct
        three_day_change = market_context.cumulative_3day_pct
        
        # Check for 3-day override
        if three_day_change <= Decimal('-2.5'):
            return DecisionType.MEDIUM
        
        # Determine based on daily change
        if daily_change >= Decimal('-1.0'):
            return DecisionType.NONE  # No dip
        elif daily_change >= Decimal('-2.0'):
            return DecisionType.SMALL  # -1% to -2%
        elif daily_change >= Decimal('-3.0'):
            return DecisionType.MEDIUM  # -2% to -3%
        else:
            return DecisionType.FULL  # Below -3%
    
    def _calculate_deployable_amounts(
        self,
        decision_type: DecisionType,
        capital_state: CapitalState,
        monthly_config: MonthlyConfig
    ) -> Tuple[Decimal, Decimal]:
        """
        Calculate base and tactical amounts to deploy
        
        Args:
            decision_type: Type of decision
            capital_state: Current capital state
            monthly_config: Monthly configuration
        
        Returns:
            Tuple of (base_amount, tactical_amount)
        """
        # Base amount (always invest if capital remaining)
        base_amount = min(
            monthly_config.daily_tranche,
            capital_state.base_remaining
        )
        
        # Tactical amount based on decision type
        tactical_pct = Decimal('0')
        if decision_type == DecisionType.SMALL:
            tactical_pct = Decimal('25')
        elif decision_type == DecisionType.MEDIUM:
            tactical_pct = Decimal('50')
        elif decision_type == DecisionType.FULL:
            tactical_pct = Decimal('100')
        
        tactical_amount = Decimal('0')
        if tactical_pct > Decimal('0'):
            tactical_to_deploy = (
                capital_state.tactical_remaining * tactical_pct / Decimal('100')
            ).quantize(Decimal('0.01'))
            tactical_amount = min(tactical_to_deploy, capital_state.tactical_remaining)
        
        return base_amount, tactical_amount
    
    def _allocate_capital(
        self,
        base_amount: Decimal,
        tactical_amount: Decimal
    ) -> List[ETFAllocation]:
        """
        Allocate base and tactical capital to ETFs
        
        Args:
            base_amount: Base capital to allocate
            tactical_amount: Tactical capital to allocate
        
        Returns:
            Combined list of allocations
        """
        all_allocations = []
        
        # Allocate base capital
        if base_amount > Decimal('0'):
            base_allocs, _ = self.allocation_engine.allocate(
                base_amount,
                self.base_allocation
            )
            all_allocations.extend(base_allocs)
        
        # Allocate tactical capital
        if tactical_amount > Decimal('0'):
            tactical_allocs, _ = self.allocation_engine.allocate(
                tactical_amount,
                self.tactical_allocation
            )
            all_allocations.extend(tactical_allocs)
        
        # Merge allocations for same ETF
        merged = self._merge_allocations(all_allocations)
        
        return merged
    
    @staticmethod
    def _merge_allocations(
        allocations: List[ETFAllocation]
    ) -> List[ETFAllocation]:
        """
        Merge multiple allocations for same ETF
        
        Args:
            allocations: List of allocations (may have duplicates)
        
        Returns:
            Merged allocations
        """
        merged_dict = {}
        
        for alloc in allocations:
            if alloc.etf_symbol in merged_dict:
                # Add to existing
                existing = merged_dict[alloc.etf_symbol]
                merged_dict[alloc.etf_symbol] = ETFAllocation(
                    etf_symbol=alloc.etf_symbol,
                    allocated_amount=existing.allocated_amount + alloc.allocated_amount,
                    allocation_pct=Decimal('0')  # Recalculated later if needed
                )
            else:
                merged_dict[alloc.etf_symbol] = alloc
        
        return list(merged_dict.values())
    
    def _generate_explanation(
        self,
        decision_type: DecisionType,
        market_context: MarketContext,
        base_amount: Decimal,
        tactical_amount: Decimal,
        actual_investable: Decimal,
        unused: Decimal
    ) -> str:
        """
        Generate human-readable explanation
        
        Args:
            decision_type: Decision type
            market_context: Market context
            base_amount: Base deployment
            tactical_amount: Tactical deployment
            actual_investable: Actual investable after unit calculation
            unused: Unused amount
        
        Returns:
            Explanation string
        """
        parts = []
        
        # Market condition
        change = market_context.daily_change_pct
        parts.append(f"NIFTY: {change}%")
        
        # Decision type explanation
        if decision_type == DecisionType.NONE:
            parts.append("No dip detected. Base capital only.")
        elif decision_type == DecisionType.SMALL:
            parts.append("Small dip (-1% to -2%). Deploying 25% tactical.")
        elif decision_type == DecisionType.MEDIUM:
            parts.append("Medium dip (-2% to -3%) or 3-day fall. Deploying 50% tactical.")
        elif decision_type == DecisionType.FULL:
            parts.append("Full dip (>-3%). Deploying 100% tactical.")
        
        # Capital breakdown
        if base_amount > Decimal('0') or tactical_amount > Decimal('0'):
            parts.append(f"Base: ₹{base_amount}, Tactical: ₹{tactical_amount}")
        
        # Actual vs suggested
        suggested = base_amount + tactical_amount
        if actual_investable < suggested:
            parts.append(f"Investable: ₹{actual_investable} (₹{unused} unused due to unit constraints)")
        
        return " | ".join(parts)
    
    def _create_none_decision(
        self,
        decision_date: date,
        market_context: MarketContext,
        capital_state: CapitalState
    ) -> DailyDecision:
        """
        Create a NONE decision (no investment today)
        
        Args:
            decision_date: Decision date
            market_context: Market context
            capital_state: Capital state
        
        Returns:
            DailyDecision with NONE type
        """
        explanation = f"NIFTY: {market_context.daily_change_pct}% | No investment today"
        
        if capital_state.base_remaining <= Decimal('0'):
            explanation += " | Base capital exhausted"
        
        return DailyDecision(
            date=decision_date,
            decision_type=DecisionType.NONE,
            nifty_change_pct=market_context.daily_change_pct,
            suggested_total_amount=Decimal('0'),
            actual_investable_amount=Decimal('0'),
            unused_amount=Decimal('0'),
            remaining_base_capital=capital_state.base_remaining,
            remaining_tactical_capital=capital_state.tactical_remaining,
            explanation=explanation,
            strategy_version=self.strategy_version,
            created_at=now_ist_naive()
        )
    
    @staticmethod
    def _create_etf_decisions(
        unit_plans: List[ETFUnitPlan]
    ) -> List[ETFDecision]:
        """
        Convert unit plans to ETF decisions
        
        Args:
            unit_plans: List of unit plans
        
        Returns:
            List of ETF decisions
        """
        decisions = []
        
        for plan in unit_plans:
            decision = ETFDecision(
                daily_decision_id=0,  # Will be set when persisting
                etf_symbol=plan.etf_symbol,
                ltp=plan.ltp,
                effective_price=plan.effective_price,
                units=plan.units,
                actual_amount=plan.actual_amount,
                status=plan.status,
                reason=plan.reason,
                created_at=now_ist_naive()
            )
            decisions.append(decision)
        
        return decisions
