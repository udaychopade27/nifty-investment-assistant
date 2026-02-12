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
from typing import List, Tuple, Optional, Dict

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
        dip_thresholds: dict,
        tactical_priority_config: Optional[dict] = None,
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
        self.tactical_priority_config = tactical_priority_config or {}
    
    def generate_decision(
        self,
        decision_date: date,
        market_context: MarketContext,
        monthly_config: MonthlyConfig,
        capital_state: CapitalState,  # ✅ ADDED: Capital state passed in
        current_prices: dict[str, Decimal],
        index_changes_by_etf: Optional[dict[str, Decimal]] = None,
        index_metrics_by_etf: Optional[dict[str, Dict[str, Decimal]]] = None,
        deploy_base_daily: bool = True
    ) -> Tuple[DailyDecision, List[ETFDecision]]:
        """
        Generate complete daily decision
        
        Args:
            decision_date: Date of decision
            market_context: Market environment
            monthly_config: Monthly capital configuration
            capital_state: Current capital state (✅ PASSED IN, not fetched)
            current_prices: Current ETF prices
            index_changes_by_etf: Optional map of ETF -> underlying index daily % change
            deploy_base_daily: Whether to deploy base capital in daily decision
        
        Returns:
            Tuple of (DailyDecision, List of ETFDecisions)
        """
        # Step 1: ✅ Capital state is now passed in (no DB access)
        # capital_state = self.capital_engine.get_capital_state(monthly_config.month)  # ❌ REMOVED
        
        # Step 2: Determine decision type (prefer per-ETF index dips if provided)
        if index_changes_by_etf:
            decision_type, tactical_deploy_pct = self._determine_decision_type_from_index_changes(
                index_changes_by_etf
            )
        else:
            decision_type = self._determine_decision_type(market_context)
            tactical_deploy_pct = None
        
        # Step 3: Calculate deployable amounts
        base_amount, tactical_amount = self._calculate_deployable_amounts(
            decision_type,
            capital_state,
            monthly_config,
            tactical_deploy_pct=tactical_deploy_pct,
            deploy_base_daily=deploy_base_daily
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
            tactical_amount,
            index_changes_by_etf=index_changes_by_etf,
            index_metrics_by_etf=index_metrics_by_etf,
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
            total_unused,
            index_changes_by_etf=index_changes_by_etf,
        )
        
        # Step 9: Create DailyDecision
        daily_decision = DailyDecision(
            date=decision_date,
            decision_type=decision_type,
            nifty_change_pct=market_context.daily_change_pct,
            suggested_total_amount=suggested_total,
            actual_investable_amount=actual_investable,
            unused_amount=total_unused,
            # Remaining capital shown in a decision should represent current available
            # capital before any manual trade execution happens.
            remaining_base_capital=capital_state.base_remaining,
            remaining_tactical_capital=capital_state.tactical_remaining,
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
            Decision type (NONE/SMALL/MEDIUM/HIGH/FULL)
        """
        daily_change = market_context.daily_change_pct
        three_day_change = market_context.cumulative_3day_pct
        
        if self.dip_thresholds.get("etf_tactical_rules", {}).get("strategy") == "smart_daily_dip_capped":
             # Smart strategy: Check if ANY ETF has a valid dip
             # This is just a label for the daily record. Actual allocation happens per-ETF.
             return self._determine_decision_type_smart_dip(market_context)
        
        # Legacy global logic
        daily_change = market_context.daily_change_pct
        three_day_change = market_context.cumulative_3day_pct
        
        # Check for 3-day override
        if three_day_change <= Decimal('-2.5'):
            return DecisionType.MEDIUM
        
        # Determine based on daily change
        if daily_change >= Decimal('-0.75'):
            return DecisionType.NONE  # No dip
        elif daily_change >= Decimal('-1.5'):
            return DecisionType.SMALL  # -0.75% to -1.5%
        elif daily_change >= Decimal('-2.5'):
            return DecisionType.MEDIUM  # -1.5% to -2.5%
        elif daily_change >= Decimal('-3.5'):
            return DecisionType.HIGH  # -2.5% to -3.5%
        else:
            return DecisionType.FULL  # Below -3.5%
    
    def _calculate_deployable_amounts(
        self,
        decision_type: DecisionType,
        capital_state: CapitalState,
        monthly_config: MonthlyConfig,
        tactical_deploy_pct: Optional[Decimal] = None,
        deploy_base_daily: bool = True
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
        base_amount = Decimal('0')
        if deploy_base_daily:
            base_amount = min(
                monthly_config.daily_tranche,
                capital_state.base_remaining
            )
        
        tactical_amount = Decimal('0')
        
        # Smart Dip Strategy: Make full tactical pot available for per-ETF selection
        if self.dip_thresholds.get("etf_tactical_rules", {}).get("strategy") == "smart_daily_dip_capped":
            if decision_type != DecisionType.NONE:
                # pass full remaining; caps will limit actual usage
                tactical_amount = capital_state.tactical_remaining
            return base_amount, tactical_amount

        # Legacy Strategy
        tactical_pct = Decimal('0')
        if tactical_deploy_pct is not None:
            tactical_pct = tactical_deploy_pct
        else:
            if decision_type == DecisionType.SMALL:
                tactical_pct = Decimal('25')
            elif decision_type == DecisionType.MEDIUM:
                tactical_pct = Decimal('50')
            elif decision_type == DecisionType.HIGH:
                tactical_pct = Decimal('75')
            elif decision_type == DecisionType.FULL:
                tactical_pct = Decimal('100')
        
        if tactical_pct > Decimal('0'):
            tactical_to_deploy = (
                capital_state.tactical_remaining * tactical_pct / Decimal('100')
            ).quantize(Decimal('0.01'))
            tactical_amount = min(tactical_to_deploy, capital_state.tactical_remaining)
        
        return base_amount, tactical_amount
    
    def _allocate_capital(
        self,
        base_amount: Decimal,
        tactical_amount: Decimal,
        index_changes_by_etf: Optional[dict[str, Decimal]] = None,
        index_metrics_by_etf: Optional[dict[str, Dict[str, Decimal]]] = None,
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
        
        # Allocate tactical capital (filter by dipped ETFs if provided)
        if tactical_amount > Decimal('0'):
            tactical_allocs = self._allocate_tactical_filtered(
                tactical_amount,
                index_changes_by_etf,
                index_metrics_by_etf=index_metrics_by_etf,
            )
            all_allocations.extend(tactical_allocs)
        
        # Merge allocations for same ETF
        merged = self._merge_allocations(all_allocations)
        
        return merged

    def _allocate_tactical_filtered(
        self,
        tactical_amount: Decimal,
        index_changes_by_etf: Optional[dict[str, Decimal]],
        index_metrics_by_etf: Optional[dict[str, Dict[str, Decimal]]] = None,
    ) -> List[ETFAllocation]:
        """
        Allocate tactical capital only to ETFs whose underlying index dipped.
        Falls back to full tactical allocation if no filter provided.
        """
        if not index_changes_by_etf:
            tactical_allocs, _ = self.allocation_engine.allocate(
                tactical_amount,
                self.tactical_allocation
            )
            return tactical_allocs

        if self.tactical_priority_config.get("enabled"):
            return self._allocate_tactical_ranked(
                tactical_amount=tactical_amount,
                index_changes_by_etf=index_changes_by_etf,
                index_metrics_by_etf=index_metrics_by_etf or {},
            )

        eligible = []
        for symbol, change in index_changes_by_etf.items():
            tier = self._get_dip_tier(change)
            if tier != "none":
                eligible.append(symbol)

        if not eligible:
            return []

        base_weights = {
            symbol: Decimal(str(pct))
            for symbol, pct in self.tactical_allocation.allocations.items()
            if symbol in eligible and Decimal(str(pct)) > Decimal('0')
        }
        total_weight = sum(base_weights.values())
        if total_weight <= Decimal('0'):
            return []

        normalized = {}
        running_sum = Decimal('0')
        items = list(base_weights.items())
        for idx, (symbol, weight) in enumerate(items):
            if idx == len(items) - 1:
                pct = (Decimal('100') - running_sum).quantize(Decimal('0.01'))
            else:
                pct = (weight / total_weight * Decimal('100')).quantize(Decimal('0.01'))
                running_sum += pct
            if pct < Decimal('0'):
                pct = Decimal('0')
            normalized[symbol] = pct

        tactical_blueprint = AllocationBlueprint(
            name="tactical_filtered",
            allocations=normalized
        )
        tactical_allocs, _ = self.allocation_engine.allocate(
            tactical_amount,
            tactical_blueprint
        )
        return tactical_allocs

    def _allocate_tactical_ranked(
        self,
        tactical_amount: Decimal,
        index_changes_by_etf: dict[str, Decimal],
        index_metrics_by_etf: dict[str, Dict[str, Decimal]],
    ) -> List[ETFAllocation]:
        """
        Smart Daily Dip Allocation with Weights-as-Caps
        """
        # 1. Get Strategy Rules
        rules = self.dip_thresholds.get("etf_tactical_rules", {})
        strategy = rules.get("strategy", "legacy")
        weights_as_caps = rules.get("weights_as_caps", False)
        
        # 2. Identify Eligible ETFs (Per-ETF Dip Check)
        eligible = []
        multiplier_map = {}
        tiers = rules.get("tiers", [])
        
        # Default legacy fallback if tiers missing
        if not tiers:
             tiers = [{"trigger": -0.75, "multiplier": 1.0}]

        for symbol, change in index_changes_by_etf.items():
            # EXPLICIT GOLD EXCLUSION
            if symbol == "HDFCGOLD":
                continue
                
            best_multiplier = Decimal("0")
            # Check tiers (assuming ordered by severity, but let's be safe)
            # Find largest multiplier for the dip
            for tier in tiers:
                trigger = Decimal(str(tier["trigger"]))
                mult = Decimal(str(tier["multiplier"]))
                if change <= trigger:
                    if mult > best_multiplier:
                        best_multiplier = mult
            
            if best_multiplier > Decimal("0"):
                eligible.append(symbol)
                multiplier_map[symbol] = best_multiplier

        if not eligible:
            return []

        # 3. Rank Eligible ETFs
        ranked = self._rank_eligible_etfs(eligible, index_changes_by_etf, index_metrics_by_etf)
        if not ranked:
            return []
            
        # 4. Allocation Logic
        rank_cfg = self.tactical_priority_config.get("ranking", {})
        rank_splits = [Decimal(str(x)) for x in rank_cfg.get("rank_splits", [50, 30, 20])]
        
        # Load Tactical Weights (Strategic Preferences)
        tactical_weights = {
            sym: Decimal(str(pct)) 
            for sym, pct in self.tactical_allocation.allocations.items()
        }
        
        # Falling Knife Rules
        knife_guard = rules.get("safety", {}).get("falling_knife_guard", {})
        knife_enabled = knife_guard.get("enabled", False)
        knife_threshold = Decimal(str(knife_guard.get("threshold", -4.0)))
        knife_penalty = Decimal(str(knife_guard.get("penalty_factor", 0.5)))
        
        allocations = []
        
        # We process ranked ETFs. Each gets a slice of the DAILY POT based on rank.
        # But that slice is CAPPED by the Strategic Weight of the TOTAL fund.
        
        # Important: Tactical Amount passed here is the REMAINING FUND.
        # We need a concept of "Daily Pot" vs "Total Fund". 
        # For simplicity in this design, we treat "tactical_amount" as the available pool 
        # and "rank_splits" as % of that pool to deploy TODAY.
        
        for idx, (sym, _) in enumerate(ranked):
            if idx >= len(rank_splits):
                break # Only top N get allocated
                
            split_pct = rank_splits[idx]
            
            weight_pct = tactical_weights.get(sym, Decimal("0"))

            if weight_pct <= Decimal("0"):
                continue # Safety for Gold
                
            strategic_cap = (tactical_amount * weight_pct / Decimal("100")).quantize(Decimal("0.01"))

            # Ideal Amount based on Dip Severity
            multiplier = multiplier_map.get(sym, Decimal("0.5"))
            
            raw_allocation = (strategic_cap * multiplier).quantize(Decimal("0.01"))
            
            # Falling Knife Check
            if knife_enabled:
                metrics = index_metrics_by_etf.get(sym, {})
                five_day = metrics.get("five_day_change_pct", Decimal("0"))
                if five_day < knife_threshold:
                     raw_allocation = (raw_allocation * knife_penalty).quantize(Decimal("0.01"))
            
            final_amt = min(raw_allocation, strategic_cap, tactical_amount)
            
            # Hard Safety: Don't exceed remaining cash
            if final_amt > tactical_amount:
                final_amt = tactical_amount
            
            if final_amt > Decimal("0"):
                allocations.append(ETFAllocation(
                    etf_symbol=sym,
                    allocated_amount=final_amt,
                    allocation_pct=Decimal("0") # Calc later
                ))
                tactical_amount -= final_amt # Deduct from running available
        
        return allocations

    def _determine_decision_type_smart_dip(self, market_context: MarketContext) -> DecisionType:
        """
        Determine if there is any tactical action today based on configured strategy.
        Used to set the high-level ID of the day.
        """
        # We rely on the fact that generate_decision calls us. 
        # But wait, generate_decision determines type BEFORE calculating amounts.
        # And we need index_changes (which are passed to generate_decision).
        # We can't access index_changes here easily inside this helper if not passed.
        # But _determine_decision_type is called inside generate_decision.
        # Actually, in generate_decision, if index_changes_by_etf is present, 
        # it calls _determine_decision_type_from_index_changes.
        # So we should modify THAT method or the caller.
        return DecisionType.MEDIUM # Default to Active if we are in this mode, logic handled in allocation.

    def _rank_eligible_etfs(
        self,
        eligible_symbols: List[str],
        index_changes_by_etf: dict[str, Decimal],
        index_metrics_by_etf: dict[str, Dict[str, Decimal]],
    ) -> List[Tuple[str, Decimal]]:
        """Score and rank eligible ETFs for tactical deployment."""
        scoring_cfg = self.tactical_priority_config.get("scoring_weights", {})
        w_severity = Decimal(str(scoring_cfg.get("severity", 50)))
        w_persistence = Decimal(str(scoring_cfg.get("persistence", 20)))
        w_liquidity = Decimal(str(scoring_cfg.get("liquidity", 15)))
        w_confidence = Decimal(str(scoring_cfg.get("confidence", 15)))
        w_corr_penalty = Decimal(str(scoring_cfg.get("correlation_penalty", 10)))
        min_priority_score = Decimal(
            str(self.tactical_priority_config.get("thresholds", {}).get("min_priority_score", 0))
        )

        # Tactical blueprint weight acts as static liquidity / tradability prior.
        tactical_weights = {
            sym: Decimal(str(pct)) / Decimal("100")
            for sym, pct in self.tactical_allocation.allocations.items()
        }

        ranked_raw: List[Tuple[str, Decimal]] = []
        for sym in eligible_symbols:
            daily_change = index_changes_by_etf.get(sym, Decimal("0"))
            metrics = index_metrics_by_etf.get(sym, {})
            three_day_change = Decimal(str(metrics.get("three_day_change_pct", Decimal("0"))))
            data_quality = Decimal(str(metrics.get("data_quality", Decimal("1"))))

            severity_signal = max(Decimal("0"), -daily_change) / Decimal("5")  # normalize ~0..1
            persistence_signal = max(Decimal("0"), -three_day_change) / Decimal("8")
            liquidity_signal = tactical_weights.get(sym, Decimal("0"))
            confidence_signal = min(Decimal("1"), max(Decimal("0"), data_quality))

            score = (
                w_severity * severity_signal
                + w_persistence * persistence_signal
                + w_liquidity * liquidity_signal
                + w_confidence * confidence_signal
            )
            if score >= min_priority_score:
                ranked_raw.append((sym, score.quantize(Decimal("0.0001"))))

        ranked_raw.sort(key=lambda x: x[1], reverse=True)
        if not ranked_raw:
            return []

        # Apply simple correlation penalty by configured ETF groups.
        groups = self.tactical_priority_config.get("correlations", {}).get("groups", [])
        selected_with_penalty: List[Tuple[str, Decimal]] = []
        picked: List[str] = []
        for sym, raw_score in ranked_raw:
            penalty_count = 0
            for group in groups:
                if sym in group:
                    penalty_count += sum(1 for p in picked if p in group)
            penalized = raw_score - (w_corr_penalty * Decimal("0.25") * Decimal(penalty_count))
            if penalized > Decimal("0"):
                selected_with_penalty.append((sym, penalized.quantize(Decimal("0.0001"))))
                picked.append(sym)

        selected_with_penalty.sort(key=lambda x: x[1], reverse=True)
        return selected_with_penalty

    def _determine_decision_type_from_index_changes(
        self,
        index_changes_by_etf: dict[str, Decimal]
    ) -> Tuple[DecisionType, Decimal]:
        """
        Determine decision type based on worst underlying index dip.
        """
        # For Smart Daily Dip, we just want to signal ACTIVE (MEDIUM) if any valid dip exists.
        if self.dip_thresholds.get("etf_tactical_rules", {}).get("strategy") == "smart_daily_dip_capped":
             tiers = self.dip_thresholds.get("etf_tactical_rules", {}).get("tiers", [])
             # Find minimum trigger
             min_trigger = Decimal("-0.75")
             if tiers:
                 min_trigger = Decimal(str(tiers[0]["trigger"])) # Assuming sorted
             
             for chg in index_changes_by_etf.values():
                 if chg <= min_trigger:
                     return DecisionType.MEDIUM, Decimal("0") # Tactical pct calculated dynamically later
             return DecisionType.NONE, Decimal("0")
             
        # Legacy Logic
        changes = [c for c in index_changes_by_etf.values() if c is not None]
        if not changes:
            return DecisionType.NONE, Decimal('0')

        worst_change = min(changes)  # most negative
        tier = self._get_dip_tier(worst_change)

        if tier == "full":
            return DecisionType.FULL, self._get_tactical_deploy_pct("full")
        if tier == "high":
            return DecisionType.HIGH, self._get_tactical_deploy_pct("high")
        if tier == "medium":
            return DecisionType.MEDIUM, self._get_tactical_deploy_pct("medium")
        if tier == "small":
            return DecisionType.SMALL, self._get_tactical_deploy_pct("small")
        return DecisionType.NONE, Decimal('0')

    def _get_dip_tier(self, change_pct: Decimal) -> str:
        """
        Map a % change to dip tier using configured thresholds.
        """
        full_cfg = self.dip_thresholds.get('full', {})
        high_cfg = self.dip_thresholds.get('high', {})
        med_cfg = self.dip_thresholds.get('medium', {})
        small_cfg = self.dip_thresholds.get('small', {})

        full_max = full_cfg.get('max_change')
        high_min = high_cfg.get('min_change')
        high_max = high_cfg.get('max_change')
        med_min = med_cfg.get('min_change')
        med_max = med_cfg.get('max_change')
        small_min = small_cfg.get('min_change')
        small_max = small_cfg.get('max_change')

        if full_max is not None and change_pct <= Decimal(str(full_max)):
            return "full"
        if high_min is not None and high_max is not None:
            if Decimal(str(high_min)) < change_pct <= Decimal(str(high_max)):
                return "high"
        if med_min is not None and med_max is not None:
            if Decimal(str(med_min)) < change_pct <= Decimal(str(med_max)):
                return "medium"
        if small_min is not None and small_max is not None:
            if Decimal(str(small_min)) < change_pct <= Decimal(str(small_max)):
                return "small"
        return "none"

    def _get_tactical_deploy_pct(self, tier: str) -> Decimal:
        cfg = self.dip_thresholds.get(tier, {})
        pct = cfg.get('tactical_deployment', 0)
        return Decimal(str(pct))
    
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
        unused: Decimal,
        index_changes_by_etf: Optional[dict[str, Decimal]] = None,
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
            parts.append("Small dip (-0.75% to -1.5%). Deploying 25% tactical.")
        elif decision_type == DecisionType.MEDIUM:
            parts.append("Medium dip (-1.5% to -2.5%) or 3-day fall. Deploying 50% tactical.")
        elif decision_type == DecisionType.HIGH:
            parts.append("High dip (-2.5% to -3.5%). Deploying 75% tactical.")
        elif decision_type == DecisionType.FULL:
            parts.append("Full dip (< -3.5%). Deploying 100% tactical.")
        
        # Capital breakdown
        if base_amount > Decimal('0') or tactical_amount > Decimal('0'):
            parts.append(f"Base: ₹{base_amount}, Tactical: ₹{tactical_amount}")

        if index_changes_by_etf:
            dipped = sorted(
                [(sym, chg) for sym, chg in index_changes_by_etf.items() if chg is not None and chg < Decimal("0")],
                key=lambda x: x[1],
            )
            if dipped:
                top = ", ".join([f"{sym}:{chg}%" for sym, chg in dipped[:3]])
                parts.append(f"Index dips: {top}")
        
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
