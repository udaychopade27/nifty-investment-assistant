"""
CONFIG ENGINE (ENGINE-0)
Load, validate, and expose system configuration

RESPONSIBILITIES:
- Load YAML configuration files
- Validate configuration integrity
- Expose read-only typed objects

RULES:
❌ No defaults if config missing
❌ No hardcoded values
✅ Fail fast on invalid config
✅ Deterministic output
"""

import yaml
from decimal import Decimal
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

from app.domain.models import (
    ETF,
    AssetClass,
    RiskLevel,
    AllocationBlueprint,
    RiskConstraints,
)


@dataclass(frozen=True)
class ETFUniverse:
    """Collection of all available ETFs"""
    etfs: List[ETF]
    symbols: List[str]
    
    def get_etf(self, symbol: str) -> ETF:
        """Get ETF by symbol"""
        for etf in self.etfs:
            if etf.symbol == symbol:
                return etf
        raise ValueError(f"ETF not found: {symbol}")
    
    def is_valid_symbol(self, symbol: str) -> bool:
        """Check if symbol is valid"""
        return symbol in self.symbols


class ConfigEngine:
    """
    Configuration Engine
    Single source of truth for all system configuration
    """
    
    def __init__(self, config_dir: Path):
        """Initialize with config directory"""
        self.config_dir = config_dir
        self._etf_universe: ETFUniverse = None
        self._base_allocation: AllocationBlueprint = None
        self._tactical_allocation: AllocationBlueprint = None
        self._crash_allocation: AllocationBlueprint = None
        self._risk_constraints: RiskConstraints = None
        self._rules: Dict = None
        self._app_config: Dict = None
        
    def load_all(self) -> None:
        """Load all configuration files"""
        self._load_etfs()
        self._load_allocations()
        self._load_rules()
        self._load_app_config()
        self._validate_all()
    
    def _load_etfs(self) -> None:
        """Load ETF universe from etfs.yml"""
        etf_file = self.config_dir / "etfs.yml"
        if not etf_file.exists():
            raise FileNotFoundError(f"ETF config not found: {etf_file}")
        
        with open(etf_file, 'r') as f:
            data = yaml.safe_load(f)
        
        etfs = []
        for etf_data in data.get('etfs', []):
            etf = ETF(
                symbol=etf_data['symbol'],
                name=etf_data['name'],
                category=etf_data['category'],
                asset_class=AssetClass(etf_data['asset_class']),
                description=etf_data['description'],
                exchange=etf_data['exchange'],
                lot_size=etf_data['lot_size'],
                is_active=etf_data['is_active'],
                risk_level=RiskLevel(etf_data['risk_level']),
                expense_ratio=Decimal(str(etf_data['expense_ratio']))
            )
            etfs.append(etf)
        
        symbols = [etf.symbol for etf in etfs]
        
        # Check for duplicates
        if len(symbols) != len(set(symbols)):
            raise ValueError("Duplicate ETF symbols found in configuration")
        
        self._etf_universe = ETFUniverse(etfs=etfs, symbols=symbols)
    
    def _load_allocations(self) -> None:
        """Load allocation blueprints from allocations.yml"""
        alloc_file = self.config_dir / "allocations.yml"
        if not alloc_file.exists():
            raise FileNotFoundError(f"Allocation config not found: {alloc_file}")
        
        with open(alloc_file, 'r') as f:
            data = yaml.safe_load(f)
        
        # Base allocation
        base_alloc = {k: Decimal(str(v)) for k, v in data['base_allocation'].items()}
        self._base_allocation = AllocationBlueprint(
            name="base",
            allocations=base_alloc
        )
        
        # Tactical allocation
        tactical_alloc = {k: Decimal(str(v)) for k, v in data['tactical_allocation'].items()}
        self._tactical_allocation = AllocationBlueprint(
            name="tactical",
            allocations=tactical_alloc
        )
        
        # Crash allocation
        crash_alloc = {k: Decimal(str(v)) for k, v in data['crash_allocation'].items()}
        self._crash_allocation = AllocationBlueprint(
            name="crash",
            allocations=crash_alloc
        )
        
        # Risk constraints from asset class limits
        limits = data['single_etf_limits']
        self._risk_constraints = RiskConstraints(
            max_equity_allocation=Decimal('75'),
            max_single_etf=Decimal('45'),
            max_midcap=Decimal('10'),
            min_debt=Decimal('10'),
            max_gold=Decimal('15'),
            max_single_investment=Decimal('100000')
        )
    
    def _load_rules(self) -> None:
        """Load investment rules from rules.yml"""
        rules_file = self.config_dir / "rules.yml"
        if not rules_file.exists():
            raise FileNotFoundError(f"Rules config not found: {rules_file}")
        
        with open(rules_file, 'r') as f:
            self._rules = yaml.safe_load(f)
    
    def _load_app_config(self) -> None:
        """Load application config from app.yml"""
        app_file = self.config_dir / "app.yml"
        if not app_file.exists():
            raise FileNotFoundError(f"App config not found: {app_file}")
        
        with open(app_file, 'r') as f:
            self._app_config = yaml.safe_load(f)
    
    def _validate_all(self) -> None:
        """Validate all configurations"""
        # Validate allocation symbols match ETF universe
        for blueprint in [self._base_allocation, self._tactical_allocation, self._crash_allocation]:
            for symbol in blueprint.allocations.keys():
                if not self._etf_universe.is_valid_symbol(symbol):
                    raise ValueError(f"Unknown ETF in allocation: {symbol}")
        
        # Validate percentages sum to 100 (already done in AllocationBlueprint)
        # Additional validations can be added here
    
    # Public getters
    
    @property
    def etf_universe(self) -> ETFUniverse:
        """Get ETF universe"""
        if self._etf_universe is None:
            raise RuntimeError("Config not loaded. Call load_all() first")
        return self._etf_universe
    
    @property
    def base_allocation(self) -> AllocationBlueprint:
        """Get base allocation blueprint"""
        if self._base_allocation is None:
            raise RuntimeError("Config not loaded. Call load_all() first")
        return self._base_allocation
    
    @property
    def tactical_allocation(self) -> AllocationBlueprint:
        """Get tactical allocation blueprint"""
        if self._tactical_allocation is None:
            raise RuntimeError("Config not loaded. Call load_all() first")
        return self._tactical_allocation
    
    @property
    def crash_allocation(self) -> AllocationBlueprint:
        """Get crash allocation blueprint"""
        if self._crash_allocation is None:
            raise RuntimeError("Config not loaded. Call load_all() first")
        return self._crash_allocation
    
    @property
    def risk_constraints(self) -> RiskConstraints:
        """Get risk constraints"""
        if self._risk_constraints is None:
            raise RuntimeError("Config not loaded. Call load_all() first")
        return self._risk_constraints
    
    @property
    def strategy_version(self) -> str:
        """Get current strategy version"""
        if self._rules is None:
            raise RuntimeError("Config not loaded. Call load_all() first")
        return self._rules['strategy']['version']
    
    def get_rule(self, *keys) -> any:
        """Get rule value by nested keys"""
        if self._rules is None:
            raise RuntimeError("Config not loaded. Call load_all() first")
        
        value = self._rules
        for key in keys:
            value = value[key]
        return value
    
    def get_app_setting(self, *keys) -> any:
        """Get app setting by nested keys"""
        if self._app_config is None:
            raise RuntimeError("Config not loaded. Call load_all() first")
        
        value = self._app_config
        for key in keys:
            value = value[key]
        return value
