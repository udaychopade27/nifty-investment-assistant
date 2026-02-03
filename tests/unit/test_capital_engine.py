"""
Unit Tests for CapitalEngine

✅ Tests the FIXED async version
✅ Comprehensive coverage of all methods
✅ Tests error cases
✅ Tests edge cases
"""

import pytest
from datetime import date
from decimal import Decimal
from typing import Optional

from app.domain.models import CapitalState, MonthlyConfig
from app.domain.services.capital_engine import CapitalEngine


# Mock Repositories for Testing
class MockMonthlyConfigRepository:
    """Mock repository for testing"""
    
    def __init__(self):
        self.configs = {}
    
    async def get_for_month(self, month: date) -> Optional[MonthlyConfig]:
        """Get config for month"""
        return self.configs.get(month)
    
    async def get_current(self) -> Optional[MonthlyConfig]:
        """Get current month config"""
        today = date.today()
        current_month = date(today.year, today.month, 1)
        return self.configs.get(current_month)
    
    def add_config(self, config: MonthlyConfig):
        """Helper to add config"""
        self.configs[config.month] = config


class MockExecutedInvestmentRepository:
    """Mock repository for testing"""
    
    def __init__(self):
        self.base_deployed = {}
        self.tactical_deployed = {}
        self.extra_deployed = {}
    
    async def get_total_base_deployed(self, month: date) -> Decimal:
        """Get base deployed"""
        return self.base_deployed.get(month, Decimal('0'))
    
    async def get_total_tactical_deployed(self, month: date) -> Decimal:
        """Get tactical deployed"""
        return self.tactical_deployed.get(month, Decimal('0'))
    
    async def get_total_extra_deployed(self, month: date) -> Decimal:
        """Get extra deployed"""
        return self.extra_deployed.get(month, Decimal('0'))
    
    def set_deployed(self, month: date, base: Decimal, tactical: Decimal, extra: Decimal):
        """Helper to set deployed amounts"""
        self.base_deployed[month] = base
        self.tactical_deployed[month] = tactical
        self.extra_deployed[month] = extra


class MockExtraCapitalRepository:
    """Mock repository for testing"""
    
    def __init__(self):
        self.extra_capital = {}
    
    async def get_total_for_month(self, month: date) -> Decimal:
        """Get extra capital"""
        return self.extra_capital.get(month, Decimal('0'))
    
    def set_extra(self, month: date, amount: Decimal):
        """Helper to set extra capital"""
        self.extra_capital[month] = amount


# Fixtures
@pytest.fixture
def month():
    """Test month"""
    return date(2026, 2, 1)


@pytest.fixture
def monthly_config(month):
    """Sample monthly config"""
    return MonthlyConfig(
        month=month,
        monthly_capital=Decimal('10000'),
        base_capital=Decimal('6000'),
        tactical_capital=Decimal('4000'),
        trading_days=20,
        daily_tranche=Decimal('300'),
        strategy_version='v1.0',
        created_at=None
    )


@pytest.fixture
def mock_repos():
    """Create mock repositories"""
    return (
        MockMonthlyConfigRepository(),
        MockExecutedInvestmentRepository(),
        MockExtraCapitalRepository()
    )


@pytest.fixture
def capital_engine(mock_repos):
    """Create CapitalEngine with mocks"""
    config_repo, investment_repo, extra_repo = mock_repos
    return CapitalEngine(
        monthly_config_repo=config_repo,
        executed_investment_repo=investment_repo,
        extra_capital_repo=extra_repo
    )


# Tests for get_capital_state
class TestGetCapitalState:
    """Tests for get_capital_state method"""
    
    @pytest.mark.asyncio
    async def test_no_deployments(self, capital_engine, mock_repos, month, monthly_config):
        """Test capital state with no deployments"""
        config_repo, _, _ = mock_repos
        config_repo.add_config(monthly_config)
        
        state = await capital_engine.get_capital_state(month)
        
        assert state.month == month
        assert state.base_total == Decimal('6000')
        assert state.base_remaining == Decimal('6000')
        assert state.tactical_total == Decimal('4000')
        assert state.tactical_remaining == Decimal('4000')
        assert state.extra_total == Decimal('0')
        assert state.extra_remaining == Decimal('0')
    
    @pytest.mark.asyncio
    async def test_partial_deployments(self, capital_engine, mock_repos, month, monthly_config):
        """Test capital state with partial deployments"""
        config_repo, investment_repo, _ = mock_repos
        config_repo.add_config(monthly_config)
        investment_repo.set_deployed(
            month,
            base=Decimal('3000'),  # 50% of base
            tactical=Decimal('1000'),  # 25% of tactical
            extra=Decimal('0')
        )
        
        state = await capital_engine.get_capital_state(month)
        
        assert state.base_remaining == Decimal('3000')
        assert state.tactical_remaining == Decimal('3000')
    
    @pytest.mark.asyncio
    async def test_full_deployment(self, capital_engine, mock_repos, month, monthly_config):
        """Test capital state when fully deployed"""
        config_repo, investment_repo, _ = mock_repos
        config_repo.add_config(monthly_config)
        investment_repo.set_deployed(
            month,
            base=Decimal('6000'),
            tactical=Decimal('4000'),
            extra=Decimal('0')
        )
        
        state = await capital_engine.get_capital_state(month)
        
        assert state.base_remaining == Decimal('0')
        assert state.tactical_remaining == Decimal('0')
    
    @pytest.mark.asyncio
    async def test_with_extra_capital(self, capital_engine, mock_repos, month, monthly_config):
        """Test capital state with extra capital injection"""
        config_repo, investment_repo, extra_repo = mock_repos
        config_repo.add_config(monthly_config)
        extra_repo.set_extra(month, Decimal('2000'))
        investment_repo.set_deployed(
            month,
            base=Decimal('3000'),
            tactical=Decimal('1000'),
            extra=Decimal('500')
        )
        
        state = await capital_engine.get_capital_state(month)
        
        assert state.extra_total == Decimal('2000')
        assert state.extra_remaining == Decimal('1500')
    
    @pytest.mark.asyncio
    async def test_negative_remaining_safety(self, capital_engine, mock_repos, month, monthly_config):
        """Test that negative remaining is clamped to zero"""
        config_repo, investment_repo, _ = mock_repos
        config_repo.add_config(monthly_config)
        # Deploy more than available (simulating data corruption)
        investment_repo.set_deployed(
            month,
            base=Decimal('7000'),  # More than 6000 available
            tactical=Decimal('5000'),  # More than 4000 available
            extra=Decimal('0')
        )
        
        state = await capital_engine.get_capital_state(month)
        
        # Should clamp to zero, not go negative
        assert state.base_remaining == Decimal('0')
        assert state.tactical_remaining == Decimal('0')
    
    @pytest.mark.asyncio
    async def test_missing_config_raises_error(self, capital_engine, month):
        """Test that missing config raises ValueError"""
        with pytest.raises(ValueError, match="No MonthlyConfig found"):
            await capital_engine.get_capital_state(month)


# Tests for get_current_capital_state
class TestGetCurrentCapitalState:
    """Tests for get_current_capital_state method"""
    
    @pytest.mark.asyncio
    async def test_current_month(self, capital_engine, mock_repos):
        """Test getting current month's capital state"""
        today = date.today()
        current_month = date(today.year, today.month, 1)
        
        config = MonthlyConfig(
            month=current_month,
            monthly_capital=Decimal('10000'),
            base_capital=Decimal('6000'),
            tactical_capital=Decimal('4000'),
            trading_days=20,
            daily_tranche=Decimal('300'),
            strategy_version='v1.0',
            created_at=None
        )
        
        config_repo, investment_repo, _ = mock_repos
        config_repo.add_config(config)
        investment_repo.set_deployed(
            current_month,
            base=Decimal('1000'),
            tactical=Decimal('500'),
            extra=Decimal('0')
        )
        
        state = await capital_engine.get_current_capital_state()
        
        assert state.month == current_month
        assert state.base_remaining == Decimal('5000')
        assert state.tactical_remaining == Decimal('3500')
    
    @pytest.mark.asyncio
    async def test_missing_current_raises_error(self, capital_engine):
        """Test that missing current config raises ValueError"""
        with pytest.raises(ValueError, match="No MonthlyConfig found for current month"):
            await capital_engine.get_current_capital_state()


# Tests for calculate_daily_tranche
class TestCalculateDailyTranche:
    """Tests for calculate_daily_tranche method"""
    
    def test_normal_calculation(self, capital_engine, monthly_config):
        """Test normal daily tranche calculation"""
        tranche = capital_engine.calculate_daily_tranche(monthly_config)
        
        # 6000 / 20 = 300
        assert tranche == Decimal('300')
    
    def test_with_decimal_result(self, capital_engine, month):
        """Test tranche calculation with decimal result"""
        config = MonthlyConfig(
            month=month,
            monthly_capital=Decimal('10000'),
            base_capital=Decimal('6000'),
            tactical_capital=Decimal('4000'),
            trading_days=23,  # Will result in decimal
            daily_tranche=Decimal('260.87'),
            strategy_version='v1.0',
            created_at=None
        )
        
        tranche = capital_engine.calculate_daily_tranche(config)
        
        # Should maintain precision
        expected = Decimal('6000') / Decimal('23')
        assert tranche == expected
    
    def test_zero_trading_days_raises_error(self, capital_engine, month):
        """Test that zero trading days raises ValueError"""
        config = MonthlyConfig(
            month=month,
            monthly_capital=Decimal('10000'),
            base_capital=Decimal('6000'),
            tactical_capital=Decimal('4000'),
            trading_days=0,  # Invalid
            daily_tranche=Decimal('0'),
            strategy_version='v1.0',
            created_at=None
        )
        
        with pytest.raises(ValueError, match="Trading days must be positive"):
            capital_engine.calculate_daily_tranche(config)


# Tests for can_deploy_tactical
class TestCanDeployTactical:
    """Tests for can_deploy_tactical method"""
    
    def test_sufficient_capital(self, capital_engine):
        """Test deployment with sufficient capital"""
        state = CapitalState(
            month=date(2026, 2, 1),
            base_total=Decimal('6000'),
            base_remaining=Decimal('3000'),
            tactical_total=Decimal('4000'),
            tactical_remaining=Decimal('2000'),
            extra_total=Decimal('0'),
            extra_remaining=Decimal('0')
        )
        
        can_deploy, reason = capital_engine.can_deploy_tactical(state, Decimal('1000'))
        
        assert can_deploy is True
        assert reason == "OK"
    
    def test_insufficient_capital(self, capital_engine):
        """Test deployment with insufficient capital"""
        state = CapitalState(
            month=date(2026, 2, 1),
            base_total=Decimal('6000'),
            base_remaining=Decimal('3000'),
            tactical_total=Decimal('4000'),
            tactical_remaining=Decimal('500'),  # Only 500 remaining
            extra_total=Decimal('0'),
            extra_remaining=Decimal('0')
        )
        
        can_deploy, reason = capital_engine.can_deploy_tactical(state, Decimal('1000'))
        
        assert can_deploy is False
        assert "Insufficient tactical capital" in reason
    
    def test_zero_amount(self, capital_engine):
        """Test deployment with zero amount"""
        state = CapitalState(
            month=date(2026, 2, 1),
            base_total=Decimal('6000'),
            base_remaining=Decimal('3000'),
            tactical_total=Decimal('4000'),
            tactical_remaining=Decimal('2000'),
            extra_total=Decimal('0'),
            extra_remaining=Decimal('0')
        )
        
        can_deploy, reason = capital_engine.can_deploy_tactical(state, Decimal('0'))
        
        assert can_deploy is False
        assert "must be positive" in reason
    
    def test_negative_amount(self, capital_engine):
        """Test deployment with negative amount"""
        state = CapitalState(
            month=date(2026, 2, 1),
            base_total=Decimal('6000'),
            base_remaining=Decimal('3000'),
            tactical_total=Decimal('4000'),
            tactical_remaining=Decimal('2000'),
            extra_total=Decimal('0'),
            extra_remaining=Decimal('0')
        )
        
        can_deploy, reason = capital_engine.can_deploy_tactical(state, Decimal('-100'))
        
        assert can_deploy is False
        assert "must be positive" in reason


# Tests for can_deploy_extra
class TestCanDeployExtra:
    """Tests for can_deploy_extra method"""
    
    def test_sufficient_extra_capital(self, capital_engine):
        """Test extra deployment with sufficient capital"""
        state = CapitalState(
            month=date(2026, 2, 1),
            base_total=Decimal('6000'),
            base_remaining=Decimal('3000'),
            tactical_total=Decimal('4000'),
            tactical_remaining=Decimal('2000'),
            extra_total=Decimal('5000'),
            extra_remaining=Decimal('3000')
        )
        
        can_deploy, reason = capital_engine.can_deploy_extra(state, Decimal('2000'))
        
        assert can_deploy is True
        assert reason == "OK"
    
    def test_insufficient_extra_capital(self, capital_engine):
        """Test extra deployment with insufficient capital"""
        state = CapitalState(
            month=date(2026, 2, 1),
            base_total=Decimal('6000'),
            base_remaining=Decimal('3000'),
            tactical_total=Decimal('4000'),
            tactical_remaining=Decimal('2000'),
            extra_total=Decimal('5000'),
            extra_remaining=Decimal('500')
        )
        
        can_deploy, reason = capital_engine.can_deploy_extra(state, Decimal('1000'))
        
        assert can_deploy is False
        assert "Insufficient extra capital" in reason


# Tests for calculate_tactical_carry_forward
class TestCalculateTacticalCarryForward:
    """Tests for calculate_tactical_carry_forward method"""
    
    def test_small_remaining_below_cap(self, capital_engine):
        """Test carry forward when remaining is below cap"""
        previous_state = CapitalState(
            month=date(2026, 1, 1),
            base_total=Decimal('6000'),
            base_remaining=Decimal('0'),
            tactical_total=Decimal('4000'),
            tactical_remaining=Decimal('1000'),  # Unused tactical
            extra_total=Decimal('0'),
            extra_remaining=Decimal('0')
        )
        
        carry_forward = capital_engine.calculate_tactical_carry_forward(
            previous_state,
            new_monthly_capital=Decimal('10000')
        )
        
        # Should carry forward all 1000 (below cap)
        assert carry_forward == Decimal('1000')
    
    def test_large_remaining_capped(self, capital_engine):
        """Test carry forward when remaining exceeds cap"""
        previous_state = CapitalState(
            month=date(2026, 1, 1),
            base_total=Decimal('6000'),
            base_remaining=Decimal('0'),
            tactical_total=Decimal('4000'),
            tactical_remaining=Decimal('8000'),  # Large unused tactical
            extra_total=Decimal('0'),
            extra_remaining=Decimal('0')
        )
        
        carry_forward = capital_engine.calculate_tactical_carry_forward(
            previous_state,
            new_monthly_capital=Decimal('10000'),
            carry_forward_cap_multiplier=Decimal('1.5')
        )
        
        # Cap = (10000 * 0.4) * 1.5 = 6000
        # Should return min(8000, 6000) = 6000
        assert carry_forward == Decimal('6000')
    
    def test_zero_remaining(self, capital_engine):
        """Test carry forward with zero remaining"""
        previous_state = CapitalState(
            month=date(2026, 1, 1),
            base_total=Decimal('6000'),
            base_remaining=Decimal('0'),
            tactical_total=Decimal('4000'),
            tactical_remaining=Decimal('0'),  # Fully deployed
            extra_total=Decimal('0'),
            extra_remaining=Decimal('0')
        )
        
        carry_forward = capital_engine.calculate_tactical_carry_forward(
            previous_state,
            new_monthly_capital=Decimal('10000')
        )
        
        assert carry_forward == Decimal('0')
    
    def test_custom_multiplier(self, capital_engine):
        """Test carry forward with custom multiplier"""
        previous_state = CapitalState(
            month=date(2026, 1, 1),
            base_total=Decimal('6000'),
            base_remaining=Decimal('0'),
            tactical_total=Decimal('4000'),
            tactical_remaining=Decimal('10000'),
            extra_total=Decimal('0'),
            extra_remaining=Decimal('0')
        )
        
        carry_forward = capital_engine.calculate_tactical_carry_forward(
            previous_state,
            new_monthly_capital=Decimal('10000'),
            carry_forward_cap_multiplier=Decimal('2.0')  # Higher cap
        )
        
        # Cap = (10000 * 0.4) * 2.0 = 8000
        # Should return min(10000, 8000) = 8000
        assert carry_forward == Decimal('8000')


# Tests for validate_capital_integrity
class TestValidateCapitalIntegrity:
    """Tests for validate_capital_integrity method"""
    
    def test_valid_state(self, capital_engine):
        """Test validation of valid capital state"""
        state = CapitalState(
            month=date(2026, 2, 1),
            base_total=Decimal('6000'),
            base_remaining=Decimal('3000'),
            tactical_total=Decimal('4000'),
            tactical_remaining=Decimal('2000'),
            extra_total=Decimal('1000'),
            extra_remaining=Decimal('500')
        )
        
        is_valid, issues = capital_engine.validate_capital_integrity(state)
        
        assert is_valid is True
        assert len(issues) == 0
    
    def test_negative_base(self, capital_engine):
        """Test validation with negative base capital"""
        state = CapitalState(
            month=date(2026, 2, 1),
            base_total=Decimal('6000'),
            base_remaining=Decimal('-100'),  # Invalid
            tactical_total=Decimal('4000'),
            tactical_remaining=Decimal('2000'),
            extra_total=Decimal('0'),
            extra_remaining=Decimal('0')
        )
        
        is_valid, issues = capital_engine.validate_capital_integrity(state)
        
        assert is_valid is False
        assert "Base capital is negative" in issues
    
    def test_remaining_exceeds_total(self, capital_engine):
        """Test validation when remaining exceeds total"""
        state = CapitalState(
            month=date(2026, 2, 1),
            base_total=Decimal('6000'),
            base_remaining=Decimal('7000'),  # Invalid: exceeds total
            tactical_total=Decimal('4000'),
            tactical_remaining=Decimal('2000'),
            extra_total=Decimal('0'),
            extra_remaining=Decimal('0')
        )
        
        is_valid, issues = capital_engine.validate_capital_integrity(state)
        
        assert is_valid is False
        assert "Base remaining exceeds total" in issues
    
    def test_multiple_issues(self, capital_engine):
        """Test validation with multiple integrity issues"""
        state = CapitalState(
            month=date(2026, 2, 1),
            base_total=Decimal('6000'),
            base_remaining=Decimal('-100'),  # Negative
            tactical_total=Decimal('4000'),
            tactical_remaining=Decimal('5000'),  # Exceeds total
            extra_total=Decimal('1000'),
            extra_remaining=Decimal('-50')  # Negative
        )
        
        is_valid, issues = capital_engine.validate_capital_integrity(state)
        
        assert is_valid is False
        assert len(issues) >= 3  # Should have multiple issues
        assert any("Base capital is negative" in issue for issue in issues)
        assert any("Tactical remaining exceeds total" in issue for issue in issues)
        assert any("Extra capital is negative" in issue for issue in issues)


# Integration-style tests
class TestCapitalEngineIntegration:
    """Integration-style tests for realistic scenarios"""
    
    @pytest.mark.asyncio
    async def test_full_month_scenario(self, capital_engine, mock_repos, month):
        """Test a complete month investment scenario"""
        # Setup: Month with 20 trading days
        config = MonthlyConfig(
            month=month,
            monthly_capital=Decimal('10000'),
            base_capital=Decimal('6000'),
            tactical_capital=Decimal('4000'),
            trading_days=20,
            daily_tranche=Decimal('300'),
            strategy_version='v1.0',
            created_at=None
        )
        
        config_repo, investment_repo, extra_repo = mock_repos
        config_repo.add_config(config)
        
        # Day 1-10: Deploy daily tranche (300 * 10 = 3000)
        investment_repo.set_deployed(
            month,
            base=Decimal('3000'),
            tactical=Decimal('0'),
            extra=Decimal('0')
        )
        
        state = await capital_engine.get_capital_state(month)
        assert state.base_remaining == Decimal('3000')
        
        # Day 11: Market dip, deploy tactical
        investment_repo.set_deployed(
            month,
            base=Decimal('3000'),
            tactical=Decimal('1500'),
            extra=Decimal('0')
        )
        
        state = await capital_engine.get_capital_state(month)
        assert state.tactical_remaining == Decimal('2500')
        
        # Mid-month: Extra capital injected
        extra_repo.set_extra(month, Decimal('2000'))
        
        state = await capital_engine.get_capital_state(month)
        assert state.extra_total == Decimal('2000')
        assert state.extra_remaining == Decimal('2000')
        
        # Validate final state integrity
        is_valid, issues = capital_engine.validate_capital_integrity(state)
        assert is_valid is True