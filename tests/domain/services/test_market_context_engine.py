"""
Unit Tests for Market Context Engine
Demonstrates testing approach for domain engines
"""

import pytest
from decimal import Decimal
from datetime import date

from app.domain.services.market_context_engine import MarketContextEngine
from app.domain.models import StressLevel


@pytest.fixture
def engine():
    """Fixture for MarketContextEngine"""
    return MarketContextEngine()


class TestMarketContextEngine:
    """Test suite for Market Context Engine"""
    
    def test_calculate_percentage_change_positive(self, engine):
        """Test percentage change calculation - market up"""
        result = engine._calculate_percentage_change(
            old_value=Decimal('22000'),
            new_value=Decimal('22440')
        )
        assert result == Decimal('2.00')
    
    def test_calculate_percentage_change_negative(self, engine):
        """Test percentage change calculation - market down"""
        result = engine._calculate_percentage_change(
            old_value=Decimal('22000'),
            new_value=Decimal('21780')
        )
        assert result == Decimal('-1.00')
    
    def test_calculate_percentage_change_zero(self, engine):
        """Test percentage change calculation - no change"""
        result = engine._calculate_percentage_change(
            old_value=Decimal('22000'),
            new_value=Decimal('22000')
        )
        assert result == Decimal('0.00')
    
    def test_calculate_percentage_change_invalid_old_value(self, engine):
        """Test percentage change with invalid old value"""
        with pytest.raises(ValueError, match="Old value must be positive"):
            engine._calculate_percentage_change(
                old_value=Decimal('0'),
                new_value=Decimal('100')
            )
    
    def test_calculate_cumulative_change(self, engine):
        """Test cumulative change over 3 days"""
        closes = [
            Decimal('22000'),
            Decimal('21800'),
            Decimal('21600')
        ]
        current = Decimal('21450')
        
        result = engine._calculate_cumulative_change(closes, current)
        expected = ((Decimal('21450') - Decimal('22000')) / Decimal('22000')) * Decimal('100')
        assert result == expected.quantize(Decimal('0.01'))
    
    def test_determine_stress_level_none(self, engine):
        """Test stress level - no stress"""
        stress = engine._determine_stress_level(
            daily_change=Decimal('-0.5'),
            cumulative_3day=Decimal('-0.8'),
            vix=Decimal('15')
        )
        assert stress == StressLevel.NONE
    
    def test_determine_stress_level_low(self, engine):
        """Test stress level - low stress"""
        stress = engine._determine_stress_level(
            daily_change=Decimal('-1.2'),
            cumulative_3day=Decimal('-1.0'),
            vix=Decimal('16')
        )
        assert stress == StressLevel.LOW
    
    def test_determine_stress_level_medium(self, engine):
        """Test stress level - medium stress"""
        stress = engine._determine_stress_level(
            daily_change=Decimal('-2.3'),
            cumulative_3day=Decimal('-2.5'),
            vix=Decimal('18')
        )
        assert stress == StressLevel.MEDIUM
    
    def test_determine_stress_level_high(self, engine):
        """Test stress level - high stress"""
        stress = engine._determine_stress_level(
            daily_change=Decimal('-3.5'),
            cumulative_3day=Decimal('-4.0'),
            vix=Decimal('22')
        )
        assert stress == StressLevel.HIGH
    
    def test_determine_stress_level_high_vix(self, engine):
        """Test stress level - high due to VIX"""
        stress = engine._determine_stress_level(
            daily_change=Decimal('-0.5'),
            cumulative_3day=Decimal('-0.5'),
            vix=Decimal('26')
        )
        assert stress == StressLevel.HIGH
    
    def test_calculate_context_full(self, engine):
        """Test full context calculation"""
        context = engine.calculate_context(
            calc_date=date(2025, 1, 29),
            nifty_close=Decimal('21780'),
            nifty_previous_close=Decimal('22000'),
            last_3_day_closes=[
                Decimal('22200'),
                Decimal('22100'),
                Decimal('22000')
            ],
            india_vix=Decimal('18.5')
        )
        
        assert context.date == date(2025, 1, 29)
        assert context.nifty_close == Decimal('21780')
        assert context.nifty_previous_close == Decimal('22000')
        assert context.daily_change_pct == Decimal('-1.00')
        assert context.india_vix == Decimal('18.5')
        assert context.stress_level == StressLevel.LOW
    
    def test_is_dip_day_true(self, engine):
        """Test dip day detection - is dip"""
        context = engine.calculate_context(
            calc_date=date(2025, 1, 29),
            nifty_close=Decimal('21780'),
            nifty_previous_close=Decimal('22000'),
            last_3_day_closes=[Decimal('22000')],
            india_vix=None
        )
        
        assert engine.is_dip_day(context) is True
    
    def test_is_dip_day_false(self, engine):
        """Test dip day detection - not dip"""
        context = engine.calculate_context(
            calc_date=date(2025, 1, 29),
            nifty_close=Decimal('22110'),
            nifty_previous_close=Decimal('22000'),
            last_3_day_closes=[Decimal('22000')],
            india_vix=None
        )
        
        assert engine.is_dip_day(context) is False
    
    def test_get_dip_magnitude_with_dip(self, engine):
        """Test dip magnitude calculation - market down"""
        context = engine.calculate_context(
            calc_date=date(2025, 1, 29),
            nifty_close=Decimal('21450'),
            nifty_previous_close=Decimal('22000'),
            last_3_day_closes=[Decimal('22000')],
            india_vix=None
        )
        
        magnitude = engine.get_dip_magnitude(context)
        assert magnitude == Decimal('2.50')
    
    def test_get_dip_magnitude_no_dip(self, engine):
        """Test dip magnitude calculation - market up"""
        context = engine.calculate_context(
            calc_date=date(2025, 1, 29),
            nifty_close=Decimal('22440'),
            nifty_previous_close=Decimal('22000'),
            last_3_day_closes=[Decimal('22000')],
            india_vix=None
        )
        
        magnitude = engine.get_dip_magnitude(context)
        assert magnitude == Decimal('0')


@pytest.mark.parametrize("daily_change,expected_stress", [
    (Decimal('-0.5'), StressLevel.NONE),
    (Decimal('-1.2'), StressLevel.LOW),
    (Decimal('-2.3'), StressLevel.MEDIUM),
    (Decimal('-3.5'), StressLevel.HIGH),
])
def test_stress_levels_parametrized(engine, daily_change, expected_stress):
    """Parametrized test for various stress levels"""
    stress = engine._determine_stress_level(
        daily_change=daily_change,
        cumulative_3day=Decimal('0'),
        vix=None
    )
    assert stress == expected_stress
