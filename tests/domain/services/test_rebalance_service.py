import pytest
from datetime import date, datetime, timedelta
from decimal import Decimal

from app.domain.services.rebalance_service import AnnualRebalanceService
from app.infrastructure.db.models import ExecutedInvestmentModel, RebalanceLogModel


class DummyConfig:
    def __init__(self, rebalance=None):
        self._rebalance = rebalance or {}

    def get_app_setting(self, *keys):
        if keys == ("rebalance",):
            return self._rebalance
        raise KeyError(keys)


class DummyCalendar:
    def is_trading_day(self, target_date: date) -> bool:
        return True

    def get_previous_trading_day(self, target_date: date) -> date:
        return target_date - timedelta(days=1)


class DummyMarketProvider:
    def __init__(self, current_prices, daily_prices=None):
        self._current_prices = current_prices
        self._daily_prices = daily_prices or {}

    async def get_current_prices(self, symbols):
        return {s: self._current_prices[s] for s in symbols if s in self._current_prices}

    async def get_prices_for_date(self, symbols, target_date: date):
        key = (target_date.isoformat(), tuple(sorted(symbols)))
        if key in self._daily_prices:
            return self._daily_prices[key]
        return {s: self._current_prices.get(s, Decimal("0")) for s in symbols}


@pytest.mark.asyncio
async def test_rebalance_single_overweight(db_session):
    run_date = date(2026, 4, 5)

    # Single overweight holding (NIFTYBEES)
    db_session.add(
        ExecutedInvestmentModel(
            etf_symbol="NIFTYBEES",
            units=100,
            executed_price=Decimal("100"),
            total_amount=Decimal("10000"),
            slippage_pct=Decimal("0"),
            capital_bucket="base",
            executed_at=datetime(2024, 1, 1),
        )
    )
    await db_session.commit()

    market_provider = DummyMarketProvider(
        current_prices={s: Decimal("100") for s in AnnualRebalanceService.TARGET_WEIGHTS},
    )
    service = AnnualRebalanceService(
        config_engine=DummyConfig(),
        market_provider=market_provider,
        nse_calendar=DummyCalendar(),
    )

    result = await service.run(db_session, run_date=run_date)

    assert result["status"] == "completed"
    sell_items = result["sell_plan"]["items"]
    assert sell_items[0]["symbol"] == "NIFTYBEES"

    buy_items = result["buy_plan"]["items"]
    buy_symbols = [item["symbol"] for item in buy_items]
    assert buy_symbols[:5] == [
        "ICICIVALUE",
        "MIDCAPETF",
        "JUNIORBEES",
        "ICICIMOM30",
        "HDFCGOLD",
    ]


@pytest.mark.asyncio
async def test_rebalance_multiple_overweight_stcg_guard(db_session):
    run_date = date(2026, 4, 5)

    # MIDCAPETF overweight (old holding -> sell allowed)
    db_session.add(
        ExecutedInvestmentModel(
            etf_symbol="MIDCAPETF",
            units=60,
            executed_price=Decimal("100"),
            total_amount=Decimal("6000"),
            slippage_pct=Decimal("0"),
            capital_bucket="base",
            executed_at=datetime(2024, 1, 1),
        )
    )
    # ICICIMOM30 overweight but recent -> should be guarded
    db_session.add(
        ExecutedInvestmentModel(
            etf_symbol="ICICIMOM30",
            units=20,
            executed_price=Decimal("100"),
            total_amount=Decimal("2000"),
            slippage_pct=Decimal("0"),
            capital_bucket="base",
            executed_at=datetime(2026, 3, 15),
        )
    )
    # NIFTYBEES small position
    db_session.add(
        ExecutedInvestmentModel(
            etf_symbol="NIFTYBEES",
            units=20,
            executed_price=Decimal("100"),
            total_amount=Decimal("2000"),
            slippage_pct=Decimal("0"),
            capital_bucket="base",
            executed_at=datetime(2024, 6, 1),
        )
    )
    await db_session.commit()

    market_provider = DummyMarketProvider(
        current_prices={s: Decimal("100") for s in AnnualRebalanceService.TARGET_WEIGHTS},
    )
    service = AnnualRebalanceService(
        config_engine=DummyConfig(),
        market_provider=market_provider,
        nse_calendar=DummyCalendar(),
    )

    result = await service.run(db_session, run_date=run_date)
    assert result["status"] == "completed"

    sell_symbols = [item["symbol"] for item in result["sell_plan"]["items"]]
    assert "MIDCAPETF" in sell_symbols
    assert "ICICIMOM30" not in sell_symbols


@pytest.mark.asyncio
async def test_rebalance_gold_band_no_action(db_session):
    run_date = date(2026, 4, 5)

    # Gold within 3-7% band -> no action
    db_session.add(
        ExecutedInvestmentModel(
            etf_symbol="HDFCGOLD",
            units=5,
            executed_price=Decimal("100"),
            total_amount=Decimal("500"),
            slippage_pct=Decimal("0"),
            capital_bucket="base",
            executed_at=datetime(2024, 1, 1),
        )
    )
    # Add other holdings to make gold weight 5%
    db_session.add(
        ExecutedInvestmentModel(
            etf_symbol="NIFTYBEES",
            units=95,
            executed_price=Decimal("100"),
            total_amount=Decimal("9500"),
            slippage_pct=Decimal("0"),
            capital_bucket="base",
            executed_at=datetime(2024, 1, 1),
        )
    )
    await db_session.commit()

    market_provider = DummyMarketProvider(
        current_prices={s: Decimal("100") for s in AnnualRebalanceService.TARGET_WEIGHTS},
    )
    service = AnnualRebalanceService(
        config_engine=DummyConfig(),
        market_provider=market_provider,
        nse_calendar=DummyCalendar(),
    )

    result = await service.run(db_session, run_date=run_date)

    assert result["status"] == "completed"
    gold_info = result["drifts"]["HDFCGOLD"]
    assert gold_info["status"] == "no_action"


@pytest.mark.asyncio
async def test_rebalance_skips_outside_window(db_session):
    run_date = date(2026, 3, 30)

    market_provider = DummyMarketProvider(
        current_prices={s: Decimal("100") for s in AnnualRebalanceService.TARGET_WEIGHTS},
    )
    service = AnnualRebalanceService(
        config_engine=DummyConfig(),
        market_provider=market_provider,
        nse_calendar=DummyCalendar(),
    )

    result = await service.run(db_session, run_date=run_date)

    assert result["status"] == "skipped"
    assert result["reason"] == "outside_rebalance_window"


@pytest.mark.asyncio
async def test_rebalance_skips_when_already_done(db_session):
    run_date = date(2026, 4, 5)
    fiscal_year = "2025-26"

    db_session.add(
        RebalanceLogModel(
            fiscal_year=fiscal_year,
            rebalance_date=run_date,
            payload={"status": "completed"},
        )
    )
    await db_session.commit()

    market_provider = DummyMarketProvider(
        current_prices={s: Decimal("100") for s in AnnualRebalanceService.TARGET_WEIGHTS},
    )
    service = AnnualRebalanceService(
        config_engine=DummyConfig(),
        market_provider=market_provider,
        nse_calendar=DummyCalendar(),
    )

    result = await service.run(db_session, run_date=run_date)

    assert result["status"] == "skipped"
    assert result["reason"] == "already_rebalanced"
