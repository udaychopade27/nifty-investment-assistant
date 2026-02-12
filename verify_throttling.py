import asyncio
import logging
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("Simulation")

# Mock dependencies
from app.realtime.runtime import RealtimeRuntime
from app.domain.services.config_engine import ConfigEngine

async def run_simulation():
    logger.info("🚀 Starting Throttling Simulation...")

    # Mock ConfigEngine
    config_engine = MagicMock(spec=ConfigEngine)
    # Return minimal config
    config_engine.get_app_setting.return_value = {
        "enabled": True,
        "redis": {"enabled": True, "tick_ttl_seconds": 60},
        "signals": {"enabled": False}
    }
    
    # Initialize Runtime
    runtime = RealtimeRuntime(config_engine)
    
    # Mock internally
    runtime._quote_store = MagicMock()
    runtime._redis_cache = AsyncMock()
    runtime._key_to_symbol = {"NSE_EQ|INF204KB14I2": "NIFTYBEES"}
    
    # Track Redis writes
    redis_write_count = 0
    
    async def mock_set_json(key, payload, ttl):
        nonlocal redis_write_count
        redis_write_count += 1
        # logger.debug(f"Redis write: {key}")
        
    runtime._redis_cache.set_json = AsyncMock(side_effect=mock_set_json)
    
    # Simulation Parameters
    instrument_key = "NSE_EQ|INF204KB14I2"
    base_price = Decimal("250.00")
    total_ticks = 100
    delay_between_ticks = 0.01  # 10ms (approx 100 ticks/sec)
    
    logger.info(f"📊 Simulating {total_ticks} ticks at ~{1/delay_between_ticks:.0f} ticks/sec...")
    
    start_time = datetime.now()
    
    for i in range(total_ticks):
        # Simulate price: mostly flat, occasionally spikes
        price = base_price
        
        # Inject spike at tick 50
        if i == 50:
            price = base_price * Decimal("1.01") # 1% jump
            logger.info("⚡ Injecting 1% price spike!")
            
        await runtime._handle_tick(instrument_key, price, datetime.now(tz=timezone.utc))
        await asyncio.sleep(delay_between_ticks)
        
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    logger.info("="*40)
    logger.info(f"⏱️  Duration: {duration:.2f} seconds")
    logger.info(f"📥 Total Ticks Input: {total_ticks}")
    logger.info(f"💾 Total Redis Writes: {redis_write_count}")
    logger.info("="*40)
    
    # Assertions
    expected_writes_max = int(duration) + 5 # 1 per sec + 1 spike + buffer
    if redis_write_count <= expected_writes_max:
        logger.info("✅ Throttling PASSED")
    else:
        logger.error(f"❌ Throttling FAILED: Writes {redis_write_count} > Expected Max {expected_writes_max}")
        
    # Verify QuoteStore got ALL ticks
    quote_store_calls = runtime._quote_store.ingest_tick.call_count
    if quote_store_calls == total_ticks:
        logger.info(f"✅ QuoteStore Integrity PASSED ({quote_store_calls} calls)")
    else:
        logger.error(f"❌ QuoteStore Integrity FAILED: {quote_store_calls} != {total_ticks}")

if __name__ == "__main__":
    asyncio.run(run_simulation())
