import csv
import asyncio
import os
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Any

# Configure logging
logger = logging.getLogger(__name__)

class MarketDataRecorder:
    """
    Asynchronously records real-time market ticks to CSV files
    organized by symbol and date.
    """
    
    def __init__(self, data_dir: str = "data/market_data/intraday"):
        self.data_dir = data_dir
        self._ensure_root_dir()
        
        # Buffer: {symbol: [tick_dict, tick_dict...]}
        self._buffer: Dict[str, List[Dict[str, Any]]] = {}
        self._buffer_lock = asyncio.Lock()
        
        # Flush configuration
        self._batch_size = 50
        self._flush_interval_seconds = 5
        self._last_flush_time = datetime.now()
        
        # Background flush task
        self._flush_task = None
        self._running = False

    def _ensure_root_dir(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    async def start(self):
        """Start the background flush task."""
        self._running = True
        self._flush_task = asyncio.create_task(self._periodic_flush_loop())
        logger.info(f"MarketDataRecorder started. Recording to {self.data_dir}")

    async def stop(self):
        """Stop the recorder and flush remaining data."""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        # Final flush
        await self.flush()
        logger.info("MarketDataRecorder stopped.")

    async def ingest_tick(self, symbol: str, price: float, volume: int = 0, bid: float = 0, ask: float = 0, ts: datetime = None):
        """Add a tick to the buffer."""
        if ts is None:
            ts = datetime.now(timezone.utc)
            
        tick_data = {
            "timestamp": ts.isoformat(),
            "price": price,
            "volume": volume,
            "bid": bid,
            "ask": ask
        }
        
        async with self._buffer_lock:
            if symbol not in self._buffer:
                self._buffer[symbol] = []
            self._buffer[symbol].append(tick_data)
            
            # Use a simpler size check to avoid strict locking overhead? 
            # Actually, we are already locked.
            should_flush = sum(len(ticks) for ticks in self._buffer.values()) >= (self._batch_size * 10) # Safety cap
            
        if should_flush:
            asyncio.create_task(self.flush())

    async def _periodic_flush_loop(self):
        while self._running:
            await asyncio.sleep(self._flush_interval_seconds)
            await self.flush()

    async def flush(self):
        """Write buffered data to disk."""
        async with self._buffer_lock:
            if not self._buffer:
                return
            
            # Swap buffer to release lock processing quickly
            current_buffer = self._buffer
            self._buffer = {}
            
        # Process writes without holding the lock
        # Group by symbol -> date
        for symbol, ticks in current_buffer.items():
            if not ticks:
                continue
                
            # We assume all ticks for a flush might span across midnight, 
            # but usually they are same day. We'll group by date just in case.
            ticks_by_date = {}
            for tick in ticks:
                try:
                    ts = datetime.fromisoformat(tick['timestamp'])
                    date_str = ts.strftime("%Y-%m-%d")
                    if date_str not in ticks_by_date:
                        ticks_by_date[date_str] = []
                    ticks_by_date[date_str].append(tick)
                except ValueError:
                    continue

            for date_str, daily_ticks in ticks_by_date.items():
                await self._write_to_csv(symbol, date_str, daily_ticks)

    async def _write_to_csv(self, symbol: str, date_str: str, ticks: List[Dict]):
        try:
            # Directory: data/market_data/intraday/{symbol}/
            symbol_dir = os.path.join(self.data_dir, symbol)
            if not os.path.exists(symbol_dir):
                os.makedirs(symbol_dir, exist_ok=True)
                
            file_path = os.path.join(symbol_dir, f"{date_str}.csv")
            file_exists = os.path.exists(file_path)
            
            # Using run_in_executor for blocking I/O
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._write_file_sync, file_path, file_exists, ticks)
            
        except Exception as e:
            logger.error(f"Failed to write recording for {symbol}: {e}")

    def _write_file_sync(self, file_path, file_exists, ticks):
        fieldnames = ["timestamp", "price", "volume", "bid", "ask"]
        with open(file_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerows(ticks)
