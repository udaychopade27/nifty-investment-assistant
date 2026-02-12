"""
Scheduled ML Retraining Job
Runs weekly to retrain models on accumulated real-time data
"""

import asyncio
import logging
from datetime import datetime, time
from app.utils.time import IST
import subprocess

logger = logging.getLogger(__name__)

class MLRetrainingScheduler:
    """Scheduler for automated ML model retraining."""
    
    def __init__(self):
        self.retrain_day = 6  # Sunday (0=Monday, 6=Sunday)
        self.retrain_time = time(22, 0)  # 10 PM IST
        self._running = False
        self._task = None
    
    async def start(self):
        """Start the retraining scheduler."""
        self._running = True
        self._task = asyncio.create_task(self._schedule_loop())
        logger.info(f"ML Retraining Scheduler started (Every Sunday at 10 PM IST)")
    
    async def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("ML Retraining Scheduler stopped")
    
    async def _schedule_loop(self):
        """Main scheduling loop."""
        while self._running:
            now = datetime.now(IST)
            
            # Check if it's Sunday at 10 PM
            if now.weekday() == self.retrain_day:
                current_time = now.time()
                target_time = self.retrain_time
                
                # Check if we're within the retraining window (10:00 PM - 10:05 PM)
                if target_time <= current_time < time(target_time.hour, target_time.minute + 5):
                    logger.info("🔄 Starting scheduled ML retraining...")
                    await self._run_retraining()
                    
                    # Sleep for 1 hour to avoid re-triggering
                    await asyncio.sleep(3600)
                    continue
            
            # Check every 5 minutes
            await asyncio.sleep(300)
    
    async def _run_retraining(self):
        """Execute the retraining script."""
        try:
            logger.info("Executing retrain_continuous.py...")
            
            # Run in subprocess to avoid blocking
            process = await asyncio.create_subprocess_exec(
                'python3', 'scripts/retrain_continuous.py',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                logger.info("✅ ML retraining completed successfully")
                logger.info(f"Output: {stdout.decode()[:500]}")
            else:
                logger.error(f"❌ ML retraining failed: {stderr.decode()}")
        
        except Exception as e:
            logger.error(f"Failed to run retraining: {e}")
    
    async def trigger_manual_retrain(self):
        """Manually trigger retraining (for API endpoint)."""
        logger.info("Manual retraining triggered")
        await self._run_retraining()
