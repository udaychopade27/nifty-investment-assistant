import subprocess
import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_step(command, description):
    logger.info(f"--- STEP: {description} ---")
    try:
        # Use docker exec if running from host, or just python3 if inside container
        # Checking if inside docker
        if os.path.exists("/.dockerenv"):
            cmd = ["python3"] + command.split()
        else:
            cmd = ["docker", "exec", "etf_app", "python3"] + command.split()
            
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            logger.error(f"Step failed: {description}")
            return False
        logger.info(f"✅ Success: {description}")
        return True
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        return False

def main():
    logger.info("Starting Full Server Setup for Options Trading...")
    
    # 1. Fetch data
    if not run_step("scripts/fetch_upstox_historical.py", "Fetching Historical Data from Upstox"):
        sys.exit(1)
        
    # 2. Train and generate all models
    if not run_step("scripts/train_on_historical.py", "Training High-Fidelity ML Models"):
        sys.exit(1)
        
    # 3. Verify ensemble logic
    # Set PYTHONPATH if needed inside container
    if os.path.exists("/.dockerenv"):
        os.environ["PYTHONPATH"] = "/app"
        if not run_step("scripts/verify_ml_ensemble.py", "Verifying ML Ensemble Integration"):
            sys.exit(1)
    else:
        # From host, pass env var
        cmd = ["docker", "exec", "-e", "PYTHONPATH=/app", "etf_app", "python3", "scripts/verify_ml_ensemble.py"]
        logger.info("--- STEP: Verifying ML Ensemble Integration ---")
        res = subprocess.run(cmd)
        if res.returncode != 0:
            sys.exit(1)

    logger.info("="*60)
    logger.info("🚀 SERVER SETUP COMPLETE!")
    logger.info("All models are trained, verified, and ready for trading.")
    logger.info("="*60)

if __name__ == "__main__":
    main()
