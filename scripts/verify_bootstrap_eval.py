import asyncio
from unittest.mock import MagicMock, patch
from app.domain.options.runtime import OptionsRuntime
import json

async def verify_eval():
    mock_config = MagicMock()
    mock_runtime = MagicMock()
    
    # Mock config to point to the new samples file
    mock_config.get_options_setting.return_value = {
        "signal": {
            "ml": {"samples_path": "data/options_paper_samples.jsonl"}
        }
    }
    
    # Avoid real model loading
    with patch('joblib.load') as mock_load:
        mock_load.return_value = MagicMock()
        runtime = OptionsRuntime(mock_config, mock_runtime)
        
        logger_mock = MagicMock()
        with patch('app.domain.options.runtime.logger', logger_mock):
            # Run evaluation
            report = runtime.evaluate_ml_walk_forward()
            print("Evaluation Report on Bootstrapped Samples:")
            print(json.dumps(report, indent=2))
            
            # Run samples status
            status = runtime.get_ml_samples_status()
            print("\nSamples Status:")
            print(json.dumps(status, indent=2))

if __name__ == "__main__":
    asyncio.run(verify_eval())
