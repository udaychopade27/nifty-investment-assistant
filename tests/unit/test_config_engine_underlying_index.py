from pathlib import Path

import pytest

from app.domain.services.config_engine import ConfigEngine


@pytest.mark.unit
def test_config_engine_loads_underlying_index():
    config_dir = Path(__file__).resolve().parents[2] / "config"
    engine = ConfigEngine(config_dir)
    engine.load_all()

    etf = engine.etf_universe.get_etf("NIFTYBEES")
    assert etf.underlying_index == "NIFTY 50"

    gold = engine.etf_universe.get_etf("HDFCGOLD")
    assert gold.underlying_index is not None
