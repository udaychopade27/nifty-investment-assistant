import pytest


@pytest.mark.asyncio
@pytest.mark.integration
async def test_config_etfs_exposes_underlying_index(client):
    resp = await client.get("/api/v1/config/etfs")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    symbols = {d["symbol"] for d in data}
    assert "NIFTYBEES" in symbols
    entry = next(d for d in data if d["symbol"] == "NIFTYBEES")
    assert entry.get("underlying_index") == "NIFTY 50"
