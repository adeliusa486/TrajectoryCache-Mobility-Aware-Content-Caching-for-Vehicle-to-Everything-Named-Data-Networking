"""Integration tests for the FastAPI monitoring interface."""

import pytest
from httpx import AsyncClient, ASGITransport

from api.main import app


@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "uptime_s" in data


@pytest.mark.asyncio
async def test_get_config():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/config")
    assert r.status_code == 200
    data = r.json()
    assert "trajectoryCache" in data
    assert "lambda_weight" in data["trajectoryCache"]


@pytest.mark.asyncio
async def test_update_config():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/config", json={"lambda_weight": 0.6})
    assert r.status_code == 200
    data = r.json()
    assert data["config"]["trajectoryCache"]["lambda_weight"] == 0.6


@pytest.mark.asyncio
async def test_update_config_invalid_lambda():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/config", json={"lambda_weight": 1.5})
    assert r.status_code == 422  # Pydantic validation error


@pytest.mark.asyncio
async def test_run_experiment_tc():
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", timeout=60.0
    ) as client:
        r = await client.post("/experiments/run", json={
            "scenario": "highway",
            "n_vehicles": 20,
            "duration_s": 20.0,
            "seed": 42,
            "policy": "tc",
        })
    assert r.status_code == 200
    data = r.json()
    assert "miss_rate" in data
    assert 0.0 <= data["miss_rate"] <= 1.0


@pytest.mark.asyncio
async def test_run_experiment_invalid_scenario():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/experiments/run", json={
            "scenario": "mars",
            "n_vehicles": 10,
            "duration_s": 10.0,
            "seed": 1,
            "policy": "tc",
        })
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_run_experiment_invalid_policy():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/experiments/run", json={
            "scenario": "highway",
            "n_vehicles": 10,
            "duration_s": 10.0,
            "seed": 1,
            "policy": "bogus_policy",
        })
    assert r.status_code == 400
