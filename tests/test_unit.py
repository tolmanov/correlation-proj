import pickle
from datetime import date
from unittest.mock import patch, Mock

import pandas as pd
import pytest
from fakeredis.aioredis import FakeRedis

from corr.main import app  # your FastAPI app


@pytest.fixture
async def fake_redis():
    redis = await FakeRedis()

    dates = pd.date_range("2024-07-01", periods=3)
    ts1 = pd.Series([100.0, 101.0, 102.0], index=dates)
    ts2 = pd.Series([200.0, 198.0, 202.0], index=dates)

    await redis.set("AAPL", pickle.dumps(ts1))
    await redis.set("GOOG", pickle.dumps(ts2))

    return redis


@pytest.fixture
async def async_client(fake_redis):
    app.state.redis_client = fake_redis

    # Proper async testing setup with ASGITransport
    from httpx import AsyncClient
    from httpx._transports.asgi import ASGITransport

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# ---------- Endpoint: / (root) ----------
@pytest.mark.asyncio
async def test_root_get(async_client):
    response = await async_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Hello": "Correlation"}


@pytest.mark.asyncio
async def test_root_head(async_client):
    response = await async_client.head("/")
    assert response.status_code == 200


# ---------- Endpoint: /tickers ----------
@pytest.mark.asyncio
async def test_get_tickers(async_client):
    response = await async_client.get("/tickers")
    assert response.status_code == 200
    assert sorted(response.json()) == ["AAPL", "GOOG"]


# ---------- Endpoint: /correlation ----------
@pytest.mark.asyncio
async def test_correlation_endpoint(async_client, fake_redis):
    payload = {
        "tickers": ["AAPL", "GOOG"],
        "start": "2024-07-01",
        "end": "2024-07-03",
        "return_period": 1,
        "min_periods": None,
    }

    response = await async_client.post("/correlation", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "tickers" in data
    assert "correlation" in data
    assert len(data["correlation"]) == 2  # 2x2 matrix
    assert len(data["correlation"][0]) == 2  # 2 columns in each row


@pytest.mark.asyncio
async def test_correlation_with_min_periods(async_client, fake_redis):
    payload = {
        "tickers": ["AAPL", "GOOG"],
        "start": "2024-07-01",
        "end": "2024-07-03",
        "return_period": 1,
        "min_periods": 2,
    }

    response = await async_client.post("/correlation", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "tickers" in data
    assert "correlation" in data
    assert len(data["correlation"]) == 2
    assert len(data["correlation"][0]) == 2


@pytest.mark.asyncio
async def test_correlation_different_return_period(async_client, fake_redis):
    payload = {
        "tickers": ["AAPL", "GOOG"],
        "start": "2024-07-01",
        "end": "2024-07-03",
        "return_period": 2,
        "min_periods": None,
    }

    response = await async_client.post("/correlation", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "tickers" in data
    assert "correlation" in data


@pytest.mark.asyncio
async def test_correlation_invalid_ticker(async_client, fake_redis):
    payload = {
        "tickers": ["AAPL", "NONEXISTENT"],
        "start": "2024-07-01",
        "end": "2024-07-03",
        "return_period": 1,
        "min_periods": None,
    }

    # This should fail because NONEXISTENT ticker doesn't exist in redis
    with pytest.raises(Exception):
        _ = await async_client.post("/correlation", json=payload)


@pytest.mark.asyncio
async def test_correlation_date_range_subset(async_client, fake_redis):
    payload = {
        "tickers": ["AAPL", "GOOG"],
        "start": "2024-07-01",
        "end": "2024-07-02",  # Only first two days
        "return_period": 1,
        "min_periods": None,
    }

    response = await async_client.post("/correlation", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "tickers" in data
    assert "correlation" in data


# ---------- Endpoint: /cache ----------
@pytest.mark.asyncio
@patch("corr.main.yf.Tickers")
async def test_cache_endpoint_success(mock_yf_tickers, async_client, fake_redis):
    # Mock yfinance data with proper MultiIndex columns structure
    mock_history_data = pd.DataFrame(
        {
            ("Close", "MSFT"): [300.0, 301.0, 302.0],
            ("Close", "TSLA"): [400.0, 398.0, 405.0],
        }
    )
    mock_history_data.columns = pd.MultiIndex.from_tuples(
        [("Close", "MSFT"), ("Close", "TSLA")]
    )
    mock_history_data.index = pd.date_range("2024-07-01", periods=3)

    # Create mock Tickers object (synchronous, not async)
    mock_tickers = Mock()
    mock_tickers.history.return_value = mock_history_data
    mock_yf_tickers.return_value = mock_tickers

    payload = {"tickers": ["MSFT", "TSLA"], "start": "2024-07-01", "end": "2024-07-03"}

    response = await async_client.post("/cache", json=payload)
    assert response.status_code == 200

    # Verify data was stored in redis
    msft_data = await fake_redis.get("MSFT")
    tsla_data = await fake_redis.get("TSLA")

    assert msft_data is not None
    assert tsla_data is not None

    # Verify the stored data
    msft_series = pickle.loads(msft_data)
    tsla_series = pickle.loads(tsla_data)

    assert len(msft_series) == 3
    assert len(tsla_series) == 3
    assert msft_series.iloc[0] == 300.0
    assert tsla_series.iloc[0] == 400.0


@pytest.mark.asyncio
@patch("corr.main.yf.Tickers")
async def test_cache_endpoint_single_ticker(mock_yf_tickers, async_client, fake_redis):
    # Mock yfinance data for single ticker
    mock_history_data = pd.DataFrame({("Close", "NVDA"): [500.0, 505.0, 510.0]})
    mock_history_data.columns = pd.MultiIndex.from_tuples([("Close", "NVDA")])
    mock_history_data.index = pd.date_range("2024-07-01", periods=3)

    mock_tickers = Mock()
    mock_tickers.history.return_value = mock_history_data
    mock_yf_tickers.return_value = mock_tickers

    payload = {"tickers": ["NVDA"], "start": "2024-07-01", "end": "2024-07-03"}

    response = await async_client.post("/cache", json=payload)
    assert response.status_code == 200

    # Verify data was stored
    nvda_data = await fake_redis.get("NVDA")
    assert nvda_data is not None

    nvda_series = pickle.loads(nvda_data)
    assert len(nvda_series) == 3
    assert nvda_series.iloc[0] == 500.0


@pytest.mark.asyncio
@patch("corr.main.yf.Tickers")
async def test_cache_endpoint_empty_tickers(mock_yf_tickers, async_client, fake_redis):
    # Mock empty yfinance response
    mock_history_data = pd.DataFrame()

    mock_tickers = Mock()
    mock_tickers.history.return_value = mock_history_data
    mock_yf_tickers.return_value = mock_tickers

    payload = {"tickers": [], "start": "2024-07-01", "end": "2024-07-03"}

    response = await async_client.post("/cache", json=payload)
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_cache_request_validation():
    # Test invalid payload structure
    from corr.main import CacheRequest
    from pydantic import ValidationError

    # Test missing required fields
    with pytest.raises(ValidationError):
        CacheRequest(tickers=["AAPL"])  # missing start and end

    # Test invalid date format (this would be caught by FastAPI validation)
    with pytest.raises(ValidationError):
        CacheRequest(tickers=["AAPL"], start="invalid-date", end="2024-07-03")


@pytest.mark.asyncio
async def test_correlation_request_validation():
    from corr.main import CorrRequest

    # Test valid request with defaults
    req = CorrRequest(
        tickers=["AAPL", "GOOG"], start=date(2024, 7, 1), end=date(2024, 7, 3)
    )
    assert req.return_period == 1  # default value
    assert req.min_periods is None  # default value

    # Test with custom values
    req = CorrRequest(
        tickers=["AAPL", "GOOG"],
        start=date(2024, 7, 1),
        end=date(2024, 7, 3),
        return_period=5,
        min_periods=10,
    )
    assert req.return_period == 5
    assert req.min_periods == 10


# ---------- Integration test: Cache then Correlate ----------
@pytest.mark.asyncio
@patch("corr.main.yf.Tickers")
async def test_cache_then_correlate_integration(
    mock_yf_tickers, async_client, fake_redis
):
    # Clear existing data
    await fake_redis.flushall()

    # Mock yfinance data with proper MultiIndex structure
    mock_history_data = pd.DataFrame(
        {
            ("Close", "AMZN"): [150.0, 152.0, 151.0, 153.0],
            ("Close", "META"): [300.0, 305.0, 302.0, 308.0],
        }
    )
    mock_history_data.columns = pd.MultiIndex.from_tuples(
        [("Close", "AMZN"), ("Close", "META")]
    )
    mock_history_data.index = pd.date_range("2024-07-01", periods=4)

    mock_tickers = Mock()
    mock_tickers.history.return_value = mock_history_data
    mock_yf_tickers.return_value = mock_tickers

    # First, cache the data
    cache_payload = {
        "tickers": ["AMZN", "META"],
        "start": "2024-07-01",
        "end": "2024-07-04",
    }

    cache_response = await async_client.post("/cache", json=cache_payload)
    assert cache_response.status_code == 200

    # Then, get correlation
    corr_payload = {
        "tickers": [
            "AMZN",
            "GOOG",
            "META",
        ],  # Checking that a missing ticker is ignored
        "start": "2024-07-01",
        "end": "2024-07-04",
        "return_period": 1,
        "min_periods": None,
    }

    corr_response = await async_client.post("/correlation", json=corr_payload)
    assert corr_response.status_code == 200

    data = corr_response.json()
    assert len(data["tickers"]) == 2
    assert data["tickers"] == ["AMZN", "META"]
    assert len(data["correlation"]) == 2
    assert len(data["correlation"][0]) == 2

    # Diagonal should be 1.0 (perfect self-correlation)
    assert data["correlation"][0][0] == 1.0
    assert data["correlation"][1][1] == 1.0


# ---------- Error handling tests ----------
@pytest.mark.asyncio
async def test_correlation_invalid_json(async_client):
    response = await async_client.post("/correlation", content="invalid json")
    assert response.status_code == 422  # Unprocessable Entity


@pytest.mark.asyncio
async def test_cache_invalid_json(async_client):
    response = await async_client.post("/cache", content="invalid json")
    assert response.status_code == 422  # Unprocessable Entity
