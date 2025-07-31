import pickle
from datetime import date
from unittest.mock import patch, Mock

import pandas as pd
import pytest
from fakeredis.aioredis import FakeRedis

from corr.main import app  # your FastAPI app
from corr import settings


@pytest.fixture
async def fake_redis():
    redis = await FakeRedis()

    dates = pd.date_range("2024-07-01", periods=3)
    ts1 = pd.Series([100.0, 101.0, 102.0], index=dates)
    ts2 = pd.Series([200.0, 198.0, 202.0], index=dates)

    await redis.set("AAPL", pickle.dumps(ts1))
    await redis.set("GOOG", pickle.dumps(ts2))

    # Add metadata for test tickers
    metadata_aapl = {"start_date": date(2024, 7, 1), "end_date": date(2024, 7, 3)}
    metadata_goog = {"start_date": date(2024, 7, 1), "end_date": date(2024, 7, 3)}

    await redis.set(f"{settings.metadata_prefix}AAPL", pickle.dumps(metadata_aapl))
    await redis.set(f"{settings.metadata_prefix}GOOG", pickle.dumps(metadata_goog))

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
    tickers = response.json()
    assert sorted(tickers) == ["AAPL", "GOOG"]
    # Ensure metadata keys are filtered out
    assert not any(ticker.startswith(settings.metadata_prefix) for ticker in tickers)


@pytest.mark.asyncio
async def test_get_tickers_filters_metadata_keys(async_client, fake_redis):
    # Add some additional metadata keys
    await fake_redis.set(
        f"{settings.metadata_prefix}TEST", pickle.dumps({"test": "data"})
    )

    response = await async_client.get("/tickers")
    assert response.status_code == 200
    tickers = response.json()

    # Should only return actual ticker names, not metadata keys
    assert sorted(tickers) == ["AAPL", "GOOG"]
    assert f"{settings.metadata_prefix}TEST" not in tickers


# ---------- Endpoint: /correlation with metadata validation ----------
@pytest.mark.asyncio
async def test_correlation_endpoint_success(async_client, fake_redis):
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
    assert data["status"] == "success"
    assert "tickers" in data
    assert "correlation" in data
    assert len(data["correlation"]) == 2  # 2x2 matrix
    assert len(data["correlation"][0]) == 2  # 2 columns in each row


@pytest.mark.asyncio
async def test_correlation_date_range_validation_start_too_early(
    async_client, fake_redis
):
    payload = {
        "tickers": ["AAPL", "GOOG"],
        "start": "2024-06-30",  # Before cached start date (2024-07-01)
        "end": "2024-07-03",
        "return_period": 1,
        "min_periods": None,
    }

    response = await async_client.post("/correlation", json=payload)
    assert response.status_code == 400

    data = response.json()
    assert data["detail"]["status"] == "fail"
    assert "before cached start date" in data["detail"]["message"]


@pytest.mark.asyncio
async def test_correlation_date_range_validation_end_too_late(async_client, fake_redis):
    payload = {
        "tickers": ["AAPL", "GOOG"],
        "start": "2024-07-01",
        "end": "2024-07-04",  # After cached end date (2024-07-03)
        "return_period": 1,
        "min_periods": None,
    }

    response = await async_client.post("/correlation", json=payload)
    assert response.status_code == 400

    data = response.json()
    assert data["detail"]["status"] == "fail"
    assert "after cached end date" in data["detail"]["message"]


@pytest.mark.asyncio
async def test_correlation_missing_ticker_metadata(async_client, fake_redis):
    # Add a ticker without metadata
    dates = pd.date_range("2024-07-01", periods=3)
    ts_no_meta = pd.Series([150.0, 151.0, 152.0], index=dates)
    await fake_redis.set("NOMETA", pickle.dumps(ts_no_meta))

    payload = {
        "tickers": ["AAPL", "NOMETA"],
        "start": "2024-07-01",
        "end": "2024-07-03",
        "return_period": 1,
        "min_periods": None,
    }

    response = await async_client.post("/correlation", json=payload)
    assert response.status_code == 400

    data = response.json()
    assert data["detail"]["status"] == "fail"
    assert "No cached data found for ticker: NOMETA" in data["detail"]["message"]


@pytest.mark.asyncio
async def test_correlation_nonexistent_ticker(async_client, fake_redis):
    payload = {
        "tickers": ["AAPL", "NONEXISTENT"],
        "start": "2024-07-01",
        "end": "2024-07-03",
        "return_period": 1,
        "min_periods": None,
    }

    response = await async_client.post("/correlation", json=payload)
    assert response.status_code == 400

    data = response.json()
    assert data["detail"]["status"] == "fail"
    assert "No cached data found for ticker: NONEXISTENT" in data["detail"]["message"]


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
    assert data["status"] == "success"
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
    assert data["status"] == "success"
    assert "tickers" in data
    assert "correlation" in data


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
    assert data["status"] == "success"
    assert "tickers" in data
    assert "correlation" in data


# ---------- Endpoint: /cache with metadata storage ----------
@pytest.mark.asyncio
@patch("corr.main.yf.Tickers")
async def test_cache_endpoint_success_with_metadata(
    mock_yf_tickers, async_client, fake_redis
):
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

    # Verify metadata was stored
    msft_metadata = await fake_redis.get(f"{settings.metadata_prefix}MSFT")
    tsla_metadata = await fake_redis.get(f"{settings.metadata_prefix}TSLA")

    assert msft_metadata is not None
    assert tsla_metadata is not None

    msft_meta = pickle.loads(msft_metadata)
    tsla_meta = pickle.loads(tsla_metadata)

    assert msft_meta["start_date"] == date(2024, 7, 1)
    assert msft_meta["end_date"] == date(2024, 7, 3)
    assert tsla_meta["start_date"] == date(2024, 7, 1)
    assert tsla_meta["end_date"] == date(2024, 7, 3)

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

    # Verify metadata was stored
    nvda_metadata = await fake_redis.get(f"{settings.metadata_prefix}NVDA")
    assert nvda_metadata is not None

    nvda_meta = pickle.loads(nvda_metadata)
    assert nvda_meta["start_date"] == date(2024, 7, 1)
    assert nvda_meta["end_date"] == date(2024, 7, 3)

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
    from src.corr.main import CacheRequest
    from pydantic import ValidationError

    # Test missing required fields
    with pytest.raises(ValidationError):
        CacheRequest(tickers=["AAPL"])  # missing start and end

    # Test invalid date format (this would be caught by FastAPI validation)
    with pytest.raises(ValidationError):
        CacheRequest(tickers=["AAPL"], start="invalid-date", end="2024-07-03")


@pytest.mark.asyncio
async def test_correlation_request_validation():
    from src.corr.main import CorrRequest

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

    # Then, get correlation with valid date range
    corr_payload = {
        "tickers": ["AMZN", "META"],  # Only use cached tickers
        "start": "2024-07-01",
        "end": "2024-07-04",
        "return_period": 1,
        "min_periods": None,
    }

    corr_response = await async_client.post("/correlation", json=corr_payload)
    assert corr_response.status_code == 200

    data = corr_response.json()
    assert data["status"] == "success"
    assert len(data["tickers"]) == 2
    assert set(data["tickers"]) == {"AMZN", "META"}
    assert len(data["correlation"]) == 2
    assert len(data["correlation"][0]) == 2

    # Diagonal should be 1.0 (perfect self-correlation)
    assert data["correlation"][0][0] == 1.0
    assert data["correlation"][1][1] == 1.0


@pytest.mark.asyncio
@patch("corr.main.yf.Tickers")
async def test_cache_then_correlate_with_missing_ticker(
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

    # Then, try to get correlation including an uncached ticker
    corr_payload = {
        "tickers": ["AMZN", "META", "GOOG"],  # GOOG is not cached
        "start": "2024-07-01",
        "end": "2024-07-04",
        "return_period": 1,
        "min_periods": None,
    }

    corr_response = await async_client.post("/correlation", json=corr_payload)
    assert corr_response.status_code == 400

    data = corr_response.json()
    assert data["detail"]["status"] == "fail"
    assert "No cached data found for ticker: GOOG" in data["detail"]["message"]


# ---------- Error handling tests ----------
@pytest.mark.asyncio
async def test_correlation_invalid_json(async_client):
    response = await async_client.post("/correlation", content="invalid json")
    assert response.status_code == 422  # Unprocessable Entity


@pytest.mark.asyncio
async def test_cache_invalid_json(async_client):
    response = await async_client.post("/cache", content="invalid json")
    assert response.status_code == 422  # Unprocessable Entity


@pytest.mark.asyncio
async def test_correlation_server_error_handling(async_client, fake_redis):
    # Create a scenario that might cause a server error
    # For example, corrupted data in Redis
    await fake_redis.set("CORRUPT", b"not-pickle-data")

    # Add metadata for the corrupt ticker
    metadata = {"start_date": date(2024, 7, 1), "end_date": date(2024, 7, 3)}
    await fake_redis.set(f"{settings.metadata_prefix}CORRUPT", pickle.dumps(metadata))

    payload = {
        "tickers": ["CORRUPT"],
        "start": "2024-07-01",
        "end": "2024-07-03",
        "return_period": 1,
        "min_periods": None,
    }

    response = await async_client.post("/correlation", json=payload)
    assert response.status_code == 500

    data = response.json()
    assert data["detail"]["status"] == "fail"
    assert "Error calculating correlation" in data["detail"]["message"]


# ---------- Metadata helper function tests ----------
@pytest.mark.asyncio
async def test_metadata_functions(fake_redis):
    from src.corr.main import get_ticker_metadata, set_ticker_metadata

    # Test setting metadata
    await set_ticker_metadata(fake_redis, "TEST", date(2024, 1, 1), date(2024, 12, 31))

    # Test getting metadata
    metadata = await get_ticker_metadata(fake_redis, "TEST")
    assert metadata is not None
    assert metadata["start_date"] == date(2024, 1, 1)
    assert metadata["end_date"] == date(2024, 12, 31)

    # Test getting non-existent metadata
    metadata = await get_ticker_metadata(fake_redis, "NONEXISTENT")
    assert metadata is None


@pytest.mark.asyncio
async def test_validate_date_range_function(fake_redis):
    from src.corr.main import validate_date_range, set_ticker_metadata

    # Set up metadata for test
    await set_ticker_metadata(fake_redis, "TEST1", date(2024, 1, 1), date(2024, 12, 31))
    await set_ticker_metadata(fake_redis, "TEST2", date(2024, 2, 1), date(2024, 11, 30))

    # Test valid range
    is_valid, message = await validate_date_range(
        fake_redis, ["TEST1", "TEST2"], date(2024, 2, 1), date(2024, 11, 30)
    )
    assert is_valid
    assert message == ""

    # Test invalid range - start too early
    is_valid, message = await validate_date_range(
        fake_redis, ["TEST1", "TEST2"], date(2023, 12, 31), date(2024, 11, 30)
    )
    assert not is_valid
    assert "before cached start date" in message

    # Test invalid range - end too late
    is_valid, message = await validate_date_range(
        fake_redis, ["TEST1", "TEST2"], date(2024, 2, 1), date(2024, 12, 31)
    )
    assert not is_valid
    assert "after cached end date" in message

    # Test missing ticker
    is_valid, message = await validate_date_range(
        fake_redis, ["TEST1", "MISSING"], date(2024, 2, 1), date(2024, 11, 30)
    )
    assert not is_valid
    assert "No cached data found for ticker: MISSING" in message
