"""
main_documented.py

This FastAPI application provides a backend service to cache and compute correlations
between stock tickers over a given date range using data from Yahoo Finance.
It uses Redis for caching both the time series data and associated metadata.
"""

import logging
import pickle
from contextlib import asynccontextmanager
from datetime import date
from logging.config import dictConfig
from typing import Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from redis.asyncio import Redis
from starlette.requests import Request

from src.corr import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize Redis client and configure logging on application startup,
    then close the Redis connection on shutdown.
    """
    print(settings.redis_url)
    app.state.redis_client = await Redis.from_url(settings.redis_url)
    dictConfig(settings.logging.to_dict())
    yield
    await app.state.redis_client.close()


# Create FastAPI app with lifespan handler
app = FastAPI(lifespan=lifespan)

# Allow CORS from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request and response models
class CacheRequest(BaseModel):
    tickers: list[str]
    start: date
    end: date


class CorrRequest(BaseModel):
    tickers: list[str]
    start: date
    end: date
    return_period: int = 1
    min_periods: int | None = None


class CorrResponse(BaseModel):
    tickers: list[str]
    correlation: list[list[float | None]]
    status: str = "success"


class ErrorResponse(BaseModel):
    status: str = "fail"
    message: str


@app.get("/", include_in_schema=False)
@app.head("/", include_in_schema=False)
async def root():
    """Simple health check endpoint."""
    return {"Hello": "Correlation"}


@app.get("/tickers")
async def get_tickers(request: Request) -> list[str]:
    """
    Retrieve list of available tickers from Redis,
    excluding internal metadata keys.
    """
    redis = request.app.state.redis_client
    tickers = await redis.keys()
    tickers = [
        t.decode() if isinstance(t, bytes) else t
        for t in tickers
        if not (
            isinstance(t, bytes) and t.decode().startswith(settings.metadata_prefix)
        )
        and not (isinstance(t, str) and t.startswith(settings.metadata_prefix))
    ]
    return list(sorted(tickers))


async def get_ticker_metadata(redis: Redis, ticker: str) -> Dict[str, Any] | None:
    """Get metadata for a specific ticker from Redis."""
    metadata_key = f"{settings.metadata_prefix}{ticker}"
    metadata = await redis.get(metadata_key)
    if metadata:
        return pickle.loads(metadata)
    return None


async def set_ticker_metadata(
    redis: Redis, ticker: str, start_date: date, end_date: date
):
    """Save metadata for a ticker to Redis."""
    metadata_key = f"{settings.metadata_prefix}{ticker}"
    metadata = {"start_date": start_date, "end_date": end_date}
    await redis.set(metadata_key, pickle.dumps(metadata))


async def validate_date_range(
    redis: Redis, tickers: list[str], start_date: date, end_date: date
) -> tuple[bool, str]:
    """
    Ensure requested date range is covered by cached metadata
    for each of the tickers.
    """
    for ticker in tickers:
        metadata = await get_ticker_metadata(redis, ticker)
        if not metadata:
            return False, f"No cached data found for ticker: {ticker}"

        cached_start = metadata.get("start_date")
        cached_end = metadata.get("end_date")

        if not cached_start or not cached_end:
            return False, f"Invalid metadata for ticker: {ticker}"

        if start_date < cached_start:
            return (
                False,
                f"Start date {start_date} is before cached start date {cached_start} for ticker: {ticker}",
            )

        if end_date > cached_end:
            return (
                False,
                f"End date {end_date} is after cached end date {cached_end} for ticker: {ticker}",
            )

    return True, ""


async def get_data_redis(tickers, redis):
    """Retrieve and combine time series data for tickers from Redis."""
    logger.info(f"Querying Redis for {len(tickers)} tickers")
    async with redis.pipeline(transaction=False) as pipe:
        for ticker in tickers:
            pipe.get(ticker)
        data = await pipe.execute()
    logger.info("Creating a DataFrame and calculating the correlation.")
    data = {t: pickle.loads(datum) for t, datum in zip(tickers, data) if datum}
    prices = pd.concat(data, axis=1)
    return prices


@app.post("/correlation", response_model=CorrResponse | ErrorResponse)
async def get_data(req: CorrRequest, request: Request):
    """
    Compute and return a correlation matrix for the given tickers over a date range.
    Validates date range against cached metadata before processing.
    """
    redis = request.app.state.redis_client

    is_valid, error_message = await validate_date_range(
        redis, req.tickers, req.start, req.end
    )
    if not is_valid:
        logger.warning(f"Date range validation failed: {error_message}")
        raise HTTPException(
            status_code=400, detail={"status": "fail", "message": error_message}
        )

    try:
        prices = await get_data_redis(req.tickers, redis)
        prices = prices.loc[req.start : req.end]
        mat = prices.pct_change(periods=req.return_period, fill_method=None).corr(
            min_periods=req.min_periods
        )
        mat = mat.replace(np.nan, None)
        logger.info("Calculation completed.")
        return CorrResponse(
            tickers=mat.columns.to_list(),
            correlation=mat.values.tolist(),
            status="success",
        )
    except Exception as e:
        logger.error(f"Error calculating correlation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "fail",
                "message": f"Error calculating correlation: {str(e)}",
            },
        )


@app.post("/cache")
async def cache(payload: CacheRequest, request: Request):
    """
    Download historical data from Yahoo Finance for selected tickers,
    store them in Redis, and save associated metadata.
    """
    redis = request.app.state.redis_client
    logger.info("Getting tickers from Yahoo")
    dat = yf.Tickers(payload.tickers)
    hist = dat.history(period=None, start=payload.start, end=payload.end)
    if hist.empty:
        return

    close_px = hist["Close"]
    logger.info("Storing them in redis")
    async with redis.pipeline(transaction=False) as pipe:
        for ticker in close_px.columns:
            pipe.set(ticker, pickle.dumps(close_px[ticker]))
        res = await pipe.execute()

    logger.info("Storing metadata for tickers")
    async with redis.pipeline(transaction=False) as pipe:
        for ticker in close_px.columns:
            await set_ticker_metadata(redis, ticker, payload.start, payload.end)

    logger.info("Saving completed.")
    return res
