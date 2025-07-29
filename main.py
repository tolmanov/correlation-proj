import logging
import pickle
from contextlib import asynccontextmanager
from datetime import date
from logging.config import dictConfig

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from redis.asyncio import Redis
from starlette.requests import Request

from config import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis_client = await Redis.from_url(settings.redis_url)
    dictConfig(settings.logging.to_dict())
    yield
    await app.state.redis_client.close()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:5500"] etc.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.get("/", include_in_schema=False)
@app.head("/", include_in_schema=False)
async def root():
    return {"Hello": "Correlation"}


def convert_to_ts(inp: list[tuple[str, float]]):
    return {pd.Timestamp(ts, unit="s"): float(val) for val, ts in inp}


@app.get("/tickers")
async def get_tickers(request: Request) -> list[str]:
    redis = request.app.state.redis_client
    tickers = await redis.keys()
    return list(sorted(tickers))


# @alru_cache()
async def get_data_redis(tickers, redis):
    logger.info(f"Queryng Redis for {len(tickers)} tickers")
    async with redis.pipeline(transaction=False) as pipe:
        for ticker in tickers:
            pipe.get(ticker)
        data = await pipe.execute()
    logger.info("Creating a DataFrame and calculating the correlation.")
    data = [pickle.loads(d) for d in data]
    prices = pd.DataFrame(dict(zip(tickers, data))).sort_index()
    return prices


@app.post("/correlation")
async def get_data(req: CorrRequest, request: Request) -> CorrResponse:
    redis = request.app.state.redis_client
    prices = await get_data_redis(req.tickers, redis)
    prices = prices.loc[req.start : req.end]
    mat = prices.pct_change(
        periods=req.return_period,
        fill_method=None,
    ).corr(min_periods=req.min_periods)
    # Needed for serialization - FIXME: Find a better way
    mat = mat.replace(np.nan, None)
    logger.info("Calculation completed.")
    return {"tickers": mat.columns.to_list(), "correlation": mat.values.tolist()}


@app.post("/cache")
async def cache(payload: CacheRequest, request: Request):
    redis = request.app.state.redis_client
    logger.info("Getting tickers from Yahoo")
    dat = yf.Tickers(payload.tickers)
    close_px = dat.history(period=None, start=payload.start, end=payload.end)["Close"]
    logger.info("Storing them in redis")
    async with redis.pipeline(transaction=False) as pipe:
        for ticker, ts_data in close_px.to_dict().items():
            pipe.set(ticker, pickle.dumps(ts_data))
        await pipe.execute()
    logger.info("Saving completed.")
