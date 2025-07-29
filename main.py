from contextlib import asynccontextmanager
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from redis.asyncio import Redis
from starlette.requests import Request

from config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.redis_client = await Redis.from_url(
        settings.redis_url, decode_responses=True
    )
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


@app.post("/correlation")
async def get_data(req: CorrRequest, request: Request) -> CorrResponse:
    redis = request.app.state.redis_client
    async with redis.pipeline(transaction=False) as pipe:
        for ticker in req.tickers:
            pipe.zrangebyscore(
                ticker,
                min=int(pd.Timestamp(req.start).timestamp()),
                max=int(pd.Timestamp(req.end).timestamp()),
                withscores=True,
            )
        data = await pipe.execute()
    prices = pd.DataFrame(dict(zip(req.tickers, map(convert_to_ts, data))))
    mat = prices.pct_change(
        periods=req.return_period,
        fill_method=None,
    ).corr(min_periods=req.min_periods)
    # Needed for serialization - FIXME: Find a better way
    mat = mat.replace(np.nan, None)
    # print(dat.option_chain(dat.options[0]).calls)
    return {"tickers": mat.columns.to_list(), "correlation": mat.values.tolist()}


@app.post("/cache")
async def cache(payload: CacheRequest, request: Request):
    redis = request.app.state.redis_client
    dat = yf.Tickers(payload.tickers)
    close_px = dat.history(period=None, start=payload.start, end=payload.end)["Close"]
    for ticker, ts in close_px.items():
        data = {str(price): dt.timestamp() for dt, price in ts.items()}
        await redis.zadd(ticker, data)  # TODO: Come up with a better key in redis
