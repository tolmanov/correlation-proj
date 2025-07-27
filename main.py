from fastapi import FastAPI

import numpy as np
import yfinance as yf
from datetime import date
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:5500"] etc.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CorrRequest(BaseModel):
    tickers: list[str]
    start: date
    end: date


@app.get("/")
async def root():
    return {"Hello": "Correlation"}


@app.post("/correlation")
async def get_data(req: CorrRequest):
    dat = yf.Tickers(req.tickers)
    data = dat.history(period=None, start=req.start, end=req.end)
    print(data["Close"])
    # TODO: Think of a period for the change
    # TODO:What if some stocks are not available for a certain period of time
    mat = data["Close"].pct_change().corr()
    mat = mat.replace(np.nan, None)
    print(mat)
    # print(dat.option_chain(dat.options[0]).calls)
    return {"tickers": mat.columns.to_list(), "correlation": mat.values.tolist()}
