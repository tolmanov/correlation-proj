# 📈 Stock Correlation Web App

This is a full-stack web application for computing the correlation between multiple stocks over a given time range. Users can select tickers, specify a date range, and view the correlation matrix directly in the browser. The system fetches historical stock data, caches it, computes the correlation matrix, and serves it via a REST API.

## Features

- 🔁 Correlation matrix calculation for multiple stocks  
- 💾 Redis-based data caching to improve speed and reduce API calls  
- 🖥️ Interactive frontend to select stocks and visualize results  
- 🌐 Fully deployed on free-tier cloud platforms

## Live Demo

- **Frontend**: [https://tolmanov.github.io/correlation-proj/](https://tolmanov.github.io/correlation-proj/)  
- **Backend**: [https://correlation-proj.onrender.com](https://correlation-proj.onrender.com)

## 🛠️ Tech Stack

- **Backend**: FastAPI, Redis, yfinance, Pandas, NumPy  
- **Frontend**: Vanilla HTML/CSS/JS  
- **Deployment**: Render (backend), GitHub Pages (frontend)

## 🧠 How It Works

1. The user selects a list of tickers and date range in the frontend.
2. The frontend sends a POST request to the backend to compute the correlation matrix.
3. The backend:
   - Checks Redis for cached data.
   - If data is missing, it fetches from Yahoo Finance and stores it.
   - Computes percentage change and correlation using `pandas`.
   - Returns the matrix as JSON.
4. The frontend renders the matrix with color gradients for easier interpretation.

## API Endpoints

### `GET /`

Health check.

### `GET /tickers`

Returns a list of all cached tickers.

### `POST /cache`

Caches historical closing prices for the requested tickers and date range.

**Payload**:

```json
{
  "tickers": ["AAPL", "MSFT"],
  "start": "2024-01-01",
  "end": "2025-07-01"
}
```

### `POST /correlation`

Calculates the correlation matrix for the given tickers and date range.

**Payload**:

```json
{
  "tickers": ["AAPL", "MSFT", "GOOG"],
  "start": "2024-01-01",
  "end": "2025-07-01"
}
```

**Response**:

```json
{
  "tickers": ["AAPL", "GOOG", "MSFT"],
  "correlation": [
    [1.0, 0.81, 0.93],
    [0.81, 1.0, 0.86],
    [0.93, 0.86, 1.0]
  ]
}
```

## 🚀 Deployment

- **Frontend**: [https://tolmanov.github.io/correlation-proj/](https://tolmanov.github.io/correlation-proj/)  
- **Backend**: [https://correlation-proj.onrender.com](https://correlation-proj.onrender.com)

## 📊 Monitoring

- **Backend Render Dashboard**:  
  [View Deployments](https://dashboard.render.com/web/srv-d2311615pdvs739f6l9g/deploys/dep-d231oa15pdvs739fvsrg)

- **Uptime Monitoring**:  
  [UptimeRobot Dashboard](https://dashboard.uptimerobot.com/monitors)

## 🧪 Local Development

### Requirements

- Python 3.12+
- Redis
- Node.js (optional, for static server)

### Backend

# Install dependencies
uv sync

# Set Redis URL (in .env or settings file)
export CORR_redis_url=redis://localhost:6379

# Run app
uv run --frozen uvicorn corr.main:app
```

### Frontend

Open `index.html` directly or serve it using a simple HTTP server:

## 📌 Limitations

_(To be added)_

## 🧠 Project Goal

This project was created as a full-stack engineering exercise. It highlights:

- Cloud deployment practices  
- API development with FastAPI  
- Async Redis caching  
- Frontend UX with native JavaScript  
- Monitoring and observability integration