from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
import json
import pandas as pd
import numpy as np
from scipy.stats import t
import requests
from datetime import datetime, timedelta

app = FastAPI()

# --- REUSED LOGIC ---
def fetch_binance_data(symbol="BTCUSDT", interval="1h", limit=1000):
    url = "https://data-api.binance.vision/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_base", "taker_quote", "ignore"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df[["timestamp", "open", "high", "low", "close", "volume"]].sort_values("timestamp").reset_index(drop=True)

def predict_next_hour(data_close, short_window=10, long_window=50, num_simulations=10000, df_t=4):
    prices = np.array(data_close)
    if len(prices) < 2: return prices[-1], prices[-1]
    log_returns = np.log(prices[1:] / prices[:-1])
    drift = np.mean(log_returns)
    volatility = np.std(log_returns, ddof=1) if len(log_returns) > 1 else 0.0
    if volatility > 0 and len(prices) > long_window:
        short_vol = np.std(log_returns[-short_window:], ddof=1)
        long_vol = np.std(log_returns[-long_window:], ddof=1)
        if long_vol > 0:
            vol_ratio = np.clip(short_vol / long_vol, 0.5, 2.0)
            adj_vol = volatility * vol_ratio
        else: adj_vol = volatility
    else: adj_vol = volatility
    std_t = np.sqrt(df_t / (df_t - 2)) if df_t > 2 else 1.0
    random_shocks = t.rvs(df=df_t, size=num_simulations) / std_t
    simulated_returns = drift - 0.5 * (adj_vol ** 2) + adj_vol * random_shocks
    simulated_prices = prices[-1] * np.exp(simulated_returns)
    return float(np.percentile(simulated_prices, 2.5)), float(np.percentile(simulated_prices, 97.5))

def calculate_metrics(results, alpha=0.05):
    if not results: return 0, 0, 0
    coverage_count = 0
    total_width = 0
    total_winkler = 0
    for res in results:
        L, U, Y = res['lower'], res['upper'], res['actual']
        width = U - L
        total_width += width
        if L <= Y <= U:
            coverage_count += 1
            winkler = width
        elif Y < L: winkler = width + (2 / alpha) * (L - Y)
        else: winkler = width + (2 / alpha) * (Y - U)
        total_winkler += winkler
    n = len(results)
    return coverage_count/n, total_width/n, total_winkler/n

@app.get("/api/data")
async def get_dashboard_data():
    try:
        # Fetch data
        df = fetch_binance_data(limit=500)
        prices = df['close'].values
        last_price = prices[-1]
        last_time = df['timestamp'].iloc[-1]
        next_time = last_time + timedelta(hours=1)
        
        # Prediction
        lower, upper = predict_next_hour(prices)
        
        # Metrics
        metrics = {"coverage": 0, "avg_width": 0, "winkler": 0}
        base_path = os.path.dirname(os.path.dirname(__file__))
        results_path = os.path.join(base_path, "backtest_results.jsonl")
        
        if os.path.exists(results_path):
            with open(results_path, "r") as f:
                results = [json.loads(line) for line in f]
            cov, aw, wink = calculate_metrics(results)
            metrics = {"coverage": cov, "avg_width": aw, "winkler": wink}
            
        # Chart data (last 50)
        chart_df = df.tail(50)
        chart_data = {
            "timestamps": chart_df['timestamp'].astype(str).tolist(),
            "prices": chart_df['close'].tolist()
        }
        
        return {
            "current_price": last_price,
            "prediction": {"lower": lower, "upper": upper, "time": str(next_time)},
            "metrics": metrics,
            "chart_data": chart_data
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/", response_class=HTMLResponse)
async def read_index():
    base_path = os.path.dirname(os.path.dirname(__file__))
    html_path = os.path.join(base_path, "public", "index.html")
    with open(html_path, "r") as f:
        return f.read()

# For local development
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
