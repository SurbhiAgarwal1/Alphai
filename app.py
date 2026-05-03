import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as graph_objects
from datetime import datetime, timedelta
import json
import os

from data import fetch_binance_data
from model import predict_next_hour
from utils import calculate_metrics

st.set_page_config(page_title="BTC Predictor", layout="wide")

# ================================
# Data Fetching
# ================================
@st.cache_data(ttl=300) # Cache for 5 minutes
def get_data():
    return fetch_binance_data(limit=500)

st.title("Predict Bitcoin's Next Hour")

try:
    df = get_data()
except Exception as e:
    st.error(f"Failed to fetch data: {e}")
    st.stop()

# ================================
# Backtest Metrics
# ================================
st.header("Metrics (from Backtest)")
if os.path.exists("backtest_results.jsonl"):
    with open("backtest_results.jsonl", "r") as f:
        results = [json.loads(line) for line in f]
    cov, awidth, wink = calculate_metrics(results)
    st.caption(f"Based on {len(results)} backtest predictions on 1h intervals")
else:
    cov, awidth, wink = 0.0, 0.0, 0.0
    st.caption("No backtest_results.jsonl found. Run backtest.py to generate metrics.")
    
col1, col2, col3 = st.columns(3)
col1.metric("Coverage (Target: ~0.95)", f"{cov:.4f}")
col2.metric("Average Width", f"{awidth:.2f}")
col3.metric("Winkler Score", f"{wink:.2f}")

# ================================
# Live Prediction
# ================================
st.header("Live Prediction")
prices = df['close'].values
last_price = prices[-1]
last_time = df['timestamp'].iloc[-1]
next_time = last_time + timedelta(hours=1)

with st.spinner("Calculating prediction..."):
    lower, upper = predict_next_hour(prices)

st.write(f"**Current Price (as of {last_time}):** ${last_price:.2f}")
st.write(f"**Predicted 95% Range for {next_time}:** ${lower:.2f} — ${upper:.2f}")

# ================================
# Chart
# ================================
fig = graph_objects.Figure()

# Plot last 50 candles close price
plot_df = df.tail(50).copy()
fig.add_trace(graph_objects.Scatter(
    x=plot_df['timestamp'],
    y=plot_df['close'],
    mode='lines',
    name='Close Price',
    line=dict(color='cyan')
))

# Add prediction band
fig.add_trace(graph_objects.Scatter(
    x=[last_time, next_time, next_time, last_time],
    y=[last_price, upper, lower, last_price],
    fill='toself',
    fillcolor='rgba(255, 0, 0, 0.3)',
    line=dict(color='rgba(255,0,0,0)'),
    name='95% Prediction Interval'
))

# Chart styling
fig.update_layout(
    template="plotly_dark",
    title="BTC/USDT - Last 50 Hours & Next Hour Prediction",
    xaxis_title="Time",
    yaxis_title="Price (USDT)",
    hovermode="x unified",
    margin=dict(l=0, r=0, t=40, b=0)
)

st.plotly_chart(fig, use_container_width=True)

# ================================
# Prediction Persistence
# ================================
st.header("Past Live Predictions")

PREDICTIONS_FILE = "live_predictions.jsonl"
current_pred = {
    "timestamp": str(next_time),
    "lower": lower,
    "upper": upper,
    "actual": None
}

past_preds = []
if os.path.exists(PREDICTIONS_FILE):
    with open(PREDICTIONS_FILE, "r") as f:
        for line in f:
            past_preds.append(json.loads(line))

# Update actuals for past predictions
updated_preds = []
df_times = df['timestamp'].astype(str).values
df_closes = df['close'].values
time_to_close = dict(zip(df_times, df_closes))

for p in past_preds:
    if p["actual"] is None:
        if p["timestamp"] in time_to_close:
            p["actual"] = float(time_to_close[p["timestamp"]])
    updated_preds.append(p)

# Only add current prediction if it's not already there
if not any(p["timestamp"] == current_pred["timestamp"] for p in updated_preds):
    updated_preds.append(current_pred)

# Save back
with open(PREDICTIONS_FILE, "w") as f:
    for p in updated_preds:
        f.write(json.dumps(p) + "\n")

if updated_preds:
    disp_df = pd.DataFrame(updated_preds).sort_values("timestamp", ascending=False)
    # Format for display
    disp_df['lower'] = disp_df['lower'].apply(lambda x: f"${x:.2f}")
    disp_df['upper'] = disp_df['upper'].apply(lambda x: f"${x:.2f}")
    disp_df['actual'] = disp_df['actual'].apply(lambda x: f"${x:.2f}" if pd.notnull(x) else "Pending")
    
    st.dataframe(disp_df, use_container_width=True)
