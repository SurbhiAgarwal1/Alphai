

## Overview

This repository contains a complete, production-ready solution for predicting the 95% confidence interval for BTC/USDT price one hour into the future. It strictly follows the challenge guidelines, balancing high coverage (~0.95) with the tightest possible range.

## Approach

1. **Modeling**: 
   - Uses **Geometric Brownian Motion (GBM)**.
   - Leverages a **Student-t distribution** (degrees of freedom = 4) instead of Normal distribution to correctly account for the "fat tails" commonly observed in crypto asset returns.
   - Simulates 10,000 possible price paths for the next hour and extracts the 2.5 and 97.5 percentiles to establish the 95% interval.

2. **No Data Leakage**:
   - The backtesting engine (`backtest.py`) rigorously avoids data leakage. When predicting the price at index $i+1$, the model is strictly provided with `data[:i+1]`. This ensures drift, volatility, and returns are evaluated only up to the present hour $i$.

3. **Metrics**:
   - Computes **Coverage**, **Average Width**, and **Winkler Score**.
   - Winkler score effectively penalizes instances where the actual price falls outside the predicted range while rewarding narrow bounds.

## Advanced Feature: Dynamic Volatility Regime Adjustment

To optimize the balance between coverage and tightness, a dynamic volatility multiplier has been implemented:
- Short-term volatility (10 hours) and long-term volatility (50 hours) are continually calculated.
- If short-term volatility > long-term volatility, the model identifies a high-variance regime and dynamically **widens** the prediction interval to maintain safety/coverage.
- Conversely, if short-term < long-term, the model **tightens** the bounds to reduce the average width and Winkler score.

## Bonus Feature: Prediction Persistence

- The Streamlit dashboard inherently saves every live prediction to `live_predictions.jsonl`.
- Whenever the dashboard is loaded or refreshed, it cross-references the live predicted timestamps with incoming historical data to retrospectively grade past live predictions.

## Setup & Deployment

Install requirements:
```bash
pip install -r requirements.txt
```

Run backtesting engine:
```bash
python backtest.py
```
This script will produce `backtest_results.jsonl` and log out the coverage, width, and Winkler score.

Run Streamlit Dashboard:
```bash
streamlit run app.py
```
