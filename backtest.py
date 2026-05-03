import json
import os
from data import fetch_binance_data
from model import predict_next_hour
from utils import calculate_metrics

def run_backtest():
    print("Fetching historical data for backtesting...")
    # Fetch maximum allowed by the public endpoint for this backtest
    df = fetch_binance_data(limit=1000)
    
    if len(df) < 100:
        print("Not enough data to run backtest.")
        return
        
    prices = df['close'].values
    timestamps = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').values
    
    results = []
    
    print(f"Running backtest over {len(prices) - 100} hours...")
    
    for i in range(100, len(prices) - 1):
        # We strictly use data up to index 'i' to predict for 'i+1'
        # No future data leakage
        train_data = prices[:i+1]
        
        try:
            lower, upper = predict_next_hour(train_data)
            actual = prices[i+1]
            ts = timestamps[i+1]
            
            res = {
                "timestamp": ts,
                "lower": lower,
                "upper": upper,
                "actual": float(actual)
            }
            results.append(res)
            
        except Exception as e:
            print(f"Error at index {i}: {e}")
            continue
            
    # Save to jsonl
    print("Saving results to backtest_results.jsonl...")
    with open("backtest_results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    coverage, avg_width, winkler = calculate_metrics(results)
    
    print("-" * 30)
    print("Backtest Complete.")
    print(f"Total Predictions: {len(results)}")
    print(f"Coverage: {coverage:.4f}")
    print(f"Average Width: {avg_width:.4f}")
    print(f"Winkler Score: {winkler:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    run_backtest()
