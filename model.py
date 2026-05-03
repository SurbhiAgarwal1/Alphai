import numpy as np
from scipy.stats import t

def predict_next_hour(data_close, short_window=10, long_window=50, num_simulations=10000, df_t=4):
    """
    Predicts the next hour's 95% confidence interval using GBM + Student-t distribution
    and Dynamic Volatility Regime Adjustment.
    """
    prices = np.array(data_close)
    if len(prices) < 2:
        return prices[-1], prices[-1] # Fallback if entirely insufficient data
        
    # Ensure we don't try to calculate moving windows larger than data length
    actual_long_window = min(long_window, len(prices) - 1)
    actual_short_window = min(short_window, actual_long_window)
        
    log_returns = np.log(prices[1:] / prices[:-1])
    
    drift = np.mean(log_returns)
    volatility = np.std(log_returns, ddof=1) if len(log_returns) > 1 else 0.0
    
    if volatility > 0 and actual_long_window > 1 and actual_short_window > 1:
        # Dynamic Volatility Regime Adjustment
        short_vol = np.std(log_returns[-actual_short_window:], ddof=1)
        long_vol = np.std(log_returns[-actual_long_window:], ddof=1)
        
        if long_vol > 0:
            vol_ratio = short_vol / long_vol
            # Limit the multiplier to prevent extreme ranges
            vol_ratio = np.clip(vol_ratio, 0.5, 2.0)
            adj_vol = volatility * vol_ratio
        else:
            adj_vol = volatility
    else:
        adj_vol = volatility
        
    # Simulate
    # GBM: S_{t+1} = S_t * exp((mu - 0.5 * sigma^2) + sigma * Z)
    # Z ~ Student-t
    
    # Scale Student-t random variables so their standard deviation is 1
    std_t = np.sqrt(df_t / (df_t - 2)) if df_t > 2 else 1.0
    random_shocks = t.rvs(df=df_t, size=num_simulations) / std_t
    
    simulated_returns = drift - 0.5 * (adj_vol ** 2) + adj_vol * random_shocks
    
    last_price = prices[-1]
    simulated_prices = last_price * np.exp(simulated_returns)
    
    lower_bound = np.percentile(simulated_prices, 2.5)
    upper_bound = np.percentile(simulated_prices, 97.5)
    
    return float(lower_bound), float(upper_bound)
