# /Users/omarabul-hassan/Desktop/projects/trading/src/distribution_builder.py
"""
Module to build probability distributions for S&P 500 EOD prices.
"""
import numpy as np
from scipy.stats import norm, t as student_t

DEBUG = True # SET TO False to reduce print output

def dprint(*args, **kwargs):
    """Debug print function that only prints if DEBUG is True."""
    if DEBUG:
        print("DEBUG distribution_builder:", *args, **kwargs)

def build_log_return_distribution(forecasted_log_mean, forecasted_log_volatility, dist_type='norm', df_t=None):
    """Builds a scipy.stats distribution object for forecasted log returns."""
    dprint(f"Building distribution. Mean={forecasted_log_mean:.6f}, Vol={forecasted_log_volatility:.6f}, Type={dist_type}, df_t={df_t}")
    if forecasted_log_volatility is None or np.isnan(forecasted_log_volatility) or forecasted_log_volatility <= 1e-8:
        print(f"Warning: Invalid forecasted_log_volatility ({forecasted_log_volatility}). Cannot build distribution.")
        return None

    if dist_type == 'norm':
        return norm(loc=forecasted_log_mean, scale=forecasted_log_volatility)
    elif dist_type == 't':
        if df_t is None or np.isnan(df_t) or df_t <= 2:
            print(f"Warning: Invalid df_t ({df_t}) for Student's t. Returning None.")
            return None
        return student_t(df=df_t, loc=forecasted_log_mean, scale=forecasted_log_volatility)
    else:
        print(f"Warning: Unsupported distribution type: {dist_type}. Defaulting to Normal.")
        return norm(loc=forecasted_log_mean, scale=forecasted_log_volatility)

def get_probability_for_price_range(log_return_dist_object, current_price, price_low_target, price_high_target):
    """Calculates probability for S&P 500 closing within a price range."""
    if log_return_dist_object is None:
        dprint("log_return_dist_object is None. Cannot calculate probability.")
        return np.nan

    dprint(f"Calculating prob. CurrentPrice={current_price:.2f}, Range=[{price_low_target:.2f}, {price_high_target:.2f}]")
    if current_price <= 0:
        print("Error: Current price must be positive.")
        return np.nan
    if price_low_target >= price_high_target:
        print("Error: Price low target must be less than price high target.")
        return np.nan

    log_return_low = np.log(price_low_target / current_price)
    log_return_high = np.log(price_high_target / current_price)
    dprint(f"Target log_return range: [{log_return_low:.6f}, {log_return_high:.6f}]")
    
    probability = log_return_dist_object.cdf(log_return_high) - log_return_dist_object.cdf(log_return_low)
    dprint(f"Calculated probability: {probability:.6f}")
    
    if np.isnan(probability):
        print("WARNING: Calculated probability is NaN.")
        dprint(f"CDF at high: {log_return_dist_object.cdf(log_return_high)}, CDF at low: {log_return_dist_object.cdf(log_return_low)}")
    return probability

if __name__ == '__main__':
    # current_debug_state = DEBUG
    # DEBUG = True
    print("--- Distribution Builder Example ---")
    example_forecasted_mean_log_return = 0.0001
    example_forecasted_daily_vol = 0.012     
    example_current_sp_price = 5000.0
    
    norm_dist = build_log_return_distribution(
        example_forecasted_mean_log_return, example_forecasted_daily_vol, 'norm')
    if norm_dist:
        prob_norm = get_probability_for_price_range(norm_dist, example_current_sp_price, 4950.0, 5050.0)
        print(f"Normal Dist Prob S&P in [4950, 5050]: {prob_norm:.4f}")

    t_dist = build_log_return_distribution(
        example_forecasted_mean_log_return, example_forecasted_daily_vol, 't', 5)
    if t_dist:
        prob_t = get_probability_for_price_range(t_dist, example_current_sp_price, 4950.0, 5050.0)
        print(f"Student's t Dist (df=5) Prob S&P in [4950, 5050]: {prob_t:.4f}")
    # DEBUG = current_debug_state