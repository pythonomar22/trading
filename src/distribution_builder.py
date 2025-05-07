# /Users/omarabul-hassan/Desktop/projects/trading/src/distribution_builder.py
"""
Module to build probability distributions for S&P 500 EOD prices.
"""
import numpy as np
from scipy.stats import norm, t as student_t

def build_log_return_distribution(forecasted_log_mean, forecasted_log_volatility, dist_type='norm', df_t=None):
    """Builds a scipy.stats distribution object for forecasted log returns."""
    print(f"DEBUG distribution_builder: Building distribution. Mean={forecasted_log_mean:.6f}, Vol={forecasted_log_volatility:.6f}, Type={dist_type}, df_t={df_t}")
    if forecasted_log_volatility is None or np.isnan(forecasted_log_volatility) or forecasted_log_volatility <= 1e-8: # Check for NaN or too small vol
        print(f"Warning distribution_builder: Invalid forecasted_log_volatility ({forecasted_log_volatility}). Cannot build distribution.")
        return None # Cannot build distribution with invalid volatility

    if dist_type == 'norm':
        return norm(loc=forecasted_log_mean, scale=forecasted_log_volatility)
    elif dist_type == 't':
        if df_t is None or np.isnan(df_t) or df_t <= 2: # df > 2 for variance to be defined
            print(f"Warning distribution_builder: Invalid df_t ({df_t}) for Student's t. Using default Normal instead or returning None.")
            # Fallback or error: For now, let's be strict.
            return None # Or fallback to norm: return norm(loc=forecasted_log_mean, scale=forecasted_log_volatility)
        return student_t(df=df_t, loc=forecasted_log_mean, scale=forecasted_log_volatility)
    else:
        print(f"Warning distribution_builder: Unsupported distribution type: {dist_type}. Defaulting to Normal.")
        return norm(loc=forecasted_log_mean, scale=forecasted_log_volatility)


def get_probability_for_price_range(log_return_dist_object, current_price, price_low_target, price_high_target):
    """Calculates probability for S&P 500 closing within a price range."""
    if log_return_dist_object is None:
        print("DEBUG distribution_builder: log_return_dist_object is None. Cannot calculate probability.")
        return np.nan # Return NaN if distribution couldn't be built

    print(f"DEBUG distribution_builder: Calculating prob. CurrentPrice={current_price:.2f}, Range=[{price_low_target:.2f}, {price_high_target:.2f}]")
    if current_price <= 0:
        print("Error distribution_builder: Current price must be positive.")
        return np.nan
    if price_low_target >= price_high_target:
        print("Error distribution_builder: Price low target must be less than price high target.")
        return np.nan

    log_return_low = np.log(price_low_target / current_price)
    log_return_high = np.log(price_high_target / current_price)
    print(f"DEBUG distribution_builder: Target log_return range: [{log_return_low:.6f}, {log_return_high:.6f}]")
    
    probability = log_return_dist_object.cdf(log_return_high) - log_return_dist_object.cdf(log_return_low)
    print(f"DEBUG distribution_builder: Calculated probability: {probability:.6f}")
    
    if np.isnan(probability):
        print("WARNING distribution_builder: Calculated probability is NaN.")
        print(f"DEBUG distribution_builder: CDF at high: {log_return_dist_object.cdf(log_return_high)}, CDF at low: {log_return_dist_object.cdf(log_return_low)}")
    return probability

# --- main example from before, slightly adjusted if needed ---
if __name__ == '__main__':
    example_forecasted_mean_log_return = 0.0001
    example_forecasted_daily_vol = 0.012     
    example_current_sp_price = 5000.0
    
    print("--- Distribution Builder Example ---")
    norm_dist = build_log_return_distribution(
        example_forecasted_mean_log_return, example_forecasted_daily_vol, 'norm')
    
    if norm_dist:
        target_low, target_high = 4950.0, 5050.0
        prob_norm = get_probability_for_price_range(norm_dist, example_current_sp_price, target_low, target_high)
        print(f"Normal Dist Prob S&P in [{target_low}, {target_high}]: {prob_norm:.4f}")

    example_df_t = 5 
    t_dist = build_log_return_distribution(
        example_forecasted_mean_log_return, example_forecasted_daily_vol, 't', example_df_t)
    
    if t_dist:
        prob_t = get_probability_for_price_range(t_dist, example_current_sp_price, target_low, target_high)
        print(f"Student's t Dist (df={example_df_t}) Prob S&P in [{target_low}, {target_high}]: {prob_t:.4f}")

    # Test NaN volatility
    print("\nTesting with NaN volatility:")
    nan_vol_dist = build_log_return_distribution(0.0001, np.nan, 'norm')
    if nan_vol_dist is None:
        print("Correctly got None for distribution with NaN volatility.")
        prob_nan = get_probability_for_price_range(nan_vol_dist, 5000, 4900, 5100)
        print(f"Probability with None distribution: {prob_nan}")