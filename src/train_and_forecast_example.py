# /Users/omarabul-hassan/Desktop/projects/trading/src/train_and_forecast_example.py
"""
Example script to demonstrate training a GARCH model, forecasting,
and building a price distribution for one period.
"""
import pandas as pd
import numpy as np
# Ensure src is in PYTHONPATH or files are in the same directory when running.
# If running from `src` directory, direct imports should work.
from data_processing import get_processed_data
from volatility_modeling import fit_garch_model, forecast_volatility, save_model, load_model
from distribution_builder import build_log_return_distribution, get_probability_for_price_range

def run_single_train_forecast_step(all_data, train_end_date_str, forecast_date_str,
                                   garch_p=1, garch_o=1, garch_q=1, garch_vol='GARCH', garch_dist='t',
                                   kalshi_target_low=None, kalshi_target_high=None):
    """
    Performs a single step of training a GARCH model and forecasting.
    """
    print(f"\n===== Running Single Train/Forecast Step =====")
    print(f"Train end date: {train_end_date_str}, Forecast date: {forecast_date_str}")

    train_end_date = pd.to_datetime(train_end_date_str)
    forecast_date = pd.to_datetime(forecast_date_str)

    if forecast_date <= train_end_date:
        print("CRITICAL ERROR: Forecast date must be after train end date.")
        return False

    # 1. Select training data
    train_data = all_data[all_data.index <= train_end_date].copy()
    print(f"DEBUG train_example: Training data shape: {train_data.shape} (from {train_data.index.min()} to {train_data.index.max()})")
    
    if train_data.empty or len(train_data) < 100: # Increased min data for GARCH
        print(f"Error: Not enough training data ({len(train_data)} rows) up to {train_end_date_str}.")
        return False

    train_returns = train_data['log_return']
    train_exog_vix = None
    model_uses_exog = True # Assuming GARCH-X for now, can be made configurable

    if model_uses_exog:
        train_exog_vix = train_data['vix_close'] / 100.0 # Scale VIX
        if train_returns.isna().any() or train_exog_vix.isna().any():
            print(f"DEBUG train_example: NaNs in train_returns ({train_returns.isna().sum()}) or train_exog_vix ({train_exog_vix.isna().sum()}) before fit.")
            # These should have been handled by data_processing, but double check
            return False

    print(f"--- Training GARCH model using data up to {train_end_date_str} ---")
    fitted_model = fit_garch_model(train_returns, 
                                   exog_series=train_exog_vix if model_uses_exog else None, 
                                   p=garch_p, o=garch_o, q=garch_q, 
                                   vol=garch_vol, dist=garch_dist)

    if not fitted_model:
        print("Model fitting failed. Aborting step.")
        return False

    model_filename = f"garch_{garch_vol}_{garch_p}{garch_o}{garch_q}_{garch_dist}_trained_on_{train_end_date.strftime('%Y%m%d')}.joblib"
    save_model(fitted_model, model_filename)

    # 2. Prepare exog for forecasting volatility for `forecast_date`
    vix_for_forecast_input = None
    if model_uses_exog:
        if forecast_date not in all_data.index:
            print(f"CRITICAL ERROR: Data for forecast_date {forecast_date_str} (needed for VIX) not found in all_data.")
            return False
        
        vix_val_on_fc_date_raw = all_data.loc[forecast_date, 'vix_close']
        print(f"DEBUG train_example: Raw VIX for forecast date {forecast_date_str}: {vix_val_on_fc_date_raw}")
        if pd.isna(vix_val_on_fc_date_raw):
            print(f"CRITICAL ERROR: VIX value for forecast date {forecast_date_str} is NaN in master data.")
            return False
        vix_for_forecast_input = vix_val_on_fc_date_raw / 100.0
        print(f"DEBUG train_example: Scaled VIX for forecast input: {vix_for_forecast_input}")
    
    print(f"--- Forecasting S&P 500 volatility for {forecast_date_str} ---")
    forecasted_stdev_series = forecast_volatility(fitted_model, 
                                                  last_obs_exog_for_forecast=vix_for_forecast_input, 
                                                  horizon=1)
    if forecasted_stdev_series is None or forecasted_stdev_series.empty or forecasted_stdev_series.isna().any():
        print("Volatility forecasting failed or produced NaN. Aborting step.")
        if forecasted_stdev_series is not None: print(f"DEBUG train_example: Forecasted stdev series: {forecasted_stdev_series}")
        return False
    
    forecasted_daily_vol = forecasted_stdev_series.iloc[0]
    print(f"Forecasted daily log return volatility for {forecast_date_str}: {forecasted_daily_vol:.6f}")

    # 3. Build probability distribution
    current_sp_price = train_data['close'].iloc[-1] # S&P close on train_end_date
    forecasted_mean_log_return = 0.0 # Assumption
    
    df_t_param_from_model = None
    if garch_dist == 't':
        try:
            df_t_param_from_model = fitted_model.params['nu']
            print(f"DEBUG train_example: Using estimated df for t-distribution: {df_t_param_from_model:.2f}")
        except (KeyError, AttributeError):
            print("Warning: Could not retrieve 'nu' (df) from fitted GARCH model. Using default df=5 for t-dist if applicable.")
            df_t_param_from_model = 5 

    print(f"--- Building probability distribution for S&P 500 EOD price on {forecast_date_str} ---")
    print(f"Current S&P 500 price (close of {train_end_date_str}): {current_sp_price:.2f}")
    
    log_return_distribution = build_log_return_distribution(
        forecasted_log_mean=forecasted_mean_log_return,
        forecasted_log_volatility=forecasted_daily_vol,
        dist_type=garch_dist if garch_dist in ['norm', 't'] else 'norm',
        df_t=df_t_param_from_model if garch_dist == 't' else None
    )
    
    if log_return_distribution is None:
        print("Failed to build log return distribution. Aborting step.")
        return False

    # 4. Calculate probability for a hypothetical Kalshi range
    if kalshi_target_low is not None and kalshi_target_high is not None:
        prob = get_probability_for_price_range(
            log_return_distribution, current_sp_price, kalshi_target_low, kalshi_target_high)
        print(f"Model-implied probability for S&P 500 EOD between {kalshi_target_low:.2f} and {kalshi_target_high:.2f} on {forecast_date_str}: {prob:.4f}")
    else:
        print("No Kalshi target range provided for probability calculation in this step.")
    
    print(f"===== Step for {forecast_date_str} Completed Successfully =====")
    return True


if __name__ == '__main__':
    print("########## Starting Full S&P 500 EOD Distribution Modeling Example ##########")
    master_data = get_processed_data()

    if master_data is not None and not master_data.empty:
        if len(master_data) >= 102: # Need at least 100 for train, 1 for prior day, 1 for forecast day
            available_dates = master_data.index.sort_values()
            
            # Let's pick a specific period for robust testing, e.g., a few years into the data
            # Ensure data is available for these specific dates
            example_train_end_str = '2005-12-30' # Example, ensure this date and next are in your data
            example_forecast_for_str = '2006-01-03' # Example, ensure this is a trading day after train_end
            
            # Fallback if specific dates aren't available, use relative dates
            if pd.to_datetime(example_train_end_str) not in available_dates or \
               pd.to_datetime(example_forecast_for_str) not in available_dates or \
               available_dates.get_loc(pd.to_datetime(example_forecast_for_str)) <= available_dates.get_loc(pd.to_datetime(example_train_end_str)):
                print(f"Warning: Example dates {example_train_end_str}/{example_forecast_for_str} not suitable. Using relative dates.")
                example_train_end_str = available_dates[-2].strftime('%Y-%m-%d')
                example_forecast_for_str = available_dates[-1].strftime('%Y-%m-%d')

            print(f"Using Train End: {example_train_end_str}, Forecast For: {example_forecast_for_str}")

            if pd.to_datetime(example_train_end_str) in master_data.index:
                 example_sp_close_on_train_end = master_data.loc[example_train_end_str, 'close']
                 hypothetical_kalshi_low = example_sp_close_on_train_end * 0.985 # 1.5% down
                 hypothetical_kalshi_high = example_sp_close_on_train_end * 1.015 # 1.5% up

                 success = run_single_train_forecast_step(
                    master_data,
                    train_end_date_str=example_train_end_str,
                    forecast_date_str=example_forecast_for_str,
                    garch_p=1, garch_o=1, garch_q=1, # GJR-GARCH style
                    garch_vol='GARCH', # 'GARCH' with o=1 is GJR for `arch` library
                    garch_dist='t',
                    kalshi_target_low=hypothetical_kalshi_low,
                    kalshi_target_high=hypothetical_kalshi_high
                 )
                 if success:
                     print("\nExample run completed successfully.")
                 else:
                     print("\nExample run encountered errors.")
            else:
                print(f"Error: Chosen train_end_date {example_train_end_str} not found in processed data index.")
        else:
            print("Not enough historical data points (<102) in master_data to run the example.")
    else:
        print("Failed to load or process data. Exiting full example.")
    print("########## Example Script Finished ##########")