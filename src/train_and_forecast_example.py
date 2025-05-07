# /Users/omarabul-hassan/Desktop/projects/trading/src/train_and_forecast_example.py
"""
Example script to demonstrate training a GARCH model, forecasting,
and building a price distribution for one period.
"""
import pandas as pd
import numpy as np
from data_processing import get_processed_data, DEBUG as DATA_DEBUG # Import DEBUG from module
from volatility_modeling import fit_garch_model, forecast_volatility, save_model, DEBUG as VOL_DEBUG
from distribution_builder import build_log_return_distribution, get_probability_for_price_range, DEBUG as DIST_DEBUG

# Set DEBUG for this script, potentially overriding module-level DEBUG for this run
SCRIPT_DEBUG = True 

def dprint(*args, **kwargs):
    """Debug print function for this script."""
    if SCRIPT_DEBUG:
        print("DEBUG train_example:", *args, **kwargs)

def run_single_train_forecast_step(all_data, train_end_date_str, forecast_date_str,
                                   garch_p=1, garch_o=1, garch_q=1, garch_vol='GARCH', garch_dist='t',
                                   kalshi_target_low=None, kalshi_target_high=None, use_exog=True):
    """Performs a single step of training a GARCH model and forecasting."""
    print(f"\n===== Running Single Train/Forecast Step =====")
    print(f"Train end date: {train_end_date_str}, Forecast date: {forecast_date_str}")

    train_end_date = pd.to_datetime(train_end_date_str)
    forecast_date = pd.to_datetime(forecast_date_str)

    if forecast_date <= train_end_date:
        print("CRITICAL ERROR: Forecast date must be after train end date.")
        return False

    train_data = all_data[all_data.index <= train_end_date].copy()
    dprint(f"Training data shape: {train_data.shape} (from {train_data.index.min()} to {train_data.index.max()})")
    
    if train_data.empty or len(train_data) < 100:
        print(f"Error: Not enough training data ({len(train_data)} rows) up to {train_end_date_str}.")
        return False

    train_returns = train_data['log_return']
    train_exog_vix = None
    if use_exog:
        train_exog_vix = train_data['vix_close'] / 100.0
        if train_returns.isna().any() or train_exog_vix.isna().any():
            dprint(f"NaNs in train_returns ({train_returns.isna().sum()}) or train_exog_vix ({train_exog_vix.isna().sum()}) before fit.")
            return False

    print(f"--- Training GARCH model using data up to {train_end_date_str} ---")
    fitted_model = fit_garch_model(train_returns, 
                                   exog_series=train_exog_vix if use_exog else None, 
                                   p=garch_p, o=garch_o, q=garch_q, 
                                   vol=garch_vol, dist=garch_dist)

    if not fitted_model:
        print("Model fitting failed. Aborting step.")
        return False

    model_filename = f"garch_{garch_vol}_{garch_p}{garch_o}{garch_q}_{garch_dist}_trained_on_{train_end_date.strftime('%Y%m%d')}.joblib"
    save_model(fitted_model, model_filename)

    vix_for_forecast_input = None
    if use_exog:
        if forecast_date not in all_data.index:
            print(f"CRITICAL ERROR: Data for forecast_date {forecast_date_str} (VIX) not found.")
            return False
        vix_val_on_fc_date_raw = all_data.loc[forecast_date, 'vix_close']
        dprint(f"Raw VIX for forecast date {forecast_date_str}: {vix_val_on_fc_date_raw}")
        if pd.isna(vix_val_on_fc_date_raw):
            print(f"CRITICAL ERROR: VIX value for forecast date {forecast_date_str} is NaN.")
            return False
        vix_for_forecast_input = vix_val_on_fc_date_raw / 100.0
        dprint(f"Scaled VIX for forecast input: {vix_for_forecast_input}")
    
    print(f"--- Forecasting S&P 500 volatility for {forecast_date_str} ---")
    forecasted_stdev_series = forecast_volatility(fitted_model, 
                                                  last_obs_exog_for_forecast=vix_for_forecast_input, 
                                                  horizon=1)
    if forecasted_stdev_series is None or forecasted_stdev_series.empty or forecasted_stdev_series.isna().any():
        print("Volatility forecasting failed or produced NaN. Aborting step.")
        if forecasted_stdev_series is not None: dprint(f"Forecasted stdev series: {forecasted_stdev_series}")
        return False
    
    forecasted_daily_vol = forecasted_stdev_series.iloc[0]
    print(f"Forecasted daily log return volatility for {forecast_date_str}: {forecasted_daily_vol:.6f}")

    current_sp_price = train_data['close'].iloc[-1]
    forecasted_mean_log_return = 0.0
    df_t_param_from_model = None
    if garch_dist == 't':
        try:
            df_t_param_from_model = fitted_model.params['nu']
            dprint(f"Using estimated df for t-distribution: {df_t_param_from_model:.2f}")
        except (KeyError, AttributeError):
            dprint("Warning: Could not retrieve 'nu' (df). Using default df=5.")
            df_t_param_from_model = 5 

    print(f"--- Building probability distribution for S&P 500 EOD price on {forecast_date_str} ---")
    print(f"Current S&P 500 price (close of {train_end_date_str}): {current_sp_price:.2f}")
    
    log_return_distribution = build_log_return_distribution(
        forecasted_log_mean, forecasted_daily_vol,
        garch_dist if garch_dist in ['norm', 't'] else 'norm',
        df_t_param_from_model if garch_dist == 't' else None)
    
    if log_return_distribution is None:
        print("Failed to build log return distribution. Aborting step.")
        return False

    if kalshi_target_low is not None and kalshi_target_high is not None:
        prob = get_probability_for_price_range(
            log_return_distribution, current_sp_price, kalshi_target_low, kalshi_target_high)
        print(f"Model-implied probability for S&P 500 EOD between {kalshi_target_low:.2f} and {kalshi_target_high:.2f} on {forecast_date_str}: {prob:.4f}")
    else:
        dprint("No Kalshi target range provided for probability calculation in this step.")
    
    print(f"===== Step for {forecast_date_str} Completed Successfully =====")
    return True

if __name__ == '__main__':
    # Control verbosity from imported modules for this script run
    # data_processing.DEBUG = SCRIPT_DEBUG # This doesn't work as DEBUG is set at module load time
    # volatility_modeling.DEBUG = SCRIPT_DEBUG 
    # distribution_builder.DEBUG = SCRIPT_DEBUG
    # Instead, you'd have to modify the DEBUG flag in each imported module directly if you want
    # this script's SCRIPT_DEBUG to control their prints when this script is run.
    # For now, their DEBUG flags are independent.
    
    print("########## Starting Full S&P 500 EOD Distribution Modeling Example ##########")
    master_data = get_processed_data()

    if master_data is not None and not master_data.empty:
        if len(master_data) >= 102:
            available_dates = master_data.index.sort_values()
            example_train_end_str = '2005-12-30'
            example_forecast_for_str = '2006-01-03'
            
            if pd.to_datetime(example_train_end_str) not in available_dates or \
               pd.to_datetime(example_forecast_for_str) not in available_dates or \
               available_dates.get_loc(pd.to_datetime(example_forecast_for_str)) <= available_dates.get_loc(pd.to_datetime(example_train_end_str)):
                dprint(f"Warning: Example dates {example_train_end_str}/{example_forecast_for_str} not suitable. Using relative dates.")
                example_train_end_str = available_dates[-2].strftime('%Y-%m-%d')
                example_forecast_for_str = available_dates[-1].strftime('%Y-%m-%d')

            print(f"Using Train End: {example_train_end_str}, Forecast For: {example_forecast_for_str}")

            if pd.to_datetime(example_train_end_str) in master_data.index:
                 example_sp_close_on_train_end = master_data.loc[example_train_end_str, 'close']
                 hypothetical_kalshi_low = example_sp_close_on_train_end * 0.985
                 hypothetical_kalshi_high = example_sp_close_on_train_end * 1.015

                 success = run_single_train_forecast_step(
                    master_data, example_train_end_str, example_forecast_for_str,
                    garch_p=1, garch_o=1, garch_q=1, garch_vol='GARCH', garch_dist='t',
                    kalshi_target_low=hypothetical_kalshi_low, kalshi_target_high=hypothetical_kalshi_high,
                    use_exog=True) # Test with exog
                 if success: print("\nExample run (with exog) completed successfully.")
                 else: print("\nExample run (with exog) encountered errors.")
                 
                 # Example without exog
                 # success_no_exog = run_single_train_forecast_step(
                 #    master_data, example_train_end_str, example_forecast_for_str,
                 #    garch_p=1, garch_o=1, garch_q=1, garch_vol='GARCH', garch_dist='t',
                 #    kalshi_target_low=hypothetical_kalshi_low, kalshi_target_high=hypothetical_kalshi_high,
                 #    use_exog=False)
                 # if success_no_exog: print("\nExample run (NO exog) completed successfully.")
                 # else: print("\nExample run (NO exog) encountered errors.")

            else:
                print(f"Error: Chosen train_end_date {example_train_end_str} not found in processed data index.")
        else:
            print("Not enough historical data points (<102) in master_data to run the example.")
    else:
        print("Failed to load or process data. Exiting full example.")
    print("########## Example Script Finished ##########")