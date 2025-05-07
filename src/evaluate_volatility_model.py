# /Users/omarabul-hassan/Desktop/projects/trading/src/evaluate_volatility_model.py
"""
Script to evaluate the performance of the GARCH volatility forecasting model
using a rolling window approach.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm

from data_processing import get_processed_data, DEBUG as DATA_DEBUG
from volatility_modeling import fit_garch_model, forecast_volatility, DEBUG as VOL_DEBUG

# Set DEBUG for this script
SCRIPT_DEBUG = False

def dprint(*args, **kwargs):
    """Debug print function for this script."""
    if SCRIPT_DEBUG:
        print("DEBUG evaluate_vol:", *args, **kwargs)

def perform_rolling_volatility_forecast(
    all_data, train_window_size, test_start_date_str, test_end_date_str,
    garch_p=1, garch_o=1, garch_q=1, garch_vol='GARCH', garch_dist='t', use_exog=True):
    """Performs rolling window GARCH fitting and 1-step ahead volatility forecasting."""
    print(f"\n===== Starting Rolling Volatility Forecast Evaluation =====")
    print(f"Train window: {train_window_size} days. Test period: {test_start_date_str} to {test_end_date_str}. Use Exog: {use_exog}")

    test_start_date = pd.to_datetime(test_start_date_str)
    test_end_date = pd.to_datetime(test_end_date_str)

    first_forecast_idx_loc = all_data.index.get_loc(test_start_date)
    if first_forecast_idx_loc < train_window_size:
        print(f"CRITICAL ERROR: test_start_date {test_start_date_str} is too early.")
        return None
    
    relevant_data_start_idx = first_forecast_idx_loc - train_window_size
    relevant_data = all_data.iloc[relevant_data_start_idx : all_data.index.get_loc(test_end_date) + 1].copy()
    dprint(f"Relevant data for rolling forecast shape: {relevant_data.shape}")

    forecast_dates = relevant_data.index[relevant_data.index >= test_start_date]
    if forecast_dates.empty:
        print("CRITICAL ERROR: No forecast dates found in the specified test period.")
        return None

    forecasted_volatilities = []
    actual_log_returns_test = []
    
    print(f"Iterating through {len(forecast_dates)} forecast dates...")

    for i, current_forecast_date in enumerate(forecast_dates):
        current_train_end_idx_loc = relevant_data.index.get_loc(current_forecast_date) - 1
        if current_train_end_idx_loc < 0:
            dprint(f"Skipping forecast for {current_forecast_date} insufficient prior data index.")
            actual_log_returns_test.append(np.nan)
            forecasted_volatilities.append(np.nan)
            continue

        current_train_start_idx_loc = max(0, current_train_end_idx_loc - train_window_size + 1)
        training_window_data = relevant_data.iloc[current_train_start_idx_loc : current_train_end_idx_loc + 1]
        
        if len(training_window_data) < train_window_size * 0.8:
            dprint(f"Skipping forecast for {current_forecast_date}, training window too small: {len(training_window_data)}")
            actual_log_returns_test.append(relevant_data.loc[current_forecast_date, 'log_return'])
            forecasted_volatilities.append(np.nan)
            continue

        train_returns = training_window_data['log_return']
        train_exog_vix = None
        if use_exog:
            train_exog_vix = training_window_data['vix_close'] / 100.0
        
        if SCRIPT_DEBUG and (i % 50 == 0 or i == len(forecast_dates) -1) : # Print progress less frequently
            print(f"Progress: {i+1}/{len(forecast_dates)} | Forecasting for: {current_forecast_date.date()} | Training on: {training_window_data.index.min().date()} to {training_window_data.index.max().date()} ({len(training_window_data)} days)")

        # Temporarily set module-level DEBUG for fitting if SCRIPT_DEBUG is on
        # original_vol_debug = VOL_DEBUG
        # if SCRIPT_DEBUG and (i % 200 == 0 or i == len(forecast_dates) -1): # Print summary less often
        #      volatility_modeling.DEBUG = True
        # else:
        #      volatility_modeling.DEBUG = False # Suppress GARCH summary for most iterations

        fitted_model = fit_garch_model(train_returns, 
                                       exog_series=train_exog_vix if use_exog else None,
                                       p=garch_p, o=garch_o, q=garch_q, 
                                       vol=garch_vol, dist=garch_dist)
        
        # volatility_modeling.DEBUG = original_vol_debug # Reset

        forecast_val = np.nan
        if fitted_model:
            vix_for_forecast_input = None
            if use_exog:
                vix_val_on_fc_date_raw = relevant_data.loc[current_forecast_date, 'vix_close']
                if pd.notna(vix_val_on_fc_date_raw):
                    vix_for_forecast_input = vix_val_on_fc_date_raw / 100.0
                else:
                    dprint(f"Warning: VIX is NaN for forecast date {current_forecast_date}. Forecast may be NaN.")
            
            stdev_series = forecast_volatility(fitted_model, 
                                               last_obs_exog_for_forecast=vix_for_forecast_input, 
                                               horizon=1)
            if stdev_series is not None and not stdev_series.empty and pd.notna(stdev_series.iloc[0]):
                forecast_val = stdev_series.iloc[0]
            else:
                dprint(f"Warning: Volatility forecast for {current_forecast_date.date()} is NaN or failed.")
        else:
            dprint(f"Warning: GARCH model fitting failed for window ending {training_window_data.index.max().date()}.")

        forecasted_volatilities.append(forecast_val)
        actual_log_returns_test.append(relevant_data.loc[current_forecast_date, 'log_return'])

    results_df = pd.DataFrame({
        'log_return_actual': actual_log_returns_test,
        'volatility_forecasted': forecasted_volatilities
    }, index=forecast_dates)
    
    results_df['realized_vol_abs_return'] = np.abs(results_df['log_return_actual'])
    results_df['realized_var_sq_return'] = results_df['log_return_actual']**2
    results_df['volatility_forecasted_var'] = results_df['volatility_forecasted']**2
    
    original_len = len(results_df)
    results_df.dropna(inplace=True)
    if SCRIPT_DEBUG and len(results_df) < original_len:
        print(f"INFO: Dropped {original_len - len(results_df)} rows from results due to NaNs in forecasts.")


    print("===== Rolling Forecast Evaluation Complete =====")
    return results_df

def evaluate_forecasts(results_df):
    """Calculates and prints evaluation metrics for volatility forecasts."""
    if results_df.empty or 'volatility_forecasted' not in results_df or 'realized_vol_abs_return' not in results_df:
        print("Error: Results DataFrame is empty or missing required columns for evaluation.")
        return

    print("\n--- Volatility Forecast Evaluation Metrics ---")
    
    forecasted_vol = results_df['volatility_forecasted']
    realized_vol_abs = results_df['realized_vol_abs_return']
    forecasted_var = results_df['volatility_forecasted_var']
    realized_var_sq = results_df['realized_var_sq_return']

    mae_vol = mean_absolute_error(realized_vol_abs, forecasted_vol)
    print(f"Mean Absolute Error (MAE) for Volatility: {mae_vol:.6f}")
    mse_var = mean_squared_error(realized_var_sq, forecasted_var)
    print(f"Mean Squared Error (MSE) for Variance: {mse_var:.8f}")
    rmse_var = np.sqrt(mse_var)
    print(f"Root Mean Squared Error (RMSE) for Variance: {rmse_var:.6f}")

    print("\nMincer-Zarnowitz Regression (Realized Var ~ Forecasted Var):")
    X_mz = sm.add_constant(forecasted_var)
    y_mz = realized_var_sq
    mz_model = sm.OLS(y_mz, X_mz).fit()
    print(mz_model.summary())
    
    plt.figure(figsize=(14, 7))
    plt.plot(results_df.index, realized_vol_abs, label='Realized Volatility (|Log Return|)', alpha=0.7, color='blue', linewidth=0.8)
    plt.plot(results_df.index, forecasted_vol, label='GARCH Forecasted Volatility', alpha=0.9, color='red', linewidth=1.2)
    plt.title('Realized vs. Forecasted Volatility (Out-of-Sample)')
    plt.xlabel('Date')
    plt.ylabel('Volatility (Daily Std Dev)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # global DEBUG_DATA, DEBUG_VOL, SCRIPT_DEBUG # Make script-level DEBUG flags accessible
    # To control module verbosity from here, you would typically pass the flag
    # or use environment variables, as direct modification after import is tricky.
    # For simplicity, edit the DEBUG flag at the top of each module for now.
    
    # SCRIPT_DEBUG can be set to False here to make this script less verbose on its own prints
    # SCRIPT_DEBUG = False 

    print("########## Starting Volatility Model Evaluation Script ##########")
    # Set module-level DEBUG flags if needed for this specific run
    # This is a bit hacky. A better way is a global config or passing debug flags.
    import data_processing
    import volatility_modeling
    data_processing.DEBUG = SCRIPT_DEBUG # Control imported module's debug state
    volatility_modeling.DEBUG = SCRIPT_DEBUG
    
    master_data = get_processed_data()

    if master_data is not None and not master_data.empty:
        TRAIN_WINDOW = 1000 
        data_start_date = master_data.index.min()
        potential_test_start = data_start_date + pd.Timedelta(days=TRAIN_WINDOW + 50)
        TEST_START_STR = "2007-01-03" 
        if pd.to_datetime(TEST_START_STR) < potential_test_start or pd.to_datetime(TEST_START_STR) not in master_data.index:
            dprint(f"Warning: Proposed TEST_START_STR {TEST_START_STR} too early/not in data. Adjusting...")
            try:
                valid_test_start_idx = master_data.index.get_loc(potential_test_start, method='bfill')
                TEST_START_STR = master_data.index[valid_test_start_idx].strftime('%Y-%m-%d')
            except KeyError:
                print("CRITICAL ERROR: Could not determine a valid test start date. Exiting.")
                exit()
        
        TEST_END_STR = master_data.index.max().strftime('%Y-%m-%d') # Use all available data
        print(f"Final Test Period: {TEST_START_STR} to {TEST_END_STR}")

        results = perform_rolling_volatility_forecast(
            master_data, TRAIN_WINDOW, TEST_START_STR, TEST_END_STR,
            garch_p=1, garch_o=1, garch_q=1, garch_vol='GARCH', garch_dist='t', use_exog=True)

        if results is not None and not results.empty:
            if SCRIPT_DEBUG:
                print("\nFirst 5 rows of forecast results:")
                print(results.head())
            evaluate_forecasts(results)
        else:
            print("Rolling forecast did not produce results or results were empty.")
    else:
        print("Failed to load or process data for evaluation. Exiting.")
    print("########## Volatility Model Evaluation Script Finished ##########")