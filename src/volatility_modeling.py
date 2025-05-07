# /Users/omarabul-hassan/Desktop/projects/trading/src/volatility_modeling.py
"""
Module for fitting GARCH models and forecasting volatility.
"""
import pandas as pd
import numpy as np
from arch import arch_model
import joblib
import os

MODELS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)

def fit_garch_model(returns_series, exog_series=None, p=1, o=0, q=1, vol='GARCH', dist='t'):
    """Fits a GARCH-type model."""
    print(f"DEBUG volatility_modeling: Attempting to fit {vol}({p},{o},{q}) with {dist} dist.")
    print(f"DEBUG volatility_modeling: Returns series length: {len(returns_series)}, NaNs: {returns_series.isna().sum()}")
    if exog_series is not None:
        print(f"DEBUG volatility_modeling: Exog series length: {len(exog_series)}, NaNs: {exog_series.isna().sum()}")
        if len(returns_series) != len(exog_series):
            print("CRITICAL ERROR: Length mismatch between returns and exog series in fit_garch_model.")
            return None

    if returns_series.var() < 1e-12 or returns_series.isna().all():
        print("Warning: Returns series has near-zero variance or is all NaNs. GARCH model cannot be fit.")
        return None
    
    # Scale returns by 100 (common practice for GARCH stability)
    scaled_returns = returns_series.dropna() * 100 # Drop NaNs again just in case
    if scaled_returns.empty:
        print("Warning: Returns series is empty after NaN drop. Cannot fit GARCH.")
        return None

    exog_data_for_fit = None
    if exog_series is not None:
        # Align exog_series with the (potentially NaN-dropped) scaled_returns index
        exog_data_for_fit = exog_series.reindex(scaled_returns.index)
        if exog_data_for_fit.isna().any():
            print(f"Warning: NaNs found in exog_series after reindexing to returns index. Count: {exog_data_for_fit.isna().sum()}. Attempting to ffill/bfill.")
            exog_data_for_fit = exog_data_for_fit.ffill().bfill() # Fill NaNs if any due to reindexing minor mismatches
            if exog_data_for_fit.isna().any():
                print("CRITICAL ERROR: Exog series still contains NaNs after ffill/bfill. Cannot fit GARCH-X.")
                return None
        exog_data_for_fit = exog_data_for_fit.values
        print(f"DEBUG volatility_modeling: Shape of exog_data_for_fit for arch_model: {exog_data_for_fit.shape if exog_data_for_fit is not None else 'None'}")


    print(f"DEBUG volatility_modeling: Final scaled returns length for model: {len(scaled_returns)}")
    
    try:
        am = arch_model(scaled_returns, x=exog_data_for_fit, p=p, o=o, q=q, vol=vol, dist=dist, rescale=False)
        model_fit = am.fit(disp='off', show_warning=True) # Show warnings from arch
        print("--- GARCH Model Fit Summary ---")
        print(model_fit.summary())
        print("--- Fitted Parameters ---")
        print(model_fit.params)
        print("-----------------------------")
        if np.isnan(model_fit.params).any() or np.isinf(model_fit.params).any():
            print("WARNING: NaN or Inf found in fitted GARCH parameters! Model may be unstable.")
        return model_fit
    except Exception as e:
        print(f"CRITICAL ERROR fitting GARCH model: {e}")
        import traceback
        traceback.print_exc()
        return None

def forecast_volatility(fitted_model_result, last_obs_exog_for_forecast=None, horizon=1):
    """Forecasts conditional volatility."""
    print(f"DEBUG volatility_modeling: forecast_volatility called. Horizon: {horizon}")
    print(f"DEBUG volatility_modeling: last_obs_exog_for_forecast received: {last_obs_exog_for_forecast}")

    if fitted_model_result is None:
        print("DEBUG volatility_modeling: No fitted model provided to forecast_volatility.")
        return None
        
    exog_for_arch_forecast = None
    model_uses_exog = fitted_model_result.model.x is not None
    print(f"DEBUG volatility_modeling: Model was fit with exogenous variables: {model_uses_exog}")

    if model_uses_exog:
        if last_obs_exog_for_forecast is None:
            print("CRITICAL ERROR: Model is GARCH-X but no exogenous variable provided for forecast period.")
            return None
        
        exog_for_arch_forecast = np.asarray(last_obs_exog_for_forecast)
        print(f"DEBUG volatility_modeling: exog_for_arch_forecast (raw asarray): {exog_for_arch_forecast}, shape: {exog_for_arch_forecast.shape}")

        if exog_for_arch_forecast.ndim == 0: # Single scalar value
             exog_for_arch_forecast = exog_for_arch_forecast.reshape(1, 1)
        elif exog_for_arch_forecast.ndim == 1:
            # Ensure it's (horizon, n_exog_vars). For horizon=1, it's (1, n_exog_vars)
            exog_for_arch_forecast = exog_for_arch_forecast.reshape(horizon, -1) 
        
        print(f"DEBUG volatility_modeling: exog_for_arch_forecast (reshaped for arch): {exog_for_arch_forecast}, shape: {exog_for_arch_forecast.shape}")
        
        num_exog_model = fitted_model_result.model.x.shape[1] if fitted_model_result.model.x.ndim >1 else 1
        if exog_for_arch_forecast.shape[1] != num_exog_model :
             print(f"CRITICAL ERROR: Mismatch in number of exog vars for forecast ({exog_for_arch_forecast.shape[1]}) vs model ({num_exog_model}).")
             return None
        if exog_for_arch_forecast.shape[0] != horizon:
             print(f"CRITICAL ERROR: Exog forecast data rows ({exog_for_arch_forecast.shape[0]}) must match horizon ({horizon}).")
             return None
        if np.isnan(exog_for_arch_forecast).any():
            print("CRITICAL ERROR: NaN found in exogenous variable provided for forecast period.")
            return None
            
    try:
        print(f"DEBUG volatility_modeling: Calling arch.forecast with exog: {exog_for_arch_forecast}")
        forecasts_obj = fitted_model_result.forecast(horizon=horizon, x=exog_for_arch_forecast, reindex=False)
        print(f"DEBUG volatility_modeling: Raw ARCH forecast object: {forecasts_obj}")
        print(f"DEBUG volatility_modeling: ARCH forecast variance: \n{forecasts_obj.variance}")
        
        # Get the variance for the last period of the horizon
        forecasted_variance_scaled = forecasts_obj.variance.iloc[-1].values
        print(f"DEBUG volatility_modeling: Extracted scaled variance for forecast: {forecasted_variance_scaled}")

        if np.isnan(forecasted_variance_scaled).any() or np.isinf(forecasted_variance_scaled).any():
            print("WARNING: NaN or Inf in forecasted variance from ARCH library.")
            # This could happen if parameters were bad or exog caused numerical issue
            return pd.Series([np.nan] * horizon, index=np.arange(1, horizon + 1))

        # Scale back: returns were *100, so variance is (vol*100)^2. StdDev = sqrt(variance)/100
        forecasted_std_dev_unscaled = np.sqrt(forecasted_variance_scaled) / 100.0
        print(f"DEBUG volatility_modeling: Unscaled forecasted std dev: {forecasted_std_dev_unscaled}")
        
        return pd.Series(forecasted_std_dev_unscaled, index=np.arange(1, horizon + 1))
        
    except Exception as e:
        print(f"CRITICAL ERROR during volatility forecasting: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- save_model and load_model remain the same as before ---
def save_model(fitted_model_result, model_name="garch_model.joblib"):
    if fitted_model_result is None:
        print("No model to save.")
        return
    file_path = os.path.join(MODELS_PATH, model_name)
    try:
        joblib.dump(fitted_model_result, file_path)
        print(f"Model saved to {file_path}")
    except Exception as e:
        print(f"Error saving model: {e}")

def load_model(model_name="garch_model.joblib"):
    file_path = os.path.join(MODELS_PATH, model_name)
    try:
        model = joblib.load(file_path)
        print(f"Model loaded from {file_path}")
        return model
    except FileNotFoundError:
        print(f"Model file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

if __name__ == '__main__':
    from data_processing import get_processed_data # Assuming data_processing.py is accessible
    
    print("--- Volatility Modeling Example ---")
    data = get_processed_data()
    if data is not None and not data.empty and 'log_return' in data.columns and 'vix_close' in data.columns:
        returns = data['log_return'].dropna()
        exog_vix = (data['vix_close'] / 100.0).reindex(returns.index).ffill().bfill()

        if len(returns) > 50: # Ensure enough data for a meaningful fit
            print("\nFitting GJR-GARCH(1,1,1) example (vol='GARCH', o=1)...")
            # For GJR, use vol='GARCH' and o=1, or directly vol='GJR' and p,q for symmetric parts
            fitted_model = fit_garch_model(returns.iloc[:-1], exog_series=exog_vix.iloc[:-1], 
                                           p=1, o=1, q=1, vol='GARCH', dist='t') 
            
            if fitted_model:
                save_model(fitted_model, "example_gjr_model.joblib")
                loaded_model = load_model("example_gjr_model.joblib")

                if loaded_model:
                    # Prepare exog for forecast: value of VIX on the day we are forecasting FOR
                    vix_for_forecast = exog_vix.iloc[-1:].values # Ensure it's array for forecast function
                    print(f"\nForecasting with VIX value: {vix_for_forecast}")
                    
                    forecasted_stdev = forecast_volatility(loaded_model, 
                                                           last_obs_exog_for_forecast=vix_for_forecast)
                    if forecasted_stdev is not None:
                        print(f"1-step ahead S&P 500 log return volatility forecast: {forecasted_stdev.iloc[0]:.6f}")
                    else:
                        print("Volatility forecasting failed in example.")
            else:
                print("Model fitting failed in example.")
        else:
            print("Not enough data for GARCH example after processing.")
    else:
        print("Could not load data or critical columns missing for volatility modeling example.")