# /Users/omarabul-hassan/Desktop/projects/trading/src/volatility_modeling.py
"""
Module for fitting GARCH models and forecasting volatility.
"""
import pandas as pd
import numpy as np
from arch import arch_model
import joblib
import os

DEBUG = True # SET TO False to reduce print output

MODELS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)

def dprint(*args, **kwargs):
    """Debug print function that only prints if DEBUG is True."""
    if DEBUG:
        print("DEBUG volatility_modeling:", *args, **kwargs)

def fit_garch_model(returns_series, exog_series=None, p=1, o=0, q=1, vol='GARCH', dist='t'):
    """Fits a GARCH-type model."""
    dprint(f"Attempting to fit {vol}({p},{o},{q}) with {dist} dist.")
    dprint(f"Returns series length: {len(returns_series)}, NaNs: {returns_series.isna().sum()}")
    if exog_series is not None:
        dprint(f"Exog series length: {len(exog_series)}, NaNs: {exog_series.isna().sum()}")
        if len(returns_series) != len(exog_series):
            print("CRITICAL ERROR: Length mismatch between returns and exog series in fit_garch_model.")
            return None

    if returns_series.var() < 1e-12 or returns_series.isna().all():
        print("Warning: Returns series has near-zero variance or is all NaNs. GARCH model cannot be fit.")
        return None
    
    scaled_returns = returns_series.dropna() * 100
    if scaled_returns.empty:
        print("Warning: Returns series is empty after NaN drop. Cannot fit GARCH.")
        return None

    exog_data_for_fit = None
    if exog_series is not None:
        exog_data_for_fit = exog_series.reindex(scaled_returns.index)
        if exog_data_for_fit.isna().any():
            dprint(f"NaNs found in exog_series after reindexing. Count: {exog_data_for_fit.isna().sum()}. Attempting ffill/bfill.")
            exog_data_for_fit = exog_data_for_fit.ffill().bfill()
            if exog_data_for_fit.isna().any():
                print("CRITICAL ERROR: Exog series still contains NaNs after ffill/bfill.")
                return None
        exog_data_for_fit = exog_data_for_fit.values
        dprint(f"Shape of exog_data_for_fit for arch_model: {exog_data_for_fit.shape if exog_data_for_fit is not None else 'None'}")

    dprint(f"Final scaled returns length for model: {len(scaled_returns)}")
    
    try:
        am = arch_model(scaled_returns, x=exog_data_for_fit, p=p, o=o, q=q, vol=vol, dist=dist, rescale=False)
        model_fit = am.fit(disp='off', show_warning=DEBUG) # Show arch warnings only if DEBUG is True
        if DEBUG:
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
        if DEBUG:
            import traceback
            traceback.print_exc()
        return None

def forecast_volatility(fitted_model_result, last_obs_exog_for_forecast=None, horizon=1):
    """Forecasts conditional volatility."""
    dprint(f"forecast_volatility called. Horizon: {horizon}")
    dprint(f"last_obs_exog_for_forecast received: {last_obs_exog_for_forecast}")

    if fitted_model_result is None:
        dprint("No fitted model provided to forecast_volatility.")
        return None
        
    exog_for_arch_forecast = None
    model_uses_exog = fitted_model_result.model.x is not None
    dprint(f"Model was fit with exogenous variables: {model_uses_exog}")

    if model_uses_exog:
        if last_obs_exog_for_forecast is None:
            print("CRITICAL ERROR: Model is GARCH-X but no exogenous variable provided for forecast period.")
            return None
        
        exog_for_arch_forecast = np.asarray(last_obs_exog_for_forecast)
        dprint(f"exog_for_arch_forecast (raw asarray): {exog_for_arch_forecast}, shape: {exog_for_arch_forecast.shape}")

        if exog_for_arch_forecast.ndim == 0:
             exog_for_arch_forecast = exog_for_arch_forecast.reshape(1, 1)
        elif exog_for_arch_forecast.ndim == 1:
            exog_for_arch_forecast = exog_for_arch_forecast.reshape(horizon, -1) 
        
        dprint(f"exog_for_arch_forecast (reshaped for arch): {exog_for_arch_forecast}, shape: {exog_for_arch_forecast.shape}")
        
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
        dprint(f"Calling arch.forecast with exog: {exog_for_arch_forecast}")
        forecasts_obj = fitted_model_result.forecast(horizon=horizon, x=exog_for_arch_forecast, reindex=False)
        dprint(f"Raw ARCH forecast object: {forecasts_obj}")
        dprint(f"ARCH forecast variance: \n{forecasts_obj.variance}")
        
        forecasted_variance_scaled = forecasts_obj.variance.iloc[-1].values
        dprint(f"Extracted scaled variance for forecast: {forecasted_variance_scaled}")

        if np.isnan(forecasted_variance_scaled).any() or np.isinf(forecasted_variance_scaled).any():
            print("WARNING: NaN or Inf in forecasted variance from ARCH library.")
            return pd.Series([np.nan] * horizon, index=np.arange(1, horizon + 1))

        forecasted_std_dev_unscaled = np.sqrt(forecasted_variance_scaled) / 100.0
        dprint(f"Unscaled forecasted std dev: {forecasted_std_dev_unscaled}")
        
        return pd.Series(forecasted_std_dev_unscaled, index=np.arange(1, horizon + 1))
        
    except Exception as e:
        print(f"CRITICAL ERROR during volatility forecasting: {e}")
        if DEBUG:
            import traceback
            traceback.print_exc()
        return None

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
    # from data_processing import get_processed_data # Not run when this file is imported
    # current_debug_state = DEBUG
    # DEBUG = True

    # Minimal example for direct testing, assumes data is pre-loaded if used
    print("--- Volatility Modeling Direct Run Example (requires manual data prep if used) ---")
    # Dummy data for minimal test
    dummy_returns = pd.Series(np.random.randn(100) / 100) # 100 days of small random returns
    dummy_vix = pd.Series(np.random.rand(100) * 20 + 10) / 100 # VIX like values
    
    print("Fitting dummy GARCH model...")
    fitted_dummy = fit_garch_model(dummy_returns, exog_series=dummy_vix, p=1,o=0,q=1, vol='GARCH', dist='Normal')
    if fitted_dummy:
        print("Forecasting with dummy model...")
        forecasted_dummy_vol = forecast_volatility(fitted_dummy, last_obs_exog_for_forecast=dummy_vix.iloc[-1:].values)
        if forecasted_dummy_vol is not None:
            print(f"Dummy forecast: {forecasted_dummy_vol.iloc[0]}")

    # DEBUG = current_debug_state