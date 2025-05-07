# /Users/omarabul-hassan/Desktop/projects/trading/src/data_processing.py
"""
Module for loading, cleaning, and preprocessing S&P 500 and VIX data.
"""
import pandas as pd
import numpy as np
import os

BASE_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

def clean_numeric_column(series, column_name="Unknown"):
    """Helper to clean numeric columns with commas."""
    if series.dtype == 'object':
        # First replace empty strings with NaN so they don't cause issues with comma removal
        series_cleaned = series.replace('', np.nan)
        # Then remove commas. If a value was originally empty, it's now NaN and unaffected by replace.
        series_cleaned = series_cleaned.str.replace(',', '', regex=False)
        try:
            return series_cleaned.astype(float)
        except ValueError as e:
            print(f"DEBUG data_processing: ValueError converting column '{column_name}' to float after cleaning. Error: {e}")
            print(f"DEBUG data_processing: Sample problematic values in '{column_name}': {series_cleaned[series_cleaned.apply(lambda x: not isinstance(x, (int, float)) and pd.notna(x))].head()}")
            # Attempt to coerce errors to NaN
            return pd.to_numeric(series_cleaned, errors='coerce')
    return series.astype(float) # If already numeric, ensure it's float

def load_snp_data(file_path=os.path.join(BASE_DATA_PATH, 'snp500.csv')):
    """
    Loads and preprocesses S&P 500 data.
    """
    print(f"DEBUG data_processing: Attempting to load S&P 500 data from {file_path}")
    try:
        df = pd.read_csv(file_path, na_filter=False) # Read empty strings as strings first
    except FileNotFoundError:
        print(f"Error: S&P 500 data file not found at {file_path}")
        return None

    print(f"DEBUG data_processing: S&P 500 raw columns: {df.columns.tolist()}")
    df.columns = [col.lower().replace(' ', '_').replace('.', '').replace('%', 'pct') for col in df.columns]
    print(f"DEBUG data_processing: S&P 500 standardized columns: {df.columns.tolist()}")
    
    if 'date' not in df.columns:
        print("Error: 'date' column not found in S&P 500 data.")
        return None
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    
    numeric_cols_to_process = ['price', 'open', 'high', 'low']
    for col in numeric_cols_to_process:
        if col in df.columns:
            print(f"DEBUG data_processing: Cleaning S&P 500 column '{col}'...")
            df[col] = clean_numeric_column(df[col], column_name=f"S&P500_{col}")
        else:
            print(f"Error: Critical S&P 500 column '{col}' not found.")
            return None

    if 'price' in df.columns:
        df.rename(columns={'price': 'close'}, inplace=True)
    else: # Should have been caught above if 'price' was missing
        print("Error: 'price' column (expected as 'close') not found.")
        return None

    df.set_index('date', inplace=True)
    df.sort_index(ascending=True, inplace=True)
    
    print(f"DEBUG data_processing: S&P 500 data shape before log returns: {df.shape}")
    if df['close'].isna().any():
        print(f"DEBUG data_processing: NaNs found in S&P500 'close' column before log returns: {df['close'].isna().sum()} NANS")
    
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    print(f"DEBUG data_processing: S&P 500 data shape after log returns: {df.shape}")
    df.dropna(subset=['log_return'], inplace=True) # Crucial: removes first row and any rows where close was NaN
    print(f"DEBUG data_processing: S&P 500 data shape after dropping NaN log_return: {df.shape}")
    
    final_cols_snp = ['open', 'high', 'low', 'close', 'log_return']
    if 'vol' in df.columns:
        print("DEBUG data_processing: Processing 'vol' column...")
        try:
            # Handle 'M', 'B', 'K' suffixes if they exist. For now, simple numeric conversion attempt.
            df['vol_cleaned'] = df['vol'].replace('', np.nan) # Treat empty strings as NaN
            df['vol_cleaned'] = clean_numeric_column(df['vol_cleaned'], column_name="S&P500_vol")
            if not df['vol_cleaned'].isna().all(): # If not all NaNs after cleaning
                 final_cols_snp.append('vol_cleaned')
                 df.rename(columns={'vol_cleaned': 'volume'}, inplace=True)
            else:
                print("DEBUG data_processing: 'vol' column is all NaNs after cleaning, not included.")
        except Exception as e:
            print(f"DEBUG data_processing: Could not process 'vol' column due to error: {e}. Not included.")
            pass
    
    missing_final_cols = [col for col in final_cols_snp if col not in df.columns and col != 'volume'] # 'volume' is optional
    if missing_final_cols:
        print(f"Error: Expected S&P columns missing after processing: {missing_final_cols}")
        return None

    return df[final_cols_snp].copy()


def load_vix_data(file_path=os.path.join(BASE_DATA_PATH, 'vix.csv')):
    """Loads and preprocesses VIX data."""
    print(f"DEBUG data_processing: Attempting to load VIX data from {file_path}")
    try:
        df = pd.read_csv(file_path, na_filter=False) # Read empty strings as strings first
    except FileNotFoundError:
        print(f"Error: VIX data file not found at {file_path}")
        return None

    print(f"DEBUG data_processing: VIX raw columns: {df.columns.tolist()}")
    df.columns = [col.lower() for col in df.columns]
    print(f"DEBUG data_processing: VIX standardized columns: {df.columns.tolist()}")
    
    if 'date' not in df.columns:
        print("Error: 'date' column not found in VIX data.")
        return None
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    
    numeric_cols_vix = ['open', 'high', 'low', 'close']
    for col in numeric_cols_vix:
        if col in df.columns:
            print(f"DEBUG data_processing: Cleaning VIX column '{col}'...")
            df[col] = clean_numeric_column(df[col], column_name=f"VIX_{col}")
        else:
            print(f"Error: Critical VIX column '{col}' not found.")
            return None
            
    df.rename(columns={'close': 'vix_close', 
                       'open': 'vix_open', 
                       'high': 'vix_high', 
                       'low': 'vix_low'}, inplace=True)
    
    df.set_index('date', inplace=True)
    df.sort_index(ascending=True, inplace=True)
    
    final_cols_vix = ['vix_close', 'vix_open', 'vix_high', 'vix_low']
    missing_final_cols = [col for col in final_cols_vix if col not in df.columns]
    if missing_final_cols:
        print(f"Error: Expected VIX columns missing after processing: {missing_final_cols}")
        return None
        
    return df[final_cols_vix].copy()


def merge_data(snp_df, vix_df):
    """Merges S&P 500 and VIX data."""
    if snp_df is None or vix_df is None:
        print("DEBUG data_processing: One or both dataframes for merging is None.")
        return None
    
    print(f"DEBUG data_processing: Merging S&P (shape {snp_df.shape}, index {snp_df.index.name}) and VIX (shape {vix_df.shape}, index {vix_df.index.name})")
    merged_df = pd.merge(snp_df, vix_df, left_index=True, right_index=True, how='inner')
    print(f"DEBUG data_processing: Merged df shape before NaN drop on critical: {merged_df.shape}")
    
    critical_cols_for_model = ['log_return', 'vix_close', 'close'] # 'close' needed for current_price
    
    for col in critical_cols_for_model:
        if col not in merged_df.columns:
            print(f"CRITICAL ERROR data_processing: Column '{col}' missing in merged_df.")
            return None
        if merged_df[col].isna().any():
            print(f"DEBUG data_processing: NaNs found in merged_df column '{col}': {merged_df[col].isna().sum()} NANS. Dropping these rows.")
            
    merged_df.dropna(subset=critical_cols_for_model, inplace=True)
    print(f"DEBUG data_processing: Merged df shape after NaN drop on critical: {merged_df.shape}")
    
    if merged_df.empty:
        print("CRITICAL ERROR data_processing: Merged dataframe is empty after processing and NaN removal.")
        return None
        
    return merged_df

def get_processed_data():
    """Orchestrates data loading and processing."""
    print("--- Starting Data Processing ---")
    snp_data = load_snp_data()
    if snp_data is None:
        print("Failed to load/process S&P 500 data.")
        return None
    print("S&P 500 data processed successfully.")
    
    vix_data = load_vix_data()
    if vix_data is None:
        print("Failed to load/process VIX data.")
        return None
    print("VIX data processed successfully.")
    
    processed_data = merge_data(snp_data, vix_data)
    if processed_data is None:
        print("Data merging failed or resulted in empty dataframe.")
        return None
        
    print(f"--- Data Processing Complete ---")
    print(f"Final processed data shape: {processed_data.shape}")
    print(f"Final data range: {processed_data.index.min()} to {processed_data.index.max()}")
    
    return processed_data

if __name__ == '__main__':
    data = get_processed_data()
    if data is not None:
        print("\nFirst 5 rows of final processed data:")
        print(data.head())
        print("\nLast 5 rows of final processed data:")
        print(data.tail())
        print("\nInfo on final processed data:")
        data.info()
        print("\nDescription of final processed data:")
        print(data.describe())
        print(f"\nNaN check in final log_return: {data['log_return'].isna().sum()}")
        print(f"NaN check in final vix_close: {data['vix_close'].isna().sum()}")