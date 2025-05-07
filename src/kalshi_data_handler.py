# /Users/omarabul-hassan/Desktop/projects/trading/src/kalshi_data_handler.py
"""
Module for interacting with the Kalshi API to fetch historical contract data,
specifically focusing on S&P 500 EOD range markets.
... (rest of docstring) ...
"""

import requests
import datetime
import time
import os
import base64 # <-------------------- ADD THIS IMPORT
import json
import re
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.backends import default_backend
from cryptography.exceptions import InvalidSignature
import pandas as pd
from urllib.parse import urlparse
import numpy as np 

DEBUG = True 

# --- Configuration (Unchanged) ---
KALSHI_API_KEY_ID = os.environ.get("KALSHI_PROD_API_KEY_ID", "c92b415e-ccc6-4093-9666-9ee8f4424260") 
PRIVATE_KEY_PATH = os.environ.get("KALSHI_PROD_PRIVATE_KEY_PATH", "/Users/omarabul-hassan/Desktop/projects/trading/privatekey.pem") 
BASE_URL = "https://api.elections.kalshi.com" 
API_PREFIX = "/trade-api/v2" 
SP500_SERIES_TICKER = "KXINX"
REQUEST_TIMEOUT = 30 
REQUEST_DELAY = 0.7 

# --- Helper Functions (Unchanged, but now base64 is available) ---
def dprint(*args, **kwargs): # ... definition ...
    if DEBUG: print("DEBUG kalshi_data_handler:", *args, **kwargs)
def load_private_key_from_file(file_path): # ... definition ...
    dprint(f"Attempting to load private key from: {file_path}")
    if not os.path.exists(file_path): print(f"CRITICAL ERROR: Private key file not found at {file_path}"); return None
    try:
        with open(file_path, "rb") as key_file: private_key = serialization.load_pem_private_key(key_file.read(), password=None, backend=default_backend())
        dprint("Private key loaded successfully."); return private_key
    except Exception as e: print(f"CRITICAL ERROR: Failed to load private key. Error: {e}"); return None
    
def sign_pss_text(private_key: rsa.RSAPrivateKey, text: str) -> str | None:
    """Signs text using RSA-PSS with SHA256.""" # Uses base64
    if private_key is None: return None
    message = text.encode('utf-8')
    try:
        signature = private_key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256())
        # base64 is now defined because we imported it at the top
        return base64.b64encode(signature).decode('utf-8') 
    except Exception as e:
        print(f"ERROR: Signing failed: {e}")
        return None

def parse_range_from_market_data(market_ticker: str, market_data: dict) -> tuple[float | None, float | None]: # ... definition unchanged ...
    dprint(f"Attempting range parse for ticker: {market_ticker}")
    match_bracket = re.search(r'-B(\d{4,})(?:-(\d{4,}))?$', market_ticker)
    if match_bracket:
        try:
            low = float(match_bracket.group(1))
            high = float(match_bracket.group(2)) if match_bracket.group(2) else low + 24.9999 
            dprint(f"Parsed range from ticker bracket '{market_ticker}': ({low}, {high})")
            return low, high
        except ValueError as e:
             dprint(f"Could not convert parsed ticker bracket range values: {e}")
    strike_type = market_data.get('strike_type')
    cap_strike = market_data.get('cap_strike')
    floor_strike = market_data.get('floor_strike') 
    if strike_type == 'less' and cap_strike is not None:
        try:
            high = float(cap_strike)
            dprint(f"Parsed range from strike_type 'less' and cap_strike: (-inf, {high})")
            return -np.inf, high
        except (ValueError, TypeError):
            dprint(f"Could not convert cap_strike '{cap_strike}' to float.")
    elif strike_type == 'greater' and floor_strike is not None: 
        try:
            low = float(floor_strike)
            dprint(f"Parsed range from strike_type 'greater' and floor_strike: ({low}, +inf)")
            return low, np.inf
        except (ValueError, TypeError):
            dprint(f"Could not convert floor_strike '{floor_strike}' to float.")
    elif strike_type == 'between' and floor_strike is not None and cap_strike is not None:
        try:
             low = float(floor_strike)
             high = float(cap_strike)
             dprint(f"Parsed range from strike_type 'between': ({low}, {high})")
             return low, high
        except (ValueError, TypeError):
             dprint(f"Could not convert floor/cap strikes for 'between': {floor_strike}, {cap_strike}")
    title = market_data.get('title', '')
    match_title = re.search(r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:to|and|-)\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)', title, re.IGNORECASE)
    if title and match_title: 
         try:
            low_str = match_title.group(1).replace(',', '')
            high_str = match_title.group(2).replace(',', '')
            low = float(low_str)
            high = float(high_str)
            dprint(f"Parsed range from title '{title}': ({low}, {high})")
            return low, high
         except ValueError as e:
            dprint(f"Could not convert parsed title range values: {e}")
    print(f"Warning: Could not determine range for market {market_ticker} from available data.")
    dprint(f"Relevant market data fields: strike_type={strike_type}, cap_strike={cap_strike}, floor_strike={floor_strike}, title='{title}'")
    return None, None
    
def format_event_ticker(target_date: datetime.date) -> str: # ... definition unchanged ...
    date_str = target_date.strftime('%y%b%d').upper()
    event_ticker = f"{SP500_SERIES_TICKER}-{date_str}H1600"
    dprint(f"Formatted event ticker for {target_date}: {event_ticker}")
    return event_ticker

# --- Kalshi API Client Class (Unchanged) ---
class KalshiAPIClient:
    # ... (Implementation unchanged) ...
    def __init__(self, api_key_id: str, private_key_path: str, base_url: str): # ... unchanged ...
        self.api_key_id = api_key_id
        self.private_key = load_private_key_from_file(private_key_path) 
        self.base_url = base_url
        self.session = requests.Session()
        self.last_api_call_time = datetime.datetime.now() - datetime.timedelta(seconds=REQUEST_DELAY)
    def _rate_limit(self): # ... unchanged ...
        now = datetime.datetime.now()
        time_since_last_call = now - self.last_api_call_time
        required_delay = datetime.timedelta(seconds=REQUEST_DELAY)
        if time_since_last_call < required_delay:
            sleep_duration = (required_delay - time_since_last_call).total_seconds()
            dprint(f"Rate limiting: sleeping for {sleep_duration:.3f} seconds.")
            time.sleep(sleep_duration)
        self.last_api_call_time = datetime.datetime.now()
    def _get_auth_headers(self, method: str, path_with_query: str) -> dict | None: # ... unchanged ...
        if self.private_key is None: return None
        timestamp_ms = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
        timestamp_str = str(timestamp_ms)
        path_parts = urlparse(path_with_query)
        path_for_signing = path_parts.path 
        if not path_for_signing.startswith(API_PREFIX):
             print(f"CRITICAL WARNING: path_for_signing '{path_for_signing}' does not start with API_PREFIX '{API_PREFIX}'.")
             path_for_signing = API_PREFIX + path_for_signing
        msg_string = timestamp_str + method.upper() + path_for_signing
        dprint(f"String to sign: {msg_string}")
        signature = sign_pss_text(self.private_key, msg_string) # Will work now
        if signature is None: return None
        headers = {'KALSHI-ACCESS-KEY': self.api_key_id, 'KALSHI-ACCESS-SIGNATURE': signature, 'KALSHI-ACCESS-TIMESTAMP': timestamp_str,'Accept': 'application/json','Content-Type': 'application/json'}
        dprint(f"Auth Headers: KEY={headers['KALSHI-ACCESS-KEY']}, SIG={headers['KALSHI-ACCESS-SIGNATURE'][:10]}..., TS={headers['KALSHI-ACCESS-TIMESTAMP']}")
        return headers
    def _make_request(self, method: str, path: str, params: dict = None, json_data: dict = None) -> dict | None: # ... unchanged ...
        self._rate_limit() 
        full_path_with_prefix = path if path.startswith(API_PREFIX) else API_PREFIX + path
        path_for_headers = full_path_with_prefix 
        if params:
            query_string = requests.models.RequestEncodingMixin._encode_params(params)
            path_for_headers += f"?{query_string}" 
        headers = self._get_auth_headers(method, path_for_headers)
        if headers is None: return None
        url = self.base_url + full_path_with_prefix
        dprint(f"Making {method} request to {url} with params: {params}, json: {json_data}")
        try:
            response = self.session.request(method=method, url=url, headers=headers, params=params, json=json_data, timeout=REQUEST_TIMEOUT)
            dprint(f"Request sent. Status Code: {response.status_code}")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            if response.status_code == 404: print(f"HTTP error: 404 Not Found for {url}.")
            else: print(f"HTTP error occurred: {http_err} - {response.status_code}")
            try: dprint(f"Error body: {response.text}")
            except: pass
            return None
        except requests.exceptions.Timeout: print(f"Request timed out after {REQUEST_TIMEOUT} seconds for {url}"); return None
        except requests.exceptions.RequestException as req_err: print(f"Request exception occurred: {req_err}"); return None
        except json.JSONDecodeError: print(f"Error decoding JSON response from {url}. Status: {response.status_code}, Body: {response.text[:500]}..."); return None
    # --- Specific Endpoint Methods (Unchanged) ---
    def get_exchange_status(self) -> dict | None: # ... definition ...
        path = "/exchange/status"
        dprint(f"Calling get_exchange_status")
        return self._make_request("GET", path)
    def get_series_list(self, category: str | None = None) -> dict | None: # ... definition ...
        path = "/series/"
        params = {}
        if category: params['category'] = category
        dprint(f"Calling get_series_list with category={category}")
        return self._make_request("GET", path, params=params if params else None)
    def get_event_with_markets(self, event_ticker: str) -> dict | None: # ... definition ...
        path = f"/events/{event_ticker}"
        params = {"with_nested_markets": "true"}
        dprint(f"Calling get_event_with_markets for {event_ticker}")
        response_data = self._make_request("GET", path, params=params)
        if response_data is not None and isinstance(response_data, dict) and 'event' in response_data:
             return response_data.get('event')
        elif response_data is not None:
             dprint(f"Unexpected response structure for get_event: {list(response_data.keys())}")
             return response_data 
        else:
            print(f"Error fetching event {event_ticker} or None response.")
            return None
    def get_market(self, market_ticker: str) -> dict | None: # ... definition ...
        path = f"/markets/{market_ticker}"
        dprint(f"Calling get_market for {market_ticker}")
        response_data = self._make_request("GET", path)
        if response_data is not None and isinstance(response_data, dict) and 'market' in response_data:
            return response_data.get('market')
        elif response_data is not None:
             dprint(f"Unexpected response structure for get_market: {list(response_data.keys())}")
             return response_data
        else:
            print(f"Error fetching market {market_ticker} or None response.")
            return None
    def get_market_candlesticks(self, market_ticker: str, series_ticker: str,
                                start_ts: int, end_ts: int,
                                period_interval: int = 60) -> list[dict] | None: # ... definition ...
        path = f"/series/{series_ticker}/markets/{market_ticker}/candlesticks"
        params = {"start_ts": start_ts, "end_ts": end_ts, "period_interval": period_interval}
        dprint(f"Calling get_market_candlesticks for {market_ticker} with params: {params}")
        response_data = self._make_request("GET", path, params=params)
        if response_data is None or 'candlesticks' not in response_data:
            dprint(f"No candlesticks found or error for {market_ticker} in range {start_ts}-{end_ts}.")
            return None
        return response_data.get('candlesticks', [])

# --- Main Loading Function (Unchanged) ---
def load_historical_kalshi_contracts_by_ticker(event_ticker: str) -> dict | None:
    # ... (Implementation unchanged) ...
    print(f"\n--- Loading Kalshi Contracts for Event Ticker: {event_ticker} ---")
    client = KalshiAPIClient(KALSHI_API_KEY_ID, PRIVATE_KEY_PATH, BASE_URL)
    if client.private_key is None: return None
    event_data = client.get_event_with_markets(event_ticker)
    if event_data is None: return None
    close_ts = event_data.get('close_ts')
    target_date = None
    if close_ts: target_date = datetime.datetime.fromtimestamp(close_ts, tz=datetime.timezone.utc).date(); print(f"Extracted target_date from event close_ts: {target_date}")
    else:
        match = re.search(r'-(\d{2}[A-Z]{3}\d{2})H\d{4}$', event_ticker)
        if match:
            try: target_date = pd.to_datetime(match.group(1), format='%y%b%d').date(); print(f"Extracted target_date from event_ticker: {target_date}")
            except ValueError: print("Could not parse date from event ticker.")
        if target_date is None: print("CRITICAL ERROR: Could not determine target date."); return None
    markets_in_event = event_data.get('markets');
    if not markets_in_event: return {} 
    print(f"Found {len(markets_in_event)} markets for event {event_ticker}.")
    contracts_data = {}
    for market_stub in markets_in_event:
        market_ticker = market_stub.get('ticker')
        if not market_ticker or not market_ticker.startswith(SP500_SERIES_TICKER): dprint(f"Skipping market stub with ticker: {market_ticker}"); continue
        dprint(f"\nProcessing Market Ticker: {market_ticker}"); market_details = client.get_market(market_ticker)
        if market_details is None: print(f"Warning: Failed to get full details for market {market_ticker}. Skipping."); continue
        range_low, range_high = parse_range_from_market_data(market_ticker, market_details)
        if range_low is None or range_high is None: continue
        yes_bid = market_details.get('yes_bid'); yes_ask = market_details.get('yes_ask'); last_price = market_details.get('last_price') 
        dprint(f"  Market Details: Bid={yes_bid}, Ask={yes_ask}, Last={last_price}")
        contracts_data[market_ticker] = {'ticker': market_ticker, 'event_ticker': event_ticker, 'series_ticker': SP500_SERIES_TICKER, 'title': market_details.get('title',''),'range_low': range_low, 'range_high': range_high, 'last_price_proxy': last_price, 'yes_bid': yes_bid, 'yes_ask': yes_ask, 'settlement': market_details.get('result') }
    print(f"--- Finished loading Kalshi data for {event_ticker}. Processed {len(contracts_data)} valid S&P contracts. ---")
    return contracts_data

# --- Main Execution Block (Unchanged) ---
if __name__ == '__main__':
    example_event_ticker = "KXINX-25MAY07H1600" 
    print(f"Attempting to load PRODUCTION data using specific event ticker: {example_event_ticker}")
    if "YOUR_PRODUCTION_API_KEY_ID_HERE" in KALSHI_API_KEY_ID or \
       "path/to/your/PRODUCTION_private_key.pem" in PRIVATE_KEY_PATH or \
       not os.path.exists(PRIVATE_KEY_PATH):
         print("\n[ERROR] PLEASE CONFIGURE **PRODUCTION** KALSHI API KEYS IN SCRIPT OR ENV VARS\n")
    else:
        historical_data = load_historical_kalshi_contracts_by_ticker(example_event_ticker)
        if historical_data is not None:
            print(f"\n--- Loaded Kalshi Contract Data Summary for {example_event_ticker} ---")
            if historical_data:
                sorted_tickers = sorted(historical_data.keys(), key=lambda t: historical_data[t].get('range_low', float('inf')))
                for ticker in sorted_tickers:
                    data = historical_data[ticker]
                    print(f"  Ticker: {ticker}")
                    print(f"    Range: ({data['range_low']} - {data['range_high']})")
                    print(f"    Yes Bid:     {data['yes_bid']}")
                    print(f"    Yes Ask:     {data['yes_ask']}")
                    print(f"    Last Price:  {data['last_price_proxy']}") 
                    print(f"    Settlement:  {data['settlement']}") 
                    print("-" * 15)
            else:
                print("  No markets found or processed for this event ticker (check API response/event status).")
        else:
            print("\nFailed to load PRODUCTION historical Kalshi data by ticker (check logs/API errors).")