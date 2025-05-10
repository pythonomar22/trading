#!/usr/bin/env python3
"""
Module for interacting with the Kalshi API to fetch historical contract data,
specifically focusing on S&P 500 EOD range markets.
"""

import requests
import datetime
import time
import os
import base64
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

# --- Configuration ---
KALSHI_API_KEY_ID = os.environ.get("KALSHI_PROD_API_KEY_ID", "c92b415e-ccc6-4093-9666-9ee8f4424260")
PRIVATE_KEY_PATH = os.environ.get("KALSHI_PROD_PRIVATE_KEY_PATH", "/Users/omarabul-hassan/Desktop/projects/trading/privatekey.pem")
BASE_URL = "https://api.elections.kalshi.com"
API_PREFIX = "/trade-api/v2"
SP500_SERIES_TICKER = "KXINX" # This seems to be the prefix for the events you're targeting
                              # even if the full series is KXINXU in some contexts.
                              # The script filters markets based on this prefix.
REQUEST_TIMEOUT = 30
REQUEST_DELAY = 0.7 # Increased slightly for more calls

# --- Helper Functions ---
def dprint(*args, **kwargs):
    if DEBUG: print("DEBUG kalshi_data_handler:", *args, **kwargs)

def load_private_key_from_file(file_path):
    dprint(f"Attempting to load private key from: {file_path}")
    if not os.path.exists(file_path): print(f"CRITICAL ERROR: Private key file not found at {file_path}"); return None
    try:
        with open(file_path, "rb") as key_file: private_key = serialization.load_pem_private_key(key_file.read(), password=None, backend=default_backend())
        dprint("Private key loaded successfully."); return private_key
    except Exception as e: print(f"CRITICAL ERROR: Failed to load private key. Error: {e}"); return None

def sign_pss_text(private_key: rsa.RSAPrivateKey, text: str) -> str | None:
    if private_key is None: return None
    message = text.encode('utf-8')
    try:
        signature = private_key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256())
        return base64.b64encode(signature).decode('utf-8')
    except Exception as e: print(f"ERROR: Signing failed: {e}"); return None

def parse_range_from_market_data(market_ticker: str, market_data: dict) -> tuple[float | None, float | None]:
    dprint(f"Attempting range parse for ticker: {market_ticker}")
    # Handle -TXXXX (threshold) markets first as they are common in your example
    # KXINX-25MAY07H1600-T5300 implies S&P <= 5300 (if 'less') or S&P > 5300 (if 'greater')
    # KXINXU-25MAY08H1600-T5624.9999 -> from your image, this is likely "S&P > 5624.9999"
    
    strike_type = market_data.get('strike_type')
    cap_strike = market_data.get('cap_strike') # For "less than or equal to"
    floor_strike = market_data.get('floor_strike') # For "greater than or equal to"

    if strike_type == 'less' and cap_strike is not None: # Kalshi often uses 'less' for 'less_than_or_equal_to'
        try:
            high = float(cap_strike)
            dprint(f"Parsed as THRESHOLD from strike_type 'less' and cap_strike: (-inf, {high})")
            return -np.inf, high
        except (ValueError, TypeError): dprint(f"Could not convert cap_strike '{cap_strike}' to float.")
    elif strike_type == 'greater' and floor_strike is not None: # Kalshi often uses 'greater' for 'greater_than_or_equal_to'
        try:
            low = float(floor_strike)
            dprint(f"Parsed as THRESHOLD from strike_type 'greater' and floor_strike: ({low}, +inf)")
            return low, np.inf
        except (ValueError, TypeError): dprint(f"Could not convert floor_strike '{floor_strike}' to float.")

    # Then handle -BXXXX (range) markets
    match_bracket = re.search(r'-B(\d{4,})(?:-B(\d{4,}))?$', market_ticker) # -BXXXX or -BXXXX-BYYYY
    if match_bracket:
        try:
            low = float(match_bracket.group(1))
            if match_bracket.group(2): # If -BXXXX-BYYYY form
                 high = float(match_bracket.group(2))
            else: # If only -BXXXX form, assume it's a 25-point range like your original script
                 high = low + 24.9999 
            dprint(f"Parsed as RANGE from ticker bracket '{market_ticker}': ({low}, {high})")
            return low, high
        except ValueError as e:
             dprint(f"Could not convert parsed ticker bracket range values: {e}")

    # Fallback for 'between' strike_type if ticker parsing didn't catch it (less common for -B)
    if strike_type == 'between' and floor_strike is not None and cap_strike is not None:
        try:
             low = float(floor_strike); high = float(cap_strike)
             dprint(f"Parsed as RANGE from strike_type 'between': ({low}, {high})")
             return low, high
        except (ValueError, TypeError): dprint(f"Could not convert floor/cap strikes for 'between': {floor_strike}, {cap_strike}")
    
    # Fallback to title parsing (less critical if ticker and strike_types are good)
    title = market_data.get('title', '')
    match_title = re.search(r'(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*(?:to|and|-)\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)', title, re.IGNORECASE)
    if title and match_title:
         try:
            low_str = match_title.group(1).replace(',', ''); high_str = match_title.group(2).replace(',', '')
            low = float(low_str); high = float(high_str)
            dprint(f"Parsed range from title '{title}': ({low}, {high})")
            return low, high
         except ValueError as e: dprint(f"Could not convert parsed title range values: {e}")
             
    print(f"Warning: Could not determine range/threshold for market {market_ticker} from available data.")
    dprint(f"Relevant market data fields: strike_type={strike_type}, cap_strike={cap_strike}, floor_strike={floor_strike}, title='{title}'")
    return None, None

def format_event_ticker(target_date: datetime.date) -> str:
    date_str = target_date.strftime('%y%b%d').upper()
    # Using the SP500_SERIES_TICKER from config which is KXINX. This matches your successful test.
    event_ticker = f"{SP500_SERIES_TICKER}-{date_str}H1600"
    dprint(f"Formatted event ticker for {target_date}: {event_ticker}")
    return event_ticker

# --- Kalshi API Client Class ---
class KalshiAPIClient:
    def __init__(self, api_key_id: str, private_key_path: str, base_url: str):
        self.api_key_id = api_key_id
        self.private_key = load_private_key_from_file(private_key_path)
        self.base_url = base_url
        self.session = requests.Session()
        self.last_api_call_time = datetime.datetime.now() - datetime.timedelta(seconds=REQUEST_DELAY)

    def _rate_limit(self):
        now = datetime.datetime.now()
        time_since_last_call = now - self.last_api_call_time
        required_delay = datetime.timedelta(seconds=REQUEST_DELAY)
        if time_since_last_call < required_delay:
            sleep_duration = (required_delay - time_since_last_call).total_seconds()
            if sleep_duration > 0: # Ensure positive sleep
                dprint(f"Rate limiting: sleeping for {sleep_duration:.3f} seconds.")
                time.sleep(sleep_duration)
        self.last_api_call_time = datetime.datetime.now()

    def _get_auth_headers(self, method: str, path_with_query: str) -> dict | None:
        if self.private_key is None: return None
        timestamp_ms = int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1000)
        timestamp_str = str(timestamp_ms)
        path_parts = urlparse(path_with_query)
        path_for_signing = path_parts.path
        if not path_for_signing.startswith(API_PREFIX):
             # This warning was helpful, but let's ensure it doesn't break things if API_PREFIX is already part of path_for_signing
             # If path_for_signing ALREADY starts with API_PREFIX due to how path was constructed, this is fine.
             # The critical part is that msg_string matches what Kalshi expects.
             # Kalshi docs example for signing: timestampt_str + method + path
             # where path is like '/trade-api/v2/portfolio/balance'
             # So path_for_signing should be the path part starting with /trade-api/v2
             if not path_for_signing.startswith("/"): path_for_signing = "/" + path_for_signing
             if not path_for_signing.startswith(API_PREFIX):
                dprint(f"CRITICAL WARNING: path_for_signing '{path_for_signing}' does not start with API_PREFIX '{API_PREFIX}'. Prepending.")
                path_for_signing = API_PREFIX + path_for_signing # This might double-prefix if path already had it.

        # Ensure path_for_signing has exactly one API_PREFIX at the start
        if path_for_signing.count(API_PREFIX) > 1 :
            path_for_signing = API_PREFIX + path_for_signing.split(API_PREFIX, 1)[-1]
        elif not path_for_signing.startswith(API_PREFIX):
             path_for_signing = API_PREFIX + path_for_signing


        msg_string = timestamp_str + method.upper() + path_for_signing
        dprint(f"String to sign: {msg_string}")
        signature = sign_pss_text(self.private_key, msg_string)
        if signature is None: return None
        headers = {'KALSHI-ACCESS-KEY': self.api_key_id, 'KALSHI-ACCESS-SIGNATURE': signature, 'KALSHI-ACCESS-TIMESTAMP': timestamp_str,'Accept': 'application/json','Content-Type': 'application/json'}
        dprint(f"Auth Headers: KEY={headers['KALSHI-ACCESS-KEY']}, SIG={headers['KALSHI-ACCESS-SIGNATURE'][:10]}..., TS={headers['KALSHI-ACCESS-TIMESTAMP']}")
        return headers

    def _make_request(self, method: str, path: str, params: dict = None, json_data: dict = None) -> dict | None:
        self._rate_limit()
        # Ensure full_path_with_prefix correctly uses API_PREFIX
        # Path for _get_auth_headers must include query params.
        # Path for the actual request URL construction must not include query params if params dict is used by requests lib.
        
        path_for_signing_and_url = path # This is the path relative to base_url, e.g., "/events/TICKER"
        if not path_for_signing_and_url.startswith(API_PREFIX):
            path_for_signing_and_url = API_PREFIX + path_for_signing_and_url

        path_for_headers_signature = path_for_signing_and_url # Start with the base path + API_PREFIX
        if params:
            query_string = requests.models.RequestEncodingMixin._encode_params(params)
            path_for_headers_signature += f"?{query_string}"
        
        headers = self._get_auth_headers(method, path_for_headers_signature)
        if headers is None: return None

        url = self.base_url + path_for_signing_and_url # path_for_signing_and_url already includes API_PREFIX
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
        except requests.exceptions.Timeout: print(f"Request timed out: {url}"); return None
        except requests.exceptions.RequestException as e: print(f"Request exception: {e}"); return None
        except json.JSONDecodeError: print(f"JSON decode error. Status: {response.status_code}, Body: {response.text[:200]}"); return None

    def get_exchange_status(self) -> dict | None:
        path = "/exchange/status" # Path relative to API_PREFIX
        dprint(f"Calling get_exchange_status")
        return self._make_request("GET", path)

    def get_event_with_markets(self, event_ticker: str) -> dict | None:
        path = f"/events/{event_ticker}" # Path relative to API_PREFIX
        params = {"with_nested_markets": "true"}
        dprint(f"Calling get_event_with_markets for {event_ticker}")
        response_data = self._make_request("GET", path, params=params)
        if response_data and 'event' in response_data: return response_data.get('event')
        dprint(f"Unexpected response or error for get_event: {response_data}"); return None

    def get_market(self, market_ticker: str) -> dict | None:
        path = f"/markets/{market_ticker}" # Path relative to API_PREFIX
        dprint(f"Calling get_market for {market_ticker}")
        response_data = self._make_request("GET", path)
        if response_data and 'market' in response_data: return response_data.get('market')
        dprint(f"Unexpected response or error for get_market: {response_data}"); return None

    def get_market_candlesticks(self, market_ticker: str, series_ticker_for_path: str,
                                start_ts: int, end_ts: int,
                                period_interval_minutes: int = 60) -> list[dict] | None:
        # Path: /series/{series_ticker}/markets/{market_ticker}/candlesticks
        # series_ticker_for_path should be the one associated with the market, e.g. KXINX or KXINXU
        path = f"/series/{series_ticker_for_path}/markets/{market_ticker}/candlesticks"
        params = {"start_ts": start_ts, "end_ts": end_ts, "period_interval": period_interval_minutes}
        dprint(f"Calling get_market_candlesticks for {market_ticker} (series for path: {series_ticker_for_path}) with params: {params}")
        response_data = self._make_request("GET", path, params=params)
        
        if response_data is None:
            dprint(f"No response from get_market_candlesticks for {market_ticker}.")
            return None
        if 'candlesticks' not in response_data:
            dprint(f"Key 'candlesticks' not in response for {market_ticker}. Response: {response_data}")
            return [] # Return empty list if key missing but response exists
            
        candlesticks = response_data.get('candlesticks', [])
        if not candlesticks:
            dprint(f"No candlesticks found for {market_ticker} in range {start_ts}-{end_ts}.")
        return candlesticks

# --- Main Loading Function ---
def load_historical_kalshi_data_with_candlesticks(event_ticker_to_load: str, 
                                                  candlestick_interval_minutes: int = 1) -> dict | None:
    print(f"\n--- Loading Kalshi Data for Event: {event_ticker_to_load} with {candlestick_interval_minutes}-min Candlesticks ---")
    client = KalshiAPIClient(KALSHI_API_KEY_ID, PRIVATE_KEY_PATH, BASE_URL)
    if client.private_key is None: return None

    event_data = client.get_event_with_markets(event_ticker_to_load)
    if event_data is None:
        print(f"Failed to load event data for {event_ticker_to_load}")
        return None

    # Determine event's open and close timestamps for candlestick fetching
    # Timestamps are in seconds
    event_open_ts = event_data.get('open_ts')
    event_close_ts = event_data.get('close_ts') # This is often the market expiration time

    if not event_open_ts or not event_close_ts:
        print(f"CRITICAL: Event {event_ticker_to_load} is missing open_ts or close_ts. Cannot fetch candlesticks.")
        # Fallback to trying to parse from event ticker if critical for date, but not ideal for ts
        match_date = re.search(r'-(\d{2}[A-Z]{3}\d{2})H(\d{2})(\d{2})$', event_ticker_to_load)
        if match_date:
            try:
                # This is a rough approximation, actual open/close can vary
                event_dt_str = match_date.group(1) + match_date.group(2) + match_date.group(3)
                # Kalshi typically uses US/Eastern for H notation
                # For simplicity, assume UTC for now if not specified, but this needs care
                # For S&P 500 EOD markets, H1600 is 4 PM ET.
                # Open is often market open, e.g. 9:30 AM ET.
                # This part is complex as open_ts might not be 00:00 of the event date.
                # Let's default to a wide range if specific open_ts is missing, but warn.
                # For now, if close_ts is available, we can try fetching for the day of close_ts.
                if event_close_ts:
                     close_datetime_utc = datetime.datetime.fromtimestamp(event_close_ts, tz=datetime.timezone.utc)
                     event_open_ts = int((close_datetime_utc.replace(hour=0, minute=0, second=0, microsecond=0)).timestamp()) # Start of close day UTC
                     dprint(f"Warning: event open_ts missing. Approximated open_ts to start of close_ts day (UTC): {event_open_ts}")
                else:
                    print("Error: Cannot determine candlestick range without event open/close timestamps.")
                    return None # Cannot proceed without timestamps for candlesticks
            except Exception as e:
                print(f"Error parsing date from event_ticker for timestamp fallback: {e}")
                return None
        else:
            print("Error: Cannot determine candlestick range without event open/close timestamps or parsable ticker.")
            return None


    markets_in_event = event_data.get('markets')
    if not markets_in_event:
        print(f"No markets found for event {event_ticker_to_load}.")
        return {}

    print(f"Found {len(markets_in_event)} markets for event {event_ticker_to_load}.")
    all_contracts_data = {}

    for market_stub in markets_in_event:
        market_ticker = market_stub.get('ticker')
        # SP500_SERIES_TICKER is "KXINX". Your example uses "KXINXU".
        # The event KXINX-25MAY07H1600 contains markets like KXINX-25MAY07H1600-T5300.
        # So filtering by SP500_SERIES_TICKER should work.
        if not market_ticker or not market_ticker.startswith(SP500_SERIES_TICKER):
            dprint(f"Skipping market stub with non-matching ticker prefix: {market_ticker} (expected: {SP500_SERIES_TICKER})")
            continue

        dprint(f"\nProcessing Market Ticker: {market_ticker}")
        market_details = client.get_market(market_ticker)
        if market_details is None:
            print(f"Warning: Failed to get full details for market {market_ticker}. Skipping.")
            continue

        range_low, range_high = parse_range_from_market_data(market_ticker, market_details)
        # If parse_range returns None, None, it means it couldn't determine, skip.
        if range_low is None and range_high is None and not (market_details.get('strike_type') in ['less', 'greater']):
            # If it's not a known threshold type AND range parsing failed, then skip
             print(f"Could not parse range/threshold for {market_ticker}, and not a clear threshold type. Skipping.")
             continue


        # Fetch candlesticks for this market
        # The 'series_ticker' in the market_details might be more specific (e.g., KXINXU)
        # The path for candlesticks is /series/{series_ticker}/markets/{market_ticker}/candlesticks
        # We need the series_ticker associated with *this market* for the path.
        market_series_ticker = market_details.get('series_ticker', SP500_SERIES_TICKER) # Fallback to global if not in market_details
        
        # Ensure we have valid timestamps for candlesticks
        # Individual markets might have their own open_ts and expiration_ts.
        # Use market's own expiration_ts if available, otherwise event_close_ts.
        # Use event_open_ts as a general start for the event's trading day.
        market_expiration_ts = market_details.get('expiration_ts', event_close_ts)
        
        # Make candlestick start/end specific to the market's active period if possible,
        # bounded by the event's overall activity period.
        # For simplicity, use event_open_ts and market_expiration_ts.
        # Ensure timestamps are integers.
        
        # Use a slightly wider window if possible, but Kalshi limits requests.
        # Let's use the event's open_ts and the market's expiration_ts.
        # The API requires end_ts to be within 5000 period_intervals after start_ts.
        # If interval is 1 min, 5000 mins = ~3.47 days. Max range.
        # If interval is 60 min, 5000 hours = ~208 days.
        # For a daily market, fetching data from event_open_ts to market_expiration_ts for that day should be fine.

        candlesticks_data = []
        if event_open_ts and market_expiration_ts:
            # Adjust start_ts to be closer to market open if known, otherwise event open
            # Adjust end_ts to be market expiration
            # Kalshi typically wants timestamps in seconds.
            
            # Ensure the range is not too large for the API
            # For a 1-minute interval, max duration is 5000 minutes.
            # For a 60-minute interval, max duration is 5000 * 60 minutes.
            max_duration_seconds = 5000 * candlestick_interval_minutes * 60
            current_duration_seconds = market_expiration_ts - event_open_ts

            if current_duration_seconds > max_duration_seconds:
                dprint(f"Candlestick range too large ({current_duration_seconds}s > {max_duration_seconds}s) for {market_ticker}. Skipping candlesticks.")
            elif market_expiration_ts < event_open_ts:
                dprint(f"Market expiration_ts ({market_expiration_ts}) is before event_open_ts ({event_open_ts}) for {market_ticker}. Skipping candlesticks.")
            else:
                candlesticks_data = client.get_market_candlesticks(
                    market_ticker,
                    series_ticker_for_path=market_series_ticker, # Use the market's specific series_ticker for path
                    start_ts=int(event_open_ts),
                    end_ts=int(market_expiration_ts),
                    period_interval_minutes=candlestick_interval_minutes
                )
                if candlesticks_data is None: candlesticks_data = [] # Ensure it's a list
        else:
            dprint(f"Missing timestamps for candlestick fetch for {market_ticker}.")


        all_contracts_data[market_ticker] = {
            'ticker': market_ticker,
            'event_ticker': event_ticker_to_load,
            'series_ticker': market_series_ticker, # Store the specific series from market_details
            'title': market_details.get('title',''),
            'range_low': range_low, # This will be -np.inf for "> X" or X for "X to Y"
            'range_high': range_high, # This will be Y for "X to Y" or +np.inf for "< Y"
            'strike_type': market_details.get('strike_type'),
            'floor_strike': market_details.get('floor_strike'),
            'cap_strike': market_details.get('cap_strike'),
            'last_price_proxy': market_details.get('last_price'),
            'yes_bid': market_details.get('yes_bid'),
            'yes_ask': market_details.get('yes_ask'),
            'settlement': market_details.get('result'),
            'candlesticks': candlesticks_data # Add the fetched candlesticks
        }
        dprint(f"  Market Details: Bid={all_contracts_data[market_ticker]['yes_bid']}, Ask={all_contracts_data[market_ticker]['yes_ask']}, Last={all_contracts_data[market_ticker]['last_price_proxy']}")
        dprint(f"  Fetched {len(candlesticks_data)} candlesticks for {market_ticker}")

    print(f"--- Finished loading Kalshi data for {event_ticker_to_load}. Processed {len(all_contracts_data)} valid S&P contracts. ---")
    return all_contracts_data

# --- Main Execution Block ---
if __name__ == '__main__':
    # Example: Use the event ticker that worked for you.
    # This event is in the future, so candlestick data might be empty or not yet available.
    # For backtesting, you'd pick an event_ticker for a PAST, SETTLED event.
    example_event_ticker = "KXINX-25MAY06H1600" 
    # To test with a past event (you'll need to find a valid one from Kalshi for KXINX series):
    # past_event_date = datetime.date(2024, 7, 1) # Example: July 1, 2024 (ensure it was a trading day)
    # example_event_ticker = format_event_ticker(past_event_date) 
    
    # For your provided KXINXU example, the event was KXINXU-25MAY08H1600
    # If your script uses SP500_SERIES_TICKER = "KXINX", then an equivalent would be:
    # example_event_ticker = "KXINX-25MAY08H1600" # If this event exists.
    # Your successful test used KXINX-25MAY07H1600. Let's stick with that.

    # Choose candlestick interval in minutes (1, 60 for hour, 1440 for day)
    # 1-minute data will result in many calls if the market was open for a long time.
    interval_minutes = 60 # Try hourly first to reduce number of data points / API calls

    print(f"Attempting to load PRODUCTION data for event: {example_event_ticker} with {interval_minutes}-min candlesticks")
    
    if "YOUR_PRODUCTION_API_KEY_ID_HERE" in KALSHI_API_KEY_ID or \
       "path/to/your/PRODUCTION_private_key.pem" in PRIVATE_KEY_PATH or \
       not os.path.exists(PRIVATE_KEY_PATH):
         print("\n[ERROR] PLEASE CONFIGURE **PRODUCTION** KALSHI API KEYS IN SCRIPT OR ENV VARS\n")
    else:
        historical_data_with_candles = load_historical_kalshi_data_with_candlesticks(
            example_event_ticker,
            candlestick_interval_minutes=interval_minutes
        )

        if historical_data_with_candles is not None:
            print(f"\n--- Loaded Kalshi Contract Data Summary for {example_event_ticker} ---")
            if historical_data_with_candles:
                # Sort by range_low if available, or ticker
                sorted_tickers = sorted(historical_data_with_candles.keys(), 
                                        key=lambda t: historical_data_with_candles[t].get('range_low', float('inf')) 
                                                      if historical_data_with_candles[t].get('range_low') is not None 
                                                      else historical_data_with_candles[t].get('floor_strike', float('inf'))
                                                           if historical_data_with_candles[t].get('floor_strike') is not None
                                                           else float('inf'))


                for ticker in sorted_tickers:
                    data = historical_data_with_candles[ticker]
                    print(f"  Ticker: {ticker} (Series for path: {data['series_ticker']})")
                    if data['range_low'] == -np.inf and data['range_high'] != np.inf :
                        print(f"    Type: Threshold (<= {data['range_high']})")
                    elif data['range_high'] == np.inf and data['range_low'] != -np.inf:
                        print(f"    Type: Threshold (>= {data['range_low']})")
                    elif data['range_low'] is not None and data['range_high'] is not None:
                        print(f"    Type: Range ({data['range_low']} - {data['range_high']})")
                    else:
                        print(f"    Type: Unknown/Other (strike_type: {data.get('strike_type')})")
                    
                    print(f"    Yes Bid:     {data['yes_bid']}")
                    print(f"    Yes Ask:     {data['yes_ask']}")
                    print(f"    Last Price:  {data['last_price_proxy']}")
                    print(f"    Settlement:  {data['settlement']}")
                    
                    if data['candlesticks']:
                        print(f"    Candlesticks ({len(data['candlesticks'])}):")
                        # Print first and last candlestick as a sample
                        if len(data['candlesticks']) > 0:
                            first_stick = data['candlesticks'][0]
                            last_stick = data['candlesticks'][-1]
                            print(f"      First: ts={first_stick.get('ts')}, o={first_stick.get('open')}, h={first_stick.get('high')}, l={first_stick.get('low')}, c={first_stick.get('close')}, v={first_stick.get('volume')}")
                            if len(data['candlesticks']) > 1:
                                print(f"      Last:  ts={last_stick.get('ts')}, o={last_stick.get('open')}, h={last_stick.get('high')}, l={last_stick.get('low')}, c={last_stick.get('close')}, v={last_stick.get('volume')}")
                    else:
                        print("    Candlesticks: None fetched or available for the chosen range/interval.")
                    print("-" * 20)
                
                # Example: Save all data to a JSON file
                # output_filename = f"{example_event_ticker}_data_with_candlesticks.json"
                # # Convert numpy types to native Python types for JSON serialization
                # def convert_numpy_types(obj):
                #     if isinstance(obj, np.integer): return int(obj)
                #     elif isinstance(obj, np.floating): return float(obj)
                #     elif isinstance(obj, np.ndarray): return obj.tolist()
                #     elif isinstance(obj, dict): return {k: convert_numpy_types(v) for k, v in obj.items()}
                #     elif isinstance(obj, list): return [convert_numpy_types(i) for i in obj]
                #     return obj
                # with open(output_filename, 'w') as f:
                #     json.dump(convert_numpy_types(historical_data_with_candles), f, indent=2, allow_nan=True) # allow_nan for inf
                # print(f"\nSaved all data to {output_filename}")

            else:
                print("  No markets found or processed for this event ticker.")
        else:
            print("\nFailed to load PRODUCTION historical Kalshi data (check logs/API errors).")