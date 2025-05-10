# /Users/omarabul-hassan/Desktop/projects/trading/src/strategy_engine.py
"""
Defines the logic for generating trading signals based on model probabilities
and Kalshi market prices.
"""
import numpy as np
import pandas as pd
from scipy.stats import rv_continuous # For type hinting distribution object

# Assuming distribution_builder is accessible (in same directory or PYTHONPATH)
try:
    from distribution_builder import get_probability_for_price_range
except ImportError:
    print("CRITICAL ERROR: Cannot import 'distribution_builder'. Ensure it's in the correct path.")
    # Define a dummy function to allow file loading, but it will fail at runtime
    def get_probability_for_price_range(*args, **kwargs):
        raise NotImplementedError("distribution_builder not imported correctly")

DEBUG = True # Set to False for less verbose output

def dprint(*args, **kwargs):
    """Debug print function for this module."""
    if DEBUG:
        print("DEBUG strategy_engine:", *args, **kwargs)

def generate_signals(
    log_return_distribution: rv_continuous | None,
    current_sp_price: float,
    kalshi_markets_data: dict,
    buy_threshold_pct_edge: float = 5.0, # e.g., Buy if P_model > P_kalshi_ask + 5%
    sell_threshold_pct_edge: float = 5.0 # e.g., Sell if P_model < P_kalshi_bid - 5%
    ) -> list[dict]:
    """
    Generates trading signals by comparing model probabilities to Kalshi market prices.

    Args:
        log_return_distribution: A scipy.stats frozen distribution object for forecasted log returns.
        current_sp_price: The S&P 500 price used as the base for the forecast (e.g., previous close).
        kalshi_markets_data: Dict output from kalshi_data_handler like:
                             { market_ticker: {'range_low': float, 'range_high': float,
                                               'yes_bid': int | None, 'yes_ask': int | None, ...} }
        buy_threshold_pct_edge: Minimum percentage point edge required to generate a BUY YES signal
                                 (P_model - P_kalshi_ask > threshold).
        sell_threshold_pct_edge: Minimum percentage point edge required to generate a SELL YES signal
                                  (P_kalshi_bid - P_model > threshold).

    Returns:
        List[dict]: A list of trade signal dictionaries, e.g.,
                    [{'ticker': 'KXINX...', 'action': 'BUY_YES', 'model_prob': 0.65, 'market_ask': 55, 'edge': 10},
                     {'ticker': 'KXINX...', 'action': 'SELL_YES', 'model_prob': 0.10, 'market_bid': 25, 'edge': 15}]
    """
    signals = []
    if log_return_distribution is None:
        print("Warning strategy_engine: No distribution provided, cannot generate signals.")
        return signals
    if not kalshi_markets_data:
        dprint("No Kalshi market data provided.")
        return signals
    if current_sp_price <= 0:
        print("Warning strategy_engine: Invalid current_sp_price.")
        return signals

    dprint(f"Generating signals based on {len(kalshi_markets_data)} Kalshi markets.")
    dprint(f"Buy Threshold Edge: {buy_threshold_pct_edge}%, Sell Threshold Edge: {sell_threshold_pct_edge}%")

    for ticker, market_data in kalshi_markets_data.items():
        range_low = market_data.get('range_low')
        range_high = market_data.get('range_high')
        yes_bid = market_data.get('yes_bid')
        yes_ask = market_data.get('yes_ask')

        # Skip if essential data is missing for this market
        if range_low is None or range_high is None:
             dprint(f"Skipping {ticker}: Missing range data.")
             continue
        # Allow trading even if bid/ask is 0/100 initially, but check for None
        if yes_bid is None or yes_ask is None:
             dprint(f"Skipping {ticker}: Missing bid/ask data (Bid={yes_bid}, Ask={yes_ask}).")
             continue
        # Skip if market is obviously wide/illiquid (e.g., 0/100), optional check
        # if yes_bid <= 1 and yes_ask >= 99:
        #     dprint(f"Skipping {ticker}: Market likely illiquid (Bid={yes_bid}, Ask={yes_ask}).")
        #     continue

        # Calculate model probability for this specific range
        p_model = get_probability_for_price_range(
            log_return_distribution,
            current_sp_price,
            range_low, # Handles -inf correctly if np.log(-inf) is handled (it becomes -inf)
            range_high # Handles +inf correctly if np.log(+inf) is handled (it becomes +inf)
        )

        if np.isnan(p_model):
            dprint(f"Skipping {ticker}: Model probability calculation failed (NaN).")
            continue
        
        p_model_pct = p_model * 100 # Convert to percentage points (0-100)

        # Market probabilities (prices in cents)
        p_kalshi_ask_pct = float(yes_ask) # Already 0-100
        p_kalshi_bid_pct = float(yes_bid) # Already 0-100
        
        dprint(f"Market: {ticker} | Range: ({range_low:.2f}-{range_high:.2f}) | P_model: {p_model_pct:.2f}% | Bid: {p_kalshi_bid_pct:.0f}c | Ask: {p_kalshi_ask_pct:.0f}c")

        # Check for BUY YES signal
        buy_edge = p_model_pct - p_kalshi_ask_pct
        if buy_edge >= buy_threshold_pct_edge:
            signal = {
                'ticker': ticker,
                'action': 'BUY_YES',
                'model_prob_pct': p_model_pct,
                'market_ask_pct': p_kalshi_ask_pct,
                'edge_pct': buy_edge
            }
            signals.append(signal)
            dprint(f"  -> BUY YES Signal generated. Edge: {buy_edge:.2f}%")

        # Check for SELL YES signal (equivalent to BUY NO)
        sell_edge = p_kalshi_bid_pct - p_model_pct
        if sell_edge >= sell_threshold_pct_edge:
             signal = {
                'ticker': ticker,
                'action': 'SELL_YES', # Represents selling the Yes contract (or buying No)
                'model_prob_pct': p_model_pct,
                'market_bid_pct': p_kalshi_bid_pct,
                'edge_pct': sell_edge
            }
             signals.append(signal)
             dprint(f"  -> SELL YES Signal generated. Edge: {sell_edge:.2f}%")

    print(f"--- Signal Generation Complete. Generated {len(signals)} signals. ---")
    return signals

if __name__ == '__main__':
    print("--- Strategy Engine Example ---")
    
    # Create a dummy distribution (e.g., Normal)
    from scipy.stats import norm
    dummy_mean = 0.0
    dummy_vol = 0.01 # 1% daily vol
    dummy_dist = norm(loc=dummy_mean, scale=dummy_vol)
    
    # Dummy current price
    dummy_sp_price = 5000.0
    
    # Dummy Kalshi market data
    dummy_kalshi_data = {
        "MKT-RANGE1": { # Model says high prob, market ask is low -> BUY YES
            'range_low': 4975.0, 'range_high': 5025.0, 'yes_bid': 50, 'yes_ask': 55, 'settlement': None
        },
         "MKT-RANGE2": { # Model says low prob, market bid is high -> SELL YES
            'range_low': 5025.0, 'range_high': 5075.0, 'yes_bid': 40, 'yes_ask': 45, 'settlement': None
        },
         "MKT-RANGE3": { # Model agrees with market, no edge -> NO TRADE
            'range_low': 4925.0, 'range_high': 4975.0, 'yes_bid': 10, 'yes_ask': 15, 'settlement': None
        },
         "MKT-RANGE4": { # Market with missing price data
            'range_low': 5075.0, 'range_high': 5125.0, 'yes_bid': None, 'yes_ask': None, 'settlement': None
        },
         "MKT-RANGE5": { # Threshold market Low (-inf, 4925) - Model High Prob
             'range_low': -np.inf, 'range_high': 4925.0, 'yes_bid': 80, 'yes_ask': 85, 'settlement': None
         },
         "MKT-RANGE6": { # Threshold market High (5125, +inf) - Model Low Prob
             'range_low': 5125.0, 'range_high': np.inf, 'yes_bid': 5, 'yes_ask': 10, 'settlement': None
         }
    }
    
    # Set DEBUG flags for dependencies for this test run
    try:
        import distribution_builder
        distribution_builder.DEBUG = True
    except ImportError: pass # Ignore if not found, will fail later anyway
        
    DEBUG = True # Ensure this module's debug prints show
    
    # Generate signals
    example_signals = generate_signals(
        log_return_distribution=dummy_dist,
        current_sp_price=dummy_sp_price,
        kalshi_markets_data=dummy_kalshi_data,
        buy_threshold_pct_edge=5.0,
        sell_threshold_pct_edge=5.0
    )
    
    print("\n--- Generated Signals: ---")
    if example_signals:
        for sig in example_signals:
            print(sig)
    else:
        print("No signals generated.")