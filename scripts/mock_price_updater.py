#!/usr/bin/env python3
"""
Auto-updating mock price generator for live dashboard testing.
Simulates realistic price movements by updating the cache file every few seconds.
Run: python scripts/mock_price_updater.py
"""

import json
import random
import time
import signal
import sys
from datetime import datetime
from pathlib import Path

# Project-local tmp folder
PROJECT_TMP = Path(__file__).resolve().parent.parent / "tmp"
CACHE_FILE = PROJECT_TMP / "td_live_prices.json"

running = True

def signal_handler(signum, frame):
    global running
    print("\nShutting down...")
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def load_cache():
    """Load existing cache or return empty dict."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_cache(cache):
    """Atomically write cache to file."""
    tmp_file = CACHE_FILE.with_suffix(".tmp")
    with open(tmp_file, 'w') as f:
        json.dump(cache, f)
    tmp_file.replace(CACHE_FILE)


def update_prices(cache):
    """Apply realistic price movements to all symbols."""
    now = datetime.now().isoformat()
    
    for symbol, data in cache.items():
        price = data.get('price', 0)
        if price <= 0:
            continue
        
        # Different volatility for different instrument types
        if symbol == 'NIFTY-I':
            # Futures: ~0.1-0.3% moves per update
            volatility = 0.002
        elif 'CE' in symbol or 'PE' in symbol:
            # Options: higher volatility, ~1-3% moves
            volatility = 0.02
        else:
            volatility = 0.01
        
        # Random walk with slight mean reversion
        change_pct = random.gauss(0, volatility)
        new_price = round(price * (1 + change_pct), 1)
        new_price = max(0.1, new_price)  # Floor at 0.1
        
        # Update bid/ask spread (typically 0.5-2% for options)
        spread_pct = 0.005 if symbol == 'NIFTY-I' else 0.01
        spread = max(0.05, new_price * spread_pct)
        
        cache[symbol] = {
            'price': new_price,
            'bid': round(new_price - spread/2, 1),
            'ask': round(new_price + spread/2, 1),
            'ts': now
        }
    
    return cache


def main():
    print("=" * 60)
    print("  MOCK PRICE UPDATER")
    print(f"  Cache file: {CACHE_FILE}")
    print(f"  Update interval: 3 seconds")
    print("  Press Ctrl+C to stop")
    print("=" * 60)
    
    # Load initial cache
    cache = load_cache()
    if not cache:
        print("No existing cache found. Run generate_mock_data.py first.")
        return
    
    print(f"Loaded {len(cache)} symbols")
    
    update_count = 0
    while running:
        try:
            cache = update_prices(cache)
            save_cache(cache)
            update_count += 1
            
            # Print status every 10 updates
            if update_count % 10 == 0:
                nifty = cache.get('NIFTY-I', {})
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Update #{update_count} | "
                      f"NIFTY-I: {nifty.get('price', 'N/A')}")
            
            time.sleep(3)
            
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)
    
    print(f"\nStopped after {update_count} updates.")


if __name__ == '__main__':
    main()