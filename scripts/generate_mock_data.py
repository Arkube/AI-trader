#!/usr/bin/env python3
"""
Generate mock live price data for today for testing the dashboard.
Run: python scripts/generate_mock_data.py
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

def generate_mock_data():
    now = datetime.now()
    today = now.date()

    # NIFTY spot around 24500
    nifty_base = 24500.0

    # Next Tuesday expiry (NIFTY weekly)
    expiry_date = today + timedelta(days=(1 - today.weekday()) % 7)
    if expiry_date <= today:
        expiry_date += timedelta(days=7)
    exp_code = expiry_date.strftime('%y%m%d')

    atm = round(nifty_base / 50) * 50  # 24500

    cache = {}

    # NIFTY-I (futures)
    cache['NIFTY-I'] = {
        'price': round(nifty_base + random.uniform(-20, 20), 1),
        'bid': round(nifty_base - 0.5, 1),
        'ask': round(nifty_base + 0.5, 1),
        'ts': now.isoformat()
    }

    # Generate ATM ± 3 strikes (7 strikes each CE/PE)
    for i in range(-3, 4):
        strike = atm + i * 50
        for opt_type in ['CE', 'PE']:
            symbol = f'NIFTY{exp_code}{strike}{opt_type}'

            # Rough option pricing
            moneyness = (nifty_base - strike) / nifty_base
            base_prem = max(5, 120 - abs(moneyness) * 2000)

            price = round(base_prem + random.uniform(-5, 5), 1)
            spread = max(0.5, price * 0.01)

            cache[symbol] = {
                'price': price,
                'bid': round(price - spread/2, 1),
                'ask': round(price + spread/2, 1),
                'ts': now.isoformat()
            }

    # Write to cache file (project-local tmp folder)
    from pathlib import Path
    PROJECT_TMP = Path(__file__).resolve().parent.parent / "tmp"
    PROJECT_TMP.mkdir(exist_ok=True)
    cache_file = PROJECT_TMP / "td_live_prices.json"
    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2)

    print(f'Generated mock data for {len(cache)} symbols')
    print(f'NIFTY-I: {cache["NIFTY-I"]["price"]}')
    print(f'ATM: {atm}, Expiry: {expiry_date} ({exp_code})')
    print('Sample options:')
    for k, v in list(cache.items())[1:6]:
        print(f'  {k}: Rs.{v["price"]} (bid:{v["bid"]} ask:{v["ask"]})')
    print(f'\nCache written to: {cache_file}')

    return cache

if __name__ == '__main__':
    generate_mock_data()