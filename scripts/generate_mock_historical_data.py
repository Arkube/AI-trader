#!/usr/bin/env python3
"""
Generate mock historical data for backtesting.
Populates minute_candles, tick_data, and option_chain tables with realistic data.
Run: python scripts/generate_mock_historical_data.py
"""

import sys
import random
from datetime import datetime, date, timedelta, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from database.db import write_df, upsert_candles, get_engine
from config.settings import STRIKE_GAP, ATM_RANGE
from backtest.option_resolver import get_nearest_expiry


def generate_mock_candles(symbol: str, start_date: date, end_date: date, base_price: float = 24500.0):
    """Generate mock 1-minute candles for a symbol over a date range."""
    candles = []
    
    current_date = start_date
    while current_date <= end_date:
        # Skip weekends
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue
        
        # Market hours: 9:15 to 15:30 IST
        day_start = datetime.combine(current_date, time(9, 15))
        day_end = datetime.combine(current_date, time(15, 30))
        
        # Random walk for the day
        price = base_price + random.uniform(-100, 100)
        daily_drift = random.uniform(-0.005, 0.005)  # ±0.5% daily drift
        
        current_time = day_start
        while current_time <= day_end:
            # Random walk with slight mean reversion
            change = random.gauss(daily_drift / 375, 0.0015)  # 375 minutes in trading day
            price = max(price * (1 + change), base_price * 0.95)
            price = min(price, base_price * 1.05)
            
            # Generate OHLC
            volatility = price * 0.001
            high = price + abs(random.gauss(0, volatility))
            low = price - abs(random.gauss(0, volatility))
            open_price = price + random.gauss(0, volatility * 0.5)
            close_price = price + random.gauss(0, volatility * 0.5)
            
            # Ensure OHLC consistency
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)
            
            volume = int(random.uniform(1000, 10000))
            vwap = (high + low + close_price) / 3
            oi = int(random.uniform(100000, 500000))
            
            candles.append({
                'timestamp': current_time,
                'symbol': symbol,
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close_price, 2),
                'volume': volume,
                'vwap': round(vwap, 2),
                'oi': oi
            })
            
            current_time += timedelta(minutes=1)
        
        # Update base price for next day with some continuity
        base_price = price
        current_date += timedelta(days=1)
    
    return pd.DataFrame(candles)


def generate_mock_ticks(symbol: str, start_date: date, end_date: date, base_price: float = 24500.0):
    """Generate mock tick data for a symbol over a date range."""
    ticks = []
    
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() >= 5:
            current_date += timedelta(days=1)
            continue
        
        day_start = datetime.combine(current_date, time(9, 15))
        day_end = datetime.combine(current_date, time(15, 30))
        
        price = base_price + random.uniform(-50, 50)
        
        current_time = day_start
        while current_time <= day_end:
            # Generate ~2-5 ticks per minute
            num_ticks = random.randint(2, 5)
            for _ in range(num_ticks):
                change = random.gauss(0, 0.0005)
                price = max(price * (1 + change), base_price * 0.95)
                price = min(price, base_price * 1.05)
                
                tick_time = current_time + timedelta(seconds=random.randint(0, 59))
                
                ticks.append({
                    'timestamp': tick_time,
                    'symbol': symbol,
                    'price': round(price, 2),
                    'volume': random.randint(1, 100),
                    'oi': random.randint(100000, 500000),
                    'bid_price': round(price - 0.5, 2),
                    'ask_price': round(price + 0.5, 2)
                })
            
            current_time += timedelta(minutes=1)
        
        base_price = price
        current_date += timedelta(days=1)
    
    return pd.DataFrame(ticks)


def generate_mock_option_chain(replay_date: date, nifty_price: float):
    """Generate mock option chain snapshot for a date."""
    expiry = get_nearest_expiry(replay_date)
    if not expiry:
        # Fallback: next Tuesday
        days_ahead = 1 - replay_date.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        expiry = replay_date + timedelta(days=days_ahead)
    
    exp_code = expiry.strftime('%y%m%d')
    atm = round(nifty_price / STRIKE_GAP['NIFTY']) * STRIKE_GAP['NIFTY']
    
    chain_data = []
    timestamp = datetime.combine(replay_date, time(10, 0))  # 10 AM snapshot
    
    for i in range(-ATM_RANGE, ATM_RANGE + 1):
        strike = int(atm + i * STRIKE_GAP['NIFTY'])
        for opt_type in ['CE', 'PE']:
            symbol = f'NIFTY{exp_code}{strike}{opt_type}'
            
            # Rough Black-Scholes approximation for premium
            moneyness = (nifty_price - strike) / nifty_price
            time_to_expiry = (expiry - replay_date).days / 365.0
            vol = 0.15  # 15% IV
            
            if opt_type == 'CE':
                intrinsic = max(nifty_price - strike, 0)
            else:
                intrinsic = max(strike - nifty_price, 0)
            
            time_value = nifty_price * vol * np.sqrt(time_to_expiry) * 0.4
            premium = max(intrinsic + time_value + random.uniform(-10, 10), 1)
            
            chain_data.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'underlying': 'NIFTY',
                'expiry': expiry,
                'strike': strike,
                'option_type': opt_type,
                'ltp': round(premium, 2),
                'oi': random.randint(1000, 50000),
                'iv': round(vol + random.uniform(-0.02, 0.02), 4),
                'delta': round(0.5 + (0.5 if opt_type == 'CE' else -0.5) * (1 - abs(moneyness) * 5), 4)
            })
    
    return pd.DataFrame(chain_data)


def main():
    print("=" * 60)
    print("  GENERATING MOCK HISTORICAL DATA FOR BACKTESTING")
    print("=" * 60)
    
    # Generate data for last 10 trading days
    end_date = date.today() - timedelta(days=1)  # Yesterday
    start_date = end_date - timedelta(days=20)   # ~10 trading days back
    
    print(f"Date range: {start_date} to {end_date}")
    
    # 1. Generate NIFTY-I minute candles
    print("\n1. Generating NIFTY-I minute candles...")
    nifty_candles = generate_mock_candles('NIFTY-I', start_date, end_date, 24500.0)
    print(f"   Generated {len(nifty_candles)} candles")
    # Use write_df with append (duplicates will be handled by DB constraints if any)
    write_df(nifty_candles, 'minute_candles', if_exists='append')
    print("   ✓ Written to minute_candles")
    
    # 2. Generate NIFTY-I tick data
    print("\n2. Generating NIFTY-I tick data...")
    nifty_ticks = generate_mock_ticks('NIFTY-I', start_date, end_date, 24500.0)
    print(f"   Generated {len(nifty_ticks)} ticks")
    write_df(nifty_ticks, 'tick_data')
    print("   ✓ Written to tick_data")
    
    # 3. Generate option chain snapshots for each day
    print("\n3. Generating option chain snapshots...")
    total_chain_rows = 0
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 5:
            # Get NIFTY close price for this day
            day_candles = nifty_candles[nifty_candles['timestamp'].dt.date == current_date]
            if not day_candles.empty:
                nifty_close = day_candles.iloc[-1]['close']
                chain = generate_mock_option_chain(current_date, nifty_close)
                write_df(chain, 'option_chain')
                total_chain_rows += len(chain)
                print(f"   {current_date}: {len(chain)} option contracts (NIFTY={nifty_close:.1f})")
        current_date += timedelta(days=1)
    
    print(f"\n   Total option chain rows: {total_chain_rows}")
    
    # 4. Generate option tick data for a few key strikes
    print("\n4. Generating option tick data for key strikes...")
    expiry = get_nearest_expiry(end_date)
    if not expiry:
        days_ahead = 1 - end_date.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        expiry = end_date + timedelta(days=days_ahead)
    exp_code = expiry.strftime('%y%m%d')
    atm = 24500
    
    option_symbols = []
    for i in [-2, -1, 0, 1, 2]:
        strike = atm + i * 50
        option_symbols.append(f'NIFTY{exp_code}{strike}CE')
        option_symbols.append(f'NIFTY{exp_code}{strike}PE')
    
    total_option_ticks = 0
    for symbol in option_symbols:
        # Base premium around 100-150
        base_prem = random.uniform(80, 150)
        opt_ticks = generate_mock_ticks(symbol, start_date, end_date, base_prem)
        write_df(opt_ticks, 'tick_data')
        total_option_ticks += len(opt_ticks)
    
    print(f"   Generated {total_option_ticks} option ticks for {len(option_symbols)} symbols")
    
    print("\n" + "=" * 60)
    print("  MOCK HISTORICAL DATA GENERATION COMPLETE")
    print("=" * 60)
    print("You can now run backtests:")
    print("  python scripts/tick_replay_backtest.py --risk high")
    print("  python scripts/tick_replay_backtest.py --risk medium")
    print("  python scripts/tick_replay_backtest.py --risk low")


if __name__ == '__main__':
    main()