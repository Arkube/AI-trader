"""
Free NSE Data Adapter
─────────────────────
Uses nsepython (free, no credentials) for:
- Option chain snapshots (via oi_chain_builder - most reliable)
- Historical data backfill (via equity_history, index_history)
- Live quotes (polling-based, via nse_eq, nse_get_index_quote)
- Market status (via nse_marketStatus)

This supplements TrueData (which provides WebSocket streaming).
Use when TrueData is unavailable or for development without creds.

Install: pip install nsepython

Note: Free NSE APIs are unreliable - they scrape NSE website which changes frequently.
      Use TrueData for production live trading. This adapter is for:
      - Development/testing without TrueData creds
      - Historical backfill when TrueData quota exceeded
      - Option chain display on dashboard
"""

import os
import sys
import time
import logging
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List, Any
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.logger import get_logger

logger = get_logger("nse_free")


class NSEFreeAdapter:
    """
    Free NSE data adapter using nsepython.
    
    Rate limits: Be respectful - NSE blocks aggressive scraping.
    Recommended: max 1 request/second, cache responses.
    
    Known limitations:
    - Option chain often returns empty (NSE blocks scrapers)
    - Historical data sometimes fails (DNS, JSON parse errors)
    - Live quotes are polling-based, not WebSocket
    - Market hours only (9:15-15:30 IST)
    """
    
    def __init__(self, cache_dir: str = "data/cache/nse_free"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_request_time = 0
        self._min_request_interval = 1.5  # 1.5 requests/second to be safe
        
    def _rate_limit(self):
        """Enforce minimum interval between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        self._last_request_time = time.time()
    
    def _cache_path(self, key: str) -> Path:
        """Get cache file path for a key."""
        safe_key = key.replace("/", "_").replace(":", "_")
        return self.cache_dir / f"{safe_key}.json"
    
    def _get_cached(self, key: str, max_age_sec: int = 300) -> Optional[Any]:
        """Get cached data if fresh."""
        path = self._cache_path(key)
        if path.exists():
            age = time.time() - path.stat().st_mtime
            if age < max_age_sec:
                try:
                    import json
                    return json.loads(path.read_text())
                except Exception:
                    pass
        return None
    
    def _set_cached(self, key: str, data: Any):
        """Cache data to disk."""
        path = self._cache_path(key)
        try:
            import json
            path.write_text(json.dumps(data, default=str))
        except Exception as e:
            logger.debug(f"Cache write failed: {e}")
    
    # ════════════════════════════════════════════════════════════════════════
    # OPTION CHAIN (using oi_chain_builder - most reliable)
    # ════════════════════════════════════════════════════════════════════════
    
    def get_option_chain(self, symbol: str = "NIFTY", use_cache: bool = True) -> pd.DataFrame:
        """
        Get option chain for NIFTY/BANKNIFTY/FINNIFTY using oi_chain_builder.
        
        Returns DataFrame with columns:
        strike, expiry, CE_ltp, CE_oi, CE_iv, CE_delta, PE_ltp, PE_oi, PE_iv, PE_delta, etc.
        
        Note: Often returns empty due to NSE blocking scrapers.
        """
        cache_key = f"option_chain_{symbol}"
        
        if use_cache:
            cached = self._get_cached(cache_key, max_age_sec=60)  # 1 min cache
            if cached:
                return pd.DataFrame(cached)
        
        self._rate_limit()
        
        try:
            from nsepython import oi_chain_builder
            
            # Try to get current expiry first
            expiry = self._get_nearest_expiry(symbol)
            if not expiry:
                logger.warning(f"Could not determine expiry for {symbol}")
                return pd.DataFrame()
            
            # Format expiry for oi_chain_builder (DD-MM-YYYY)
            try:
                exp_date = datetime.strptime(expiry, "%Y-%m-%d")
                expiry_fmt = exp_date.strftime("%d-%m-%Y")
            except Exception:
                expiry_fmt = expiry
            
            raw = oi_chain_builder(symbol, expiry=expiry_fmt, oi_mode='full')
            
            if not raw or 'records' not in raw or 'data' not in raw['records']:
                logger.warning(f"No option chain data for {symbol} (expiry: {expiry})")
                return pd.DataFrame()
            
            records = raw['records']['data']
            expiry_dates = raw['records'].get('expiryDates', [])
            
            # Parse into flat DataFrame
            rows = []
            for rec in records:
                strike = rec.get('strikePrice')
                exp = rec.get('expiryDate')
                
                ce = rec.get('CE', {})
                pe = rec.get('PE', {})
                
                rows.append({
                    'symbol': symbol,
                    'strike': strike,
                    'expiry': exp,
                    'timestamp': datetime.now(),
                    
                    # Call data
                    'ce_ltp': ce.get('lastPrice'),
                    'ce_oi': ce.get('openInterest'),
                    'ce_iv': ce.get('impliedVolatility'),
                    'ce_delta': ce.get('delta'),
                    'ce_gamma': ce.get('gamma'),
                    'ce_theta': ce.get('theta'),
                    'ce_vega': ce.get('vega'),
                    'ce_volume': ce.get('totalTradedVolume'),
                    'ce_bid': ce.get('bidprice'),
                    'ce_ask': ce.get('askPrice'),
                    
                    # Put data
                    'pe_ltp': pe.get('lastPrice'),
                    'pe_oi': pe.get('openInterest'),
                    'pe_iv': pe.get('impliedVolatility'),
                    'pe_delta': pe.get('delta'),
                    'pe_gamma': pe.get('gamma'),
                    'pe_theta': pe.get('theta'),
                    'pe_vega': pe.get('vega'),
                    'pe_volume': pe.get('totalTradedVolume'),
                    'pe_bid': pe.get('bidprice'),
                    'pe_ask': pe.get('askPrice'),
                    
                    # Underlying
                    'underlying_value': raw['records'].get('underlyingValue'),
                })
            
            df = pd.DataFrame(rows)
            
            if use_cache and not df.empty:
                self._set_cached(cache_key, df.to_dict('records'))
            
            logger.info(f"Fetched option chain for {symbol}: {len(df)} strikes, expiry: {expiry}")
            return df
            
        except Exception as e:
            logger.error(f"Option chain fetch failed for {symbol}: {e}")
            return pd.DataFrame()
    
    def _get_nearest_expiry(self, symbol: str) -> Optional[str]:
        """Get nearest expiry date for symbol."""
        try:
            from nsepython import nse_expirydetails
            expiries = nse_expirydetails(symbol)
            if expiries and isinstance(expiries, list) and len(expiries) > 0:
                # Return first (nearest) expiry
                return expiries[0]
        except Exception as e:
            logger.debug(f"Could not fetch expiry for {symbol}: {e}")
        
        # Fallback: calculate next Thursday (weekly expiry)
        today = date.today()
        days_ahead = 3 - today.weekday()  # Thursday = 3
        if days_ahead <= 0:
            days_ahead += 7
        return (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    
    def get_atm_strike(self, symbol: str = "NIFTY") -> Optional[int]:
        """Get ATM strike for symbol."""
        chain = self.get_option_chain(symbol)
        if chain.empty:
            return None
        underlying = chain['underlying_value'].iloc[0] if 'underlying_value' in chain.columns else None
        if underlying is None:
            return None
        gap = 50 if symbol == "NIFTY" else 100
        return int(round(underlying / gap) * gap)
    
    def get_expiry_dates(self, symbol: str = "NIFTY") -> List[str]:
        """Get list of expiry dates for symbol."""
        try:
            from nsepython import nse_expirydetails
            expiries = nse_expirydetails(symbol)
            if expiries and isinstance(expiries, list):
                return [str(e) for e in expiries]
        except Exception:
            pass
        return []
    
    # ════════════════════════════════════════════════════════════════════════
    # LIVE QUOTES (Polling-based)
    # ════════════════════════════════════════════════════════════════════════
    
    def get_live_quote(self, symbol: str) -> Optional[Dict]:
        """
        Get live quote for equity/index.
        
        Note: This is polling-based (REST), not WebSocket.
        Latency ~500ms-2s. Not suitable for high-frequency trading.
        """
        cache_key = f"quote_{symbol}"
        cached = self._get_cached(cache_key, max_age_sec=5)  # 5 sec cache
        if cached:
            return cached
        
        self._rate_limit()
        
        try:
            from nsepython import nse_eq, nse_get_index_quote
            
            # Handle index vs equity
            if symbol in ["NIFTY 50", "NIFTY BANK", "NIFTY FIN SERVICE", "NIFTY"]:
                # Index quote
                quote = nse_get_index_quote(symbol)
            else:
                # Equity quote
                quote = nse_eq(symbol)
            
            if not quote or not isinstance(quote, dict) or len(quote) == 0:
                return None
            
            # Normalize to standard format (handle different key formats)
            price = (quote.get('lastPrice') or quote.get('last_price') or 
                    quote.get('LTP') or quote.get('ltp'))
            change = quote.get('change') or quote.get('CHANGE')
            pct_change = quote.get('pChange') or quote.get('p_change') or quote.get('PCHANGE')
            open_price = quote.get('open') or quote.get('OPEN')
            high = quote.get('dayHigh') or quote.get('high') or quote.get('HIGH')
            low = quote.get('dayLow') or quote.get('low') or quote.get('LOW')
            prev_close = quote.get('previousClose') or quote.get('prev_close') or quote.get('PREVCLOSE')
            volume = quote.get('totalTradedVolume') or quote.get('volume') or quote.get('VOLUME')
            
            result = {
                'symbol': symbol,
                'price': float(price) if price else None,
                'change': float(change) if change else None,
                'pct_change': float(pct_change) if pct_change else None,
                'open': float(open_price) if open_price else None,
                'high': float(high) if high else None,
                'low': float(low) if low else None,
                'prev_close': float(prev_close) if prev_close else None,
                'volume': int(volume) if volume else 0,
                'timestamp': datetime.now().isoformat(),
                'source': 'nse_free'
            }
            
            self._set_cached(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Live quote failed for {symbol}: {e}")
            return None
    
    def get_index_quotes(self) -> Dict[str, Dict]:
        """Get quotes for major indices."""
        indices = ["NIFTY 50", "NIFTY BANK", "NIFTY FIN SERVICE"]
        quotes = {}
        for idx in indices:
            q = self.get_live_quote(idx)
            if q:
                quotes[idx] = q
        return quotes
    
    # ════════════════════════════════════════════════════════════════════════
    # HISTORICAL DATA (using equity_history, index_history)
    # ═══════════════════════════════════════════════════════════════════════
    
    def get_historical_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data.
        
        Args:
            symbol: NSE symbol (e.g., "RELIANCE", "NIFTY 50")
            start_date: Start date
            end_date: End date
            interval: "1d" (daily), "1m" not supported by free API
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        cache_key = f"historical_{symbol}_{start_date}_{end_date}_{interval}"
        cached = self._get_cached(cache_key, max_age_sec=3600)  # 1 hour cache
        if cached:
            df = pd.DataFrame(cached)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        
        self._rate_limit()
        
        try:
            from nsepython import equity_history, index_history
            
            # Format dates as DD-MM-YYYY
            start_str = start_date.strftime("%d-%m-%Y")
            end_str = end_date.strftime("%d-%m-%Y")
            
            # nsepython historical data
            if symbol in ["NIFTY 50", "NIFTY BANK", "NIFTY FIN SERVICE", "NIFTY"]:
                # Index historical
                data = index_history(symbol, start_str, end_str)
            else:
                # Equity historical - needs series parameter
                data = equity_history(symbol, "EQ", start_str, end_str)
            
            if not data or not isinstance(data, dict) or 'data' not in data:
                logger.warning(f"No historical data for {symbol}")
                return pd.DataFrame()
            
            rows = []
            for row in data['data']:
                # Handle different column name formats
                ts = (row.get('CH_TIMESTAMP') or row.get('CH_DATE') or 
                      row.get('DATE') or row.get('date'))
                open_price = (row.get('CH_OPENING_PRICE') or row.get('OPEN') or 
                             row.get('open'))
                high = (row.get('CH_TRADE_HIGH_PRICE') or row.get('HIGH') or 
                       row.get('high'))
                low = (row.get('CH_TRADE_LOW_PRICE') or row.get('LOW') or 
                      row.get('low'))
                close = (row.get('CH_CLOSING_PRICE') or row.get('CLOSE') or 
                        row.get('close'))
                volume = (row.get('CH_TOT_TRADED_QTY') or row.get('VOLUME') or 
                         row.get('volume') or 0)
                
                if ts and close:
                    rows.append({
                        'timestamp': pd.to_datetime(ts),
                        'open': float(open_price) if open_price else None,
                        'high': float(high) if high else None,
                        'low': float(low) if low else None,
                        'close': float(close) if close else None,
                        'volume': int(volume) if volume else 0,
                    })
            
            df = pd.DataFrame(rows)
            if not df.empty:
                df = df.sort_values('timestamp').reset_index(drop=True)
            
            self._set_cached(cache_key, df.to_dict('records'))
            logger.info(f"Fetched {len(df)} historical bars for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Historical data failed for {symbol}: {e}")
            return pd.DataFrame()
    
    # ═══════════════════════════════════════════════════════════════════════
    # MARKET STATUS
    # ══════════════════════════════════════════════════════════════════════
    
    def get_market_status(self) -> Dict:
        """Get market status (open/closed, trading hours)."""
        cache_key = "market_status"
        cached = self._get_cached(cache_key, max_age_sec=60)
        if cached:
            return cached
        
        self._rate_limit()
        
        try:
            from nsepython import nse_marketStatus
            status = nse_marketStatus()
            
            # Parse the complex status structure
            market_state = "UNKNOWN"
            is_open = False
            trade_date = None
            
            if isinstance(status, dict) and 'marketState' in status:
                for mkt in status['marketState']:
                    if mkt.get('market') == 'Capital Market':
                        market_state = mkt.get('marketStatus', 'UNKNOWN')
                        is_open = market_state.upper() == 'OPEN'
                        trade_date = mkt.get('tradeDate')
                        break
            
            result = {
                'market_state': market_state,
                'trade_date': trade_date,
                'index': 'NIFTY 50',
                'timestamp': datetime.now().isoformat(),
                'is_open': is_open
            }
            
            self._set_cached(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Market status failed: {e}")
            return {'market_state': 'UNKNOWN', 'is_open': False}
    
    # ═════════════════════════════════════════════════════════════════════
    # UTILITY: Convert to TrueData-compatible format
    # ═══════════════════════════════════════════════════════════════════
    
    def to_truedata_format(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Convert NSE free data format to match TrueData schema.
        Useful for backtesting with unified data format.
        """
        if df.empty:
            return df
        
        # Ensure required columns
        required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                df[col] = np.nan
        
        df['symbol'] = symbol
        df = df[required + ['symbol']].copy()
        df = df.dropna(subset=['timestamp', 'close'])
        return df
    
    def get_option_chain_for_backtest(
        self,
        symbol: str = "NIFTY",
        expiry: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get option chain formatted for backtesting.
        Returns one row per strike with CE/PE data.
        """
        chain = self.get_option_chain(symbol)
        if chain.empty:
            return pd.DataFrame()
        
        if expiry:
            chain = chain[chain['expiry'] == expiry]
        
        # Select relevant columns for backtesting
        cols = [
            'symbol', 'strike', 'expiry', 'timestamp',
            'ce_ltp', 'ce_oi', 'ce_iv', 'ce_delta', 'ce_volume',
            'pe_ltp', 'pe_oi', 'pe_iv', 'pe_delta', 'pe_volume',
            'underlying_value'
        ]
        available = [c for c in cols if c in chain.columns]
        return chain[available].copy()


# ════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def get_nse_free_adapter() -> NSEFreeAdapter:
    """Get singleton instance."""
    if not hasattr(get_nse_free_adapter, '_instance'):
        get_nse_free_adapter._instance = NSEFreeAdapter()
    return get_nse_free_adapter._instance


# ═════════════════════════════════════════════════════════════════
# TEST / DEMO
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test NSE Free Adapter")
    parser.add_argument("--option-chain", action="store_true", help="Test option chain")
    parser.add_argument("--quote", action="store_true", help="Test live quote")
    parser.add_argument("--historical", action="store_true", help="Test historical data")
    parser.add_argument("--status", action="store_true", help="Test market status")
    parser.add_argument("--symbol", default="NIFTY", help="Symbol to test")
    args = parser.parse_args()
    
    adapter = NSEFreeAdapter()
    
    if args.option_chain:
        print(f"\n=== Option Chain for {args.symbol} ===")
        df = adapter.get_option_chain(args.symbol)
        print(f"Rows: {len(df)}")
        if not df.empty:
            print(df[['strike', 'expiry', 'ce_ltp', 'ce_oi', 'pe_ltp', 'pe_oi']].head(10))
            print(f"\nExpiries: {adapter.get_expiry_dates(args.symbol)}")
            print(f"ATM Strike: {adapter.get_atm_strike(args.symbol)}")
    
    if args.quote:
        print(f"\n=== Live Quote for {args.symbol} ===")
        quote = adapter.get_live_quote(args.symbol)
        print(quote)
    
    if args.historical:
        print(f"\n=== Historical Data for {args.symbol} ===")
        end = date.today()
        start = end - timedelta(days=30)
        df = adapter.get_historical_data(args.symbol, start, end)
        print(f"Rows: {len(df)}")
        if not df.empty:
            print(df.tail())
    
    if args.status:
        print("\n=== Market Status ===")
        status = adapter.get_market_status()
        print(status)
    
    if not any([args.option_chain, args.quote, args.historical, args.status]):
        # Run all tests
        print("Running all tests...")
        print("\n1. Market Status:")
        print(adapter.get_market_status())
        
        print("\n2. Index Quotes:")
        print(adapter.get_index_quotes())
        
        print("\n3. Option Chain (NIFTY):")
        df = adapter.get_option_chain("NIFTY")
        print(f"   {len(df)} strikes, expiries: {adapter.get_expiry_dates('NIFTY')[:3]}...")
        
        print("\n4. Historical (last 10 days):")
        end = date.today()
        start = end - timedelta(days=10)
        hist = adapter.get_historical_data("NIFTY", start, end)
        print(f"   {len(hist)} bars")