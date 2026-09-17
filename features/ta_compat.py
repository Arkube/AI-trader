"""
Technical Indicators - Fallback Implementation
──────────────────────────────────────────────
Pure pandas/numpy implementations of technical indicators
used when pandas_ta is not available (e.g., Python 3.14+ without numba).
"""

import numpy as np
import pandas as pd


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=length, min_periods=length).mean()
    avg_loss = loss.rolling(window=length, min_periods=length).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD - Moving Average Convergence Divergence"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return pd.DataFrame({
        'macd': macd_line,
        'histogram': histogram,
        'signal': signal_line
    })


def stochrsi(close: pd.Series, length: int = 14) -> pd.DataFrame:
    """Stochastic RSI"""
    rsi_values = rsi(close, length)
    min_rsi = rsi_values.rolling(window=length, min_periods=length).min()
    max_rsi = rsi_values.rolling(window=length, min_periods=length).max()
    
    stoch_k = 100 * (rsi_values - min_rsi) / (max_rsi - min_rsi).replace(0, np.nan)
    stoch_d = stoch_k.rolling(window=3, min_periods=3).mean()
    
    return pd.DataFrame({
        'stochrsi_k': stoch_k,
        'stochrsi_d': stoch_d
    })


def willr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Williams %R"""
    highest_high = high.rolling(window=length, min_periods=length).max()
    lowest_low = low.rolling(window=length, min_periods=length).min()
    
    wr = -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)
    return wr


def roc(close: pd.Series, length: int = 10) -> pd.Series:
    """Rate of Change"""
    return 100 * (close - close.shift(length)) / close.shift(length).replace(0, np.nan)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.DataFrame:
    """Average Directional Index"""
    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    plus_dm = pd.Series(plus_dm, index=close.index)
    minus_dm = pd.Series(minus_dm, index=close.index)
    
    # Smoothed TR and DM
    atr = tr.rolling(window=length, min_periods=length).mean()
    plus_di = 100 * (plus_dm.rolling(window=length, min_periods=length).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(window=length, min_periods=length).mean() / atr.replace(0, np.nan))
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.rolling(window=length, min_periods=length).mean()
    
    return pd.DataFrame({
        'adx': adx_val,
        'di_plus': plus_di,
        'di_minus': minus_di
    })


def cci(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 20) -> pd.Series:
    """Commodity Channel Index"""
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(window=length, min_periods=length).mean()
    mean_dev = typical_price.rolling(window=length, min_periods=length).apply(
        lambda x: np.mean(np.abs(x - np.mean(x)))
    )
    
    cci_val = (typical_price - sma_tp) / (0.015 * mean_dev).replace(0, np.nan)
    return cci_val


def ema(close: pd.Series, length: int) -> pd.Series:
    """Exponential Moving Average"""
    return close.ewm(span=length, adjust=False).mean()


def sma(close: pd.Series, length: int) -> pd.Series:
    """Simple Moving Average"""
    return close.rolling(window=length, min_periods=length).mean()


def bbands(close: pd.Series, length: int = 20, std: float = 2) -> pd.DataFrame:
    """Bollinger Bands"""
    middle = close.rolling(window=length, min_periods=length).mean()
    std_dev = close.rolling(window=length, min_periods=length).std()
    
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    
    return pd.DataFrame({
        'lower': lower,
        'middle': middle,
        'upper': upper
    })


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Average True Range"""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=length, min_periods=length).mean()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On Balance Volume"""
    direction = np.where(close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0))
    obv_val = (direction * volume).cumsum()
    return pd.Series(obv_val, index=close.index)


def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int = 14) -> pd.Series:
    """Money Flow Index"""
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
    
    pos_mf = positive_flow.rolling(window=length, min_periods=length).sum()
    neg_mf = negative_flow.rolling(window=length, min_periods=length).sum()
    
    mfi_val = 100 - (100 / (1 + pos_mf / neg_mf.replace(0, np.nan)))
    return mfi_val


# Module-level functions that match pandas_ta API
def _get_func(name: str):
    """Get function by name, matching pandas_ta API"""
    funcs = {
        'rsi': rsi,
        'macd': macd,
        'stochrsi': stochrsi,
        'willr': willr,
        'roc': roc,
        'adx': adx,
        'cci': cci,
        'ema': ema,
        'sma': sma,
        'bbands': bbands,
        'atr': atr,
        'obv': obv,
        'mfi': mfi,
    }
    return funcs.get(name)


# Create a mock ta module for compatibility
class _TACompat:
    def __getattr__(self, name):
        func = _get_func(name)
        if func is None:
            raise AttributeError(f"module 'ta_compat' has no attribute '{name}'")
        return func


ta = _TACompat()