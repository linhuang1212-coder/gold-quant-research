"""
Backtest Runner — Data Loading & Experiment Execution
======================================================
Eliminates the ~20 copies of "load data → prepare indicators → reset state → run"
scattered across experiment scripts.
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import research_config as config
from indicators import prepare_indicators, get_orb_strategy, calc_rsi, calc_adx
import indicators as signals_mod
from backtest.engine import BacktestEngine, TradeRecord
from backtest.stats import calc_stats


# ═══════════════════════════════════════════════════════════════
# Default data paths
# ═══════════════════════════════════════════════════════════════

_M15_CANDIDATES = [
    Path("data/download/xauusd-m15-bid-2015-01-01-2026-05-06.csv"),
    Path("data/download/xauusd-m15-bid-2015-01-01-2026-05-05.csv"),
    Path("data/download/xauusd-m15-bid-2015-01-01-2026-04-27.csv"),
    Path("data/download/xauusd-m15-bid-2015-01-01-2026-04-10.csv"),
    Path("data/download/xauusd-m15-bid-2015-01-01-2026-03-25.csv"),
]
M15_CSV_PATH = next((p for p in _M15_CANDIDATES if p.exists()), _M15_CANDIDATES[0])

_H1_CANDIDATES = [
    Path("data/download/xauusd-h1-bid-2015-01-01-2026-05-06.csv"),
    Path("data/download/xauusd-h1-bid-2015-01-01-2026-05-05.csv"),
    Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-27.csv"),
    Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-10.csv"),
    Path("data/download/xauusd-h1-bid-2015-01-01-2026-03-25.csv"),
]
H1_CSV_PATH = next((p for p in _H1_CANDIDATES if p.exists()), _H1_CANDIDATES[0])

M15_ASK_PATH = Path("data/download/xauusd-m15-ask-2015-01-01-2026-04-27.csv")
H1_ASK_PATH = Path("data/download/xauusd-h1-ask-2015-01-01-2026-04-27.csv")
M15_SPREAD_PATH = Path("data/download/xauusd-m15-spread-2015-01-01-2026-04-27.csv")
H1_SPREAD_PATH = Path("data/download/xauusd-h1-spread-2015-01-01-2026-04-27.csv")


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def load_csv(path: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """Load OHLCV CSV with timestamp(ms) format."""
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'}, inplace=True)

    if 'Volume' not in df.columns:
        df['Volume'] = 0

    df['is_flat'] = (df['Open'] == df['High']) & (df['High'] == df['Low']) & (df['Low'] == df['Close'])

    if start:
        df = df[df.index >= pd.Timestamp(start, tz='UTC')]
    if end:
        df = df[df.index <= pd.Timestamp(end, tz='UTC')]

    return df


def load_m15(csv_path: Path = M15_CSV_PATH) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"M15 CSV not found: {csv_path}")
    df = load_csv(str(csv_path))
    print(f"  M15: {csv_path} ({len(df)} bars, {df.index[0]} -> {df.index[-1]})")
    return df


def load_spread_series(spread_path: Path = M15_SPREAD_PATH) -> Optional[pd.Series]:
    """Load historical spread data as a Series indexed by timestamp(ms)."""
    if not spread_path.exists():
        print(f"  Spread file not found: {spread_path}, using fixed spread")
        return None
    df = pd.read_csv(spread_path)
    series = pd.Series(df['spread_avg'].values, index=df['timestamp'].values)
    series = series.sort_index()
    print(f"  Spread: {len(series)} bars, mean=${series.mean():.4f}, median=${series.median():.4f}")
    return series


def load_h1_aligned(h1_path: Path, m15_start: pd.Timestamp) -> pd.DataFrame:
    df = load_csv(str(h1_path))
    warmup_start = m15_start - pd.Timedelta(hours=200)
    df = df[df.index >= warmup_start]
    print(f"  H1: {len(df)} bars, {df.index[0]} -> {df.index[-1]}")
    return df


_D1_SPOT_CACHE: Dict[str, pd.DataFrame] = {}


def load_d1_spot(
    start: Optional[str] = None,
    end: Optional[str] = None,
    *,
    session_close_utc: int = 0,
    m15_path: Optional[Path] = None,
    tz_naive: bool = False,
) -> pd.DataFrame:
    """Spot XAUUSD D1 OHLCV by resampling HistData M15 bid bars.

    Use this instead of ``data/xauusd_daily_yf.csv`` (which is COMEX GC=F futures)
    for any D1-level trend / EMA-alignment / regime filter that interacts with
    M15/H1 spot signals.  Eliminates the 5-10 USD futures-vs-spot basis that
    silently biases D1 EMA20 / SMA200 filters.

    Args:
        start, end: ISO date strings (e.g. "2015-01-01"); inclusive.
        session_close_utc: UTC hour anchoring the daily bar close.
            * 0  = UTC midnight (default; natural for 24h forex)
            * 21 = NY close (matches CME RTH boundary, useful for replicating
                   futures-style "overnight" semantics on spot data)
        m15_path: override M15 CSV (defaults to ``M15_CSV_PATH``).
        tz_naive: if True, drop timezone from the result index (matches the
            shape produced by ``pd.read_csv(parse_dates=True, index_col='Date')``
            on the old ``xauusd_daily_yf.csv``).

    Returns:
        DataFrame with columns ``Open, High, Low, Close, Volume`` indexed by
        the daily bar close timestamp (UTC unless ``tz_naive``).
    """
    path = m15_path or M15_CSV_PATH
    if not path.exists():
        raise FileNotFoundError(f"M15 CSV not found for D1 resample: {path}")

    cache_key = f"{path}|{session_close_utc}"
    if cache_key not in _D1_SPOT_CACHE:
        m15 = load_csv(str(path))
        if session_close_utc == 0:
            d1 = m15.resample('1D', label='left', closed='left').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
                'Volume': 'sum',
            })
        else:
            offset = pd.Timedelta(hours=session_close_utc)
            shifted = m15.copy()
            shifted.index = shifted.index - offset
            d1 = shifted.resample('1D', label='left', closed='left').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
                'Volume': 'sum',
            })
            d1.index = d1.index + offset
        d1 = d1.dropna(subset=['Close'])
        _D1_SPOT_CACHE[cache_key] = d1
        print(f"  D1 spot (M15 resampled, session_close={session_close_utc:02d}:00 UTC): "
              f"{len(d1)} bars, {d1.index[0].date()} -> {d1.index[-1].date()}")

    df = _D1_SPOT_CACHE[cache_key].copy()
    if start:
        df = df[df.index >= pd.Timestamp(start, tz='UTC')]
    if end:
        df = df[df.index <= pd.Timestamp(end, tz='UTC')]
    if tz_naive:
        df.index = df.index.tz_localize(None)
    return df


def check_data_gaps(df: pd.DataFrame, label: str, expected_freq: str):
    """Detect and report gaps in time series data.

    Args:
        expected_freq: 'M15' (15min) or 'H1' (60min)
    """
    if len(df) < 2:
        return
    freq_minutes = 15 if expected_freq == 'M15' else 60
    diffs = df.index.to_series().diff().dropna()
    non_flat = df[~df.get('is_flat', False).astype(bool)] if 'is_flat' in df.columns else df
    if len(non_flat) < 2:
        return
    diffs_nf = non_flat.index.to_series().diff().dropna()
    max_gap = diffs_nf.max()
    threshold = pd.Timedelta(hours=72)
    large_gaps = diffs_nf[diffs_nf > threshold]
    if len(large_gaps) > 0:
        print(f"  [WARN] {label}: {len(large_gaps)} gaps > 72h detected (max: {max_gap})")
        for ts, gap in large_gaps.head(5).items():
            print(f"       {ts - gap} -> {ts}  ({gap})")
        if len(large_gaps) > 5:
            print(f"       ... and {len(large_gaps) - 5} more")
    else:
        total_bars = len(non_flat)
        date_range = (non_flat.index[-1] - non_flat.index[0]).total_seconds() / 60
        expected_bars = date_range / freq_minutes
        coverage = total_bars / expected_bars if expected_bars > 0 else 1
        if coverage < 0.90:
            print(f"  [WARN] {label}: coverage={coverage:.1%} (expected ~{expected_bars:.0f} bars, got {total_bars})")


def add_atr_percentile(h1_df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling ATR percentile column if not present."""
    if 'atr_percentile' not in h1_df.columns:
        h1_df['atr_percentile'] = h1_df['ATR'].rolling(500, min_periods=50).rank(pct=True)
        h1_df['atr_percentile'] = h1_df['atr_percentile'].fillna(0.5)
    return h1_df


def _wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1, dtype=float)
    return series.rolling(period).apply(
        lambda x: np.dot(x, weights[-len(x):]) / weights[-len(x):].sum(), raw=True
    )


def _hma(series: pd.Series, period: int) -> pd.Series:
    half_period = max(2, period // 2)
    sqrt_period = max(2, int(np.sqrt(period)))
    wma_half = _wma(series, half_period)
    wma_full = _wma(series, period)
    diff = 2 * wma_half - wma_full
    return _wma(diff, sqrt_period)


def _kama(series: pd.Series, period: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    close = series.values.astype(float)
    result = np.full_like(close, np.nan)
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)

    if len(close) <= period:
        return pd.Series(result, index=series.index)

    result[period - 1] = close[period - 1]
    for i in range(period, len(close)):
        direction = abs(close[i] - close[i - period])
        volatility = sum(abs(close[j] - close[j - 1]) for j in range(i - period + 1, i + 1))
        if volatility == 0:
            er = 0
        else:
            er = direction / volatility
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        if np.isnan(result[i - 1]):
            result[i] = close[i]
        else:
            result[i] = result[i - 1] + sc * (close[i] - result[i - 1])
    return pd.Series(result, index=series.index)


def prepare_indicators_custom(
    df: pd.DataFrame, kc_ema=25, kc_mult=1.2, ema_trend=100, kc_ma_type: str = "ema"
) -> pd.DataFrame:
    """Recalculate indicators with custom Keltner/EMA parameters."""
    import numpy as _np
    df = df.copy()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['EMA100'] = df['Close'].ewm(span=ema_trend).mean()
    df['EMA9'] = df['Close'].ewm(span=9).mean()
    df['EMA12'] = df['Close'].ewm(span=12).mean()
    df['EMA21'] = df['Close'].ewm(span=21).mean()
    df['EMA26'] = df['Close'].ewm(span=26).mean()
    df['ATR'] = (df['High'] - df['Low']).rolling(14).mean()
    if kc_ma_type == "hma":
        df['KC_mid'] = _hma(df['Close'], int(kc_ema))
    elif kc_ma_type == "kama":
        df['KC_mid'] = _kama(df['Close'], int(kc_ema))
    else:
        df['KC_mid'] = df['Close'].ewm(span=kc_ema).mean()
    df['KC_upper'] = df['KC_mid'] + kc_mult * df['ATR']
    df['KC_lower'] = df['KC_mid'] - kc_mult * df['ATR']
    _bb_std = df['Close'].rolling(20).std()
    _bb_mid = df['Close'].rolling(20).mean()
    df['BB_upper'] = _bb_mid + 2 * _bb_std
    df['BB_lower'] = _bb_mid - 2 * _bb_std
    df['squeeze'] = ((df['BB_upper'] < df['KC_upper']) & (df['BB_lower'] > df['KC_lower'])).astype(float)
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    df['RSI2'] = calc_rsi(df['Close'], 2)
    df['RSI14'] = calc_rsi(df['Close'], 14)
    df['ADX'] = calc_adx(df, 14)
    df['Vol_MA20'] = df['Volume'].rolling(20).mean()

    # Pinbar detection
    body = (df['Close'] - df['Open']).abs()
    full_range = df['High'] - df['Low']
    upper_wick = df['High'] - df[['Close', 'Open']].max(axis=1)
    lower_wick = df[['Close', 'Open']].min(axis=1) - df['Low']
    body_ratio = body / full_range.replace(0, _np.nan)
    df['pinbar_bull'] = ((lower_wick > body * 2) & (lower_wick > upper_wick * 1.5)
                         & (body_ratio < 0.35)).astype(float)
    df['pinbar_bear'] = ((upper_wick > body * 2) & (upper_wick > lower_wick * 1.5)
                         & (body_ratio < 0.35)).astype(float)

    # Swing High/Low support/resistance (vectorized for speed)
    _swing_n = 10
    _win = 2 * _swing_n + 1
    _roll_max = df['High'].rolling(_win, center=True).max()
    df['swing_high'] = df['High'].where(df['High'] == _roll_max, _np.nan)
    _roll_min = df['Low'].rolling(_win, center=True).min()
    df['swing_low'] = df['Low'].where(df['Low'] == _roll_min, _np.nan)
    df['nearest_resistance'] = df['swing_high'].ffill()
    df['nearest_support'] = df['swing_low'].ffill()
    df['dist_to_resistance'] = (df['nearest_resistance'] - df['Close']) / df['ATR'].replace(0, _np.nan)
    df['dist_to_support'] = (df['Close'] - df['nearest_support']) / df['ATR'].replace(0, _np.nan)

    # Top/Bottom Fractal
    h1_, h2_, h3_ = df['High'].shift(2), df['High'].shift(1), df['High']
    l1_, l2_, l3_ = df['Low'].shift(2), df['Low'].shift(1), df['Low']
    c3_ = df['Close']
    o3_ = df['Open']
    df['top_fractal'] = ((h2_ > h1_) & (h2_ > h3_) & (c3_ < o3_)).astype(float)
    df['bot_fractal'] = ((l2_ < l1_) & (l2_ < l3_) & (c3_ > o3_)).astype(float)

    # Inside Bar — 3rd bar confirms direction
    prev_h_ = df['High'].shift(1)
    prev_l_ = df['Low'].shift(1)
    pprev_h_ = df['High'].shift(2)
    pprev_l_ = df['Low'].shift(2)
    is_inside_ = (prev_h_ <= pprev_h_) & (prev_l_ >= pprev_l_)
    df['inside_bar_bull'] = (is_inside_ & (df['Close'] > pprev_h_)).astype(float)
    df['inside_bar_bear'] = (is_inside_ & (df['Close'] < pprev_l_)).astype(float)

    # 2B / Engulfing
    prev_body_top_ = df[['Close', 'Open']].shift(1).max(axis=1)
    prev_body_bot_ = df[['Close', 'Open']].shift(1).min(axis=1)
    cur_body_top_ = df[['Close', 'Open']].max(axis=1)
    cur_body_bot_ = df[['Close', 'Open']].min(axis=1)
    df['engulf_bull'] = ((df['Close'] > df['Open'])
                         & (df['Open'].shift(1) > df['Close'].shift(1))
                         & (cur_body_top_ >= prev_body_top_)
                         & (cur_body_bot_ <= prev_body_bot_)).astype(float)
    df['engulf_bear'] = ((df['Close'] < df['Open'])
                         & (df['Open'].shift(1) < df['Close'].shift(1))
                         & (cur_body_top_ >= prev_body_top_)
                         & (cur_body_bot_ <= prev_body_bot_)).astype(float)

    # Daily range from open
    _date_ = df.index.date
    _date_series_ = pd.Series(_date_, index=df.index)
    _daily_open_ = df.groupby(_date_series_)['Open'].transform('first')
    df['daily_range_up'] = df['High'] - _daily_open_
    df['daily_range_down'] = _daily_open_ - df['Low']
    df['daily_max_range'] = df[['daily_range_up', 'daily_range_down']].max(axis=1)

    # PA confluence count
    df['pa_bull_count'] = (df['pinbar_bull'] + df['bot_fractal']
                           + df['inside_bar_bull'] + df['engulf_bull'])
    df['pa_bear_count'] = (df['pinbar_bear'] + df['top_fractal']
                           + df['inside_bar_bear'] + df['engulf_bear'])

    return df


def add_dual_kc(
    h1_df: pd.DataFrame,
    fast_ema: int = 15,
    fast_mult: float = 0.8,
    slow_ema: int = 35,
    slow_mult: float = 1.6,
) -> pd.DataFrame:
    """Add secondary (slow) KC channels for dual-KC mode."""
    h1_df = h1_df.copy()
    atr = h1_df['ATR'] if 'ATR' in h1_df.columns else (h1_df['High'] - h1_df['Low']).rolling(14).mean()

    slow_mid = h1_df['Close'].ewm(span=slow_ema).mean()
    h1_df['KC_slow_mid'] = slow_mid
    h1_df['KC_slow_upper'] = slow_mid + slow_mult * atr
    h1_df['KC_slow_lower'] = slow_mid - slow_mult * atr

    fast_mid = h1_df['Close'].ewm(span=fast_ema).mean()
    h1_df['KC_fast_mid'] = fast_mid
    h1_df['KC_fast_upper'] = fast_mid + fast_mult * atr
    h1_df['KC_fast_lower'] = fast_mid - fast_mult * atr

    return h1_df


# ═══════════════════════════════════════════════════════════════
# Data bundle (load once, reuse across variants)
# ═══════════════════════════════════════════════════════════════

class DataBundle:
    """Pre-loaded and indicator-prepared data, ready for engine instantiation."""

    def __init__(self, m15_df: pd.DataFrame, h1_df: pd.DataFrame):
        self.m15_df = m15_df
        self.h1_df = h1_df

    @classmethod
    def load_default(cls, start: str = "2015-01-01", end: Optional[str] = None) -> 'DataBundle':
        """Load with default indicators (prepare_indicators)."""
        print("\nLoading data...")
        m15_raw = load_m15()
        if start:
            m15_raw = m15_raw[m15_raw.index >= pd.Timestamp(start, tz='UTC')]
        if end:
            m15_raw = m15_raw[m15_raw.index <= pd.Timestamp(end, tz='UTC')]

        h1_raw = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])

        print("Preparing indicators...")
        print("  M15...", end='', flush=True)
        m15_df = prepare_indicators(m15_raw)
        print(" done")
        print("  H1...", end='', flush=True)
        h1_df = prepare_indicators(h1_raw)
        print(" done")
        h1_df = add_atr_percentile(h1_df)

        print(f"  M15: {len(m15_df)} bars, H1: {len(h1_df)} bars")
        check_data_gaps(m15_df, "M15", "M15")
        check_data_gaps(h1_df, "H1", "H1")
        return cls(m15_df, h1_df)

    @classmethod
    def load_custom(cls, kc_ema=25, kc_mult=1.2, ema_trend=100,
                    start: str = "2015-01-01", end: Optional[str] = None,
                    kc_ma_type: str = "ema") -> 'DataBundle':
        """Load with custom indicator parameters."""
        print("\nLoading data...")
        m15_raw = load_m15()
        if start:
            m15_raw = m15_raw[m15_raw.index >= pd.Timestamp(start, tz='UTC')]
        if end:
            m15_raw = m15_raw[m15_raw.index <= pd.Timestamp(end, tz='UTC')]

        h1_raw = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])

        print(f"Preparing custom indicators (KC_ema={kc_ema}, KC_mult={kc_mult}, EMA={ema_trend}, MA={kc_ma_type})...")
        m15_df = prepare_indicators_custom(m15_raw, kc_ema=kc_ema, kc_mult=kc_mult, ema_trend=ema_trend, kc_ma_type=kc_ma_type)
        h1_df = prepare_indicators_custom(h1_raw, kc_ema=kc_ema, kc_mult=kc_mult, ema_trend=ema_trend, kc_ma_type=kc_ma_type)
        h1_df = add_atr_percentile(h1_df)

        print(f"  M15: {len(m15_df)} bars, H1: {len(h1_df)} bars")
        check_data_gaps(m15_df, "M15", "M15")
        check_data_gaps(h1_df, "H1", "H1")
        return cls(m15_df, h1_df)

    @classmethod
    def load_raw(cls, start: str = "2015-01-01", end: Optional[str] = None) -> 'DataBundle':
        """Load raw data without indicators (for custom prep later)."""
        print("\nLoading raw data...")
        m15_raw = load_m15()
        if start:
            m15_raw = m15_raw[m15_raw.index >= pd.Timestamp(start, tz='UTC')]
        if end:
            m15_raw = m15_raw[m15_raw.index <= pd.Timestamp(end, tz='UTC')]
        h1_raw = load_h1_aligned(H1_CSV_PATH, m15_raw.index[0])
        print(f"  M15: {len(m15_raw)} bars, H1: {len(h1_raw)} bars")
        return cls(m15_raw, h1_raw)

    def slice(self, start: str, end: str) -> 'DataBundle':
        """Return a time-windowed subset."""
        ts = pd.Timestamp(start, tz='UTC')
        te = pd.Timestamp(end, tz='UTC')
        m15 = self.m15_df[(self.m15_df.index >= ts) & (self.m15_df.index < te)]
        h1 = self.h1_df[(self.h1_df.index >= ts) & (self.h1_df.index < te)]
        return DataBundle(m15, h1)


# ═══════════════════════════════════════════════════════════════
# Run helpers
# ═══════════════════════════════════════════════════════════════

def run_variant(data: DataBundle, label: str, *, verbose: bool = True, **engine_kwargs) -> Dict:
    """Run a single backtest variant and return stats dict.

    Handles global state reset automatically.
    """
    if verbose:
        print(f"\n  [{label}]", flush=True)

    get_orb_strategy().reset_daily()
    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False

    t0 = time.time()
    engine = BacktestEngine(data.m15_df, data.h1_df, label=label, **engine_kwargs)
    trades = engine.run()
    elapsed = time.time() - t0

    stats = calc_stats(trades, engine.equity_curve)
    stats['label'] = label
    stats['rsi_filtered'] = engine.rsi_filtered_count
    stats['rsi_total'] = engine.rsi_total_signals
    stats['h1_entries'] = engine.h1_entry_count
    stats['m15_entries'] = engine.m15_entry_count
    stats['skipped_choppy'] = engine.skipped_choppy
    stats['skipped_neutral_m15'] = engine.skipped_neutral_m15
    stats['skipped_ema_slope'] = getattr(engine, 'skipped_ema_slope', 0)
    stats['atr_spike_tightens'] = getattr(engine, 'atr_spike_tighten_count', 0)
    stats['skipped_kc_bw'] = getattr(engine, 'skipped_kc_bw', 0)
    stats['skipped_session'] = getattr(engine, 'skipped_session', 0)
    stats['time_decay_tp'] = getattr(engine, 'time_decay_tp_count', 0)
    stats['skipped_min_bars'] = getattr(engine, 'skipped_min_bars', 0)
    stats['skipped_adx_gray'] = getattr(engine, 'skipped_adx_gray', 0)
    stats['escalated_cooldowns'] = getattr(engine, 'escalated_cooldowns', 0)
    stats['breakeven_triggered'] = getattr(engine, 'breakeven_triggered', 0)
    stats['skipped_pinbar'] = getattr(engine, 'skipped_pinbar', 0)
    stats['skipped_sr'] = getattr(engine, 'skipped_sr', 0)
    stats['pinbar_sr_entries'] = getattr(engine, 'pinbar_sr_entries', 0)
    stats['skipped_fractal'] = getattr(engine, 'skipped_fractal', 0)
    stats['skipped_inside_bar'] = getattr(engine, 'skipped_inside_bar', 0)
    stats['skipped_engulf'] = getattr(engine, 'skipped_engulf', 0)
    stats['skipped_pa_confluence'] = getattr(engine, 'skipped_pa_confluence', 0)
    stats['skipped_daily_range'] = getattr(engine, 'skipped_daily_range', 0)
    stats['fractal_sr_entries'] = getattr(engine, 'fractal_sr_entries', 0)
    stats['inside_bar_sr_entries'] = getattr(engine, 'inside_bar_sr_entries', 0)
    stats['engulf_sr_entries'] = getattr(engine, 'engulf_sr_entries', 0)
    stats['skipped_squeeze'] = getattr(engine, 'skipped_squeeze', 0)
    stats['skipped_consecutive'] = getattr(engine, 'skipped_consecutive', 0)
    stats['partial_tp_count'] = getattr(engine, 'partial_tp_count', 0)
    stats['profit_dd_exit_count'] = getattr(engine, 'profit_dd_exit_count', 0)
    stats['adaptive_hold_triggered'] = getattr(engine, 'adaptive_hold_triggered', 0)
    stats['session_entry_counts'] = getattr(engine, 'session_entry_counts', {})
    # R16 counters
    stats['timeout_profit_lock'] = getattr(engine, 'timeout_profit_lock_count', 0)
    stats['timeout_adverse_exit'] = getattr(engine, 'timeout_adverse_exit_count', 0)
    stats['timeout_momentum_exit'] = getattr(engine, 'timeout_momentum_exit_count', 0)
    stats['timeout_dynamic_extend'] = getattr(engine, 'timeout_dynamic_extend_count', 0)
    stats['timeout_dynamic_cut'] = getattr(engine, 'timeout_dynamic_cut_count', 0)
    # R17 counters
    stats['dd_pause_count'] = getattr(engine, 'dd_pause_count', 0)
    stats['dd_reduce_count'] = getattr(engine, 'dd_reduce_count', 0)
    stats['equity_filter_skip'] = getattr(engine, 'equity_filter_skip_count', 0)
    stats['maxloss_cap_count'] = getattr(engine, 'maxloss_cap_count', 0)
    stats['final_capital'] = getattr(engine, '_current_capital', 0)
    stats['equity_peak'] = getattr(engine, '_equity_peak', 0)
    stats['skipped_no_signal'] = getattr(engine, 'skipped_no_signal', 0)
    stats['elapsed_s'] = round(elapsed, 1)
    stats['_trades'] = trades
    stats['_equity_curve'] = engine.equity_curve

    if verbose:
        print(f"    {stats['n']} trades (H1={engine.h1_entry_count}, M15={engine.m15_entry_count}), "
              f"Sharpe={stats['sharpe']:.2f}, PnL=${stats['total_pnl']:.0f}, {elapsed:.0f}s")

    return stats


def _worker_run_variant(args):
    """Subprocess worker: run a single variant in isolation."""
    m15_df, h1_df, label, engine_kwargs = args
    get_orb_strategy().reset_daily()
    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False

    t0 = time.time()
    engine = BacktestEngine(m15_df, h1_df, label=label, **engine_kwargs)
    trades = engine.run()
    elapsed = time.time() - t0

    stats = calc_stats(trades, engine.equity_curve)
    stats['label'] = label
    stats['elapsed_s'] = round(elapsed, 1)
    stats['n_trades_raw'] = len(trades)
    for attr in ['rsi_filtered_count', 'rsi_total_signals', 'h1_entry_count',
                 'm15_entry_count', 'skipped_choppy', 'skipped_neutral_m15',
                 'skipped_ema_slope', 'atr_spike_tighten_count', 'skipped_kc_bw',
                 'skipped_session', 'time_decay_tp_count', 'skipped_min_bars',
                 'skipped_adx_gray', 'escalated_cooldowns', 'breakeven_triggered',
                 'skipped_pinbar', 'skipped_sr', 'pinbar_sr_entries',
                 'skipped_fractal', 'skipped_inside_bar', 'skipped_engulf',
                 'skipped_pa_confluence', 'skipped_daily_range',
                 'fractal_sr_entries', 'inside_bar_sr_entries', 'engulf_sr_entries',
                 'skipped_squeeze', 'skipped_consecutive',
                 'partial_tp_count', 'profit_dd_exit_count', 'adaptive_hold_triggered',
                 'timeout_profit_lock_count', 'timeout_adverse_exit_count',
                 'timeout_momentum_exit_count', 'timeout_dynamic_extend_count',
                 'timeout_dynamic_cut_count', 'dd_pause_count', 'dd_reduce_count',
                 'equity_filter_skip_count']:
        stats[attr.replace('_count', '')] = getattr(engine, attr, 0)
    stats['session_entry_counts'] = getattr(engine, 'session_entry_counts', {})
    stats['final_capital'] = getattr(engine, '_current_capital', 0)
    stats['equity_peak'] = getattr(engine, '_equity_peak', 0)
    return stats


def _max_parallel_workers() -> int:
    """Determine safe number of workers (leave 1 core free)."""
    return max(1, min(os.cpu_count() - 1, 6))


def run_variants(data: DataBundle, variants: List[Dict]) -> List[Dict]:
    """Run multiple variants sequentially.

    Each item in variants is a dict with 'label' + engine kwargs.
    Example:
        variants = [
            {"label": "Baseline"},
            {"label": "Trail 0.8/0.25", "trailing_activate_atr": 0.8, "trailing_distance_atr": 0.25},
        ]
    """
    results = []
    for i, v in enumerate(variants, 1):
        label = v.pop('label', f'V{i}')
        print(f"\n  [{i}/{len(variants)}] {label}", flush=True)
        stats = run_variant(data, label, **v)
        v['label'] = label  # restore
        results.append(stats)
    return results


def run_variants_parallel(data: DataBundle, variants: List[Dict],
                          max_workers: Optional[int] = None) -> List[Dict]:
    """Run multiple variants in parallel using ProcessPoolExecutor.

    Returns results in the same order as input variants.
    Note: stats['_trades'] and stats['_equity_curve'] are NOT available
    in parallel mode (not picklable / too large to transfer).
    """
    if max_workers is None:
        max_workers = _max_parallel_workers()

    tasks = []
    labels = []
    for i, v in enumerate(variants, 1):
        label = v.pop('label', f'V{i}')
        labels.append(label)
        tasks.append((data.m15_df, data.h1_df, label, dict(v)))
        v['label'] = label  # restore

    print(f"  Running {len(tasks)} variants in parallel ({max_workers} workers)...", flush=True)
    results = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {pool.submit(_worker_run_variant, t): i for i, t in enumerate(tasks)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            stats = future.result()
            results[idx] = stats
            print(f"    [{idx+1}/{len(tasks)}] {labels[idx]}: "
                  f"Sharpe={stats['sharpe']:.2f}, PnL=${stats['total_pnl']:.0f}, "
                  f"{stats['elapsed_s']}s", flush=True)

    return results


def run_kfold(data: DataBundle, engine_kwargs: Dict, n_folds: int = 6,
              label_prefix: str = "", parallel: bool = False) -> List[Dict]:
    """Run K-Fold cross validation with fixed time windows.

    Set parallel=True to run folds concurrently (significant speedup).
    """
    folds = [
        ("Fold1", "2015-01-01", "2017-01-01"),
        ("Fold2", "2017-01-01", "2019-01-01"),
        ("Fold3", "2019-01-01", "2021-01-01"),
        ("Fold4", "2021-01-01", "2023-01-01"),
        ("Fold5", "2023-01-01", "2025-01-01"),
        ("Fold6", "2025-01-01", "2026-04-01"),
    ][:n_folds]

    valid_folds = []
    for fold_name, start, end in folds:
        fold_data = data.slice(start, end)
        if len(fold_data.m15_df) < 1000 or len(fold_data.h1_df) < 200:
            continue
        label = f"{label_prefix}{fold_name}" if label_prefix else fold_name
        valid_folds.append((fold_name, start, end, fold_data, label))

    if not parallel or len(valid_folds) <= 1:
        results = []
        for fold_name, start, end, fold_data, label in valid_folds:
            stats = run_variant(fold_data, label, **engine_kwargs)
            stats['fold'] = fold_name
            stats['test_start'] = start
            stats['test_end'] = end
            results.append(stats)
        return results

    max_workers = _max_parallel_workers()
    tasks = []
    meta = []
    for fold_name, start, end, fold_data, label in valid_folds:
        tasks.append((fold_data.m15_df, fold_data.h1_df, label, dict(engine_kwargs)))
        meta.append((fold_name, start, end))

    print(f"  K-Fold parallel: {len(tasks)} folds, {max_workers} workers", flush=True)
    results = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        future_to_idx = {pool.submit(_worker_run_variant, t): i for i, t in enumerate(tasks)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            stats = future.result()
            fold_name, start, end = meta[idx]
            stats['fold'] = fold_name
            stats['test_start'] = start
            stats['test_end'] = end
            results[idx] = stats
            print(f"    {fold_name}: Sharpe={stats['sharpe']:.2f}, "
                  f"PnL=${stats['total_pnl']:.0f}, {stats['elapsed_s']}s", flush=True)

    return results


# ═══════════════════════════════════════════════════════════════
# Common config presets
# ═══════════════════════════════════════════════════════════════

C12_KWARGS = {
    "trailing_activate_atr": 0.8,
    "trailing_distance_atr": 0.25,
    "sl_atr_mult": 4.5,
    "tp_atr_mult": 8.0,
    "keltner_adx_threshold": 18,
    "regime_config": {
        'low': {'trail_act': 1.0, 'trail_dist': 0.35},
        'normal': {'trail_act': 0.8, 'trail_dist': 0.25},
        'high': {'trail_act': 0.6, 'trail_dist': 0.20},
    },
}

V3_REGIME = {
    'low': {'trail_act': 1.0, 'trail_dist': 0.35},
    'normal': {'trail_act': 0.8, 'trail_dist': 0.25},
    'high': {'trail_act': 0.6, 'trail_dist': 0.20},
}

# ── L8 Legacy Preset (frozen 2026-04-28) ─────────────────────────────────────
# History: L5.1 (04-13) → L6 (04-18) → L7 (04-22) → L8 (04-26) → L8_BASE+Cap80 (04-28)
# Kept for historical comparison. DO NOT use for new experiments.
L8_PARITY_KWARGS = {
    "trailing_activate_atr": 0.14,
    "trailing_distance_atr": 0.025,
    "sl_atr_mult": 3.5,
    "tp_atr_mult": 8.0,
    "keltner_adx_threshold": 14,
    "regime_config": {
        'low':    {'trail_act': 0.22, 'trail_dist': 0.04},
        'normal': {'trail_act': 0.14, 'trail_dist': 0.025},
        'high':   {'trail_act': 0.06, 'trail_dist': 0.008},
    },
    "intraday_adaptive": True,
    "choppy_threshold": 0.50,
    "time_decay_tp": False,
    "rsi_adx_filter": 40,
    "keltner_max_hold_m15": 20,
    "max_positions": 1,
    "live_atr_percentile": True,
}

# ── R202 preset (frozen 2026-05-14, superseded by Trail-First) ────────────────
# SL=6.0 + Cap=$70 proved harmful in R233 ablation (-0.7% and -1.7% respectively).
# Kept for comparison. DO NOT use for new experiments.
R202_PARITY_KWARGS = {
    "trailing_activate_atr": 0.06,
    "trailing_distance_atr": 0.015,
    "sl_atr_mult": 6.0,
    "tp_atr_mult": 8.0,
    "keltner_adx_threshold": 14,
    "regime_config": {
        'low':    {'trail_act': 0.06, 'trail_dist': 0.015},
        'normal': {'trail_act': 0.06, 'trail_dist': 0.015},
        'high':   {'trail_act': 0.06, 'trail_dist': 0.015},
    },
    "maxloss_cap": 70,
    "maxloss_cap_atr_mult": 4.0,
    "min_lot_size": 0.04,
    "max_lot_size": 0.04,
    "intraday_adaptive": True,
    "choppy_threshold": 0.50,
    "time_decay_tp": False,
    "rsi_adx_filter": 40,
    "keltner_max_hold_m15": 20,
    "max_positions": 1,
    "live_atr_percentile": True,
}

# ── LIVE_PARITY: Trail-First (R234 STRONG_PASS 6/6, deployed 2026-05-14) ─────
# History: L8 (04-28) → R202 (05-14 AM) → Trail-First (05-14 PM)
# R233 ablation showed: Trail change = +4.8%, SL 6.0 = -0.7%, Cap $70 = -1.7%
# Trail-First = keep R202 trail (best single change), revert SL to 3.5, remove Cap
# R234 validation: Sharpe 5.675, WF 19/19, MC P(>0)=100%, Sensitivity Δ=0.203
#
# 2026-05-15: Merged independent optimizations deployed to live:
#   max_hold 20→8: R155 sweep (hold_grid=[4,8,...128]), K-Fold 6.04>5.77
#   maxloss_cap=0: R233 ablation (Cap $70 = -1.7%), Trail-First removes Cap
#   lots 0.04: R193 (Sharpe 7.191, Trail 83.6%)
#   ATR_SL_MAX 50→150: 2026-04-23 bugfix, R30/R39 validated
# Session ADX (R178): fully mirrored via keltner_session_adx parameter.
LIVE_PARITY_KWARGS = {
    "trailing_activate_atr": 0.06,
    "trailing_distance_atr": 0.015,
    "sl_atr_mult": 3.5,
    "tp_atr_mult": 8.0,
    "keltner_session_adx": {
        "asia":    (0, 7, 14),      # R178: 保留原值, 实盘1月 27笔 WR=93% +$222
        "london":  (8, 12, 10),     # R178b: 101笔额外信号 WR=88.1% Sharpe=6.68
        "ny":      (13, 17, 16),    # NY时段需更强趋势确认
        "evening": (18, 23, 14),    # R178: 保留原值, 实盘1月 12笔 WR=92% +$102
    },
    "regime_config": {
        'low':    {'trail_act': 0.06, 'trail_dist': 0.015},
        'normal': {'trail_act': 0.06, 'trail_dist': 0.015},
        'high':   {'trail_act': 0.06, 'trail_dist': 0.015},
    },
    "min_lot_size": 0.04,
    "max_lot_size": 0.04,
    "intraday_adaptive": True,
    "choppy_threshold": 0.50,
    "time_decay_tp": False,
    "rsi_adx_filter": 40,
    "keltner_max_hold_m15": 8,
    "maxloss_cap": 0,
    "max_positions": 1,
    "live_atr_percentile": True,
}

# Realistic execution cost preset — calibrated from 91 real EA trades (2026-04/05).
# Entry slippage: BUY +$0.67, SELL +$0.17 (empirical sampling from live data).
# Spread: session-aware (Asia $0.40, London/NY $0.25, Late $0.45) + exit slippage.
REALISTIC_COST_KWARGS = {
    **LIVE_PARITY_KWARGS,
    "spread_model": "realistic",
    "slippage_model": "empirical",
    "slippage_buy": 0.67,
    "slippage_sell": 0.17,
}

# Legacy L5.1 preset — kept for historical comparison only.
# DO NOT use for new experiments; use LIVE_PARITY_KWARGS instead.
L51_PARITY_KWARGS = {
    "trailing_activate_atr": 0.28,
    "trailing_distance_atr": 0.06,
    "sl_atr_mult": 3.5,
    "tp_atr_mult": 8.0,
    "keltner_adx_threshold": 18,
    "regime_config": {
        'low':    {'trail_act': 0.40, 'trail_dist': 0.10},
        'normal': {'trail_act': 0.28, 'trail_dist': 0.06},
        'high':   {'trail_act': 0.12, 'trail_dist': 0.02},
    },
    "intraday_adaptive": True,
    "choppy_threshold": 0.50,
    "time_decay_tp": False,
    "rsi_adx_filter": 40,
    "keltner_max_hold_m15": 20,
    "max_positions": 1,
    "live_atr_percentile": True,
}

TRUE_BASELINE_KWARGS = {
    "trailing_activate_atr": 1.5,
    "trailing_distance_atr": 0.5,
    "sl_atr_mult": 2.5,
    "tp_atr_mult": 3.0,
    "keltner_adx_threshold": 24,
}


# ═══════════════════════════════════════════════════════════════
# JSON serialization helper
# ═══════════════════════════════════════════════════════════════

def screen_then_validate(
    data: DataBundle,
    h1_df: pd.DataFrame,
    signal_func,
    param_grid: List[Dict],
    engine_kwargs_func,
    *,
    bt_defaults: Optional[Dict] = None,
    filter_mode: str = 'eliminate',
    min_screen_sharpe: float = 0.0,
    top_k: int = 3,
    rank_by: str = 'sharpe',
    kfold_validate: bool = True,
    kfold_min_sharpe: float = 1.0,
    label_prefix: str = "",
    verbose: bool = True,
) -> Dict:
    """Two-tier screening: fast eliminate bad combos, then full engine validation.

    Tier 1 uses fast_backtest_signals (NumPy loop on H1) to eliminate clearly
    non-viable combos (Sharpe < threshold). This avoids accidentally discarding
    good strategies that the simplified H1 model underestimates.

    Tier 2 runs ALL survivors through the full BacktestEngine. K-Fold is only
    done on candidates exceeding kfold_min_sharpe in the full engine run.

    Args:
        data: DataBundle for full engine runs
        h1_df: H1 DataFrame with indicators for fast screening
        signal_func: callable(df, **sig_params) -> (signals, atr)
        param_grid: list of dicts with 'sig_params' and optionally 'bt_params', 'label'
        engine_kwargs_func: callable(sig_params, bt_params) -> dict of BacktestEngine kwargs
            Translates screened params into full engine config.
        bt_defaults: default backtest params for tier 1
        filter_mode: 'eliminate' = keep all with sharpe >= min_screen_sharpe;
                     'top_k' = keep only top_k (legacy, may miss good strategies)
        min_screen_sharpe: elimination threshold for tier 1 (default 0.0)
        top_k: number of candidates in top_k mode (ignored in eliminate mode)
        rank_by: metric for tier 1 ranking
        kfold_validate: run K-Fold on qualifying candidates in tier 2
        kfold_min_sharpe: only K-Fold candidates with full-engine sharpe >= this
        label_prefix: prefix for labels
        verbose: print progress

    Returns:
        Dict with 'screen_results', 'validated', 'kfold', and 'timing'.
    """
    from backtest.fast_screen import screen_grid
    import time as _time

    t0 = _time.time()

    if filter_mode == 'eliminate':
        if verbose:
            print(f"\n  === Two-Tier Screening (eliminate mode): "
                  f"{len(param_grid)} combos, min_sharpe={min_screen_sharpe} ===")
            print(f"  Tier 1: Fast screening on H1...", flush=True)

        survivors = screen_grid(
            h1_df, signal_func, param_grid,
            bt_defaults=bt_defaults, min_sharpe=min_screen_sharpe,
            rank_by=rank_by, label_prefix=label_prefix, verbose=verbose,
        )
    else:
        if verbose:
            print(f"\n  === Two-Tier Screening (top_k mode): "
                  f"{len(param_grid)} combos -> top {top_k} ===")
            print(f"  Tier 1: Fast screening on H1...", flush=True)

        survivors = screen_grid(
            h1_df, signal_func, param_grid,
            bt_defaults=bt_defaults, top_k=top_k,
            rank_by=rank_by, label_prefix=label_prefix, verbose=verbose,
        )

    tier1_time = _time.time() - t0

    if verbose:
        print(f"\n  Tier 2: Full engine validation on {len(survivors)} candidates...",
              flush=True)

    validated = []
    kfold_results = []
    for idx, r in enumerate(survivors):
        engine_kw = engine_kwargs_func(r['sig_params'], r['bt_params'])
        label = f"V_{r['label']}"

        stats = run_variant(data, label, verbose=verbose, **engine_kw)
        engine_sharpe = stats.get('sharpe', 0)
        validated.append({
            'label': label,
            'screen_rank': r['rank'],
            'screen_sharpe': r['stats']['sharpe'],
            'engine_sharpe': engine_sharpe,
            'engine_stats': {k: v for k, v in stats.items() if not k.startswith('_')},
            'sig_params': r['sig_params'],
            'bt_params': r['bt_params'],
        })

        if kfold_validate and engine_sharpe >= kfold_min_sharpe:
            kf = run_kfold(data, engine_kw, n_folds=6, label_prefix=f"KF_{r['label']}_")
            sharpes = [f['sharpe'] for f in kf]
            kfold_results.append({
                'label': label,
                'sharpes': [round(s, 2) for s in sharpes],
                'mean_sharpe': round(np.mean(sharpes), 2) if sharpes else 0,
                'min_sharpe': round(min(sharpes), 2) if sharpes else 0,
                'pass_count': sum(1 for s in sharpes if s > 0),
                'n_folds': len(sharpes),
            })
            if verbose:
                kfr = kfold_results[-1]
                print(f"    K-Fold: {kfr['pass_count']}/{kfr['n_folds']} pass, "
                      f"mean={kfr['mean_sharpe']:.2f}, min={kfr['min_sharpe']:.2f}")

        if verbose and (idx + 1) % 10 == 0:
            elapsed = _time.time() - t0
            print(f"    Progress: {idx+1}/{len(survivors)} validated ({elapsed:.0f}s)",
                  flush=True)

    total_time = _time.time() - t0
    if verbose:
        n_kfold_pass = sum(1 for k in kfold_results if k['pass_count'] >= 5)
        print(f"\n  === Done: {tier1_time:.1f}s tier1 + "
              f"{total_time - tier1_time:.1f}s tier2 = {total_time:.1f}s total ===")
        print(f"  {len(param_grid)} screened -> {len(survivors)} validated -> "
              f"{len(kfold_results)} K-Fold tested -> {n_kfold_pass} passed 5/6")

    return {
        'screen_survivors': [{k: v for k, v in r.items()} for r in survivors],
        'validated': validated,
        'kfold': kfold_results if kfold_validate else [],
        'timing': {
            'tier1_s': round(tier1_time, 1),
            'tier2_s': round(total_time - tier1_time, 1),
            'total_s': round(total_time, 1),
            'n_screened': len(param_grid),
            'n_survived': len(survivors),
            'n_validated': len(validated),
            'n_kfold_tested': len(kfold_results),
        },
    }


def sanitize_for_json(results: List[Dict]) -> List[Dict]:
    """Convert numpy types and drop non-serializable fields for JSON output."""
    safe = []
    for r in results:
        sr = {}
        for k, v in r.items():
            if k.startswith('_'):
                continue
            if isinstance(v, (np.integer,)):
                sr[k] = int(v)
            elif isinstance(v, (np.floating,)):
                sr[k] = float(v)
            else:
                sr[k] = v
        safe.append(sr)
    return safe
