"""
H4 Backtest Engine
====================
Single-timeframe engine for H4 strategies on XAUUSD.

Design rationale (vs the M15+H1 BacktestEngine):
  - Driven by H4 bars (6 bars/day max), not M15
  - No intraday filters (Choppy Gate, session, slot contention)
  - SL/TP in ATR multiples, appropriate for multi-day holds
  - Trailing stop, max hold (in H4 bars), cooldown
  - Realistic spread cost
  - Reuses TradeRecord/Position from backtest.engine for compatibility

Strategies are pluggable via signal functions that receive an H4 DataFrame
window and return a signal dict or None.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pathlib import Path

from backtest.engine import Position, TradeRecord


# ═══════════════════════════════════════════════════════════════
# Indicators (H4-specific prep)
# ═══════════════════════════════════════════════════════════════

def prepare_h4_indicators(df: pd.DataFrame, kc_ema: int = 20, kc_mult: float = 2.0) -> pd.DataFrame:
    """Compute technical indicators on H4 OHLCV data."""
    df = df.copy()

    # ATR
    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift(1)).abs(),
        (df['Low'] - df['Close'].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df['ATR'] = tr.ewm(span=14, adjust=False).mean()

    # EMAs
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()

    # Keltner Channel
    df['KC_mid'] = df['Close'].ewm(span=kc_ema, adjust=False).mean()
    df['KC_upper'] = df['KC_mid'] + kc_mult * df['ATR']
    df['KC_lower'] = df['KC_mid'] - kc_mult * df['ATR']

    # Bollinger Bands
    bb_mid = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_upper'] = bb_mid + 2 * bb_std
    df['BB_lower'] = bb_mid - 2 * bb_std
    df['BB_bandwidth'] = (df['BB_upper'] - df['BB_lower']) / bb_mid

    # Donchian Channel
    df['DC_upper'] = df['High'].rolling(20).max()
    df['DC_lower'] = df['Low'].rolling(20).min()
    df['DC_mid'] = (df['DC_upper'] + df['DC_lower']) / 2

    # RSI(14)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI14'] = 100 - 100 / (1 + rs)

    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    # ADX
    plus_dm = df['High'].diff().clip(lower=0)
    minus_dm = (-df['Low'].diff()).clip(lower=0)
    plus_dm[df['High'].diff() <= (-df['Low'].diff())] = 0
    minus_dm[(-df['Low'].diff()) <= df['High'].diff()] = 0
    atr_smooth = tr.ewm(span=14, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr_smooth.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr_smooth.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df['ADX'] = dx.ewm(span=14, adjust=False).mean()

    # CCI(20)
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df['CCI'] = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))

    # EMA slopes (normalized by ATR)
    df['EMA20_slope'] = (df['EMA20'] - df['EMA20'].shift(3)) / df['ATR'].replace(0, np.nan)
    df['EMA50_slope'] = (df['EMA50'] - df['EMA50'].shift(3)) / df['ATR'].replace(0, np.nan)

    return df


# ═══════════════════════════════════════════════════════════════
# Signal function type
# ═══════════════════════════════════════════════════════════════

SignalFunc = Callable[[pd.DataFrame], Optional[Dict]]


# ═══════════════════════════════════════════════════════════════
# H4 Engine
# ═══════════════════════════════════════════════════════════════

class H4BacktestEngine:
    """Single-timeframe H4 backtest engine.

    Parameters:
        h4_df: H4 OHLCV DataFrame with indicators (from prepare_h4_indicators)
        signal_funcs: list of (name, func) tuples; func receives H4 window -> signal dict
        sl_atr_mult: stop loss in ATR multiples (0 = use signal-provided SL)
        tp_atr_mult: take profit in ATR multiples (0 = use signal-provided TP)
        trailing_activate_atr: trailing activation threshold in ATR
        trailing_distance_atr: trailing distance in ATR
        max_hold: max bars to hold (H4 bars; 0 = unlimited)
        cooldown_bars: bars to wait after a trade closes before next entry
        max_positions: max simultaneous open positions
        spread_cost: spread cost per trade in dollars
        lot_size: fixed lot size
        window_size: bars of history passed to signal functions
    """

    WINDOW = 100

    def __init__(
        self,
        h4_df: pd.DataFrame,
        signal_funcs: List[Tuple[str, SignalFunc]],
        *,
        sl_atr_mult: float = 3.0,
        tp_atr_mult: float = 6.0,
        trailing_activate_atr: float = 0.0,
        trailing_distance_atr: float = 0.0,
        max_hold: int = 30,
        cooldown_bars: int = 2,
        max_positions: int = 1,
        spread_cost: float = 0.30,
        lot_size: float = 0.02,
        window_size: int = 100,
        adx_min: float = 0.0,
        # Filters matching BacktestEngine LIVE_PARITY parity
        choppy_gate: bool = True,
        choppy_adx_threshold: float = 20.0,
        atr_pctl_floor: float = 0.30,
        atr_pctl_window: int = 50,
        rule_b_sigma: float = 2.5,
        rule_b_lookback: int = 60,
        # Entry slippage model calibrated from 91 real EA trades
        slippage_model: str = "none",       # "none" | "fixed" | "empirical" | "realistic"
        slippage_buy: float = 0.67,
        slippage_sell: float = 0.17,
        slippage_seed: int = 42,
    ):
        self.h4_df = h4_df
        self.signal_funcs = signal_funcs
        self._sl_atr_mult = sl_atr_mult
        self._tp_atr_mult = tp_atr_mult
        self._trail_act = trailing_activate_atr
        self._trail_dist = trailing_distance_atr
        self._max_hold = max_hold
        self._cooldown = cooldown_bars
        self._max_positions = max_positions
        self._spread = spread_cost
        self._lot_size = lot_size
        self._window = window_size
        self._adx_min = adx_min
        # Filters
        self._choppy_gate = choppy_gate
        self._choppy_adx_threshold = choppy_adx_threshold
        self._atr_pctl_floor = atr_pctl_floor
        self._atr_pctl_window = atr_pctl_window
        self._rule_b_sigma = rule_b_sigma
        self._rule_b_lookback = rule_b_lookback

        # Entry slippage
        self._slippage_model = slippage_model
        self._slippage_buy = slippage_buy
        self._slippage_sell = slippage_sell
        self._slippage_rng = np.random.RandomState(slippage_seed)
        self._empirical_buy_slips = np.array([
            -3.43, -1.29, -1.17, -0.82, -0.59, -0.53, -0.28, -0.22,
            0.18, 0.19, 0.23, 0.24, 0.26, 0.26, 0.29, 0.30, 0.32, 0.34,
            0.35, 0.41, 0.41, 0.49, 0.51, 0.61, 0.62, 0.62, 0.64, 0.64,
            0.67, 0.69, 0.73, 0.81, 0.82, 0.83, 0.87, 0.87, 0.87, 1.08,
            1.08, 1.12, 1.20, 1.23, 1.33, 1.86, 1.91, 2.26, 2.97, 3.02,
            3.20, 4.33,
        ])
        self._empirical_sell_slips = np.array([
            -2.59, -2.04, -1.61, -0.97, -0.78, -0.63, -0.61, -0.60,
            -0.51, -0.25, -0.24, -0.06, -0.05, -0.05, -0.04, -0.04, -0.04,
            0.00, 0.00, 0.17, 0.19, 0.26, 0.28, 0.28, 0.32, 0.40, 0.43,
            0.46, 0.47, 0.53, 0.58, 0.61, 0.82, 0.91, 0.97, 0.99, 1.04,
            1.10, 1.55, 2.33, 3.19,
        ])
        self.total_entry_slippage = 0.0
        self.slippage_count = 0

        self.positions: List[Position] = []
        self.trades: List[TradeRecord] = []
        self._cooldown_until = 0

        # Diagnostics
        self.total_signals = 0
        self.filtered_cooldown = 0
        self.filtered_adx = 0
        self.filtered_max_pos = 0
        self.filtered_choppy = 0
        self.filtered_atr_pctl = 0
        self.filtered_rule_b = 0

    def _calc_entry_slippage(self, direction: str) -> float:
        """Entry slippage from 91 real EA trades. Positive = worse for trader."""
        if self._slippage_model == "none":
            return 0.0
        if self._slippage_model == "fixed":
            return self._slippage_buy if direction == 'BUY' else self._slippage_sell
        if self._slippage_model == "empirical":
            pool = self._empirical_buy_slips if direction == 'BUY' else self._empirical_sell_slips
            return float(self._slippage_rng.choice(pool))
        if self._slippage_model == "realistic":
            if self._slippage_rng.rand() < 0.5:
                pool = self._empirical_buy_slips if direction == 'BUY' else self._empirical_sell_slips
                return float(self._slippage_rng.choice(pool))
            return self._slippage_buy if direction == 'BUY' else self._slippage_sell
        return 0.0

    def _compute_atr_pctl(self, atr: np.ndarray) -> np.ndarray:
        """Rolling percentile rank of ATR over window."""
        n = len(atr)
        out = np.full(n, 0.5)
        w = self._atr_pctl_window
        for i in range(w, n):
            window = atr[i - w:i]
            valid = window[~np.isnan(window)]
            if len(valid) >= 10:
                out[i] = float(np.sum(valid < atr[i])) / len(valid)
        return out

    def _check_rule_b(self, atr: np.ndarray, i: int) -> bool:
        """Return True if Rule B blocks entry (ATR spike > mean + sigma*std)."""
        lb = self._rule_b_lookback
        if i < lb:
            return False
        window = atr[max(0, i - lb):i]
        valid = window[~np.isnan(window)]
        if len(valid) < 10:
            return False
        mean = float(valid.mean())
        std = float(valid.std())
        if std > 0 and atr[i] > mean + self._rule_b_sigma * std:
            return True
        return False

    def run(self) -> List[TradeRecord]:
        """Run backtest over all H4 bars. Returns list of TradeRecord."""
        h4 = self.h4_df
        n_bars = len(h4)
        high = h4['High'].values
        low = h4['Low'].values
        close = h4['Close'].values
        atr = h4['ATR'].values
        times = h4.index
        adx = h4['ADX'].values if 'ADX' in h4.columns else np.zeros(n_bars)

        # Pre-compute ATR percentile array for filter
        atr_pctl = self._compute_atr_pctl(atr) if self._atr_pctl_floor > 0 else np.ones(n_bars)

        progress_step = max(1, n_bars // 10)
        print(f'  Backtest: {times[self._window]} -> {times[-1]}')
        print(f'  H4 bars: {n_bars}')

        for i in range(self._window, n_bars):
            if (i - self._window) % progress_step == 0:
                pct = int(100 * (i - self._window) / (n_bars - self._window))
                print(f'    {pct}%...', end='', flush=True)

            bar_time = times[i]
            h, l, c = high[i], low[i], close[i]
            cur_atr = atr[i]

            if cur_atr <= 0 or np.isnan(cur_atr):
                continue

            # ── Manage open positions ──
            closed_indices = []
            for pi, pos in enumerate(self.positions):
                pos.bars_held += 1
                exit_price = None
                reason = ""

                if pos.direction == 'BUY':
                    if l <= pos.sl_price:
                        exit_price = pos.sl_price
                        reason = "SL"
                    elif pos.tp_price > 0 and h >= pos.tp_price:
                        exit_price = pos.tp_price
                        reason = "TP"
                    else:
                        pos.extreme_price = max(pos.extreme_price, h)
                        if self._trail_act > 0 and self._trail_dist > 0:
                            profit = pos.extreme_price - pos.entry_price
                            if profit >= pos.entry_atr * self._trail_act:
                                trail = pos.extreme_price - pos.entry_atr * self._trail_dist
                                if trail > pos.trailing_stop_price:
                                    pos.trailing_stop_price = trail
                                if pos.trailing_stop_price > 0 and l <= pos.trailing_stop_price:
                                    exit_price = pos.trailing_stop_price
                                    reason = "Trail"
                        if exit_price is None and self._max_hold > 0 and pos.bars_held >= self._max_hold:
                            exit_price = c
                            reason = "Timeout"
                else:  # SELL
                    if h >= pos.sl_price:
                        exit_price = pos.sl_price
                        reason = "SL"
                    elif pos.tp_price > 0 and l <= pos.tp_price:
                        exit_price = pos.tp_price
                        reason = "TP"
                    else:
                        pos.extreme_price = min(pos.extreme_price, l)
                        if self._trail_act > 0 and self._trail_dist > 0:
                            profit = pos.entry_price - pos.extreme_price
                            if profit >= pos.entry_atr * self._trail_act:
                                trail = pos.extreme_price + pos.entry_atr * self._trail_dist
                                if pos.trailing_stop_price == 0 or trail < pos.trailing_stop_price:
                                    pos.trailing_stop_price = trail
                                if pos.trailing_stop_price > 0 and h >= pos.trailing_stop_price:
                                    exit_price = pos.trailing_stop_price
                                    reason = "Trail"
                        if exit_price is None and self._max_hold > 0 and pos.bars_held >= self._max_hold:
                            exit_price = c
                            reason = "Timeout"

                if exit_price is not None:
                    if pos.direction == 'BUY':
                        pnl = (exit_price - pos.entry_price) * self._lot_size * 100 - self._spread
                    else:
                        pnl = (pos.entry_price - exit_price) * self._lot_size * 100 - self._spread

                    self.trades.append(TradeRecord(
                        strategy=pos.strategy,
                        direction=pos.direction,
                        entry_price=pos.entry_price,
                        exit_price=exit_price,
                        entry_time=pos.entry_time,
                        exit_time=bar_time,
                        lots=self._lot_size,
                        pnl=round(pnl, 2),
                        exit_reason=reason,
                        bars_held=pos.bars_held,
                    ))
                    closed_indices.append(pi)
                    self._cooldown_until = i + self._cooldown

            for pi in reversed(closed_indices):
                self.positions.pop(pi)

            # ── Check for new entries ──
            if i < self._cooldown_until:
                continue

            if len(self.positions) >= self._max_positions:
                continue

            window = h4.iloc[max(0, i - self._window + 1):i + 1]

            for strat_name, sig_func in self.signal_funcs:
                if len(self.positions) >= self._max_positions:
                    break

                sig = sig_func(window)
                if sig is None:
                    continue

                self.total_signals += 1
                direction = sig.get('signal', '')
                if direction not in ('BUY', 'SELL'):
                    continue

                # ADX filter
                if self._adx_min > 0 and adx[i] < self._adx_min:
                    self.filtered_adx += 1
                    continue

                # Choppy Gate: low ADX = choppy market
                if self._choppy_gate and adx[i] < self._choppy_adx_threshold:
                    self.filtered_choppy += 1
                    continue

                # ATR Percentile Floor
                if self._atr_pctl_floor > 0 and atr_pctl[i] < self._atr_pctl_floor:
                    self.filtered_atr_pctl += 1
                    continue

                # Rule B: extreme volatility spike
                if self._rule_b_sigma > 0 and self._check_rule_b(atr, i):
                    self.filtered_rule_b += 1
                    continue

                # Determine SL/TP
                if self._sl_atr_mult > 0:
                    sl_dist = cur_atr * self._sl_atr_mult
                else:
                    sl_dist = sig.get('sl', cur_atr * 3.0)

                if self._tp_atr_mult > 0:
                    tp_dist = cur_atr * self._tp_atr_mult
                else:
                    tp_dist = sig.get('tp', 0)

                entry_price = c

                # Apply realistic entry slippage
                slip = self._calc_entry_slippage(direction)
                if slip != 0.0:
                    if direction == 'BUY':
                        entry_price += slip
                    else:
                        entry_price -= slip
                    self.total_entry_slippage += abs(slip)
                    self.slippage_count += 1

                pos = Position(
                    strategy=sig.get('strategy', strat_name),
                    direction=direction,
                    entry_price=entry_price,
                    entry_time=bar_time,
                    lots=self._lot_size,
                    sl_distance=sl_dist,
                    tp_distance=tp_dist,
                    entry_atr=cur_atr,
                )
                self.positions.append(pos)

        print(' done!')

        # Close any remaining positions at last bar
        for pos in self.positions:
            c = close[-1]
            if pos.direction == 'BUY':
                pnl = (c - pos.entry_price) * self._lot_size * 100 - self._spread
            else:
                pnl = (pos.entry_price - c) * self._lot_size * 100 - self._spread
            self.trades.append(TradeRecord(
                strategy=pos.strategy, direction=pos.direction,
                entry_price=pos.entry_price, exit_price=c,
                entry_time=pos.entry_time, exit_time=times[-1],
                lots=self._lot_size, pnl=round(pnl, 2),
                exit_reason="EndOfData", bars_held=pos.bars_held,
            ))
        self.positions.clear()

        return self.trades


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

_H4_CANDIDATES = [
    Path("data/download/xauusd-h4-bid-2015-01-01-2026-05-13.csv"),
    Path("data/download/xauusd-h4-bid-2015-01-01-2026-05-06.csv"),
    Path("data/download/xauusd-h4-bid-2015-01-01-2026-04-27.csv"),
    Path("data/download/xauusd-h4-bid-2015-01-01-2026-04-10.csv"),
]


def load_h4(start: str = "2015-01-01", end: Optional[str] = None) -> pd.DataFrame:
    """Load H4 OHLCV data from CSV."""
    csv_path = next((p for p in _H4_CANDIDATES if p.exists()), None)
    if csv_path is None:
        raise FileNotFoundError(f"H4 CSV not found. Candidates: {_H4_CANDIDATES}")

    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('datetime', inplace=True)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low',
                        'close': 'Close', 'volume': 'Volume'}, inplace=True)

    if start:
        df = df[df.index >= pd.Timestamp(start, tz='UTC')]
    if end:
        df = df[df.index <= pd.Timestamp(end, tz='UTC')]

    print(f"  H4: {csv_path.name} ({len(df)} bars, {df.index[0]} -> {df.index[-1]})")
    return df


def load_h4_with_indicators(start: str = "2015-01-01", end: Optional[str] = None,
                            kc_ema: int = 20, kc_mult: float = 2.0) -> pd.DataFrame:
    """Load H4 data and compute indicators."""
    print("\nLoading H4 data...")
    raw = load_h4(start, end)
    print("  Computing H4 indicators...", end='', flush=True)
    df = prepare_h4_indicators(raw, kc_ema=kc_ema, kc_mult=kc_mult)
    print(f" done ({len(df)} bars)")
    return df
