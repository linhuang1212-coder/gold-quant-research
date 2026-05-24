#!/usr/bin/env python3
"""R244 — Live Portfolio Missing-Strategy Revalidation.

R243 validated Keltner + M30 RSI14 via BacktestEngine baseline. This script
covers the 3 remaining live strategies that R243 did NOT exercise:

  - TSMOM       (H1, Score 480/960 method, R205/R206 candidate)
  - SESS_BO     (H1, GMT12 breakout, D1 EMA20 filter ON)
  - Dual Thrust (H1, k=0.5, n_bars=6)

All use current live parameters as documented in
.cursor/rules/trading-project-architecture.md (synced 2026-05-15):

  | Strategy    | sl_atr | tp_atr | trail_act | trail_dist | max_hold |
  |-------------|--------|--------|-----------|------------|----------|
  | TSMOM       | 3.5    | 8.0    | 0.14      | 0.025      | 30 bars  |
  | SESS_BO     | 4.5    | 4.0    | 0.06      | 0.01       | 20 bars  |
  | Dual Thrust | 6.0    | 8.0    | 0.06      | 0.01       | 20 bars  |

Outputs:
  results/r244_missing_strategies/
    SUMMARY.md
    {strat}_full.json
    {strat}_era.json
    {strat}_recent7mo.json
"""
from __future__ import annotations
import sys
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings('ignore')

from backtest.runner import load_csv, H1_CSV_PATH

OUTPUT_DIR = Path("results/r244_missing_strategies")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOT = 0.04
PV = 100
SPREAD = 0.30

ERA_SEGMENTS = {
    "Pre-COVID (2015-2019)":      ("2015-01-01", "2020-01-01"),
    "COVID+Recovery (2020-2021)": ("2020-01-01", "2022-01-01"),
    "Tightening (2022-2023)":     ("2022-01-01", "2024-01-01"),
    "Recent (2024-2026)":         ("2024-01-01", "2026-06-01"),
}
RECENT_WINDOW = ("2025-09-01", "2026-04-10")
DATA_END = "2026-04-10"

T0 = time.time()


def stamp(msg: str):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════
# Shared indicators
# ═══════════════════════════════════════════════════════════════

def compute_atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(c)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    atr = np.zeros(n)
    if n <= period:
        return atr
    atr[period] = np.mean(tr[1:period + 1])
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def calc_stats(trades: List[Dict], label: str = "") -> Dict:
    if not trades:
        return {'label': label, 'n': 0, 'sharpe': 0.0, 'pnl': 0.0,
                'win_rate': 0.0, 'max_dd': 0.0, 'days': 0}
    pnls = np.array([t['pnl'] for t in trades])
    n = len(pnls)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    win_rate = float(len(wins) / n) if n else 0.0
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0.0) + t['pnl']
    daily_arr = np.array(list(daily.values()))
    sharpe = 0.0
    if len(daily_arr) > 1 and daily_arr.std(ddof=1) > 0:
        sharpe = float(daily_arr.mean() / daily_arr.std(ddof=1) * np.sqrt(252))
    cum = np.cumsum(pnls)
    max_dd = float((np.maximum.accumulate(cum) - cum).max()) if len(cum) else 0.0
    pf = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else 0.0
    return {
        'label': label,
        'n': int(n),
        'sharpe': round(sharpe, 3),
        'pnl': round(float(pnls.sum()), 2),
        'win_rate': round(win_rate, 4),
        'max_dd': round(max_dd, 2),
        'pf': round(pf, 3),
        'days': int(len(daily_arr)),
    }


def filter_period(trades: List[Dict], start: str, end: str) -> List[Dict]:
    ts = pd.Timestamp(start, tz='UTC')
    te = pd.Timestamp(end, tz='UTC')
    return [t for t in trades if ts <= pd.Timestamp(t['entry_time']) < te]


# ═══════════════════════════════════════════════════════════════
# TSMOM (R205/R206 candidate: LB=480/960, SL=3.5, TP=8.0, MH=30, RB=2.5)
# ═══════════════════════════════════════════════════════════════

def compute_score(close: np.ndarray, fast_lb: int, slow_lb: int) -> np.ndarray:
    n = len(close)
    score = np.zeros(n)
    for i in range(n):
        s = 0.0
        if i >= fast_lb and close[i - fast_lb] > 0:
            ret = close[i] / close[i - fast_lb] - 1.0
            s += 0.5 * (1.0 if ret > 0 else (-1.0 if ret < 0 else 0.0))
        if i >= slow_lb and close[i - slow_lb] > 0:
            ret = close[i] / close[i - slow_lb] - 1.0
            s += 0.5 * (1.0 if ret > 0 else (-1.0 if ret < 0 else 0.0))
        score[i] = s
    return score


def bt_tsmom(h1: pd.DataFrame, fast_lb=480, slow_lb=960,
             sl_atr_mult=3.5, tp_atr_mult=8.0, max_hold=30,
             trail_act_atr=0.14, trail_dist_atr=0.025,
             rule_b_sigma=2.0, rule_b_lb=90, rule_b_skip=8,
             atr_floor=0.1, cap_atr_mult=6.5) -> List[Dict]:
    """Live TSMOM: Score 480/960 + R243-recommended Rule B σ=2.0 + maxloss cap."""
    h = h1['High'].values
    l = h1['Low'].values
    c = h1['Close'].values
    times = h1.index
    n = len(c)

    atr = compute_atr(h, l, c, 14)
    score = compute_score(c, fast_lb, slow_lb)

    # Rule B blocking
    atr_s = pd.Series(atr)
    rb_mean = atr_s.rolling(rule_b_lb).mean().values
    rb_std = atr_s.rolling(rule_b_lb).std(ddof=1).values
    skip_until = -1
    skip_delta_bars = rule_b_skip  # H1 bars

    trades = []
    pos = None
    last_exit = -999

    for i in range(slow_lb + 1, n):
        # Rule B check
        if i >= rule_b_lb and not pd.isna(rb_mean[i]) and not pd.isna(rb_std[i]) \
           and rb_std[i] > 0:
            if atr[i] > rb_mean[i] + rule_b_sigma * rb_std[i]:
                skip_until = max(skip_until, i + skip_delta_bars)

        if pos is not None:
            sl = pos['atr'] * sl_atr_mult
            tp = pos['atr'] * tp_atr_mult
            bars_held = i - pos['bar']

            # MaxLoss cap (R53: cap_atr_mult × ATR × LOT × PV at running PnL)
            cap_limit = cap_atr_mult * pos['atr'] * LOT * PV if cap_atr_mult > 0 else 0
            if cap_limit > 0:
                if pos['dir'] == 'BUY':
                    float_pnl = (c[i] - pos['entry']) * LOT * PV
                else:
                    float_pnl = (pos['entry'] - c[i]) * LOT * PV
                if -float_pnl > cap_limit:
                    pnl = -cap_limit - SPREAD * LOT * PV
                    trades.append({'dir': pos['dir'], 'entry_time': pos['time'].isoformat(),
                                   'exit_time': times[i].isoformat(),
                                   'pnl': pnl, 'reason': 'MaxLossCap', 'bars': bars_held})
                    pos = None; last_exit = i; continue

            # Trail stop
            if pos['dir'] == 'BUY':
                gain = c[i] - pos['entry']
                if gain >= trail_act_atr * pos['atr']:
                    new_sl = c[i] - trail_dist_atr * pos['atr']
                    pos['trail_sl'] = max(pos.get('trail_sl', pos['entry'] - sl), new_sl)
                sl_price = pos.get('trail_sl', pos['entry'] - sl)
                if l[i] <= sl_price:
                    pnl = (sl_price - pos['entry'] - SPREAD) * LOT * PV
                    trades.append({'dir': 'BUY', 'entry_time': pos['time'].isoformat(),
                                   'exit_time': times[i].isoformat(),
                                   'pnl': pnl, 'reason': 'SL/Trail', 'bars': bars_held})
                    pos = None; last_exit = i; continue
                if h[i] >= pos['entry'] + tp:
                    pnl = (tp - SPREAD) * LOT * PV
                    trades.append({'dir': 'BUY', 'entry_time': pos['time'].isoformat(),
                                   'exit_time': times[i].isoformat(),
                                   'pnl': pnl, 'reason': 'TP', 'bars': bars_held})
                    pos = None; last_exit = i; continue
            else:
                gain = pos['entry'] - c[i]
                if gain >= trail_act_atr * pos['atr']:
                    new_sl = c[i] + trail_dist_atr * pos['atr']
                    pos['trail_sl'] = min(pos.get('trail_sl', pos['entry'] + sl), new_sl)
                sl_price = pos.get('trail_sl', pos['entry'] + sl)
                if h[i] >= sl_price:
                    pnl = (pos['entry'] - sl_price - SPREAD) * LOT * PV
                    trades.append({'dir': 'SELL', 'entry_time': pos['time'].isoformat(),
                                   'exit_time': times[i].isoformat(),
                                   'pnl': pnl, 'reason': 'SL/Trail', 'bars': bars_held})
                    pos = None; last_exit = i; continue
                if l[i] <= pos['entry'] - tp:
                    pnl = (tp - SPREAD) * LOT * PV
                    trades.append({'dir': 'SELL', 'entry_time': pos['time'].isoformat(),
                                   'exit_time': times[i].isoformat(),
                                   'pnl': pnl, 'reason': 'TP', 'bars': bars_held})
                    pos = None; last_exit = i; continue

            if bars_held >= max_hold:
                if pos['dir'] == 'BUY':
                    pnl = (c[i] - pos['entry'] - SPREAD) * LOT * PV
                else:
                    pnl = (pos['entry'] - c[i] - SPREAD) * LOT * PV
                trades.append({'dir': pos['dir'], 'entry_time': pos['time'].isoformat(),
                               'exit_time': times[i].isoformat(),
                               'pnl': pnl, 'reason': 'Time', 'bars': bars_held})
                pos = None; last_exit = i
            continue

        if i - last_exit < 2:
            continue
        if i <= skip_until:
            continue
        if atr[i] < atr_floor:
            continue

        # Entry: score change of sign
        if score[i] > 0 and score[i - 1] <= 0:
            pos = {'dir': 'BUY', 'entry': c[i] + SPREAD / 2, 'bar': i,
                   'time': times[i], 'atr': atr[i]}
        elif score[i] < 0 and score[i - 1] >= 0:
            pos = {'dir': 'SELL', 'entry': c[i] - SPREAD / 2, 'bar': i,
                   'time': times[i], 'atr': atr[i]}

    return trades


# ═══════════════════════════════════════════════════════════════
# SESS_BO (GMT12 breakout, D1 EMA20 filter, lookback=4)
# ═══════════════════════════════════════════════════════════════

def bt_sess_bo(h1: pd.DataFrame, session_hour=12, lookback=4,
               sl_atr_mult=4.5, tp_atr_mult=4.0, max_hold=20,
               trail_act_atr=0.06, trail_dist_atr=0.01,
               d1_ema20_filter=True, atr_floor=0.1,
               cap_atr_mult=5.0) -> List[Dict]:
    h = h1['High'].values
    l = h1['Low'].values
    c = h1['Close'].values
    times = h1.index
    hours = h1.index.hour
    n = len(c)

    atr = compute_atr(h, l, c, 14)

    d1_dir = None
    if d1_ema20_filter:
        daily = h1['Close'].resample('D').last().dropna()
        d1_ema20 = daily.ewm(span=20, adjust=False).mean()
        d1_direction = pd.Series(np.where(daily > d1_ema20, 1, -1), index=daily.index)
        d1_dir = d1_direction.reindex(h1.index, method='ffill').values

    trades = []
    pos = None
    last_exit = -999

    for i in range(lookback + 14, n):
        if pos is not None:
            sl = pos['atr'] * sl_atr_mult
            tp = pos['atr'] * tp_atr_mult
            bars_held = i - pos['bar']

            # MaxLoss cap
            cap_limit = cap_atr_mult * pos['atr'] * LOT * PV if cap_atr_mult > 0 else 0
            if cap_limit > 0:
                if pos['dir'] == 'BUY':
                    float_pnl = (c[i] - pos['entry']) * LOT * PV
                else:
                    float_pnl = (pos['entry'] - c[i]) * LOT * PV
                if -float_pnl > cap_limit:
                    pnl = -cap_limit - SPREAD * LOT * PV
                    trades.append({'dir': pos['dir'], 'entry_time': pos['time'].isoformat(),
                                   'exit_time': times[i].isoformat(),
                                   'pnl': pnl, 'reason': 'MaxLossCap', 'bars': bars_held})
                    pos = None; last_exit = i; continue

            if pos['dir'] == 'BUY':
                gain = c[i] - pos['entry']
                if gain >= trail_act_atr * pos['atr']:
                    new_sl = c[i] - trail_dist_atr * pos['atr']
                    pos['trail_sl'] = max(pos.get('trail_sl', pos['entry'] - sl), new_sl)
                sl_price = pos.get('trail_sl', pos['entry'] - sl)
                if l[i] <= sl_price:
                    pnl = (sl_price - pos['entry'] - SPREAD) * LOT * PV
                    trades.append({'dir': 'BUY', 'entry_time': pos['time'].isoformat(),
                                   'exit_time': times[i].isoformat(),
                                   'pnl': pnl, 'reason': 'SL/Trail', 'bars': bars_held})
                    pos = None; last_exit = i; continue
                if h[i] >= pos['entry'] + tp:
                    pnl = (tp - SPREAD) * LOT * PV
                    trades.append({'dir': 'BUY', 'entry_time': pos['time'].isoformat(),
                                   'exit_time': times[i].isoformat(),
                                   'pnl': pnl, 'reason': 'TP', 'bars': bars_held})
                    pos = None; last_exit = i; continue
            else:
                gain = pos['entry'] - c[i]
                if gain >= trail_act_atr * pos['atr']:
                    new_sl = c[i] + trail_dist_atr * pos['atr']
                    pos['trail_sl'] = min(pos.get('trail_sl', pos['entry'] + sl), new_sl)
                sl_price = pos.get('trail_sl', pos['entry'] + sl)
                if h[i] >= sl_price:
                    pnl = (pos['entry'] - sl_price - SPREAD) * LOT * PV
                    trades.append({'dir': 'SELL', 'entry_time': pos['time'].isoformat(),
                                   'exit_time': times[i].isoformat(),
                                   'pnl': pnl, 'reason': 'SL/Trail', 'bars': bars_held})
                    pos = None; last_exit = i; continue
                if l[i] <= pos['entry'] - tp:
                    pnl = (tp - SPREAD) * LOT * PV
                    trades.append({'dir': 'SELL', 'entry_time': pos['time'].isoformat(),
                                   'exit_time': times[i].isoformat(),
                                   'pnl': pnl, 'reason': 'TP', 'bars': bars_held})
                    pos = None; last_exit = i; continue

            if bars_held >= max_hold:
                if pos['dir'] == 'BUY':
                    pnl = (c[i] - pos['entry'] - SPREAD) * LOT * PV
                else:
                    pnl = (pos['entry'] - c[i] - SPREAD) * LOT * PV
                trades.append({'dir': pos['dir'], 'entry_time': pos['time'].isoformat(),
                               'exit_time': times[i].isoformat(),
                               'pnl': pnl, 'reason': 'Time', 'bars': bars_held})
                pos = None; last_exit = i
            continue

        if hours[i] != session_hour:
            continue
        if i - last_exit < 2:
            continue
        if np.isnan(atr[i]) or atr[i] < atr_floor:
            continue

        # Breakout signal: close above/below max/min of last `lookback` bars
        hh = max(h[i - j] for j in range(1, lookback + 1))
        ll = min(l[i - j] for j in range(1, lookback + 1))
        signal = None
        if c[i] > hh:
            signal = 'BUY'
        elif c[i] < ll:
            signal = 'SELL'
        if signal is None:
            continue

        # D1 EMA20 filter
        if d1_ema20_filter and d1_dir is not None:
            d1v = d1_dir[i] if i < len(d1_dir) else 0
            if signal == 'BUY' and d1v == -1:
                continue
            if signal == 'SELL' and d1v == 1:
                continue

        if signal == 'BUY':
            pos = {'dir': 'BUY', 'entry': c[i] + SPREAD / 2, 'bar': i,
                   'time': times[i], 'atr': atr[i]}
        else:
            pos = {'dir': 'SELL', 'entry': c[i] - SPREAD / 2, 'bar': i,
                   'time': times[i], 'atr': atr[i]}

    return trades


# ═══════════════════════════════════════════════════════════════
# Dual Thrust (k=0.5, n_bars=6, live params)
# ═══════════════════════════════════════════════════════════════

def bt_dual_thrust(h1: pd.DataFrame, n_bars=6, k_up=0.5, k_down=0.5,
                   sl_atr_mult=6.0, tp_atr_mult=8.0, max_hold=20,
                   trail_act_atr=0.06, trail_dist_atr=0.01,
                   atr_floor=0.1, cap_atr_mult=5.0) -> List[Dict]:
    h = h1['High'].values
    l = h1['Low'].values
    c = h1['Close'].values
    o = h1['Open'].values
    times = h1.index
    n = len(c)

    atr = compute_atr(h, l, c, 14)

    trades = []
    pos = None
    last_exit = -999

    for i in range(n_bars + 14, n):
        if pos is not None:
            sl = pos['atr'] * sl_atr_mult
            tp = pos['atr'] * tp_atr_mult
            bars_held = i - pos['bar']

            # MaxLoss cap
            cap_limit = cap_atr_mult * pos['atr'] * LOT * PV if cap_atr_mult > 0 else 0
            if cap_limit > 0:
                if pos['dir'] == 'BUY':
                    float_pnl = (c[i] - pos['entry']) * LOT * PV
                else:
                    float_pnl = (pos['entry'] - c[i]) * LOT * PV
                if -float_pnl > cap_limit:
                    pnl = -cap_limit - SPREAD * LOT * PV
                    trades.append({'dir': pos['dir'], 'entry_time': pos['time'].isoformat(),
                                   'exit_time': times[i].isoformat(),
                                   'pnl': pnl, 'reason': 'MaxLossCap', 'bars': bars_held})
                    pos = None; last_exit = i; continue

            if pos['dir'] == 'BUY':
                gain = c[i] - pos['entry']
                if gain >= trail_act_atr * pos['atr']:
                    new_sl = c[i] - trail_dist_atr * pos['atr']
                    pos['trail_sl'] = max(pos.get('trail_sl', pos['entry'] - sl), new_sl)
                sl_price = pos.get('trail_sl', pos['entry'] - sl)
                if l[i] <= sl_price:
                    pnl = (sl_price - pos['entry'] - SPREAD) * LOT * PV
                    trades.append({'dir': 'BUY', 'entry_time': pos['time'].isoformat(),
                                   'exit_time': times[i].isoformat(),
                                   'pnl': pnl, 'reason': 'SL/Trail', 'bars': bars_held})
                    pos = None; last_exit = i; continue
                if h[i] >= pos['entry'] + tp:
                    pnl = (tp - SPREAD) * LOT * PV
                    trades.append({'dir': 'BUY', 'entry_time': pos['time'].isoformat(),
                                   'exit_time': times[i].isoformat(),
                                   'pnl': pnl, 'reason': 'TP', 'bars': bars_held})
                    pos = None; last_exit = i; continue
            else:
                gain = pos['entry'] - c[i]
                if gain >= trail_act_atr * pos['atr']:
                    new_sl = c[i] + trail_dist_atr * pos['atr']
                    pos['trail_sl'] = min(pos.get('trail_sl', pos['entry'] + sl), new_sl)
                sl_price = pos.get('trail_sl', pos['entry'] + sl)
                if h[i] >= sl_price:
                    pnl = (pos['entry'] - sl_price - SPREAD) * LOT * PV
                    trades.append({'dir': 'SELL', 'entry_time': pos['time'].isoformat(),
                                   'exit_time': times[i].isoformat(),
                                   'pnl': pnl, 'reason': 'SL/Trail', 'bars': bars_held})
                    pos = None; last_exit = i; continue
                if l[i] <= pos['entry'] - tp:
                    pnl = (tp - SPREAD) * LOT * PV
                    trades.append({'dir': 'SELL', 'entry_time': pos['time'].isoformat(),
                                   'exit_time': times[i].isoformat(),
                                   'pnl': pnl, 'reason': 'TP', 'bars': bars_held})
                    pos = None; last_exit = i; continue

            if bars_held >= max_hold:
                if pos['dir'] == 'BUY':
                    pnl = (c[i] - pos['entry'] - SPREAD) * LOT * PV
                else:
                    pnl = (pos['entry'] - c[i] - SPREAD) * LOT * PV
                trades.append({'dir': pos['dir'], 'entry_time': pos['time'].isoformat(),
                               'exit_time': times[i].isoformat(),
                               'pnl': pnl, 'reason': 'Time', 'bars': bars_held})
                pos = None; last_exit = i
            continue

        if i - last_exit < 2:
            continue
        if np.isnan(atr[i]) or atr[i] < atr_floor:
            continue

        # Dual Thrust signal: high-close range over n_bars
        hh = max(h[i - j] for j in range(1, n_bars + 1))
        hc = max(c[i - j] for j in range(1, n_bars + 1))
        lc = min(c[i - j] for j in range(1, n_bars + 1))
        ll = min(l[i - j] for j in range(1, n_bars + 1))
        rng = max(hh - lc, hc - ll)
        up_trigger = o[i] + k_up * rng
        dn_trigger = o[i] - k_down * rng

        if c[i] > up_trigger:
            pos = {'dir': 'BUY', 'entry': c[i] + SPREAD / 2, 'bar': i,
                   'time': times[i], 'atr': atr[i]}
        elif c[i] < dn_trigger:
            pos = {'dir': 'SELL', 'entry': c[i] - SPREAD / 2, 'bar': i,
                   'time': times[i], 'atr': atr[i]}

    return trades


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def evaluate(name: str, trades: List[Dict]) -> Dict:
    """Compute full + era + recent stats for a strategy."""
    full = calc_stats(trades, f"{name}_full")
    recent = calc_stats(filter_period(trades, *RECENT_WINDOW), f"{name}_recent7mo")
    era = {}
    for label, (start, end) in ERA_SEGMENTS.items():
        era[label] = calc_stats(filter_period(trades, start, end), f"{name}_{label}")
    return {'full': full, 'recent_7mo': recent, 'era': era}


def fmt_stats(s: Dict) -> str:
    return (f"n={s['n']:>5}  Sharpe={s['sharpe']:>6.3f}  "
            f"PnL=${s['pnl']:>10.0f}  WR={s['win_rate']*100:>5.1f}%  "
            f"PF={s['pf']:>5.2f}  MaxDD=${s['max_dd']:>7.0f}")


def print_report(name: str, r: Dict):
    print(f'\n  === {name} ===', flush=True)
    print(f'    Full:        {fmt_stats(r["full"])}', flush=True)
    print(f'    Recent-7mo:  {fmt_stats(r["recent_7mo"])}', flush=True)
    for era_name, es in r['era'].items():
        print(f'    {era_name:<28} {fmt_stats(es)}', flush=True)


def main():
    print('=' * 80, flush=True)
    print('R244 — Missing Strategies Revalidation (TSMOM + SESS_BO + Dual Thrust)', flush=True)
    print('=' * 80, flush=True)
    stamp(f'Loading H1 from {H1_CSV_PATH}')

    h1 = load_csv(str(H1_CSV_PATH))
    h1 = h1[h1.index <= pd.Timestamp(DATA_END, tz='UTC')]
    stamp(f'H1 bars: {len(h1)}, {h1.index[0]} -> {h1.index[-1]}')

    results = {}

    stamp('Running TSMOM (Score 480/960, SL=3.5, TP=8.0, MH=30, Rule B σ=2.0)...')
    tsmom_trades = bt_tsmom(h1)
    stamp(f'  TSMOM trades: {len(tsmom_trades)}')
    results['TSMOM'] = evaluate('TSMOM', tsmom_trades)
    print_report('TSMOM', results['TSMOM'])
    with open(OUTPUT_DIR / 'tsmom_trades.json', 'w', encoding='utf-8') as f:
        json.dump(tsmom_trades, f, indent=1)

    stamp('Running SESS_BO (GMT12, lookback=4, SL=4.5, TP=4.0, D1 EMA20 ON)...')
    sess_trades = bt_sess_bo(h1)
    stamp(f'  SESS_BO trades: {len(sess_trades)}')
    results['SESS_BO'] = evaluate('SESS_BO', sess_trades)
    print_report('SESS_BO', results['SESS_BO'])
    with open(OUTPUT_DIR / 'sess_bo_trades.json', 'w', encoding='utf-8') as f:
        json.dump(sess_trades, f, indent=1)

    stamp('Running Dual Thrust (k=0.5, n_bars=6, SL=6.0, TP=8.0)...')
    dt_trades = bt_dual_thrust(h1)
    stamp(f'  Dual Thrust trades: {len(dt_trades)}')
    results['DUAL_THRUST'] = evaluate('DUAL_THRUST', dt_trades)
    print_report('DUAL_THRUST', results['DUAL_THRUST'])
    with open(OUTPUT_DIR / 'dual_thrust_trades.json', 'w', encoding='utf-8') as f:
        json.dump(dt_trades, f, indent=1)

    # Save consolidated JSON
    with open(OUTPUT_DIR / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)

    # Write SUMMARY.md
    md = ['# R244 — Missing Strategies Revalidation Report (with MaxLoss Cap)', '',
          '_Coverage: TSMOM + SESS_BO + Dual Thrust (the 3 live strategies R243 did NOT cover)_', '',
          '_Cap mechanism: cap_atr_mult x ATR x LOT x PV (matches BacktestEngine R53 spec)_',
          '_TSMOM cap_atr=6.5 | SESS_BO cap_atr=5.0 | Dual Thrust cap_atr=5.0_', '',
          f'_Data: H1 bid, {h1.index[0].date()} -> {h1.index[-1].date()}_', '']
    for name in ['TSMOM', 'SESS_BO', 'DUAL_THRUST']:
        r = results[name]
        md.append(f'## {name}')
        md.append('')
        md.append('| Window | N | Sharpe | PnL | WR | PF | MaxDD |')
        md.append('|---|---|---|---|---|---|---|')
        for label, s in [('Full', r['full']), ('Recent 7mo', r['recent_7mo'])]:
            md.append(f'| {label} | {s["n"]} | {s["sharpe"]} | ${s["pnl"]:.0f} | '
                      f'{s["win_rate"]*100:.1f}% | {s["pf"]} | ${s["max_dd"]:.0f} |')
        for era_label, es in r['era'].items():
            md.append(f'| {era_label} | {es["n"]} | {es["sharpe"]} | ${es["pnl"]:.0f} | '
                      f'{es["win_rate"]*100:.1f}% | {es["pf"]} | ${es["max_dd"]:.0f} |')
        md.append('')
    md.append('---')
    md.append('_Combine with R243 (Keltner + M30 RSI14) for full 5-strategy live portfolio coverage._')
    (OUTPUT_DIR / 'SUMMARY.md').write_text('\n'.join(md), encoding='utf-8')

    stamp(f'DONE in {(time.time()-T0)/60:.1f} minutes')
    stamp(f'Output: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
