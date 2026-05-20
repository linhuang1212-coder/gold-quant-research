#!/usr/bin/env python3
"""R237: Full Strategy Revalidation — BacktestEngine + LIVE_PARITY_KWARGS
==========================================================================
All strategies previously tested with standalone engines (M30BacktestEngine,
H4BacktestEngine, or inline loops) must pass the full validation pipeline:

  Module A: 11 M30 strategies (m30_rsi14 excluded — already validated in R236)
  Module B: 5 H4 strategies (via enhanced H4BacktestEngine with filters)
  Module C: 4 H1 non-Keltner strategies (PSAR, SESS_BO, DualThrust, Chandelier)

Each strategy is evaluated with:
  1. Full-sample daily Sharpe (sqrt(252))
  2. K-Fold 6 validation (>=4/6 positive = PASS)
  3. Holdout OOS (2025-05 onward, retention >= 50%, Sharpe > 0.5)
  4. Era stability (4 eras)

Execution: multiprocessing.Pool for parallel evaluation.
"""
from __future__ import annotations
import sys
import json
import time
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Callable, Optional, Tuple
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.runner import DataBundle, LIVE_PARITY_KWARGS
from backtest.engine import BacktestEngine, TradeRecord
from backtest.m30_engine import load_m30_with_indicators
from backtest.h4_engine import H4BacktestEngine, load_h4_with_indicators

OUTPUT_DIR = Path("results/r237_full_revalidation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HOLDOUT_START = "2025-05-01"

ERA_SEGMENTS = {
    "Pre-COVID (2015-2019)":     ("2015-01-01", "2020-01-01"),
    "COVID+Recovery (2020-2021)": ("2020-01-01", "2022-01-01"),
    "Tightening (2022-2023)":    ("2022-01-01", "2024-01-01"),
    "Recent (2024-2026)":        ("2024-01-01", "2026-06-01"),
}


def pf(msg):
    print(msg, flush=True)


# ═══════════════════════════════════════════════════════════════
# Utility functions
# ═══════════════════════════════════════════════════════════════

def calc_daily_sharpe(trades):
    """Correct daily Sharpe: aggregate PnL by day, annualize with sqrt(252)."""
    if not trades or len(trades) < 5:
        return -999, 0, 0, 0, 0, 0, 0
    pnls = np.array([t.pnl for t in trades])
    n = len(pnls)
    total = float(pnls.sum())
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    win_rate = len(wins) / n
    profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else 99.0

    daily_pnl = {}
    for t in trades:
        day = pd.Timestamp(t.exit_time).date()
        daily_pnl[day] = daily_pnl.get(day, 0.0) + t.pnl
    daily_series = np.array(list(daily_pnl.values()))
    n_days = len(daily_series)
    if n_days < 10:
        return -999, total, win_rate, profit_factor, n, n_days, 0

    daily_mean = float(daily_series.mean())
    daily_std = float(daily_series.std(ddof=1))
    sharpe = daily_mean / max(daily_std, 1e-9) * np.sqrt(252)

    cum = np.cumsum(pnls)
    max_dd = float((np.maximum.accumulate(cum) - cum).max())

    return round(sharpe, 3), round(total, 2), round(win_rate, 4), round(profit_factor, 3), n, n_days, round(max_dd, 2)


def kfold_validate(trades, k=6):
    """K-Fold cross-validation on trade sequence."""
    if len(trades) < k * 5:
        return {'verdict': 'SKIP', 'pass_count': 0, 'total': 0, 'sharpes': []}
    pnls = np.array([t.pnl for t in trades])
    fold_size = len(pnls) // k
    kf_pass = 0
    sharpes = []
    for fold in range(k):
        s = fold * fold_size
        e = s + fold_size if fold < k - 1 else len(pnls)
        fp = pnls[s:e]
        if len(fp) < 3:
            continue
        sh = float(fp.mean() / max(fp.std(ddof=1), 1e-9) * np.sqrt(252))
        sharpes.append(round(sh, 3))
        if sh > 0:
            kf_pass += 1
    verdict = 'PASS' if kf_pass >= 4 else 'FAIL'
    return {'verdict': verdict, 'pass_count': kf_pass, 'total': len(sharpes), 'sharpes': sharpes}


def era_stability(trades):
    """Evaluate strategy across 4 market eras."""
    results = {}
    for era_name, (start, end) in ERA_SEGMENTS.items():
        start_ts = pd.Timestamp(start, tz='UTC')
        end_ts = pd.Timestamp(end, tz='UTC')
        era_trades = [t for t in trades
                      if start_ts <= pd.Timestamp(t.entry_time) < end_ts]
        sharpe, pnl, wr, pf_val, n, _, _ = calc_daily_sharpe(era_trades)
        results[era_name] = {'n': n, 'sharpe': sharpe, 'pnl': pnl, 'win_rate': wr}
    return results


def validate_strategy(trades, strategy_name):
    """Full validation pipeline for a strategy's trades."""
    sharpe, pnl, wr, pf_val, n, n_days, max_dd = calc_daily_sharpe(trades)
    kf = kfold_validate(trades)

    holdout_cutoff = pd.Timestamp(HOLDOUT_START, tz='UTC')
    train_trades = [t for t in trades if pd.Timestamp(t.entry_time) < holdout_cutoff]
    oos_trades = [t for t in trades if pd.Timestamp(t.entry_time) >= holdout_cutoff]
    train_sharpe = calc_daily_sharpe(train_trades)[0]
    oos_sharpe = calc_daily_sharpe(oos_trades)[0]
    oos_n = len(oos_trades)

    if train_sharpe > 0 and oos_sharpe > 0:
        retention = oos_sharpe / train_sharpe
        oos_verdict = 'PASS' if retention >= 0.5 and oos_sharpe > 0.5 else 'FAIL'
    else:
        retention = 0
        oos_verdict = 'FAIL'

    eras = era_stability(trades)
    positive_eras = sum(1 for v in eras.values() if v['sharpe'] > 0)

    final_verdict = 'PASS' if kf['verdict'] == 'PASS' and oos_verdict == 'PASS' else 'FAIL'

    return {
        'strategy': strategy_name,
        'n_trades': n,
        'n_days': n_days,
        'sharpe': sharpe,
        'pnl': pnl,
        'win_rate': wr,
        'profit_factor': pf_val,
        'max_dd': max_dd,
        'kfold': kf,
        'oos_sharpe': oos_sharpe,
        'oos_n': oos_n,
        'oos_retention': round(retention, 3) if retention else 0,
        'oos_verdict': oos_verdict,
        'eras': eras,
        'positive_eras': positive_eras,
        'final_verdict': final_verdict,
    }


# ═══════════════════════════════════════════════════════════════
# MODULE A: M30 Strategies via BacktestEngine
# ═══════════════════════════════════════════════════════════════

def m30_sig_kc_breakout(df):
    if len(df) < 55:
        return None
    row = df.iloc[-1]
    c = float(row['Close'])
    kc_u = float(row.get('KC_upper', c + 9999))
    kc_l = float(row.get('KC_lower', c - 9999))
    if pd.isna(kc_u) or pd.isna(kc_l) or float(row.get('ATR', 0)) <= 0:
        return None
    if c > kc_u:
        return {'strategy': 'm30_kc', 'signal': 'BUY'}
    if c < kc_l:
        return {'strategy': 'm30_kc', 'signal': 'SELL'}
    return None


def m30_sig_ema_fast_cross(df):
    if len(df) < 55:
        return None
    row = df.iloc[-1]
    prev = df.iloc[-2]
    e9 = float(row.get('EMA9', 0))
    e20 = float(row.get('EMA20', 0))
    e9p = float(prev.get('EMA9', 0))
    e20p = float(prev.get('EMA20', 0))
    if e9 == 0 or e20 == 0 or float(row.get('ATR', 0)) <= 0:
        return None
    if e9 > e20 and e9p <= e20p:
        return {'strategy': 'm30_ema_fast', 'signal': 'BUY'}
    if e9 < e20 and e9p >= e20p:
        return {'strategy': 'm30_ema_fast', 'signal': 'SELL'}
    return None


def m30_sig_ema_cross(df):
    if len(df) < 55:
        return None
    row = df.iloc[-1]
    prev = df.iloc[-2]
    e20 = float(row.get('EMA20', 0))
    e50 = float(row.get('EMA50', 0))
    e20p = float(prev.get('EMA20', 0))
    e50p = float(prev.get('EMA50', 0))
    if e20 == 0 or e50 == 0 or float(row.get('ATR', 0)) <= 0:
        return None
    if e20 > e50 and e20p <= e50p:
        return {'strategy': 'm30_ema_cross', 'signal': 'BUY'}
    if e20 < e50 and e20p >= e50p:
        return {'strategy': 'm30_ema_cross', 'signal': 'SELL'}
    return None


def m30_sig_macd_cross(df):
    if len(df) < 55:
        return None
    row = df.iloc[-1]
    prev = df.iloc[-2]
    macd = float(row.get('MACD', 0))
    sig = float(row.get('MACD_signal', 0))
    macd_p = float(prev.get('MACD', 0))
    sig_p = float(prev.get('MACD_signal', 0))
    if float(row.get('ATR', 0)) <= 0:
        return None
    if macd > sig and macd_p <= sig_p:
        return {'strategy': 'm30_macd', 'signal': 'BUY'}
    if macd < sig and macd_p >= sig_p:
        return {'strategy': 'm30_macd', 'signal': 'SELL'}
    return None


def m30_sig_rsi6_extreme(df):
    if len(df) < 55:
        return None
    row = df.iloc[-1]
    rsi6 = float(row.get('RSI6', 50))
    c = float(row['Close'])
    ema200 = float(row.get('EMA200', c))
    if pd.isna(rsi6) or float(row.get('ATR', 0)) <= 0 or pd.isna(ema200):
        return None
    if rsi6 < 15 and c > ema200:
        return {'strategy': 'm30_rsi6', 'signal': 'BUY'}
    if rsi6 > 85 and c < ema200:
        return {'strategy': 'm30_rsi6', 'signal': 'SELL'}
    return None


def m30_sig_cci_momentum(df):
    if len(df) < 55:
        return None
    row = df.iloc[-1]
    cci = float(row.get('CCI', 0))
    if pd.isna(cci) or float(row.get('ATR', 0)) <= 0:
        return None
    if cci > 200:
        return {'strategy': 'm30_cci', 'signal': 'BUY'}
    if cci < -200:
        return {'strategy': 'm30_cci', 'signal': 'SELL'}
    return None


def m30_sig_stochastic(df):
    if len(df) < 55:
        return None
    row = df.iloc[-1]
    prev = df.iloc[-2]
    k = float(row.get('Stoch_K', 50))
    d = float(row.get('Stoch_D', 50))
    k_p = float(prev.get('Stoch_K', 50))
    d_p = float(prev.get('Stoch_D', 50))
    if float(row.get('ATR', 0)) <= 0:
        return None
    if k < 20 and k > d and k_p <= d_p:
        return {'strategy': 'm30_stoch', 'signal': 'BUY'}
    if k > 80 and k < d and k_p >= d_p:
        return {'strategy': 'm30_stoch', 'signal': 'SELL'}
    return None


def m30_sig_bb_squeeze(df):
    if len(df) < 55:
        return None
    row = df.iloc[-1]
    prev = df.iloc[-2]
    c = float(row['Close'])
    bb_u = float(row.get('BB_upper', c + 9999))
    bb_l = float(row.get('BB_lower', c - 9999))
    bw = float(row.get('BB_bandwidth', 1.0))
    bw_p = float(prev.get('BB_bandwidth', 1.0))
    if pd.isna(bb_u) or float(row.get('ATR', 0)) <= 0:
        return None
    if bw > bw_p and bw_p < 0.02:
        if c > bb_u:
            return {'strategy': 'm30_squeeze', 'signal': 'BUY'}
        if c < bb_l:
            return {'strategy': 'm30_squeeze', 'signal': 'SELL'}
    return None


def m30_sig_mean_revert(df):
    if len(df) < 55:
        return None
    row = df.iloc[-1]
    c = float(row['Close'])
    bb_u = float(row.get('BB_upper', c))
    bb_l = float(row.get('BB_lower', c))
    rsi = float(row.get('RSI14', 50))
    if pd.isna(bb_u) or float(row.get('ATR', 0)) <= 0:
        return None
    if c < bb_l and rsi < 25:
        return {'strategy': 'm30_mean_rev', 'signal': 'BUY'}
    if c > bb_u and rsi > 75:
        return {'strategy': 'm30_mean_rev', 'signal': 'SELL'}
    return None


def m30_sig_inside_bar(df):
    if len(df) < 55:
        return None
    row = df.iloc[-1]
    prev = df.iloc[-2]
    pprev = df.iloc[-3]
    h, l = float(row['High']), float(row['Low'])
    ph, pl = float(prev['High']), float(prev['Low'])
    pph, ppl = float(pprev['High']), float(pprev['Low'])
    if float(row.get('ATR', 0)) <= 0:
        return None
    # Inside bar = prev range contained within pprev range
    if ph <= pph and pl >= ppl:
        if h > ph:
            return {'strategy': 'm30_inside_bar', 'signal': 'BUY'}
        if l < pl:
            return {'strategy': 'm30_inside_bar', 'signal': 'SELL'}
    return None


def m30_sig_engulfing(df):
    if len(df) < 55:
        return None
    row = df.iloc[-1]
    prev = df.iloc[-2]
    o, c = float(row['Open']), float(row['Close'])
    po, pc = float(prev['Open']), float(prev['Close'])
    if float(row.get('ATR', 0)) <= 0:
        return None
    # Bullish engulfing
    if pc < po and c > o and c > po and o < pc:
        return {'strategy': 'm30_engulf', 'signal': 'BUY'}
    # Bearish engulfing
    if pc > po and c < o and c < po and o > pc:
        return {'strategy': 'm30_engulf', 'signal': 'SELL'}
    return None


M30_STRATEGIES = [
    ('m30_kc',         m30_sig_kc_breakout,    {'sl': 2.0, 'tp': 4.0, 'trail_act': 0.30, 'trail_dist': 0.08, 'max_hold_m15': 48, 'cooldown': 4}),
    ('m30_ema_fast',   m30_sig_ema_fast_cross, {'sl': 2.0, 'tp': 4.0, 'trail_act': 0.30, 'trail_dist': 0.08, 'max_hold_m15': 48, 'cooldown': 4}),
    ('m30_ema_cross',  m30_sig_ema_cross,      {'sl': 2.0, 'tp': 4.0, 'trail_act': 0.30, 'trail_dist': 0.08, 'max_hold_m15': 48, 'cooldown': 4}),
    ('m30_macd',       m30_sig_macd_cross,     {'sl': 2.0, 'tp': 4.0, 'trail_act': 0.30, 'trail_dist': 0.08, 'max_hold_m15': 48, 'cooldown': 4}),
    ('m30_rsi6',       m30_sig_rsi6_extreme,   {'sl': 3.0, 'tp': 5.0, 'trail_act': 0.30, 'trail_dist': 0.08, 'max_hold_m15': 48, 'cooldown': 4}),
    ('m30_cci',        m30_sig_cci_momentum,   {'sl': 2.0, 'tp': 4.0, 'trail_act': 0.30, 'trail_dist': 0.08, 'max_hold_m15': 48, 'cooldown': 4}),
    ('m30_stoch',      m30_sig_stochastic,     {'sl': 2.0, 'tp': 4.0, 'trail_act': 0.30, 'trail_dist': 0.08, 'max_hold_m15': 48, 'cooldown': 4}),
    ('m30_squeeze',    m30_sig_bb_squeeze,     {'sl': 2.0, 'tp': 4.0, 'trail_act': 0.30, 'trail_dist': 0.08, 'max_hold_m15': 48, 'cooldown': 4}),
    ('m30_mean_rev',   m30_sig_mean_revert,    {'sl': 2.0, 'tp': 4.0, 'trail_act': 0.30, 'trail_dist': 0.08, 'max_hold_m15': 48, 'cooldown': 4}),
    ('m30_inside_bar', m30_sig_inside_bar,     {'sl': 2.0, 'tp': 4.0, 'trail_act': 0.30, 'trail_dist': 0.08, 'max_hold_m15': 48, 'cooldown': 4}),
    ('m30_engulf',     m30_sig_engulfing,      {'sl': 2.0, 'tp': 4.0, 'trail_act': 0.30, 'trail_dist': 0.08, 'max_hold_m15': 48, 'cooldown': 4}),
]


def run_m30_strategy(args):
    """Worker function for M30 strategy validation. Designed for multiprocessing."""
    name, sig_func, params, m15_path, h1_path, m30_path = args
    try:
        # Reload data in worker process
        m15_df = pd.read_pickle(m15_path)
        h1_df = pd.read_pickle(h1_path)
        m30_df = pd.read_pickle(m30_path)

        kwargs = {**LIVE_PARITY_KWARGS}
        kwargs['m30_df'] = m30_df
        kwargs['m30_enabled'] = True
        kwargs['m30_signal_func'] = sig_func
        kwargs['m30_strategy_name'] = name
        kwargs['m30_sl_atr_mult'] = params['sl']
        kwargs['m30_tp_atr_mult'] = params['tp']
        kwargs['m30_trail_activate_atr'] = params['trail_act']
        kwargs['m30_trail_distance_atr'] = params['trail_dist']
        kwargs['m30_max_hold_m15'] = params['max_hold_m15']
        kwargs['m30_cooldown_bars'] = params['cooldown']
        # M30-only mode
        kwargs['keltner_session_adx'] = {
            "asia": (0, 7, 999), "london": (8, 12, 999),
            "ny": (13, 17, 999), "evening": (18, 23, 999),
        }
        kwargs['rsi_adx_filter'] = 0.001
        kwargs['label'] = f'm30_only_{name}'

        engine = BacktestEngine(m15_df, h1_df, **kwargs)
        trades = engine.run()
        m30_trades = [t for t in trades if t.strategy == name]

        result = validate_strategy(m30_trades, name)
        result['engine_stats'] = {
            'm30_total_signals': engine.m30_total_signals,
            'm30_entries': engine.m30_entries,
            'm30_skipped_choppy': engine.m30_skipped_choppy,
            'm30_skipped_atr': engine.m30_skipped_atr,
            'm30_skipped_slot': engine.m30_skipped_slot,
            'm30_skipped_cooldown': engine.m30_skipped_cooldown,
        }
        return result
    except Exception as e:
        return {'strategy': name, 'error': str(e), 'traceback': traceback.format_exc(), 'final_verdict': 'ERROR'}


# ═══════════════════════════════════════════════════════════════
# MODULE B: H4 Strategies via Enhanced H4BacktestEngine
# ═══════════════════════════════════════════════════════════════

def h4_sig_kc_breakout(df):
    if len(df) < 50:
        return None
    row = df.iloc[-1]
    c = float(row['Close'])
    kc_u = float(row.get('KC_upper', c + 9999))
    kc_l = float(row.get('KC_lower', c - 9999))
    if pd.isna(kc_u) or float(row.get('ATR', 0)) <= 0:
        return None
    if c > kc_u:
        return {'strategy': 'h4_kc', 'signal': 'BUY'}
    if c < kc_l:
        return {'strategy': 'h4_kc', 'signal': 'SELL'}
    return None


def h4_sig_ema_cross(df):
    if len(df) < 50:
        return None
    row = df.iloc[-1]
    prev = df.iloc[-2]
    e20 = float(row.get('EMA20', 0))
    e50 = float(row.get('EMA50', 0))
    e20p = float(prev.get('EMA20', 0))
    e50p = float(prev.get('EMA50', 0))
    if e20 == 0 or e50 == 0 or float(row.get('ATR', 0)) <= 0:
        return None
    if e20 > e50 and e20p <= e50p:
        return {'strategy': 'h4_ema_cross', 'signal': 'BUY'}
    if e20 < e50 and e20p >= e50p:
        return {'strategy': 'h4_ema_cross', 'signal': 'SELL'}
    return None


def h4_sig_macd_cross(df):
    if len(df) < 50:
        return None
    row = df.iloc[-1]
    prev = df.iloc[-2]
    macd = float(row.get('MACD', 0))
    sig = float(row.get('MACD_signal', 0))
    macd_p = float(prev.get('MACD', 0))
    sig_p = float(prev.get('MACD_signal', 0))
    if float(row.get('ATR', 0)) <= 0:
        return None
    if macd > sig and macd_p <= sig_p:
        return {'strategy': 'h4_macd', 'signal': 'BUY'}
    if macd < sig and macd_p >= sig_p:
        return {'strategy': 'h4_macd', 'signal': 'SELL'}
    return None


def h4_sig_cci_momentum(df):
    if len(df) < 50:
        return None
    row = df.iloc[-1]
    cci = float(row.get('CCI', 0))
    if pd.isna(cci) or float(row.get('ATR', 0)) <= 0:
        return None
    if cci > 100:
        return {'strategy': 'h4_cci', 'signal': 'BUY'}
    if cci < -100:
        return {'strategy': 'h4_cci', 'signal': 'SELL'}
    return None


def h4_sig_bb_squeeze(df):
    if len(df) < 50:
        return None
    row = df.iloc[-1]
    prev = df.iloc[-2]
    c = float(row['Close'])
    bb_u = float(row.get('BB_upper', c + 9999))
    bb_l = float(row.get('BB_lower', c - 9999))
    bw = float(row.get('BB_bandwidth', 1.0))
    bw_p = float(prev.get('BB_bandwidth', 1.0))
    if pd.isna(bb_u) or float(row.get('ATR', 0)) <= 0:
        return None
    if bw > bw_p and bw_p < 0.03:
        if c > bb_u:
            return {'strategy': 'h4_squeeze', 'signal': 'BUY'}
        if c < bb_l:
            return {'strategy': 'h4_squeeze', 'signal': 'SELL'}
    return None


H4_STRATEGIES = [
    ('h4_kc',        h4_sig_kc_breakout, {'sl': 5.0, 'tp': 6.0, 'trail_act': 0.3, 'trail_dist': 0.08, 'max_hold': 30}),
    ('h4_ema_cross', h4_sig_ema_cross,   {'sl': 3.0, 'tp': 4.0, 'trail_act': 0.3, 'trail_dist': 0.08, 'max_hold': 30}),
    ('h4_macd',      h4_sig_macd_cross,  {'sl': 2.0, 'tp': 6.0, 'trail_act': 0.3, 'trail_dist': 0.08, 'max_hold': 30}),
    ('h4_cci',       h4_sig_cci_momentum, {'sl': 4.0, 'tp': 6.0, 'trail_act': 0.3, 'trail_dist': 0.08, 'max_hold': 30}),
    ('h4_squeeze',   h4_sig_bb_squeeze,  {'sl': 4.0, 'tp': 4.0, 'trail_act': 0.3, 'trail_dist': 0.08, 'max_hold': 30}),
]


def run_h4_strategy(args):
    """Worker function for H4 strategy validation."""
    name, sig_func, params, h4_path = args
    try:
        h4_df = pd.read_pickle(h4_path)

        engine = H4BacktestEngine(
            h4_df,
            [(name, sig_func)],
            sl_atr_mult=params['sl'],
            tp_atr_mult=params['tp'],
            trailing_activate_atr=params['trail_act'],
            trailing_distance_atr=params['trail_dist'],
            max_hold=params['max_hold'],
            cooldown_bars=2,
            max_positions=1,
            spread_cost=0.30,
            lot_size=0.02,
            choppy_gate=True,
            choppy_adx_threshold=20.0,
            atr_pctl_floor=0.30,
            atr_pctl_window=50,
            rule_b_sigma=2.5,
            rule_b_lookback=60,
        )
        trades = engine.run()

        result = validate_strategy(trades, name)
        result['engine_stats'] = {
            'total_signals': engine.total_signals,
            'filtered_choppy': engine.filtered_choppy,
            'filtered_atr_pctl': engine.filtered_atr_pctl,
            'filtered_rule_b': engine.filtered_rule_b,
            'filtered_adx': engine.filtered_adx,
        }
        return result
    except Exception as e:
        return {'strategy': name, 'error': str(e), 'traceback': traceback.format_exc(), 'final_verdict': 'ERROR'}


# ═══════════════════════════════════════════════════════════════
# MODULE C: H1 Non-Keltner Strategies via BacktestEngine
# ═══════════════════════════════════════════════════════════════

H1_STRATEGIES = [
    ('psar',        {'sl': 6.0, 'tp': 6.0, 'trail_act': 0.06, 'trail_dist': 0.01}),
    ('sess_bo',     {'sl': 4.5, 'tp': 4.0, 'trail_act': 0.06, 'trail_dist': 0.01}),
    ('dual_thrust', {'sl': 6.0, 'tp': 8.0, 'trail_act': 0.06, 'trail_dist': 0.01}),
    ('chandelier',  {'sl': 4.5, 'tp': 8.0, 'trail_act': 0.06, 'trail_dist': 0.01}),
]


def run_h1_strategy(args):
    """Worker function for H1 non-Keltner strategy validation."""
    name, params, m15_path, h1_path = args
    try:
        import indicators as signals_mod
        m15_df = pd.read_pickle(m15_path)
        h1_df = pd.read_pickle(h1_path)

        # Enable only this strategy
        signals_mod.PSAR_ENABLED = (name == 'psar')
        signals_mod.SESS_BO_ENABLED = (name == 'sess_bo')
        signals_mod.DUAL_THRUST_ENABLED = (name == 'dual_thrust')
        signals_mod.CHANDELIER_ENABLED = (name == 'chandelier')

        kwargs = {**LIVE_PARITY_KWARGS}
        kwargs['sl_atr_mult'] = params['sl']
        kwargs['tp_atr_mult'] = params['tp']
        kwargs['trailing_activate_atr'] = params['trail_act']
        kwargs['trailing_distance_atr'] = params['trail_dist']
        kwargs['label'] = f'h1_{name}'

        engine = BacktestEngine(m15_df, h1_df, **kwargs)
        trades = engine.run()

        strat_trades = [t for t in trades if t.strategy == name]

        # Reset flags
        signals_mod.PSAR_ENABLED = False
        signals_mod.SESS_BO_ENABLED = False
        signals_mod.DUAL_THRUST_ENABLED = False
        signals_mod.CHANDELIER_ENABLED = False

        result = validate_strategy(strat_trades, name)
        result['total_trades_all'] = len(trades)
        return result
    except Exception as e:
        return {'strategy': name, 'error': str(e), 'traceback': traceback.format_exc(), 'final_verdict': 'ERROR'}


# ═══════════════════════════════════════════════════════════════
# Main Execution
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    pf('=' * 80)
    pf('R237: FULL STRATEGY REVALIDATION')
    pf('  BacktestEngine + LIVE_PARITY_KWARGS + Full Filter Stack')
    pf('  Module A: 11 M30 strategies | Module B: 5 H4 strategies | Module C: 4 H1 strategies')
    pf('=' * 80)

    # ── Load all data and serialize for multiprocessing ──
    pf('\n[1/4] Loading data...')
    data = DataBundle.load_default()
    m30_df = load_m30_with_indicators()
    h4_df = load_h4_with_indicators()
    pf(f'  M15: {len(data.m15_df)} bars | H1: {len(data.h1_df)} bars')
    pf(f'  M30: {len(m30_df)} bars | H4: {len(h4_df)} bars')

    # Serialize DataFrames for worker processes
    cache_dir = OUTPUT_DIR / '_cache'
    cache_dir.mkdir(exist_ok=True)
    m15_path = str(cache_dir / 'm15.pkl')
    h1_path = str(cache_dir / 'h1.pkl')
    m30_path = str(cache_dir / 'm30.pkl')
    h4_path = str(cache_dir / 'h4.pkl')

    data.m15_df.to_pickle(m15_path)
    data.h1_df.to_pickle(h1_path)
    m30_df.to_pickle(m30_path)
    h4_df.to_pickle(h4_path)
    pf('  Data cached for workers.')

    n_workers = min(cpu_count(), 8)
    pf(f'  Workers: {n_workers}')

    all_results = {}

    # ════════════════════════════════════════════════════════════
    # MODULE A: M30 Strategies
    # ════════════════════════════════════════════════════════════
    pf(f'\n\n{"="*80}')
    pf('MODULE A: M30 STRATEGIES (11 strategies via BacktestEngine)')
    pf(f'{"="*80}')

    m30_args = [
        (name, sig_func, params, m15_path, h1_path, m30_path)
        for name, sig_func, params in M30_STRATEGIES
    ]

    pf(f'\n  Running {len(m30_args)} M30 strategies in parallel...')
    t1 = time.time()
    with Pool(n_workers) as pool:
        m30_results = pool.map(run_m30_strategy, m30_args)
    pf(f'  Module A complete in {time.time()-t1:.0f}s')

    pf(f'\n  {"Strategy":<16} {"Verdict":<8} {"Sharpe":>8} {"N":>5} {"KF":>6} {"OOS_Sh":>8} {"PnL":>10}')
    pf(f'  {"-"*16} {"-"*8} {"-"*8} {"-"*5} {"-"*6} {"-"*8} {"-"*10}')
    for r in m30_results:
        if 'error' in r:
            pf(f'  {r["strategy"]:<16} ERROR    {r["error"][:40]}')
        else:
            kf_str = f'{r["kfold"]["pass_count"]}/{r["kfold"]["total"]}'
            pf(f'  {r["strategy"]:<16} {r["final_verdict"]:<8} {r["sharpe"]:>8.3f} {r["n_trades"]:>5} {kf_str:>6} {r["oos_sharpe"]:>8.3f} {r["pnl"]:>10.1f}')
    all_results['m30'] = m30_results

    # ════════════════════════════════════════════════════════════
    # MODULE B: H4 Strategies
    # ════════════════════════════════════════════════════════════
    pf(f'\n\n{"="*80}')
    pf('MODULE B: H4 STRATEGIES (5 strategies via enhanced H4BacktestEngine)')
    pf(f'{"="*80}')

    h4_args = [
        (name, sig_func, params, h4_path)
        for name, sig_func, params in H4_STRATEGIES
    ]

    pf(f'\n  Running {len(h4_args)} H4 strategies in parallel...')
    t2 = time.time()
    with Pool(n_workers) as pool:
        h4_results = pool.map(run_h4_strategy, h4_args)
    pf(f'  Module B complete in {time.time()-t2:.0f}s')

    pf(f'\n  {"Strategy":<16} {"Verdict":<8} {"Sharpe":>8} {"N":>5} {"KF":>6} {"OOS_Sh":>8} {"PnL":>10}')
    pf(f'  {"-"*16} {"-"*8} {"-"*8} {"-"*5} {"-"*6} {"-"*8} {"-"*10}')
    for r in h4_results:
        if 'error' in r:
            pf(f'  {r["strategy"]:<16} ERROR    {r["error"][:40]}')
        else:
            kf_str = f'{r["kfold"]["pass_count"]}/{r["kfold"]["total"]}'
            pf(f'  {r["strategy"]:<16} {r["final_verdict"]:<8} {r["sharpe"]:>8.3f} {r["n_trades"]:>5} {kf_str:>6} {r["oos_sharpe"]:>8.3f} {r["pnl"]:>10.1f}')
    all_results['h4'] = h4_results

    # ════════════════════════════════════════════════════════════
    # MODULE C: H1 Non-Keltner Strategies
    # ════════════════════════════════════════════════════════════
    pf(f'\n\n{"="*80}')
    pf('MODULE C: H1 NON-KELTNER STRATEGIES (4 strategies via BacktestEngine)')
    pf(f'{"="*80}')

    h1_args = [
        (name, params, m15_path, h1_path)
        for name, params in H1_STRATEGIES
    ]

    pf(f'\n  Running {len(h1_args)} H1 strategies sequentially (shared indicators module)...')
    t3 = time.time()
    h1_results = []
    for args in h1_args:
        pf(f'    {args[0]}...', )
        r = run_h1_strategy(args)
        h1_results.append(r)
    pf(f'  Module C complete in {time.time()-t3:.0f}s')

    pf(f'\n  {"Strategy":<16} {"Verdict":<8} {"Sharpe":>8} {"N":>5} {"KF":>6} {"OOS_Sh":>8} {"PnL":>10}')
    pf(f'  {"-"*16} {"-"*8} {"-"*8} {"-"*5} {"-"*6} {"-"*8} {"-"*10}')
    for r in h1_results:
        if 'error' in r:
            pf(f'  {r["strategy"]:<16} ERROR    {r["error"][:40]}')
        else:
            kf_str = f'{r["kfold"]["pass_count"]}/{r["kfold"]["total"]}'
            pf(f'  {r["strategy"]:<16} {r["final_verdict"]:<8} {r["sharpe"]:>8.3f} {r["n_trades"]:>5} {kf_str:>6} {r["oos_sharpe"]:>8.3f} {r["pnl"]:>10.1f}')
    all_results['h1'] = h1_results

    # ════════════════════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════════════════════
    pf(f'\n\n{"="*80}')
    pf('FINAL SUMMARY')
    pf(f'{"="*80}')

    all_strats = m30_results + h4_results + h1_results
    passed = [r for r in all_strats if r.get('final_verdict') == 'PASS']
    failed = [r for r in all_strats if r.get('final_verdict') == 'FAIL']
    errors = [r for r in all_strats if r.get('final_verdict') == 'ERROR']

    pf(f'\n  Total: {len(all_strats)} strategies')
    pf(f'  PASS:  {len(passed)}')
    pf(f'  FAIL:  {len(failed)}')
    pf(f'  ERROR: {len(errors)}')

    if passed:
        pf(f'\n  PASSED strategies (deployment candidates):')
        for r in passed:
            pf(f'    {r["strategy"]:<16} Sharpe={r["sharpe"]:.3f}, OOS={r["oos_sharpe"]:.3f}, N={r["n_trades"]}')

    if failed:
        pf(f'\n  FAILED strategies:')
        for r in failed:
            pf(f'    {r["strategy"]:<16} Sharpe={r.get("sharpe","?")}, KF={r.get("kfold",{}).get("verdict","?")}, OOS={r.get("oos_verdict","?")}')

    if errors:
        pf(f'\n  ERROR strategies:')
        for r in errors:
            pf(f'    {r["strategy"]:<16} {r.get("error","unknown")}')

    elapsed = time.time() - t0
    pf(f'\n  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)')

    # Save results
    output_file = OUTPUT_DIR / 'r237_results.json'

    # Convert non-serializable types
    def serialize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=serialize)
    pf(f'\n  Results saved: {output_file}')

    # Cleanup cache
    import shutil
    shutil.rmtree(cache_dir, ignore_errors=True)

    pf('\nDone.')


if __name__ == '__main__':
    main()
