#!/usr/bin/env python3
"""R240b: Squeeze strategy comprehensive validation (remote VPS)
==================================================================

Builds on R240 (which found SHORT failed, LONG passed). Runs:

Phase 1 — Time-window stability: same logic across 1/3/5/10 year windows
                                 for SHORT, LONG, BOTH
Phase 2 — Parameter scan: min_squeeze_bars (2..5), sl_atr (1.0..3.0),
                          trail_act_atr (0.10..0.30) — for each of S/L/B
Phase 3 — Filter validation: add session filter (Asia/London/NY/Evening),
          ADX filter (>20, >25), D1 trend filter (EMA20 slope)

Designed for parallel execution. Uses ProcessPoolExecutor with N workers.
"""
from __future__ import annotations
import sys
import json
import time
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.runner import DataBundle
from backtest.engine import BacktestEngine
import indicators as signals_mod

OUTPUT_DIR = Path("results/r240b_squeeze_comprehensive")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# Signal generator (parameterized)
# ═══════════════════════════════════════════════════════════════

# Mutated per run by the worker — kept in module-level so monkey-patched
# scan_all_signals can read it.
_RUN_CFG = {
    'enabled_long': False,
    'enabled_short': False,
    'min_squeeze_bars': 3,
    'sl_atr': 1.5,
    'trail_act_atr': 0.20,
    'trail_dist_atr': 0.04,
    'min_atr': 0.1,
    # Filters
    'session_filter': None,    # None, 'asia', 'london', 'ny', 'evening'
    'adx_filter': 0,           # 0 = off, else min ADX
    'd1_trend_filter': False,  # require D1 EMA20 slope alignment
}

_RUN_STATE = {
    'count': 0,
    'last_release_bar': None,
}


def _reset_state():
    _RUN_STATE['count'] = 0
    _RUN_STATE['last_release_bar'] = None


def _in_session(hour: int, session: str) -> bool:
    if session is None:
        return True
    if session == 'asia':
        return 0 <= hour <= 7
    if session == 'london':
        return 8 <= hour <= 12
    if session == 'ny':
        return 13 <= hour <= 17
    if session == 'evening':
        return 18 <= hour <= 23
    return True


def squeeze_signal(df: pd.DataFrame, direction: str) -> Optional[Dict]:
    """BB-in-KC squeeze release detector."""
    cfg = _RUN_CFG
    if df is None or len(df) < 50:
        return None

    row = df.iloc[-1]
    sq_val = row.get('squeeze', 0)
    if pd.isna(sq_val):
        return None
    squeeze_now = float(sq_val) > 0.5

    atr_val = row.get('ATR', 0)
    if pd.isna(atr_val):
        _RUN_STATE['count'] = 0
        return None
    atr = float(atr_val)
    if atr <= cfg['min_atr']:
        _RUN_STATE['count'] = 0
        return None

    if squeeze_now:
        _RUN_STATE['count'] += 1
        return None

    saved_count = _RUN_STATE['count']
    _RUN_STATE['count'] = 0

    if saved_count < cfg['min_squeeze_bars']:
        return None

    bar_time = df.index[-1]
    if _RUN_STATE['last_release_bar'] == bar_time:
        return None
    _RUN_STATE['last_release_bar'] = bar_time

    # ── Filters ──
    if cfg['session_filter']:
        hour = pd.Timestamp(bar_time).hour
        if not _in_session(hour, cfg['session_filter']):
            return None

    if cfg['adx_filter'] > 0:
        adx_val = row.get('ADX', 0)
        if pd.isna(adx_val) or float(adx_val) < cfg['adx_filter']:
            return None

    if cfg['d1_trend_filter']:
        # Use EMA100 slope (proxy for D1 trend on H1 bars)
        if len(df) >= 21:
            ema_now = row.get('EMA100', 0)
            ema_prev = df.iloc[-21].get('EMA100', 0)
            if pd.isna(ema_now) or pd.isna(ema_prev) or ema_prev <= 0:
                return None
            slope_up = float(ema_now) > float(ema_prev)
            if direction == 'BUY' and not slope_up:
                return None
            if direction == 'SELL' and slope_up:
                return None

    strategy_name = 'squeeze_long' if direction == 'BUY' else 'squeeze_short'
    sl = round(atr * cfg['sl_atr'], 2)
    close = float(row.get('Close', 0))

    return {
        'strategy': strategy_name,
        'signal': direction,
        'sl': sl,
        'tp': 0,
        'close': close,
        'reason': f"Squeeze release {direction}",
        'trailing_activate': round(atr * cfg['trail_act_atr'], 2),
        'trailing_distance': round(atr * cfg['trail_dist_atr'], 2),
    }


def patched_scan(df, timeframe='H1', h1_adx=None):
    if timeframe != 'H1':
        return []
    signals = []
    if _RUN_CFG['enabled_long']:
        s = squeeze_signal(df, 'BUY')
        if s:
            signals.append(s)
    if _RUN_CFG['enabled_short']:
        s = squeeze_signal(df, 'SELL')
        if s:
            signals.append(s)
    return signals


# ═══════════════════════════════════════════════════════════════
# Stats
# ═══════════════════════════════════════════════════════════════

def calc_stats(trades, strategy_filter=None):
    if strategy_filter:
        trades = [t for t in trades if t.strategy == strategy_filter]
    if not trades:
        return {'n': 0, 'pnl': 0.0, 'sharpe': 0.0, 'win_rate': 0.0,
                'avg_pnl': 0.0, 'max_dd': 0.0, 'median_pnl': 0.0}
    pnls = np.array([t.pnl for t in trades])
    n = len(pnls)
    cum = np.cumsum(pnls)
    dd = np.maximum.accumulate(cum) - cum
    std = max(pnls.std(ddof=1), 1e-9) if n > 1 else 1e-9
    sharpe = float(pnls.mean() / std * np.sqrt(252 * 4)) if n > 1 else 0.0
    return {
        'n': n,
        'pnl': round(float(pnls.sum()), 2),
        'sharpe': round(sharpe, 3),
        'win_rate': round(100 * (pnls > 0).sum() / n, 2),
        'avg_pnl': round(float(pnls.mean()), 2),
        'median_pnl': round(float(np.median(pnls)), 2),
        'max_dd': round(float(dd.max()), 2),
    }


# ═══════════════════════════════════════════════════════════════
# Single backtest run (in-process)
# ═══════════════════════════════════════════════════════════════

def run_single(data, label, *, run_cfg, start_date=None, end_date=None):
    """Run a single backtest with given run_cfg."""
    _RUN_CFG.update(run_cfg)
    _reset_state()

    signals_mod.scan_all_signals = patched_scan
    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False
    from indicators import get_orb_strategy
    get_orb_strategy().reset_daily()

    bt_kwargs = {
        'sl_atr_mult': 0,
        'tp_atr_mult': 0,
        'trailing_activate_atr': _RUN_CFG['trail_act_atr'],
        'trailing_distance_atr': _RUN_CFG['trail_dist_atr'],
        'max_positions': 1,
        'min_lot_size': 0.04,
        'max_lot_size': 0.04,
        'maxloss_cap': 0,
        'live_atr_percentile': True,
    }

    if start_date or end_date:
        bundle = data.slice(start_date or '2015-01-01', end_date or '2027-01-01')
    else:
        bundle = data

    engine = BacktestEngine(bundle.m15_df, bundle.h1_df, **bt_kwargs)
    trades = engine.run()

    return {
        'label': label,
        'cfg': dict(run_cfg),
        'start': start_date,
        'end': end_date,
        'overall': calc_stats(trades),
        'long': calc_stats(trades, 'squeeze_long'),
        'short': calc_stats(trades, 'squeeze_short'),
    }


# ═══════════════════════════════════════════════════════════════
# Phase runners
# ═══════════════════════════════════════════════════════════════

def phase1_time_windows(data):
    """Run SHORT/LONG/BOTH across 1y, 3y, 5y, 10y windows."""
    print('\n' + '=' * 80)
    print('Phase 1: Time-Window Stability')
    print('=' * 80)

    # Use last bar in data to set windows
    end = data.h1_df.index[-1].strftime('%Y-%m-%d')
    end_ts = pd.Timestamp(end)
    windows = {
        '1y': ((end_ts - pd.DateOffset(years=1)).strftime('%Y-%m-%d'), end),
        '3y': ((end_ts - pd.DateOffset(years=3)).strftime('%Y-%m-%d'), end),
        '5y': ((end_ts - pd.DateOffset(years=5)).strftime('%Y-%m-%d'), end),
        '10y': ('2015-01-01', end),
    }

    sides = [
        ('SHORT', {'enabled_long': False, 'enabled_short': True}),
        ('LONG',  {'enabled_long': True, 'enabled_short': False}),
        ('BOTH',  {'enabled_long': True, 'enabled_short': True}),
    ]

    results = []
    base = {'min_squeeze_bars': 3, 'sl_atr': 1.5,
            'trail_act_atr': 0.20, 'trail_dist_atr': 0.04,
            'session_filter': None, 'adx_filter': 0, 'd1_trend_filter': False}
    for side_label, side_cfg in sides:
        for win_label, (s, e) in windows.items():
            cfg = {**base, **side_cfg}
            label = f'{side_label}_{win_label}'
            t0 = time.time()
            res = run_single(data, label, run_cfg=cfg, start_date=s, end_date=e)
            res['time_window'] = win_label
            res['side'] = side_label
            elapsed = time.time() - t0
            ov = res['overall']
            print(f'  {label:<15}  n={ov["n"]:>4}  PnL={ov["pnl"]:>+9.2f}  '
                  f'Sharpe={ov["sharpe"]:>+6.3f}  WR={ov["win_rate"]:>5.1f}%  '
                  f'MaxDD={ov["max_dd"]:>6.2f}  ({elapsed:.1f}s)')
            results.append(res)

    with open(OUTPUT_DIR / 'phase1_time_windows.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    print(f'  -> saved phase1_time_windows.json')
    return results


def phase2_param_scan(data):
    """Scan min_squeeze_bars x sl_atr x trail_act for SHORT and LONG."""
    print('\n' + '=' * 80)
    print('Phase 2: Parameter Scan (SHORT & LONG, 10y data)')
    print('=' * 80)

    grid = list(itertools.product(
        [2, 3, 4, 5],            # min_squeeze_bars
        [1.0, 1.5, 2.0, 3.0],    # sl_atr
        [0.10, 0.20, 0.30],      # trail_act_atr
    ))
    print(f'  Grid size: {len(grid)} combos x 2 sides = {len(grid)*2} runs')

    base = {'trail_dist_atr': 0.04,
            'session_filter': None, 'adx_filter': 0, 'd1_trend_filter': False}
    results = []
    sides = [
        ('SHORT', {'enabled_long': False, 'enabled_short': True}),
        ('LONG',  {'enabled_long': True, 'enabled_short': False}),
    ]
    total = len(grid) * len(sides)
    done = 0
    for side_label, side_cfg in sides:
        for sb, sla, tra in grid:
            cfg = {**base, **side_cfg,
                   'min_squeeze_bars': sb, 'sl_atr': sla, 'trail_act_atr': tra}
            label = f'{side_label}_sb{sb}_sl{sla}_tr{tra}'
            t0 = time.time()
            res = run_single(data, label, run_cfg=cfg)
            res['side'] = side_label
            res['params'] = {'min_squeeze_bars': sb, 'sl_atr': sla, 'trail_act_atr': tra}
            elapsed = time.time() - t0
            ov = res['overall']
            done += 1
            print(f'  [{done:>3}/{total}] {label:<40}  '
                  f'n={ov["n"]:>4}  PnL={ov["pnl"]:>+9.2f}  '
                  f'Sharpe={ov["sharpe"]:>+6.3f}  WR={ov["win_rate"]:>5.1f}%  ({elapsed:.1f}s)')
            results.append(res)

    with open(OUTPUT_DIR / 'phase2_param_scan.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    print(f'  -> saved phase2_param_scan.json')

    # Top 10 by Sharpe per side
    for side in ['SHORT', 'LONG']:
        side_res = [r for r in results if r['side'] == side]
        side_res.sort(key=lambda r: r['overall']['sharpe'], reverse=True)
        print(f'\n  Top 10 {side} by Sharpe:')
        for r in side_res[:10]:
            ov = r['overall']
            p = r['params']
            print(f'    sb={p["min_squeeze_bars"]} sl={p["sl_atr"]} tr={p["trail_act_atr"]}  '
                  f'n={ov["n"]:>4} PnL={ov["pnl"]:>+8.2f} Sharpe={ov["sharpe"]:>+6.3f} '
                  f'WR={ov["win_rate"]:>5.1f}% MaxDD={ov["max_dd"]:>6.2f}')

    return results


def phase3_filters(data):
    """Test filters: session, ADX, D1 trend — for SHORT only."""
    print('\n' + '=' * 80)
    print('Phase 3: Filter Validation (focus on SHORT — can filters rescue it?)')
    print('=' * 80)

    base = {'enabled_long': False, 'enabled_short': True,
            'min_squeeze_bars': 3, 'sl_atr': 1.5,
            'trail_act_atr': 0.20, 'trail_dist_atr': 0.04}

    filters = [
        ('no_filter',         {}),
        ('asia_only',         {'session_filter': 'asia'}),
        ('london_only',       {'session_filter': 'london'}),
        ('ny_only',           {'session_filter': 'ny'}),
        ('evening_only',      {'session_filter': 'evening'}),
        ('adx20',             {'adx_filter': 20}),
        ('adx25',             {'adx_filter': 25}),
        ('d1_trend',          {'d1_trend_filter': True}),
        ('d1_trend+adx20',    {'d1_trend_filter': True, 'adx_filter': 20}),
        ('london+adx20',      {'session_filter': 'london', 'adx_filter': 20}),
        ('ny+adx20',          {'session_filter': 'ny', 'adx_filter': 20}),
    ]
    print(f'  Total filter combos: {len(filters)}')

    results = []
    for label, flt in filters:
        cfg = {**base, **flt}
        t0 = time.time()
        res = run_single(data, label, run_cfg=cfg)
        elapsed = time.time() - t0
        ov = res['overall']
        print(f'  {label:<22}  n={ov["n"]:>4}  PnL={ov["pnl"]:>+9.2f}  '
              f'Sharpe={ov["sharpe"]:>+6.3f}  WR={ov["win_rate"]:>5.1f}%  '
              f'MaxDD={ov["max_dd"]:>6.2f}  ({elapsed:.1f}s)')
        res['filter_label'] = label
        results.append(res)

    with open(OUTPUT_DIR / 'phase3_filters.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str, ensure_ascii=False)
    print(f'  -> saved phase3_filters.json')

    # Also try filters on LONG to see if any boost
    print('\n  Bonus: same filters on LONG side')
    base_long = {**base, 'enabled_long': True, 'enabled_short': False}
    results_long = []
    for label, flt in filters:
        cfg = {**base_long, **flt}
        t0 = time.time()
        res = run_single(data, label, run_cfg=cfg)
        elapsed = time.time() - t0
        ov = res['overall']
        print(f'  LONG_{label:<22}  n={ov["n"]:>4}  PnL={ov["pnl"]:>+9.2f}  '
              f'Sharpe={ov["sharpe"]:>+6.3f}  WR={ov["win_rate"]:>5.1f}%  '
              f'MaxDD={ov["max_dd"]:>6.2f}  ({elapsed:.1f}s)')
        res['filter_label'] = label
        results_long.append(res)

    with open(OUTPUT_DIR / 'phase3_filters_long.json', 'w', encoding='utf-8') as f:
        json.dump(results_long, f, indent=2, default=str, ensure_ascii=False)
    print(f'  -> saved phase3_filters_long.json')

    return results, results_long


def main():
    t_start = time.time()
    print('=' * 80)
    print('R240b: Squeeze Strategy Comprehensive Validation')
    print('=' * 80)

    data = DataBundle.load_default()
    print(f'  Data span: {data.h1_df.index[0]} -> {data.h1_df.index[-1]}')
    print(f'  H1 bars: {len(data.h1_df)}, M15 bars: {len(data.m15_df)}')

    # Phase 1: Time windows
    p1 = phase1_time_windows(data)

    # Phase 2: Param scan
    p2 = phase2_param_scan(data)

    # Phase 3: Filters
    p3_short, p3_long = phase3_filters(data)

    elapsed = time.time() - t_start
    print('\n' + '=' * 80)
    print(f'TOTAL TIME: {elapsed/60:.1f} min ({elapsed:.0f}s)')
    print('=' * 80)


if __name__ == '__main__':
    main()
