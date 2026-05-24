#!/usr/bin/env python3
"""R241: M30 RSI14 slot analysis — how much does m30 get blocked by Keltner?
=============================================================================

Phase 1: Run m30_rsi14 in isolation (no Keltner), collect signal timestamps
Phase 2: Run Keltner in isolation, collect entry timestamps  
Phase 3: Overlap analysis — what % of m30 signals occur within ±2h of a Keltner entry?
Phase 4: Test max_positions 5 vs 6 vs "reserved slot" — how many more m30 trades fire?
"""
from __future__ import annotations
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.runner import DataBundle, LIVE_PARITY_KWARGS
from backtest.engine import BacktestEngine
from backtest.m30_engine import load_m30_with_indicators
import indicators as signals_mod

OUTPUT_DIR = Path("results/r241_m30_slot")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def m30_sig(df):
    """M30 RSI14 trend signal — same as live."""
    if len(df) < 55:
        return None
    row = df.iloc[-1]
    rsi = float(row.get('RSI14', 50))
    c = float(row['Close'])
    ema50 = float(row.get('EMA50', c))
    slope = float(row.get('EMA50_slope', 0))
    if pd.isna(rsi) or float(row.get('ATR', 0)) <= 0:
        return None
    if rsi < 30 and c > ema50 and slope > 0:
        return {'strategy': 'm30_rsi14', 'signal': 'BUY'}
    if rsi > 70 and c < ema50 and slope < 0:
        return {'strategy': 'm30_rsi14', 'signal': 'SELL'}
    return None


def calc_stats(trades, strat_filter=None):
    if strat_filter:
        trades = [t for t in trades if t.strategy == strat_filter]
    if not trades:
        return {'n': 0, 'pnl': 0, 'sharpe': 0, 'win_rate': 0, 'max_dd': 0}
    pnls = np.array([t.pnl for t in trades])
    n = len(pnls)
    cum = np.cumsum(pnls)
    dd = np.maximum.accumulate(cum) - cum
    std = max(pnls.std(ddof=1), 1e-9) if n > 1 else 1e-9
    return {
        'n': n,
        'pnl': round(float(pnls.sum()), 2),
        'sharpe': round(float(pnls.mean() / std * np.sqrt(252 * 4)), 3),
        'win_rate': round(100 * (pnls > 0).sum() / n, 1),
        'max_dd': round(float(dd.max()), 2),
    }


def run_m30_isolated(data, m30_df):
    """Phase 1: M30 RSI14 isolated (no Keltner competition)."""
    kwargs = {**LIVE_PARITY_KWARGS}
    kwargs['max_positions'] = 1
    kwargs['m30_enabled'] = True
    kwargs['m30_df'] = m30_df
    kwargs['m30_signal_func'] = m30_sig
    kwargs['m30_sl_atr_mult'] = 8.0
    kwargs['m30_tp_atr_mult'] = 8.0

    # Disable all H1 strategies — monkey-patch scan to return nothing
    orig = signals_mod.scan_all_signals
    signals_mod.scan_all_signals = lambda *a, **kw: []

    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False
    from indicators import get_orb_strategy
    get_orb_strategy().reset_daily()

    engine = BacktestEngine(data.m15_df, data.h1_df, **kwargs)
    trades = engine.run()

    signals_mod.scan_all_signals = orig

    m30_trades = [t for t in trades if t.strategy == 'm30_rsi14']
    return m30_trades


def run_keltner_isolated(data):
    """Phase 2: Keltner isolated (standard live parity)."""
    kwargs = {**LIVE_PARITY_KWARGS}
    kwargs['max_positions'] = 1

    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False
    from indicators import get_orb_strategy
    get_orb_strategy().reset_daily()

    engine = BacktestEngine(data.m15_df, data.h1_df, **kwargs)
    trades = engine.run()

    kc_trades = [t for t in trades if t.strategy == 'keltner']
    return kc_trades


def run_combined(data, m30_df, max_pos):
    """Phase 4: Combined run with given max_positions."""
    kwargs = {**LIVE_PARITY_KWARGS}
    kwargs['max_positions'] = max_pos
    kwargs['m30_enabled'] = True
    kwargs['m30_df'] = m30_df
    kwargs['m30_signal_func'] = m30_sig
    kwargs['m30_sl_atr_mult'] = 8.0
    kwargs['m30_tp_atr_mult'] = 8.0

    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False
    from indicators import get_orb_strategy
    get_orb_strategy().reset_daily()

    engine = BacktestEngine(data.m15_df, data.h1_df, **kwargs)
    trades = engine.run()
    return trades


def overlap_analysis(m30_trades, kc_trades, window_hours=2):
    """Phase 3: What % of m30 entries fall within ±window_hours of a Keltner entry?"""
    if not m30_trades or not kc_trades:
        return {'overlap_pct': 0, 'overlap_count': 0, 'total_m30': len(m30_trades)}

    kc_times = pd.DatetimeIndex(
        sorted([pd.Timestamp(t.entry_time).tz_localize(None)
                if pd.Timestamp(t.entry_time).tz else pd.Timestamp(t.entry_time)
                for t in kc_trades])
    )

    overlap = 0
    for t in m30_trades:
        ts = pd.Timestamp(t.entry_time)
        if ts.tz:
            ts = ts.tz_localize(None)
        diffs_h = np.abs((kc_times - ts).total_seconds() / 3600.0)
        if diffs_h.min() <= window_hours:
            overlap += 1

    return {
        'overlap_count': overlap,
        'total_m30': len(m30_trades),
        'total_kc': len(kc_trades),
        'overlap_pct': round(100 * overlap / len(m30_trades), 1),
    }


def main():
    t0 = time.time()
    print('=' * 80)
    print('R241: M30 RSI14 Slot Analysis')
    print('=' * 80)

    data = DataBundle.load_default()
    print(f'  H1: {len(data.h1_df)} bars, M15: {len(data.m15_df)} bars')

    # Load M30 data
    print('Loading M30 data...')
    m30_df = load_m30_with_indicators()
    print(f'  M30: {len(m30_df)} bars')

    # ── Phase 1: M30 isolated ──
    print('\n' + '=' * 80)
    print('Phase 1: M30 RSI14 isolated (no Keltner)')
    print('=' * 80)
    m30_iso = run_m30_isolated(data, m30_df)
    s = calc_stats(m30_iso)
    print(f'  M30 isolated: n={s["n"]} PnL={s["pnl"]:+.2f} Sharpe={s["sharpe"]:+.3f} WR={s["win_rate"]}% MaxDD={s["max_dd"]}')

    # ── Phase 2: Keltner isolated ──
    print('\n' + '=' * 80)
    print('Phase 2: Keltner isolated')
    print('=' * 80)
    kc_iso = run_keltner_isolated(data)
    s = calc_stats(kc_iso)
    print(f'  Keltner isolated: n={s["n"]} PnL={s["pnl"]:+.2f} Sharpe={s["sharpe"]:+.3f} WR={s["win_rate"]}%')

    # ── Phase 3: Overlap ──
    print('\n' + '=' * 80)
    print('Phase 3: Signal overlap')
    print('=' * 80)
    for window in [1, 2, 4, 6]:
        ov = overlap_analysis(m30_iso, kc_iso, window_hours=window)
        print(f'  ±{window}h: {ov["overlap_count"]}/{ov["total_m30"]} = {ov["overlap_pct"]}% overlap')

    # ── Phase 4: Combined with different max_positions ──
    print('\n' + '=' * 80)
    print('Phase 4: Combined runs with different MAX_POSITIONS')
    print('=' * 80)
    results = {}
    for mp in [1, 2, 3, 5, 6]:
        print(f'\n  MAX_POSITIONS = {mp}')
        trades = run_combined(data, m30_df, mp)
        m30_s = calc_stats(trades, 'm30_rsi14')
        kc_s = calc_stats(trades, 'keltner')
        all_s = calc_stats(trades)
        print(f'    Keltner : n={kc_s["n"]:>5} PnL={kc_s["pnl"]:>+10.2f} Sharpe={kc_s["sharpe"]:>+7.3f} WR={kc_s["win_rate"]:>5.1f}%')
        print(f'    M30     : n={m30_s["n"]:>5} PnL={m30_s["pnl"]:>+10.2f} Sharpe={m30_s["sharpe"]:>+7.3f} WR={m30_s["win_rate"]:>5.1f}%')
        print(f'    Combined: n={all_s["n"]:>5} PnL={all_s["pnl"]:>+10.2f} Sharpe={all_s["sharpe"]:>+7.3f}')
        results[f'mp{mp}'] = {
            'max_positions': mp,
            'keltner': kc_s, 'm30': m30_s, 'combined': all_s,
            'skipped_m30_slot': getattr(trades, 'm30_skipped_slot', None),
        }

    # ── Save ──
    all_results = {
        'm30_isolated': calc_stats(m30_iso),
        'kc_isolated': calc_stats(kc_iso),
        'overlap_2h': overlap_analysis(m30_iso, kc_iso, 2),
        'overlap_4h': overlap_analysis(m30_iso, kc_iso, 4),
        'combined_by_mp': results,
    }
    with open(OUTPUT_DIR / 'r241_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f'\n  Saved {OUTPUT_DIR}/r241_results.json')

    elapsed = time.time() - t0
    print(f'\nTotal time: {elapsed/60:.1f} min ({elapsed:.0f}s)')


if __name__ == '__main__':
    main()
