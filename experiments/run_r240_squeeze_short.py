#!/usr/bin/env python3
"""R240: Squeeze Short (BB-in-KC release) — Backtest & Validation
==================================================================

Goal: Validate R23-S1 short-leg as a standalone live strategy.

Paper trading (Apr-May 2026) shows the SHORT leg of the squeeze straddle
is the profitable side (12 trades, +$198, 92% WR) while the LONG leg is
loss-making (-$86). This script runs a full historical backtest to
determine whether that asymmetry holds long-term or is just a recent
market regime artifact.

Phase 1: Backtest BB-in-KC squeeze release (>=3 squeeze bars then release)
         - Variant A: SHORT only on release (bear continuation hypothesis)
         - Variant B: LONG only on release (bull continuation hypothesis)
         - Variant C: Both legs (straddle, current paper behavior)

Phase 2: Per-period breakdown (yearly stats, full vs recent regimes)
Phase 3: Signal overlap with Keltner (for correlation-check todo)

Engine: BacktestEngine via monkey-patched scan_all_signals.
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
import indicators as signals_mod

OUTPUT_DIR = Path("results/r240_squeeze_short")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# Squeeze release signal logic
# ═══════════════════════════════════════════════════════════════
# State across scan_all_signals calls (engine processes bar-by-bar)
_squeeze_state = {
    'count': 0,
    'last_release_bar': None,
}

# Config — controlled per variant
SQUEEZE_CFG = {
    'enabled_long': False,
    'enabled_short': False,
    'min_squeeze_bars': 3,
    'sl_atr': 1.5,          # paper s1: 1.5 ATR tight SL
    'tp_atr': 0.0,          # no fixed TP, trail handles exit
    'trail_act_atr': 0.20,  # paper s1: 0.20 / 0.04
    'trail_dist_atr': 0.04,
    'min_atr': 0.1,
}


def _reset_squeeze_state():
    _squeeze_state['count'] = 0
    _squeeze_state['last_release_bar'] = None


def check_squeeze_release_signal(df: pd.DataFrame, direction: str) -> Optional[Dict]:
    """Detect BB-in-KC squeeze release. Returns signal dict or None.

    direction: 'BUY' or 'SELL' — which side to fire on release.
    Mirrors the paper logic in gold-quant-trading/paper_trader.py s1_*
    """
    if df is None or len(df) < 50:
        return None

    row = df.iloc[-1]
    sq_val = row.get('squeeze', 0)
    if pd.isna(sq_val):
        return None
    squeeze_now = float(sq_val) > 0.5

    atr_val = row.get('ATR', 0)
    if pd.isna(atr_val):
        _squeeze_state['count'] = 0
        return None
    atr = float(atr_val)
    if atr <= SQUEEZE_CFG['min_atr']:
        _squeeze_state['count'] = 0
        return None

    if squeeze_now:
        _squeeze_state['count'] += 1
        return None

    saved_count = _squeeze_state['count']
    _squeeze_state['count'] = 0

    if saved_count < SQUEEZE_CFG['min_squeeze_bars']:
        return None

    bar_time = df.index[-1]
    if _squeeze_state['last_release_bar'] == bar_time:
        return None
    _squeeze_state['last_release_bar'] = bar_time

    strategy_name = 'squeeze_long' if direction == 'BUY' else 'squeeze_short'
    sl = round(atr * SQUEEZE_CFG['sl_atr'], 2)
    tp = round(atr * SQUEEZE_CFG['tp_atr'], 2) if SQUEEZE_CFG['tp_atr'] > 0 else 0
    close = float(row.get('Close', 0))

    return {
        'strategy': strategy_name,
        'signal': direction,
        'sl': sl,
        'tp': tp,
        'close': close,
        'reason': f"Squeeze release {direction} after {saved_count} bars",
        # Custom trail params (engine reads these via signal dict)
        'trailing_activate': round(atr * SQUEEZE_CFG['trail_act_atr'], 2),
        'trailing_distance': round(atr * SQUEEZE_CFG['trail_dist_atr'], 2),
    }


# ═══════════════════════════════════════════════════════════════
# Monkey-patch scan_all_signals — squeeze-only variant
# ═══════════════════════════════════════════════════════════════

_original_scan = signals_mod.scan_all_signals


def patched_scan_squeeze_only(df, timeframe='H1', h1_adx=None):
    """Returns ONLY squeeze signals — disable Keltner/etc for isolation tests."""
    signals = []
    if timeframe == 'H1':
        if SQUEEZE_CFG['enabled_long']:
            sig = check_squeeze_release_signal(df, 'BUY')
            if sig:
                signals.append(sig)
        if SQUEEZE_CFG['enabled_short']:
            sig = check_squeeze_release_signal(df, 'SELL')
            if sig:
                signals.append(sig)
    return signals


def patched_scan_squeeze_plus_keltner(df, timeframe='H1', h1_adx=None):
    """Squeeze + all existing strategies — for overlap analysis."""
    signals = _original_scan(df, timeframe, h1_adx)
    if timeframe == 'H1':
        if SQUEEZE_CFG['enabled_long']:
            sig = check_squeeze_release_signal(df, 'BUY')
            if sig:
                signals.append(sig)
        if SQUEEZE_CFG['enabled_short']:
            sig = check_squeeze_release_signal(df, 'SELL')
            if sig:
                signals.append(sig)
    return signals


# ═══════════════════════════════════════════════════════════════
# Stats
# ═══════════════════════════════════════════════════════════════

def calc_stats(trades, strategy_filter=None):
    if strategy_filter:
        trades = [t for t in trades if t.strategy == strategy_filter]
    if not trades:
        return {'n': 0, 'pnl': 0, 'sharpe': 0, 'win_rate': 0,
                'avg_pnl': 0, 'max_dd': 0, 'median_pnl': 0}
    pnls = np.array([t.pnl for t in trades])
    n = len(pnls)
    cum = np.cumsum(pnls)
    dd = np.maximum.accumulate(cum) - cum
    sharpe = float(pnls.mean() / max(pnls.std(ddof=1), 1e-9) * np.sqrt(252 * 4)) if n > 1 else 0
    return {
        'n': n,
        'pnl': round(float(pnls.sum()), 2),
        'sharpe': round(sharpe, 3),
        'win_rate': round(100 * (pnls > 0).sum() / n, 2),
        'avg_pnl': round(float(pnls.mean()), 2),
        'median_pnl': round(float(np.median(pnls)), 2),
        'max_dd': round(float(dd.max()), 2),
    }


def calc_yearly_stats(trades, strategy_filter=None):
    if strategy_filter:
        trades = [t for t in trades if t.strategy == strategy_filter]
    by_year = {}
    for t in trades:
        y = pd.Timestamp(t.entry_time).year
        by_year.setdefault(y, []).append(t)
    return {y: calc_stats(yt) for y, yt in sorted(by_year.items())}


def save(name, data):
    p = OUTPUT_DIR / f'{name}.json'
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str, ensure_ascii=False)
    print(f'  -> saved {p}')


def run_squeeze_variant(data, label, *, enabled_long, enabled_short):
    """Run a squeeze-only backtest (no Keltner-specific filters)."""
    global _squeeze_state
    _reset_squeeze_state()

    SQUEEZE_CFG['enabled_long'] = enabled_long
    SQUEEZE_CFG['enabled_short'] = enabled_short

    signals_mod.scan_all_signals = patched_scan_squeeze_only

    # Minimal kwargs: drop all Keltner-specific gates (session ADX, choppy
    # intraday, rsi_adx filter, regime_config) because they would block
    # squeeze entries that aren't Keltner-driven.
    kwargs = {
        'sl_atr_mult': 0,           # signal provides its own SL
        'tp_atr_mult': 0,           # no fixed TP, trail handles exit
        'trailing_activate_atr': SQUEEZE_CFG['trail_act_atr'],
        'trailing_distance_atr': SQUEEZE_CFG['trail_dist_atr'],
        'max_positions': 1,
        'min_lot_size': 0.04,
        'max_lot_size': 0.04,
        'maxloss_cap': 0,
        'live_atr_percentile': True,
    }

    print(f'\n  Running variant: {label}')
    print(f'    enabled_long={enabled_long}, enabled_short={enabled_short}')

    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False
    from indicators import get_orb_strategy
    get_orb_strategy().reset_daily()

    engine = BacktestEngine(data.m15_df, data.h1_df, **kwargs)
    trades = engine.run()

    stats_all = calc_stats(trades)
    stats_long = calc_stats(trades, 'squeeze_long')
    stats_short = calc_stats(trades, 'squeeze_short')
    yearly = calc_yearly_stats(trades)

    return {
        'label': label,
        'enabled_long': enabled_long,
        'enabled_short': enabled_short,
        'overall': stats_all,
        'long_only': stats_long,
        'short_only': stats_short,
        'yearly': yearly,
        'n_trades': len(trades),
    }


def main():
    t_start = time.time()
    print('=' * 80)
    print('R240: Squeeze Short Backtest')
    print('=' * 80)

    data = DataBundle.load_default()
    print(f'  Data span: {data.h1_df.index[0]} -> {data.h1_df.index[-1]}')
    print(f'  H1 bars: {len(data.h1_df)}, M15 bars: {len(data.m15_df)}')

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Squeeze-only variants
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 1: Squeeze variants (isolation)')
    print('=' * 80)

    results = {}

    # Variant A: SHORT only
    results['short_only'] = run_squeeze_variant(
        data, 'SHORT_only', enabled_long=False, enabled_short=True
    )

    # Variant B: LONG only
    results['long_only'] = run_squeeze_variant(
        data, 'LONG_only', enabled_long=True, enabled_short=False
    )

    # Variant C: Both (straddle-like, though single-position cap limits to one direction per release)
    results['both'] = run_squeeze_variant(
        data, 'BOTH', enabled_long=True, enabled_short=True
    )

    print('\n' + '=' * 80)
    print('Summary:')
    print(f'  {"Variant":<15} {"N":>6} {"PnL":>10} {"Sharpe":>8} {"WR%":>7} {"AvgPnL":>8} {"MaxDD":>9}')
    for key, res in results.items():
        s = res['overall']
        print(f'  {res["label"]:<15} {s["n"]:>6} {s["pnl"]:>10.2f} {s["sharpe"]:>8.3f} '
              f'{s["win_rate"]:>6.1f}% {s["avg_pnl"]:>8.2f} {s["max_dd"]:>9.2f}')

    save('phase1_squeeze_variants', results)

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: Squeeze short yearly breakdown
    # ═══════════════════════════════════════════════════════════════
    print('\n' + '=' * 80)
    print('Phase 2: Squeeze Short — Yearly Breakdown')
    print('=' * 80)
    yearly = results['short_only']['yearly']
    print(f'  {"Year":<6} {"N":>5} {"PnL":>10} {"Sharpe":>8} {"WR%":>7} {"AvgPnL":>8}')
    for y, s in yearly.items():
        print(f'  {y:<6} {s["n"]:>5} {s["pnl"]:>10.2f} {s["sharpe"]:>8.3f} '
              f'{s["win_rate"]:>6.1f}% {s["avg_pnl"]:>8.2f}')

    elapsed = time.time() - t_start
    print(f'\n  Total time: {elapsed:.1f}s')

    # Decision summary
    print('\n' + '=' * 80)
    print('Decision Summary (Success criteria: 10y Sharpe>=0.5, WR>=65%, MaxDD<=$500)')
    print('=' * 80)
    short_stats = results['short_only']['overall']
    long_stats = results['long_only']['overall']
    print(f'  SHORT: Sharpe={short_stats["sharpe"]:.3f}, WR={short_stats["win_rate"]:.1f}%, '
          f'MaxDD=${short_stats["max_dd"]:.2f}, n={short_stats["n"]}')
    print(f'  LONG : Sharpe={long_stats["sharpe"]:.3f}, WR={long_stats["win_rate"]:.1f}%, '
          f'MaxDD=${long_stats["max_dd"]:.2f}, n={long_stats["n"]}')

    short_pass = (short_stats['sharpe'] >= 0.5
                  and short_stats['win_rate'] >= 65
                  and short_stats['max_dd'] <= 500)
    print(f'\n  Short-only deploy decision: {"PASS" if short_pass else "FAIL"}')

    print('=' * 80)


if __name__ == '__main__':
    main()
