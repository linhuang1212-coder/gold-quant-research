#!/usr/bin/env python3
"""R240c: Squeeze LONG vs Keltner signal overlap analysis.

Runs the live-parity engine with both Keltner (active live strategy) and
squeeze_long enabled simultaneously, then compares entry timestamps to
measure how independent the two strategies are.

Overlap definition: a squeeze_long entry is "overlapping" if there is a
Keltner BUY entry within +/- 2 H1 bars. Low overlap (<20%) means squeeze_long
adds new tradable signals.
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.runner import DataBundle, LIVE_PARITY_KWARGS
from backtest.engine import BacktestEngine
import indicators as signals_mod

OUTPUT_DIR = Path("results/r240c_correlation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


_RUN_STATE = {'count': 0, 'last_release_bar': None}


def squeeze_long_signal(df: pd.DataFrame):
    """LONG-only squeeze release signal with optimal params (sb=3, sl=2.0)."""
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
    if atr <= 0.1:
        _RUN_STATE['count'] = 0
        return None

    if squeeze_now:
        _RUN_STATE['count'] += 1
        return None

    saved = _RUN_STATE['count']
    _RUN_STATE['count'] = 0
    if saved < 3:
        return None

    bar_time = df.index[-1]
    if _RUN_STATE['last_release_bar'] == bar_time:
        return None
    _RUN_STATE['last_release_bar'] = bar_time

    return {
        'strategy': 'squeeze_long', 'signal': 'BUY',
        'sl': round(atr * 2.0, 2), 'tp': 0,
        'close': float(row.get('Close', 0)),
        'reason': f"Squeeze release BUY after {saved} bars",
        'trailing_activate': round(atr * 0.20, 2),
        'trailing_distance': round(atr * 0.04, 2),
    }


_orig_scan = signals_mod.scan_all_signals


def patched_scan(df, timeframe='H1', h1_adx=None):
    signals = _orig_scan(df, timeframe, h1_adx)
    if timeframe == 'H1':
        s = squeeze_long_signal(df)
        if s:
            signals.append(s)
    return signals


def main():
    print('=' * 80)
    print('R240c: Squeeze LONG vs Keltner signal overlap')
    print('=' * 80)

    data = DataBundle.load_default()
    signals_mod.scan_all_signals = patched_scan
    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False
    from indicators import get_orb_strategy
    get_orb_strategy().reset_daily()

    kwargs = {**LIVE_PARITY_KWARGS}
    kwargs['max_positions'] = 5  # Allow both to enter

    print('Running combined backtest (Keltner + squeeze_long)...')
    engine = BacktestEngine(data.m15_df, data.h1_df, **kwargs)
    trades = engine.run()
    print(f'  Total trades: {len(trades)}')

    by_strat = {}
    for t in trades:
        by_strat.setdefault(t.strategy, []).append(t)
    for s, lst in sorted(by_strat.items()):
        pnls = np.array([t.pnl for t in lst])
        sharpe = float(pnls.mean() / max(pnls.std(ddof=1), 1e-9) * np.sqrt(252*4)) if len(pnls) > 1 else 0
        print(f'  {s:<20} n={len(lst):>4} PnL={pnls.sum():+9.2f} Sharpe={sharpe:+6.3f} WR={100*(pnls>0).mean():.1f}%')

    # Overlap analysis: squeeze_long vs keltner BUY
    sq_trades = [t for t in trades if t.strategy == 'squeeze_long']
    kc_trades = [t for t in trades if t.strategy == 'keltner' and t.direction == 'BUY']
    kc_all = [t for t in trades if t.strategy == 'keltner']
    print(f'\nsqueeze_long entries: {len(sq_trades)}')
    print(f'Keltner BUY entries: {len(kc_trades)}')
    print(f'Keltner ALL entries: {len(kc_all)}')

    if not sq_trades:
        print('No squeeze_long trades — overlap check N/A')
        return

    # Build set of keltner buy timestamps (rounded to H1)
    kc_times = sorted([pd.Timestamp(t.entry_time).floor('h') for t in kc_trades])
    kc_arr = pd.DatetimeIndex(kc_times)

    overlap_2h = 0
    overlap_6h = 0
    for sq in sq_trades:
        sq_t = pd.Timestamp(sq.entry_time).floor('h')
        # find min absolute diff in hours to any keltner buy
        if len(kc_arr) == 0:
            continue
        diffs_h = np.abs((kc_arr - sq_t).total_seconds() / 3600.0)
        min_diff = diffs_h.min()
        if min_diff <= 2:
            overlap_2h += 1
        if min_diff <= 6:
            overlap_6h += 1

    pct_2h = 100 * overlap_2h / len(sq_trades)
    pct_6h = 100 * overlap_6h / len(sq_trades)
    print(f'\nOverlap (within +/-2h of Keltner BUY): {overlap_2h}/{len(sq_trades)} = {pct_2h:.1f}%')
    print(f'Overlap (within +/-6h of Keltner BUY): {overlap_6h}/{len(sq_trades)} = {pct_6h:.1f}%')

    # Save
    result = {
        'total_squeeze_long': len(sq_trades),
        'total_keltner_buy': len(kc_trades),
        'total_keltner_all': len(kc_all),
        'overlap_2h_count': overlap_2h,
        'overlap_2h_pct': round(pct_2h, 2),
        'overlap_6h_count': overlap_6h,
        'overlap_6h_pct': round(pct_6h, 2),
    }
    with open(OUTPUT_DIR / 'correlation.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nSaved {OUTPUT_DIR}/correlation.json')

    decision = 'PASS (independent signal source)' if pct_2h < 20 else 'FAIL (high overlap with Keltner)'
    print(f'\nDecision: {decision}')


if __name__ == '__main__':
    main()
