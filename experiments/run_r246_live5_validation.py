#!/usr/bin/env python3
"""R246 — 5-Strategy LIVE Portfolio Validation (BacktestEngine)
================================================================
Counterfactual to R209v2: re-run with PSAR + Chandelier DISABLED, matching
the actual live config in trading-project-architecture.md:
  - Active: Keltner + TSMOM (M30, separate) + SESS_BO + Dual Thrust + M15 RSI / ORB
  - Disabled: PSAR, Chandelier

NOTE: TSMOM in live system is M30-only and not via BacktestEngine; R209v2 also
omits it. So R246 covers: Keltner + M15 RSI + ORB + SESS_BO + Dual Thrust.

This confirms (or refutes) whether disabling PSAR + Chandelier improves
the live portfolio Sharpe through slot contention reduction.
"""
from __future__ import annotations
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.runner import DataBundle, LIVE_PARITY_KWARGS
from backtest.engine import BacktestEngine
import indicators as signals_mod
import research_config as config

# Reuse the patched signal functions from R209v2
from experiments.run_r209v2_non_keltner_audit import (
    check_psar_signal, check_sess_bo_signal,
    check_dual_thrust_signal, check_chandelier_signal,
    _compute_psar_series,
)

OUTPUT_DIR = Path("results/r246_live5_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LIVE_PERIOD = ("2026-03-25", "2026-05-13")

# Match LIVE config from trading-project-architecture.md exactly:
# Keltner + TSMOM + SESS_BO + Dual Thrust + M30 RSI14 (M30, runs separately)
# PSAR: OFF, Chandelier: OFF
LIVE5_STRAT_CONFIGS = {
    'psar': {'enabled': False},  # DISABLED per architecture
    'sess_bo': {
        'enabled': True, 'broker_gmt_offset': 0, 'session_hour_gmt': 12,
        'lookback_bars': 4, 'sl_atr': 4.5, 'tp_atr': 4.0,
    },
    'dual_thrust': {
        'enabled': True, 'n_bars': 6, 'k_up': 0.5, 'k_down': 0.5,
        'sl_atr': 6.0, 'tp_atr': 8.0,
        'trail_act_atr': 0.06, 'trail_dist_atr': 0.01,
    },
    'chandelier': {'enabled': False},  # DISABLED per architecture
}

# Monkey-patch with the live5 config
import experiments.run_r209v2_non_keltner_audit as r209_mod

r209_mod.LIVE_STRAT_CONFIGS = LIVE5_STRAT_CONFIGS


def patch_indicators_for_live5():
    """Monkey-patch indicators.scan_all_signals to add live5 strategies."""
    original_scan = signals_mod.scan_all_signals

    def patched_scan(df_h1, df_m15=None, **kwargs):
        signals = original_scan(df_h1, df_m15, **kwargs) or []
        extras = []
        # Only the LIVE-active strategies (PSAR/Chandelier skipped via enabled=False)
        for fn in [check_psar_signal, check_sess_bo_signal,
                   check_dual_thrust_signal, check_chandelier_signal]:
            sig = fn(df_h1)
            if sig is not None:
                extras.append(sig)
        return signals + extras

    signals_mod.scan_all_signals = patched_scan
    print('  Patched scan_all_signals: SESS_BO + Dual Thrust ACTIVE; PSAR + Chandelier DISABLED')


def calc_stats(trades, label):
    if not trades:
        return {'label': label, 'n': 0, 'sharpe': 0.0, 'pnl': 0.0,
                'win_rate': 0.0, 'avg_pnl': 0.0, 'max_dd': 0.0}
    pnls = np.array([t['pnl'] for t in trades])
    n = len(pnls)
    total_pnl = pnls.sum()
    wr = (pnls > 0).mean() * 100
    cum = np.cumsum(pnls)
    dd = (np.maximum.accumulate(cum) - cum).max()
    sharpe = pnls.mean() / pnls.std() * np.sqrt(252) if pnls.std() > 0 else 0
    return {
        'label': label, 'n': n, 'pnl': round(total_pnl, 2),
        'sharpe': round(sharpe, 3), 'win_rate': round(wr, 2),
        'avg_pnl': round(pnls.mean(), 2), 'max_dd': round(dd, 2),
    }


def run_engine(data, start=None, end=None):
    kwargs_all = {**LIVE_PARITY_KWARGS}
    kwargs_all['max_positions'] = 4
    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False
    from indicators import get_orb_strategy
    get_orb_strategy().reset_daily()
    eng = BacktestEngine(data.m15_df, data.h1_df, **kwargs_all)
    print(f'  M15 bars: {len(data.m15_df)}, H1 bars: {len(data.h1_df)}')
    trades = eng.run()
    return trades


def _trade_attr(t, name, default=None):
    """Trade objects may be dataclasses or dicts."""
    if hasattr(t, name):
        return getattr(t, name)
    if isinstance(t, dict):
        return t.get(name, default)
    return default


def per_strat_stats(trades, label=''):
    by_strat = defaultdict(list)
    for t in trades:
        by_strat[_trade_attr(t, 'strategy', 'unknown')].append(t)
    rows = {}
    for s, ts in by_strat.items():
        # Normalize trades to dicts with pnl
        norm = [{'pnl': _trade_attr(tt, 'pnl', 0)} for tt in ts]
        rows[s] = calc_stats(norm, f'{label}_{s}')
    return rows, by_strat


def filter_period(trades, start, end):
    s = pd.Timestamp(start, tz='UTC')
    e = pd.Timestamp(end, tz='UTC')
    out = []
    for t in trades:
        et = _trade_attr(t, 'exit_time') or _trade_attr(t, 'entry_time')
        if et is None:
            continue
        if isinstance(et, str):
            et = pd.Timestamp(et)
        if et.tzinfo is None:
            et = et.tz_localize('UTC')
        if s <= et <= e:
            out.append(t)
    return out


def main():
    t0 = time.time()
    print('=' * 80)
    print('R246: 5-Strategy LIVE Portfolio Validation')
    print('=' * 80)
    print('  Matches live config: PSAR + Chandelier DISABLED')
    print('  Active strategies through engine: Keltner + M15_RSI + ORB + SESS_BO + Dual Thrust')

    patch_indicators_for_live5()

    print('\nLoading data...')
    data = DataBundle.load_default()
    print('Preparing indicators...')

    # Run full backtest
    print('\n' + '=' * 80)
    print('Phase 1: Full History (LIVE config — no PSAR, no Chandelier)')
    print('=' * 80)
    trades = run_engine(data)
    print(f'\n  Total trades: {len(trades)}')

    strat_stats, by_strat = per_strat_stats(trades, 'full')
    print(f"  Strategies seen: {sorted(strat_stats.keys())}")
    print()
    print(f"  {'Strategy':<15} {'N':>6} {'PnL':>10} {'Sharpe':>8} {'WR%':>7} {'MaxDD':>10}")
    for s in sorted(strat_stats.keys()):
        st = strat_stats[s]
        print(f"  {s:<15} {st['n']:>6} {st['pnl']:>10.2f} {st['sharpe']:>8.3f} "
              f"{st['win_rate']:>6.1f}% {st['max_dd']:>10.2f}")

    # Portfolio-level stats (normalize)
    norm = [{'pnl': _trade_attr(t, 'pnl', 0)} for t in trades]
    port_stats = calc_stats(norm, 'live5_portfolio')
    print(f"\n  PORTFOLIO: N={port_stats['n']}, Sharpe={port_stats['sharpe']}, "
          f"PnL=${port_stats['pnl']}, MaxDD=${port_stats['max_dd']}")

    # Live period
    live_trades = filter_period(trades, *LIVE_PERIOD)
    live_strat_stats, _ = per_strat_stats(live_trades, 'live')
    live_norm = [{'pnl': _trade_attr(t, 'pnl', 0)} for t in live_trades]
    live_port_stats = calc_stats(live_norm, 'live5_portfolio_live_period')
    print(f"\n  === Live Period ({LIVE_PERIOD[0]} -> {LIVE_PERIOD[1]}) ===")
    print(f"  {'Strategy':<15} {'N':>6} {'PnL':>10} {'Sharpe':>8} {'WR%':>7}")
    for s in sorted(live_strat_stats.keys()):
        st = live_strat_stats[s]
        print(f"  {s:<15} {st['n']:>6} {st['pnl']:>10.2f} {st['sharpe']:>8.3f} "
              f"{st['win_rate']:>6.1f}%")
    print(f"\n  LIVE-PORT: N={live_port_stats['n']}, Sharpe={live_port_stats['sharpe']}, "
          f"PnL=${live_port_stats['pnl']}")

    # Save
    output = {
        'config': 'live5 (Keltner + M15_RSI + ORB + SESS_BO + Dual Thrust)',
        'phase1_full': strat_stats,
        'phase1_portfolio': port_stats,
        'live_period_strats': live_strat_stats,
        'live_period_portfolio': live_port_stats,
        'runtime_sec': round(time.time() - t0, 1),
    }
    out_json = OUTPUT_DIR / 'R246_results.json'
    out_json.write_text(json.dumps(output, indent=2))
    print(f'\nSaved -> {out_json}')
    print(f'Total runtime: {(time.time() - t0):.0f}s')


if __name__ == '__main__':
    main()
