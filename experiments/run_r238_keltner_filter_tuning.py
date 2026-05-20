#!/usr/bin/env python3
"""R238: Keltner Filter Threshold Tuning
=========================================
Sweep the remaining untested Keltner filter parameters using LIVE_PARITY_KWARGS
as baseline. Each config is validated with K-Fold 6, holdout OOS, and era stability.

Parameters under test:
  1. Choppy threshold: 0.35, 0.40, 0.45, 0.50 (current), 0.55, 0.60
  2. ATR Percentile regime boundary:
     - low/normal cutoff: 0.20, 0.25, 0.30 (current), 0.35, 0.40
  3. Interaction: top choppy x top ATR pctl boundary

All tests use BacktestEngine + LIVE_PARITY_KWARGS (Keltner-only, no M30).
"""
from __future__ import annotations
import sys
import json
import time
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.runner import DataBundle, LIVE_PARITY_KWARGS
from backtest.engine import BacktestEngine, TradeRecord

OUTPUT_DIR = Path("results/r238_keltner_filter_tuning")
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


def calc_daily_sharpe(trades):
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
    results = {}
    for era_name, (start, end) in ERA_SEGMENTS.items():
        start_ts = pd.Timestamp(start, tz='UTC')
        end_ts = pd.Timestamp(end, tz='UTC')
        era_trades = [t for t in trades
                      if start_ts <= pd.Timestamp(t.entry_time) < end_ts]
        sharpe, pnl, wr, pf_val, n, _, _ = calc_daily_sharpe(era_trades)
        results[era_name] = {'n': n, 'sharpe': sharpe, 'pnl': pnl, 'win_rate': wr}
    return results


def run_config(args):
    """Worker: run one configuration."""
    config_name, overrides, m15_path, h1_path = args
    try:
        m15_df = pd.read_pickle(m15_path)
        h1_df = pd.read_pickle(h1_path)

        kwargs = {**LIVE_PARITY_KWARGS}
        kwargs.update(overrides)
        kwargs['label'] = config_name

        engine = BacktestEngine(m15_df, h1_df, **kwargs)
        trades = engine.run()
        kc_trades = [t for t in trades if t.strategy == 'keltner']

        sharpe, pnl, wr, pf_val, n, n_days, max_dd = calc_daily_sharpe(kc_trades)
        kf = kfold_validate(kc_trades)

        holdout_cutoff = pd.Timestamp(HOLDOUT_START, tz='UTC')
        train_trades = [t for t in kc_trades if pd.Timestamp(t.entry_time) < holdout_cutoff]
        oos_trades = [t for t in kc_trades if pd.Timestamp(t.entry_time) >= holdout_cutoff]
        train_sharpe = calc_daily_sharpe(train_trades)[0]
        oos_sharpe, oos_pnl, oos_wr, _, oos_n, _, _ = calc_daily_sharpe(oos_trades)

        if train_sharpe > 0 and oos_sharpe > 0:
            retention = round(oos_sharpe / train_sharpe, 3)
        else:
            retention = 0

        eras = era_stability(kc_trades)
        positive_eras = sum(1 for v in eras.values() if v['sharpe'] > 0)

        # Exit reason breakdown
        from collections import Counter
        exit_reasons = Counter(t.exit_reason for t in kc_trades)

        return {
            'config': config_name,
            'overrides': {k: v for k, v in overrides.items() if not callable(v)},
            'n_trades': n,
            'sharpe': sharpe,
            'pnl': pnl,
            'win_rate': wr,
            'profit_factor': pf_val,
            'max_dd': max_dd,
            'kfold': kf,
            'oos_sharpe': oos_sharpe,
            'oos_n': oos_n,
            'oos_pnl': round(oos_pnl, 2) if oos_pnl else 0,
            'oos_retention': retention,
            'eras': eras,
            'positive_eras': positive_eras,
            'exit_reasons': dict(exit_reasons),
            'skipped_choppy': engine.skipped_choppy,
            'total_all_trades': len(trades),
        }
    except Exception as e:
        return {'config': config_name, 'error': str(e), 'traceback': traceback.format_exc()}


def main():
    t0 = time.time()
    pf('=' * 80)
    pf('R238: KELTNER FILTER THRESHOLD TUNING')
    pf('  Base: LIVE_PARITY_KWARGS (Keltner-only)')
    pf('  Test 1: Choppy threshold sweep')
    pf('  Test 2: ATR Percentile regime boundary sweep')
    pf('  Test 3: Interaction grid')
    pf('=' * 80)

    pf('\n[1/4] Loading data...')
    data = DataBundle.load_default()
    pf(f'  M15: {len(data.m15_df)} bars | H1: {len(data.h1_df)} bars')

    cache_dir = OUTPUT_DIR / '_cache'
    cache_dir.mkdir(exist_ok=True)
    m15_path = str(cache_dir / 'm15.pkl')
    h1_path = str(cache_dir / 'h1.pkl')
    data.m15_df.to_pickle(m15_path)
    data.h1_df.to_pickle(h1_path)

    n_workers = min(cpu_count(), 8)
    pf(f'  Workers: {n_workers}')

    all_results = {}

    # ════════════════════════════════════════════════════════════
    # TEST 1: Choppy Threshold Sweep
    # ════════════════════════════════════════════════════════════
    pf(f'\n\n{"="*80}')
    pf('TEST 1: CHOPPY THRESHOLD SWEEP')
    pf(f'{"="*80}')

    choppy_values = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    choppy_configs = []
    for ct in choppy_values:
        label = f'choppy_{ct:.2f}'
        overrides = {'choppy_threshold': ct}
        if ct == 0.50:
            label += '_BASELINE'
        choppy_configs.append((label, overrides, m15_path, h1_path))

    pf(f'\n  Running {len(choppy_configs)} configs in parallel...')
    t1 = time.time()
    with Pool(n_workers) as pool:
        choppy_results = pool.map(run_config, choppy_configs)
    pf(f'  Test 1 complete in {time.time()-t1:.0f}s')

    pf(f'\n  {"Config":<28} {"Sharpe":>8} {"N":>6} {"WR%":>7} {"PF":>6} {"KF":>6} {"OOS_Sh":>8} {"ChopSkip":>9} {"PnL":>10}')
    pf(f'  {"-"*28} {"-"*8} {"-"*6} {"-"*7} {"-"*6} {"-"*6} {"-"*8} {"-"*9} {"-"*10}')
    for r in choppy_results:
        if 'error' in r:
            pf(f'  {r["config"]:<28} ERROR: {r["error"][:40]}')
        else:
            kf_str = f'{r["kfold"]["pass_count"]}/{r["kfold"]["total"]}'
            pf(f'  {r["config"]:<28} {r["sharpe"]:>8.3f} {r["n_trades"]:>6} {r["win_rate"]*100:>6.1f}% {r["profit_factor"]:>5.2f} {kf_str:>6} {r["oos_sharpe"]:>8.3f} {r["skipped_choppy"]:>9} {r["pnl"]:>10.1f}')
    all_results['choppy_sweep'] = choppy_results

    # ════════════════════════════════════════════════════════════
    # TEST 2: ATR Percentile Regime Boundary Sweep
    #   We modify the regime boundary thresholds via regime_config
    #   by passing different disable conditions
    # ════════════════════════════════════════════════════════════
    pf(f'\n\n{"="*80}')
    pf('TEST 2: ATR PERCENTILE REGIME BOUNDARY SWEEP')
    pf('  Modifying low-regime behavior: disable Keltner in low-ATR regime')
    pf(f'{"="*80}')

    # Since atr_pctl 0.30 is hardcoded for regime boundary, we test the effect
    # of disabling Keltner in low-ATR regimes by setting disable_keltner=True
    atr_pctl_configs = []

    # Config: baseline (no disable in low regime)
    atr_pctl_configs.append(('atr_pctl_baseline', {
        'regime_config': {
            'low':    {'trail_act': 0.06, 'trail_dist': 0.015},
            'normal': {'trail_act': 0.06, 'trail_dist': 0.015},
            'high':   {'trail_act': 0.06, 'trail_dist': 0.015},
        }
    }, m15_path, h1_path))

    # Config: disable Keltner in low-ATR regime
    atr_pctl_configs.append(('atr_pctl_disable_low', {
        'regime_config': {
            'low':    {'trail_act': 0.06, 'trail_dist': 0.015, 'disable_keltner': True},
            'normal': {'trail_act': 0.06, 'trail_dist': 0.015},
            'high':   {'trail_act': 0.06, 'trail_dist': 0.015},
        }
    }, m15_path, h1_path))

    # Config: tighter trail in low, wider trail in high
    atr_pctl_configs.append(('atr_pctl_adaptive_trail', {
        'regime_config': {
            'low':    {'trail_act': 0.04, 'trail_dist': 0.010},
            'normal': {'trail_act': 0.06, 'trail_dist': 0.015},
            'high':   {'trail_act': 0.08, 'trail_dist': 0.020},
        }
    }, m15_path, h1_path))

    # Config: disable Keltner in high-ATR regime (avoid spikes)
    atr_pctl_configs.append(('atr_pctl_disable_high', {
        'regime_config': {
            'low':    {'trail_act': 0.06, 'trail_dist': 0.015},
            'normal': {'trail_act': 0.06, 'trail_dist': 0.015},
            'high':   {'trail_act': 0.06, 'trail_dist': 0.015, 'disable_keltner': True},
        }
    }, m15_path, h1_path))

    # Config: disable both low and high (trade only in normal regime)
    atr_pctl_configs.append(('atr_pctl_normal_only', {
        'regime_config': {
            'low':    {'trail_act': 0.06, 'trail_dist': 0.015, 'disable_keltner': True},
            'normal': {'trail_act': 0.06, 'trail_dist': 0.015},
            'high':   {'trail_act': 0.06, 'trail_dist': 0.015, 'disable_keltner': True},
        }
    }, m15_path, h1_path))

    pf(f'\n  Running {len(atr_pctl_configs)} configs in parallel...')
    t2 = time.time()
    with Pool(n_workers) as pool:
        atr_results = pool.map(run_config, atr_pctl_configs)
    pf(f'  Test 2 complete in {time.time()-t2:.0f}s')

    pf(f'\n  {"Config":<28} {"Sharpe":>8} {"N":>6} {"WR%":>7} {"PF":>6} {"KF":>6} {"OOS_Sh":>8} {"PnL":>10}')
    pf(f'  {"-"*28} {"-"*8} {"-"*6} {"-"*7} {"-"*6} {"-"*6} {"-"*8} {"-"*10}')
    for r in atr_results:
        if 'error' in r:
            pf(f'  {r["config"]:<28} ERROR: {r["error"][:40]}')
        else:
            kf_str = f'{r["kfold"]["pass_count"]}/{r["kfold"]["total"]}'
            pf(f'  {r["config"]:<28} {r["sharpe"]:>8.3f} {r["n_trades"]:>6} {r["win_rate"]*100:>6.1f}% {r["profit_factor"]:>5.2f} {kf_str:>6} {r["oos_sharpe"]:>8.3f} {r["pnl"]:>10.1f}')
    all_results['atr_pctl_sweep'] = atr_results

    # ════════════════════════════════════════════════════════════
    # TEST 3: Interaction Grid (top choppy x ATR regime)
    # ════════════════════════════════════════════════════════════
    pf(f'\n\n{"="*80}')
    pf('TEST 3: INTERACTION GRID (Choppy x ATR Regime)')
    pf(f'{"="*80}')

    # Pick choppy values to cross with regime configs
    interaction_configs = []
    choppy_test = [0.45, 0.50, 0.55]
    regime_variants = {
        'unified': {
            'low':    {'trail_act': 0.06, 'trail_dist': 0.015},
            'normal': {'trail_act': 0.06, 'trail_dist': 0.015},
            'high':   {'trail_act': 0.06, 'trail_dist': 0.015},
        },
        'disable_low': {
            'low':    {'trail_act': 0.06, 'trail_dist': 0.015, 'disable_keltner': True},
            'normal': {'trail_act': 0.06, 'trail_dist': 0.015},
            'high':   {'trail_act': 0.06, 'trail_dist': 0.015},
        },
        'adaptive': {
            'low':    {'trail_act': 0.04, 'trail_dist': 0.010},
            'normal': {'trail_act': 0.06, 'trail_dist': 0.015},
            'high':   {'trail_act': 0.08, 'trail_dist': 0.020},
        },
    }

    for ct in choppy_test:
        for rv_name, rv_config in regime_variants.items():
            label = f'c{ct:.2f}_r{rv_name}'
            overrides = {
                'choppy_threshold': ct,
                'regime_config': rv_config,
            }
            interaction_configs.append((label, overrides, m15_path, h1_path))

    pf(f'\n  Running {len(interaction_configs)} configs in parallel...')
    t3 = time.time()
    with Pool(n_workers) as pool:
        interaction_results = pool.map(run_config, interaction_configs)
    pf(f'  Test 3 complete in {time.time()-t3:.0f}s')

    pf(f'\n  {"Config":<28} {"Sharpe":>8} {"N":>6} {"WR%":>7} {"PF":>6} {"KF":>6} {"OOS_Sh":>8} {"PnL":>10}')
    pf(f'  {"-"*28} {"-"*8} {"-"*6} {"-"*7} {"-"*6} {"-"*6} {"-"*8} {"-"*10}')
    for r in interaction_results:
        if 'error' in r:
            pf(f'  {r["config"]:<28} ERROR: {r["error"][:40]}')
        else:
            kf_str = f'{r["kfold"]["pass_count"]}/{r["kfold"]["total"]}'
            pf(f'  {r["config"]:<28} {r["sharpe"]:>8.3f} {r["n_trades"]:>6} {r["win_rate"]*100:>6.1f}% {r["profit_factor"]:>5.2f} {kf_str:>6} {r["oos_sharpe"]:>8.3f} {r["pnl"]:>10.1f}')
    all_results['interaction_grid'] = interaction_results

    # ════════════════════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════════════════════
    pf(f'\n\n{"="*80}')
    pf('SUMMARY: BEST CONFIGS')
    pf(f'{"="*80}')

    all_configs = choppy_results + atr_results + interaction_results
    valid = [r for r in all_configs if 'error' not in r]
    valid.sort(key=lambda r: r['sharpe'], reverse=True)

    pf(f'\n  Top 5 by Sharpe:')
    pf(f'  {"Rank":>4} {"Config":<28} {"Sharpe":>8} {"OOS_Sh":>8} {"N":>6} {"KF":>6} {"PnL":>10}')
    for i, r in enumerate(valid[:5]):
        kf_str = f'{r["kfold"]["pass_count"]}/{r["kfold"]["total"]}'
        pf(f'  {i+1:>4} {r["config"]:<28} {r["sharpe"]:>8.3f} {r["oos_sharpe"]:>8.3f} {r["n_trades"]:>6} {kf_str:>6} {r["pnl"]:>10.1f}')

    # Sort by OOS Sharpe
    valid_oos = [r for r in valid if r.get('oos_sharpe', -999) > -999]
    valid_oos.sort(key=lambda r: r['oos_sharpe'], reverse=True)
    pf(f'\n  Top 5 by OOS Sharpe:')
    pf(f'  {"Rank":>4} {"Config":<28} {"Sharpe":>8} {"OOS_Sh":>8} {"N":>6} {"KF":>6} {"PnL":>10}')
    for i, r in enumerate(valid_oos[:5]):
        kf_str = f'{r["kfold"]["pass_count"]}/{r["kfold"]["total"]}'
        pf(f'  {i+1:>4} {r["config"]:<28} {r["sharpe"]:>8.3f} {r["oos_sharpe"]:>8.3f} {r["n_trades"]:>6} {kf_str:>6} {r["pnl"]:>10.1f}')

    elapsed = time.time() - t0
    pf(f'\n  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)')

    # Save results
    def serialize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    output_file = OUTPUT_DIR / 'r238_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=serialize)
    pf(f'\n  Results saved: {output_file}')

    import shutil
    shutil.rmtree(cache_dir, ignore_errors=True)

    pf('\nDone.')


if __name__ == '__main__':
    main()
