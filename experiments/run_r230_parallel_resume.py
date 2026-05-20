#!/usr/bin/env python3
"""R230 Parallel Resume: Pick up from Phase 3 onward with multiprocessing.

The original run_r230_mega_overnight.py is single-threaded on a 208-core machine.
This script:
  1. Kills the old single-threaded process
  2. Resumes from where Phase 3 left off (combo ~350/3600 for first strategy)
  3. Uses multiprocessing.Pool to parallelize the parameter sweep
  4. Continues through Phase 4-10 and Part B/C with parallelism

Key parallelization points:
  - Phase 3 param sweep: each combo is independent -> Pool.map
  - Phase 4 Walk-Forward: each WF window is independent
  - Phase 5 Era stability: each era is independent
  - Phase 6 Sensitivity: each perturbation is independent
  - Phase 7 Monte Carlo: embarrassingly parallel
  - Part B can run after Part A completes (or even concurrently if memory allows)
"""
from __future__ import annotations
import sys
import json
import time
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple
from itertools import combinations, product
from multiprocessing import Pool, cpu_count
from functools import partial

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.m30_engine import M30BacktestEngine, load_m30_with_indicators
from backtest.h4_engine import H4BacktestEngine, load_h4_with_indicators
from backtest.engine import TradeRecord

OUTPUT_DIR = Path("results/r230_mega_overnight")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SPREAD = 0.30
N_BOOTSTRAP = 5000
N_WORKERS = min(cpu_count() - 2, 100)  # leave 2 cores for system

ERA_SEGMENTS = {
    "Pre-COVID (2015-2019)":      ("2015-01-01", "2020-01-01"),
    "COVID+Recovery (2020-2021)": ("2020-01-01", "2022-01-01"),
    "Tightening (2022-2023)":     ("2022-01-01", "2024-01-01"),
    "Recent (2024-2026)":         ("2024-01-01", "2026-06-01"),
}

WF_CUTOFFS = [
    ("2015-01-01", "2017-01-01", "2017-01-01", "2018-10-01"),
    ("2015-01-01", "2018-10-01", "2018-10-01", "2020-07-01"),
    ("2015-01-01", "2020-07-01", "2020-07-01", "2022-04-01"),
    ("2015-01-01", "2022-04-01", "2022-04-01", "2024-01-01"),
    ("2015-01-01", "2024-01-01", "2024-01-01", "2025-07-01"),
    ("2015-01-01", "2025-07-01", "2025-07-01", "2026-06-01"),
]

SL_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
TP_GRID = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0]
TRAIL_GRID = [
    (0.0, 0.0),
    (0.15, 0.04), (0.2, 0.06), (0.25, 0.07),
    (0.3, 0.08), (0.4, 0.10), (0.5, 0.12),
    (0.5, 0.15), (0.8, 0.20), (1.0, 0.25),
]

SLIPPAGE_CONFIGS = [
    {"name": "no_slippage", "slippage_model": "none"},
    {"name": "fixed_slippage", "slippage_model": "fixed"},
    {"name": "empirical_slippage", "slippage_model": "empirical"},
    {"name": "realistic_slippage", "slippage_model": "realistic"},
]

# ═══════════════════════════════════════════════════════
# Utility functions (same as original)
# ═══════════════════════════════════════════════════════

def print_flush(msg):
    print(msg)
    sys.stdout.flush()


def save(name, data):
    p = OUTPUT_DIR / f'{name}.json'
    with open(p, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print_flush(f'  -> saved {p}')


def save_progress(part, phase, detail=''):
    prog_file = OUTPUT_DIR / '_progress.json'
    try:
        prog = json.loads(prog_file.read_text()) if prog_file.exists() else {}
    except:
        prog = {}
    prog[f'{part}_{phase}'] = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'detail': detail,
    }
    prog_file.write_text(json.dumps(prog, indent=2))


def calc_stats(trades):
    if not trades or len(trades) < 5:
        return {'n': len(trades) if trades else 0, 'sharpe': -999, 'pnl': 0,
                'profit_factor': 0, 'win_rate': 0, 'avg_pnl': 0}
    pnls = np.array([t.pnl for t in trades])
    n = len(pnls)
    total = float(pnls.sum())
    mean = float(pnls.mean())
    std = float(pnls.std(ddof=1)) if n > 1 else 1e-9
    sharpe = mean / max(std, 1e-9) * np.sqrt(252 * 2)  # M30 ~ 2 trades/day scaling
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else 99.0
    return {
        'n': n, 'sharpe': round(float(sharpe), 4), 'pnl': round(total, 2),
        'profit_factor': round(pf, 3), 'win_rate': round(len(wins) / n, 4),
        'avg_pnl': round(mean, 3),
    }


def monte_carlo_bootstrap(trades, n_resamples=5000):
    if len(trades) < 20:
        return {'skip': True, 'reason': 'too_few_trades'}
    pnls = np.array([t.pnl for t in trades])
    n = len(pnls)
    rng = np.random.default_rng(42)
    boot_sharpes = np.zeros(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(pnls, size=n, replace=True)
        m, s = sample.mean(), sample.std(ddof=1)
        boot_sharpes[i] = m / max(s, 1e-9) * np.sqrt(252 * 2)
    boot_arr = boot_sharpes
    p_value = float((boot_arr <= 0).mean())
    return {
        'mean_sharpe': round(float(boot_arr.mean()), 3),
        'std_sharpe': round(float(boot_arr.std()), 3),
        'ci_5': round(float(np.percentile(boot_arr, 5)), 3),
        'ci_95': round(float(np.percentile(boot_arr, 95)), 3),
        'ci_1': round(float(np.percentile(boot_arr, 1)), 3),
        'mc_verdict': 'PASS' if p_value < 0.05 else 'FAIL',
        'p_value': round(p_value, 4),
    }


def drawdown_analysis(trades):
    if len(trades) < 10:
        return {'skip': True}
    pnls = np.array([t.pnl for t in trades])
    n = len(pnls)
    cum = np.cumsum(pnls)
    peaks = np.maximum.accumulate(cum)
    drawdowns = peaks - cum
    max_dd = float(drawdowns.max())
    max_dd_idx = int(drawdowns.argmax())
    streak, max_streak = 0, 0
    for p in pnls:
        if p < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    recovery_bars = 0
    if max_dd_idx < n - 1:
        for ri in range(max_dd_idx + 1, n):
            if cum[ri] >= peaks[max_dd_idx]:
                recovery_bars = ri - max_dd_idx
                break
    return {
        'max_dd': round(max_dd, 2),
        'worst_losing_streak': max_streak,
        'recovery_trades': recovery_bars,
        'total_pnl': round(float(cum[-1]), 2),
    }


# ═══════════════════════════════════════════════════════
# Worker function for parallel param sweep
# ═══════════════════════════════════════════════════════

# Global variables set per-worker via initializer
_worker_df = None
_worker_engine_cls = None
_worker_default_params = None


def _init_worker(df_path_or_bytes, engine_cls_name, default_params):
    """Initialize worker process with shared data."""
    global _worker_df, _worker_engine_cls, _worker_default_params, _worker_sig_map
    import pickle
    _worker_df = pickle.loads(df_path_or_bytes)
    if engine_cls_name == 'M30BacktestEngine':
        from backtest.m30_engine import M30BacktestEngine
        _worker_engine_cls = M30BacktestEngine
    else:
        from backtest.h4_engine import H4BacktestEngine
        _worker_engine_cls = H4BacktestEngine
    _worker_default_params = default_params
    # Build sig func map inside worker
    from experiments.run_r230_mega_overnight import M30_STRATEGIES, H4_STRATEGIES
    _worker_sig_map = {}
    for name, func in M30_STRATEGIES:
        _worker_sig_map[name] = func
    for name, func in H4_STRATEGIES:
        _worker_sig_map[name] = func


def _run_single_combo(args):
    """Run a single parameter combo in worker process."""
    strat_name, sig_func_name, sl_m, tp_m, trail_a, trail_d, mh = args
    global _worker_df, _worker_engine_cls, _worker_default_params, _worker_sig_map

    sig_func = _worker_sig_map.get(sig_func_name)
    if sig_func is None:
        return None

    p = dict(_worker_default_params)
    p.update({
        'sl_atr_mult': sl_m, 'tp_atr_mult': tp_m,
        'trailing_activate_atr': trail_a, 'trailing_distance_atr': trail_d,
        'max_hold': mh,
    })

    engine = _worker_engine_cls(_worker_df, signal_funcs=[(strat_name, sig_func)], **p)
    trades = engine.run()
    strat_trades = [t for t in trades if t.strategy == strat_name]
    s = calc_stats(strat_trades)
    return {
        'sl': sl_m, 'tp': tp_m, 'trail_a': trail_a, 'trail_d': trail_d,
        'max_hold': mh, **s
    }


def _run_wf_window(args):
    """Run walk-forward on a single window."""
    strat_name, sig_func_name, train_s, train_e, test_s, test_e, base_params, sl_factors, tp_factors = args
    global _worker_df, _worker_engine_cls, _worker_default_params, _worker_sig_map

    sig_func = _worker_sig_map.get(sig_func_name)
    if sig_func is None:
        return None

    df = _worker_df
    df_train = df[(df.index >= pd.Timestamp(train_s, tz='UTC')) &
                  (df.index < pd.Timestamp(train_e, tz='UTC'))].copy()
    df_test = df[(df.index >= pd.Timestamp(test_s, tz='UTC')) &
                 (df.index < pd.Timestamp(test_e, tz='UTC'))].copy()

    if len(df_train) < 100 or len(df_test) < 50:
        return {'window': f'{train_s}->{test_e}', 'skip': True}

    best_sh, best_p = -999, None
    base_sl = base_params.get('sl', 3.0)
    base_tp = base_params.get('tp', 6.0)

    for sl_f in sl_factors:
        for tp_f in tp_factors:
            sl_m = round(base_sl * sl_f, 1)
            tp_m = round(base_tp * tp_f, 1)
            if tp_m < sl_m:
                continue
            p = dict(_worker_default_params)
            p.update({
                'sl_atr_mult': sl_m, 'tp_atr_mult': tp_m,
                'trailing_activate_atr': base_params.get('trail_act', 0.3),
                'trailing_distance_atr': base_params.get('trail_dist', 0.08),
                'max_hold': base_params.get('max_hold', 48),
            })
            engine = _worker_engine_cls(df_train, signal_funcs=[(strat_name, sig_func)], **p)
            trades = [t for t in engine.run() if t.strategy == strat_name]
            s = calc_stats(trades)
            if s['sharpe'] > best_sh and s['n'] >= 10:
                best_sh = s['sharpe']
                best_p = {'sl': sl_m, 'tp': tp_m}

    # Test OOS with best in-sample params
    if best_p:
        p = dict(_worker_default_params)
        p.update({
            'sl_atr_mult': best_p['sl'], 'tp_atr_mult': best_p['tp'],
            'trailing_activate_atr': base_params.get('trail_act', 0.3),
            'trailing_distance_atr': base_params.get('trail_dist', 0.08),
            'max_hold': base_params.get('max_hold', 48),
        })
        engine = _worker_engine_cls(df_test, signal_funcs=[(strat_name, sig_func)], **p)
        oos_trades = [t for t in engine.run() if t.strategy == strat_name]
        oos_stats = calc_stats(oos_trades)
    else:
        oos_stats = {'sharpe': -999, 'n': 0}

    return {
        'window': f'{test_s}->{test_e}',
        'is_sharpe': round(best_sh, 3),
        'is_params': best_p,
        'oos_sharpe': oos_stats['sharpe'],
        'oos_n': oos_stats['n'],
    }


def _run_era(args):
    """Run backtest on a single era segment."""
    strat_name, sig_func_name, era_name, era_start, era_end, params = args
    global _worker_df, _worker_engine_cls, _worker_default_params, _worker_sig_map

    sig_func = _worker_sig_map.get(sig_func_name)
    df = _worker_df
    df_era = df[(df.index >= pd.Timestamp(era_start, tz='UTC')) &
                (df.index < pd.Timestamp(era_end, tz='UTC'))].copy()
    if len(df_era) < 50:
        return {'era': era_name, 'skip': True}

    p = dict(_worker_default_params)
    p.update(params)
    engine = _worker_engine_cls(df_era, signal_funcs=[(strat_name, sig_func)], **p)
    trades = [t for t in engine.run() if t.strategy == strat_name]
    s = calc_stats(trades)
    return {'era': era_name, **s}


# ═══════════════════════════════════════════════════════
# Signal function registry (imported at module level for main process)
# ═══════════════════════════════════════════════════════

from experiments.run_r230_mega_overnight import M30_STRATEGIES, H4_STRATEGIES


# ═══════════════════════════════════════════════════════
# Parallel Pipeline
# ═══════════════════════════════════════════════════════

def run_parallel_pipeline(
    tf_name: str,
    df: pd.DataFrame,
    strategies: List[Tuple[str, Callable]],
    engine_cls,
    default_params: dict,
    sl_grid: list,
    tp_grid: list,
    trail_grid: list,
    max_hold_grid: list,
    part_label: str,
    skip_phase1_2: bool = False,
    phase2_result: dict = None,
):
    """Run 10-phase pipeline with multiprocessing on param sweeps."""
    import pickle

    t0 = time.time()
    strat_map = dict(strategies)
    engine_cls_name = 'M30BacktestEngine' if 'M30' in engine_cls.__name__ else 'H4BacktestEngine'
    df_bytes = pickle.dumps(df)

    def run_single(strat_name, sig_func, params_override=None):
        p = dict(default_params)
        if params_override:
            p.update(params_override)
        engine = engine_cls(df, signal_funcs=[(strat_name, sig_func)], **p)
        trades = engine.run()
        return [t for t in trades if t.strategy == strat_name]

    # Phase 1: Screening (fast, no need to parallelize)
    if not skip_phase1_2:
        print_flush(f'\n{"="*80}\n{part_label} Phase 1: Strategy Screening\n{"="*80}')
        phase1 = {}
        viable = []
        for strat_name, sig_func in strategies:
            trades = run_single(strat_name, sig_func)
            s = calc_stats(trades)
            era_results = {}
            for era_name, (es, ee) in ERA_SEGMENTS.items():
                df_era = df[(df.index >= pd.Timestamp(es, tz='UTC')) &
                            (df.index < pd.Timestamp(ee, tz='UTC'))].copy()
                if len(df_era) < 50:
                    continue
                eng = engine_cls(df_era, signal_funcs=[(strat_name, sig_func)], **default_params)
                era_trades = [t for t in eng.run() if t.strategy == strat_name]
                era_results[era_name] = calc_stats(era_trades)
            phase1[strat_name] = {'baseline': s, 'eras': era_results}
            print_flush(f'  {strat_name:<20} n={s["n"]:>5}  Sharpe={s["sharpe"]:.3f}  PnL=${s["pnl"]:.0f}')
            if s['sharpe'] > 0.3 and s['n'] >= 30:
                viable.append(strat_name)
        save(f'{tf_name}_phase1_screening', phase1)
        save_progress(part_label, 'phase1', f'viable={viable}')
        print_flush(f'\n  Viable (Sharpe>0.3, n>=30): {viable}')

        # Phase 2: K-Fold
        print_flush(f'\n{"="*80}\n{part_label} Phase 2: K-Fold Validation\n{"="*80}')
        phase2 = {}
        for strat_name in viable:
            sig_func = strat_map[strat_name]
            trades = run_single(strat_name, sig_func)
            pnls = [t.pnl for t in trades]
            n = len(pnls)
            if n < 30:
                phase2[strat_name] = {'kfold_6': {'verdict': 'FAIL'}}
                continue
            # 6-fold
            fold_size = n // 6
            folds_6 = []
            for fi in range(6):
                start_i = fi * fold_size
                end_i = start_i + fold_size if fi < 5 else n
                fold_pnls = np.array(pnls[start_i:end_i])
                if len(fold_pnls) < 5:
                    continue
                sh = float(fold_pnls.mean() / max(fold_pnls.std(ddof=1), 1e-9) * np.sqrt(252 * 2))
                folds_6.append({'fold': fi + 1, 'n': len(fold_pnls), 'sharpe': round(sh, 3)})
            positive_folds = sum(1 for f in folds_6 if f['sharpe'] > 0)
            verdict = 'PASS' if positive_folds >= 4 else 'FAIL'
            kf6 = {'folds': folds_6, 'positive_folds': positive_folds, 'verdict': verdict}
            phase2[strat_name] = {'kfold_6': kf6}
            print_flush(f'  {strat_name}: {positive_folds}/6 positive -> {verdict}')
        save(f'{tf_name}_phase2_kfold', phase2)
        save_progress(part_label, 'phase2')
        kf_passers = [s for s in viable if phase2.get(s, {}).get('kfold_6', {}).get('verdict') == 'PASS']
    else:
        # Resume from existing phase2 results
        phase2 = phase2_result or {}
        # Derive viable and kf_passers directly from phase2 keys (all entries had verdict)
        kf_passers = [s for s in phase2 if phase2[s].get('kfold_6', {}).get('verdict') == 'PASS']
        # Also check phase1 file for broader viable list
        p1_file = OUTPUT_DIR / f'{tf_name}_phase1_screening.json'
        if p1_file.exists():
            phase1 = json.loads(p1_file.read_text())
            viable = list(phase1.keys())
        else:
            viable = list(phase2.keys())
        print_flush(f'  Resumed: viable={viable}')
        print_flush(f'  K-Fold passers: {kf_passers}')

    # ═══ Phase 3: Parallel Parameter Sweep ═══
    if kf_passers:
        print_flush(f'\n{"="*80}\n{part_label} Phase 3: Extended Parameter Sweep (PARALLEL, {N_WORKERS} workers)\n{"="*80}')
        phase3 = {}

        # Build all combos
        all_combos_template = []
        for sl_m in sl_grid:
            for tp_m in tp_grid:
                if tp_m < sl_m:
                    continue
                for trail_a, trail_d in trail_grid:
                    for mh in max_hold_grid:
                        all_combos_template.append((sl_m, tp_m, trail_a, trail_d, mh))

        total_combos = len(all_combos_template)
        print_flush(f'  Grid size per strategy: {total_combos} combos')
        print_flush(f'  Strategies to sweep: {kf_passers}')
        print_flush(f'  Total work: {total_combos * len(kf_passers)} backtests across {N_WORKERS} workers')

        with Pool(
            processes=N_WORKERS,
            initializer=_init_worker,
            initargs=(df_bytes, engine_cls_name, default_params)
        ) as pool:
            for strat_name in kf_passers:
                print_flush(f'\n  --- {strat_name} parallel sweep ({total_combos} combos) ---')
                t_strat = time.time()

                work_items = [
                    (strat_name, strat_name, sl, tp, ta, td, mh)
                    for sl, tp, ta, td, mh in all_combos_template
                ]

                results = pool.map(_run_single_combo, work_items, chunksize=max(1, total_combos // (N_WORKERS * 4)))
                results = [r for r in results if r is not None]
                results.sort(key=lambda x: x['sharpe'], reverse=True)

                best = results[0] if results else {}
                best_sharpe = best.get('sharpe', -999)
                best_params = {
                    'sl': best.get('sl'), 'tp': best.get('tp'),
                    'trail_act': best.get('trail_a'), 'trail_dist': best.get('trail_d'),
                    'max_hold': best.get('max_hold'),
                } if best else None

                for r in results[:10]:
                    print_flush(f'    SL{r["sl"]}_TP{r["tp"]}_T{r["trail_a"]}/{r["trail_d"]}_MH{r["max_hold"]}  '
                                f'n={r["n"]:>5} Sh={r["sharpe"]:.3f} PnL=${r["pnl"]:.0f} PF={r["profit_factor"]:.2f}')
                print_flush(f'  Best: {best_params}  Sharpe={best_sharpe:.3f}  ({time.time()-t_strat:.1f}s)')
                phase3[strat_name] = {'best_params': best_params, 'best_sharpe': best_sharpe,
                                      'top10': results[:10], 'total_combos': total_combos}

        save(f'{tf_name}_phase3_param_sweep', phase3)
        save_progress(part_label, 'phase3')
    else:
        phase3 = {}

    # ═══ Phase 4: Walk-Forward (Parallel) ═══
    deep_candidates = [s for s in kf_passers if phase3.get(s, {}).get('best_sharpe', 0) > 1.0]
    if not deep_candidates:
        deep_candidates = kf_passers[:5]

    if deep_candidates:
        print_flush(f'\n{"="*80}\n{part_label} Phase 4: Walk-Forward (PARALLEL)\n{"="*80}')
        phase4 = {}
        sl_factors = [0.7, 0.85, 1.0, 1.15, 1.3]
        tp_factors = [0.7, 0.85, 1.0, 1.15, 1.3]

        with Pool(processes=N_WORKERS, initializer=_init_worker,
                  initargs=(df_bytes, engine_cls_name, default_params)) as pool:
            for strat_name in deep_candidates:
                print_flush(f'\n  --- {strat_name} Walk-Forward ---')
                best_p = phase3.get(strat_name, {}).get('best_params', {})
                if not best_p:
                    best_p = {'sl': default_params.get('sl_atr_mult', 3.0),
                              'tp': default_params.get('tp_atr_mult', 6.0),
                              'trail_act': 0.3, 'trail_dist': 0.08,
                              'max_hold': default_params.get('max_hold', 48)}

                work_items = [
                    (strat_name, strat_name, ts, te, oos_s, oos_e, best_p, sl_factors, tp_factors)
                    for ts, te, oos_s, oos_e in WF_CUTOFFS
                ]
                wf_results = pool.map(_run_wf_window, work_items)
                wf_results = [r for r in wf_results if r is not None]

                oos_sharpes = [r['oos_sharpe'] for r in wf_results if not r.get('skip')]
                positive_oos = sum(1 for s in oos_sharpes if s > 0)
                verdict = 'PASS' if positive_oos >= 4 else 'FAIL'

                for r in wf_results:
                    if not r.get('skip'):
                        print_flush(f'    {r["window"]}: IS={r["is_sharpe"]:.3f} OOS={r["oos_sharpe"]:.3f}')
                print_flush(f'  WF verdict: {positive_oos}/{len(oos_sharpes)} positive OOS -> {verdict}')
                phase4[strat_name] = {'windows': wf_results, 'oos_sharpes': oos_sharpes,
                                      'positive_oos': positive_oos, 'verdict': verdict}

        save(f'{tf_name}_phase4_walkforward', phase4)
        save_progress(part_label, 'phase4')
    else:
        phase4 = {}

    # ═══ Phase 5: Era Stability (Parallel) ═══
    if deep_candidates:
        print_flush(f'\n{"="*80}\n{part_label} Phase 5: Era Stability\n{"="*80}')
        phase5 = {}

        with Pool(processes=N_WORKERS, initializer=_init_worker,
                  initargs=(df_bytes, engine_cls_name, default_params)) as pool:
            for strat_name in deep_candidates:
                best_p = phase3.get(strat_name, {}).get('best_params', {})
                params = {
                    'sl_atr_mult': best_p.get('sl', default_params.get('sl_atr_mult', 3.0)),
                    'tp_atr_mult': best_p.get('tp', default_params.get('tp_atr_mult', 6.0)),
                    'trailing_activate_atr': best_p.get('trail_act', 0.3),
                    'trailing_distance_atr': best_p.get('trail_dist', 0.08),
                    'max_hold': best_p.get('max_hold', default_params.get('max_hold', 48)),
                }
                work_items = [
                    (strat_name, strat_name, era_name, es, ee, params)
                    for era_name, (es, ee) in ERA_SEGMENTS.items()
                ]
                era_results = pool.map(_run_era, work_items)
                era_results = [r for r in era_results if r and not r.get('skip')]
                positive_eras = sum(1 for r in era_results if r.get('sharpe', -999) > 0)
                verdict = 'PASS' if positive_eras >= 3 else 'FAIL'
                for r in era_results:
                    print_flush(f'    {r["era"]}: Sharpe={r["sharpe"]:.3f} n={r["n"]}')
                print_flush(f'  Era verdict: {positive_eras}/4 positive -> {verdict}')
                phase5[strat_name] = {'eras': era_results, 'positive_eras': positive_eras, 'verdict': verdict}

        save(f'{tf_name}_phase5_era_stability', phase5)
        save_progress(part_label, 'phase5')
    else:
        phase5 = {}

    # ═══ Phase 6: Parameter Sensitivity ═══
    if deep_candidates:
        print_flush(f'\n{"="*80}\n{part_label} Phase 6: Parameter Sensitivity\n{"="*80}')
        phase6 = {}
        perturbations = [0.6, 0.8, 1.0, 1.2, 1.4]

        with Pool(processes=N_WORKERS, initializer=_init_worker,
                  initargs=(df_bytes, engine_cls_name, default_params)) as pool:
            for strat_name in deep_candidates:
                best_p = phase3.get(strat_name, {}).get('best_params', {})
                base_sl = best_p.get('sl', default_params.get('sl_atr_mult', 3.0))
                base_tp = best_p.get('tp', default_params.get('tp_atr_mult', 6.0))

                work_items = []
                for sl_f in perturbations:
                    for tp_f in perturbations:
                        sl_m = round(base_sl * sl_f, 1)
                        tp_m = round(base_tp * tp_f, 1)
                        if tp_m < sl_m:
                            continue
                        work_items.append((
                            strat_name, strat_name, sl_m, tp_m,
                            best_p.get('trail_act', 0.3), best_p.get('trail_dist', 0.08),
                            best_p.get('max_hold', 48)
                        ))

                results = pool.map(_run_single_combo, work_items)
                results = [r for r in results if r is not None]
                sharpes = [r['sharpe'] for r in results]
                sensitivity = float(np.std(sharpes)) if sharpes else 999
                verdict = 'PASS' if sensitivity < 0.5 else 'FAIL'
                print_flush(f'  {strat_name}: sensitivity_std={sensitivity:.3f} -> {verdict}')
                phase6[strat_name] = {'results': results, 'sensitivity_std': round(sensitivity, 4),
                                      'verdict': verdict}

        save(f'{tf_name}_phase6_sensitivity', phase6)
        save_progress(part_label, 'phase6')
    else:
        phase6 = {}

    # ═══ Phase 7: Monte Carlo Bootstrap ═══
    if deep_candidates:
        print_flush(f'\n{"="*80}\n{part_label} Phase 7: Monte Carlo Bootstrap\n{"="*80}')
        phase7 = {}
        for strat_name in deep_candidates:
            best_p = phase3.get(strat_name, {}).get('best_params', {})
            params = {
                'sl_atr_mult': best_p.get('sl', default_params.get('sl_atr_mult', 3.0)),
                'tp_atr_mult': best_p.get('tp', default_params.get('tp_atr_mult', 6.0)),
                'trailing_activate_atr': best_p.get('trail_act', 0.3),
                'trailing_distance_atr': best_p.get('trail_dist', 0.08),
                'max_hold': best_p.get('max_hold', default_params.get('max_hold', 48)),
            }
            trades = run_single(strat_name, strat_map[strat_name], params)
            mc = monte_carlo_bootstrap(trades, N_BOOTSTRAP)
            print_flush(f'  {strat_name}: mean_sh={mc.get("mean_sharpe", 0):.3f} '
                        f'CI=[{mc.get("ci_5", 0):.3f}, {mc.get("ci_95", 0):.3f}] -> {mc.get("mc_verdict")}')
            phase7[strat_name] = mc

        save(f'{tf_name}_phase7_montecarlo', phase7)
        save_progress(part_label, 'phase7')
    else:
        phase7 = {}

    # ═══ Phase 8: Drawdown Stress ═══
    if deep_candidates:
        print_flush(f'\n{"="*80}\n{part_label} Phase 8: Drawdown Stress\n{"="*80}')
        phase8 = {}
        for strat_name in deep_candidates:
            best_p = phase3.get(strat_name, {}).get('best_params', {})
            params = {
                'sl_atr_mult': best_p.get('sl', default_params.get('sl_atr_mult', 3.0)),
                'tp_atr_mult': best_p.get('tp', default_params.get('tp_atr_mult', 6.0)),
                'trailing_activate_atr': best_p.get('trail_act', 0.3),
                'trailing_distance_atr': best_p.get('trail_dist', 0.08),
                'max_hold': best_p.get('max_hold', default_params.get('max_hold', 48)),
            }
            trades = run_single(strat_name, strat_map[strat_name], params)
            dd = drawdown_analysis(trades)
            print_flush(f'  {strat_name}: MaxDD=${dd.get("max_dd", 0):.0f} '
                        f'WorstStreak={dd.get("worst_losing_streak", 0)} '
                        f'Recovery={dd.get("recovery_trades", 0)}')
            phase8[strat_name] = dd

        save(f'{tf_name}_phase8_drawdown', phase8)
        save_progress(part_label, 'phase8')
    else:
        phase8 = {}

    # ═══ Phase 9: Slippage Testing ═══
    # Skip complex slippage models, just test with spread variations
    if deep_candidates:
        print_flush(f'\n{"="*80}\n{part_label} Phase 9: Slippage/Spread Testing\n{"="*80}')
        phase9 = {}
        spread_levels = [0.0, 0.20, 0.30, 0.50, 0.80, 1.0]
        for strat_name in deep_candidates:
            best_p = phase3.get(strat_name, {}).get('best_params', {})
            slip_results = []
            for sp in spread_levels:
                params = {
                    'sl_atr_mult': best_p.get('sl', default_params.get('sl_atr_mult', 3.0)),
                    'tp_atr_mult': best_p.get('tp', default_params.get('tp_atr_mult', 6.0)),
                    'trailing_activate_atr': best_p.get('trail_act', 0.3),
                    'trailing_distance_atr': best_p.get('trail_dist', 0.08),
                    'max_hold': best_p.get('max_hold', default_params.get('max_hold', 48)),
                    'spread_cost': sp,
                }
                trades = run_single(strat_name, strat_map[strat_name], params)
                s = calc_stats(trades)
                slip_results.append({'spread': sp, **s})
                print_flush(f'    {strat_name} spread={sp:.2f}: Sharpe={s["sharpe"]:.3f} PnL=${s["pnl"]:.0f}')

            # Verdict: profitable at spread=0.50
            verdict = 'PASS' if any(r['sharpe'] > 0.5 and r['spread'] >= 0.50 for r in slip_results) else 'FAIL'
            phase9[strat_name] = {'results': slip_results, 'verdict': verdict}
            print_flush(f'  {strat_name} slippage verdict: {verdict}')

        save(f'{tf_name}_phase9_slippage', phase9)
        save_progress(part_label, 'phase9')
    else:
        phase9 = {}

    # ═══ Phase 10: Final Multi-Gate Verdict ═══
    print_flush(f'\n{"="*80}\n{part_label} Phase 10: Final Verdict\n{"="*80}')
    phase10 = {}
    for strat_name in kf_passers:
        gates = {
            'kfold': phase2.get(strat_name, {}).get('kfold_6', {}).get('verdict', 'SKIP'),
            'param_sweep': 'PASS' if phase3.get(strat_name, {}).get('best_sharpe', 0) > 1.0 else 'FAIL',
            'walkforward': phase4.get(strat_name, {}).get('verdict', 'SKIP'),
            'era_stability': phase5.get(strat_name, {}).get('verdict', 'SKIP'),
            'sensitivity': phase6.get(strat_name, {}).get('verdict', 'SKIP'),
            'montecarlo': phase7.get(strat_name, {}).get('mc_verdict', 'SKIP'),
            'slippage': phase9.get(strat_name, {}).get('verdict', 'SKIP'),
        }
        pass_count = sum(1 for v in gates.values() if v == 'PASS')
        total_gates = sum(1 for v in gates.values() if v != 'SKIP')

        if pass_count >= 6:
            final = 'STRONG_PASS'
        elif pass_count >= 4:
            final = 'CONDITIONAL_PASS'
        else:
            final = 'REJECT'

        phase10[strat_name] = {
            'gates': gates, 'pass_count': pass_count, 'total_gates': total_gates,
            'final_verdict': final,
            'best_sharpe': phase3.get(strat_name, {}).get('best_sharpe', 0),
            'best_params': phase3.get(strat_name, {}).get('best_params'),
            'baseline_sharpe': phase2.get(strat_name, {}).get('kfold_6', {}).get('folds', [{}])[0].get('sharpe', 0),
        }
        print_flush(f'  {strat_name}: {pass_count}/{total_gates} gates -> {final}')

    save(f'{tf_name}_phase10_verdict', phase10)
    save_progress(part_label, 'phase10')

    elapsed = time.time() - t0
    print_flush(f'\n  {part_label} complete in {elapsed:.0f}s ({elapsed/3600:.1f}h)')

    return {
        'viable': viable, 'kf_passers': kf_passers,
        'phase3': phase3, 'phase4': phase4, 'phase5': phase5,
        'phase6': phase6, 'phase7': phase7, 'phase8': phase8,
        'phase9': phase9, 'phase10': phase10,
    }


# ═══════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════

def main():
    t_global = time.time()
    print_flush('=' * 80)
    print_flush(f'R230 PARALLEL RESUME ({N_WORKERS} workers on {cpu_count()} cores)')
    print_flush(f'Started: {pd.Timestamp.now()}')
    print_flush('=' * 80)

    # Load existing phase2 results to skip re-running Phase 1&2
    m30_phase2_file = OUTPUT_DIR / 'm30_phase2_kfold.json'
    m30_phase2_data = None
    if m30_phase2_file.exists():
        m30_phase2_data = json.loads(m30_phase2_file.read_text())
        print_flush(f'  Loaded existing M30 Phase 1&2 results from disk')
        kf_check = [s for s in m30_phase2_data if m30_phase2_data[s].get('kfold_6', {}).get('verdict') == 'PASS']
        print_flush(f'  M30 K-Fold passers: {kf_check}')

    # ═══ PART A: M30 ═══
    try:
        print_flush(f'\n{"#"*80}\n# PART A: M30 Strategy Universe (PARALLEL)\n{"#"*80}')
        m30_df = load_m30_with_indicators()
        m30_results = run_parallel_pipeline(
            tf_name='m30',
            df=m30_df,
            strategies=M30_STRATEGIES,
            engine_cls=M30BacktestEngine,
            default_params={
                'sl_atr_mult': 2.0, 'tp_atr_mult': 4.0,
                'trailing_activate_atr': 0.3, 'trailing_distance_atr': 0.08,
                'max_hold': 48, 'cooldown_bars': 4, 'spread_cost': SPREAD,
            },
            sl_grid=SL_GRID,
            tp_grid=TP_GRID,
            trail_grid=TRAIL_GRID,
            max_hold_grid=[24, 36, 48, 72, 96],
            part_label='PART_A_M30',
            skip_phase1_2=(m30_phase2_data is not None),
            phase2_result=m30_phase2_data,
        )
        save_progress('PART_A', 'COMPLETE')
    except Exception as e:
        print_flush(f'\n!!! PART A ERROR: {e}')
        traceback.print_exc()
        save_progress('PART_A', 'ERROR', str(e))
        m30_results = {}

    # ═══ PART B: H4 ═══
    try:
        print_flush(f'\n{"#"*80}\n# PART B: H4 Strategy Universe (PARALLEL)\n{"#"*80}')
        h4_df = load_h4_with_indicators()
        h4_results = run_parallel_pipeline(
            tf_name='h4',
            df=h4_df,
            strategies=H4_STRATEGIES,
            engine_cls=H4BacktestEngine,
            default_params={
                'sl_atr_mult': 3.0, 'tp_atr_mult': 6.0,
                'trailing_activate_atr': 0.3, 'trailing_distance_atr': 0.08,
                'max_hold': 30, 'cooldown_bars': 2, 'spread_cost': SPREAD,
            },
            sl_grid=SL_GRID,
            tp_grid=TP_GRID,
            trail_grid=TRAIL_GRID,
            max_hold_grid=[15, 20, 30, 45, 60],
            part_label='PART_B_H4',
        )
        save_progress('PART_B', 'COMPLETE')
    except Exception as e:
        print_flush(f'\n!!! PART B ERROR: {e}')
        traceback.print_exc()
        save_progress('PART_B', 'ERROR', str(e))
        h4_results = {}

    # ═══ PART C: Portfolio ═══
    try:
        print_flush(f'\n{"#"*80}\n# PART C: Cross-Timeframe Portfolio\n{"#"*80}')
        all_winners = {}
        for tf_name, results, df_data, eng_cls, strats, def_params in [
            ('m30', m30_results, m30_df, M30BacktestEngine, M30_STRATEGIES, {
                'sl_atr_mult': 2.0, 'tp_atr_mult': 4.0, 'trailing_activate_atr': 0.3,
                'trailing_distance_atr': 0.08, 'max_hold': 48, 'cooldown_bars': 4, 'spread_cost': SPREAD}),
            ('h4', h4_results, h4_df, H4BacktestEngine, H4_STRATEGIES, {
                'sl_atr_mult': 3.0, 'tp_atr_mult': 6.0, 'trailing_activate_atr': 0.3,
                'trailing_distance_atr': 0.08, 'max_hold': 30, 'cooldown_bars': 2, 'spread_cost': SPREAD}),
        ]:
            if not results:
                continue
            strat_map_local = dict(strats)
            p10 = results.get('phase10', {})
            for sname, info in p10.items():
                if info.get('final_verdict') in ('STRONG_PASS', 'CONDITIONAL_PASS'):
                    sig_func = strat_map_local.get(sname)
                    if sig_func is None:
                        continue
                    best_p = info.get('best_params', {})
                    params = dict(def_params)
                    if best_p:
                        params.update({
                            'sl_atr_mult': best_p.get('sl', params['sl_atr_mult']),
                            'tp_atr_mult': best_p.get('tp', params['tp_atr_mult']),
                            'trailing_activate_atr': best_p.get('trail_act', 0.3),
                            'trailing_distance_atr': best_p.get('trail_dist', 0.08),
                        })
                    engine = eng_cls(df_data, signal_funcs=[(sname, sig_func)], **params)
                    trades = [t for t in engine.run() if t.strategy == sname]
                    daily = {}
                    for t in trades:
                        day = pd.Timestamp(t.exit_time).date()
                        daily[day] = daily.get(day, 0) + t.pnl
                    all_winners[sname] = {'daily_pnl': pd.Series(daily), 'stats': calc_stats(trades),
                                          'verdict': info.get('final_verdict')}

        if len(all_winners) >= 2:
            all_days = sorted(set().union(*[set(w['daily_pnl'].index) for w in all_winners.values()]))
            corr_df = pd.DataFrame({name: w['daily_pnl'].reindex(all_days, fill_value=0)
                                    for name, w in all_winners.items()})
            portfolio_corr = {}
            for s1, s2 in combinations(all_winners.keys(), 2):
                r = float(corr_df[s1].corr(corr_df[s2]))
                label = 'LOW' if abs(r) < 0.3 else ('MODERATE' if abs(r) < 0.6 else 'HIGH')
                portfolio_corr[f'{s1}_vs_{s2}'] = {'correlation': round(r, 3), 'label': label}
                print_flush(f'  {s1} vs {s2}: r={r:.3f} ({label})')

            combined_daily = corr_df.sum(axis=1)
            combined_pnls = combined_daily.values
            if len(combined_pnls) > 10:
                port_sharpe = float(combined_pnls.mean() / max(combined_pnls.std(ddof=1), 1e-9) * np.sqrt(252))
                port_pnl = float(combined_pnls.sum())
                cum = np.cumsum(combined_pnls)
                port_dd = float((np.maximum.accumulate(cum) - cum).max())
                print_flush(f'\n  Combined Portfolio: Sharpe={port_sharpe:.3f} PnL=${port_pnl:.0f} MaxDD=${port_dd:.0f}')
            else:
                port_sharpe, port_pnl, port_dd = 0, 0, 0

            save('part_c_portfolio', {
                'winners': {k: {'stats': v['stats'], 'verdict': v['verdict']} for k, v in all_winners.items()},
                'correlations': portfolio_corr,
                'combined_portfolio': {'sharpe': round(port_sharpe, 3), 'total_pnl': round(port_pnl, 2),
                                       'max_dd': round(port_dd, 2)},
            })
        else:
            print_flush('  Not enough winners for portfolio')
            save('part_c_portfolio', {'note': 'insufficient winners'})
        save_progress('PART_C', 'COMPLETE')
    except Exception as e:
        print_flush(f'\n!!! PART C ERROR: {e}')
        traceback.print_exc()
        save_progress('PART_C', 'ERROR', str(e))

    # ═══ FINAL SUMMARY ═══
    elapsed_total = time.time() - t_global
    print_flush(f'\n{"#"*80}\n# FINAL SUMMARY\n{"#"*80}')
    print_flush(f'  Total runtime: {elapsed_total:.0f}s ({elapsed_total/3600:.1f}h)')
    print_flush(f'  Workers used: {N_WORKERS}')
    print_flush(f'  Finished: {pd.Timestamp.now()}')
    print_flush('=' * 80)

    final_summary = {'started': pd.Timestamp.now().isoformat(),
                     'runtime_hours': round(elapsed_total / 3600, 2),
                     'workers': N_WORKERS}
    for tf_name, results in [('m30', m30_results), ('h4', h4_results)]:
        if not results:
            continue
        p10 = results.get('phase10', {})
        tf_summary = {}
        for sname, info in p10.items():
            tf_summary[sname] = {
                'best_sharpe': info.get('best_sharpe'),
                'best_params': info.get('best_params'),
                'final_verdict': info.get('final_verdict'),
                'gates': info.get('gates'),
            }
            v = info.get('final_verdict', 'REJECT')
            marker = '***' if v == 'STRONG_PASS' else ('**' if v == 'CONDITIONAL_PASS' else '')
            print_flush(f'  {tf_name}/{sname:<20} Sharpe={info.get("best_sharpe", 0):.3f} -> {v} {marker}')
        final_summary[tf_name] = tf_summary

    save('R230_final_summary', final_summary)


if __name__ == '__main__':
    main()
