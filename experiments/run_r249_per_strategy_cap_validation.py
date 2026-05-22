#!/usr/bin/env python3
"""R249 — Per-Strategy cap_atr=1.0 Deep Validation (PARALLEL)
=================================================================
Validates the **selective** application of cap_atr=1.0:
  - TSMOM, SESS_BO, Dual Thrust  → cap_atr_mult=1.0
  - Keltner, M15_RSI, ORB, PSAR  → no cap (None)

This is the R247 recommendation. R248 tested *global* cap=1.0 and got
3/6 (REJECT). R249 tests the *selective* per-strategy approach.

Compares 3 configurations:
  - baseline_cap0:   all strategies cap=0 (R246 baseline)
  - global_cap1:     all strategies cap_atr_mult=1.0 (R248 setup)
  - per_strategy:    cap=1.0 only on TSMOM/SESS_BO/Dual Thrust (R247 reco)

Gates (per `parameter-change-protocol.md`):
  1. K-Fold 6/6
  2. Walk-Forward 8 windows
  3. Era stability (4 eras)
  4. Parameter sensitivity (already done by R247)
  5. Monte Carlo
  6. Realistic cost

Uses multiprocessing for VPS parallelism (16 workers).
"""
from __future__ import annotations
import sys
import os
import json
import time
import multiprocessing as mp
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUTPUT_DIR = Path("results/r249_per_strategy_cap")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ERAS = [
    ('Pre-COVID', '2015-01-01', '2019-12-31'),
    ('COVID+Recovery', '2020-01-01', '2021-12-31'),
    ('Tightening', '2022-01-01', '2023-12-31'),
    ('Recent', '2024-01-01', '2026-05-13'),
]

PER_STRAT_CAPS = {
    'tsmom': 1.0,
    'sess_bo': 1.0,
    'dual_thrust': 1.0,
}

CONFIGS = {
    'baseline_cap0': {
        'maxloss_cap_atr_mult': 0,
        'maxloss_cap_atr_mult_by_strategy': None,
    },
    'global_cap1': {
        'maxloss_cap_atr_mult': 1.0,
        'maxloss_cap_atr_mult_by_strategy': None,
    },
    'per_strategy': {
        'maxloss_cap_atr_mult': 0,
        'maxloss_cap_atr_mult_by_strategy': PER_STRAT_CAPS,
    },
}


# ═══════════════════════════════════════════════════════════════
# Worker setup
# ═══════════════════════════════════════════════════════════════
_data = None
_init_done = False


def _worker_init():
    global _data, _init_done
    from backtest.runner import DataBundle
    import indicators as signals_mod
    from experiments.run_r209v2_non_keltner_audit import (
        check_psar_signal, check_sess_bo_signal,
        check_dual_thrust_signal, check_chandelier_signal,
    )
    import experiments.run_r209v2_non_keltner_audit as r209_mod

    LIVE5_CONFIG = {
        'psar': {'enabled': False},
        'sess_bo': {'enabled': True, 'broker_gmt_offset': 0,
                    'session_hour_gmt': 12, 'lookback_bars': 4,
                    'sl_atr': 4.5, 'tp_atr': 4.0},
        'dual_thrust': {'enabled': True, 'n_bars': 6, 'k_up': 0.5, 'k_down': 0.5,
                        'sl_atr': 6.0, 'tp_atr': 8.0,
                        'trail_act_atr': 0.06, 'trail_dist_atr': 0.01},
        'chandelier': {'enabled': False},
    }

    r209_mod.LIVE_STRAT_CONFIGS = LIVE5_CONFIG
    original_scan = signals_mod.scan_all_signals

    def patched_scan(df_h1, df_m15=None, **kwargs):
        signals = original_scan(df_h1, df_m15, **kwargs) or []
        extras = []
        for fn in [check_psar_signal, check_sess_bo_signal,
                   check_dual_thrust_signal, check_chandelier_signal]:
            sig = fn(df_h1)
            if sig is not None:
                extras.append(sig)
        return signals + extras

    signals_mod.scan_all_signals = patched_scan

    _data = DataBundle.load_default()
    _init_done = True
    print(f'  [worker {os.getpid()}] initialized', flush=True)


def _trade_attr(t, name, default=None):
    if hasattr(t, name):
        return getattr(t, name)
    if isinstance(t, dict):
        return t.get(name, default)
    return default


def _calc_stats(trades):
    if not trades:
        return {'n': 0, 'sharpe': 0.0, 'pnl': 0.0, 'win_rate': 0.0, 'max_dd': 0.0}
    pnls = np.array([_trade_attr(t, 'pnl', 0) for t in trades])
    wr = float((pnls > 0).mean() * 100)
    cum = np.cumsum(pnls)
    dd = float((np.maximum.accumulate(cum) - cum).max())
    sharpe = float(pnls.mean() / pnls.std() * np.sqrt(252)) if pnls.std() > 0 else 0.0
    return {'n': len(pnls), 'sharpe': round(sharpe, 3),
            'pnl': round(float(pnls.sum()), 2),
            'win_rate': round(wr, 2), 'max_dd': round(dd, 2)}


def _stats_by_strategy(trades):
    by_strat = {}
    for t in trades:
        s = _trade_attr(t, 'strategy', 'unknown')
        by_strat.setdefault(s, []).append(t)
    return {k: _calc_stats(v) for k, v in by_strat.items()}


def _filter_period(trades, start, end):
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


def _run_single(task):
    """Run one BacktestEngine instance."""
    from backtest.engine import BacktestEngine
    from backtest.runner import LIVE_PARITY_KWARGS, REALISTIC_COST_KWARGS
    import indicators as signals_mod
    from indicators import get_orb_strategy

    if not _init_done:
        _worker_init()

    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False
    get_orb_strategy().reset_daily()

    s = pd.Timestamp(task['start'], tz='UTC')
    e = pd.Timestamp(task['end'], tz='UTC')
    m15 = _data.m15_df[(_data.m15_df.index >= s) & (_data.m15_df.index <= e)]
    h1 = _data.h1_df[(_data.h1_df.index >= s) & (_data.h1_df.index <= e)]

    preset = LIVE_PARITY_KWARGS if task['preset'] == 'live' else REALISTIC_COST_KWARGS
    cfg = CONFIGS[task['config']]
    kwargs = {**preset, **cfg, 'maxloss_cap': 0, 'max_positions': 4}

    t0 = time.time()
    try:
        eng = BacktestEngine(m15, h1, **kwargs)
        trades = eng.run()
        cap_fires = getattr(eng, 'maxloss_cap_count', 0)
        cap_fires_by_strat = dict(getattr(eng, 'maxloss_cap_count_by_strategy', {}))
    except Exception as ex:
        return {'id': task['id'], 'error': str(ex), 'elapsed': time.time() - t0}

    if task.get('test_filter_start'):
        trades = _filter_period(trades, task['test_filter_start'], task['end'])

    stats = _calc_stats(trades)
    eras = {}
    by_strat = {}
    if task['kind'] in ('full', 'realistic'):
        for era_name, es, ee in ERAS:
            eras[era_name] = _calc_stats(_filter_period(trades, es, ee))
        by_strat = _stats_by_strategy(trades)

    boot = None
    if task['kind'] == 'full':
        pnls = np.array([_trade_attr(t, 'pnl', 0) for t in trades])
        if len(pnls) >= 10:
            rng = np.random.default_rng(42 + hash(task['config']) % 1000)
            sharpes = np.zeros(1000)
            for i in range(1000):
                sample = rng.choice(pnls, size=len(pnls), replace=True)
                sharpes[i] = sample.mean() / sample.std() * np.sqrt(252) if sample.std() > 0 else 0
            boot = {
                'mean': round(float(sharpes.mean()), 3),
                'std': round(float(sharpes.std()), 3),
                'p_positive': round(float((sharpes > 0).mean()), 4),
                'sharpes': sharpes.tolist(),
            }

    return {
        'id': task['id'],
        'kind': task['kind'],
        'config': task['config'],
        'start': task['start'],
        'end': task['end'],
        'preset': task['preset'],
        'stats': stats,
        'eras': eras,
        'by_strategy': by_strat,
        'cap_fires': cap_fires,
        'cap_fires_by_strategy': cap_fires_by_strat,
        'bootstrap': boot,
        'elapsed': round(time.time() - t0, 1),
    }


# ═══════════════════════════════════════════════════════════════
# Task builder
# ═══════════════════════════════════════════════════════════════
def build_tasks():
    tasks = []
    configs_to_test = list(CONFIGS.keys())

    fold_boundaries = pd.date_range('2015-01-01', '2026-05-13', periods=7)
    for fold_idx in range(6):
        start = fold_boundaries[fold_idx].strftime('%Y-%m-%d')
        end = fold_boundaries[fold_idx + 1].strftime('%Y-%m-%d')
        for cfg in configs_to_test:
            tasks.append({
                'id': f'kfold_{fold_idx+1}_{cfg}',
                'kind': 'kfold', 'config': cfg, 'preset': 'live',
                'start': start, 'end': end,
            })

    # Walk-Forward (8 windows × 3 configs = 24 tasks)
    test_starts = ['2019-01-01', '2020-01-01', '2021-01-01', '2022-01-01',
                   '2023-01-01', '2024-01-01', '2025-01-01', '2026-01-01']
    test_ends = ['2019-12-31', '2020-12-31', '2021-12-31', '2022-12-31',
                 '2023-12-31', '2024-12-31', '2025-12-31', '2026-05-13']
    for i, (ts, te) in enumerate(zip(test_starts, test_ends)):
        ext_start = (pd.Timestamp(ts) - pd.Timedelta(days=120)).strftime('%Y-%m-%d')
        for cfg in configs_to_test:
            tasks.append({
                'id': f'wf_{i+1}_{cfg}',
                'kind': 'wf', 'config': cfg, 'preset': 'live',
                'start': ext_start, 'end': te,
                'test_filter_start': ts,
            })

    # Full history (era + bootstrap + per-strategy breakdown)
    for cfg in configs_to_test:
        tasks.append({
            'id': f'full_{cfg}',
            'kind': 'full', 'config': cfg, 'preset': 'live',
            'start': '2015-01-01', 'end': '2026-05-13',
        })

    # Realistic cost
    for cfg in configs_to_test:
        tasks.append({
            'id': f'realistic_{cfg}',
            'kind': 'realistic', 'config': cfg, 'preset': 'realistic',
            'start': '2015-01-01', 'end': '2026-05-13',
        })

    return tasks


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    print('=' * 80)
    print('R249: Per-Strategy cap_atr=1.0 Validation (PARALLEL, 16 workers)')
    print('=' * 80)
    print('\nConfigurations under test:')
    for k, v in CONFIGS.items():
        print(f'  {k}: {v}')

    tasks = build_tasks()
    print(f'\n  Total tasks: {len(tasks)}  (3 configs × {len(tasks)//3} runs each)')

    mp.set_start_method('fork', force=True)
    with mp.Pool(processes=16, initializer=_worker_init) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(_run_single, tasks, chunksize=1)):
            results.append(r)
            print(f'  [{i+1}/{len(tasks)}] {r["id"]} ({r["elapsed"]}s) '
                  f'S={r["stats"]["sharpe"]:+.2f} N={r["stats"]["n"]} '
                  f'caps={r["cap_fires"]}', flush=True)

    by_id = {r['id']: r for r in results}
    configs_to_test = list(CONFIGS.keys())

    # ─── K-Fold analysis ────
    print('\n' + '=' * 80)
    print('K-FOLD (6 folds × 3 configs)')
    print('=' * 80)
    print(f"  {'Fold':>4}  " + "  ".join(f"{c[:18]:>20}" for c in configs_to_test))
    kfold_summary = {}
    for cfg in configs_to_test:
        kfold_summary[cfg] = []
    for fold_idx in range(6):
        row = f"  {fold_idx+1:>4}  "
        for cfg in configs_to_test:
            r = by_id[f'kfold_{fold_idx+1}_{cfg}']
            kfold_summary[cfg].append(r['stats'])
            row += f"  S={r['stats']['sharpe']:+5.2f} N={r['stats']['n']:>5}"
        print(row)

    # Compare per_strategy vs baseline and global
    print('\n  per_strategy vs baseline_cap0:')
    wins_vs_baseline = 0
    for fold_idx in range(6):
        s_ps = kfold_summary['per_strategy'][fold_idx]['sharpe']
        s_bl = kfold_summary['baseline_cap0'][fold_idx]['sharpe']
        diff = s_ps - s_bl
        win = diff > 0
        wins_vs_baseline += int(win)
        print(f"    Fold {fold_idx+1}: per_strategy {s_ps:+.2f}  baseline {s_bl:+.2f}  Δ={diff:+.2f}  {'WIN' if win else '.'}")
    print(f"  per_strategy wins {wins_vs_baseline}/6 vs baseline")
    pass_kfold = wins_vs_baseline >= 4 and all(s['sharpe'] > 0 for s in kfold_summary['per_strategy'])

    # ─── Walk-Forward ────
    print('\n' + '=' * 80)
    print('WALK-FORWARD (8 windows × 3 configs)')
    print('=' * 80)
    wins_vs_baseline_wf = 0
    wf_data = []
    for i in range(8):
        rps = by_id[f'wf_{i+1}_per_strategy']['stats']
        rbl = by_id[f'wf_{i+1}_baseline_cap0']['stats']
        rgl = by_id[f'wf_{i+1}_global_cap1']['stats']
        diff = rps['sharpe'] - rbl['sharpe']
        win = diff > 0
        wins_vs_baseline_wf += int(win)
        wf_data.append({'window': i+1, 'baseline': rbl, 'global_cap1': rgl,
                       'per_strategy': rps, 'delta_vs_baseline': round(diff, 3),
                       'win': win})
        print(f"  W{i+1} end={by_id[f'wf_{i+1}_per_strategy']['end']}  "
              f"baseline S={rbl['sharpe']:+.2f}  global S={rgl['sharpe']:+.2f}  "
              f"per_strategy S={rps['sharpe']:+.2f}  Δ={diff:+.2f}  {'WIN' if win else '.'}")
    print(f"  per_strategy wins {wins_vs_baseline_wf}/8 vs baseline")
    pass_wf = wins_vs_baseline_wf >= 6

    # ─── Full + era ────
    full = {c: by_id[f'full_{c}'] for c in configs_to_test}
    real = {c: by_id[f'realistic_{c}'] for c in configs_to_test}

    print('\n' + '=' * 80)
    print('FULL HISTORY (11 years)')
    print('=' * 80)
    print(f"  {'Config':<20}  {'Sharpe':>8} {'N':>7} {'PnL':>12} {'MaxDD':>10} {'Caps':>6}")
    for c in configs_to_test:
        s = full[c]['stats']
        print(f"  {c:<20}  {s['sharpe']:>+8.2f} {s['n']:>7} ${s['pnl']:>10,.0f} ${s['max_dd']:>8,.0f} {full[c]['cap_fires']:>6}")

    print('\n  Per-strategy breakdown (full history, per_strategy config):')
    for k, v in full['per_strategy']['by_strategy'].items():
        cap_n = full['per_strategy']['cap_fires_by_strategy'].get(k, 0)
        print(f"    {k:<15}  S={v['sharpe']:+.2f}  N={v['n']:>6}  PnL=${v['pnl']:>9,.0f}  caps={cap_n}")

    print('\n  Per-strategy breakdown (full history, baseline_cap0):')
    for k, v in full['baseline_cap0']['by_strategy'].items():
        print(f"    {k:<15}  S={v['sharpe']:+.2f}  N={v['n']:>6}  PnL=${v['pnl']:>9,.0f}")

    print('\n' + '=' * 80)
    print('ERA STABILITY')
    print('=' * 80)
    for era_name, _, _ in ERAS:
        print(f"  {era_name:<20}  " + "  ".join(
            f"{c[:18]:>15}: S={full[c]['eras'][era_name]['sharpe']:+.2f}"
            for c in configs_to_test
        ))
    era_all_pos = all(full['per_strategy']['eras'][e]['sharpe'] > 0 for e, _, _ in ERAS)

    # ─── Monte Carlo ────
    print('\n' + '=' * 80)
    print('MONTE CARLO (1000 resamples each)')
    print('=' * 80)
    mc = {c: full[c]['bootstrap'] for c in configs_to_test}
    bs_ps = np.array(mc['per_strategy']['sharpes'])
    bs_bl = np.array(mc['baseline_cap0']['sharpes'])
    bs_gl = np.array(mc['global_cap1']['sharpes'])
    p_ps_beats_bl = float((bs_ps > bs_bl).mean())
    p_ps_beats_gl = float((bs_ps > bs_gl).mean())
    for c in configs_to_test:
        b = mc[c]
        print(f"  {c:<20}  mean={b['mean']:.2f} std={b['std']:.2f}  P(>0)={b['p_positive']:.2%}")
    print(f"  P(per_strategy > baseline_cap0) = {p_ps_beats_bl:.2%}")
    print(f"  P(per_strategy > global_cap1) = {p_ps_beats_gl:.2%}")
    pass_mc = mc['per_strategy']['p_positive'] > 0.99 and p_ps_beats_bl > 0.90

    # Strip raw arrays from saved data
    for c in configs_to_test:
        mc[c] = {k: v for k, v in mc[c].items() if k != 'sharpes'}

    # ─── Realistic cost ────
    print('\n' + '=' * 80)
    print('REALISTIC COST')
    print('=' * 80)
    for c in configs_to_test:
        s = real[c]['stats']
        print(f"  {c:<20}  S={s['sharpe']:+.2f}  N={s['n']:>6}  PnL=${s['pnl']:>9,.0f}")
    pass_realistic = real['per_strategy']['stats']['sharpe'] > 1.0
    # Relaxed: per_strategy beats baseline under realistic cost
    pass_realistic_v2 = real['per_strategy']['stats']['sharpe'] > real['baseline_cap0']['stats']['sharpe']

    # ─── Final ────
    gates = {
        'gate1_kfold': pass_kfold,
        'gate2_walk_forward': pass_wf,
        'gate3_era_stability': era_all_pos,
        'gate5_monte_carlo': pass_mc,
        'gate6_realistic_cost_absolute': pass_realistic,
        'gate6b_realistic_cost_relative': pass_realistic_v2,
    }
    n_pass = sum(gates.values())

    print('\n' + '=' * 80)
    print(f'FINAL: {n_pass}/{len(gates)} gates passed')
    print('=' * 80)
    for g, p in gates.items():
        print(f'  {"PASS" if p else "FAIL"}: {g}')

    if n_pass >= 5:
        verdict = 'DEPLOY per_strategy cap (TSMOM/SESS_BO/Dual Thrust = 1.0)'
    elif n_pass >= 4:
        verdict = 'HOLD — investigate failing gates'
    else:
        verdict = 'REJECT per_strategy cap'
    print(f'\n  Verdict: {verdict}')

    output = {
        'configs': CONFIGS,
        'gates': gates,
        'n_pass': n_pass,
        'verdict': verdict,
        'phase1_kfold': {'summary': kfold_summary, 'pass': pass_kfold,
                         'per_strategy_wins_vs_baseline': wins_vs_baseline},
        'phase2_walk_forward': {'windows': wf_data, 'pass': pass_wf,
                                'per_strategy_wins_vs_baseline': wins_vs_baseline_wf},
        'phase3_era_stability': {c: full[c]['eras'] for c in configs_to_test},
        'phase5_monte_carlo': {'mc': mc, 'p_ps_beats_bl': p_ps_beats_bl,
                               'p_ps_beats_gl': p_ps_beats_gl},
        'phase6_realistic_cost': {c: real[c]['stats'] for c in configs_to_test},
        'full_history': {c: {'stats': full[c]['stats'],
                            'by_strategy': full[c]['by_strategy'],
                            'cap_fires_by_strategy': full[c]['cap_fires_by_strategy']}
                        for c in configs_to_test},
        'all_results': results,
        'runtime_sec': round(time.time() - t0, 1),
    }
    out_json = OUTPUT_DIR / 'R249_results.json'
    out_json.write_text(json.dumps(output, indent=2, default=str))
    print(f'\nSaved -> {out_json}')
    print(f'Total runtime: {(time.time() - t0):.0f}s')


if __name__ == '__main__':
    main()
