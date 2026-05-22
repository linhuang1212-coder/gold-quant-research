#!/usr/bin/env python3
"""R248 — cap_atr=1.0 Deep Multi-Gate Validation (PARALLEL)
============================================================
Per `parameter-change-protocol.md`, validates cap_atr=1.0 against
the 6 mandatory gates. Uses multiprocessing for VPS parallelism.

Gates:
  1. K-Fold 6/6 — ≥5/6 positive Sharpe, cap=1.0 > cap=0 in ≥4/6 folds
  2. Walk-Forward >70% — cap=1.0 outperforms in ≥6/8 OOS windows
  3. Era stability — all 4 eras positive Sharpe at cap=1.0
  4. Parameter sensitivity Δ < 0.5 — Sharpe stable within ±0.5 cap perturbation
  5. Monte Carlo P(Sharpe>0) > 99% — bootstrap-confirmed
  6. Realistic cost test Sharpe > 1.0 — survives spread+slippage

Parallelism: 16 worker processes via multiprocessing.Pool
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
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUTPUT_DIR = Path("results/r248_cap1_deep_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ERAS = [
    ('Pre-COVID', '2015-01-01', '2019-12-31'),
    ('COVID+Recovery', '2020-01-01', '2021-12-31'),
    ('Tightening', '2022-01-01', '2023-12-31'),
    ('Recent', '2024-01-01', '2026-05-13'),
]


# ═══════════════════════════════════════════════════════════════
# Worker-side init + helpers (run in each subprocess)
# ═══════════════════════════════════════════════════════════════
_data = None
_init_done = False


def _worker_init():
    """Per-worker setup: load data + patch indicators."""
    global _data, _init_done
    from backtest.runner import DataBundle, LIVE_PARITY_KWARGS, REALISTIC_COST_KWARGS
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
    """One BacktestEngine run with given config + data slice.

    task = {
        'id': str,
        'kind': 'kfold'|'wf'|'full'|'realistic',
        'start': ISO date,
        'end': ISO date,
        'cap': float,
        'kwargs_preset': 'live' | 'realistic',
        'test_filter_start': optional (for WF — filter trades to test period)
    }
    """
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

    preset = LIVE_PARITY_KWARGS if task['kwargs_preset'] == 'live' else REALISTIC_COST_KWARGS
    kwargs = {**preset, 'maxloss_cap_atr_mult': task['cap'], 'maxloss_cap': 0,
              'max_positions': 4}

    t0 = time.time()
    try:
        eng = BacktestEngine(m15, h1, **kwargs)
        trades = eng.run()
        cap_fires = getattr(eng, 'maxloss_cap_count', 0)
    except Exception as ex:
        return {'id': task['id'], 'error': str(ex), 'elapsed': time.time() - t0}

    # Optional filter (for WF)
    if task.get('test_filter_start'):
        trades = _filter_period(trades, task['test_filter_start'], task['end'])

    stats = _calc_stats(trades)
    # Era breakdown for full runs
    eras = {}
    if task['kind'] in ('full', 'realistic'):
        for era_name, es, ee in ERAS:
            eras[era_name] = _calc_stats(_filter_period(trades, es, ee))

    # Bootstrap distribution (only for full)
    boot = None
    if task['kind'] == 'full':
        pnls = np.array([_trade_attr(t, 'pnl', 0) for t in trades])
        if len(pnls) >= 10:
            rng = np.random.default_rng(42 + int(task['cap'] * 10))
            sharpes = np.zeros(1000)
            for i in range(1000):
                sample = rng.choice(pnls, size=len(pnls), replace=True)
                sharpes[i] = sample.mean() / sample.std() * np.sqrt(252) if sample.std() > 0 else 0
            boot = {
                'mean': round(float(sharpes.mean()), 3),
                'std': round(float(sharpes.std()), 3),
                'p_positive': round(float((sharpes > 0).mean()), 4),
                'sharpes': sharpes.tolist(),  # for cross-comparison
            }

    elapsed = time.time() - t0
    return {
        'id': task['id'],
        'kind': task['kind'],
        'start': task['start'],
        'end': task['end'],
        'cap': task['cap'],
        'preset': task['kwargs_preset'],
        'stats': stats,
        'cap_fires': cap_fires,
        'eras': eras,
        'bootstrap': boot,
        'elapsed': round(elapsed, 1),
    }


# ═══════════════════════════════════════════════════════════════
# Main (parallel orchestration)
# ═══════════════════════════════════════════════════════════════
def build_tasks():
    """Build the full list of tasks to run in parallel."""
    tasks = []

    # K-Fold (k=6) — 12 tasks
    fold_boundaries = pd.date_range('2015-01-01', '2026-05-13', periods=7)
    for fold_idx in range(6):
        start = fold_boundaries[fold_idx].strftime('%Y-%m-%d')
        end = fold_boundaries[fold_idx + 1].strftime('%Y-%m-%d')
        for cap in [0, 1.0]:
            tasks.append({
                'id': f'kfold_{fold_idx+1}_cap{cap}',
                'kind': 'kfold', 'start': start, 'end': end,
                'cap': cap, 'kwargs_preset': 'live',
            })

    # Walk-Forward — 8 windows × 2 caps = 16 tasks
    test_starts = ['2019-01-01', '2020-01-01', '2021-01-01', '2022-01-01',
                   '2023-01-01', '2024-01-01', '2025-01-01', '2026-01-01']
    test_ends = ['2019-12-31', '2020-12-31', '2021-12-31', '2022-12-31',
                 '2023-12-31', '2024-12-31', '2025-12-31', '2026-05-13']
    for i, (ts, te) in enumerate(zip(test_starts, test_ends)):
        # Need warmup period for indicators
        ext_start = (pd.Timestamp(ts) - pd.Timedelta(days=120)).strftime('%Y-%m-%d')
        for cap in [0, 1.0]:
            tasks.append({
                'id': f'wf_{i+1}_cap{cap}',
                'kind': 'wf', 'start': ext_start, 'end': te,
                'cap': cap, 'kwargs_preset': 'live',
                'test_filter_start': ts,
            })

    # Full history (era stability + bootstrap) — 2 tasks
    for cap in [0, 1.0]:
        tasks.append({
            'id': f'full_cap{cap}',
            'kind': 'full', 'start': '2015-01-01', 'end': '2026-05-13',
            'cap': cap, 'kwargs_preset': 'live',
        })

    # Realistic cost — 2 tasks
    for cap in [0, 1.0]:
        tasks.append({
            'id': f'realistic_cap{cap}',
            'kind': 'realistic', 'start': '2015-01-01', 'end': '2026-05-13',
            'cap': cap, 'kwargs_preset': 'realistic',
        })

    return tasks


def main():
    t0 = time.time()
    print('=' * 80)
    print('R248: cap_atr=1.0 Deep Multi-Gate Validation (PARALLEL, 16 workers)')
    print('=' * 80)

    tasks = build_tasks()
    print(f'\n  Total tasks: {len(tasks)}')
    print(f'  Workers: 16')
    print(f'  Expected ~5-10 min total')

    # Use mp.Pool with worker init
    mp.set_start_method('fork', force=True)
    with mp.Pool(processes=16, initializer=_worker_init) as pool:
        print('\n  Spinning up 16 workers (each loads data)...')
        # imap to get results as completed
        results = []
        for i, r in enumerate(pool.imap_unordered(_run_single, tasks, chunksize=1)):
            results.append(r)
            print(f'  [{i+1}/{len(tasks)}] {r["id"]} ({r["elapsed"]}s) '
                  f'Sharpe={r["stats"]["sharpe"]:+.2f} N={r["stats"]["n"]}',
                  flush=True)

    # ─── Aggregate ────────────────────────────────────────────
    by_id = {r['id']: r for r in results}

    # K-Fold analysis
    kfold = []
    for fold_idx in range(6):
        r0 = by_id[f'kfold_{fold_idx+1}_cap0']
        r1 = by_id[f'kfold_{fold_idx+1}_cap1.0']
        d_sharpe = r1['stats']['sharpe'] - r0['stats']['sharpe']
        kfold.append({
            'fold': fold_idx + 1,
            'start': r0['start'], 'end': r0['end'],
            'cap0': r0['stats'], 'cap1.0': r1['stats'],
            'delta_sharpe': round(d_sharpe, 3),
            'cap1_wins': d_sharpe > 0,
        })
    cap1_wins_kf = sum(1 for f in kfold if f['cap1_wins'])
    cap1_pos_kf = sum(1 for f in kfold if f['cap1.0']['sharpe'] > 0)
    pass_kfold = cap1_wins_kf >= 4 and cap1_pos_kf >= 5

    # Walk-Forward
    wf = []
    for i in range(8):
        r0 = by_id[f'wf_{i+1}_cap0']
        r1 = by_id[f'wf_{i+1}_cap1.0']
        d_sharpe = r1['stats']['sharpe'] - r0['stats']['sharpe']
        wf.append({
            'window': i + 1,
            'test_start': r0.get('start'), 'end': r0['end'],
            'cap0': r0['stats'], 'cap1.0': r1['stats'],
            'delta_sharpe': round(d_sharpe, 3),
            'cap1_wins': d_sharpe > 0,
        })
    cap1_wins_wf = sum(1 for w in wf if w['cap1_wins'])
    pass_wf = cap1_wins_wf >= 6

    # Era stability + Monte Carlo
    full0 = by_id['full_cap0']
    full1 = by_id['full_cap1.0']
    era_all_pos = all(full1['eras'][k]['sharpe'] > 0 for k in full1['eras'])
    mc = {
        'cap0': full0['bootstrap'],
        'cap1.0': full1['bootstrap'],
    }
    if mc['cap0'] and mc['cap1.0']:
        bs0 = np.array(mc['cap0']['sharpes'])
        bs1 = np.array(mc['cap1.0']['sharpes'])
        mc['p_cap1_beats_cap0'] = round(float((bs1 > bs0).mean()), 4)
        pass_mc = mc['cap1.0']['p_positive'] > 0.99 and mc['p_cap1_beats_cap0'] > 0.90
        # Don't save the raw sharpe arrays to JSON (huge)
        mc['cap0'] = {k: v for k, v in mc['cap0'].items() if k != 'sharpes'}
        mc['cap1.0'] = {k: v for k, v in mc['cap1.0'].items() if k != 'sharpes'}
    else:
        pass_mc = False

    # Realistic cost
    real0 = by_id['realistic_cap0']
    real1 = by_id['realistic_cap1.0']
    pass_realistic = real1['stats']['sharpe'] > 1.0

    # Parameter sensitivity (from R247)
    p4 = {'pass': False, 'note': 'R247 not loaded'}
    r247_path = Path('results/r247_engine_cap_sweep/R247_results.json')
    if r247_path.exists():
        r247 = json.loads(r247_path.read_text())
        rows = r247.get('phase_a_live5_portfolio_sweep', [])
        s_by_cap = {r['cap_atr']: r['port_sharpe'] for r in rows}
        if 1.0 in s_by_cap and 1.5 in s_by_cap:
            d = abs(s_by_cap[1.0] - s_by_cap[1.5])
            p4 = {
                'cap_1.0_sharpe': s_by_cap[1.0],
                'cap_1.5_sharpe': s_by_cap[1.5],
                'sensitivity_at_+0.5': round(d, 3),
                'pass': d < 0.5,
            }

    gates = {
        'gate1_kfold': pass_kfold,
        'gate2_walkforward': pass_wf,
        'gate3_era_stability': era_all_pos,
        'gate4_param_sensitivity': p4.get('pass', False),
        'gate5_monte_carlo': pass_mc,
        'gate6_realistic_cost': pass_realistic,
    }
    n_pass = sum(gates.values())

    print('\n' + '=' * 80)
    print('SUMMARY')
    print('=' * 80)

    print('\n  K-Fold (6 folds):')
    print(f'  {"Fold":>4}  {"cap=0":>20}  {"cap=1.0":>20}  {"ΔS":>7}  win?')
    for f in kfold:
        print(f"  {f['fold']:>4}  S={f['cap0']['sharpe']:+5.2f} N={f['cap0']['n']:>5} | "
              f"S={f['cap1.0']['sharpe']:+5.2f} N={f['cap1.0']['n']:>5} | "
              f"{f['delta_sharpe']:+5.2f}  {'Y' if f['cap1_wins'] else '.'}")
    print(f'  cap=1.0 wins {cap1_wins_kf}/6, positive in {cap1_pos_kf}/6 folds → '
          f'{"PASS" if pass_kfold else "FAIL"}')

    print('\n  Walk-Forward (8 windows):')
    for w in wf:
        print(f"  W{w['window']}  end={w['end']}  "
              f"S0={w['cap0']['sharpe']:+5.2f} S1={w['cap1.0']['sharpe']:+5.2f} "
              f"ΔS={w['delta_sharpe']:+5.2f}  {'Y' if w['cap1_wins'] else '.'}")
    print(f'  cap=1.0 wins {cap1_wins_wf}/8 → {"PASS" if pass_wf else "FAIL"}')

    print('\n  Era Stability (cap=1.0):')
    for k in ['Pre-COVID', 'COVID+Recovery', 'Tightening', 'Recent']:
        e1 = full1['eras'][k]
        e0 = full0['eras'][k]
        print(f"  {k:<22}  cap=0 S={e0['sharpe']:+5.2f}  cap=1.0 S={e1['sharpe']:+5.2f}")
    print(f'  All 4 eras cap=1.0 positive: {"YES" if era_all_pos else "NO"} → '
          f'{"PASS" if era_all_pos else "FAIL"}')

    print('\n  Monte Carlo (1000 resamples each):')
    if mc.get('cap0') and mc.get('cap1.0'):
        print(f"  cap=0   bootstrap: {mc['cap0']['mean']:.2f} ± {mc['cap0']['std']:.2f}, "
              f"P(>0)={mc['cap0']['p_positive']:.2%}")
        print(f"  cap=1.0 bootstrap: {mc['cap1.0']['mean']:.2f} ± {mc['cap1.0']['std']:.2f}, "
              f"P(>0)={mc['cap1.0']['p_positive']:.2%}")
        print(f"  P(cap=1.0 > cap=0) = {mc['p_cap1_beats_cap0']:.2%}")
    print(f'  → {"PASS" if pass_mc else "FAIL"}')

    print('\n  Realistic Cost:')
    print(f"  cap=0  : Sharpe={real0['stats']['sharpe']:+.2f}  PnL=${real0['stats']['pnl']:.0f}")
    print(f"  cap=1.0: Sharpe={real1['stats']['sharpe']:+.2f}  PnL=${real1['stats']['pnl']:.0f}")
    print(f'  → {"PASS" if pass_realistic else "FAIL"} (need Sharpe > 1.0)')

    print(f'\n  Parameter Sensitivity (cap=1.0 vs cap=1.5 from R247): {p4}')

    print('\n' + '=' * 80)
    print(f'FINAL: {n_pass}/6 gates passed')
    print('=' * 80)
    for g, p in gates.items():
        print(f'  {"PASS" if p else "FAIL"}: {g}')

    if n_pass >= 5:
        verdict = 'DEPLOY (with paper trade 1-2 weeks)'
    elif n_pass >= 4:
        verdict = 'HOLD — gather more evidence on failing gates'
    else:
        verdict = 'REJECT'
    print(f'\n  Verdict: {verdict}')

    output = {
        'gates': gates,
        'n_pass': n_pass,
        'verdict': verdict,
        'phase1_kfold': {'folds': kfold, 'pass': pass_kfold,
                         'cap1_wins': cap1_wins_kf, 'cap1_positive': cap1_pos_kf},
        'phase2_walk_forward': {'windows': wf, 'pass': pass_wf,
                                'cap1_wins': cap1_wins_wf},
        'phase3_era_stability': {'cap0': full0['eras'], 'cap1.0': full1['eras'],
                                 'pass': era_all_pos},
        'phase4_param_sensitivity': p4,
        'phase5_monte_carlo': {'mc': mc, 'pass': pass_mc},
        'phase6_realistic_cost': {'cap0': real0['stats'], 'cap1.0': real1['stats'],
                                  'pass': pass_realistic},
        'full_history': {'cap0': full0['stats'], 'cap1.0': full1['stats']},
        'all_results': results,
        'runtime_sec': round(time.time() - t0, 1),
    }
    out_json = OUTPUT_DIR / 'R248_results.json'
    out_json.write_text(json.dumps(output, indent=2, default=str))
    print(f'\nSaved -> {out_json}')
    print(f'Total runtime: {(time.time() - t0):.0f}s')


if __name__ == '__main__':
    main()
