#!/usr/bin/env python3
"""R239c: Keltner 0.06 lot Full Deployment Validation
======================================================
Per constraints.md: lot change -> must re-validate Cap + Portfolio MaxDD.
Full pipeline: K-Fold 6 + Walk-Forward + Monte Carlo + Era + Cap sweep.

Configs tested:
  1. 0.04 baseline (current live, no cap)
  2. 0.06 no cap
  3. 0.06 + cap $70
  4. 0.06 + cap $80
  5. 0.06 + cap $100

All use BacktestEngine + LIVE_PARITY_KWARGS.
"""
from __future__ import annotations
import sys
import json
import time
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.runner import DataBundle, LIVE_PARITY_KWARGS
from backtest.engine import BacktestEngine

OUTPUT_DIR = Path("results/r239c_lot06_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HOLDOUT_START = "2025-05-01"

ERA_SEGMENTS = {
    "Pre-COVID (2015-2019)":      ("2015-01-01", "2020-01-01"),
    "COVID+Recovery (2020-2021)":  ("2020-01-01", "2022-01-01"),
    "Tightening (2022-2023)":      ("2022-01-01", "2024-01-01"),
    "Recent (2024-2026)":          ("2024-01-01", "2026-06-01"),
}

KFOLD_BOUNDARIES = [
    ("2015-01-01", "2016-11-01"),
    ("2016-11-01", "2018-09-01"),
    ("2018-09-01", "2020-07-01"),
    ("2020-07-01", "2022-05-01"),
    ("2022-05-01", "2024-03-01"),
    ("2024-03-01", "2026-06-01"),
]

WF_TRAIN_MONTHS = 24
WF_TEST_MONTHS = 3


def pf(msg):
    print(msg, flush=True)


def daily_sharpe(trades):
    if not trades or len(trades) < 5:
        return -999
    daily = {}
    for t in trades:
        d = pd.Timestamp(t.exit_time).date()
        daily[d] = daily.get(d, 0.0) + t.pnl
    ds = np.array(list(daily.values()))
    if len(ds) < 10:
        return -999
    return round(float(ds.mean() / max(ds.std(ddof=1), 1e-9) * np.sqrt(252)), 3)


def full_metrics(trades):
    if not trades or len(trades) < 5:
        return {'sharpe': -999, 'n': 0, 'pnl': 0, 'wr': 0, 'pf': 0, 'max_dd': 0, 'pnl_per_trade': 0}
    pnls = np.array([t.pnl for t in trades])
    n = len(pnls)
    total = float(pnls.sum())
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    wr = len(wins) / n
    pf_val = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else 99.0
    cum = np.cumsum(pnls)
    max_dd = float((np.maximum.accumulate(cum) - cum).max())
    sh = daily_sharpe(trades)
    return {
        'sharpe': sh, 'n': n, 'pnl': round(total, 2), 'wr': round(wr, 4),
        'pf': round(pf_val, 3), 'max_dd': round(max_dd, 2),
        'pnl_per_trade': round(total / n, 2),
        'avg_win': round(float(wins.mean()), 2) if len(wins) else 0,
        'avg_loss': round(float(losses.mean()), 2) if len(losses) else 0,
    }


def run_engine(m15_df, h1_df, overrides):
    kw = {**LIVE_PARITY_KWARGS}
    kw.update(overrides)
    engine = BacktestEngine(m15_df, h1_df, **kw)
    trades = engine.run()
    kc = [t for t in trades if t.strategy == 'keltner']
    return kc, engine


def kfold_validate(m15_df, h1_df, overrides, k=6):
    results = []
    for i, (start, end) in enumerate(KFOLD_BOUNDARIES[:k]):
        start_ts = pd.Timestamp(start, tz='UTC')
        end_ts = pd.Timestamp(end, tz='UTC')
        fold_m15 = m15_df[(m15_df.index >= start_ts) & (m15_df.index < end_ts)]
        fold_h1 = h1_df[(h1_df.index >= start_ts) & (h1_df.index < end_ts)]
        if len(fold_m15) < 100 or len(fold_h1) < 100:
            results.append({'fold': i + 1, 'sharpe': -999, 'n': 0})
            continue
        trades, _ = run_engine(fold_m15, fold_h1, overrides)
        m = full_metrics(trades)
        results.append({'fold': i + 1, 'sharpe': m['sharpe'], 'n': m['n'], 'pnl': m['pnl']})
    sharpes = [r['sharpe'] for r in results if r['sharpe'] > -999]
    pass_count = sum(1 for s in sharpes if s > 0)
    return {
        'folds': results,
        'pass_count': pass_count,
        'total': len(sharpes),
        'verdict': 'PASS' if pass_count >= 4 else 'FAIL',
        'mean_sharpe': round(float(np.mean(sharpes)), 3) if sharpes else -999,
        'min_sharpe': round(float(np.min(sharpes)), 3) if sharpes else -999,
    }


def walk_forward(m15_df, h1_df, overrides):
    results = []
    all_dates = m15_df.index
    start_date = all_dates[0]
    end_date = all_dates[-1]
    wf_start = start_date + pd.DateOffset(months=WF_TRAIN_MONTHS)
    current = wf_start
    while current + pd.DateOffset(months=WF_TEST_MONTHS) <= end_date:
        train_start = current - pd.DateOffset(months=WF_TRAIN_MONTHS)
        test_end = current + pd.DateOffset(months=WF_TEST_MONTHS)
        test_m15 = m15_df[(m15_df.index >= current) & (m15_df.index < test_end)]
        test_h1 = h1_df[(h1_df.index >= current) & (h1_df.index < test_end)]
        if len(test_m15) < 50:
            current += pd.DateOffset(months=WF_TEST_MONTHS)
            continue
        trades, _ = run_engine(test_m15, test_h1, overrides)
        sh = daily_sharpe(trades)
        results.append({
            'period': f'{current.strftime("%Y-%m")}',
            'sharpe': sh, 'n': len(trades),
            'pnl': round(sum(t.pnl for t in trades), 2),
        })
        current += pd.DateOffset(months=WF_TEST_MONTHS)
    pass_count = sum(1 for r in results if r['sharpe'] > 0)
    return {
        'windows': len(results),
        'pass': pass_count,
        'ratio': f'{pass_count}/{len(results)}',
        'verdict': 'PASS' if pass_count / max(len(results), 1) >= 0.70 else 'FAIL',
    }


def monte_carlo(trades, n_sims=500, seed=42):
    if len(trades) < 20:
        return {'verdict': 'SKIP', 'p_positive': 0}
    rng = np.random.RandomState(seed)
    pnls = np.array([t.pnl for t in trades])
    n = len(pnls)
    positive = 0
    sharpes = []
    chunk = max(1, n // 200)
    usable = (n // chunk) * chunk
    for _ in range(n_sims):
        sample = rng.choice(pnls, size=n, replace=True)
        cum = np.cumsum(sample)
        total = float(cum[-1])
        if total > 0:
            positive += 1
        daily_approx = sample[:usable].reshape(-1, chunk).sum(axis=1)
        if len(daily_approx) > 5 and daily_approx.std() > 0:
            sh = float(daily_approx.mean() / daily_approx.std() * np.sqrt(252))
            sharpes.append(sh)
    p_pos = round(positive / n_sims * 100, 1)
    return {
        'n_sims': n_sims,
        'p_positive': p_pos,
        'mc_mean_sharpe': round(float(np.mean(sharpes)), 3) if sharpes else 0,
        'mc_5pct_sharpe': round(float(np.percentile(sharpes, 5)), 3) if sharpes else 0,
        'verdict': 'PASS' if p_pos >= 95 else 'FAIL',
    }


def era_stability(trades):
    results = {}
    for era_name, (start, end) in ERA_SEGMENTS.items():
        start_ts = pd.Timestamp(start, tz='UTC')
        end_ts = pd.Timestamp(end, tz='UTC')
        era_trades = [t for t in trades if start_ts <= pd.Timestamp(t.entry_time) < end_ts]
        m = full_metrics(era_trades)
        results[era_name] = m
    positive = sum(1 for v in results.values() if v['sharpe'] > 0)
    return results, positive


def loss_tail(trades):
    losses = sorted([t.pnl for t in trades if t.pnl < 0])
    if not losses:
        return {}
    arr = np.array(losses)
    return {
        'p50': round(float(np.percentile(arr, 50)), 2),
        'p75': round(float(np.percentile(arr, 75)), 2),
        'p90': round(float(np.percentile(arr, 90)), 2),
        'p95': round(float(np.percentile(arr, 95)), 2),
        'p99': round(float(np.percentile(arr, 99)), 2),
        'worst': round(float(arr.min()), 2),
        'n_loss': len(losses),
    }


def sensitivity_test(m15_df, h1_df, base_overrides, seed=42):
    """MC parameter perturbation: +-15% on key params, 30 runs."""
    rng = np.random.RandomState(seed)
    base_kw = {**LIVE_PARITY_KWARGS, **base_overrides}
    perturb_keys = ['trailing_activate_atr', 'trailing_distance_atr', 'sl_atr_mult', 'tp_atr_mult']
    sharpes = []
    for _ in range(30):
        kw = {**base_kw}
        for pk in perturb_keys:
            if pk in kw and isinstance(kw[pk], (int, float)) and kw[pk] > 0:
                kw[pk] = kw[pk] * rng.uniform(0.85, 1.15)
        try:
            engine = BacktestEngine(m15_df, h1_df, **kw)
            trades = engine.run()
            kc = [t for t in trades if t.strategy == 'keltner']
            sh = daily_sharpe(kc)
            sharpes.append(sh)
        except Exception:
            pass
    if not sharpes:
        return {'verdict': 'SKIP'}
    return {
        'n_runs': len(sharpes),
        'mean': round(float(np.mean(sharpes)), 3),
        'min': round(float(np.min(sharpes)), 3),
        'max': round(float(np.max(sharpes)), 3),
        'std': round(float(np.std(sharpes)), 3),
        'verdict': 'PASS' if min(sharpes) > 0 else 'FAIL',
    }


def validate_config(args):
    config_name, overrides, m15_path, h1_path, run_sensitivity = args
    try:
        m15_df = pd.read_pickle(m15_path)
        h1_df = pd.read_pickle(h1_path)

        pf(f'\n  [{config_name}] Running full-sample backtest...')
        trades, engine = run_engine(m15_df, h1_df, overrides)
        m = full_metrics(trades)
        exit_reasons = dict(Counter(t.exit_reason for t in trades))
        tail = loss_tail(trades)

        pf(f'    Full: Sharpe={m["sharpe"]}, N={m["n"]}, PnL=${m["pnl"]}, MaxDD=${m["max_dd"]}')

        # OOS
        holdout = pd.Timestamp(HOLDOUT_START, tz='UTC')
        oos_trades = [t for t in trades if pd.Timestamp(t.entry_time) >= holdout]
        oos_m = full_metrics(oos_trades)
        pf(f'    OOS: Sharpe={oos_m["sharpe"]}, N={oos_m["n"]}, PnL=${oos_m["pnl"]}')

        # K-Fold
        pf(f'    K-Fold 6...')
        kf = kfold_validate(m15_df, h1_df, overrides)
        pf(f'    KF: {kf["verdict"]} ({kf["pass_count"]}/{kf["total"]}), mean={kf["mean_sharpe"]}, min={kf["min_sharpe"]}')

        # Walk-Forward
        pf(f'    Walk-Forward (24m train / 3m test)...')
        wf = walk_forward(m15_df, h1_df, overrides)
        pf(f'    WF: {wf["verdict"]} ({wf["ratio"]})')

        # Monte Carlo
        pf(f'    Monte Carlo (500 sims)...')
        try:
            mc = monte_carlo(trades)
        except Exception as mc_err:
            pf(f'    MC error (non-fatal): {mc_err}')
            mc = {'verdict': 'SKIP', 'p_positive': 0, 'mc_mean_sharpe': 0, 'mc_5pct_sharpe': 0, 'n_sims': 0}
        pf(f'    MC: {mc["verdict"]} (P(>0)={mc.get("p_positive",0)}%, 5pct Sharpe={mc.get("mc_5pct_sharpe","N/A")})')

        # Era Stability
        eras, positive_eras = era_stability(trades)
        era_verdict = 'PASS' if positive_eras >= 3 else 'FAIL'
        pf(f'    Eras: {positive_eras}/4 positive → {era_verdict}')

        # Sensitivity (only for key configs)
        sens = {'verdict': 'SKIP'}
        if run_sensitivity:
            pf(f'    Sensitivity (30 MC param perturbations)...')
            sens = sensitivity_test(m15_df, h1_df, overrides)
            pf(f'    Sens: {sens["verdict"]} (min={sens.get("min","N/A")}, std={sens.get("std","N/A")})')

        # Overall verdict
        gates = [kf['verdict'], wf['verdict'], mc['verdict'], era_verdict]
        pass_gates = sum(1 for g in gates if g == 'PASS')
        overall = 'STRONG_PASS' if pass_gates == 4 else ('PASS' if pass_gates >= 3 else 'FAIL')

        pf(f'    *** {config_name}: {overall} ({pass_gates}/4 gates) ***')

        return {
            'config': config_name,
            'overrides': {k: v for k, v in overrides.items() if not callable(v)},
            'full': m,
            'oos': oos_m,
            'kfold': kf,
            'walk_forward': wf,
            'monte_carlo': mc,
            'eras': eras,
            'era_positive': positive_eras,
            'era_verdict': era_verdict,
            'sensitivity': sens,
            'exit_reasons': exit_reasons,
            'loss_tail': tail,
            'cap_hits': engine.maxloss_cap_count,
            'overall': overall,
            'gates_passed': pass_gates,
        }
    except Exception as e:
        pf(f'    ERROR: {e}')
        return {'config': config_name, 'error': str(e), 'traceback': traceback.format_exc()}


def main():
    t0 = time.time()
    pf('=' * 80)
    pf('R239c: KELTNER 0.06 LOT FULL DEPLOYMENT VALIDATION')
    pf('  Per constraints.md: lot change -> re-validate Cap + Portfolio MaxDD')
    pf('  Pipeline: K-Fold 6 + Walk-Forward + Monte Carlo + Era + Cap + Sensitivity')
    pf('=' * 80)

    pf('\n[1] Loading data...')
    data = DataBundle.load_default()
    pf(f'  M15: {len(data.m15_df)} bars | H1: {len(data.h1_df)} bars')

    cache_dir = OUTPUT_DIR / '_cache'
    cache_dir.mkdir(exist_ok=True)
    m15_path = str(cache_dir / 'm15.pkl')
    h1_path = str(cache_dir / 'h1.pkl')
    data.m15_df.to_pickle(m15_path)
    data.h1_df.to_pickle(h1_path)

    configs = [
        # Baseline
        ('A_0.04_no_cap_BASELINE', {
            'min_lot_size': 0.04, 'max_lot_size': 0.04, 'maxloss_cap': 0,
        }, True),
        # Proposed: 0.06 no cap
        ('B_0.06_no_cap', {
            'min_lot_size': 0.06, 'max_lot_size': 0.06, 'maxloss_cap': 0,
        }, True),
        # 0.06 + cap $70 (R239b best OOS)
        ('C_0.06_cap_70', {
            'min_lot_size': 0.06, 'max_lot_size': 0.06, 'maxloss_cap': 70,
        }, True),
        # 0.06 + cap $80 (legacy reference)
        ('D_0.06_cap_80', {
            'min_lot_size': 0.06, 'max_lot_size': 0.06, 'maxloss_cap': 80,
        }, False),
        # 0.06 + cap $100 (wider)
        ('E_0.06_cap_100', {
            'min_lot_size': 0.06, 'max_lot_size': 0.06, 'maxloss_cap': 100,
        }, False),
    ]

    # Run sequentially (each does internal K-Fold/WF which is CPU intensive)
    pf(f'\n[2] Running {len(configs)} configs through full validation pipeline...')
    all_results = []
    for name, overrides, run_sens in configs:
        t1 = time.time()
        result = validate_config((name, overrides, m15_path, h1_path, run_sens))
        elapsed = time.time() - t1
        pf(f'  {name} done in {elapsed:.0f}s')
        all_results.append(result)

    # ════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ════════════════════════════════════════════════════════════
    pf(f'\n\n{"="*80}')
    pf('SUMMARY')
    pf(f'{"="*80}')

    pf(f'\n  {"Config":<26} {"Verdict":<14} {"Sharpe":>7} {"OOS_Sh":>7} {"N":>6} {"KF":>5} {"WF":>6} {"MC%":>5} {"Era":>4} {"MaxDD":>8} {"PnL":>10}')
    pf(f'  {"-"*26} {"-"*14} {"-"*7} {"-"*7} {"-"*6} {"-"*5} {"-"*6} {"-"*5} {"-"*4} {"-"*8} {"-"*10}')
    for r in all_results:
        if 'error' in r:
            pf(f'  {r["config"]:<26} ERROR')
            continue
        kf_str = f'{r["kfold"]["pass_count"]}/{r["kfold"]["total"]}'
        pf(f'  {r["config"]:<26} {r["overall"]:<14} {r["full"]["sharpe"]:>7.3f} {r["oos"]["sharpe"]:>7.3f} '
           f'{r["full"]["n"]:>6} {kf_str:>5} {r["walk_forward"]["ratio"]:>6} '
           f'{r["monte_carlo"]["p_positive"]:>4.0f}% {r["era_positive"]:>3}/4 '
           f'{r["full"]["max_dd"]:>8.1f} {r["full"]["pnl"]:>10.0f}')

    # Detail: Cap impact
    pf(f'\n  Cap Impact (0.06 lot):')
    pf(f'  {"Config":<26} {"Sharpe":>7} {"CapHit":>7} {"CapHit%":>8} {"MaxDD":>8} {"dPnL vs NoCap":>14}')
    b_pnl = None
    for r in all_results:
        if 'error' in r:
            continue
        if r['config'].startswith('B_'):
            b_pnl = r['full']['pnl']
        if r['config'].startswith(('B_', 'C_', 'D_', 'E_')):
            cap_pct = round(r['cap_hits'] / max(r['full']['n'], 1) * 100, 1)
            dpnl = r['full']['pnl'] - b_pnl if b_pnl is not None else 0
            pf(f'  {r["config"]:<26} {r["full"]["sharpe"]:>7.3f} {r["cap_hits"]:>7} {cap_pct:>7.1f}% {r["full"]["max_dd"]:>8.1f} {dpnl:>+13.0f}')

    # Loss tail comparison
    pf(f'\n  Loss Tail (0.06 lot, no cap):')
    for r in all_results:
        if r.get('config', '').startswith('B_') and 'loss_tail' in r:
            for k, v in r['loss_tail'].items():
                pf(f'    {k}: {v}')

    # Sensitivity comparison
    pf(f'\n  Sensitivity (MC +-15% param perturbation):')
    for r in all_results:
        if 'error' in r:
            continue
        s = r.get('sensitivity', {})
        if s.get('verdict') != 'SKIP':
            pf(f'  {r["config"]:<26} {s["verdict"]}: mean={s["mean"]}, min={s["min"]}, std={s["std"]}')

    # Recommendation
    pf(f'\n\n{"="*80}')
    pf('RECOMMENDATION')
    pf(f'{"="*80}')
    strong = [r for r in all_results if r.get('overall') == 'STRONG_PASS' and r['config'].startswith(('B_', 'C_', 'D_', 'E_'))]
    regular = [r for r in all_results if r.get('overall') == 'PASS' and r['config'].startswith(('B_', 'C_', 'D_', 'E_'))]
    if strong:
        best = max(strong, key=lambda r: r['oos']['sharpe'])
        pf(f'  STRONG_PASS: {best["config"]} (OOS Sharpe={best["oos"]["sharpe"]}, MaxDD=${best["full"]["max_dd"]})')
    elif regular:
        best = max(regular, key=lambda r: r['oos']['sharpe'])
        pf(f'  PASS (not strong): {best["config"]} (OOS Sharpe={best["oos"]["sharpe"]})')
    else:
        pf(f'  NO PASS: 0.06 lot does not pass full validation pipeline. Stay at 0.04.')

    total = time.time() - t0
    pf(f'\n  Total runtime: {total:.0f}s ({total/60:.1f}min)')

    def serialize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(OUTPUT_DIR / 'r239c_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=serialize)
    pf(f'  Results saved: {OUTPUT_DIR / "r239c_results.json"}')

    import shutil
    shutil.rmtree(cache_dir, ignore_errors=True)
    pf('\nDone.')


if __name__ == '__main__':
    main()
