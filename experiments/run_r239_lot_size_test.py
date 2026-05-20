#!/usr/bin/env python3
"""R239: Keltner Lot Size 0.04 -> 0.06 Impact Test
====================================================
Compare Keltner performance at different lot sizes using LIVE_PARITY_KWARGS.

Configs:
  A) 0.04 lot baseline (current live)
  B) 0.06 lot (proposed)
  C) 0.06 lot + MaxLoss Cap $40
  D) 0.06 lot + MaxLoss Cap $60
  E) 0.06 lot + MaxLoss Cap $80
  F) 0.06 lot + realistic costs (spread + slippage)
  G) 0.04 lot + realistic costs (for comparison)

All use BacktestEngine + LIVE_PARITY_KWARGS, Keltner-only.
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

OUTPUT_DIR = Path("results/r239_lot_size_test")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HOLDOUT_START = "2025-05-01"

ERA_SEGMENTS = {
    "Pre-COVID (2015-2019)":      ("2015-01-01", "2020-01-01"),
    "COVID+Recovery (2020-2021)":  ("2020-01-01", "2022-01-01"),
    "Tightening (2022-2023)":      ("2022-01-01", "2024-01-01"),
    "Recent (2024-2026)":          ("2024-01-01", "2026-06-01"),
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
        return {'verdict': 'SKIP', 'pass_count': 0, 'total': 0}
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


def run_config(args):
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
        oos_sharpe, oos_pnl, oos_wr, _, oos_n, _, oos_dd = calc_daily_sharpe(oos_trades)

        eras = {}
        for era_name, (start, end) in ERA_SEGMENTS.items():
            start_ts = pd.Timestamp(start, tz='UTC')
            end_ts = pd.Timestamp(end, tz='UTC')
            era_t = [t for t in kc_trades if start_ts <= pd.Timestamp(t.entry_time) < end_ts]
            s, p, w, pf2, en, _, dd = calc_daily_sharpe(era_t)
            eras[era_name] = {'n': en, 'sharpe': s, 'pnl': p, 'win_rate': w, 'max_dd': dd}

        exit_reasons = Counter(t.exit_reason for t in kc_trades)
        pnl_per_trade = round(pnl / n, 2) if n > 0 else 0
        avg_win = float(np.mean([t.pnl for t in kc_trades if t.pnl > 0])) if any(t.pnl > 0 for t in kc_trades) else 0
        avg_loss = float(np.mean([t.pnl for t in kc_trades if t.pnl <= 0])) if any(t.pnl <= 0 for t in kc_trades) else 0

        return {
            'config': config_name,
            'overrides': {k: v for k, v in overrides.items() if not callable(v)},
            'n_trades': n,
            'sharpe': sharpe,
            'pnl': pnl,
            'pnl_per_trade': pnl_per_trade,
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'win_rate': wr,
            'profit_factor': pf_val,
            'max_dd': max_dd,
            'kfold': kf,
            'oos_sharpe': oos_sharpe,
            'oos_n': oos_n,
            'oos_pnl': round(oos_pnl, 2) if oos_pnl else 0,
            'oos_dd': oos_dd,
            'eras': eras,
            'exit_reasons': dict(exit_reasons),
            'maxloss_cap_count': engine.maxloss_cap_count,
        }
    except Exception as e:
        return {'config': config_name, 'error': str(e), 'traceback': traceback.format_exc()}


def main():
    t0 = time.time()
    pf('=' * 80)
    pf('R239: KELTNER LOT SIZE 0.04 -> 0.06 TEST')
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

    n_workers = min(cpu_count(), 8)
    pf(f'  Workers: {n_workers}')

    configs = [
        # A: Current live baseline
        ('A_lot_0.04_baseline', {
            'min_lot_size': 0.04, 'max_lot_size': 0.04,
            'maxloss_cap': 0,
        }),
        # B: Proposed 0.06, no cap
        ('B_lot_0.06_no_cap', {
            'min_lot_size': 0.06, 'max_lot_size': 0.06,
            'maxloss_cap': 0,
        }),
        # C: 0.06 + cap $40
        ('C_lot_0.06_cap_40', {
            'min_lot_size': 0.06, 'max_lot_size': 0.06,
            'maxloss_cap': 40,
        }),
        # D: 0.06 + cap $60
        ('D_lot_0.06_cap_60', {
            'min_lot_size': 0.06, 'max_lot_size': 0.06,
            'maxloss_cap': 60,
        }),
        # E: 0.06 + cap $80
        ('E_lot_0.06_cap_80', {
            'min_lot_size': 0.06, 'max_lot_size': 0.06,
            'maxloss_cap': 80,
        }),
        # F: 0.06 + realistic costs
        ('F_lot_0.06_realistic', {
            'min_lot_size': 0.06, 'max_lot_size': 0.06,
            'maxloss_cap': 0,
            'spread_model': 'realistic',
            'slippage_model': 'empirical',
        }),
        # G: 0.04 + realistic costs (for fair comparison)
        ('G_lot_0.04_realistic', {
            'min_lot_size': 0.04, 'max_lot_size': 0.04,
            'maxloss_cap': 0,
            'spread_model': 'realistic',
            'slippage_model': 'empirical',
        }),
    ]

    tasks = [(name, overrides, m15_path, h1_path) for name, overrides in configs]

    pf(f'\n[2] Running {len(tasks)} configs in parallel...')
    t1 = time.time()
    with Pool(n_workers) as pool:
        results = pool.map(run_config, tasks)
    pf(f'  Complete in {time.time()-t1:.0f}s')

    # ════════════════════════════════════════════════════════════
    # RESULTS
    # ════════════════════════════════════════════════════════════
    pf(f'\n\n{"="*80}')
    pf('RESULTS')
    pf(f'{"="*80}')

    header = f'  {"Config":<28} {"Sharpe":>7} {"N":>6} {"WR%":>7} {"PF":>6} {"KF":>5} {"OOS_Sh":>7} {"MaxDD":>8} {"$/trade":>8} {"PnL":>10}'
    pf(header)
    pf(f'  {"-"*28} {"-"*7} {"-"*6} {"-"*7} {"-"*6} {"-"*5} {"-"*7} {"-"*8} {"-"*8} {"-"*10}')

    for r in results:
        if 'error' in r:
            pf(f'  {r["config"]:<28} ERROR: {r["error"][:50]}')
            continue
        kf_str = f'{r["kfold"]["pass_count"]}/{r["kfold"]["total"]}'
        pf(f'  {r["config"]:<28} {r["sharpe"]:>7.3f} {r["n_trades"]:>6} {r["win_rate"]*100:>6.1f}% {r["profit_factor"]:>5.2f} {kf_str:>5} {r["oos_sharpe"]:>7.3f} {r["max_dd"]:>8.1f} {r["pnl_per_trade"]:>8.2f} {r["pnl"]:>10.1f}')

    # Detailed comparison: A vs B
    pf(f'\n\n{"="*80}')
    pf('DETAILED: 0.04 vs 0.06 (no cap, zero spread)')
    pf(f'{"="*80}')
    a = next((r for r in results if r['config'].startswith('A_')), None)
    b = next((r for r in results if r['config'].startswith('B_')), None)
    if a and b and 'error' not in a and 'error' not in b:
        pf(f'\n  {"Metric":<25} {"0.04 lot":>12} {"0.06 lot":>12} {"Delta":>12} {"Delta%":>10}')
        pf(f'  {"-"*25} {"-"*12} {"-"*12} {"-"*12} {"-"*10}')

        metrics = [
            ('Sharpe', a['sharpe'], b['sharpe']),
            ('Trades', a['n_trades'], b['n_trades']),
            ('Win Rate', a['win_rate'], b['win_rate']),
            ('Profit Factor', a['profit_factor'], b['profit_factor']),
            ('Total PnL ($)', a['pnl'], b['pnl']),
            ('PnL/Trade ($)', a['pnl_per_trade'], b['pnl_per_trade']),
            ('Avg Win ($)', a['avg_win'], b['avg_win']),
            ('Avg Loss ($)', a['avg_loss'], b['avg_loss']),
            ('Max DD ($)', a['max_dd'], b['max_dd']),
            ('OOS Sharpe', a['oos_sharpe'], b['oos_sharpe']),
            ('OOS PnL ($)', a['oos_pnl'], b['oos_pnl']),
        ]
        for name, va, vb in metrics:
            delta = vb - va
            pct = (delta / abs(va) * 100) if va != 0 else 0
            pf(f'  {name:<25} {va:>12.2f} {vb:>12.2f} {delta:>+12.2f} {pct:>+9.1f}%')

        # Era comparison
        pf(f'\n  Era Stability:')
        pf(f'  {"Era":<30} {"0.04 Sharpe":>12} {"0.06 Sharpe":>12} {"0.04 PnL":>10} {"0.06 PnL":>10}')
        pf(f'  {"-"*30} {"-"*12} {"-"*12} {"-"*10} {"-"*10}')
        for era in ERA_SEGMENTS:
            ea = a['eras'].get(era, {})
            eb = b['eras'].get(era, {})
            pf(f'  {era:<30} {ea.get("sharpe",0):>12.3f} {eb.get("sharpe",0):>12.3f} {ea.get("pnl",0):>10.1f} {eb.get("pnl",0):>10.1f}')

        # Exit reason comparison
        pf(f'\n  Exit Reasons:')
        all_reasons = set(list(a['exit_reasons'].keys()) + list(b['exit_reasons'].keys()))
        pf(f'  {"Reason":<25} {"0.04 lot":>10} {"0.06 lot":>10}')
        for reason in sorted(all_reasons):
            pf(f'  {reason:<25} {a["exit_reasons"].get(reason,0):>10} {b["exit_reasons"].get(reason,0):>10}')

    # Detailed: realistic cost comparison
    pf(f'\n\n{"="*80}')
    pf('DETAILED: Realistic Costs (0.04 vs 0.06)')
    pf(f'{"="*80}')
    f_res = next((r for r in results if r['config'].startswith('F_')), None)
    g_res = next((r for r in results if r['config'].startswith('G_')), None)
    if f_res and g_res and 'error' not in f_res and 'error' not in g_res:
        pf(f'\n  {"Metric":<25} {"0.04+cost":>12} {"0.06+cost":>12} {"Delta":>12}')
        pf(f'  {"-"*25} {"-"*12} {"-"*12} {"-"*12}')
        for name, va, vb in [
            ('Sharpe', g_res['sharpe'], f_res['sharpe']),
            ('Trades', g_res['n_trades'], f_res['n_trades']),
            ('PnL ($)', g_res['pnl'], f_res['pnl']),
            ('PnL/Trade ($)', g_res['pnl_per_trade'], f_res['pnl_per_trade']),
            ('Max DD ($)', g_res['max_dd'], f_res['max_dd']),
            ('OOS Sharpe', g_res['oos_sharpe'], f_res['oos_sharpe']),
        ]:
            delta = vb - va
            pf(f'  {name:<25} {va:>12.2f} {vb:>12.2f} {delta:>+12.2f}')

    # MaxLoss Cap analysis
    pf(f'\n\n{"="*80}')
    pf('MAXLOSS CAP ANALYSIS (0.06 lot)')
    pf(f'{"="*80}')
    cap_configs = [r for r in results if r['config'].startswith(('B_', 'C_', 'D_', 'E_'))]
    if cap_configs:
        pf(f'\n  {"Config":<28} {"Sharpe":>7} {"N":>6} {"MaxDD":>8} {"Cap Hits":>9} {"PnL":>10}')
        pf(f'  {"-"*28} {"-"*7} {"-"*6} {"-"*8} {"-"*9} {"-"*10}')
        for r in cap_configs:
            if 'error' not in r:
                pf(f'  {r["config"]:<28} {r["sharpe"]:>7.3f} {r["n_trades"]:>6} {r["max_dd"]:>8.1f} {r["maxloss_cap_count"]:>9} {r["pnl"]:>10.1f}')

    elapsed = time.time() - t0
    pf(f'\n\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)')

    # Save
    def serialize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(OUTPUT_DIR / 'r239_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=serialize)
    pf(f'\nResults saved: {OUTPUT_DIR / "r239_results.json"}')

    import shutil
    shutil.rmtree(cache_dir, ignore_errors=True)
    pf('\nDone.')


if __name__ == '__main__':
    main()
