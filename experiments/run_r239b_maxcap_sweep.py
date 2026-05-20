#!/usr/bin/env python3
"""R239b: 0.06 lot MaxLoss Cap fine sweep + loss tail analysis."""
from __future__ import annotations
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.runner import DataBundle, LIVE_PARITY_KWARGS
from backtest.engine import BacktestEngine

OUTPUT_DIR = Path("results/r239_lot_size_test")
HOLDOUT_START = "2025-05-01"


def pf(msg):
    print(msg, flush=True)


def calc_metrics(trades):
    if not trades or len(trades) < 5:
        return dict(sharpe=-999, n=0, pnl=0, max_dd=0, wr=0)
    pnls = np.array([t.pnl for t in trades])
    n = len(pnls)
    daily = {}
    for t in trades:
        d = pd.Timestamp(t.exit_time).date()
        daily[d] = daily.get(d, 0.0) + t.pnl
    ds = np.array(list(daily.values()))
    sh = float(ds.mean() / max(ds.std(ddof=1), 1e-9) * np.sqrt(252)) if len(ds) >= 10 else -999
    cum = np.cumsum(pnls)
    max_dd = float((np.maximum.accumulate(cum) - cum).max())
    return dict(
        sharpe=round(sh, 3),
        n=n,
        pnl=round(float(pnls.sum()), 2),
        max_dd=round(max_dd, 2),
        wr=round(len(pnls[pnls > 0]) / n, 4),
    )


def loss_tail(trades):
    losses = sorted([t.pnl for t in trades if t.pnl < 0])
    if not losses:
        return {}
    arr = np.array(losses)
    pcts = [50, 75, 90, 95, 99]
    out = {f'p{p}': round(float(np.percentile(arr, p)), 2) for p in pcts}
    out['worst'] = round(float(arr.min()), 2)
    out['n_loss'] = len(losses)
    return out


def run_one(args):
    cap, m15_path, h1_path = args
    try:
        m15_df = pd.read_pickle(m15_path)
        h1_df = pd.read_pickle(h1_path)
        kw = {**LIVE_PARITY_KWARGS,
              'min_lot_size': 0.06, 'max_lot_size': 0.06,
              'maxloss_cap': cap, 'label': f'cap_{cap}'}
        eng = BacktestEngine(m15_df, h1_df, **kw)
        trades = eng.run()
        kc = [t for t in trades if t.strategy == 'keltner']
        m = calc_metrics(kc)
        holdout = pd.Timestamp(HOLDOUT_START, tz='UTC')
        oos = [t for t in kc if pd.Timestamp(t.entry_time) >= holdout]
        oos_m = calc_metrics(oos)
        tail = loss_tail(kc) if cap == 0 else {}
        return {
            'cap': cap,
            **m,
            'oos_sharpe': oos_m['sharpe'],
            'cap_hits': eng.maxloss_cap_count,
            'cap_hit_pct': round(eng.maxloss_cap_count / max(m['n'], 1) * 100, 2),
            'tail': tail,
        }
    except Exception as e:
        return {'cap': cap, 'error': str(e)}


def main():
    t0 = time.time()
    pf('=' * 72)
    pf('R239b: 0.06 lot MaxLoss Cap sweep')
    pf('=' * 72)

    data = DataBundle.load_default()
    cache = OUTPUT_DIR / '_cache'
    cache.mkdir(exist_ok=True)
    m15p, h1p = str(cache / 'm15.pkl'), str(cache / 'h1.pkl')
    data.m15_df.to_pickle(m15p)
    data.h1_df.to_pickle(h1p)

    caps = [0, 50, 60, 70, 80, 90, 100, 120, 150]
    tasks = [(c, m15p, h1p) for c in caps]
    nw = min(cpu_count(), 8)
    with Pool(nw) as pool:
        results = pool.map(run_one, tasks)

    pf('\n  Cap ($)   Sharpe    N   CapHit%   MaxDD      PnL   OOS_Sh')
    pf('  ' + '-' * 68)
    for r in sorted(results, key=lambda x: x.get('cap', 0)):
        if 'error' in r:
            pf(f'  {r["cap"]:>6}   ERROR {r["error"][:40]}')
            continue
        pf(f'  {r["cap"]:>6}   {r["sharpe"]:>6.3f} {r["n"]:>5} {r["cap_hit_pct"]:>7.1f}% {r["max_dd"]:>8.1f} {r["pnl"]:>9.0f} {r["oos_sharpe"]:>7.3f}')

    base = next(r for r in results if r.get('cap') == 0 and 'error' not in r)
    if base.get('tail'):
        pf('\n  Loss tail (0.06 lot, NO cap):')
        for k, v in base['tail'].items():
            pf(f'    {k}: {v}')

    pf('\n  Cap vs baseline (cap=0):')
    pf(f'  {"Cap":>6} {"dSharpe":>8} {"dMaxDD":>8} {"dPnL":>10} {"Hits":>6}')
    for r in sorted(results, key=lambda x: x.get('cap', 0)):
        if r.get('cap', 0) == 0 or 'error' in r:
            continue
        pf(f'  {r["cap"]:>6} {r["sharpe"]-base["sharpe"]:>+8.3f} {r["max_dd"]-base["max_dd"]:>+8.1f} {r["pnl"]-base["pnl"]:>+10.0f} {r["cap_hits"]:>6}')

    # Proportional caps from EA $80 @ 0.03 lot
    pf('\n  Reference: EA doc MaxLoss $80 @ 0.03 lot -> scaled:')
    pf('    0.04 lot -> $106.7 | 0.06 lot -> $160.0')

    out = OUTPUT_DIR / 'r239b_maxcap_sweep.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    pf(f'\n  Saved: {out}')
    pf(f'  Runtime: {time.time()-t0:.0f}s')
    import shutil
    shutil.rmtree(cache, ignore_errors=True)


if __name__ == '__main__':
    main()
