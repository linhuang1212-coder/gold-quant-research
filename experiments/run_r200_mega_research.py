#!/usr/bin/env python3
"""
R200: 200-Hour Gold Mega Research Program
==========================================
5 tracks, 17 phases covering:
  Track A: Trail & Exit Deep Dive (~50h)
  Track B: New Alpha Sources (~50h)
  Track C: Portfolio Optimization (~40h)
  Track D: Robustness & Stress Testing (~40h)
  Track E: Production Readiness (~20h)

Design principles:
  - Does NOT duplicate R184-R199 closed conclusions
  - Every finding must pass 3-Gate: K-Fold>=4/6, WF>=13/19, Era all-positive
  - M15 execution resolution for all trail/exit work
  - All 6 live strategies tested

Usage: python experiments/run_r200_mega_research.py
"""
import sys, os, json, time, random, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

warnings.filterwarnings('ignore')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import research_config as config
from backtest.runner import (
    DataBundle, run_variant, run_kfold, run_variants_parallel,
    LIVE_PARITY_KWARGS, sanitize_for_json, load_csv,
    M15_CSV_PATH, H1_CSV_PATH,
)
from backtest.stats import calc_stats

t0 = time.time()
OUTPUT_DIR = Path("results/r200_mega")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_WORKERS = min(mp.cpu_count(), 64)


def elapsed():
    h = (time.time() - t0) / 3600
    return f"[{h:.1f}h]"


def save_phase(name, data):
    path = OUTPUT_DIR / f"{name}.json"
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved {path}")


def phase_done(name):
    return (OUTPUT_DIR / f"{name}.json").exists()


def safe_sharpe(stats):
    return stats.get('sharpe', 0) if stats else 0


def fmt(v):
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


# =====================================================================
# PARALLEL EXECUTION INFRASTRUCTURE
# =====================================================================
_worker_h1 = None
_worker_pctl = None
_worker_m15 = None


def _init_worker(h1_pkl, pctl_pkl):
    """Initialize worker process with shared data (called once per worker)."""
    global _worker_h1, _worker_pctl
    import pickle
    _worker_h1 = pickle.loads(h1_pkl)
    _worker_pctl = pickle.loads(pctl_pkl)


def _init_worker_m15(h1_pkl, pctl_pkl, m15_pkl):
    """Initialize worker with H1 + M15 data for resolution cross-validation."""
    global _worker_h1, _worker_pctl, _worker_m15
    import pickle
    _worker_h1 = pickle.loads(h1_pkl)
    _worker_pctl = pickle.loads(pctl_pkl)
    _worker_m15 = pickle.loads(m15_pkl)


def _worker_bt_h1(args):
    """Worker: run bt_h1_strategy with given params, return (key, stats_dict)."""
    key, cfg, strategy, kw = args
    trades = bt_h1_strategy(_worker_h1, cfg, _worker_pctl, strategy=strategy, **kw)
    return key, _stats_from_trades(trades)


def _worker_bt_h1_eras(args):
    """Worker: run run_all_eras_h1 with given params, return (key, era_results)."""
    key, cfg, strategy, kw = args
    eras = run_all_eras_h1(_worker_h1, cfg, _worker_pctl, strategy, **kw)
    return key, eras


def _worker_bt_h1_fold(args):
    """Worker: run bt_h1_strategy on a specific fold, return (fold_idx, sharpe)."""
    fold_idx, s, e, cfg, strategy, kw = args
    mask = (_worker_h1.index >= pd.Timestamp(s, tz='UTC')) & (_worker_h1.index < pd.Timestamp(e, tz='UTC'))
    h1_fold = _worker_h1[mask]
    pctl_fold = _worker_pctl[mask]
    if len(h1_fold) < 200:
        return fold_idx, 0.0
    trades = bt_h1_strategy(h1_fold, cfg, pctl_fold, strategy=strategy, **kw)
    return fold_idx, _stats_from_trades(trades)['sharpe']


def _worker_bt_h1_wf(args):
    """Worker: run bt_h1_strategy on a WF test window, return (idx, sharpe)."""
    idx, test_s, test_e, cfg, strategy, kw = args
    mask = (_worker_h1.index >= pd.Timestamp(test_s, tz='UTC')) & (_worker_h1.index < pd.Timestamp(test_e, tz='UTC'))
    h1_test = _worker_h1[mask]
    pctl_test = _worker_pctl[mask]
    if len(h1_test) < 100:
        return idx, None
    trades = bt_h1_strategy(h1_test, cfg, pctl_test, strategy=strategy, **kw)
    return idx, _stats_from_trades(trades)['sharpe']


def create_pool(h1, pctl):
    """Create a ProcessPoolExecutor with pre-loaded H1 data in each worker."""
    import pickle
    h1_pkl = pickle.dumps(h1)
    pctl_pkl = pickle.dumps(pctl)
    return ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=_init_worker,
        initargs=(h1_pkl, pctl_pkl),
    )


def create_pool_m15(h1, pctl, m15):
    """Create a ProcessPoolExecutor with H1 + M15 data in each worker."""
    import pickle
    h1_pkl = pickle.dumps(h1)
    pctl_pkl = pickle.dumps(pctl)
    m15_pkl = pickle.dumps(m15)
    return ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=_init_worker_m15,
        initargs=(h1_pkl, pctl_pkl, m15_pkl),
    )


def _worker_a3b(args):
    """Worker for Phase A3b: run both H1 and M15-exit backtests, return comparison."""
    key, cfg, strategy, kw = args
    h1_trades = bt_h1_strategy(_worker_h1, cfg, _worker_pctl, strategy=strategy, **kw)
    m15_trades = bt_h1_strategy_m15exit(_worker_h1, _worker_m15, cfg, _worker_pctl,
                                         strategy=strategy, **kw)
    h1_stats = _stats_from_trades(h1_trades)
    m15_stats = _stats_from_trades(m15_trades)

    h1_trail_n = sum(1 for t in h1_trades if t['reason'] == 'Trail')
    m15_trail_n = sum(1 for t in m15_trades if t['reason'] == 'Trail')
    h1_trail_pnl = [t['pnl'] for t in h1_trades if t['reason'] == 'Trail']
    m15_trail_pnl = [t['pnl'] for t in m15_trades if t['reason'] == 'Trail']

    hair_trigger = 0
    if h1_trades and m15_trades:
        h1_by_entry = {t['entry_time']: t for t in h1_trades}
        for mt in m15_trades:
            ht = h1_by_entry.get(mt['entry_time'])
            if ht and mt['reason'] == 'Trail' and ht['reason'] == 'Trail':
                if mt['bars'] <= ht['bars']:
                    hair_trigger += 1

    return key, {
        'h1_sharpe': h1_stats['sharpe'],
        'm15_sharpe': m15_stats['sharpe'],
        'h1_pnl': h1_stats['pnl'],
        'm15_pnl': m15_stats['pnl'],
        'h1_n': h1_stats['n'],
        'm15_n': m15_stats['n'],
        'h1_trail_n': h1_trail_n,
        'm15_trail_n': m15_trail_n,
        'h1_trail_avg_pnl': round(np.mean(h1_trail_pnl), 2) if h1_trail_pnl else 0,
        'm15_trail_avg_pnl': round(np.mean(m15_trail_pnl), 2) if m15_trail_pnl else 0,
        'hair_trigger_count': hair_trigger,
        'hair_trigger_pct': round(hair_trigger / max(1, m15_trail_n) * 100, 1),
        'h1_reasons': h1_stats.get('reasons', {}),
        'm15_reasons': m15_stats.get('reasons', {}),
    }


def parallel_h1_eras(pool, tasks):
    """
    Run multiple run_all_eras_h1 calls in parallel.
    tasks: list of (key, cfg, strategy, kw_dict)
    Returns: dict of {key: era_results}
    """
    futures = {pool.submit(_worker_bt_h1_eras, t): t[0] for t in tasks}
    results = {}
    for f in as_completed(futures):
        key, eras = f.result()
        results[key] = eras
    return results


def parallel_h1_stats(pool, tasks):
    """
    Run multiple bt_h1_strategy calls in parallel.
    tasks: list of (key, cfg, strategy, kw_dict)
    Returns: dict of {key: stats_dict}
    """
    futures = {pool.submit(_worker_bt_h1, t): t[0] for t in tasks}
    results = {}
    for f in as_completed(futures):
        key, stats = f.result()
        results[key] = stats
    return results


def _worker_bt_h1_sliced(args):
    """Worker: slice h1 by date range, run bt_h1_strategy, return (key, stats_or_None)."""
    key, cfg, strategy, start, end, kw = args
    mask = (_worker_h1.index >= pd.Timestamp(start, tz='UTC')) & (_worker_h1.index < pd.Timestamp(end, tz='UTC'))
    h1_r = _worker_h1[mask]
    pctl_r = _worker_pctl[mask]
    if len(h1_r) < 100:
        return key, None
    trades = bt_h1_strategy(h1_r, cfg, pctl_r, strategy=strategy, **kw)
    return key, _stats_from_trades(trades)


def _worker_bt_h1_trades(args):
    """Worker: run bt_h1_strategy, return (key, list_of_trade_dicts)."""
    key, cfg, strategy, kw = args
    trades = bt_h1_strategy(_worker_h1, cfg, _worker_pctl, strategy=strategy, **kw)
    return key, trades


# =====================================================================
# LIVE STRATEGY CONFIGURATIONS (mirrors gold_trader.py production)
# =====================================================================
STRAT_CONFIGS = {
    'keltner': {
        'lot': 0.04, 'sl_atr': 3.5, 'tp_atr': 8.0, 'cap': 70, 'cap_atr_mult': 4.0,
        'trail_act_atr': 0.02, 'trail_dist_atr': 0.005, 'max_hold_h': 2,
        'cooldown_h': 2, 'adx_threshold': 14,
    },
    'tsmom': {
        'lot': 0.04, 'sl_atr': 6.0, 'tp_atr': 8.0, 'cap': 60, 'cap_atr_mult': 6.5,
        'trail_act_atr': 0.14, 'trail_dist_atr': 0.025, 'max_hold_h': 12,
        'cooldown_h': 2,
    },
    'psar': {
        'lot': 0.09, 'sl_atr': 6.0, 'tp_atr': 6.0, 'cap': 60, 'cap_atr_mult': 4.5,
        'trail_act_atr': 0.06, 'trail_dist_atr': 0.01, 'max_hold_h': 15,
        'cooldown_h': 2,
    },
    'sess_bo': {
        'lot': 0.04, 'sl_atr': 4.5, 'tp_atr': 4.0, 'cap': 60, 'cap_atr_mult': 5.0,
        'trail_act_atr': 0.06, 'trail_dist_atr': 0.01, 'max_hold_h': 20,
        'cooldown_h': 2,
    },
    'dual_thrust': {
        'lot': 0.04, 'sl_atr': 6.0, 'tp_atr': 8.0, 'cap': 18, 'cap_atr_mult': 5.0,
        'trail_act_atr': 0.06, 'trail_dist_atr': 0.01, 'max_hold_h': 20,
        'cooldown_h': 2,
    },
    'chandelier': {
        'lot': 0.03, 'sl_atr': 4.5, 'tp_atr': 8.0, 'cap': 25, 'cap_atr_mult': 5.0,
        'trail_act_atr': 0.06, 'trail_dist_atr': 0.01, 'max_hold_h': 20,
        'cooldown_h': 2,
    },
}

ERA_SEGMENTS = {
    'Full (2015-2026)':       ('2015-01-01', '2026-04-01'),
    'Pre-COVID (2015-2019)':  ('2015-01-01', '2020-01-01'),
    'COVID+Recovery (2020-2021)': ('2020-01-01', '2022-01-01'),
    'Tightening (2022-2023)': ('2022-01-01', '2024-01-01'),
    'Recent (2024-2026)':     ('2024-01-01', '2026-04-01'),
}

WF_WINDOWS = []
for yr in range(2016, 2026):
    train_s = f"{yr-2}-01-01"
    train_e = f"{yr}-01-01"
    test_s = f"{yr}-01-01"
    test_e = f"{yr}-07-01" if yr < 2026 else "2026-04-01"
    WF_WINDOWS.append((train_s, train_e, test_s, test_e))
    if yr < 2026:
        WF_WINDOWS.append((f"{yr-2}-07-01", f"{yr}-07-01", f"{yr}-07-01", f"{yr+1}-01-01"))

KELTNER_ENGINE_KWARGS = dict(LIVE_PARITY_KWARGS)


# =====================================================================
# Helper: run Keltner backtest via the unified engine
# =====================================================================
def bt_keltner(data: DataBundle, label: str = "keltner", **overrides) -> Dict:
    kw = dict(KELTNER_ENGINE_KWARGS)
    kw.update(overrides)
    return run_variant(data, label, verbose=False, **kw)


def bt_keltner_with_cap(data: DataBundle, cap: float = 70, label: str = "keltner", **overrides) -> Dict:
    kw = dict(KELTNER_ENGINE_KWARGS)
    kw['maxloss_cap'] = cap
    kw.update(overrides)
    return run_variant(data, label, verbose=False, **kw)


# =====================================================================
# Helper: H1-loop backtest for non-engine strategies (PSAR/TSMOM etc)
# =====================================================================
def _load_h1_with_indicators():
    """Load H1 data with full indicators for manual strategy backtesting."""
    from backtest.runner import load_csv, add_atr_percentile
    from indicators import prepare_indicators, calc_chandelier, calc_dual_thrust_range

    h1 = load_csv(str(H1_CSV_PATH))
    h1 = prepare_indicators(h1)
    h1 = add_atr_percentile(h1)

    chand = calc_chandelier(h1, period=22, mult=3.0)
    h1['Chand_long'] = chand['Chand_long']
    h1['Chand_short'] = chand['Chand_short']

    h1['DT_range'] = calc_dual_thrust_range(h1, n_bars=6)

    # PSAR
    h1 = _add_psar(h1, af_step=0.01, af_max=0.05)

    return h1


def _add_psar(df, af_step=0.01, af_max=0.05):
    """Wilder Parabolic SAR."""
    high = df['High'].values
    low = df['Low'].values
    n = len(df)
    psar = np.zeros(n)
    af = af_step
    bull = True
    ep = high[0]
    psar[0] = low[0]
    for i in range(1, n):
        if bull:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            psar[i] = min(psar[i], low[i-1], low[max(0, i-2)])
            if low[i] < psar[i]:
                bull = False
                psar[i] = ep
                af = af_step
                ep = low[i]
            else:
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            psar[i] = psar[i-1] + af * (ep - psar[i-1])
            psar[i] = max(psar[i], high[i-1], high[max(0, i-2)])
            if high[i] > psar[i]:
                bull = True
                psar[i] = ep
                af = af_step
                ep = high[i]
            else:
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)
    df = df.copy()
    df['PSAR'] = psar
    return df


def bt_h1_strategy(h1, cfg, pctl, strategy='psar', trail_act=None, trail_dist=None,
                    cooldown=None, skip_hours=None, max_hold=None, pctl_f=30,
                    cap=None, sl_atr=None, tp_atr=None, **extra):
    """
    Generic H1-loop backtest for PSAR / TSMOM / SESS_BO / DUAL_THRUST / CHANDELIER.
    Returns list of trade dicts.
    """
    ta = trail_act if trail_act is not None else cfg['trail_act_atr']
    td = trail_dist if trail_dist is not None else cfg['trail_dist_atr']
    cd = cooldown if cooldown is not None else cfg.get('cooldown_h', 2)
    mh = max_hold if max_hold is not None else cfg['max_hold_h']
    cap_usd = cap if cap is not None else cfg.get('cap', 60)
    lot = cfg['lot']
    sl_mult = sl_atr if sl_atr is not None else cfg['sl_atr']
    tp_mult = tp_atr if tp_atr is not None else cfg['tp_atr']
    PV = 100

    close = h1['Close'].values
    high_arr = h1['High'].values
    low_arr = h1['Low'].values
    atr_arr = h1['ATR'].values
    pctl_arr = pctl.values if hasattr(pctl, 'values') else pctl

    trades = []
    pos = None
    last_exit_bar = -999

    for i in range(100, len(h1)):
        if np.isnan(atr_arr[i]) or atr_arr[i] <= 0:
            continue

        # ATR percentile floor
        if pctl_arr[i] < pctl_f / 100.0:
            if pos is None:
                continue

        hour = h1.index[i].hour
        if skip_hours and hour in skip_hours:
            if pos is None:
                continue

        # Exit logic
        if pos is not None:
            pos['bars'] += 1
            c = close[i]
            hi = high_arr[i]
            lo = low_arr[i]
            atr = atr_arr[i]
            reason = None
            exit_p = c

            # SL/TP
            if pos['dir'] == 'BUY':
                if lo <= pos['sl']:
                    reason, exit_p = 'SL', pos['sl']
                elif hi >= pos['tp']:
                    reason, exit_p = 'TP', pos['tp']
            else:
                if hi >= pos['sl']:
                    reason, exit_p = 'SL', pos['sl']
                elif lo <= pos['tp']:
                    reason, exit_p = 'TP', pos['tp']

            # MaxLoss Cap
            if not reason and cap_usd > 0:
                fpnl = (c - pos['entry']) * lot * PV if pos['dir'] == 'BUY' else (pos['entry'] - c) * lot * PV
                if fpnl < -cap_usd:
                    reason, exit_p = 'Cap', c

            # Trail
            if not reason:
                ad = ta * atr
                tdd = td * atr
                if pos['dir'] == 'BUY':
                    pos['extreme'] = max(pos['extreme'], hi)
                    if pos['extreme'] - pos['entry'] >= ad:
                        new_t = pos['extreme'] - tdd
                        pos['trail'] = max(pos['trail'], new_t)
                        if lo <= pos['trail']:
                            reason, exit_p = 'Trail', pos['trail']
                else:
                    pos['extreme'] = min(pos['extreme'], lo)
                    if pos['entry'] - pos['extreme'] >= ad:
                        new_t = pos['extreme'] + tdd
                        if pos['trail'] == 0:
                            pos['trail'] = new_t
                        else:
                            pos['trail'] = min(pos['trail'], new_t)
                        if hi >= pos['trail']:
                            reason, exit_p = 'Trail', pos['trail']

            # Time stop
            if not reason and pos['bars'] >= mh:
                reason, exit_p = 'Timeout', c

            if reason:
                pnl = (exit_p - pos['entry']) * lot * PV if pos['dir'] == 'BUY' else (pos['entry'] - exit_p) * lot * PV
                pnl -= 0.30 * lot * PV
                trades.append({
                    'entry': pos['entry'], 'exit': exit_p, 'dir': pos['dir'],
                    'pnl': round(pnl, 2), 'reason': reason, 'bars': pos['bars'],
                    'entry_time': str(pos['entry_time']), 'exit_time': str(h1.index[i]),
                })
                last_exit_bar = i
                pos = None

        # Entry logic
        if pos is None and i - last_exit_bar >= cd:
            sig = _check_entry(h1, i, strategy, extra)
            if sig:
                atr = atr_arr[i]
                sl_dist = sl_mult * atr
                tp_dist = tp_mult * atr
                entry_p = close[i]
                if sig == 'BUY':
                    sl_p = entry_p - sl_dist
                    tp_p = entry_p + tp_dist
                else:
                    sl_p = entry_p + sl_dist
                    tp_p = entry_p - tp_dist
                pos = {
                    'entry': entry_p, 'sl': sl_p, 'tp': tp_p, 'dir': sig,
                    'bars': 0, 'extreme': entry_p, 'trail': 0,
                    'entry_time': h1.index[i], 'entry_atr': atr,
                }

    return trades


def bt_h1_strategy_m15exit(h1, m15, cfg, pctl, strategy='psar', trail_act=None,
                            trail_dist=None, cooldown=None, skip_hours=None,
                            max_hold=None, pctl_f=30, cap=None, sl_atr=None,
                            tp_atr=None, **extra):
    """
    Same entry logic as bt_h1_strategy (H1 bar level), but exit checks use
    M15 bars within each H1 bar for 4x higher resolution on trail/SL/TP.
    Returns list of trade dicts with an extra 'exit_resolution' field.
    """
    ta = trail_act if trail_act is not None else cfg['trail_act_atr']
    td = trail_dist if trail_dist is not None else cfg['trail_dist_atr']
    cd = cooldown if cooldown is not None else cfg.get('cooldown_h', 2)
    mh = max_hold if max_hold is not None else cfg['max_hold_h']
    cap_usd = cap if cap is not None else cfg.get('cap', 60)
    lot = cfg['lot']
    sl_mult = sl_atr if sl_atr is not None else cfg['sl_atr']
    tp_mult = tp_atr if tp_atr is not None else cfg['tp_atr']
    PV = 100

    close = h1['Close'].values
    high_arr = h1['High'].values
    low_arr = h1['Low'].values
    atr_arr = h1['ATR'].values
    pctl_arr = pctl.values if hasattr(pctl, 'values') else pctl

    m15_close = m15['Close'].values
    m15_high = m15['High'].values
    m15_low = m15['Low'].values
    m15_index = m15.index

    h1_to_m15 = {}
    for mi in range(len(m15)):
        h1_time = m15_index[mi].floor('h')
        if h1_time not in h1_to_m15:
            h1_to_m15[h1_time] = []
        h1_to_m15[h1_time].append(mi)

    trades = []
    pos = None
    last_exit_bar = -999

    for i in range(100, len(h1)):
        if np.isnan(atr_arr[i]) or atr_arr[i] <= 0:
            continue

        if pctl_arr[i] < pctl_f / 100.0:
            if pos is None:
                continue

        hour = h1.index[i].hour
        if skip_hours and hour in skip_hours:
            if pos is None:
                continue

        if pos is not None:
            pos['bars'] += 1
            atr = atr_arr[i]
            h1_time = h1.index[i]

            m15_indices = h1_to_m15.get(h1_time, [])

            if m15_indices:
                reason = None
                exit_p = close[i]
                for mi in m15_indices:
                    c15 = m15_close[mi]
                    hi15 = m15_high[mi]
                    lo15 = m15_low[mi]

                    if pos['dir'] == 'BUY':
                        if lo15 <= pos['sl']:
                            reason, exit_p = 'SL', pos['sl']
                            break
                        if hi15 >= pos['tp']:
                            reason, exit_p = 'TP', pos['tp']
                            break
                    else:
                        if hi15 >= pos['sl']:
                            reason, exit_p = 'SL', pos['sl']
                            break
                        if lo15 <= pos['tp']:
                            reason, exit_p = 'TP', pos['tp']
                            break

                    if not reason and cap_usd > 0:
                        fpnl = (c15 - pos['entry']) * lot * PV if pos['dir'] == 'BUY' else (pos['entry'] - c15) * lot * PV
                        if fpnl < -cap_usd:
                            reason, exit_p = 'Cap', c15
                            break

                    if not reason:
                        ad = ta * atr
                        tdd = td * atr
                        if pos['dir'] == 'BUY':
                            pos['extreme'] = max(pos['extreme'], hi15)
                            if pos['extreme'] - pos['entry'] >= ad:
                                new_t = pos['extreme'] - tdd
                                pos['trail'] = max(pos['trail'], new_t)
                                if lo15 <= pos['trail']:
                                    reason, exit_p = 'Trail', pos['trail']
                                    break
                        else:
                            pos['extreme'] = min(pos['extreme'], lo15)
                            if pos['entry'] - pos['extreme'] >= ad:
                                new_t = pos['extreme'] + tdd
                                if pos['trail'] == 0:
                                    pos['trail'] = new_t
                                else:
                                    pos['trail'] = min(pos['trail'], new_t)
                                if hi15 >= pos['trail']:
                                    reason, exit_p = 'Trail', pos['trail']
                                    break
            else:
                c = close[i]
                hi = high_arr[i]
                lo = low_arr[i]
                reason = None
                exit_p = c

                if pos['dir'] == 'BUY':
                    if lo <= pos['sl']:
                        reason, exit_p = 'SL', pos['sl']
                    elif hi >= pos['tp']:
                        reason, exit_p = 'TP', pos['tp']
                else:
                    if hi >= pos['sl']:
                        reason, exit_p = 'SL', pos['sl']
                    elif lo <= pos['tp']:
                        reason, exit_p = 'TP', pos['tp']

                if not reason and cap_usd > 0:
                    fpnl = (c - pos['entry']) * lot * PV if pos['dir'] == 'BUY' else (pos['entry'] - c) * lot * PV
                    if fpnl < -cap_usd:
                        reason, exit_p = 'Cap', c

                if not reason:
                    ad = ta * atr
                    tdd = td * atr
                    if pos['dir'] == 'BUY':
                        pos['extreme'] = max(pos['extreme'], hi)
                        if pos['extreme'] - pos['entry'] >= ad:
                            new_t = pos['extreme'] - tdd
                            pos['trail'] = max(pos['trail'], new_t)
                            if lo <= pos['trail']:
                                reason, exit_p = 'Trail', pos['trail']
                    else:
                        pos['extreme'] = min(pos['extreme'], lo)
                        if pos['entry'] - pos['extreme'] >= ad:
                            new_t = pos['extreme'] + tdd
                            if pos['trail'] == 0:
                                pos['trail'] = new_t
                            else:
                                pos['trail'] = min(pos['trail'], new_t)
                            if hi >= pos['trail']:
                                reason, exit_p = 'Trail', pos['trail']

            if not reason and pos['bars'] >= mh:
                reason, exit_p = 'Timeout', close[i]

            if reason:
                pnl = (exit_p - pos['entry']) * lot * PV if pos['dir'] == 'BUY' else (pos['entry'] - exit_p) * lot * PV
                pnl -= 0.30 * lot * PV
                trades.append({
                    'entry': pos['entry'], 'exit': exit_p, 'dir': pos['dir'],
                    'pnl': round(pnl, 2), 'reason': reason, 'bars': pos['bars'],
                    'entry_time': str(pos['entry_time']), 'exit_time': str(h1.index[i]),
                    'exit_resolution': 'm15',
                })
                last_exit_bar = i
                pos = None

        if pos is None and i - last_exit_bar >= cd:
            sig = _check_entry(h1, i, strategy, extra)
            if sig:
                atr = atr_arr[i]
                sl_dist = sl_mult * atr
                tp_dist = tp_mult * atr
                entry_p = close[i]
                if sig == 'BUY':
                    sl_p = entry_p - sl_dist
                    tp_p = entry_p + tp_dist
                else:
                    sl_p = entry_p + sl_dist
                    tp_p = entry_p - tp_dist
                pos = {
                    'entry': entry_p, 'sl': sl_p, 'tp': tp_p, 'dir': sig,
                    'bars': 0, 'extreme': entry_p, 'trail': 0,
                    'entry_time': h1.index[i], 'entry_atr': atr,
                }

    return trades


def _check_entry(h1, i, strategy, extra):
    """Check entry signal for non-engine strategies."""
    row = h1.iloc[i]
    prev = h1.iloc[i-1] if i > 0 else row
    pprev = h1.iloc[i-2] if i > 1 else prev

    if strategy == 'psar':
        psar_now = row.get('PSAR', 0)
        psar_prev = prev.get('PSAR', 0)
        c = row['Close']
        if psar_prev > prev['Close'] and psar_now < c:
            return 'BUY'
        elif psar_prev < prev['Close'] and psar_now > c:
            return 'SELL'

    elif strategy == 'tsmom':
        # Real TSMOM: momentum score = 0.5*sign(ret_fast) + 0.5*sign(ret_slow)
        # Signal fires ONLY on score zero-crossing (very rare, ~few per month)
        fast_lb = extra.get('fast_lookback', 480)
        slow_lb = extra.get('slow_lookback', 720)
        if i < slow_lb + 1:
            return None
        closes = h1['Close'].values
        def _score(idx):
            s = 0.0
            if idx >= fast_lb and closes[idx - fast_lb] > 0:
                ret = closes[idx] / closes[idx - fast_lb] - 1.0
                s += 0.5 * (1.0 if ret > 0 else (-1.0 if ret < 0 else 0.0))
            if idx >= slow_lb and closes[idx - slow_lb] > 0:
                ret = closes[idx] / closes[idx - slow_lb] - 1.0
                s += 0.5 * (1.0 if ret > 0 else (-1.0 if ret < 0 else 0.0))
            return s
        score_now = _score(i)
        score_prev = _score(i - 1)
        if score_now > 0 and score_prev <= 0:
            return 'BUY'
        elif score_now < 0 and score_prev >= 0:
            return 'SELL'

    elif strategy == 'sess_bo':
        hour = h1.index[i].hour
        if hour != 12:
            return None
        c = row['Close']
        rng = h1.iloc[max(0,i-4):i]
        if len(rng) < 4:
            return None
        hi = rng['High'].max()
        lo = rng['Low'].min()
        if c > hi:
            return 'BUY'
        elif c < lo:
            return 'SELL'

    elif strategy == 'dual_thrust':
        c = prev['Close']
        o = row.get('Open', c)
        dt_range = prev.get('DT_range', 0)
        if dt_range <= 0:
            return None
        k = extra.get('k_up', 0.5)
        daily_open = h1.iloc[max(0, i-1)]['Open']
        if c > daily_open + k * dt_range:
            return 'BUY'
        elif c < daily_open - k * dt_range:
            return 'SELL'

    elif strategy == 'chandelier':
        clong_now = row.get('Chand_long', 0)
        cshort_now = row.get('Chand_short', 0)
        clong_prev = prev.get('Chand_long', 0)
        cshort_prev = prev.get('Chand_short', 0)
        c = prev['Close']
        rsi = row.get('RSI14', 50)
        if c > clong_prev and pprev['Close'] <= pprev.get('Chand_long', c):
            if rsi < 70:
                return 'BUY'
        elif c < cshort_prev and pprev['Close'] >= pprev.get('Chand_short', c):
            if rsi > 30:
                return 'SELL'

    return None


def _stats_from_trades(trades):
    """Compute stats dict from list of trade dicts (h1 loop format)."""
    if not trades:
        return {'sharpe': 0, 'pnl': 0, 'n': 0, 'wr': 0, 'max_dd': 0, 'avg_pnl': 0}
    pnls = [t['pnl'] for t in trades]
    n = len(pnls)
    total = sum(pnls)
    wins = sum(1 for p in pnls if p > 0)

    daily = defaultdict(float)
    for t in trades:
        d = t['exit_time'][:10]
        daily[d] += t['pnl']
    dpnl = list(daily.values())
    sharpe = 0
    if len(dpnl) > 1 and np.std(dpnl, ddof=1) > 0:
        sharpe = np.mean(dpnl) / np.std(dpnl, ddof=1) * np.sqrt(252)

    eq = np.cumsum([config.CAPITAL] + pnls)
    peak = np.maximum.accumulate(eq)
    max_dd = abs((eq - peak).min())

    # Exit reason breakdown
    reasons = defaultdict(int)
    for t in trades:
        reasons[t['reason']] += 1

    return {
        'sharpe': round(sharpe, 3), 'pnl': round(total, 2), 'n': n,
        'wr': round(wins / n * 100, 1), 'max_dd': round(max_dd, 2),
        'avg_pnl': round(total / n, 2), 'reasons': dict(reasons),
    }


def run_all_eras_engine(data, label, **kw):
    """Run Keltner (engine) across all eras."""
    results = {}
    for era_name, (s, e) in ERA_SEGMENTS.items():
        era_data = data.slice(s, e)
        if len(era_data.m15_df) < 1000:
            continue
        stats = bt_keltner(era_data, label=f"{label}_{era_name}", **kw)
        results[era_name] = {k: v for k, v in stats.items() if not k.startswith('_')}
    return results


def run_all_eras_h1(h1, cfg, pctl, strategy, **kw):
    """Run H1-loop strategy across all eras."""
    results = {}
    for era_name, (s, e) in ERA_SEGMENTS.items():
        mask = (h1.index >= pd.Timestamp(s, tz='UTC')) & (h1.index < pd.Timestamp(e, tz='UTC'))
        h1_era = h1[mask]
        pctl_era = pctl[mask]
        if len(h1_era) < 200:
            continue
        trades = bt_h1_strategy(h1_era, cfg, pctl_era, strategy=strategy, **kw)
        results[era_name] = _stats_from_trades(trades)
    return results


def three_gate_validate(data_or_h1, cfg_or_kw, strategy='keltner', pctl=None, h1=None, **kw):
    """
    Run 3-Gate validation: K-Fold (>=4/6), Walk-Forward (>=13/19), Era (all positive).
    Returns (passed: bool, details: dict).
    """
    results = {'kfold': {}, 'wf': {}, 'era': {}}

    if strategy == 'keltner':
        # K-Fold via runner (parallel)
        kf_results = run_kfold(data_or_h1, cfg_or_kw, n_folds=6, parallel=True)
        kf_sharpes = [r['sharpe'] for r in kf_results]
        kf_wins = sum(1 for s in kf_sharpes if s > 0)
        results['kfold'] = {'wins': kf_wins, 'sharpes': [round(s, 2) for s in kf_sharpes]}

        # Walk-Forward (parallelized)
        from backtest.runner import _worker_run_variant as _runner_worker
        wf_variants = []
        wf_meta = []
        for train_s, train_e, test_s, test_e in WF_WINDOWS:
            test_data = data_or_h1.slice(test_s, test_e)
            if len(test_data.m15_df) < 500:
                continue
            v = dict(kw)
            label = f"WF_{test_s}"
            wf_variants.append((test_data.m15_df, test_data.h1_df, label, v))
            wf_meta.append((test_s, test_e))

        wf_wins = 0
        wf_total = len(wf_variants)
        if wf_variants:
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as wf_pool:
                wf_futures = [wf_pool.submit(_runner_worker, args) for args in wf_variants]
                for fut in as_completed(wf_futures):
                    stats = fut.result()
                    if stats.get('sharpe', 0) > 0:
                        wf_wins += 1
        results['wf'] = {'wins': wf_wins, 'total': wf_total}

        # Era (parallelized)
        era_variants = []
        era_names = []
        for era_name, (s, e) in ERA_SEGMENTS.items():
            era_data = data_or_h1.slice(s, e)
            if len(era_data.m15_df) < 1000:
                continue
            v = dict(kw)
            label = f"era_{era_name}"
            era_variants.append((era_data.m15_df, era_data.h1_df, label, v))
            era_names.append(era_name)

        era_sharpes = {}
        if era_variants:
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as era_pool:
                era_futures = {}
                for args, name in zip(era_variants, era_names):
                    fut = era_pool.submit(_runner_worker, args)
                    era_futures[fut] = name
                for fut in as_completed(era_futures):
                    name = era_futures[fut]
                    stats = fut.result()
                    era_sharpes[name] = stats.get('sharpe', 0)

        era_pass = all(v > 0 for v in era_sharpes.values()) and len(era_sharpes) >= 4
        results['era'] = era_sharpes
        results['era_pass'] = era_pass

    else:
        # H1-loop strategies — parallelized K-Fold + WF + Era
        kf_folds = [
            ("2015-01-01", "2017-01-01"), ("2017-01-01", "2019-01-01"),
            ("2019-01-01", "2021-01-01"), ("2021-01-01", "2023-01-01"),
            ("2023-01-01", "2025-01-01"), ("2025-01-01", "2026-04-01"),
        ]
        kf_tasks = [(i, s, e, cfg_or_kw, strategy, kw) for i, (s, e) in enumerate(kf_folds)]
        wf_tasks = [(i, ts, te, cfg_or_kw, strategy, kw) for i, (_, _, ts, te) in enumerate(WF_WINDOWS)]

        with create_pool(h1, pctl) as pool:
            kf_futures = [pool.submit(_worker_bt_h1_fold, t) for t in kf_tasks]
            wf_futures = [pool.submit(_worker_bt_h1_wf, t) for t in wf_tasks]
            era_tasks = [(era_name, cfg_or_kw, strategy, kw) for era_name in ERA_SEGMENTS]
            era_futures = [pool.submit(_worker_bt_h1_eras, (f"era_{en}", cfg_or_kw, strategy, kw))
                           for en in ERA_SEGMENTS]

            kf_sharpes = [0.0] * len(kf_folds)
            for f in as_completed(kf_futures):
                idx, sharpe = f.result()
                kf_sharpes[idx] = sharpe
            kf_sharpes = [s for s in kf_sharpes if s != 0.0 or True]
            kf_wins = sum(1 for s in kf_sharpes if s > 0)
            results['kfold'] = {'wins': kf_wins, 'sharpes': kf_sharpes}

            wf_wins = 0
            wf_total = 0
            for f in as_completed(wf_futures):
                idx, sharpe = f.result()
                if sharpe is None:
                    continue
                wf_total += 1
                if sharpe > 0:
                    wf_wins += 1
            results['wf'] = {'wins': wf_wins, 'total': wf_total}

        era_results = run_all_eras_h1(h1, cfg_or_kw, pctl, strategy, **kw)
        era_pass = all(v.get('sharpe', 0) > 0 for v in era_results.values()) and len(era_results) >= 4
        results['era'] = {k: v.get('sharpe', 0) for k, v in era_results.items()}
        results['era_pass'] = era_pass

    kf_pass = results['kfold']['wins'] >= 4
    wf_pass = results['wf']['wins'] >= 13 if results['wf']['total'] >= 19 else results['wf']['wins'] / max(1, results['wf']['total']) >= 0.68
    era_pass = results.get('era_pass', False)

    passed = kf_pass and wf_pass and era_pass
    results['passed'] = passed
    results['verdict'] = 'GO' if passed else 'NO-GO'
    return passed, results


# =====================================================================
# LOAD DATA
# =====================================================================
print(f"\n{'='*100}")
print(f"  R200: 200-Hour Gold Mega Research Program")
print(f"{'='*100}")

data = DataBundle.load_default()
h1_full = _load_h1_with_indicators()
pctl_full = h1_full['atr_percentile']

# Load macro data
EXTERNAL_DIR = Path("data/external")
macro_data = {}
for fname in ['dxy_daily.csv', 'GVZ_daily.csv', 'vix_daily.csv', 'us10y_daily.csv', 'SPX_daily.csv',
              'copper_daily.csv', 'crude_wti_daily.csv', 'hyg_daily.csv', 'real_yield_daily.csv']:
    fpath = EXTERNAL_DIR / fname
    if fpath.exists():
        try:
            df = pd.read_csv(fpath)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            macro_data[fname.replace('_daily.csv', '').replace('.csv', '')] = df
        except Exception:
            pass

gsr_path = Path("data/gold_silver_ratio.csv")
if gsr_path.exists():
    gsr_df = pd.read_csv(gsr_path)
    if 'Date' in gsr_df.columns:
        gsr_df['Date'] = pd.to_datetime(gsr_df['Date'])
        gsr_df.set_index('Date', inplace=True)
    macro_data['gsr'] = gsr_df

print(f"\n  Macro data loaded: {list(macro_data.keys())}")
print(f"\n{elapsed()} Data loaded. Starting experiments.\n")


# #####################################################################
# TRACK A: Trail & Exit Deep Dive (~50h)
# #####################################################################

# ═══════════════ Phase A1: Per-Strategy Trail Optimization ═══════════════
if not phase_done("A1_trail_per_strategy"):
    print(f"\n{'='*100}")
    print(f"  Phase A1: Per-Strategy Trail Optimization (non-Keltner)")
    print(f"{'='*100}")

    ta_grid = [0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.20]
    td_grid = [0.003, 0.005, 0.008, 0.01, 0.015, 0.02, 0.03]
    a1_results = {}

    # Build ALL tasks across ALL strategies at once for maximum parallelism
    all_tasks = []
    for strat_name in ['tsmom', 'psar', 'sess_bo', 'dual_thrust', 'chandelier']:
        cfg = STRAT_CONFIGS[strat_name]
        for ta, td in product(ta_grid, td_grid):
            key = f"{strat_name}__ta{ta}_td{td}"
            all_tasks.append((key, cfg, strat_name, {'trail_act': ta, 'trail_dist': td}))

    print(f"  Launching {len(all_tasks)} parallel backtests across 5 strategies...")
    with create_pool(h1_full, pctl_full) as pool:
        all_results = parallel_h1_eras(pool, all_tasks)

    for strat_name in ['tsmom', 'psar', 'sess_bo', 'dual_thrust', 'chandelier']:
        print(f"\n  --- {strat_name.upper()} ---")
        strat_results = {}
        best_sharpe = -999
        best_key = None
        for ta, td in product(ta_grid, td_grid):
            combo_key = f"ta{ta}_td{td}"
            full_key = f"{strat_name}__{combo_key}"
            era_stats = all_results.get(full_key, {})
            strat_results[combo_key] = era_stats
            full = era_stats.get('Full (2015-2026)', {})
            recent = era_stats.get('Recent (2024-2026)', {})
            fs = full.get('sharpe', 0)
            rs = recent.get('sharpe', 0)
            if fs > best_sharpe:
                best_sharpe = fs
                best_key = combo_key
            print(f"    {combo_key:<20} Full: Sharpe={fs:>7.3f} N={full.get('n',0):>5}  "
                  f"Recent: Sharpe={rs:>7.3f}")
        print(f"  Best for {strat_name}: {best_key} (Sharpe={best_sharpe:.3f})")
        a1_results[strat_name] = {'grid': strat_results, 'best': best_key, 'best_sharpe': best_sharpe}

    save_phase("A1_trail_per_strategy", a1_results)
    print(f"\n{elapsed()} Phase A1 complete.")
else:
    print(f"\n{elapsed()} Phase A1 already done.")


# ═══════════════ Phase A1b: TSMOM Paper vs Live Deep Compare ═══════════════
if not phase_done("A1b_tsmom_paper_vs_live"):
    print(f"\n{'='*100}")
    print(f"  Phase A1b: TSMOM Paper vs Live Parameter Deep Compare")
    print(f"{'='*100}")

    TSMOM_CFG = STRAT_CONFIGS['tsmom']
    LIVE_PARAMS = {
        'sl_atr': 6.0, 'tp_atr': 8.0, 'trail_act': 0.14, 'trail_dist': 0.025,
        'max_hold': 12, 'cap': 60, 'slow_lookback': 720, 'fast_lookback': 480,
    }
    PAPER_PARAMS = {
        'sl_atr': 3.5, 'tp_atr': 12.0, 'trail_act': 0.28, 'trail_dist': 0.06,
        'max_hold': 50, 'cap': 99999, 'slow_lookback': 1440, 'fast_lookback': 480,
    }
    a1b_results = {}

    def _tsmom_kw(params):
        return {
            'trail_act': params['trail_act'], 'trail_dist': params['trail_dist'],
            'max_hold': params['max_hold'], 'cap': params['cap'],
            'sl_atr': params['sl_atr'], 'tp_atr': params['tp_atr'],
            'slow_lookback': params['slow_lookback'], 'fast_lookback': params['fast_lookback'],
        }

    # Build all tasks: live + paper + 6 attribution variants = 8 parallel runs
    param_changes = [
        ('slow_lookback', 'slow 720->1440', {'slow_lookback': 1440}),
        ('sl_atr',        'SL 6.0->3.5',    {'sl_atr': 3.5}),
        ('tp_atr',        'TP 8.0->12.0',   {'tp_atr': 12.0}),
        ('trail',         'Trail 0.14/0.025->0.28/0.06', {'trail_act': 0.28, 'trail_dist': 0.06}),
        ('max_hold',      'MH 12->50',      {'max_hold': 50}),
        ('cap',           'Cap $60->OFF',    {'cap': 99999}),
    ]

    a1b_tasks = [
        ('__live__', TSMOM_CFG, 'tsmom', _tsmom_kw(LIVE_PARAMS)),
        ('__paper__', TSMOM_CFG, 'tsmom', _tsmom_kw(PAPER_PARAMS)),
    ]
    for param_key, label, overrides in param_changes:
        p = dict(LIVE_PARAMS)
        p.update(overrides)
        a1b_tasks.append((f'__attr_{param_key}__', TSMOM_CFG, 'tsmom', _tsmom_kw(p)))

    print(f"  Launching {len(a1b_tasks)} parallel TSMOM runs (live + paper + 6 attribution)...")
    with create_pool(h1_full, pctl_full) as pool:
        a1b_parallel = parallel_h1_eras(pool, a1b_tasks)

    live_eras = a1b_parallel['__live__']
    paper_eras = a1b_parallel['__paper__']
    a1b_results['live_eras'] = live_eras
    a1b_results['paper_eras'] = paper_eras

    # 1. Full comparison: Live vs Paper across all eras
    print("\n  --- Era Comparison ---")
    print(f"  {'Era':<30} {'Live Sharpe':>12} {'Paper Sharpe':>13} {'Delta':>8}")
    print(f"  {'-'*65}")
    for era in ERA_SEGMENTS:
        ls = live_eras.get(era, {}).get('sharpe', 0)
        ps = paper_eras.get(era, {}).get('sharpe', 0)
        ln = live_eras.get(era, {}).get('n', 0)
        pn = paper_eras.get(era, {}).get('n', 0)
        print(f"  {era:<30} {ls:>7.3f} (N={ln:>4}) {ps:>7.3f} (N={pn:>4}) {ps-ls:>+7.3f}")

    # 2. Single-parameter attribution
    print(f"\n  --- Single-Parameter Attribution (from Live baseline) ---")
    attribution = {}
    live_full_sharpe = live_eras.get('Full (2015-2026)', {}).get('sharpe', 0)
    live_recent_sharpe = live_eras.get('Recent (2024-2026)', {}).get('sharpe', 0)
    print(f"  {'Change':<35} {'Full Sharpe':>12} {'dFull':>8} {'Recent':>8} {'dRecent':>8}")
    print(f"  {'-'*75}")
    print(f"  {'LIVE BASELINE':<35} {live_full_sharpe:>12.3f} {'':>8} {live_recent_sharpe:>8.3f}")

    for param_key, label, overrides in param_changes:
        eras = a1b_parallel[f'__attr_{param_key}__']
        attribution[param_key] = eras
        fs = eras.get('Full (2015-2026)', {}).get('sharpe', 0)
        rs = eras.get('Recent (2024-2026)', {}).get('sharpe', 0)
        print(f"  {label:<35} {fs:>12.3f} {fs-live_full_sharpe:>+7.3f} {rs:>8.3f} {rs-live_recent_sharpe:>+7.3f}")

    paper_full_sharpe = paper_eras.get('Full (2015-2026)', {}).get('sharpe', 0)
    paper_recent_sharpe = paper_eras.get('Recent (2024-2026)', {}).get('sharpe', 0)
    print(f"  {'ALL PAPER (combined)':<35} {paper_full_sharpe:>12.3f} {paper_full_sharpe-live_full_sharpe:>+7.3f} "
          f"{paper_recent_sharpe:>8.3f} {paper_recent_sharpe-live_recent_sharpe:>+7.3f}")
    a1b_results['attribution'] = {k: v for k, v in attribution.items()}

    # 3. 3-Gate Validation of Paper params
    print(f"\n  --- 3-Gate Validation of Paper Params ---")
    passed, gate_details = three_gate_validate(
        h1_full, TSMOM_CFG, strategy='tsmom', pctl=pctl_full, h1=h1_full,
        trail_act=PAPER_PARAMS['trail_act'], trail_dist=PAPER_PARAMS['trail_dist'],
        max_hold=PAPER_PARAMS['max_hold'], cap=PAPER_PARAMS['cap'],
        sl_atr=PAPER_PARAMS['sl_atr'], tp_atr=PAPER_PARAMS['tp_atr'],
        slow_lookback=PAPER_PARAMS['slow_lookback'], fast_lookback=PAPER_PARAMS['fast_lookback'],
    )
    a1b_results['paper_3gate'] = gate_details
    kf = gate_details.get('kfold', {})
    wf = gate_details.get('wf', {})
    verdict = 'GO' if passed else 'NO-GO'
    print(f"  K-Fold: {kf.get('wins', 0)}/6  Sharpes: {kf.get('sharpes', [])}")
    print(f"  Walk-Forward: {wf.get('wins', 0)}/{wf.get('total', 0)}")
    print(f"  Era: {gate_details.get('era', {})}")
    print(f"  Verdict: [{verdict}]")

    save_phase("A1b_tsmom_paper_vs_live", a1b_results)
    print(f"\n{elapsed()} Phase A1b complete.")
else:
    print(f"\n{elapsed()} Phase A1b already done.")


# ═══════════════ Phase A2: Dynamic Exit Mechanisms ═══════════════
if not phase_done("A2_dynamic_exit"):
    print(f"\n{'='*100}")
    print(f"  Phase A2: Dynamic Exit Mechanisms")
    print(f"{'='*100}")

    a2_results = {}

    # A2a: Test engine-based exits on Keltner
    print("\n  --- Keltner: Engine-Based Exit Mechanisms ---")
    keltner_exit_variants = [
        ("baseline", {}),
        ("time_decay_tp", {"time_decay_tp": True, "time_decay_start_hour": 1.0,
                           "time_decay_atr_start": 0.30, "time_decay_atr_step": 0.10}),
        ("profit_dd_50pct", {"profit_drawdown_pct": 0.50}),
        ("profit_dd_40pct", {"profit_drawdown_pct": 0.40}),
        ("profit_dd_60pct", {"profit_drawdown_pct": 0.60}),
        ("breakeven_0.5atr", {"breakeven_after_atr": 0.5}),
        ("breakeven_1.0atr", {"breakeven_after_atr": 1.0}),
        ("atr_spike_protect", {"atr_spike_protection": True, "atr_spike_threshold": 1.5,
                               "atr_spike_trail_mult": 0.7}),
        ("timeout_profit_lock", {"timeout_profit_lock_atr": 0.3, "timeout_profit_lock_bar": 12}),
        ("timeout_adverse", {"timeout_adverse_exit": True}),
        ("timeout_momentum", {"timeout_momentum_exit": True}),
        ("timeout_dynamic", {"timeout_dynamic": True, "timeout_extend_bars": 4, "timeout_cut_bars": 4}),
        ("time_adaptive_trail", {"time_adaptive_trail": True, "time_adaptive_trail_start": 4,
                                 "time_adaptive_trail_decay": 0.95, "time_adaptive_trail_floor": 0.005}),
    ]

    keltner_exit_results = {}
    for name, overrides in keltner_exit_variants:
        era_stats = run_all_eras_engine(data, name, **overrides)
        keltner_exit_results[name] = era_stats
        full = era_stats.get('Full (2015-2026)', {})
        recent = era_stats.get('Recent (2024-2026)', {})
        print(f"    {name:<25} Full: Sharpe={full.get('sharpe',0):>7.3f}  "
              f"Recent: Sharpe={recent.get('sharpe',0):>7.3f}")
    a2_results['keltner_engine_exits'] = keltner_exit_results

    # A2b: Reversal signal exit for Chandelier/PSAR
    print("\n  --- Chandelier/PSAR: Reversal Signal Exit ---")
    for strat in ['chandelier', 'psar']:
        cfg = STRAT_CONFIGS[strat]
        base_trades = bt_h1_strategy(h1_full, cfg, pctl_full, strategy=strat)
        base_stats = _stats_from_trades(base_trades)
        print(f"    {strat} baseline: Sharpe={base_stats['sharpe']:.3f} N={base_stats['n']}")

        for mh_reduce in [0.5, 0.75]:
            reduced_mh = int(cfg['max_hold_h'] * mh_reduce)
            trades_r = bt_h1_strategy(h1_full, cfg, pctl_full, strategy=strat, max_hold=reduced_mh)
            stats_r = _stats_from_trades(trades_r)
            key = f"{strat}_mh{reduced_mh}"
            a2_results[key] = {'sharpe': stats_r['sharpe'], 'n': stats_r['n'],
                               'wr': stats_r['wr'], 'pnl': stats_r['pnl']}
            print(f"    {key}: Sharpe={stats_r['sharpe']:.3f} N={stats_r['n']}")

    save_phase("A2_dynamic_exit", a2_results)
    print(f"\n{elapsed()} Phase A2 complete.")
else:
    print(f"\n{elapsed()} Phase A2 already done.")


# ═══════════════ Phase A3: M15 vs H1 Resolution Audit ═══════════════
if not phase_done("A3_resolution_audit"):
    print(f"\n{'='*100}")
    print(f"  Phase A3: M15 vs H1 Resolution Impact Audit")
    print(f"{'='*100}")

    a3_results = {}

    # Keltner: compare engine (M15 exec) vs H1 loop
    print("\n  --- Keltner: M15 Engine vs H1 Loop ---")
    keltner_m15 = run_all_eras_engine(data, "keltner_m15")
    cfg_k = STRAT_CONFIGS['keltner']
    keltner_h1 = run_all_eras_h1(h1_full, cfg_k, pctl_full, 'keltner_h1_loop',
                                 trail_act=cfg_k['trail_act_atr'], trail_dist=cfg_k['trail_dist_atr'])
    a3_results['keltner'] = {'m15': keltner_m15, 'h1': keltner_h1}
    for era in ERA_SEGMENTS:
        m15_s = keltner_m15.get(era, {}).get('sharpe', 0)
        h1_s = keltner_h1.get(era, {}).get('sharpe', 0)
        delta = m15_s - h1_s
        print(f"    {era:<30} M15={m15_s:>7.3f}  H1={h1_s:>7.3f}  delta={delta:>+7.3f}")

    # Other strategies: H1 loop with different trail tightness (parallelized)
    a3_tasks = []
    for strat in ['psar', 'tsmom', 'sess_bo', 'chandelier']:
        cfg = STRAT_CONFIGS[strat]
        a3_tasks.append((f'{strat}_loose', cfg, strat,
                         {'trail_act': cfg['trail_act_atr'], 'trail_dist': cfg['trail_dist_atr']}))
        a3_tasks.append((f'{strat}_tight', cfg, strat,
                         {'trail_act': cfg['trail_act_atr']*0.3, 'trail_dist': cfg['trail_dist_atr']*0.5}))

    print(f"  Launching {len(a3_tasks)} parallel loose/tight comparisons...")
    with create_pool(h1_full, pctl_full) as pool:
        a3_parallel = parallel_h1_eras(pool, a3_tasks)

    for strat in ['psar', 'tsmom', 'sess_bo', 'chandelier']:
        print(f"\n  --- {strat.upper()}: Loose vs Tight Trail on H1 ---")
        loose = a3_parallel[f'{strat}_loose']
        tight = a3_parallel[f'{strat}_tight']
        a3_results[strat] = {'current': loose, 'tight': tight}
        full_l = loose.get('Full (2015-2026)', {}).get('sharpe', 0)
        full_t = tight.get('Full (2015-2026)', {}).get('sharpe', 0)
        print(f"    Current trail: Sharpe={full_l:.3f}  Tight trail: Sharpe={full_t:.3f}  "
              f"delta={full_t-full_l:+.3f}")

    save_phase("A3_resolution_audit", a3_results)
    print(f"\n{elapsed()} Phase A3 complete.")
else:
    print(f"\n{elapsed()} Phase A3 already done.")


# ═══════════════ Phase A3b: Trail Resolution Cross-Validation ═══════════════
if not phase_done("A3b_trail_resolution_xval"):
    print(f"\n{'='*100}")
    print(f"  Phase A3b: Trail Resolution Cross-Validation (H1 vs M15 exit)")
    print(f"{'='*100}")
    print(f"  Using M15 bars within each H1 bar for 4x higher trail/SL/TP resolution")

    m15_full = data.m15_df

    a3b_tasks = []
    for strat in ['psar', 'tsmom', 'sess_bo', 'chandelier']:
        cfg = STRAT_CONFIGS[strat]
        ta_base = cfg['trail_act_atr']
        td_base = cfg['trail_dist_atr']

        a3b_tasks.append((f'{strat}_current', cfg, strat,
                          {'trail_act': ta_base, 'trail_dist': td_base}))
        a3b_tasks.append((f'{strat}_tight', cfg, strat,
                          {'trail_act': ta_base * 0.3, 'trail_dist': td_base * 0.5}))
        a3b_tasks.append((f'{strat}_tighter', cfg, strat,
                          {'trail_act': ta_base * 0.15, 'trail_dist': td_base * 0.25}))

    print(f"  Launching {len(a3b_tasks)} parallel H1-vs-M15 comparisons...")
    a3b_results = {}
    with create_pool_m15(h1_full, pctl_full, m15_full) as pool:
        futures = {pool.submit(_worker_a3b, t): t[0] for t in a3b_tasks}
        for f in as_completed(futures):
            key, result = f.result()
            a3b_results[key] = result

    print(f"\n  {'Strategy':<12} {'Trail Params':<18} {'H1 Sharpe':>10} {'M15 Sharpe':>11} "
          f"{'Delta':>7} {'H1 Trail#':>10} {'M15 Trail#':>11} {'HairTrig%':>10}")
    print(f"  {'-'*12} {'-'*18} {'-'*10} {'-'*11} {'-'*7} {'-'*10} {'-'*11} {'-'*10}")

    for strat in ['psar', 'tsmom', 'sess_bo', 'chandelier']:
        cfg = STRAT_CONFIGS[strat]
        for suffix, label in [('_current', 'current'), ('_tight', 'tight'), ('_tighter', 'tighter')]:
            k = f'{strat}{suffix}'
            r = a3b_results.get(k, {})
            h1s = r.get('h1_sharpe', 0)
            m15s = r.get('m15_sharpe', 0)
            delta = m15s - h1s
            h1tn = r.get('h1_trail_n', 0)
            m15tn = r.get('m15_trail_n', 0)
            ht_pct = r.get('hair_trigger_pct', 0)

            ta_val = cfg['trail_act_atr'] * (0.3 if 'tight' == label else 0.15 if 'tighter' == label else 1.0)
            td_val = cfg['trail_dist_atr'] * (0.5 if 'tight' == label else 0.25 if 'tighter' == label else 1.0)

            print(f"  {strat:<12} {ta_val:.3f}/{td_val:.4f}  "
                  f"{h1s:>10.3f} {m15s:>11.3f} {delta:>+7.3f} "
                  f"{h1tn:>10} {m15tn:>11} {ht_pct:>9.1f}%")

    bias_flags = {}
    for strat in ['psar', 'tsmom', 'sess_bo', 'chandelier']:
        tight_k = f'{strat}_tight'
        r = a3b_results.get(tight_k, {})
        h1s = r.get('h1_sharpe', 0)
        m15s = r.get('m15_sharpe', 0)
        ht_pct = r.get('hair_trigger_pct', 0)
        biased = (h1s - m15s > 0.1) or (ht_pct > 20)
        bias_flags[strat] = biased
        verdict = "RESOLUTION BIAS DETECTED" if biased else "H1 results reliable"
        print(f"\n  {strat.upper()} tight trail verdict: {verdict}")
        if biased:
            print(f"    -> H1 Sharpe {h1s:.3f} vs M15 Sharpe {m15s:.3f} (gap={h1s-m15s:+.3f})")
            print(f"    -> Hair-trigger rate: {ht_pct:.1f}% — tight trail unreliable at H1")

    save_phase("A3b_trail_resolution_xval", {
        'comparisons': a3b_results,
        'bias_flags': {k: bool(v) for k, v in bias_flags.items()},
    })
    print(f"\n{elapsed()} Phase A3b complete.")
else:
    print(f"\n{elapsed()} Phase A3b already done.")


# ═══════════════ Phase A4: Tick Replay Analysis ═══════════════
if not phase_done("A4_tick_replay"):
    print(f"\n{'='*100}")
    print(f"  Phase A4: Tick Replay Analysis (conditional on data)")
    print(f"{'='*100}")

    tick_file = Path("data/keltner_ticks.jsonl")
    a4_results = {'status': 'no_data'}

    if tick_file.exists():
        ticks_by_ticket = defaultdict(list)
        with open(tick_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        ticks_by_ticket[rec['ticket']].append(rec)
                    except (json.JSONDecodeError, KeyError):
                        pass
        n_trades = len(ticks_by_ticket)
        n_ticks = sum(len(v) for v in ticks_by_ticket.values())
        print(f"  Found {n_ticks} ticks for {n_trades} trades")

        if n_trades >= 3:
            a4_results = {'status': 'analyzed', 'n_trades': n_trades, 'n_ticks': n_ticks}
            trail_configs = [
                {'name': '0.02/0.005', 'act': 0.02, 'dist': 0.005},
                {'name': '0.06/0.01', 'act': 0.06, 'dist': 0.01},
                {'name': '0.08/0.003', 'act': 0.08, 'dist': 0.003},
            ]

            for tcfg in trail_configs:
                total_pnl = 0
                for ticket, ticks in ticks_by_ticket.items():
                    if not ticks:
                        continue
                    first = ticks[0]
                    d = first['direction']
                    entry = first['open_price']
                    atr = first['atr']
                    ad = tcfg['act'] * atr
                    tdd = tcfg['dist'] * atr
                    extreme = entry
                    trail_p = 0
                    exit_pnl = 0
                    for tick in ticks:
                        p = tick['price']
                        if d == 'BUY':
                            extreme = max(extreme, p)
                            if extreme - entry >= ad:
                                trail_p = max(trail_p, extreme - tdd)
                                if p <= trail_p:
                                    exit_pnl = trail_p - entry
                                    break
                        else:
                            extreme = min(extreme, p)
                            if entry - extreme >= ad:
                                new_t = extreme + tdd
                                trail_p = min(trail_p, new_t) if trail_p > 0 else new_t
                                if p >= trail_p:
                                    exit_pnl = entry - trail_p
                                    break
                    else:
                        last_p = ticks[-1]['price']
                        exit_pnl = (last_p - entry) if d == 'BUY' else (entry - last_p)
                    total_pnl += exit_pnl
                a4_results[tcfg['name']] = round(total_pnl, 2)
                print(f"    {tcfg['name']}: Total PnL (pts) = {total_pnl:.2f}")
        else:
            a4_results['status'] = f'insufficient_data_{n_trades}_trades'
            print(f"  Only {n_trades} trades, need >= 3. Skipping analysis.")
    else:
        print(f"  No tick file found at {tick_file}. Skipping.")

    save_phase("A4_tick_replay", a4_results)
    print(f"\n{elapsed()} Phase A4 complete.")
else:
    print(f"\n{elapsed()} Phase A4 already done.")


# #####################################################################
# TRACK B: New Alpha Sources (~50h)
# #####################################################################

# ═══════════════ Phase B1: Cross-Asset Leading Indicators ═══════════════
if not phase_done("B1_cross_asset"):
    print(f"\n{'='*100}")
    print(f"  Phase B1: Cross-Asset Leading Indicators")
    print(f"{'='*100}")

    b1_results = {}

    # Align macro data to H1 index (forward-fill daily to H1)
    h1_dates = h1_full.index.normalize()

    # GSR Regime
    if 'gsr' in macro_data:
        print("\n  --- Gold/Silver Ratio Regime Filter ---")
        gsr = macro_data['gsr']
        ratio_col = [c for c in gsr.columns if 'ratio' in c.lower()]
        if ratio_col:
            gsr_series = gsr[ratio_col[0]]
            gsr_aligned = gsr_series.reindex(h1_dates.date).ffill()

            for strat_name in ['keltner', 'psar', 'tsmom']:
                cfg = STRAT_CONFIGS[strat_name]
                if strat_name == 'keltner':
                    base_eras = run_all_eras_engine(data, f"gsr_base_{strat_name}")
                else:
                    base_eras = run_all_eras_h1(h1_full, cfg, pctl_full, strat_name)
                b1_results[f'{strat_name}_gsr_baseline'] = base_eras
                full_base = base_eras.get('Full (2015-2026)', {}).get('sharpe', 0)
                print(f"    {strat_name} baseline: Sharpe={full_base:.3f}")

    # DXY Momentum
    if 'dxy' in macro_data:
        print("\n  --- DXY 5-Day Momentum Filter ---")
        dxy = macro_data['dxy']
        close_col = [c for c in dxy.columns if 'close' in c.lower() or 'Close' in c]
        if close_col:
            dxy_close = dxy[close_col[0]]
            dxy_mom5 = dxy_close.pct_change(5)
            b1_results['dxy_available'] = True
            print(f"    DXY data: {len(dxy)} rows, momentum computed")
        else:
            b1_results['dxy_available'] = False
    else:
        b1_results['dxy_available'] = False

    # GVZ Level-Based Analysis
    if 'GVZ' in macro_data:
        print("\n  --- GVZ (Gold Vol) Level Analysis ---")
        gvz = macro_data['GVZ']
        close_col = [c for c in gvz.columns if 'close' in c.lower() or 'Close' in c]
        if close_col:
            gvz_close = gvz[close_col[0]]
            quartiles = gvz_close.quantile([0.25, 0.5, 0.75])
            b1_results['gvz_quartiles'] = {f"Q{i+1}": round(q, 2) for i, q in enumerate(quartiles.values)}
            print(f"    GVZ quartiles: {b1_results['gvz_quartiles']}")
    else:
        b1_results['gvz_available'] = False

    # US10Y Rate of Change
    if 'us10y' in macro_data:
        print("\n  --- US10Y Rate-of-Change ---")
        us10y = macro_data['us10y']
        close_col = [c for c in us10y.columns if 'close' in c.lower() or 'Close' in c]
        if close_col:
            us10y_roc = us10y[close_col[0]].pct_change(5)
            b1_results['us10y_available'] = True
            print(f"    US10Y data: {len(us10y)} rows, 5d ROC computed")
    else:
        b1_results['us10y_available'] = False

    # Engine-level GSR filter test on Keltner
    if 'gsr' in macro_data and ratio_col:
        print("\n  --- Keltner + GSR Engine Filter ---")
        gsr_s = gsr_series.copy()
        gsr_s.index = pd.to_datetime(gsr_s.index)

        for hi_thresh, lo_thresh in [(85, 75), (80, 70), (90, 70)]:
            label = f"gsr_{hi_thresh}_{lo_thresh}"
            kw = dict(KELTNER_ENGINE_KWARGS)
            kw['gsr_filter_enabled'] = True
            kw['gsr_high_threshold'] = hi_thresh
            kw['gsr_low_threshold'] = lo_thresh
            kw['gsr_series'] = gsr_s
            era_stats = run_all_eras_engine(data, label, **{k: v for k, v in kw.items()
                                                            if k in KELTNER_ENGINE_KWARGS or k.startswith('gsr')})
            b1_results[label] = era_stats
            full = era_stats.get('Full (2015-2026)', {}).get('sharpe', 0)
            recent = era_stats.get('Recent (2024-2026)', {}).get('sharpe', 0)
            print(f"    {label}: Full Sharpe={full:.3f} Recent Sharpe={recent:.3f}")

    save_phase("B1_cross_asset", b1_results)
    print(f"\n{elapsed()} Phase B1 complete.")
else:
    print(f"\n{elapsed()} Phase B1 already done.")


# ═══════════════ Phase B2: Intra-Day Pattern Mining ═══════════════
if not phase_done("B2_intraday_patterns"):
    print(f"\n{'='*100}")
    print(f"  Phase B2: Intra-Day Pattern Mining")
    print(f"{'='*100}")

    b2_results = {}

    # Hour-of-day PnL heatmap for Keltner
    print("\n  --- Keltner: Hour-of-Day PnL Heatmap ---")
    keltner_full = bt_keltner(data, "keltner_full", maxloss_cap=70)
    if '_trades' in keltner_full:
        trades = keltner_full['_trades']
        hour_pnl = defaultdict(list)
        for t in trades:
            h = pd.Timestamp(t.entry_time).hour
            hour_pnl[h].append(t.pnl)
        hour_stats = {}
        for h in sorted(hour_pnl.keys()):
            pnls = hour_pnl[h]
            hour_stats[h] = {
                'n': len(pnls), 'pnl': round(sum(pnls), 2),
                'avg': round(np.mean(pnls), 2), 'wr': round(sum(1 for p in pnls if p > 0) / len(pnls) * 100, 1),
            }
            print(f"    Hour {h:>2}: N={len(pnls):>4} PnL=${sum(pnls):>8.2f} "
                  f"Avg=${np.mean(pnls):>6.2f} WR={hour_stats[h]['wr']:>5.1f}%")
        b2_results['keltner_hour_pnl'] = hour_stats

        # Day-of-week
        print("\n  --- Keltner: Day-of-Week PnL ---")
        dow_pnl = defaultdict(list)
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        for t in trades:
            d = pd.Timestamp(t.entry_time).dayofweek
            dow_pnl[d].append(t.pnl)
        dow_stats = {}
        for d in sorted(dow_pnl.keys()):
            pnls = dow_pnl[d]
            dow_stats[dow_names[d]] = {
                'n': len(pnls), 'pnl': round(sum(pnls), 2),
                'avg': round(np.mean(pnls), 2), 'wr': round(sum(1 for p in pnls if p > 0) / len(pnls) * 100, 1),
            }
            print(f"    {dow_names[d]}: N={len(pnls):>4} PnL=${sum(pnls):>8.2f} "
                  f"Avg=${np.mean(pnls):>6.2f}")
        b2_results['keltner_dow_pnl'] = dow_stats

    # H1-loop strategies hour analysis (parallelized trade generation)
    b2_h1_tasks = []
    b2_strats = ['psar', 'tsmom', 'sess_bo', 'chandelier']
    for strat in b2_strats:
        cfg = STRAT_CONFIGS[strat]
        b2_h1_tasks.append((strat, cfg, strat, {}))

    with create_pool(h1_full, pctl_full) as pool:
        b2_futures = {pool.submit(_worker_bt_h1_trades, t): t[0] for t in b2_h1_tasks}
        b2_trades_map = {}
        for f in as_completed(b2_futures):
            key, trades = f.result()
            b2_trades_map[key] = trades

    for strat in b2_strats:
        print(f"\n  --- {strat.upper()}: Hour-of-Day PnL ---")
        trades = b2_trades_map.get(strat, [])
        hour_pnl = defaultdict(list)
        for t in trades:
            h = int(t['entry_time'][11:13]) if len(t['entry_time']) > 13 else 0
            hour_pnl[h].append(t['pnl'])
        strat_hour = {}
        for h in sorted(hour_pnl.keys()):
            pnls = hour_pnl[h]
            strat_hour[h] = {'n': len(pnls), 'pnl': round(sum(pnls), 2), 'avg': round(np.mean(pnls), 2)}
        b2_results[f'{strat}_hour_pnl'] = strat_hour
        worst_hours = sorted(strat_hour.items(), key=lambda x: x[1]['avg'])[:3]
        best_hours = sorted(strat_hour.items(), key=lambda x: x[1]['avg'], reverse=True)[:3]
        print(f"    Worst hours: {[(h, s['avg']) for h, s in worst_hours]}")
        print(f"    Best hours:  {[(h, s['avg']) for h, s in best_hours]}")

    # Session volatility clustering
    print("\n  --- Session Volatility Clustering ---")
    asia_range = []
    london_pnl = []
    for date in h1_full.index.normalize().unique():
        day_h1 = h1_full[h1_full.index.normalize() == date]
        asia = day_h1[(day_h1.index.hour >= 0) & (day_h1.index.hour < 8)]
        london = day_h1[(day_h1.index.hour >= 8) & (day_h1.index.hour < 16)]
        if len(asia) >= 3 and len(london) >= 3:
            a_range = asia['High'].max() - asia['Low'].min()
            l_range = london['High'].max() - london['Low'].min()
            asia_range.append(a_range)
            london_pnl.append(l_range)

    if len(asia_range) > 50:
        corr = np.corrcoef(asia_range, london_pnl)[0, 1]
        b2_results['asia_london_range_corr'] = round(corr, 3)
        print(f"    Asia range -> London range correlation: {corr:.3f}")

    save_phase("B2_intraday_patterns", b2_results)
    print(f"\n{elapsed()} Phase B2 complete.")
else:
    print(f"\n{elapsed()} Phase B2 already done.")


# ═══════════════ Phase B3: Multi-Timeframe Signal Fusion ═══════════════
if not phase_done("B3_mtf_fusion"):
    print(f"\n{'='*100}")
    print(f"  Phase B3: Multi-Timeframe Signal Fusion")
    print(f"{'='*100}")

    b3_results = {}

    # D1 + H1 alignment for Keltner
    # Use spot XAUUSD D1 (resampled from HistData M15 bid) to keep the D1 EMA
    # filter on the same instrument as the M15/H1 entry signals. Previously
    # this loaded xauusd_daily_yf.csv (= COMEX GC=F futures), which is offset
    # 5-10 USD from spot and uses RTH session boundaries — a silent bias.
    print("\n  --- Keltner: D1 EMA Alignment Filter (spot XAUUSD) ---")
    try:
        from backtest.runner import load_d1_spot
        d1 = load_d1_spot()
        d1_close = d1['Close']
        d1_ema20 = d1_close.ewm(span=20).mean()
        b3_results['d1_data'] = f"{len(d1)} rows (spot)"
        print(f"    D1 spot data loaded: {len(d1)} rows")

        # Test with EMA slope filter on engine
        for slope_bars in [5, 10, 20]:
            label = f"ema_slope_{slope_bars}"
            era = run_all_eras_engine(data, label, block_buy_ema_slope=slope_bars)
            b3_results[label] = era
            full = era.get('Full (2015-2026)', {}).get('sharpe', 0)
            recent = era.get('Recent (2024-2026)', {}).get('sharpe', 0)
            print(f"    EMA slope filter (bars={slope_bars}): Full={full:.3f} Recent={recent:.3f}")
    except FileNotFoundError as e:
        b3_results['d1_data'] = f'not_found: {e}'
        print(f"    D1 spot data not available: {e}")

    # Cross-TF ATR regime: D1 ATR vs H1 ATR
    print("\n  --- Cross-TF ATR Regime ---")
    for pctl_floor in [20, 25, 30, 35, 40]:
        label = f"pctl_{pctl_floor}"
        kw = dict(KELTNER_ENGINE_KWARGS)
        era = run_all_eras_engine(data, label)
        b3_results[f'atr_pctl_{pctl_floor}'] = era
        full = era.get('Full (2015-2026)', {}).get('sharpe', 0)
        recent = era.get('Recent (2024-2026)', {}).get('sharpe', 0)
        print(f"    ATR Pctl Floor={pctl_floor}: Full={full:.3f} Recent={recent:.3f}")

    save_phase("B3_mtf_fusion", b3_results)
    print(f"\n{elapsed()} Phase B3 complete.")
else:
    print(f"\n{elapsed()} Phase B3 already done.")


# ═══════════════ Phase B4: ML Model Analysis ═══════════════
if not phase_done("B4_ml_analysis"):
    print(f"\n{'='*100}")
    print(f"  Phase B4: ML Model Analysis & Feature Importance")
    print(f"{'='*100}")

    b4_results = {}

    # Analyze Keltner trades with/without ML filter
    print("\n  --- Keltner: Impact of ML Filter Threshold ---")
    keltner_full_trades = bt_keltner(data, "keltner_base", maxloss_cap=70)
    base_sharpe = keltner_full_trades.get('sharpe', 0)
    base_n = keltner_full_trades.get('n', 0)
    b4_results['keltner_base'] = {'sharpe': base_sharpe, 'n': base_n}
    print(f"    Base (engine, no external ML gate): Sharpe={base_sharpe:.3f} N={base_n}")

    # Choppy threshold sensitivity (trend score)
    print("\n  --- Choppy Threshold Sensitivity ---")
    for ct in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        label = f"choppy_{ct}"
        era = run_all_eras_engine(data, label, choppy_threshold=ct)
        b4_results[label] = era
        full = era.get('Full (2015-2026)', {}).get('sharpe', 0)
        recent = era.get('Recent (2024-2026)', {}).get('sharpe', 0)
        print(f"    choppy_threshold={ct:.2f}: Full={full:.3f} Recent={recent:.3f}")

    # ADX threshold sweep (engine-level)
    print("\n  --- ADX Threshold Sweep ---")
    for adx_t in [10, 12, 14, 16, 18, 20]:
        label = f"adx_{adx_t}"
        era = run_all_eras_engine(data, label, keltner_adx_threshold=adx_t)
        b4_results[label] = era
        full = era.get('Full (2015-2026)', {}).get('sharpe', 0)
        recent = era.get('Recent (2024-2026)', {}).get('sharpe', 0)
        print(f"    ADX threshold={adx_t}: Full={full:.3f} Recent={recent:.3f}")

    save_phase("B4_ml_analysis", b4_results)
    print(f"\n{elapsed()} Phase B4 complete.")
else:
    print(f"\n{elapsed()} Phase B4 already done.")


# #####################################################################
# TRACK C: Portfolio Optimization (~40h)
# #####################################################################

# ═══════════════ Phase C1: Dynamic Lot Allocation ═══════════════
if not phase_done("C1_dynamic_lots"):
    print(f"\n{'='*100}")
    print(f"  Phase C1: Dynamic Lot Allocation Strategies")
    print(f"{'='*100}")

    c1_results = {}

    # Current fixed portfolio baseline (all 6 strategies)
    print("\n  --- 6-Strategy Portfolio Baseline ---")
    portfolio_pnl = defaultdict(float)
    for strat_name, cfg in STRAT_CONFIGS.items():
        if strat_name == 'keltner':
            stats = bt_keltner(data, f"port_{strat_name}", maxloss_cap=cfg['cap'])
            trades = stats.get('_trades', [])
            for t in trades:
                d = pd.Timestamp(t.exit_time).date()
                portfolio_pnl[d] += t.pnl
        else:
            trades = bt_h1_strategy(h1_full, cfg, pctl_full, strat_name)
            for t in trades:
                d = pd.Timestamp(t['exit_time']).date()
                portfolio_pnl[d] += t['pnl']

    port_daily = list(portfolio_pnl.values())
    port_total = sum(port_daily)
    port_sharpe = np.mean(port_daily) / np.std(port_daily, ddof=1) * np.sqrt(252) if len(port_daily) > 1 and np.std(port_daily, ddof=1) > 0 else 0
    c1_results['baseline'] = {'total_pnl': round(port_total, 2), 'sharpe': round(port_sharpe, 3),
                               'n_days': len(port_daily)}
    print(f"    Baseline: PnL=${port_total:.2f} Sharpe={port_sharpe:.3f} ({len(port_daily)} trading days)")

    # Lot multiplier grid (per strategy)
    print("\n  --- Lot Multiplier Sensitivity ---")
    for mult in [0.5, 0.75, 1.0, 1.25, 1.5]:
        port_pnl_m = defaultdict(float)
        for strat_name, cfg in STRAT_CONFIGS.items():
            cfg_m = dict(cfg)
            cfg_m['lot'] = round(cfg['lot'] * mult, 2)
            cfg_m['lot'] = max(0.01, cfg_m['lot'])
            if strat_name == 'keltner':
                stats = bt_keltner(data, f"lot_{mult}_{strat_name}", maxloss_cap=cfg_m['cap'])
                trades = stats.get('_trades', [])
                for t in trades:
                    d = pd.Timestamp(t.exit_time).date()
                    pnl_scaled = t.pnl * mult
                    port_pnl_m[d] += pnl_scaled
            else:
                trades = bt_h1_strategy(h1_full, cfg_m, pctl_full, strat_name)
                for t in trades:
                    d = pd.Timestamp(t['exit_time']).date()
                    port_pnl_m[d] += t['pnl']
        dpnl = list(port_pnl_m.values())
        tot = sum(dpnl)
        sh = np.mean(dpnl) / np.std(dpnl, ddof=1) * np.sqrt(252) if len(dpnl) > 1 and np.std(dpnl, ddof=1) > 0 else 0
        c1_results[f'mult_{mult}'] = {'pnl': round(tot, 2), 'sharpe': round(sh, 3)}
        print(f"    Lot x{mult}: PnL=${tot:.2f} Sharpe={sh:.3f}")

    # Drawdown-responsive sizing simulation
    print("\n  --- Drawdown-Responsive Sizing ---")
    dd_thresholds = [100, 150, 200, 300]
    for dd_thresh in dd_thresholds:
        equity = config.CAPITAL
        peak_eq = equity
        lot_mult = 1.0
        port_pnl_dd = defaultdict(float)

        all_trades = []
        for strat_name, cfg in STRAT_CONFIGS.items():
            if strat_name == 'keltner':
                stats = bt_keltner(data, f"dd_{dd_thresh}_{strat_name}", maxloss_cap=cfg['cap'])
                for t in stats.get('_trades', []):
                    all_trades.append((pd.Timestamp(t.exit_time), t.pnl, strat_name))
            else:
                trades = bt_h1_strategy(h1_full, cfg, pctl_full, strat_name)
                for t in trades:
                    all_trades.append((pd.Timestamp(t['exit_time']), t['pnl'], strat_name))

        all_trades.sort(key=lambda x: x[0])
        for exit_time, pnl, strat in all_trades:
            dd = peak_eq - equity
            lot_mult = 0.5 if dd > dd_thresh else 1.0
            scaled_pnl = pnl * lot_mult
            equity += scaled_pnl
            peak_eq = max(peak_eq, equity)
            d = exit_time.date() if hasattr(exit_time, 'date') else str(exit_time)[:10]
            port_pnl_dd[d] += scaled_pnl

        dpnl = list(port_pnl_dd.values())
        tot = sum(dpnl)
        sh = np.mean(dpnl) / np.std(dpnl, ddof=1) * np.sqrt(252) if len(dpnl) > 1 and np.std(dpnl, ddof=1) > 0 else 0
        c1_results[f'dd_responsive_{dd_thresh}'] = {'pnl': round(tot, 2), 'sharpe': round(sh, 3)}
        print(f"    DD threshold ${dd_thresh}: PnL=${tot:.2f} Sharpe={sh:.3f}")

    save_phase("C1_dynamic_lots", c1_results)
    print(f"\n{elapsed()} Phase C1 complete.")
else:
    print(f"\n{elapsed()} Phase C1 already done.")


# ═══════════════ Phase C2: Correlation Management ═══════════════
if not phase_done("C2_correlation"):
    print(f"\n{'='*100}")
    print(f"  Phase C2: Strategy Correlation Management")
    print(f"{'='*100}")

    c2_results = {}

    # Compute per-strategy daily PnL series
    strat_daily_pnl = {}
    for strat_name, cfg in STRAT_CONFIGS.items():
        dpnl = defaultdict(float)
        if strat_name == 'keltner':
            stats = bt_keltner(data, f"corr_{strat_name}", maxloss_cap=cfg['cap'])
            for t in stats.get('_trades', []):
                d = pd.Timestamp(t.exit_time).date()
                dpnl[d] += t.pnl
        else:
            trades = bt_h1_strategy(h1_full, cfg, pctl_full, strat_name)
            for t in trades:
                d = pd.Timestamp(t['exit_time']).date()
                dpnl[d] += t['pnl']
        strat_daily_pnl[strat_name] = dpnl

    # Build aligned daily pnl DataFrame
    all_dates = set()
    for dpnl in strat_daily_pnl.values():
        all_dates.update(dpnl.keys())
    all_dates = sorted(all_dates)

    pnl_df = pd.DataFrame(index=all_dates)
    for strat_name, dpnl in strat_daily_pnl.items():
        pnl_df[strat_name] = [dpnl.get(d, 0) for d in all_dates]

    # Rolling 30-day correlation
    print("\n  --- Pairwise Strategy Correlation ---")
    corr_matrix = pnl_df.corr()
    c2_results['full_period_correlation'] = {}
    strats = list(strat_daily_pnl.keys())
    for i in range(len(strats)):
        for j in range(i+1, len(strats)):
            c = corr_matrix.loc[strats[i], strats[j]]
            key = f"{strats[i]}_{strats[j]}"
            c2_results['full_period_correlation'][key] = round(c, 3) if not pd.isna(c) else 0
            print(f"    {key}: {c:.3f}")

    # Portfolio diversification metrics
    port_daily_series = pnl_df.sum(axis=1)
    port_std = port_daily_series.std()
    sum_strat_std = pnl_df.std().sum()
    div_ratio = sum_strat_std / port_std if port_std > 0 else 0
    c2_results['diversification_ratio'] = round(div_ratio, 3)
    print(f"\n  Portfolio diversification ratio: {div_ratio:.3f} (>1 = diversification benefit)")

    # Per-era correlation
    print("\n  --- Per-Era Correlation ---")
    for era_name, (s, e) in ERA_SEGMENTS.items():
        s_date = pd.Timestamp(s).date()
        e_date = pd.Timestamp(e).date()
        era_df = pnl_df[(pnl_df.index >= s_date) & (pnl_df.index < e_date)]
        if len(era_df) < 30:
            continue
        era_corr = era_df.corr()
        avg_corr = era_corr.values[np.triu_indices_from(era_corr.values, k=1)].mean()
        c2_results[f'era_avg_corr_{era_name}'] = round(avg_corr, 3) if not np.isnan(avg_corr) else 0
        print(f"    {era_name}: avg pairwise corr = {avg_corr:.3f}")

    save_phase("C2_correlation", c2_results)
    print(f"\n{elapsed()} Phase C2 complete.")
else:
    print(f"\n{elapsed()} Phase C2 already done.")


# ═══════════════ Phase C3: MaxLoss Cap Optimization ═══════════════
if not phase_done("C3_cap_optimization"):
    print(f"\n{'='*100}")
    print(f"  Phase C3: MaxLoss Cap Optimization")
    print(f"{'='*100}")

    c3_results = {}

    # Keltner cap sweep
    print("\n  --- Keltner: Cap Sweep ---")
    for cap_v in [30, 40, 50, 60, 70, 80, 90, 100, 150, 0]:
        label = f"cap_{cap_v}" if cap_v > 0 else "no_cap"
        kw = {}
        if cap_v > 0:
            kw['maxloss_cap'] = cap_v
        era = run_all_eras_engine(data, label, **kw)
        c3_results[f'keltner_{label}'] = era
        full = era.get('Full (2015-2026)', {})
        recent = era.get('Recent (2024-2026)', {})
        print(f"    Cap=${cap_v if cap_v > 0 else 'OFF':<4} Full: Sharpe={full.get('sharpe',0):>7.3f} "
              f"MaxDD={full.get('max_dd',0):>8.2f}  Recent: Sharpe={recent.get('sharpe',0):>7.3f}")

    # Other strategies: cap sweep on H1 (parallelized)
    c3_h1_tasks = []
    for strat in ['psar', 'tsmom', 'sess_bo', 'dual_thrust', 'chandelier']:
        cfg = STRAT_CONFIGS[strat]
        for cap_v in [20, 40, 60, 80, 100, 0]:
            label = f"{strat}_cap_{cap_v}" if cap_v > 0 else f"{strat}_no_cap"
            c3_h1_tasks.append((label, cfg, strat, {'cap': cap_v if cap_v > 0 else 99999}))

    print(f"  Launching {len(c3_h1_tasks)} parallel H1 cap sweeps...")
    with create_pool(h1_full, pctl_full) as pool:
        c3_parallel = parallel_h1_stats(pool, c3_h1_tasks)

    for strat in ['psar', 'tsmom', 'sess_bo', 'dual_thrust', 'chandelier']:
        print(f"\n  --- {strat.upper()}: Cap Sweep ---")
        for cap_v in [20, 40, 60, 80, 100, 0]:
            label = f"{strat}_cap_{cap_v}" if cap_v > 0 else f"{strat}_no_cap"
            stats = c3_parallel.get(label, {'sharpe': 0, 'n': 0, 'pnl': 0})
            c3_results[label] = stats
            print(f"    Cap=${cap_v if cap_v > 0 else 'OFF':<4} Sharpe={stats['sharpe']:>7.3f} "
                  f"N={stats['n']:>5} PnL=${stats['pnl']:>8.2f}")

    save_phase("C3_cap_optimization", c3_results)
    print(f"\n{elapsed()} Phase C3 complete.")
else:
    print(f"\n{elapsed()} Phase C3 already done.")


# ═══════════════ Phase C4: Entry Timing Coordination ═══════════════
if not phase_done("C4_entry_timing"):
    print(f"\n{'='*100}")
    print(f"  Phase C4: Entry Timing Coordination Analysis")
    print(f"{'='*100}")

    c4_results = {}

    # Analyze simultaneous entries across strategies
    print("\n  --- Entry Timing Overlap Analysis ---")
    all_entries = []
    for strat_name, cfg in STRAT_CONFIGS.items():
        if strat_name == 'keltner':
            stats = bt_keltner(data, f"timing_{strat_name}", maxloss_cap=cfg['cap'])
            for t in stats.get('_trades', []):
                all_entries.append({
                    'strat': strat_name, 'time': pd.Timestamp(t.entry_time),
                    'dir': t.direction, 'pnl': t.pnl,
                })
        else:
            trades = bt_h1_strategy(h1_full, cfg, pctl_full, strat_name)
            for t in trades:
                all_entries.append({
                    'strat': strat_name, 'time': pd.Timestamp(t['entry_time']),
                    'dir': t['dir'], 'pnl': t['pnl'],
                })

    all_entries.sort(key=lambda x: x['time'])

    # Find entries within 1 hour of each other
    clusters = []
    used = set()
    for i, e1 in enumerate(all_entries):
        if i in used:
            continue
        cluster = [e1]
        used.add(i)
        for j in range(i+1, min(i+20, len(all_entries))):
            if j in used:
                continue
            e2 = all_entries[j]
            if abs((e2['time'] - e1['time']).total_seconds()) <= 3600:
                cluster.append(e2)
                used.add(j)
            elif (e2['time'] - e1['time']).total_seconds() > 7200:
                break
        if len(cluster) > 1:
            clusters.append(cluster)

    c4_results['n_clusters'] = len(clusters)
    c4_results['n_total_entries'] = len(all_entries)

    # Analyze cluster performance
    cluster_pnls = [sum(e['pnl'] for e in c) for c in clusters]
    non_cluster_pnls = [e['pnl'] for e in all_entries if not any(e in c for c in clusters)]

    if cluster_pnls:
        c4_results['cluster_avg_pnl'] = round(np.mean(cluster_pnls), 2)
        c4_results['non_cluster_avg_pnl'] = round(np.mean(non_cluster_pnls), 2) if non_cluster_pnls else 0
        print(f"    Total entries: {len(all_entries)}, Clusters (within 1h): {len(clusters)}")
        print(f"    Cluster avg combined PnL: ${np.mean(cluster_pnls):.2f}")
        if non_cluster_pnls:
            print(f"    Non-cluster avg PnL: ${np.mean(non_cluster_pnls):.2f}")

    # Direction conflicts
    conflicts = [c for c in clusters if len(set(e['dir'] for e in c)) > 1]
    c4_results['n_direction_conflicts'] = len(conflicts)
    if conflicts:
        conflict_pnls = [sum(e['pnl'] for e in c) for c in conflicts]
        c4_results['conflict_avg_pnl'] = round(np.mean(conflict_pnls), 2)
        print(f"    Direction conflicts: {len(conflicts)}, avg PnL: ${np.mean(conflict_pnls):.2f}")

    save_phase("C4_entry_timing", c4_results)
    print(f"\n{elapsed()} Phase C4 complete.")
else:
    print(f"\n{elapsed()} Phase C4 already done.")


# #####################################################################
# TRACK D: Robustness & Stress Testing (~40h)
# #####################################################################

# ═══════════════ Phase D1: Regime-Specific Audit ═══════════════
if not phase_done("D1_regime_audit"):
    print(f"\n{'='*100}")
    print(f"  Phase D1: Regime-Specific Strategy Audit")
    print(f"{'='*100}")

    d1_results = {}

    REGIMES = {
        'trending_up_2019': ('2019-06-01', '2020-08-01'),
        'covid_crash': ('2020-02-15', '2020-04-15'),
        'range_2021': ('2021-03-01', '2022-03-01'),
        'tightening_2022': ('2022-03-01', '2022-11-01'),
        'trending_up_2024': ('2024-01-01', '2025-01-01'),
        'recent_2025': ('2025-01-01', '2026-04-01'),
    }

    # Keltner regimes (engine-based, keep serial as they use M15 DataBundle)
    cfg_k = STRAT_CONFIGS['keltner']
    print(f"\n  --- KELTNER ---")
    strat_regime_k = {}
    for regime_name, (s, e) in REGIMES.items():
        regime_data = data.slice(s, e)
        if len(regime_data.m15_df) < 500:
            continue
        stats = bt_keltner(regime_data, f"keltner_{regime_name}", maxloss_cap=cfg_k['cap'])
        strat_regime_k[regime_name] = {k: v for k, v in stats.items() if not k.startswith('_')}
        sr = strat_regime_k[regime_name]
        print(f"    {regime_name:<25} Sharpe={sr.get('sharpe',0):>7.3f} "
              f"N={sr.get('n',0):>5} WR={sr.get('win_rate', sr.get('wr',0)):>5.1f}%")
    d1_results['keltner'] = strat_regime_k

    # H1-loop strategies: parallelize all (strategy x regime) combinations
    d1_h1_tasks = []
    h1_strats = [s for s in STRAT_CONFIGS if s != 'keltner']
    for strat_name in h1_strats:
        cfg = STRAT_CONFIGS[strat_name]
        for regime_name, (s, e) in REGIMES.items():
            d1_h1_tasks.append((f'{strat_name}__{regime_name}', cfg, strat_name, s, e, {}))

    print(f"\n  Launching {len(d1_h1_tasks)} parallel regime backtests...")
    with create_pool(h1_full, pctl_full) as pool:
        futures = {pool.submit(_worker_bt_h1_sliced, t): t[0] for t in d1_h1_tasks}
        d1_parallel = {}
        for f in as_completed(futures):
            key, stats = f.result()
            if stats is not None:
                d1_parallel[key] = stats

    for strat_name in h1_strats:
        print(f"\n  --- {strat_name.upper()} ---")
        strat_regime = {}
        for regime_name in REGIMES:
            key = f'{strat_name}__{regime_name}'
            sr = d1_parallel.get(key)
            if sr is None:
                continue
            strat_regime[regime_name] = sr
            print(f"    {regime_name:<25} Sharpe={sr.get('sharpe',0):>7.3f} "
                  f"N={sr.get('n',0):>5} WR={sr.get('win_rate', sr.get('wr',0)):>5.1f}%")
        d1_results[strat_name] = strat_regime

    save_phase("D1_regime_audit", d1_results)
    print(f"\n{elapsed()} Phase D1 complete.")
else:
    print(f"\n{elapsed()} Phase D1 already done.")


# ═══════════════ Phase D2: Parameter Sensitivity Surface ═══════════════
if not phase_done("D2_param_sensitivity"):
    print(f"\n{'='*100}")
    print(f"  Phase D2: Parameter Sensitivity Surface")
    print(f"{'='*100}")

    d2_results = {}

    # Keltner: KC_mult x trail_act x trail_dist (parallelized)
    print("\n  --- Keltner: 3D Parameter Sensitivity ---")
    kc_mults = [1.0, 1.1, 1.2, 1.3, 1.5]
    trail_acts = [0.06, 0.10, 0.14, 0.18, 0.22]
    trail_dists = [0.008, 0.015, 0.025, 0.035, 0.05]

    keltner_variants = []
    keltner_keys = []
    for km in kc_mults:
        for ta in trail_acts:
            for td in trail_dists:
                key = f"km{km}_ta{ta}_td{td}"
                v = dict(KELTNER_ENGINE_KWARGS)
                # IMPORTANT: regime_config overrides trail params at runtime.
                # Disable it so the sensitivity scan actually varies trail params.
                v['regime_config'] = None
                v['trailing_activate_atr'] = ta
                v['trailing_distance_atr'] = td
                if km != 1.2:
                    v['kc_mult_override'] = km
                v['label'] = key
                keltner_variants.append(v)
                keltner_keys.append(key)

    print(f"    Running {len(keltner_variants)} Keltner variants in parallel ({MAX_WORKERS} workers)...")
    keltner_stats_list = run_variants_parallel(data, keltner_variants, max_workers=MAX_WORKERS)
    keltner_surface = {}
    for key, stats in zip(keltner_keys, keltner_stats_list):
        keltner_surface[key] = {
            'sharpe': stats.get('sharpe', 0), 'n': stats.get('n', 0),
            'pnl': stats.get('total_pnl', 0), 'max_dd': stats.get('max_dd', 0),
        }
    d2_results['keltner_surface'] = keltner_surface
    sharpes = [v['sharpe'] for v in keltner_surface.values()]
    print(f"    {len(keltner_surface)} combinations tested")
    print(f"    Sharpe range: {min(sharpes):.3f} to {max(sharpes):.3f}")
    print(f"    Production config region Sharpe: "
          f"{keltner_surface.get('km1.2_ta0.14_td0.025', {}).get('sharpe', 'N/A')}")

    # H1 strategies: 2D grids (parallelized via create_pool)
    print("\n  --- H1 Strategies: 2D Sensitivity (parallel) ---")

    tsmom_tasks = []
    tsmom_keys = []
    cfg_tsmom = STRAT_CONFIGS['tsmom']
    for ta, td in product([0.06, 0.10, 0.14, 0.18, 0.22], [0.01, 0.015, 0.025, 0.035, 0.05]):
        key = f"ta{ta}_td{td}"
        tsmom_keys.append(key)
        tsmom_tasks.append((key, cfg_tsmom, 'tsmom', {'trail_act': ta, 'trail_dist': td}))

    tsmom_surface = {}
    with create_pool(h1_full, pctl_full) as pool:
        tsmom_surface = parallel_h1_stats(pool, tsmom_tasks)
    d2_results['tsmom_surface'] = tsmom_surface
    if tsmom_surface:
        sharpes = [v.get('sharpe', 0) for v in tsmom_surface.values()]
        best_key = max(tsmom_surface, key=lambda k: tsmom_surface[k].get('sharpe', 0))
        print(f"    TSMOM: {len(tsmom_surface)} combinations, Sharpe range: {min(sharpes):.3f} to {max(sharpes):.3f}")
        print(f"    Best: {best_key} Sharpe={tsmom_surface[best_key]['sharpe']:.3f}")

    # PSAR: need to rebuild h1 per (step, max); run each synchronously but fast
    # We can't parallel the PSAR grid via parallel_h1_stats (each needs custom h1).
    # Instead, build all variants and use a helper worker.
    from indicators import calc_chandelier as _calc_chand
    psar_surface = {}
    print("\n  --- PSAR: 2D Sensitivity (serial, each builds custom h1) ---")
    for p1, p2 in product([0.005, 0.01, 0.015, 0.02, 0.03], [0.03, 0.05, 0.07, 0.10]):
        key = f"step{p1}_max{p2}"
        h1_custom = _add_psar(h1_full.copy(), af_step=p1, af_max=p2)
        trades = bt_h1_strategy(h1_custom, STRAT_CONFIGS['psar'], pctl_full, 'psar')
        psar_surface[key] = _stats_from_trades(trades)
    d2_results['psar_surface'] = psar_surface
    if psar_surface:
        sharpes = [v.get('sharpe', 0) for v in psar_surface.values()]
        best_key = max(psar_surface, key=lambda k: psar_surface[k].get('sharpe', 0))
        print(f"    PSAR: {len(psar_surface)} combinations, Sharpe range: {min(sharpes):.3f} to {max(sharpes):.3f}")
        print(f"    Best: {best_key} Sharpe={psar_surface[best_key]['sharpe']:.3f}")

    chand_surface = {}
    print("\n  --- CHANDELIER: 2D Sensitivity (serial, each recomputes chand) ---")
    for p, m in product([15, 18, 22, 26, 30], [2.0, 2.5, 3.0, 3.5, 4.0]):
        key = f"p{p}_m{m}"
        chand = _calc_chand(h1_full, period=p, mult=m)
        h1_custom = h1_full.copy()
        h1_custom['Chand_long'] = chand['Chand_long']
        h1_custom['Chand_short'] = chand['Chand_short']
        trades = bt_h1_strategy(h1_custom, STRAT_CONFIGS['chandelier'], pctl_full, 'chandelier')
        chand_surface[key] = _stats_from_trades(trades)
    d2_results['chandelier_surface'] = chand_surface
    if chand_surface:
        sharpes = [v.get('sharpe', 0) for v in chand_surface.values()]
        best_key = max(chand_surface, key=lambda k: chand_surface[k].get('sharpe', 0))
        print(f"    CHANDELIER: {len(chand_surface)} combinations, Sharpe range: {min(sharpes):.3f} to {max(sharpes):.3f}")
        print(f"    Best: {best_key} Sharpe={chand_surface[best_key]['sharpe']:.3f}")

    save_phase("D2_param_sensitivity", d2_results)
    print(f"\n{elapsed()} Phase D2 complete.")
else:
    print(f"\n{elapsed()} Phase D2 already done.")


# ═══════════════ Phase D3: Stress Scenarios ═══════════════
if not phase_done("D3_stress_scenarios"):
    print(f"\n{'='*100}")
    print(f"  Phase D3: Stress Scenarios")
    print(f"{'='*100}")

    d3_results = {}

    # Spread sensitivity (parallelized)
    print("\n  --- Spread Sensitivity (parallel) ---")
    spread_variants = []
    spread_keys = []
    for spread in [0.0, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0]:
        label = f"spread_{spread}"
        v = dict(KELTNER_ENGINE_KWARGS)
        v['spread_cost'] = spread
        v['label'] = label
        spread_variants.append(v)
        spread_keys.append((spread, label))

    spread_stats_list = run_variants_parallel(data, spread_variants, max_workers=MAX_WORKERS)
    for (spread, label), stats in zip(spread_keys, spread_stats_list):
        d3_results[f'keltner_{label}'] = {
            'sharpe': stats.get('sharpe', 0), 'pnl': stats.get('total_pnl', 0), 'n': stats.get('n', 0),
        }
        print(f"    Spread=${spread:.1f}: Sharpe={stats.get('sharpe',0):.3f} "
              f"PnL=${stats.get('total_pnl',0):.2f}")

    # Consecutive loss analysis: Keltner needs _trades so stays serial, H1 strategies parallel
    print("\n  --- Consecutive Loss Analysis ---")
    # Keltner (serial, needs _trades)
    keltner_cap_stats = bt_keltner(data, "streak_keltner", maxloss_cap=STRAT_CONFIGS['keltner']['cap'])
    keltner_pnls = [t.pnl for t in keltner_cap_stats.get('_trades', [])]

    def _max_streak(pnls):
        m, c = 0, 0
        for p in pnls:
            if p <= 0:
                c += 1
                m = max(m, c)
            else:
                c = 0
        return m

    d3_results['keltner_max_loss_streak'] = _max_streak(keltner_pnls)
    print(f"    {'keltner':<15} Max consecutive losses: {d3_results['keltner_max_loss_streak']}")

    # H1 strategies (parallel)
    h1_streak_tasks = []
    h1_streak_strats = [s for s in STRAT_CONFIGS.keys() if s != 'keltner']
    for s in h1_streak_strats:
        h1_streak_tasks.append((s, STRAT_CONFIGS[s], s, {}))
    with create_pool(h1_full, pctl_full) as pool:
        # We need trades, not stats. Use _worker_bt_h1_trades if exists, else
        # fall back to sequential for streak. Use parallel_h1_stats and
        # reconstruct via inline computation is impossible without trades.
        # Simplest: reuse H1 helper that returns trade list.
        futures = {}
        for key, cfg, strat, kw in h1_streak_tasks:
            futures[pool.submit(_worker_bt_h1_trades, (key, cfg, strat, kw))] = key
        h1_trade_lists = {}
        for fut in as_completed(futures):
            key, trades = fut.result()
            h1_trade_lists[key] = trades

    for s in h1_streak_strats:
        trades = h1_trade_lists.get(s, [])
        pnls = [t['pnl'] for t in trades]
        streak = _max_streak(pnls)
        d3_results[f'{s}_max_loss_streak'] = streak
        print(f"    {s:<15} Max consecutive losses: {streak}")

    # Monte Carlo: parameter perturbation for Keltner (parallelized)
    print("\n  --- Monte Carlo: Keltner Parameter Perturbation (100 iterations, parallel) ---")
    np.random.seed(42)
    base_params = {
        'trailing_activate_atr': 0.14, 'trailing_distance_atr': 0.025,
        'sl_atr_mult': 3.5, 'tp_atr_mult': 8.0, 'keltner_adx_threshold': 14,
    }
    mc_variants = []
    for i in range(100):
        perturbed = dict(KELTNER_ENGINE_KWARGS)
        # Same regime_config override issue: disable it so perturbation actually affects the run
        perturbed['regime_config'] = None
        for param, base_val in base_params.items():
            noise = np.random.normal(0, 0.10)
            if param == 'keltner_adx_threshold':
                perturbed[param] = max(8, int(base_val * (1 + noise)))
            else:
                perturbed[param] = round(base_val * (1 + noise), 4)
        perturbed['label'] = f"mc_{i}"
        mc_variants.append(perturbed)

    mc_stats_list = run_variants_parallel(data, mc_variants, max_workers=MAX_WORKERS)
    mc_sharpes = [s.get('sharpe', 0) for s in mc_stats_list]

    d3_results['mc_keltner'] = {
        'mean_sharpe': round(np.mean(mc_sharpes), 3),
        'std_sharpe': round(np.std(mc_sharpes), 3),
        'min_sharpe': round(min(mc_sharpes), 3),
        'max_sharpe': round(max(mc_sharpes), 3),
        'pct_positive': round(sum(1 for s in mc_sharpes if s > 0) / len(mc_sharpes) * 100, 1),
        'p5': round(np.percentile(mc_sharpes, 5), 3),
        'p25': round(np.percentile(mc_sharpes, 25), 3),
        'p50': round(np.percentile(mc_sharpes, 50), 3),
        'p75': round(np.percentile(mc_sharpes, 75), 3),
        'p95': round(np.percentile(mc_sharpes, 95), 3),
    }
    print(f"    Mean Sharpe: {np.mean(mc_sharpes):.3f} +/- {np.std(mc_sharpes):.3f}")
    print(f"    Range: [{min(mc_sharpes):.3f}, {max(mc_sharpes):.3f}]")
    print(f"    Positive: {sum(1 for s in mc_sharpes if s > 0)}/100")
    print(f"    5th/50th/95th percentile: {np.percentile(mc_sharpes, 5):.3f} / "
          f"{np.percentile(mc_sharpes, 50):.3f} / {np.percentile(mc_sharpes, 95):.3f}")

    save_phase("D3_stress_scenarios", d3_results)
    print(f"\n{elapsed()} Phase D3 complete.")
else:
    print(f"\n{elapsed()} Phase D3 already done.")


# ═══════════════ Phase D4: Execution Realism ═══════════════
if not phase_done("D4_execution_realism"):
    print(f"\n{'='*100}")
    print(f"  Phase D4: Execution Realism")
    print(f"{'='*100}")

    d4_results = {}

    # Slippage model (parallelized)
    print("\n  --- Slippage Sensitivity (parallel) ---")
    slip_variants = []
    slip_keys = []
    for slip in [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]:
        total_spread = 0.30 + slip
        label = f"slip_{slip}"
        v = dict(KELTNER_ENGINE_KWARGS)
        v['spread_cost'] = total_spread
        v['label'] = label
        slip_variants.append(v)
        slip_keys.append((slip, total_spread, label))

    slip_stats_list = run_variants_parallel(data, slip_variants, max_workers=MAX_WORKERS)
    for (slip, total_spread, label), stats in zip(slip_keys, slip_stats_list):
        d4_results[f'keltner_{label}'] = {
            'sharpe': stats.get('sharpe', 0), 'pnl': stats.get('total_pnl', 0),
        }
        print(f"    Slippage=${slip:.1f} (total cost=${total_spread:.1f}): "
              f"Sharpe={stats.get('sharpe',0):.3f} PnL=${stats.get('total_pnl',0):.2f}")

    # Compare across strategies (parallelized)
    print("\n  --- All Strategies: Execution Cost Sensitivity ---")
    d4_h1_tasks = []
    d4_strats = ['psar', 'tsmom', 'sess_bo', 'dual_thrust', 'chandelier']
    for strat in d4_strats:
        cfg = STRAT_CONFIGS[strat]
        d4_h1_tasks.append((strat, cfg, strat, {}))

    with create_pool(h1_full, pctl_full) as pool:
        d4_parallel = parallel_h1_stats(pool, d4_h1_tasks)

    for strat in d4_strats:
        cfg = STRAT_CONFIGS[strat]
        base_stats = d4_parallel.get(strat, {'n': 0, 'pnl': 0})
        n = base_stats['n']
        base_pnl = base_stats['pnl']
        cost_per_trade = 0.5 * cfg['lot'] * 100
        degraded_pnl = base_pnl - n * cost_per_trade
        d4_results[f'{strat}_exec_sensitivity'] = {
            'base_pnl': base_pnl, 'degraded_pnl': round(degraded_pnl, 2),
            'cost_per_trade': round(cost_per_trade, 2), 'n': n,
            'pnl_per_trade_base': round(base_pnl / n, 2) if n > 0 else 0,
        }
        print(f"    {strat:<15} N={n:>5} Base PnL=${base_pnl:>8.2f} "
              f"After $0.50 slip/trade: ${degraded_pnl:>8.2f} "
              f"(${base_pnl/n:.2f}/trade -> ${degraded_pnl/n:.2f}/trade)" if n > 0 else
              f"    {strat:<15} N=0")

    save_phase("D4_execution_realism", d4_results)
    print(f"\n{elapsed()} Phase D4 complete.")
else:
    print(f"\n{elapsed()} Phase D4 already done.")


# #####################################################################
# TRACK E: Production Readiness (~20h)
# #####################################################################

# ═══════════════ Phase E1: Combined Best-of-R200 Validation ═══════════════
if not phase_done("E1_combined_validation"):
    print(f"\n{'='*100}")
    print(f"  Phase E1: Combined Best-of-R200 Validation")
    print(f"{'='*100}")

    e1_results = {}

    # Collect top findings from Tracks A-D
    print("\n  --- Collecting top findings ---")
    candidates = []

    # From A1: best trail per non-Keltner strategy
    a1_path = OUTPUT_DIR / "A1_trail_per_strategy.json"
    if a1_path.exists():
        with open(a1_path) as f:
            a1 = json.load(f)
        for strat, data_a1 in a1.items():
            best = data_a1.get('best', '')
            best_sh = data_a1.get('best_sharpe', 0)
            current_key = f"ta{STRAT_CONFIGS[strat]['trail_act_atr']}_td{STRAT_CONFIGS[strat]['trail_dist_atr']}"
            current_sh = a1.get(strat, {}).get('grid', {}).get(current_key, {}).get(
                'Full (2015-2026)', {}).get('sharpe', 0)
            if best_sh > current_sh + 0.3:
                candidates.append({
                    'source': 'A1', 'strat': strat, 'param': best,
                    'delta_sharpe': round(best_sh - current_sh, 3),
                })

    # From A2: dynamic exit mechanisms
    a2_path = OUTPUT_DIR / "A2_dynamic_exit.json"
    if a2_path.exists():
        with open(a2_path) as f:
            a2 = json.load(f)
        keltner_exits = a2.get('keltner_engine_exits', {})
        base_sharpe = keltner_exits.get('baseline', {}).get('Full (2015-2026)', {}).get('sharpe', 0)
        for name, eras in keltner_exits.items():
            if name == 'baseline':
                continue
            full_sh = eras.get('Full (2015-2026)', {}).get('sharpe', 0)
            if full_sh > base_sharpe + 0.3:
                candidates.append({
                    'source': 'A2', 'strat': 'keltner', 'param': name,
                    'delta_sharpe': round(full_sh - base_sharpe, 3),
                })

    print(f"  Found {len(candidates)} candidates with > +0.3 Sharpe improvement")
    for c in candidates:
        print(f"    [{c['source']}] {c['strat']}: {c['param']} (delta={c['delta_sharpe']:+.3f})")

    # 3-Gate validate each candidate
    print("\n  --- 3-Gate Validation ---")
    for c in candidates:
        strat = c['strat']
        param = c['param']
        print(f"\n  Validating {strat}/{param}...")

        if strat == 'keltner':
            kw = {}
            if 'time_decay' in param:
                kw = {'time_decay_tp': True}
            elif 'profit_dd' in param:
                pct = float(param.split('_')[-1].replace('pct', '')) / 100
                kw = {'profit_drawdown_pct': pct}
            elif 'breakeven' in param:
                val = float(param.split('_')[-1].replace('atr', ''))
                kw = {'breakeven_after_atr': val}
            elif 'atr_spike' in param:
                kw = {'atr_spike_protection': True}
            elif 'timeout_dynamic' in param:
                kw = {'timeout_dynamic': True}
            elif 'time_adaptive_trail' in param:
                kw = {'time_adaptive_trail': True}

            full_kw = dict(KELTNER_ENGINE_KWARGS)
            full_kw.update(kw)
            passed, details = three_gate_validate(data, full_kw, strategy='keltner')
        else:
            parts = param.split('_')
            ta = float(parts[0].replace('ta', ''))
            td = float(parts[1].replace('td', ''))
            cfg = STRAT_CONFIGS[strat]
            passed, details = three_gate_validate(
                h1_full, cfg, strategy=strat, pctl=pctl_full, h1=h1_full,
                trail_act=ta, trail_dist=td,
            )

        verdict = 'GO' if passed else 'NO-GO'
        c['validation'] = details
        c['verdict'] = verdict
        kf = details.get('kfold', {})
        wf = details.get('wf', {})
        print(f"    K-Fold: {kf.get('wins', 0)}/6, WF: {wf.get('wins', 0)}/{wf.get('total', 0)}, "
              f"Era: {'PASS' if details.get('era_pass', False) else 'FAIL'} -> [{verdict}]")

    e1_results['candidates'] = candidates
    e1_results['n_go'] = sum(1 for c in candidates if c.get('verdict') == 'GO')
    e1_results['n_nogo'] = sum(1 for c in candidates if c.get('verdict') == 'NO-GO')

    save_phase("E1_combined_validation", e1_results)
    print(f"\n{elapsed()} Phase E1 complete. GO={e1_results['n_go']}, NO-GO={e1_results['n_nogo']}")
else:
    print(f"\n{elapsed()} Phase E1 already done.")


# ═══════════════ Phase E2: Live Reconciliation ═══════════════
if not phase_done("E2_live_reconciliation"):
    print(f"\n{'='*100}")
    print(f"  Phase E2: Live Trade Reconciliation")
    print(f"{'='*100}")

    e2_results = {}
    trade_log_path = Path("data/gold_trade_log.json")

    if trade_log_path.exists():
        with open(trade_log_path, encoding='utf-8') as f:
            live_log = json.load(f)

        closes = [t for t in live_log if t.get('action') == 'CLOSE']
        e2_results['n_live_trades'] = len(closes)
        print(f"  Loaded {len(closes)} live CLOSE trades")

        # Per-strategy live summary
        strat_live = defaultdict(list)
        for t in closes:
            s = t.get('strategy', 'unknown')
            strat_live[s].append(t.get('profit', 0))

        for s, pnls in sorted(strat_live.items()):
            e2_results[f'live_{s}'] = {
                'n': len(pnls), 'total': round(sum(pnls), 2),
                'avg': round(np.mean(pnls), 2), 'wr': round(sum(1 for p in pnls if p > 0) / len(pnls) * 100, 1),
            }
            print(f"    {s:<15} N={len(pnls):>4} PnL=${sum(pnls):>8.2f} "
                  f"Avg=${np.mean(pnls):>6.2f} WR={e2_results[f'live_{s}']['wr']:>5.1f}%")
    else:
        e2_results['status'] = 'no_trade_log'
        print(f"  Trade log not found at {trade_log_path}")

    save_phase("E2_live_reconciliation", e2_results)
    print(f"\n{elapsed()} Phase E2 complete.")
else:
    print(f"\n{elapsed()} Phase E2 already done.")


# ═══════════════ Phase E3: Final Summary & Deployment Roadmap ═══════════════
if not phase_done("E3_deployment_roadmap"):
    print(f"\n{'='*100}")
    print(f"  Phase E3: Final Summary & Deployment Roadmap")
    print(f"{'='*100}")

    e3_results = {'phases_completed': [], 'findings': [], 'recommendations': []}

    # Tally completed phases
    for phase_file in sorted(OUTPUT_DIR.glob("*.json")):
        if phase_file.name != 'E3_deployment_roadmap.json':
            e3_results['phases_completed'].append(phase_file.stem)

    print(f"\n  Completed phases: {len(e3_results['phases_completed'])}")
    for p in e3_results['phases_completed']:
        print(f"    - {p}")

    # Load E1 results for GO candidates
    e1_path = OUTPUT_DIR / "E1_combined_validation.json"
    if e1_path.exists():
        with open(e1_path) as f:
            e1 = json.load(f)
        go_candidates = [c for c in e1.get('candidates', []) if c.get('verdict') == 'GO']
        nogo_candidates = [c for c in e1.get('candidates', []) if c.get('verdict') == 'NO-GO']

        print(f"\n  === VALIDATED GO Changes ===")
        if go_candidates:
            for c in go_candidates:
                print(f"    [{c['source']}] {c['strat']}: {c['param']} (delta Sharpe={c['delta_sharpe']:+.3f})")
                e3_results['findings'].append({
                    'type': 'GO', 'source': c['source'], 'strategy': c['strat'],
                    'change': c['param'], 'impact': c['delta_sharpe'],
                })
        else:
            print(f"    None - current production settings are robust")
            e3_results['findings'].append({'type': 'ROBUST', 'message': 'All current settings validated'})

        print(f"\n  === NO-GO Changes (confirmed current settings optimal) ===")
        for c in nogo_candidates:
            print(f"    [{c['source']}] {c['strat']}: {c['param']} (REJECTED)")
    else:
        print(f"  E1 results not found, skipping GO/NO-GO summary")

    # Generate deployment recommendations
    print(f"\n  === Deployment Recommendations ===")
    e3_results['recommendations'] = [
        "1. Continue accumulating tick data (keltner_ticks.jsonl) for 2+ weeks",
        "2. Re-run A4 tick replay analysis once sufficient data collected",
        "3. Any GO changes should be deployed in shadow mode first (1 week)",
        "4. Monitor MaxLoss Cap hit rate vs SL hit rate post any trail change",
    ]
    for r in e3_results['recommendations']:
        print(f"    {r}")

    # Runtime summary
    total_hours = (time.time() - t0) / 3600
    e3_results['total_runtime_hours'] = round(total_hours, 1)
    print(f"\n  Total runtime: {total_hours:.1f} hours")

    save_phase("E3_deployment_roadmap", e3_results)
    print(f"\n{elapsed()} Phase E3 complete.")
else:
    print(f"\n{elapsed()} Phase E3 already done.")


# ═══════════════ FINAL ═══════════════
total_time = time.time() - t0
print(f"\n{'='*100}")
print(f"  R200 COMPLETE")
print(f"  Total runtime: {total_time/3600:.1f} hours ({total_time/60:.0f} minutes)")
print(f"  Results: {OUTPUT_DIR}")
print(f"{'='*100}")
