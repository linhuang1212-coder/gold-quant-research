#!/usr/bin/env python3
"""
R127 — Full Parameter Sensitivity Grid
========================================
Comprehensive parameter sweep for all 4 strategies.
~2000 backtests to find robust parameter plateaus vs overfit peaks.

Phase 1: PSAR parameter grid
Phase 2: TSMOM parameter grid
Phase 3: SESS_BO parameter grid
Phase 4: L8_MAX parameter grid (via engine)
Phase 5: Plateau detection and robustness scoring
Phase 6: Top configs bootstrap confidence intervals
"""
import sys, os, time, json, warnings, itertools
import numpy as np
import pandas as pd
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS

OUTPUT_DIR = Path("results/r127_param_sensitivity")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30
UNIT_LOT = 0.01

t0 = time.time()

# ═══════════════════════════════════════════════════════════════
# Parameter grids
# ═══════════════════════════════════════════════════════════════

SL_ATR_GRID = [2.0, 3.0, 4.0, 4.5, 5.0, 6.0]
TP_ATR_GRID = [4.0, 6.0, 8.0, 12.0, 16.0, 20.0]
TRAIL_ACT_GRID = [0.08, 0.12, 0.14, 0.18, 0.22, 0.28]
TRAIL_DIST_GRID = [0.015, 0.025, 0.04, 0.06]
MAX_HOLD_GRID = [10, 15, 20, 25]

L8_SL_GRID = [2.5, 3.0, 3.5, 4.0, 5.0]
L8_TRAIL_ACT_GRID = [0.08, 0.12, 0.14, 0.18, 0.22]
L8_TRAIL_DIST_GRID = [0.015, 0.025, 0.04, 0.06]

# ═══════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════

def compute_atr(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def _mk(pos, exit_px, exit_time, reason, bar_i, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_px,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_i - pos['bar']}

def _run_exit_with_cap(pos, i, hi, lo, cl, spread, lot, pv, times,
                       sl_atr, tp_atr, trail_act, trail_dist, max_hold, cap):
    atr = pos['atr']
    sl = atr * sl_atr; tp = atr * tp_atr; bars = i - pos['bar']
    if pos['dir'] == 'BUY':
        pnl_now = (cl - pos['entry'] - spread) * lot * pv
        if cap > 0 and pnl_now < -cap:
            return _mk(pos, cl, times[i], "MaxLossCap", i, -cap)
        if lo <= pos['entry'] - sl:
            return _mk(pos, pos['entry'] - sl, times[i], "SL", i, -sl * lot * pv)
        if hi >= pos['entry'] + tp:
            return _mk(pos, pos['entry'] + tp, times[i], "TP", i, tp * lot * pv)
        extreme = pos.get('extreme', pos['entry']); extreme = max(extreme, hi); pos['extreme'] = extreme
        if extreme - pos['entry'] >= atr * trail_act:
            trail_price = extreme - atr * trail_dist
            if cl <= trail_price:
                return _mk(pos, trail_price, times[i], "Trail", i, (trail_price - pos['entry'] - spread) * lot * pv)
        if bars >= max_hold:
            return _mk(pos, cl, times[i], "TimeExit", i, pnl_now)
    else:
        pnl_now = (pos['entry'] - cl - spread) * lot * pv
        if cap > 0 and pnl_now < -cap:
            return _mk(pos, cl, times[i], "MaxLossCap", i, -cap)
        if hi >= pos['entry'] + sl:
            return _mk(pos, pos['entry'] + sl, times[i], "SL", i, -sl * lot * pv)
        if lo <= pos['entry'] - tp:
            return _mk(pos, pos['entry'] - tp, times[i], "TP", i, tp * lot * pv)
        extreme = pos.get('extreme', pos['entry']); extreme = min(extreme, lo); pos['extreme'] = extreme
        if pos['entry'] - extreme >= atr * trail_act:
            trail_price = extreme + atr * trail_dist
            if cl >= trail_price:
                return _mk(pos, trail_price, times[i], "Trail", i, (pos['entry'] - trail_price - spread) * lot * pv)
        if bars >= max_hold:
            return _mk(pos, cl, times[i], "TimeExit", i, pnl_now)
    return None


# ═══════════════════════════════════════════════════════════════
# Strategy backtests (parameterized)
# ═══════════════════════════════════════════════════════════════

def add_psar(df, af_step=0.01, af_max=0.05):
    h, l, c = df['High'].values, df['Low'].values, df['Close'].values
    n = len(df)
    psar = np.empty(n); psar[:] = np.nan
    af = af_step; rising = True; ep = h[0]; psar[0] = l[0]
    for i in range(1, n):
        prev = psar[i-1]
        if rising:
            psar[i] = prev + af * (ep - prev)
            psar[i] = min(psar[i], l[i-1], l[max(0, i-2)])
            if l[i] < psar[i]:
                rising = False; psar[i] = ep; ep = l[i]; af = af_step
            else:
                if h[i] > ep: ep = h[i]; af = min(af + af_step, af_max)
        else:
            psar[i] = prev + af * (ep - prev)
            psar[i] = max(psar[i], h[i-1], h[max(0, i-2)])
            if h[i] > psar[i]:
                rising = True; psar[i] = ep; ep = h[i]; af = af_step
            else:
                if l[i] < ep: ep = l[i]; af = min(af + af_step, af_max)
    df['PSAR'] = psar

def bt_psar(h1_df, spread, lot, sl_atr=4.5, tp_atr=16.0, trail_act=0.20,
            trail_dist=0.04, max_hold=20, maxloss_cap=0):
    df = h1_df.copy(); add_psar(df); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR', 'PSAR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; psar = df['PSAR'].values; times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if psar[i-1] > c[i-1] and psar[i] < c[i]:
            pos = {'dir': 'BUY', 'entry': c[i] + spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif psar[i-1] < c[i-1] and psar[i] > c[i]:
            pos = {'dir': 'SELL', 'entry': c[i] - spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades

def bt_tsmom(h1_df, spread, lot, fast=480, slow=720, sl_atr=4.5, tp_atr=6.0,
             trail_act=0.14, trail_dist=0.025, max_hold=20, maxloss_cap=0):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; times = df.index; n = len(df)
    max_lb = max(fast, slow)
    score = np.full(n, np.nan)
    for i in range(max_lb, n):
        s = 0.0
        if c[i-fast] > 0: s += 0.5 * np.sign(c[i] / c[i-fast] - 1.0)
        if c[i-slow] > 0: s += 0.5 * np.sign(c[i] / c[i-slow] - 1.0)
        score[i] = s
    trades = []; pos = None; last_exit = -999
    for i in range(max_lb + 1, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            if pos['dir'] == 'BUY' and score[i] < 0:
                pnl = (c[i] - pos['entry'] - spread) * lot * PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            elif pos['dir'] == 'SELL' and score[i] > 0:
                pnl = (pos['entry'] - c[i] - spread) * lot * PV
                trades.append(_mk(pos, c[i], times[i], "Reversal", i, pnl)); pos = None; last_exit = i; continue
            continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        if np.isnan(score[i]) or np.isnan(score[i-1]): continue
        if score[i] > 0 and score[i-1] <= 0:
            pos = {'dir': 'BUY', 'entry': c[i] + spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif score[i] < 0 and score[i-1] >= 0:
            pos = {'dir': 'SELL', 'entry': c[i] - spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades

def bt_sess_bo(h1_df, spread, lot, session_hour=12, lookback=4, sl_atr=4.5,
               tp_atr=4.0, trail_act=0.14, trail_dist=0.025, max_hold=20, maxloss_cap=0):
    df = h1_df.copy(); df['ATR'] = compute_atr(df)
    df = df.dropna(subset=['ATR'])
    c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
    atr = df['ATR'].values; hours = df.index.hour; times = df.index; n = len(df)
    trades = []; pos = None; last_exit = -999
    for i in range(lookback, n):
        if pos is not None:
            result = _run_exit_with_cap(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                                        sl_atr, tp_atr, trail_act, trail_dist, max_hold, maxloss_cap)
            if result:
                trades.append(result); pos = None; last_exit = i; continue
            continue
        if hours[i] != session_hour: continue
        if i - last_exit < 2: continue
        if np.isnan(atr[i]) or atr[i] < 0.1: continue
        hh = max(h[i-j] for j in range(1, lookback + 1))
        ll = min(lo[i-j] for j in range(1, lookback + 1))
        if c[i] > hh:
            pos = {'dir': 'BUY', 'entry': c[i] + spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
        elif c[i] < ll:
            pos = {'dir': 'SELL', 'entry': c[i] - spread/2, 'bar': i, 'time': times[i], 'atr': atr[i]}
    return trades


# ═══════════════════════════════════════════════════════════════
# Metric helpers
# ═══════════════════════════════════════════════════════════════

def trades_to_daily_pnl(trades):
    if not trades:
        return np.array([])
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    dates = sorted(daily.keys())
    return np.array([daily[d] for d in dates])

def sharpe(arr):
    if len(arr) < 10:
        return 0.0
    s = np.std(arr, ddof=1)
    return float(np.mean(arr) / s * np.sqrt(252)) if s > 0 else 0.0

def max_dd(arr):
    if len(arr) == 0:
        return 0.0
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max())

def metrics_from_trades(trades):
    pnls = [t['pnl'] for t in trades]
    daily = trades_to_daily_pnl(trades)
    total = sum(pnls)
    wins = [p for p in pnls if p > 0]
    wr = len(wins) / len(pnls) if pnls else 0
    return {
        'n_trades': len(trades),
        'total_pnl': round(total, 2),
        'sharpe': round(sharpe(daily), 3),
        'max_dd': round(max_dd(daily), 2),
        'win_rate': round(wr, 3),
        'avg_pnl': round(total / len(pnls), 2) if pnls else 0,
    }


# ═══════════════════════════════════════════════════════════════
# Tiered sweep logic
# ═══════════════════════════════════════════════════════════════

def run_tiered_sweep(strat_name, bt_func, h1_df, defaults):
    """Three-pass tiered sweep: sl/tp → trail → max_hold."""
    all_results = []

    # Pass 1: sl_atr x tp_atr
    print(f"  Pass 1: sl_atr x tp_atr (36 configs)...", flush=True)
    pass1 = []
    for sl, tp in itertools.product(SL_ATR_GRID, TP_ATR_GRID):
        params = {**defaults, 'sl_atr': sl, 'tp_atr': tp}
        trades = bt_func(h1_df, SPREAD, UNIT_LOT, **params)
        m = metrics_from_trades(trades)
        rec = {'params': params, **m}
        pass1.append(rec)
        all_results.append(rec)
    pass1.sort(key=lambda x: x['sharpe'], reverse=True)
    top5_sltp = pass1[:5]
    print(f"    Top 5 sl/tp Sharpe: {[r['sharpe'] for r in top5_sltp]}", flush=True)

    # Pass 2: for top 5, sweep trail_act x trail_dist
    print(f"  Pass 2: trail sweep for top 5 (120 configs)...", flush=True)
    pass2 = []
    for base in top5_sltp:
        for ta, td in itertools.product(TRAIL_ACT_GRID, TRAIL_DIST_GRID):
            params = {**base['params'], 'trail_act': ta, 'trail_dist': td}
            trades = bt_func(h1_df, SPREAD, UNIT_LOT, **params)
            m = metrics_from_trades(trades)
            rec = {'params': params, **m}
            pass2.append(rec)
            all_results.append(rec)
    pass2.sort(key=lambda x: x['sharpe'], reverse=True)
    top3_trail = pass2[:3]
    print(f"    Top 3 trail Sharpe: {[r['sharpe'] for r in top3_trail]}", flush=True)

    # Pass 3: for top 3, sweep max_hold
    print(f"  Pass 3: max_hold sweep for top 3 (12 configs)...", flush=True)
    pass3 = []
    for base in top3_trail:
        for mh in MAX_HOLD_GRID:
            params = {**base['params'], 'max_hold': mh}
            trades = bt_func(h1_df, SPREAD, UNIT_LOT, **params)
            m = metrics_from_trades(trades)
            rec = {'params': params, **m}
            pass3.append(rec)
            all_results.append(rec)

    all_results.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"    Best overall Sharpe: {all_results[0]['sharpe']} "
          f"(n_trades={all_results[0]['n_trades']})", flush=True)

    return all_results


def run_l8_sweep(data_bundle):
    """Tiered sweep for L8_MAX via engine."""
    all_results = []

    # Pass 1: sl_atr_mult x trailing_activate_atr (5x5=25)
    print("  Pass 1: sl_atr_mult x trail_act (25 configs)...", flush=True)
    pass1 = []
    for sl, ta in itertools.product(L8_SL_GRID, L8_TRAIL_ACT_GRID):
        kw = {**LIVE_PARITY_KWARGS,
              'sl_atr_mult': sl,
              'trailing_activate_atr': ta,
              'spread_cost': SPREAD,
              'initial_capital': 2000,
              'min_lot_size': UNIT_LOT,
              'max_lot_size': UNIT_LOT}
        try:
            result = run_variant(data_bundle, "L8_MAX", verbose=False, **kw)
            raw_trades = result.get('_trades', [])
            trades = [{'pnl': t.pnl, 'exit_time': t.exit_time,
                        'dir': t.direction, 'entry': t.entry_price,
                        'exit': t.exit_price, 'reason': t.exit_reason,
                        'entry_time': t.entry_time, 'bars': 0}
                       for t in raw_trades]
        except Exception as e:
            print(f"    WARN: sl={sl} ta={ta} failed: {e}", flush=True)
            trades = []
        m = metrics_from_trades(trades)
        rec = {'params': {'sl_atr_mult': sl, 'trailing_activate_atr': ta}, **m}
        pass1.append(rec)
        all_results.append(rec)
    pass1.sort(key=lambda x: x['sharpe'], reverse=True)
    top5 = pass1[:5]
    print(f"    Top 5 Sharpe: {[r['sharpe'] for r in top5]}", flush=True)

    # Pass 2: for top 5, sweep trailing_distance_atr (5x4=20)
    print("  Pass 2: trail_dist sweep for top 5 (20 configs)...", flush=True)
    pass2 = []
    for base in top5:
        for td in L8_TRAIL_DIST_GRID:
            kw = {**LIVE_PARITY_KWARGS,
                  'sl_atr_mult': base['params']['sl_atr_mult'],
                  'trailing_activate_atr': base['params']['trailing_activate_atr'],
                  'trailing_distance_atr': td,
                  'spread_cost': SPREAD,
                  'initial_capital': 2000,
                  'min_lot_size': UNIT_LOT,
                  'max_lot_size': UNIT_LOT}
            try:
                result = run_variant(data_bundle, "L8_MAX", verbose=False, **kw)
                raw_trades = result.get('_trades', [])
                trades = [{'pnl': t.pnl, 'exit_time': t.exit_time,
                            'dir': t.direction, 'entry': t.entry_price,
                            'exit': t.exit_price, 'reason': t.exit_reason,
                            'entry_time': t.entry_time, 'bars': 0}
                           for t in raw_trades]
            except Exception as e:
                print(f"    WARN: td={td} failed: {e}", flush=True)
                trades = []
            m = metrics_from_trades(trades)
            params = {**base['params'], 'trailing_distance_atr': td}
            rec = {'params': params, **m}
            pass2.append(rec)
            all_results.append(rec)

    all_results.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"    Best overall Sharpe: {all_results[0]['sharpe']}", flush=True)
    return all_results


# ═══════════════════════════════════════════════════════════════
# Phase 5: Plateau detection
# ═══════════════════════════════════════════════════════════════

def compute_plateau_scores(results, param_keys):
    """
    For each config, compare its Sharpe to the mean Sharpe of ±1-step neighbors.
    Plateau = Sharpe ≈ neighbor mean (robust). Peak = Sharpe >> neighbor mean (overfit).
    Returns results with 'plateau_score' added (lower = more robust).
    """
    grid_vals = {}
    for k in param_keys:
        vals = sorted(set(r['params'].get(k, 0) for r in results))
        grid_vals[k] = vals

    def neighbors(params):
        nbrs = []
        for r in results:
            if r['params'] == params:
                continue
            diffs = 0
            for k in param_keys:
                v1 = params.get(k, 0)
                v2 = r['params'].get(k, 0)
                vals = grid_vals.get(k, [])
                if v1 == v2:
                    continue
                i1 = vals.index(v1) if v1 in vals else -1
                i2 = vals.index(v2) if v2 in vals else -1
                if abs(i1 - i2) == 1:
                    diffs += 1
                else:
                    diffs = 999
                    break
            if 0 < diffs <= 2:
                nbrs.append(r['sharpe'])
        return nbrs

    for r in results:
        nbr_sharpes = neighbors(r['params'])
        if len(nbr_sharpes) >= 2:
            nbr_mean = np.mean(nbr_sharpes)
            r['plateau_score'] = round(abs(r['sharpe'] - nbr_mean), 4)
            r['neighbor_mean_sharpe'] = round(nbr_mean, 3)
            r['n_neighbors'] = len(nbr_sharpes)
        else:
            r['plateau_score'] = None
            r['neighbor_mean_sharpe'] = None
            r['n_neighbors'] = 0

    return results


# ═══════════════════════════════════════════════════════════════
# Phase 6: Bootstrap
# ═══════════════════════════════════════════════════════════════

def bootstrap_sharpe(bt_func, h1_df, params, n_resamples=100, data_bundle=None, is_l8=False):
    """Resample daily PnL 100 times, return 5th/50th/95th Sharpe."""
    if is_l8 and data_bundle is not None:
        kw = {**LIVE_PARITY_KWARGS, **params,
              'spread_cost': SPREAD, 'initial_capital': 2000,
              'min_lot_size': UNIT_LOT, 'max_lot_size': UNIT_LOT}
        try:
            result = run_variant(data_bundle, "L8_MAX", verbose=False, **kw)
            raw_trades = result.get('_trades', [])
            trades = [{'pnl': t.pnl, 'exit_time': t.exit_time} for t in raw_trades]
        except Exception:
            return {'p5': 0, 'p50': 0, 'p95': 0}
    else:
        trades = bt_func(h1_df, SPREAD, UNIT_LOT, **params)

    daily = trades_to_daily_pnl(trades)
    if len(daily) < 20:
        return {'p5': 0, 'p50': 0, 'p95': 0}

    rng = np.random.RandomState(42)
    sharpes = []
    for _ in range(n_resamples):
        sample = rng.choice(daily, size=len(daily), replace=True)
        sharpes.append(sharpe(sample))

    return {
        'p5': round(float(np.percentile(sharpes, 5)), 3),
        'p50': round(float(np.percentile(sharpes, 50)), 3),
        'p95': round(float(np.percentile(sharpes, 95)), 3),
    }


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    results = {'experiment': 'R127_param_sensitivity', 'strategies': {}}

    # Load data
    print("=" * 60, flush=True)
    print("R127 — Full Parameter Sensitivity Grid", flush=True)
    print("=" * 60, flush=True)

    from backtest.runner import load_csv
    # H1 spot XAUUSD bid (HistData) only — do NOT silently fall back to
    # data/xauusd_h1_yf.csv (= GC=F futures), the schema and price level differ.
    h1_candidates = [
        Path("data/download/xauusd-h1-bid-2015-01-01-2026-05-06.csv"),
        Path("data/download/xauusd-h1-bid-2015-01-01-2026-05-05.csv"),
        Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-27.csv"),
        Path("data/download/xauusd-h1-bid-2015-01-01-2026-04-10.csv"),
        Path("data/download/xauusd-h1-bid-2015-01-01-2026-03-25.csv"),
    ]
    h1_path = next((p for p in h1_candidates if p.exists()), None)
    if h1_path is None:
        print(f"ERROR: No HistData H1 bid CSV found in {h1_candidates[0].parent}", flush=True)
        return
    h1_df = load_csv(str(h1_path))
    h1_df.sort_index(inplace=True)
    print(f"H1 data: {len(h1_df)} bars ({h1_df.index[0]} ~ {h1_df.index[-1]})", flush=True)

    # Load DataBundle for L8_MAX
    data_bundle = None
    try:
        data_bundle = DataBundle.load_default()
        print(f"DataBundle loaded for L8_MAX", flush=True)
    except Exception as e:
        print(f"WARN: DataBundle load failed ({e}), skipping L8_MAX", flush=True)

    # ── Phase 1: PSAR ─────────────────────────────────────────
    print(f"\n{'─'*50}", flush=True)
    print("Phase 1: PSAR parameter grid", flush=True)
    psar_defaults = {'sl_atr': 4.5, 'tp_atr': 16.0, 'trail_act': 0.20,
                     'trail_dist': 0.04, 'max_hold': 20, 'maxloss_cap': 0}
    psar_results = run_tiered_sweep("PSAR", bt_psar, h1_df, psar_defaults)
    results['strategies']['PSAR'] = {
        'total_configs': len(psar_results),
        'top20': psar_results[:20],
    }
    print(f"  PSAR: {len(psar_results)} configs tested", flush=True)

    # ── Phase 2: TSMOM ────────────────────────────────────────
    print(f"\n{'─'*50}", flush=True)
    print("Phase 2: TSMOM parameter grid", flush=True)
    tsmom_defaults = {'fast': 480, 'slow': 720, 'sl_atr': 4.5, 'tp_atr': 6.0,
                      'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 20, 'maxloss_cap': 0}
    tsmom_results = run_tiered_sweep("TSMOM", bt_tsmom, h1_df, tsmom_defaults)
    results['strategies']['TSMOM'] = {
        'total_configs': len(tsmom_results),
        'top20': tsmom_results[:20],
    }
    print(f"  TSMOM: {len(tsmom_results)} configs tested", flush=True)

    # ── Phase 3: SESS_BO ──────────────────────────────────────
    print(f"\n{'─'*50}", flush=True)
    print("Phase 3: SESS_BO parameter grid", flush=True)
    sessbo_defaults = {'session_hour': 12, 'lookback': 4, 'sl_atr': 4.5,
                       'tp_atr': 4.0, 'trail_act': 0.14, 'trail_dist': 0.025,
                       'max_hold': 20, 'maxloss_cap': 0}
    sessbo_results = run_tiered_sweep("SESS_BO", bt_sess_bo, h1_df, sessbo_defaults)
    results['strategies']['SESS_BO'] = {
        'total_configs': len(sessbo_results),
        'top20': sessbo_results[:20],
    }
    print(f"  SESS_BO: {len(sessbo_results)} configs tested", flush=True)

    # ── Phase 4: L8_MAX ───────────────────────────────────────
    print(f"\n{'─'*50}", flush=True)
    print("Phase 4: L8_MAX parameter grid (via engine)", flush=True)
    if data_bundle is not None:
        l8_results = run_l8_sweep(data_bundle)
        results['strategies']['L8_MAX'] = {
            'total_configs': len(l8_results),
            'top20': l8_results[:20],
        }
        print(f"  L8_MAX: {len(l8_results)} configs tested", flush=True)
    else:
        l8_results = []
        results['strategies']['L8_MAX'] = {'skipped': True, 'reason': 'DataBundle unavailable'}
        print("  L8_MAX: SKIPPED (no DataBundle)", flush=True)

    # ── Phase 5: Plateau detection ────────────────────────────
    print(f"\n{'─'*50}", flush=True)
    print("Phase 5: Plateau detection & robustness scoring", flush=True)

    h1_param_keys = ['sl_atr', 'tp_atr', 'trail_act', 'trail_dist', 'max_hold']
    l8_param_keys = ['sl_atr_mult', 'trailing_activate_atr', 'trailing_distance_atr']

    for name, res_list, pkeys in [
        ('PSAR', psar_results, h1_param_keys),
        ('TSMOM', tsmom_results, h1_param_keys),
        ('SESS_BO', sessbo_results, h1_param_keys),
        ('L8_MAX', l8_results, l8_param_keys),
    ]:
        if not res_list:
            continue
        compute_plateau_scores(res_list, pkeys)
        scored = [r for r in res_list if r.get('plateau_score') is not None]
        scored.sort(key=lambda x: x['plateau_score'])
        plateau_top = scored[:10]
        peak_top = sorted(scored, key=lambda x: x['plateau_score'], reverse=True)[:5]

        results['strategies'][name]['plateau_top10'] = plateau_top
        results['strategies'][name]['peak_top5'] = peak_top

        print(f"  {name}: {len(scored)} configs scored", flush=True)
        if plateau_top:
            print(f"    Most robust (lowest plateau_score): "
                  f"Sharpe={plateau_top[0]['sharpe']}, score={plateau_top[0]['plateau_score']}", flush=True)
        if peak_top:
            print(f"    Likely overfit (highest plateau_score): "
                  f"Sharpe={peak_top[0]['sharpe']}, score={peak_top[0]['plateau_score']}", flush=True)

    # ── Phase 6: Bootstrap confidence intervals ───────────────
    print(f"\n{'─'*50}", flush=True)
    print("Phase 6: Bootstrap confidence intervals (100 resamples)", flush=True)

    for name, res_list, bt_func, is_l8 in [
        ('PSAR', psar_results, bt_psar, False),
        ('TSMOM', tsmom_results, bt_tsmom, False),
        ('SESS_BO', sessbo_results, bt_sess_bo, False),
        ('L8_MAX', l8_results, None, True),
    ]:
        if not res_list:
            continue
        top5 = res_list[:5]
        bootstrap_results = []
        for idx, cfg in enumerate(top5):
            ci = bootstrap_sharpe(bt_func, h1_df, cfg['params'],
                                  data_bundle=data_bundle, is_l8=is_l8)
            bootstrap_results.append({
                'rank': idx + 1,
                'params': cfg['params'],
                'sharpe': cfg['sharpe'],
                'bootstrap_ci': ci,
            })
            print(f"  {name} #{idx+1}: Sharpe={cfg['sharpe']} "
                  f"CI=[{ci['p5']}, {ci['p50']}, {ci['p95']}]", flush=True)
        results['strategies'][name]['bootstrap_top5'] = bootstrap_results

    # ── Save ──────────────────────────────────────────────────
    elapsed = time.time() - t0
    results['elapsed_sec'] = round(elapsed, 1)
    results['total_backtests'] = sum(
        results['strategies'].get(s, {}).get('total_configs', 0)
        for s in ['PSAR', 'TSMOM', 'SESS_BO', 'L8_MAX']
    )

    out_path = OUTPUT_DIR / "r127_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n{'='*60}", flush=True)
    print(f"R127 complete — {results['total_backtests']} backtests in {elapsed:.0f}s", flush=True)
    print(f"Saved: {out_path}", flush=True)


if __name__ == '__main__':
    main()
