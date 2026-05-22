#!/usr/bin/env python3
"""
R128 — Regime-Conditional Parameter Optimization
==================================================
Find optimal parameters per market regime, then simulate
regime-switching parameter selection vs static defaults.

Phase 1: Assign regime labels (bull/bear/normal) to each trading day
Phase 2: Per-regime parameter optimization (sl_atr x tp_atr grid)
Phase 3: Regime-switching simulation vs single-param vs default
Phase 4: K-Fold (5 folds) cross-validation of regime-adaptive approach
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

OUTPUT_DIR = Path("results/r128_regime_params")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30
UNIT_LOT = 0.01

SL_ATR_GRID = [2.0, 3.0, 4.0, 4.5, 5.0, 6.0]
TP_ATR_GRID = [4.0, 6.0, 8.0, 12.0, 16.0, 20.0]

t0 = time.time()


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
# Regime labeling
# ═══════════════════════════════════════════════════════════════

def load_regime_labels(h1_df):
    """
    Load regime labels from R121 results if available,
    otherwise compute simple ATR-percentile-based regime inline.
    Returns dict: date → regime ('bull'/'bear'/'normal').
    """
    r121_path = Path("results/r121_regime_detection/r121_results.json")
    if r121_path.exists():
        print("  Loading regime labels from R121...", flush=True)
        try:
            with open(r121_path) as f:
                r121 = json.load(f)
            labels = {}
            for entry in r121.get('regime_labels', r121.get('daily_regimes', [])):
                d = entry.get('date', entry.get('Date'))
                r = entry.get('regime', entry.get('label', 'normal'))
                if d:
                    labels[pd.Timestamp(d).date()] = r.lower()
            if labels:
                print(f"    Loaded {len(labels)} regime labels from R121", flush=True)
                return labels
        except Exception as e:
            print(f"    WARN: R121 parse failed ({e}), computing inline", flush=True)

    print("  Computing inline ATR-percentile regime labels...", flush=True)
    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
    daily_atr = df.groupby(df.index.date)['ATR'].mean().dropna()

    rolling_pct = daily_atr.rolling(60, min_periods=20).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5,
        raw=False
    )

    labels = {}
    for d, pct in rolling_pct.items():
        if np.isnan(pct):
            labels[d] = 'normal'
        elif pct < 0.3:
            labels[d] = 'bull'
        elif pct > 0.7:
            labels[d] = 'bear'
        else:
            labels[d] = 'normal'

    regime_counts = {}
    for r in labels.values():
        regime_counts[r] = regime_counts.get(r, 0) + 1
    print(f"    Regime distribution: {regime_counts}", flush=True)
    return labels


def filter_trades_by_regime(trades, regime_labels, target_regime):
    """Keep only trades whose entry date matches the target regime."""
    filtered = []
    for t in trades:
        d = pd.Timestamp(t['entry_time']).date()
        if regime_labels.get(d, 'normal') == target_regime:
            filtered.append(t)
    return filtered


# ═══════════════════════════════════════════════════════════════
# Phase 2: Per-regime parameter grid
# ═══════════════════════════════════════════════════════════════

STRAT_DEFAULTS = {
    'PSAR': {'sl_atr': 4.5, 'tp_atr': 16.0, 'trail_act': 0.20, 'trail_dist': 0.04,
             'max_hold': 20, 'maxloss_cap': 0},
    'TSMOM': {'fast': 480, 'slow': 720, 'sl_atr': 4.5, 'tp_atr': 6.0,
              'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 20, 'maxloss_cap': 0},
    'SESS_BO': {'session_hour': 12, 'lookback': 4, 'sl_atr': 4.5, 'tp_atr': 4.0,
                'trail_act': 0.14, 'trail_dist': 0.025, 'max_hold': 20, 'maxloss_cap': 0},
}

STRAT_FUNCS = {
    'PSAR': bt_psar,
    'TSMOM': bt_tsmom,
    'SESS_BO': bt_sess_bo,
}

REGIMES = ['bull', 'normal', 'bear']


def find_best_params_per_regime(strat_name, bt_func, h1_df, defaults, regime_labels):
    """Sweep sl_atr x tp_atr (36 configs), evaluate on regime-filtered trades."""
    best_per_regime = {}
    for regime in REGIMES:
        best_sharpe = -999
        best_params = None
        best_metrics = None
        for sl, tp in itertools.product(SL_ATR_GRID, TP_ATR_GRID):
            params = {**defaults, 'sl_atr': sl, 'tp_atr': tp}
            all_trades = bt_func(h1_df, SPREAD, UNIT_LOT, **params)
            regime_trades = filter_trades_by_regime(all_trades, regime_labels, regime)
            m = metrics_from_trades(regime_trades)
            if m['sharpe'] > best_sharpe and m['n_trades'] >= 5:
                best_sharpe = m['sharpe']
                best_params = {'sl_atr': sl, 'tp_atr': tp}
                best_metrics = m
        best_per_regime[regime] = {
            'best_params': best_params if best_params else {'sl_atr': defaults['sl_atr'], 'tp_atr': defaults['tp_atr']},
            'metrics': best_metrics if best_metrics else {'n_trades': 0, 'sharpe': 0, 'total_pnl': 0},
        }
    return best_per_regime


# ═══════════════════════════════════════════════════════════════
# Phase 3: Regime-switching simulation
# ═══════════════════════════════════════════════════════════════

def simulate_regime_switching(strat_name, bt_func, h1_df, defaults,
                              regime_labels, regime_params):
    """
    Compare three approaches:
      1. Default params (static defaults)
      2. Best overall params (best single sl/tp)
      3. Regime-adaptive (switch params by regime label)
    """
    # 1) Default
    default_trades = bt_func(h1_df, SPREAD, UNIT_LOT, **defaults)
    default_m = metrics_from_trades(default_trades)

    # 2) Best overall (sweep once, pick best)
    best_sharpe = -999
    best_overall_params = None
    for sl, tp in itertools.product(SL_ATR_GRID, TP_ATR_GRID):
        params = {**defaults, 'sl_atr': sl, 'tp_atr': tp}
        trades = bt_func(h1_df, SPREAD, UNIT_LOT, **params)
        m = metrics_from_trades(trades)
        if m['sharpe'] > best_sharpe:
            best_sharpe = m['sharpe']
            best_overall_params = {'sl_atr': sl, 'tp_atr': tp}
    best_overall_trades = bt_func(h1_df, SPREAD, UNIT_LOT,
                                  **{**defaults, **best_overall_params})
    best_overall_m = metrics_from_trades(best_overall_trades)

    # 3) Regime-adaptive: run with each regime's best params, merge trades
    adaptive_trades = []
    for regime in REGIMES:
        rp = regime_params[regime]['best_params']
        params = {**defaults, **rp}
        all_trades = bt_func(h1_df, SPREAD, UNIT_LOT, **params)
        regime_trades = filter_trades_by_regime(all_trades, regime_labels, regime)
        adaptive_trades.extend(regime_trades)
    adaptive_trades.sort(key=lambda t: t['entry_time'])
    adaptive_m = metrics_from_trades(adaptive_trades)

    return {
        'default': {'params': {'sl_atr': defaults['sl_atr'], 'tp_atr': defaults['tp_atr']}, **default_m},
        'best_overall': {'params': best_overall_params, **best_overall_m},
        'regime_adaptive': {'regime_params': {r: regime_params[r]['best_params'] for r in REGIMES}, **adaptive_m},
    }


# ═══════════════════════════════════════════════════════════════
# Phase 4: K-Fold cross-validation
# ═══════════════════════════════════════════════════════════════

def kfold_regime_adaptive(strat_name, bt_func, h1_df, defaults, n_folds=5):
    """
    Time-series K-Fold: train regime params on train set, evaluate on test set.
    """
    dates = sorted(set(h1_df.index.date))
    n = len(dates)
    fold_size = n // n_folds
    fold_results = []

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = min((fold + 1) * fold_size, n)
        test_dates = set(dates[test_start:test_end])
        train_dates = set(dates[:test_start]) | set(dates[test_end:])

        if len(train_dates) < 60 or len(test_dates) < 20:
            continue

        train_mask = h1_df.index.map(lambda x: x.date() in train_dates)
        test_mask = h1_df.index.map(lambda x: x.date() in test_dates)
        train_df = h1_df[train_mask].copy()
        test_df = h1_df[test_mask].copy()

        if len(train_df) < 200 or len(test_df) < 50:
            continue

        # Compute regime labels on train set
        train_regimes = load_regime_labels_from_df(train_df)

        # Find best params per regime on train set
        regime_params = find_best_params_per_regime(
            strat_name, bt_func, train_df, defaults, train_regimes)

        # Compute regime labels on test set
        test_regimes = load_regime_labels_from_df(test_df)

        # Evaluate on test set: default vs best-overall vs regime-adaptive
        default_trades = bt_func(test_df, SPREAD, UNIT_LOT, **defaults)
        default_m = metrics_from_trades(default_trades)

        # Best overall from train
        best_sharpe = -999
        best_params = None
        for sl, tp in itertools.product(SL_ATR_GRID, TP_ATR_GRID):
            params = {**defaults, 'sl_atr': sl, 'tp_atr': tp}
            trades = bt_func(train_df, SPREAD, UNIT_LOT, **params)
            m = metrics_from_trades(trades)
            if m['sharpe'] > best_sharpe:
                best_sharpe = m['sharpe']
                best_params = {'sl_atr': sl, 'tp_atr': tp}
        best_oos_trades = bt_func(test_df, SPREAD, UNIT_LOT,
                                  **{**defaults, **best_params})
        best_oos_m = metrics_from_trades(best_oos_trades)

        # Regime-adaptive on test
        adaptive_trades = []
        for regime in REGIMES:
            rp = regime_params[regime]['best_params']
            params = {**defaults, **rp}
            all_trades = bt_func(test_df, SPREAD, UNIT_LOT, **params)
            rt = filter_trades_by_regime(all_trades, test_regimes, regime)
            adaptive_trades.extend(rt)
        adaptive_trades.sort(key=lambda t: t['entry_time'])
        adaptive_m = metrics_from_trades(adaptive_trades)

        fold_results.append({
            'fold': fold + 1,
            'train_days': len(train_dates),
            'test_days': len(test_dates),
            'default_sharpe': default_m['sharpe'],
            'best_overall_sharpe': best_oos_m['sharpe'],
            'regime_adaptive_sharpe': adaptive_m['sharpe'],
            'regime_params': {r: regime_params[r]['best_params'] for r in REGIMES},
        })
        print(f"    Fold {fold+1}: default={default_m['sharpe']:.3f} "
              f"best_overall={best_oos_m['sharpe']:.3f} "
              f"adaptive={adaptive_m['sharpe']:.3f}", flush=True)

    return fold_results


def load_regime_labels_from_df(df):
    """Compute regime labels from a DataFrame's ATR percentile."""
    df = df.copy()
    df['ATR'] = compute_atr(df)
    daily_atr = df.groupby(df.index.date)['ATR'].mean().dropna()

    rolling_pct = daily_atr.rolling(60, min_periods=20).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5,
        raw=False
    )

    labels = {}
    for d, pct in rolling_pct.items():
        if np.isnan(pct):
            labels[d] = 'normal'
        elif pct < 0.3:
            labels[d] = 'bull'
        elif pct > 0.7:
            labels[d] = 'bear'
        else:
            labels[d] = 'normal'
    return labels


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    results = {'experiment': 'R128_regime_params', 'strategies': {}}

    print("=" * 60, flush=True)
    print("R128 — Regime-Conditional Parameter Optimization", flush=True)
    print("=" * 60, flush=True)

    # Load data — H1 spot XAUUSD bid only, no GC=F futures fallback.
    from backtest.runner import load_csv
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

    # ── Phase 1: Regime labels ────────────────────────────────
    print(f"\n{'─'*50}", flush=True)
    print("Phase 1: Assign regime labels", flush=True)
    regime_labels = load_regime_labels(h1_df)
    regime_counts = {}
    for r in regime_labels.values():
        regime_counts[r] = regime_counts.get(r, 0) + 1
    results['regime_summary'] = regime_counts
    print(f"  Total days labeled: {len(regime_labels)}", flush=True)

    # ── Phase 2: Per-regime optimization ──────────────────────
    print(f"\n{'─'*50}", flush=True)
    print("Phase 2: Per-regime parameter optimization (36 configs each)", flush=True)

    for strat_name in ['PSAR', 'TSMOM', 'SESS_BO']:
        print(f"\n  {strat_name}:", flush=True)
        bt_func = STRAT_FUNCS[strat_name]
        defaults = STRAT_DEFAULTS[strat_name]
        regime_params = find_best_params_per_regime(
            strat_name, bt_func, h1_df, defaults, regime_labels)

        for regime in REGIMES:
            bp = regime_params[regime]['best_params']
            bm = regime_params[regime]['metrics']
            print(f"    {regime}: sl={bp['sl_atr']} tp={bp['tp_atr']} "
                  f"→ Sharpe={bm['sharpe']} ({bm['n_trades']} trades)", flush=True)

        results['strategies'][strat_name] = {
            'regime_params': regime_params,
        }

    # ── Phase 3: Regime-switching simulation ──────────────────
    print(f"\n{'─'*50}", flush=True)
    print("Phase 3: Regime-switching simulation", flush=True)

    for strat_name in ['PSAR', 'TSMOM', 'SESS_BO']:
        print(f"\n  {strat_name}:", flush=True)
        bt_func = STRAT_FUNCS[strat_name]
        defaults = STRAT_DEFAULTS[strat_name]
        regime_params = results['strategies'][strat_name]['regime_params']

        comparison = simulate_regime_switching(
            strat_name, bt_func, h1_df, defaults,
            regime_labels, regime_params)

        results['strategies'][strat_name]['comparison'] = comparison
        print(f"    Default Sharpe:   {comparison['default']['sharpe']}", flush=True)
        print(f"    Best-overall:     {comparison['best_overall']['sharpe']}", flush=True)
        print(f"    Regime-adaptive:  {comparison['regime_adaptive']['sharpe']}", flush=True)

    # ── Phase 4: K-Fold validation ────────────────────────────
    print(f"\n{'─'*50}", flush=True)
    print("Phase 4: K-Fold (5 folds) cross-validation", flush=True)

    for strat_name in ['PSAR', 'TSMOM', 'SESS_BO']:
        print(f"\n  {strat_name}:", flush=True)
        bt_func = STRAT_FUNCS[strat_name]
        defaults = STRAT_DEFAULTS[strat_name]

        fold_results = kfold_regime_adaptive(strat_name, bt_func, h1_df, defaults)
        results['strategies'][strat_name]['kfold'] = fold_results

        if fold_results:
            avg_default = np.mean([f['default_sharpe'] for f in fold_results])
            avg_best = np.mean([f['best_overall_sharpe'] for f in fold_results])
            avg_adapt = np.mean([f['regime_adaptive_sharpe'] for f in fold_results])
            print(f"    Avg across folds: default={avg_default:.3f} "
                  f"best={avg_best:.3f} adaptive={avg_adapt:.3f}", flush=True)
            results['strategies'][strat_name]['kfold_summary'] = {
                'avg_default_sharpe': round(avg_default, 3),
                'avg_best_overall_sharpe': round(avg_best, 3),
                'avg_regime_adaptive_sharpe': round(avg_adapt, 3),
                'n_folds': len(fold_results),
            }

    # ── Save ──────────────────────────────────────────────────
    elapsed = time.time() - t0
    results['elapsed_sec'] = round(elapsed, 1)

    out_path = OUTPUT_DIR / "r128_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n{'='*60}", flush=True)
    print(f"R128 complete in {elapsed:.0f}s", flush=True)
    print(f"Saved: {out_path}", flush=True)


if __name__ == '__main__':
    main()
