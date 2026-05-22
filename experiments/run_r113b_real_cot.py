#!/usr/bin/env python3
"""
R113-B: COT Signal with REAL CFTC Data
=======================================
Uses actual CFTC Commitment of Traders weekly report for Gold (COMEX).
Data: 1986-2026, ~1445 weekly reports.

Tests:
  1. Net Speculative Z-Score as direction filter
  2. Net Change Z-Score as momentum signal  
  3. COT + existing strategy portfolio filter
  4. K-Fold validation (5 folds)
"""
import sys, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r113b_real_cot")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data")

PV = 100; SPREAD = 0.30; UNIT_LOT = 0.01
t0 = time.time()

FOLDS = [
    ("Fold1", "2006-01-01", "2010-01-01"),
    ("Fold2", "2010-01-01", "2014-01-01"),
    ("Fold3", "2014-01-01", "2018-01-01"),
    ("Fold4", "2018-01-01", "2022-01-01"),
    ("Fold5", "2022-01-01", "2027-01-01"),
]


def sharpe(arr, ann=252):
    if len(arr) < 10: return 0.0
    s = np.std(arr, ddof=1)
    return float(np.mean(arr) / s * np.sqrt(ann)) if s > 0 else 0.0


def max_dd(arr):
    if len(arr) == 0: return 0.0
    eq = np.cumsum(arr)
    return float((np.maximum.accumulate(eq) - eq).max())


def metrics(trades, ann=252):
    if not trades:
        return {'n': 0, 'sharpe': 0, 'pnl': 0, 'max_dd': 0, 'wr': 0, 'avg_pnl': 0, 'pf': 0}
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0) + t['pnl']
    dates = sorted(daily.keys())
    arr = np.array([daily[d] for d in dates])
    wins = [t['pnl'] for t in trades if t['pnl'] > 0]
    losses = [t['pnl'] for t in trades if t['pnl'] <= 0]
    tw = sum(wins) if wins else 0
    tl = abs(sum(losses)) if losses else 0.01
    return {
        'n': len(trades), 'sharpe': round(sharpe(arr, ann), 3),
        'pnl': round(sum(t['pnl'] for t in trades), 2),
        'max_dd': round(max_dd(arr), 2),
        'wr': round(len(wins) / len(trades) * 100, 1),
        'avg_pnl': round(sum(t['pnl'] for t in trades) / len(trades), 3),
        'pf': round(tw / tl, 2),
    }


def kfold_test(run_func, data, folds, **kwargs):
    results = []
    for fname, start, end in folds:
        fdata = data[(data.index >= start) & (data.index < end)]
        if len(fdata) < 60:
            results.append({'name': fname, 'sharpe': 0, 'n': 0})
            continue
        trades = run_func(fdata, **kwargs)
        m = metrics(trades)
        results.append({'name': fname, 'sharpe': m['sharpe'], 'n': m['n'],
                        'pnl': m['pnl'], 'wr': m['wr']})
    sharpes = [r['sharpe'] for r in results]
    pos = sum(1 for s in sharpes if s > 0)
    return {'folds': results, 'positive': pos, 'mean': round(np.mean(sharpes), 3), 'pass': pos >= 3}


def main():
    print("=" * 80)
    print("  R113-B: COT Signal with REAL CFTC Data")
    print("=" * 80)

    # Load data — use spot XAUUSD D1 (M15 resampled) so that the trades simulated
    # in this experiment match the broker price we actually execute against.
    # Previously used xauusd_daily_yf.csv (= COMEX GC=F futures), which sits
    # 5-10 USD off spot and biases SL/TP/ATR-based PnL calculations.
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from backtest.runner import load_d1_spot
    gold = load_d1_spot(tz_naive=True)
    gold = gold.dropna(subset=['Close'])

    tr = pd.concat([gold['High'] - gold['Low'],
                     (gold['High'] - gold['Close'].shift()).abs(),
                     (gold['Low'] - gold['Close'].shift()).abs()], axis=1).max(axis=1)
    gold['ATR14'] = tr.rolling(14).mean()
    gold['SMA50'] = gold['Close'].rolling(50).mean()
    gold['SMA200'] = gold['Close'].rolling(200).mean()

    cot = pd.read_csv(DATA_DIR / "cot_gold_weekly.csv", index_col=0, parse_dates=True)
    if cot.index.tz is not None:
        cot.index = cot.index.tz_localize(None)

    print(f"  Gold: {len(gold)} daily bars ({gold.index[0].date()} ~ {gold.index[-1].date()})")
    print(f"  COT:  {len(cot)} weekly reports ({cot.index[0].date()} ~ {cot.index[-1].date()})")

    cot_daily = cot[['net_spec', 'net_spec_z', 'net_change', 'net_change_z', 'net_pct_oi']].reindex(gold.index, method='ffill')
    df = gold.copy()
    for col in cot_daily.columns:
        df[col] = cot_daily[col]
    df = df.dropna(subset=['net_spec_z', 'ATR14'])
    print(f"  Merged: {len(df)} daily bars with COT data")

    # ── Phase 1: COT Z-Score Directional Filter ──
    print("\n" + "─" * 70)
    print("  Phase 1: COT Z-Score as Direction Filter")
    print("─" * 70)

    def bt_cot_z(data, z_long=0.5, z_short=-0.5, trend_filter=True,
                 sl_atr=4.0, tp_atr=3.0, max_hold=20):
        c = data['Close'].values; h = data['High'].values; lo = data['Low'].values
        atr = data['ATR14'].values; cz = data['net_spec_z'].values
        sma50 = data['SMA50'].values
        times = data.index; n = len(data)
        trades = []; pos = None; last_exit = -999
        for i in range(1, n):
            if pos is not None:
                bars = i - pos['bar']
                sl = pos['atr'] * sl_atr; tp = pos['atr'] * tp_atr
                if pos['dir'] == 'BUY':
                    if lo[i] <= pos['entry'] - sl:
                        trades.append({'dir':'BUY','entry_time':pos['time'],'exit_time':times[i],'pnl':-sl*UNIT_LOT*PV,'reason':'SL'})
                        pos = None; last_exit = i; continue
                    if h[i] >= pos['entry'] + tp:
                        trades.append({'dir':'BUY','entry_time':pos['time'],'exit_time':times[i],'pnl':tp*UNIT_LOT*PV,'reason':'TP'})
                        pos = None; last_exit = i; continue
                else:
                    if h[i] >= pos['entry'] + sl:
                        trades.append({'dir':'SELL','entry_time':pos['time'],'exit_time':times[i],'pnl':-sl*UNIT_LOT*PV,'reason':'SL'})
                        pos = None; last_exit = i; continue
                    if lo[i] <= pos['entry'] - tp:
                        trades.append({'dir':'SELL','entry_time':pos['time'],'exit_time':times[i],'pnl':tp*UNIT_LOT*PV,'reason':'TP'})
                        pos = None; last_exit = i; continue
                if bars >= max_hold:
                    pnl = ((c[i]-pos['entry']-SPREAD) if pos['dir']=='BUY' else (pos['entry']-c[i]-SPREAD)) * UNIT_LOT * PV
                    trades.append({'dir':pos['dir'],'entry_time':pos['time'],'exit_time':times[i],'pnl':pnl,'reason':'Time'})
                    pos = None; last_exit = i
                continue
            if i - last_exit < 2: continue
            if np.isnan(atr[i]) or atr[i] < 0.1 or np.isnan(cz[i]): continue
            trend_ok = not trend_filter or not np.isnan(sma50[i])
            if cz[i] > z_long and trend_ok and (not trend_filter or c[i] > sma50[i]):
                pos = {'dir':'BUY','entry':c[i]+SPREAD/2,'bar':i,'time':times[i],'atr':atr[i]}
            elif cz[i] < z_short and trend_ok and (not trend_filter or c[i] < sma50[i]):
                pos = {'dir':'SELL','entry':c[i]-SPREAD/2,'bar':i,'time':times[i],'atr':atr[i]}
        return trades

    grid = []
    for zl in [-0.5, 0.0, 0.5, 1.0, 1.5]:
        for zs in [0.5, 0.0, -0.5, -1.0, -1.5]:
            for tf in [True, False]:
                for sl in [3.0, 4.0, 5.0]:
                    for tp in [2.0, 3.0, 4.0, 6.0]:
                        for mh in [10, 15, 20, 30]:
                            trades = bt_cot_z(df, z_long=zl, z_short=zs, trend_filter=tf,
                                              sl_atr=sl, tp_atr=tp, max_hold=mh)
                            m = metrics(trades)
                            if m['n'] >= 30:
                                grid.append({'params':{'zl':zl,'zs':zs,'tf':tf,'sl':sl,'tp':tp,'mh':mh}, **m})

    grid.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"  Grid: {len(grid)} valid configs (>=30 trades)")
    for i, g in enumerate(grid[:10]):
        p = g['params']
        tf_str = "T" if p['tf'] else "F"
        print(f"    #{i+1}: zL={p['zl']:>4} zS={p['zs']:>5} trend={tf_str} SL={p['sl']} TP={p['tp']} MH={p['mh']:2d} "
              f"-> Sharpe={g['sharpe']:6.3f}, n={g['n']:4d}, PnL=${g['pnl']:>8.0f}, WR={g['wr']:.1f}%, PF={g['pf']}")

    # K-Fold top 3
    print("\n  K-Fold Validation (top 3):")
    kf_results = []
    for rank, g in enumerate(grid[:3]):
        p = g['params']
        kf = kfold_test(bt_cot_z, df, FOLDS,
                        z_long=p['zl'], z_short=p['zs'], trend_filter=p['tf'],
                        sl_atr=p['sl'], tp_atr=p['tp'], max_hold=p['mh'])
        status = "PASS" if kf['pass'] else "FAIL"
        fold_str = " ".join(f"{f['sharpe']:6.3f}" for f in kf['folds'])
        print(f"    #{rank+1}: {fold_str} -> {kf['positive']}/5 [{status}] mean={kf['mean']}")
        kf_results.append({'rank': rank+1, 'params': p, 'fullsample': g, 'kfold': kf})

    # ── Phase 2: COT Net Change (Momentum) ──
    print("\n" + "─" * 70)
    print("  Phase 2: COT Net Change Momentum")
    print("─" * 70)

    def bt_cot_change(data, change_long=1.0, change_short=-1.0,
                      sl_atr=4.0, tp_atr=3.0, max_hold=15):
        c = data['Close'].values; h = data['High'].values; lo = data['Low'].values
        atr = data['ATR14'].values; nz = data['net_change_z'].values
        sma50 = data['SMA50'].values
        times = data.index; n = len(data)
        trades = []; pos = None; last_exit = -999
        for i in range(1, n):
            if pos is not None:
                bars = i - pos['bar']
                sl = pos['atr'] * sl_atr; tp = pos['atr'] * tp_atr
                if pos['dir'] == 'BUY':
                    if lo[i] <= pos['entry'] - sl:
                        trades.append({'dir':'BUY','entry_time':pos['time'],'exit_time':times[i],'pnl':-sl*UNIT_LOT*PV,'reason':'SL'})
                        pos = None; last_exit = i; continue
                    if h[i] >= pos['entry'] + tp:
                        trades.append({'dir':'BUY','entry_time':pos['time'],'exit_time':times[i],'pnl':tp*UNIT_LOT*PV,'reason':'TP'})
                        pos = None; last_exit = i; continue
                else:
                    if h[i] >= pos['entry'] + sl:
                        trades.append({'dir':'SELL','entry_time':pos['time'],'exit_time':times[i],'pnl':-sl*UNIT_LOT*PV,'reason':'SL'})
                        pos = None; last_exit = i; continue
                    if lo[i] <= pos['entry'] - tp:
                        trades.append({'dir':'SELL','entry_time':pos['time'],'exit_time':times[i],'pnl':tp*UNIT_LOT*PV,'reason':'TP'})
                        pos = None; last_exit = i; continue
                if bars >= max_hold:
                    pnl = ((c[i]-pos['entry']-SPREAD) if pos['dir']=='BUY' else (pos['entry']-c[i]-SPREAD)) * UNIT_LOT * PV
                    trades.append({'dir':pos['dir'],'entry_time':pos['time'],'exit_time':times[i],'pnl':pnl,'reason':'Time'})
                    pos = None; last_exit = i
                continue
            if i - last_exit < 2: continue
            if np.isnan(atr[i]) or atr[i] < 0.1 or np.isnan(nz[i]): continue
            if nz[i] > change_long and (np.isnan(sma50[i]) or c[i] > sma50[i]):
                pos = {'dir':'BUY','entry':c[i]+SPREAD/2,'bar':i,'time':times[i],'atr':atr[i]}
            elif nz[i] < change_short and (np.isnan(sma50[i]) or c[i] < sma50[i]):
                pos = {'dir':'SELL','entry':c[i]-SPREAD/2,'bar':i,'time':times[i],'atr':atr[i]}
        return trades

    grid2 = []
    for cl in [0.5, 1.0, 1.5, 2.0]:
        for cs in [-0.5, -1.0, -1.5, -2.0]:
            for sl in [3.0, 4.0, 5.0]:
                for tp in [2.0, 3.0, 4.0]:
                    for mh in [10, 15, 20]:
                        trades = bt_cot_change(df, change_long=cl, change_short=cs,
                                               sl_atr=sl, tp_atr=tp, max_hold=mh)
                        m = metrics(trades)
                        if m['n'] >= 30:
                            grid2.append({'params':{'cl':cl,'cs':cs,'sl':sl,'tp':tp,'mh':mh}, **m})

    grid2.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"  Grid: {len(grid2)} valid configs")
    for i, g in enumerate(grid2[:5]):
        p = g['params']
        print(f"    #{i+1}: cL={p['cl']} cS={p['cs']} SL={p['sl']} TP={p['tp']} MH={p['mh']} "
              f"-> Sharpe={g['sharpe']:6.3f}, n={g['n']}, PnL=${g['pnl']:>8.0f}, WR={g['wr']:.1f}%")

    if grid2:
        p = grid2[0]['params']
        kf2 = kfold_test(bt_cot_change, df, FOLDS,
                         change_long=p['cl'], change_short=p['cs'],
                         sl_atr=p['sl'], tp_atr=p['tp'], max_hold=p['mh'])
        status = "PASS" if kf2['pass'] else "FAIL"
        fold_str = " ".join(f"{f['sharpe']:6.3f}" for f in kf2['folds'])
        print(f"  K-Fold: {fold_str} -> {kf2['positive']}/5 [{status}]")
    else:
        kf2 = None

    # ── Phase 3: Contrarian (Sell when crowd bullish) ──
    print("\n" + "─" * 70)
    print("  Phase 3: COT Contrarian (Sell when crowd extreme bullish)")
    print("─" * 70)

    def bt_cot_contrarian(data, z_sell=1.5, z_buy=-1.5,
                          sl_atr=4.0, tp_atr=3.0, max_hold=20):
        c = data['Close'].values; h = data['High'].values; lo = data['Low'].values
        atr = data['ATR14'].values; cz = data['net_spec_z'].values
        times = data.index; n = len(data)
        trades = []; pos = None; last_exit = -999
        for i in range(1, n):
            if pos is not None:
                bars = i - pos['bar']
                sl = pos['atr'] * sl_atr; tp = pos['atr'] * tp_atr
                if pos['dir'] == 'BUY':
                    if lo[i] <= pos['entry'] - sl:
                        trades.append({'dir':'BUY','entry_time':pos['time'],'exit_time':times[i],'pnl':-sl*UNIT_LOT*PV,'reason':'SL'})
                        pos = None; last_exit = i; continue
                    if h[i] >= pos['entry'] + tp:
                        trades.append({'dir':'BUY','entry_time':pos['time'],'exit_time':times[i],'pnl':tp*UNIT_LOT*PV,'reason':'TP'})
                        pos = None; last_exit = i; continue
                else:
                    if h[i] >= pos['entry'] + sl:
                        trades.append({'dir':'SELL','entry_time':pos['time'],'exit_time':times[i],'pnl':-sl*UNIT_LOT*PV,'reason':'SL'})
                        pos = None; last_exit = i; continue
                    if lo[i] <= pos['entry'] - tp:
                        trades.append({'dir':'SELL','entry_time':pos['time'],'exit_time':times[i],'pnl':tp*UNIT_LOT*PV,'reason':'TP'})
                        pos = None; last_exit = i; continue
                if bars >= max_hold:
                    pnl = ((c[i]-pos['entry']-SPREAD) if pos['dir']=='BUY' else (pos['entry']-c[i]-SPREAD)) * UNIT_LOT * PV
                    trades.append({'dir':pos['dir'],'entry_time':pos['time'],'exit_time':times[i],'pnl':pnl,'reason':'Time'})
                    pos = None; last_exit = i
                continue
            if i - last_exit < 2: continue
            if np.isnan(atr[i]) or atr[i] < 0.1 or np.isnan(cz[i]): continue
            if cz[i] < z_buy:
                pos = {'dir':'BUY','entry':c[i]+SPREAD/2,'bar':i,'time':times[i],'atr':atr[i]}
            elif cz[i] > z_sell:
                pos = {'dir':'SELL','entry':c[i]-SPREAD/2,'bar':i,'time':times[i],'atr':atr[i]}
        return trades

    grid3 = []
    for zs in [1.0, 1.5, 2.0, 2.5]:
        for zb in [-1.0, -1.5, -2.0, -2.5]:
            for sl in [3.0, 4.0, 5.0]:
                for tp in [2.0, 3.0, 4.0, 6.0]:
                    for mh in [10, 15, 20, 30]:
                        trades = bt_cot_contrarian(df, z_sell=zs, z_buy=zb,
                                                   sl_atr=sl, tp_atr=tp, max_hold=mh)
                        m = metrics(trades)
                        if m['n'] >= 20:
                            grid3.append({'params':{'zs':zs,'zb':zb,'sl':sl,'tp':tp,'mh':mh}, **m})

    grid3.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"  Grid: {len(grid3)} valid configs")
    for i, g in enumerate(grid3[:5]):
        p = g['params']
        print(f"    #{i+1}: zS={p['zs']} zB={p['zb']} SL={p['sl']} TP={p['tp']} MH={p['mh']} "
              f"-> Sharpe={g['sharpe']:6.3f}, n={g['n']}, PnL=${g['pnl']:>8.0f}, WR={g['wr']:.1f}%")

    if grid3:
        p = grid3[0]['params']
        kf3 = kfold_test(bt_cot_contrarian, df, FOLDS,
                         z_sell=p['zs'], z_buy=p['zb'],
                         sl_atr=p['sl'], tp_atr=p['tp'], max_hold=p['mh'])
        status = "PASS" if kf3['pass'] else "FAIL"
        fold_str = " ".join(f"{f['sharpe']:6.3f}" for f in kf3['folds'])
        print(f"  K-Fold: {fold_str} -> {kf3['positive']}/5 [{status}]")
    else:
        kf3 = None

    # ── Phase 4: COT as Portfolio-Level Filter ──
    print("\n" + "─" * 70)
    print("  Phase 4: COT as Macro Regime Filter for Existing Portfolio")
    print("─" * 70)
    print("  Concept: Only allow LONG entries when COT Z > threshold")
    print("           Only allow SHORT entries when COT Z < threshold")

    cz = df['net_spec_z'].dropna()
    for threshold in [0.0, 0.5, 1.0, -0.5, -1.0]:
        long_days = (cz > threshold).sum()
        short_days = (cz < -abs(threshold)).sum()
        neutral = len(cz) - long_days - short_days
        print(f"  z_thresh={threshold:>5.1f}: LONG_OK={long_days:5d} ({long_days/len(cz)*100:.1f}%), "
              f"SHORT_OK={short_days:5d} ({short_days/len(cz)*100:.1f}%), "
              f"NEUTRAL={neutral:5d} ({neutral/len(cz)*100:.1f}%)")

    # ── Summary ──
    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)

    results = {
        'phase1_z_score': {'grid_top10': grid[:10], 'kfold_top3': kf_results},
        'phase2_change': {'grid_top5': grid2[:5], 'kfold': kf2},
        'phase3_contrarian': {'grid_top5': grid3[:5], 'kfold': kf3},
        'elapsed_s': round(elapsed, 1),
        'data_info': {
            'gold_bars': len(df),
            'cot_reports': len(cot),
            'cot_period': f"{cot.index[0].date()} ~ {cot.index[-1].date()}",
            'current_net_spec': int(cot['net_spec'].iloc[-1]),
            'current_z': round(float(cot['net_spec_z'].dropna().iloc[-1]), 2),
        }
    }

    out_file = OUTPUT_DIR / "r113b_results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"  Saved: {out_file}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"\n  Current COT status: Net Spec = {results['data_info']['current_net_spec']:,} contracts")
    print(f"  Current Z-Score: {results['data_info']['current_z']}")
    print(f"  Interpretation: {'BEARISH' if results['data_info']['current_z'] < -1 else 'BULLISH' if results['data_info']['current_z'] > 1 else 'NEUTRAL'} regime")


if __name__ == '__main__':
    main()
