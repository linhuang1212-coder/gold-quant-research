#!/usr/bin/env python3
"""
R242 — Donchian60 Comprehensive Re-evaluation (2026-05-21)
============================================================
Context:
  - R102 (2026-04): ch=60/SL4/TP3 K-Fold 6/6, Sharpe 7.079
  - R163 (2026-05): Confirmation, portfolio K-Fold
  - Paper trading: 15 trades, WR=93%, PnL=-$5.43 (1 SL = -$74.55)
  - Constraints doc: "Donchian Channel — 全部负Sharpe, 永久否决" (different variant)
  - R45: daily PnL correlation 0.445 with Keltner → diversification concern
  
  Current live portfolio changed: now Keltner + TSMOM + DualThrust + SESS_BO + M30_RSI14
  Need fresh validation against THIS portfolio.

Phases:
  1. Donchian60 standalone full-period + year-by-year
  2. Parameter sensitivity (ch ±10, SL ±1, TP ±1)
  3. K-Fold 6-fold validation (2-year folds)
  4. Signal overlap with Keltner (±1h, ±2h, ±4h)
  5. Combined portfolio test: current 5-strat vs 6-strat with Donchian
  6. Time-window stability (3 eras)
  7. Paper trading reality check: expected vs observed
"""
from __future__ import annotations
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

from backtest.runner import DataBundle, LIVE_PARITY_KWARGS
from backtest.engine import BacktestEngine
import indicators as signals_mod

OUTPUT_DIR = Path("results/r242_donchian")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PV = 100
SPREAD = 0.30
UNIT_LOT = 0.01

FOLDS = [
    ("Fold1", "2015-01-01", "2017-01-01"),
    ("Fold2", "2017-01-01", "2019-01-01"),
    ("Fold3", "2019-01-01", "2021-01-01"),
    ("Fold4", "2021-01-01", "2023-01-01"),
    ("Fold5", "2023-01-01", "2025-01-01"),
    ("Fold6", "2025-01-01", "2026-05-01"),
]

ERAS = [
    ("Pre-COVID", "2015-01-01", "2020-01-01"),
    ("COVID+Inflation", "2020-01-01", "2023-01-01"),
    ("Recent", "2023-01-01", "2026-05-06"),
]

t0 = time.time()


def compute_atr(df, period=14):
    h, l, c = df['High'], df['Low'], df['Close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _mk(pos, exit_px, exit_time, reason, bar_i, pnl):
    return {'dir': pos['dir'], 'entry': pos['entry'], 'exit': exit_px,
            'entry_time': pos['time'], 'exit_time': exit_time,
            'pnl': pnl, 'reason': reason, 'bars': bar_i - pos['bar']}


def _run_exit(pos, i, hi, lo, cl, spread, lot, pv, times,
              sl_atr, tp_atr, trail_act, trail_dist, max_hold):
    atr = pos['atr']
    sl = atr * sl_atr
    tp = atr * tp_atr
    bars = i - pos['bar']
    if pos['dir'] == 'BUY':
        if lo <= pos['entry'] - sl:
            return _mk(pos, pos['entry'] - sl, times[i], "SL", i, -sl * lot * pv)
        if hi >= pos['entry'] + tp:
            return _mk(pos, pos['entry'] + tp, times[i], "TP", i, tp * lot * pv)
        extreme = pos.get('extreme', pos['entry'])
        extreme = max(extreme, hi)
        pos['extreme'] = extreme
        act_dist = atr * trail_act
        if extreme - pos['entry'] >= act_dist:
            trail_price = extreme - atr * trail_dist
            if cl <= trail_price:
                pnl = (trail_price - pos['entry'] - spread) * lot * pv
                return _mk(pos, trail_price, times[i], "Trail", i, pnl)
        if bars >= max_hold:
            pnl = (cl - pos['entry'] - spread) * lot * pv
            return _mk(pos, cl, times[i], "TimeExit", i, pnl)
    else:
        if hi >= pos['entry'] + sl:
            return _mk(pos, pos['entry'] + sl, times[i], "SL", i, -sl * lot * pv)
        if lo <= pos['entry'] - tp:
            return _mk(pos, pos['entry'] - tp, times[i], "TP", i, tp * lot * pv)
        extreme = pos.get('extreme', pos['entry'])
        extreme = min(extreme, lo)
        pos['extreme'] = extreme
        act_dist = atr * trail_act
        if pos['entry'] - extreme >= act_dist:
            trail_price = extreme + atr * trail_dist
            if cl >= trail_price:
                pnl = (pos['entry'] - trail_price - spread) * lot * pv
                return _mk(pos, trail_price, times[i], "Trail", i, pnl)
        if bars >= max_hold:
            pnl = (pos['entry'] - cl - spread) * lot * pv
            return _mk(pos, cl, times[i], "TimeExit", i, pnl)
    return None


def bt_donchian(h1_df, spread=SPREAD, lot=UNIT_LOT,
                channel=60, sl_atr=4.0, tp_atr=3.0,
                trail_act=0.14, trail_dist=0.025, max_hold=20,
                cooldown=2):
    """Standalone Donchian backtest — matches R102/R163/live signal logic."""
    df = h1_df.copy()
    df['ATR'] = compute_atr(df)
    df['HH'] = df['High'].rolling(channel).max()
    df['LL'] = df['Low'].rolling(channel).min()
    df = df.dropna(subset=['ATR', 'HH', 'LL'])
    c = df['Close'].values
    h = df['High'].values
    lo = df['Low'].values
    atr = df['ATR'].values
    hh = df['HH'].values
    ll = df['LL'].values
    times = df.index
    n = len(df)
    trades = []
    entry_times = []
    pos = None
    last_exit = -999
    for i in range(1, n):
        if pos is not None:
            result = _run_exit(pos, i, h[i], lo[i], c[i], spread, lot, PV, times,
                               sl_atr, tp_atr, trail_act, trail_dist, max_hold)
            if result:
                trades.append(result)
                pos = None
                last_exit = i
                continue
            continue
        if i - last_exit < cooldown:
            continue
        if np.isnan(atr[i]) or atr[i] < 0.1:
            continue
        if c[i] > hh[i - 1]:
            pos = {'dir': 'BUY', 'entry': c[i] + spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
            entry_times.append(times[i])
        elif c[i] < ll[i - 1]:
            pos = {'dir': 'SELL', 'entry': c[i] - spread / 2, 'bar': i, 'time': times[i], 'atr': atr[i]}
            entry_times.append(times[i])
    return trades, entry_times


def trades_to_daily(trades):
    if not trades:
        return pd.Series(dtype=float)
    daily = {}
    for t in trades:
        d = pd.Timestamp(t['exit_time']).normalize()
        if hasattr(d, 'tz') and d.tzinfo is not None:
            d = d.tz_localize(None)
        daily[d] = daily.get(d, 0) + t['pnl']
    dates = sorted(daily.keys())
    return pd.Series([daily[d] for d in dates], index=pd.DatetimeIndex(dates))


def calc_metrics(trades):
    if not trades:
        return {'n': 0, 'pnl': 0, 'sharpe': 0, 'wr': 0, 'max_dd': 0, 'avg_pnl': 0}
    daily = trades_to_daily(trades)
    arr = daily.values
    n = len(trades)
    pnl_total = sum(t['pnl'] for t in trades)
    wr = sum(1 for t in trades if t['pnl'] > 0) / n * 100
    avg = pnl_total / n
    if len(arr) >= 10 and np.std(arr, ddof=1) > 0:
        sh = float(np.mean(arr) / np.std(arr, ddof=1) * np.sqrt(252))
    else:
        sh = 0.0
    eq = np.cumsum(arr)
    dd = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) > 0 else 0
    return {
        'n': n,
        'pnl': round(pnl_total, 2),
        'sharpe': round(sh, 3),
        'wr': round(wr, 1),
        'max_dd': round(dd, 2),
        'avg_pnl': round(avg, 3),
    }


def run_keltner_engine(data, h1_slice=None):
    """Run Keltner via BacktestEngine, return entry times."""
    kwargs = {**LIVE_PARITY_KWARGS, 'max_positions': 1}

    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False
    from indicators import get_orb_strategy
    get_orb_strategy().reset_daily()

    m15 = data.m15_df
    h1 = data.h1_df
    if h1_slice is not None:
        start, end = h1_slice
        h1 = h1[(h1.index >= start) & (h1.index < end)]
        m15 = m15[(m15.index >= start) & (m15.index < end)]

    engine = BacktestEngine(m15, h1, **kwargs)
    trades = engine.run()
    kc_trades = [t for t in trades if t.strategy == 'keltner']
    kc_entries = [pd.Timestamp(t.entry_time) for t in kc_trades]
    return kc_trades, kc_entries


def print_metrics(label, m):
    print(f"    {label:20s}: n={m['n']:>5} PnL=${m['pnl']:>+10.2f} "
          f"Sharpe={m['sharpe']:>+7.3f} WR={m['wr']:>5.1f}% "
          f"MaxDD=${m['max_dd']:>8.2f} AvgPnL=${m['avg_pnl']:>+7.3f}")


def main():
    print("=" * 80)
    print("  R242 — Donchian60 Comprehensive Re-evaluation")
    print("=" * 80)

    print("\nLoading data...")
    data = DataBundle.load_default()
    h1_df_raw = data.h1_df.copy()
    print(f"  H1: {len(data.h1_df)} bars ({data.h1_df.index[0].date()} ~ {data.h1_df.index[-1].date()})")

    results = {}

    # ══════════════════════════════════════════════════════════════
    # Phase 1: Standalone full-period + year-by-year
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("Phase 1: Donchian60 Standalone (ch=60, SL=4, TP=3, Trail 0.14/0.025, MH=20)")
    print("=" * 80)

    donch_trades, donch_entries = bt_donchian(h1_df_raw)
    m = calc_metrics(donch_trades)
    print_metrics("Donchian60 Full", m)
    results['phase1_full'] = m

    # Year-by-year
    print("\n  Year-by-year breakdown:")
    yearly = {}
    for year in range(2015, 2027):
        yt = [t for t in donch_trades if pd.Timestamp(t['exit_time']).year == year]
        if yt:
            ym = calc_metrics(yt)
            yearly[year] = ym
            print(f"    {year}: n={ym['n']:>4} PnL=${ym['pnl']:>+8.2f} Sharpe={ym['sharpe']:>+7.3f} WR={ym['wr']:.0f}%")
    results['phase1_yearly'] = yearly

    # Exit reasons
    reasons = {}
    for t in donch_trades:
        r = t.get('reason', 'Unknown')
        reasons[r] = reasons.get(r, 0) + 1
    print("\n  Exit reasons:")
    for r, cnt in sorted(reasons.items(), key=lambda x: -x[1]):
        pnl_r = sum(t['pnl'] for t in donch_trades if t.get('reason') == r)
        print(f"    {r:12s}: {cnt:>4} ({cnt/len(donch_trades)*100:.1f}%) PnL=${pnl_r:>+.2f}")
    results['phase1_exit_reasons'] = reasons

    # ══════════════════════════════════════════════════════════════
    # Phase 2: Parameter sensitivity
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("Phase 2: Parameter Sensitivity (around ch=60, SL=4, TP=3)")
    print("=" * 80)

    param_grid = {
        'channel': [40, 50, 60, 70, 80],
        'sl_atr': [3.0, 3.5, 4.0, 4.5, 5.0],
        'tp_atr': [2.0, 2.5, 3.0, 3.5, 4.0],
        'max_hold': [15, 20, 25, 30],
    }
    total = 1
    for v in param_grid.values():
        total *= len(v)
    print(f"  {total} combinations...")

    grid = []
    tested = 0
    for ch, sl, tp, mh in product(param_grid['channel'], param_grid['sl_atr'],
                                   param_grid['tp_atr'], param_grid['max_hold']):
        trades, _ = bt_donchian(h1_df_raw, channel=ch, sl_atr=sl, tp_atr=tp, max_hold=mh)
        m = calc_metrics(trades)
        grid.append({'params': {'ch': ch, 'sl': sl, 'tp': tp, 'mh': mh}, **m})
        tested += 1
        if tested % 100 == 0:
            print(f"    {tested}/{total}...", flush=True)

    grid.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"\n  Top 10:")
    for i, g in enumerate(grid[:10]):
        p = g['params']
        print(f"    #{i+1}: ch={p['ch']:>2} SL={p['sl']:.1f} TP={p['tp']:.1f} MH={p['mh']:>2} "
              f"→ Sharpe={g['sharpe']:>+7.3f} n={g['n']:>4} PnL=${g['pnl']:>+.2f} WR={g['wr']:.0f}%")

    # Check where our live params (ch=60/SL4/TP3/MH20) rank
    live_rank = next((i for i, g in enumerate(grid) if
                      g['params'] == {'ch': 60, 'sl': 4.0, 'tp': 3.0, 'mh': 20}), None)
    print(f"\n  Live params ch=60/SL4/TP3/MH20 rank: #{live_rank+1 if live_rank is not None else '?'}/{total}")
    results['phase2_top10'] = grid[:10]
    results['phase2_live_rank'] = live_rank + 1 if live_rank is not None else None
    results['phase2_total'] = total

    # ══════════════════════════════════════════════════════════════
    # Phase 3: K-Fold validation
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("Phase 3: K-Fold Validation (6 folds × top 3 + live params)")
    print("=" * 80)

    candidates = []
    # Add live params if not in top 3
    live_params = {'ch': 60, 'sl': 4.0, 'tp': 3.0, 'mh': 20}
    top3_params = [g['params'] for g in grid[:3]]
    if live_params not in top3_params:
        candidates = top3_params + [live_params]
    else:
        candidates = top3_params[:4]

    kfold_results = {}
    for p in candidates:
        label = f"ch{p['ch']}/SL{p['sl']}/TP{p['tp']}/MH{p['mh']}"
        fold_sharpes = []
        fold_details = []
        for fname, start, end in FOLDS:
            fold_h1 = h1_df_raw[(h1_df_raw.index >= start) & (h1_df_raw.index < end)]
            if len(fold_h1) < 100:
                fold_sharpes.append(0.0)
                fold_details.append({'fold': fname, 'sharpe': 0, 'n': 0, 'pnl': 0})
                continue
            trades, _ = bt_donchian(fold_h1, channel=p['ch'], sl_atr=p['sl'],
                                    tp_atr=p['tp'], max_hold=p['mh'])
            m = calc_metrics(trades)
            fold_sharpes.append(m['sharpe'])
            fold_details.append({'fold': fname, **m})

        positive = sum(1 for s in fold_sharpes if s > 0)
        mean_sh = np.mean(fold_sharpes)
        min_sh = min(fold_sharpes)
        passed = positive >= 5
        kfold_results[label] = {
            'params': p,
            'fold_sharpes': [round(s, 3) for s in fold_sharpes],
            'fold_details': fold_details,
            'positive': positive,
            'mean_sharpe': round(mean_sh, 3),
            'min_sharpe': round(min_sh, 3),
            'pass_5of6': passed,
        }
        status = "PASS" if passed else ("MARGINAL" if positive >= 4 else "FAIL")
        print(f"  {label:28s}: {[f'{s:+.2f}' for s in fold_sharpes]} "
              f"→ {positive}/6 [{status}] mean={mean_sh:+.3f} min={min_sh:+.3f}")
    results['phase3_kfold'] = kfold_results

    # ══════════════════════════════════════════════════════════════
    # Phase 4: Signal overlap with Keltner
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("Phase 4: Signal Overlap — Donchian vs Keltner")
    print("=" * 80)

    print("  Running Keltner engine...", flush=True)
    kc_trades_obj, kc_entries = run_keltner_engine(data)
    kc_times = pd.DatetimeIndex(sorted([
        t.tz_localize(None) if t.tz else t for t in kc_entries
    ]))
    donch_times = pd.DatetimeIndex(sorted([
        t.tz_localize(None) if hasattr(t, 'tz') and t.tz else t for t in donch_entries
    ]))

    print(f"  Donchian entries: {len(donch_entries)}, Keltner entries: {len(kc_entries)}")

    overlap_results = {}
    for window_h in [1, 2, 4, 6, 8]:
        if len(donch_times) == 0 or len(kc_times) == 0:
            overlap_results[f'{window_h}h'] = {'overlap': 0, 'pct': 0}
            continue
        overlap = 0
        for dt in donch_times:
            diffs_h = np.abs((kc_times - dt).total_seconds() / 3600.0)
            if diffs_h.min() <= window_h:
                overlap += 1
        pct = round(100 * overlap / len(donch_times), 1)
        overlap_results[f'{window_h}h'] = {'overlap': overlap, 'total': len(donch_times), 'pct': pct}
        print(f"  ±{window_h}h: {overlap}/{len(donch_times)} = {pct}% overlap")

    # Daily PnL correlation
    donch_daily = trades_to_daily(donch_trades)
    kc_daily_trades = [{'exit_time': t.exit_time, 'pnl': t.pnl} for t in kc_trades_obj]
    kc_daily = trades_to_daily(kc_daily_trades)

    if len(donch_daily) > 10 and len(kc_daily) > 10:
        combined = pd.DataFrame({'donchian': donch_daily, 'keltner': kc_daily}).fillna(0)
        daily_corr = round(float(combined['donchian'].corr(combined['keltner'])), 4)
        print(f"\n  Daily PnL correlation: {daily_corr:+.4f}")
        overlap_results['daily_pnl_corr'] = daily_corr
    else:
        overlap_results['daily_pnl_corr'] = None
    results['phase4_overlap'] = overlap_results

    # Direction agreement on overlapping signals
    if len(donch_times) > 0 and len(kc_times) > 0:
        same_dir = 0
        opp_dir = 0
        donch_dirs = {str(donch_entries[i]): donch_trades[i]['dir']
                      for i in range(min(len(donch_entries), len(donch_trades)))}
        for dt in donch_times:
            diffs_h = np.abs((kc_times - dt).total_seconds() / 3600.0)
            if diffs_h.min() <= 2:
                nearest_idx = diffs_h.argmin()
                nearest_kc_time = kc_times[nearest_idx]
                kc_trade = next((t for t in kc_trades_obj
                                if abs((pd.Timestamp(t.entry_time).tz_localize(None)
                                       if pd.Timestamp(t.entry_time).tz
                                       else pd.Timestamp(t.entry_time))
                                      - nearest_kc_time).total_seconds() < 10), None)
                if kc_trade:
                    donch_dir = donch_dirs.get(str(dt))
                    kc_dir = kc_trade.direction
                    if donch_dir and kc_dir:
                        if donch_dir == kc_dir:
                            same_dir += 1
                        else:
                            opp_dir += 1
        total_overlap = same_dir + opp_dir
        if total_overlap > 0:
            print(f"  Direction agreement (±2h overlaps): {same_dir}/{total_overlap} same dir "
                  f"({100*same_dir/total_overlap:.0f}%), {opp_dir} opposite")
            overlap_results['direction'] = {
                'same': same_dir, 'opposite': opp_dir,
                'same_pct': round(100 * same_dir / total_overlap, 1)
            }

    # ══════════════════════════════════════════════════════════════
    # Phase 5: Combined portfolio test
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("Phase 5: Portfolio Integration — Current 5-strat vs 6-strat (+Donchian)")
    print("=" * 80)

    print("  Running current 5-strategy portfolio via BacktestEngine...", flush=True)

    def run_full_engine(data_bundle, max_pos=5):
        kwargs = {**LIVE_PARITY_KWARGS, 'max_positions': max_pos}
        signals_mod._friday_close_price = None
        signals_mod._gap_traded_today = False
        signals_mod.SESS_BO_ENABLED = True
        signals_mod.DUAL_THRUST_ENABLED = True
        signals_mod.PSAR_ENABLED = False
        signals_mod.CHANDELIER_ENABLED = False
        from indicators import get_orb_strategy
        get_orb_strategy().reset_daily()
        engine = BacktestEngine(data_bundle.m15_df, data_bundle.h1_df, **kwargs)
        return engine.run()

    all_trades_5 = run_full_engine(data)
    strat_counts = {}
    for t in all_trades_5:
        s = t.strategy
        strat_counts[s] = strat_counts.get(s, 0) + 1

    print("  Current 5-strat breakdown:")
    all_5_daily = {}
    for t in all_trades_5:
        d = pd.Timestamp(t.exit_time).normalize()
        if d.tz:
            d = d.tz_localize(None)
        all_5_daily[d] = all_5_daily.get(d, 0) + t.pnl
    arr_5 = np.array([all_5_daily[d] for d in sorted(all_5_daily.keys())])
    sh_5 = float(np.mean(arr_5) / np.std(arr_5, ddof=1) * np.sqrt(252)) if len(arr_5) > 10 and np.std(arr_5, ddof=1) > 0 else 0
    pnl_5 = float(arr_5.sum())
    dd_5 = float((np.maximum.accumulate(np.cumsum(arr_5)) - np.cumsum(arr_5)).max())
    print(f"    5-strat: n={len(all_trades_5)} Sharpe={sh_5:+.3f} PnL=${pnl_5:+.2f} MaxDD=${dd_5:.2f}")
    for s, cnt in sorted(strat_counts.items(), key=lambda x: -x[1]):
        strat_pnl = sum(t.pnl for t in all_trades_5 if t.strategy == s)
        print(f"      {s:15s}: {cnt:>5} trades, PnL=${strat_pnl:>+10.2f}")

    # Add Donchian as 6th strategy — merge daily PnL
    donch_daily_dict = {}
    for t in donch_trades:
        d = pd.Timestamp(t['exit_time']).normalize()
        if hasattr(d, 'tz') and d.tzinfo is not None:
            d = d.tz_localize(None)
        lot_scale = 0.03 / UNIT_LOT
        donch_daily_dict[d] = donch_daily_dict.get(d, 0) + t['pnl'] * lot_scale

    all_6_daily = dict(all_5_daily)
    for d, pnl in donch_daily_dict.items():
        all_6_daily[d] = all_6_daily.get(d, 0) + pnl
    arr_6 = np.array([all_6_daily[d] for d in sorted(all_6_daily.keys())])
    sh_6 = float(np.mean(arr_6) / np.std(arr_6, ddof=1) * np.sqrt(252)) if len(arr_6) > 10 and np.std(arr_6, ddof=1) > 0 else 0
    pnl_6 = float(arr_6.sum())
    dd_6 = float((np.maximum.accumulate(np.cumsum(arr_6)) - np.cumsum(arr_6)).max())
    print(f"\n    6-strat (+Donchian@0.03): Sharpe={sh_6:+.3f} PnL=${pnl_6:+.2f} MaxDD=${dd_6:.2f}")
    print(f"    Delta: Sharpe {sh_6 - sh_5:+.3f}, PnL ${pnl_6 - pnl_5:+.2f}, MaxDD ${dd_6 - dd_5:+.2f}")

    results['phase5_portfolio'] = {
        '5strat': {'sharpe': round(sh_5, 3), 'pnl': round(pnl_5, 2), 'max_dd': round(dd_5, 2), 'n': len(all_trades_5)},
        '6strat_donch003': {'sharpe': round(sh_6, 3), 'pnl': round(pnl_6, 2), 'max_dd': round(dd_6, 2)},
        'delta_sharpe': round(sh_6 - sh_5, 3),
        'delta_pnl': round(pnl_6 - pnl_5, 2),
    }

    # Portfolio K-Fold
    print("\n  Portfolio K-Fold (5 vs 6 strat):")
    port_kfold_5 = []
    port_kfold_6 = []
    for fname, start, end in FOLDS:
        fold_h1 = h1_df_raw[(h1_df_raw.index >= start) & (h1_df_raw.index < end)]
        if len(fold_h1) < 100:
            port_kfold_5.append(0.0)
            port_kfold_6.append(0.0)
            continue

        # 5-strat fold
        fold_5_daily = {}
        for t in all_trades_5:
            et = pd.Timestamp(t.exit_time)
            if et.tz:
                et = et.tz_localize(None)
            if start <= str(et) < end:
                d = et.normalize()
                fold_5_daily[d] = fold_5_daily.get(d, 0) + t.pnl
        arr = np.array(list(fold_5_daily.values())) if fold_5_daily else np.array([0.0])
        sh = float(np.mean(arr) / np.std(arr, ddof=1) * np.sqrt(252)) if len(arr) > 10 and np.std(arr, ddof=1) > 0 else 0
        port_kfold_5.append(sh)

        # 6-strat fold (add Donchian)
        donch_fold_trades, _ = bt_donchian(fold_h1)
        fold_6_daily = dict(fold_5_daily)
        for t in donch_fold_trades:
            d = pd.Timestamp(t['exit_time']).normalize()
            if hasattr(d, 'tz') and d.tzinfo is not None:
                d = d.tz_localize(None)
            lot_scale = 0.03 / UNIT_LOT
            fold_6_daily[d] = fold_6_daily.get(d, 0) + t['pnl'] * lot_scale
        arr = np.array(list(fold_6_daily.values())) if fold_6_daily else np.array([0.0])
        sh = float(np.mean(arr) / np.std(arr, ddof=1) * np.sqrt(252)) if len(arr) > 10 and np.std(arr, ddof=1) > 0 else 0
        port_kfold_6.append(sh)

    for i, (fname, _, _) in enumerate(FOLDS):
        delta = port_kfold_6[i] - port_kfold_5[i]
        marker = "+" if delta > 0 else "-"
        print(f"    {fname}: 5strat={port_kfold_5[i]:+.3f} 6strat={port_kfold_6[i]:+.3f} [{marker}{abs(delta):.3f}]")

    wins_6 = sum(1 for a, b in zip(port_kfold_6, port_kfold_5) if a > b)
    print(f"  6-strat wins: {wins_6}/6 folds")
    results['phase5_kfold'] = {
        '5strat_folds': [round(s, 3) for s in port_kfold_5],
        '6strat_folds': [round(s, 3) for s in port_kfold_6],
        '6strat_wins': wins_6,
    }

    # ══════════════════════════════════════════════════════════════
    # Phase 6: Time-window stability
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("Phase 6: Time-Window Stability (3 eras)")
    print("=" * 80)

    era_results = {}
    for era_name, start, end in ERAS:
        era_h1 = h1_df_raw[(h1_df_raw.index >= start) & (h1_df_raw.index < end)]
        trades, _ = bt_donchian(era_h1)
        m = calc_metrics(trades)
        era_results[era_name] = m
        print_metrics(era_name, m)
    results['phase6_eras'] = era_results

    all_positive = all(er['sharpe'] > 0 for er in era_results.values())
    print(f"\n  All eras positive Sharpe: {'YES' if all_positive else 'NO'}")

    # ══════════════════════════════════════════════════════════════
    # Phase 7: Paper trading reality check
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("Phase 7: Paper Trading Reality Check")
    print("=" * 80)

    paper_n = 15
    paper_pnl = -5.43
    paper_wins = 14
    paper_losses = 1
    paper_biggest_loss = -74.55

    backtest_m = calc_metrics(donch_trades)
    bt_avg_loss = np.mean([t['pnl'] for t in donch_trades if t['pnl'] < 0]) if any(t['pnl'] < 0 for t in donch_trades) else 0
    bt_avg_win = np.mean([t['pnl'] for t in donch_trades if t['pnl'] > 0]) if any(t['pnl'] > 0 for t in donch_trades) else 0
    bt_max_loss = min(t['pnl'] for t in donch_trades) if donch_trades else 0

    print(f"  Paper:    n={paper_n}, WR={paper_wins/paper_n*100:.0f}%, PnL=${paper_pnl:+.2f}, "
          f"Biggest loss=${paper_biggest_loss:+.2f}")
    print(f"  Backtest: n={backtest_m['n']}, WR={backtest_m['wr']:.0f}%, "
          f"AvgWin=${bt_avg_win:+.2f}, AvgLoss=${bt_avg_loss:+.2f}, "
          f"MaxLoss=${bt_max_loss:+.2f}")

    if bt_avg_win > 0 and abs(bt_avg_loss) > 0:
        rr = abs(bt_avg_win / bt_avg_loss)
        print(f"  Win/Loss ratio: {rr:.2f}")
        results['phase7_paper'] = {
            'paper': {'n': paper_n, 'pnl': paper_pnl, 'wr': round(paper_wins/paper_n*100, 1)},
            'backtest_avg_win': round(bt_avg_win, 2),
            'backtest_avg_loss': round(bt_avg_loss, 2),
            'backtest_max_loss': round(bt_max_loss, 2),
            'win_loss_ratio': round(rr, 2),
        }

    # ══════════════════════════════════════════════════════════════
    # Final Verdict
    # ══════════════════════════════════════════════════════════════
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    # Decision criteria:
    # 1. Standalone Sharpe > 2 in full period
    # 2. K-Fold >= 5/6 positive
    # 3. Signal overlap (±2h) < 30% with Keltner
    # 4. Portfolio Sharpe improves or stays flat
    # 5. All 3 eras have positive Sharpe
    # 6. Daily PnL correlation < 0.3 with Keltner

    checks = {
        'standalone_sharpe_gt2': backtest_m['sharpe'] > 2,
        'kfold_5of6': any(kf['positive'] >= 5 for kf in kfold_results.values()),
        'overlap_2h_lt30': overlap_results.get('2h', {}).get('pct', 100) < 30,
        'portfolio_sharpe_up': sh_6 >= sh_5,
        'all_eras_positive': all_positive,
        'daily_corr_lt03': (overlap_results.get('daily_pnl_corr') is not None and
                            abs(overlap_results.get('daily_pnl_corr', 1)) < 0.3),
    }

    for check, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}: {check}")

    pass_count = sum(checks.values())
    total_checks = len(checks)

    if pass_count >= 5:
        verdict = "DEPLOY"
    elif pass_count >= 4:
        verdict = "CAUTIOUS_DEPLOY"
    elif pass_count >= 3:
        verdict = "PAPER_ONLY"
    else:
        verdict = "REJECT"

    print(f"\n  Score: {pass_count}/{total_checks}")
    print(f"  VERDICT: {verdict}")
    results['verdict'] = {
        'checks': {k: v for k, v in checks.items()},
        'pass_count': pass_count,
        'total_checks': total_checks,
        'decision': verdict,
    }

    # Save
    elapsed = time.time() - t0
    results['elapsed_s'] = round(elapsed, 1)
    out_file = OUTPUT_DIR / 'r242_results.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nTotal time: {elapsed/60:.1f} min ({elapsed:.0f}s)")
    print(f"Saved: {out_file}")


if __name__ == '__main__':
    main()
