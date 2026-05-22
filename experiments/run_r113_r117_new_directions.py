#!/usr/bin/env python3
"""
R113-R117 — 新方向探索套件 (5 experiments in 1)
================================================
R113: COT 持仓报告信号 (CFTC Commitment of Traders)
R114: 实际利率 + DXY 宏观条件化
R115: 波动率突破 / 压缩 (ATR regime)
R116: London Fix 定盘价模式
R117: Google Trends 散户情绪反向

所有实验共享 daily gold 数据, 独立输出结果.
"""
import sys, os, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path("results/r113_r117_new_directions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path("data")

PV = 100
SPREAD = 0.30
UNIT_LOT = 0.01

FOLDS = [
    ("Fold1", "2006-01-01", "2010-01-01"),
    ("Fold2", "2010-01-01", "2014-01-01"),
    ("Fold3", "2014-01-01", "2018-01-01"),
    ("Fold4", "2018-01-01", "2022-01-01"),
    ("Fold5", "2022-01-01", "2026-06-01"),
]

t0 = time.time()


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
    fold_sharpes = []
    fold_trades = []
    for fname, start, end in folds:
        fdata = data[(data.index >= start) & (data.index < end)]
        if len(fdata) < 60:
            fold_sharpes.append(0.0); fold_trades.append(0)
            continue
        trades = run_func(fdata, **kwargs)
        m = metrics(trades)
        fold_sharpes.append(m['sharpe'])
        fold_trades.append(m['n'])
    pos = sum(1 for s in fold_sharpes if s > 0)
    return {
        'fold_sharpes': [round(s, 3) for s in fold_sharpes],
        'fold_trades': fold_trades,
        'positive': pos, 'mean': round(np.mean(fold_sharpes), 3),
        'pass': pos >= 3,
    }


# ═══════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════

def load_gold_daily():
    # Use spot XAUUSD D1 (HistData M15 resampled) — same instrument as the live
    # broker. Previous version loaded xauusd_daily_yf.csv (= COMEX GC=F futures),
    # which is offset 5-10 USD from spot.
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from backtest.runner import load_d1_spot
    df = load_d1_spot(tz_naive=True)
    df = df.dropna(subset=['Close'])
    tr = pd.concat([df['High'] - df['Low'],
                     (df['High'] - df['Close'].shift()).abs(),
                     (df['Low'] - df['Close'].shift()).abs()], axis=1).max(axis=1)
    df['ATR14'] = tr.rolling(14).mean()
    df['ATR_pct'] = df['ATR14'] / df['Close'] * 100
    df['ret_1d'] = df['Close'].pct_change()
    df['ret_5d'] = df['Close'].pct_change(5)
    df['ret_20d'] = df['Close'].pct_change(20)
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()
    return df


# ═══════════════════════════════════════════════════════════════
# R113: COT (Commitment of Traders)
# ═══════════════════════════════════════════════════════════════

def download_cot_data():
    """Download COT gold futures data from CFTC."""
    print("  Downloading COT data from Quandl/CFTC...")
    try:
        url = "https://data.nasdaq.com/api/v3/datasets/CFTC/088691_FO_ALL/data.csv?api_key=DEMO_KEY"
        cot = pd.read_csv(url, parse_dates=['Date'], index_col='Date')
        cot = cot.sort_index()
        if len(cot) > 50:
            print(f"    Quandl: {len(cot)} rows ({cot.index[0].date()} ~ {cot.index[-1].date()})")
            return cot
    except Exception as e:
        print(f"    Quandl failed: {e}")

    print("  Generating synthetic COT proxy from price data...")
    gold = load_gold_daily()
    cot = pd.DataFrame(index=gold.index)
    ret20 = gold['Close'].pct_change(20)
    vol20 = gold['ret_1d'].rolling(20).std()
    cot['net_spec'] = (ret20 / vol20.replace(0, np.nan)).fillna(0)
    cot['net_spec_z'] = (cot['net_spec'] - cot['net_spec'].rolling(52*5).mean()) / \
                         cot['net_spec'].rolling(52*5).std()
    cot = cot.resample('W-FRI').last().dropna()
    print(f"    Synthetic proxy: {len(cot)} weeks")
    return cot


def run_r113(gold):
    print("\n" + "=" * 70)
    print("  R113: COT Commitment of Traders Signal")
    print("=" * 70)

    cot_raw = download_cot_data()

    if 'Non Commercial Long' in cot_raw.columns:
        cot_raw['net_spec'] = cot_raw['Non Commercial Long'] - cot_raw['Non Commercial Short']
        rm = cot_raw['net_spec'].rolling(52).mean()
        rs = cot_raw['net_spec'].rolling(52).std()
        cot_raw['net_spec_z'] = (cot_raw['net_spec'] - rm) / rs
    elif 'net_spec_z' not in cot_raw.columns:
        print("  ERROR: Cannot find COT columns")
        return {'status': 'no_data'}

    cot_weekly = cot_raw[['net_spec_z']].dropna()
    cot_daily = cot_weekly.reindex(gold.index, method='ffill')

    merged = gold.copy()
    merged['cot_z'] = cot_daily['net_spec_z']
    merged = merged.dropna(subset=['cot_z', 'ATR14'])

    print(f"  Merged data: {len(merged)} days")
    print(f"  COT Z range: [{merged['cot_z'].min():.2f}, {merged['cot_z'].max():.2f}]")

    def bt_cot_filter(df, z_long=0.5, z_short=-0.5, sl_atr=4.0, tp_atr=3.0, max_hold=20):
        c = df['Close'].values; h = df['High'].values; lo = df['Low'].values
        atr = df['ATR14'].values; cot_z = df['cot_z'].values
        sma50 = df['SMA50'].values; times = df.index; n = len(df)
        trades = []; pos = None; last_exit = -999
        for i in range(1, n):
            if pos is not None:
                bars = i - pos['bar']
                sl = pos['atr'] * sl_atr; tp = pos['atr'] * tp_atr
                if pos['dir'] == 'BUY':
                    if lo[i] <= pos['entry'] - sl:
                        pnl = -sl * UNIT_LOT * PV
                        trades.append({'dir':'BUY','entry_time':pos['time'],'exit_time':times[i],'pnl':pnl,'reason':'SL'})
                        pos = None; last_exit = i; continue
                    if h[i] >= pos['entry'] + tp:
                        pnl = tp * UNIT_LOT * PV
                        trades.append({'dir':'BUY','entry_time':pos['time'],'exit_time':times[i],'pnl':pnl,'reason':'TP'})
                        pos = None; last_exit = i; continue
                else:
                    if h[i] >= pos['entry'] + sl:
                        pnl = -sl * UNIT_LOT * PV
                        trades.append({'dir':'SELL','entry_time':pos['time'],'exit_time':times[i],'pnl':pnl,'reason':'SL'})
                        pos = None; last_exit = i; continue
                    if lo[i] <= pos['entry'] - tp:
                        pnl = tp * UNIT_LOT * PV
                        trades.append({'dir':'SELL','entry_time':pos['time'],'exit_time':times[i],'pnl':pnl,'reason':'TP'})
                        pos = None; last_exit = i; continue
                if bars >= max_hold:
                    if pos['dir'] == 'BUY':
                        pnl = (c[i] - pos['entry'] - SPREAD) * UNIT_LOT * PV
                    else:
                        pnl = (pos['entry'] - c[i] - SPREAD) * UNIT_LOT * PV
                    trades.append({'dir':pos['dir'],'entry_time':pos['time'],'exit_time':times[i],'pnl':pnl,'reason':'Time'})
                    pos = None; last_exit = i
                continue
            if i - last_exit < 2: continue
            if np.isnan(atr[i]) or atr[i] < 0.1 or np.isnan(cot_z[i]): continue
            if np.isnan(sma50[i]): continue
            if cot_z[i] > z_long and c[i] > sma50[i]:
                pos = {'dir':'BUY','entry':c[i]+SPREAD/2,'bar':i,'time':times[i],'atr':atr[i]}
            elif cot_z[i] < z_short and c[i] < sma50[i]:
                pos = {'dir':'SELL','entry':c[i]-SPREAD/2,'bar':i,'time':times[i],'atr':atr[i]}
        return trades

    grid = []
    for zl in [0.0, 0.5, 1.0, 1.5]:
        for zs in [-0.0, -0.5, -1.0, -1.5]:
            for sl in [3.0, 4.0, 5.0]:
                for tp in [2.0, 3.0, 4.0, 6.0]:
                    for mh in [10, 15, 20, 30]:
                        trades = bt_cot_filter(merged, z_long=zl, z_short=zs,
                                               sl_atr=sl, tp_atr=tp, max_hold=mh)
                        m = metrics(trades)
                        if m['n'] >= 30:
                            grid.append({'params':{'z_long':zl,'z_short':zs,'sl':sl,'tp':tp,'mh':mh}, **m})

    grid.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"\n  Grid: {len(grid)} configs with >=30 trades")
    for i, g in enumerate(grid[:5]):
        p = g['params']
        print(f"    #{i+1}: zL={p['z_long']} zS={p['z_short']} SL={p['sl']} TP={p['tp']} MH={p['mh']} "
              f"-> Sharpe={g['sharpe']}, n={g['n']}, PnL=${g['pnl']}, WR={g['wr']}%")

    kf = None
    if grid:
        p = grid[0]['params']
        kf = kfold_test(bt_cot_filter, merged, FOLDS,
                        z_long=p['z_long'], z_short=p['z_short'],
                        sl_atr=p['sl'], tp_atr=p['tp'], max_hold=p['mh'])
        status = "PASS" if kf['pass'] else "FAIL"
        print(f"  K-Fold: {kf['fold_sharpes']} -> {kf['positive']}/5 [{status}]")

    return {'grid_top5': grid[:5], 'kfold': kf, 'total_configs': len(grid)}


# ═══════════════════════════════════════════════════════════════
# R114: Real Rates + DXY Macro Conditioning
# ═══════════════════════════════════════════════════════════════

def run_r114(gold):
    print("\n" + "=" * 70)
    print("  R114: Real Rates + DXY Macro Conditioning")
    print("=" * 70)

    macro_path = DATA_DIR / "aligned_daily.csv"
    has_macro = macro_path.exists()
    df = gold.copy()

    if has_macro:
        macro = pd.read_csv(macro_path, parse_dates=['Date'], index_col='Date')
        if macro.index.tz is not None:
            macro.index = macro.index.tz_localize(None)
        for col in ['DXY', 'TIPS_10Y', 'VIX']:
            if col in macro.columns:
                df[col] = macro[col].reindex(df.index, method='ffill')
        print(f"  Macro data loaded: DXY={'DXY' in df.columns}, TIPS={'TIPS_10Y' in df.columns}")
    else:
        print("  No macro file found. Using price-derived proxies.")

    if 'DXY' not in df.columns or df['DXY'].isna().all():
        df['DXY_proxy'] = -df['ret_20d'].fillna(0) * 100
        df['DXY_mom'] = df['DXY_proxy'].rolling(20).mean()
    else:
        df['DXY_mom'] = df['DXY'].pct_change(20).fillna(0) * 100

    df['gold_mom'] = df['ret_20d'].fillna(0) * 100
    df['divergence'] = df['gold_mom'] + df['DXY_mom']
    df['atr_rank'] = df['ATR_pct'].rolling(252).rank(pct=True)

    df = df.dropna(subset=['ATR14', 'divergence'])
    print(f"  Data: {len(df)} days")

    def bt_macro_cond(data, div_long=1.0, div_short=-1.0, atr_min=0.3,
                      sl_atr=4.0, tp_atr=3.0, max_hold=15):
        c = data['Close'].values; h = data['High'].values; lo = data['Low'].values
        atr = data['ATR14'].values; div = data['divergence'].values
        atr_rk = data['atr_rank'].values if 'atr_rank' in data.columns else np.ones(len(data)) * 0.5
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
            if np.isnan(atr[i]) or atr[i] < 0.1 or np.isnan(div[i]): continue
            if np.isnan(atr_rk[i]) or atr_rk[i] < atr_min: continue
            if div[i] > div_long:
                pos = {'dir':'BUY','entry':c[i]+SPREAD/2,'bar':i,'time':times[i],'atr':atr[i]}
            elif div[i] < div_short:
                pos = {'dir':'SELL','entry':c[i]-SPREAD/2,'bar':i,'time':times[i],'atr':atr[i]}
        return trades

    grid = []
    for dl in [0.5, 1.0, 1.5, 2.0]:
        for ds in [-0.5, -1.0, -1.5, -2.0]:
            for am in [0.0, 0.2, 0.3, 0.5]:
                for sl in [3.0, 4.0, 5.0]:
                    for tp in [2.0, 3.0, 4.0]:
                        for mh in [10, 15, 20]:
                            trades = bt_macro_cond(df, div_long=dl, div_short=ds,
                                                   atr_min=am, sl_atr=sl, tp_atr=tp, max_hold=mh)
                            m = metrics(trades)
                            if m['n'] >= 30:
                                grid.append({'params':{'dl':dl,'ds':ds,'am':am,'sl':sl,'tp':tp,'mh':mh}, **m})

    grid.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"\n  Grid: {len(grid)} configs")
    for i, g in enumerate(grid[:5]):
        p = g['params']
        print(f"    #{i+1}: dL={p['dl']} dS={p['ds']} atrMin={p['am']} SL={p['sl']} TP={p['tp']} MH={p['mh']} "
              f"-> Sharpe={g['sharpe']}, n={g['n']}, PnL=${g['pnl']}, WR={g['wr']}%")

    kf = None
    if grid:
        p = grid[0]['params']
        kf = kfold_test(bt_macro_cond, df, FOLDS,
                        div_long=p['dl'], div_short=p['ds'], atr_min=p['am'],
                        sl_atr=p['sl'], tp_atr=p['tp'], max_hold=p['mh'])
        status = "PASS" if kf['pass'] else "FAIL"
        print(f"  K-Fold: {kf['fold_sharpes']} -> {kf['positive']}/5 [{status}]")

    return {'grid_top5': grid[:5], 'kfold': kf, 'total_configs': len(grid)}


# ═══════════════════════════════════════════════════════════════
# R115: Volatility Breakout / Compression
# ═══════════════════════════════════════════════════════════════

def run_r115(gold):
    print("\n" + "=" * 70)
    print("  R115: Volatility Breakout / Compression")
    print("=" * 70)

    df = gold.copy()
    df['atr_rank_60'] = df['ATR_pct'].rolling(60).rank(pct=True)
    df['atr_rank_252'] = df['ATR_pct'].rolling(252).rank(pct=True)
    df['atr_expansion'] = df['ATR14'] / df['ATR14'].rolling(20).mean()
    df['bb_width'] = df['Close'].rolling(20).std() / df['Close'].rolling(20).mean()
    df['bb_squeeze'] = df['bb_width'].rolling(120).rank(pct=True)
    df = df.dropna(subset=['atr_rank_252', 'bb_squeeze', 'ATR14'])
    print(f"  Data: {len(df)} days")

    def bt_vol_breakout(data, squeeze_max=0.2, expansion_min=1.3,
                        sl_atr=3.0, tp_atr=4.0, max_hold=10):
        """Enter when vol compresses then expands (breakout)."""
        c = data['Close'].values; h = data['High'].values; lo = data['Low'].values
        atr = data['ATR14'].values
        squeeze = data['bb_squeeze'].values
        expansion = data['atr_expansion'].values
        sma50 = data['SMA50'].values if 'SMA50' in data.columns else np.full(len(data), np.nan)
        times = data.index; n = len(data)
        trades = []; pos = None; last_exit = -999
        prev_squeeze = np.full(n, np.nan)
        for i in range(5, n):
            prev_squeeze[i] = squeeze[i-5]
        for i in range(6, n):
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
            if np.isnan(atr[i]) or atr[i] < 0.1: continue
            if np.isnan(prev_squeeze[i]) or np.isnan(expansion[i]): continue
            was_compressed = prev_squeeze[i] < squeeze_max
            now_expanding = expansion[i] > expansion_min
            if not (was_compressed and now_expanding): continue
            if not np.isnan(sma50[i]) and c[i] > sma50[i]:
                pos = {'dir':'BUY','entry':c[i]+SPREAD/2,'bar':i,'time':times[i],'atr':atr[i]}
            elif not np.isnan(sma50[i]) and c[i] < sma50[i]:
                pos = {'dir':'SELL','entry':c[i]-SPREAD/2,'bar':i,'time':times[i],'atr':atr[i]}
        return trades

    grid = []
    for sq in [0.1, 0.15, 0.2, 0.25, 0.3]:
        for ex in [1.1, 1.2, 1.3, 1.5, 1.8]:
            for sl in [2.0, 3.0, 4.0]:
                for tp in [2.0, 3.0, 4.0, 6.0]:
                    for mh in [5, 10, 15, 20]:
                        trades = bt_vol_breakout(df, squeeze_max=sq, expansion_min=ex,
                                                 sl_atr=sl, tp_atr=tp, max_hold=mh)
                        m = metrics(trades)
                        if m['n'] >= 20:
                            grid.append({'params':{'sq':sq,'ex':ex,'sl':sl,'tp':tp,'mh':mh}, **m})

    grid.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"\n  Grid: {len(grid)} configs")
    for i, g in enumerate(grid[:5]):
        p = g['params']
        print(f"    #{i+1}: sq={p['sq']} ex={p['ex']} SL={p['sl']} TP={p['tp']} MH={p['mh']} "
              f"-> Sharpe={g['sharpe']}, n={g['n']}, PnL=${g['pnl']}, WR={g['wr']}%")

    kf = None
    if grid:
        p = grid[0]['params']
        kf = kfold_test(bt_vol_breakout, df, FOLDS,
                        squeeze_max=p['sq'], expansion_min=p['ex'],
                        sl_atr=p['sl'], tp_atr=p['tp'], max_hold=p['mh'])
        status = "PASS" if kf['pass'] else "FAIL"
        print(f"  K-Fold: {kf['fold_sharpes']} -> {kf['positive']}/5 [{status}]")

    return {'grid_top5': grid[:5], 'kfold': kf, 'total_configs': len(grid)}


# ═══════════════════════════════════════════════════════════════
# R116: London Fix Anomaly
# ═══════════════════════════════════════════════════════════════

def run_r116(gold):
    print("\n" + "=" * 70)
    print("  R116: London Fix / Time-of-Day Anomaly (Daily proxy)")
    print("=" * 70)

    df = gold.copy()
    df['dow'] = df.index.dayofweek
    df['month'] = df.index.month
    df['overnight_ret'] = (df['Open'] - df['Close'].shift()) / df['Close'].shift() * 100
    df['intraday_ret'] = (df['Close'] - df['Open']) / df['Open'] * 100
    df = df.dropna(subset=['overnight_ret', 'intraday_ret', 'ATR14'])

    print(f"  Data: {len(df)} days\n")

    print("  Overnight vs Intraday returns:")
    print(f"    Overnight mean: {df['overnight_ret'].mean():.4f}%")
    print(f"    Intraday mean:  {df['intraday_ret'].mean():.4f}%")

    print("\n  By day-of-week (0=Mon, 4=Fri):")
    for dow in range(5):
        sub = df[df['dow'] == dow]
        print(f"    {['Mon','Tue','Wed','Thu','Fri'][dow]}: overnight={sub['overnight_ret'].mean():.4f}%, "
              f"intraday={sub['intraday_ret'].mean():.4f}%, n={len(sub)}")

    print("\n  By month:")
    for m in range(1, 13):
        sub = df[df['month'] == m]
        if len(sub) > 0:
            total = sub['ret_1d'].mean() * 100
            print(f"    {m:2d}: mean_ret={total:.4f}%, n={len(sub)}")

    def bt_overnight(data, direction='BUY', dow_filter=None, sl_atr=2.0, tp_atr=1.5):
        """Buy at close, sell at next open (overnight hold)."""
        c = data['Close'].values; o = data['Open'].values
        atr = data['ATR14'].values; dows = data['dow'].values
        times = data.index; n = len(data)
        trades = []
        for i in range(1, n):
            if np.isnan(atr[i-1]) or atr[i-1] < 0.1: continue
            if dow_filter is not None and dows[i-1] not in dow_filter: continue
            if direction == 'BUY':
                pnl = (o[i] - c[i-1] - SPREAD) * UNIT_LOT * PV
                sl = atr[i-1] * sl_atr
                if o[i] - c[i-1] < -sl:
                    pnl = -sl * UNIT_LOT * PV
            else:
                pnl = (c[i-1] - o[i] - SPREAD) * UNIT_LOT * PV
                sl = atr[i-1] * sl_atr
                if c[i-1] - o[i] < -sl:
                    pnl = -sl * UNIT_LOT * PV
            trades.append({'dir':direction,'entry_time':times[i-1],'exit_time':times[i],'pnl':pnl,'reason':'Overnight'})
        return trades

    def bt_intraday(data, direction='BUY', dow_filter=None, sl_atr=2.0, tp_atr=1.5):
        """Buy at open, sell at close (intraday hold)."""
        c = data['Close'].values; o = data['Open'].values
        atr = data['ATR14'].values; dows = data['dow'].values
        times = data.index; n = len(data)
        trades = []
        for i in range(n):
            if np.isnan(atr[i]) or atr[i] < 0.1: continue
            if dow_filter is not None and dows[i] not in dow_filter: continue
            if direction == 'BUY':
                pnl = (c[i] - o[i] - SPREAD) * UNIT_LOT * PV
            else:
                pnl = (o[i] - c[i] - SPREAD) * UNIT_LOT * PV
            trades.append({'dir':direction,'entry_time':times[i],'exit_time':times[i],'pnl':pnl,'reason':'Intraday'})
        return trades

    results_116 = {}
    for label, func, direction, dow in [
        ('Overnight_BUY_All', bt_overnight, 'BUY', None),
        ('Overnight_SELL_All', bt_overnight, 'SELL', None),
        ('Intraday_BUY_All', bt_intraday, 'BUY', None),
        ('Intraday_SELL_All', bt_intraday, 'SELL', None),
        ('Overnight_BUY_MonTue', bt_overnight, 'BUY', [0, 1]),
        ('Overnight_BUY_ThuFri', bt_overnight, 'BUY', [3, 4]),
        ('Intraday_BUY_WedThu', bt_intraday, 'BUY', [2, 3]),
    ]:
        trades = func(df, direction=direction, dow_filter=dow)
        m = metrics(trades)
        results_116[label] = m
        print(f"  {label:30s}: n={m['n']:5d}, Sharpe={m['sharpe']:6.3f}, PnL=${m['pnl']:8.0f}, WR={m['wr']:.1f}%")

    best_label = max(results_116, key=lambda k: results_116[k]['sharpe'])
    print(f"\n  Best: {best_label} (Sharpe={results_116[best_label]['sharpe']})")

    best_trades_func = bt_overnight if 'Overnight' in best_label else bt_intraday
    best_dir = 'BUY' if 'BUY' in best_label else 'SELL'
    best_dow = None
    if 'MonTue' in best_label: best_dow = [0, 1]
    elif 'ThuFri' in best_label: best_dow = [3, 4]
    elif 'WedThu' in best_label: best_dow = [2, 3]

    kf = kfold_test(best_trades_func, df, FOLDS, direction=best_dir, dow_filter=best_dow)
    status = "PASS" if kf['pass'] else "FAIL"
    print(f"  K-Fold ({best_label}): {kf['fold_sharpes']} -> {kf['positive']}/5 [{status}]")

    return {'strategies': results_116, 'best': best_label, 'kfold': kf}


# ═══════════════════════════════════════════════════════════════
# R117: Google Trends / Sentiment Proxy
# ═══════════════════════════════════════════════════════════════

def run_r117(gold):
    print("\n" + "=" * 70)
    print("  R117: Sentiment / Retail Crowd Proxy")
    print("=" * 70)

    df = gold.copy()
    df['vol_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    df['ret_5d_abs'] = df['ret_5d'].abs()
    df['momentum_crowd'] = df['ret_20d'].rolling(10).mean()
    df['crowd_z'] = (df['momentum_crowd'] - df['momentum_crowd'].rolling(252).mean()) / \
                     df['momentum_crowd'].rolling(252).std()
    df['extreme_up'] = (df['crowd_z'] > 2.0).astype(float)
    df['extreme_down'] = (df['crowd_z'] < -2.0).astype(float)
    df = df.dropna(subset=['crowd_z', 'ATR14'])
    print(f"  Data: {len(df)} days")
    print(f"  Crowd Z range: [{df['crowd_z'].min():.2f}, {df['crowd_z'].max():.2f}]")
    print(f"  Extreme up days: {df['extreme_up'].sum():.0f} ({df['extreme_up'].mean()*100:.1f}%)")
    print(f"  Extreme down days: {df['extreme_down'].sum():.0f} ({df['extreme_down'].mean()*100:.1f}%)")

    def bt_contrarian(data, z_sell=2.0, z_buy=-2.0, sl_atr=3.0, tp_atr=2.0, max_hold=10):
        """Contrarian: sell when crowd extremely bullish, buy when extremely bearish."""
        c = data['Close'].values; h = data['High'].values; lo = data['Low'].values
        atr = data['ATR14'].values; cz = data['crowd_z'].values
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

    grid = []
    for zs in [1.5, 2.0, 2.5, 3.0]:
        for zb in [-1.5, -2.0, -2.5, -3.0]:
            for sl in [2.0, 3.0, 4.0]:
                for tp in [1.5, 2.0, 3.0, 4.0]:
                    for mh in [5, 10, 15, 20]:
                        trades = bt_contrarian(df, z_sell=zs, z_buy=zb,
                                               sl_atr=sl, tp_atr=tp, max_hold=mh)
                        m = metrics(trades)
                        if m['n'] >= 20:
                            grid.append({'params':{'zs':zs,'zb':zb,'sl':sl,'tp':tp,'mh':mh}, **m})

    grid.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"\n  Grid: {len(grid)} configs")
    for i, g in enumerate(grid[:5]):
        p = g['params']
        print(f"    #{i+1}: zS={p['zs']} zB={p['zb']} SL={p['sl']} TP={p['tp']} MH={p['mh']} "
              f"-> Sharpe={g['sharpe']}, n={g['n']}, PnL=${g['pnl']}, WR={g['wr']}%")

    kf = None
    if grid:
        p = grid[0]['params']
        kf = kfold_test(bt_contrarian, df, FOLDS,
                        z_sell=p['zs'], z_buy=p['zb'],
                        sl_atr=p['sl'], tp_atr=p['tp'], max_hold=p['mh'])
        status = "PASS" if kf['pass'] else "FAIL"
        print(f"  K-Fold: {kf['fold_sharpes']} -> {kf['positive']}/5 [{status}]")

    return {'grid_top5': grid[:5], 'kfold': kf, 'total_configs': len(grid)}


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("  R113-R117: New Directions Research Suite")
    print("=" * 80)

    gold = load_gold_daily()
    print(f"  Gold daily: {len(gold)} bars ({gold.index[0].date()} ~ {gold.index[-1].date()})")

    all_results = {}

    all_results['R113_COT'] = run_r113(gold)
    all_results['R114_Macro'] = run_r114(gold)
    all_results['R115_Vol'] = run_r115(gold)
    all_results['R116_Fix'] = run_r116(gold)
    all_results['R117_Sentiment'] = run_r117(gold)

    elapsed = time.time() - t0
    all_results['elapsed_s'] = round(elapsed, 1)

    # Summary
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    for exp_id, res in all_results.items():
        if exp_id == 'elapsed_s': continue
        kf = res.get('kfold')
        if kf:
            status = "PASS" if kf['pass'] else "FAIL"
            top = res.get('grid_top5', res.get('strategies', []))
            best_sharpe = top[0]['sharpe'] if isinstance(top, list) and top else 'N/A'
            print(f"  {exp_id:20s}: K-Fold {kf['positive']}/5 [{status}], "
                  f"mean={kf['mean']}, best_sharpe={best_sharpe}")
        else:
            print(f"  {exp_id:20s}: No K-Fold data")

    out_file = OUTPUT_DIR / "r113_r117_results.json"
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n  R113-R117 COMPLETE -- {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Saved: {out_file}")


if __name__ == '__main__':
    main()
