"""R256: Keltner 体制依赖深挖 (接 R255-B).

R255-B 意外发现 keltner-only 裸回测真实成本下 6 折仅近年(F6 25-26)正, 其余 5 折负。
且单 Q1 2024 就 316 笔 → 历史负 Sharpe 是真实系统性亏损, 非样本薄噪声。

本实验: 跑一次 keltner-only 全周期(2015-2026, 实测 by_strategy 成本), 每笔 pnl join 进场时
H1 体制(ATR / ADX / atr_percentile), 按 年份 / ATR分位 / ADX / 绝对ATR 分桶, 定位 edge 活在哪。
目的: ① 确认体制依赖真伪 ② 找能挡掉烂时段的体制过滤器(如 ATR/ADX 下限)。

注: 回测已含 LIVE_PARITY 的 keltner_session_adx(时段ADX阈值10-16)等过滤; 看是否还需 ATR 下限。

执行: python experiments/run_r256_keltner_regime.py
"""
import sys
import time
import warnings
from pathlib import Path

_PROJ_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

OUTDIR = Path(_PROJ_ROOT) / 'results' / 'r256_keltner_regime'
OUTDIR.mkdir(parents=True, exist_ok=True)


def _sh(pnls):
    pnls = np.asarray(pnls, dtype=float)
    if len(pnls) < 2 or pnls.std() == 0:
        return 0.0
    return float(pnls.mean()/pnls.std()*np.sqrt(252))


def _agg(df):
    p = df['pnl'].values
    return dict(n=len(p), wr=round(float((p > 0).mean()*100), 1),
                pnl=round(float(p.sum()), 0), mean=round(float(p.mean()), 2),
                sharpe=round(_sh(p), 2))


def main():
    t0 = time.time()
    from backtest.runner import DataBundle, REALISTIC_COST_BY_STRATEGY_KWARGS
    import indicators as sm
    _o = sm.scan_all_signals
    sm.scan_all_signals = lambda h1, m15=None, **k: [
        s for s in (_o(h1, m15, **k) or []) if s.get('strategy') == 'keltner']
    from backtest.engine import BacktestEngine
    from indicators import get_orb_strategy

    data = DataBundle.load_default(start='2015-01-01')
    h1 = data.h1_df
    print(f'H1 {len(h1)} bars {h1.index.min()}..{h1.index.max()}', flush=True)

    sm._friday_close_price = None
    sm._gap_traded_today = False
    get_orb_strategy().reset_daily()
    eng = BacktestEngine(data.m15_df, h1, **{**REALISTIC_COST_BY_STRATEGY_KWARGS,
                                             'max_positions': 1})
    trades = eng.run()
    kt = [t for t in trades if getattr(t, 'strategy', None) == 'keltner']
    print(f'keltner trades: {len(kt)}  ({time.time()-t0:.0f}s)', flush=True)

    # join regime at entry via H1 bar at/before entry_time
    idx = h1.index
    rows = []
    for t in kt:
        et = pd.Timestamp(t.entry_time)
        if et.tzinfo is None:
            et = et.tz_localize('UTC')
        j = idx.searchsorted(et, side='right') - 1
        if j < 0:
            continue
        bar = h1.iloc[j]
        rows.append({'entry': et, 'year': et.year, 'pnl': float(t.pnl),
                     'dir': t.direction, 'reason': getattr(t, 'exit_reason', ''),
                     'ATR': float(bar['ATR']), 'ADX': float(bar['ADX']),
                     'atr_pct': float(bar.get('atr_percentile', np.nan))})
    df = pd.DataFrame(rows)
    print(f'joined {len(df)} trades\n')

    # A. per year
    print('### A. 逐年 (keltner-only, 实测成本)')
    print(f'{"year":6}{"n":>6}{"WR%":>7}{"totPnL":>10}{"mean":>8}{"Sharpe":>8}{"meanATR":>9}{"meanADX":>9}')
    for y in sorted(df['year'].unique()):
        s = df[df.year == y]; a = _agg(s)
        print(f'{y:<6}{a["n"]:>6}{a["wr"]:>7.1f}{a["pnl"]:>10.0f}{a["mean"]:>8.2f}'
              f'{a["sharpe"]:>8.2f}{s["ATR"].mean():>9.2f}{s["ADX"].mean():>9.1f}')
    aa = _agg(df)
    print(f'{"ALL":<6}{aa["n"]:>6}{aa["wr"]:>7.1f}{aa["pnl"]:>10.0f}{aa["mean"]:>8.2f}{aa["sharpe"]:>8.2f}')

    # B. by atr_percentile quintile
    print('\n### B. 按 ATR 分位 (atr_percentile, 进场时)')
    df['atr_q'] = pd.qcut(df['atr_pct'], 5, labels=['Q1低','Q2','Q3','Q4','Q5高'], duplicates='drop')
    print(f'{"atr_q":8}{"n":>7}{"WR%":>7}{"totPnL":>10}{"mean":>8}{"Sharpe":>8}{"meanATR":>9}')
    for q in df['atr_q'].cat.categories:
        s = df[df.atr_q == q]; a = _agg(s)
        print(f'{q:8}{a["n"]:>7}{a["wr"]:>7.1f}{a["pnl"]:>10.0f}{a["mean"]:>8.2f}{a["sharpe"]:>8.2f}{s["ATR"].mean():>9.2f}')

    # C. by absolute ATR level
    print('\n### C. 按绝对 ATR ($/oz, 进场时)')
    bins = [0, 6, 10, 15, 22, 999]; labs = ['<6', '6-10', '10-15', '15-22', '>22']
    df['atr_lvl'] = pd.cut(df['ATR'], bins=bins, labels=labs)
    print(f'{"ATR$":8}{"n":>7}{"WR%":>7}{"totPnL":>10}{"mean":>8}{"Sharpe":>8}')
    for q in labs:
        s = df[df.atr_lvl == q];
        if len(s) == 0: continue
        a = _agg(s)
        print(f'{q:8}{a["n"]:>7}{a["wr"]:>7.1f}{a["pnl"]:>10.0f}{a["mean"]:>8.2f}{a["sharpe"]:>8.2f}')

    # D. by ADX bucket
    print('\n### D. 按 ADX (趋势强度, 进场时)')
    bins = [0, 20, 25, 30, 40, 999]; labs = ['<20', '20-25', '25-30', '30-40', '>40']
    df['adx_b'] = pd.cut(df['ADX'], bins=bins, labels=labs)
    print(f'{"ADX":8}{"n":>7}{"WR%":>7}{"totPnL":>10}{"mean":>8}{"Sharpe":>8}')
    for q in labs:
        s = df[df.adx_b == q]
        if len(s) == 0: continue
        a = _agg(s)
        print(f'{q:8}{a["n"]:>7}{a["wr"]:>7.1f}{a["pnl"]:>10.0f}{a["mean"]:>8.2f}{a["sharpe"]:>8.2f}')

    # E. counterfactual: drop low-ATR trades, what happens to total?
    print('\n### E. 反事实: 加 ATR 下限过滤 (只保留进场 ATR >= 阈值的 keltner 单)')
    print(f'{"ATR下限":10}{"保留n":>8}{"剔除n":>8}{"保留totPnL":>12}{"保留mean":>10}{"保留Sharpe":>11}')
    for thr in [0, 6, 8, 10, 12, 15]:
        s = df[df.ATR >= thr]; a = _agg(s)
        print(f'>= ${thr:<7}{a["n"]:>8}{len(df)-a["n"]:>8}{a["pnl"]:>12.0f}{a["mean"]:>10.2f}{a["sharpe"]:>11.2f}')

    df.to_json(OUTDIR / 'keltner_trades_regime.json', orient='records', date_format='iso')
    print(f'\nElapsed: {time.time()-t0:.0f}s | saved {OUTDIR}')


if __name__ == '__main__':
    main()
