"""R256-B: Keltner ATR 下限 + maxloss cap 组合的 K-Fold 复核 (接 R256).

R256 发现 keltner edge 由 ATR 决定, ATR<$6 的 75% 单子净亏。本实验跨 6 折忠实验证
"ATR 下限过滤"(引擎内抑制低 ATR keltner 信号 = 等同实盘那道门), 叠加 $80 cap。

配置: base / ATR≥6 / ATR≥8 / cap80 / ATR≥6+cap80 / ATR≥8+cap80
判据: ATR 下限应在**每一折**(尤其低波动 F1-F5)提升或不伤 Sharpe、把亏损折拉回; 高波动折不误伤。

忠实做法: patched scan 在当前 H1 ATR < 阈值时抑制 keltner 信号 (df_h1.iloc[-1] = 当前 bar)。

执行: python experiments/run_r256b_keltner_atrfloor_kfold.py
"""
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

_PROJ_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd

DATA_START = '2015-01-01'
FOLDS = [
    ("F1_15-17", "2015-01-01", "2017-01-01"),
    ("F2_17-19", "2017-01-01", "2019-01-01"),
    ("F3_19-21", "2019-01-01", "2021-01-01"),
    ("F4_21-23", "2021-01-01", "2023-01-01"),
    ("F5_23-25", "2023-01-01", "2025-01-01"),
    ("F6_25-26", "2025-01-01", "2026-04-09"),
]
# (label, atr_floor, maxloss_cap)
CFGS = [
    ('base',        0, 0),
    ('ATR>=6',      6, 0),
    ('ATR>=8',      8, 0),
    ('cap80',       0, 80),
    ('ATR>=6+cap80', 6, 80),
    ('ATR>=8+cap80', 8, 80),
]

OUTDIR = Path(_PROJ_ROOT) / 'results' / 'r256_keltner_regime'
OUTDIR.mkdir(parents=True, exist_ok=True)

_data = None
_init_done = False
_CFG = {'atr_floor': 0.0}   # per-task, set in _run before eng.run()


def _worker_init():
    global _data, _init_done
    if _PROJ_ROOT not in sys.path:
        sys.path.insert(0, _PROJ_ROOT)
    from backtest.runner import DataBundle
    import indicators as signals_mod
    _orig = signals_mod.scan_all_signals

    def keltner_atrfloor(df_h1, df_m15=None, **kw):
        sigs = [s for s in (_orig(df_h1, df_m15, **kw) or []) if s.get('strategy') == 'keltner']
        fl = _CFG['atr_floor']
        if fl > 0 and sigs is not None and len(df_h1):
            atr = df_h1['ATR'].iloc[-1]
            if pd.notna(atr) and atr < fl:
                return []   # 低波动: 抑制 keltner 信号 (= 实盘 ATR 下限门)
        return sigs
    signals_mod.scan_all_signals = keltner_atrfloor

    _data = DataBundle.load_default(start=DATA_START)
    _init_done = True
    print(f'  [worker {os.getpid()}] init H1 {len(_data.h1_df)}', flush=True)


def _g(t, n, d=None):
    return getattr(t, n) if hasattr(t, n) else (t.get(n, d) if isinstance(t, dict) else d)


def _stats(trades):
    p = np.array([_g(t, 'pnl', 0) or 0 for t in trades], dtype=float)
    if len(p) == 0:
        return dict(n=0, wr=0.0, pnl=0.0, sharpe=0.0, worst=0.0, worst5=0.0)
    sh = float(p.mean()/p.std()*np.sqrt(252)) if p.std() > 0 else 0.0
    return dict(n=len(p), wr=round(float((p > 0).mean()*100), 1), pnl=round(float(p.sum()), 0),
                sharpe=round(sh, 2), worst=round(float(p.min()), 0),
                worst5=round(float(np.sort(p)[:5].sum()), 0))


def _run(task):
    from backtest.engine import BacktestEngine
    from backtest.runner import REALISTIC_COST_BY_STRATEGY_KWARGS
    import indicators as signals_mod
    from indicators import get_orb_strategy
    if not _init_done:
        _worker_init()
    _CFG['atr_floor'] = task['atr_floor']
    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False
    get_orb_strategy().reset_daily()

    s = pd.Timestamp(task['start'], tz='UTC'); e = pd.Timestamp(task['end'], tz='UTC')
    m15 = _data.m15_df[(_data.m15_df.index >= s) & (_data.m15_df.index <= e)]
    h1 = _data.h1_df[(_data.h1_df.index >= s) & (_data.h1_df.index <= e)]
    kwargs = {**REALISTIC_COST_BY_STRATEGY_KWARGS, 'max_positions': 1,
              'maxloss_cap': task['cap'], 'maxloss_cap_atr_mult': 0}
    try:
        eng = BacktestEngine(m15, h1, **kwargs)
        kt = [t for t in eng.run() if _g(t, 'strategy') == 'keltner']
    except Exception as ex:
        import traceback
        return {'id': task['id'], 'error': f'{ex}\n{traceback.format_exc()[:400]}'}
    return {'cfg': task['cfg'], 'fold': task['fold'], 'stats': _stats(kt)}


def main():
    t0 = time.time()
    print('=' * 92)
    print('R256-B: Keltner ATR下限 + cap 组合 K-Fold (忠实: 引擎内抑制低ATR信号; by_strategy 成本)')
    print('=' * 92)
    tasks = []
    for (lab, fl, cap) in CFGS:
        for f in FOLDS:
            tasks.append({'id': f'{lab}|{f[0]}', 'cfg': lab, 'fold': f[0],
                          'start': f[1], 'end': f[2], 'atr_floor': fl, 'cap': cap})
    with mp.Pool(min(12, mp.cpu_count()), initializer=_worker_init) as pool:
        results = pool.map(_run, tasks)
    for r in results:
        if 'error' in r:
            print(f'[ERR] {r["id"]}: {r["error"][:200]}')
    M = {(r['cfg'], r['fold']): r['stats'] for r in results if 'error' not in r}

    folds = [f[0] for f in FOLDS]
    # per-config: per-fold Sharpe and totPnL
    print('\n### 逐折 Sharpe (每折: 该配置 keltner Sharpe)')
    print(f'{"config":14}' + ''.join(f'{f[5:]:>9}' for f in folds) + f'{"|meanSh":>9}{"posFolds":>9}')
    for (lab, _, _) in CFGS:
        shs = [M[(lab, f)]['sharpe'] for f in folds if (lab, f) in M]
        cells = ''.join(f'{M[(lab,f)]["sharpe"]:>9.2f}' for f in folds if (lab, f) in M)
        pos = sum(1 for x in shs if x > 0)
        print(f'{lab:14}{cells}{np.mean(shs):>9.2f}{pos:>6}/6')

    print('\n### 逐折 totPnL')
    print(f'{"config":14}' + ''.join(f'{f[5:]:>9}' for f in folds) + f'{"|sumPnL":>10}')
    for (lab, _, _) in CFGS:
        cells = ''.join(f'{M[(lab,f)]["pnl"]:>9.0f}' for f in folds if (lab, f) in M)
        tot = sum(M[(lab, f)]['pnl'] for f in folds if (lab, f) in M)
        print(f'{lab:14}{cells}{tot:>10.0f}')

    print('\n### 聚合 (跨 6 折)')
    print(f'{"config":14}{"sumPnL":>10}{"meanSh":>9}{"minFoldSh":>10}{"posFolds":>9}{"worstTrade":>11}{"sumWorst5":>10}')
    base_pnl = sum(M[('base', f)]['pnl'] for f in folds if ('base', f) in M)
    for (lab, _, _) in CFGS:
        present = [f for f in folds if (lab, f) in M]
        shs = [M[(lab, f)]['sharpe'] for f in present]
        tot = sum(M[(lab, f)]['pnl'] for f in present)
        ww = min(M[(lab, f)]['worst'] for f in present)
        w5 = sum(M[(lab, f)]['worst5'] for f in present)
        pos = sum(1 for x in shs if x > 0)
        d = tot - base_pnl
        print(f'{lab:14}{tot:>10.0f}{np.mean(shs):>9.2f}{min(shs):>10.2f}{pos:>6}/6  '
              f'{ww:>10.0f}{w5:>10.0f}   Δ{d:+.0f}')

    print(f'\nElapsed: {time.time()-t0:.0f}s')
    with open(OUTDIR / 'atrfloor_kfold.json', 'w', encoding='utf-8') as f:
        json.dump({'cfgs': CFGS, 'folds': FOLDS,
                   'matrix': {f'{k[0]}|{k[1]}': v for k, v in M.items()}}, f,
                  indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
