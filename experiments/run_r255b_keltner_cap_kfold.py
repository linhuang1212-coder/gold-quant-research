"""R255-B: Keltner MaxLoss Cap 的 K-Fold 跨折复核 (接 R255).

R255 (单周期 2022-26) 显示固定 $80 cap 优于 no-cap (Sharpe 1.292→1.357, 尾部砍 42%)。
部署前用 6-fold (2年/折) 确认 $80 在每一折都 ≥ no-cap 且削尾, 避免单窗口最优。

设置同 R255: keltner-only scan, REALISTIC_COST_BY_STRATEGY (keltner 0.66/1.00 实测滑点)。
对比 cap ∈ {0=no-cap, 60, 80, 100}, 折 = R242 同款 6 折。

判据: $80 在尽量多折 Sharpe ≥ no-cap 且 worst(最差单笔) 收窄 → K-Fold 通过。

执行: python experiments/run_r255b_keltner_cap_kfold.py
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
CAPS = [0, 60, 80, 100]   # 0 = no-cap

OUTDIR = Path(_PROJ_ROOT) / 'results' / 'r255_keltner_cap'
OUTDIR.mkdir(parents=True, exist_ok=True)

_data = None
_init_done = False


def _worker_init():
    global _data, _init_done
    if _PROJ_ROOT not in sys.path:
        sys.path.insert(0, _PROJ_ROOT)
    from backtest.runner import DataBundle
    import indicators as signals_mod
    _orig = signals_mod.scan_all_signals

    def keltner_only(df_h1, df_m15=None, **kw):
        return [s for s in (_orig(df_h1, df_m15, **kw) or []) if s.get('strategy') == 'keltner']
    signals_mod.scan_all_signals = keltner_only

    _data = DataBundle.load_default(start=DATA_START)
    _init_done = True
    print(f'  [worker {os.getpid()}] init, H1 {len(_data.h1_df)} '
          f'{_data.h1_df.index.min()}..{_data.h1_df.index.max()}', flush=True)


def _g(t, n, d=None):
    return getattr(t, n) if hasattr(t, n) else (t.get(n, d) if isinstance(t, dict) else d)


def _stats(trades):
    pnls = np.array([_g(t, 'pnl', 0) or 0 for t in trades], dtype=float)
    if len(pnls) == 0:
        return dict(n=0, wr=0.0, pnl=0.0, sharpe=0.0, dd=0.0, worst=0.0, worst5=0.0)
    cum = np.cumsum(pnls)
    dd = float((np.maximum.accumulate(cum) - cum).max())
    sh = float(pnls.mean()/pnls.std()*np.sqrt(252)) if pnls.std() > 0 else 0.0
    return dict(n=len(pnls), wr=round(float((pnls > 0).mean()*100), 1),
                pnl=round(float(pnls.sum()), 1), sharpe=round(sh, 3), dd=round(dd, 1),
                worst=round(float(pnls.min()), 1), worst5=round(float(np.sort(pnls)[:5].sum()), 1))


def _run(task):
    from backtest.engine import BacktestEngine
    from backtest.runner import REALISTIC_COST_BY_STRATEGY_KWARGS
    import indicators as signals_mod
    from indicators import get_orb_strategy
    if not _init_done:
        _worker_init()
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
        trades = eng.run()
    except Exception as ex:
        import traceback
        return {'id': task['id'], 'error': f'{ex}\n{traceback.format_exc()[:400]}'}
    kt = [t for t in trades if _g(t, 'strategy') == 'keltner']
    return {'fold': task['fold'], 'cap': task['cap'], 'stats': _stats(kt)}


def main():
    t0 = time.time()
    print('=' * 80)
    print('R255-B: Keltner Cap K-Fold (keltner-only, by_strategy cost)')
    print('=' * 80)
    tasks = [{'id': f'{f[0]}_cap{c}', 'fold': f[0], 'start': f[1], 'end': f[2], 'cap': c}
             for f in FOLDS for c in CAPS]
    with mp.Pool(min(12, mp.cpu_count()), initializer=_worker_init) as pool:
        results = pool.map(_run, tasks)
    ok = [r for r in results if 'error' not in r]
    for r in results:
        if 'error' in r:
            print(f'[ERR] {r["id"]}: {r["error"][:200]}')
    M = {(r['fold'], r['cap']): r['stats'] for r in ok}

    # Per-fold table: Sharpe + worst for each cap, mark whether $80 beats no-cap
    print(f'\n{"fold":10} | ' + ' | '.join(f'cap{c if c else "0":>3} Sh/worst' for c in CAPS))
    print('-' * 80)
    win80 = 0; tail80 = 0
    for f in FOLDS:
        fn = f[0]
        cells = []
        for c in CAPS:
            st = M.get((fn, c), {})
            cells.append(f'{st.get("sharpe",0):>5.2f}/{st.get("worst",0):>6.0f}')
        base = M.get((fn, 0), {}); c80 = M.get((fn, 80), {})
        sh_ok = c80.get('sharpe', -9) >= base.get('sharpe', 0) - 0.03   # 容忍 0.03
        tl_ok = c80.get('worst', -9e9) > base.get('worst', -9e9)        # 尾部收窄
        win80 += int(sh_ok); tail80 += int(tl_ok)
        mark = ('Sh' + ('+' if sh_ok else '-')) + ('/T' + ('+' if tl_ok else '-'))
        print(f'{fn:10} | ' + ' | '.join(f'{c:>14}' for c in cells) + f'  [$80 {mark}]')

    # aggregate: each cap's mean Sharpe + total worst across folds
    print('\n聚合 (跨 6 折):')
    print(f'{"cap":8} {"meanSharpe":>11} {"sumPnL":>10} {"minSharpe":>10} {"worstWorst":>11} {"sumWorst5":>10}')
    for c in CAPS:
        sh = [M[(f[0], c)]['sharpe'] for f in FOLDS if (f[0], c) in M]
        pnl = sum(M[(f[0], c)]['pnl'] for f in FOLDS if (f[0], c) in M)
        ww = min(M[(f[0], c)]['worst'] for f in FOLDS if (f[0], c) in M)
        w5 = sum(M[(f[0], c)]['worst5'] for f in FOLDS if (f[0], c) in M)
        lab = 'no_cap' if c == 0 else f'${c}'
        print(f'{lab:8} {np.mean(sh):>11.3f} {pnl:>10.0f} {min(sh):>10.3f} {ww:>11.0f} {w5:>10.0f}')

    print(f'\n$80 vs no-cap: Sharpe 不输的折数 {win80}/6, 尾部收窄的折数 {tail80}/6')
    print(f'Elapsed: {time.time()-t0:.0f}s')
    with open(OUTDIR / 'cap_kfold.json', 'w', encoding='utf-8') as f:
        json.dump({'folds': FOLDS, 'caps': CAPS,
                   'matrix': {f'{k[0]}|{k[1]}': v for k, v in M.items()}}, f,
                  indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
