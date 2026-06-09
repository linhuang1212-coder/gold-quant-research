"""R255: Keltner MaxLoss Cap 重验 (真实成本下, 接 R253/R254).

背景: 实盘 keltner maxloss_cap=None (Trail-First, 当初 R202 回测 no-cap Sharpe 5.675 > cap 5.532)。
但 2026-06-08 一笔 keltner SELL 逆势扛到 2h 时间止损, 亏 -$220 (ATR 高 $23 → 止损天然宽),
吃掉当天 9 笔小赢。用户问: 加个 cap 是否能砍掉这种尾部而不伤总 Sharpe?
而且"no-cap 更好"来自同一套回测框架, 该框架刚在 dual_thrust 上被证明低估成本 → 值得在真实成本下重测。

设计: scan 过滤成 keltner-only (干净隔离), 在 REALISTIC_COST_BY_STRATEGY (keltner 0.66/1.00 实测滑点) 下,
并行扫两组 cap:
  fixed_$   : maxloss_cap ∈ {0,40,60,80,100,120}  (0=现状 no-cap)
  atr_mult  : maxloss_cap_atr_mult ∈ {0,1.5,2.0,2.5,3.0}  (cap = mult×ATR×lots×PV, 随波动自适应)
注: 引擎 keltner SL=3.5×ATR (LIVE_PARITY); cap 在 SL/trailing 之前每 bar 查。

判据: 好的 cap 应**砍掉 maxDD + 最差单笔/尾部**, 同时**总 PnL/Sharpe 不显著低于** no-cap 基线。

执行: python experiments/run_r255_keltner_cap_sweep.py
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

PERIOD_START = '2022-01-01'
PERIOD_END = '2026-04-09'
RECENT = ('2024-01-01', '2026-04-09')

FIXED_CAPS = [0, 40, 60, 80, 100, 120]        # $ (0 = no cap = 现状)
ATR_MULTS = [1.5, 2.0, 2.5, 3.0]              # ×ATR

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
        sigs = _orig(df_h1, df_m15, **kw) or []
        return [s for s in sigs if s.get('strategy') == 'keltner']
    signals_mod.scan_all_signals = keltner_only

    _data = DataBundle.load_default(start=PERIOD_START)
    _init_done = True
    print(f'  [worker {os.getpid()}] init done, H1 {len(_data.h1_df)}', flush=True)


def _g(t, n, d=None):
    return getattr(t, n) if hasattr(t, n) else (t.get(n, d) if isinstance(t, dict) else d)


def _stats(trades):
    pnls = np.array([_g(t, 'pnl', 0) or 0 for t in trades], dtype=float)
    if len(pnls) == 0:
        return dict(n=0, wr=0.0, pnl=0.0, mean=0.0, sharpe=0.0, dd=0.0,
                    worst=0.0, worst5=0.0)
    cum = np.cumsum(pnls)
    dd = float((np.maximum.accumulate(cum) - cum).max())
    sh = float(pnls.mean() / pnls.std() * np.sqrt(252)) if pnls.std() > 0 else 0.0
    worst5 = float(np.sort(pnls)[:5].sum())
    return dict(n=len(pnls), wr=round(float((pnls > 0).mean()*100), 1),
                pnl=round(float(pnls.sum()), 1), mean=round(float(pnls.mean()), 2),
                sharpe=round(sh, 3), dd=round(dd, 1),
                worst=round(float(pnls.min()), 1), worst5=round(worst5, 1))


def _filt(trades, a, b):
    s = pd.Timestamp(a, tz='UTC'); e = pd.Timestamp(b, tz='UTC')
    out = []
    for t in trades:
        et = _g(t, 'exit_time')
        if et is None:
            continue
        if isinstance(et, str):
            et = pd.Timestamp(et)
        if et.tzinfo is None:
            et = et.tz_localize('UTC')
        if s <= et <= e:
            out.append(t)
    return out


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

    s = pd.Timestamp(PERIOD_START, tz='UTC'); e = pd.Timestamp(PERIOD_END, tz='UTC')
    m15 = _data.m15_df[(_data.m15_df.index >= s) & (_data.m15_df.index <= e)]
    h1 = _data.h1_df[(_data.h1_df.index >= s) & (_data.h1_df.index <= e)]

    kwargs = {**REALISTIC_COST_BY_STRATEGY_KWARGS, 'max_positions': 1,
              'maxloss_cap': task['fixed'], 'maxloss_cap_atr_mult': task['atr_mult']}
    t0 = time.time()
    try:
        eng = BacktestEngine(m15, h1, **kwargs)
        trades = eng.run()
        cap_n = eng.maxloss_cap_count
    except Exception as ex:
        import traceback
        return {'id': task['id'], 'error': f'{ex}\n{traceback.format_exc()[:500]}'}
    kt = [t for t in trades if _g(t, 'strategy') == 'keltner']
    return {'id': task['id'], 'label': task['label'], 'group': task['group'],
            'cap_triggers': cap_n,
            'all': _stats(kt), 'recent': _stats(_filt(kt, *RECENT)),
            'elapsed': round(time.time()-t0, 1)}


def main():
    t0 = time.time()
    print('=' * 80)
    print(f'R255: Keltner MaxLoss Cap 重验  {PERIOD_START}~{PERIOD_END} (keltner-only, by_strategy cost)')
    print('=' * 80)
    tasks = []
    for c in FIXED_CAPS:
        lab = 'no_cap' if c == 0 else f'fixed_${c}'
        tasks.append({'id': lab, 'label': lab, 'group': 'fixed', 'fixed': c, 'atr_mult': 0})
    for m in ATR_MULTS:
        tasks.append({'id': f'atr_{m}x', 'label': f'{m}xATR', 'group': 'atr',
                      'fixed': 0, 'atr_mult': m})

    n_workers = min(10, mp.cpu_count())
    with mp.Pool(n_workers, initializer=_worker_init) as pool:
        results = pool.map(_run, tasks)

    ok = [r for r in results if 'error' not in r]
    for r in results:
        if 'error' in r:
            print(f'[ERR] {r["id"]}: {r["error"][:300]}', flush=True)

    base = next((r for r in ok if r['label'] == 'no_cap'), None)
    bpnl = base['all']['pnl'] if base else 0

    def show(group, title):
        print(f'\n### {title}')
        print(f'{"cap":12} {"n":>4} {"WR%":>6} {"totPnL":>9} {"Δvs no-cap":>11} {"Sharpe":>7} '
              f'{"maxDD":>8} {"worst":>8} {"worst5":>9} {"capHit":>7} | {"rec_PnL":>8} {"rec_Sh":>7}')
        rows = [r for r in ok if r['group'] == group]
        if group == 'atr' and base:
            rows = [base] + rows
        for r in rows:
            a = r['all']; rc = r['recent']
            dpnl = a['pnl'] - bpnl
            print(f'{r["label"]:12} {a["n"]:>4} {a["wr"]:>6.1f} {a["pnl"]:>9.1f} {dpnl:>+11.1f} '
                  f'{a["sharpe"]:>7.3f} {a["dd"]:>8.1f} {a["worst"]:>8.1f} {a["worst5"]:>9.1f} '
                  f'{r["cap_triggers"]:>7} | {rc["pnl"]:>8.1f} {rc["sharpe"]:>7.3f}')

    show('fixed', '固定 $ cap (0 = 现状 no-cap)')
    show('atr', 'ATR 比例 cap (cap = mult×ATR×lots×PV; no_cap 作基线)')

    print(f'\n判读: cap 好 = worst/worst5/maxDD 明显变小 且 totPnL/Sharpe 不显著低于 no-cap。')
    print(f'Elapsed: {time.time()-t0:.0f}s')
    with open(OUTDIR / 'cap_sweep.json', 'w', encoding='utf-8') as f:
        json.dump({'period': [PERIOD_START, PERIOD_END], 'results': ok}, f,
                  indent=2, default=str, ensure_ascii=False)


if __name__ == '__main__':
    main()
