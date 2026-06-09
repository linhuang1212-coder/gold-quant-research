"""R253: DualThrust 真实滑点重验 (live loss diagnosis).

背景: 实盘 dual_thrust 每个时间窗都亏 (全期-$182/30d-$180/7d-$86, WR 47-50%),
与 R251/R252 验证 (OOS Sharpe +0.665, PBO 0%, DSR 1.0) 矛盾。

诊断 (2026-06-08, 243 笔实盘成交): dual_thrust 真实入场滑点
  BUY +$1.68 (中位 1.08) / SELL +$1.89 (中位 1.11)
远高于 REALISTIC_COST_KWARGS 校准值 (BUY +0.67 / SELL +0.17)。
原因: dual_thrust 是突破策略, 进场在点差最宽/续走最猛的瞬间, 滑点天然更大;
而校准用的 91 笔以 keltner 为主, 低估了突破策略的真实成本。

本实验: 把 act010_dist010 (实盘配置) 在 4 个成本档下重测, 看 edge 是否在真实滑点下崩塌。
  nocost            — LIVE_PARITY (无滑点) 上界
  calib_0.67/0.17   — R251/R252 用的校准值 (fixed)
  measured_med      — 实测中位 1.08/1.11 (fixed, 较保守下界)
  measured_mean     — 实测均值 1.70/1.90 (fixed, 期望成本)

执行: python experiments/run_r253_dt_realcost_revalidation.py
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

import numpy as np
import pandas as pd

PERIOD_START = '2022-01-01'
PERIOD_END = '2026-05-13'

ERAS = [
    ('2022-2023', '2022-01-01', '2023-12-31'),
    ('2024-2025', '2024-01-01', '2025-12-31'),
    ('2026',      '2026-01-01', '2026-05-13'),
]

# dual_thrust trail 配置: 实盘用的 act=0.10/dist=0.01, + baseline(trail off) 作参照
CONFIGS = {
    'act010_dist010': {'dual_thrust_trail_enabled': True,
                       'dual_thrust_trail_act': 0.10, 'dual_thrust_trail_dist': 0.01},
    'baseline_off':   {'dual_thrust_trail_enabled': False},
}

OUTDIR = Path(_PROJ_ROOT) / 'results' / 'r253_dt_realcost'
OUTDIR.mkdir(parents=True, exist_ok=True)

_data = None
_init_done = False


def _worker_init():
    """与 R251/R252 同款 patch, 注入 dual_thrust 信号。"""
    global _data, _init_done
    if _PROJ_ROOT not in sys.path:
        sys.path.insert(0, _PROJ_ROOT)
    from backtest.runner import DataBundle
    import indicators as signals_mod
    from experiments.run_r209v2_non_keltner_audit import (
        check_psar_signal, check_sess_bo_signal,
        check_dual_thrust_signal, check_chandelier_signal,
    )
    import experiments.run_r209v2_non_keltner_audit as r209_mod

    LIVE5_CONFIG = {
        'psar': {'enabled': False},
        'sess_bo': {'enabled': True, 'broker_gmt_offset': 0,
                    'session_hour_gmt': 12, 'lookback_bars': 4,
                    'sl_atr': 4.5, 'tp_atr': 4.0},
        'dual_thrust': {'enabled': True, 'n_bars': 6, 'k_up': 0.5, 'k_down': 0.5,
                        'sl_atr': 6.0, 'tp_atr': 8.0,
                        'trail_act_atr': 0.06, 'trail_dist_atr': 0.01},
        'chandelier': {'enabled': False},
    }
    r209_mod.LIVE_STRAT_CONFIGS = LIVE5_CONFIG
    original_scan = signals_mod.scan_all_signals

    def patched_scan(df_h1, df_m15=None, **kwargs):
        signals = original_scan(df_h1, df_m15, **kwargs) or []
        extras = []
        for fn in [check_psar_signal, check_sess_bo_signal,
                   check_dual_thrust_signal, check_chandelier_signal]:
            sig = fn(df_h1)
            if sig is not None:
                extras.append(sig)
        return signals + extras
    signals_mod.scan_all_signals = patched_scan

    _data = DataBundle.load_default(start=PERIOD_START)
    _init_done = True
    print(f'  [worker {os.getpid()}] init done, H1 bars: {len(_data.h1_df)}', flush=True)


def _build_cost_presets():
    from backtest.runner import LIVE_PARITY_KWARGS, REALISTIC_COST_KWARGS

    def fixed_cost(sb, ss):
        # realistic 时段点差 + 固定方向滑点 (apples-to-apples 只变滑点幅度)
        return {**REALISTIC_COST_KWARGS, 'slippage_model': 'fixed',
                'slippage_buy': sb, 'slippage_sell': ss}

    return {
        'nocost':          dict(LIVE_PARITY_KWARGS),
        'calib_0.67_0.17': fixed_cost(0.67, 0.17),
        'measured_med':    fixed_cost(1.08, 1.11),
        'measured_mean':   fixed_cost(1.70, 1.90),
    }


def _trade_attr(t, name, default=None):
    if hasattr(t, name):
        return getattr(t, name)
    if isinstance(t, dict):
        return t.get(name, default)
    return default


def _calc_stats(trades):
    if not trades:
        return {'n': 0, 'sharpe': 0.0, 'pnl': 0.0, 'win_rate': 0.0,
                'mean_pnl': 0.0, 'max_dd': 0.0}
    pnls = np.array([_trade_attr(t, 'pnl', 0) for t in trades], dtype=float)
    wr = float((pnls > 0).mean() * 100)
    cum = np.cumsum(pnls)
    dd = float((np.maximum.accumulate(cum) - cum).max())
    sharpe = (float(pnls.mean() / pnls.std() * np.sqrt(252))
              if pnls.std() > 0 else 0.0)
    return {'n': len(pnls), 'sharpe': round(sharpe, 3),
            'pnl': round(float(pnls.sum()), 2),
            'win_rate': round(wr, 2),
            'mean_pnl': round(float(pnls.mean()), 2),
            'max_dd': round(dd, 2)}


def _filter_period(trades, start, end):
    s = pd.Timestamp(start, tz='UTC'); e = pd.Timestamp(end, tz='UTC')
    out = []
    for t in trades:
        et = _trade_attr(t, 'exit_time') or _trade_attr(t, 'entry_time')
        if et is None:
            continue
        if isinstance(et, str):
            et = pd.Timestamp(et)
        if et.tzinfo is None:
            et = et.tz_localize('UTC')
        if s <= et <= e:
            out.append(t)
    return out


def _run_single(task):
    from backtest.engine import BacktestEngine
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

    presets = _build_cost_presets()
    kwargs = {**presets[task['cost']], **CONFIGS[task['config']],
              'maxloss_cap': 0, 'max_positions': 4}

    t0 = time.time()
    try:
        eng = BacktestEngine(m15, h1, **kwargs)
        all_trades = eng.run()
    except Exception as ex:
        import traceback
        return {'id': task['id'], 'error': f'{ex}\n{traceback.format_exc()[:600]}'}

    dt = [t for t in all_trades if _trade_attr(t, 'strategy') == 'dual_thrust']
    eras = {nm: _calc_stats(_filter_period(dt, a, b)) for nm, a, b in ERAS}
    return {'id': task['id'], 'config': task['config'], 'cost': task['cost'],
            'dt': _calc_stats(dt), 'eras': eras, 'elapsed': round(time.time() - t0, 1)}


def main():
    t0 = time.time()
    print('=' * 78)
    print('R253: DualThrust 真实滑点重验')
    print('=' * 78)
    presets = _build_cost_presets()
    tasks = [{'id': f'{c}__{k}', 'config': c, 'cost': k}
             for c in CONFIGS for k in presets]
    print(f'Period: {PERIOD_START} ~ {PERIOD_END} | configs={len(CONFIGS)} × costs={len(presets)} = {len(tasks)} tasks\n')

    n_workers = min(8, mp.cpu_count())
    with mp.Pool(n_workers, initializer=_worker_init) as pool:
        results = pool.map(_run_single, tasks)

    ok = [r for r in results if 'error' not in r]
    for r in results:
        if 'error' in r:
            print(f'[ERR] {r["id"]}: {r["error"][:300]}', flush=True)

    COST_ORDER = ['nocost', 'calib_0.67_0.17', 'measured_med', 'measured_mean']
    by = {(r['config'], r['cost']): r for r in ok}

    for cfg in CONFIGS:
        print('\n' + '=' * 78)
        print(f'CONFIG: {cfg}   (dual_thrust trades only, {PERIOD_START}~{PERIOD_END})')
        print('-' * 78)
        print(f'{"cost":18} {"n":>4} {"WR%":>6} {"totPnL":>10} {"meanPnL":>9} {"Sharpe":>8} {"maxDD":>9}')
        for cost in COST_ORDER:
            r = by.get((cfg, cost))
            if not r:
                continue
            d = r['dt']
            print(f'{cost:18} {d["n"]:>4} {d["win_rate"]:>6.1f} {d["pnl"]:>10.1f} '
                  f'{d["mean_pnl"]:>9.2f} {d["sharpe"]:>8.3f} {d["max_dd"]:>9.1f}')
        # 分时代 (act010 配置)
    print('\n' + '=' * 78)
    print('act010_dist010 — 分时代 totPnL / meanPnL (看近年 2026 是否最差)')
    print('-' * 78)
    print(f'{"cost":18} | ' + ' | '.join(f'{nm:>16}' for nm, _, _ in ERAS))
    for cost in COST_ORDER:
        r = by.get(('act010_dist010', cost))
        if not r:
            continue
        cells = []
        for nm, _, _ in ERAS:
            s = r['eras'][nm]
            cells.append(f'{s["pnl"]:>7.0f}/{s["mean_pnl"]:>+5.2f}({s["n"]})')
        print(f'{cost:18} | ' + ' | '.join(f'{c:>16}' for c in cells))

    outfile = OUTDIR / 'revalidation.json'
    with open(outfile, 'w', encoding='utf-8') as f:
        json.dump({'period': [PERIOD_START, PERIOD_END], 'eras': ERAS,
                   'results': ok}, f, indent=2, default=str, ensure_ascii=False)
    print(f'\nElapsed: {time.time()-t0:.0f}s | Saved: {outfile}', flush=True)


if __name__ == '__main__':
    main()
