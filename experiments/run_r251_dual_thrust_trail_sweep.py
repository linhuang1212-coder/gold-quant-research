"""R251: DualThrust trailing 经验调参 sweep.

目标: 用户痛点 "只赚几块就走" — 找到最优 dual_thrust trail_act_atr / trail_dist_atr 组合.

参数网格 (5×5 + baseline):
  trail_act_atr  ∈ {0.06, 0.10, 0.20, 0.40, 0.80}  (浮盈激活阈值, 越大越晚激活)
  trail_dist_atr ∈ {0.01, 0.03, 0.08, 0.15, 0.30}  (跟踪距离, 越大越宽容回撤)
  + baseline (trail off, 仅 SL/TP 出场)

执行: python experiments/run_r251_dual_thrust_trail_sweep.py
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

# ═══════════════════════════════════════════════════════════════
# Sweep 网格
# ═══════════════════════════════════════════════════════════════
TRAIL_ACT_GRID = [0.06, 0.10, 0.20, 0.40, 0.80]
TRAIL_DIST_GRID = [0.01, 0.03, 0.08, 0.15, 0.30]

CONFIGS = {'baseline_off': {'dual_thrust_trail_enabled': False}}
for act in TRAIL_ACT_GRID:
    for dist in TRAIL_DIST_GRID:
        if dist > act:
            continue  # 跳过 dist > act 的无意义组合 (永远不会激活)
        name = f'act{int(act*100):03d}_dist{int(dist*1000):03d}'
        CONFIGS[name] = {
            'dual_thrust_trail_enabled': True,
            'dual_thrust_trail_act': act,
            'dual_thrust_trail_dist': dist,
        }

OUTDIR = Path(__file__).resolve().parents[1] / 'results' / 'r251_dual_thrust_trail'
OUTDIR.mkdir(parents=True, exist_ok=True)

ERAS = [
    ('2019-2021', '2019-01-01', '2021-12-31'),
    ('2022-2023', '2022-01-01', '2023-12-31'),
    ('2024-2025', '2024-01-01', '2025-12-31'),
    ('2026',      '2026-01-01', '2026-05-13'),
]

# ═══════════════════════════════════════════════════════════════
# Worker init: patch scan_all_signals 加 dual_thrust signal generator
# ═══════════════════════════════════════════════════════════════
_data = None
_init_done = False


def _worker_init():
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

    _data = DataBundle.load_default()
    _init_done = True
    print(f'  [worker {os.getpid()}] init done', flush=True)


def _trade_attr(t, name, default=None):
    if hasattr(t, name):
        return getattr(t, name)
    if isinstance(t, dict):
        return t.get(name, default)
    return default


def _calc_stats(trades):
    if not trades:
        return {'n': 0, 'sharpe': 0.0, 'pnl': 0.0, 'win_rate': 0.0, 'max_dd': 0.0}
    pnls = np.array([_trade_attr(t, 'pnl', 0) for t in trades])
    wr = float((pnls > 0).mean() * 100)
    cum = np.cumsum(pnls)
    dd = float((np.maximum.accumulate(cum) - cum).max())
    sharpe = (float(pnls.mean() / pnls.std() * np.sqrt(252))
              if pnls.std() > 0 else 0.0)
    return {'n': len(pnls), 'sharpe': round(sharpe, 3),
            'pnl': round(float(pnls.sum()), 2),
            'win_rate': round(wr, 2),
            'max_dd': round(dd, 2),
            'mean_pnl': round(float(pnls.mean()), 2),
            'median_pnl': round(float(np.median(pnls)), 2)}


def _filter_period(trades, start, end):
    s = pd.Timestamp(start, tz='UTC')
    e = pd.Timestamp(end, tz='UTC')
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
    from backtest.runner import LIVE_PARITY_KWARGS, REALISTIC_COST_KWARGS
    import indicators as signals_mod
    from indicators import get_orb_strategy

    if not _init_done:
        _worker_init()
    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False
    get_orb_strategy().reset_daily()

    s = pd.Timestamp(task['start'], tz='UTC')
    e = pd.Timestamp(task['end'], tz='UTC')
    m15 = _data.m15_df[(_data.m15_df.index >= s) & (_data.m15_df.index <= e)]
    h1 = _data.h1_df[(_data.h1_df.index >= s) & (_data.h1_df.index <= e)]

    preset = LIVE_PARITY_KWARGS if task['preset'] == 'live' else REALISTIC_COST_KWARGS
    cfg = CONFIGS[task['config']]
    kwargs = {**preset, **cfg, 'maxloss_cap': 0, 'max_positions': 4}

    t0 = time.time()
    try:
        eng = BacktestEngine(m15, h1, **kwargs)
        all_trades = eng.run()
        trail_n = getattr(eng, 'dual_thrust_trail_triggered', 0)
    except Exception as ex:
        import traceback
        return {'id': task['id'], 'error': f'{ex}\n{traceback.format_exc()[:500]}',
                'elapsed': time.time() - t0}

    # 只关注 dual_thrust trades
    dt_trades = [t for t in all_trades if _trade_attr(t, 'strategy') == 'dual_thrust']
    all_stats = _calc_stats(all_trades)
    dt_stats = _calc_stats(dt_trades)

    dt_eras = {}
    for era_name, es, ee in ERAS:
        dt_eras[era_name] = _calc_stats(_filter_period(dt_trades, es, ee))

    # 按出场原因分组 (dual_thrust)
    by_reason = {}
    for t in dt_trades:
        r = _trade_attr(t, 'reason', 'unknown') or 'unknown'
        by_reason.setdefault(r, []).append(t)
    reason_stats = {k: _calc_stats(v) for k, v in by_reason.items()}

    return {
        'id': task['id'], 'config': task['config'], 'preset': task['preset'],
        'all_stats': all_stats, 'dt_stats': dt_stats,
        'dt_eras': dt_eras, 'dt_by_reason': reason_stats,
        'dt_trail_triggered': trail_n,
        'elapsed': round(time.time() - t0, 1),
    }


def build_tasks():
    tasks = []
    for cfg in CONFIGS:
        for preset in ['live', 'realistic']:
            tasks.append({
                'id': f'{cfg}_{preset}', 'config': cfg, 'preset': preset,
                'start': '2015-01-01', 'end': '2026-05-13',
            })
    return tasks


def main():
    t0 = time.time()
    print('=' * 80)
    print('R251: DualThrust Trailing Sweep')
    print('=' * 80)
    print(f'\nGrid: act × dist = {len(TRAIL_ACT_GRID)} × {len(TRAIL_DIST_GRID)}')
    print(f'Configurations: {len(CONFIGS)} (含 1 baseline)')
    print(f'已剔除 dist > act 无意义组合')

    tasks = build_tasks()
    print(f'\nTotal tasks: {len(tasks)} ({len(CONFIGS)} configs × 2 presets)')

    n_workers = min(32, mp.cpu_count())
    print(f'Workers: {n_workers}\n', flush=True)
    with mp.Pool(n_workers, initializer=_worker_init) as pool:
        results = pool.map(_run_single, tasks)

    # 组织 by config
    by_config = {}
    for r in results:
        if 'error' in r:
            print(f'[ERR] {r["id"]}: {r["error"][:200]}', flush=True)
            continue
        by_config.setdefault(r['config'], {})[r['preset']] = r

    # 排行: 按 Realistic dt_pnl 降序
    rows = []
    for cfg, presets in by_config.items():
        live = presets.get('live', {}).get('dt_stats', {})
        real = presets.get('realistic', {}).get('dt_stats', {})
        rows.append({
            'cfg': cfg,
            'live_n': live.get('n', 0),
            'live_sh': live.get('sharpe', 0),
            'live_pnl': live.get('pnl', 0),
            'live_wr': live.get('win_rate', 0),
            'live_dd': live.get('max_dd', 0),
            'real_n': real.get('n', 0),
            'real_sh': real.get('sharpe', 0),
            'real_pnl': real.get('pnl', 0),
            'real_wr': real.get('win_rate', 0),
            'real_dd': real.get('max_dd', 0),
            'trail_n': presets.get('live', {}).get('dt_trail_triggered', 0),
        })
    rows.sort(key=lambda x: -x['real_pnl'])

    print('\n' + '=' * 145, flush=True)
    print(f'DualThrust Trailing 排行 (按 Realistic PnL 降序)')
    print(f'{"Config":24} | {"Live#":>5} {"LiveSh":>7} {"LivePnL":>9} {"LiveWR":>7} {"LiveDD":>8} | '
          f'{"Real#":>5} {"RealSh":>7} {"RealPnL":>9} {"RealWR":>7} {"RealDD":>8} | {"Trail":>5}')
    print('-' * 145)
    for r in rows:
        print(f'{r["cfg"]:24} | {r["live_n"]:>5} {r["live_sh"]:>7} {r["live_pnl"]:>9.2f} '
              f'{r["live_wr"]:>6.1f}% {r["live_dd"]:>8.2f} | '
              f'{r["real_n"]:>5} {r["real_sh"]:>7} {r["real_pnl"]:>9.2f} '
              f'{r["real_wr"]:>6.1f}% {r["real_dd"]:>8.2f} | {r["trail_n"]:>5}')

    elapsed = time.time() - t0
    print(f'\nElapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)', flush=True)

    outfile = OUTDIR / 'sweep_results.json'
    with open(outfile, 'w', encoding='utf-8') as f:
        json.dump({'meta': {'configs': CONFIGS, 'grid_act': TRAIL_ACT_GRID,
                            'grid_dist': TRAIL_DIST_GRID, 'eras': ERAS,
                            'elapsed_sec': elapsed},
                   'rows': rows, 'by_config': by_config,
                   'raw_results': results}, f, indent=2, default=str, ensure_ascii=False)
    print(f'Saved: {outfile}', flush=True)


if __name__ == '__main__':
    main()
