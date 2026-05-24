"""R250 Layer 1 - Track A: Keltner H1 趋势过滤 (4 配置).

测试 4 种过滤模式 (baseline / A1 / A2 / A1+A2) 在 Live5 组合上的全样本表现。
输出: Sharpe / PnL / MaxDD / Win Rate / Keltner trade 数 / 趋势过滤次数

执行方式 (VPS):
  python experiments/run_r250_layer1_track_a.py

依赖:
  - backtest/engine.py (R250 扩展版, 已支持 keltner_trend_filter_mode)
  - 数据: data/download/*.csv (M15/H1/spread/D1)
"""
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

# 确保项目根目录在 sys.path 中 (multiprocessing worker 需要)
_PROJ_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════
# 配置: 4 种趋势过滤模式 (含 baseline)
# ═══════════════════════════════════════════════════════════════
CONFIGS = {
    'baseline_off': {'keltner_trend_filter_mode': 'off'},
    'A1_ema9_21':   {'keltner_trend_filter_mode': 'a1'},
    'A2_ema50_5':   {'keltner_trend_filter_mode': 'a2', 'keltner_trend_filter_ema50_lookback': 5},
    'A1+A2_both':   {'keltner_trend_filter_mode': 'a1+a2', 'keltner_trend_filter_ema50_lookback': 5},
}

ERAS = [
    ('2024-2025', '2024-01-01', '2025-12-31'),
    ('2026Q1',    '2026-01-01', '2026-03-31'),
    ('2026Q2',    '2026-04-01', '2026-05-13'),
]

OUTDIR = Path(__file__).resolve().parents[1] / 'results' / 'r250_efficiency_overhaul'
OUTDIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# Worker init (复用 R249 的 Live5 patch)
# ═══════════════════════════════════════════════════════════════
_data = None
_init_done = False


def _worker_init():
    global _data, _init_done
    # multiprocessing worker 重新加载 sys.path
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
    sharpe = float(pnls.mean() / pnls.std() * np.sqrt(252)) if pnls.std() > 0 else 0.0
    return {'n': len(pnls), 'sharpe': round(sharpe, 3),
            'pnl': round(float(pnls.sum()), 2),
            'win_rate': round(wr, 2), 'max_dd': round(dd, 2)}


def _stats_by_strategy(trades):
    by_strat = {}
    for t in trades:
        s = _trade_attr(t, 'strategy', 'unknown')
        by_strat.setdefault(s, []).append(t)
    return {k: _calc_stats(v) for k, v in by_strat.items()}


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
    """单 task = 一个 config × 一个时间段"""
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
        trades = eng.run()
        trend_filtered = getattr(eng, 'keltner_trend_filtered_count', 0)
    except Exception as ex:
        import traceback
        return {'id': task['id'], 'error': f'{ex}\n{traceback.format_exc()[:500]}',
                'elapsed': time.time() - t0}

    stats = _calc_stats(trades)
    eras = {era_name: _calc_stats(_filter_period(trades, es, ee))
            for era_name, es, ee in ERAS}
    by_strat = _stats_by_strategy(trades)

    return {
        'id': task['id'],
        'kind': task['kind'],
        'config': task['config'],
        'preset': task['preset'],
        'start': task['start'], 'end': task['end'],
        'stats': stats,
        'eras': eras,
        'by_strategy': by_strat,
        'trend_filtered_count': trend_filtered,
        'elapsed': round(time.time() - t0, 1),
    }


def build_tasks():
    tasks = []
    full_start, full_end = '2015-01-01', '2026-05-13'
    for cfg in CONFIGS.keys():
        # Live parity full history
        tasks.append({
            'id': f'full_live_{cfg}', 'kind': 'full', 'config': cfg,
            'preset': 'live', 'start': full_start, 'end': full_end,
        })
        # Realistic cost full history
        tasks.append({
            'id': f'realistic_{cfg}', 'kind': 'realistic', 'config': cfg,
            'preset': 'realistic', 'start': full_start, 'end': full_end,
        })
    return tasks


def main():
    t0 = time.time()
    print('=' * 80)
    print('R250 Layer 1 - Track A: Keltner H1 Trend Filter')
    print('=' * 80)
    print('\nConfigurations:')
    for k, v in CONFIGS.items():
        print(f'  {k}: {v}')

    tasks = build_tasks()
    print(f'\nTotal tasks: {len(tasks)} (× {len(CONFIGS)} configs × 2 presets)')

    n_workers = min(16, mp.cpu_count())
    print(f'Workers: {n_workers}\n')
    with mp.Pool(n_workers, initializer=_worker_init) as pool:
        results = pool.map(_run_single, tasks)

    # 整理结果
    by_config = {}
    for r in results:
        if 'error' in r:
            print(f'  [ERROR] {r["id"]}: {r["error"][:200]}')
            continue
        cfg = r['config']
        kind = r['kind']
        by_config.setdefault(cfg, {})[kind] = r

    # 打印榜单
    print('\n' + '=' * 100)
    print(f'{"Config":18} | {"Live Sharpe":>12} | {"Live PnL":>10} | {"Realistic Sh":>13} | {"Realistic PnL":>14} | {"Trend Filt":>11}')
    print('-' * 100)
    for cfg in CONFIGS.keys():
        r_live = by_config.get(cfg, {}).get('full', {})
        r_real = by_config.get(cfg, {}).get('realistic', {})
        l_sh = r_live.get('stats', {}).get('sharpe', 'n/a')
        l_pnl = r_live.get('stats', {}).get('pnl', 'n/a')
        r_sh = r_real.get('stats', {}).get('sharpe', 'n/a')
        r_pnl = r_real.get('stats', {}).get('pnl', 'n/a')
        tf = r_live.get('trend_filtered_count', 0)
        print(f'{cfg:18} | {l_sh:>12} | {l_pnl:>10} | {r_sh:>13} | {r_pnl:>14} | {tf:>11}')

    elapsed = time.time() - t0
    print(f'\nElapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)')

    # 保存 raw results
    outfile = OUTDIR / 'track_a_results.json'
    with open(outfile, 'w', encoding='utf-8') as f:
        json.dump({'meta': {'configs': CONFIGS, 'eras': ERAS, 'elapsed_sec': elapsed},
                   'results': results, 'by_config': by_config}, f, indent=2, default=str, ensure_ascii=False)
    print(f'Saved: {outfile}')


if __name__ == '__main__':
    main()
