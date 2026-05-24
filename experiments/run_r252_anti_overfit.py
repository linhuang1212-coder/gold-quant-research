"""R252: 反过拟合综合验证 (Anti-Overfitting Comprehensive Validation)

回应核心质疑: A1 (R250) + DT trail (R251) 是不是为本周 -$316 亏损过拟合?

三个独立检验:

Phase A — OOS Holdout (最关键):
  IS = 2015-01-01 ~ 2024-12-31 (10 年, R250/R251 验证过)
  OOS = 2025-01-01 ~ 2026-05-06 (~16 月, 完全未触过)
  配置: baseline / A1_only / DT_trail_only / A1+DT_trail
  preset: live / realistic
  通过条件: OOS 上 A1+DT_trail 的 Sharpe/PnL 仍胜过 baseline

Phase D — A1 EMA 邻域扫描:
  9 组 (fast, slow):
    (7,18) (7,21) (7,25)
    (9,18) (9,21) (9,25)    ← (9,21) 是 R250 baseline
    (12,18) (12,21) (12,25)
  通过条件: EMA(9,21) 不是孤立 peak, 而是 plateau (邻近都 PnL > baseline)
  实现: monkey-patch h1_df['EMA9'/'EMA21'] 为指定 EMA, 用 keltner_trend_filter_mode='a1'

输出: results/r252_anti_overfit/{phaseA, phaseD}_results.json

执行: python experiments/run_r252_anti_overfit.py
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
from multiprocessing import Pool

_PROJ_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import numpy as np
import pandas as pd

OUTDIR = Path(__file__).resolve().parents[1] / 'results' / 'r252_anti_overfit'
OUTDIR.mkdir(parents=True, exist_ok=True)

PERIODS = {
    'IS_10y':  ('2015-01-01', '2025-01-01'),
    'OOS_16m': ('2025-01-01', '2026-05-07'),
}

PHASE_A_CONFIGS = {
    'baseline':      {'keltner_trend_filter_mode': 'off',
                      'dual_thrust_trail_enabled': False},
    'A1_only':       {'keltner_trend_filter_mode': 'a1',
                      'dual_thrust_trail_enabled': False},
    'DT_trail_only': {'keltner_trend_filter_mode': 'off',
                      'dual_thrust_trail_enabled': True,
                      'dual_thrust_trail_act': 0.10,
                      'dual_thrust_trail_dist': 0.01},
    'A1+DT_trail':   {'keltner_trend_filter_mode': 'a1',
                      'dual_thrust_trail_enabled': True,
                      'dual_thrust_trail_act': 0.10,
                      'dual_thrust_trail_dist': 0.01},
}

PHASE_D_EMA_PAIRS = [
    (7, 18), (7, 21), (7, 25),
    (9, 18), (9, 21), (9, 25),
    (12, 18), (12, 21), (12, 25),
]

_data_cache = {}
_init_done = False


def _worker_init():
    global _init_done
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

    _data_cache['full'] = DataBundle.load_default(start='2015-01-01')
    _init_done = True
    print(f'  [worker {os.getpid()}] init done, full H1 bars: {len(_data_cache["full"].h1_df)}', flush=True)


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
            'mean_pnl': round(float(pnls.mean()), 2)}


def _stats_by_strategy(trades):
    by = {}
    for t in trades:
        s = _trade_attr(t, 'strategy', 'unknown') or 'unknown'
        by.setdefault(s, []).append(t)
    return {s: _calc_stats(ts) for s, ts in by.items()}


def _slice_data(data, start: str, end: str):
    s = pd.Timestamp(start, tz='UTC')
    e = pd.Timestamp(end, tz='UTC')
    m15 = data.m15_df[(data.m15_df.index >= s) & (data.m15_df.index < e)]
    h1 = data.h1_df[(data.h1_df.index >= s) & (data.h1_df.index < e)]
    return m15, h1


def _override_ema(h1_df, fast: int, slow: int) -> pd.DataFrame:
    """Phase D: 用指定 (fast, slow) 覆盖 EMA9/EMA21 列, 用于 A1 EMA 邻域扫描"""
    h1_new = h1_df.copy()
    h1_new['EMA9'] = h1_new['Close'].ewm(span=fast, adjust=False).mean()
    h1_new['EMA21'] = h1_new['Close'].ewm(span=slow, adjust=False).mean()
    return h1_new


def _run_one(task):
    from backtest.engine import BacktestEngine
    from backtest.runner import LIVE_PARITY_KWARGS, REALISTIC_COST_KWARGS
    import indicators as signals_mod
    from indicators import get_orb_strategy

    if not _init_done:
        _worker_init()
    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False
    get_orb_strategy().reset_daily()

    data = _data_cache['full']
    period = task['period']
    start, end = PERIODS[period]
    m15, h1 = _slice_data(data, start, end)

    if task['phase'] == 'D':
        fast, slow = task['ema_pair']
        if fast > 0 and slow > 0:
            h1 = _override_ema(h1, fast, slow)

    preset = LIVE_PARITY_KWARGS if task['preset'] == 'live' else REALISTIC_COST_KWARGS
    cfg = task['cfg']
    kwargs = {**preset, **cfg, 'maxloss_cap': 0, 'max_positions': 4}

    t0 = time.time()
    try:
        eng = BacktestEngine(m15, h1, **kwargs)
        trades = eng.run()
        trail_n = getattr(eng, 'dual_thrust_trail_triggered', 0)
        kc_filt = getattr(eng, 'keltner_trend_filtered_count', 0)
    except Exception as ex:
        return {'id': task['id'], 'error': str(ex), 'elapsed': round(time.time() - t0, 1)}
    elapsed = time.time() - t0

    return {
        'id': task['id'],
        'phase': task['phase'],
        'config': task.get('config_name'),
        'ema_pair': task.get('ema_pair'),
        'period': period,
        'preset': task['preset'],
        'all_stats': _calc_stats(trades),
        'by_strategy': _stats_by_strategy(trades),
        'dt_trail_triggered': int(trail_n),
        'keltner_trend_filtered': int(kc_filt),
        'elapsed': round(elapsed, 1),
    }


def build_tasks():
    tasks = []

    for period in PERIODS:
        for preset in ['live', 'realistic']:
            for cfg_name, cfg in PHASE_A_CONFIGS.items():
                tasks.append({
                    'id': f'A_{period}_{preset}_{cfg_name}',
                    'phase': 'A', 'period': period, 'preset': preset,
                    'config_name': cfg_name, 'cfg': cfg,
                })

    period = 'IS_10y'
    for preset in ['live', 'realistic']:
        for ema_pair in PHASE_D_EMA_PAIRS:
            tasks.append({
                'id': f'D_{ema_pair[0]}x{ema_pair[1]}_{preset}',
                'phase': 'D', 'period': period, 'preset': preset,
                'ema_pair': ema_pair,
                'config_name': f'a1_ema_{ema_pair[0]}x{ema_pair[1]}',
                'cfg': {'keltner_trend_filter_mode': 'a1',
                        'dual_thrust_trail_enabled': False},
            })

    for preset in ['live', 'realistic']:
        tasks.append({
            'id': f'D_OFF_{preset}',
            'phase': 'D', 'period': period, 'preset': preset,
            'ema_pair': (0, 0),
            'config_name': 'A1_OFF_baseline',
            'cfg': {'keltner_trend_filter_mode': 'off',
                    'dual_thrust_trail_enabled': False},
        })

    return tasks


def main():
    tasks = build_tasks()
    print(f'\n=== R252 Anti-Overfit ===')
    print(f'  Phase A: {len([t for t in tasks if t["phase"]=="A"])} runs (OOS holdout)')
    print(f'  Phase D: {len([t for t in tasks if t["phase"]=="D"])} runs (EMA neighbor scan)')
    print(f'  Total:   {len(tasks)} runs')
    print(f'  Workers: 28 (200+ cores avail, leave headroom for data loading)')

    t_start = time.time()
    with Pool(processes=28, initializer=_worker_init) as pool:
        results = pool.map(_run_one, tasks)
    elapsed = time.time() - t_start
    print(f'\n=== Done in {elapsed/60:.1f} min ===')

    phase_a = [r for r in results if r.get('phase') == 'A']
    phase_d = [r for r in results if r.get('phase') == 'D']

    with open(OUTDIR / 'phaseA_results.json', 'w', encoding='utf-8') as f:
        json.dump(phase_a, f, ensure_ascii=False, indent=2, default=str)
    with open(OUTDIR / 'phaseD_results.json', 'w', encoding='utf-8') as f:
        json.dump(phase_d, f, ensure_ascii=False, indent=2, default=str)

    print('\n=== Phase A: OOS Holdout (realistic preset) ===')
    print(f'{"config":<15} {"period":<10} {"n":>6} {"sharpe":>8} {"pnl":>12} {"maxdd":>10}')
    for r in phase_a:
        if r['preset'] != 'realistic': continue
        s = r['all_stats']
        print(f'{r["config"]:<15} {r["period"]:<10} {s["n"]:>6} {s["sharpe"]:>8.3f} {s["pnl"]:>12.2f} {s["max_dd"]:>10.2f}')

    print('\n=== Phase D: A1 EMA Neighbor (realistic, IS) ===')
    print(f'{"fast":>5}x{"slow":<5} {"n":>6} {"sharpe":>8} {"pnl":>12} {"dt_pnl":>10}')
    for r in phase_d:
        if r['preset'] != 'realistic': continue
        s = r['all_stats']
        dt = r['by_strategy'].get('dual_thrust', {}).get('pnl', 0)
        kc = r['by_strategy'].get('keltner', {}).get('pnl', 0)
        pair = r['ema_pair']
        print(f'{pair[0]:>5}x{pair[1]:<5} {s["n"]:>6} {s["sharpe"]:>8.3f} {s["pnl"]:>12.2f}  kc={kc:>7.0f}')


if __name__ == '__main__':
    main()
