"""R250 Layer 2: A1 + C1 联合验证 + scope 细分 (6 关验证).

复用 R249 完整 6 关 framework: K-Fold + WF + Era + Bootstrap + Realistic Cost.

配置:
  L2_baseline       : 完全 baseline
  L2_A1_only        : 仅 Track A (EMA9>EMA21)
  L2_C1_only        : 仅 Track C (BE 0.6 ATR, buf=0, scope=all)
  L2_A1+C1_all      : A1 + BE 全策略
  L2_A1+C1_keltner  : A1 + BE 仅 keltner
  L2_A1+C1_no_dt    : A1 + BE 全策略但排除 dual_thrust

执行: python experiments/run_r250_layer2_combined.py
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
# 6 个联合配置
# ═══════════════════════════════════════════════════════════════
CONFIGS = {
    'L2_baseline': {},
    'L2_A1_only': {
        'keltner_trend_filter_mode': 'a1',
    },
    'L2_C1_only': {
        'breakeven_after_atr': 0.6,
        'breakeven_buffer_atr': 0.0,
        'breakeven_strategies': {'all'},
    },
    'L2_A1+C1_all': {
        'keltner_trend_filter_mode': 'a1',
        'breakeven_after_atr': 0.6,
        'breakeven_buffer_atr': 0.0,
        'breakeven_strategies': {'all'},
    },
    'L2_A1+C1_keltner': {
        'keltner_trend_filter_mode': 'a1',
        'breakeven_after_atr': 0.6,
        'breakeven_buffer_atr': 0.0,
        'breakeven_strategies': {'keltner'},
    },
    'L2_A1+C1_no_dt': {
        'keltner_trend_filter_mode': 'a1',
        'breakeven_after_atr': 0.6,
        'breakeven_buffer_atr': 0.0,
        'breakeven_strategies': {'keltner', 'sess_bo', 'tsmom', 'm15_rsi', 'orb', 'donchian', 'm30_rsi14'},
    },
}

ERAS = [
    ('2019-2021', '2019-01-01', '2021-12-31'),
    ('2022-2023', '2022-01-01', '2023-12-31'),
    ('2024-2025', '2024-01-01', '2025-12-31'),
    ('2026',      '2026-01-01', '2026-05-13'),
]

OUTDIR = Path(__file__).resolve().parents[1] / 'results' / 'r250_efficiency_overhaul'
OUTDIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# Worker init
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
        be_n = getattr(eng, 'breakeven_triggered', 0)
        tf_n = getattr(eng, 'keltner_trend_filtered_count', 0)
    except Exception as ex:
        import traceback
        return {'id': task['id'], 'error': f'{ex}\n{traceback.format_exc()[:500]}',
                'elapsed': time.time() - t0}

    if task.get('test_filter_start'):
        trades = _filter_period(trades, task['test_filter_start'], task['end'])

    stats = _calc_stats(trades)
    eras = {}
    by_strat = {}
    if task['kind'] in ('full', 'realistic'):
        for era_name, es, ee in ERAS:
            eras[era_name] = _calc_stats(_filter_period(trades, es, ee))
        by_strat = _stats_by_strategy(trades)

    boot = None
    if task['kind'] == 'full':
        pnls = np.array([_trade_attr(t, 'pnl', 0) for t in trades])
        if len(pnls) >= 10:
            rng = np.random.default_rng(42 + hash(task['config']) % 1000)
            sharpes = np.zeros(1000)
            for i in range(1000):
                sample = rng.choice(pnls, size=len(pnls), replace=True)
                sharpes[i] = sample.mean() / sample.std() * np.sqrt(252) if sample.std() > 0 else 0
            boot = {
                'mean': round(float(sharpes.mean()), 3),
                'std': round(float(sharpes.std()), 3),
                'p_positive': round(float((sharpes > 0).mean()), 4),
            }

    return {
        'id': task['id'], 'kind': task['kind'], 'config': task['config'],
        'preset': task['preset'], 'start': task['start'], 'end': task['end'],
        'stats': stats, 'eras': eras, 'by_strategy': by_strat,
        'be_triggered': be_n, 'trend_filtered': tf_n, 'bootstrap': boot,
        'elapsed': round(time.time() - t0, 1),
    }


def build_tasks():
    tasks = []
    configs_to_test = list(CONFIGS.keys())

    # K-Fold (6 folds × 6 configs = 36 tasks)
    fold_boundaries = pd.date_range('2015-01-01', '2026-05-13', periods=7)
    for fold_idx in range(6):
        start = fold_boundaries[fold_idx].strftime('%Y-%m-%d')
        end = fold_boundaries[fold_idx + 1].strftime('%Y-%m-%d')
        for cfg in configs_to_test:
            tasks.append({
                'id': f'kfold_{fold_idx+1}_{cfg}',
                'kind': 'kfold', 'config': cfg, 'preset': 'live',
                'start': start, 'end': end,
            })

    # Walk-Forward (8 windows × 6 configs = 48 tasks)
    test_starts = ['2019-01-01', '2020-01-01', '2021-01-01', '2022-01-01',
                   '2023-01-01', '2024-01-01', '2025-01-01', '2026-01-01']
    test_ends = ['2019-12-31', '2020-12-31', '2021-12-31', '2022-12-31',
                 '2023-12-31', '2024-12-31', '2025-12-31', '2026-05-13']
    for i, (ts, te) in enumerate(zip(test_starts, test_ends)):
        ext_start = (pd.Timestamp(ts) - pd.Timedelta(days=120)).strftime('%Y-%m-%d')
        for cfg in configs_to_test:
            tasks.append({
                'id': f'wf_{i+1}_{cfg}',
                'kind': 'wf', 'config': cfg, 'preset': 'live',
                'start': ext_start, 'end': te,
                'test_filter_start': ts,
            })

    # Full + Realistic (12 tasks)
    for cfg in configs_to_test:
        tasks.append({
            'id': f'full_{cfg}', 'kind': 'full', 'config': cfg, 'preset': 'live',
            'start': '2015-01-01', 'end': '2026-05-13',
        })
        tasks.append({
            'id': f'realistic_{cfg}', 'kind': 'realistic', 'config': cfg, 'preset': 'realistic',
            'start': '2015-01-01', 'end': '2026-05-13',
        })

    return tasks


def main():
    t0 = time.time()
    print('=' * 80)
    print('R250 Layer 2: A1 + C1 联合验证 (6 关 K-Fold/WF/Era/MC/Realistic)')
    print('=' * 80)
    print('\nConfigurations:')
    for k, v in CONFIGS.items():
        print(f'  {k}: {v}')

    tasks = build_tasks()
    n_kfold = sum(1 for t in tasks if t['kind'] == 'kfold')
    n_wf = sum(1 for t in tasks if t['kind'] == 'wf')
    n_full = sum(1 for t in tasks if t['kind'] == 'full')
    n_real = sum(1 for t in tasks if t['kind'] == 'realistic')
    print(f'\nTotal: {len(tasks)} tasks (KFold={n_kfold}, WF={n_wf}, Full={n_full}, Realistic={n_real})')

    n_workers = min(32, mp.cpu_count())
    print(f'Workers: {n_workers}\n', flush=True)
    with mp.Pool(n_workers, initializer=_worker_init) as pool:
        results = pool.map(_run_single, tasks)

    # 整理
    by_config = {cfg: {'kfold': [], 'wf': [], 'full': None, 'realistic': None}
                 for cfg in CONFIGS}
    for r in results:
        if 'error' in r:
            print(f'  [ERROR] {r["id"]}: {r["error"][:200]}', flush=True)
            continue
        cfg = r['config']
        kind = r['kind']
        if kind in ('kfold', 'wf'):
            by_config[cfg][kind].append(r)
        else:
            by_config[cfg][kind] = r

    # 综合榜单
    print('\n' + '=' * 130, flush=True)
    print(f'{"Config":22} | {"K-Fold Pass":>11} | {"Live Sh":>8} | {"Live PnL":>10} | {"DD":>8} | {"Real Sh":>8} | {"Real PnL":>10} | {"BE触发":>7} | {"过滤":>5}')
    print('-' * 130)
    for cfg in CONFIGS:
        kf = by_config[cfg]['kfold']
        kf_pos = sum(1 for r in kf if r.get('stats', {}).get('sharpe', 0) > 0)
        wf = by_config[cfg]['wf']
        wf_pos = sum(1 for r in wf if r.get('stats', {}).get('sharpe', 0) > 0)
        full = by_config[cfg]['full'] or {}
        real = by_config[cfg]['realistic'] or {}
        live_sh = full.get('stats', {}).get('sharpe', 'n/a')
        live_pnl = full.get('stats', {}).get('pnl', 'n/a')
        live_dd = full.get('stats', {}).get('max_dd', 'n/a')
        real_sh = real.get('stats', {}).get('sharpe', 'n/a')
        real_pnl = real.get('stats', {}).get('pnl', 'n/a')
        be_n = full.get('be_triggered', 0)
        tf_n = full.get('trend_filtered', 0)
        print(f'{cfg:22} | KF {kf_pos}/{len(kf)} WF {wf_pos}/{len(wf)} | {live_sh:>8} | {live_pnl:>10} | {live_dd:>8} | {real_sh:>8} | {real_pnl:>10} | {be_n:>7} | {tf_n:>5}')

    elapsed = time.time() - t0
    print(f'\nElapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)', flush=True)

    outfile = OUTDIR / 'layer2_results.json'
    with open(outfile, 'w', encoding='utf-8') as f:
        json.dump({'meta': {'configs': {k: {kk: list(vv) if isinstance(vv, set) else vv for kk, vv in v.items()} for k, v in CONFIGS.items()},
                            'eras': ERAS, 'elapsed_sec': elapsed},
                   'results': results, 'by_config': by_config}, f, indent=2, default=str, ensure_ascii=False)
    print(f'Saved: {outfile}', flush=True)


if __name__ == '__main__':
    main()
