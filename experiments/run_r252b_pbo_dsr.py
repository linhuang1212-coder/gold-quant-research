"""R252-B: PBO (CSCV) + Deflated Sharpe Ratio for R251 5x5 trail sweep.

Goal: Quantify SELECTION BIAS in the R251 grid search, complementing R252-A's
true OOS holdout test. R252-A proved DT trail works on unseen 16-month data;
R252-B asks "given that we picked the best of N=20 grid configs, how likely
is that pick to fail in resampled OOS?"

Two metrics, both Bailey & Lopez de Prado:
  PBO (CSCV, 2015): split daily PnL into 8 blocks, run C(8,4)=70 train/test
    swaps, count how often the IS-best variant ranks below median in OOS.
    pbo<0.20 LOW, 0.20-0.40 MEDIUM, >0.40 HIGH.
  DSR (Deflated Sharpe, 2014): for the chosen variant act010_dist010, compute
    P(true Sharpe > 0 | observed Sharpe AND we tested N=20 variants).
    dsr>0.95 means significant after accounting for selection.

Why NOT Bonferroni: neighboring (act, dist) configs have ~90% overlapping
trades (same dual_thrust signal, slightly different trail), so Bonferroni
assumption of independence is severely violated. DSR handles correlation
implicitly via the variance of trial Sharpes.

Output:
  results/r252b_pbo_dsr/
    daily_pnl_by_config.json   — dense daily PnL series, dt-only, both presets
    pbo_results.json            — PBO output for live + realistic
    dsr_results.json            — DSR output for chosen variant + key alts
    summary.md                  — human-readable verdict

Run on VPS:
  cd /root/gold-quant-research && nohup python -u experiments/run_r252b_pbo_dsr.py \
    > results/r252b_pbo_dsr/run.log 2>&1 &
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path
from multiprocessing import Pool

_PROJ_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import numpy as np
import pandas as pd

OUTDIR = Path(__file__).resolve().parents[1] / 'results' / 'r252b_pbo_dsr'
OUTDIR.mkdir(parents=True, exist_ok=True)

PERIOD_START = '2015-01-01'
PERIOD_END = '2026-05-13'

# Mirror R251 grid exactly (5x5 act x dist with dist <= act constraint, plus baseline).
TRAIL_ACT_GRID = [0.06, 0.10, 0.20, 0.40, 0.80]
TRAIL_DIST_GRID = [0.01, 0.03, 0.08, 0.15, 0.30]

CONFIGS = {'baseline_off': {'dual_thrust_trail_enabled': False}}
for _act in TRAIL_ACT_GRID:
    for _dist in TRAIL_DIST_GRID:
        if _dist > _act:
            continue
        _name = f'act{int(_act*100):03d}_dist{int(_dist*1000):03d}'
        CONFIGS[_name] = {
            'dual_thrust_trail_enabled': True,
            'dual_thrust_trail_act': _act,
            'dual_thrust_trail_dist': _dist,
        }

CHOSEN = 'act010_dist010'

_data = None
_init_done = False


def _worker_init():
    """Same patch logic as R251/R252-A so dual_thrust signals are emitted."""
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


def _trade_attr(t, name, default=None):
    if hasattr(t, name):
        return getattr(t, name)
    if isinstance(t, dict):
        return t.get(name, default)
    return default


def _run_one(task):
    """Run a single (config, preset) and return raw dt trade records."""
    from backtest.engine import BacktestEngine
    from backtest.runner import LIVE_PARITY_KWARGS, REALISTIC_COST_KWARGS
    import indicators as signals_mod
    from indicators import get_orb_strategy

    if not _init_done:
        _worker_init()
    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False
    get_orb_strategy().reset_daily()

    s = pd.Timestamp(PERIOD_START, tz='UTC')
    e = pd.Timestamp(PERIOD_END, tz='UTC')
    m15 = _data.m15_df[(_data.m15_df.index >= s) & (_data.m15_df.index <= e)]
    h1 = _data.h1_df[(_data.h1_df.index >= s) & (_data.h1_df.index <= e)]

    preset = LIVE_PARITY_KWARGS if task['preset'] == 'live' else REALISTIC_COST_KWARGS
    cfg = CONFIGS[task['config']]
    kwargs = {**preset, **cfg, 'maxloss_cap': 0, 'max_positions': 4}

    t0 = time.time()
    try:
        eng = BacktestEngine(m15, h1, **kwargs)
        all_trades = eng.run()
    except Exception as ex:
        import traceback
        return {'id': task['id'], 'error': f'{ex}\n{traceback.format_exc()[:600]}',
                'elapsed': round(time.time() - t0, 1)}

    # Only dual_thrust trades — trail_act/trail_dist only affect this strategy.
    dt_trades = [
        {'exit_time': str(_trade_attr(t, 'exit_time')),
         'pnl': float(_trade_attr(t, 'pnl', 0))}
        for t in all_trades
        if _trade_attr(t, 'strategy') == 'dual_thrust'
    ]

    return {
        'id': task['id'],
        'config': task['config'],
        'preset': task['preset'],
        'n_trades': len(dt_trades),
        'total_pnl': round(sum(t['pnl'] for t in dt_trades), 2),
        'trades': dt_trades,
        'elapsed': round(time.time() - t0, 1),
    }


def build_tasks():
    return [
        {'id': f'{cfg}_{preset}', 'config': cfg, 'preset': preset}
        for cfg in CONFIGS
        for preset in ['live', 'realistic']
    ]


def trades_to_dense_daily(trades, idx):
    """Build a dense daily PnL series aligned to idx (DatetimeIndex)."""
    if not trades:
        return np.zeros(len(idx), dtype=float)
    df = pd.DataFrame(trades)
    df['date'] = pd.to_datetime(df['exit_time'], utc=True, errors='coerce').dt.normalize()
    df = df.dropna(subset=['date'])
    daily = df.groupby('date')['pnl'].sum()
    return daily.reindex(idx, fill_value=0.0).to_numpy(dtype=float)


def main():
    print('=' * 80, flush=True)
    print('R252-B: PBO (CSCV) + DSR for R251 trail sweep', flush=True)
    print('=' * 80, flush=True)
    print(f'Configs: {len(CONFIGS)} ({len(CONFIGS)-1} trail-on + 1 baseline)', flush=True)
    print(f'Presets: live + realistic', flush=True)
    print(f'Period:  {PERIOD_START} ~ {PERIOD_END}', flush=True)

    tasks = build_tasks()
    print(f'Tasks:   {len(tasks)}', flush=True)
    n_workers = min(int(os.environ.get('R252B_WORKERS', 32)), len(tasks))
    print(f'Workers: {n_workers}\n', flush=True)

    t0 = time.time()
    with Pool(processes=n_workers, initializer=_worker_init) as pool:
        results = pool.map(_run_one, tasks)
    elapsed = time.time() - t0
    print(f'\n=== Backtests done in {elapsed/60:.1f} min ===\n', flush=True)

    # ---------- Build dense daily PnL series per (preset, config) ----------
    daily_idx = pd.date_range(PERIOD_START, PERIOD_END, freq='D', tz='UTC').normalize()

    daily_by = {'live': {}, 'realistic': {}}
    summary_rows = []
    errors = []
    for r in results:
        if 'error' in r:
            errors.append(r)
            print(f'[ERR] {r["id"]}: {r["error"][:200]}', flush=True)
            continue
        series = trades_to_dense_daily(r['trades'], daily_idx)
        daily_by[r['preset']][r['config']] = series.tolist()
        summary_rows.append({
            'config': r['config'], 'preset': r['preset'],
            'n_trades': r['n_trades'], 'total_pnl': r['total_pnl'],
            'sharpe_daily': round(_sharpe(series), 3),
        })

    if errors:
        print(f'\n!! {len(errors)} task(s) errored. Aborting analysis.\n', flush=True)
        raise SystemExit(1)

    # ---------- Save raw daily series ----------
    with open(OUTDIR / 'daily_pnl_by_config.json', 'w', encoding='utf-8') as f:
        json.dump({
            'meta': {
                'configs': CONFIGS,
                'period_start': PERIOD_START,
                'period_end': PERIOD_END,
                'n_days': len(daily_idx),
                'chosen': CHOSEN,
            },
            'summary': summary_rows,
            'daily': daily_by,
        }, f, ensure_ascii=False, indent=2)

    # ---------- PBO + DSR ----------
    from backtest.stats import compute_pbo, deflated_sharpe

    pbo_results = {}
    dsr_results = {}
    for preset in ['live', 'realistic']:
        print(f'\n=== {preset.upper()} preset ===', flush=True)

        all_variants = daily_by[preset]
        trail_only = {k: v for k, v in all_variants.items() if k != 'baseline_off'}

        # PBO with all variants (baseline competing too)
        print(f'  Running PBO/CSCV (n_partitions=8, {len(all_variants)} variants)...', flush=True)
        pbo_all = compute_pbo(all_variants, n_partitions=8)
        # PBO with trail-only variants (selection within trail grid)
        pbo_trail = compute_pbo(trail_only, n_partitions=8)
        pbo_results[preset] = {
            'all_variants': {k: v for k, v in pbo_all.items()
                             if k != 'logit_distribution'},
            'trail_only': {k: v for k, v in pbo_trail.items()
                           if k != 'logit_distribution'},
            'n_variants_all': len(all_variants),
            'n_variants_trail_only': len(trail_only),
        }
        print(f'    PBO (all={len(all_variants)} variants):   {pbo_all["pbo"]:.1%} '
              f'({pbo_all["overfit_risk"]}) over {pbo_all["n_combinations"]} splits', flush=True)
        print(f'    PBO (trail-only N={len(trail_only)}):  {pbo_trail["pbo"]:.1%} '
              f'({pbo_trail["overfit_risk"]})', flush=True)

        # DSR for chosen variant + a few alternatives, n_trials = number of trail-on variants
        n_trials = len(trail_only)
        chosen_series = all_variants[CHOSEN]
        # Variance of all trail-only variants' Sharpes (Bailey & LdP recommend this)
        trail_sharpes = np.array([_sharpe(np.array(v)) for v in trail_only.values()])
        sharpes_var = float(np.var(trail_sharpes, ddof=1)) if len(trail_sharpes) > 1 else None
        # Convert annualized variance to daily (DSR formula uses daily SR space)
        sharpes_var_daily = sharpes_var / 252.0 if sharpes_var is not None else None

        dsr_results[preset] = {
            'n_trials': n_trials,
            'all_trial_sharpes': sorted([(k, round(_sharpe(np.array(v)), 3))
                                         for k, v in trail_only.items()],
                                        key=lambda x: -x[1]),
            'trial_sharpes_mean': float(np.mean(trail_sharpes)),
            'trial_sharpes_std': float(np.std(trail_sharpes, ddof=1)),
            'sharpes_var_annualized': sharpes_var,
            'sharpes_var_daily': sharpes_var_daily,
            'variants': {},
        }

        # DSR for top variants
        for variant in [CHOSEN, 'act080_dist010', 'act006_dist010', 'act010_dist030']:
            if variant not in all_variants:
                continue
            v_series = all_variants[variant]
            dsr_using_var = deflated_sharpe(v_series, n_trials=n_trials,
                                            all_sharpes_var=sharpes_var_daily)
            dsr_naive = deflated_sharpe(v_series, n_trials=n_trials)
            dsr_results[preset]['variants'][variant] = {
                'with_trial_var': dsr_using_var,
                'naive_sr_var': dsr_naive,
            }
            print(f'    DSR[{variant:18}]: SR={dsr_using_var["sharpe_obs"]:.3f}, '
                  f'SR*={dsr_using_var["sr_star"]:.3f}, DSR={dsr_using_var["dsr"]:.3f} '
                  f'(passed={dsr_using_var["passed"]})', flush=True)

    with open(OUTDIR / 'pbo_results.json', 'w', encoding='utf-8') as f:
        json.dump(pbo_results, f, ensure_ascii=False, indent=2)
    with open(OUTDIR / 'dsr_results.json', 'w', encoding='utf-8') as f:
        json.dump(dsr_results, f, ensure_ascii=False, indent=2)

    # ---------- Markdown summary ----------
    write_summary(pbo_results, dsr_results, summary_rows, elapsed)
    print(f'\n=== R252-B done. Outputs in {OUTDIR} ===', flush=True)
    print(f'    Total elapsed: {(time.time() - t0)/60:.1f} min', flush=True)


def _sharpe(daily):
    arr = np.asarray(daily, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return 0.0
    sd = float(np.std(arr, ddof=1))
    if sd <= 0:
        return 0.0
    return float(np.mean(arr) / sd * np.sqrt(252))


def write_summary(pbo_results, dsr_results, summary_rows, elapsed_sec):
    lines = []
    lines.append('# R252-B: PBO (CSCV) + DSR — Selection Bias Validation\n')
    lines.append(f'**Date**: {time.strftime("%Y-%m-%d")}')
    lines.append(f'**Runtime**: {elapsed_sec/60:.1f} min')
    lines.append(f'**Period**: {PERIOD_START} ~ {PERIOD_END}\n')

    lines.append('## TL;DR\n')
    for preset in ['live', 'realistic']:
        pall = pbo_results[preset]['all_variants']
        ptr = pbo_results[preset]['trail_only']
        dchose = dsr_results[preset]['variants'].get(CHOSEN, {}).get('with_trial_var', {})
        dsr_val = dchose.get('dsr', 0.0)
        sr_obs = dchose.get('sharpe_obs', 0.0)
        sr_star = dchose.get('sr_star', 0.0)
        lines.append(f'### {preset.upper()} preset')
        lines.append(f'- **PBO (all={pbo_results[preset]["n_variants_all"]} variants)**: '
                     f'{pall["pbo"]:.1%} ({pall["overfit_risk"]} risk)')
        lines.append(f'- **PBO (trail-only N={pbo_results[preset]["n_variants_trail_only"]})**: '
                     f'{ptr["pbo"]:.1%} ({ptr["overfit_risk"]} risk)')
        lines.append(f'- **DSR for {CHOSEN}**: '
                     f'SR_obs={sr_obs:.3f}, SR*(expected max under null)={sr_star:.3f}, '
                     f'**DSR={dsr_val:.3f}** {"PASS (>0.95)" if dsr_val > 0.95 else "FAIL"}')
        lines.append('')

    lines.append('## Decision Rules (Bailey & Lopez de Prado)\n')
    lines.append('| Metric | Threshold | Interpretation |')
    lines.append('|---|---|---|')
    lines.append('| PBO | < 20% LOW, 20-40% MED, > 40% HIGH | Probability that IS-best variant ranks below median in OOS |')
    lines.append('| DSR | > 0.95 PASS | Probability the chosen variant\'s edge is real after accounting for N trials |')
    lines.append('')

    lines.append('## Per-config summary\n')
    lines.append('| config | preset | n_trades | total_pnl | sharpe |')
    lines.append('|---|---|---|---|---|')
    for r in sorted(summary_rows, key=lambda x: (x['preset'], -x['total_pnl'])):
        lines.append(f'| {r["config"]} | {r["preset"]} | {r["n_trades"]} | '
                     f'${r["total_pnl"]:.0f} | {r["sharpe_daily"]:.3f} |')
    lines.append('')

    lines.append('## DSR Detail by Variant\n')
    for preset in ['live', 'realistic']:
        lines.append(f'### {preset.upper()}')
        lines.append(f'- Mean Sharpe across {dsr_results[preset]["n_trials"]} trial variants: '
                     f'{dsr_results[preset]["trial_sharpes_mean"]:.3f}')
        lines.append(f'- Std Sharpe across trials: {dsr_results[preset]["trial_sharpes_std"]:.3f}')
        lines.append('')
        lines.append('| variant | Sharpe (annual) | DSR | Pass |')
        lines.append('|---|---|---|---|')
        for v_name, v in dsr_results[preset]['variants'].items():
            d = v['with_trial_var']
            pass_mark = 'YES' if d.get('passed') else 'no'
            lines.append(f'| {v_name} | {d.get("sharpe_obs", 0):.3f} | '
                         f'{d.get("dsr", 0):.3f} | {pass_mark} |')
        lines.append('')

    lines.append('## Methodology Notes\n')
    lines.append('- **PBO**: 8 time-blocks, C(8,4)=70 combinatorial IS/OOS splits. '
                 'Each split: identify IS-best by Sharpe, look up its OOS rank. '
                 'PBO = fraction of splits where IS-best ranks below median in OOS.')
    lines.append('- **DSR**: Computes E[max(SR)] under null assumption that all N trial '
                 'Sharpes come from a centered distribution with empirical variance. '
                 'DSR = P(true SR > sr_star). Variance of trial Sharpes is used '
                 'instead of asymptotic SR std so that correlation between configs '
                 '(neighboring grid points share trades) is implicitly captured.')
    lines.append('- **Why not Bonferroni**: Neighboring (act, dist) configs have ~90% '
                 'overlapping trades; Bonferroni assumes independence so it would '
                 'wildly over-correct. DSR with empirical variance handles this.')
    lines.append('- **PnL series**: dual_thrust trades only, aggregated to dense '
                 'daily index (zero-padded for no-trade days).')
    lines.append('')

    with open(OUTDIR / 'summary.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    main()
