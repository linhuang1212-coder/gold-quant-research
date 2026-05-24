#!/usr/bin/env python3
"""R245 — Cap_ATR Sweep
==========================
Sweep cap_atr_mult from 0 (off) to sl_atr+1.5 to find Sharpe-optimal cap level
for each strategy. Builds on R244 v2 (with cap) by reusing bt_tsmom/bt_sess_bo/bt_dual_thrust.

For each strategy:
    - Sweep cap_atr in [0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
    - Track full / recent_7mo Sharpe + PnL + MaxDD + cap_fire_count
    - Plot curve + identify optimal cap

The KEY insight: cap is only effective when cap_atr < sl_atr (otherwise SL fires first).
This sweep finds the "tighter than SL" cap value that maximizes Sharpe.
"""
from __future__ import annotations
import sys
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Reuse R244 v2 strategy functions
from experiments.run_r244_missing_strategies import (
    bt_tsmom, bt_sess_bo, bt_dual_thrust,
    load_csv, calc_stats, H1_CSV_PATH, DATA_END,
)

OUTPUT_DIR = Path("results/r245_cap_atr_sweep")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CAP_VALUES = [0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]


ERAS = [
    ('Pre-COVID', '2015-01-01', '2019-12-31'),
    ('COVID+Recovery', '2020-01-01', '2021-12-31'),
    ('Tightening', '2022-01-01', '2023-12-31'),
    ('Recent', '2024-01-01', '2026-05-13'),
]


def _bench(name: str, strat_fn, h1: pd.DataFrame, sl_atr: float, base_kwargs: dict):
    """Run sweep for one strategy with era breakdown."""
    print(f'\n=== Sweep: {name} (sl_atr={sl_atr}) ===')
    rows = []
    for cap in CAP_VALUES:
        kwargs = {**base_kwargs, 'cap_atr_mult': cap}
        t0 = time.time()
        trades = strat_fn(h1, **kwargs)
        dt = time.time() - t0

        cap_n = sum(1 for t in trades if t['reason'] == 'MaxLossCap')
        st_full = calc_stats(trades, f'{name}_full')
        recent_trades = [t for t in trades if t['exit_time'] >= '2025-10-01']
        st_recent = calc_stats(recent_trades, f'{name}_recent_7mo')

        # Era breakdown
        era_stats = {}
        for era_name, start, end in ERAS:
            era_trades = [t for t in trades if start <= t['exit_time'][:10] <= end]
            era_stats[era_name] = calc_stats(era_trades, f'{name}_{era_name}')

        rows.append({
            'cap_atr': cap,
            'effective': 'Yes (tighter than SL)' if 0 < cap < sl_atr else (
                'No (cap off)' if cap == 0 else 'Marginal (looser than SL)'),
            'full_n': st_full['n'],
            'full_sharpe': st_full['sharpe'],
            'full_pnl': st_full['pnl'],
            'full_pf': st_full.get('pf', 0),
            'full_max_dd': st_full['max_dd'],
            'recent_n': st_recent['n'],
            'recent_sharpe': st_recent['sharpe'],
            'recent_pnl': st_recent['pnl'],
            'eras': {k: {'n': v['n'], 'sharpe': v['sharpe'], 'pnl': v['pnl']}
                     for k, v in era_stats.items()},
            'cap_fires': cap_n,
            'cap_fire_pct': cap_n / len(trades) * 100 if trades else 0,
            'wall_sec': round(dt, 2),
        })
        era_sharpes = ' '.join(f"{k[:3]}={v['sharpe']:+5.2f}" for k, v in era_stats.items())
        print(
            f"  cap={cap:>4}  S_full={st_full['sharpe']:+6.3f}  PnL=${st_full['pnl']:>+7.0f}  "
            f"DD=${st_full['max_dd']:>6.0f}  S_rec={st_recent['sharpe']:+5.2f}  "
            f"caps={cap_n:>3} ({cap_n/max(1, len(trades))*100:.1f}%)  | {era_sharpes}"
        )

    return rows


def find_optimal(rows: list, metric: str):
    """Find cap_atr that maximizes the given metric."""
    valid = [r for r in rows if r['cap_atr'] > 0]
    if not valid:
        return None
    best = max(valid, key=lambda r: r[metric])
    return best


def main():
    print('Loading H1 data...')
    h1 = load_csv(str(H1_CSV_PATH))
    h1 = h1[h1.index <= pd.Timestamp(DATA_END, tz='UTC')]
    print(f'  bars: {len(h1)},  {h1.index[0].date()} -> {h1.index[-1].date()}')

    results = {}

    # TSMOM (sl_atr=3.5, original cap=6.5 → ineffective)
    results['TSMOM'] = _bench('TSMOM', bt_tsmom, h1, sl_atr=3.5, base_kwargs={
        'fast_lb': 480, 'slow_lb': 960,
        'sl_atr_mult': 3.5, 'tp_atr_mult': 8.0, 'max_hold': 30,
        'trail_act_atr': 0.14, 'trail_dist_atr': 0.025,
        'rule_b_sigma': 2.0, 'rule_b_lb': 90, 'rule_b_skip': 8,
        'atr_floor': 0.1,
    })

    # SESS_BO (sl_atr=4.5, original cap=5.0 → marginal)
    results['SESS_BO'] = _bench('SESS_BO', bt_sess_bo, h1, sl_atr=4.5, base_kwargs={
        'session_hour': 12, 'lookback': 4,
        'sl_atr_mult': 4.5, 'tp_atr_mult': 4.0, 'max_hold': 20,
        'trail_act_atr': 0.06, 'trail_dist_atr': 0.01,
        'd1_ema20_filter': True, 'atr_floor': 0.1,
    })

    # Dual Thrust (sl_atr=6.0, original cap=5.0 → tighter, EFFECTIVE)
    results['DUAL_THRUST'] = _bench('DUAL_THRUST', bt_dual_thrust, h1, sl_atr=6.0, base_kwargs={
        'n_bars': 6, 'k_up': 0.5, 'k_down': 0.5,
        'sl_atr_mult': 6.0, 'tp_atr_mult': 8.0, 'max_hold': 20,
        'trail_act_atr': 0.06, 'trail_dist_atr': 0.01,
        'atr_floor': 0.1,
    })

    # Write JSON
    out = {name: rows for name, rows in results.items()}
    out_json = OUTPUT_DIR / 'sweep_results.json'
    out_json.write_text(json.dumps(out, indent=2))
    print(f'\nSaved -> {out_json}')

    # Build markdown report
    md = ['# R245 — Cap_ATR Sweep Report', '']
    md.append('_Sweeping cap_atr_mult to find Sharpe-optimal maxloss cap level._')
    md.append('_Cap formula: cap_dollars = cap_atr_mult x ATR x LOT x PV (same as BacktestEngine)._')
    md.append('')

    for strat, rows in results.items():
        sl_atr = {'TSMOM': 3.5, 'SESS_BO': 4.5, 'DUAL_THRUST': 6.0}[strat]
        md.append(f'## {strat} (sl_atr={sl_atr})')
        md.append('')
        md.append('| cap_atr | Effective | Full N | Full Sharpe | Full PnL | Full MaxDD | Recent Sharpe | Cap fires | Cap % |')
        md.append('|---|---|---|---|---|---|---|---|---|')

        # Build rows with bolded best
        best_full = max(rows, key=lambda r: r['full_sharpe'])
        best_rec = max(rows, key=lambda r: r['recent_sharpe'])
        for r in rows:
            mark_full = ' **<- max full**' if r is best_full else ''
            mark_rec = ' **<- max recent**' if r is best_rec else ''
            md.append(
                f"| {r['cap_atr']} | {r['effective']} | {r['full_n']} | "
                f"{r['full_sharpe']:.3f}{mark_full} | ${r['full_pnl']:.0f} | "
                f"${r['full_max_dd']:.0f} | {r['recent_sharpe']:.2f}{mark_rec} | "
                f"{r['cap_fires']} | {r['cap_fire_pct']:.1f}% |"
            )
        md.append('')

        # Recommendation
        baseline = next((r for r in rows if r['cap_atr'] == 0), rows[0])
        live_cap = {'TSMOM': 6.5, 'SESS_BO': 5.0, 'DUAL_THRUST': 5.0}[strat]
        live_row = next((r for r in rows if r['cap_atr'] == live_cap), None)
        md.append(f'**Live current**: cap_atr={live_cap}')
        if live_row:
            md.append(
                f"- Live cap: Sharpe={live_row['full_sharpe']:.3f}, PnL=${live_row['full_pnl']:.0f}, Recent={live_row['recent_sharpe']:.2f}"
            )
        md.append(f"- Baseline (cap off): Sharpe={baseline['full_sharpe']:.3f}, PnL=${baseline['full_pnl']:.0f}, Recent={baseline['recent_sharpe']:.2f}")
        md.append(f"- **Sharpe-optimal full**: cap_atr={best_full['cap_atr']} (Sharpe={best_full['full_sharpe']:.3f}, PnL=${best_full['full_pnl']:.0f})")
        md.append(f"- **Sharpe-optimal recent**: cap_atr={best_rec['cap_atr']} (Sharpe={best_rec['recent_sharpe']:.3f}, PnL=${best_rec['recent_pnl']:.0f})")

        # Era robustness check on best_full and best_rec
        md.append('')
        md.append('**Era robustness of optimal cap (Sharpe per era)**')
        md.append('')
        md.append('| cap_atr | Pre-COVID | COVID+Recovery | Tightening | Recent | All positive? |')
        md.append('|---|---|---|---|---|---|')
        check_caps = sorted({0, best_full['cap_atr'], best_rec['cap_atr'], live_cap})
        for cap_val in check_caps:
            r = next((r for r in rows if r['cap_atr'] == cap_val), None)
            if not r:
                continue
            eras = r['eras']
            all_pos = all(eras[k]['sharpe'] > 0 for k in eras)
            era_cells = ' | '.join(
                f"S={eras[k]['sharpe']:+.2f} N={eras[k]['n']} PnL=${eras[k]['pnl']:.0f}"
                for k in ['Pre-COVID', 'COVID+Recovery', 'Tightening', 'Recent']
            )
            md.append(f"| **{cap_val}** | {era_cells} | {'✓ ALL POS' if all_pos else '✗ has negative'} |")
        md.append('')

        # Verdict
        improvement_full = best_full['full_sharpe'] - baseline['full_sharpe']
        improvement_rec = best_rec['recent_sharpe'] - baseline['recent_sharpe']
        # Era robustness of optimal
        best_full_eras = best_full['eras']
        all_pos_full = all(best_full_eras[k]['sharpe'] > 0 for k in best_full_eras)
        if improvement_full > 0.1 and all_pos_full:
            verdict = (f'**RETUNE recommended** — optimal cap_atr={best_full["cap_atr"]} improves full Sharpe by '
                       f'{improvement_full:+.2f} AND all eras positive (era-robust).')
        elif improvement_full > 0.1:
            verdict = (f'**TENTATIVE retune** — cap_atr={best_full["cap_atr"]} improves Sharpe but has at least one '
                       f'negative era → possible over-fit.')
        elif improvement_rec > 0.2:
            verdict = f'**CONSIDER retune** — improves recent Sharpe by {improvement_rec:+.2f} but full marginal.'
        else:
            verdict = '**Current setting OK** — sweep shows no significant improvement.'
        md.append(f'**Verdict**: {verdict}')
        md.append('')

    out_md = OUTPUT_DIR / 'SUMMARY.md'
    out_md.write_text('\n'.join(md), encoding='utf-8')
    print(f'Saved -> {out_md}')


if __name__ == '__main__':
    main()
