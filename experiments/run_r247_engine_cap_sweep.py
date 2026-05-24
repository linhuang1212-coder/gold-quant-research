#!/usr/bin/env python3
"""R247 — BacktestEngine Cap_ATR Sweep + PSAR/Chandelier Revival Test
=====================================================================
Two phases:

PHASE A — Live 5-Strategy Portfolio Cap Sweep
  - Same setup as R246 (Keltner + M15_RSI + ORB + SESS_BO + Dual Thrust)
  - Sweep maxloss_cap_atr_mult in [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
  - Identify portfolio-level Sharpe-optimal cap
  - Q: Does R245's cap=1.0 finding hold when full engine filters are applied?

PHASE B — PSAR + Chandelier Isolated + Cap Sweep
  - Reuse R209v2 isolated phase setup
  - For psar and chandelier separately, sweep cap_atr in [0, 1.0, 1.5, 2.0, 3.0]
  - Era-robust check: all 4 eras positive?
  - Q: Can PSAR/Chandelier be revived with cap=1.0?
"""
from __future__ import annotations
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.runner import DataBundle, LIVE_PARITY_KWARGS
from backtest.engine import BacktestEngine
import indicators as signals_mod
import research_config as config

from experiments.run_r209v2_non_keltner_audit import (
    check_psar_signal, check_sess_bo_signal,
    check_dual_thrust_signal, check_chandelier_signal,
)

OUTPUT_DIR = Path("results/r247_engine_cap_sweep")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LIVE_PERIOD = ("2026-03-25", "2026-05-13")

ERAS = [
    ('Pre-COVID', '2015-01-01', '2019-12-31'),
    ('COVID+Recovery', '2020-01-01', '2021-12-31'),
    ('Tightening', '2022-01-01', '2023-12-31'),
    ('Recent', '2024-01-01', '2026-05-13'),
]

# Cap sweep values — tuned for ~50 min total runtime
CAP_SWEEP_A = [0, 1.0, 1.5, 2.0, 3.0]   # 5 portfolio runs × ~8min each = 40min
CAP_SWEEP_B = [0, 1.0, 2.0, 3.0]         # 4 caps × 2 strategies isolated = 8 × ~1.5min = 12min

# 5-strategy LIVE config (PSAR + Chandelier OFF)
LIVE5_CONFIG = {
    'psar': {'enabled': False},
    'sess_bo': {
        'enabled': True, 'broker_gmt_offset': 0, 'session_hour_gmt': 12,
        'lookback_bars': 4, 'sl_atr': 4.5, 'tp_atr': 4.0,
    },
    'dual_thrust': {
        'enabled': True, 'n_bars': 6, 'k_up': 0.5, 'k_down': 0.5,
        'sl_atr': 6.0, 'tp_atr': 8.0,
        'trail_act_atr': 0.06, 'trail_dist_atr': 0.01,
    },
    'chandelier': {'enabled': False},
}

# Full 7-strategy config for Phase B isolation
FULL_CONFIG = {
    'psar': {
        'enabled': True, 'psar_af_step': 0.01, 'psar_af_max': 0.05,
        'min_atr': 0.1, 'sl_atr': 4.0, 'tp_atr': 6.0,
        'trail_act_atr': 0.08, 'trail_dist_atr': 0.015,
    },
    'sess_bo': {
        'enabled': True, 'broker_gmt_offset': 0, 'session_hour_gmt': 12,
        'lookback_bars': 4, 'sl_atr': 4.5, 'tp_atr': 4.0,
    },
    'dual_thrust': {
        'enabled': True, 'n_bars': 6, 'k_up': 0.5, 'k_down': 0.5,
        'sl_atr': 4.5, 'tp_atr': 8.0,
        'trail_act_atr': 0.14, 'trail_dist_atr': 0.025,
    },
    'chandelier': {
        'enabled': True, 'chand_period': 22, 'chand_mult': 3.0,
        'rsi_filter': True, 'sl_atr': 4.5, 'tp_atr': 8.0,
        'trail_act_atr': 0.14, 'trail_dist_atr': 0.025,
    },
}


import experiments.run_r209v2_non_keltner_audit as r209_mod


def patch_indicators(strat_config):
    """Monkey-patch signals_mod.scan_all_signals with the given config."""
    r209_mod.LIVE_STRAT_CONFIGS = strat_config

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


def _trade_attr(t, name, default=None):
    if hasattr(t, name):
        return getattr(t, name)
    if isinstance(t, dict):
        return t.get(name, default)
    return default


def calc_stats(trades, label=''):
    if not trades:
        return {'label': label, 'n': 0, 'sharpe': 0.0, 'pnl': 0.0,
                'win_rate': 0.0, 'max_dd': 0.0}
    pnls = np.array([_trade_attr(t, 'pnl', 0) for t in trades])
    n = len(pnls)
    wr = (pnls > 0).mean() * 100
    cum = np.cumsum(pnls)
    dd = (np.maximum.accumulate(cum) - cum).max()
    sharpe = pnls.mean() / pnls.std() * np.sqrt(252) if pnls.std() > 0 else 0
    return {
        'label': label, 'n': n, 'pnl': round(float(pnls.sum()), 2),
        'sharpe': round(sharpe, 3), 'win_rate': round(wr, 2),
        'max_dd': round(dd, 2),
    }


def filter_period(trades, start, end):
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


def per_strat(trades):
    by = defaultdict(list)
    for t in trades:
        by[_trade_attr(t, 'strategy', 'unknown')].append(t)
    return by


def run_one(data, kwargs, max_positions=4):
    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False
    from indicators import get_orb_strategy
    get_orb_strategy().reset_daily()

    k = {**kwargs, 'max_positions': max_positions}
    eng = BacktestEngine(data.m15_df, data.h1_df, **k)
    trades = eng.run()
    cap_count = getattr(eng, 'maxloss_cap_count', 0)
    return trades, cap_count


def phase_a(data):
    """Phase A: 5-strategy live portfolio cap sweep."""
    print('\n' + '=' * 80)
    print('PHASE A: 5-Strategy Live Portfolio — Cap_ATR Sweep')
    print('=' * 80)
    print(f'  Cap values: {CAP_SWEEP_A}')

    patch_indicators(LIVE5_CONFIG)

    results = []
    for cap in CAP_SWEEP_A:
        t0 = time.time()
        kwargs = {**LIVE_PARITY_KWARGS, 'maxloss_cap_atr_mult': cap, 'maxloss_cap': 0}
        trades, cap_fires = run_one(data, kwargs, max_positions=4)
        elapsed = time.time() - t0

        # Per-strategy stats
        by = per_strat(trades)
        per_s = {s: calc_stats(ts, s) for s, ts in by.items()}

        # Portfolio
        port = calc_stats(trades, 'portfolio')

        # Live period
        live_trades = filter_period(trades, *LIVE_PERIOD)
        live_port = calc_stats(live_trades, 'live_port')

        row = {
            'cap_atr': cap,
            'cap_fires_total': cap_fires,
            'port_n': port['n'],
            'port_sharpe': port['sharpe'],
            'port_pnl': port['pnl'],
            'port_max_dd': port['max_dd'],
            'live_n': live_port['n'],
            'live_sharpe': live_port['sharpe'],
            'live_pnl': live_port['pnl'],
            'per_strat': per_s,
            'wall_sec': round(elapsed, 1),
        }
        results.append(row)
        print(f"\n  cap={cap}  N={port['n']:>6}  Sharpe={port['sharpe']:+5.2f}  "
              f"PnL=${port['pnl']:>+9.0f}  DD=${port['max_dd']:>6.0f}  "
              f"caps_fired={cap_fires}  ({elapsed:.0f}s)")
        # Per-strategy
        for s in sorted(per_s.keys()):
            ps = per_s[s]
            print(f"    {s:<12} N={ps['n']:>5}  Sharpe={ps['sharpe']:+5.2f}  PnL=${ps['pnl']:>+8.0f}")

    return results


def phase_b(data):
    """Phase B: PSAR + Chandelier isolated, cap sweep."""
    print('\n' + '=' * 80)
    print('PHASE B: PSAR + Chandelier Isolated — Cap Revival Test')
    print('=' * 80)

    targets = ['psar', 'chandelier']
    results = {}

    for target in targets:
        print(f'\n  === {target.upper()} (isolated) ===')
        # Configure: only this strategy enabled
        cfg = {k: {**v} for k, v in FULL_CONFIG.items()}
        for s in cfg:
            cfg[s]['enabled'] = (s == target)
        patch_indicators(cfg)

        rows = []
        for cap in CAP_SWEEP_B:
            t0 = time.time()
            # Disable Keltner SL/TP override for isolated, max_positions=1
            kwargs = {**LIVE_PARITY_KWARGS,
                      'sl_atr_mult': 0, 'tp_atr_mult': 0,
                      'maxloss_cap_atr_mult': cap, 'maxloss_cap': 0}
            trades, cap_fires = run_one(data, kwargs, max_positions=1)
            elapsed = time.time() - t0

            target_trades = [t for t in trades if _trade_attr(t, 'strategy') == target]
            st_full = calc_stats(target_trades, f'{target}_full')
            era_stats = {}
            for era_name, start, end in ERAS:
                era_t = filter_period(target_trades, start, end)
                era_stats[era_name] = calc_stats(era_t, f'{era_name}')
            all_pos = all(era_stats[k]['sharpe'] > 0 for k in era_stats)

            row = {
                'cap_atr': cap,
                'n': st_full['n'],
                'sharpe': st_full['sharpe'],
                'pnl': st_full['pnl'],
                'max_dd': st_full['max_dd'],
                'cap_fires': cap_fires,
                'eras': {k: {'n': v['n'], 'sharpe': v['sharpe'], 'pnl': v['pnl']}
                         for k, v in era_stats.items()},
                'all_eras_positive': all_pos,
                'wall_sec': round(elapsed, 1),
            }
            rows.append(row)
            era_str = ' '.join(f"{k[:3]}={v['sharpe']:+5.2f}" for k, v in era_stats.items())
            print(f"    cap={cap}  N={st_full['n']:>5}  Sharpe={st_full['sharpe']:+6.3f}  "
                  f"PnL=${st_full['pnl']:>+7.0f}  All_pos={'YES' if all_pos else 'no'}  "
                  f"| {era_str}  ({elapsed:.0f}s)")
        results[target] = rows

    return results


def main():
    t0 = time.time()
    print('=' * 80)
    print('R247: Engine-Level Cap_ATR Sweep + PSAR/Chandelier Revival')
    print('=' * 80)

    print('\nLoading data...')
    data = DataBundle.load_default()

    phase_a_results = phase_a(data)
    phase_b_results = phase_b(data)

    output = {
        'phase_a_live5_portfolio_sweep': phase_a_results,
        'phase_b_psar_chandelier_revival': phase_b_results,
        'runtime_sec': round(time.time() - t0, 1),
    }
    out_json = OUTPUT_DIR / 'R247_results.json'
    out_json.write_text(json.dumps(output, indent=2))
    print(f'\nSaved -> {out_json}')

    # ─── Build markdown summary ────────────────────────────────────
    md = ['# R247 — Engine Cap Sweep + PSAR/Chandelier Revival', '']
    md.append('## Phase A: 5-Strategy Live Portfolio Cap Sweep')
    md.append('')
    md.append('_Validates whether R245 standalone finding (cap_atr=1.0 optimal) holds via BacktestEngine._')
    md.append('')
    md.append('| cap_atr | Port N | Port Sharpe | Port PnL | Port MaxDD | Cap fires | Live Sharpe | Live PnL |')
    md.append('|---|---|---|---|---|---|---|---|')
    best = max(phase_a_results, key=lambda r: r['port_sharpe'])
    for r in phase_a_results:
        mark = ' **<- best**' if r is best else ''
        md.append(f"| **{r['cap_atr']}** | {r['port_n']} | {r['port_sharpe']:.3f}{mark} | "
                  f"${r['port_pnl']:.0f} | ${r['port_max_dd']:.0f} | {r['cap_fires_total']} | "
                  f"{r['live_sharpe']:.2f} | ${r['live_pnl']:.0f} |")
    md.append('')
    md.append(f"**Best: cap_atr={best['cap_atr']} with Sharpe {best['port_sharpe']}**")
    md.append('')
    md.append('### Per-Strategy Sharpe across cap values (Phase A)')
    md.append('')
    strats = sorted({s for r in phase_a_results for s in r['per_strat']})
    md.append('| cap_atr | ' + ' | '.join(strats) + ' |')
    md.append('|---|' + '---|' * len(strats))
    for r in phase_a_results:
        row = [f"**{r['cap_atr']}**"]
        for s in strats:
            ps = r['per_strat'].get(s, {})
            row.append(f"S={ps.get('sharpe', 0):.2f} N={ps.get('n', 0)} PnL=${ps.get('pnl', 0):.0f}")
        md.append('| ' + ' | '.join(row) + ' |')
    md.append('')

    md.append('## Phase B: PSAR + Chandelier Revival Test (Isolated + Cap)')
    md.append('')
    for target in ['psar', 'chandelier']:
        rows = phase_b_results[target]
        md.append(f'### {target.upper()}')
        md.append('')
        md.append('| cap_atr | N | Full Sharpe | PnL | MaxDD | Pre-COVID | COVID | Tightening | Recent | All eras positive? |')
        md.append('|---|---|---|---|---|---|---|---|---|---|')
        for r in rows:
            era_cells = ' | '.join(
                f"{r['eras'][k]['sharpe']:+.2f}"
                for k in ['Pre-COVID', 'COVID+Recovery', 'Tightening', 'Recent']
            )
            mark = ' ✓ REVIVE' if r['all_eras_positive'] and r['sharpe'] > 0.5 else (
                ' ✓ all pos' if r['all_eras_positive'] else '')
            md.append(f"| **{r['cap_atr']}** | {r['n']} | {r['sharpe']:.3f}{mark} | "
                      f"${r['pnl']:.0f} | ${r['max_dd']:.0f} | {era_cells} | "
                      f"{'YES' if r['all_eras_positive'] else 'no'} |")
        md.append('')

        # Verdict
        positive_caps = [r for r in rows if r['all_eras_positive'] and r['sharpe'] > 0.5]
        if positive_caps:
            best = max(positive_caps, key=lambda r: r['sharpe'])
            md.append(f"**{target.upper()} verdict**: ✓ Revivable at cap_atr={best['cap_atr']} "
                      f"(Sharpe {best['sharpe']:.2f}, all eras positive)")
        else:
            md.append(f"**{target.upper()} verdict**: ✗ Cannot revive — no cap gives era-robust positive Sharpe > 0.5")
        md.append('')

    out_md = OUTPUT_DIR / 'SUMMARY.md'
    out_md.write_text('\n'.join(md), encoding='utf-8')
    print(f'Saved -> {out_md}')
    print(f'Total runtime: {(time.time() - t0):.0f}s')


if __name__ == '__main__':
    main()
