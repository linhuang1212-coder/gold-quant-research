#!/usr/bin/env python3
"""R243 — Factor Re-validation triggered by IC report 2026-05-21
==================================================================
Context:
  - 20+ days of IC monitoring on 160 live trades + 832 paper trades
  - IC report findings (data/ic_report.json, archived 2026-05-21):
    * Rule B disabled (config.RULEB_ENABLED=False) pending re-test
    * RSI14 / RSI2 IC decaying: first_half=-0.29 → second_half=+0.04 (reversed!)
    * ADX IC strengthening: first_half=-0.09 → second_half=+0.20 (trend regime)
    * atr_percentile IC decaying: -0.142 → -0.023
    * ATR / Vol_MA20 / MACD_hist remain stable (IR>1.0)

Phases:
  Phase 0 — Baseline backtest (LIVE_PARITY + m30_rsi14 enabled)
            * Full period (2015-2026-04) + Most-recent era (2025-09 to 2026-04)
  Phase 1 — Rule B parameter sweep via POST-HOC FILTER
            (Rule B is NOT in engine for portfolio-level filtering,
             so we simulate by filtering baseline trade entries that fall
             inside Rule B cooldown windows.)
            sigma ∈ [2.0, 2.5, 3.0, 3.5, 4.0]
            lookback ∈ [60, 90, 120]
            skip_hours ∈ [4, 8, 12]
            Plus baseline (no Rule B) = 45 + 1 configs
  Phase 2 — RSI decay validation
            * m30_rsi14 only (m30_only mode) era-segmented
            * m15_rsi (custom rsi thresholds) era-segmented
            * Compare era Sharpe to detect "alpha has gone" signal
  Phase 3 — ATR_PCTL_FLOOR threshold sweep via post-hoc filter
            threshold ∈ [0, 15, 20, 25, 30, 35, 40] (in percent)
  Phase 4 — ADX entry filter sweep via engine
            keltner_adx_threshold ∈ [0, 14, 18, 22, 26, 30]
            (Disables session-aware ADX to use global threshold)

Output:
  results/r243_factor_revalidation/
    phase0_baseline.json
    phase1_ruleb_sweep.json
    phase2_rsi_decay.json
    phase3_atr_pctl_sweep.json
    phase4_adx_sweep.json
    SUMMARY.md
"""
from __future__ import annotations
import sys
import os
import json
import time
import warnings
from pathlib import Path
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

from backtest.runner import DataBundle, LIVE_PARITY_KWARGS
from backtest.engine import BacktestEngine, TradeRecord
from backtest.m30_engine import load_m30_with_indicators
from backtest.stats import calc_stats, aggregate_daily_pnl

OUTPUT_DIR = Path("results/r243_factor_revalidation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

T0 = time.time()

# ─────────────────────────────────────────────────────────────────
# Eras (matched to IC report semantics)
# ─────────────────────────────────────────────────────────────────
ERAS = [
    ("Pre-COVID",     "2015-01-01", "2020-01-01"),
    ("COVID-Inflate", "2020-01-01", "2023-01-01"),
    ("Recent",        "2023-01-01", "2026-04-10"),
    ("Most-Recent",   "2025-09-01", "2026-04-10"),  # ~7 months ≈ IC report window
]

MOST_RECENT_START = "2025-09-01"
DATA_END = "2026-04-10"


def stamp(msg: str):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def trades_to_records(trades: List[TradeRecord]) -> List[Dict]:
    """Convert TradeRecord objects to JSON-serializable dicts."""
    out = []
    for t in trades:
        out.append({
            'strategy': t.strategy,
            'direction': t.direction,
            'entry_price': float(t.entry_price),
            'exit_price': float(t.exit_price),
            'entry_time': pd.Timestamp(t.entry_time).isoformat(),
            'exit_time': pd.Timestamp(t.exit_time).isoformat(),
            'lots': float(t.lots),
            'pnl': float(t.pnl),
            'exit_reason': t.exit_reason,
            'bars_held': int(t.bars_held),
        })
    return out


def stats_summary(trades: List[TradeRecord], label: str = "") -> Dict:
    """Compact stat summary (subset of calc_stats)."""
    if not trades:
        return {'label': label, 'n': 0, 'sharpe': 0.0, 'pnl': 0.0,
                'win_rate': 0.0, 'max_dd': 0.0, 'days': 0}
    pnls = np.array([t.pnl for t in trades])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    win_rate = float(len(wins) / len(pnls)) if len(pnls) > 0 else 0.0
    daily = {}
    for t in trades:
        d = pd.Timestamp(t.exit_time).date()
        daily[d] = daily.get(d, 0.0) + t.pnl
    daily_arr = np.array(list(daily.values()))
    sharpe = 0.0
    if len(daily_arr) > 1 and daily_arr.std(ddof=1) > 0:
        sharpe = float(daily_arr.mean() / daily_arr.std(ddof=1) * np.sqrt(252))
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    max_dd = float((peak - cum).max()) if len(cum) > 0 else 0.0
    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(abs(losses.mean())) if len(losses) > 0 else 0.0
    rr = float(avg_win / avg_loss) if avg_loss > 0 else 0.0
    pf = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else 0.0
    return {
        'label': label,
        'n': int(len(pnls)),
        'sharpe': round(sharpe, 3),
        'pnl': round(float(pnls.sum()), 2),
        'win_rate': round(win_rate, 4),
        'max_dd': round(max_dd, 2),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'rr': round(rr, 3),
        'pf': round(pf, 3),
        'days': int(len(daily_arr)),
    }


def m30_sig_rsi14_trend(df):
    """M30 RSI14 trend-confirmed signal (from R236)."""
    if len(df) < 55:
        return None
    row = df.iloc[-1]
    rsi = float(row.get('RSI14', 50))
    c = float(row['Close'])
    ema50 = float(row.get('EMA50', c))
    slope = float(row.get('EMA50_slope', 0))
    if pd.isna(rsi) or float(row.get('ATR', 0)) <= 0:
        return None
    if rsi < 30 and c > ema50 and slope > 0:
        return {'strategy': 'm30_rsi14', 'signal': 'BUY'}
    if rsi > 70 and c < ema50 and slope < 0:
        return {'strategy': 'm30_rsi14', 'signal': 'SELL'}
    return None


# ═══════════════════════════════════════════════════════════════
# Phase 0: Baseline
# ═══════════════════════════════════════════════════════════════

def run_baseline(data: DataBundle, m30_df: pd.DataFrame, label: str) -> Tuple[List[TradeRecord], Dict]:
    """Run baseline backtest = LIVE_PARITY_KWARGS + m30_rsi14 enabled."""
    kwargs = dict(LIVE_PARITY_KWARGS)
    kwargs.update({
        'm30_df': m30_df,
        'm30_enabled': True,
        'm30_signal_func': m30_sig_rsi14_trend,
        'm30_strategy_name': 'm30_rsi14',
        'm30_sl_atr_mult': 8.0,
        'm30_tp_atr_mult': 8.0,
        'm30_trail_activate_atr': 0.50,
        'm30_trail_distance_atr': 0.15,
        'm30_max_hold_m15': 96,
        'm30_cooldown_bars': 8,
        'label': label,
    })
    import indicators as signals_mod
    from indicators import get_orb_strategy
    get_orb_strategy().reset_daily()
    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False
    engine = BacktestEngine(data.m15_df, data.h1_df, **kwargs)
    trades = engine.run()
    return trades, stats_summary(trades, label)


def phase0(full_data: DataBundle, m30_df: pd.DataFrame) -> Dict:
    stamp("PHASE 0 ── Baseline backtest")
    # Full period
    full_trades, full_stats = run_baseline(full_data, m30_df, "baseline_full")
    stamp(f"  Full: n={full_stats['n']}, Sharpe={full_stats['sharpe']}, "
          f"PnL=${full_stats['pnl']:.0f}")

    # Most-recent era (slice m30_df accordingly)
    recent_data = full_data.slice(MOST_RECENT_START, DATA_END)
    recent_m30 = m30_df[(m30_df.index >= pd.Timestamp(MOST_RECENT_START, tz='UTC')) &
                        (m30_df.index <  pd.Timestamp(DATA_END, tz='UTC'))]
    recent_trades, recent_stats = run_baseline(recent_data, recent_m30, "baseline_recent7mo")
    stamp(f"  Recent-7mo: n={recent_stats['n']}, Sharpe={recent_stats['sharpe']}, "
          f"PnL=${recent_stats['pnl']:.0f}")

    # Per-strategy breakdown for full period
    strat_breakdown = {}
    for s in set(t.strategy for t in full_trades):
        ts = [t for t in full_trades if t.strategy == s]
        strat_breakdown[s] = stats_summary(ts, s)

    result = {
        'full_period': full_stats,
        'recent_7mo': recent_stats,
        'strategy_breakdown_full': strat_breakdown,
        'full_trades_json': trades_to_records(full_trades),  # save raw for phase 1/3
        'recent_trades_json': trades_to_records(recent_trades),
    }
    (OUTPUT_DIR / "phase0_baseline.json").write_text(
        json.dumps({k: v for k, v in result.items() if not k.endswith('_json')},
                   indent=2), encoding='utf-8')
    (OUTPUT_DIR / "phase0_trades_full.json").write_text(
        json.dumps(result['full_trades_json'], indent=1), encoding='utf-8')
    (OUTPUT_DIR / "phase0_trades_recent.json").write_text(
        json.dumps(result['recent_trades_json'], indent=1), encoding='utf-8')

    return result


# ═══════════════════════════════════════════════════════════════
# Phase 1: Rule B post-hoc filter sweep
# ═══════════════════════════════════════════════════════════════

def compute_ruleb_skip_windows(h1_df: pd.DataFrame,
                                sigma: float,
                                lookback: int,
                                skip_hours: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """For each H1 bar, check if ATR > rolling_mean + sigma * rolling_std.
    Returns list of (start, end) windows during which entries are blocked
    (with 8h sliding cooldown — matches live behavior of RULEB_SKIP_HOURS)."""
    atr = h1_df['ATR'].values
    n = len(atr)
    if n < lookback + 1:
        return []
    rolling_mean = pd.Series(atr).rolling(lookback).mean().values
    rolling_std = pd.Series(atr).rolling(lookback).std(ddof=1).values
    times = h1_df.index

    windows = []
    skip_delta = pd.Timedelta(hours=skip_hours)
    last_end: Optional[pd.Timestamp] = None
    for i in range(lookback, n):
        m = rolling_mean[i]
        s = rolling_std[i]
        if pd.isna(m) or pd.isna(s) or s <= 0:
            continue
        if atr[i] > m + sigma * s:
            trig_time = times[i]
            # if extending an existing window, just extend; else start new
            if last_end is not None and trig_time <= last_end:
                new_end = trig_time + skip_delta
                if new_end > last_end:
                    windows[-1] = (windows[-1][0], new_end)
                    last_end = new_end
            else:
                end = trig_time + skip_delta
                windows.append((trig_time, end))
                last_end = end
    return windows


def filter_trades_by_windows(trades_json: List[Dict],
                              windows: List[Tuple[pd.Timestamp, pd.Timestamp]]
                              ) -> List[Dict]:
    """Remove trades whose entry_time falls inside any block window."""
    if not windows:
        return trades_json
    times = [(w[0].value, w[1].value) for w in windows]
    times.sort()
    keep = []
    for t in trades_json:
        et = pd.Timestamp(t['entry_time']).value
        blocked = False
        for ws, we in times:
            if ws <= et < we:
                blocked = True
                break
            if ws > et:
                break
        if not blocked:
            keep.append(t)
    return keep


def stats_from_json_trades(trades_json: List[Dict], label: str = "") -> Dict:
    if not trades_json:
        return {'label': label, 'n': 0, 'sharpe': 0.0, 'pnl': 0.0,
                'win_rate': 0.0, 'max_dd': 0.0, 'days': 0}
    pnls = np.array([t['pnl'] for t in trades_json])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    win_rate = float(len(wins) / len(pnls)) if len(pnls) > 0 else 0.0
    daily = {}
    for t in trades_json:
        d = pd.Timestamp(t['exit_time']).date()
        daily[d] = daily.get(d, 0.0) + t['pnl']
    daily_arr = np.array(list(daily.values()))
    sharpe = 0.0
    if len(daily_arr) > 1 and daily_arr.std(ddof=1) > 0:
        sharpe = float(daily_arr.mean() / daily_arr.std(ddof=1) * np.sqrt(252))
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    max_dd = float((peak - cum).max()) if len(cum) > 0 else 0.0
    pf = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else 0.0
    return {
        'label': label,
        'n': int(len(pnls)),
        'sharpe': round(sharpe, 3),
        'pnl': round(float(pnls.sum()), 2),
        'win_rate': round(win_rate, 4),
        'max_dd': round(max_dd, 2),
        'pf': round(pf, 3),
        'days': int(len(daily_arr)),
    }


def phase1(phase0_result: Dict, h1_df: pd.DataFrame) -> Dict:
    stamp("PHASE 1 ── Rule B parameter sweep (post-hoc)")

    full_trades = phase0_result['full_trades_json']
    recent_trades = phase0_result['recent_trades_json']

    sigmas = [2.0, 2.5, 3.0, 3.5, 4.0]
    lookbacks = [60, 90, 120]
    skip_hours_list = [4, 8, 12]

    results_full = []
    results_recent = []

    # Baseline (no Rule B)
    baseline_full = stats_from_json_trades(full_trades, "no_ruleb")
    baseline_recent = stats_from_json_trades(recent_trades, "no_ruleb")
    results_full.append({**baseline_full, 'sigma': None, 'lookback': None, 'skip_hours': None,
                          'blocked_pct': 0.0, 'windows': 0})
    results_recent.append({**baseline_recent, 'sigma': None, 'lookback': None, 'skip_hours': None,
                            'blocked_pct': 0.0, 'windows': 0})

    h1_recent = h1_df[(h1_df.index >= pd.Timestamp(MOST_RECENT_START, tz='UTC')) &
                       (h1_df.index <  pd.Timestamp(DATA_END, tz='UTC'))]

    total = len(sigmas) * len(lookbacks) * len(skip_hours_list)
    done = 0
    for sigma, lb, sh in product(sigmas, lookbacks, skip_hours_list):
        done += 1
        windows_full = compute_ruleb_skip_windows(h1_df, sigma, lb, sh)
        windows_recent = [(s, e) for s, e in windows_full
                          if s >= pd.Timestamp(MOST_RECENT_START, tz='UTC') and
                          s < pd.Timestamp(DATA_END, tz='UTC')]

        kept_full = filter_trades_by_windows(full_trades, windows_full)
        kept_recent = filter_trades_by_windows(recent_trades, windows_recent)

        blocked_full = len(full_trades) - len(kept_full)
        blocked_recent = len(recent_trades) - len(kept_recent)
        block_pct_full = blocked_full / max(len(full_trades), 1) * 100
        block_pct_recent = blocked_recent / max(len(recent_trades), 1) * 100

        lbl = f"σ{sigma}_lb{lb}_cd{sh}h"
        stats_full = stats_from_json_trades(kept_full, lbl)
        stats_recent = stats_from_json_trades(kept_recent, lbl)
        stats_full.update({'sigma': sigma, 'lookback': lb, 'skip_hours': sh,
                            'blocked_pct': round(block_pct_full, 2),
                            'blocked_trades': blocked_full,
                            'windows': len(windows_full)})
        stats_recent.update({'sigma': sigma, 'lookback': lb, 'skip_hours': sh,
                              'blocked_pct': round(block_pct_recent, 2),
                              'blocked_trades': blocked_recent,
                              'windows': len(windows_recent)})
        results_full.append(stats_full)
        results_recent.append(stats_recent)

        if done % 10 == 0 or done == total:
            stamp(f"  Rule B sweep {done}/{total}: {lbl} "
                  f"full Sharpe={stats_full['sharpe']:.2f} (Δ={stats_full['sharpe']-baseline_full['sharpe']:+.2f}) "
                  f"blocked={block_pct_full:.1f}%")

    # Rank by Sharpe gain
    def rank_by_sharpe(arr):
        return sorted(arr, key=lambda x: x['sharpe'], reverse=True)
    ranked_full = rank_by_sharpe(results_full)
    ranked_recent = rank_by_sharpe(results_recent)

    payload = {
        'baseline_full': baseline_full,
        'baseline_recent': baseline_recent,
        'all_configs_full': results_full,
        'all_configs_recent': results_recent,
        'top5_full': ranked_full[:5],
        'top5_recent': ranked_recent[:5],
        'bottom3_full': ranked_full[-3:],
        'bottom3_recent': ranked_recent[-3:],
    }
    (OUTPUT_DIR / "phase1_ruleb_sweep.json").write_text(
        json.dumps(payload, indent=2), encoding='utf-8')

    stamp(f"  PHASE 1 done. Top full: {ranked_full[0]['label']} "
          f"Sharpe={ranked_full[0]['sharpe']} (baseline={baseline_full['sharpe']})")
    stamp(f"  Top recent: {ranked_recent[0]['label']} "
          f"Sharpe={ranked_recent[0]['sharpe']} (baseline={baseline_recent['sharpe']})")
    return payload


# ═══════════════════════════════════════════════════════════════
# Phase 2: RSI decay validation (era-segmented, isolate signals)
# ═══════════════════════════════════════════════════════════════

def run_m30_only(data: DataBundle, m30_df: pd.DataFrame, label: str
                  ) -> Tuple[List[TradeRecord], Dict]:
    """m30_rsi14 in isolation (Keltner/RSI effectively disabled)."""
    kwargs = dict(LIVE_PARITY_KWARGS)
    kwargs.update({
        'm30_df': m30_df,
        'm30_enabled': True,
        'm30_signal_func': m30_sig_rsi14_trend,
        'm30_strategy_name': 'm30_rsi14',
        'm30_sl_atr_mult': 8.0,
        'm30_tp_atr_mult': 8.0,
        'm30_trail_activate_atr': 0.50,
        'm30_trail_distance_atr': 0.15,
        'm30_max_hold_m15': 96,
        'm30_cooldown_bars': 8,
        # Disable other strategies via impossible ADX thresholds & RSI filter
        'keltner_session_adx': {
            'asia':    (0, 7, 999),
            'london':  (8, 12, 999),
            'ny':      (13, 17, 999),
            'evening': (18, 23, 999),
        },
        'rsi_adx_filter': 0.001,
        'label': label,
    })
    import indicators as signals_mod
    from indicators import get_orb_strategy
    get_orb_strategy().reset_daily()
    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False
    engine = BacktestEngine(data.m15_df, data.h1_df, **kwargs)
    trades = engine.run()
    m30_trades = [t for t in trades if t.strategy == 'm30_rsi14']
    return m30_trades, stats_summary(m30_trades, label)


def run_m15_rsi_only(data: DataBundle, label: str) -> Tuple[List[TradeRecord], Dict]:
    """m15_rsi (RSI2) in isolation."""
    kwargs = dict(LIVE_PARITY_KWARGS)
    kwargs.update({
        'keltner_session_adx': {
            'asia':    (0, 7, 999),
            'london':  (8, 12, 999),
            'ny':      (13, 17, 999),
            'evening': (18, 23, 999),
        },
        'label': label,
    })
    import indicators as signals_mod
    from indicators import get_orb_strategy
    get_orb_strategy().reset_daily()
    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False
    engine = BacktestEngine(data.m15_df, data.h1_df, **kwargs)
    trades = engine.run()
    rsi_trades = [t for t in trades if t.strategy == 'm15_rsi']
    return rsi_trades, stats_summary(rsi_trades, label)


def phase2(full_data: DataBundle, m30_df: pd.DataFrame) -> Dict:
    stamp("PHASE 2 ── RSI decay validation (era-segmented)")

    # m30_rsi14: era-segmented
    m30_results = {}
    for era_name, e_start, e_end in ERAS:
        era_data = full_data.slice(e_start, e_end)
        era_m30 = m30_df[(m30_df.index >= pd.Timestamp(e_start, tz='UTC')) &
                         (m30_df.index <  pd.Timestamp(e_end, tz='UTC'))]
        trades, st = run_m30_only(era_data, era_m30, f"m30_rsi14_{era_name}")
        m30_results[era_name] = st
        stamp(f"  m30_rsi14 [{era_name}]: n={st['n']} Sharpe={st['sharpe']} "
              f"PnL=${st['pnl']:.0f} WR={st['win_rate']*100:.1f}%")

    # m15_rsi: era-segmented
    rsi_results = {}
    for era_name, e_start, e_end in ERAS:
        era_data = full_data.slice(e_start, e_end)
        trades, st = run_m15_rsi_only(era_data, f"m15_rsi_{era_name}")
        rsi_results[era_name] = st
        stamp(f"  m15_rsi [{era_name}]: n={st['n']} Sharpe={st['sharpe']} "
              f"PnL=${st['pnl']:.0f} WR={st['win_rate']*100:.1f}%")

    # Decay verdict
    def decay_verdict(era_stats: Dict[str, Dict]) -> str:
        eras_order = ['Pre-COVID', 'COVID-Inflate', 'Recent', 'Most-Recent']
        sharpes = [era_stats[e]['sharpe'] for e in eras_order if e in era_stats]
        if len(sharpes) < 3 or sharpes[0] == 0:
            return "INCONCLUSIVE"
        early_avg = (sharpes[0] + sharpes[1]) / 2
        recent_avg = sharpes[-1]
        if recent_avg < 0 and early_avg > 0.3:
            return "DECAYED"
        if recent_avg < early_avg * 0.3:
            return "SEVERELY_DECAYED"
        if recent_avg < early_avg * 0.6:
            return "WEAKENED"
        return "STABLE"

    m30_decay = decay_verdict(m30_results)
    rsi_decay = decay_verdict(rsi_results)
    stamp(f"  m30_rsi14 decay verdict: {m30_decay}")
    stamp(f"  m15_rsi decay verdict: {rsi_decay}")

    payload = {
        'm30_rsi14_by_era': m30_results,
        'm15_rsi_by_era': rsi_results,
        'm30_decay_verdict': m30_decay,
        'm15_decay_verdict': rsi_decay,
    }
    (OUTPUT_DIR / "phase2_rsi_decay.json").write_text(
        json.dumps(payload, indent=2), encoding='utf-8')
    return payload


# ═══════════════════════════════════════════════════════════════
# Phase 3: ATR_PCTL_FLOOR threshold sweep (post-hoc)
# ═══════════════════════════════════════════════════════════════

def phase3(phase0_result: Dict, h1_df: pd.DataFrame) -> Dict:
    stamp("PHASE 3 ── ATR_PCTL_FLOOR threshold sweep (post-hoc)")
    full_trades = phase0_result['full_trades_json']
    recent_trades = phase0_result['recent_trades_json']

    # Build a fast lookup of atr_percentile at each trade entry
    h1_atr_pct = h1_df['atr_percentile']

    def lookup_atr_pct(entry_time_iso: str) -> float:
        t = pd.Timestamp(entry_time_iso)
        if t.tzinfo is None:
            t = t.tz_localize('UTC')
        # Floor to H1 boundary
        t_floor = t.floor('1H')
        try:
            return float(h1_atr_pct.loc[t_floor])
        except KeyError:
            # find nearest preceding H1 bar
            idx = h1_atr_pct.index.searchsorted(t_floor) - 1
            if idx < 0:
                return 0.5
            return float(h1_atr_pct.iloc[idx])

    # Annotate trades with atr_pct
    for t in full_trades:
        t['_atr_pct'] = lookup_atr_pct(t['entry_time'])
    for t in recent_trades:
        t['_atr_pct'] = lookup_atr_pct(t['entry_time'])

    thresholds = [0, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    results_full = []
    results_recent = []
    baseline_full = stats_from_json_trades(full_trades, "no_floor")
    baseline_recent = stats_from_json_trades(recent_trades, "no_floor")

    for thr in thresholds:
        kept_full = [t for t in full_trades if t['_atr_pct'] >= thr]
        kept_recent = [t for t in recent_trades if t['_atr_pct'] >= thr]
        lbl = f"floor_{int(thr*100)}pct"
        s_full = stats_from_json_trades(kept_full, lbl)
        s_recent = stats_from_json_trades(kept_recent, lbl)
        s_full['threshold_pct'] = round(thr * 100, 1)
        s_recent['threshold_pct'] = round(thr * 100, 1)
        s_full['blocked'] = len(full_trades) - len(kept_full)
        s_recent['blocked'] = len(recent_trades) - len(kept_recent)
        results_full.append(s_full)
        results_recent.append(s_recent)
        stamp(f"  floor={thr*100:.0f}%: full Sharpe={s_full['sharpe']} "
              f"(Δ={s_full['sharpe']-baseline_full['sharpe']:+.2f}) "
              f"n={s_full['n']} | recent Sharpe={s_recent['sharpe']} "
              f"(Δ={s_recent['sharpe']-baseline_recent['sharpe']:+.2f})")

    payload = {
        'thresholds_full': results_full,
        'thresholds_recent': results_recent,
        'baseline_full': baseline_full,
        'baseline_recent': baseline_recent,
    }
    (OUTPUT_DIR / "phase3_atr_pctl_sweep.json").write_text(
        json.dumps(payload, indent=2), encoding='utf-8')
    return payload


# ═══════════════════════════════════════════════════════════════
# Phase 4: ADX entry filter sweep (live engine)
# ═══════════════════════════════════════════════════════════════

def run_with_adx_threshold(data: DataBundle, m30_df: pd.DataFrame,
                            adx_thr: float, label: str
                            ) -> Tuple[List[TradeRecord], Dict]:
    """Replace session_adx with global keltner_adx_threshold."""
    kwargs = dict(LIVE_PARITY_KWARGS)
    kwargs.update({
        'm30_df': m30_df,
        'm30_enabled': True,
        'm30_signal_func': m30_sig_rsi14_trend,
        'm30_strategy_name': 'm30_rsi14',
        'm30_sl_atr_mult': 8.0,
        'm30_tp_atr_mult': 8.0,
        'm30_trail_activate_atr': 0.50,
        'm30_trail_distance_atr': 0.15,
        'm30_max_hold_m15': 96,
        'm30_cooldown_bars': 8,
        'keltner_session_adx': None,
        'keltner_adx_threshold': adx_thr,
        'label': label,
    })
    import indicators as signals_mod
    from indicators import get_orb_strategy
    get_orb_strategy().reset_daily()
    signals_mod._friday_close_price = None
    signals_mod._gap_traded_today = False
    engine = BacktestEngine(data.m15_df, data.h1_df, **kwargs)
    trades = engine.run()
    return trades, stats_summary(trades, label)


def phase4(full_data: DataBundle, m30_df: pd.DataFrame) -> Dict:
    stamp("PHASE 4 ── ADX entry filter sweep")
    adx_values = [0, 14, 18, 22, 26, 30]
    results_full = []
    results_recent = []

    recent_data = full_data.slice(MOST_RECENT_START, DATA_END)
    recent_m30 = m30_df[(m30_df.index >= pd.Timestamp(MOST_RECENT_START, tz='UTC')) &
                        (m30_df.index <  pd.Timestamp(DATA_END, tz='UTC'))]

    for adx in adx_values:
        lbl_full = f"adx{adx}_full"
        trades_full, st_full = run_with_adx_threshold(full_data, m30_df, adx, lbl_full)
        st_full['adx_threshold'] = adx
        results_full.append(st_full)
        stamp(f"  ADX={adx} full: n={st_full['n']} Sharpe={st_full['sharpe']} PnL=${st_full['pnl']:.0f}")

        lbl_recent = f"adx{adx}_recent"
        trades_recent, st_recent = run_with_adx_threshold(recent_data, recent_m30, adx, lbl_recent)
        st_recent['adx_threshold'] = adx
        results_recent.append(st_recent)
        stamp(f"  ADX={adx} recent: n={st_recent['n']} Sharpe={st_recent['sharpe']} PnL=${st_recent['pnl']:.0f}")

    payload = {
        'adx_sweep_full': results_full,
        'adx_sweep_recent': results_recent,
    }
    (OUTPUT_DIR / "phase4_adx_sweep.json").write_text(
        json.dumps(payload, indent=2), encoding='utf-8')
    return payload


# ═══════════════════════════════════════════════════════════════
# Summary report
# ═══════════════════════════════════════════════════════════════

def write_summary(p0: Dict, p1: Dict, p2: Dict, p3: Dict, p4: Dict):
    md = ["# R243 — Factor Re-validation Report", "",
          "_Triggered by IC report 2026-05-21 (160 live + 832 paper trades)_", "",
          f"**Data range**: 2015-01-01 to {DATA_END}", "",
          f"**Eras tested**: {', '.join(e[0] for e in ERAS)}", "",
          "---", "",
          "## Phase 0 — Baseline (LIVE_PARITY + m30_rsi14)", ""]

    b_full = p0['full_period']
    b_rec = p0['recent_7mo']
    md.append(f"| Window | Trades | Sharpe | PnL | MaxDD | WinRate |")
    md.append(f"|---|---|---|---|---|---|")
    md.append(f"| Full (2015-2026) | {b_full['n']} | {b_full['sharpe']} | "
              f"${b_full['pnl']:.0f} | ${b_full['max_dd']:.0f} | {b_full['win_rate']*100:.1f}% |")
    md.append(f"| Most-recent 7mo | {b_rec['n']} | {b_rec['sharpe']} | "
              f"${b_rec['pnl']:.0f} | ${b_rec['max_dd']:.0f} | {b_rec['win_rate']*100:.1f}% |")
    md.append("")

    md.append("### Strategy breakdown (full period)")
    md.append(f"| Strategy | Trades | Sharpe | PnL | WinRate |")
    md.append(f"|---|---|---|---|---|")
    for s, st in sorted(p0['strategy_breakdown_full'].items(),
                         key=lambda x: -x[1]['pnl']):
        md.append(f"| {s} | {st['n']} | {st['sharpe']} | ${st['pnl']:.0f} | "
                   f"{st['win_rate']*100:.1f}% |")
    md.append("")

    # Phase 1
    md.append("## Phase 1 — Rule B Parameter Sweep")
    md.append("")
    md.append("Top 5 configs (full period, by Sharpe):")
    md.append(f"| Config | Sharpe | Δ vs baseline | PnL | Blocked% | Windows |")
    md.append(f"|---|---|---|---|---|---|")
    bs_full = p1['baseline_full']['sharpe']
    for r in p1['top5_full']:
        d = r['sharpe'] - bs_full
        md.append(f"| {r['label']} | {r['sharpe']} | {d:+.2f} | ${r['pnl']:.0f} | "
                   f"{r['blocked_pct']:.1f}% | {r.get('windows', 0)} |")
    md.append("")
    md.append("Top 5 configs (most-recent 7mo, by Sharpe):")
    md.append(f"| Config | Sharpe | Δ vs baseline | PnL | Blocked% | Windows |")
    md.append(f"|---|---|---|---|---|---|")
    bs_rec = p1['baseline_recent']['sharpe']
    for r in p1['top5_recent']:
        d = r['sharpe'] - bs_rec
        md.append(f"| {r['label']} | {r['sharpe']} | {d:+.2f} | ${r['pnl']:.0f} | "
                   f"{r['blocked_pct']:.1f}% | {r.get('windows', 0)} |")
    md.append("")

    # Phase 2
    md.append("## Phase 2 — RSI Decay Validation (m30_rsi14 / m15_rsi by era)")
    md.append("")
    md.append("### m30_rsi14 isolated")
    md.append(f"| Era | Trades | Sharpe | PnL | WinRate |")
    md.append(f"|---|---|---|---|---|")
    for e, _, _ in ERAS:
        s = p2['m30_rsi14_by_era'].get(e, {})
        md.append(f"| {e} | {s.get('n', 0)} | {s.get('sharpe', 0)} | "
                   f"${s.get('pnl', 0):.0f} | {s.get('win_rate', 0)*100:.1f}% |")
    md.append(f"\n**Decay verdict**: {p2['m30_decay_verdict']}")
    md.append("")
    md.append("### m15_rsi (RSI2) isolated")
    md.append(f"| Era | Trades | Sharpe | PnL | WinRate |")
    md.append(f"|---|---|---|---|---|")
    for e, _, _ in ERAS:
        s = p2['m15_rsi_by_era'].get(e, {})
        md.append(f"| {e} | {s.get('n', 0)} | {s.get('sharpe', 0)} | "
                   f"${s.get('pnl', 0):.0f} | {s.get('win_rate', 0)*100:.1f}% |")
    md.append(f"\n**Decay verdict**: {p2['m15_decay_verdict']}")
    md.append("")

    # Phase 3
    md.append("## Phase 3 — ATR_PCTL_FLOOR Threshold Sweep")
    md.append("")
    md.append(f"| Threshold | Full Sharpe | Δ | Full PnL | Blocked | Recent Sharpe | Δ | Recent PnL |")
    md.append(f"|---|---|---|---|---|---|---|---|")
    bs_full = p3['baseline_full']['sharpe']
    bs_rec = p3['baseline_recent']['sharpe']
    full_map = {r['threshold_pct']: r for r in p3['thresholds_full']}
    rec_map = {r['threshold_pct']: r for r in p3['thresholds_recent']}
    for thr_pct in sorted(full_map.keys()):
        rf = full_map[thr_pct]
        rr = rec_map.get(thr_pct, {})
        df = rf['sharpe'] - bs_full
        dr = rr.get('sharpe', 0) - bs_rec
        md.append(f"| {thr_pct:.0f}% | {rf['sharpe']} | {df:+.2f} | ${rf['pnl']:.0f} | "
                   f"{rf['blocked']} | {rr.get('sharpe', 0)} | {dr:+.2f} | "
                   f"${rr.get('pnl', 0):.0f} |")
    md.append("")

    # Phase 4
    md.append("## Phase 4 — ADX Entry Filter Sweep")
    md.append("")
    md.append(f"| ADX threshold | Full Sharpe | Full PnL | Full N | Recent Sharpe | Recent PnL | Recent N |")
    md.append(f"|---|---|---|---|---|---|---|")
    full_map4 = {r['adx_threshold']: r for r in p4['adx_sweep_full']}
    rec_map4 = {r['adx_threshold']: r for r in p4['adx_sweep_recent']}
    for adx in sorted(full_map4.keys()):
        rf = full_map4[adx]
        rr = rec_map4.get(adx, {})
        md.append(f"| {adx} | {rf['sharpe']} | ${rf['pnl']:.0f} | {rf['n']} | "
                   f"{rr.get('sharpe', 0)} | ${rr.get('pnl', 0):.0f} | {rr.get('n', 0)} |")
    md.append("")

    # Key recommendations
    md.append("---")
    md.append("## Key Findings & Recommendations")
    md.append("")
    md.append("_Generated from Phase 1-4 analysis. Human review required before deploying._")

    (OUTPUT_DIR / "SUMMARY.md").write_text("\n".join(md), encoding='utf-8')
    stamp("SUMMARY.md written")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    stamp("=" * 78)
    stamp("R243 — Factor Re-validation (IC-triggered, 2026-05-22)")
    stamp("=" * 78)

    stamp("Loading data...")
    full_data = DataBundle.load_default(start="2015-01-01", end=DATA_END)
    m30_df = load_m30_with_indicators()
    stamp(f"  M15: {len(full_data.m15_df)} bars  "
          f"H1: {len(full_data.h1_df)} bars  M30: {len(m30_df)} bars")

    # Phase 0
    p0 = phase0(full_data, m30_df)

    # Phase 1
    p1 = phase1(p0, full_data.h1_df)

    # Phase 2
    p2 = phase2(full_data, m30_df)

    # Phase 3
    p3 = phase3(p0, full_data.h1_df)

    # Phase 4
    p4 = phase4(full_data, m30_df)

    # Summary
    write_summary(p0, p1, p2, p3, p4)
    stamp(f"DONE in {(time.time() - T0) / 60:.1f} minutes")
    stamp(f"Output: {OUTPUT_DIR.resolve()}")


if __name__ == '__main__':
    main()
