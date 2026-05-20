#!/usr/bin/env python3
"""R236b: M30 RSI14 Parameter Ablation — Live vs Proposed
==========================================================
Ablation comparison between current live params and R236 proposed params,
using BacktestEngine with full filter stack (LIVE_PARITY_KWARGS).

Compares:
  A. Live current:   Trail 0.30/0.08, MaxHold 24 M30 bars (48 M15)
  B. R236 moderate:  Trail 0.50/0.15, MaxHold 48 M30 bars (96 M15)
  C. No trail:       Trail 0/0, MaxHold 48 M30 bars (96 M15)
  D. Tight trail:    Trail 0.15/0.04, MaxHold 24 M30 bars (48 M15)
  E. Live trail + longer hold: Trail 0.30/0.08, MaxHold 48 (96 M15)
  F. Moderate trail + shorter hold: Trail 0.50/0.15, MaxHold 24 (48 M15)

Each config runs through full BacktestEngine (Choppy + ATR Pctl + Rule B).
K-Fold 6 + Holdout OOS on each variant.

Per constraints.md parameter-change-protocol:
  "改 SL/TP/MH/Trail → 重测: 入场过滤器, Cap, K-Fold"
"""
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.runner import DataBundle, LIVE_PARITY_KWARGS
from backtest.engine import BacktestEngine, TradeRecord
from backtest.m30_engine import load_m30_with_indicators

OUTPUT_DIR = Path("results/r236b_ablation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HOLDOUT_START = "2025-05-01"


def m30_sig_rsi14_trend(df):
    """M30 RSI14 trend signal."""
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


CONFIGS = {
    'A_live_current': {
        'trail_activate': 0.30, 'trail_distance': 0.08,
        'max_hold_m15': 48, 'label': 'Live (0.30/0.08, MH24)',
    },
    'B_moderate': {
        'trail_activate': 0.50, 'trail_distance': 0.15,
        'max_hold_m15': 96, 'label': 'Moderate (0.50/0.15, MH48)',
    },
    'C_no_trail': {
        'trail_activate': 0.0, 'trail_distance': 0.0,
        'max_hold_m15': 96, 'label': 'No Trail (0/0, MH48)',
    },
    'D_tight_trail': {
        'trail_activate': 0.15, 'trail_distance': 0.04,
        'max_hold_m15': 48, 'label': 'Tight (0.15/0.04, MH24)',
    },
    'E_live_trail_long_hold': {
        'trail_activate': 0.30, 'trail_distance': 0.08,
        'max_hold_m15': 96, 'label': 'Live trail + Long hold (0.30/0.08, MH48)',
    },
    'F_moderate_trail_short_hold': {
        'trail_activate': 0.50, 'trail_distance': 0.15,
        'max_hold_m15': 48, 'label': 'Moderate + Short hold (0.50/0.15, MH24)',
    },
}


def calc_daily_sharpe(trades):
    if not trades or len(trades) < 5:
        return {'sharpe': -999, 'pnl': 0, 'win_rate': 0, 'pf': 0, 'n': len(trades) if trades else 0,
                'n_days': 0, 'max_dd': 0, 'avg_pnl': 0, 'calmar': 0}
    pnls = np.array([t.pnl for t in trades])
    n = len(pnls)
    total = float(pnls.sum())
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    win_rate = len(wins) / n
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else 99.0

    daily_pnl = {}
    for t in trades:
        day = pd.Timestamp(t.exit_time).date()
        daily_pnl[day] = daily_pnl.get(day, 0.0) + t.pnl
    daily_series = np.array(list(daily_pnl.values()))
    n_days = len(daily_series)
    if n_days < 10:
        return {'sharpe': -999, 'pnl': round(total, 2), 'win_rate': round(win_rate, 4),
                'pf': round(pf, 3), 'n': n, 'n_days': n_days, 'max_dd': 0, 'avg_pnl': 0, 'calmar': 0}

    daily_mean = float(daily_series.mean())
    daily_std = float(daily_series.std(ddof=1))
    sharpe = daily_mean / max(daily_std, 1e-9) * np.sqrt(252)

    cum = np.cumsum(pnls)
    max_dd = float((np.maximum.accumulate(cum) - cum).max())
    calmar = (total / n_days * 252) / max(max_dd, 1) if max_dd > 0 else 0

    return {
        'sharpe': round(sharpe, 3), 'pnl': round(total, 2),
        'win_rate': round(win_rate, 4), 'pf': round(pf, 3),
        'n': n, 'n_days': n_days, 'max_dd': round(max_dd, 2),
        'avg_pnl': round(float(pnls.mean()), 3),
        'calmar': round(calmar, 2),
    }


def kfold_validate(trades, k=6):
    if len(trades) < k * 5:
        return {'verdict': 'SKIP', 'pass_count': 0, 'total': 0, 'sharpes': []}
    pnls = np.array([t.pnl for t in trades])
    fold_size = len(pnls) // k
    kf_pass = 0
    sharpes = []
    for fold in range(k):
        s = fold * fold_size
        e = s + fold_size if fold < k - 1 else len(pnls)
        fp = pnls[s:e]
        if len(fp) < 3:
            continue
        sh = float(fp.mean() / max(fp.std(ddof=1), 1e-9) * np.sqrt(252))
        sharpes.append(round(sh, 3))
        if sh > 0:
            kf_pass += 1
    verdict = 'PASS' if kf_pass >= 4 else 'FAIL'
    return {'verdict': verdict, 'pass_count': kf_pass, 'total': len(sharpes), 'sharpes': sharpes}


def run_config(m15_df, h1_df, m30_df, config_name, config):
    """Run one ablation config."""
    kwargs = {**LIVE_PARITY_KWARGS}
    kwargs['m30_df'] = m30_df
    kwargs['m30_enabled'] = True
    kwargs['m30_signal_func'] = m30_sig_rsi14_trend
    kwargs['m30_strategy_name'] = 'm30_rsi14'
    kwargs['m30_sl_atr_mult'] = 8.0
    kwargs['m30_tp_atr_mult'] = 8.0
    kwargs['m30_trail_activate_atr'] = config['trail_activate']
    kwargs['m30_trail_distance_atr'] = config['trail_distance']
    kwargs['m30_max_hold_m15'] = config['max_hold_m15']
    kwargs['m30_cooldown_bars'] = 8
    # M30-only: disable Keltner/RSI to isolate M30 signal
    kwargs['keltner_session_adx'] = {
        "asia": (0, 7, 999), "london": (8, 12, 999),
        "ny": (13, 17, 999), "evening": (18, 23, 999),
    }
    kwargs['rsi_adx_filter'] = 0.001
    kwargs['label'] = config_name

    engine = BacktestEngine(m15_df, h1_df, **kwargs)
    all_trades = engine.run()
    m30_trades = [t for t in all_trades if t.strategy == 'm30_rsi14']
    return m30_trades, engine


def main():
    t0 = time.time()
    print('=' * 80)
    print('R236b: M30 RSI14 PARAMETER ABLATION')
    print('  Live (0.30/0.08) vs Moderate (0.50/0.15) vs No-Trail vs Tight')
    print('  All through BacktestEngine full filter stack')
    print('=' * 80)

    print('\nLoading data...')
    data = DataBundle.load_default()
    m30_df = load_m30_with_indicators()

    results = {}
    holdout_cutoff = pd.Timestamp(HOLDOUT_START, tz='UTC')

    for config_name, config in CONFIGS.items():
        print(f'\n{"="*60}')
        print(f'  {config_name}: {config["label"]}')
        print(f'{"="*60}')

        trades, engine = run_config(data.m15_df, data.h1_df, m30_df, config_name, config)

        # Full period stats
        full_stats = calc_daily_sharpe(trades)
        kf = kfold_validate(trades)

        # Train/OOS split
        train_trades = [t for t in trades if pd.Timestamp(t.entry_time) < holdout_cutoff]
        oos_trades = [t for t in trades if pd.Timestamp(t.entry_time) >= holdout_cutoff]
        train_stats = calc_daily_sharpe(train_trades)
        oos_stats = calc_daily_sharpe(oos_trades)

        if train_stats['sharpe'] > 0 and oos_stats['sharpe'] > 0:
            retention = oos_stats['sharpe'] / train_stats['sharpe']
        else:
            retention = 0

        print(f'\n  Full period: n={full_stats["n"]}, Sharpe={full_stats["sharpe"]}, '
              f'PnL=${full_stats["pnl"]}, WR={full_stats["win_rate"]*100:.1f}%, '
              f'PF={full_stats["pf"]}, MaxDD=${full_stats["max_dd"]}')
        print(f'  K-Fold 6: {kf["verdict"]} ({kf["pass_count"]}/6) {kf["sharpes"]}')
        print(f'  Train: n={train_stats["n"]}, Sharpe={train_stats["sharpe"]}, PnL=${train_stats["pnl"]}')
        print(f'  OOS:   n={oos_stats["n"]}, Sharpe={oos_stats["sharpe"]}, PnL=${oos_stats["pnl"]}')
        print(f'  Retention: {retention*100:.1f}%')
        print(f'  Signals={engine.m30_total_signals}, Entries={engine.m30_entries}, '
              f'Choppy_skip={engine.m30_skipped_choppy}, ATR_skip={engine.m30_skipped_atr}')

        results[config_name] = {
            'config': config,
            'full': full_stats,
            'kfold': kf,
            'train': train_stats,
            'oos': oos_stats,
            'retention': round(retention, 3),
            'signals': engine.m30_total_signals,
            'entries': engine.m30_entries,
        }

    # ═══════════════════════════════════════════════════════════════
    # Summary comparison table
    # ═══════════════════════════════════════════════════════════════
    print(f'\n\n{"="*80}')
    print('ABLATION SUMMARY')
    print(f'{"="*80}')

    header = (f'{"Config":<35} {"Sharpe":<8} {"PnL":<10} {"N":<6} {"WR":<7} '
              f'{"KF":<8} {"OOS Sh":<8} {"Ret%":<7} {"MaxDD":<8}')
    print(header)
    print('-' * len(header))

    baseline_sharpe = results['A_live_current']['full']['sharpe']

    for name, r in results.items():
        f = r['full']
        delta = f['sharpe'] - baseline_sharpe if baseline_sharpe > -900 else 0
        delta_str = f'({"+" if delta >= 0 else ""}{delta:.1f})'
        print(f'{r["config"]["label"]:<35} {f["sharpe"]:<8} ${f["pnl"]:<9} {f["n"]:<6} '
              f'{f["win_rate"]*100:.1f}%  {r["kfold"]["verdict"]:<8} '
              f'{r["oos"]["sharpe"]:<8} {r["retention"]*100:.0f}%    ${f["max_dd"]:<7}')

    # Recommendation
    print(f'\n{"="*80}')
    print('RECOMMENDATION')
    print(f'{"="*80}')

    best_name = max(results.keys(),
                    key=lambda k: results[k]['full']['sharpe'] if results[k]['kfold']['verdict'] == 'PASS' else -999)
    best = results[best_name]
    live = results['A_live_current']

    print(f'\n  Current live:  Sharpe={live["full"]["sharpe"]}, PnL=${live["full"]["pnl"]}, '
          f'KF={live["kfold"]["verdict"]}, OOS={live["oos"]["sharpe"]}')
    print(f'  Best variant:  {best["config"]["label"]}')
    print(f'                 Sharpe={best["full"]["sharpe"]}, PnL=${best["full"]["pnl"]}, '
          f'KF={best["kfold"]["verdict"]}, OOS={best["oos"]["sharpe"]}')

    if best['full']['sharpe'] > live['full']['sharpe'] * 1.1:
        print(f'\n  >>> RECOMMEND CHANGE to {best_name}')
        print(f'      Sharpe improvement: +{best["full"]["sharpe"] - live["full"]["sharpe"]:.2f} '
              f'({(best["full"]["sharpe"]/max(live["full"]["sharpe"],0.01)-1)*100:.1f}%)')
    else:
        print(f'\n  >>> KEEP CURRENT — improvement < 10%, not worth the risk')

    # Save
    with open(OUTPUT_DIR / 'ablation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\n  Saved: {OUTPUT_DIR / "ablation_results.json"}')

    elapsed = time.time() - t0
    print(f'\n  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)')


if __name__ == '__main__':
    main()
