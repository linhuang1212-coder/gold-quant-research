#!/usr/bin/env python3
"""R236: M30 RSI14 Integrated Validation via BacktestEngine
=============================================================
Validates m30_rsi14 through the FULL BacktestEngine filter stack:
  - Choppy Gate (intraday_adaptive + choppy_threshold=0.50)
  - ATR Percentile Floor (live_atr_percentile, skip <30th pctl)
  - Rule B (ATR > 2.5σ over 60 bars → skip)
  - Slot competition (max_positions=1 → M30 competes with Keltner)
  - Realistic spread ($0.30)
  - Next-bar Open entry (pending signal mechanism)

Tests:
  1. LIVE_PARITY_KWARGS + M30 RSI14 enabled → full 10-year K-Fold 6
  2. Holdout OOS (2025-05 → 2026-05)
  3. M30-only mode (disable Keltner) for isolated signal validation

Pass criteria (R209v2 methodology):
  - K-Fold: ≥4/6 positive Sharpe folds
  - OOS Sharpe retention ≥ 50%
  - OOS Sharpe > 0.5
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

OUTPUT_DIR = Path("results/r236_m30_integrated")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HOLDOUT_START = "2025-05-01"


def m30_sig_rsi14_trend(df):
    """M30 RSI14 trend signal — confirmed REAL_SIGNAL in R235."""
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


def calc_daily_sharpe(trades):
    """Correct daily Sharpe: aggregate PnL by day, annualize with sqrt(252)."""
    if not trades or len(trades) < 5:
        return -999, 0, 0, 0, 0, 0
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
        return -999, total, win_rate, pf, n, n_days

    daily_mean = float(daily_series.mean())
    daily_std = float(daily_series.std(ddof=1))
    sharpe = daily_mean / max(daily_std, 1e-9) * np.sqrt(252)

    cum = np.cumsum(pnls)
    max_dd = float((np.maximum.accumulate(cum) - cum).max())

    return round(sharpe, 3), round(total, 2), round(win_rate, 4), round(pf, 3), n, n_days


def kfold_validate(trades, k=6):
    """K-Fold cross-validation on trade sequence."""
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


def run_backtest(m15_df, h1_df, m30_df, label, m30_only=False):
    """Run BacktestEngine with M30 RSI14 integrated."""
    kwargs = {**LIVE_PARITY_KWARGS}

    # M30 integration params
    kwargs['m30_df'] = m30_df
    kwargs['m30_enabled'] = True
    kwargs['m30_signal_func'] = m30_sig_rsi14_trend
    kwargs['m30_strategy_name'] = 'm30_rsi14'
    kwargs['m30_sl_atr_mult'] = 8.0
    kwargs['m30_tp_atr_mult'] = 8.0
    kwargs['m30_trail_activate_atr'] = 0.50
    kwargs['m30_trail_distance_atr'] = 0.15
    kwargs['m30_max_hold_m15'] = 96  # 48 M30 bars = 96 M15 bars
    kwargs['m30_cooldown_bars'] = 8  # 8 M30 bars = 4 hours

    if m30_only:
        # Disable all H1/M15 strategies by setting impossible ADX thresholds
        kwargs['keltner_session_adx'] = {
            "asia": (0, 7, 999),
            "london": (8, 12, 999),
            "ny": (13, 17, 999),
            "evening": (18, 23, 999),
        }
        kwargs['rsi_adx_filter'] = 0.001  # effectively disable M15 RSI

    kwargs['label'] = label

    engine = BacktestEngine(m15_df, h1_df, **kwargs)
    trades = engine.run()

    m30_trades = [t for t in trades if t.strategy == 'm30_rsi14']
    other_trades = [t for t in trades if t.strategy != 'm30_rsi14']

    return trades, m30_trades, other_trades, engine


def main():
    t0 = time.time()
    print('=' * 80)
    print('R236: M30 RSI14 INTEGRATED VALIDATION (BacktestEngine)')
    print('  Full filter stack: Choppy + ATR Pctl + Rule B + Slot Competition')
    print('=' * 80)

    # Load data
    print('\nLoading data...')
    data = DataBundle.load_default()
    m30_df = load_m30_with_indicators()
    print(f'  M15: {len(data.m15_df)} bars')
    print(f'  H1:  {len(data.h1_df)} bars')
    print(f'  M30: {len(m30_df)} bars')

    results = {}

    # ═══════════════════════════════════════════════════════════════
    # Test 1: M30-only mode (isolate M30 signal quality under full filters)
    # ═══════════════════════════════════════════════════════════════
    print(f'\n\n{"="*80}')
    print('TEST 1: M30 RSI14 ONLY (Keltner/RSI disabled, full filter stack active)')
    print(f'{"="*80}')

    all_trades, m30_trades, _, engine = run_backtest(
        data.m15_df, data.h1_df, m30_df, "m30_only", m30_only=True)

    sharpe, pnl, wr, pf, n, n_days = calc_daily_sharpe(m30_trades)
    print(f'\n  M30 RSI14 trades: {n}')
    print(f'  Daily Sharpe: {sharpe}')
    print(f'  Total PnL: ${pnl}')
    print(f'  Win Rate: {wr*100:.1f}%')
    print(f'  Profit Factor: {pf}')
    print(f'  Trading days: {n_days}')
    print(f'\n  Engine stats:')
    print(f'    M30 signals generated: {engine.m30_total_signals}')
    print(f'    M30 entries: {engine.m30_entries}')
    print(f'    M30 skipped (choppy): {engine.m30_skipped_choppy}')
    print(f'    M30 skipped (ATR/RuleB): {engine.m30_skipped_atr}')
    print(f'    M30 skipped (slot full): {engine.m30_skipped_slot}')
    print(f'    M30 skipped (cooldown): {engine.m30_skipped_cooldown}')

    # K-Fold on full period
    kf = kfold_validate(m30_trades)
    print(f'\n  K-Fold 6: {kf["verdict"]} ({kf["pass_count"]}/{kf["total"]})')
    print(f'    Fold Sharpes: {kf["sharpes"]}')

    # Holdout split
    holdout_cutoff = pd.Timestamp(HOLDOUT_START, tz='UTC')
    train_trades = [t for t in m30_trades if pd.Timestamp(t.entry_time) < holdout_cutoff]
    oos_trades = [t for t in m30_trades if pd.Timestamp(t.entry_time) >= holdout_cutoff]

    train_sharpe, train_pnl, train_wr, _, train_n, _ = calc_daily_sharpe(train_trades)
    oos_sharpe, oos_pnl, oos_wr, _, oos_n, _ = calc_daily_sharpe(oos_trades)

    print(f'\n  TRAIN (2015 - 2025-05): n={train_n}, Sharpe={train_sharpe}, PnL=${train_pnl}, WR={train_wr*100:.1f}%')
    print(f'  OOS   (2025-05 - end):  n={oos_n}, Sharpe={oos_sharpe}, PnL=${oos_pnl}, WR={oos_wr*100:.1f}%')

    if train_sharpe > 0 and oos_sharpe > 0:
        retention = oos_sharpe / train_sharpe
        print(f'  Sharpe retention: {retention*100:.1f}%')
        oos_verdict = 'PASS' if retention >= 0.5 and oos_sharpe > 0.5 else 'FAIL'
    elif oos_sharpe <= 0:
        retention = 0
        oos_verdict = 'FAIL'
    else:
        retention = 0
        oos_verdict = '?'
    print(f'  OOS Verdict: {oos_verdict}')

    results['m30_only'] = {
        'n': n, 'sharpe': sharpe, 'pnl': pnl, 'win_rate': wr,
        'kfold': kf, 'train_sharpe': train_sharpe, 'oos_sharpe': oos_sharpe,
        'oos_pnl': oos_pnl, 'retention': round(retention, 3) if retention else 0,
        'oos_verdict': oos_verdict,
        'signals': engine.m30_total_signals,
        'entries': engine.m30_entries,
        'skipped_choppy': engine.m30_skipped_choppy,
        'skipped_atr': engine.m30_skipped_atr,
        'skipped_slot': engine.m30_skipped_slot,
    }

    # ═══════════════════════════════════════════════════════════════
    # Test 2: Combined mode (M30 + Keltner competing for same slot)
    # ═══════════════════════════════════════════════════════════════
    print(f'\n\n{"="*80}')
    print('TEST 2: COMBINED (M30 RSI14 + Keltner, slot competition, max_pos=1)')
    print(f'{"="*80}')

    all_trades, m30_trades, keltner_trades, engine = run_backtest(
        data.m15_df, data.h1_df, m30_df, "combined", m30_only=False)

    print(f'\n  Total trades: {len(all_trades)}')
    print(f'  - Keltner: {len(keltner_trades)}')
    print(f'  - M30 RSI14: {len(m30_trades)}')

    sharpe_all, pnl_all, wr_all, pf_all, n_all, _ = calc_daily_sharpe(all_trades)
    sharpe_m30, pnl_m30, wr_m30, pf_m30, n_m30, _ = calc_daily_sharpe(m30_trades)

    print(f'\n  Combined portfolio:')
    print(f'    Sharpe: {sharpe_all}, PnL: ${pnl_all}, WR: {wr_all*100:.1f}%, n={n_all}')
    print(f'\n  M30 RSI14 subset:')
    print(f'    Sharpe: {sharpe_m30}, PnL: ${pnl_m30}, WR: {wr_m30*100:.1f}%, n={n_m30}')
    print(f'\n  M30 slot stats:')
    print(f'    M30 skipped (slot full): {engine.m30_skipped_slot}')

    results['combined'] = {
        'total_n': n_all, 'total_sharpe': sharpe_all, 'total_pnl': pnl_all,
        'm30_n': n_m30, 'm30_sharpe': sharpe_m30, 'm30_pnl': pnl_m30,
        'keltner_n': len(keltner_trades),
        'm30_skipped_slot': engine.m30_skipped_slot,
    }

    # ═══════════════════════════════════════════════════════════════
    # Test 3: Combined mode with max_positions=2 (no slot competition)
    # ═══════════════════════════════════════════════════════════════
    print(f'\n\n{"="*80}')
    print('TEST 3: COMBINED (M30 + Keltner, max_pos=2, no slot competition)')
    print(f'{"="*80}')

    kwargs_mp2 = {**LIVE_PARITY_KWARGS}
    kwargs_mp2['m30_df'] = m30_df
    kwargs_mp2['m30_enabled'] = True
    kwargs_mp2['m30_signal_func'] = m30_sig_rsi14_trend
    kwargs_mp2['m30_strategy_name'] = 'm30_rsi14'
    kwargs_mp2['m30_sl_atr_mult'] = 8.0
    kwargs_mp2['m30_tp_atr_mult'] = 8.0
    kwargs_mp2['m30_trail_activate_atr'] = 0.50
    kwargs_mp2['m30_trail_distance_atr'] = 0.15
    kwargs_mp2['m30_max_hold_m15'] = 96
    kwargs_mp2['m30_cooldown_bars'] = 8
    kwargs_mp2['max_positions'] = 2
    kwargs_mp2['label'] = "combined_mp2"

    engine2 = BacktestEngine(data.m15_df, data.h1_df, **kwargs_mp2)
    all_trades2 = engine2.run()
    m30_trades2 = [t for t in all_trades2 if t.strategy == 'm30_rsi14']
    keltner_trades2 = [t for t in all_trades2 if t.strategy != 'm30_rsi14']

    sharpe_all2, pnl_all2, wr_all2, _, n_all2, _ = calc_daily_sharpe(all_trades2)
    sharpe_m302, pnl_m302, wr_m302, _, n_m302, _ = calc_daily_sharpe(m30_trades2)

    print(f'\n  Total trades: {n_all2}')
    print(f'  - Keltner: {len(keltner_trades2)}')
    print(f'  - M30 RSI14: {n_m302}')
    print(f'\n  Combined portfolio (max_pos=2):')
    print(f'    Sharpe: {sharpe_all2}, PnL: ${pnl_all2}, WR: {wr_all2*100:.1f}%')
    print(f'\n  M30 RSI14 subset:')
    print(f'    Sharpe: {sharpe_m302}, PnL: ${pnl_m302}, WR: {wr_m302*100:.1f}%')

    results['combined_mp2'] = {
        'total_n': n_all2, 'total_sharpe': sharpe_all2, 'total_pnl': pnl_all2,
        'm30_n': n_m302, 'm30_sharpe': sharpe_m302, 'm30_pnl': pnl_m302,
        'keltner_n': len(keltner_trades2),
    }

    # ═══════════════════════════════════════════════════════════════
    # Final Summary
    # ═══════════════════════════════════════════════════════════════
    print(f'\n\n{"="*80}')
    print('FINAL SUMMARY')
    print(f'{"="*80}')
    print(f'\n  {"Test":<30} {"Sharpe":<10} {"PnL":<12} {"N trades":<10} {"Verdict"}')
    print(f'  {"-"*75}')

    # M30 only verdict
    m30_final = 'PASS' if (kf['verdict'] == 'PASS' and oos_verdict == 'PASS') else 'FAIL'
    print(f'  {"M30-only (full filters)":<30} {sharpe:<10} ${pnl:<11} {n:<10} {m30_final}')
    print(f'  {"Combined (slot=1)":<30} {sharpe_all:<10} ${pnl_all:<11} {n_all:<10} -')
    print(f'  {"Combined (slot=2)":<30} {sharpe_all2:<10} ${pnl_all2:<11} {n_all2:<10} -')

    print(f'\n  M30 RSI14 Integrated Verdict: {"[PASS] READY FOR PAPER TRADE" if m30_final == "PASS" else "[FAIL] NOT READY"}')
    print(f'    K-Fold: {kf["verdict"]} ({kf["pass_count"]}/6)')
    print(f'    OOS Sharpe: {oos_sharpe} (retention: {retention*100:.1f}%)')
    print(f'    Filter impact: {engine.m30_total_signals} signals → {engine.m30_entries} entries '
          f'({engine.m30_entries/max(engine.m30_total_signals,1)*100:.1f}% pass rate)')

    results['final_verdict'] = m30_final

    # Save
    with open(OUTPUT_DIR / 'r236_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\n  Saved: {OUTPUT_DIR / "r236_results.json"}')

    elapsed = time.time() - t0
    print(f'\n  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)')
    print('R236 complete.')


if __name__ == '__main__':
    main()
