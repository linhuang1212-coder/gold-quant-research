#!/usr/bin/env python3
"""R232: Holdout OOS Test - Complete out-of-sample validation on last 12 months.

Split:
  - Training: 2015-01-01 to 2025-05-01 (params were selected on this period)
  - Holdout:  2025-05-01 to 2026-05-13 (completely untouched, OOS)

Tests 4 strategies × 3 trail configs = 12 combinations.
Uses FIXED engine (next-bar-Open entry, no same-bar trailing).

Added m30_rsi14 (R235 confirmed REAL_SIGNAL) + no_trail config.
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.m30_engine import M30BacktestEngine, load_m30_with_indicators
from backtest.engine import TradeRecord


HOLDOUT_START = "2025-05-01"


def calc_stats(trades):
    """Comprehensive stats with correct daily Sharpe."""
    if not trades or len(trades) < 5:
        return {'n': len(trades) if trades else 0, 'sharpe': -999, 'pnl': 0,
                'win_rate': 0, 'profit_factor': 0, 'max_dd': 0, 'avg_bars': 0,
                'avg_pnl': 0, 'n_days': 0, 'calmar': 0}

    pnls = np.array([t.pnl for t in trades])
    bars_arr = np.array([t.bars_held for t in trades])
    n = len(pnls)
    total = float(pnls.sum())
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else 99.0

    # Daily Sharpe
    daily_pnl = {}
    for t in trades:
        day = pd.Timestamp(t.exit_time).date()
        daily_pnl[day] = daily_pnl.get(day, 0.0) + t.pnl
    daily_series = np.array(list(daily_pnl.values()))
    n_days = len(daily_series)

    if n_days >= 10:
        daily_mean = float(daily_series.mean())
        daily_std = float(daily_series.std(ddof=1))
        sharpe = daily_mean / max(daily_std, 1e-9) * np.sqrt(252)
    else:
        sharpe = -999

    # Max drawdown
    cum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cum)
    dd = running_max - cum
    max_dd = float(dd.max())

    # Calmar ratio (annualized return / max DD)
    if max_dd > 0 and n_days >= 10:
        ann_return = total / n_days * 252
        calmar = ann_return / max_dd
    else:
        calmar = 0

    return {
        'n': n,
        'sharpe': round(sharpe, 3),
        'pnl': round(total, 2),
        'win_rate': round(len(wins) / n, 4),
        'profit_factor': round(pf, 3),
        'max_dd': round(max_dd, 2),
        'avg_bars': round(float(bars_arr.mean()), 1),
        'avg_pnl': round(float(pnls.mean()), 3),
        'n_days': n_days,
        'calmar': round(calmar, 2),
    }


# ═══════════════════════════════════════════════════════════════
# Signal Functions
# ═══════════════════════════════════════════════════════════════

def m30_sig_kc_breakout(df):
    if len(df) < 30:
        return None
    row = df.iloc[-1]
    c = float(row['Close'])
    kc_u = float(row.get('KC_upper', 0))
    kc_l = float(row.get('KC_lower', 0))
    if pd.isna(kc_u) or kc_u == 0 or float(row.get('ATR', 0)) <= 0:
        return None
    if c > kc_u:
        return {'strategy': 'm30_kc', 'signal': 'BUY'}
    if c < kc_l:
        return {'strategy': 'm30_kc', 'signal': 'SELL'}
    return None


def m30_sig_ema_fast_cross(df):
    if len(df) < 25:
        return None
    curr, prev = df.iloc[-1], df.iloc[-2]
    e9 = float(curr['EMA9'])
    e20 = float(curr['EMA20'])
    e9p = float(prev['EMA9'])
    e20p = float(prev['EMA20'])
    if pd.isna(e9) or pd.isna(e20) or float(curr.get('ATR', 0)) <= 0:
        return None
    if e9 > e20 and e9p <= e20p:
        return {'strategy': 'm30_ema_fast', 'signal': 'BUY'}
    if e9 < e20 and e9p >= e20p:
        return {'strategy': 'm30_ema_fast', 'signal': 'SELL'}
    return None


def m30_sig_rsi6_extreme(df):
    if len(df) < 20:
        return None
    row = df.iloc[-1]
    rsi6 = float(row.get('RSI6', 50))
    c = float(row['Close'])
    ema200 = float(row.get('EMA200', c))
    if pd.isna(rsi6) or float(row.get('ATR', 0)) <= 0 or pd.isna(ema200):
        return None
    if rsi6 < 15 and c > ema200:
        return {'strategy': 'm30_rsi6', 'signal': 'BUY'}
    if rsi6 > 85 and c < ema200:
        return {'strategy': 'm30_rsi6', 'signal': 'SELL'}
    return None


def m30_sig_rsi14_trend(df):
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


STRATEGIES = [
    ('m30_kc', m30_sig_kc_breakout),
    ('m30_ema_fast', m30_sig_ema_fast_cross),
    ('m30_rsi6', m30_sig_rsi6_extreme),
    ('m30_rsi14', m30_sig_rsi14_trend),
]

TRAIL_CONFIGS = {
    'aggressive': {'trailing_activate_atr': 0.15, 'trailing_distance_atr': 0.04},
    'moderate':   {'trailing_activate_atr': 0.50, 'trailing_distance_atr': 0.15},
    'no_trail':   {'trailing_activate_atr': 0.0,  'trailing_distance_atr': 0.0},
}

BEST_PARAMS = {
    'm30_kc':       {'sl_atr_mult': 3.0, 'tp_atr_mult': 3.0, 'max_hold': 96},
    'm30_ema_fast': {'sl_atr_mult': 2.5, 'tp_atr_mult': 3.0, 'max_hold': 24},
    'm30_rsi6':     {'sl_atr_mult': 3.0, 'tp_atr_mult': 5.0, 'max_hold': 24},
    'm30_rsi14':    {'sl_atr_mult': 8.0, 'tp_atr_mult': 8.0, 'max_hold': 48},
}


def run_holdout_test(df_full, strat_name, sig_func, trail_name, trail_params):
    """Run backtest on holdout period only, but feed full data for indicator warmup."""
    params = {**BEST_PARAMS[strat_name], **trail_params,
              'cooldown_bars': 4, 'spread_cost': 0.30, 'lot_size': 0.02}

    engine = M30BacktestEngine(df_full, signal_funcs=[(strat_name, sig_func)], **params)
    all_trades = engine.run()

    holdout_cutoff = pd.Timestamp(HOLDOUT_START, tz='UTC')
    holdout_trades = [t for t in all_trades
                      if t.strategy == strat_name and pd.Timestamp(t.entry_time) >= holdout_cutoff]
    train_trades = [t for t in all_trades
                    if t.strategy == strat_name and pd.Timestamp(t.entry_time) < holdout_cutoff]

    return train_trades, holdout_trades


def main():
    print('=' * 80)
    print('R232: HOLDOUT OUT-OF-SAMPLE TEST')
    print(f'  Holdout period: {HOLDOUT_START} -> end of data')
    print(f'  Engine: Fixed (next-bar Open entry, no same-bar trailing)')
    print('=' * 80)

    print('\nLoading full M30 data...')
    df = load_m30_with_indicators()
    print(f'  Total bars: {len(df)}')
    print(f'  Date range: {df.index[0]} -> {df.index[-1]}')

    holdout_bars = len(df[df.index >= pd.Timestamp(HOLDOUT_START, tz='UTC')])
    print(f'  Holdout bars: {holdout_bars} ({holdout_bars/48:.0f} trading days)')

    results = {}

    for strat_name, sig_func in STRATEGIES:
        for trail_name, trail_params in TRAIL_CONFIGS.items():
            key = f'{strat_name}_{trail_name}'
            print(f'\n{"─"*60}')
            print(f'  {key}')
            print(f'{"─"*60}')

            train_trades, holdout_trades = run_holdout_test(
                df, strat_name, sig_func, trail_name, trail_params)

            train_stats = calc_stats(train_trades)
            oos_stats = calc_stats(holdout_trades)

            print(f'\n  TRAIN (2015 - 2025-05):')
            print(f'    Trades: {train_stats["n"]}, Sharpe: {train_stats["sharpe"]}, '
                  f'WinRate: {train_stats["win_rate"]*100:.1f}%, PnL: ${train_stats["pnl"]}, '
                  f'PF: {train_stats["profit_factor"]}, MaxDD: ${train_stats["max_dd"]}')

            print(f'\n  HOLDOUT OOS (2025-05 - 2026-05):')
            print(f'    Trades: {oos_stats["n"]}, Sharpe: {oos_stats["sharpe"]}, '
                  f'WinRate: {oos_stats["win_rate"]*100:.1f}%, PnL: ${oos_stats["pnl"]}, '
                  f'PF: {oos_stats["profit_factor"]}, MaxDD: ${oos_stats["max_dd"]}')

            # Degradation check
            if train_stats['sharpe'] > 0 and oos_stats['sharpe'] > 0:
                degrad = oos_stats['sharpe'] / train_stats['sharpe']
                print(f'    Sharpe retention: {degrad*100:.1f}%')
                if degrad >= 0.5:
                    print(f'    [PASS] (>50% retention)')
                else:
                    print(f'    [FAIL] (<50% retention, likely overfit)')
            elif oos_stats['sharpe'] <= 0:
                print(f'    [FAIL] (negative OOS Sharpe)')

            results[key] = {
                'train': train_stats,
                'holdout': oos_stats,
            }

            # Show holdout sample trades
            if holdout_trades:
                print(f'\n  Holdout sample trades (first 5):')
                for t in holdout_trades[:5]:
                    print(f'    {t.entry_time} -> {t.exit_time} | {t.direction} | '
                          f'entry={t.entry_price:.2f} exit={t.exit_price:.2f} | '
                          f'pnl=${t.pnl:.2f} | {t.exit_reason} | bars={t.bars_held}')

    # ═══════════════════════════════════════════════════════════════
    # Portfolio analysis on holdout
    # ═══════════════════════════════════════════════════════════════
    print(f'\n\n{"="*80}')
    print('PORTFOLIO ANALYSIS (Holdout OOS, moderate trail)')
    print(f'{"="*80}')

    # Combine all moderate-trail strategies
    all_holdout_pnl = {}
    for strat_name, sig_func in STRATEGIES:
        key = f'{strat_name}_moderate'
        if key in results and results[key]['holdout']['n'] > 0:
            # We need to re-run to get actual trades for portfolio
            pass

    # Re-run all with moderate trail for portfolio
    portfolio_daily = {}
    for strat_name, sig_func in STRATEGIES:
        trail_params = TRAIL_CONFIGS['moderate']
        _, holdout_trades = run_holdout_test(df, strat_name, sig_func, 'moderate', trail_params)
        for t in holdout_trades:
            day = pd.Timestamp(t.exit_time).date()
            if day not in portfolio_daily:
                portfolio_daily[day] = 0.0
            portfolio_daily[day] += t.pnl

    if portfolio_daily:
        port_series = np.array(list(portfolio_daily.values()))
        port_total = float(port_series.sum())
        port_mean = float(port_series.mean())
        port_std = float(port_series.std(ddof=1))
        port_sharpe = port_mean / max(port_std, 1e-9) * np.sqrt(252)
        cum = np.cumsum(port_series)
        port_dd = float((np.maximum.accumulate(cum) - cum).max())
        n_days = len(port_series)
        win_days = (port_series > 0).sum()

        print(f'\n  3-strategy portfolio (moderate trail):')
        print(f'    Trading days: {n_days}')
        print(f'    Total PnL: ${port_total:.2f}')
        print(f'    Daily Sharpe: {port_sharpe:.3f}')
        print(f'    Max Drawdown: ${port_dd:.2f}')
        print(f'    Win days: {win_days}/{n_days} ({win_days/n_days*100:.1f}%)')
        print(f'    Avg daily PnL: ${port_mean:.2f}')
        print(f'    Calmar: {port_total/n_days*252/max(port_dd,1):.2f}')

    # Summary table
    print(f'\n\n{"="*80}')
    print('SUMMARY TABLE')
    print(f'{"="*80}')
    header = f"{'Strategy':<25} {'Train Sharpe':<14} {'OOS Sharpe':<12} {'Retention':<12} {'OOS PnL':<12} {'OOS WR':<8} {'Verdict'}"
    print(header)
    print('-' * 95)

    for key, data in results.items():
        train_s = data['train']['sharpe']
        oos_s = data['holdout']['sharpe']
        if train_s > 0 and oos_s > 0:
            ret = f'{oos_s/train_s*100:.0f}%'
            verdict = 'PASS' if oos_s / train_s >= 0.5 else 'FAIL'
        elif oos_s <= 0:
            ret = 'N/A'
            verdict = 'FAIL'
        else:
            ret = 'N/A'
            verdict = '?'
        oos_pnl = f"${data['holdout']['pnl']}"
        oos_wr = f"{data['holdout']['win_rate']*100:.1f}%"
        print(f'{key:<25} {train_s:<14} {oos_s:<12} {ret:<12} {oos_pnl:<12} {oos_wr:<8} {verdict}')

    # Save results
    out = Path('results/r232_holdout_oos')
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'holdout_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\n  Saved: {out / "holdout_results.json"}')


if __name__ == '__main__':
    main()
