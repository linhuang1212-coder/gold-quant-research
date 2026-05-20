#!/usr/bin/env python3
"""R231: Engine Validation - Compare old (same-bar Close) vs new (next-bar Open) entry.

Tests 3 representative strategies:
  1. m30_kc (Keltner breakout - momentum)
  2. m30_ema_fast (EMA cross - trend)
  3. m30_rsi6 (RSI extreme - mean reversion)

Uses CORRECT Sharpe calculation: daily PnL aggregation, sqrt(252).
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.m30_engine import M30BacktestEngine, load_m30_with_indicators, prepare_m30_indicators
from backtest.engine import TradeRecord


def _load_m30_from_m15_resample() -> pd.DataFrame:
    """Fallback: create M30 from M15 data by resampling."""
    m15_candidates = [
        Path("data/download/xauusd-m15-bid-2015-01-01-2026-04-10.csv"),
    ]
    csv_path = next((p for p in m15_candidates if p.exists()), None)
    if csv_path is None:
        raise FileNotFoundError("Neither M30 nor M15 data found locally.")

    print(f'  Resampling M30 from M15: {csv_path.name}')
    df = pd.read_csv(csv_path)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('datetime', inplace=True)
    df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low',
                       'close': 'Close', 'volume': 'Volume'}, inplace=True)

    m30 = df.resample('30min').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()
    print(f'  Resampled: {len(m30)} M30 bars')
    return m30


def load_data() -> pd.DataFrame:
    """Try loading M30 directly, fallback to M15 resample."""
    try:
        return load_m30_with_indicators()
    except FileNotFoundError:
        print("  M30 CSV not found, falling back to M15 resample...")
        raw = _load_m30_from_m15_resample()
        return prepare_m30_indicators(raw)


def calc_sharpe_correct(trades):
    """Correct Sharpe: aggregate to daily PnL, then annualize with sqrt(252)."""
    if not trades or len(trades) < 10:
        return {'n': len(trades), 'sharpe_daily': -999, 'sharpe_old': -999,
                'pnl': 0, 'win_rate': 0, 'avg_pnl': 0, 'profit_factor': 0}

    pnls = np.array([t.pnl for t in trades])
    n = len(pnls)
    total = float(pnls.sum())
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else 99.0

    # Old (wrong) Sharpe: per-trade, sqrt(252*2)
    mean_t = float(pnls.mean())
    std_t = float(pnls.std(ddof=1)) if n > 1 else 1e-9
    sharpe_old = mean_t / max(std_t, 1e-9) * np.sqrt(252 * 2)

    # Correct Sharpe: daily PnL aggregation
    daily_pnl = {}
    for t in trades:
        day = pd.Timestamp(t.exit_time).date()
        daily_pnl[day] = daily_pnl.get(day, 0.0) + t.pnl

    daily_series = np.array(list(daily_pnl.values()))
    n_days = len(daily_series)
    if n_days < 10:
        sharpe_daily = -999
    else:
        daily_mean = float(daily_series.mean())
        daily_std = float(daily_series.std(ddof=1))
        sharpe_daily = daily_mean / max(daily_std, 1e-9) * np.sqrt(252)

    return {
        'n': n,
        'sharpe_daily': round(sharpe_daily, 3),
        'sharpe_old': round(sharpe_old, 3),
        'pnl': round(total, 2),
        'win_rate': round(len(wins) / n, 4),
        'avg_pnl': round(mean_t, 3),
        'profit_factor': round(pf, 3),
        'n_trading_days': n_days,
        'max_dd': round(float((np.maximum.accumulate(np.cumsum(pnls)) - np.cumsum(pnls)).max()), 2),
    }


# Signal functions (same as R230)
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


STRATEGIES = [
    ('m30_kc', m30_sig_kc_breakout),
    ('m30_ema_fast', m30_sig_ema_fast_cross),
    ('m30_rsi6', m30_sig_rsi6_extreme),
]

# Best params from R230
BEST_PARAMS = {
    'm30_kc': {'sl_atr_mult': 3.0, 'tp_atr_mult': 3.0, 'trailing_activate_atr': 0.15,
               'trailing_distance_atr': 0.04, 'max_hold': 96},
    'm30_ema_fast': {'sl_atr_mult': 2.5, 'tp_atr_mult': 3.0, 'trailing_activate_atr': 0.15,
                     'trailing_distance_atr': 0.04, 'max_hold': 24},
    'm30_rsi6': {'sl_atr_mult': 3.0, 'tp_atr_mult': 5.0, 'trailing_activate_atr': 0.15,
                 'trailing_distance_atr': 0.04, 'max_hold': 24},
}


def main():
    print('=' * 80)
    print('R231: Engine Validation - next-bar-Open entry vs old same-bar-Close')
    print('=' * 80)

    print('\nLoading M30 data...')
    df = load_data()
    print(f'  Loaded {len(df)} bars')

    results = {}

    for strat_name, sig_func in STRATEGIES:
        print(f'\n{"─"*60}')
        print(f'  Strategy: {strat_name}')
        print(f'{"─"*60}')

        params = BEST_PARAMS[strat_name]
        common = {
            'cooldown_bars': 4,
            'spread_cost': 0.30,
            'lot_size': 0.02,
        }
        common.update(params)

        # Run with new engine (next-bar Open entry)
        engine = M30BacktestEngine(df, signal_funcs=[(strat_name, sig_func)], **common)
        trades = engine.run()
        strat_trades = [t for t in trades if t.strategy == strat_name]
        stats_new = calc_sharpe_correct(strat_trades)

        print(f'\n  [NEW] Next-bar Open entry:')
        print(f'    Trades: {stats_new["n"]}')
        print(f'    Sharpe (daily, correct): {stats_new["sharpe_daily"]}')
        print(f'    Sharpe (old formula):    {stats_new["sharpe_old"]}')
        print(f'    PnL: ${stats_new["pnl"]}')
        print(f'    Win Rate: {stats_new["win_rate"]*100:.1f}%')
        print(f'    Profit Factor: {stats_new["profit_factor"]}')
        print(f'    Max DD: ${stats_new["max_dd"]}')
        print(f'    Trading Days: {stats_new["n_trading_days"]}')

        # Show some sample trades
        if strat_trades:
            print(f'\n  Sample trades (first 5):')
            for t in strat_trades[:5]:
                print(f'    {t.entry_time} -> {t.exit_time} | {t.direction} | '
                      f'entry={t.entry_price:.2f} exit={t.exit_price:.2f} | '
                      f'pnl=${t.pnl:.2f} | {t.exit_reason} | bars={t.bars_held}')

        results[strat_name] = {
            'new_engine': stats_new,
        }

    # Summary comparison
    print(f'\n\n{"="*80}')
    print('SUMMARY: Sharpe Comparison')
    print(f'{"="*80}')
    print(f'{"Strategy":<15} {"Sharpe(daily)":<15} {"Sharpe(old)":<15} {"Ratio":<10} {"PnL":<12} {"WinRate":<10}')
    print(f'{"-"*77}')

    for strat_name in ['m30_kc', 'm30_ema_fast', 'm30_rsi6']:
        s = results[strat_name]['new_engine']
        ratio = s['sharpe_old'] / max(s['sharpe_daily'], 0.01) if s['sharpe_daily'] > 0 else 'N/A'
        ratio_str = f'{ratio:.2f}x' if isinstance(ratio, float) else ratio
        print(f'{strat_name:<15} {s["sharpe_daily"]:<15} {s["sharpe_old"]:<15} {ratio_str:<10} ${s["pnl"]:<10} {s["win_rate"]*100:.1f}%')

    print(f'\n  Note: R230 reported Sharpe used the OLD formula.')
    print(f'  The "Ratio" column shows how much the old formula inflates Sharpe.')

    # Save results
    out = Path('results/r231_engine_validation')
    out.mkdir(parents=True, exist_ok=True)
    with open(out / 'validation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f'\n  Saved: {out / "validation_results.json"}')


if __name__ == '__main__':
    main()
