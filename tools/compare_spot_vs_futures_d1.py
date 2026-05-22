"""Spot XAUUSD vs COMEX GC=F daily comparison.

Quantifies the basis between:
  - Spot D1   : HistData M15 bid → resampled to D1 (load_d1_spot)
  - Futures D1: data/xauusd_daily_yf.csv (yfinance GC=F)

Run any time after refreshing data to confirm the basis is still in the
expected 5-10 USD range. If the gap widens significantly, it usually
indicates contango/backwardation shift worth investigating before relying
on legacy GC=F-based filters.

Usage:
    python tools/compare_spot_vs_futures_d1.py
    python tools/compare_spot_vs_futures_d1.py --since 2024-01-01
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.runner import load_d1_spot


def load_futures_d1(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df.dropna(subset=['Close']).sort_index()
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--since', default='2018-01-01',
                        help='Lower bound for the comparison window (default: 2018-01-01)')
    parser.add_argument('--futures-csv', default='data/xauusd_daily_yf.csv',
                        help='Path to GC=F daily CSV (default: data/xauusd_daily_yf.csv)')
    args = parser.parse_args()

    futures_path = Path(args.futures_csv)
    if not futures_path.exists():
        print(f"[ERROR] Futures CSV not found: {futures_path}")
        print("        Run scripts/download_silver.py to refresh.")
        return 1

    print(f"Spot D1   : HistData M15 resampled (load_d1_spot, UTC midnight anchor)")
    print(f"Futures D1: {futures_path}")
    print(f"Window    : >= {args.since}")
    print()

    spot = load_d1_spot(start=args.since, tz_naive=True)
    fut = load_futures_d1(futures_path)
    fut = fut[fut.index >= pd.Timestamp(args.since)]

    spot_idx = spot.index.normalize()
    fut_idx = fut.index.normalize()
    spot_n = spot.copy()
    fut_n = fut.copy()
    spot_n.index = spot_idx
    fut_n.index = fut_idx

    common = spot_n.index.intersection(fut_n.index)
    if len(common) < 20:
        print(f"[WARN] Only {len(common)} overlapping days — comparison may be unreliable.")

    s = spot_n.loc[common, 'Close']
    f = fut_n.loc[common, 'Close']
    basis = (f - s).rename('futures_minus_spot')

    print(f"Days compared      : {len(common):,}")
    print(f"Spot close range   : {s.min():.2f} ~ {s.max():.2f}")
    print(f"Futures close range: {f.min():.2f} ~ {f.max():.2f}")
    print()
    print("Basis (Futures - Spot, USD):")
    print(f"  Mean   : {basis.mean():+.2f}")
    print(f"  Median : {basis.median():+.2f}")
    print(f"  Std    : {basis.std():.2f}")
    print(f"  P05/P95: {basis.quantile(0.05):+.2f} / {basis.quantile(0.95):+.2f}")
    print(f"  Min/Max: {basis.min():+.2f} / {basis.max():+.2f}")
    print()

    pct = (basis / s * 100).rename('basis_pct')
    print("Basis as % of spot:")
    print(f"  Mean   : {pct.mean():+.3f}%")
    print(f"  Median : {pct.median():+.3f}%")
    print(f"  P05/P95: {pct.quantile(0.05):+.3f}% / {pct.quantile(0.95):+.3f}%")
    print()

    # EMA20 divergence — what would the D1 EMA filter actually see?
    ema_s = s.ewm(span=20).mean()
    ema_f = f.ewm(span=20).mean()
    ema_diff = (ema_f - ema_s)
    direction_disagree = ((s > ema_s) != (f > ema_f)).sum()
    print("D1 EMA20 trend filter divergence:")
    print(f"  |EMA_fut - EMA_spot| mean : {ema_diff.abs().mean():.2f}")
    print(f"  Bullish/bearish disagree  : {direction_disagree} days "
          f"({direction_disagree / max(len(common), 1) * 100:.1f}%)")
    print()

    by_year = pd.DataFrame({'basis': basis, 'spot': s})
    by_year['year'] = by_year.index.year
    grouped = by_year.groupby('year')['basis'].agg(['mean', 'std', 'min', 'max', 'count'])
    print("Basis by year:")
    print(grouped.round(2).to_string())
    print()

    recent = basis.tail(10)
    print("Most recent 10 days:")
    for ts, b in recent.items():
        print(f"  {ts.date()}  basis={b:+6.2f}  spot={s.loc[ts]:8.2f}  fut={f.loc[ts]:8.2f}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
