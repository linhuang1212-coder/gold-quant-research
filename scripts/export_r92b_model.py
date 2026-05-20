#!/usr/bin/env python3
"""
Export R92-B ML Exit Model with Wilder ATR (12 features)
========================================================
Re-trains XGBoost on ALL historical L8_MAX trades using Wilder ATR
(replacing simple range ATR) for the 5 ATR-dependent features.

This should be run AFTER Fix #3 (Wilder ATR) is applied to
gold-quant-trading/strategies/signals.py, so the deployed model
matches the live feature computation.

Run from gold-quant-research root:
    python scripts/export_r92b_model.py

Outputs:
    gold-quant-trading/data/l8_ml_exit_model.json       (model)
    gold-quant-trading/data/l8_ml_exit_model.meta.json  (metadata)
"""
import sys, os, io, json, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from copy import deepcopy

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

import xgboost as xgb
from sklearn.metrics import roc_auc_score

OUTPUT_PATH = Path(r"C:\Users\hlin2\gold-quant-trading\data\l8_ml_exit_model.json")
BACKUP_DIR = Path(r"C:\Users\hlin2\gold-quant-trading\data\model_backups")
TRAIN_CUTOFF = '2026-05-01'

FEATURE_COLS = [
    'atr_14', 'adx_14', 'rsi_14', 'rsi_2',
    'kc_breakout_strength', 'volume_ratio', 'atr_percentile',
    'ema9_ema21_cross', 'close_ema100_dist',
    'hour_of_day', 'day_of_week', 'direction',
]

XGB_PARAMS = {
    'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.05,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'random_state': 42, 'eval_metric': 'logloss', 'verbosity': 1,
}

HOLDOUT_SPLIT = '2023-01-01'


def calc_atr_wilder(df, period=14):
    """Wilder True Range + EMA(alpha=1/period)"""
    h, l, pc = df['High'], df['Low'], df['Close'].shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def compute_h1_features(h1_df):
    """Compute R92-B H1 indicators using Wilder ATR."""
    df = h1_df.copy()
    df['ATR_W'] = calc_atr_wilder(df, 14)

    ema20 = df['Close'].ewm(span=20, adjust=False).mean()
    kc_width = 2 * 1.5 * df['ATR_W']
    df['KC_breakout_strength'] = (df['Close'] - ema20) / kc_width.replace(0, np.nan)

    range_vol = df['High'] - df['Low']
    df['Volume_ratio'] = range_vol / range_vol.rolling(20).mean().replace(0, np.nan)

    df['ATR_percentile'] = df['ATR_W'].rolling(252, min_periods=50).rank(pct=True)
    df['ATR_percentile'] = df['ATR_percentile'].fillna(0.5)

    ema9 = df['Close'].ewm(span=9, adjust=False).mean()
    ema21 = df['Close'].ewm(span=21, adjust=False).mean()
    df['EMA9_EMA21_cross'] = (ema9 - ema21) / df['ATR_W'].replace(0, np.nan)

    ema100 = df['Close'].ewm(span=100, adjust=False).mean()
    df['Close_EMA100_dist'] = (df['Close'] - ema100) / df['ATR_W'].replace(0, np.nan)

    return df


def build_feature_matrix(trades, h1_ind, cutoff_ts):
    """Build labeled feature matrix from trades."""
    h1_idx = h1_ind.index
    if h1_idx.tz is not None:
        h1_ind = h1_ind.copy()
        h1_ind.index = h1_idx.tz_localize(None)

    samples = []
    for trade in trades:
        entry_time = trade.entry_time if hasattr(trade, 'entry_time') else trade.get('entry_time')
        pnl = trade.pnl if hasattr(trade, 'pnl') else trade.get('pnl', 0)
        direction = trade.direction if hasattr(trade, 'direction') else trade.get('direction', '')
        if not direction:
            direction = trade.dir if hasattr(trade, 'dir') else trade.get('dir', 'BUY')

        if entry_time is None:
            continue
        if isinstance(entry_time, str):
            entry_time = pd.Timestamp(entry_time)
        if entry_time.tzinfo is None:
            entry_time = entry_time.tz_localize('UTC')
        if entry_time >= cutoff_ts:
            continue

        entry_naive = entry_time.tz_localize(None)
        h1_time = entry_naive.floor('h')

        if h1_time in h1_ind.index:
            row = h1_ind.loc[h1_time]
        else:
            loc = h1_ind.index.get_indexer([h1_time], method='ffill')
            if loc[0] < 0:
                continue
            row = h1_ind.iloc[loc[0]]

        sample = {
            'atr_14': float(row.get('ATR_W', np.nan)),
            'adx_14': float(row.get('ADX', np.nan)),
            'rsi_14': float(row.get('RSI14', np.nan)),
            'rsi_2': float(row.get('RSI2', np.nan)),
            'kc_breakout_strength': float(row.get('KC_breakout_strength', np.nan)),
            'volume_ratio': float(row.get('Volume_ratio', np.nan)),
            'atr_percentile': float(row.get('ATR_percentile', np.nan)),
            'ema9_ema21_cross': float(row.get('EMA9_EMA21_cross', np.nan)),
            'close_ema100_dist': float(row.get('Close_EMA100_dist', np.nan)),
            'hour_of_day': h1_time.hour,
            'day_of_week': h1_time.weekday(),
            'direction': 1 if direction == 'BUY' else -1,
            'label': 1 if pnl > 0 else 0,
            'pnl': pnl,
            'entry_time': str(entry_time),
        }
        samples.append(sample)

    return pd.DataFrame(samples)


def main():
    t0 = time.time()
    print("=" * 70)
    print("  Export R92-B ML Exit Model (Wilder ATR, 12 features)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Train cutoff: {TRAIN_CUTOFF}")
    print("=" * 70)

    from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS

    # ── 加载数据 ──
    print("\n  Loading DataBundle...", flush=True)
    data = DataBundle.load_default()
    h1_df = data.h1_df
    print(f"  H1: {len(h1_df)} bars ({h1_df.index[0]} -> {h1_df.index[-1]})")

    # ── 计算 Wilder ATR 特征 ──
    print("\n  Computing Wilder ATR features...", flush=True)
    h1_ind = compute_h1_features(h1_df)

    # ── 跑 L8_MAX 回测 ──
    print("\n  Running L8_MAX backtest...", flush=True)
    kw = {**LIVE_PARITY_KWARGS, 'maxloss_cap': 35,
          'min_lot_size': 0.02, 'max_lot_size': 0.02}
    result = run_variant(data, "R92B_EXPORT", verbose=False, **kw)
    trades = result.get('_trades', [])
    print(f"  {len(trades)} trades, Sharpe={result.get('sharpe', 0):.2f}, "
          f"PnL=${result.get('total_pnl', 0):.0f}")

    # ── 构建特征矩阵 ──
    print("\n  Building feature matrix...", flush=True)
    cutoff_ts = pd.Timestamp(TRAIN_CUTOFF, tz='UTC')
    feat_df = build_feature_matrix(trades, h1_ind, cutoff_ts)
    print(f"  Training samples: {len(feat_df)} (cutoff < {TRAIN_CUTOFF})")
    print(f"  Win rate: {feat_df['label'].mean()*100:.1f}%")

    X = feat_df[FEATURE_COLS].copy()
    y = feat_df['label'].values

    med = X.median()
    X_filled = X.fillna(med)
    print(f"  NaN rows filled: {X.isna().any(axis=1).sum()}")

    # ── Holdout 验证 ──
    print("\n  Holdout validation...", flush=True)
    entry_times = pd.to_datetime(feat_df['entry_time'])
    if entry_times.dt.tz is not None:
        entry_times = entry_times.dt.tz_localize(None)
    split_ts = pd.Timestamp(HOLDOUT_SPLIT)
    train_mask = entry_times < split_ts
    test_mask = entry_times >= split_ts

    if train_mask.sum() >= 50 and test_mask.sum() >= 20:
        Xtr = X_filled[train_mask]
        ytr = y[train_mask]
        Xte = X_filled[test_mask]
        yte = y[test_mask]
        val_model = xgb.XGBClassifier(**{**XGB_PARAMS, 'verbosity': 0})
        val_model.fit(Xtr.values, ytr)
        val_probs = val_model.predict_proba(Xte.values)[:, 1]
        val_auc = roc_auc_score(yte, val_probs) if len(np.unique(yte)) > 1 else 0
        val_acc = float((val_model.predict(Xte.values) == yte).mean())
        print(f"  Holdout AUC: {val_auc:.4f}  Accuracy: {val_acc:.4f}")
        print(f"  Train/Test: {train_mask.sum()}/{test_mask.sum()}")
    else:
        val_auc = 0
        print(f"  Insufficient data for holdout validation")

    # ── 全量训练 ──
    print("\n  Training full model...", flush=True)
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_filled.values, y)

    train_proba = model.predict_proba(X_filled.values)[:, 1]
    train_auc = roc_auc_score(y, train_proba)
    train_acc = float((model.predict(X_filled.values) == y).mean())

    # ── 备份旧模型 ──
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        backup_name = f"l8_ml_exit_model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        backup_path = BACKUP_DIR / backup_name
        import shutil
        shutil.copy2(OUTPUT_PATH, backup_path)
        print(f"\n  旧模型已备份: {backup_path}")

    # ── 保存新模型 ──
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(OUTPUT_PATH))

    meta = {
        'model': 'R92B_ML_Exit_WilderATR',
        'features': FEATURE_COLS,
        'threshold': 0.65,
        'xgb_params': XGB_PARAMS,
        'train_cutoff': TRAIN_CUTOFF,
        'holdout_split': HOLDOUT_SPLIT,
        'n_samples': len(feat_df),
        'win_rate': round(float(feat_df['label'].mean()), 4),
        'train_auc': round(train_auc, 4),
        'train_acc': round(train_acc, 4),
        'holdout_auc': round(val_auc, 4),
        'atr_type': 'wilder_ewm_alpha_1_14',
        'exported_at': datetime.now().isoformat(),
        'robustness': 'R92-B 5/5 PASS (original), re-export with Wilder ATR',
        'median_fill': {k: round(v, 6) for k, v in med.to_dict().items()},
    }
    meta_path = OUTPUT_PATH.with_suffix('.meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n  {'='*50}")
    print(f"  Model saved: {OUTPUT_PATH}")
    print(f"  Meta saved:  {meta_path}")
    print(f"  Features: {FEATURE_COLS}")
    print(f"  Train samples: {len(feat_df)}")
    print(f"  Train AUC: {train_auc:.4f}")
    print(f"  Holdout AUC: {val_auc:.4f}")
    print(f"  Threshold: 0.65")
    print(f"  File size: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  {'='*50}")


if __name__ == "__main__":
    main()
