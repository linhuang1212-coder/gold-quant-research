#!/usr/bin/env python3
"""
ATR 对比 Harness: 量化旧ATR vs Wilder ATR 对 R92-B 决策的影响
===============================================================
对每个历史 L8_MAX 入场信号，分别用两套 ATR 计算 R92-B 的 12 个特征，
用已部署模型推理，输出决策不一致率。

Run from gold-quant-research root:
    python scripts/compare_atr_r92b.py
"""
import sys, os, io, json, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
warnings.filterwarnings('ignore')

MODEL_PATH = Path(r"C:\Users\hlin2\gold-quant-trading\data\l8_ml_exit_model.json")
THRESHOLD = 0.65

FEATURE_COLS = [
    'atr_14', 'adx_14', 'rsi_14', 'rsi_2',
    'kc_breakout_strength', 'volume_ratio', 'atr_percentile',
    'ema9_ema21_cross', 'close_ema100_dist',
    'hour_of_day', 'day_of_week', 'direction',
]

OUTPUT_DIR = Path("results/compare_atr_r92b")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def calc_atr_simple(df, period=14):
    """旧版: (High - Low).rolling(14).mean()"""
    return (df['High'] - df['Low']).rolling(period).mean()


def calc_atr_wilder(df, period=14):
    """新版: Wilder True Range + EMA(alpha=1/period)"""
    h, l, pc = df['High'], df['Low'], df['Close'].shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def compute_features_with_atr(h1_df, atr_series, entry_time, direction):
    """用给定的 ATR 序列计算 R92-B 12 个特征 (复制 ml_filter.py 逻辑)"""
    if isinstance(entry_time, str):
        entry_time = pd.Timestamp(entry_time)
    h1_time = entry_time.floor('h')
    # align tz with h1_df index
    if h1_df.index.tz is not None and h1_time.tzinfo is None:
        h1_time = h1_time.tz_localize(h1_df.index.tz)
    elif h1_df.index.tz is None and h1_time.tzinfo is not None:
        h1_time = h1_time.tz_localize(None)

    if h1_time in h1_df.index:
        loc = h1_df.index.get_loc(h1_time)
    else:
        idx_arr = h1_df.index.get_indexer([h1_time], method='ffill')
        if idx_arr[0] < 0:
            return None
        loc = idx_arr[0]

    if loc < 252:
        return None

    latest = h1_df.iloc[loc]
    atr_14 = float(atr_series.iloc[loc])
    if pd.isna(atr_14) or atr_14 <= 0:
        return None

    adx_14 = float(latest.get('ADX', np.nan))
    rsi_14 = float(latest.get('RSI14', np.nan))
    rsi_2 = float(latest.get('RSI2', np.nan))

    close = float(latest['Close'])
    ema20 = float(h1_df['Close'].iloc[:loc+1].ewm(span=20, adjust=False).mean().iloc[-1])
    kc_width = 2.0 * 1.5 * atr_14
    kc_breakout_strength = (close - ema20) / kc_width if kc_width > 0 else np.nan

    cur_range = float(latest['High'] - latest['Low'])
    avg_range = float((h1_df['High'] - h1_df['Low']).iloc[max(0,loc-19):loc+1].mean())
    volume_ratio = cur_range / avg_range if avg_range > 0 else np.nan

    atr_tail = atr_series.iloc[max(0, loc-251):loc+1].dropna()
    atr_percentile = float((atr_tail <= atr_14).sum() / len(atr_tail)) if len(atr_tail) >= 20 else np.nan

    ema9 = float(latest.get('EMA9', np.nan))
    ema21 = float(latest.get('EMA21', np.nan))
    if not any(np.isnan(v) for v in [ema9, ema21]) and atr_14 > 0:
        ema9_ema21_cross = (ema9 - ema21) / atr_14
    else:
        ema9_ema21_cross = np.nan

    ema100 = float(latest.get('EMA100', np.nan))
    close_ema100_dist = (close - ema100) / atr_14 if not np.isnan(ema100) and atr_14 > 0 else np.nan

    hour_of_day = h1_time.hour
    day_of_week = h1_time.weekday()

    return {
        'atr_14': atr_14, 'adx_14': adx_14, 'rsi_14': rsi_14, 'rsi_2': rsi_2,
        'kc_breakout_strength': kc_breakout_strength, 'volume_ratio': volume_ratio,
        'atr_percentile': atr_percentile, 'ema9_ema21_cross': ema9_ema21_cross,
        'close_ema100_dist': close_ema100_dist,
        'hour_of_day': hour_of_day, 'day_of_week': day_of_week,
        'direction': direction,
    }


def is_gap_bar(h1_df, loc):
    """判断 bar 是否有跳空 (|Open - prev_Close| > 0.5 * ATR)"""
    if loc < 1:
        return False
    gap = abs(h1_df['Open'].iloc[loc] - h1_df['Close'].iloc[loc - 1])
    atr_approx = (h1_df['High'].iloc[loc] - h1_df['Low'].iloc[loc])
    return gap > 0.5 * atr_approx if atr_approx > 0 else False


def main():
    t0 = time.time()
    print("=" * 70)
    print("  ATR 对比 Harness: Simple Range vs Wilder ATR → R92-B 决策漂移")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ── 加载模型 ──
    import xgboost as xgb
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_PATH))
    print(f"\n  R92-B model loaded: {MODEL_PATH}")
    print(f"  Threshold: {THRESHOLD}")

    # ── 加载数据 ──
    from backtest.runner import DataBundle, run_variant, LIVE_PARITY_KWARGS
    print("\n  Loading DataBundle...", flush=True)
    data = DataBundle.load_default()
    h1_df = data.h1_df
    print(f"  H1: {len(h1_df)} bars ({h1_df.index[0]} -> {h1_df.index[-1]})")

    # ── 跑 L8_MAX 拿所有信号 ──
    print("\n  Running L8_MAX backtest...", flush=True)
    kw = {**LIVE_PARITY_KWARGS, 'maxloss_cap': 35,
          'min_lot_size': 0.02, 'max_lot_size': 0.02}
    result = run_variant(data, "ATR_HARNESS", verbose=False, **kw)
    trades = result.get('_trades', [])
    print(f"  {len(trades)} trades")

    # ── 计算两套 ATR ──
    h1_raw = h1_df.copy()
    atr_old = calc_atr_simple(h1_raw, 14)
    atr_new = calc_atr_wilder(h1_raw, 14)

    # ── 预计算 ATR 差异统计 ──
    valid = atr_old.notna() & atr_new.notna() & (atr_old > 0)
    pct_diff = ((atr_new[valid] - atr_old[valid]) / atr_old[valid] * 100)
    print(f"\n  ATR 差异分布 (Wilder vs Simple, % change):")
    print(f"    p5={pct_diff.quantile(0.05):.2f}%  p50={pct_diff.quantile(0.50):.2f}%  "
          f"p95={pct_diff.quantile(0.95):.2f}%  max={pct_diff.max():.2f}%")

    # ── 对每个信号计算两套特征并推理 ──
    print(f"\n  Computing features for {len(trades)} trades...", flush=True)
    records = []
    n_skip = 0

    for i, trade in enumerate(trades):
        entry_time = trade.entry_time if hasattr(trade, 'entry_time') else trade.get('entry_time')
        if entry_time is None:
            n_skip += 1
            continue
        if isinstance(entry_time, str):
            entry_time = pd.Timestamp(entry_time)

        direction_str = trade.direction if hasattr(trade, 'direction') else trade.get('direction', '')
        if not direction_str:
            direction_str = trade.dir if hasattr(trade, 'dir') else trade.get('dir', 'BUY')
        direction = 1 if direction_str == 'BUY' else -1

        feat_old = compute_features_with_atr(h1_raw, atr_old, entry_time, direction)
        feat_new = compute_features_with_atr(h1_raw, atr_new, entry_time, direction)

        if feat_old is None or feat_new is None:
            n_skip += 1
            continue

        df_old = pd.DataFrame([feat_old])[FEATURE_COLS]
        df_new = pd.DataFrame([feat_new])[FEATURE_COLS]
        med_old = df_old.median()
        med_new = df_new.median()
        df_old = df_old.fillna(med_old)
        df_new = df_new.fillna(med_new)

        prob_old = float(model.predict_proba(df_old.values)[0][1])
        prob_new = float(model.predict_proba(df_new.values)[0][1])

        allow_old = prob_old >= THRESHOLD
        allow_new = prob_new >= THRESHOLD

        h1_time = entry_time.floor('h')
        if h1_raw.index.tz is not None and h1_time.tzinfo is None:
            h1_time = h1_time.tz_localize(h1_raw.index.tz)
        elif h1_raw.index.tz is None and h1_time.tzinfo is not None:
            h1_time = h1_time.tz_localize(None)
        if h1_time in h1_raw.index:
            loc = h1_raw.index.get_loc(h1_time)
        else:
            idx_arr = h1_raw.index.get_indexer([h1_time], method='ffill')
            loc = idx_arr[0]

        records.append({
            'entry_time': str(entry_time),
            'direction': direction_str,
            'prob_old': prob_old,
            'prob_new': prob_new,
            'allow_old': allow_old,
            'allow_new': allow_new,
            'disagreement': allow_old != allow_new,
            'prob_diff': abs(prob_new - prob_old),
            'atr_old': float(atr_old.iloc[loc]) if loc < len(atr_old) else np.nan,
            'atr_new': float(atr_new.iloc[loc]) if loc < len(atr_new) else np.nan,
            'is_gap': is_gap_bar(h1_raw, loc),
            'pnl': trade.pnl if hasattr(trade, 'pnl') else trade.get('pnl', 0),
        })

        if (i + 1) % 2000 == 0:
            print(f"    processed {i+1}/{len(trades)}...", flush=True)

    df = pd.DataFrame(records)
    print(f"\n  Processed: {len(df)} signals ({n_skip} skipped)")

    # ── 汇总结果 ──
    print(f"\n{'='*70}")
    print(f"  RESULTS")
    print(f"{'='*70}")

    n_total = len(df)
    n_disagree = df['disagreement'].sum()
    pct_disagree = n_disagree / n_total * 100 if n_total > 0 else 0

    print(f"\n  决策不一致率: {n_disagree}/{n_total} = {pct_disagree:.2f}%")

    if pct_disagree < 3:
        verdict = "SAFE: < 3%, 可直接 re-export 上线"
    elif pct_disagree < 10:
        verdict = "CAUTION: 3-10%, 需抽样检查不一致 bar"
    else:
        verdict = "HOLD: > 10%, 暂缓上线"
    print(f"  判定: {verdict}")

    # gap vs 非 gap
    gap_df = df[df['is_gap']]
    nongap_df = df[~df['is_gap']]
    if len(gap_df) > 0:
        gap_disagree = gap_df['disagreement'].sum() / len(gap_df) * 100
        print(f"\n  Gap bar 不一致率:    {gap_df['disagreement'].sum()}/{len(gap_df)} = {gap_disagree:.2f}%")
    else:
        print(f"\n  Gap bar: 无")
    if len(nongap_df) > 0:
        nongap_disagree = nongap_df['disagreement'].sum() / len(nongap_df) * 100
        print(f"  Non-gap 不一致率:   {nongap_df['disagreement'].sum()}/{len(nongap_df)} = {nongap_disagree:.2f}%")

    # 概率偏移分布
    prob_diffs = df['prob_diff']
    print(f"\n  概率偏移 |prob_new - prob_old| 分布:")
    print(f"    p50={prob_diffs.quantile(0.50):.4f}  p95={prob_diffs.quantile(0.95):.4f}  "
          f"max={prob_diffs.max():.4f}")

    # 不一致 case 详情
    if n_disagree > 0:
        disagree_df = df[df['disagreement']].head(10)
        print(f"\n  前 {len(disagree_df)} 个不一致 case:")
        print(f"  {'entry_time':25s} {'dir':5s} {'prob_old':>8s} {'prob_new':>8s} "
              f"{'old':>5s} {'new':>5s} {'gap':>4s} {'pnl':>8s}")
        for _, r in disagree_df.iterrows():
            print(f"  {r['entry_time']:25s} {r['direction']:5s} "
                  f"{r['prob_old']:8.4f} {r['prob_new']:8.4f} "
                  f"{'ALLOW' if r['allow_old'] else 'SKIP':>5s} "
                  f"{'ALLOW' if r['allow_new'] else 'SKIP':>5s} "
                  f"{'Y' if r['is_gap'] else 'N':>4s} "
                  f"${r['pnl']:7.2f}")

    # 新 ATR 对滤后策略表现的影响
    if n_disagree > 0:
        old_allowed_pnl = df[df['allow_old']]['pnl'].sum()
        new_allowed_pnl = df[df['allow_new']]['pnl'].sum()
        print(f"\n  滤后 PnL 对比:")
        print(f"    旧 ATR 模型通过的信号总 PnL: ${old_allowed_pnl:.2f}")
        print(f"    新 ATR 特征（旧模型）通过的信号总 PnL: ${new_allowed_pnl:.2f}")
        print(f"    差值: ${new_allowed_pnl - old_allowed_pnl:.2f}")

    # ── 保存详细结果 ──
    out_path = OUTPUT_DIR / "atr_comparison_detail.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  详细结果: {out_path}")

    summary = {
        'total_signals': n_total,
        'disagreements': int(n_disagree),
        'disagreement_pct': round(pct_disagree, 4),
        'verdict': verdict,
        'prob_diff_p50': round(float(prob_diffs.quantile(0.50)), 6),
        'prob_diff_p95': round(float(prob_diffs.quantile(0.95)), 6),
        'prob_diff_max': round(float(prob_diffs.max()), 6),
        'gap_bar_disagreement_pct': round(float(gap_disagree), 4) if len(gap_df) > 0 else None,
        'nongap_disagreement_pct': round(float(nongap_disagree), 4) if len(nongap_df) > 0 else None,
        'timestamp': datetime.now().isoformat(),
    }
    summary_path = OUTPUT_DIR / "atr_comparison_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  摘要: {summary_path}")

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
