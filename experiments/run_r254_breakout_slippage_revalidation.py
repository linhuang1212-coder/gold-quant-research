"""R254: 突破类策略真实滑点重验 (sess_bo / donchian, 接 R253).

R253 证明 dual_thrust 的 edge 是滑点低估造成的幻觉。sess_bo 和 donchian 同为突破策略
(进场在区间突破瞬间 = 最宽点差 + 动量续走), 极可能有同一盲点。本实验:

  Part 1 (主引擎): keltner(对照) + sess_bo + dual_thrust(交叉验证), 三套成本:
    nocost            — 无滑点上界
    global_realistic  — 旧 REALISTIC_COST_KWARGS (全局 0.67/0.17, 低估突破类)
    by_strategy       — 新 REALISTIC_COST_BY_STRATEGY_KWARGS (突破类 1.70/1.90 实测)
  交叉验证: dual_thrust 在 by_strategy 下应 ≈ R253 measured_mean (~-$1492), 证明分档滑点接对了。

  Part 2 (bt_donchian): donchian 走独立回测, 对 spread 做敏感度阶梯, 找盈亏平衡点。
    donchian 验证用 SPREAD=0.30 (UNIT_LOT=0.01) — 对突破策略明显偏低。

执行: python experiments/run_r254_breakout_slippage_revalidation.py
"""
import sys
import time
from pathlib import Path

_PROJ_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd

PERIOD_START = '2022-01-01'
PERIOD_END = '2026-04-09'
RECENT = ('2024-01-01', '2026-04-09')


def _stats(trades, attr='pnl'):
    def g(t, n):
        return getattr(t, n) if hasattr(t, n) else (t.get(n) if isinstance(t, dict) else None)
    pnls = np.array([g(t, 'pnl') or 0 for t in trades], dtype=float)
    if len(pnls) == 0:
        return dict(n=0, wr=0.0, pnl=0.0, mean=0.0, sharpe=0.0, dd=0.0)
    wr = float((pnls > 0).mean() * 100)
    cum = np.cumsum(pnls)
    dd = float((np.maximum.accumulate(cum) - cum).max())
    sh = float(pnls.mean() / pnls.std() * np.sqrt(252)) if pnls.std() > 0 else 0.0
    return dict(n=len(pnls), wr=round(wr, 1), pnl=round(float(pnls.sum()), 1),
                mean=round(float(pnls.mean()), 2), sharpe=round(sh, 3), dd=round(dd, 1))


def _filt(trades, a, b):
    s = pd.Timestamp(a, tz='UTC'); e = pd.Timestamp(b, tz='UTC')
    out = []
    for t in trades:
        et = getattr(t, 'exit_time', None) if hasattr(t, 'exit_time') else (
            t.get('exit_time') if isinstance(t, dict) else None)
        if et is None:
            continue
        if isinstance(et, str):
            et = pd.Timestamp(et)
        if et.tzinfo is None:
            et = et.tz_localize('UTC')
        if s <= et <= e:
            out.append(t)
    return out


def part1_engine():
    from backtest.runner import (DataBundle, LIVE_PARITY_KWARGS,
                                  REALISTIC_COST_KWARGS, REALISTIC_COST_BY_STRATEGY_KWARGS)
    from backtest.engine import BacktestEngine
    import indicators as signals_mod
    from indicators import get_orb_strategy
    from experiments.run_r209v2_non_keltner_audit import (
        check_psar_signal, check_sess_bo_signal,
        check_dual_thrust_signal, check_chandelier_signal)
    import experiments.run_r209v2_non_keltner_audit as r209_mod

    r209_mod.LIVE_STRAT_CONFIGS = {
        'psar': {'enabled': False},
        'sess_bo': {'enabled': True, 'broker_gmt_offset': 0, 'session_hour_gmt': 12,
                    'lookback_bars': 4, 'sl_atr': 4.5, 'tp_atr': 4.0},
        'dual_thrust': {'enabled': True, 'n_bars': 6, 'k_up': 0.5, 'k_down': 0.5,
                        'sl_atr': 6.0, 'tp_atr': 8.0, 'trail_act_atr': 0.06, 'trail_dist_atr': 0.01},
        'chandelier': {'enabled': False},
    }
    _orig = signals_mod.scan_all_signals

    def patched(df_h1, df_m15=None, **kw):
        sigs = _orig(df_h1, df_m15, **kw) or []
        for fn in (check_psar_signal, check_sess_bo_signal,
                   check_dual_thrust_signal, check_chandelier_signal):
            s = fn(df_h1)
            if s is not None:
                sigs.append(s)
        return sigs
    signals_mod.scan_all_signals = patched

    data = DataBundle.load_default(start=PERIOD_START)
    s = pd.Timestamp(PERIOD_START, tz='UTC'); e = pd.Timestamp(PERIOD_END, tz='UTC')
    m15 = data.m15_df[(data.m15_df.index >= s) & (data.m15_df.index <= e)]
    h1 = data.h1_df[(data.h1_df.index >= s) & (data.h1_df.index <= e)]

    COSTS = {
        'nocost': dict(LIVE_PARITY_KWARGS),
        'global_realistic': dict(REALISTIC_COST_KWARGS),
        'by_strategy': dict(REALISTIC_COST_BY_STRATEGY_KWARGS),
    }
    STRATS = ['keltner', 'sess_bo', 'dual_thrust']
    out = {st: {} for st in STRATS}
    for cname, preset in COSTS.items():
        signals_mod._friday_close_price = None
        signals_mod._gap_traded_today = False
        get_orb_strategy().reset_daily()
        kwargs = {**preset, 'maxloss_cap': 0, 'max_positions': 4}
        t0 = time.time()
        eng = BacktestEngine(m15, h1, **kwargs)
        trades = eng.run()
        for st in STRATS:
            tr = [t for t in trades if (getattr(t, 'strategy', None) if hasattr(t, 'strategy')
                                        else t.get('strategy')) == st]
            out[st][cname] = {'all': _stats(tr), 'recent': _stats(_filt(tr, *RECENT))}
        print(f"  [{cname}] done {time.time()-t0:.0f}s", flush=True)
    return out


def part2_donchian():
    from backtest.runner import DataBundle
    from experiments.run_r242_donchian_comprehensive import bt_donchian, calc_metrics
    data = DataBundle.load_default(start=PERIOD_START)
    s = pd.Timestamp(PERIOD_START, tz='UTC'); e = pd.Timestamp(PERIOD_END, tz='UTC')
    h1 = data.h1_df[(data.h1_df.index >= s) & (data.h1_df.index <= e)].copy()
    rs = pd.Timestamp(RECENT[0], tz='UTC'); re_ = pd.Timestamp(RECENT[1], tz='UTC')
    h1_recent = h1[(h1.index >= rs) & (h1.index <= re_)].copy()

    SPREADS = [0.30, 0.60, 1.00, 1.50, 2.00, 3.00]  # 0.30 = 验证用; 突破真实成本应更高
    rows = []
    for sp in SPREADS:
        tr_all, _ = bt_donchian(h1, spread=sp)
        tr_rec, _ = bt_donchian(h1_recent, spread=sp)
        rows.append((sp, calc_metrics(tr_all), calc_metrics(tr_rec)))
    return rows


def main():
    t0 = time.time()
    print('=' * 80)
    print(f'R254: 突破类滑点重验  period {PERIOD_START}~{PERIOD_END}  recent {RECENT[0]}~{RECENT[1]}')
    print('=' * 80)

    print('\n### Part 1 — 主引擎 (keltner 对照 / sess_bo / dual_thrust)  [UNIT_LOT 0.04]\n')
    eng = part1_engine()
    COSTS = ['nocost', 'global_realistic', 'by_strategy']
    for st in ['keltner', 'sess_bo', 'dual_thrust']:
        print(f'\n--- {st} ---')
        print(f'{"cost":18} {"n":>5} {"WR%":>6} {"totPnL":>10} {"meanPnL":>9} {"Sharpe":>8} | '
              f'{"recent_n":>8} {"rec_PnL":>9} {"rec_mean":>9}')
        for c in COSTS:
            a = eng[st][c]['all']; r = eng[st][c]['recent']
            print(f'{c:18} {a["n"]:>5} {a["wr"]:>6.1f} {a["pnl"]:>10.1f} {a["mean"]:>9.2f} '
                  f'{a["sharpe"]:>8.3f} | {r["n"]:>8} {r["pnl"]:>9.1f} {r["mean"]:>9.2f}')

    print('\n\n### Part 2 — donchian spread 敏感度 (bt_donchian, UNIT_LOT 0.01)\n')
    rows = part2_donchian()
    print(f'{"spread":>7} {"n":>5} {"WR%":>6} {"totPnL":>10} {"meanPnL":>9} {"Sharpe":>8} | '
          f'{"rec_n":>6} {"rec_PnL":>9} {"rec_mean":>9}')
    for sp, a, r in rows:
        tag = '  <- 验证用' if abs(sp - 0.30) < 1e-9 else ''
        print(f'{sp:>7.2f} {a["n"]:>5} {a["wr"]:>6.1f} {a["pnl"]:>10.1f} {a["avg_pnl"]:>9.3f} '
              f'{a["sharpe"]:>8.3f} | {r["n"]:>6} {r["pnl"]:>9.1f} {r["avg_pnl"]:>9.3f}{tag}')

    print(f'\nElapsed: {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
