"""R257: keltner macro/趋势过滤验证 (接 R256, 回答"加趋势/方向过滤对 keltner 有没有用").

实盘发现 keltner BUY 长期亏(-$575)、SELL 净赚(+$166)(但混淆了已修的超持仓 bug)。
本实验在**当前实盘配置之上**(ATR≥6 下限 + $80 cap)干净回测各过滤, 每个拆 BUY/SELL:
  base       — 仅 ATR≥6 + cap80 (= 现在实盘)
  a1         — +EMA9/EMA21 排列 (BUY 要 EMA9>EMA21)
  a2         — +EMA50 斜率对齐
  a1a2       — +两者
  slow       — +EMA100 斜率对齐 (慢趋势=macro 代理: BUY 要 EMA100 上行)
  buy_slow   — 仅对 BUY 加 EMA100 斜率门 (SELL 不动; 测"专修 BUY 侧")
  sell_only  — 砍掉所有 keltner BUY (极端: 测 BUY 是否纯拖累)

注: macro(DXY/VIX)无历史数据接回测, 用慢趋势(EMA100 斜率)做方向代理。
执行: python experiments/run_r257_keltner_trend_filter.py
"""
import json, multiprocessing as mp, os, sys, time
from pathlib import Path
_PROJ_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np, pandas as pd

PERIOD_START, PERIOD_END = '2015-01-01', '2026-04-09'
RECENT = ('2024-01-01', '2026-04-09')
ATR_FLOOR = 6.0
EMA50_LB = 5
EMA100_LB = 5
FILTERS = ['base', 'a1', 'a2', 'a1a2', 'slow', 'buy_slow', 'sell_only']

OUTDIR = Path(_PROJ_ROOT) / 'results' / 'r257_keltner_filter'
OUTDIR.mkdir(parents=True, exist_ok=True)
_data = None; _init = False
_CFG = {'filter': 'base'}


def _passes(sig, df):
    """根据 _CFG['filter'] 判定该 keltner 信号是否通过 (含 ATR 下限)。"""
    d = sig.get('signal')
    last = df.iloc[-1]
    atr = last.get('ATR')
    if pd.isna(atr) or atr < ATR_FLOOR:
        return False
    fl = _CFG['filter']
    if fl == 'base':
        return True
    if fl == 'sell_only':
        return d == 'SELL'

    def a1_ok():  # EMA9 vs EMA21 排列
        ema9, ema21 = last.get('EMA9'), last.get('EMA21')
        if pd.isna(ema9) or pd.isna(ema21): return True
        return (ema9 > ema21) if d == 'BUY' else (ema9 < ema21)

    def a2_ok():  # SMA50 斜率 (EMA50 列不存在, 用 SMA50 代替): 最近 LB 根不单调反向
        e = df['SMA50'].iloc[-EMA50_LB:].values
        if len(e) < 2 or np.any(pd.isna(e)): return True
        diffs = np.diff(e)
        return not (np.all(diffs < 0) if d == 'BUY' else np.all(diffs > 0))

    def slow_ok():  # EMA100 斜率 (慢趋势 = macro 代理)
        e = df['EMA100'].iloc[-EMA100_LB:].values
        if len(e) < 2 or np.any(pd.isna(e)): return True
        return (e[-1] > e[0]) if d == 'BUY' else (e[-1] < e[0])

    if fl == 'a1':       return a1_ok()
    if fl == 'a2':       return a2_ok()
    if fl == 'a1a2':     return a1_ok() and a2_ok()
    if fl == 'slow':     return slow_ok()
    if fl == 'buy_slow': return slow_ok() if d == 'BUY' else True
    return True


def _worker_init():
    global _data, _init
    if _PROJ_ROOT not in sys.path:
        sys.path.insert(0, _PROJ_ROOT)
    from backtest.runner import DataBundle
    import indicators as sm
    _o = sm.scan_all_signals
    def patched(df_h1, df_m15=None, **kw):
        out = []
        for s in (_o(df_h1, df_m15, **kw) or []):
            if s.get('strategy') == 'keltner':
                if _passes(s, df_h1):
                    out.append(s)
        return out
    sm.scan_all_signals = patched
    _data = DataBundle.load_default(start=PERIOD_START)
    _init = True
    print(f'  [w{os.getpid()}] init', flush=True)


def _g(t, n, d=None):
    return getattr(t, n) if hasattr(t, n) else (t.get(n, d) if isinstance(t, dict) else d)


def _st(trades):
    p = np.array([_g(t, 'pnl', 0) or 0 for t in trades], float)
    if len(p) == 0:
        return dict(n=0, wr=0, pnl=0, mean=0, sharpe=0, worst=0)
    sh = float(p.mean()/p.std()*np.sqrt(252)) if p.std() > 0 else 0.0
    return dict(n=len(p), wr=round(float((p > 0).mean()*100), 0), pnl=round(float(p.sum()), 0),
                mean=round(float(p.mean()), 2), sharpe=round(sh, 2), worst=round(float(p.min()), 0))


def _recent(trades):
    a = pd.Timestamp(RECENT[0], tz='UTC'); b = pd.Timestamp(RECENT[1], tz='UTC')
    out = []
    for t in trades:
        et = _g(t, 'exit_time')
        if et is None: continue
        et = pd.Timestamp(et)
        if et.tzinfo is None: et = et.tz_localize('UTC')
        if a <= et <= b: out.append(t)
    return out


def _run(task):
    from backtest.engine import BacktestEngine
    from backtest.runner import REALISTIC_COST_BY_STRATEGY_KWARGS
    import indicators as sm
    from indicators import get_orb_strategy
    if not _init: _worker_init()
    _CFG['filter'] = task['filter']
    sm._friday_close_price = None; sm._gap_traded_today = False
    get_orb_strategy().reset_daily()
    s = pd.Timestamp(PERIOD_START, tz='UTC'); e = pd.Timestamp(PERIOD_END, tz='UTC')
    m15 = _data.m15_df[(_data.m15_df.index >= s) & (_data.m15_df.index <= e)]
    h1 = _data.h1_df[(_data.h1_df.index >= s) & (_data.h1_df.index <= e)]
    kwargs = {**REALISTIC_COST_BY_STRATEGY_KWARGS, 'max_positions': 1,
              'maxloss_cap': 80.0, 'maxloss_cap_atr_mult': 0}
    try:
        kt = [t for t in BacktestEngine(m15, h1, **kwargs).run() if _g(t, 'strategy') == 'keltner']
    except Exception as ex:
        import traceback
        return {'filter': task['filter'], 'error': f'{ex}\n{traceback.format_exc()[:400]}'}
    buy = [t for t in kt if _g(t, 'direction') == 'BUY']
    sell = [t for t in kt if _g(t, 'direction') == 'SELL']
    return {'filter': task['filter'], 'all': _st(kt), 'buy': _st(buy), 'sell': _st(sell),
            'rec_all': _st(_recent(kt)), 'rec_buy': _st(_recent(buy)), 'rec_sell': _st(_recent(sell))}


def main():
    t0 = time.time()
    print('=' * 96)
    print(f'R257: keltner 趋势/方向过滤 (ATR≥{ATR_FLOOR}+cap80 之上, {PERIOD_START}~{PERIOD_END})')
    print('=' * 96)
    with mp.Pool(min(8, mp.cpu_count()), initializer=_worker_init) as pool:
        res = pool.map(_run, [{'filter': f} for f in FILTERS])
    R = {r['filter']: r for r in res if 'error' not in r}
    for r in res:
        if 'error' in r: print(f"[ERR] {r['filter']}: {r['error'][:200]}")

    def show(key, title):
        print(f'\n### {title}')
        print(f'{"filter":10}| {"ALL n/PnL/Sh":>22} | {"BUY n/PnL/Sh":>22} | {"SELL n/PnL/Sh":>22}')
        for f in FILTERS:
            if f not in R: continue
            a, b, s = R[f][key], R[f][key.replace('all', 'buy')], R[f][key.replace('all', 'sell')]
            def c(x): return f'{x["n"]:>4}/{x["pnl"]:>7.0f}/{x["sharpe"]:>5.2f}'
            print(f'{f:10}| {c(a):>22} | {c(b):>22} | {c(s):>22}')

    show('all', f'全周期 {PERIOD_START}~{PERIOD_END} (keltner; 拆 BUY/SELL)')
    show('rec_all', f'近年 {RECENT[0]}~{RECENT[1]}')
    print('\n判读: 好的过滤 = ALL 的 PnL/Sharpe 升 且 BUY 侧明显改善, 而 SELL 不被砍太多。')
    print(f'Elapsed: {time.time()-t0:.0f}s')
    with open(OUTDIR / 'filter_results.json', 'w', encoding='utf-8') as f:
        json.dump(R, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
