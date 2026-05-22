---
description: Architecture of the live trading system (gold-quant-trading project)
globs: "**/*"
alwaysApply: true
---

# Live Trading System Architecture

## Project Location
`C:\Users\hlin2\gold-quant-trading`

## Architecture: Python-Driven with MT4 Execution Bridge

The live trading system is **entirely Python-driven**. MT4 only acts as an order executor.

```
Python (gold_trader.py) — 主进程，每15秒扫描
  ├── strategies/signals.py — 所有策略信号计算（Keltner, TSMOM, SESS_BO, DualThrust, M30_RSI14）
  ├── strategies/exit_logic.py — 出场逻辑（trailing stop, maxloss cap）
  ├── mt4_bridge.py — 通过文件桥接与MT4通信
  ├── config.py — 所有参数集中管理
  ├── risk_manager.py — 风控
  ├── data_provider.py — 数据获取
  └── position_tracker.py — 持仓跟踪

GoldBridge_EA.mq4 — MT4端，纯"下单执行器"
  - 每500ms检查 DWX/commands.json
  - 执行 OPEN/CLOSE/MODIFY 指令
  - 写回 heartbeat/positions/account/bars JSON
  - 不做任何策略逻辑
```

## Communication: File Bridge (DWX)

- 路径: `C:\Users\hlin2\AppData\Roaming\MetaQuotes\Terminal\35EEC3EFDB656AF6FC775F21FEAD053B\MQL4\Files\DWX\`
- Python写 `commands.json` → EA读取执行 → EA写 `response.json`
- EA每30秒写 `bars_h1.json` 和 `bars_m15.json`（200根K线）
- EA每5秒写 `heartbeat.json`, `account.json`, `positions.json`

## Live Portfolio (synced 2026-05-15 from config.py)

| Strategy | Lots | MaxLoss Cap | SL/TP (ATR) | Trail Act/Dist | Max Hold | Notes |
|----------|------|-------------|-------------|----------------|----------|-------|
| Keltner (Trail-First) | 0.04 | None (SL兜底) | 3.5 / 8.0 | **0.06 / 0.015** | 2 H1 bars | R202 trail, R178 session ADX, ML filter (0.65) |
| TSMOM | 0.04 | $60 (cap_atr=**1.0**) | 3.5 / 8.0 | 0.14 / 0.025 | 30 bars | Score 480/960, R205/R206, **R249 cap_atr 6.5→1.0** (2026-05-22) |
| SESS_BO | 0.04 | $60 (cap_atr=**1.0**) | 4.5 / 4.0 | 0.06 / 0.01 | 20 bars | GMT12 breakout, D1 EMA20 filter ON, **R249 cap_atr 5.0→1.0** (engine Sharpe 0.54→0.94) |
| Dual Thrust | 0.04 | $18 (cap_atr=**1.0**) | 6.0 / 8.0 | 0.06 / 0.01 | 20 bars | k=0.5, n_bars=6, **R249 cap_atr 5.0→1.0** (量增33%, MaxDD-18%) |
| M30 RSI14 | 0.04 | None | 8.0 / 8.0 | 0.30 / 0.08 | 24 bars | RSI14 mean-reversion on M30 |
| PSAR | OFF | - | - | - | - | R209v2: 10年Sharpe=-0.37, 关闭 |
| Chandelier | OFF | - | - | - | - | R209v2: 10年Sharpe=0.07, 关闭 |

Capital: $5,000 | MAX_POSITIONS: **5** | MAX_TOTAL_LOSS: $3,750
Active strategies: Keltner + TSMOM + SESS_BO + Dual Thrust + M30_RSI14

### Risk Controls
- MAX_LOT_CAP_BY_LOSSES: {0 losses: 0.13, 1: 0.08, 2: 0.05, 3: 0.03}
- Rule B: ATR > **2.5σ** (60-bar) → skip 8 hours (R205: 3.0→2.5)
- R187: ATR percentile floor = 30th (lb=300), skip low-vol entries
- R162: ATR > 2× rolling mean → halve lot size
- COOLDOWN_MINUTES: 30 (per strategy after losing close)
- DAILY_MAX_LOSS / DAILY_MAX_LOSSES: $9999 / 9999 (effectively disabled)

### ML Filters (live)
- **Keltner**: ml_filter.py, 12 features, threshold=0.65, model=l8_ml_exit_model.json
- **Non-Keltner**: ml_filter_r173.py, 4 features (ATR/ADX/RSI14/squeeze), threshold=0.70, model=r173_ml_filter.json, **LIVE** (not shadow)

### Session ADX (R178, deployed)
- Asia (0-7 UTC): ADX=14
- London (8-12 UTC): ADX=10
- NY (13-17 UTC): ADX=16
- Evening (18-23 UTC): ADX=14

## Key Implementation Details

### Signal Generation (strategies/signals.py)
- All indicators computed in Python using pandas (ATR, ADX, RSI, KC, EMA, PSAR, etc.)
- `prepare_indicators(df)` computes all technical indicators on H1 DataFrame
- `check_all_signals(df_h1, df_m15)` returns list of signal dicts
- Each signal: `{'strategy': 'keltner', 'signal': 'BUY'/'SELL', 'sl': ..., 'tp': ..., 'reason': ...}`

### Order Execution (mt4_bridge.py)
- `MT4Bridge.send_order(action, type, lots, sl, tp, ...)` writes commands.json
- Atomic writes via temp file + os.replace to prevent EA reading partial JSON

### Data Source
- EA writes H1/M15 bar data to `bars_h1.json`/`bars_m15.json` every 30 seconds
- Python reads these files via `data_provider.py`

### TSMOM Signal (IMPORTANT - R53/R56 Score method, NOT SMA crossover)
```python
def _tsmom_score(closes, idx, fast_lb, slow_lb):
    s = 0.0
    if idx >= fast_lb and closes[idx - fast_lb] > 0:
        ret = closes[idx] / closes[idx - fast_lb] - 1.0
        s += 0.5 * (1.0 if ret > 0 else (-1.0 if ret < 0 else 0.0))
    if idx >= slow_lb and closes[idx - slow_lb] > 0:
        ret = closes[idx] / closes[idx - slow_lb] - 1.0
        s += 0.5 * (1.0 if ret > 0 else (-1.0 if ret < 0 else 0.0))
    return s
```

## ML Exit Filter Integration Point

Since ALL strategy logic is in Python, ML Exit filter should be implemented as:
1. A Python module `strategies/ml_filter.py` with trained XGBoost model
2. Called AFTER signal generation, BEFORE sending to bridge
3. No MQL4 changes needed, no HTTP service needed, no separate process needed

```python
# In the signal processing flow:
sig = check_keltner_signal(df)
if sig and ml_filter.allows(sig, df):
    signals.append(sig)
```
