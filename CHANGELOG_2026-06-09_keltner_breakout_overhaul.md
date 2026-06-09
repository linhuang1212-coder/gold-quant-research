# 变更记录 — 2026-06-08/09 实盘亏损诊断 + keltner/突破类大修

> 起因: 实盘持续亏钱。从"今天 keltner 为什么亏"一路深挖, 诊断出多个根因并修复 + 用真实成本回测验证。
> 涉及两仓: **gold-quant-trading(实盘 VPS)** 改 config/runner/signals; **gold-quant-research** 改 engine/runner + 加 R253-R257 实验。

---

## 一、实盘改动 (VPS: gold-quant-trading)

| # | 文件 | 改动 | 原因 | 备份 |
|---|---|---|---|---|
| Fix A | config.py | `KELTNER_V1_DYNAMIC_MAX_HOLD = True→False` | 动态 max-hold 把 2h 放大到 4/6h 致超持仓巨亏; 改回固定 2h | config.py.bak.20260608 |
| Fix B | gold_runner.py:630 | 周五平仓窗 `20:50→20:30 UTC` | 旧窗太晚, 经纪商拒单致周末扛单 | gold_runner.py.bak.20260607 |
| — | config.py | dual_thrust `lots 0.04→0.01` (后又 `enabled→False`) | R253/R254: 真实滑点下 edge 是幻觉, 净亏 | config.py.bak.20260608b |
| — | config.py | sess_bo `enabled→False` | R254: 同 dual_thrust, 突破类滑点幻觉, 比 DT 更弱 | config.py.bak.20260609 |
| — | config.py:102 | 新增 `KELTNER_ATR_FLOOR = 6.0` | R256: keltner edge 由波动率决定, ATR<$6 净亏; 低波动不交易 | config.py.bak.20260609b |
| — | config.py keltner 块 | `maxloss_cap None→80.0` | R255/R255-B: 削尾(最差单笔 -601→-347)不伤 Sharpe | config.py.bak.20260609b |
| — | strategies/signals.py:511 | `check_keltner_signal` 加 ATR 下限门 | 落地 KELTNER_ATR_FLOOR (低 ATR 跳过 keltner 信号) | signals.py.bak.20260609 |

**当前实盘策略**: keltner(ATR≥6 门 + $80 cap)/ tsmom / m30_rsi14 / donchian(0.02)。
**已停用**: dual_thrust, sess_bo (+ 历史 psar/chandelier/macd/orb/m15_rsi/gap_fill)。
**运维**: runner 跑在 session 0(跨会话存活); watchdog.py 未运行(无自愈, 待补); 重启用 schtasks 拉 _start_runner.bat 回 session 0。

## 二、研发/引擎改动 (gold-quant-research)

| 文件 | 改动 | 说明 |
|---|---|---|
| backtest/engine.py | 加 `slippage_by_strategy` 参数 (+ `_calc_entry_slippage` 按策略查表) | 按策略/突破性分档滑点; 向后兼容(None=旧全局行为) |
| backtest/runner.py | 加 `SLIPPAGE_BY_STRATEGY` + `REALISTIC_COST_BY_STRATEGY_KWARGS` | 实测校准: 突破类 1.70/1.90, 均值回归 0.66/1.00 |

## 三、实验与结论 (gold-quant-research/experiments + results)

| 实验 | 结论 |
|---|---|
| R253 dt 真实滑点重验 | dual_thrust edge 是滑点低估幻觉: 校准0.67/0.17→实测1.7/1.9, +$7089/Sh1.49 → −$1492/Sh−0.32 |
| R254 突破类重验 | sess_bo+dual_thrust 同病翻负; **donchian 更稳**(慢突破); keltner 对照组稳活(方法精准非无差别杀) |
| R255 / R255-B keltner cap | 固定 $80 cap 优于 no-cap(Sharpe↑尾部↓); **ATR 比例 cap 反更差**(高波动放宽方向错); K-Fold 6/6 通过 |
| R256 / R256-B keltner 体制 | **keltner edge 几乎全由 ATR 决定**: ATR<$6 的 75% 单净亏 −$19465; ATR≥$6 门聚合 Sharpe −1.17→+1.08(K-Fold 验证) |
| R257 keltner 趋势/macro 过滤 | **趋势/macro 过滤无用**(略降收益); 干净回测 keltner BUY 不弱(实盘 BUY-bleed 是已修 bug); 已部署修复已足够 |

**核心认知**: 实盘 keltner 现在赚钱很大程度是踩中 2024-26 极端高波动金市; ATR 下限是"波动率回归"的保险。
**方法论**: 回测低估执行成本(尤其突破类滑点)会造出"假 edge"; 真实成本 + K-Fold 是照妖镜。

## 四、待办 (未做)

- 观察 1-2 周, 用干净的"修复后"实盘数据验证回测改善是否兑现。
- 把 watchdog.py 起起来(补进程自愈安全网)。
- donchian 用实测滑点单独重验(它也是突破类)。
- (Stage 5) VPS↔repo 三层漂移系统性收敛。
