# Fix #3 (Wilder ATR) 部署手册

> 编写日期: 2026-05-17
> 状态: 待部署 (Fix #1 观察期结束后)

## 前置条件

- Fix #1 (未收K线丢弃) 已上线并观察 3-5 天，确认入场时间分钟数集中在 :00-:01
- Fix #2 (Swing High/Low) 暂不上线，留到 R173 重训后

## ATR Harness 结果摘要 (2026-05-17)

| 指标 | 值 |
|------|-----|
| 总信号数 | 21,390 |
| 决策不一致率 | 6.40% (1,369) |
| Gap bar 不一致率 | 11.19% (16/143) |
| Non-gap 不一致率 | 6.37% (1,353/21,247) |
| 概率偏移 p50/p95/max | 0.022 / 0.126 / 0.387 |

### 不一致信号质量分析

**ALLOW → SKIP (861 个, 被新特征 block):**
- 81.9% 的被 block 信号实际 PnL > 0 → 新特征+旧模型在这些 case 上偏保守了
- 总 PnL = +$236，平均 PnL = +$0.27
- 结论: 不换模型会损失一些小正 PnL 信号，但每个信号平均只亏 $0.27

**SKIP → ALLOW (508 个, 被新特征放行):**
- 68.5% PnL > 0，但总 PnL = -$1,508，平均 PnL = -$2.97
- 结论: 新放行的信号里有一些大亏损 case

**Gap bar 子集 (16 个):**
- 几乎是随机分布，样本太小无统计意义
- 无明显 red flag

**关键认知:**

1. -2.7% PnL 测的是"新特征+旧模型"的 fallback 场景，不代表"新特征+新模型"的生产表现。
2. 两个方向的可恢复性不对称:
   - ALLOW→SKIP (861个, -$232): 重训后模型重新适配新特征 scale，绝大部分应该能拿回来
   - SKIP→ALLOW (508个, -$1,508): 旧模型正确拒绝的信号，但在新 ATR 特征空间里可能呈现"好信号"模式，重训后未必完全反转
3. 新模型相对旧模型的 PnL 预期区间: 0 ~ -$1,500，realistic 落在 -$300 ~ -$500
4. **D+10 监控要点**: 如果新模型 ALLOW 了一些旧模型会 SKIP 的亏损交易，大概率是 SKIP→ALLOW 残留而非 bug，不触发 rollback。判断标准仍以 2 周 checklist 整体指标为准。

## 部署流程 (分两阶段)

### D 日: 上线 Fix #3 (保留旧模型)

1. 在 `gold-quant-trading/strategies/signals.py` 应用 Wilder ATR:
   ```python
   # 添加函数
   def calc_atr_wilder(df, period=14):
       h, l, pc = df['High'], df['Low'], df['Close'].shift(1)
       tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
       return tr.ewm(alpha=1/period, adjust=False).mean()

   # 替换 prepare_indicators 中的
   df['ATR'] = calc_atr_wilder(df, 14)  # 原: (df['High'] - df['Low']).rolling(14).mean()
   ```

2. 确认 `calc_adx` 内部的 ATR 不受影响 (它有自己的 TR 计算)

3. 不换模型，接受临时 ~2.7% PnL 漂移作为 isolation 成本

4. 观察 3 天:
   - ATR 值是否合理 (比旧版略大 3-5% 是正常的)
   - 没有异常大的 SL 距离
   - R92-B 的 allow/skip 比例与 D-1 相差不超过 5pp

### D+3 日: 换新模型

1. 在远程服务器运行:
   ```bash
   cd /root/gold-quant-research
   python3 scripts/export_r92b_model.py    # R92-B 模型
   python3 scripts/export_r173_model.py    # R173 模型 (shadow)
   ```

2. 两个脚本会自动:
   - 备份旧模型到 `data/model_backups/`
   - 用 Wilder ATR 特征重训
   - 输出 holdout AUC 对比

3. 将新模型复制到实盘:
   - `l8_ml_exit_model.json` → `gold-quant-trading/data/`
   - `r173_ml_filter.json` → `gold-quant-trading/data/`

4. 确认 holdout AUC 不低于旧模型 > 0.02

## 监控 Checklist (D 日起 2 周)

- [ ] R92-B allow/skip 比例: 与 baseline 偏差 < 5pp
- [ ] 日均信号触发数: 与 baseline 偏差 < 20%
- [ ] 平均 SL 距离 (pips): 预期 +5-20%，不超过 ATR_SL_MAX=150
- [ ] PnL: 在 baseline ±5% 内
- [ ] 无连续 3 笔以上同向反常亏损
- [ ] R173 shadow log: 概率分布无剧变

## 回滚方案

1. `data/model_backups/` 里有带时间戳的旧模型备份
2. `signals.py` 的 ATR 行改回 `(df['High'] - df['Low']).rolling(14).mean()` 即可
3. 回滚后所有依赖 ATR 的指标 (KC, dist_to_resistance 等) 自动恢复

## 文件清单

| 文件 | 项目 | 用途 |
|------|------|------|
| `scripts/compare_atr_r92b.py` | research | ATR 对比 harness |
| `scripts/export_r92b_model.py` | research | R92-B 重训导出 |
| `scripts/export_r173_model.py` | research | R173 重训导出 |
| `results/compare_atr_r92b/` | research | harness 结果归档 |
| `strategies/signals.py` | trading | ATR 修改位置 |
| `data/l8_ml_exit_model.json` | trading | R92-B 模型 |
| `data/r173_ml_filter.json` | trading | R173 模型 |
| `data/model_backups/` | trading | 模型备份目录 |
