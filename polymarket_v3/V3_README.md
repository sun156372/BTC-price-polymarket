# Polymarket BTC 狙击策略 V3

**先读 [`V3_ANALYSIS.md`](V3_ANALYSIS.md)。** 它记录了 V3 的每一个参数是怎么测出来的，也记录了**为什么 V3 默认拒绝交易**。这份 README 只讲怎么用。

```
polymarket_v3/
├── run_v3.py                    唯一入口（selfcheck / ladder / fit / evaluate / paper）
├── V3_ANALYSIS.md               参数分析与优化报告（证据、方法、结论）
├── config_v3/
│   ├── strategy_v3.json         全部参数，每项标注依据
│   ├── standardiser_v3.json     拟合好的特征标准化器（由 fit 生成）
│   └── snapshot_example.json    纸面交易的输入样例，覆盖 5 种闸门结果
├── engine_v3/
│   ├── data.py                  数据读取 + 列契约 + 防未来函数规则
│   ├── signals.py               信号、标准化器、盈亏平衡数学
│   ├── strategy.py              决策闸门（唯一的决策路径）
│   ├── paper.py                 纸面交易执行器
│   └── config.py                类型化配置 + 启动即校验
└── tools_v3/
    ├── analyze_v3.py            walk-forward 信号搜索（600 配置 × 8 折）
    └── backtest_v3.py           TP/SL 路径研究 + 成本阶梯
```

---

## 核心结论（放在最前面，因为决定你怎么用）

| 指标 | 实测值 |
|---|---|
| 信号 | **反向短期动量**（5/15分钟上涨 → 买 DOWN） |
| 池化方向准确率 | **55.19%**（2,301 笔 / 237 天 / 8 折，7 折为正） |
| 最差折 | 47.83% |
| 频率 | **9.7 笔/天**（阈值 \|z\| ≥ 2.0） |
| 盈亏平衡所需准确率 | 进场价 0.535 时为 55.24% |
| **只剩的生存空间** | **进场价 ≤ 0.535**（真实盘口中位 1 分价差，T-240 中位 ask 0.500） |

**真实盘口已验证这条路径可行**：288 个市场的真实报价显示中位价差 1 分、80.4% 报价贴最小 tick，T-240 的中位 ask 是 0.500，taker 进场约 0.505 —— 而 V3 的数字模型预测 0.504。**模型与实测相差 0.1 分钱**，见分析报告第十一节。

---

## 命令

### 1. 自检（19 项断言，不需要数据、不联网）

```bash
cd polymarket_v3
python run_v3.py selfcheck
```

覆盖：盈亏平衡等式（与手续费曲线逐点比对）、手续费在两端归零/中间峰值、阶梯单调性、以及**六道闸门各自真的拦住单子**（价格超限、未实测成交价、陈旧报价、价差过宽、冷却、并发持仓）。

### 2. 查看盈亏平衡阶梯

```bash
python run_v3.py ladder
```

打印 0.45 → 0.75 每个进场价所需的准确率，并标出 0.535 这个闸门位置。

### 3. 拟合特征标准化器（**上线前必须做一次**）

```bash
python run_v3.py fit
```

**为什么必须做**：信号是"三个特征 z 分数的均值"。如果线上不做标准化，原始值（约 0.002）永远到不了 \|z\| ≥ 2，策略会**看起来在运行、实际一笔都不下**。这一步把训练期的 mean/std 落盘，纸面交易会加载同一个文件——保证**线上跑的就是测过的那个策略**。

### 4. 重跑 walk-forward 分析

```bash
python run_v3.py evaluate --folds 8
# 或直接调工具，参数更全
python tools_v3/analyze_v3.py --folds 9 --out ../output/v3
```

输出 `output/v3/v3_sweep.csv`（600 个配置的全部指标，含每折准确率）。

### 5. TP/SL 路径研究（建模的，注意假设）

```bash
python tools_v3/backtest_v3.py --folds 4 --sims 24
```

输出 `output/v3/v3_tp_sl_study.csv`。结论见分析报告第八节：**在这份数据上 TP/SL 不可判定**，且保守成本档下 49 个组合全部为负。

### 6. 胜率 / 盈亏比 / 频率专项报告（objective 要求的三项）

```bash
python tools_v3/edge_report_v3.py --folds 8
```

输出 `output/v3/v3_edge_report.csv` 并打印四张表：

1. **逐折胜率**（池化 0.5523，逐折标准差 0.0312，含把"胜率 > 50%"和"能赚钱"区分开的那一折）
2. **盈亏比矩阵**（PF 随进场价从 1.1501 掉到 0.9034，临界点 entry 0.5349）
3. **频率 vs edge**（阈值 0.75–3.0 的胜率、PF、每笔期望）
4. **日收益优化**（按"每天赚多少"而不是"每笔盈亏比"重排阈值 —— 结论与第 2 张表不同）

### 7. TP/SL 敏感性（判断"最优止盈止损"能不能信）

```bash
python tools_v3/tp_sensitivity_v3.py --sims 12
```

用 6 个同样合理的路径假设重跑同一组止盈止损，看"最优"会不会变：

- **持有到结算在 6 个假设下取值完全相同（+0.0221）** —— 唯一稳定的策略
- 紧止盈跨越 −0.085 → +0.201，符号在假设之间翻转
- 6 个假设给出 2 个不同的"最优退出" → **任何最优 TP/SL 推荐都不可辩护**

脚本还会打印**模型自校验**：把 TP/SL 都禁用时必须是 100% 结算，且 P&L
逐位等于闭式解。没有这个校验的障碍模型结果不要引用。

### 8. 真实盘口检验（**最重要的一步**）

```bash
python tools_v3/execution_check_v3.py
```

用 `markets/snapshots/trades/deltas.parquet`（真实 L2 盘口 + 68 万笔成交）测出 V3 之前只能假设的三件事：

- **价差**：中位 **1 分**，80.4% 报价贴最小 tick，90.5% 在 5 分内
- **深度**：10–50 USDC 的订单走价差以外成本**中位 0.00 分**
- **真实进场价**：T-240（窗口内 60–120s）中位 ask **0.500**，taker 进场约 **0.505**

结论：**V3 原先的 15 分钱保守档不现实**，`1 分价差 + 1 分滑点` 才是规划口径。
输出 `output/v3/v3_execution_check.csv`。

**限制**：只有 1 天（288 窗口）。它能定价成交，不能定 edge。

### 9. 纸面交易

```bash
python run_v3.py paper --snapshot config_v3/snapshot_example.json
```

样例覆盖 5 种结果，跑出来是这样：

```
btc-updown-5m-1766038500   hold   execution:unmeasured_fill    UP   2.00
btc-updown-5m-1766038800  enter                     ready    UP   2.09
btc-updown-5m-1766039100   hold  execution:entry_above_cap    UP   2.35
btc-updown-5m-1766039400   hold     signal:below_threshold   -0.13
btc-updown-5m-1766039700   hold  execution:spread_too_wide  DOWN  -2.23
```

日志写到 `output/v3/v3_paper_log.csv`，**包括所有被拒绝的单子**——拒绝原因就是模拟盘与实盘差距的度量。

#### 快照格式

```json
{
  "slug": "btc-updown-5m-1766038800",
  "ts": 1766038560,
  "end_ts": 1766039100,
  "features": {"btc_ret_15m": -0.0044, "ind_roc_5": -0.0058, "ind_roc_15": -0.0101},
  "quote": {"bid": 0.49, "ask": 0.50, "ts": 1766038560},
  "measured_fill": 0.50,
  "won": true
}
```

* `measured_fill`：**真实成交价**。为 `null` 时 V3 拒绝进场（`execution:unmeasured_fill`）。
* `won`：结算结果，用于更新虚拟账本（可选，回填历史时用）。

---

## 六道执行闸门

V3 的决策顺序（`engine_v3/strategy.py`，回测与实盘共用同一条路径）：

| # | 闸门 | 默认 | 触发时的拒绝原因 |
|---|---|---|---|
| 1 | 并发持仓 | 1 | `risk:position_open` |
| 2 | 冷却时间 | 300s | `risk:cooldown` |
| 3 | 信号强度 | \|z\| ≥ 2.0 | `signal:below_threshold` |
| 4 | 报价新鲜度 | ≤ 5s | `execution:quote_stale` |
| 5 | 价差 | ≤ 4 分 | `execution:spread_too_wide` |
| 6 | **实测成交价存在** | 必须 | `execution:unmeasured_fill` |
| 7 | **进场价上限** | **0.535** | `execution:entry_above_cap` |
| 8 | 滑点 | ≤ 3 分 | `execution:slippage_too_high` |

第 6、7 两条是 V3 相对 v1/v2 最重要的改动：

* **第 6 条**把"模拟盘与实盘的差距"变成一行代码——**没观测到真实成交价就不下单**，而不是拿模型价凑一个漂亮的回测。
* **第 7 条**把盈亏平衡写进闸门。进场价 0.54 时需要 55.74% 准确率，而信号只有 55.19%，所以 0.54 以上的成交**无论信号多强都是负期望**，直接拒绝。

---

## 配置

改 `config_v3/strategy_v3.json`。每一项都带 `_evidence` / `_comment` 说明依据。改完**必须重跑 `evaluate`**，否则配置和证据就脱钩了。

最值得动的三个：

| 参数 | 默认 | 改动的后果 |
|---|---|---|
| `signal.threshold_z` | 2.0 | 降到 1.0 → 43 笔/天但准确率 53.7%，edge 只剩 1.5 点，付不起摩擦；升到 2.5 → 准确率反而降 |
| `signal.features` | 短周期三件套 | 换成 `blend_all` 准确率略降但更稳（分析报告第 4 节） |
| `execution.max_entry_price` | 0.535 | 这是**数学结论不是偏好**。只有当你测出更高的准确率才能抬高它 |

---

## 安全边界

* `safety.paper_only = true`，`live_trading_enabled = false`。
* **这个包里没有任何下单客户端、没有 API key、没有签名代码。** 现金是 dataclass 里的一个 float。
* 配置校验会拒绝 `live_trading_enabled` 与 `paper_only` 同时为真。

---

## 上线前必须过的四关（来自分析报告第十一节）

1. ~~**导出 UP/DOWN 代币真实成交与盘口**~~ → **已完成**（`markets/snapshots/trades/deltas.parquet`）。
   结果：中位价差 **1 分**、T-240 中位 ask **0.500**、10–50 USDC 额外成本中位 **0.00 分**。
   **原先"拿不到 0.535 进场价"的担忧不成立。**
2. **把盘口采集从 1 天扩到 ≥ 30 天，并与信号同期。** ← **这是现在优先级最高的一关。**
   edge 来自 264.8 天但无价格，成本来自 1 天但有真实价格；**两者不同期是当前最大的剩余风险。**
3. **纸上跟踪 ≥ 2 周**（可用真实报价驱动），记录三个比例：
   `unmeasured_fill` / `entry_above_cap` / `spread_too_wide`。
4. 同期盘口到手后，回填 `ladder` 与 `execution_check_v3.py`，把 `max_entry_price`
   从单点值升级为**分布**。全部通过后，才把 `paper_only` 改为 `false`，并从最小 `notional` 起。

**当前状态：第 1 关已过，第 2 关未过。** 所以结论从"因价差未知不能上线"变成
**"成本不再是否决项，但 edge 与盘口不同期，仍需同期数据"**——详见分析报告第 11.7 节。
