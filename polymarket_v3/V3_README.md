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
| **只剩的生存空间** | **进场价 ≤ 0.535，滑点 ≤ 3 分钱** |

**所以 V3 的行为是：大多数时候说"不交易"。** 这不是保守，是唯一的正期望做法。

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

### 8. 纸面交易

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

## 上线前必须过的四关（来自分析报告第十节）

1. **导出 UP/DOWN 代币真实成交与盘口**（`price, size, side, timestamp`）。这一件事的价值大于所有参数搜索。
2. **纸上跟踪 ≥ 2 周**，记录三个比例：`unmeasured_fill` / `entry_above_cap` / `spread_too_wide`。
3. 用真实进场价分布回填 `ladder`，确认 edge 仍为正。
4. 全部通过后，才把 `paper_only` 改为 `false`，并从最小 `notional` 起。

**在当前数据下，第 1 关过不了，所以 V3 不应上真金。** 这不是代码问题，是数据问题：v1 的 orderbook 显示 99.77% 的价差 ≥ 0.50，如果现在仍是那样，0.535 的进场价根本拿不到，策略归零。
