# 末秒狙击策略优化 — 数据接入说明（第 1 步）

## 为什么先要这一步

你的数据在 `D:\polymarket_data`，共 1.88 GB。**我这个会话看不到 D 盘**——实测本机只有 `C:`，`Test-Path D:\` 返回 false（D 盘可能是外接盘、或属于另一台机器、或没挂载）。

这意味着两件事我必须说清楚：

1. **我无法读取你的数据，也无法替你跑出回测结果。** 任何声称"我跑完了、胜率是 X%"的说法都是编的，我不会给你编的数字。
2. **但真正卡住回测的从来不是数据量，而是列名和值格式。** `timestamp` 还是 `t`？epoch 是秒还是毫秒？盘口是 `best_bid` 还是 `bids` 数组？猜错一个，代码就在第一行崩掉。所以我先把"看清数据"这件事做成了工具，并且**用真实文件测过**。

## 第一步：在你自己的终端跑这个（不联网、不写 D 盘）

```bash
cd C:\Users\孙奕涵\Desktop\polymarket\polymarket_sniper
python tools\inspect_dataset.py "D:\polymarket_data" --sample-rows 3 --out audit.json
```

它做四件事，全部经过内存安全设计：

| 做了什么 | 怎么做到不爆内存 |
|---|---|
| 列出每个子文件夹的文件数、总大小、扩展名分布 | 只走目录，不读内容 |
| 每个文件报告**真实列名 + dtype + 行数** | parquet 只读 footer 元数据；csv/jsonl 只读前 20MB 采样 |
| 每文件打印前 3 行**原始内容** | 有字节上限 |
| **自动识别时间列的单位**（秒/毫秒/微秒/纳秒/ISO） | 只看采样值的量级 |

常用参数：

```bash
python tools\inspect_dataset.py "D:\polymarket_data" --max-files 2      # 每个文件夹只看 2 个文件
python tools\inspect_dataset.py "D:\polymarket_data" --only orderbooks,crypto_prices
python tools\inspect_dataset.py "D:\polymarket_data" --bytes-per-file 5mb
```

**把这个 `audit.json`（或终端输出）发我**，我就能按你的真实列名写出能一次跑通的分析与回测代码，而不是让你来回试错。

## 第二步（你跑完第一步后，我立刻做）

拿到真实 schema，我会补齐这三个模块并针对你的数据跑出结果：

| 模块 | 内容 |
|---|---|
| `core/signal_mining.py` | 末1-3分钟错误定价规律：现货急跌 → UP 是否被过度抛售（<0.3）；反转成功样本的特征（价差、深度、OBI、剩余时间、跌幅大小） |
| `core/optimizer.py` | 入场时点扫描（T-60/-30/-15/-5）+ `movePct` / `windowMin` / `sumTarget` 网格优化，含**真实手续费** `shares×0.07×p×(1-p)` 与 VWAP 滑点 |
| `tools/run_analysis.py` | 一键跑完：对齐 → 信号挖掘 → 参数优化 → 输出报告 |

## 关于你的三个默认参数，先说结论方向（待数据验证）

| 参数 | 你的默认 | 我的判断 | 理由 |
|---|---|---|---|
| `movePct=0.15` | 0.15 | **很可能过大** | 15% 是三秒内的巨幅波动，BTC 极少出现。若原意是 0.15%，正确写法是 `0.0015`；若真是 15%，那信号几乎不会触发。这是最需要向数据求证的一条 |
| `windowMin=2` | 2 分钟 | **偏长，值得扫** | 越靠近结算，现货方向越决定性（同样跌幅的信息量随剩余时间下降而上升）；但越晚入场，盘口越薄、滑点越大。T-60/-30/-15/-5 必须用真实盘口分别算清楚 |
| `sumTarget=0.95` | 0.95 | **方向对，但要扣成本** | UP+DOWN≤0.95 锁定的是 5 分钱的毛差；扣掉双边手续费（p≈0.5 时单边约 1.75% of notional）与滑点后，净空间可能只剩 1-2 分。**必须用真实手续费重算保本点**，否则"锁利"可能是锁亏 |

另外提醒一个结构性风险：`UP + DOWN ≤ 0.95` 的机会在真实市场里往往是**因为结算结果存在分歧**（某一方信息更早），而不是无风险套利。这条我会用你的 `trades` 与 `price_history` 反查：这些"便宜的双边"最终是哪一边赢。

## 已经就绪并且在本地验证过的部分

| 文件 | 作用 | 验证状态 |
|---|---|---|
| `tools/inspect_dataset.py` | 数据审计（列名/行数/时间单位/样例） | **19/19 通过** |
| `core/dataset_loader.py` | 分块读取 + 时间单位归一 + 现货↔盘口 as-of 对齐 | **46/46 通过** |

`dataset_loader` 已覆盖的坑（都用测试固定住了）：

* 列名自动识别：`timestamp`/`t`/`ts`/`open_time`、`best_bid`/`bid`/`bids`、`best_ask`/`asks`、`market_slug`/`slug`/`condition_id`
* 时间单位自动判定：秒 / 毫秒 / 微秒 / 纳秒 / ISO-8601
* 盘口四种格式：`[[price,size]]`、JSON 字符串、`{"price":..,"size":..}` 列表、`{"0.52":100}` 映射
* 流式读取：`chunk_size` 上限强制生效，`max_rows` 可截断，**任何函数都不会整文件载入内存**
* **as-of 对齐严格向后取价**（不允许用到未来的现货价）——这是最容易让回测虚高的一处，已单独断言
* 交叉/越界盘口（bid≤0、ask≥1、bid≥ask）按数据损坏处理并丢弃

CSV / JSONL 路径**零第三方依赖**（用标准库跑通）；只有 parquet 需要 `pip install pyarrow`。

## 内存安全的具体做法（回应你的提醒）

1. 一律 `iter_rows(..., chunk_size=N)` 流式读，逐块处理完即释放。
2. 内存里只保留**现货序列**（每秒一行）用于 as-of 查询，不保留订单簿全量。
3. 需要落盘的联合数据集**边对齐边写 CSV**，不攒在内存里。
4. 若你的文件是 parquet，工具**只读 footer 拿 schema 和行数**，采样只读第一个 row group。
5. 提供 `--max-files`、`--bytes-per-file`、`--max-rows` 三重上限，先小后大。
