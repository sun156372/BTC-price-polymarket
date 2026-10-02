# Polymarket BTC 5分钟市场 · 数据集 + 1200 参数组合 + 命令行参数

三样东西全部就绪：

1. **1200 个参数组合** —— 已生成、已校验（不依赖网络）
2. **可直接粘贴的模拟盘命令行** —— 已生成、已校验（不依赖网络）
3. **BTC 5分钟市场数据集** —— 两条获取路径：公开数据集（走国内镜像，本机已验证可下载）/ 直连 API 抓取脚本（海外服务器）

---

## 一、最快路径：公开数据集（镜像下载，无需 Polymarket 连通性）

本机（DSH 沙箱）**无法直连 Polymarket API**，但**可以访问 `hf-mirror.com`**。已实测下载可用的公开数据集：

| 数据集 | 内容 | 大小 | 用途 |
|---|---|---|---|
| [Sigrex/polymarket_btc_up_down_5m_251218_260909](https://hf-mirror.com/datasets/Sigrex/polymarket_btc_up_down_5m_251218_260909) | **71,602 个已结算的 BTC 5分钟市场**（2025-12-18 → 2026-09-08），42 列，含 `label_up` / `winning_outcome` / `start_ts` / `end_ts` / Chainlink 结算价 / Binance 现货 | 56.7 MB | **主力数据源** |
| [whodisidk/polymarket-btc-updown-market-data](https://hf-mirror.com/datasets/whodisidk/polymarket-btc-updown-market-data) | 逐日归档的 **100 毫秒级盘口快照**（books/external/lifecycle/resolution 五张表），2026-03-27 → 2026-05-24 | 每条约 330 MB | 需要盘口级精度时 |

### 下载 + 提取 500 个市场

```bash
# 1) 从镜像下载完整数据集（支持断点续传，中断后重跑同一条命令即可）
python3 scripts/hf_fetch.py "Sigrex/polymarket_btc_up_down_5m_251218_260909" \
  "polymarket_btc_up_down_5m_251218_260909.csv" work/sigrex_btc5m.csv

# 2) 抽出最近 500 个市场，转成本项目格式
python3 scripts/extract_dataset.py --input work/sigrex_btc5m.csv --count 500

# 3) 校验产物
python3 scripts/verify_dataset.py
```

镜像测速约 **40–110 KB/s**，56.7 MB 大约需要 10–25 分钟；`hf_fetch.py` 支持 HTTP Range 断点续传和自动重试，慢但不会白跑。

### ⚠️ 必须知道的数据限制（我没有替你造假数据）

上面这个公开数据集每个市场**只有一个真实价格点**（`implied_up_last`，收盘附近的 Up 代币价格），**没有窗口内的逐秒价格序列**。所以提取出来的 `btc5m_prices.csv`：

* ✅ 时间戳、窗口、**结算结果（UP/DOWN）全部是真实的**
* ✅ 那个价格点所在的观察时点（默认 `--time 15`）有**真实**的 `up_price` / `down_price`
* ❌ 其他 11 个观察时点**留空**，而不是编造数值

原因：Polymarket 公开 API 对已收盘市场的 `prices-history` 只返回 **2–30 个价格点**，本身就很稀疏；历史盘口官方不提供。每个观察点的真实覆盖度写在 `data/data_source.json` 里，一眼可见。

> ⚠️ **前视偏差警告（很重要）**：`implied_up_last` 是**收盘附近**的价格。如果你用 `--time 240` 去读它，等于用了当时还不存在的信息——**胜率会虚高**。实测在 500 个真实市场上跑 `--time 15` 得到 93% 胜率、盈亏比 29，这个数字**不代表真实策略表现**，只说明数据链路通了。要做多观察时点的参数扫描，必须用下面方案 A 或 B 拿到真实的逐时点价格。`sim_template.py` 在覆盖率不足时会自动打印这条警告。

### 想要每个观察时点都有真实价格，二选一

**方案 A：海外服务器直连 API 抓取**（推荐，数据最新最全）

```bash
cd /path/to/polymarket
./run_scrape.sh                      # 默认抓 500 个市场
```

`scripts/scrape_btc5m.py` 会用确定性 slug（`btc-updown-5m-{unix时间戳}`，按 300 秒向下取整）逐个找市场，从 CLOB 拉 UP/DOWN 两个代币的价格历史，按 `--offsets` 对齐到每个观察点，并用 Gamma 的 `outcomePrices` 判定结算结果。带限速（`--rps`）、指数退避重试、断点续抓缓存（`.cache/`）。

**方案 B：镜像下载 100 毫秒盘口归档**（需要耐心：每条约 330 MB）

```bash
python3 scripts/hf_fetch.py "whodisidk/polymarket-btc-updown-market-data" \
  "market_parquet_2026-04-01.tar.gz" work/whodisidk_2026-04-01.tar.gz
```

一个日档覆盖一整天约 288 个 BTC 5分钟市场，且含逐秒盘口（`books_100ms`）+ 结算表（`resolution`），可以算出任意观察时点的真实买卖价。代价是体积和下载时间。

---

## 二、1200 个参数组合

采样方式：**分层随机采样**，保证 8 个观察时间 × 10 个阈值每个格子都被覆盖，而非盲抽。校验 15/15 通过。

| 维度 | 命令行 | 取值 | 个数 |
|---|---|---|---|
| 观察时间点 | `--time` | 240, 210, 180, 150, 120, 90, 60, 30（秒，收盘前） | 8 |
| 价格偏移阈值 | `--threshold` | 0.01, 0.02, …, 0.10（%） | 10 |
| 止盈偏移 | `--tp` | 0.02, 0.05, 0.08, 0.10, 0.15, 0.20 | 6 |
| 止损偏移 | `--sl` | 0.02, 0.05, 0.08, 0.10, 0.15, 0.20 | 6 |
| 最小入场间隔 | `--min-gap` | 0, 30, 60, 90, 120（秒） | 5 |

* 全网格 = 8 × 10 × 6 × 6 × 5 = **14400 个组合**（完整清单：`params/param_combinations_full_grid.csv`）
* 抽样输出 = **1200 个组合**（种子 42，可复现）

```bash
python3 scripts/generate_params.py --count 3000 --seed 7    # 抽 3000 个
python3 scripts/generate_params.py --include-full           # 同时导出 14400 全网格
```

产出（`params/`）：`param_combinations.csv`、`param_combinations.json`、`param_combinations_full_grid.csv`、`grid_definition.json`。

---

## 三、服务器模拟盘命令行（可直接复制粘贴）

`params/commands.txt` 每行一条：

```
./sim --time 120 --threshold 0.1 --tp 0.15 --sl 0.1 --min-gap 0
./sim --time 240 --threshold 0.01 --tp 0.02 --sl 0.15 --min-gap 90
./sim --time 150 --threshold 0.06 --tp 0.08 --sl 0.08 --min-gap 30
```

换成你自己的程序：

```bash
python3 scripts/build_commands.py --prefix "python3 sim.py"
python3 scripts/build_commands.py --prefix "./sim --symbol BTCUSDT --data data/btc5m_prices.csv"
# 旗标名不同也能改：
python3 scripts/build_commands.py --prefix "./sim" \
  --time-flag --observe --threshold-flag --thr --tp-flag --take-profit --sl-flag --stop-loss --min-gap-flag --gap
```

| 文件（`params/`） | 用途 |
|---|---|
| `commands.txt` | **每行一条完整命令**，直接粘贴 |
| `commands_args.txt` | 只有参数部分（配 `xargs`） |
| `commands.csv` | combo_id + 参数 + 命令行，便于和绩效结果 join |
| `commands.jsonl` | 一行一个 JSON，便于喂队列 |
| `commands.sh` | bash 循环跑完 1200 个，日志写 `sweep_results.log` |

批量执行：

```bash
cd params && bash commands.sh                 # 顺序
cat commands_args.txt | xargs -P 4 -I {} ./sim {}   # 并行 4 路
```

### 先验证你的程序能接住这些参数

`templates/btc5m_sim_template.py` 是一个最小骨架，**接口就是这套旗标**，读上面两个 CSV 并打印每组的交易数、胜率、总盈亏、盈亏比、最大回撤。用它确认参数能跑通、再替换成你自己的模拟盘：

```bash
python3 templates/btc5m_sim_template.py --time 15 --threshold 0.05 --tp 0.10 --sl 0.05 --min-gap 0
```

---

## 四、脚本清单

| 文件 | 作用 | 状态 |
|---|---|---|
| `scripts/scrape_btc5m.py` | 直连 API 抓 500 个市场（限速/重试/缓存） | 已用本地假接口端到端验证 11/11 |
| `scripts/http_util.py` | HTTP 层：令牌桶限速 + 指数退避重试，requests 优先、stdlib 兜底 | — |
| `scripts/hf_search.py` | 在 hf-mirror 搜索数据集 | 已跑通 |
| `scripts/hf_inspect.py` | 列出数据集文件与体积 | 已跑通 |
| `scripts/hf_fetch.py` | 镜像下载（Range 断点续传 + 重试） | 已跑通 |
| `scripts/extract_dataset.py` | 公开数据集 → 本项目格式 + provenance | 已验证 16/16 |
| `scripts/verify_dataset.py` | 校验 `data/` 产物（连续性、标签平衡、时间戳一致性等） | — |
| `scripts/generate_params.py` | 生成参数组合 | 已验证 15/15 |
| `scripts/build_commands.py` | 组合 → 命令行 | 已验证 |
| `scripts/sim_template.py` | 模拟盘骨架（消费这套旗标） | 已被测通 |
| `scripts/test_mock_pipeline.py` | 抓取链路离线自测 | 11/11 通过 |
| `scripts/test_realdata_path.py` | 提取 + 模拟链路自测 | 16/16 通过 |

抓取脚本在没有 `requests` 的环境会自动退回标准库 `urllib`，裸 Python 也能跑。

---

## 五、常见报错处理

| 现象 | 处理 |
|---|---|
| `429 Too Many Requests` | `--rps 2 --workers 3` |
| 连接超时 / 抓不到数据 | 该机器出网受限；本机就是这种情况，改用镜像路径或换海外服务器 |
| 镜像下载很慢/中断 | `hf_fetch.py` 支持断点续传，重跑同一条命令即可接着下 |
| 抓到市场数 < `--count` | 正常，部分窗口没开盘；加大 `--max-back`（默认 `count×4`） |
| 最近几个市场 `result=UNKNOWN` | 链上结算延迟 5–10 分钟，稍后重跑（有缓存，很快） |
| 币安接口连不上 | 加 `--no-btc` |

---

## 六、下一步

1. 跑上面「一、最快路径」的三条命令，拿到 `data/btc5m_markets.csv` + `data/btc5m_prices.csv`。
2. 把 CSV 发我，我做数据质量校验（缺口、价差、UP+DOWN 是否≈1、结算一致性），并按需把 1200 个组合接成扫描输入。
3. 用 `bash params/commands.sh` 跑完扫描，把 `sweep_results.log` 发我，我按胜率、盈亏比、最大回撤、期望值筛出真正的好参数——**参数优劣只能由你服务器跑出的绩效来定，这一步不能靠猜**。

数据来源：[Sigrex BTC 5m 数据集](https://hf-mirror.com/datasets/Sigrex/polymarket_btc_up_down_5m_251218_260909)、[whodisidk 100ms 盘口归档](https://hf-mirror.com/datasets/whodisidk/polymarket-btc-updown-market-data)、镜像 [hf-mirror.com](https://hf-mirror.com)。
