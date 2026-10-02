# Polymarket 5分钟末秒狙击量化系统

纯本地代码，**默认不发起任何网络请求**。实时接入需要你自己注入 transport 与 broker——这是刻意设计的边界：回测引擎在结构上不可能误连外网，实盘也必须显式开启。

```
polymarket_sniper/
├── config/
│   ├── config.json          # 全局配置：行情端点、策略、执行、风控、手续费、回测
│   └── loader.py            # 带类型校验的配置加载（非法值直接报错，不静默）
├── data/
│   ├── param_combinations.csv        # 你的 1200 组参数
│   ├── param_combinations_full_grid.csv  # 14400 全网格
│   ├── commands.txt / commands_args.txt  # 服务器命令行
├── core/
│   ├── market_data.py       # 模块1 WebSocket+REST 行情、深度识别
│   ├── execution.py         # 模块4 滑点、自适应路由、拆单
│   ├── risk_control.py      # 模块3 动态风控、反向对冲
│   ├── backtest_engine.py   # 模块2a 回测引擎（手续费+VWAP滑点）
│   └── param_selector.py    # 模块2b 1200 组参数扫描与排序
├── strategies/
│   └── sniper_strategy.py   # 模块5 主策略：T-10s 末秒狙击
├── utils/
│   ├── fee_calculator.py    # 手续费函数
│   ├── vwap_calculator.py   # VWAP 函数
│   └── logger.py            # 日志与序列化
├── tools/
│   └── make_demo_params.py  # 按数据集实际有的时间点生成参数文件
├── output/                  # 回测/扫描结果
└── main.py                  # 入口：selfcheck / sweep / backtest / paper
```

## 先跑这个

```bash
cd polymarket_sniper
python main.py selfcheck
```

**40/40 通过**。它离线验证全部数学与决策链路：手续费公式、VWAP 走单、拆单、OBI、流动性评分、流动性墙、滑点容忍闸门、动态止损、反向对冲、末秒信号、数据质量闸门、回测引擎。不需要任何数据，不联网。

## 五大模块实现要点

### 1. 多源行情与深度识别（`core/market_data.py`）

* `OrderBook` 维护单个 outcome 的 L2 本地簿，支持快照与增量（sequence 乱序丢弃）。
* `MarketDataHub` 融合 WebSocket 流与 REST 快照，**交叉校验**：中间价偏离超过 `max_divergence`、任一源超时、盘口交叉、缺双边报价、现货价过期——任一命中就判定不可交易并给出原因。
* 派生指标：中间价、微价格（按量加权的五档中间价）、价差(bps)、**前5档流动性评分**（形态 + 规模两部分，均衡的簿得分高于单档堆量）、**OBI=(买量-卖量)/(买量+卖量)**、**流动性墙**（单档占前N档总量 ≥ `min_share`）。
* 传输层是注入式的：`OfflineRestTransport` / `RecordingStreamTransport` 供离线测试；`LiveWebSocketTransport` 可选，代码里**没有任何地方实例化它**。

### 2. 参数选择与回测引擎（`core/backtest_engine.py` + `core/param_selector.py`）

* 读取 `data/param_combinations.csv` 的**全部 1200 组**（时间点/阈值/止盈/止损/间隔）。
* **真实手续费**：`fee = shares × 0.07 × p × (1−p)`（即 700bps × p × (1−p)），实现于 `utils/fee_calculator.py`。峰值在 p=0.5、在两端趋于 0——这是这条曲线的特征，代码里用测试固定住了。同时提供 maker 返佣侧（返佣比例默认 25%，可配）。
* **VWAP 滑点**：每笔成交都沿订单簿逐档吃单计算 VWAP，滑点 = VWAP 与对手价之差（bps）。数据集没有深度时用 `SyntheticBookModel` 合成可复现的盘口，并在报告里标注 `book_source: synthetic`——**合成簿的滑点只代表方法，不代表那个市场的真实深度**。
* 缺失价格**跳过而不插值**，并在指标里计数（`skipped_no_price`）。
* 指标：净盈亏、毛盈亏、总手续费、手续费占毛盈亏比、平均滑点bps、胜率、盈亏比、期望值、最大回撤、Sharpe、Sortino、TP/SL/结算离场分布、对冲笔数、换手。
* `param_selector` 排序可选指标（`--rank-metric`），并支持 `--min-trades` 过滤——3 笔 100% 胜率的组合是噪声，不是优势。

### 3. 动态风险控制与反向对冲（`core/risk_control.py`）

* 止损不是固定值：`stop = clamp(base + mult×波动率 + |OBI|×base, min, max)`，`mult` 默认 2.5。
* 触发条件包括价格不利位移、OBI 极端（`obi_extreme_abs`）、OBI 反向翻转、跟踪回撤（`trailing_stop_prob`）、窗口结束强制平仓。
* **反向开仓对冲**：当信号强烈反转（`signal_strength ≥ hedge_min_signal_strength`）时，不是止损卖出，而是在**反向一侧开仓**，比例由 `hedge_ratio` 给出并按信心与失衡度动态缩放，上限 `max_hedge_ratio`，最多补仓一次。理由写在决策原因里：在已经走远的簿上卖出是最贵的离场方式。

### 4. 精确滑点控制（`core/execution.py`）

* `estimate_execution` 给出成交前 VWAP 估算、手续费、总成本、有效价格、滑点bps。
* **自适应订单路由** `choose_route`：临近收盘 ≤2s 或紧急度高 → Taker；价差过宽或波动率过高 → Maker；OBI 对我方不利 → Taker；否则按 `maker_first` 挂单，并在盘口内侧改善 1 tick。
* **拆单**：超过 `split_order_threshold_notional` 或超过可见深度的一半即拆分，子单数量按参与率(`split_participation_rate`)自适应，2~6 笔。
* **滑点容忍闸门**：市价单必须携带 `slippageTolerance`；超限或深度不足会直接拦下并给出原因，而不是硬吃。`check_fill` 还会在事后比对真实成交价与预估，偏差超 `max_slippage_bps` 视为异常成交并告警。
* 一个刻意的语义选择：滑点默认以**对手价**（touch）为基准度量"走单冲击"，而不是中间价。二元盘口 0.50/0.54 很常见，若按中间价算，光跨价差就会吃掉全部容忍度，容忍度就失去意义了。需要总量口径时用 `reference_side="mid"`。

### 5. 实盘策略主程序（`strategies/sniper_strategy.py` + `main.py`）

* T-10s 进入扫描窗口（`scan_window_seconds`）。
* 对比**币安现货**与 Polymarket 隐含概率：`implied_move_probability` 用逻辑斯蒂把「现货位移 / 剩余时间可波动幅度」映射成公允概率——剩余时间越少，同样的位移越决定性。
* 只有当「现货已确认方向」且「Polymarket 未跟上」时狙击：要求 `edge = 公允概率 − 市场概率` 落在 `[probability_gap_min, probability_gap_max]`。**上界同样重要**：edge 过大说明市场不是滞后而是不同意，那是在跟有信息的对手盘做对家。
* 入场前依次过：数据质量闸门 → 扫描窗口 → 方向 → 阈值/现货确认 → 价格带 → edge 区间 → 最小入场间隔 → 执行规划（滑点闸门）。每一步的拦截原因都会记录。
* 下单只交给注入的 `broker`，且只有 `mode=live` 才会真正调用；paper 模式永远停在执行层。

## 用法

```bash
python main.py selfcheck                    # 离线自检（先跑这个）
python main.py sweep --prices <csv> --markets <csv>   # 1200 组扫描 + 排序
python main.py backtest --combo c00001       # 单组回测
python main.py backtest --time 15 --threshold 0.05 --tp 0.10 --sl 0.05 --min-gap 0
python main.py paper --limit 200 --dump-decisions output\decisions.csv   # 复盘决策链路
```

常用筛选：`--times 30,60` 只跑这些时间点；`--combos 200` 只跑前 200 组；`--rank-metric sharpe`；`--min-trades 20`。

## 实测结果（用你 `../data` 里的真实数据）

* **T-15s 单组回测**（真实市场，500 个市场）：成交 1 笔，毛盈亏 10.87，**手续费 1.93（占毛盈亏 17.7%）**，平均滑点 31.1bps。499 个市场因 edge / 价格带不满足被跳过。这一笔就是全部——**它不是策略有效的证据**，原因见下面第 2 条。
* **1200 组全网格扫描**：0 笔成交——因为该数据集**只在 T-15s 有真实价格**，而网格区间是 T-30s~T-240s。系统如实报出原因，而不是拿空结果硬排个"最优参数"给你。
* **1200 组 T-15s 参数扫描**（`output/demo_param_combinations.csv`）：720 组产生成交，排序正常输出 `sweep_results.csv` 与 `sweep_ranking.json`。
* **paper 复盘**：默认 T-10s 窗口下 500 条观测全部 held，原因如实打印（数据在 T-15s，不在窗口内）；把窗口放到 `--scan-window 20` 后，被拦下的原因是「价格 0.9951 在入场带 [0.08, 0.92] 之外」——**近收盘价根本不该作为入场价**，闸门正确地拒绝了它。

## ⚠️ 关于这批数据的两个坑（必须知道）

1. **只有 T-15s 有真实价格**：当前数据集每个市场只有一个近收盘价格点（`implied_up_last`），其余 11 个观察时点为空。想让 T-30s~T-240s 的网格真正跑起来，需要海外服务器跑 `../scripts/scrape_btc5m.py` 抓逐时点价格，或下载 100ms 盘口归档。
2. **那是近收盘价，存在前视偏差**：拿它当 T-240s 的入场价，等于用了当时还不存在的信息。上面 T-15s 回测 100% 胜率就是这个原因，**不代表策略有效**。`--time 15` 之外的结果都不可信，除非换成真实逐时点数据。

## 边界与未做的事

* 没有把任何东西接到真实下单接口；`LiveWebSocketTransport` 需要你自己装 `websockets` 并显式实例化。
* `SyntheticBookModel` 的盘口是**模型**，用固定 tick 与随时间衰减的深度合成，只用于让滑点可计算、可复现；接入真实盘口后应替换为真实簿。
* 没有自动 `git push`，没有任何外网调用。
