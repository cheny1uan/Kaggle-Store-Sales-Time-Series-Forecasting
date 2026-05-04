# Store Sales Time Series Forecasting

Kaggle `Store Sales - Time Series Forecasting`   
目标是根据历史销量、门店信息、商品类别、促销、油价、节假日和交易量，预测未来 15 天每个门店-商品组合的销量。

## 最终结果

| 版本 | 提交文件 | Public Score | Rank | 说明 |
|---|---|---:|---:|---|
| Final | `submissions/v4_logblend_55since_45full.csv` | **0.40387** | **100** | 最终提交，`v4_since2015` 与 `v4_full` 的 log-space blend |
| v4_since2015 | `submissions/ensemble_v4_since2015.csv` | 0.38540 | - | 2015 之后窗口训练，长滞后版本 |
| v4_full | `submissions/ensemble_v4_full.csv` | 0.38599 | - | 全历史训练版本 |
| v3_full | `submissions/ensemble_v3_full.csv` | 0.40439 | 103 | 递推集成成型版本 |
| v2_plus_full | `submissions/ensemble_v2_plus_full.csv` | 0.40583 | - | 在 v2 基础上继续加强特征和校准 |
| v2_full | `submissions/ensemble_v2_full.csv` | 0.40659 | - | 早期稳定基线版本 |

## 项目思路

这个项目不是一开始就追求“复杂模型”，而是按“先把系统做稳，再逐步增加有效信息”的顺序推进。

整体路线可以概括为：

1. 先做一个能稳定跑通的 baseline。
2. 再把销量预测拆成递推预测问题，逐日生成未来 15 天结果。
3. 引入交易量子模型，让交易量先预测，再作为销量模型的输入之一。
4. 逐步补充更有业务含义的特征，例如油价、节假日、发薪日、地震事件、零销量规则。
5. 用多模型、多随机种子、验证集权重搜索、校准和最终提交级别 blend 提升分数。
6. 在最后一轮，把“全历史”和“近年窗口”两种训练视角合在一起，拿到最终的 0.40387 / Rank 100。

## 版本演进

| 版本 | 更新思路 | 主要方法 | 结果 |
|---|---|---|---|
| v1 | 先把数据管道和递推框架跑通 | 合并 train/test/stores/oil/holidays/transactions，构造日期特征、基础 lag/rolling、LightGBM baseline、递推预测 | 作为工程起点 |
| v2 | 把单模型升级为多模型集成 | sales / transactions 分开建模，LightGBM + XGBoost，多随机种子，recursive forecast，基于验证集做初步融合 | `0.40659` |
| v2_plus | 继续补业务信息，提升稳定性 | zero forecasting、油价插值、发薪日、地震、特征重要性筛选、简单校准、submission-level blend | `0.40583`，后续 blend 到 `0.40564` |
| v3 | 把整条流水线系统化，减少手工拼接 | 更完整的递推框架、交易量单独预测、权重搜索、校准策略搜索、stacker 实验、提交级别融合 | `0.40439`，Rank 103 |
| v4 | 继续提高上限，转向更长历史和双窗口训练 | 长滞后特征、命名节假日特征、2015 之后窗口、全历史窗口、log-space blend | 最终 `0.40387`，Rank 100 |

### 这几类尝试的作用

- **递推预测**：比赛要求预测未来 15 天，不能直接用未来真实值，所以每天都要用前一天预测出的结果继续滚动。
- **交易量单独建模**：交易量和销量存在很强的同步关系，先把交易量预测稳，销量模型会更有依赖。
- **zero forecasting**：某些 store-family 历史上长期为 0，直接置 0 比让模型“猜”更稳。
- **油价插值**：油价有缺失，插值后时间序列更连续，特征更容易被模型利用。
- **发薪日 / 节假日 / 地震**：这些是强业务事件，属于高收益特征。
- **长滞后特征**：`364/365/728` 这类年度周期特征对这个题很关键。
- **双窗口训练**：全历史保留长期信息，`since2015` 窗口更贴近后期分布，最后用 log blend 合并。

## 最终采用的方法

### 1. 数据整合

将 Kaggle 原始数据合并为统一特征表：

- `train.csv`
- `test.csv`
- `stores.csv`
- `oil.csv`
- `holidays_events.csv`
- `transactions.csv`

再补充：

- 门店静态信息
- 油价及其变化特征
- 节假日 / 事件特征
- 交易量特征
- 日期时间特征

### 2. 特征工程

当前版本里保留了以下几类特征：

- 日期特征：`day`、`month`、`dayofweek`、`dayofyear`、`weekofyear`、`is_weekend`、`doy_sin/cos`
- 门店 / 商品静态特征：`store_nbr`、`family`、`city`、`state`、`type`、`cluster`
- 销量 lag / rolling：`sales_lag_1/7/14/28/56/91/182/364/365/728`
- 交易量 lag / rolling：`transactions_lag_1/7/14/28`
- 油价特征：`dcoilwtico`、差分、涨跌幅、插值后的连续值
- 节假日特征：全国 / 区域 / 本地 / 事件 / 工作日 / 转移节假日
- 命名节假日特征：earthquake、christmas、mothers_day、labor_day、new_year、soccer、dead_day、black_friday、cyber_monday
- 促销交互特征：`onpromotion` 与油价、family mean、store-family mean 的交互
- 目标聚合特征：按 `family`、`store_nbr`、`store_nbr+family`、`dayofweek` 等维度做均值 / 标准差统计

### 3. 递推预测

我最终没有把问题当成“直接回归一个 15 天向量”，而是做成了 **day-by-day recursive forecast**：

- 第 1 天先预测
- 把第 1 天预测值写回历史
- 再预测第 2 天
- 直到滚动完 15 天

这样做的好处是：

- 更符合比赛真实场景
- 可以在每一天都重新注入 lag / rolling 特征
- 交易量和销量可以形成互相依赖的闭环

### 4. 模型层

主模型是树模型路线：

- LightGBM
- XGBoost

并且不是只跑一个模型，而是做了：

- 多随机种子
- 多窗口训练
- 验证集权重搜索
- 递推结果融合
- log-space blend

### 5. 校准与后处理

在验证集上，我额外做了一个轻量校准层：

- 按 `store_family` / `family_store` / `store` / `family` / `global` 这些粒度尝试残差修正
- 选择验证集上最稳的方案
- 最终使用 `store_family` 校准策略

同时做了：

- 非负裁剪
- zero forecasting 覆盖
- submission-level log blend

## 文件夹结构

```text
.
|-- data/                      # Kaggle 原始数据，未提交到 GitHub
|-- notebooks/                 
|-- outputs/                   # 运行日志、特征重要性、验证输出，未提交
|-- scripts/
|   |-- run_ensemble_v1.py     # v1 baseline 入口
|   |-- run_ensemble_v2.py     # v2 入口
|   |-- run_ensemble_v2_plus.py # v2_plus 入口
|   |-- run_ensemble_v3.py     # v3 入口
|   |-- run_ensemble_v4.py     # v4 入口
|   `-- blend_submissions.py    # 旧版提交融合工具
|-- src/
|   |-- data/
|   |   `-- load_data.py       # 读取、合并、插值 Kaggle 原始数据
|   |-- ensemble_v1/           # 最早的 baseline 模块
|   |-- ensemble_v2/           # v2 模块
|   |-- ensemble_v2_plus/      # v2_plus 模块
|   |-- ensemble_v3/           # 当前主线实现，承载 v4 逻辑
|   |-- features/
|   |   `-- make_features.py   # 通用 lag / rolling / target aggregate 工具
|   |-- models/
|   |   `-- train_lgbm.py      # LightGBM 通用训练辅助函数
|   `-- utils/
|       `-- metrics.py         # RMSLE 等指标
|-- submissions/               # Kaggle 提交文件，未提交
|-- PROJECT_REPORT_CN.md       # 中文项目总结
|-- SUBMISSION_QUEUE.md        # 当前推荐提交清单
`-- README.md
```

> 说明：目录名里有些保留了 `v3`，但当前主线逻辑已经是最后的 v4 迭代。

## 运行方法

### 1. 安装依赖

```powershell
pip install -r requirements.txt
```

### 2. 跑 v4 版本

快速验证：

```powershell
python scripts/run_ensemble_v4.py --mode v4_fast
```

完整训练：

```powershell
python scripts/run_ensemble_v4.py --mode v4_full
```

更推荐的最终窗口版本：

```powershell
python scripts/run_ensemble_v4.py --mode v4_since2015
```

### 3. 当前推荐提交

```text
submissions/v4_logblend_55since_45full.csv
```

如果这个文件不方便提交，再退回：

```text
submissions/ensemble_v4_since2015.csv
submissions/ensemble_v4_full.csv
```

## 我做过但没有进入最终提交的尝试

- 更激进的 stacker
- 最近 365 天窗口单独训练
- 更复杂的后校准组合
- 更大范围的 submission-level 权重扫描

这些尝试里，只有少数对验证集有帮助，最后没有直接作为最终提交版本。

## 备注

- `data/`、`outputs/`、`submissions/` 都不建议上传到 GitHub。
- GitHub 仓库只保留代码、配置、说明文档和依赖文件。
- 当前最好结果已经达到 **0.40387 / Rank 100**。
