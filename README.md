# Store Sales Time Series Forecasting

机器学习课程期末项目，任务来自 Kaggle `Store Sales - Time Series Forecasting`。项目目标是根据历史销量、门店信息、商品品类、促销、油价、节假日等数据，预测未来 15 天不同门店和商品系列的销量。

## 当前最好结果

| 版本 | 提交文件 | Public Score | 说明 |
|---|---|---:|---|
| v2 | `ensemble_v2_full.csv` | 0.40659 | 递推预测 + 多模型候选 |
| v2_plus | `ensemble_v2_plus_full.csv` | 0.40583 | 在 v2 基础上加入预测校准 |
| blend | `blend_70plus_30v2.csv` | **0.40564** | 70% v2_plus + 30% v2，当前最好 |
| blend | `blend_85plus_15v2.csv` | 0.40570 | 85% v2_plus + 15% v2 |

当前推荐提交文件：

```text
submissions/blend_70plus_30v2.csv
```

## 方法概述

本项目采用表格型时间序列建模方案，将销量预测问题转化为监督学习回归问题。

核心方法包括：

- 使用 LightGBM / XGBoost 对 `log1p(sales)` 建模。
- 使用 day-by-day recursive forecast 逐日递推未来 15 天销量。
- 构造销量 lag 特征，例如 `sales_lag_1`、`sales_lag_7`、`sales_lag_14`、`sales_lag_28`。
- 构造 rolling 特征，例如 7 日 / 28 日均值和标准差。
- 加入门店、商品、城市、州、门店类型等静态信息。
- 对油价进行插值，并构造油价变化特征。
- 加入节假日、发薪日、地震事件、星期、月份等时间和业务特征。
- 对历史长期为 0 的门店-商品组合加入 zero forecasting 规则。
- 在 v2_plus 中加入轻量级校准层，对递推预测结果做保守修正。
- 最后对 v2 和 v2_plus 的提交结果进行加权融合，得到当前最好提交。

## 项目结构

```text
.
|-- data/                         # Kaggle 原始数据，已被 .gitignore 忽略
|-- notebooks/                    # 可选探索 notebook
|-- outputs/                      # 中间输出和分析结果，已被 .gitignore 忽略
|-- scripts/
|   |-- run_ensemble_v1.py         # 早期稳定版本入口
|   |-- run_ensemble_v2.py         # v2 训练和提交入口
|   |-- run_ensemble_v2_plus.py    # v2_plus 训练、校准和提交入口
|   `-- blend_submissions.py       # 融合 v2 与 v2_plus 提交文件
|-- src/
|   |-- data/
|   |   `-- load_data.py           # 读取并合并 Kaggle 原始表
|   |-- ensemble_v1/               # 早期集成版本
|   |-- ensemble_v2/               # 当前主模型版本
|   |   |-- config.py              # 参数配置
|   |   |-- ensemble.py            # 预测融合工具
|   |   |-- features.py            # 特征组装和 zero forecasting
|   |   |-- models.py              # LightGBM / XGBoost 训练与预测
|   |   `-- pipeline.py            # 训练、验证、递推预测和提交生成流程
|   |-- ensemble_v2_plus/          # v2 的轻量校准版本
|   |   |-- config.py              # v2_plus 参数配置
|   |   |-- ensemble.py            # 融合工具
|   |   |-- features.py            # 特征工程
|   |   |-- models.py              # 模型训练与预测
|   |   `-- pipeline.py            # v2_plus 训练、递推、校准流程
|   |-- features/
|   |   `-- make_features.py       # 通用日期、聚合、lag、rolling 特征
|   |-- models/
|   |   `-- train_lgbm.py          # LightGBM 通用辅助函数
|   `-- utils/
|       `-- metrics.py             # RMSLE 评价指标
|-- submissions/                   # Kaggle 提交文件，已被 .gitignore 忽略
|-- PROJECT_REPORT_CN.md           # 中文项目总结
|-- GIT_SUBMIT.md                  # GitHub Desktop 提交说明
|-- README.md
`-- requirements.txt
```

## 数据文件

请将 Kaggle 下载的数据放在 `data/` 目录下：

```text
train.csv
test.csv
stores.csv
oil.csv
holidays_events.csv
transactions.csv
sample_submission.csv
```

这些原始数据文件不会上传到 GitHub，避免超过 GitHub 文件大小限制。

## 运行方式

安装依赖：

```powershell
pip install -r requirements.txt
```

运行 v2：

```powershell
python scripts/run_ensemble_v2.py --mode fast
python scripts/run_ensemble_v2.py --mode full
```

运行 v2_plus：

```powershell
python scripts/run_ensemble_v2_plus.py --mode fast
python scripts/run_ensemble_v2_plus.py --mode full
```

生成融合提交：

```powershell
python scripts/blend_submissions.py
```

默认会生成：

```text
submissions/blend_50plus_50v2.csv
submissions/blend_70plus_30v2.csv
submissions/blend_85plus_15v2.csv
```

## 备注

- `data/`、`outputs/`、`submissions/` 均已在 `.gitignore` 中忽略。
- GitHub 仓库只保存代码、配置、依赖和说明文档。
- 当前最推荐的 Kaggle 提交是 `blend_70plus_30v2.csv`。
