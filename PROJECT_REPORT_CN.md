# 商店销量时间序列预测项目总结

## 1. 项目目标

本项目完成 Kaggle `Store Sales - Time Series Forecasting` 比赛任务。比赛要求根据历史销售数据、门店信息、商品品类、促销信息、油价、节假日和交易量等数据，预测未来 15 天每个门店和商品系列的销量。

本项目的核心目标有两个：

1. 构建一套可以稳定复现的机器学习预测流程。
2. 在 Kaggle Public Leaderboard 上取得尽可能低的 RMSLE 分数。

评价指标为 RMSLE。相比普通 RMSE，RMSLE 对低销量商品更敏感，因此模型不仅要预测高销量商品的趋势，也要避免对低销量或零销量组合产生过高预测。

## 2. 总体建模思路

本项目最终采用表格型时间序列建模方案。基本思路是将时间序列预测问题转换为监督学习回归问题：

- 每一行样本对应某一天、某个门店、某个商品系列。
- 目标变量为该样本对应的 `sales`。
- 输入特征包括日期特征、门店特征、商品特征、外部事件特征、促销特征、历史销量 lag 和 rolling 特征。
- 模型使用 LightGBM / XGBoost 对 `log1p(sales)` 建模。
- 预测时使用逐日递推方式生成未来 15 天销量。

这种方案相比复杂深度学习模型更容易解释、调试和复现，也更适合作为课程项目的最终交付版本。

## 3. 数据理解与预处理

原始数据包括：

- `train.csv`：历史训练数据，包含日期、门店、商品系列、促销数和销量。
- `test.csv`：未来 15 天需要预测的样本。
- `stores.csv`：门店所在城市、州、类型和 cluster。
- `oil.csv`：每日油价。
- `holidays_events.csv`：节假日和特殊事件。
- `transactions.csv`：门店每日交易量。
- `sample_submission.csv`：Kaggle 提交格式模板。

预处理阶段主要完成：

- 统一日期格式。
- 合并门店、油价、节假日和交易量数据。
- 对油价缺失值进行线性插值和前后填充。
- 将节假日按全国、地区、城市等层级转换为特征。
- 保证训练集、验证集和测试集的特征列一致。

## 4. 特征工程

### 4.1 日期特征

日期特征用于帮助模型识别周期规律和特殊日期效应，主要包括：

- 年、月、日。
- 星期几。
- 一年中的第几天。
- 一年中的第几周。
- 是否周末。
- 周期性正弦 / 余弦编码。
- 是否发薪日。
- 距离上一个 / 下一个发薪日的天数。

其中发薪日特征基于业务背景构造，因为工资发放前后往往会影响超市消费行为。

### 4.2 节假日与特殊事件特征

节假日和特殊事件对销量有明显影响。本项目构造了：

- 全国节假日特征。
- 州级节假日特征。
- 城市级节假日特征。
- 活动事件数量。
- 是否处于特殊事件影响窗口。

同时加入了地震事件相关窗口特征，用于标记异常时期对销量的影响。

### 4.3 油价特征

油价数据存在非连续日期和缺失值。本项目先补全完整日期，再进行插值，并构造：

- 当前油价。
- 原始油价缺失标记。
- 油价一阶差分。
- 油价百分比变化。
- 高油价标记。

这些特征用于表达宏观经济环境变化对消费需求的影响。

### 4.4 历史销量特征

历史销量是该任务中最重要的信号之一。本项目加入：

- `sales_lag_1`
- `sales_lag_7`
- `sales_lag_14`
- `sales_lag_28`
- `sales_roll_mean_7`
- `sales_roll_mean_28`
- `sales_roll_std_7`
- `sales_roll_std_28`

lag 特征表达过去固定时间点的销量，rolling 特征表达近期趋势和波动情况。

### 4.5 静态聚合特征

为了让模型获得更稳定的历史先验，本项目基于训练集构造了多组聚合统计特征：

- 门店维度销量均值和标准差。
- 商品系列维度销量均值和标准差。
- 门店-商品系列维度销量均值和标准差。
- 商品系列-星期维度销量统计。
- 门店-星期维度销量统计。
- 城市、州、门店类型维度销量统计。
- 商品系列-月份、门店-月份维度销量统计。

这些特征帮助模型理解不同门店和不同商品系列的基础销量水平。

### 4.6 促销特征

促销是销量预测中的重要因素。本项目加入：

- 促销数量。
- 促销数量的对数变换。
- 是否有促销。
- 当日总促销量。
- 门店当日促销总量。
- 商品系列当日促销总量。
- 当前样本促销占门店当日促销比例。
- 当前样本促销占商品系列当日促销比例。

这些特征用于表达促销活动的局部强度和相对强度。

### 4.7 Zero Forecasting 规则

部分门店和商品系列在历史上长期没有销量。对于这类组合，如果测试集中没有促销，本项目将预测值强制置为 0。

该规则能够减少无效预测，尤其对 RMSLE 指标比较有帮助。

## 5. 模型迭代过程

### 5.1 baseline 阶段

最初版本的目标是跑通完整流程，包括：

- 读取原始数据。
- 合并基础特征。
- 训练 LightGBM。
- 生成 Kaggle 提交文件。

这一阶段分数不高，但保证了项目管道完整，为后续迭代打下基础。

### 5.2 加入 lag / rolling 特征

第二阶段加入销量 lag 和 rolling 特征。该阶段显著提升了模型对时间序列规律的捕捉能力，也是后续所有版本保留的核心特征。

### 5.3 加入多源业务特征

第三阶段逐步加入油价、节假日、发薪日、地震事件、促销统计和静态聚合特征。经过特征重要性观察后，保留了对验证结果更稳定的特征，并剔除了部分低贡献特征。

### 5.4 从批量预测改为递推预测

这是项目中最关键的一次改造。

测试集是连续 15 天未来日期。如果直接一次性预测所有测试样本，第 2 天之后的 lag 和 rolling 特征无法使用前一天的预测结果，训练验证方式与真实提交场景会不一致。

因此最终采用 day-by-day recursive forecast：

1. 预测第 1 天。
2. 将第 1 天预测结果写回历史序列。
3. 用更新后的历史序列生成第 2 天 lag / rolling 特征。
4. 继续预测第 2 天。
5. 重复直到第 15 天结束。

递推预测使验证方式更接近 Kaggle 测试场景，是分数大幅提升的关键。

### 5.5 ensemble_v2

v2 版本在稳定递推框架上加入：

- 两个不同随机种子的 LightGBM。
- 一个 XGBoost 辅助模型。
- 基于递推验证分数的自动融合权重。
- 如果融合结果不如最佳单模型，则自动回退到最佳单模型。

v2 的 Kaggle Public Score 为 **0.40659**。

### 5.6 ensemble_v2_plus

v2_plus 在 v2 基础上加入轻量级预测校准层。校准层根据验证集中的预测偏差，对递推预测结果进行保守修正。

具体做法：

- 先用 v2 递推得到验证集预测。
- 计算不同校准策略在验证尾段上的效果。
- 候选策略包括全局校准、family 校准、store 校准、store-family 校准等。
- 如果校准策略优于原始预测，则使用该策略；否则自动回退。

最终 full 验证中：

- 原始递推验证 RMSLE：**0.40638**
- 校准后验证 RMSLE：**0.39079**

v2_plus 的 Kaggle Public Score 为 **0.40583**。

### 5.7 提交文件融合

由于 v2_plus 在线上只小幅优于 v2，说明校准层存在一定验证集过拟合。因此最后采用提交文件加权融合，提高泛化稳定性。

生成了三组融合：

- `blend_50plus_50v2.csv`
- `blend_70plus_30v2.csv`
- `blend_85plus_15v2.csv`

Kaggle Public Score：

- `blend_85plus_15v2.csv`：**0.40570**
- `blend_70plus_30v2.csv`：**0.40564**

当前最好版本为 `blend_70plus_30v2.csv`。

## 6. 最终文件结构

```text
scripts/
|-- run_ensemble_v1.py
|-- run_ensemble_v2.py
|-- run_ensemble_v2_plus.py
`-- blend_submissions.py

src/
|-- data/
|   `-- load_data.py
|-- ensemble_v1/
|-- ensemble_v2/
|-- ensemble_v2_plus/
|-- features/
|   `-- make_features.py
|-- models/
|   `-- train_lgbm.py
`-- utils/
    `-- metrics.py
```

主要文件作用：

- `src/data/load_data.py`：读取并合并原始 Kaggle 数据。
- `src/features/make_features.py`：构造日期、lag、rolling、聚合统计等通用特征。
- `src/ensemble_v2/features.py`：v2 特征组装和 zero forecasting。
- `src/ensemble_v2/models.py`：LightGBM / XGBoost 模型训练与预测。
- `src/ensemble_v2/pipeline.py`：v2 的训练、验证、递推预测和提交生成流程。
- `src/ensemble_v2_plus/pipeline.py`：v2_plus 的递推预测和校准流程。
- `scripts/run_ensemble_v2.py`：运行 v2。
- `scripts/run_ensemble_v2_plus.py`：运行 v2_plus。
- `scripts/blend_submissions.py`：融合 v2 与 v2_plus 提交文件。
- `src/utils/metrics.py`：RMSLE 评价指标。

## 7. 运行方式

安装依赖：

```powershell
pip install -r requirements.txt
```

快速调试：

```powershell
python scripts/run_ensemble_v2_plus.py --mode fast
```

完整训练：

```powershell
python scripts/run_ensemble_v2_plus.py --mode full
```

生成融合提交：

```powershell
python scripts/blend_submissions.py
```

推荐最终提交文件：

```text
submissions/blend_70plus_30v2.csv
```

## 8. 成绩记录

| 阶段 | 文件 | Public Score | 说明 |
|---|---|---:|---|
| baseline | baseline / early submission | 约 2.67 | 跑通完整流程 |
| early ensemble | ensemble_v1 early | 约 2.44 | 初步加入集成 |
| stable recursive | `ensemble_v1_full.csv` | 0.42660 | 递推预测稳定版 |
| v2 | `ensemble_v2_full.csv` | 0.40659 | 多模型候选和递推验证 |
| v2_plus | `ensemble_v2_plus_full.csv` | 0.40583 | 加入预测校准 |
| final blend | `blend_70plus_30v2.csv` | **0.40564** | 当前最好提交 |

当前最好提交时间：2026-05-02  
当前最好提交文件：`blend_70plus_30v2.csv`  
当前最好 Public Score：**0.40564**

## 9. 项目总结

本项目从一个简单 baseline 开始，逐步完成了完整的数据处理、特征工程、模型训练、递推预测、校准和提交融合流程。

最终经验总结如下：

1. 时间序列任务中，lag 和 rolling 特征是核心。
2. 验证方式必须尽量接近真实测试场景，递推预测比一次性预测更可靠。
3. 特征不是越多越好，训练集和测试集的一致性更重要。
4. 轻量校准可以带来提升，但需要通过融合降低过拟合风险。
5. 稳定、可复现、可解释的工程流程比复杂但难调的结构更适合作为课程项目最终版本。
