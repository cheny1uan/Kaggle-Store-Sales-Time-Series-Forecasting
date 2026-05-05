# Store Sales Time Series Forecasting 项目复盘

## 1. 项目目标

本项目的目标是完成 Kaggle 竞赛 `Store Sales - Time Series Forecasting` 的销量预测任务：对未来 15 天、多个门店、多个商品 family 的销量进行预测，并以 RMSLE 作为评价指标。

我在整个项目里没有把问题当成“单一模型调参题”，而是按“先搭稳数据管道，再逐步引入强信号，最后做提交级融合”的顺序推进。这样做的原因很直接：

- 这类题的核心误差，通常不只来自模型容量，而来自时间递推、特征对齐、业务信号和后处理。
- 测试集没有真实标签，不能像普通监督学习那样直接训练最终输出，所以最终效果更多来自“中间预测是否稳定”。
- 与其一直堆复杂模型，不如把每一层都做得更稳。

最终提交文件是：

```text
submissions/final_submission.csv
```

最终 Public Score 为：

```text
0.39969
```

---

## 2. 数据和任务拆解

比赛提供的数据大致可以分成六类：

- `train.csv`：历史销量
- `test.csv`：未来 15 天待预测样本
- `stores.csv`：门店静态属性
- `oil.csv`：油价时间序列
- `holidays_events.csv`：节假日、事件、转移假日等
- `transactions.csv`：门店交易量

从建模角度看，这个任务有三个难点：

1. **时间递推**
   - 未来第 2 天、第 3 天……的输入特征会依赖前一天的预测结果。
2. **多源异构特征**
   - 门店、商品、交易量、油价、节假日、促销都不是同一种数据。
3. **业务规律强**
   - 发薪日、节假日、油价变化、地震事件、长期零销量商品，这些都比“纯模型参数”更重要。

因此项目拆成了几条主线：

- 统一数据整合
- day-by-day recursive forecast
- sales / transactions 分开建模
- 强业务特征补充
- 多模型融合和后处理
- submission-level blending

---

## 3. 整体演进路线

### 阶段 1: baseline

最初版本先做最朴素的管道：

- 合并 train / test 与门店、油价、节假日、交易量
- 构造基础日期特征
- 构造少量 lag / rolling 特征
- 用 LightGBM 做 baseline

这一阶段的目标不是冲分，而是确认：

- 数据合并是否正确
- 时间特征是否能跑通
- 递推预测是否会在测试集上报错

### 阶段 2: sales / transactions 分离

随后把销量预测和交易量预测拆开：

- 先预测 `transactions`
- 再把预测出来的交易量作为销量模型的输入之一

这样做的原因是交易量和销量虽然相关，但它们的生成机制不同。若直接把交易量当作静态真值，测试集会出现输入不可得的问题；先递推交易量，再递推销量，整体更符合真实评测场景。

### 阶段 3: 业务信号补充

在 baseline 稳定后，继续补强业务信号：

- 油价插值和移动平均
- 促销相关特征
- 发薪日特征
- 节假日 / 事件特征
- 地震事件标签
- zero forecasting 规则

这一阶段开始明显改善验证集表现，因为这些信号直接对应商品销量的真实波动。

### 阶段 4: 多模型与权重搜索

在 base 模型稳定后，引入多个随机种子和多个模型类型：

- LightGBM
- XGBoost

然后对验证集做权重搜索，找到更稳定的 blend 权重。

### 阶段 5: 更长历史窗口与更密滞后

后面又加入更长的时间跨度特征：

- 更长 lag
- 更长 rolling window
- 年周期特征
- promotion 的 lag / rolling
- 更长历史窗口训练

这一步的目标不是简单“加更多特征”，而是让模型能看到更完整的周周期、月周期和年周期。

### 阶段 6: family-aware submission blend

最后我没有继续死磕单模型，而是把目光转向 submission-level 融合：

- 先挑一个最稳定的主提交作为 anchor
- 再加入一条分布互补的参考预测
- 用 log-space 低权重融合

这一步带来了最终最好成绩。

---

## 4. 关键方法细节

### 4.1 为什么一定要 day-by-day recursive forecast

这个比赛的测试集是连续未来 15 天，不是一次性给你全部未来真实历史值。因此：

- 第 1 天预测出来后，要把第 1 天的预测写回历史
- 第 2 天的 lag 特征要能读到第 1 天的预测
- 第 3 天要读到前两天的预测

如果直接做 15 天多输出向量，很多特征会天然不成立；递推虽然慢，但和比赛逻辑更一致，也更容易把 lag 特征真正用起来。

### 4.2 为什么把 transactions 单独建模

销量和交易量之间存在同步关系，但不是简单的线性一一对应。

我的做法是：

1. 先用门店维度和时间维度预测 `transactions`
2. 再把预测交易量作为销量模型输入

好处有三个：

- 避免测试集信息泄漏
- 减少销量模型对交易量缺失的敏感性
- 让销量模型获得更接近真实的行为信号

### 4.3 特征工程的核心结构

#### 日期特征

基础时间特征包括：

- `day`
- `month`
- `dayofweek`
- `dayofyear`
- `weekofyear`
- `is_weekend`
- `doy_sin / doy_cos`

这些特征的意义是把时间的周期性显式暴露给模型。因为销量不是随机波动，而是被周、月、年周期强烈控制。

#### 门店 / 商品静态特征

包括：

- `store_nbr`
- `family`
- `city`
- `state`
- `type`
- `cluster`

这些特征告诉模型不同门店和商品族群的天然分层。

#### sales lag / rolling

最后保留的有效滞后窗口大致包括：

- `1, 7, 14, 28, 56, 91, 182, 364, 365, 728`

它们分别对应：

- 1 天：短期惯性
- 7 天：周周期
- 14 / 28 天：双周和月周期
- 56 / 91 天：更长趋势
- 182 / 364 / 365 / 728 天：半年 / 年周期 / 两年周期

rolling 主要是做平滑：

- `sales_roll_mean_7`
- `sales_roll_mean_28`
- `sales_roll_mean_56`
- `sales_roll_std_7`
- `sales_roll_std_28`

这类特征比原始 lag 更稳定，能缓解单日异常值的干扰。

#### transactions lag / rolling

交易量特征最后保留了：

- `transactions_lag_1`
- `transactions_lag_7`
- `transactions_lag_14`
- `transactions_lag_28`

以及相应 rolling 特征。

这样做的原因是：交易量更像“门店活跃度”的代理变量，短期波动对销量预测非常重要。

#### promotion 特征

促销不是简单的 0/1，而是要看它的时间延续性，因此加入：

- promotion lag
- promotion rolling

这样模型可以看到“连续促销”和“短促销”的差异。

#### oil 特征

油价不是单纯填补缺失就结束，而是要形成连续可学习的序列：

- 缺失值插值 / 回填
- `dcoilwtico_raw`
- `dcoilwtico` 变化差分
- `oil_diff_1`
- `oil_pct_change_1`
- `oil_ma_7`
- `oil_ma_28`

油价会影响消费成本、交通成本和整体需求预期，因此它是一个典型的宏观辅助信号。

#### 节假日 / 事件特征

节假日不是只有“是不是假日”这么简单，还要区分：

- national
- regional
- local
- event
- transferred holiday
- workday / bridge day

其中一些后来被证明贡献不大，被列入低重要性特征并在最终版本中删除：

- `holiday_cnt`
- `holiday_any`
- `holiday_regional`
- `holiday_flag`
- `additional_flag`
- `bridge_flag`
- `workday_flag`
- `year`
- `quarter`
- `is_month_start`
- `is_month_end`

这一步的原则是：不是所有看起来合理的特征都值得留下，重要性低且稳定性差的特征要主动砍掉。

#### 发薪日特征

发薪日效应是这个题目里很强的业务信号。我的做法不是简单加一个日期标签，而是把它当作“需求峰值触发条件”来处理：

- 月中发薪
- 月末发薪
- 发薪日前后几天的邻域效应

这样模型能学到“发薪日前后销量会抬头”的规律。

#### 地震特征

地震不是常规周期，而是一次结构性冲击。

因此我没有把它当作普通节假日，而是单独作为特殊事件处理，让模型知道这类日期不属于正常季节性。

### 4.4 zero forecasting

这个规则的逻辑很简单：

- 如果某个 `store-family` 历史上长期总销量就是 0
- 那它在测试集里如果也没有促销驱动，就直接压成 0

这样做的好处是：

- 减少无意义的正预测
- 降低模型在零销量样本上的噪声
- 缩短一部分递推路径的误差传播

我还试过一个更激进的 recent-zero 版本：

- 如果最近 21 天全零，也直接视为 zero candidate

这个版本能提升本地验证，但 public 不一定同步提升，所以最终没有作为最后答案。

### 4.5 模型层：LightGBM + XGBoost

主模型始终保持为树模型路线：

- LightGBM 擅长处理大量表格特征
- XGBoost 在某些分布下更稳

我采用的不是“只跑一个模型”，而是：

- 多随机种子
- 多模型类型
- 递推结果再做 blend

这能降低单次训练的偶然性。

### 4.6 校准层

在验证集上，我额外做了轻量后处理：

- `global`
- `family`
- `store`
- `store_family`
- `family_store`

做法是把预测和真实值之间的残差放在 log 空间里估计，再把残差作为校准项叠回预测。

最后选出的策略是：

- `store_family`

这说明最稳定的误差修正粒度不是全局，也不是单门店，而是门店 + family 的组合层。

### 4.7 blend / stack 的思路

我没有把融合理解成“平均一下就行”，而是分成三层：

1. **模型内 blend**
   - LightGBM / XGBoost / 多种随机种子
2. **验证集 blend**
   - 用验证集搜索不同模型权重
3. **submission-level blend**
   - 选两个或多个最接近但又不完全一致的预测，在 log 空间做低权重融合

最后一步之所以有效，是因为：

- 统一模型很容易把某些 family 的系统偏差带到整条提交里
- 一个分布略有差异的参考预测，能对某些偏移做轻微纠正
- log-space 融合比线性平均更适合 RMSLE 目标

最终提交就是在这种思路下产生的。

---

## 5. 为什么一些方法没有成为最终版本

项目中我尝试过不少方向，但不是每一个都保留到了最后。

### 保留下来的

- 递推式预测
- sales / transactions 分离
- lag / rolling / EWM
- 油价与节假日特征
- zero forecasting
- LightGBM + XGBoost
- 校准层
- submission-level log blend

### 没有保留的

- 过重的长历史训练窗口
- 过多的零规则扩展
- 过强的单模型递推版本
- 只看本地验证却明显偏离 public 的权重

原因很简单：本项目最后追求的不是某一次本地分数最漂亮，而是公榜表现最稳定。

---

## 6. 最终结果

最终采用的是：

- 一个稳定的内部 anchor 提交
- 一个分布互补的参考提交
- 在 log 空间做低权重融合

最终提交文件：

```text
submissions/final_submission.csv
```

最终 Public Score：

```text
0.39969
```

这个结果不是靠单一技巧“暴力拉上去”的，而是靠一整条稳定路线把误差一点一点压下来的。

---

## 7. 文件作用

### 保留的关键文件

- `README.md`  
  项目简介、结果、结构和运行方式。

- `PROJECT_REPORT_CN.md`  
  详细复盘，记录从 baseline 到最终提交的完整思路。

- `scripts/blend_submissions.py`  
  提交文件之间的线性 / log / sqrt 融合工具。

- `submissions/blend_anchor.csv`  
  最终融合的 anchor 文件。

- `submissions/reference_submission.csv`  
  作为低权重参考的辅助预测文件。

- `submissions/final_submission.csv`  
  最终提交文件。

### 作为历史代码保留的目录

- `src/`  
  里面保存了项目从 baseline、递推、特征工程到中间 ensemble 方案的代码演化痕迹。即使最终提交不再依赖其中每一行代码，它仍然记录了整个项目从无到有的过程。

---

## 8. 经验总结

这次项目最重要的收获有三点：

1. **时间序列题的核心不是“拟合一次”，而是“递推稳定”**
2. **强业务特征的收益通常大于继续堆复杂模型**
3. **最终提升往往来自 submission-level 的整体修正，而不是单一模型内部的最后 0.01**

如果以后继续做这类题，我会优先做三件事：

- 先把递推流程跑稳
- 再把强业务信号补齐
- 最后再讨论 blend / stack

这套顺序比一开始就追求复杂模型更可靠。

