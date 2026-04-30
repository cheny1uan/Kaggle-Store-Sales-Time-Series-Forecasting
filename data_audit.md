# Store Sales 数据体检报告

项目：Kaggle - Store Sales - Time Series Forecasting  
检查时间：2026-04-30

## 1. 数据文件概览
- `train.csv`：116.16 MB
- `test.csv`：0.97 MB
- `stores.csv`：0.00 MB
- `oil.csv`：0.02 MB
- `holidays_events.csv`：0.02 MB
- `transactions.csv`：1.48 MB
- `sample_submission.csv`：0.33 MB

## 2. 各表规模
- `train`：3,000,888 行，6 列
- `test`：28,512 行，5 列
- `stores`：54 行，5 列
- `oil`：1,218 行，2 列
- `holidays_events`：350 行，6 列
- `transactions`：83,488 行，3 列
- `sample_submission`：28,512 行，2 列

## 3. 训练集检查
- 时间范围：2013-01-01 ~ 2017-08-15
- 门店数：54
- 商品家族数：33
- `sales` 最小值：0.0000
- `sales` 最大值：124717.0000
- `sales` 均值：357.7757
- `sales` 中位数：11.0000
- `sales` 为 0 的比例：31.30%
- `sales` 大于 0 的比例：68.70%
- `onpromotion` 取值很多，是一个重要特征

`sales` 分位数：
- 1%：0.0000
- 5%：0.0000
- 25%：0.0000
- 50%：11.0000
- 75%：195.8473
- 95%：1965.0000
- 99%：5507.0000

## 4. 测试集检查
- 时间范围：2017-08-16 ~ 2017-08-31
- 门店数：54
- 商品家族数：33
- `id` 唯一：是
- `onpromotion` 无缺失

## 5. 门店表检查
- 门店数：54
- 门店类型数：5
- 城市数：22
- 州/省数：16
- `cluster` 数：17

## 6. 油价表检查
- 日期范围：2013-01-01 ~ 2017-08-31
- `dcoilwtico` 最小值：26.1900
- `dcoilwtico` 最大值：110.6200
- `dcoilwtico` 缺失值：43
- 日期有少量缺口，后续需要补齐

## 7. 节假日表检查
- 记录数：350
- `type` 数：6
- `locale` 数：3
- `locale_name` 数：24
- `transferred` 取值：`False`, `True`

节假日类型分布：
- Holiday：221
- Event：56
- Additional：51
- Transfer：12
- Bridge：5
- Work Day：5

## 8. 交易表检查
- 日期范围：2013-01-01 ~ 2017-08-15
- 门店数：54
- 交易额最小值：0
- 交易额最大值：8359
- 交易额均值：1694.60

## 9. 合并关系
- `train.csv` 和 `test.csv` 可通过 `date + store_nbr + family` 对齐
- `stores.csv` 通过 `store_nbr` 合并
- `oil.csv` 通过 `date` 合并
- `holidays_events.csv` 需要先转成日期级特征
- `transactions.csv` 可按 `date + store_nbr` 合并

## 10. 体检结论
1. 这是典型的“时间序列 + 多表特征”任务。
2. `sales` 分布明显右偏，后面建模要重视 RMSLE 思路。
3. 最优先的特征是日期、节假日、油价、门店信息、滞后特征。
4. `transactions.csv` 值得加入特征工程。
5. 验证必须按时间切分，不能随机打乱。

## 11. 下一步建议
- 先做基础合并和日期特征
- 再做 `lag` / `rolling` 特征
- 先跑 `LightGBM` baseline，再考虑深度学习或融合
