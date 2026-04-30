# Baseline 项目目录清单

## 根目录
```text
D:\ML-Kaggle\
├─ data\
├─ src\
├─ scripts\
├─ notebooks\
├─ outputs\
├─ submissions\
└─ plan.md
```

## 目录说明

### `data/`
原始比赛数据统一放这里，保持不修改或少修改。

建议包含：
- `train.csv`
- `test.csv`
- `stores.csv`
- `oil.csv`
- `holidays_events.csv`
- `transactions.csv`
- `sample_submission.csv`

### `src/`
放核心代码。

建议拆分为：
- `src/data/`：读取、清洗、合并数据
- `src/features/`：特征工程
- `src/models/`：模型训练与预测
- `src/utils/`：工具函数、指标、保存加载

### `scripts/`
放一键运行脚本。

建议文件：
- `run_baseline.py`
- `run_train.py`
- `run_predict.py`

### `notebooks/`
放探索性分析和临时实验。

建议用途：
- 数据检查
- 特征可视化
- 模型试验记录

### `outputs/`
放所有中间结果和实验产物。

建议子目录：
- `outputs/figures/`
- `outputs/models/`
- `outputs/predictions/`

### `submissions/`
放最终提交到 Kaggle 的预测文件。

### `plan.md`
项目路线和分工说明。

## Baseline 开发顺序
1. 读取数据并检查字段
2. 构造基础时间特征
3. 做时间切分验证
4. 训练 LightGBM baseline
5. 生成提交文件
6. 再加入 lag、rolling、节假日、油价等特征

## 约定
- 原始数据放 `data/`
- 代码放 `src/` 和 `scripts/`
- 图表和模型放 `outputs/`
- 提交文件放 `submissions/`
- 不把实验产物和原始数据混在一起
