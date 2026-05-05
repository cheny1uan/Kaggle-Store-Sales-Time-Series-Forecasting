# Store Sales Time Series Forecasting

这是 Kaggle 比赛 `Store Sales - Time Series Forecasting` 的最终整理版仓库。
当前仓库只保留可复现的最终结果、核心脚本和说明文档，文件名也已经统一成更标准的命名。

## 最终结果

| 项目 | 文件 | Public Score | 说明 |
|---|---|---:|---|
| Final | `submissions/final_submission.csv` | **0.39969** | 最终提交文件 |

## 目录结构

```text
.
|-- data/                 # Kaggle 原始数据，不建议上传 GitHub
|-- notebooks/            # 参考 notebook
|-- outputs/              # 中间结果、权重报告、日志
|-- scripts/
|   `-- blend_submissions.py  # 生成最终 blend 的工具脚本
|-- src/                  # 保留的特征工程与建模代码
|-- submissions/
|   |-- blend_anchor.csv
|   |-- final_submission.csv
|   `-- reference_submission.csv
|-- PROJECT_REPORT_CN.md
|-- SUBMISSION_QUEUE.md
`-- README.md
```

## 如何复现最终提交

```powershell
pip install -r requirements.txt
python scripts/blend_submissions.py `
  --base submissions/blend_anchor.csv `
  --plus submissions/reference_submission.csv `
  --weights 0.08 `
  --methods log `
  --output submissions/final_submission.csv
```

## 说明

- `blend_anchor.csv` 是最终融合前的主锚点结果。
- `reference_submission.csv` 是与锚点互补的参考结果。
- `final_submission.csv` 是最终提交文件。
- `data/` 只保留本地训练数据，不建议提交到 GitHub。
- `outputs/` 只放中间产物和分析结果，清理后可以保持为空目录。
- 更详细的项目演进、方法细节和每一阶段尝试，请看 [PROJECT_REPORT_CN.md](./PROJECT_REPORT_CN.md)。
