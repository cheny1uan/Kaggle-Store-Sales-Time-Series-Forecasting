# Hybrid v2 代码结构

## 入口
- `scripts/run_hybrid_v2.py`

## 核心模块
- `src/hybrid_v2/config.py`：参数配置
- `src/hybrid_v2/io.py`：数据读写
- `src/hybrid_v2/rules.py`：zero forecasting 规则
- `src/hybrid_v2/features.py`：日期、节假日、油价、发薪日、地震、Fourier 特征
- `src/hybrid_v2/trend.py`：趋势/周期底模
- `src/hybrid_v2/residual.py`：残差树模型
- `src/hybrid_v2/pipeline.py`：训练、验证、预测总流程

## 设计思路
1. 用线性趋势模型拟合长期走势和周期项
2. 用 LightGBM 拟合残差
3. 加入 zero forecasting 规则
4. 保留逐步扩展空间，后续可以再加递推预测

## 运行方法
```bash
python scripts/run_hybrid_v2.py --mode fast
python scripts/run_hybrid_v2.py --mode full
```

## 与 baseline 的关系
- baseline 继续保留，作为稳定对照组
- `Hybrid v2` 是一条新的实验线，结构更接近混合模型

