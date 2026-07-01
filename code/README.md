# Core Runtime Only

这个目录现在只保留项目运行核心，不再包含任何数据整理、统计分析、论文作图代码。

## 保留内容

- 配置采样
- `hebo` / `bo` / `tpe`
- `bestconfig` / `promisetune` / `flash`
- 最小运行入口

## 目录

- `run_core.py`
  最小入口，只分发上述 6 类调优器
- `config_sampling.py`
  独立的配置采样工具，保留随机采样、LHS、DDS
- `tuner/`
  6 个调优器源码，以及依赖感知公共基类
- `utils/`
  运行这些调优器需要的最小辅助代码快照

## 最小运行方式

```bash
python Deptuner/code/run_core.py \
  --config path/to/config.ini \
  --db_host 127.0.0.1 \
  --fidelity_type single_fidelity \
  --tuning_method hebo \
  --run 1
```

## 支持的方法

- `hebo`
- `bo`
- `tpe`
- `flash`
- `bestconfig`
- `promisetune`
- `promise`

## 说明

- 这里不再保留任何结果汇总、采样后评估数据导出、排名、ablation、表图渲染脚本。
- `tuner/` 下保留的是源码快照，运行入口 `run_core.py` 直接调用项目里的核心实现。
