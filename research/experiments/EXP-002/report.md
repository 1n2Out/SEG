# EXP-002：baseline 热启动对照

任务：[TASK-001](../../../handoff/TASK-001-detail-fusion-screening.md) · 想法：IDEA-001
运行目录：`screening_baseline_1113_1036` · 提交：`b448503bd5894f28e6f999abcb9eacc6ff6cdfd6`

## 目的

检验继续微调本身的影响，给新模块提供同起点、同轮数的公平对照。

## 与协议的差异

采用 [protocol](../../../docs/protocol.md) 中的热启动筛选：从 EXP-001 第 55 轮 best 加载全部原参数，重置优化器微调 10 轮，原参数 lr 1e-5；不挂新模块。
固定使用第 10 轮比较，不以本次 best 替代。

## 结果

数值来源：[summary.json](summary.json)、[验证集总表](../index.md)及对应的 `val_per_patient.csv`。
第 10 轮 val Mean Dice 为 81.20%；相对原 EXP-001 为 -0.58 个百分点。
空 ET 假阳性为 3/5，原 EXP-001 为 2/5；继续微调没有带来整体收益。

## 观察

分析脚本：[compare-screening.py](../../../VM-UNet-main/analysis/compare-screening.py)，运行：`python -B analysis/compare-screening.py`（源码目录）。
本地产物：`results/analysis/task-001/comparison-summary.json` 和五份逐患者配对 CSV；只使用 val。

原 checkpoint 与热启动对照的表现不同，因此模块筛选必须比较 EXP-002；不能把相对热启动对照的增益等同于相对原 baseline 的增益。
主候选相对本对照的分析见 [EXP-003](../EXP-003/report.md)。

## 结论与建议

对照组已完成，供 TASK-001 的第 10 轮配对比较使用。这些数字只作热启动筛选，不替代完整协议训练。
