# EXP-007：细节回注叠加筛选——热启动对照

任务：[TASK-004](../../../handoff/archive/task-004-detail-stacking.md) · 基座：IDEA-001 · 无叠加件
运行目录：`screening-stack-control_1114_1954` · 训练提交：`18498f9b7e333c7dd8e1fb7b15f6a19873160fb3`

## 目的

测量 EXP-006 best 在相同微调安排下的对照结果。

## 与协议的差异

采用协议中的热启动筛选安排：EXP-006 第 95 轮 best 初始化、微调 10 轮、全部参数 lr=1e-5，固定评价第 10 轮。不增加新部件。
其余使用共同协议；结果仅用于筛选，不作为完整训练的论文指标。使用 GPU 2。

## 结果（仅 val）

依据 [summary.json](summary.json)、[逐患者 CSV](val_per_patient.csv) 和 [总表](../index.md)。
固定第 10 轮：Mean Dice 83.09%，ET Dice 76.25%，Mean HD95 17.10 mm，空 ET 假阳性 2/5。

## 逐患者观察

- 三个候选均未满足任务单的全部筛选条件；具体差值和配对观察见各候选报告。
- 作为三个候选的唯一筛选对照，不能用 EXP-006 原 best 替代该对照。
- 脚本：[compare-stacking.py](../../../VM-UNet-main/analysis/compare-stacking.py)，复用 `compare-screening.py` 核验；输入含 55 个不同的 val 患者、公共配置及指标一致性均通过。
- 复现：`python analysis/compare-stacking.py`（源码目录）；配对 CSV 和四项判定见 `results/analysis/task-004/comparison-summary.json` 及同目录 CSV。

## 结论与建议

对照实验完成；按 TASK-004 的“都未通过”分支，不启动第二阶段完整训练。
不执行 FINAL；IDEA 的采纳或放弃由用户决定。
