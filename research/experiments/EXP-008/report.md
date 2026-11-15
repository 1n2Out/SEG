# EXP-008：细节回注叠加筛选——边界辅助监督

任务：[TASK-004](../../../handoff/archive/task-004-detail-stacking.md) · 基座：IDEA-001 · 叠加件：IDEA-005
运行目录：`screening-stack-boundary_1114_1954` · 训练提交：`18498f9b7e333c7dd8e1fb7b15f6a19873160fb3`

## 目的

边界监督能否在 detail gated 上进一步改善分割。

## 与协议的差异

采用协议中的热启动筛选安排：EXP-006 第 95 轮 best 初始化、微调 10 轮、已有参数 lr=1e-5、新参数 lr=1e-4，固定评价第 10 轮。边界辅助损失权重为 0.5。
其余使用共同协议；结果仅用于筛选，不作为完整训练的论文指标。使用 GPU 3。

## 结果（仅 val）

依据 [summary.json](summary.json)、[逐患者 CSV](val_per_patient.csv) 和 [总表](../index.md)。
相对 EXP-007：Mean Dice +0.009 个百分点，ET Dice -0.418 个百分点，Mean HD95 -1.039 mm；空 ET 假阳性保持 2/5。

## 逐患者观察

- 55 例患者逐例配对：Mean Dice 上升 15 例、下降 40 例；中位差 -0.259 个百分点。空 ET 假阳性仍为对照的同两例。
- Mean HD95 改善，但 ET 和非空 ET Dice 都下降；不能据此宣称整体提升。
- 脚本：[compare-stacking.py](../../../VM-UNet-main/analysis/compare-stacking.py)，复用 `compare-screening.py` 核验；输入含 55 个不同的 val 患者、公共配置及指标一致性均通过。
- 复现：`python analysis/compare-stacking.py`（源码目录）；配对 CSV 和四项判定见 `results/analysis/task-004/comparison-summary.json` 及同目录 CSV。

## 结论与建议

未通过：Mean Dice 未达到至少 +0.5 个百分点，且 ET Dice 下降。按 TASK-004 决策表不进入完整训练，不调参重跑。
不执行 FINAL；IDEA 的采纳或放弃由用户决定。
