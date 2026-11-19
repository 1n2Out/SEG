# EXP-015：ET 常数回调筛选

任务 [TASK-007](../../../handoff/archive/task-007-et-rollback-screening.md) · 想法 [IDEA-009](../../ideas/idea-009-et-rollback.md)

运行目录 `brats2020_et_rollback_constant_screen_1118_1559` · 训练提交 `8d94b9d` · RTX 6000 GPU 2

## 目的

检验共享的可学习回调系数能否恢复 EXP-006 被细节残差误删的 ET，并为局部回调提供对照。

## 与协议的差异

热启动加载 EXP-006 第 95 轮 best，冻结原模型且保持 eval。仅训练常数回调，初始学习率为 `1e-4`，固定评价第 10 轮 val。配置和训练曲线见 [config.json](config.json) 与 [history.jsonl](history.jsonl)。

## 结果与结论

- 固定第 10 轮的 55 例逐患者指标与 EXP-006 一致。50 例非空 ET 的 Dice 全部持平，恢复真 ET 和新增假阳性体素均为 0。Mean HD95 持平。见 [实验摘要](summary.json)、[逐患者 CSV](val_per_patient.csv) 和 [配对汇总](../EXP-016/comparison-summary.json)。
- 第 10 轮 checkpoint 的共享系数为负，推理时截断到 0。此模型在本轮验证中没有产生有效回调。
- Mean Dice 和非空 ET 逐例收益未达筛选门槛。按任务单保留 EXP-006。
