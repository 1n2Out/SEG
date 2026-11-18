# EXP-011：完整训练：gated + 深监督

任务：[TASK-005](../../../handoff/archive/task-005-detail-options-full-training.md) · 想法：IDEA-001、IDEA-007
运行目录：`brats2020_detail_gated_ds_1115_1955` · 训练提交：`bb68b586df9903fb222fc6015a575e481690c07b`

## 目的

检验 gated + 深监督 相对 [EXP-006](../EXP-006/report.md) 的验证集表现。

## 与协议的差异

无。VMamba 预训练初始化、最多 100 轮、单卡 batch 16、seed 42；按协议于第 90 轮早停，按 val best 选择第 90 轮。
训练在 RTX 6000 GPU 3 串行执行；源码与配置无未提交改动，run.json 中的 dirty 如为 true，仅来自交接文档。

## 结果（仅 val）

以 [summary.json](summary.json)、[总表](../index.md)和 [val CSV](val_per_patient.csv) 为准。Mean Dice 83.02%，相对 EXP-006 -0.03 个百分点；ET Dice +0.97 个百分点。
Mean HD95 15.47 mm（差值 -1.49 mm，不作通过门槛）；空 ET 假阳性 2/5。

## 逐患者观察

- Mean Dice：35 例上升、20 例下降、0 例持平；配对差中位数 +0.24 个百分点。
- 非空 ET：30 例上升、18 例下降、2 例持平；平均 Dice 差 +1.07 个百分点。
- ET 的改善被 TC Dice -1.05 个百分点抵消，整体 Mean Dice 未增加；空 ET 假阳性病例与 EXP-006 相同。
- 分析：[compare-detail-options.py](../../../VM-UNet-main/analysis/compare-detail-options.py)；[配对 JSON](../../../VM-UNet-main/results/analysis/task-005/exp-011/comparison-summary.json) 与同目录 `val-paired.csv`。

## 结论与建议

四项条件依次为：Mean Dice 至少 +0.3 个百分点（未通过）、ET Dice 不下降（通过）、非空 ET 上升多于下降（通过）、空 ET 假阳性不增加（通过）。
候选未通过。按 TASK-005 建议保留 EXP-006，不调参重跑；本结果不等于证明该选项在其他种子或任务上无效。
未执行 FINAL，未修改 IDEA 状态或用户决定。
