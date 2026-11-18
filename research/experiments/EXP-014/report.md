# EXP-014：完整训练：detail concat

任务：[TASK-005](../../../handoff/task-005-detail-options-full-training.md) · 想法：IDEA-001（门控消融）
运行目录：`brats2020_detail_concat_1115_1955` · 训练提交：`bb68b586df9903fb222fc6015a575e481690c07b`

## 目的

检验 detail concat 相对 [EXP-006](../EXP-006/report.md) 的验证集表现。

## 与协议的差异

无。VMamba 预训练初始化、最多 100 轮、单卡 batch 16、seed 42；完成 100 轮，按 val best 选择第 75 轮。
训练在 RTX 6000 GPU 3 串行执行；源码与配置无未提交改动，run.json 中的 dirty 如为 true，仅来自交接文档。

## 结果（仅 val）

以 [summary.json](summary.json)、[总表](../index.md)和 [val CSV](val_per_patient.csv) 为准。Mean Dice 82.23%，相对 EXP-006 -0.83 个百分点；ET Dice -1.60 个百分点。
Mean HD95 20.53 mm（差值 +3.57 mm，不作通过门槛）；空 ET 假阳性 3/5。

## 逐患者观察

- Mean Dice：30 例上升、25 例下降、0 例持平；配对差中位数 +0.16 个百分点。
- 非空 ET：32 例上升、15 例下降、3 例持平；平均 Dice 差 +0.25 个百分点。
- 比 EXP-006 多出空 ET 假阳性 `BraTS20_Training_330`，空 ET 部分对 gated 优势贡献 0.61 个百分点；非空 ET 平均 Dice 由 concat 略高，不能宣称门控对所有病例更好。
- 分析：[compare-detail-options.py](../../../VM-UNet-main/analysis/compare-detail-options.py)；[配对 JSON](../../../VM-UNet-main/results/analysis/task-005/exp-014/comparison-summary.json) 与同目录 `val-paired.csv`。

## 结论与建议

gated 的 Mean Dice 比 concat 高 0.83 个百分点，超过任务设定的 0.3 门槛；按本次单种子 val 消融，门控可写为贡献。
建议保留 gated，表述限定在本次协议与数据划分，并披露额外空 ET 假阳性对差距的贡献；最终表述由用户确认。
未执行 FINAL，未修改 IDEA 状态或用户决定。
