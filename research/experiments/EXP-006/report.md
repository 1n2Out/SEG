# EXP-006：detail gated 完整训练

任务：[TASK-002](../../../handoff/archive/task-002-detail-gated-full-training.md) · 想法：IDEA-001
运行目录：`brats2020_detail_gated_1113_1530` · 训练源码提交：`4ede60987d3b78367ab5a55b2ebba03ee17db923`

## 目的

按完整协议验证 detail gated 相对 EXP-001 的收益；不把热启动筛选结果当作完整训练结果。

## 与协议的差异

无。VMamba 预训练初始化、100 轮、单卡 batch 16、seed 42；相对 baseline 仅新增 detail_fusion.gated。
源码来源闸门通过；训练时未提交的内容仅为交接文档，模型、损失和协议均未改动。

## 结果（仅 val）

以 [summary.json](summary.json)、[总表](../index.md)和逐患者 CSV 为准；按 val Mean Dice 选中的 best 为第 95 轮。
Mean Dice 为 83.05%，相对 EXP-001 +1.27 个百分点；ET Dice +2.44 个百分点，Mean HD95 -4.06 mm。
空 ET 假阳性保持 2/5，且仍是同两例；TASK-002 四项验证集条件全部满足。

## 逐患者与训练曲线

- 43/55 例 Mean Dice 上升，配对差中位数 +1.04 个百分点；固定模型的患者 bootstrap 95% 区间为 [0.47, 2.05] 个百分点。
- 非空 ET 的平均 Dice +2.68 个百分点；39 例上升、8 例下降、3 例持平，增益不是由空 ET 规则变化产生。
- 原 100 轮轨迹的前 60 轮最佳为 82.25%，到最终 best 另增 0.80 个百分点；这不是独立的 60 轮余弦调度实验，不能据此直接改协议。
- 分析：[compare-full-training.py](../../../VM-UNet-main/analysis/compare-full-training.py)；默认命令 `python -B analysis/compare-full-training.py` 仅分析 val，产物在 `results/analysis/exp-006/`。
- [训练与验证曲线](../../../VM-UNet-main/results/analysis/exp-006/curves/training_curves.png)：`python -B plot_brats.py curves --work-dirs results/brats2020_1102_1734 results/brats2020_detail_gated_1113_1530 --names Baseline Detail-Gated --formats png --output-dir results/analysis/exp-006/curves`。

## 结论与建议

完整训练在验证集上有改善，单 seed 不能证明跨种子稳定性；val 用于选 best，上述患者重采样区间仅作描述。
可以保留该完整训练结果；门控的独立贡献仍需完整 concat 对照，由用户另行决定，不自动开展新训练。
用户授权的一次 FINAL 测试单独记录于 [final/report.md](final/report.md)，不进入开发分析或实验总表。
