# EXP-018：对侧半球参照与细节回注

训练提交 `0570fef` · 运行目录 `brats2020_contralateral_1124_1755` · [TASK-009](../../../handoff/task-009-contralateral-full.md) · [IDEA-011](../../ideas/idea-011-contralateral-reference.md)

## 目的与协议

检验在 EXP-006 的半分辨率解码特征加入对侧参照能否提高验证集 Mean Dice。遵循[统一协议](../../../docs/protocol.md)，以 seed 42 从相同 VMamba 预训练独立初始化。A 端曾接触 EXP-006 的 FINAL 结果，本轮设计与比较使用 train 和 val。

## 验证结果

完成 100 轮，第 75 轮取得 val best。EXP-018 的 WT、TC、ET Dice 分别为 90.72%、82.14%、76.56%，Mean Dice 为 83.14%，Mean HD95 为 13.12 mm。[收集结果](summary.json) · [总表](../index.md)

相对 EXP-006，WT、TC、ET Dice 分别变化 −0.17、+0.03、+0.40 个百分点，Mean Dice 增加 0.088 个百分点。Mean HD95 下降 3.84 mm，三个分区均下降，以 ET 的 7.19 mm 降幅最大。相对原始 VM-UNet（EXP-001），Mean Dice 增加 1.35 个百分点，Mean HD95 下降 7.90 mm。[配对汇总](comparison-summary.json) · [EXP-006 汇总](../EXP-006/summary.json)

## 逐病例观察

- 三组的 55 个 val 病例及分区真值体素数一致。相对 EXP-006，50 例非空 ET 的配对 Mean Dice 平均变化为 −0.018 个百分点（27 例上升，23 例下降），5 例空 ET 平均上升 1.146 个百分点。两组对 55 例整体均值的贡献分别为 −0.016 和 +0.104 个百分点。[配对汇总](comparison-summary.json)
- 三组空 ET 误检均为 2/5，病例为 272 和 281。相对 EXP-006，两例的 ET 误检体素数分别从 694 降至 409、从 7,120 降至 3,964。[EXP-006 逐病例结果](../EXP-006/val_per_patient.csv) · [EXP-018 逐病例结果](val_per_patient.csv)
- 相对 EXP-006，病例 250、148、299 的 Mean Dice 降幅最大，分别为 5.84、5.72、4.81 个百分点。病例 250 主要下降在 TC，148 在 TC 和 ET 均下降，299 主要下降在 ET。[配对汇总](comparison-summary.json) · [EXP-006 逐病例结果](../EXP-006/val_per_patient.csv) · [EXP-018 逐病例结果](val_per_patient.csv)

## 改善集中在哪些病例

固定两组各自的最佳检查点，55 例中有 34 例的三分区平均 HD95 下降、16 例上升、5 例持平。患者配对差的中位数为 −0.11 mm，均值为 −3.84 mm。降幅最大的四例（298、232、148、268）合计贡献净降幅的 89.1%，其余 51 例平均下降 0.45 mm。正式指标保留全部病例。[验证复核](validation-audit.json)

病例 298 的 ET 真值只有 94 个体素。EXP-006 的预测为空，按协议计入 373.13 mm 的空预测惩罚。EXP-018 预测 18 个体素，命中 11 个，ET Dice 为 19.64%，HD95 为 23.16 mm。该例贡献整体 Mean HD95 净降幅的 54.9%，也贡献全组 ET Dice 增量中的 0.357 个百分点（总增量 0.398）。其余 54 例的 Mean HD95 平均下降 1.77 mm。[验证复核](validation-audit.json) · [评价协议](../../../docs/protocol.md)

对固定检查点做 20,000 次逐患者配对 bootstrap，Mean Dice 差的 95% 百分位区间为 [−0.52, +0.76] 个百分点，Mean HD95 差为 [−8.92, −0.70] mm。区间描述这批验证患者的抽样波动，选轮和重复训练带来的不确定性仍待独立评估。[复核脚本](../../../VM-UNet-main/analysis/analyze-contralateral-results.py)

## 同轮次与镜像假设复核

- 对齐两组的 21 次验证，EXP-018 有 18 次 Mean HD95 更低。第 60–100 轮的 9 次验证全部更低，病例 298 的 ET 在 EXP-018 中每次都有命中，在 EXP-006 中每次 Dice 为零。扣除该例后，9 次中有 5 次 HD95 更低，平均差为 −0.57 mm。同一期间 Mean Dice 有 5 次领先、4 次落后，平均差为 −0.008 个百分点。这些轮次共享患者和训练轨迹。[各轮配对记录](validation-audit.json)
- 按各病例的解剖左右轴翻转 WT 真值，计算镜像后仍落入 WT 的体素占原 WT 的比例。该比例与患者 Mean Dice 差、Mean HD95 差的 Spearman 秩相关分别是 −0.105、+0.101。镜像重叠越高、收益越弱的趋势很弱，单凭该比例难以解释病例差异。对侧参照本身的贡献仍需同容量原图消融区分。[逐病例复核](validation-audit.json)

## 结论与建议

这次单种子验证中，对侧参照模型在多个轮次取得较低的 Mean HD95，净降幅主要来自少数大误差病例。Mean Dice 增益低于 +0.30 个百分点，非空 ET 组配对均值低于零，两个判据均未达到。同容量原图消融是否开展，交用户裁决。[TASK-009 判据](../../../handoff/task-009-contralateral-full.md#通过标准与后续)
