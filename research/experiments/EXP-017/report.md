# EXP-017：全主干四方向分组融合与细节回注

[TASK-008](../../../handoff/archive/task-008-direction-full.md) · [IDEA-010](../../ideas/idea-010-direction-difference-fusion.md) · 训练提交 `acd6227` · 运行目录 `brats2020-direction-all_1123_2147`

## 目的与协议

检验在 EXP-006 的 gated 细节回注基础上，对全部 15 个 VSSBlock 加入方向融合能否提高验证集 Mean Dice。遵循[统一协议](../../../docs/protocol.md)，seed 42，从 VMamba 权重独立训练。方向融合分支使用 FP32，其余主干沿用 AMP。A 端此前已接触 EXP-006 的 FINAL 结果，本轮设计和比较使用 train/val。

## 验证结果

第 30 轮取得 val best，第 80 轮触发早停。下表中 Dice 为百分数，HD95 单位为 mm。[收集结果](summary.json) · [总表](../index.md)

| 实验 | Mean Dice | WT Dice | TC Dice | ET Dice | Mean HD95 | 空 ET 误检 |
|---|---:|---:|---:|---:|---:|---:|
| EXP-001 原 VM-UNet | 81.78 | 90.27 | 81.36 | 73.72 | 21.02 | 2/5 |
| EXP-006 细节回注 | 83.05 | 90.89 | 82.10 | 76.16 | 16.96 | 2/5 |
| EXP-017 细节回注 + 方向融合 | 81.71 | 90.17 | 80.27 | 74.68 | 15.75 | 2/5 |

55 例按病例编号配对，各分区真值体素数一致。相对 EXP-006，Mean Dice 下降 1.34 个百分点，WT、TC、ET 分别下降 0.72、1.83、1.48 个百分点。50 例非空 ET 患者中，逐病例三分区 Mean Dice 有 16 例上升、34 例下降，等权平均下降 1.50 个百分点。三组的空 ET 误检均为病例 272、281。[配对汇总](comparison-summary.json) · [分析脚本](../../../VM-UNet-main/analysis/compare-direction-fusion.py)

## 原因核查

- 对齐训练轮次后，第 30 轮 EXP-017 比 EXP-006 高 1.42 个百分点；从第 35 轮到第 80 轮，之后的 10 次验证均落后。即使两组都只选前 80 轮的最佳结果，EXP-006 仍以第 75 轮的 82.93% 领先 1.22 个百分点。这段时间 EXP-017 的训练损失仍在下降。[EXP-006 轨迹](../EXP-006/history.jsonl) · [EXP-017 轨迹](history.jsonl)
- 对有对应区域真值的患者等权平均，WT 精确率升 1.33、召回率降 2.48 个百分点；TC 精确率降 0.76、召回率降 2.76 个百分点；ET 精确率和召回率分别降 1.92 和 1.16 个百分点。病例 250、148、232 合计贡献整体 Mean Dice 降幅中的 0.68 个百分点，余下 52 例平均仍降 0.70 个百分点。[配对汇总](comparison-summary.json)
- HD95 的改善主要来自 WT。WT 平均 HD95 降 3.49 mm、TC 降 0.46 mm、ET 升 0.67 mm。WT 中病例 268 的 HD95 单例就降了 67.71 mm。[EXP-006 逐病例结果](../EXP-006/val_per_patient.csv) · [EXP-017 逐病例结果](val_per_patient.csv)
- 在最佳权重上，对全部 55 例各取轴向深度 1/4、1/2、3/4 的切片，读取每个位置、每个通道组四方向归一化权重的最大值，再取平均。浅层编码器前四块为 25.8%～26.4%（均分为 25%），最深两块为 99.44% 和 99.95%；仅统计与 WT 重叠的网格，最深两块仍为 99.82% 和 99.97%。这说明深层学到了高度集中的方向分配。该行为可能削弱多方向信息互补，单种子观察无法判定其对 Dice 的因果效应。[权重统计](direction-weights.json) · [复核脚本](https://github.com/1n2Out/SEG/blob/exp-017-direction-fusion/VM-UNet-main/analysis/inspect-direction-weights.py)

病例 250 说明了 TC 精确率和召回率为何会同时下降。该患者 TC 真值为 3,265 个体素。EXP-006 和 EXP-017 分别预测了 3,606 和 10,584 个体素，但召回率从 63.6% 降到 59.1%。EXP-017 多分出了背景或其他区域，同时漏掉了更多真实核心体素。[EXP-006 逐病例结果](../EXP-006/val_per_patient.csv) · [EXP-017 逐病例结果](val_per_patient.csv)

深层方向权重集中可能削弱互补信息，进而影响 TC 召回率，这一机制仍待验证。

## 结论与建议

EXP-017 的 Mean HD95 较 EXP-006 改善 1.22 mm，但三个分区的 Dice 均下降，降幅最大的是 TC。组合相对 EXP-001 的 Mean Dice 也低 0.08 个百分点。按 TASK-008 判据，本次单种子验证优先保留 EXP-006。研究取舍见[决策记录](../../decisions.md)。
