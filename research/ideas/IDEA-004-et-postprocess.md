# IDEA-004：ET 小体积后处理

状态：**放弃**（2026-11-14） · 属于评价端的方法

## 动机

按本地的空区域规则，空 ET 患者只要被预测出任何 ET，该患者的 ET Dice 就记 0、HD95 记 373 mm。一例这样的误判会让 Mean Dice 下降约 0.6 个百分点。

EXP-001 在 val 上有 2/5 例空 ET 患者被误判出 ET（见 [summary.json](../experiments/EXP-001/summary.json) 中的 `et_empty_false_positive`）。

## 核心思想

如果一名患者预测出的 ET 体素数少于阈值 T，就把这些 ET 全部改为坏死（标签 1）。这样 WT 和 TC 不受影响，只有 ET 变化。

## 依据

- **No New-Net** 和 **nnU-Net（BraTS 2020）** 都用过这个做法，各队伍用过的阈值有 50、200、300 体素。
- **局限：** nnU-Net 的作者也指出，这样做是在迎合 BraTS 的评价规则。在需要发现微小强化灶的临床场景中，未必合适。

## 验证计划

- **不需要训练，也不需要重新推理：** `val_per_patient.csv` 里已经有预测和真值的 ET 体素数，可以直接在 val 上扫描阈值 T。
- **阈值只在 val 上选定**，test 只在 FINAL 阶段用同一个阈值报告一次。
- **注意误伤：** val 中有 2 例真实 ET 小于 200 体素，阈值过大会把它们误伤。
- 脚本：[et-postprocess-scan.py](../../VM-UNet-main/analysis/et-postprocess-scan.py)，运行 `python -B analysis/et-postprocess-scan.py`，产物在 `results/analysis/et-postprocess/scan.csv`。

## 结论

只在 val 上扫描了 EXP-001 和 EXP-006（阈值 0–2000 体素）：
- 两者的空 ET 假阳性都是体积较大的预测，不是零星小点；EXP-006 在 2000 体素时仍保留 1 例。
- 能去掉假阳性的阈值同时会误删真实的小 ET：EXP-006 上 Mean Dice 最多 +0.13 个百分点，代价是 3 例非空 ET 被清空、Mean HD95 变差。
- 统一施加时 baseline 获益更多（EXP-001 在 1500 体素时 +1.00，同样误删 2 例），反而缩小方法间的差距。
- 收益不稳定，且所需阈值远大于文献的 50–300 体素。因此放弃按患者 ET 总体积的后处理（见 [decisions](../decisions.md) 2026-11-14）。按连通域或 ET 置信度过滤的变体需要重新推理，未评估。
