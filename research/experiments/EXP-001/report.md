# EXP-001：Baseline（VM-UNet，预训练初始化）

任务：无（早于新流程） · 运行目录：`brats2020_1102_1734` · 提交：无记录（见"代码版本"）

## 目的

建立 VM-UNet 四模态 baseline 的训练与三维评价基准，作为后续所有方法的对照组。

## 与协议的差异

无。本实验的配置就是 [protocol](../../../docs/protocol.md) 的来源：预训练初始化，100 轮，单卡 batch 16，AMP。

## 代码版本

- 这次运行早于仓库的首次提交，所以 `run.json` 里没有 git 记录。
- 统一换行符之后，拿 `run.json` 中的逐文件 SHA256 与 `cc14022` 比对：
  - 数据、损失、指标、推理和模型代码完全一致；
  - 只有 `run_brats.py` 和 `train_brats.py` 不同。这两个文件后来加了热启动和扩展分派，未配置这些功能时 baseline 的行为不变。

## 结果

数值以 [summary.json](summary.json) 和 [总表](../index.md) 为准：val Mean Dice 为 81.78%，best 在第 55 轮，三个区域中 ET 最弱。

## 观察（只用 val）

- **后半程进入平台。** 第 55 轮之后的 9 次验证都没有刷新 best（见 `history.jsonl`），也没有触发早停。
- **空 ET 假阳性。**
  - 5 例空 ET 患者中有 2 例被预测出 ET。按空区域规则，每例都会让 Mean Dice 下降约 0.6 个百分点。
  - 这 2 例的 HD95 各记 373 mm，合计给 ET 的平均 HD95 贡献了约 13.6 mm。这是 ET 的 HD95 偏高的主要原因，见 [IDEA-004](../../ideas/IDEA-004-et-postprocess.md)。
- **非空 ET 的 Dice** 为 75.09%，已经接近 4×4 块常数上限的量级。这是 [IDEA-001](../../ideas/IDEA-001-detail-fusion.md) 的动机依据。

## 测试集

这次运行在封存规则建立之前，于 2026-11-03 评价过一次测试集，结果存放在 `final/`。开发期间不读取。

## 结论

作为后续所有实验的对照基准。
