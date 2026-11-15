# STATE
覆盖式更新，不超过 40 行。最后更新：2026-11-15，A。

## 协议与基线

- **工作分支：** `stage003-detail-stacking`；两端共用该分支。
- **用户要求：** 仅明确要求时提交和推送。使用 `vm_unet` 和 RTX 6000（GPU 2–5），不使用 A100；单种子、单次训练。
- **协议：** [docs/protocol.md](docs/protocol.md)、`VM-UNet-main/configs/brats2020.json`；预训练初始化、100 轮、单卡 batch 16；不使用相邻切片；方法可附加训练期辅助损失（写在方法文档里）。
- **对照：** baseline 为 [EXP-001](research/experiments/EXP-001/report.md)。结果见 [总表](research/experiments/index.md)。
- **正式创新点：** [IDEA-001](research/ideas/IDEA-001-detail-fusion.md) detail gated，2026-11-15 采纳；代表实验 [EXP-006](research/experiments/EXP-006/report.md)（val best 第 95 轮，已做过一次 FINAL）。后续优化以 EXP-006 为基座和对照。

## 当前任务

- **[TASK-005](handoff/task-005-detail-options-full-training.md)：** 4 个完整训练，对照 EXP-006：gated + 深监督（[IDEA-007](research/ideas/IDEA-007-deep-supervision.md)）、gated + 输入端补强（[IDEA-008](research/ideas/IDEA-008-input-stem.md)）、两者组合、concat 门控消融。已推送，待 B 执行。

## 交接

- **已推送** 到 `stage003-detail-stacking`：`146a146`（上一轮决定落档与代码清理）、`fdb7996`（TASK-005 的代码、配置、测试与文档）。
- **A 端检查：** CPU 173 passed（含 2 项双进程用例）、8 skipped（CUDA 用例，交 B 执行）。

## 测试集约束

A 已接触测试集。此后在 BraTS 上提出的优化 IDEA 须注明这一点，且不得以测试集现象为动机；优化后的最终模型只测一次 test。

## 训练中

无。

## 待决事项与下一步

- **用户：** TASK-005 完成后，按回执裁决最终方法和门控的论文表述。
- **B：** 拉取后按 TASK-005 先检查，再在 GPU 2–5 并行训练 4 个配置（单个约 13–14 h），收集、分析并填回执；不执行 FINAL，不修改 IDEA 状态。
- **A：** 补齐 IDEA-001 动机统计脚本（ET 边界占比、块常数上限），入库 `analysis/`，只用 train/val；TASK-005 完成后按结果整理最终方法与消融表。
