# STATE
覆盖式更新，不超过 40 行。最后更新：2026-11-15，A。

## 协议与基线

- **工作分支：** `stage003-detail-stacking`；两端共用该分支。
- **用户要求：** 仅明确要求时提交和推送。使用 `vm_unet` 和 RTX 6000（GPU 2–5），不使用 A100；单种子、单次训练。
- **协议：** [docs/protocol.md](docs/protocol.md)、`VM-UNet-main/configs/brats2020.json`；预训练初始化、100 轮、单卡 batch 16；不使用相邻切片；方法可附加训练期辅助损失（写在方法文档里）。
- **对照：** baseline 为 [EXP-001](research/experiments/EXP-001/report.md)。结果见 [总表](research/experiments/index.md)。
- **正式创新点：** [IDEA-001](research/ideas/IDEA-001-detail-fusion.md) detail gated，2026-11-15 采纳；代表实验 [EXP-006](research/experiments/EXP-006/report.md)（val best 第 95 轮，已做过一次 FINAL）。后续优化以 EXP-006 为基座和对照。

## 当前任务

- **[TASK-005](handoff/task-005-detail-options-full-training.md)：** 4 个完整训练，对照 EXP-006：gated + 深监督（[IDEA-007](research/ideas/IDEA-007-deep-supervision.md)）、gated + 输入端补强（[IDEA-008](research/ideas/IDEA-008-input-stem.md)）、两者组合、concat 门控消融。A 端代码、配置、测试和文档已完成，提交推送后由 B 执行。

## 本地未推送的改动（A 端，未提交）

- **上一轮（已暂存）：** 放弃 IDEA-002/005/006、授权 A 读取 FINAL、采纳 IDEA-001 等决定落档；删除 `shallow_refinement` 扩展和 `boundary_weight`、`context_slices` 选项；文档整理。
- **本轮（未暂存）：** `detail_fusion` 新增 `deep_supervision`、`input_stem` 两个选项（默认关闭时与 EXP-006 逐位相同）、3 个正式配置和对应测试；IDEA-007/008、TASK-005、方法文档、协议、codemap、IDEA 索引同步；decisions 追加本轮两条决定。
- **检查：** A 端 CPU 171 passed、10 skipped（8 项 CUDA 用例交 B；2 项双进程用例单独开启后 2 passed）。B 端在推送前看不到以上改动。

## 测试集约束

A 已接触测试集。此后在 BraTS 上提出的优化 IDEA 须注明这一点，且不得以测试集现象为动机；优化后的最终模型只测一次 test。

## 训练中

无。

## 待决事项与下一步

- **用户：** 决定何时提交推送；B 需要拉取后才能执行 TASK-005。
- **B：** 拉取后按 TASK-005 先检查，再在 GPU 2–5 并行训练 4 个配置（单个约 13–14 h），收集、分析并填回执；不执行 FINAL，不修改 IDEA 状态。
- **A：** 补齐 IDEA-001 动机统计脚本（ET 边界占比、块常数上限），入库 `analysis/`，只用 train/val；TASK-005 完成后按结果整理最终方法与消融表。
