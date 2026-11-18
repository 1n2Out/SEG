# STATE
覆盖式更新，不超过 40 行。最后更新：2026-11-18，A。

## 协议与基线

- **工作分支：** `stage003-detail-stacking`；两端共用该分支。
- **用户要求：** 仅明确要求时提交和推送。使用 `vm_unet` 和 RTX 6000（GPU 2–5），不使用 A100；单种子、单次训练。
- **协议：** [docs/protocol.md](docs/protocol.md)、`VM-UNet-main/configs/brats2020.json`；预训练初始化、100 轮、单卡 batch 16；不使用相邻切片；不附加辅助损失。
- **对照：** baseline 为 [EXP-001](research/experiments/EXP-001/report.md)。结果见 [总表](research/experiments/index.md)。
- **正式创新点：** [IDEA-001](research/ideas/IDEA-001-detail-fusion.md) detail gated，2026-11-15 采纳；代表实验 [EXP-006](research/experiments/EXP-006/report.md)（val best 第 95 轮，已做过一次 FINAL）。门控消融为 [EXP-014](research/experiments/EXP-014/report.md)（concat）。

## 当前任务

- **[TASK-005](handoff/task-005-detail-options-full-training.md)：** B 已执行并填回执。用户已放弃 IDEA-007、IDEA-008（decisions 2026-11-18）；最终方法与门控表述待裁决后关闭并归档任务单。
- **代码清理（A，未提交）：** 删除深监督、输入端补强的代码、3 个配置、专用测试，相关源码与文档回到 `146a146` 的状态；协议去掉"可附加辅助损失"。`detail_fusion` 对两个退役字段只接受 `false`，EXP-014 的 checkpoint 仍可加载。保留 gated、concat 正式配置与 4 个筛选配置。

## 交接

- **A 端检查：** CPU 152 passed、4 skipped（CUDA 用例）；双进程与热启动 26 passed。
- **B 端已拉取的提交：** `6785299`（TASK-005 结果、EXP-011～014、`analysis/compare-detail-options.py`）。
- **B 端注意：** 清理提交推送后，EXP-011～013 的 checkpoint 不能再用当前代码加载；如需重新评价，在独立副本中检出 `6785299`。

## 测试集约束

A 已接触测试集。此后在 BraTS 上提出的优化 IDEA 须注明这一点，且不得以测试集现象为动机；优化后的最终模型只测一次 test。

## 训练状态

无训练在运行。TASK-005 运行凭据见任务回执与 `VM-UNet-main/logs/task-005-*`。

## 待决事项与下一步

- **用户：** 是否确认 EXP-006 为最终方法；门控的论文表述（gated 领先 0.83 个百分点，其中约 0.61 来自一例空 ET 假阳性）；是否提交、推送本次清理。尚未批准新的 FINAL。
- **A：** 补齐 IDEA-001 动机统计脚本（ET 边界占比、块常数上限），入库 `analysis/`，只用 train/val；按裁决整理最终方法与消融表。
