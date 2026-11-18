# STATE
覆盖式更新，不超过 40 行。最后更新：2026-11-18，B。

## 协议与基线

- **工作分支：** `stage003-detail-stacking`；两端共用该分支。
- **用户要求：** 仅明确要求时提交和推送。使用 `vm_unet` 和 RTX 6000（GPU 2–5），不使用 A100；单种子、单次训练。
- **协议：** [docs/protocol.md](docs/protocol.md)、`VM-UNet-main/configs/brats2020.json`；预训练初始化、100 轮、单卡 batch 16；不使用相邻切片；方法可附加训练期辅助损失（写在方法文档里）。
- **对照：** baseline 为 [EXP-001](research/experiments/EXP-001/report.md)。结果见 [总表](research/experiments/index.md)。
- **正式创新点：** [IDEA-001](research/ideas/IDEA-001-detail-fusion.md) detail gated，2026-11-15 采纳；代表实验 [EXP-006](research/experiments/EXP-006/report.md)（val best 第 95 轮，已做过一次 FINAL）。后续优化以 EXP-006 为基座和对照。

## 当前任务

- **[TASK-005](handoff/task-005-detail-options-full-training.md)：** 4 项训练已正常结束，已按 val best 收集并完成逐患者配对、报告与回执，待用户裁决；检查结果与运行目录见任务回执。
- **结果：** [EXP-011](research/experiments/EXP-011/report.md)、[EXP-012](research/experiments/EXP-012/report.md)、[EXP-013](research/experiments/EXP-013/report.md) 均未通过任务门槛；[EXP-014](research/experiments/EXP-014/report.md) 的 concat 低于 gated，门控达到任务贡献门槛。建议保留 EXP-006，限定单种子结论并披露空 ET 病例影响，不调参重跑；未自动修改 IDEA 状态或 decisions。

## 交接

- **训练提交：** `bb68b586df9903fb222fc6015a575e481690c07b`，已拉取最新分支；未改模型、损失、数据处理或协议。
- **交接内容：** STATE、TASK-005 回执、EXP-011～014 的收集文件和报告、实验总表、新增分析源码 `VM-UNet-main/analysis/compare-detail-options.py`；用户已明确授权本轮提交和推送。
- **复现分析：** 源码目录运行 `python -B analysis/compare-detail-options.py`，只读 val；产物 `VM-UNet-main/results/analysis/task-005/comparison-summary.json` 与各 EXP 子目录的配对 JSON/CSV，包含输入 SHA256、四条件判断和空 ET 贡献分解。

## 测试集约束

A 已接触测试集。此后在 BraTS 上提出的优化 IDEA 须注明这一点，且不得以测试集现象为动机；优化后的最终模型只测一次 test。

## 训练状态

- 无本任务训练在运行。GPU 3 串行队列于 2026-11-18 00:18（上海）完成全部四项；EXP-011/013 为协议内早停，非异常中断，EXP-012/014 完成 100 轮。
- **运行凭据：** `VM-UNet-main/logs/task-005-queue.{py,log,json}`；检查日志 `task-005-checks-{cpu,cuda,distributed}.log`、smoke 日志 `task-005-smoke-*.log`（同目录）。未使用 A100，未执行 FINAL。

## 待决事项与下一步

- **用户：** 按 TASK-005 回执确认是否保留 EXP-006 及门控的论文表述；尚未批准新的 FINAL。
- **B：** 本轮分析源码与结果文档随回执入库，等待最终方法和论文表述裁决。
- **A：** 补齐 IDEA-001 动机统计脚本（ET 边界占比、块常数上限），入库 `analysis/`，只用 train/val；TASK-005 完成后按结果整理最终方法与消融表。
