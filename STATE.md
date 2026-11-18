# STATE
覆盖式更新，不超过 40 行。最后更新：2026-11-18，B。

## 协议与基线

- **工作分支：** `stage003-detail-stacking`；两端共用该分支。
- **用户要求：** 仅明确要求时提交和推送。使用 `vm_unet` 和 RTX 6000（GPU 2–5），不使用 A100；单种子、单次训练。
- **协议：** [docs/protocol.md](docs/protocol.md)、`VM-UNet-main/configs/brats2020.json`；预训练初始化、100 轮、单卡 batch 16；不使用相邻切片；不附加辅助损失。
- **对照：** baseline 为 [EXP-001](research/experiments/EXP-001/report.md)。结果见 [总表](research/experiments/index.md)。
- **正式创新点：** [IDEA-001](research/ideas/IDEA-001-detail-fusion.md) detail gated，2026-11-15 采纳；代表实验 [EXP-006](research/experiments/EXP-006/report.md)（val best 第 95 轮，已做过一次 FINAL）。门控消融为 [EXP-014](research/experiments/EXP-014/report.md)（concat）。

## 当前任务

- **[TASK-006](handoff/task-006-et-residual-audit.md)：** 已完成一次完整 val 推理、逐例复现核验、离线计数交叉核验及绘图；[报告与统计归档](research/experiments/EXP-006/et-residual-audit/report.md)。专项检查 3 passed；55 例 WT/TC/ET 体素数和 Dice 均通过。
- **TASK-006 结论：** 主要作用为 ET 误检抑制；非空组 46 升/2 降/2 平，存在小 ET 抹除反例。残差前主干与支路共同训练，差值不能解释为独立 baseline 提升；研究方向未变更。
- **[TASK-005](handoff/task-005-detail-options-full-training.md)：** B 已执行并填回执。用户已放弃 IDEA-007、IDEA-008（decisions 2026-11-18）；最终方法与门控表述待裁决后关闭并归档任务单。
- **代码清理（`fa76453`）：** 删除深监督、输入端补强的代码、配置和专用测试，协议恢复为不附加辅助损失。EXP-014 的 checkpoint 仍可加载。
- **IDEA-001 动机复核：** train/val 标签分析和 val 非空 ET 配对已完成，见 [IDEA-001](research/ideas/IDEA-001-detail-fusion.md)。

## 交接

- **B 端执行提交：** 已拉取 `295abb8a34b01ac3cd28dac3ce085267bcee69a6`；推理开始时工作区干净，EXP-006 第 95 轮 checkpoint、原配置和固定 val 划分核对通过。
- **B 端交付：** STATE、TASK-006 回执、EXP-006 下的诊断报告与统计文件、新增离线核验源码 `analysis/summarize-et-residual-audit.py`；用户已明确授权本轮提交和推送。
- **B 端注意：** 如需重新评价 EXP-011～013 的 checkpoint，在独立副本中检出 `6785299`。

## 测试集约束

A 已接触测试集。此后在 BraTS 上提出的优化 IDEA 须注明这一点，且不得以测试集现象为动机；优化后的最终模型只测一次 test。

## 训练状态

无本任务训练或推理在运行。TASK-006 用 RTX 6000 GPU 3；产物 `VM-UNet-main/results/analysis/task-006/`、日志 `VM-UNet-main/logs/task-006-*`；四张图各含 PNG/PDF/SVG。未训练、未执行 FINAL。

## 待决事项与下一步

- **用户：** 是否确认 EXP-006 为最终方法；门控的论文表述（gated 领先 0.83 个百分点，其中约 0.61 来自一例空 ET 假阳性）。尚未批准新的 FINAL。
- **A/用户：** 依据 TASK-006 核验结果限定机制表述，同时披露误检抑制与小病灶漏分；残差强度或模态判别仅作为待裁决假设。
- **B：** TASK-006 执行完毕，报告和统计归档已就绪；最终方法与消融表按用户裁决整理。
