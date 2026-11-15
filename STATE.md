# STATE
覆盖式更新，不超过 40 行。最后更新：2026-11-15，B。

## 协议与基线

- **工作分支：** `stage003-detail-stacking`；两端共用该分支。
- **用户要求：** 仅明确要求时提交和推送。使用 `vm_unet` 和 RTX 6000（GPU 2–5），不使用 A100；单种子、单次训练。
- **协议：** [docs/protocol.md](docs/protocol.md)、`VM-UNet-main/configs/brats2020.json`；预训练初始化、100 轮、单卡 batch 16；允许细节支路读取上下各 1 张相邻切片。
- **对照：** baseline 为 [EXP-001](research/experiments/EXP-001/report.md)，叠加基座为 [EXP-006](research/experiments/EXP-006/report.md)（val best 第 95 轮）。结果见 [总表](research/experiments/index.md)。

## 任务结果

- **[TASK-004](handoff/archive/task-004-detail-stacking.md) 已完成并归档：** 四组筛选全部完成 10 轮，已固定收集第 10 轮；三个候选均未通过 Mean Dice 增益与 ET 上升门槛，按决策表不启动完整训练。
- **报告：** [EXP-007 对照](research/experiments/EXP-007/report.md)、[EXP-008 边界监督](research/experiments/EXP-008/report.md)、[EXP-009 相邻切片](research/experiments/EXP-009/report.md)、[EXP-010 两者叠加](research/experiments/EXP-010/report.md)。
- **可追溯性：** 四组训练提交均为 `18498f9`，源码及配置干净；运行目录见各 `summary.json`，同名日志和 PID 文件在 `VM-UNet-main/logs/`。
- **配对分析：** `VM-UNet-main/analysis/compare-stacking.py`，产物在 `VM-UNet-main/results/analysis/task-004/`；55 例患者、公共配置与 CSV 指标一致性已核验。
- **工程检查：** CPU 78 passed、CUDA 4 passed、双进程 2 passed，无 skipped；组合与对照 smoke 通过，来源 epoch 95，new=13/0。
- **限制：** GPU 4/5 有其他用户进程，耗时不作严格性能比较；未进行重复种子、调参重跑或 FINAL。
- **交付范围：** 分析脚本、收集产物、报告、链接和回执随本次交接提交；IDEA-005/006 状态未变更。
- **IDEA-004：** 已放弃，依据见 [IDEA-004](research/ideas/IDEA-004-et-postprocess.md)；主表一律使用原始输出。

## 训练中

无。EXP-007～010 均已结束；未创建 EXP-011 或后续完整训练。

## 待决事项与下一步

- **用户：** 决定是否继续探索新的叠加件；本轮筛选不足以支持边界监督或相邻切片的完整训练。
- **A：** 阅读 TASK-004 回执及 EXP-008～010 报告；如需新方案，由用户决定后另发任务。
- **B：** 等待下一项授权任务；不修改协议、IDEA 状态或执行测试集评价。
