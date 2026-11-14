# STATE

覆盖式更新，不超过 40 行。最后更新：2026-11-14，A。

## 协议与基线

- **工作分支：** `stage003-detail-stacking`（从 `main` 新建）。两端都在该分支上工作；阶段成果出现后合并到 `main`。
- **用户要求：** 仅在用户明确要求时提交和推送。使用 `vm_unet` 和 RTX 6000（GPU 2–5），不使用 A100。单种子、单次训练，不做重复种子。
- **协议：** [docs/protocol.md](docs/protocol.md)，配置为 `VM-UNet-main/configs/brats2020.json`（预训练初始化、100 轮、单卡 batch 16）。2026-11-14 起，细节支路可读上下各 1 张相邻切片；叠加筛选从 EXP-006 出发。
- **对照：** baseline 为 [EXP-001](research/experiments/EXP-001/report.md)；叠加基座为 [EXP-006](research/experiments/EXP-006/report.md)（detail gated，val best 第 95 轮）。全部实验见 [总表](research/experiments/index.md)。

## 进行中

| 任务 | 状态 | 负责 | 说明 |
|---|---|---|---|
| [TASK-004](handoff/task-004-detail-stacking.md) | 待执行 | B | 四组热启动筛选（对照、边界监督、相邻切片、两者都加），按决策表接完整训练 |

- **A 已完成：** IDEA-005、IDEA-006 的代码、配置、方法文档和测试；A 端 CPU 测试 193 passed（含双进程检查），CUDA 测试待 B 执行。
- **IDEA-004：** 已放弃（val 阈值扫描结论见 [IDEA-004](research/ideas/IDEA-004-et-postprocess.md)）；主表一律报告原始输出。

## 训练中

无。

## 待决事项

1. 第二阶段完整训练结束后，由用户确定最终方法并决定是否发起 FINAL；中间模型不测 test。

## 下一步

- **B：** 拉取 `stage003-detail-stacking`，按 TASK-004 执行检查、smoke 和四组筛选；收集、报告后按决策表直接启动完整训练（用户已确认），并填写回执。
- **A：** 等待 TASK-004 的筛选报告。
