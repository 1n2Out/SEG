# STATE

覆盖式更新，不超过 40 行。最后更新：2026-11-12，A。

## 协议与基线

- **工作分支：** `main`。你新建 stage 分支后，在这里更新分支名，两端都执行 `git switch` 切过去。
- **协议：** [docs/protocol.md](docs/protocol.md)，配置为 `VM-UNet-main/configs/brats2020.json`（预训练初始化、100 轮、单卡 batch 16）。
- **对照基线：** [EXP-001](research/experiments/EXP-001/report.md)，val Mean Dice 81.78%（第 55 轮）。全部实验见 [总表](research/experiments/index.md)。

## 进行中

| 任务 | 状态 | 负责 | 说明 |
|---|---|---|---|
| [TASK-001](handoff/TASK-001-detail-fusion-screening.md) | 待执行 | B | detail_fusion 热启动筛选，对应 EXP-002 到 EXP-005 |

## 训练中

无。

## 待用户决定

1. 是否允许使用相邻切片（2.5D）。
2. ET 后处理（IDEA-004）在主表中的报告口径。
3. 之后的完整训练是否统一改为 60 轮。若改，需要先按 60 轮重跑一次 baseline。

## 下一步

- **B：** 首次开工时先核对 [docs/env.md](docs/env.md) 中 B 段的路径，然后执行 TASK-001。
- **A：** 等待 TASK-001 的回执。空闲时可以在 `analysis/` 中补上 IDEA-001 的标签统计脚本。
