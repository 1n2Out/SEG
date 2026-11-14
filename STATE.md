# STATE

覆盖式更新，不超过 40 行。最后更新：2026-11-14，B。

## 协议与基线

- **工作分支：** `stage002-detail-gated-full-training`，由本地 `main` 创建；TASK-001 已合并到 `main`。
- **用户要求：** 暂不推送；仅在用户明确要求时提交和推送。使用 `vm_unet` 和 RTX 6000，不使用 A100。
- **协议：** [docs/protocol.md](docs/protocol.md)，配置为 `VM-UNet-main/configs/brats2020.json`（预训练初始化、100 轮、单卡 batch 16）。
- **对照基线：** [EXP-001](research/experiments/EXP-001/report.md)，val Mean Dice 81.78%（第 55 轮）。全部实验见 [总表](research/experiments/index.md)。

## 本轮已完成

| 任务 | 状态 | 负责 | 说明 |
|---|---|---|---|
| [TASK-001](handoff/archive/task-001-detail-fusion-screening.md) | 关闭 | B | 热启动筛选、结果收集、配对分析和报告完成 |
| [TASK-002](handoff/archive/task-002-detail-gated-full-training.md) | 关闭 | B | 完整训练、val best 收集、配对分析和报告完成 |
| [TASK-003](handoff/archive/task-003-detail-gated-final.md) | 关闭 | B | 一次 FINAL 测试及报告完成；测试结果独立保存在 EXP-006/final/ |

## 训练中

无进行中任务或训练。文档交付已完成，任务单已归档；本轮改动未提交或推送。

## 后续研究待决（不属于本轮未完成项）

1. 是否基于验证集结果开展完整 concat 对照与后续消融；门控独立贡献仍未证实，见 [EXP-006](research/experiments/EXP-006/report.md)。
2. 是否允许使用相邻切片（2.5D）。
3. ET 后处理（IDEA-004）在主表中的报告口径。
4. 之后的完整训练是否统一改为 60 轮。若改，需要先按 60 轮重跑一次 baseline。

## 交付状态

- **B：** 本轮工作与文档均已收尾，无待办；仅在用户布置新任务后继续。
- **后续约束：** 不自动提交、推送、训练或重复测试；测试结果不用于选方案、调参或动机分析。
