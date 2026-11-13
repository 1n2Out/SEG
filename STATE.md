# STATE

覆盖式更新，不超过 40 行。最后更新：2026-11-13，B。

## 协议与基线

- **工作分支：** `main`。TASK-001 阶段分支已合并，A 端接手时切换到 `main`。
- **用户要求：** 仅在用户明确要求时提交和推送；本轮由用户手动推送。仅使用 RTX 6000，不使用 A100。
- **协议：** [docs/protocol.md](docs/protocol.md)，配置为 `VM-UNet-main/configs/brats2020.json`（预训练初始化、100 轮、单卡 batch 16）。
- **对照基线：** [EXP-001](research/experiments/EXP-001/report.md)，val Mean Dice 81.78%（第 55 轮）。全部实验见 [总表](research/experiments/index.md)。

## 进行中

| 任务 | 状态 | 负责 | 说明 |
|---|---|---|---|
| [TASK-001](handoff/TASK-001-detail-fusion-screening.md) | 执行完成，待裁决 | B | 四组筛选、收集、配对分析、报告和回执完成；gated 通过四项标准，门控额外收益未证实 |

## 训练中

无。EXP-002 至 EXP-005 均完成 10 轮和最终验证，结果见[实验总表](research/experiments/index.md)。

## 待用户决定

1. 是否批准 gated 的完整协议训练，并保留 concat 对照；依据见 [EXP-003](research/experiments/EXP-003/report.md)。
2. 是否允许使用相邻切片（2.5D）。
3. ET 后处理（IDEA-004）在主表中的报告口径。
4. 之后的完整训练是否统一改为 60 轮。若改，需要先按 60 轮重跑一次 baseline。

## 下一步

- **B：** 本轮交付完成，等待用户裁决和下一任务；完整训练需另行安排。
- **A：** 从本分支读取筛选结果、分析脚本和报告；用户裁决后发起下一任务，并补上 IDEA-001 的标签统计脚本。
