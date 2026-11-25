# STATE
最后更新：2026-11-25，B。

## 研究状态

- 用户已确定第二项工作的正式创新点一为 [IDEA-001 语义引导的多分辨率细节回注](research/ideas/IDEA-001-detail-fusion.md)，定稿模型为 [EXP-006](research/experiments/EXP-006/report.md)。论文对照为独立训练的 [EXP-001](research/experiments/EXP-001/report.md) 和直接拼接的门控消融 [EXP-014](research/experiments/EXP-014/report.md)。
- [TASK-008](handoff/archive/task-008-direction-full.md) 已关闭。[EXP-017](research/experiments/EXP-017/report.md) 作为负结果归档，用户已[确认放弃 IDEA-010](research/decisions.md)。
- [TASK-009](handoff/task-009-contralateral-full.md#回执b-填写) 已完成训练和验证分析。相对 EXP-006，Mean Dice 小幅上升，非空 ET 组略降，两个通过判据均未达到。Mean HD95 的净降幅主要来自少数病例。是否开展同容量原图消融、如何选择第二数据集，待用户裁决。[EXP-018 报告](research/experiments/EXP-018/report.md)

## 交接约束

- 两端共用 `main`。仅在用户明确要求时提交和推送。
- 正式实验遵循[统一协议](docs/protocol.md)，B 端使用 `vm_unet` 和 RTX 6000（GPU 2–5）。退役部件的复核版本见[代码导航](docs/codemap.md)。
- EXP-006 已完成一次获批的 FINAL。A 已接触测试集，后续 BraTS 优化须披露这一点，只用 train/val 做设计。尚未批准新的 FINAL。
