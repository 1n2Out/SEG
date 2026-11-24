# STATE
最后更新：2026-11-24，A。

## 研究状态

- 用户已确定第二项工作的正式创新点一为 [IDEA-001 语义引导的多分辨率细节回注](research/ideas/IDEA-001-detail-fusion.md)，定稿模型为 [EXP-006](research/experiments/EXP-006/report.md)。论文对照为独立训练的 [EXP-001](research/experiments/EXP-001/report.md) 和直接拼接的门控消融 [EXP-014](research/experiments/EXP-014/report.md)。
- [TASK-008](handoff/task-008-direction-full.md#回执b-填写) 已完成训练和验证分析，待用户验收。[EXP-017](research/experiments/EXP-017/report.md) 第 80 轮早停，第 30 轮 val best；相对 EXP-006，Mean Dice 下降 1.34 个百分点，Mean HD95 改善 1.22 mm。
- 用户已[确认放弃 IDEA-010](research/decisions.md)，保留 EXP-006 作为对照。[EXP-017](research/experiments/EXP-017/report.md) 保留为负结果。第二创新点的新候选、论文图表和第二数据集待安排。

## 交接约束

- 常规交接使用 `main`。TASK-008 的训练代码和回执在任务单指定的英文分支上交接。仅在用户明确要求时提交和推送。
- 正式实验遵循[统一协议](docs/protocol.md)，B 端使用 `vm_unet` 和 RTX 6000（GPU 2–5）。退役部件的复核版本见[代码导航](docs/codemap.md)。
- EXP-006 已完成一次获批的 FINAL。A 已接触测试集，后续 BraTS 优化须披露这一点，只用 train/val 做设计。尚未批准新的 FINAL。
