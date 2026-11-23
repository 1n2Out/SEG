# STATE
最后更新：2026-11-23，A。

## 研究状态

- 用户已确定第二项工作的正式创新点一为 [IDEA-001 语义引导的多分辨率细节回注](research/ideas/IDEA-001-detail-fusion.md)，定稿模型为 [EXP-006](research/experiments/EXP-006/report.md)。论文对照为独立训练的 [EXP-001](research/experiments/EXP-001/report.md) 和直接拼接的门控消融 [EXP-014](research/experiments/EXP-014/report.md)。
- [IDEA-010 四方向分组融合](research/ideas/idea-010-direction-difference-fusion.md) 已实现。A 端 CPU 检查通过，[TASK-008](handoff/task-008-direction-full.md) 待交接 B 端，尚未启动训练。
- 下一步：交接后由 B 端验证真实 CUDA/AMP 和正式 batch，再运行一次 EXP-017 完整训练，按 val best 与 EXP-006 比较。论文图表、第二数据集待安排。

## 交接约束

- 常规交接使用 `main`。TASK-008 的训练代码和回执在任务单指定的英文分支上交接。仅在用户明确要求时提交和推送。
- 正式实验遵循[统一协议](docs/protocol.md)，B 端使用 `vm_unet` 和 RTX 6000（GPU 2–5）。退役部件的复核版本见[代码导航](docs/codemap.md)。
- EXP-006 已完成一次获批的 FINAL。A 已接触测试集，后续 BraTS 优化须披露这一点，只用 train/val 做设计。尚未批准新的 FINAL。
