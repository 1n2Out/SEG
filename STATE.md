# STATE
最后更新：2026-11-18，A。

## 研究状态

- 用户同意恢复 ET 残差强度筛选。[IDEA-009](research/ideas/idea-009-et-rollback.md) 已实现，[TASK-007](handoff/task-007-et-rollback-screening.md) 待交 B 端执行。见[决策记录](research/decisions.md)。
- 已采纳的创新点为 [IDEA-001 多分辨率细节回注与语义门控](research/ideas/IDEA-001-detail-fusion.md)。独立基线为 [EXP-001](research/experiments/EXP-001/report.md)，完整模型为 [EXP-006](research/experiments/EXP-006/report.md)，门控消融为 [EXP-014](research/experiments/EXP-014/report.md)。
- [TASK-006](handoff/archive/task-006-et-residual-audit.md) 已验收归档。固定 EXP-006 权重下的 ET 误检抑制及小 ET 漏分反例见[残差核验报告](research/experiments/EXP-006/et-residual-audit/report.md)。
- [TASK-005](handoff/archive/task-005-detail-options-full-training.md) 已按用户要求关闭归档。EXP-006 是否定为最终方法、门控的论文表述仍待用户裁决。

## 交接约束

- 两端共用 `stage003-detail-stacking`，TASK-006 的 B 端结果提交为 `20a50db`。仅在用户明确要求时提交和推送。
- 正式训练按[实验协议](docs/protocol.md)及 `VM-UNet-main/configs/brats2020.json` 执行，单种子、单次训练；B 端使用 `vm_unet` 和 RTX 6000（GPU 2–5）。复核退役选项 EXP-011～013 时在独立副本中检出 `6785299`。
- EXP-006 已完成一次获批的 FINAL。A 已接触测试集，后续 BraTS 优化须披露这一点，只用 train/val 做设计。尚未批准新的 FINAL。
