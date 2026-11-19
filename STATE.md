# STATE
最后更新：2026-11-19，B。

## 研究状态

- [TASK-007](handoff/archive/task-007-et-rollback-screening.md) 已完成并归档。[EXP-015 常数回调](research/experiments/EXP-015/report.md)与 EXP-006 的逐病例指标一致，[EXP-016 局部回调](research/experiments/EXP-016/report.md)增加了少量 ET 假阳性。两组均未达筛选门槛，本轮筛选保留 EXP-006。
- 已采纳的创新点为 [IDEA-001 多分辨率细节回注与语义门控](research/ideas/IDEA-001-detail-fusion.md)。独立基线为 [EXP-001](research/experiments/EXP-001/report.md)，完整模型为 [EXP-006](research/experiments/EXP-006/report.md)，门控消融为 [EXP-014](research/experiments/EXP-014/report.md)。
- [TASK-006](handoff/archive/task-006-et-residual-audit.md) 已验收归档。固定 EXP-006 权重下的 ET 误检抑制及小 ET 漏分反例见[残差核验报告](research/experiments/EXP-006/et-residual-audit/report.md)。
- [TASK-005](handoff/archive/task-005-detail-options-full-training.md) 已按用户要求关闭归档。EXP-006 是否定为最终方法、门控的论文表述仍待用户裁决。

## 交接约束

- 两端共用 `stage003-detail-stacking`。仅在用户明确要求时提交和推送。
- 正式训练按[实验协议](docs/protocol.md)及 `VM-UNet-main/configs/brats2020.json` 执行，单种子、单次训练；B 端使用 `vm_unet` 和 RTX 6000（GPU 2–5）。复核退役选项 EXP-011～013 时在独立副本中检出 `6785299`。
- EXP-006 已完成一次获批的 FINAL。A 已接触测试集，后续 BraTS 优化须披露这一点，只用 train/val 做设计。尚未批准新的 FINAL。

## TASK-007 交接

- 两组训练均已正常结束，固定第 10 轮 val 已收集。运行目录、来源权重和验收项见[任务回执](handoff/archive/task-007-et-rollback-screening.md#b-端回执)，逐病例统计见[配对汇总](research/experiments/EXP-016/comparison-summary.json)。
