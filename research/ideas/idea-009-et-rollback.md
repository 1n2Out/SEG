# IDEA-009：ET 残差强度门控

状态：已实现 · 基座：[IDEA-001](IDEA-001-detail-fusion.md) · 筛选：[TASK-007](../../handoff/task-007-et-rollback-screening.md)

## 依据

[TASK-006](../experiments/EXP-006/et-residual-audit/report.md) 表明，细节残差主要削减 ET 误检，也会删去真 ET，尤其在真值内侧边界。检验局部门控能否保住部分真 ET，同时维持空 ET 患者的表现。A 已接触过测试集，本方案依据仅为 train/val。

## 设计和对照

在同一前向中保留主干分数与加残差后的分数。若主干预测为 ET、加残差后转为非 ET，把最终 ET 分数相对于最强非 ET 类的差值向主干差值插值。局部系数取决于全分辨率融合特征和两个 ET 分数差；常数对照为全空间共享一个可学习系数。[实现说明](../../docs/methods/detail_fusion.md#et-残差强度门控)

冻结 EXP-006，仅训练回调模块。与原 EXP-006 和常数回调做固定轮次配对，检查局部特征带来的增益及恢复 ET 的真假阳性。[运行与验收标准](../../handoff/task-007-et-rollback-screening.md)
