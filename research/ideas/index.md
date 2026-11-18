# 创新点索引

状态只取以下几种：**候选**、**已实现**、**筛选中**、**采纳**、**放弃**。改为"采纳"或"放弃"需要用户确认，并记入 [decisions](../decisions.md)。每个立项的 IDEA 一个文件，不超过 60 行，按 [模板](TEMPLATE.md) 写。

| ID | 名称 | 状态 | 一句话 |
|---|---|---|---|
| [IDEA-001](IDEA-001-detail-fusion.md) | 多分辨率细节回注 + 语义门控 | 采纳 | 两级细节回注改善验证集分割，残差主要削减 ET 误检，也有误删小 ET 的病例 |
| [IDEA-002](IDEA-002-shallow-refinement.md) | 浅层细节回注 | 放弃 | 只在 1/4 尺度补连接；不属于 IDEA-001，不再保留为对照 |
| [IDEA-003](IDEA-003-modality-difference.md) | 区域条件的模态差异补偿 | 候选 | 用 T1ce−T1 等模态差异补偿 ET 和 TC；暂缓 |
| [IDEA-004](IDEA-004-et-postprocess.md) | ET 小体积后处理 | 放弃 | 空 ET 误判体积大，去掉它们会误删真实小 ET；统一施加时 baseline 获益更多 |
| [IDEA-005](IDEA-005-boundary-supervision.md) | 细节支路的边界辅助监督 | 放弃 | 给补回的 1/1 特征加边界带监督；筛选中 ET 多数病例下降，HD95 改善由少数病例带动 |
| [IDEA-006](IDEA-006-slice-context.md) | 细节支路读取相邻切片 | 放弃 | 只给细节支路补层间信息；筛选中无方向性信号，与边界监督叠加也不互补 |
| [IDEA-007](IDEA-007-deep-supervision.md) | 多尺度深监督（粗—细两级） | 放弃 | 五处同损失监督；完整训练 ET 升、TC 降，Mean Dice 无增益（EXP-011） |
| [IDEA-008](IDEA-008-input-stem.md) | 输入端卷积补强 | 放弃 | 1/2 细节零初始化加到 patch embedding；完整训练 Mean Dice 与 ET 均下降（EXP-012） |
| [IDEA-009](idea-009-et-rollback.md) | ET 残差强度门控 | 已实现 | 按局部特征回调 ET 抑制，与可学习的常数回调配对筛选 |
| — | 区域存在性感知监督 | 放弃 | 非空 ET 有改善，但空 ET 假阳性增多，总体退步；源码在 `VM-UNet-main/archives/` |

## 候选池（未立项，只保留取舍结论）

| 候选 | 取舍 |
|---|---|
| 区域化 sigmoid 输出头 | 推理和评价都要改，改动面大 → 暂缓 |
| 只在 1/4 尺度及以下更换融合模块 | 只是对已有特征重新加权，预期收益与噪声同一量级 → 不单独做 |
| STNet SFF | 适合作为 1/4 尺度的融合消融 → 暂缓 |
| SAB/CAB 替换跳跃连接处的相加 | 成本极小，可作为叠加件 → 暂缓 |
| SDI（U-Net v2） | 新意会被归给 VM-UNetV2 → 低优先 |
| FreqFusion、DySample | 针对边界和上采样，对应 HD95；替换上采样会破坏零初始化热启动 → 可选 |
| 卷积 FFN、延长训练 | 预训练权重覆盖不到；baseline 的 best 已在第 55 轮 → 不做首选 |
