# IDEA-011 对侧半球参照的多模态差异融合

状态：**已实现** · [方法](../../docs/methods/contralateral-reference.md) · [TASK-009 / EXP-018](../../handoff/task-009-contralateral-full.md)

## 动机

同一张轴向 MRI 同时包含两侧半球。为每个位置提供同一患者对侧近似对应位置的四模态信号，可以检验个体内强度差是否有助于区分病灶和正常结构。

按固定 train 清单和 NIfTI affine 确认左右轴，以数组中心近似中线。逐患者统计 WT 镜像落在 WT 外且两侧 FLAIR 非零的比例，并统计这些配对中本侧更亮的比例。ET 单独用 T1ce 做同样的强度比较，参照仍须位于 WT 外。[分析脚本](../../VM-UNet-main/analysis/contralateral-reference.py) · [训练集汇总](../../VM-UNet-main/results/analysis/contralateral-reference/summary.json)

上述统计检验有标签病灶是否存在参照与强度差。正常组织的不对称也能产生差异，跨中线病灶和占位变形会削弱参照价值。

## 结构

- 在随机增强前，按解剖左右轴对单张切片逐模态计算带符号差值（本侧减镜像对侧），再对原图、差值图、标签同步增强。推理按同一解剖轴构造差值。标签只用于监督及离线统计。
- 差值图平均池化到 1/2 尺度，经轻量 3×3 卷积提取差异特征。拼接该特征与 EXP-006 的半分辨率解码语义，预测单通道空间门控，让网络学习在哪些位置采用对侧参照。
- 门控后的差异特征经零初始化 1×1 投影，加到已有的 1/2 融合特征上，再走原 1/1 解码与四类输出。每张切片只运行一次主干，训练沿用原 CE 和区域 Dice。

## 文献依据

Zhang 等在《Segmenting Brain Tumors with Symmetry》（ML4H Workshop 2017）中验证过脑对称先验。Ma 等在《Symmetry Awareness Encoded Deep Learning Framework for Brain Imaging Analysis》（MICCAI 2024）中用跨半球注意力与预训练学习对称特征。本候选的贡献定位为在 2D 四模态 VM-UNet 中，按解码语义选择带方向的模态内差异，并用同容量对照验证这种参照的价值。

## 验证计划

- 首轮只新增一次完整训练，从相同 VMamba 预训练初始化，与 EXP-006 比较。沿用正式协议和原始输出，检查 val 分区 Dice、Mean HD95、空 ET 误检，并按患者配对。
- 建议沿用 TASK-008 的门槛。相对 EXP-006 的 val Mean Dice 至少增加 0.30 个百分点，且非空 ET 患者的配对 Mean Dice 平均差为正，再投入一次消融。若均值小幅上升，只报告病例来源。若持平或下降，结束本轮。
- 消融保留相同卷积、门控和参数量，把差值图换成本侧四模态图，独立完整训练。另按 WT 镜像重叠程度查看病例差异，优先检查跨中线失败例及正常对侧新增误检。

## 结论

已完成 train 动机核验。模型的验证集增益待完整训练。A 曾接触既有 FINAL，候选依据仅来自 train、既有 val 和公开文献。
