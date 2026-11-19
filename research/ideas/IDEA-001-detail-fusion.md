# IDEA-001：多分辨率细节回注与语义门控

状态：**采纳** · 定位：第二项工作的正式创新点一 · 方法文档：[detail_fusion](../../docs/methods/detail_fusion.md) · 实验：[EXP-006](../experiments/EXP-006/report.md) · 决定：[decisions](../decisions.md) 2026-11-19

## 动机

**结构瓶颈。** VM-UNet 只通过 4×4 的 patch token 看图像：
- 输入端是步长为 4 的卷积；
- 输出端是"线性层 + 像素重排"的 4 倍上采样，之后只接 1×1 卷积。

全分辨率输出端缺少跨像素的局部融合，每个 4×4 输出块由一个解码 token 投影生成。

**标签分析。** 对 train/val 的轴向标签做 3×3 平面腐蚀，统计各区域被腐蚀体素的合计占比。ET 的平面边界占比高于 WT 和 TC。再以 2×2 和 4×4 块做四类多数投票，按患者计算 Dice 后等权平均。两种网格下 ET 的标签近似损失均最大，为 1/2 和 1/1 尺度的细节回注提供动机。块多数投票衡量标签的粗网格近似误差。

**逐患者分析。** 在非空 ET 的 val 病例上，把相对 EXP-001 的 ET Dice 配对差对边界占比和 ET 体素数的对数做回归。两种回注的斜率点估计都为正，逐一移除单个病例后方向一致。按患者重采样，concat 的 95% 区间跨零，gated 的下界接近零。薄层受益更多属于探索性观察。

统计由 [detail-motivation.py](../../VM-UNet-main/analysis/detail-motivation.py) 生成，见[分析汇总](../../VM-UNet-main/results/analysis/detail-motivation/summary.json)和[离线绘图命令](../../docs/plotting.md#细节回注动机-detail-motivation)。

## 核心思想

在原 logits 上叠加一个零初始化的残差分支：
1. 用轻量卷积从输入图像中提取 1/2 和 1/1 尺度的细节；
2. 把最终的解码语义逐级上采样后，与这些细节融合；
3. 由语义门控决定细节在哪些位置起作用。

主干网络和损失都保持不变。

## 依据

- **Swin-UMamba（MICCAI 2024）** 在 Mamba 编码器前加了步长 2 的卷积 stem，并从输入直接接一条全分辨率跳跃连接；**U-Net v2 / VM-UNetV2** 研究语义与细节特征的融合。
- 空间门控借鉴 EMCAD（CVPR 2024）的 LGAG，归一化和权重生成方式见[方法文档](../../docs/methods/detail_fusion.md)。
- **贡献定位。** 粗网格下 ET 标签的近似损失为结构设计提供动机。在 VM-UNet 解码端逐级融合原图细节，并以残差修正分割输出；卷积支路和 LGAG 门控均有既有工作基础。

## 验证记录

- 热启动筛选（[TASK-001](../../handoff/archive/task-001-detail-fusion-screening.md)）通过，对照为同样微调 10 轮的 baseline。
- 完整训练（[TASK-002](../../handoff/archive/task-002-detail-gated-full-training.md)）的验证集结果见 EXP-006。
- 同一 EXP-006 权重下，残差修正前后的逐患者与分区统计已通过 [TASK-006](../../handoff/archive/task-006-et-residual-audit.md) 核验。

## 论文表述

- **完整模型效果。** 同一训练协议下，EXP-006 相对独立训练的 EXP-001 提升了验证集 Mean Dice 和 ET Dice。[完整训练对照](../experiments/EXP-006/report.md)
- **残差的直接作用。** 在固定 EXP-006 权重的验证集诊断中，残差主要削减共同训练主干的 ET 误检，多数非空 ET 患者受益。修正也删去部分真实 ET，包含小病灶最终预测为空的反例。该比较只衡量同一模型内部的输出变化。[逐患者核验报告](../experiments/EXP-006/et-residual-audit/report.md)
- **测试集：** 只评价过一次，见 [FINAL 报告](../experiments/EXP-006/final/report.md)。Dice 提升且 ET 最明显；Mean HD95 未同步改善，患者重采样区间跨 0，论文须如实呈现。
- **门控消融。** 单种子验证中，gated 的整体 Mean Dice 高于 concat。差距主要来自一例空 ET 误检，非空 ET 平均 Dice 则以 concat 略高。[EXP-014 配对分析](../experiments/EXP-014/report.md)
