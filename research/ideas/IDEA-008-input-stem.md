# IDEA-008：输入端卷积补强（与细节支路共享）

状态：**放弃** · 基座：[IDEA-001](IDEA-001-detail-fusion.md) · 任务：[TASK-005](../../handoff/task-005-detail-options-full-training.md) · 决定：[decisions](../decisions.md) 2026-11-15、2026-11-18

> A 已于 2026-11-15 接触测试集。本 IDEA 的动机只来自模型结构分析和业界通行做法，不来自测试集现象。

## 动机

- **patch 瓶颈有两端。** 输入端 `PatchEmbed2D` 是 4×4、步长 4 的不重叠卷积加 LayerNorm，token 进入第一个 VSS 块之前没有任何重叠的局部非线性处理；输出端是线性 4 倍展开（IDEA-001 的动机）。IDEA-001 只补了输出端。
- **业界做法：**
  - VMamba 官方代码的新版本把 4×4 不重叠嵌入换成了两层 3×3、步长 2 的卷积 stem（`patchembed_version="v2"`，引用前核对论文表述）；
  - Early Convolutions Help Transformers See Better（NeurIPS 2021）表明卷积 stem 能改善 ViT 的优化和精度；
  - Swin-UMamba（MICCAI 2024）也在 Mamba 编码器前使用卷积 stem。
- **预期要保守。** 4×4 线性嵌入本身不丢信息（96 个输出通道多于 4×4×4=64 个输入值），收益只能来自给编码器更好的局部非线性特征。在预训练主干上，这类收益通常小于从头训练。

## 核心思想

细节支路已有的 1/2 细节 H 再降到 1/4，经零初始化投影加到 patch embedding 的输出上，初始输出不变。这样，一条共享的高分辨率卷积通路同时补强 patch 化主干的输入端和输出端。实现见提交 `6785299`。

## 依据

- **已有组件：** 卷积 stem（VMamba 新版本、Early Convolutions、Swin-UMamba）、零初始化残差接入。
- **可讲的部分：** 与 IDEA-001 的动机直接对应，同一个 patch 瓶颈的两端由同一条卷积通路补强。它与 Swin-UMamba 更接近，论文中要讲清差异：零初始化残差接入、与细节支路共享、不改动预训练嵌入。原创性需核查。
- **注意：** 解码器不使用 `skip_list[0]`，注入只经编码器起作用：先过第一阶段的 VSS 块，再经 1/8 及更深的跳跃连接影响解码。

## 验证计划

- **对照组：** EXP-006；组合效果看 gated + 两者。
- **验证方式：** 直接完整训练。也可以从 EXP-006 热启动筛选，但主干学习率低时编码器来不及适应新特征，筛选会低估效果。
- **通过标准与决策：** 见 TASK-005。

## 结论

放弃。完整训练未通过 TASK-005 门槛，见 [EXP-012](../experiments/EXP-012/report.md)（单件）和 [EXP-013](../experiments/EXP-013/report.md)（与 IDEA-007 组合），对照 EXP-006，数字来自 `VM-UNet-main/analysis/compare-detail-options.py` 的产物。代码、配置与测试已删除，实现见提交 `6785299`。

- **均值下降。** Mean Dice −0.81、ET Dice −0.92 个百分点；空 ET 假阳性不变，下降不来自空 ET 病例。
- **逐例与均值不一致。** Mean Dice 31 例上升、24 例下降，中位差 +0.19 个百分点，均值由少数病例的明显下降带动。
