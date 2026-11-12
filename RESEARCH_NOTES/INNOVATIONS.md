# 创新点

更新：2026-11-12。先验证收益，再用消融解释机制；工程验收不等于分割效果提升。

## CURRENT

| 方向 | 状态 | 优先级 |
| --- | --- | --- |
| 创新二（升级）：多分辨率细节回注与语义门控融合 | 已实现，CPU 与双进程检查通过；待服务器 CUDA 检查、smoke 和热启动筛选 | 当前优先 |
| 创新二（新版）：浅层细节回注与语义门控 | 已实现、工程验收通过，效果待验证 | 降为“仅 1/4 尺度”对照，完整训练暂缓 |
| 评价端：ET 小体积后处理 | 候选，未实现；可先用已有逐患者表格零训练模拟 | 并行，报告口径待定 |
| 创新一：区域条件的模态差异补偿 | 候选，未实现 | 备选（可借 STNet TFF 实现） |

公共协议沿用 [EXP-001 baseline](EXPERIMENTS_BraTS.md)：二维四模态输入、三维患者级评价，固定 259/55/55 划分；同一预训练初始化、batch=16、100 轮，按验证 Mean Dice 选 best。选型依据验证集，方案固定后再评价测试集。热启动筛选（见创新二升级版）只用于决定哪些变体值得完整训练，论文数字仍来自上述正式协议。

## 决策记录：2026-11-11/12

**背景。** 用户要求以最小时间成本先提点，再根据有效模块组织创新故事；接受“模块缝合 + 轻微改造”。11-11 先给出候选，未改代码；11-12 用户认可“先做最可能提点的结构改造”，并接受把融合模块挂在该改造上。本节记录取舍依据，便于后续回溯。

### 定位依据

**结构瓶颈。** VM-UNet 只通过 4×4 patch token 看图像：输入端 `PatchEmbed2D` 是步长 4 的卷积，输出端 `Final_PatchExpand2D` 是“线性层 + 像素重排 + LN”的 4 倍扩展，之后只接 1×1 卷积。全分辨率上没有空间卷积，每个 4×4 输出块只由一个 token 决定；1/2、1/1 尺度的特征完全缺失。原创新二只补 1/4 尺度的 `skip_list[0]`，不增加分辨率。

**标签统计。** 只读 train（259 例）和 val（55 例）标签，未触碰 test。“距边界”用平面内 3×3 腐蚀计算；“块常数上限”指每个平面块按多数表决只判一类后重建，再按本地协议计算患者级三维 Dice（空对空记 1）并对患者平均。256 输入的补零为每侧 8 像素，块网格与模型 token 网格对齐。

| 统计（train / val） | WT | TC | ET |
| --- | ---: | ---: | ---: |
| 距边界 ≤1 像素的体素比例 | 20.2% / 18.8% | 21.9% / 20.3% | 52.9% / 52.6% |
| 距边界 ≤2 像素的体素比例 | 36.1% / 33.9% | 39.3% / 36.7% | 79.7% / 77.3% |
| 4×4 块常数上限 Dice | 92.6% / 93.1% | 91.7% / 92.3% | 81.3% / 79.9% |
| 2×2 块常数上限 Dice | 96.2% / 96.5% | 95.9% / 96.2% | 90.4% / 90.4% |

ET 是薄层结构；非空患者的 4×4 上限为 79.8% / 77.9%，baseline 非空 ET Dice（验证 75.09%、测试 79.85%）已接近该量级。模型输出并不严格按块判类，所以这不是严格上限，但说明 1/4 分辨率很可能是 ET 的瓶颈，可作为论文动机图。统计脚本是一次性分析，未入库。

**失分来源。** 空 ET 患者：val 5 例、train 18 例；val 另有 4 例 ET 小于 1000 体素（其中 2 例小于 200）。按本地规则，空 ET 患者只要预测出 ET，该患者 ET Dice 记 0、HD95 记 373.13 mm；一例这样的误判使 Mean Dice 下降约 0.61 个百分点。baseline 在 val、test 上各有 2 例；测试集 ET HD95 的 24.49 mm 中约 13.6 mm 来自这两例。EXP-002 的整体下降也来自这里（见 ARCHIVE）。

**训练轮数。** EXP-001、EXP-002 的最佳均在第 55 轮，后半程未刷新 best。

### 候选与取舍

| 候选 | 判断 | 决定 |
| --- | --- | --- |
| 第 0 步：ET 小体积后处理（No New-Net、nnU-Net 的 BraTS 做法） | 零训练；用已有 `per_patient.csv` 即可模拟；理想情况下 baseline 验证 Mean Dice 81.78→约 82.99（上限估计）。对所有方法同样有效，不拉开方法差距 | 并行做，主表口径待定 |
| EXP-002 + 后处理 | 乐观估计验证约 84.1，比加同一后处理的 baseline（约 83.0）高约 1.1 | 待用户决定是否复活 |
| C1 多分辨率细节回注 | 直接针对上述瓶颈；零初始化可热启动筛选；是创新二的自然升级 | **采纳为主线，已实现** |
| C2 相邻切片（2.5D） | 需打破“不使用相邻切片”的设定；文献结论不一（BraTS17 有 ET +5.6，BraTS20 无提升） | 待用户决定 |
| C3 深监督 | 预期收益小（约 0.2–0.5），可叠加 | 暂缓 |
| C4 区域化 sigmoid 输出头 | 推理、评价都要改，改动面大 | 暂缓 |
| VSS 块加卷积 FFN、延长训练 | 预训练覆盖不到且变慢；最佳已在第 55 轮 | 不作为首选 |
| simple/gated 浅层回注的两次完整训练 | 只补 1/4，预计提升空间有限；每次约 11 小时 | 暂缓，改用热启动筛选 |

### 融合模块评估

现有融合点（3 处 `x + skip` 相加及 64² 浅层回注）都在 1/4 及以下，换融合模块只能重新加权已有特征，常见收益约 0.2–0.5，与单 seed、55 例验证集的噪声同量级；解码器已有 SS2D 全局建模。融合模块最有价值的位置是 C1 的新注入点：既有新信息进来，又需要语义筛掉噪声。

| 模块 | 判断 | 决定 |
| --- | --- | --- |
| LGAG（EMCAD，CVPR 2024） | 分组卷积注意力门，轻量；笔记已引用 EMCAD，叙事一致 | **采纳为 C1 的语义门控** |
| STNet SFF（ICME 2023） | 全像素注意力：64² 便宜，128² 需 PyTorch 2.0 的 SDPA，256² 不划算；1/4 处不增分辨率 | 暂缓；可作 1/4 尺度消融，与浅层回注 gated 对比 |
| STNet TFF | 把 (T1, T1ce)、(T2, FLAIR) 当作配对输入做差异门控；ET 本身就是 T1ce 相对 T1 的强化，故事性最强，可接创新一；风险是放大血管、脉络丛强化，加重空 ET 假阳性 | C1 筛选后再做；1/2 尺度输出可与 C1 共用 |
| CGAFusion、AFF/iAFF、Attention Gate | 同类门控融合 | LGAG 不理想时替换 |
| SAB/CAB（MALUNet，UltraLight VM-UNet、H-vmunet 沿用） | 替换 3 处跳跃相加最顺，成本极小 | 暂缓，作便宜叠加件 |
| SDI（U-Net v2，VM-UNetV2 已用） | 新意会归给 VM-UNetV2 | 低优先 |
| FreqFusion、DySample | 边界、上采样专项，对应 HD95 | 可选 |
| UCTransNet 通道 Transformer、逐模态编码器的跨模态 Mamba 融合 | 较重，或与 SS2D 重复，或训练成本翻倍 | 不推荐 |

### 待用户决定

1. 是否用 ET 后处理复活 EXP-002（与“已停用”的结论相反）。
2. 是否允许使用相邻切片，这决定 C2 能否做。
3. 后处理在主表中的口径：所有方法都加，或主表报原始输出、最终方法单列“+后处理”。
4. 后续完整训练是否统一改为 60 轮：需先按 60 轮重跑一次 baseline（约 6.7 小时），之后每次约省 4.5 小时。

## 创新二（升级）：多分辨率细节回注与语义门控融合

#### 1. Research Motivation

见上方“定位依据”：VM-UNet 的 4×4 token 化与线性 4 倍扩展使全分辨率细节无法恢复，而 ET 约一半体素位于边界一像素内。拟检验：补回 1/2、1/1 尺度的细节，能否提升 ET、TC 的 Dice 与 HD95，且不增加空 ET 假阳性。

#### 2. Core Idea

在原 logits 上叠加零初始化的残差 logits 分支：轻量卷积从输入图像提取 1/2、1/1 细节，最终解码语义逐级上采样后与细节融合；语义门控决定细节在哪里起作用。原主干、原 CE + Dice 损失不变。

#### 3. Theoretical Basis

[Swin-UMamba（MICCAI 2024）](https://arxiv.org/abs/2402.03302)在 Mamba 编码器前加步长 2 的卷积 stem，并从输入加全分辨率跳跃连接；其轻量版 Swin-UMamba† 去掉了 1/2 特征。[U-Net v2](https://arxiv.org/abs/2311.17791) 与 [VM-UNetV2](https://arxiv.org/abs/2403.09157) 研究语义与细节的融合。门控沿用 [EMCAD（CVPR 2024）](https://openaccess.thecvf.com/content/CVPR2024/html/Rahman_EMCAD_Efficient_Multi-scale_Convolutional_Attention_Decoding_for_Medical_Image_Segmentation_CVPR_2024_paper.html)的 LGAG。高分辨率卷积细节支路和门控都是已有组件，不作为原创贡献；可讲的是“以量化的 patch 分辨率瓶颈为依据、面向薄层 ET 的改造”，原创性需进一步核查。

#### 4. Technical Design

X 为输入（4 通道、256²），D 为最终 1/4 解码特征（64²、96 通道）。

1. 细节支路：X 经残差卷积块得到 1/1 细节 F（16 通道），再经步长 2 的残差卷积块得到 1/2 细节 H（32 通道）。
2. 1/2 融合：D 经 1×1 卷积与 2 倍像素重排上采样为 S₂；gated 模式下 H ← H·σ(LGAG(S₂, H))；以残差卷积块融合 [S₂, H]。
3. 1/1 融合：结果再上采样为 S₁；gated 模式下 F ← F·σ(LGAG(S₁, F))；融合 [S₁, F]。
4. 1×1 卷积输出四类残差 logits，加到原 logits 上。

LGAG 的两处改动：BatchNorm 换成 GroupNorm，使结果与 batch 组成无关；sigmoid 前不做归一化，使整张切片的门控可以整体关闭。卷积块为两层 3×3 卷积加残差投影，GroupNorm + GELU。只有最后的 1×1 输出卷积初始化为零：初始输出与 baseline 逐位相同，第一步只有输出卷积学习，之后整个支路开始学习。模块在主干之后构建并隔离随机数，相同 seed 下主干初始权重与后续随机序列与 baseline 一致。

| 模式 | 计算 | 新增参数 | 卷积计算量（每张 256² 切片） |
| --- | --- | ---: | ---: |
| `off` / 省略配置 | 原 baseline | 0 | 0 |
| `concat` | 细节与语义直接拼接融合 | 69,476 | 约 1.49 GMACs |
| `gated` | 拼接前用 LGAG 筛选细节 | 70,462 | 约 1.52 GMACs |

实现：[detail_fusion](../VM-UNet-main/brats/innovations/detail_fusion/block.py)。正式配置：[gated.json](../VM-UNet-main/configs/detail_fusion/gated.json)、[concat.json](../VM-UNet-main/configs/detail_fusion/concat.json)，只比 baseline 多出模块字段和输出目录。使用、筛选命令和移除方法见 [使用说明](../VM-UNet-main/docs/DETAIL_FUSION.md)。

**热启动筛选。** 训练入口新增 `warm_start`：从 EXP-001 的 `results/brats2020_1102_1734/best.pth`（第 55 轮，验证 81.78%）加载全部原有参数，挂上零初始化模块微调 10 轮。要求同一患者划分，只允许新模块参数缺失；与 `pretrained` 互斥。原有参数学习率 1e-5，新模块参数 1e-4，两组都按余弦在 10 轮内降到 0；第 1、5、10 轮验证；其余设置同 EXP-001。

| 筛选配置 | 用途 | 必要性 |
| --- | --- | --- |
| [baseline.json](../VM-UNet-main/configs/screening/baseline.json) | 对照：不挂模块，同样微调 10 轮，排除“多训几轮”的影响 | 必跑 |
| [detail_gated.json](../VM-UNet-main/configs/screening/detail_gated.json) | 主候选 | 必跑 |
| [detail_concat.json](../VM-UNet-main/configs/screening/detail_concat.json) | 门控消融 | 建议 |
| [shallow_gated.json](../VM-UNet-main/configs/screening/shallow_gated.json) | 只补 1/4 尺度的现有方案 | 可选 |

对照组按 EXP-001 每轮约 6.7 分钟估算约 1.2–1.5 小时；带模块变体的实际耗时以 `history.jsonl` 的 `train_seconds` 为准。多卡可并行。

**判读规则。** 与对照组比较同一轮次的验证 Mean Dice、三区域 Dice、HD95、空 ET 假阳性病例数，并用逐患者 CSV 看配对差。主候选比对照组高 0.5 个百分点以上，且 ET 与 HD95 方向一致，才送去完整训练（约 11 小时）；0.5 以内在 55 例、单 seed 下基本是噪声。论文消融预设为 baseline、浅层回注 gated（仅 1/4）、concat、gated；有收益后再考虑只保留 1/2 或 1/1 的消融和多 seed 复验。

#### 5. Research Conclusion

**尚无实验结论。** 2026-11-12 在隔离的 Linux CPU 环境（Python 3.10、torch 1.13.1+cpu，与 Windows `vmunet-local` 同版本）完成工程检查：新增 38 项 CPU 检查通过（配置、零初始化恒等、两步学习、参数量、逐切片独立、原损失不变、构建时 baseline 权重与随机数一致、前向接线、热启动加载规则、学习率分组、启动器预检），两进程 Gloo 检查通过；全量测试 151 passed、8 skipped（6 项需真实 CUDA；2 项双进程检查需显式开启，已单独开启通过）。训练入口在不配置新字段时，baseline 的模型构建、初始化和优化器保持原样。

尚未执行：服务器上的 [CUDA 检查](../VM-UNet-main/tests/test_detail_fusion_cuda.py)（真实 selective scan 与 AMP 下初始输出逐位相同、两步学习、checkpoint 恢复、热启动后输出与来源模型相同）、筛选配置的 smoke 和热启动筛选。Windows 本地可选复核新增 CPU 测试。

## 创新二（新版）：浅层细节回注与语义门控

#### 1. Research Motivation

当前 `skip_list[0]` 为 1/4 分辨率、96 通道的浅层特征，未直接连接最后一级解码层；该解码层恰好具有相同尺寸。拟检验增加直接连接能否改善局部细节恢复。

#### 2. Core Idea

将浅层信息经轻量提取和语义筛选后，残差回注最终解码特征。保留原主干及原 CE+Dice，集中验证结构收益。

#### 3. Theoretical Basis

[VM-UNetV2](https://arxiv.org/abs/2403.09157)研究语义与细节融合，[EMCAD（CVPR 2024）](https://openaccess.thecvf.com/content/CVPR2024/html/Rahman_EMCAD_Efficient_Multi-scale_Convolutional_Attention_Decoding_for_Medical_Image_Segmentation_CVPR_2024_paper.html)采用轻量多尺度卷积解码。具体连接与门控的价值需通过消融验证，不将已有基础组件本身作为原创贡献。

#### 4. Technical Design

S 为最浅层特征，D 为最终 1/4 分辨率解码特征。

| 模式 | 计算 | 新增参数 |
| --- | --- | ---: |
| `off` / 省略配置 | 原 baseline | 0 |
| `simple` | `D′ = D + α · Conv1×1(S)` | 9,408 |
| `gated` | `D′ = D + α · G · R` | 26,520 |

完整模块：S 经 1×1 卷积降至 24 通道，经过并行 3×3/5×5 深度卷积与 GELU，拼接并投影回 96 通道得到 R；`G = sigmoid(Conv1×1([D,R]))`。α 为逐通道可学习系数，初始为零，其他卷积正常初始化；初始输出与同 seed baseline 一致。模块在训练和推理时均执行，输出仍为四类 logits。

实现：[shallow_refinement](../VM-UNet-main/brats/innovations/shallow_refinement/model.py)。配置：[simple.json](../VM-UNet-main/configs/shallow_refinement/simple.json)、[gated.json](../VM-UNet-main/configs/shallow_refinement/gated.json)。开关、训练和删除方法见 [使用说明](../VM-UNet-main/docs/SHALLOW_REFINEMENT.md)。

#### 5. Research Conclusion

**工程验收通过，正式效果待验证。** 已通过 117 项检查及独立小清单训练、恢复、验证；确认零残差初始化等价、分支可学习、checkpoint 可恢复。见 [验收报告](../VM-UNet-main/results/verification/shallow_refinement_pipeline/verification_report.json)。

原计划比较 baseline、simple、gated 的完整训练。**2026-11-12 调整：** 两次完整训练暂缓；gated 以 [shallow_gated.json](../VM-UNet-main/configs/screening/shallow_gated.json) 进入热启动筛选（可选），作为“只补 1/4 尺度”的对照。若升级版有效，本方案在论文中作为仅 1/4 回注的消融；同一次运行只能启用一个模型扩展。

## 创新一：区域条件的模态差异补偿

#### 1. Research Motivation

四模态在 Patch Embedding 阶段融合，后续不再显式区分模态。拟检验：补充模态特有信息，能否改善 TC、ET 识别。早期融合导致信息损失尚未被证实。

#### 2. Core Idea

主干学习融合语义，轻量分支保留模态差异，由解码区域预测决定补偿位置和强度。

#### 3. Theoretical Basis

[RFNet（ICCV 2021）](https://openaccess.thecvf.com/content/ICCV2021/html/Ding_RFNet_Region-Aware_Fusion_Network_for_Incomplete_Multi-Modal_Brain_Tumor_Segmentation_ICCV_2021_paper.html)及[子区域感知模态融合（2026）](https://arxiv.org/abs/2601.15734)已有区域模态加权设计。拟研究差异是面向已有融合主干的残差补偿，原创性需进一步核查。

#### 4. Technical Design

四模态分别经共享轻量卷积，计算各自相对均值的特征差异；用解码器预测的 WT/TC/ET 概率进行空间加权，在一个解码尺度残差注入。训练和推理均使用预测区域条件。

最小对照：baseline、普通模态加权、去掉区域条件、完整模块；同时比较参数量与耗时。

**2026-11-12 补充：** 可借 [STNet（ICME 2023）](https://arxiv.org/abs/2304.11422)的 TFF 实现模态差异：把 (T1, T1ce)、(T2, FLAIR) 当作配对输入，以差分生成空间门控后融合，结果以零初始化残差接入。其 1/2 尺度输出可与创新二升级版的细节支路共用。主要风险是放大血管等非肿瘤强化，加重空 ET 假阳性。

#### 5. Research Conclusion

尚无实验结论。暂缓实现；待创新二升级版筛选后，再决定是否以 TFF 路线实现。

## 评价端：ET 小体积后处理

**动机。** 空 ET 假阳性是 baseline 与 EXP-002 的主要失分来源（见“定位依据”）。BraTS 上常见做法是：患者预测的 ET 体素少于阈值 T 时，把 ET 全部改为坏死（标签 1），WT、TC 不变，只有 ET 变化（[No New-Net](https://arxiv.org/abs/1809.10483)、[nnU-Net BraTS 2020](https://arxiv.org/abs/2011.00848)；其他队伍用过 50、200、300 体素）。nnU-Net 作者也指出，这类后处理迎合 BraTS 评价规则，在需要发现微小强化灶的场景下未必合适。

**做法。** 逐患者 CSV 已含预测、真值的 ET 体素数及 ET 指标，可在验证集上扫描 T，零训练、无需重新推理；阈值只在验证集上选定，test 最后用同一阈值报告一次。验证集有 2 例真实 ET 小于 200 体素，阈值过大会误伤。估计值（上限，假设去掉全部假阳性且不误伤）：baseline 验证 81.78→约 82.99，EXP-002 验证 81.08→约 84.1。

**状态。** 未实现；主表口径与是否用于复活 EXP-002 待用户决定（见“待用户决定”）。

## ARCHIVE

### 旧创新二：区域存在性感知的分割监督

**状态：首次 full 实验整体指标下降，已停用。** 设计为存在性辅助头 + 非空区域 Dice + 空区域 Top-k 惩罚，保留 CE；辅助项权重各 0.1，Top-k 比例 0.01。

| 指标 | Baseline | Full |
| --- | ---: | ---: |
| 验证 Mean Dice | 81.78% | 81.08% |
| 测试 Mean Dice | 83.19% | 82.74% |
| 验证非空 ET Dice | 75.09% | 77.88% |
| 验证空 ET 假阳性病例 | 2/5 | 5/5 |

非空 ET 有局部收益，但新增空患者假阳性使总体下降。仅完成 seed=42 的联合实验，未做单组件消融，不能确定具体归因。

依据：[EXP-002 结果与分析](EXPERIMENTS_BraTS.md)；损失理论参考：[Dice Loss 与空标签](https://arxiv.org/abs/2207.09521)。旧实现、配置和测试已移出活动代码并[归档](../VM-UNet-main/results/archives/region_presence_source_before_replacement.zip)，历史结果与原权重保留；旧模型推理使用导出的纯分割权重，见[退役说明](../VM-UNet-main/docs/REGION_PRESENCE.md)。

2026-11-11 分析：失分集中在空 ET 假阳性，ET 小体积后处理可能抵消；是否借此复活待用户决定，见决策记录。
