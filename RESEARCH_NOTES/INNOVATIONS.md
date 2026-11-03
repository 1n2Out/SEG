# 创新点

更新：2026-11-03。先验证收益，再用消融解释机制；工程验收不等于分割效果提升。

## CURRENT

| 方向 | 状态 | 优先级 |
| --- | --- | --- |
| 创新一：区域条件的模态差异补偿 | 候选，未实现 | 备选 |
| 创新二（新版）：浅层细节回注与语义门控 | 已实现、工程验收通过，效果待验证 | 当前优先 |

公共协议沿用 [EXP-001 baseline](EXPERIMENTS_BraTS.md)：二维四模态输入、三维患者级评价，固定 259/55/55 划分；同一预训练初始化、batch=16、100 轮，按验证 Mean Dice 选 best。选型依据验证集，方案固定后再评价测试集。

### 创新一：区域条件的模态差异补偿

#### 1. Research Motivation

四模态在 Patch Embedding 阶段融合，后续不再显式区分模态。拟检验：补充模态特有信息，能否改善 TC、ET 识别。早期融合导致信息损失尚未被证实。

#### 2. Core Idea

主干学习融合语义，轻量分支保留模态差异，由解码区域预测决定补偿位置和强度。

#### 3. Theoretical Basis

[RFNet（ICCV 2021）](https://openaccess.thecvf.com/content/ICCV2021/html/Ding_RFNet_Region-Aware_Fusion_Network_for_Incomplete_Multi-Modal_Brain_Tumor_Segmentation_ICCV_2021_paper.html)及[子区域感知模态融合（2026）](https://arxiv.org/abs/2601.15734)已有区域模态加权设计。拟研究差异是面向已有融合主干的残差补偿，原创性需进一步核查。

#### 4. Technical Design

四模态分别经共享轻量卷积，计算各自相对均值的特征差异；用解码器预测的 WT/TC/ET 概率进行空间加权，在一个解码尺度残差注入。训练和推理均使用预测区域条件。

最小对照：baseline、普通模态加权、去掉区域条件、完整模块；同时比较参数量与耗时。

#### 5. Research Conclusion

尚无实验结论。暂缓实现，优先验证新版创新二。

### 创新二（新版）：浅层细节回注与语义门控

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

下一步比较 baseline、simple、gated 的完整训练最佳验证结果，关注 Mean Dice、各区域 Dice/HD95，以及 ET 空患者假阳性和非空患者召回。若 gated 不优于 simple，则优先保留简单方案；有收益后再拆分组件消融并做多 seed 复验。

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
