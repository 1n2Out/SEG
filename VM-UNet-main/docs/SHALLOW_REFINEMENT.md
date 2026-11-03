# 浅层细节回注：简单残差与多尺度语义门控

本方案替换已停用的区域存在性感知监督。使用原 VM-UNet 主干和原 `BraTSCEDiceLoss`，仅在最后一级解码特征与最终上采样之间插入可选模块。是否提点需要正式对照实验验证。

## 配置与启动

| 模式 | 行为 | 配置 |
| --- | --- | --- |
| 不含配置字段 / `off` | 原 VMUNet | `configs/brats2020.json` |
| `simple` | 浅层 1×1 投影 + 零初始化残差系数 | `configs/shallow_refinement/simple.json` |
| `gated` | 3×3/5×5 局部提取 + 语义门控 + 零初始化残差系数 | `configs/shallow_refinement/gated.json` |

```json
"shallow_refinement": {
  "mode": "gated",
  "hidden_channels": 24
}
```

`hidden_channels` 仅影响 gated 分支的压缩通道数；残差系数固定从零初始化，不提供额外调参开关。配置省略的默认值在读取时补齐并保存到 checkpoint，恢复训练仍要求相同配置。

两个实验配置仅比 baseline 增加模块设置，并分别使用 `results/brats2020_shallow_simple_*`、`results/brats2020_shallow_gated_*` 输出目录。患者清单、四模态输入、预训练文件、batch=16、100 轮、学习率、调度、数据增强、验证间隔与早停保持一致。新增模块不使用旧创新实验的训练权重作为初始化。

在 `VM-UNet-main/` 下执行：

```bash
# 预览配置和输出位置。
python run_brats.py train --config configs/shallow_refinement/gated.json --dry-run

# GPU 上检查真实前后向与一例验证。
python run_brats.py smoke --config configs/shallow_refinement/gated.json

# 正式训练完整门控版；启动简单版时将 gated.json 换为 simple.json。
python run_brats.py train --config configs/shallow_refinement/gated.json --background

# 使用同一配置和原目录续跑。
python run_brats.py resume --config configs/shallow_refinement/gated.json --run-dir results/实际训练目录 --background

# 方法固定后再评价测试集，run-dir 传训练目录。
python run_brats.py test --config configs/shallow_refinement/gated.json --run-dir results/实际训练目录
```

## 结构与初始化

输入为最浅层 `skip_list[0]` 特征 S 和最后一级解码特征 D。当前二者均为 BHWC、1/4 分辨率、96 通道；256×256 图像对应 64×64 特征。S 是 Patch Embedding 后的特征，仍通过原编码器参与计算；新模块增加其通向最后解码层的直接连接。

- **simple：** `D' = D + alpha * Conv1x1(S)`。
- **gated：** S 经 `Conv1x1(96→24) + GELU`，并行通过 `DWConv3x3 + GELU` 和 `DWConv5x5 + GELU`，拼接后以 `Conv1x1(48→96)` 得到 R；`G = sigmoid(Conv1x1(concat(D,R)))`；最后 `D' = D + alpha * G * R`。

alpha 为每通道一个可学习参数，初始全部为零；其他卷积正常初始化。第一个反向步骤可更新 alpha，其他分支获得零但非缺失梯度；alpha 非零后，分支卷积开始获得梯度。不能同时把残差投影也初始化为零，否则可能阻断启动学习。前向始终计算分支，不根据 alpha 值跳过路径，以保持 DDP 参数图稳定。

模块在 CPU 上构建于原主干之后，使用独立的临时随机数状态，保证相同 seed 下主干初始权重及后续 RNG 序列一致。零残差保持初始网络输出与 baseline 一致，这不保证训练后性能不下降。

两种模式在训练和推理时都返回原四类 logits，并都执行回注。损失始终是原 CE + WT/TC/ET Dice，不含存在性分类或 Top-k 负项，不新增辅助监督。原预训练加载器仍只加载 `vmunet.*`，新分支从上述初始化开始学习。

| 模式 | 新增参数 | 总参数（当前主干） |
| --- | ---: | ---: |
| baseline / off | 0 | 27,429,172 |
| simple | 9,408 | 27,438,580 |
| gated，hidden=24 | 26,520 | 27,455,692 |

以上是参数计数；训练及推理耗时增加量需实测。与旧辅助头不同，新分支参与推理，不能在训练后删除其权重并声称输出等价。

## 管理、消融与移除

实现集中在 [shallow_refinement/](../brats/innovations/shallow_refinement/)：`__init__.py` 负责配置，`block.py` 负责残差与门控，`model.py` 是 VMUNet 子类。原 `vmamba.py`、`vmunet.py`、损失、数据与三维评价实现不变；训练入口负责按配置选择模型。

最小对照为 baseline、simple、gated。simple 检验直接浅层连接是否有用，gated 检验完整多尺度与门控设计是否比简单连接更有效。只有观察到收益后，才进一步拆分多尺度或门控的独立贡献。

停用时用原 baseline 配置，或设 `mode: "off"`。彻底移除时，先保留需要复现实验的模块和配置，然后使用不含 `shallow_refinement` 字段的 baseline 配置，删除模块目录、两个实验配置和专用测试。训练入口的可选导入在 baseline 下不会执行；可进一步清理这些分支。

启用模式的 checkpoint 包含 `shallow_refinement.*` 权重，评价和恢复须保留匹配模块；不能通过关闭开关把它当作 baseline checkpoint。当前入口会拒绝旧 `region_presence` 配置，旧实验复查与权重导出见 [旧模块退役说明](REGION_PRESENCE.md)。历史实验结果和研究笔记未因替换代码而改写。

## 工程检查

```bash
python -m pytest -q tests/test_shallow_refinement.py
CUDA_VISIBLE_DEVICES=2 python -m pytest -q tests/test_shallow_refinement_cuda.py
BRATS_TEST_DISTRIBUTED=1 python -m pytest -q tests/test_shallow_refinement_distributed.py
```

检查覆盖配置隔离、原损失及梯度不变、零残差初始恒等、两步后分支学习、真实 CUDA/AMP、checkpoint/优化器/AMP 状态恢复，以及两进程 DDP 中零残差分支能正常启动。无 CUDA 的环境跳过 GPU 检查；Gloo 检查需显式启用。工程验收不作为正式效果实验。

恢复检查要求模型、优化器及 AMP 状态精确一致，且加载后前向输出一致、能继续更新。当前环境中，同一个模型在相同随机种子下重复 CUDA 反向仍存在微小数值差异，原 baseline 也观察到该现象；因此不宣称独立运行或中断恢复后的所有后续训练步均逐位相同。

2026-11-03 工程验收完成：113 项 CPU/回归检查、3 项真实 CUDA 检查和 1 项两进程 Gloo 检查通过。另在 GPU 2 上使用独立三病例清单完成 gated 两轮训练、第 1 轮后中断恢复、checkpoint 重新加载验证，以及 simple 的真实前后向与单病例验证。gated 重新加载的区域指标与保存前完全一致，alpha 已从零学到非零值。小清单仅取自原训练/验证集，未评价正式测试集。详见 [验收报告](../results/verification/shallow_refinement_pipeline/verification_report.json)。
