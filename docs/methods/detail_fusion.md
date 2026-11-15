# 方法：多分辨率细节回注 `detail_fusion`

研究动机与状态见 [IDEA-001](../../research/ideas/IDEA-001-detail-fusion.md)；两个选项部件见 [IDEA-007](../../research/ideas/IDEA-007-deep-supervision.md)（深监督）和 [IDEA-008](../../research/ideas/IDEA-008-input-stem.md)（输入端补强）。本文只写结构、配置和工程检查。实现代码在 `VM-UNet-main/brats/innovations/detail_fusion/`。

## 配置

```json
"detail_fusion": {"mode": "gated", "full_channels": 16, "half_channels": 32,
                  "deep_supervision": false, "input_stem": false}
```

| 字段 | 取值 | 含义 |
|---|---|---|
| `mode` | `off`（或不写该字段）/ `concat` / `gated` | `off` 为原 VMUNet；`concat` 补出 1/2、1/1 细节支路并与解码语义拼接；`gated` 在拼接前先用语义门控筛选细节 |
| `full_channels`、`half_channels` | 整数，至少为 2 | 1/1、1/2 尺度的通道数 |
| `deep_supervision` | `true` / `false` | 训练时对多个尺度附加辅助损失（IDEA-007）；推理不变 |
| `input_stem` | `true` / `false` | 把 1/2 细节降到 1/4，零初始化后加到 patch embedding 上（IDEA-008） |

省略的字段会补上默认值，并写入 checkpoint。两个选项只能在 `concat` 或 `gated` 下开启；都关闭时模型与 EXP-006 逐位相同。

| 配置 | 用途 |
|---|---|
| `configs/detail_fusion/concat.json`、`gated.json` | 完整训练；相对 baseline 只多出模块字段和输出目录 |
| `configs/detail_fusion/gated-ds.json`、`gated-stem.json`、`gated-ds-stem.json` | 完整训练；相对 `gated.json` 只多出选项和输出目录 |
| `configs/screening/detail_gated.json`、`detail_concat.json` | IDEA-001 的热启动筛选，来源为 EXP-001 |
| `configs/screening/stack-control.json` | 在 EXP-006 上叠加新部件时的筛选对照：从 EXP-006 热启动，不加新部件 |

筛选配置的约定见 [protocol](../protocol.md) 中的"热启动筛选"，`pretrained` 必须为 `null`。在 EXP-006 上叠加时，在 `warm_start.new_parameters` 中列出新部件的前缀，只有它们使用新模块学习率；对照的列表为空。新部件前缀：深监督为 `detail_fusion.aux_half.`、`detail_fusion.aux_decoder.`，输入端补强为 `detail_fusion.input_stem.`。来源中已有的张量不能声明为新，来源缺少的张量也必须在声明之内，否则启动报错。

## 结构

记输入为 X（4 通道，256²），最终 1/4 尺度的解码特征为 D（64²，96 通道，即 `forward_final` 之前的特征）。

1. **细节支路。** X 经残差卷积块得到 1/1 细节 F（16 通道）；F 再经步长 2 的残差卷积块得到 1/2 细节 H（32 通道）。
2. **1/2 尺度融合。** D 经 1×1 卷积和 2 倍像素重排上采样得到 S₂。gated 模式下先做 `H ← H·σ(LGAG(S₂, H))`，再用残差卷积块融合 [S₂, H]。
3. **1/1 尺度融合。** 把上一步结果用同样方式上采样得到 S₁，再融合 [S₁, F]；gated 模式同样先做门控。
4. **输出。** 1×1 卷积输出 4 类残差 logits，加到原 logits 上。

- **LGAG 门控**（来自 EMCAD，CVPR 2024）：语义和细节各经过一个分组 3×3 卷积（输出 C/2 通道），相加后 ReLU，再经 1×1 卷积和 sigmoid 得到一张空间权重图。相对原版有两处改动：
  - BatchNorm 换成 GroupNorm，使结果与 batch 组成无关；
  - sigmoid 之前不做归一化，使门控能把整张切片整体关闭。
- **卷积块。** 两层 3×3 卷积加残差投影，GroupNorm + GELU。
- **初始化。** 只有最后的 1×1 输出卷积初始化为零，所以初始输出与 baseline 逐位相同。第一步只有输出卷积会更新，之后整个支路都开始学习。

### 选项一：深监督 `deep_supervision`

训练时在 5 处使用同一种损失（CE + WT/TC/ET 区域 Dice）。权重相对最终输出逐级减半：

| 监督点 | 来源 | 权重 |
|---|---|---:|
| 最终输出 | 原 logits + 残差 logits | 1 |
| 1/2 | 1/2 融合后的特征，经 1×1 辅助头 | 1/2 |
| 粗预测（1/4 token） | 主干自己的 logits，即加残差之前的 `forward_final` 输出 | 1/4 |
| 1/8、1/16 | 解码器对应阶段的输出，各经 1×1 辅助头 | 1/8、1/16 |

- 1/32 不监督。nnU-Net 还会把权重归一化到和为 1，这只差一个常数因子，对 AdamW 的更新基本没有影响。
- 辅助 logits 先双线性上采样到标签尺寸再算损失（与 mmsegmentation 的辅助头做法相同），标签不下采样，薄层 ET 不会因此走样。
- 训练态模型返回 `(logits, (1/2, 粗预测, 1/8, 1/16))`，评价态只返回 logits。辅助头不参与推理，推理计算量不变。
- 日志中的损失分量为 `ce`、`dice`（最终输出）和 `deep`（加权后的辅助项之和）。

### 选项二：输入端补强 `input_stem`

1/2 细节 H 经步长 2 的残差卷积块（32→64 通道）降到 1/4，再经零初始化的 1×1 卷积投影到 96 通道，加到 patch embedding 的输出上，之后照常进入编码器。

- H 由细节支路和输入端共享，两条路径的梯度都会回到细节支路的卷积。
- 投影初始为零，所以初始输出与 EXP-006 结构逐位相同。第一步只有投影更新，之后整个补强块开始学习。
- 解码器不使用 `skip_list[0]`（见 [codemap](../codemap.md)），所以注入的特征只经编码器起作用：先过第一阶段的 VSS 块，再通过 1/8 及更深的跳跃连接影响解码。

### 参数量与计算量

| 配置 | 新增参数（相对 baseline） | 卷积计算量（每张 256² 切片） |
|---|---:|---:|
| concat | 69,476 | ≈1.49 GMACs |
| gated | 70,462 | ≈1.52 GMACs |
| gated + 深监督 | 72,906（辅助头 2,444 只在训练时使用） | 推理同 gated；训练另加约 0.003 GMACs |
| gated + 输入端补强 | 134,430 | ≈1.78 GMACs |
| gated + 两者 | 136,874（辅助头 2,444 只在训练时使用） | ≈1.78 GMACs |

baseline 总参数为 27,429,172。实际耗时以运行记录中的 `train_seconds` 为准。

## 工程检查

```bash
python -m pytest -q tests/test_detail_fusion.py tests/test_warm_start.py          # A、B 两端都要跑
CUDA_VISIBLE_DEVICES=<卡> python -m pytest -q tests/test_detail_fusion_cuda.py      # 仅 B 端
BRATS_TEST_DISTRIBUTED=1 python -m pytest -q tests/test_detail_fusion_distributed.py # A 端 CPU 也能跑
```

- **CPU 检查** 覆盖：配置校验、零初始化下输出不变、两步内分支开始学习、参数量、逐切片独立、原损失不变、baseline 的随机数序列不变；热启动部分覆盖划分校验、新张量声明、学习率分组和启动器预检。选项部分覆盖：
  - 开启选项不改变共享支路的初始化，两个选项互不影响对方的初始化；
  - 投影为零时编码器与原实现逐位相同；自写的编码、解码循环与原实现一致；
  - 深监督不改变输出，训练态返回 4 个尺度，损失按权重在标签尺寸上计算；
  - 训练时每个支路参数都有梯度（DDP 要求参数集合固定）。
- **CUDA 检查** 覆盖：真实 selective scan 加 AMP 时初始输出逐位相同、两步学习、checkpoint 恢复，以及热启动后输出与来源模型相同；三种选项组合都覆盖，训练态在相同 drop-path 随机数下与 gated 输出逐位相同。
- **双进程检查** 覆盖零初始化支路，以及输入端投影和辅助头在 DDP 下的梯度同步。

## 消融与移除

- **消融** 的设计与可宣称的结论见各 IDEA。
- **停用** 时改用 baseline 配置即可；停用某个选项时把它设为 `false`。
- **彻底移除** 时，删除模块目录、相关配置、专用测试，以及 `train_brats.py` 里的分派分支，同时删除本文。
