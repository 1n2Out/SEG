# 方法：多分辨率细节回注 `detail_fusion`

研究动机与状态见 [IDEA-001](../../research/ideas/IDEA-001-detail-fusion.md)；两个叠加选项见 [IDEA-005](../../research/ideas/IDEA-005-boundary-supervision.md)（边界辅助监督）和 [IDEA-006](../../research/ideas/IDEA-006-slice-context.md)（相邻切片）。本文只写结构、配置和工程检查。实现代码在 `VM-UNet-main/brats/innovations/detail_fusion/`。

## 配置

```json
"detail_fusion": {"mode": "gated", "full_channels": 16, "half_channels": 32,
                  "boundary_weight": 0.0, "context_slices": 0}
```

| 字段 | 取值 | 含义 |
|---|---|---|
| `mode` | `off`（或不写该字段）/ `concat` / `gated` | `off` 为原 VMUNet；`concat` 补出 1/2、1/1 细节支路并与解码语义拼接；`gated` 在拼接前先用语义门控筛选细节 |
| `full_channels`、`half_channels` | 整数，至少为 2 | 1/1、1/2 尺度的通道数 |
| `boundary_weight` | 非负数，0 表示关闭 | 边界辅助损失的权重（IDEA-005） |
| `context_slices` | 0 或 1 | 细节支路额外读取的上下相邻切片数（IDEA-006） |

- 两个选项只在 `concat` 或 `gated` 下可用，默认都关闭，此时模型与 EXP-006 完全相同。
- 省略的字段会补上默认值，并写入 checkpoint。

| 配置 | 用途 |
|---|---|
| `configs/detail_fusion/concat.json`、`gated.json` | 完整训练；相对 baseline 只多出模块字段和输出目录 |
| `configs/detail_fusion/gated-boundary.json`、`gated-context.json`、`gated-boundary-context.json` | 叠加件的完整训练；相对 `gated.json` 只多出选项和输出目录 |
| `configs/screening/detail_gated.json`、`detail_concat.json` | IDEA-001 的热启动筛选，来源为 EXP-001 |
| `configs/screening/stack-control.json`、`stack-boundary.json`、`stack-context.json`、`stack-boundary-context.json` | 叠加件的热启动筛选，来源为 EXP-006；`control` 不加选项 |

筛选配置的约定见 [protocol](../protocol.md) 中的"热启动筛选"，`pretrained` 必须为 `null`。叠加筛选在 `warm_start.new_parameters` 中列出新部件（`detail_fusion.boundary_head.`、`detail_fusion.context.`），只有它们使用新模块学习率；`control` 列表为空。来源中已有的张量不能声明为新，来源缺少的张量也必须在声明之内，否则启动报错。

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

### 选项一：边界辅助监督（`boundary_weight > 0`）

- 在 1/1 融合后的特征上加一个 1×1 卷积头，输出 WT、TC、ET 三张边界 logits，初始化为零。
- 边界真值在训练时由增强后的标签现算：每个区域做 3×3 膨胀减 3×3 腐蚀，得到内外各 1 像素宽的边界带。
- 总损失为 `CE + 区域 Dice + w × (BCE + 整批 soft Dice)`，w 取 0.5，不调参。整批 Dice 是因为多数单张切片没有边界。
- 训练态模型返回 `(logits, boundary_logits)`，由 `BoundaryAuxLoss` 处理；评价和推理只返回 logits，推理计算量不变。`history.jsonl` 多出 `train_boundary_loss` 一项。

### 选项二：相邻切片（`context_slices = 1`）

- 输入改为 12 通道：当前切片在前，其后依次为上一张、下一张；超出体积时复制边缘切片。训练的 `BraTSSlices` 和三维推理的 `predict_volume` 共用 `brats/dataset.py::stack_slices`，保证堆叠方式一致。
- 主干只取前 4 个通道，因此主干、预训练映射和 baseline 都不受影响。
- 细节支路中，12 通道输入经一个残差卷积块和零初始化的 1×1 投影，加到 F 上；`stem_full` 仍只读当前切片，EXP-006 的权重可以直接加载。

| 模式 | 新增参数 | 总参数 | 卷积计算量（每张 256² 切片） |
|---|---:|---:|---:|
| off | 0 | 27,429,172 | 0 |
| concat | 69,476 | 27,498,648 | ≈1.49 GMACs |
| gated | 70,462 | 27,499,634 | ≈1.52 GMACs |
| gated + 边界 | 70,513 | 27,499,685 | 推理不变 |
| gated + 相邻切片 | 75,054 | 27,504,226 | 另加约 0.3 GMACs |
| gated + 两者 | 75,105 | 27,504,277 | 同上 |

两个选项相对 gated 的新增张量数分别为 2 和 11，两者都加为 13，热启动日志中的 `new=` 应与之一致。实际耗时以运行记录中的 `train_seconds` 为准。

## 工程检查

```bash
python -m pytest -q tests/test_detail_fusion.py tests/test_warm_start.py tests/test_brats_torch.py   # A、B 两端都要跑
CUDA_VISIBLE_DEVICES=<卡> python -m pytest -q tests/test_detail_fusion_cuda.py                   # 仅 B 端
BRATS_TEST_DISTRIBUTED=1 python -m pytest -q tests/test_detail_fusion_distributed.py              # A 端 CPU 也能跑
```

- **CPU 检查** 覆盖：配置校验、零初始化下输出不变、两步内分支开始学习、参数量、逐切片独立、原损失不变、baseline 的随机数序列不变；选项部分覆盖共享支路初始化不变、边界带真值、辅助损失组成、相邻切片的堆叠顺序与边缘、增强对所有切片一致、推理与训练堆叠一致；热启动部分覆盖划分校验、新张量声明、学习率分组和启动器预检。
- **CUDA 检查** 覆盖：真实 selective scan 加 AMP 时初始输出逐位相同、两步学习、checkpoint 恢复、热启动后输出与来源模型相同，以及从训练过的 gated 模型叠加两个选项后输出不变、新部件开始学习。
- **双进程检查** 覆盖原支路和两个选项都打开时的 DDP 梯度同步。

## 消融与移除

- **消融行** 按叠加顺序排列：baseline、gated（EXP-006）、gated 加单个选项、gated 加两个选项，运行清单见 [TASK-004](../../handoff/task-004-detail-stacking.md)。
- **停用** 时改用 baseline 配置即可；停用某个选项时把它设回默认值。
- **彻底移除** 时，删除模块目录、相关配置、专用测试，以及 `train_brats.py` 里的分派分支、`input_context` 和边界损失分派，同时删除本文。相邻切片堆叠函数属于数据层，没有方法使用时可一并删除。
