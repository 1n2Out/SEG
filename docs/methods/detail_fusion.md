# 方法：多分辨率细节回注 `detail_fusion`

本方法为第二项工作的正式创新点一，定稿模型为 [EXP-006](../../research/experiments/EXP-006/report.md)。研究依据和论文表述见 [IDEA-001](../../research/ideas/IDEA-001-detail-fusion.md)。本文只写结构、配置和工程检查。实现代码在 `VM-UNet-main/brats/innovations/detail_fusion/`。

## 配置

```json
"detail_fusion": {"mode": "gated", "full_channels": 16, "half_channels": 32}
```

| 字段 | 取值 | 含义 |
|---|---|---|
| `mode` | `off`（或不写该字段）/ `concat` / `gated` | `off` 为原 VMUNet；`concat` 补出 1/2、1/1 细节支路并与解码语义拼接；`gated` 在拼接前先用语义门控筛选细节 |
| `full_channels`、`half_channels` | 整数，至少为 2 | 1/1、1/2 尺度的通道数 |
| `contralateral` | 可选布尔值，默认关闭 | gated 模式的半分辨率[对侧参照分支](contralateral-reference.md) |

省略的字段会补上默认值，并写入 checkpoint。已退役的 `deep_supervision`、`input_stem`（IDEA-007/008）只接受 `false` 并被忽略，以便加载 EXP-014 等由当时代码训练的 checkpoint。

| 配置 | 用途 |
|---|---|
| `configs/detail_fusion/gated.json`、`configs/detail_fusion/concat.json` | 分别对应定稿模型 EXP-006 和门控消融 EXP-014 |
| `configs/screening/baseline.json`、`detail_concat.json`、`detail_gated.json` | IDEA-001 的三组热启动筛选，均从 EXP-001 出发 |

筛选配置和学习率分组遵循[热启动筛选协议](../protocol.md)。

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

| 模式 | 新增参数 | 总参数 | 卷积计算量（每张 256² 切片） |
|---|---:|---:|---:|
| off | 0 | 27,429,172 | 0 |
| concat | 69,476 | 27,498,648 | ≈1.49 GMACs |
| gated | 70,462 | 27,499,634 | ≈1.52 GMACs |

实际耗时以运行记录中的 `train_seconds` 为准。

## 工程检查

```bash
python -m pytest -q tests/test_detail_fusion.py tests/test_warm_start.py          # A、B 两端都要跑
CUDA_VISIBLE_DEVICES=<卡> python -m pytest -q tests/test_detail_fusion_cuda.py      # 仅 B 端
BRATS_TEST_DISTRIBUTED=1 python -m pytest -q tests/test_detail_fusion_distributed.py # A 端 CPU 也能跑
```

- **CPU 检查** 覆盖：配置校验、零初始化下输出不变、两步内分支开始学习、参数量、逐切片独立、原损失不变、baseline 的随机数序列不变；热启动部分覆盖划分校验、新张量声明、学习率分组和启动器预检。
- **CUDA 检查** 覆盖：真实 selective scan 加 AMP 时初始输出逐位相同、两步学习、checkpoint 恢复，以及热启动后输出与来源模型相同。
- **双进程检查** 覆盖零初始化支路在 DDP 下的梯度同步。

## 消融与移除

- **消融** 的设计与可宣称的结论见 [IDEA-001](../../research/ideas/IDEA-001-detail-fusion.md)。
- **停用** 时改用 baseline 配置即可。
- **彻底移除** 时，删除模块目录、相关配置、专用测试，以及 `train_brats.py` 里的分派分支，同时删除本文。
