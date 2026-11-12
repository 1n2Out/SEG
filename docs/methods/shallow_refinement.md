# 方法：浅层细节回注 `shallow_refinement`

研究定位见 [IDEA-002](../../research/ideas/IDEA-002-shallow-refinement.md)：它只在 1/4 尺度补连接，作为 [detail_fusion](detail_fusion.md) 的"不提高分辨率"对照。实现代码在 `VM-UNet-main/brats/innovations/shallow_refinement/`。

## 配置

| `shallow_refinement.mode` | 行为 | 配置 |
|---|---|---|
| 不写该字段，或 `off` | 原 VMUNet | `configs/brats2020.json` |
| `simple` | `D′ = D + α·Conv1×1(S)` | `configs/shallow_refinement/simple.json` |
| `gated` | `D′ = D + α·G·R` | `configs/shallow_refinement/gated.json`；筛选用 `configs/screening/shallow_gated.json` |

```json
"shallow_refinement": {"mode": "gated", "hidden_channels": 24}
```

## 结构

- **输入。** S 是最浅层特征 `skip_list[0]`，D 是最后一级解码特征。两者都是 1/4 尺度、96 通道、BHWC 格式。
- **gated 分支：**
  1. S 经 `Conv1×1(96→24)` 和 GELU；
  2. 并行通过两路深度卷积 `DWConv3×3 + GELU` 和 `DWConv5×5 + GELU`；
  3. 两路拼接后经 `Conv1×1(48→96)` 得到 R；
  4. 门控 `G = sigmoid(Conv1×1([D, R]))`。
- **α 系数。** α 是逐通道的可学习系数，初始化为 0，其他卷积正常初始化。注意不能把投影层也初始化为零，否则分支无法开始学习。

| 模式 | 新增参数 | 总参数 |
|---|---:|---:|
| simple | 9,408 | 27,438,580 |
| gated（hidden=24） | 26,520 | 27,455,692 |

## 工程检查

```bash
python -m pytest -q tests/test_shallow_refinement.py
CUDA_VISIBLE_DEVICES=<卡> python -m pytest -q tests/test_shallow_refinement_cuda.py      # 仅 B 端
BRATS_TEST_DISTRIBUTED=1 python -m pytest -q tests/test_shallow_refinement_distributed.py
```

- **已知现象。** 当前环境下，同一模型、同一 seed 重复做 CUDA 反向，结果仍有微小数值差异，baseline 也一样。因此不能宣称"独立运行"或"中断后续跑"每一步都逐位相同。
- **移除。** 方法确认不再需要时，删除模块目录、两个配置、筛选配置和专用测试，同时删除本文。
