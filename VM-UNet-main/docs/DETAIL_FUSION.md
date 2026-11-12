# 多分辨率细节回注：语义门控融合（detail_fusion）

本方案是浅层细节回注的升级版：shallow_refinement 只在 1/4 分辨率补连接，detail_fusion 补上 VM-UNet 缺失的 1/2 和 1/1 尺度。使用原 VM-UNet 主干和原 `BraTSCEDiceLoss`，新分支以零初始化残差 logits 接入，训练和推理时都执行。是否提点需要服务器实验验证；动机、决策和筛选结论记录在 [INNOVATIONS.md](../../RESEARCH_NOTES/INNOVATIONS.md)。

## 动机

VM-UNet 只通过 4×4 patch token 看图像：输入端是步长 4 的卷积，输出端是“线性层 + 像素重排 + LN”的 4 倍扩展，之后只接 1×1 卷积。全分辨率上没有任何空间卷积，每个 4×4 输出块只由一个 token 决定。

训练/验证集标签统计（不含 test）显示 ET 是薄层结构：约 53% 的 ET 体素距边界不超过 1 个像素，约 78% 不超过 2 个像素。若每个 4×4 块只能判一类，ET Dice 上限约 80%，2×2 块约 90%；WT/TC 在 4×4 块下约 92–93%。baseline 非空 ET Dice（75–80%）已接近 4×4 量级，1/4 分辨率可能是 ET 的瓶颈。统计脚本与结果不在本仓库，数值见研究笔记。

## 配置与启动

| 模式 | 行为 | 正式配置 |
| --- | --- | --- |
| 不含配置字段 / `off` | 原 VMUNet | `configs/brats2020.json` |
| `concat` | 1/2、1/1 细节支路，与解码语义直接拼接融合 | `configs/detail_fusion/concat.json` |
| `gated` | 同上，拼接前用 LGAG 式语义门控筛选细节 | `configs/detail_fusion/gated.json` |

```json
"detail_fusion": {
  "mode": "gated",
  "full_channels": 16,
  "half_channels": 32
}
```

`full_channels`、`half_channels` 是 1/1、1/2 尺度的通道数，至少为 2。省略的默认值在读取时补齐并写入 checkpoint，恢复训练仍要求相同配置。一次运行只能配置一个模型扩展；同时写 `shallow_refinement` 和 `detail_fusion` 会报错。

正式配置只比 baseline 多出模块字段和输出目录，其余与 EXP-001 协议一致：同一预训练文件、batch=16、100 轮、学习率、调度、增强、验证间隔与早停。

## 热启动筛选

正式训练每次约 11 小时。先用热启动筛选决定哪些变体值得完整训练：从 baseline 最佳权重出发，挂上零初始化模块微调 10 轮。零初始化保证起点就是 baseline 最佳模型的输出。筛选结果只用于决定是否投入完整训练，论文数字仍须来自正式协议。

| 筛选配置 | 内容 |
| --- | --- |
| `configs/screening/baseline.json` | 对照组：不挂模块，同样微调 10 轮，排除“多训几轮”的影响 |
| `configs/screening/detail_gated.json` | 主候选 |
| `configs/screening/detail_concat.json` | 门控消融 |
| `configs/screening/shallow_gated.json` | 可选：只补 1/4 尺度的现有方案 |

四个筛选配置共用同一协议，仅模块和输出目录不同：

- `warm_start.checkpoint` 指向 EXP-001 的 `results/brats2020_1102_1734/best.pth`（第 55 轮，验证 Mean Dice 81.78%）。相对路径从源码目录解析。
- 原有参数学习率 1e-5，新模块参数学习率 `new_parameter_lr`=1e-4，两组都按余弦在 10 轮内降到 0。
- `pretrained` 必须为 null：热启动已加载全部原有参数，与预训练初始化互斥。
- 第 1、5、10 轮验证；早停设置沿用但 10 轮内不会触发。

热启动要求 checkpoint 的患者划分校验和与当前清单一致；只允许新模块参数缺失，其余缺失、多余或形状不符都直接报错。加载报告（来源路径、SHA256、来源轮次与分数、新参数列表）写入 `run.json` 的 `warm_start_report`。恢复训练和测试使用本次运行自己的 checkpoint，不再需要热启动文件。

在 `VM-UNet-main/` 下执行。GPU 在各配置的 `launch.gpus` 中修改：每个运行只用一张卡，保持与 EXP-001 相同的 batch=16；不同运行可以放在不同卡上并行，但不要多个运行挤在同一张卡上，否则 `train_seconds` 无法用来比较耗时。

```bash
# 预览命令；正式启动前在 GPU 上检查一次热启动加载与前后向。
python run_brats.py train --config configs/screening/detail_gated.json --dry-run
python run_brats.py smoke --config configs/screening/detail_gated.json

# 对照组与主候选必跑，门控消融建议跑，shallow_gated 可选。
python run_brats.py train --config configs/screening/baseline.json --background
python run_brats.py train --config configs/screening/detail_gated.json --background
python run_brats.py train --config configs/screening/detail_concat.json --background
python run_brats.py train --config configs/screening/shallow_gated.json --background
```

判读：比较各运行与对照组在同一轮次的验证 Mean Dice、三区域 Dice 和 HD95，并用逐患者 CSV 看配对差。主候选比对照组高 0.5 个百分点以上，且 ET 与 HD95 方向一致，才送去完整训练；55 例、单 seed 下 0.5 以内基本是噪声。`history.jsonl` 的 `train_seconds` 同时给出实际耗时开销。

## 正式训练

```bash
python run_brats.py smoke --config configs/detail_fusion/gated.json
python run_brats.py train --config configs/detail_fusion/gated.json --background
python run_brats.py resume --config configs/detail_fusion/gated.json --run-dir results/实际训练目录 --background
# 方法固定后再评价测试集。
python run_brats.py test --config configs/detail_fusion/gated.json --run-dir results/实际训练目录
```

## 结构与初始化

记输入图像为 X（4 通道、256²），最终 1/4 解码特征为 D（64²、96 通道，即 `forward_final` 之前的特征）。

1. 细节支路：X 经残差卷积块得到 1/1 细节 F（16 通道），再经步长 2 的残差卷积块得到 1/2 细节 H（32 通道）。
2. 1/2 融合：D 经 1×1 卷积和 2 倍像素重排上采样为 S₂（32 通道）；gated 模式下 H ← H·σ(门控(S₂, H))；再以残差卷积块融合 [S₂, H]。
3. 1/1 融合：上一步结果同样上采样为 S₁（16 通道）；gated 模式下 F ← F·σ(门控(S₁, F))；再融合 [S₁, F]。
4. 1×1 卷积输出四类残差 logits，与原 logits 相加。

门控沿用 EMCAD（CVPR 2024）LGAG 的形式：语义与细节各经 C/2 输出、C/2 分组的 3×3 卷积，相加后 ReLU，再经 1×1 卷积和 sigmoid 得到一张空间权重图。改动有两处：BatchNorm 换成 GroupNorm，使结果与 batch 组成无关；sigmoid 前不做归一化，使整张切片的门控可以整体关闭。卷积块均为两层 3×3 卷积加残差投影，GroupNorm + GELU。

只有最后的 1×1 输出卷积初始化为零，其余正常初始化：第一步只有输出卷积获得梯度，之后整个支路开始学习。分支前向始终执行，参数集合固定，便于 DDP。模块在原主干之后构建并隔离随机数，相同 seed 下主干初始权重及后续随机序列与 baseline 一致；零残差使初始输出与 baseline 相同，但不保证训练后不下降。

| 模式 | 新增参数 | 总参数（当前主干） |
| --- | ---: | ---: |
| baseline / off | 0 | 27,429,172 |
| concat | 69,476 | 27,498,648 |
| gated | 70,462 | 27,499,634 |

新增计算主要在 1/1、1/2 尺度的 3×3 卷积：按卷积层统计，每张 256² 切片 concat 约 1.49 GMACs、gated 约 1.52 GMACs（不含归一化和激活）。训练和推理的实际耗时增加量需在服务器实测。

## 管理、消融与移除

实现集中在 [detail_fusion/](../brats/innovations/detail_fusion/)：`__init__.py` 负责配置（不依赖 PyTorch），`block.py` 是细节支路与门控，`model.py` 是 VMUNet 子类。原 `vmamba.py`、`vmunet.py`、损失、数据与三维评价未改。训练入口新增模型扩展分派、热启动加载和新参数学习率分组；不配置这些字段时，baseline 的模型构建、初始化和优化器与修改前一致。

最小对照为 baseline、concat、gated：concat 检验多分辨率细节本身，gated 检验语义门控。有收益后再考虑只保留 1/2 或 1/1 的消融。

停用时用 baseline 配置或设 `mode: "off"`。启用模式的 checkpoint 含 `detail_fusion.*` 权重，评价和恢复须保留模块。彻底移除时，先保留需要复现实验的代码和配置，再删除模块目录、`configs/detail_fusion/`、相关筛选配置和专用测试，并移除 `train_brats.py` 中的分派分支。

## 工程检查

```bash
python -m pytest -q tests/test_detail_fusion.py tests/test_warm_start.py
CUDA_VISIBLE_DEVICES=2 python -m pytest -q tests/test_detail_fusion_cuda.py
BRATS_TEST_DISTRIBUTED=1 python -m pytest -q tests/test_detail_fusion_distributed.py
```

CPU 检查覆盖：配置校验与默认值、正式与筛选配置只差应有字段、一次只允许一个扩展、零初始化恒等与两步学习、参数量、逐切片独立性、原损失不变、构建时 baseline 权重与随机数一致、前向接线；热启动的划分校验、缺失/多余/形状错误、新参数学习率分组和启动器预检。CUDA 检查在真实 selective scan 与 AMP 下核对初始输出逐位相同、两步学习、checkpoint 恢复，以及热启动后输出与来源模型逐位相同。

2026-11-12 在隔离的 Linux CPU 环境（Python 3.10、torch 1.13.1+cpu，与 Windows `vmunet-local` 同版本）中，新增 CPU 测试和两进程 Gloo 测试通过，已有回归测试结果见 INNOVATIONS.md。Windows `vmunet-local`、真实 CUDA 测试、GPU smoke 和服务器训练尚未执行。
