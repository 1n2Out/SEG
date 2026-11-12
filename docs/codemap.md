# 代码导航与修改约定

源码在 `VM-UNet-main/`，所有 Python 入口都在该目录下运行。相对路径一律按源码目录解析。

## 目录

```text
SEG/                          # git 仓库根目录
├── AGENTS.md  CLAUDE.md  STATE.md
├── docs/  research/  handoff/
├── datasets/BraTS2020/       # 原始数据（不入库）
└── VM-UNet-main/
    ├── run_brats.py          # 统一启动：train / resume / smoke / test / prepare
    ├── train_brats.py        # 底层入口：训练、恢复、评价、预测、smoke
    ├── collect_brats.py      # 把运行结果收集到 research/experiments/
    ├── prepare_brats.py      # 生成清单、校验数据、建缓存
    ├── plot_brats.py         # 离线绘图（不加载模型、不用 GPU）
    ├── check_local.py        # Windows CPU 冒烟检查
    ├── brats/                # 数据、损失、推理、指标、训练工具、来源记录
    │   └── innovations/      # 可选模型扩展，每个方法一个子包
    ├── models/vmunet/        # VMUNet 封装与 VSSM / SS2D 主干
    ├── configs/              # brats2020.json 为正式协议；子目录为各方法和筛选配置
    ├── analysis/             # 研究分析脚本（必须入库）
    ├── data/brats2020/       # 划分清单、数据审计（入库）；cache/（不入库）
    ├── archives/             # 已退役方案的源码归档
    ├── tests/                # CPU 测试；*_cuda.py 和 *_distributed.py 需要 B 端执行
    └── results/  logs/       # 运行产物（不入库）
```

## 按目标找代码

| 修改目标 | 位置 |
|---|---|
| 患者清单、缓存、几何检查 | `prepare_brats.py`、`brats/data.py` |
| 切片读取、增强、损失 | `brats/dataset.py`、`brats/losses.py` |
| 三维推理、指标、预测导出 | `brats/engine.py`、`brats/metrics.py` |
| 训练循环、DDP、恢复、选 best、热启动 | `train_brats.py`、`brats/training.py` |
| 启动、GPU 选择、来源检查、测试闸门 | `run_brats.py`、`brats/provenance.py` |
| 结果收集与总表 | `collect_brats.py` |
| 输入输出封装、预训练加载 | `models/vmunet/vmunet.py::VMUNet` |
| 编解码、跳跃连接、selective scan | `models/vmunet/vmamba.py::VSSM / VSSBlock / SS2D` |
| 模型扩展的注册与分派 | `train_brats.py::innovation_settings / build_model`，以及 `EXTENSIONS` |

## 新增模型扩展的约定

现有的 `detail_fusion` 和 `shallow_refinement` 都按以下方式实现，新方法照做：

1. **子包结构。** 在 `brats/innovations/<name>/` 下建子包：
   - `__init__.py`：负责配置解析和默认值，不导入 torch；
   - `block.py`：模块本体；
   - `model.py`：继承 `VMUNet` 的子类。
2. **注册。** 在 `train_brats.py` 的 `EXTENSIONS` 和分派函数中注册。配置用一个同名字段开启，`mode: "off"` 等同于 baseline。**一次运行只允许启用一个扩展。**
3. **不影响 baseline。** 不写扩展字段时，模型构建、初始化、随机数序列和优化器必须与 baseline 完全一致。新模块在主干之后构建，并隔离自己的随机数。
4. **新分支零初始化，以残差方式接入。** 初始输出要与 baseline 相同，这是能做热启动筛选的前提。参数集合保持固定，保证 DDP 正常工作。
5. **配套交付。** 包括：正式配置和筛选配置（相对 baseline 只多出模块字段和输出目录）、CPU 测试、CUDA 测试（交给 B 执行）、`docs/methods/<name>.md`。

## 修改禁区

- **主干结构。** 编码通道为 `[96,192,384,768]`，跳跃连接用相加，`skip_list[0]` 未被使用。改宽度或拓扑时，必须同时检查相加处的尺寸、上下采样和预训练权重映射。不要顺手删除会消耗随机数的旧初始化代码，否则随机序列会变。
- **数据与评价。** 不得修改划分、预处理、评价口径和空区域规则（见 [protocol](protocol.md)）。修改需要用户决定。
- **checkpoint 兼容。** 启用了扩展的 checkpoint 带有该扩展的权重，评价和续跑时必须保留对应模块。
- **Windows 端。** 不添加模拟 selective scan 的 CPU 后端，也不自动下载权重。
- **已退役方案。** 区域存在性监督的配置会被入口拒绝；源码只在 `archives/` 中，恢复时放到独立副本里，不覆盖现有代码。
