# VM-UNet 脑肿瘤分割实验

本项目以 [VM-UNet](https://arxiv.org/abs/2402.02491) 为 baseline，服务于硕士论文第二项工作。当前先在 **BraTS2020** 上确定创新点并开展实验，第二数据集根据创新点另行选择。

## 当前状态

- 采用 **2D 四模态切片训练、3D 患者级评价**：同一轴向层的 T1/T1ce/T2/FLAIR 构成四通道输入，不使用相邻切片。
- 已完成 baseline 和区域存在性感知 full 的各 100 轮正式训练及本地测试。旧创新方案已停用。浅层细节回注提供 simple/gated 两种配置；2026-11-12 新增多分辨率细节回注（detail_fusion，concat/gated）及热启动筛选配置，当前优先筛选 detail_fusion gated。两者正式效果均待验证。
- 仅保留 369 例有标注病例，按患者固定划分为 train/val/test = 259/55/55；125 例官方无标签 validation 数据及专属清单已删除。
- 本机完整 369 例 v2 缓存已准备并核对，约 29.65 GB，保留用于后续训练和评价。
- Windows CPU 开发链路和 Linux CUDA/训练恢复链路已有验收记录；浅层细节回注的小清单工程检查见 [使用与验收说明](docs/SHALLOW_REFINEMENT.md)。工程验收与正式实验分别记录。
- 当前 baseline/simple/gated 配置共用 GPU 2、batch=16、最多 100 轮、AMP 和 `../../package/vmamba_small_e238_ema.pth` 预训练初始化；该文件在当前工作区存在。迁移时需同时携带或重新指定路径，不自动下载。

## 阅读与使用

| 内容 | 唯一维护入口 |
|---|---|
| AI 项目导航、修改边界 | [PROJECT_CONTEXT.md](../PROJECT_CONTEXT.md) |
| BraTS 数据、损失、评价协议和服务器命令 | [BRATS2020.md](BRATS2020.md) |
| Windows 环境重建、日常检查及验收记录 | [WINDOWS_LOCAL.md](WINDOWS_LOCAL.md) |
| 病例图、数据分布图、训练曲线及绘图配置 | [离线绘图指南](docs/PLOTTING.md) |
| 当前浅层细节回注方案与启动命令 | [SHALLOW_REFINEMENT.md](docs/SHALLOW_REFINEMENT.md) |
| 多分辨率细节回注、热启动筛选与启动命令 | [DETAIL_FUSION.md](docs/DETAIL_FUSION.md) |
| 正式实验记录与结果索引 | [EXPERIMENTS_BraTS.md](../RESEARCH_NOTES/EXPERIMENTS_BraTS.md) |
| 旧区域存在性方案、权重兼容与源码恢复 | [REGION_PRESENCE.md](docs/REGION_PRESENCE.md) |
| 源码归档与完整性校验 | [archives/README.md](archives/README.md) |
| 原论文、旧实验路径及资源链接 | [原仓库参考说明](docs/UPSTREAM_REFERENCE.md) |

Windows 本地仅做环境配置、代码修改和非训练验证；训练、完整模型评价/预测及 `--mode smoke` 在 Linux 执行。本机 Linux 环境为 `vm_unet`，运行时进入本源码目录。不自动下载权重。

当前启动参数以 `configs/` 为准，已有实验以各运行目录的 `config.json`、`run.json` 和实际结果为准；上层导航与个人指南中的早期 60 轮、随机初始化描述属于历史记录。

## 工作区与源码目录

以下结构以 `Seg/` 为根；运行 Python 入口时进入源码目录 `VM-UNet-main/`。预训练文件位于 `Seg/` 的上一级 `package/` 中。

```text
Seg/
├── PROJECT_CONTEXT.md
├── RESEARCH_NOTES/           # 研究笔记；按其 README 约定维护
├── datasets/                 # 原始数据：BraTS2020/
└── VM-UNet-main/             # 源码目录
    ├── run_brats.py          # 统一启动、恢复与测试
    ├── prepare_brats.py      # 清单、校验与缓存准备
    ├── train_brats.py        # 训练、恢复、评价和预测
    ├── check_local.py        # Windows CPU 非训练检查
    ├── plot_brats.py         # 离线病例、统计和训练曲线绘图
    ├── brats/                # BraTS 数据、损失、推理和指标
    ├── configs/brats2020.json
    ├── models/vmunet/        # VMUNet 与 VSSM/SS2D
    ├── data/brats2020/       # 固定清单、数据审计及正式缓存
    ├── archives/             # 退役方案源码归档；独立于实验输出
    ├── results/              # 正式实验、图表及轻量工程验收记录
    ├── logs/                 # 正式训练日志
    ├── tests/                # 数据、几何、指标等回归测试
    └── docs/                 # 方案、绘图和上游参考说明
```

外层 `datasets/` 是原始数据目录；源码内 `data/` 是清单和缓存目录。BraTS JSON 清单、审计记录、数据缓存及正式实验权重均保留；`results/`、`logs/` 存放实验输出，退役方案源码存放于 `archives/`。

BraTS 主流程未使用的 8 个旧 ISIC/Synapse 文件已从源码目录移出。历史记录中的 `others/archives/vmunet_upstream_legacy_2026-11-01.zip` 未随当前 Linux 副本携带；原资源与恢复说明见 [上游参考](docs/UPSTREAM_REFERENCE.md)。本机保留的区域存在性源码归档见 [archives/](archives/README.md)。

## 来源与许可

原论文：[VM-UNet: Vision Mamba UNet for Medical Image Segmentation](https://arxiv.org/abs/2402.02491)。模型参考 [VMamba](https://github.com/MzeroMiko/VMamba)；原仓库同时致谢 [Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet)。当前副本缺少原 `LICENSE` 文件，需从原始来源补回。原始数据遵循各自的数据使用约定。


