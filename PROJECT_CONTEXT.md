# PROJECT_CONTEXT.md

> AI 项目导航，更新于 2026-11-02。源码位于工作区子目录 `VM-UNet-main/`；下文代码路径均相对于源码目录。

## 项目定位与当前决策

- **关键科研宗旨（后续研究的取舍依据）**：科研时间紧迫，需要以较少的时间完成足以支撑硕士毕业论文的工作量。优先级为 **创新性“故事” > 工作量堆叠（图表、实验等） > 底层的真实创新性 > 实验指标**。优先形成动机清晰、逻辑连贯、有依据且可验证的创新叙事，选择实现成本和实验周期可控的方案；围绕核心论点安排足够的对比、消融、分析与图表，形成完整的论文支撑材料；指标用于检验方法效果，避免为小幅涨点投入长周期调参。研究叙事与结论以实际实验和分析为依据，后续选题、模块设计与实验投入均遵循此优先级。
- 以 VM-UNet 为 baseline，服务于硕士论文第二项工作。先在 BraTS2020 上确定创新点和开展实验，再根据创新点选择第二数据集。
- 已实现 BraTS 数据准备、训练、恢复、三维评价和预测入口；尚未正式训练，创新模块尚未确定。
- 当前方法是 **2D 四模态轴向切片分割，3D 患者级评价**，不使用相邻切片。
- 当前正式训练预算已确定为 **全切片 + 最多 60 轮**：GPU 2（RTX 6000 Ada）、每卡 batch=8、`num_workers=4`、`amp=true`、随机初始化；不抽样删减阴性切片。验证间隔 5 轮及早停 `patience=10/min_delta=0.0001` 保持原值，余弦调度随总轮数采用 `T_max=60`。baseline、改进与消融共用该协议。短测外推约 7.3 小时，排期预留 8–10 小时；尚未正式训练，详见 [训练时间分析](RESEARCH_NOTES/TRAINING_TIME_gpu2_2026-11-02.md)。
- 原始数据在工作区 `datasets/BraTS2020/`。仅保留 369 例有标注患者，固定划分 259/55/55，使用 v2 清单；125 例官方无标签 validation 数据及专属清单已于 2026-11-01 删除。
- 2026-11-02 已在 Linux `vm_unet` 生成并核对全部 369 例正式 v2 缓存，位于 `VM-UNet-main/data/brats2020/cache/`（约 29.65 GB）；历史双卡配置 smoke 和训练入口预检通过，当前 GPU 2 的 AMP + workers=4 已通过真实数据短测及一例完整验证推理检查。正式测试集尚未模型评价。

## 文档入口与验证边界

**研究文档维护限制**：操作 `RESEARCH_NOTES/` 前先阅读其 [README](RESEARCH_NOTES/README.md)。未经用户明确要求，不得新增、修改、追加、删除、重命名或移动该目录及子目录中的文档；代码、配置或训练任务不自动授权同步更新研究笔记。

| 查阅内容 | 入口 |
|---|---|
| 项目状态、目录解释 | [README.md](VM-UNet-main/README.md) |
| 数据、损失、评价、权重、服务器命令 | [BRATS2020.md](VM-UNet-main/BRATS2020.md) |
| Windows 环境、日常检查、验收记录 | [WINDOWS_LOCAL.md](VM-UNet-main/WINDOWS_LOCAL.md) |
| 原仓库实验与资源 | [UPSTREAM_REFERENCE.md](VM-UNet-main/docs/UPSTREAM_REFERENCE.md) |
| 创新点动机、理论依据、模块设计与归档（当前为模板） | [INNOVATIONS.md](RESEARCH_NOTES/INNOVATIONS.md) |
| BraTS2020 实验动机、训练配置与结果（当前为模板） | [EXPERIMENTS_BraTS.md](RESEARCH_NOTES/EXPERIMENTS_BraTS.md) |
| 科研发现、历史决策与后续计划 | [HISTORY.md](RESEARCH_NOTES/HISTORY.md) |

用户约定：Windows 仅做环境配置、代码调整和非训练验证，使用 `vmunet-local` CPU 环境；训练、完整 VM-UNet 前后向、模型评价/预测及 `--mode smoke` 在 Linux 服务器运行。本机 Linux 使用 `vm_unet`，用户已于 2026-11-02 授权补齐依赖及验证训练/测试链路。不自动下载权重，不添加模拟 selective scan 后端。

Linux `vm_unet` 已补齐数据依赖，`pip check` 通过，回归测试 67 passed。真实 selective scan、完整 CUDA 前后向、真实病例 smoke、双卡短训练及中断恢复、测试入口和预测导出已通过；验收仅使用独立的原训练/验证集三病例小清单，没有评估正式测试集或开展正式 baseline 训练。详见 [环境检查](RESEARCH_NOTES/ENV_CHECK_vm_unet_2026-11-02.md)。仅记录实际执行结果，不将链路验收写成正式实验完成。

## 代码导航

| 修改目标 | 关键位置 |
|---|---|
| 患者清单、特殊标签别名、缓存和几何 | `prepare_brats.py`、`brats/data.py` |
| 切片、增强、损失 | `brats/dataset.py`、`brats/losses.py` |
| 三维推理、指标与导出 | `brats/engine.py`、`brats/metrics.py` |
| 训练、DDP、恢复、best 选择和初始参数 | `train_brats.py`、`configs/brats2020.json` |
| 离线病例/分布/曲线绘图、样式和用法 | `plot_brats.py`、`brats/plotting.py`、`configs/plotting.json`、`docs/PLOTTING.md` |
| 一键前台/后台训练、缓存、续跑与测试 | `run_brats.py`；GPU/目录等统一在 `configs/brats2020.json` 的 `launch` 中配置 |
| 早停控制与训练/后台/测试命令 | `brats/training.py`、工作区 `RESEARCH_NOTES/个人指南.md` |
| 输入/输出封装、预训练加载 | `models/vmunet/vmunet.py::VMUNet` |
| 编解码、跳跃连接和选择性扫描 | `models/vmunet/vmamba.py::VSSM/VSSBlock/SS2D` |
| 本地数据检查与回归测试 | `check_local.py`、`tests/` |

## 实验与修改边界

- 输入顺序 T1/T1ce/T2/FLAIR；缓存图像为 `[Z,4,X,Y]`。运行时只读缓存；数组、空间元信息和对应清单需一起迁移。当前仅保留有标注患者清单；无标签预测数据如需使用须另行准备独立清单。
- 网络输出四类 logits；训练标签 4→3，评价与导出还原 3→4。WT/TC/ET 是重叠区域，不是三个互斥类别。损失与空区域规则以 `BRATS2020.md` 为准。
- 旧工作区记录有 `pre_trained_weights/vmamba_small_e238_ema.pth`，但当前 Linux 工作区未携带该文件；当前配置仍为 `pretrained: null`，随机初始化不受影响。加载器筛选键名/形状并适配输入卷积，不代表完整加载原预训练网络；实际文件加载待验证。
- 编码通道 `[96,192,384,768]`，跳跃连接相加，当前 `skip_list[0]` 未使用。改宽度或拓扑须联动检查相加尺寸、上下采样及权重映射；不要顺手删除消耗随机数的旧初始化代码。BraTS 输入尺寸需覆盖原图且可被 32 整除。
- baseline、改进和消融共用患者清单、公共数据处理、初始化及早停策略。按内部验证集实际最高三维 Mean Dice 选 best，默认早停为连续 10 次验证无超过 0.0001 的有效提升；测试集仅在方法确定后评估。支持 torchrun DDP，batch_size 为每卡 batch。恢复要求原配置、原工作目录及原卡数。双卡 DDP/NCCL 及真实 checkpoint 中断恢复已在小清单上验收，完整数据长训练稳定性尚无实测结论。
- 绘图独立于训练，手动执行，不加载模型或使用 GPU；训练默认额外记录 CE/Dice 分量和验证各区域 Dice，不新增前向/验证，不自动绘图。绘图样式配置独立，不进入 checkpoint 的训练配置比较。病例图和统计图可读原始数据或已有 v2 缓存；曲线读实际训练日志，旧日志缺失分量不补造。
- `BraTS20_Training_355` 的特殊标签名由显式别名识别，不重命名原始文件。数据审计保存在 `data/brats2020/data_audit.json`，为删除官方无标签数据前的历史记录；MRI 强度全量校验在正式缓存准备阶段完成。
- 原 ISIC/Synapse 参考代码有 RGB、掩码缩放、旧统计量和不同模型选择逻辑，不能直接继承为新数据协议；未被 BraTS 引用的 8 个旧文件已移出源码目录，归档于 `others/archives/vmunet_upstream_legacy_2026-11-01.zip`，详见上游参考文档。功能适配/公共修复与论文创新分开记录。

