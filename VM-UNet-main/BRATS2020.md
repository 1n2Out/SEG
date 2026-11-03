# BraTS2020 二维四模态实验

本入口用于 VM-UNet baseline 和 BraTS2020 创新实验。baseline 与旧区域存在性感知 full 已完成各 100 轮训练及本地测试，结果见 [实验记录](../RESEARCH_NOTES/EXPERIMENTS_BraTS.md)。当前方案为 [浅层细节回注](docs/SHALLOW_REFINEMENT.md)，正式效果待验证；第二数据集另行选择。原 ISIC/Synapse 参考信息见 [原仓库说明](docs/UPSTREAM_REFERENCE.md)。

采用 **2D 四模态切片训练、3D 患者级评价**：每个输入仅包含同一轴向层的四个模态，不使用相邻切片；推理后按层重建三维分割。

## 当前本地数据与划分

源码目录为工作区内的 `VM-UNet-main/`；原始数据默认位于源码目录的上一级 `datasets/BraTS2020/`。所有默认路径均相对于脚本位置，支持通过参数覆盖，迁移服务器时无需保留 Windows 绝对路径。

训练划分保存于 `data/brats2020/split_seed42.json`。当前仅保留有标注数据：

| 集合 | 病例数 | 用途 |
|---|---:|---|
| 内部 train | 259 | 训练 |
| 内部 val | 55 | 模型选择、超参数选择 |
| 内部 test | 55 | 最终独立评估 |

125 例官方无标签 validation 数据及其专属清单 `official_val.json` 已于 2026-11-01 删除，不参与当前实验；未生成对应缓存。

这是对 369 例有标注病例排序后、使用独立 Python RNG（seed=42）进行的患者级随机划分，val/test 各取四舍五入的 15%，没有按肿瘤分级分层。训练清单采用 v2 格式，患者成员与原固定划分一致。

训练随机种子可以单独修改，重复实验时复用同一训练清单。清单包含相对路径、完整成员及 SHA256；移除官方无标签数据不影响训练清单和模型恢复。准备脚本仍拒绝覆盖同一路径下不同的训练划分。训练入口的 `--manifest` 仅指训练清单。无标签预测功能仍保留，如需使用须另行准备数据与独立预测清单。

`BraTS20_Training_355` 的标签为 `W39_1998.09.19_Segm.nii`，通过显式别名识别，原始文件没有重命名。其他病例缺少模态/标签时直接报错，不静默跳过。

`data/brats2020/data_audit.json` 保留删除官方无标签数据前的一次历史读取检查：494 例、2,345 个 NIfTI 文件；检查所有头信息、未压缩文件长度、qform/sform 几何对齐及全部 369 例标签体素。全部形状为 240×240×155、标签取值为 0/1/2/4，未发现检查范围内的错误。此次检查使用现有 numpy 和 NIfTI-1 头读取，不依赖 nibabel；并未扫描所有 MRI 强度或验证完整训练链路。当前保留 369 例、1,845 个 NIfTI 文件；MRI 非有限值及生产读取流程在生成缓存时进一步检查。

## 数据与模型协议

- 输入模态顺序固定为 **T1、T1ce、T2、FLAIR**，四通道，不转换 RGB。
- 每病例、每模态，使用该模态非零体素的均值/标准差进行 z-score，零背景保持零；不使用跨患者拟合统计量或标签裁剪。
- `nibabel` 读取 NIfTI，检查维度、affine 和标签值；不重新配准、不改变方向、不重采样。
- 缓存 `image.npy` 为 float16 的 `[Z,4,X,Y]`，`label.npy` 为 uint8 的 `[Z,X,Y]`。v2 元信息保存模态顺序、原始 affine、空间单位、qform/sform、XYZ 间距（毫米）、源文件签名和预处理版本。训练读取时转换 float32；预测导出从缓存恢复空间信息。
- 所有轴向切片均参与训练与推理，包括无肿瘤切片。训练增强为同步的 90° 旋转和面内翻转。
- 240×240 面内居中补零到 256×256，不缩放图像；推理裁去补零后恢复原始 XYZ 体积。若输入尺寸变化，必须仍能覆盖原图且可被 32 整除。
- 四个互斥输出为 `0/1/2/3`，原始标签 `4→3`，使用 **Cross Entropy + WT/TC/ET soft Dice**，两项等权；Dice 在每张切片上计算，再平均切片和区域。导出和评估恢复为 `0/1/2/4`。
- 网络深度沿用仓库配置 `[2,2,2,2] / [2,2,2,1]`。当前正式配置为 AdamW、lr=1e-4、weight_decay=0.01、**全切片、最多 100 epochs**、每卡 batch=16、workers=4、AMP 开启、余弦调度 `T_max=100`；默认启用验证集早停（patience=10、min_delta=0.0001）。当前使用 GPU 2（RTX 6000 Ada），预训练初始化；baseline、simple、gated 共用上述设置。已有 baseline 和旧创新 full 均完成 100 轮，最佳验证模型均为第 55 轮；当前浅层方案尚无正式结果。

## 预训练与初始化

当前配置使用 **预训练初始化**，路径为 `../../package/vmamba_small_e238_ema.pth`（约 711 MB，相对于源码目录）；本机文件存在，正式运行配置也记录了该预训练路径。此前静态检查其参数键名得到编码器深度 `[2,2,27,2]`，与当前 `[2,2,2,2]` 不同，因此不能视为完整复现该预训练网络。

在 JSON 的 `pretrained` 字段指定路径；相对路径按源码目录解析，设为 `null` 才使用随机初始化。`VMUNet.load_from()` 筛选兼容键名和形状，保留编码器到解码器的映射；RGB 输入卷积取通道均值、重复四次并乘 3/4，其余形状不匹配的张量跳过。实际加载应检查 `loaded/adapted/skipped` 报告。baseline、改进和消融必须共用初始化策略；迁移时需携带预训练文件或重新指定路径。

## 评估口径

本地三维患者级评估：

| 区域 | 原始标签 | 主指标 |
|---|---|---|
| WT | 1、2、4 | Dice、HD95(mm) |
| TC | 1、4 | Dice、HD95(mm) |
| ET | 4 | Dice、HD95(mm) |

先对每名患者的完整三维体积计算，再对患者等权平均；Mean Dice 为 WT/TC/ET 均值。另输出 Precision、Recall、Specificity。Dice 范围为 0~1，转百分数需乘 100。

**显式本地空区域规则**：两者均空，Dice/Precision/Recall=1、HD95=0；恰有一者为空，Dice/Precision/Recall=0、HD95 为 `norm(shape_xyz * spacing_xyz_mm)`，当前网格约 373.13 mm。HD95 采用 6 邻域表面，合并两个方向的表面最近距离后取 95 百分位。失败病例不删除；CSV 另记录预测与标注区域体素数。该规则是固定的本地实验协议，尚未核对为 BraTS 官方服务器完全一致实现；不能把本地数字直接当成官方榜单分数。

官方任务/区域说明：https://www.med.upenn.edu/cbica/brats2020/tasks.html

输出：

绘图入口、数据分布分析和训练曲线的完整使用方法见 [离线绘图指南](docs/PLOTTING.md)。绘图不自动随训练运行，训练日志增加 CE/Dice 分量和分区域验证 Dice，损失目标与验证频率不变。

- `summary.json`：各区域均值、患者间总体标准差（ddof=0）、Mean Dice、评估协议、划分校验和、模型参数量、推理耗时及 CUDA 峰值分配显存。
- `per_patient.csv/json`：逐患者、逐区域指标及体素数，可供后续配对检验和误差分析。
- 可选 `predictions/*.nii.gz`：原始形状/空间坐标、uint8、0/1/2/4 标签。

耗时包含缓存切片访问、补零、设备传输、前向和重建，不包含指标计算和 NIfTI 导出，未做独立预热；它是运行记录，不是严格延迟基准。不自动报告可能漏算 selective scan 的 FLOPs；正式效率对比需另行统一测量。

## 开发与训练环境

Windows 仅用于环境配置、代码调整和非训练验证，本地不运行训练或 `--mode smoke`。安装、检查命令与验收记录统一见 [WINDOWS_LOCAL.md](WINDOWS_LOCAL.md)。

### Linux 服务器环境准备

服务器需配置可用的 CUDA、PyTorch、torchvision、timm、einops 和 `mamba_ssm` selective scan，再安装 `requirements-brats.txt` 中的数据/评价依赖；运行测试时使用 `requirements-dev.txt`。

2026-11-02 的环境记录为 Python 3.8.20、Torch 2.0.1+cu118、torchvision 0.15.2+cu118、timm 0.4.12、einops 0.8.1、mamba-ssm/causal-conv1d 1.0.1；补齐 nibabel 5.2.1 后，真实数据 smoke、双卡短训练及中断恢复、测试入口和 NIfTI 预测导出通过。原 `ENV_CHECK_vm_unet_2026-11-02.md` 未随当前副本保留；正式运行环境以各实验的 `run.json` 为准。已配置相同 CUDA 核心栈的本机环境可使用 `python -m pip install -r requirements-linux-vm_unet.txt` 补齐已验证依赖；其中辅助包版本兼顾本机既有 TensorFlow 2.13.1，不能将此文件当作任意服务器的 CUDA 环境安装器。

以下是既有资源的版本信息，不是其他平台已验证的安装组合。

| 资源 | 版本或目标平台 |
|---|---|
| 原仓库环境记录 | Python 3.8、Torch 1.13.0、torchvision 0.14.0、timm 0.4.12；详见 [原仓库参考](docs/UPSTREAM_REFERENCE.md) |
| `others/packages/causal_conv1d-1.0.1+cu118torch2.0cxx11abiFALSE-cp38-cp38-linux_x86_64.whl` | Python 3.8、Torch 2.0、CUDA 11.8、Linux x86_64 |
| `others/packages/mamba_ssm-1.0.1+cu118torch2.0cxx11abiFALSE-cp38-cp38-linux_x86_64.whl` | Python 3.8、Torch 2.0、CUDA 11.8、Linux x86_64 |

`others/` 在旧工作区位于源码目录的上一级，当前 Linux 工作区未携带该目录。这些扩展 wheel 不能直接用于 Torch 1.13 或 Windows CPU 环境。其他服务器安装前须统一 Python、Torch、CUDA 和扩展包版本并单独验证。

## 在 Linux 训练环境执行

以下训练、恢复和 smoke 命令用于配置完成的 Linux 服务器。本机已有全部 369 例正式 v2 缓存（`data/brats2020/cache/`，约 29.65 GB）。训练切片 40,145 张，val/test 各 8,525 张。正式 baseline 位于 `results/brats2020_1102_1734/`；独立三病例短训练仅用于程序验收，不作为科研结果。

以下命令均在源码目录执行。`--data-root` 仅在准备阶段使用；训练、评估、预测均读取 `--cache-dir`，无需原始 NIfTI。数据根目录变化时保留其内部目录结构，以匹配清单中的相对路径。

日常推荐使用统一启动器：修改 `configs/brats2020.json` 中的 `launch`（GPU、路径）及训练参数后，执行 `python run_brats.py` 前台训练，或 `python run_brats.py --background` 后台训练。`prepare`、`smoke`、`resume`、`test` 子命令分别对应准备缓存、链路检查、续跑和最终测试，完整简短指令见 [个人指南](../RESEARCH_NOTES/个人指南.md)。相对路径按源码目录解析；`launch` 不进入 checkpoint 的训练参数比较，GPU 数量等恢复约束仍由训练入口检查。后台通过独立会话和标准流重定向脱离终端，日志追加写入，PID 保存于同名 `.pid` 文件；启动成功只代表进程已创建，应检查日志确认训练进展。以下保留底层入口命令供参考。

服务器可选择两种上传方式：

1. **上传源码与原始训练数据**：上传 `BraTS2020_TrainingData`，保留其下 `MICCAI_BraTS2020_TrainingData/病例目录` 结构。在服务器运行 `prepare_brats.py --cache`，通过 `--data-root` 指向包含训练数据目录的根目录。
2. **上传源码、训练清单与完整 v2 缓存**：缓存需包含训练、内部验证、内部测试全部病例的 `image.npy`、`label.npy`、`metadata.json`。可直接进行训练/评估及预测导出，不上传原始 MRI，也不需要重跑准备命令。当前本机已有完整缓存，可以采用此方式迁移；必须一起保留固定清单、数组与空间元信息。

```bash
# 仅确认目录并生成/核对固定划分，不需要 torch/nibabel；本地已执行
python prepare_brats.py

# 生产 NIfTI 校验与缓存，369 例约需 30 GB 十进制磁盘空间
python prepare_brats.py --cache

# 前向、loss、反向有限值及 1 个验证体积重建检查；不会做优化器更新
python train_brats.py --mode smoke --work-dir results/brats2020_smoke

# 正式训练；不自动评估内部测试集
python train_brats.py --work-dir results/brats2020_seed42

# 单机双卡 DDP；batch_size 是每卡 batch，全局 batch 为其两倍
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nnodes=1 --nproc_per_node=2 train_brats.py --work-dir results/brats2020_seed42_2gpu

# 恢复：原配置、原工作目录，同时保留 latest.pth 和 best.pth
python train_brats.py --work-dir results/brats2020_seed42 --resume results/brats2020_seed42/latest.pth

# 固定方法与超参数后，再单独评估内部测试集
python train_brats.py --mode evaluate --checkpoint results/brats2020_seed42/best.pth --work-dir results/brats2020_seed42 --split test --save-predictions

```

自定义缓存路径时，例如：

```bash
# 服务器准备原始数据
python prepare_brats.py --data-root /srv/data/BraTS2020 --cache --cache-dir /srv/cache/brats2020
# 或已上传完整缓存，直接执行训练
python train_brats.py --cache-dir /srv/cache/brats2020
```

缓存完成标记最后写入；中断后重跑可继续。原始文件的大小/修改时间签名仅在准备阶段检查，训练/评估检查病例、模态顺序、缓存形状、类型和空间元数据。源文件或缓存协议变化时，使用 `--cache --rebuild-cache` 显式重建；不修改原始数据。v1 缓存需重建为 v2，以补齐独立导出的空间信息。不支持并发写同一个缓存目录。

best 模型由内部验证集实际最高三维 Mean Dice 选择，第 1 轮、每 5 轮及最后一轮验证。`early_stopping` 默认 `enabled=true, patience=10, min_delta=0.0001`：分数超过上次有效提升的参考值加 min_delta 时重置计数，连续 10 次验证无有效提升后停止；未验证的 epoch 不计数。best 选择独立于早停阈值。checkpoint 包含模型、优化器、调度器、AMP scaler、各进程随机数状态、早停状态、卡数、配置和划分校验和；恢复要求原工作目录、配置、划分和卡数，已早停的 latest 恢复后直接退出。旧配置缺少早停字段时保持固定轮数。

支持 `torchrun` 启动 DDP，每卡一个进程，使用 DistributedSampler 并逐轮 `set_epoch`；主进程独占完整验证集评价和 checkpoint/日志写入。训练 loss 按各进程实际采样数汇总；为对齐步数，每轮可能重复最多 `world_size-1` 张切片，数量记录在 `run.json`。训练 checkpoint 保存不带 DDP 前缀的权重，可直接单卡评估；evaluate/predict 使用单进程 `python`，多进程启动会报错。学习率不自动随卡数缩放。CUDA 算子的完全确定性仍取决于运行环境。`run.json` 保存环境信息、卡数、全局 batch 和源码 SHA256（排除测试、归档、结果和临时目录），`history.jsonl` 保存训练和早停记录。未实现测试时增强或后处理。

单卡/多卡训练、nohup、恢复和测试的历史操作指南见 [个人指南](../RESEARCH_NOTES/个人指南.md)，其中早期训练参数不代表当前配置；浅层方案的现行命令见 [SHALLOW_REFINEMENT.md](docs/SHALLOW_REFINEMENT.md)。

## 验证状态

Windows 非训练测试的历史结果见 [Windows 验收记录](WINDOWS_LOCAL.md#本机验收2026-10-31)。Linux CUDA、双卡训练、中断恢复及预测导出已有历史验收；当前浅层方案的检查范围与记录见 [工程检查](docs/SHALLOW_REFINEMENT.md#工程检查) 和 [小清单验收报告](results/verification/shallow_refinement_pipeline/verification_report.json)。正式 baseline 和旧创新 full 的训练及测试结果见 [实验记录](../RESEARCH_NOTES/EXPERIMENTS_BraTS.md)。

2026-11-03 清理了两套小清单工程验收的临时权重，保留配置、划分、报告、日志和评价指标。它们不能再从原临时 checkpoint 续跑；需要复查时重新执行验收。正式实验权重和数据缓存均保留，清理明细见 [cleanup_20261103.json](results/cleanup_20261103.json)。
