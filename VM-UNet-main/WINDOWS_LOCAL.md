# Windows 本地开发环境

本环境用于代码阅读/修改、BraTS 数据处理、普通网络模块的 CPU 检查及一个真实病例的小样本检查。完整 VM-UNet 前后向、训练、`--mode smoke`、模型评估/预测在 Linux 服务器的专属环境运行。BraTS 实验协议与服务器操作见 [BRATS2020.md](BRATS2020.md)，本页集中维护本地操作和验收记录。

## 日常使用

在支持 Conda 的 PowerShell / Anaconda Prompt 中执行：

```powershell
conda activate vmunet-local
cd D:\ZJGSU\ThesisWork\Task2\VM-UNet-main\VM-UNet-main
python -c "import sys; print(sys.executable)"

# 非训练测试，排除现有反向传播测试
python -B -m pytest tests -q -p no:cacheprovider --basetemp .test_tmp_brats_windows -k "not loss_is_finite_and_differentiable"

# 模型模块导入、PatchEmbed/Merging/Expand 和 loss 的 CPU 前向检查
python -B check_local.py

# 在上面的检查基础上，仅处理固定训练集的第一名患者
python -B check_local.py --sample
```

解释器应为 `D:\Dev_Env\Anaconda\envs\vmunet-local\python.exe`，IDE 也选择此解释器。无需设置 CUDA 环境变量。未激活终端时可以使用完整解释器路径，例如：

```powershell
& 'D:\Dev_Env\Anaconda\envs\vmunet-local\python.exe' -B check_local.py
```

`--sample` 使用项目现有的 NIfTI 预处理函数，检查四模态读取、归一化、标签映射、补零、增强和 `num_workers=0` 的 CPU DataLoader；抽查首/中/末切片，并对真实标签做 NIfTI 导出/回读。导出的是原始标签，不是模型预测。它不创建模型训练任务、不执行 backward 或优化器更新。

默认原始数据目录为工作区的 `datasets/BraTS2020/`，可用 `--data-root` 覆盖。只读原始文件，不修改固定清单。单病例缓存及报告在 `results/local_checks/`，已加入忽略规则，与正式 `data/brats2020/cache/` 分开。该目录的 `report.json` 记录最近一次成功检查；带 `--sample` 的检查额外记录病例与数据结果。重复检查会复用通过源文件校验的单病例缓存。

## 环境重建

本机安装位置为 `D:\Dev_Env\Anaconda\envs\vmunet-local`。使用以下命令可重建同名环境；仅在环境不存在时创建：

```powershell
cd D:\ZJGSU\ThesisWork\Task2\VM-UNet-main\VM-UNet-main
conda env create -f environment-windows-local.yml
conda activate vmunet-local
python -m pip check
```

已存在的环境只需补齐本地依赖时：

```powershell
conda activate vmunet-local
python -m pip install -r requirements-windows-local.txt
```

| 依赖 | 本地版本 |
|---|---|
| Python | 3.10.20 |
| PyTorch / torchvision | 1.13.1+cpu / 0.14.1+cpu |
| timm / einops | 0.4.12 / 0.8.1 |
| NumPy / SciPy | 1.26.4 / 1.13.1 |
| nibabel / pytest | 5.3.2 / 8.4.2 |
| matplotlib / Pillow / h5py | 3.9.4 / 10.4.0 / 3.11.0 |

Python 3.10 兼容原仓库的 PyTorch 1.13 系列，NumPy 保持 `<2`。CPU wheel 的版本配对依据 [PyTorch 官方历史版本说明](https://pytorch.org/get-started/previous-versions/#v1131)。直接依赖固定在 `requirements-windows-local.txt`；它复用现有 BraTS 和开发测试依赖，不改变服务器的依赖文件。

没有安装 CUDA Toolkit、Triton、`mamba_ssm`、`causal_conv1d`，不自动下载预训练权重。`VMUNet`/`vmamba` 可导入；涉及 selective scan 的 `SS2D`、`VSSBlock` 和完整 VM-UNet 前向仍需服务器环境。没有添加 CPU 模拟扫描算子。原 ISIC/Synapse 训练入口的工具依赖（例如 medpy、SimpleITK、tensorboardX）不属于本环境的验收范围。

## 本机验收（2026-10-31）

- PowerShell 中 `conda activate vmunet-local` 成功；全部表内开发依赖可导入，`torch.version.cuda` 为 `None`。
- `python -m pip check`：`No broken requirements found.`
- 上述非训练 pytest 命令：**26 passed、1 deselected**，没有跳过缺失依赖的测试；被排除项为反向传播测试。
- `check_local.py --sample`：普通 Patch 模块及 loss 的纯前向检查通过；真实病例 `BraTS20_Training_001` 的四模态缓存为 `[155,4,240,240]`，原始标签为 `0/1/2/4`，CPU batch 为 `[2,4,256,256]`。
- 抽查切片 `0/77/154`、增强、标签导出/回读及 affine 一致性均通过；检查前后源文件签名一致。单病例数组缓存约 80 MB，与正式缓存分开。

具体机器与依赖信息见 `results/local_checks/report.json`。未执行完整模型 selective scan、训练、模型评估或反向传播；这些检查仍留在服务器环境完成。

### 清理后复核（2026-11-01）

- 移出未被 BraTS 引用的旧代码后，按上述非训练 pytest 命令复核：**66 passed、1 deselected**（排除 loss 反向传播测试）。测试临时目录在复核后清除。
- 53 个保留文件（核心代码、测试、配置、数据清单、审计、预训练权重、病例图、训练集统计及检查报告）的 SHA256 与清理前一致。
- `.gitignore` 在独立临时 Git 仓库中核验：24 个应忽略路径与 16 个应保留路径均符合预期；未初始化当前项目的 Git 仓库。

### 离线绘图补充验收（2026-10-31）

- 新增 `plot_brats.py case/stats/curves`，完整使用方法见 [绘图指南](docs/PLOTTING.md)。无需本地训练或 GPU。
- 非训练全量回归：**66 passed、1 deselected**；新增 7 项绘图/统计/日志/纯前向 loss 分量检查。排除的仍为既有反向传播测试。
- 真实 `BraTS20_Training_001` 的原始数据及已有缓存路径均已生成多平面图；PNG/PDF/SVG 导出通过，并检查图片布局。
- 完整固定训练集 **259 例**标签统计通过，每患者每原始标签的体素数均与独立 `data_audit.json` 一致，未包含 val/test；生成类别比例、区域体积、阳性切片数量和轴向出现比例图及源数据 CSV。
- 曲线工具用明确标注为测试夹具的日志验证，覆盖真实验证轮次、分量缺失、旧版区域指标回读及未完成末行；没有正式训练曲线。
- 修改仅增加训练 loss 分量和已有验证结果的日志记录；损失总值的纯前向一致性通过，CUDA/DDP 的实际训练记录仍待服务器验证。未启动模型训练、模型评价、smoke 或反向传播。
