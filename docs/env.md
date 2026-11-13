# 运行环境

## A 端：Windows 本地（CPU）

用途：读代码、改代码，跑 CPU 测试，做数据检查和绘图。**不做**完整的 VM-UNet 前后向、训练、smoke 或模型评价，这些依赖 selective scan，只能在 B 端运行。

```powershell
conda activate vmunet-local        # 解释器 D:\Dev_Env\Anaconda\envs\vmunet-local\python.exe
cd D:\ZJGSU\ThesisWork\Task2\Seg\SEG\VM-UNet-main
python -B -m pytest tests -q -p no:cacheprovider --basetemp .test_tmp -k "not loss_is_finite_and_differentiable"
python -B check_local.py           # 模块导入，以及 Patch 模块和 loss 的 CPU 前向
python -B check_local.py --sample  # 另外用训练集第 1 例检查读取、归一化、增强和标签导出
```

- 被排除的那个测试需要反向传播，CUDA 测试和双进程测试会自动跳过，这些交给 B 端执行。
- `check_local.py --sample` 的缓存和报告写在 `results/local_checks/`，与正式缓存分开。

**重建环境：**

```powershell
conda env create -f environment-windows-local.yml
python -m pip check
```

如果只是补齐依赖，运行 `pip install -r requirements-windows-local.txt`。

| 依赖 | 版本 |
|---|---|
| Python | 3.10.20 |
| PyTorch / torchvision | 1.13.1+cpu / 0.14.1+cpu |
| timm / einops | 0.4.12 / 0.8.1 |
| NumPy / SciPy / nibabel | 1.26.4 / 1.13.1 / 5.3.2 |
| pytest / matplotlib | 8.4.2 / 3.9.4 |

本机没有安装 CUDA、Triton、`mamba_ssm`、`causal_conv1d`。`VMUNet` 能导入，但涉及 `SS2D` 的前向无法运行。

## B 端：Linux 服务器（GPU）

| 项目 | 值 |
|---|---|
| 仓库根目录 | `/home/hzh/hwh/Seg`（源码目录为 `Seg/VM-UNet-main`） |
| 预训练权重 | `/home/hzh/hwh/package/vmamba_small_e238_ema.pth`（配置中写作 `../../package/...`） |
| Conda 环境 | `vm_unet`；解释器 `/home/hzh/anaconda3/envs/vm_unet/bin/python` |
| GPU | 6 张：0、1 为 A100 80GB，2、3、4、5 为 RTX 6000 Ada 48GB；编号按 `CUDA_DEVICE_ORDER=PCI_BUS_ID`。用户指定只使用 RTX 6000，不使用 A100 |
| 核心依赖 | Python 3.8.20，Torch 2.0.1+cu118，torchvision 0.15.2+cu118，timm 0.4.12，einops 0.8.1，mamba-ssm / causal-conv1d 1.0.1，NumPy 1.24.3，SciPy 1.10.1，nibabel 5.2.1，pytest 8.3.5 |
| 数据缓存 | `VM-UNet-main/data/brats2020/cache/`，按固定清单生成；完整缓存约 30 GB |

- 补齐依赖：先配好 CUDA、PyTorch 和 Mamba 核心栈，再运行 `pip install -r requirements-linux-vm_unet.txt`。这个文件只能补齐依赖，**不能**用来在新服务器上从零装 CUDA 环境。
- 换服务器时：必须统一 Python、Torch、CUDA 和扩展包的版本并重新验证，缓存要和清单一起迁移。
- 每次实验的实际环境，以对应运行目录里的 `run.json` 为准。
