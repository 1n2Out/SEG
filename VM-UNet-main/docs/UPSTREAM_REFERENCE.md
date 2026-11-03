# 原 VM-UNet 仓库参考

本页保留原项目来源、资源和旧实验操作说明。当前项目使用 BraTS2020，日常入口见 [项目首页](../README.md)，实验协议见 [BRATS2020.md](../BRATS2020.md)。本页的原作者结果不代表当前论文的实验结果。

2026-11-01 的清理记录将未被 BraTS 流程引用的 8 个旧文件归档至 `others/archives/vmunet_upstream_legacy_2026-11-01.zip`：`train.py`、`train_synapse.py`、`engine.py`、`engine_synapse.py`、`utils.py`、`datasets/dataset.py`、`configs/config_setting.py`、`configs/config_setting_synapse.py`。历史记录称归档保留原相对路径，并由 `ARCHIVE_INFO.json` 记录各文件 SHA256。当前 Linux 副本未携带该归档，本次无法重新核验。若需研究旧入口，先从原工作区取回归档并按原路径恢复到独立副本，使该副本能够导入当前 `models/`。下文为归档前的路径与历史用法，当前源码目录不再直接提供这些入口。

## 来源与致谢

原论文：[VM-UNet: Vision Mamba UNet for Medical Image Segmentation](https://arxiv.org/abs/2402.02491)。原工作提出基于视觉状态空间块的 U 形分割网络，并在 ISIC17、ISIC18 和 Synapse 上开展实验。模型参考 [VMamba](https://github.com/MzeroMiko/VMamba)，原仓库同时致谢 [Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet)。当前副本缺少原 `LICENSE` 文件，需从原始来源补回。

## 原环境记录

原 README 列出 Python 3.8、PyTorch 1.13.0、torchvision 0.14.0、CUDA 11.7 PyTorch wheel、timm 0.4.12、Triton 2.0.0、causal_conv1d 1.0.0、mamba_ssm 1.0.1，并引用带 cu118/torch1.13 标记的扩展 wheel。其他工具包括 tensorboardX、scikit-learn、matplotlib、thop、h5py、SimpleITK、scikit-image 和 medpy。

这是原仓库环境记录，未经本项目 Linux 服务器验证，不作为可直接执行的安装清单。当前工作区的扩展 wheel 标记为 **cu118/torch2.0/cp38/Linux**，与上述 Torch 1.13 环境不同。服务器准备要求见 [BraTS 环境说明](../BRATS2020.md#linux-服务器环境准备)。Windows 使用独立的 [CPU 开发环境](../WINDOWS_LOCAL.md)。

原扩展包资源：[Baidu](https://pan.baidu.com/s/1Tibn8Xh4FMwj0ths8Ufazw?pwd=uu5k)、[Google Drive](https://drive.google.com/drive/folders/1ZJjc7sdyd-6KfI7c8R6rDN8bcTz3QkCx?usp=sharing)。

## 旧实验路径与入口

以下路径均相对于原源码目录，按归档时的旧配置记录。

| 内容 | 归档参考代码使用的路径 |
|---|---|
| ISIC17 | `data/isic2017/` |
| ISIC18 | `data/isic2018/` |
| ISIC 子目录 | `train/images/`、`train/masks/`、`val/images/`、`val/masks/` |
| Synapse 训练切片 | `data/Synapse/train_npz/` |
| Synapse 测试体积 | `data/Synapse/test_vol_h5/` |
| Synapse 清单 | `data/Synapse/lists/lists_Synapse/train.txt`、`test_vol.txt` |
| VMamba 预训练文件 | `pre_trained_weights/vmamba_small_e238_ema.pth` |

归档内 ISIC 配置为 `configs/config_setting.py`；Synapse 配置为 `configs/config_setting_synapse.py`。恢复后的原入口为 `python train.py` 和 `python train_synapse.py`，仅用于配置完整的 Linux/CUDA 环境。这些入口及依赖未纳入当前 Windows 验收。

旧配置的 `only_test_and_save_figs`、`best_ckpt_path`、`img_save_path` 控制原入口的测试与图片导出，不能用于 `train_brats.py`。旧入口的模型选择、数据划分和 checkpoint 格式与 BraTS 不同。

## 原资源链接

- ISIC17/18 数据：[Baidu](https://pan.baidu.com/s/1Y0YupaH21yDN5uldl7IcZA?pwd=dybm)。原仓库描述其为 7:3 划分；本项目未下载或重新核验。
- Synapse 数据：[Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet)、[Baidu](https://pan.baidu.com/s/1JCXBfRL9y1cjfJUKtbEhiQ?pwd=9jti)。
- VMamba 预训练权重：[Baidu](https://pan.baidu.com/s/1ci_YvPPEiUT2bIIK5x8Igw?pwd=wnyy)、[Google Drive](https://drive.google.com/drive/folders/1ZJjc7sdyd-6KfI7c8R6rDN8bcTz3QkCx?usp=sharing)。
- 原作者训练 checkpoint：[Baidu](https://pan.baidu.com/s/1lygUOFo6fMF_wS_dskwpBQ?pwd=5z00)、[Google Drive](https://drive.google.com/drive/folders/1ZJjc7sdyd-6KfI7c8R6rDN8bcTz3QkCx?usp=sharing)。
- 原作者结果：[Baidu](https://pan.baidu.com/s/1FzDpfBQGz56BH6v2APoymA?pwd=8bqf)。

这些链接作为原资源索引保留，本次整理未检查远端可用性，也未自动下载任何内容。
