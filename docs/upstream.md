# 原 VM-UNet 参考

原论文为 [VM-UNet: Vision Mamba UNet for Medical Image Segmentation](https://arxiv.org/abs/2402.02491)，原作者在 ISIC17、ISIC18 和 Synapse 上做实验。模型部分参考了 [VMamba](https://github.com/MzeroMiko/VMamba)，原仓库同时致谢 [Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet)。当前副本缺少原 `LICENSE` 文件，需要从原始来源补回。原作者的实验结果不代表本项目的结果。

## 原环境记录

原仓库列出的环境：Python 3.8、PyTorch 1.13.0、torchvision 0.14.0、timm 0.4.12、Triton 2.0.0、causal_conv1d 1.0.0、mamba_ssm 1.0.1。本项目实际使用的环境见 [env](env.md)。

## 旧入口

原 ISIC 和 Synapse 的入口（`train.py`、`train_synapse.py`、`engine*.py`、`utils.py`、`configs/config_setting*.py`）已从源码中移除。它们的数据划分、模型选择方式和 checkpoint 格式都与 BraTS 协议不同，**不能继承**。历史归档 `others/archives/vmunet_upstream_legacy_2026-11-01.zip` 不在当前仓库中。

## 原资源

- VMamba 预训练权重：[Baidu](https://pan.baidu.com/s/1ci_YvPPEiUT2bIIK5x8Igw?pwd=wnyy)、[Google Drive](https://drive.google.com/drive/folders/1ZJjc7sdyd-6KfI7c8R6rDN8bcTz3QkCx?usp=sharing)
- 扩展包 wheel：[Baidu](https://pan.baidu.com/s/1Tibn8Xh4FMwj0ths8Ufazw?pwd=uu5k)
- ISIC17/18 数据：[Baidu](https://pan.baidu.com/s/1Y0YupaH21yDN5uldl7IcZA?pwd=dybm)
- Synapse 数据：[Swin-UNet](https://github.com/HuCaoFighting/Swin-Unet)
- 原作者 checkpoint：[Baidu](https://pan.baidu.com/s/1lygUOFo6fMF_wS_dskwpBQ?pwd=5z00)

以上链接都没有核验过是否仍然可用。
