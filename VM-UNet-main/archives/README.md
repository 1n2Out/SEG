# 源码归档

这里存放已退役方案的源码。恢复时只能解压到独立的副本中，不要覆盖现有实现。

## 区域存在性感知监督（2026-11-03 停用）

- **文件：** `region_presence_source_before_replacement.zip`，共 13 个文件，包括当时的训练入口、模块、3 个配置、专用测试和原说明文档。
- **SHA256：** `6e126b5342ea53b93d89af84c162a19d66f6c097b192eac500b4893903f13d32`，可用 `sha256sum` 核验。
- **入口拒绝：** 当前训练入口会拒绝 `region_presence` 配置。
- **旧权重（仅在服务器上）：** 旧运行目录 `results/brats2020_region_presence_full_1103_1004/` 中，`exports/*_inference.pth` 是去掉辅助头之后的纯分割权重，可以直接用 `train_brats.py --mode evaluate --checkpoint ...` 加载（不要再传旧配置），但不能用于续跑。
- **复现或续跑旧训练：** 在独立副本中恢复归档源码，并使用原来的 `latest.pth`。
