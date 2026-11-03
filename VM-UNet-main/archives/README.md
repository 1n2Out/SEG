# 源码归档

本目录保存退役方案的源码，独立于 `results/` 实验输出。归档代码仅在独立副本中恢复，避免覆盖当前实现。

## 区域存在性感知监督

- 文件：[region_presence_source_before_replacement.zip](region_presence_source_before_replacement.zip)
- 内容：替换前的训练入口、旧模块、三个配置、专用测试和原说明，共 13 个文件。
- SHA256：`6e126b5342ea53b93d89af84c162a19d66f6c097b192eac500b4893903f13d32`
- 2026-11-03 从 `results/archives/` 迁入，ZIP 完整性和迁移前后 SHA256 核对通过；旧路径保留相对符号链接，兼容既有研究笔记。
- 正式实验权重保留于 `results/brats2020_region_presence_full_1103_1004/`。旧模型推理及恢复方式见 [退役说明](../docs/REGION_PRESENCE.md)。小清单工程验收权重已清理，不能用于恢复。

可在源码目录执行：

```bash
sha256sum archives/region_presence_source_before_replacement.zip
```

历史 ISIC/Synapse 归档未随当前副本携带，见 [上游参考说明](../docs/UPSTREAM_REFERENCE.md)。
