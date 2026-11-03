# 区域存在性感知监督：已停用并移出活动代码

2026-11-03 按用户要求，用浅层细节回注方案替换。旧模块、三个运行配置和专用测试已从活动源码中删除；当前训练入口会拒绝 `region_presence` 配置，避免静默使用错误模型。

新方案见 [SHALLOW_REFINEMENT.md](SHALLOW_REFINEMENT.md)。已有正式实验目录、日志、研究笔记和原训练权重保留。

## 旧模型仍可用于推理

在 `results/brats2020_region_presence_full_1103_1004/exports/` 保存了：

- `best_inference.pth`：第 55 轮最佳模型的纯分割权重；
- `latest_inference.pth`：第 100 轮的纯分割权重。

导出逐张量核对了所有 `vmunet.*` 权重与原文件完全一致，只移除评价时不执行的存在性辅助头及训练状态。导出文件使用原 VMUNet，可由当前 `train_brats.py --mode evaluate --checkpoint ...` 加载；不要再传旧创新配置。它们不能用于 resume。

## 追溯和恢复源码

删除前的模块、配置、专用测试、训练入口和原使用说明已保存为 [源码归档](../archives/region_presence_source_before_replacement.zip)，校验与恢复说明见 [归档索引](../archives/README.md)。旧的 `results/archives/` 路径保留兼容链接，供已有研究笔记引用。如果需要重现旧训练或续跑旧模型，请在独立源码副本中恢复归档，使用正式实验的原 `latest.pth` 和配置；不要覆盖当前新方案代码。
