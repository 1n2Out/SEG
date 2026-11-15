# STATE
覆盖式更新，不超过 40 行。最后更新：2026-11-15，A。

## 协议与基线

- **工作分支：** `stage003-detail-stacking`；两端共用该分支。
- **用户要求：** 仅明确要求时提交和推送。使用 `vm_unet` 和 RTX 6000（GPU 2–5），不使用 A100；单种子、单次训练。
- **协议：** [docs/protocol.md](docs/protocol.md)、`VM-UNet-main/configs/brats2020.json`；预训练初始化、100 轮、单卡 batch 16；不使用相邻切片。
- **对照：** baseline 为 [EXP-001](research/experiments/EXP-001/report.md)。结果见 [总表](research/experiments/index.md)。
- **正式创新点：** [IDEA-001](research/ideas/IDEA-001-detail-fusion.md) detail gated，2026-11-15 采纳；代表实验 [EXP-006](research/experiments/EXP-006/report.md)（val best 第 95 轮，已做过一次 FINAL）。后续优化以 EXP-006 为基座和对照。

## 本地未推送的改动（A 端，已暂存、未提交）

- **决定落档：** 放弃 IDEA-002/005/006、授权 A 读取 FINAL、采纳 IDEA-001，见 [decisions](research/decisions.md) 2026-11-15 各条。
- **代码清理：** 删除 `shallow_refinement` 扩展（旧配置会被拒绝）以及 `boundary_weight`、`context_slices` 两个选项的代码、配置和测试；保留 `warm_start.new_parameters` 和 `configs/screening/stack-control.json`，供在 EXP-006 上叠加新部件时使用。
- **分析脚本：** `analysis/compare-screening.py`、`compare-full-training.py` 读写时显式用 UTF-8。
- **文档整理：** 协议、方法文档、codemap、IDEA-001/002/003/005/006 与索引已同步；去掉重复事实和过时引用。
- **检查：** A 端 CPU 151 passed、4 skipped（CUDA 用例）；双进程检查 1 passed。B 端在推送前看不到以上改动。

## 测试集约束

A 已接触测试集。此后在 BraTS 上提出的优化 IDEA 须注明这一点，且不得以测试集现象（如个别病例的 HD95 离群）为动机；优化后的最终模型只测一次 test。

## 训练中

无。

## 待决事项与下一步

- **用户：** 在另一个窗口构思 IDEA-001 的优化方向；决定是否补做 concat 完整训练（约 13 h），用于证明门控的独立贡献；决定何时提交推送。
- **A：** 按选定方向起草优化 IDEA 和任务单；补齐 IDEA-001 动机统计（ET 边界占比、块常数上限）的脚本，入库 `analysis/`，只用 train/val。
- **B：** 暂无任务。A 推送后再拉取，并跑一遍 CPU 与 CUDA 检查；不修改协议、IDEA 状态，也不执行测试集评价。
