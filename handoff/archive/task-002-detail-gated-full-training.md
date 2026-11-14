# TASK-002：detail gated 完整训练（附加任务）

状态：**关闭（执行完成）** · 想法：[IDEA-001](../../research/ideas/IDEA-001-detail-fusion.md) · 发起：用户，2026-11-13 · 执行：B · 分支：见 STATE.md

## 目的

基于 [TASK-001](task-001-detail-fusion-screening.md) 的筛选结果，完成一次 detail gated 的完整协议训练，检验其相对 EXP-001 的验证集收益。
筛选结论见 [EXP-003](../../research/experiments/EXP-003/report.md)；本任务不预设门控本身有效。

## 运行清单

| EXP | 配置 | 说明 | 必要性 |
|---|---|---|---|
| EXP-006 | `configs/detail_fusion/gated.json` | detail gated，VMamba 预训练初始化，按完整协议独立训练 | 必跑，仅此一组 |

## 协议与边界

- 沿用 [protocol](../../docs/protocol.md)：最多 100 轮，单卡 batch 16，lr 1e-4，seed 42，AMP 与早停按配置执行。
- 初始化使用已有 `../../package/vmamba_small_e238_ema.pth`；不加载筛选 checkpoint，不使用 `warm_start`。
- 对照为已有 EXP-001；concat 完整训练及其他消融需另行安排。
- 使用本地 `vm_unet` 和 RTX 6000，不使用 A100；不修改模型、损失、数据划分或评价口径。
- 本轮范围：完善文档、检查、smoke、启动后台训练并登记；训练结果收集、分析和 EXP-006 报告由用户回叫后执行。
- 本任务不含 FINAL；用户于 2026-11-14 追加的测试授权另记 [TASK-003](task-003-detail-gated-final.md)。不推送远端。

## 步骤

1. 核对正式配置公共参数与 baseline 一致，检查预训练权重、缓存、代码来源闸门和可用 GPU。
2. 模型和环境未变，沿用 TASK-001 已通过的 CPU/CUDA 检查；本轮使用正式配置单独执行 smoke。
3. `python run_brats.py smoke --config configs/detail_fusion/gated.json --gpus 2`，须实际通过。
4. `python run_brats.py train --config configs/detail_fusion/gated.json --gpus 2 --dry-run`，核对输出目录。
5. `python run_brats.py train --config configs/detail_fusion/gated.json --gpus 2 --background`，确认进入训练循环后结束本轮。
6. 用户回叫后：`python collect_brats.py --run-dir results/<运行目录> --exp EXP-006 --title "完整训练：detail gated"`，按 val best 收集，不指定 `--epoch 10`。

## 通过标准与决策

| 结果 | 建议 |
|---|---|
| Mean Dice 高于 EXP-001，ET 不下降、Mean HD95 不变差、空 ET 假阳性不增加 | 报告完整训练收益，建议用户决定后续消融 |
| 指标有取舍或没有整体收益 | 如实报告逐患者差异及负结果，由用户决定后续方向，不自动调参重跑 |
| 只有 gated 完整结果，缺少 concat 完整对照 | 不判断门控的独立贡献，不把筛选数字作为完整训练消融结果 |

---

## 回执（B 填写）

- **实际提交：** 训练源码 `4ede609`；仅完善交接文档，源码和配置保持已提交状态。
- **检查：** 公共参数与 baseline 一致；权重、369 例缓存和来源闸门检查通过。正式 smoke 前后向及单例 val 通过，产物 `results/brats2020_detail_gated_smoke_1113_1529/`；dry-run 通过。
- **运行目录：** `results/brats2020_detail_gated_1113_1530`；RTX 6000 GPU 2，PID 2284172；日志 `logs/brats2020_detail_gated_1113_1530.log`。
- **异常与修复：** 无；使用 `vm_unet` 和 RTX 6000 GPU 2。
- **结论：** 正常完成 100 轮，val best 为第 95 轮；最佳权重结果收集、逐患者配对分析和 [EXP-006 报告](../../research/experiments/EXP-006/report.md)完成，四项验证集条件全部满足。
- **建议：** 根据验证集结果可保留 gated 并由用户决定完整 concat 对照；不自动训练、提交或推送。独立 FINAL 结果见 TASK-003。
