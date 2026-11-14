# TASK-003：detail gated 一次性 FINAL 测试与结果分析

状态：**关闭（执行完成）** · 授权：用户于 2026-11-14 要求“请测试并查看结果，然后进行分析” · 执行：B

## 目的

对 [TASK-002](task-002-detail-gated-full-training.md) 的 EXP-006 收集完整训练结果，并在用户授权的 FINAL 阶段评价一次封存测试集。

## 固定评价对象

- 运行：`results/brats2020_detail_gated_1113_1530`；训练完成 100 轮，权重固定为 val 选出的第 95 轮 `best.pth`。
- 配置：`configs/detail_fusion/gated.json`；源码提交 `4ede609`，执行前来源闸门须通过。
- 评价前未发现该运行的 `test_*` 目录；本任务只执行一次 test，不按测试结果重新选权重或调参。
- baseline 使用 EXP-001 已有的 val 结果和历史 FINAL 测试结果，不重新测试 baseline。
- 使用本地 `vm_unet` 和 RTX 6000，不使用 A100；不提交或推送。

## 步骤

1. 确认训练正常结束、best 与历史记录一致、源码和配置可追溯；收集 val best 到 EXP-006。
2. `python run_brats.py test --config configs/detail_fusion/gated.json --run-dir results/brats2020_detail_gated_1113_1530 --gpus 2 --confirm-test`。
3. `python collect_brats.py --run-dir results/brats2020_detail_gated_1113_1530 --exp EXP-006 --title "完整训练：detail gated" --include-test --force`。
4. 分别按 val / test 与 EXP-001 做逐患者配对比较，核对病例集合、空 ET 假阳性及各区域指标。
5. val 分析和报告放在 EXP-006 主目录；测试分析与报告放在 `EXP-006/final/`，测试分数不进入实验总表。
6. 更新 TASK-002 回执、STATE 和本任务回执；结果不用于选择后续模型、超参数或后处理。

## 验收

- 测试仅使用固定的第 95 轮 best，完整覆盖既定 55 例 test；数据划分与训练记录匹配。
- 各指标和配对分析都可追溯到脚本生成的文件；分析脚本保留在 `analysis/`。
- 如实报告验证与测试表现及差异；不将单 seed 结果解释为重复训练稳定性，不判断缺少完整 concat 对照的门控贡献。

## 回执

- **来源核验：** 训练正常结束且无报错，来源闸门通过；训练时仅交接文档未提交，源码和配置无变动。
- **FINAL 目录：** `results/brats2020_detail_gated_1113_1530/test_1114_1113/evaluation_test`；固定第 95 轮，完整 55 例，导出 55 份预测；仅执行一次。
- **指标与分析：** 已通过收集器归档到 `EXP-006/final/`，并完成配对分析与[独立测试报告](../../research/experiments/EXP-006/final/report.md)。Dice 有改善，Mean HD95 略变差；完整呈现不确定性，不据此调参。
- **异常与修复：** 无。
