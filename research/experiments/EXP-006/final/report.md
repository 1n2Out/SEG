# EXP-006 FINAL：一次性测试集评价

授权任务：[TASK-003](../../../../handoff/archive/task-003-detail-gated-final.md)，用户于 2026-11-14 要求测试与分析。
对象：val 选出的第 95 轮 `best.pth`；固定 55 例 test，仅评价一次，不按 test 选权重或调参。
源结果：[test_summary.json](test_summary.json)、[test_per_patient.csv](test_per_patient.csv)；对照为 EXP-001 已有 FINAL 结果，未重新测试 baseline。

## 指标

| 指标 | EXP-001 | EXP-006 | 差值（改进−基线） |
|---|---:|---:|---:|
| Mean Dice (%) | 83.19 | 83.94 | +0.75 pp |
| WT Dice (%) | 88.96 | 89.05 | +0.08 pp |
| TC Dice (%) | 82.91 | 83.77 | +0.86 pp |
| ET Dice (%) | 77.68 | 78.99 | +1.30 pp |
| Mean HD95 (mm) | 14.87 | 15.23 | +0.36 mm |
| ET HD95 (mm) | 24.49 | 27.70 | +3.20 mm |
| 空 ET 假阳性 | 2/4 | 2/4 | 不变 |

## 配对分析与限制

- 42/55 例 Mean Dice 上升，13 例下降；配对差中位数 +1.09 个百分点，bootstrap 95% 区间 [-0.19, 1.64] 个百分点，跨过 0，整体平均增益存在不确定性。
- 非空 ET 共 51 例，36 例上升、14 例下降、1 例持平；平均差 +1.41 个百分点。两例空 ET 假阳性身份与 baseline 相同。
- WT/TC 的 HD95 均值改善，但 ET HD95 变差，导致 Mean HD95 略升；因此不能写成所有指标全面改善。
- ET HD95 变差最明显的两例为 `BraTS20_Training_222`（1.00→71.65 mm）、`BraTS20_Training_358`（1.00→64.99 mm）；二者对 Mean HD95 差值贡献约 +0.82 mm，未剔除任何失败病例。
- 区间来自 10,000 次配对患者重采样（seed 42），描述固定模型的患者差异，不包含训练种子不确定性。
- 分析脚本：[compare-full-training.py](../../../../VM-UNet-main/analysis/compare-full-training.py)，命令 `python -B analysis/compare-full-training.py --include-test`；测试产物独立保存于 `results/analysis/exp-006/final/`。

## 结论

本次固定模型的测试 Dice 有改善，ET 改善方向较明确；Mean HD95 未同步改善，且整体 Mean Dice 的患者抽样区间跨 0，应完整呈现这一取舍。
缺少完整 concat 对照，不能归因于门控；测试结果只作最终报告，不用于选择后续架构、参数、训练轮数或后处理。
