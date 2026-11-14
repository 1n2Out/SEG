# EXP-003：detail gated 热启动筛选

任务：[TASK-001](../../../handoff/archive/task-001-detail-fusion-screening.md) · 想法：IDEA-001
运行目录：`screening_detail_gated_1113_1036` · 提交：`b448503bd5894f28e6f999abcb9eacc6ff6cdfd6`

## 目的

判断多分辨率细节回注与语义门控是否达到 TASK-001 的完整训练准入标准。

## 与协议的差异

采用 [protocol](../../../docs/protocol.md) 的 10 轮热启动筛选；原参数 lr 1e-5，新分支 lr 1e-4，固定比较第 10 轮。启用 detail_fusion.gated，其余公共配置与 EXP-002 相同。

## 结果

数值来源：[summary.json](summary.json)、[验证集总表](../index.md)及对应的 `val_per_patient.csv`。
相对 [EXP-002](../EXP-002/report.md)：Mean Dice +0.92 个百分点，ET Dice +1.78 个百分点，Mean HD95 -2.08 mm；空 ET 假阳性维持 3/5。
TASK-001 四项筛选标准全部满足；相对原 EXP-001 的 Mean Dice 仅 +0.33 个百分点，且空 ET 假阳性多于原来的 2/5，不能混淆两个对照。

## 观察

分析脚本：[compare-screening.py](../../../VM-UNet-main/analysis/compare-screening.py)，运行：`python -B analysis/compare-screening.py`（源码目录）。
本地产物：`results/analysis/task-001/comparison-summary.json` 和五份逐患者配对 CSV；只使用 val。

- 相对 EXP-002，47/55 例患者 Mean Dice 上升，配对差中位数 +0.77 个百分点。
- 非空 ET 共 50 例，42 例上升、5 例下降、3 例持平；平均增益 +1.96 个百分点，改善方向较一致。
- 相对 concat，Mean Dice 仅 +0.03 个百分点，只有 21/55 例上升，ET Dice -0.27 个百分点；不足以证明门控的独立收益。
- 相对 shallow，Mean Dice +0.55 个百分点，45/55 例上升；支持继续验证多分辨率支路，但不是对分辨率作用的单因素证明。

## 结论与建议

**热启动筛选通过。** 建议用户批准后，由 A 发起 gated 的完整协议训练任务，并保留 concat 对照来检验门控贡献。
单 seed 热启动结果不代表完整训练提升或统计显著性；不更新 IDEA 的采纳状态，不启动完整训练或 FINAL 测试。
