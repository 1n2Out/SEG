# EXP-004：detail concat 门控消融

任务：[TASK-001](../../../handoff/archive/task-001-detail-fusion-screening.md) · 想法：IDEA-001
运行目录：`screening_detail_concat_1113_1036` · 提交：`b448503bd5894f28e6f999abcb9eacc6ff6cdfd6`

## 目的

去掉语义门控，检验细节支路本身是否有效，以及门控能否提供额外收益。

## 与协议的差异

采用 [protocol](../../../docs/protocol.md) 的 10 轮热启动筛选；原参数 lr 1e-5、新分支 lr 1e-4。启用 detail_fusion.concat，固定比较第 10 轮。

## 结果

数值来源：[summary.json](summary.json)、[验证集总表](../index.md)及对应的 `val_per_patient.csv`。
相对 EXP-002：Mean Dice +0.89 个百分点，ET Dice +2.05 个百分点，Mean HD95 -1.34 mm；空 ET 假阳性维持 3/5。

## 观察

分析脚本：[compare-screening.py](../../../VM-UNet-main/analysis/compare-screening.py)，运行：`python -B analysis/compare-screening.py`（源码目录）。
本地产物：`results/analysis/task-001/comparison-summary.json` 和五份逐患者配对 CSV；只使用 val。

- 相对 EXP-002，46/55 例患者 Mean Dice 上升；非空 ET 中 41/50 例上升，说明改善并非只来自少数患者。
- gated 的平均 Mean Dice 仅比 concat 高 0.03 个百分点，其 ET Dice 反而更低；本轮不支持“门控有效”的独立叙事。
- gated 的 Mean HD95 均值更好，但逐患者中位差方向相反；不据均值优势作普遍改善的推断。

## 结论与建议

结果支持继续验证细节分支；门控增益尚未证实。建议完整训练保留 concat 对照，由用户决定是否纳入下一任务。
这些数字只作热启动筛选，不替代完整协议训练。
