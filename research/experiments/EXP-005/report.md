# EXP-005：shallow gated 尺度对照

任务：[TASK-001](../../../handoff/TASK-001-detail-fusion-screening.md) · 想法：IDEA-002
运行目录：`screening_shallow_gated_1113_1147` · 提交：`b448503bd5894f28e6f999abcb9eacc6ff6cdfd6`

## 目的

以仅 1/4 尺度的浅层回注作为对照，检验多分辨率细节支路是否值得继续验证。

## 与协议的差异

采用 [protocol](../../../docs/protocol.md) 的 10 轮热启动筛选；原参数 lr 1e-5、新分支 lr 1e-4。启用 shallow_refinement.gated，固定比较第 10 轮。
运行启动时仅 STATE 和任务单存在未提交文档改动；源码哈希与 EXP-002 至 EXP-004 相同。

## 结果

数值来源：[summary.json](summary.json)、[验证集总表](../index.md)及对应的 `val_per_patient.csv`。
相对 EXP-002：Mean Dice +0.37 个百分点，低于 0.5 个百分点的筛选幅度；空 ET 假阳性维持 3/5。

## 观察

分析脚本：[compare-screening.py](../../../VM-UNet-main/analysis/compare-screening.py)，运行：`python -B analysis/compare-screening.py`（源码目录）。
本地产物：`results/analysis/task-001/comparison-summary.json` 和五份逐患者配对 CSV；只使用 val。

- Mean Dice 仅 30/55 例上升，配对差中位数 +0.03 个百分点。
- 非空 ET 中 23 例上升、24 例下降、3 例持平；方向不一致。
- detail gated 相对本组 Mean Dice +0.55 个百分点，支持继续验证更高分辨率支路；由于结构和参数量也不同，不能据此把分辨率认定为唯一原因。

## 结论与建议

未达到同等 0.5 个百分点的筛选幅度，且非空 ET 没有稳定同向改善。建议保留为尺度对照，不推荐单独升级为主线。
这些数字只作热启动筛选；方法的采纳或放弃由用户决定。
