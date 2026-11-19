# EXP-016：ET 局部回调筛选

任务 [TASK-007](../../../handoff/archive/task-007-et-rollback-screening.md) · 想法 [IDEA-009](../../ideas/idea-009-et-rollback.md)

运行目录 `brats2020_et_rollback_local_screen_1118_1559` · 训练提交 `8d94b9d` · RTX 6000 GPU 3

## 目的

检验结合局部特征预测回调系数能否比共享常数更准确地恢复被残差误删的 ET。

## 与协议的差异

热启动加载 EXP-006 第 95 轮 best，冻结原模型且保持 eval。仅训练局部回调，初始学习率为 `1e-4`，固定评价第 10 轮 val。配置和训练曲线见 [config.json](config.json) 与 [history.jsonl](history.jsonl)。

## 结果与结论

- 相对 EXP-006，Mean Dice 下降 0.000157 个百分点，ET Dice 下降 0.000247 个百分点，Mean HD95 下降 0.000108 mm。Mean Dice 相对常数回调同样下降 0.000157 个百分点。见 [实验摘要](summary.json) 和 [配对汇总](comparison-summary.json)。
- 50 例非空 ET 中，0 例上升、1 例下降、49 例持平。没有恢复真 ET，`BraTS20_Training_232` 新增 6 个 ET 假阳性体素。空 ET 患者的假阳性状态不变。见 [逐患者 CSV](val_per_patient.csv) 和 [配对汇总](comparison-summary.json)。
- 仅通过空 ET 假阳性一项门槛。局部特征的增益未获验证，按任务单结束筛选，保留 EXP-006。

配对汇总由 [compare-et-rollback.py](../../../VM-UNet-main/analysis/compare-et-rollback.py) 生成，默认输出 `results/analysis/task-007/summary.json`。
