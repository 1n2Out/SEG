# TASK-007：ET 残差强度门控筛选

状态：待执行 · 想法：[IDEA-009](../research/ideas/idea-009-et-rollback.md) · 发起：A，2026-11-18 · 分支见 [STATE](../STATE.md)

## 目的与运行清单

检验局部特征能否比共享常数更准确地恢复被残差误删的 ET，并用 [EXP-006](../research/experiments/EXP-006/summary.json) 检查净收益。两组热启动载入同一个第 95 轮 best，原模型冻结且处于 eval，仅更新回调模块。训练和筛选范围为 train/val，沿用 CE + 区域 Dice。

| EXP | 配置 | 说明 |
|---|---|---|
| EXP-006 | 已归档 | 固定基准 |
| EXP-015 | `configs/screening/et-rollback-constant.json` | 常数回调，固定第 10 轮 |
| EXP-016 | `configs/screening/et-rollback-local.json` | 局部回调，固定第 10 轮 |

## 运行

按 [runbook](../docs/runbook.md) 进入源码目录并激活 `vm_unet`。以下示例用 GPU 3，可换成 GPU 2–5 中空闲的 RTX 6000。

```bash
CUDA_VISIBLE_DEVICES=3 python -m pytest -q tests/test_detail_fusion.py tests/test_warm_start.py tests/test_detail_fusion_cuda.py &&
for mode in constant local; do
  cfg="configs/screening/et-rollback-${mode}.json"
  python run_brats.py smoke --config "$cfg" --gpus 3 || break
  python run_brats.py train --config "$cfg" --gpus 3 --dry-run || break
done
```

检查全部通过且 CUDA 项未跳过后，分别用同一配置去掉 `--dry-run` 启动训练。核对两组 `run.json` 的热启动来源轮次为 95、权重 SHA256 一致，新增张量均属 `detail_fusion.et_rollback.`。既有 10 轮耗时见 [EXP-007](../research/experiments/EXP-007/summary.json) 和 [EXP-008](../research/experiments/EXP-008/summary.json)，以本次首轮计时校正。

```bash
python collect_brats.py --run-dir "$constant_run" --exp EXP-015 --title "筛选：ET 常数回调" --epoch 10
python collect_brats.py --run-dir "$local_run" --exp EXP-016 --title "筛选：ET 局部回调" --epoch 10
python analysis/compare-et-rollback.py
```

将两个变量设为对应实际运行目录。汇总位于 `results/analysis/task-007/summary.json`，记录非空 ET 逐患者升降、恢复真 ET 与新增假阳性体素、空 ET 状态变化及 Mean HD95。

## 验收和决策

候选相对 EXP-006 同时满足四项才通过。Mean Dice 至少升 0.3 个百分点，ET Dice 不降，非空 ET 逐病例上升数多于下降数，无新出现假阳性的空 ET 病例。Mean HD95 单列。局部方案还须在 Mean Dice 上优于常数方案。

| 结果 | 决策 |
|---|---|
| 局部通过四项且优于常数 | B 归档结果，交用户决定是否正式训练 |
| 常数与局部相当或更优 | 报告实际差值，局部特征的增益未获验证 |
| 常数单独通过四项 | 记录常数回调的收益，交用户裁决 |
| 两组均未达门槛 | 基于本轮证据保留 EXP-006，结束筛选 |
| 归档病例、来源权重、体素计数核验失败 | 先定位配置或环境差异，暂缓解读 |

## 回执（B 填写）

- 实际提交、权重 SHA256、配置和运行目录：
- CPU / CUDA / smoke / dry-run 结果与耗时：
- EXP 报告、汇总和逐项达标情况：
