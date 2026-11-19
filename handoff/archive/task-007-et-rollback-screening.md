# TASK-007：ET 残差强度门控筛选

状态：已执行并归档 · 想法：[IDEA-009](../../research/ideas/idea-009-et-rollback.md) · 发起：A，2026-11-18 · 分支见 [STATE](../../STATE.md)

## 目的与运行清单

检验局部特征能否比共享常数更准确地恢复被残差误删的 ET，并用 [EXP-006](../../research/experiments/EXP-006/summary.json) 检查净收益。两组热启动载入同一个第 95 轮 best，原模型冻结且处于 eval，仅更新回调模块。训练和筛选范围为 train/val，沿用 CE + 区域 Dice。

| EXP | 配置 | 说明 |
|---|---|---|
| EXP-006 | 已归档 | 固定基准 |
| EXP-015 | `configs/screening/et-rollback-constant.json` | 常数回调，固定第 10 轮 |
| EXP-016 | `configs/screening/et-rollback-local.json` | 局部回调，固定第 10 轮 |

## 运行

按 [runbook](../../docs/runbook.md) 进入源码目录并激活 `vm_unet`。以下示例用 GPU 3，可换成 GPU 2–5 中空闲的 RTX 6000。

```bash
CUDA_VISIBLE_DEVICES=3 python -m pytest -q tests/test_detail_fusion.py tests/test_warm_start.py tests/test_detail_fusion_cuda.py &&
for mode in constant local; do
  cfg="configs/screening/et-rollback-${mode}.json"
  python run_brats.py smoke --config "$cfg" --gpus 3 || break
  python run_brats.py train --config "$cfg" --gpus 3 --dry-run || break
done
```

检查全部通过且 CUDA 项未跳过后，分别用同一配置去掉 `--dry-run` 启动训练。核对两组 `run.json` 的热启动来源轮次为 95、权重 SHA256 一致，新增张量均属 `detail_fusion.et_rollback.`。既有 10 轮耗时见 [EXP-007](../../research/experiments/EXP-007/summary.json) 和 [EXP-008](../../research/experiments/EXP-008/summary.json)，以本次首轮计时校正。

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

## B 端回执

- 训练提交 `8d94b9d249428c756d80d22c0af19bc4c150a82f`。两组加载 EXP-006 第 95 轮的权重，SHA256 `b21fb1e9ca543297803384b6500592f0560630730e2f446bd54115d709ad604f`。`run.json` 中的划分和新增张量已核对，两组共用固定划分，常数新增 1 个张量，局部新增 4 个，均属 `detail_fusion.et_rollback.`。
- 使用运行清单的两份配置。实际运行目录位于 `VM-UNet-main/results/`，EXP-015 为 `brats2020_et_rollback_constant_screen_1118_1559`（RTX 6000 GPU 2），EXP-016 为 `brats2020_et_rollback_local_screen_1118_1559`（RTX 6000 GPU 3）。
- CPU/CUDA 专项合计 60 passed，耗时 19.40 s。GPU 3 的常数与局部 smoke 完成前后向及 1 例 val，分别耗时 8.55 s 和 7.50 s。两组 dry-run 均通过，常数另在 GPU 2 复核启动计划。
- 两组正常完成 10 轮，收集固定第 10 轮 val。纯训练耗时见各 EXP 摘要。日志在 `VM-UNet-main/logs/`，与运行目录同名。两组第 10 轮 checkpoint 中的 282 个冻结张量逐一与源权重核对一致。
- 55 例验证病例、标注体素计数和逐病例 Dice 与摘要均核验通过。[配对汇总](../../research/experiments/EXP-016/comparison-summary.json)与 `results/analysis/task-007/summary.json` 字节一致。
- [EXP-015](../../research/experiments/EXP-015/report.md)通过 ET Dice 和空 ET 假阳性两项。[EXP-016](../../research/experiments/EXP-016/report.md)仅通过空 ET 假阳性一项，Mean Dice 低于常数方案。两组均未通过筛选，保留 EXP-006。两组评价范围为 val。
