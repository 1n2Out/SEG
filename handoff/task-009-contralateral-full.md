# TASK-009 对侧半球参照完整训练

状态：训练与分析完成，待用户验收 · 发起：A，2026-11-24 · [IDEA-011](../research/ideas/idea-011-contralateral-reference.md) · [结构与配置](../docs/methods/contralateral-reference.md)

## 目的与运行清单

检验在 EXP-006 的 1/2 解码特征加入对侧参照，能否提高 val 患者级 Mean Dice。仅新增 EXP-018 一组完整训练，从 VMamba 预训练独立初始化，按[统一协议](../docs/protocol.md)选 val best。主对照为 [EXP-006](../research/experiments/EXP-006/summary.json)，原始 VM-UNet 参照为 [EXP-001](../research/experiments/EXP-001/summary.json)。

## B 端执行

1. 收到用户训练指令后，按 [runbook](../docs/runbook.md) 同步包含本任务的提交，确认源码与配置干净，检查 EXP-018 是否已有运行或 PID。激活 `vm_unet`，进入 `VM-UNet-main`，在 RTX 6000 的 GPU 2–5 中选一张空闲卡。
2. 依次执行检查，CUDA 专项须显示 1 passed，smoke 须完成正式 batch 的前向、反向及一名 val 患者的整例评价。核对 dry-run 的配置、工作目录与卡号。

```bash
gpu=2  # 改为实际选中的空闲卡
cfg=configs/detail_fusion/contralateral.json
python -m pytest -q tests/test_brats_torch.py::test_contralateral_training_and_volume_geometry tests/test_detail_fusion.py::test_contralateral_formal_config_and_learning
CUDA_VISIBLE_DEVICES="$gpu" python -m pytest -q -rs tests/test_detail_fusion_cuda.py::test_contralateral_real_backbone_amp_and_checkpoint
python run_brats.py smoke --config "$cfg" --gpus "$gpu"
python run_brats.py train --config "$cfg" --gpus "$gpu" --dry-run
```

3. 检查通过后启动，记录启动器输出的目录、PID 和日志。核对运行提交、患者划分及预训练文件哈希与共享权重加载数量，后两项对照 EXP-006。确认进入首轮。耗时可先参考 EXP-006 的 13.26 小时，再按实际轮时估算。中断时按 runbook 在同一目录续跑。

```bash
python run_brats.py train --config "$cfg" --gpus "$gpu" --background
```

4. 完成 100 轮或触发协议早停后，收集 val best 并按病例配对。用实际运行目录替换下面的占位符。按[报告模板](../research/experiments/TEMPLATE.md)写 EXP-018 报告，列出 WT、TC、ET Dice、Mean Dice、Mean HD95、空 ET 误检变化、非空 ET 组配对 Mean Dice 和退步最多病例，填写回执并更新 STATE。

```bash
python collect_brats.py --run-dir "results/<实际运行目录>" --exp EXP-018 --title "完整训练：对侧半球参照与细节回注"
python analysis/compare-direction-fusion.py --candidate EXP-018
python analysis/analyze-contralateral-results.py
```

## 通过标准与后续

以未舍入的配对结果对照 EXP-006。非空 ET 组按每名患者三个区域 Dice 的平均值计算配对差，再对患者取均值。

| val 结果 | 后续 |
|---|---|
| Mean Dice 至少 +0.30 个百分点且非空 ET 组配对均值为正 | 报告分区收益与代价，提交用户裁决是否开展同容量原图消融 |
| Mean Dice 上升，仅部分满足上述条件 | 报告幅度、空 ET 病例及其余病例贡献，交用户判断 |
| Mean Dice 持平或下降 | 报告失败病例与分区代价，优先保留 EXP-006 |

## 回执（B 填写）

- EXP-018 在 `main` 的 `0570fef8878d8a9851d01de5d407727d78a9f72e` 上启动，使用 RTX 6000 Ada GPU 5，PID `4183571`。在 `VM-UNet-main/` 下，运行目录为 `results/brats2020_contralateral_1124_1755/`，日志和 PID 文件为 `logs/brats2020_contralateral_1124_1755.log`、`logs/brats2020_contralateral_1124_1755.pid`。
- CPU 检查 3 passed。CUDA 专项的初始缩放值 128 使第三次反传中的差异卷积和门控梯度下溢。测试改用与正式训练一致的默认值 65,536，原有断言全部通过（1 passed）。正式尺寸 smoke 完成前向、反向和单例 val，dry-run 核对通过。
- `run.json` 与 EXP-006 的患者划分、预训练文件 SHA256、共享加载的 208 个张量（25,842,528 个参数）一致。EXP-018 共 27,501,939 个参数，单卡全局 batch 16。
- 训练正常完成 100 轮，核对 `latest.pth` 为第 100 轮、`best.pth` 为第 75 轮。已收集 val best，完成与 EXP-006、EXP-001 的逐病例配对。Mean Dice 小幅上升，两个判据均未达到，后续消融待用户裁决。[EXP-018 报告](../research/experiments/EXP-018/report.md) · [配对汇总](../research/experiments/EXP-018/comparison-summary.json)
- 深入复核使用 EXP-006、EXP-018 原始运行目录中的逐轮验证结果和固定 val 标签缓存。上面的最后一条命令可再生成病例贡献、配对抽样区间、同轮次对照和镜像重叠相关性。[验证复核](../research/experiments/EXP-018/validation-audit.json)
