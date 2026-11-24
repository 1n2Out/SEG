# TASK-009 对侧半球参照完整训练

状态：待执行 · 发起：A，2026-11-24 · [IDEA-011](../research/ideas/idea-011-contralateral-reference.md) · [结构与配置](../docs/methods/contralateral-reference.md)

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
```

## 通过标准与后续

以未舍入的配对结果对照 EXP-006。非空 ET 组按每名患者三个区域 Dice 的平均值计算配对差，再对患者取均值。

| val 结果 | 后续 |
|---|---|
| Mean Dice 至少 +0.30 个百分点且非空 ET 组配对均值为正 | 报告分区收益与代价，提交用户裁决是否开展同容量原图消融 |
| Mean Dice 上升，仅部分满足上述条件 | 报告幅度、空 ET 病例及其余病例贡献，交用户判断 |
| Mean Dice 持平或下降 | 报告失败病例与分区代价，优先保留 EXP-006 |

## 回执（B 填写）

- 训练提交、GPU、目录、PID / 日志：
- 检查、异常与修复：
- EXP-018 报告、配对结果及建议：
