# TASK-008：四方向分组融合完整训练

状态：已关闭归档 · 发起：A，2026-11-23 · [IDEA-010](../../research/ideas/idea-010-direction-difference-fusion.md) · [方法](https://github.com/1n2Out/SEG/blob/exp-017-direction-fusion/docs/methods/direction-fusion.md)

## 目的与运行清单

| EXP | 配置或参照 | 用途 |
|---|---|---|
| EXP-017 | [direction-all.json](https://github.com/1n2Out/SEG/blob/exp-017-direction-fusion/VM-UNet-main/configs/detail_fusion/direction-all.json) | 唯一新训练，全部 15 个 VSSBlock 的方向融合 + gated 细节回注 |
| [EXP-006](../../research/experiments/EXP-006/summary.json) | 归档完整训练的 val best | 主对照，评价第二点在第一点之上的净收益 |
| [EXP-001](../../research/experiments/EXP-001/summary.json) | 归档完整训练的 val best | 参照两项组合相对原 VM-UNet 的总效果 |

按[统一协议](../../docs/protocol.md)从 VMamba 预训练独立开始，单卡 batch 16、seed 42、最多 100 轮、沿用早停和 val best。跳过热启动筛选。范围为 train/val。

## B 端步骤

按 [runbook](../../docs/runbook.md) 检查工作区与在跑的任务，复核时在独立副本中运行 `git fetch origin tag exp-017-direction-fusion` 和 `git switch --detach exp-017-direction-fusion`。确认本任务单和正式配置都在该提交内，源码与配置干净。先检查 EXP-017 的回执、已有运行目录和 PID，避免重复开跑。进入 VM-UNet-main，激活 vm_unet，按 nvidia-smi 从 2–5 选一张空闲卡。

~~~bash
cfg=configs/detail_fusion/direction-all.json
gpu=3  # 换成选好的物理卡号
CUDA_VISIBLE_DEVICES="$gpu" python -m pytest -q -rs tests/test_detail_fusion_cuda.py::test_all_direction_amp_against_gated_and_gradients
python run_brats.py smoke --config "$cfg" --gpus "$gpu"
python run_brats.py train --config "$cfg" --gpus "$gpu" --dry-run
~~~

逐项确认后再执行下一项。CUDA 专项须显示 1 passed，覆盖真实扫描、AMP 初始输出及两步反传；smoke 须在正式尺寸和 batch 下完成前向、反向及单例 val。若显存不足，记录失败位置及可用显存信息后交回 A。检查 dry-run 输出的卡号、来源告警、配置与目录。

~~~bash
python run_brats.py train --config "$cfg" --gpus "$gpu" --background
~~~

用正式启动器实际打印的 Output directory、Log 和 PID file 填回执，按 Follow 命令查看日志，确认进程进入首轮训练。记录本次 run.json 的 git.commit。对照 EXP-006 原运行的 run.json 核对 manifest_sha256、pretrained_report.source_sha256 和共享权重加载数。参数总数应为 27,539,282。以 [EXP-006 耗时](../../research/experiments/EXP-006/summary.json) 作参考，按首轮计时修正。中断时按 runbook 从原运行目录续跑。

~~~bash
python collect_brats.py --run-dir "results/<本次运行目录>" --exp EXP-017 --title "完整训练：全主干方向融合与细节回注"
~~~

收集正常结束或按协议早停后的 val best。用两个实验的 summary.json 和 val_per_patient.csv 按 case_id 对齐，并核对区域真值体素数。报告 Mean、WT/TC/ET Dice、Mean HD95 和空 ET 误检病例变化。非空 ET 组取 ET_target_voxels > 0 的患者，先算每人的三分区 Dice 平均配对差，再对患者等权平均，统一换算为百分点。分析脚本按项目约定入库，按[报告模板](../../research/experiments/TEMPLATE.md)写结论、填回执并更新 STATE。

## 判定建议

| 相对 EXP-006 的 val Mean Dice | 后续 |
|---|---|
| ≥ +0.3 个百分点，非空 ET 组配对 Mean Dice 也上升 | 作为有收益的组合交用户判断 |
| 正增益但未达到上述条件 | 报告幅度与病例来源，交用户判断 |
| 持平或下降 | 报告分区代价，优先保留 EXP-006 |

一次训练只支持该种子下组合对照。额外消融、正式采纳与 FINAL 由用户另行决定。

## 回执（B 填写）

- EXP-017 训练提交为 `acd6227d945dc8b380c30652f6221f403bda0b31`，使用 `vm_unet` 和 3 号 NVIDIA RTX 6000 Ada，PID 为 `500132`。启动日志已确认第 1 轮首批完成。
- 以下路径相对 `VM-UNet-main/`。运行目录为 `results/brats2020-direction-all_1123_2147/`，日志为 `logs/brats2020-direction-all_1123_2147.log`，PID 文件为 `logs/brats2020-direction-all_1123_2147.pid`。
- 已核对两次运行的 `run.json`。与 EXP-006 的划分清单、预训练文件 SHA256 和加载的 208 个张量（25,842,528 个参数）一致。本模型共 27,539,282 个参数。首轮训练 569.3 秒，80 轮累计训练 12.69 小时（不计验证）。
- CUDA 专项连续两次在第二步反传失败，首个失败参数为 `vmunet.layers.0.blocks.0.self_attention.direction_fusion.describe.weight`，梯度为零。初始 AMP 前向与 gated 逐元素一致，梯度有限性检查通过。
- 修复前固定合成输入种子 42，第二步反传时测试缩放 128 下有 15 个块的描述器权重梯度全零，缩放 65,536 下有 3 个，FP32 下为 0 个。
- B 端将方向融合分支的计算保留为 FP32。原 CUDA/AMP 专项两次通过，CPU 方向测试 2 passed。3 号卡上 16 × 4 × 256 × 256 的 smoke 完成前向、反向和单例 val，产物在 `results/brats2020-direction-all_smoke_1123_2145/`。修复提交后的 dry-run 指向正确配置与 GPU，无来源告警。
- 第 80 轮正常早停，已收集第 30 轮 val best，核对三组的 55 个病例及分区真值体素数。相对 EXP-006，Mean Dice 和非空 ET 组的配对 Mean Dice 均下降，按任务判据优先保留 EXP-006。[实验报告](../../research/experiments/EXP-017/report.md) · [配对汇总](../../research/experiments/EXP-017/comparison-summary.json)
- 同轮次对照、分区误差、深层方向权重及待验证的机制假设见[原因核查](../../research/experiments/EXP-017/report.md#原因核查)。
