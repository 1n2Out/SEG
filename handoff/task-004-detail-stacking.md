# TASK-004：细节回注基座上的叠加筛选与完整训练

状态：待执行 · 想法：[IDEA-005](../research/ideas/IDEA-005-boundary-supervision.md)、[IDEA-006](../research/ideas/IDEA-006-slice-context.md)（基座 IDEA-001） · 发起：A，2026-11-14 · 分支：见 STATE.md

## 目的

检验边界辅助监督和细节支路相邻切片能否在 EXP-006 上继续提升 val 指标；通过的组合按完整协议训练一次（决定见 decisions 2026-11-14）。

## 运行清单

| EXP | 配置 | 说明 | 必要性 |
|---|---|---|---|
| EXP-007 | `configs/screening/stack-control.json` | 对照：EXP-006 best 出发，不加新部件，同样微调 10 轮 | 必跑 |
| EXP-008 | `configs/screening/stack-boundary.json` | 加边界辅助监督 | 必跑 |
| EXP-009 | `configs/screening/stack-context.json` | 加相邻切片 | 必跑 |
| EXP-010 | `configs/screening/stack-boundary-context.json` | 两者都加 | 必跑 |
| EXP-011 起 | `configs/detail_fusion/gated-*.json` | 第二阶段完整训练，按决策表选择 | 条件 |

## 步骤

1. **检查**（必须 passed，skipped 不算）：
   - `python -m pytest -q tests/test_detail_fusion.py tests/test_warm_start.py tests/test_brats_torch.py`
   - `CUDA_VISIBLE_DEVICES=<卡> python -m pytest -q tests/test_detail_fusion_cuda.py`
   - `BRATS_TEST_DISTRIBUTED=1 python -m pytest -q tests/test_detail_fusion_distributed.py`
   - 确认 `results/brats2020_detail_gated_1113_1530/best.pth` 存在，且为第 95 轮。
2. **smoke：** `python run_brats.py smoke --config configs/screening/stack-boundary-context.json --gpus <卡>`。日志中热启动行应显示 `epoch 95` 和 `new=13`；`stack-control` 也跑一次，应为 `new=0`。
3. **筛选：** 四个配置分别放在 GPU 2、3、4、5，后台并行；先 `--dry-run` 核对输出目录。EXP-003 单次训练约 1.3 h（不含验证）。
4. **收集：** `python collect_brats.py --run-dir results/<运行目录> --exp EXP-00x --title "<标题>" --epoch 10`，固定比较第 10 轮。逐患者配对分析以 EXP-007 为对照，脚本放进 `analysis/`（可在 `compare-screening.py` 基础上扩展）。
5. **完整训练：** 按下面的决策表直接启动，不再等待确认；每个运行一张 RTX 6000，约 13–14 h。收集时按 val best，不加 `--epoch`。
6. **不执行 FINAL。** 最终组合模型的测试另发任务，中间模型都不测 test。

## 通过标准与决策

**筛选通过：** 候选相对 EXP-007 同时满足：Mean Dice 至少高 0.5 个百分点；ET Dice 上升；Mean HD95 不变差；空 ET 假阳性病例数不增加。

| 筛选结果 | 第二阶段 |
|---|---|
| EXP-010（两者都加）通过 | 完整训练 `gated-boundary-context`（最终候选），同时完整训练 `gated-boundary` 和 `gated-context` 作为消融行 |
| EXP-010 未通过，只有一个单件通过 | 只完整训练该单件，作为最终候选 |
| EXP-010 未通过，两个单件都通过 | 两个单件都完整训练；最终候选由用户按 val 裁决 |
| 都未通过 | 不做完整训练；如实报告，回执交回用户 |

**完整训练的结论：** 以 EXP-006 为对照，四项条件（Mean Dice 更高、ET 不下降、Mean HD95 不变差、空 ET 假阳性不增加）逐项报告。满足时建议用户确定最终方法并发起 FINAL；不满足时如实报告，不调参重跑。

---

## 回执（B 填写）

- **实际提交：**
- **运行目录：**
- **异常与修复：**
- **结论：**
- **建议：**
