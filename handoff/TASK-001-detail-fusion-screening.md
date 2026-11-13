# TASK-001：detail_fusion 热启动筛选

状态：**执行完成，待用户裁决** · 想法：[IDEA-001](../research/ideas/IDEA-001-detail-fusion.md) · 发起：A，2026-11-12 · 分支：见 STATE.md

## 目的

用 10 轮热启动微调，判断多分辨率细节回注（gated）是否值得做约 11 小时的完整训练。

## 运行清单

热启动来源为 EXP-001 的 `results/brats2020_1102_1734/best.pth`，规则见 [protocol](../docs/protocol.md) 中的"热启动筛选"。

| EXP | 配置 | 说明 | 必要性 |
|---|---|---|---|
| EXP-002 | `configs/screening/baseline.json` | 对照组：不挂模块，同样微调 10 轮 | 必跑 |
| EXP-003 | `configs/screening/detail_gated.json` | 主候选 | 必跑 |
| EXP-004 | `configs/screening/detail_concat.json` | 门控消融 | 建议 |
| EXP-005 | `configs/screening/shallow_gated.json` | 仅 1/4 尺度（IDEA-002） | 可选 |

## 步骤

1. **检查**，约 30 分钟：
   - `python -m pytest -q tests/test_detail_fusion.py tests/test_warm_start.py tests/test_provenance.py tests/test_collect_brats.py`
   - `CUDA_VISIBLE_DEVICES=<卡> python -m pytest -q tests/test_detail_fusion_cuda.py`：必须 passed，skipped 不算通过。
   - 确认 `results/brats2020_1102_1734/best.pth` 存在。
2. **Smoke：** 每个配置各跑一次 `python run_brats.py smoke --config <配置> --gpus <卡>`。新模块零初始化，所以四个配置在第 1 例上的 Dice 应当基本一致。
3. **训练：** 每个配置执行 `python run_brats.py train --config <配置> --gpus <卡> --background`。每个运行占一张卡，可以多卡并行，单个约 1.2–1.5 小时。
4. **收集：** 每个运行执行 `python collect_brats.py --run-dir results/<运行目录> --exp EXP-00x --title "筛选：<名称>" --epoch 10`。四个 EXP 都用第 10 轮，保证可比。

## 通过标准与决策

EXP-003（gated）与 EXP-002（对照）在第 10 轮比较。

| 结果 | 决策 |
|---|---|
| 同时满足以下四条：Mean Dice 至少高 0.5 个百分点；ET Dice 上升；Mean HD95 不变差；空 ET 假阳性不多于对照 | 建议对 gated 做完整训练（新任务单） |
| 有提升但不到 0.5 个百分点 | 视为噪声。检查逐患者配对差中，非空 ET 的 Dice 是否稳定为正，再交给用户决定 |
| 没有提升，或变差 | 建议放弃 IDEA-001，记录负结果 |

附带结论，只用于论文叙事：
- 若 gated 优于 concat，说明门控有效；
- 若 detail 优于 shallow，说明更高的分辨率才是关键。

---

## 回执（B 填写）

- **实际提交：** 训练源码提交 `b448503bd5894f28e6f999abcb9eacc6ff6cdfd6`；本任务未修改模型、损失或实验协议。
- **运行目录：** EXP-002：`results/screening_baseline_1113_1036`，GPU 2，PID 1361389。
  EXP-003：`results/screening_detail_gated_1113_1036`，GPU 3，PID 1361616。
  EXP-004：`results/screening_detail_concat_1113_1036`，GPU 5，PID 1362311。
  EXP-005：`results/screening_shallow_gated_1113_1147`，GPU 2，PID 1555645；已由队列接续完成。
- **异常与修复：** 缓存缺失，按原清单重建 369 例；`vm_unet` 下指定 CPU 检查 43 passed、shallow 补充 18 passed、CUDA 6 passed。只使用 RTX 6000。
- **结论：** 四组 10 轮训练与验证、固定第 10 轮收集和五组配对分析均完成；gated 相对热启动对照通过四项筛选标准。数值及限制见 [EXP-003 报告](../research/experiments/EXP-003/report.md)。
- **建议：** 用户批准后由 A 发起 gated 完整训练，并保留 concat 对照；门控独立收益未证实。筛选结果、报告和分析脚本随本轮阶段分支交付。
