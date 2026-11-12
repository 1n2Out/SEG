# TASK-001：detail_fusion 热启动筛选

状态：**待执行** · 想法：[IDEA-001](../research/ideas/IDEA-001-detail-fusion.md) · 发起：A，2026-11-12 · 分支：见 STATE.md

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

- **实际提交：**
- **运行目录：**
- **异常与修复：**
- **结论：**
- **建议：**
