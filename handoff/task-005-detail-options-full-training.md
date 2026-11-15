# TASK-005：细节回注选项的完整训练（深监督、输入端补强、门控消融）

状态：待执行 · 想法：[IDEA-007](../research/ideas/IDEA-007-deep-supervision.md)、[IDEA-008](../research/ideas/IDEA-008-input-stem.md)（基座 IDEA-001） · 发起：A，2026-11-15 · 分支：见 STATE.md

## 目的

按完整协议检验深监督、输入端补强及两者组合能否在 EXP-006 上继续提升 val 指标，并补齐 concat 门控消融（决定见 decisions 2026-11-15）。

## 运行清单

| EXP | 配置 | 说明 | 必要性 |
|---|---|---|---|
| EXP-011 | `configs/detail_fusion/gated-ds.json` | gated + 深监督 | 必跑 |
| EXP-012 | `configs/detail_fusion/gated-stem.json` | gated + 输入端补强 | 必跑 |
| EXP-013 | `configs/detail_fusion/gated-ds-stem.json` | 两者都加 | 必跑 |
| EXP-014 | `configs/detail_fusion/concat.json` | 门控消融（无门控） | 必跑 |

全部从 VMamba 预训练出发，协议与 EXP-006 相同（配置相对 `gated.json` 只多出选项和输出目录）。对照为已有的 EXP-006，不重跑。

## 步骤

1. **检查**（必须 passed，skipped 不算）：
   - `python -m pytest -q tests/test_detail_fusion.py tests/test_warm_start.py tests/test_brats_torch.py`
   - `CUDA_VISIBLE_DEVICES=<卡> python -m pytest -q tests/test_detail_fusion_cuda.py`：8 项
   - `BRATS_TEST_DISTRIBUTED=1 python -m pytest -q tests/test_detail_fusion_distributed.py`：2 项
2. **smoke：** 4 个配置各跑一次 `python run_brats.py smoke --config <配置> --gpus <卡>`，均应完成前后向和 1 例 val。
3. **训练：** 4 个配置分别占 GPU 2、3、4、5，后台并行；先 `--dry-run` 核对输出目录。单个约 13–14 h，两个选项的额外开销很小。其他用户占卡时，训练耗时不作严格比较。
4. **收集：** `python collect_brats.py --run-dir results/<运行目录> --exp EXP-0xx --title "<标题>"`，按 val best，不加 `--epoch`。标题依次为"完整训练：gated + 深监督""完整训练：gated + 输入端补强""完整训练：gated + 深监督 + 输入端补强""完整训练：detail concat"。
5. **分析：** 以 EXP-006 为对照做逐患者配对，只用 val。脚本可在 `analysis/compare-full-training.py` 基础上扩展，入库。
6. **不执行 FINAL。** 最终方法由用户确认后另发任务。

## 通过标准与决策

**候选通过：** EXP-011～013 各自相对 EXP-006（val best），同时满足：
1. Mean Dice 至少高 0.3 个百分点（单种子下更小的差异视为无差别）；
2. ET Dice 不下降；
3. 非空 ET 逐例配对：Dice 上升的病例多于下降的病例；
4. 空 ET 假阳性病例数不增加。

Mean HD95 逐项报告，不作门槛：它由少数病例主导（见 IDEA-005 的结论）。门控消融比较 EXP-014 与 EXP-006 的 Mean Dice、ET Dice 和逐患者配对。

| 结果 | 决策或建议 |
|---|---|
| 有候选通过 | 列出全部通过者，建议 Mean Dice 最高者为最终方法，交用户确认 |
| 候选都未通过 | EXP-006 仍为最终方法；如实报告，不调参重跑 |
| gated 比 concat 至少高 0.3 个百分点 | 门控可写成贡献 |
| 两者差距在 ±0.3 个百分点以内 | 论文不单独宣称门控；是否简化为 concat 由用户决定 |
| concat 至少高 0.3 个百分点 | 如实报告，基座选择交用户 |

---

## 回执（B 填写）

- **实际提交：** `<git.commit>`
- **运行目录：** 每个 EXP 对应的运行目录
- **异常与修复：** 遇到的问题，以及 `[B] fix` 提交的哈希
- **检查结果：**
- **结论：** 对照上面的通过标准逐项判断，并链接 EXP 报告
- **建议：**
