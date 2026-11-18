# TASK-006：细节支路对 ET 的逐患者修正分析

状态：B 已执行，全部验收通过；后续研究建议待用户裁决。发起：A，2026-11-18。分支见 [STATE](../STATE.md)。

## 目的与运行清单

检验 [IDEA-001](../research/ideas/IDEA-001-detail-fusion.md) 的细节支路在推理时补回了哪些 ET、消除了哪些误检，又在哪里引入错误。复用 [EXP-006](../research/experiments/EXP-006/summary.json)，只做一次完整 val 推理。

| 项目 | 取值 |
|---|---|
| 权重 | `VM-UNet-main/results/brats2020_detail_gated_1113_1530/best.pth`，val best 第 95 轮 |
| 病例 | 固定清单的 55 例 val，含全部轴向切片。50 例非空 ET 和 5 例空 ET 分开汇总 |
| 参照 | 同一 EXP-006 前向中的主干 logits 和加残差后的最终 logits。主干与支路共同训练 |
| 核验依据 | [EXP-006 逐患者 CSV](../research/experiments/EXP-006/val_per_patient.csv)及[评价协议](../docs/protocol.md) |

## 步骤与验收

在 B 端同步本次改动后，进入源码目录执行。示例使用 GPU 3，按空闲情况选 2–5 中的一张。

```bash
conda activate vm_unet
cd /home/hzh/hwh/Seg/VM-UNet-main
python -B -m pytest -q tests/test-et-residual-audit.py &&
python -B analysis/et-residual-audit.py --gpus 3 &&
python -B plot_brats.py et-residual-audit
```

1. [专项检查](../VM-UNet-main/tests/test-et-residual-audit.py)覆盖单次前向、补边裁剪、类别映射、区域统计和出图。[分析入口](../VM-UNet-main/analysis/et-residual-audit.py)只读取固定 val 清单。
2. B 按 [runbook](../docs/runbook.md) 选空闲 RTX 6000，通过 `--gpus` 指定。核对 checkpoint 轮次、固定划分、原配置和病例编号，每批切片一次前向取两套四类预测，还原到原始体积。范围限 val，无训练或 FINAL。
3. 验收最终预测：逐例核对归档 CSV 的 WT/TC/ET 标注及预测体素数、Dice。整数应一致，浮点仅容许数值舍入误差；有差异先记录病例并排查权重、推理设置及映射，再解释修正结果。
4. 每位患者按下表计数，记录修正前后 ET Dice、预测体积和每种翻转的原始标签转移。四项计数之和应等于两张 ET 掩膜异或的体素数。

| 真值 | 主干预测 → 最终预测 | 计数 |
|---|---|---|
| ET | 非 ET → ET | 补回漏分 |
| 非 ET | ET → 非 ET | 消除误检 |
| ET | ET → 非 ET | 新增漏分 |
| 非 ET | 非 ET → ET | 新增误检 |

5. 对非空 ET 病例，令 M 为 ET 真值，逐轴向切片用 3×3 邻域腐蚀、膨胀，得到四个互斥区域：腐蚀后的内部、M 内侧一像素边界、M 外侧一像素边界、膨胀范围外。按区域分别统计翻转数与机会数。例如，内侧补回率的分母为该区域原有漏分数，内侧新增漏分率的分母为该区域原本分对的 ET 数。分母为零记缺失并报告有效病例数。
6. 对空 ET 病例，逐例列出修正前后 ET 预测体积及空/非空状态。汇总以患者为单位，报告 ET Dice 配对差的均值、中位数和升/降/持平例数，保留所有病例。
7. 产物位于 `results/analysis/task-006/`：`patients.csv`、`regions.csv`、`transitions.csv`、`summary.json`，图在 `figures/`。非空组按 ET Dice 配对差选最大改善、最接近中位数、最大退步病例（缺少方向记 null），并列取最小病例编号。每例展示 ET 翻转最多的轴向切片，并列取最小索引。重画只运行最后一条命令；另起推理可用 `--output 新目录`，绘图对应传 `--analysis-dir 新目录`。

## 结果解释

| 验证集观察 | 回执应回答的问题 |
|---|---|
| 收益集中在边界 | 补回漏分和消除误检各贡献多少，多少患者有净改善？ |
| 边界受益且远处新增误检 | 哪些区域原本正确却被改成 ET，是否值得研究残差修正强度？ |
| 改动集中在核心内部 | 真值及预测转移是否集中在增强与非增强核心之间，是否值得研究模态判别？ |
| 收益集中在少数病例、空 ET 翻转或缺少一致方向 | 如实报告病例分布与反例，限定机制解释。 |

该诊断解释固定权重下残差对输出的直接作用；与独立训练的 EXP-001 的性能差异另见原实验。后续结构变更由用户裁决。

## 回执（B 填写）

- **实际执行：** 提交 `295abb8a34b01ac3cd28dac3ce085267bcee69a6`、EXP-006 第 95 轮 `best.pth`，`vm_unet`、RTX 6000 GPU 3；按上述三条命令执行。产物 `results/analysis/task-006/`，图为四组 PNG/PDF/SVG；日志 `logs/task-006-{et-residual-audit,plot,summary}.log`。
- **验收：** 专项检查 3 passed；55 例 WT/TC/ET 体素数与归档完全一致、Dice 误差不超过 `1e-12`；820 行区域及 1,122 行转移表计数、体积守恒和配对汇总通过离线交叉核验。无推理异常；绘图缓存目录警告未影响产物，未改模型或协议，未训练、未执行 FINAL。
- **结论：** 见 [报告与统计归档](../research/experiments/EXP-006/et-residual-audit/report.md)。主要作用为 ET 误检抑制，非空组 46 升/2 降/2 平，同时存在小 ET 抹除；不能把共同训练主干的残差前后差值当成独立 baseline 提升。离线核验源码 `analysis/summarize-et-residual-audit.py`，命令 `python -B analysis/summarize-et-residual-audit.py`；建议同时披露收益与反例，后续结构由用户决定；用户已明确授权本轮提交和推送。
