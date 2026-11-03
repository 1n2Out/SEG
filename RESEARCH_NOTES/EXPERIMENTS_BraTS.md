# BraTS2020 实验记录

## 公共配置与评价

| 项目 | 设置 |
| --- | --- |
| 数据 | 369 例，按患者固定划分 train/val/test = 259/55/55，seed=42 |
| 输入 | T1/T1ce/T2/FLAIR，二维轴向切片，256×256；保留全部 40,145 张训练切片 |
| 预处理与增强 | 各模态非零体素 Z-score；居中零填充；随机 90° 倍数旋转、翻转 |
| 主干与初始化 | VM-UNet，编码深度 `[2,2,2,2]`，解码深度 `[2,2,2,1]`，drop path=0.2；`vmamba_small_e238_ema.pth` 预训练初始化 |
| 优化 | AdamW，lr=1e-4，weight decay=0.01；余弦调度 T_max=100 |
| 训练 | 单 RTX 6000 Ada，batch=16，workers=4，AMP；推理 batch=8；两次均完成 100 轮 |
| 选模与早停 | 第 1 轮及每 5 轮验证，按三维验证 Mean Dice 选 best；patience=10 次验证，min_delta=1e-4，两次均未触发 |

采用 `local-brats2020-volume-v1`：逐患者计算三维指标后等权平均；Mean 为 WT/TC/ET 的宏平均。测试均为 **55 例本地保留患者**。表中 ± 为患者间总体 SD（ddof=0）；目前均为 seed=42 单次训练。

空掩膜规则：两者均空时 Dice=1、HD95=0；仅一方为空时 Dice=0、HD95 取物理体积对角线（本数据为 373.13 mm）。无 TTA 或后处理。完整划分见 [split.json](../VM-UNet-main/results/brats2020_1102_1734/split.json)。

## 结果总览

| 实验 | 最佳 epoch | 验证 Mean Dice ↑ | 测试 Mean Dice ↑ | 测试 Mean HD95 ↓ | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| EXP-001 Baseline | 55 | 81.78% | 83.19% | 14.87 mm | 后续对照基准 |
| EXP-002 区域存在性感知 full | 55 | 81.08% | 82.74% | 16.01 mm | 整体退步，已停用 |

## EXP-001：Baseline

### Motivation

建立四模态 VM-UNet 的训练与三维评价基准。

### Training Configuration

损失为 **CE + WT/TC/ET Soft Dice 均值**。参数量 27,429,172；100 轮训练循环累计 11.15 h，不含验证与保存。

训练目录：`results/brats2020_1102_1734`。依据：[配置](../VM-UNet-main/results/brats2020_1102_1734/config.json)、[运行信息](../VM-UNet-main/results/brats2020_1102_1734/run.json)、[训练记录](../VM-UNet-main/results/brats2020_1102_1734/history.jsonl)。

### Result

| 测试区域 | Dice ↑（%） | HD95 ↓（mm） |
| --- | ---: | ---: |
| WT | 88.96 ± 7.66 | 10.26 ± 17.02 |
| TC | 82.91 ± 16.81 | 9.85 ± 18.51 |
| ET | 77.68 ± 23.81 | 24.49 ± 84.23 |

**结论：** 第 55 轮最佳，后期验证进入平台；保留为后续实验基准。

结果：[验证汇总](../VM-UNet-main/results/brats2020_1102_1734/validation/epoch_055/summary.json)、[测试汇总](../VM-UNet-main/results/brats2020_1102_1734/test_1103_0902/evaluation_test/summary.json)、[逐患者指标](../VM-UNet-main/results/brats2020_1102_1734/test_1103_0902/evaluation_test/per_patient.csv)。

## EXP-002：区域存在性感知监督（full，已停用）

### Motivation

同时减少 ET 空区域假阳性和微小 ET 漏检。

### Training Configuration

公共配置同 EXP-001，新增存在性辅助头与条件化分割损失：

`L = CE + Dice_positive + 0.1 × EmptyFP_topk + 0.1 × PresenceBCE`

辅助头 hidden=64；Top-k 比例 0.01，正负组分别归一化。评价时不执行辅助头。参数量 27,441,719；100 轮训练循环累计 11.39 h。

训练目录：`results/brats2020_region_presence_full_1103_1004`。依据：[配置](../VM-UNet-main/results/brats2020_region_presence_full_1103_1004/config.json)、[运行信息](../VM-UNet-main/results/brats2020_region_presence_full_1103_1004/run.json)、[训练记录](../VM-UNet-main/results/brats2020_region_presence_full_1103_1004/history.jsonl)。

### Result

| 测试区域 | Dice ↑（%） | HD95 ↓（mm） |
| --- | ---: | ---: |
| WT | 88.95 ± 8.17 | 9.65 ± 18.81 |
| TC | 83.52 ± 17.94 | 6.56 ± 9.41 |
| ET | 75.74 ± 26.05 | 31.82 ± 96.07 |

**结论：** 验证 Mean Dice 下降 0.71 个百分点，测试下降 0.45 个百分点；当前 full 方案未获整体收益，已停用。

主要失分在空 ET 患者：验证集假阳性病例 **2/5 → 5/5**，测试集 **2/4 → 4/4**。非空 ET 的平均 Dice 虽有改善（验证 **75.09% → 77.88%**，测试 **79.85% → 81.68%**），仍未抵消空患者失分。仅做了联合实验，无法确定各组件贡献。

训练后半程未刷新 best，不优先延长 epoch。旧方案由浅层细节回注替换，设计及状态见 [INNOVATIONS.md](INNOVATIONS.md)，历史权重使用见 [退役说明](../VM-UNet-main/docs/REGION_PRESENCE.md)。

结果：[验证汇总](../VM-UNet-main/results/brats2020_region_presence_full_1103_1004/validation/epoch_055/summary.json)、[验证逐患者指标](../VM-UNet-main/results/brats2020_region_presence_full_1103_1004/validation/epoch_055/per_patient.csv)、[测试汇总](../VM-UNet-main/results/brats2020_region_presence_full_1103_1004/test_1103_2155_01/evaluation_test/summary.json)、[测试逐患者指标](../VM-UNet-main/results/brats2020_region_presence_full_1103_1004/test_1103_2155_01/evaluation_test/per_patient.csv)、[验证曲线](../VM-UNet-main/results/figures/curves_region_presence_full_1103_1004/validation_regions.png)。
