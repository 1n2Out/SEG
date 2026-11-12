# 离线绘图

所有图都由 `plot_brats.py` 生成。它不导入 torch、不加载模型、不使用 GPU，A、B 两端都能运行。训练不会自动画图。样式配置在 `configs/plotting.json`（dpi、格式、字体、标签颜色），与训练配置分开，修改它不影响续跑。

各子命令的完整参数用 `python plot_brats.py <子命令> --help` 查看。图片默认输出到 `results/figures/`（不入库）。论文要用的图，在 IDEA 或 report 里记下生成命令即可，不提交图片。

## 病例图 `case`

```bash
python plot_brats.py case --case-id BraTS20_Training_001                                   # 三个平面，各取 WT 面积最大的切片
python plot_brats.py case --case-id BraTS20_Training_001 --plane axial --slices 60 67 75
python plot_brats.py case --case-id BraTS20_Training_001 --plane axial --region ET --radius 2   # ET 最大切片，加左右各 2 张相邻切片
python plot_brats.py case --case-id BraTS20_Training_001 --plane axial --slices 67 --crop 40 40 180 180
```

- **病例范围。** 默认只允许画训练集病例。画 val 病例要加 `--split val`；**test 病例只在 FINAL 阶段画。**
- **方向。** 按原始 XYZ 体素轴显示，不重新定向，不标注临床上的左右方向。
- **显示窗。** 每张切片取非零像素的 1%–99% 分位作为显示范围。
- **裁剪。** `--slices` 和 `--crop` 都要求只指定一个平面。`--crop` 只影响显示，不改变训练输入。
- **输出。** 每个位置一张图，另有 `case_slices.npz`、`slices.csv` 和 `case_metadata.json`。

## 数据分布 `stats`

```bash
python plot_brats.py stats                       # 训练集 259 例，只读标签和头信息
python plot_brats.py stats --limit 3 --output-dir results/figures/stats_preview   # 部分预览，元数据里会标明
```

- **输出。** 类别比例、各区域体积、阳性切片数、轴向分布，每张图都附带对应的 CSV。
- **区域重叠。** WT、TC、ET 是相互重叠的区域，不能画成堆叠比例图。
- **背景。** 全网格比例包含了脑外背景。

## 训练曲线 `curves`

```bash
python plot_brats.py curves --work-dirs results/<运行目录>
python plot_brats.py curves --work-dirs results/<运行A> results/<运行B> --names Baseline Improved --output-dir results/figures/comparison
```

- **数据来源。** 只读取 `history.jsonl`，旧日志还会读 `validation/epoch_*/summary.json`。也可以直接读 `research/experiments/EXP-xxx/history.jsonl`（要把该目录作为 `--work-dirs` 传入）。
- **输出。** 训练 loss 与验证 Mean Dice、各区域验证 Dice、loss 分量、学习率与累计训练时间，另有 `curve_data.csv`。
- **作图原则。** 只在真实做过验证的轮次画点，不平滑，也不补造缺失的轮次。累计训练时间不含验证时间。
