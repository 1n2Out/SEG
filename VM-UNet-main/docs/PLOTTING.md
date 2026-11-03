# BraTS 离线绘图使用说明

三类绘图共用 `plot_brats.py`，独立于训练入口，不创建模型、不导入 torch、不使用 GPU，不自动修改数据或生成正式缓存。Windows CPU / Linux 均可使用。训练不会自动绘图；新增的 CE、Dice 分量日志仅复用原有 loss 计算，没有额外模型前向或验证。

## 运行目录与依赖

以下命令均在源码目录运行：

```powershell
conda activate vmunet-local
cd D:\ZJGSU\ThesisWork\Task2\VM-UNet-main\VM-UNet-main
```

现有本地环境已安装绘图依赖，无需重复安装。其他机器只需按需安装独立依赖：

```bash
python -m pip install -r requirements-plotting.txt
```

默认数据根目录、清单和绘图配置按脚本位置解析，不依赖启动目录；用户显式传入的相对路径按当前终端目录解析。所有示例均无需执行训练或模型测试。

## 1. 病例图：训练前可用

```bash
# 默认读取原始 NIfTI，显示四模态和 FLAIR+真实标签
# 在三个平面分别选取 WT 面积最大的切片，无该区域时选择中心切片
python plot_brats.py case --case-id BraTS20_Training_001

# 手动指定轴向切片，索引从 0 开始
python plot_brats.py case --case-id BraTS20_Training_001 --plane axial --slices 60 67 75

# 选择 ET 最大切片，并导出左右各 2 张相邻切片
python plot_brats.py case --case-id BraTS20_Training_001 --plane axial --region ET --radius 2

# 局部放大：显示像素坐标的 X0 Y0 X1 Y1，右/上边界不包含
python plot_brats.py case --case-id BraTS20_Training_001 --plane axial --slices 67 --crop 40 40 180 180 --output-dir results/figures/case_zoom

# 读取已有单病例检查缓存，无需重新读取原始 MRI
python plot_brats.py case --case-id BraTS20_Training_001 --cache-dir results/local_checks/sample_cache
```

默认仅选择训练集病例；如果手动指定内部 val/test 的病例，需同时指定 `--split val` / `--split test`，防止误用数据划分。省略病例 ID 时使用所选集合的第一例。官方无标签预测清单不适用于这两个有标注数据绘图命令。

默认输出根目录为 `results/figures/case/`，每名患者独立一个子目录：

- `axial_067.png/pdf/svg` 等：每个位置独立一张图。
- `case_slices.npz`：绘图使用的归一化模态切片和原始标签数组。
- `slices.csv`：每张展示切片在裁剪范围内的标签像素数。
- `case_metadata.json`：划分校验和、数据来源、空间信息、切片选择、裁剪和绘图样式。

病例图下方统一显示英文全称及缩写，避免只写数字标签；图例放在影像之外，避免遮挡：

| 原始标签 | 图例 | 中文含义 |
|---|---|---|
| 1 | Necrotic and non-enhancing tumor core (NCR/NET) | 坏死与非增强肿瘤核心 |
| 2 | Peritumoral edema (ED) | 肿瘤周围水肿 |
| 4 | Enhancing tumor (ET) | 增强肿瘤 |

颜色保持由 `label_colors` 配置；CSV 中的原始标签字段名保留，元数据额外保存标签名称映射。

显示使用原始 XYZ 体素轴：axial 固定 Z，coronal 固定 Y，sagittal 固定 X，转置后 `origin=lower`。不重定向、不重采样，不标注临床放射学左右方向；非标准方向数据应先确认方向再解释切面。图像按当前生产预处理做每病例每模态非零 z-score，并使用 float16 量化；显示窗为每张切片非零像素的 1%~99% 分位，零背景显示黑色。`--crop` 是展示裁剪，不改变训练输入。`--slices` 和 `--crop` 均要求指定单一平面。

## 2. 数据分布图：训练前生成，数据不变即可复用

```bash
# 完整统计固定训练集的 259 名患者；只读取标签和 MRI 头信息
python plot_brats.py stats

# 先用 3 例检查效果：图片标题和元数据明确标记为部分预览
python plot_brats.py stats --limit 3 --output-dir results/figures/stats_preview

# 使用完整生产缓存；不会创建或补齐缓存
python plot_brats.py stats --cache-dir data/brats2020/cache

# 非默认数据位置
python plot_brats.py stats --data-root /srv/data/BraTS2020
```

默认输出 `results/figures/stats/train/`（其他集合使用对应 split 目录）：

| 文件名 | 内容 |
|---|---|
| `class_balance.*` | 全图网格类别比例，以及仅肿瘤标签的比例 |
| `region_volumes.*` | 逐患者 WT/TC/ET 真实区域体积箱线图，单位 mL |
| `positive_slices.*` | 逐患者每个区域阳性轴向切片数量箱线图 |
| `axial_presence.*` | 各存储 Z 索引上存在对应区域的患者比例 |
| `class_balance.csv` | 原始标签计数和比例 |
| `patient_statistics.csv` | 逐患者类别计数、区域体积、阳性切片数量 |
| `slice_statistics.csv` | 每患者、每轴向切片的区域体素数，包含阴性切片 |
| `axial_presence.csv` | 轴向分布图的实际数据 |
| `stats_metadata.json` | 划分、完整/部分统计标记、患者列表、源文件大小/时间签名及样式 |

这里会重新读取真实标签，不依赖旧审计文件中的计数。WT=1/2/4、TC=1/4、ET=4 是重叠区域，不能作为互斥类别画堆叠比例。全网格比例包括脑外背景，不能称为“占脑组织比例”。体积用 affine 行列式并换算空间单位计算。仅当所有患者轴向深度一致时绘制按原始索引汇总的分布；相同索引并不代表解剖位置完全配准。

优先用训练集分析数据难点。测试集不用于决定模块、分组阈值或超参数。`--limit` 仅供预览，不是正式全集结果。

## 3. 训练曲线：训练记录日志，手动生成图片

```bash
# 按当前启动器的默认工作目录绘制；需要已产生 history.jsonl
python plot_brats.py curves --work-dirs results/brats2020

# 多次实验叠加比较，自定义图例与输出目录
python plot_brats.py curves --work-dirs results/baseline results/improved --names Baseline Improved --output-dir results/figures/comparison
```

训练中也可读取当前日志快照，但绘图需要手动运行；希望完全避免共享资源竞争时，在训练后或把日志拷回本机后绘制。曲线命令不读取影像、不加载 checkpoint、不执行推理。复制日志到本机时建议保留 `history.jsonl` 和 `run.json`；旧日志若没有 `val_dice`，还需保留 `validation/epoch_*/summary.json` 才能补齐分区域曲线。

默认输出 `results/figures/curves/`：

| 文件名 | 内容 |
|---|---|
| `training_curves.*` | 总训练 loss、验证 Mean Dice；星形标记最高验证分数 |
| `validation_regions.*` | WT/TC/ET 验证 Dice 曲线 |
| `loss_components.*` | CE loss、WT/TC/ET 平均 soft Dice loss |
| `schedule_and_time.*` | 学习率，以及累计训练时间对应的验证 Mean Dice |
| `curve_data.csv` | 多次实验合并后的实际绘图数据 |
| `curves_metadata.json` | 工作目录、日志快照 SHA256、最高分轮次、早停状态和缺失字段说明 |

只在真实验证轮次绘制点，不补造未验证 epoch 的分数，不做平滑。存在早停时用竖向虚线标记。累计训练时间为 `train_seconds` 求和，不包含验证、checkpoint 写入或恢复间隔，不等于完整实验墙钟时间。

新增训练记录为 `train_ce_loss`、`train_dice_loss`、`val_dice`、`best_epoch`、`best_val_mean_dice`。损失公式、验证频率、best 选择及恢复配置比较规则不变；DDP 对各 loss 分量按实际采样数汇总。旧日志没有分量时显示缺失，不反推或补零。正在写入但尚未完成的最后一行会警告并跳过，其他损坏行报错。

目前没有正式训练日志，因此真实收敛曲线需要服务器首次训练后生成；测试夹具中的示例数值仅用于程序验证。

## 共用配置与导出

绘图样式位于 `configs/plotting.json`，独立于 `configs/brats2020.json`，修改绘图样式不会影响断点续跑：

- `dpi`：默认 200。
- `formats`：默认同时输出 PNG、PDF、SVG。
- `font_family`：默认 DejaVu Sans；图中文字使用英文以减少跨平台字体问题。
- `overlay_alpha`：标签叠加透明度，默认 0.45。
- `label_colors`：原始标签 1/2/4 的统一颜色。

可以只对一次命令覆盖格式、分辨率和输出目录：

```bash
python plot_brats.py case --case-id BraTS20_Training_001 --formats png --dpi 300 --output-dir results/figures/case_300dpi
python plot_brats.py stats --formats png pdf --dpi 300
python plot_brats.py curves --work-dirs results/brats2020 --plot-config configs/plotting.json
```

同一输出目录内同名图表会覆盖；比较不同切片选择、不同统计范围或不同实验组合时，用不同 `--output-dir` 保留各版本。建议不要把图输出到尚未启动训练的工作目录，因为训练入口要求新实验目录为空。

分别运行 `python plot_brats.py case --help`、`stats --help`、`curves --help` 查看完整参数。生成内容默认放在已忽略的 `results/figures/` 下。
