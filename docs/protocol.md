# 实验协议（BraTS2020）

baseline、改进和消融共用本协议。参数的权威值在 `VM-UNet-main/configs/brats2020.json`，本文只解释含义和规则。修改协议需要用户决定，并记入 [decisions](../research/decisions.md)。

## 数据与划分

- 共 369 例有标注病例，原始数据放在 `datasets/BraTS2020/`。官方无标签 validation 数据不使用。
- 按患者固定划分，**train/val/test = 259/55/55**：病例排序后用独立 RNG（seed=42）随机划分，没有按分级分层。清单为 `VM-UNet-main/data/brats2020/split_seed42.json`（v2 格式，含 SHA256），划分不可修改。
- val 用于选 best、选方案和调参；**test 只在方法确定后评价一次**（见 AGENTS §3）。
- `BraTS20_Training_355` 的标签文件名不规范，靠显式别名识别，原始文件没有重命名。其他病例缺模态或标签时直接报错。

## 预处理与输入

- 模态顺序固定为 **T1、T1ce、T2、FLAIR**，四通道输入。
- 每个病例的每个模态，用该模态非零体素的均值和标准差做 z-score，背景保持为零。
- 不重配准、不重采样、不改方向。
- 缓存位于 `data/brats2020/cache/`（约 30 GB）：`image.npy` 为 float16 的 `[Z,4,X,Y]`，`label.npy` 为 uint8 的 `[Z,X,Y]`，元信息里保存 affine 和体素间距。数组、元信息和清单必须一起迁移。
- **2D 轴向切片训练**，主干只读当前切片。方法分支可在配置中显式开启相邻切片（上下各 1 张，越界时复制边缘切片），训练与三维推理按同一方式堆叠。全部切片都参与，包括无肿瘤切片：训练集 40,145 张，val 和 test 各 8,525 张。
- 240×240 居中补零到 256×256，不缩放。输入尺寸若要改，必须仍能覆盖原图，并能被 32 整除。
- 训练增强：图像和标签同步做 90° 倍数旋转和翻转。

## 模型与训练

| 项目 | 设置 |
|---|---|
| 主干 | VM-UNet，编码深度 `[2,2,2,2]`，解码深度 `[2,2,2,1]`，drop path 0.2，参数量 27,429,172 |
| 初始化 | VMamba `vmamba_small_e238_ema.pth` 预训练，路径相对源码目录为 `../../package/`。只加载形状兼容的部分（原网络深度为 `[2,2,27,2]`）；RGB 输入卷积按通道取均值、复制 4 份后乘 3/4 |
| 输出与损失 | 4 类互斥 logits，原始标签 4 映射为 3。损失为 CE 与 WT/TC/ET soft Dice 等权相加，Dice 按切片计算后平均。方法可附加辅助损失（写在方法文档里），主损失不变 |
| 优化 | AdamW，lr 1e-4，weight decay 0.01，余弦调度（T_max 等于总轮数） |
| 训练量 | 最多 100 轮；单卡 batch 16，workers 4，AMP 开启；推理 batch 8 |
| 选 best 与早停 | 第 1 轮及每 5 轮在 val 上做三维评价，按 Mean Dice 最高选 best。早停 patience 为 10 次验证，min_delta 1e-4 |
| 随机种子 | 42，单次训练 |

**续跑：** 必须使用原配置、原目录和原卡数，不能中途修改总轮数。DDP 下 batch 指每卡 batch，学习率不随卡数自动缩放。

**热启动筛选（只用于决定是否值得完整训练）：** 从任务单指定的来源 best 加载全部已有参数（IDEA-001 的筛选用 EXP-001，叠加件用 EXP-006），挂上零初始化的新模块微调 10 轮。只有来源中没有的参数算新模块（叠加时用 `warm_start.new_parameters` 显式声明），lr 1e-4；其余参数 lr 1e-5，两组都按余弦降到 0；第 1、5、10 轮验证。筛选时必须同时跑"从同一来源出发、不加新模块、同样微调 10 轮"的对照组。**论文数字只来自完整协议的训练。**

## 评价

三维、按患者计算，再对患者等权平均。

| 区域 | 原始标签 | 指标 |
|---|---|---|
| WT | 1、2、4 | Dice、HD95 (mm) |
| TC | 1、4 | Dice、HD95 (mm) |
| ET | 4 | Dice、HD95 (mm) |

- Mean Dice 是三个区域的宏平均。另外输出 Precision、Recall、Specificity。报告中的 ± 是患者间的总体标准差（ddof=0）。
- **空区域规则：**
  - 预测和标注都为空：Dice、Precision、Recall 记 1，HD95 记 0。
  - 只有一方为空：Dice、Precision、Recall 记 0，HD95 记体积对角线长度（本数据约 373.13 mm）。
  - 因此空 ET 患者一旦被误判出 ET，会让 Mean Dice 下降约 0.6 个百分点。
- HD95：6 邻域表面，两个方向的距离合并后取 95 百分位。
- 不做 TTA，不做后处理。失败病例不删除。
- 这是本地协议，不等同于 BraTS 官方评测服务器的分数（[官方区域定义](https://www.med.upenn.edu/cbica/brats2020/tasks.html)）。
- 推理耗时只是运行记录，不能当作严格的延迟基准。不自动报告 FLOPs，因为可能漏算 selective scan。

## 运行产物（`VM-UNet-main/results/<运行目录>/`）

| 文件 | 内容 |
|---|---|
| `config.json`、`split.json` | 实际使用的配置和划分 |
| `run.json` | 环境信息、git 提交与工作区状态、源码 SHA256、预训练或热启动加载报告 |
| `history.jsonl` | 每轮的 loss 分量、验证 Dice 和早停状态 |
| `validation/epoch_XXX/` | `summary.json` 和逐患者 `per_patient.csv`（含各区域的目标体素数与预测体素数） |
| `best.pth`、`latest.pth` | checkpoint，含优化器、调度器、AMP 和随机数状态 |
| `test_*/evaluation_test/` | 仅 FINAL 任务产生 |
