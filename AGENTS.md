# AGENTS.md：协作规则（所有 AI 共用）

本项目以 VM-UNet 为 baseline，在 BraTS2020 上做 2D 四模态分割改进，服务于硕士论文第二项工作。第二数据集待创新点确定后再选。

**研究宗旨。** 时间紧，取舍按以下顺序：创新"故事" > 实验指标 > 工作量（图表、对比、消融） > 底层原创性。优先选动机清晰、成本可控、能验证的方案。叙事必须以实际实验为依据；不为小幅涨点投入长周期调参。

## 1. 角色

| 机器 | 角色 | 当前工具 | 负责 | 不做 |
|---|---|---|---|---|
| Windows 本地 | **A：构建** | Claude Code | 方案设计、代码与配置、CPU 测试、方法文档、任务单 | 训练、CUDA 运行、读取测试集结果 |
| Linux 服务器（B 机经 VSCode Remote 连接） | **B：执行与分析** | Codex | GPU 检查、训练、结果收集、实验报告、运行问题修复 | 改研究方向；改模型、损失或协议 |

角色按运行环境判定：在 Windows 上就是 A，在 Linux 上就是 B。工具可以更换，规则只认角色。

## 2. 每次开始

1. `git pull`。B 端若有训练在跑，先确认这次拉取不会改到正在使用的代码（见 [runbook](docs/runbook.md)）。
2. 读 [STATE.md](STATE.md)，了解当前在做什么、下一步是什么。
3. 只读当前任务需要的文档（见 §4 索引），不要通读全部。

## 3. 硬性规则

1. **测试集封存。** 选方案、调参数、做动机分析，只能用 train 和 val。`run_brats.py test`（需加 `--confirm-test`）和 `research/experiments/*/final/` 只在用户批准的 FINAL 任务中使用。
2. **结果可追溯。** 正式训练启动时，`VM-UNet-main` 下的源码和配置不能有未提交的改动；文档改动不受限。启动器会检查，并把 commit 记入 `run.json`。续跑和测试时，启动器还会核对代码自该提交以来是否有变化。会进入论文的实验不得使用 `--allow-dirty` 或 `--allow-code-change`。
3. **数字只来自文件。** 指标只引用 `collect_brats.py` 生成的 `summary.json`、`index.md` 和逐患者 CSV，不手抄、不估算。没运行的不写成已完成，工程检查通过也不写成效果提升。
4. **协议只有一个来源。** 实验协议以 [docs/protocol.md](docs/protocol.md) 和 `VM-UNet-main/configs/brats2020.json` 为准。baseline、改进和消融共用同一协议；修改协议要由用户决定，并记入 decisions。
5. **决定权在用户。** `research/decisions.md` 只记录用户确认过的决定。IDEA 状态改为"采纳"或"放弃"、修改协议、执行 FINAL 测试，都要用户确认。
6. **不进仓库的东西。** 数据、缓存、权重、`results/`、`logs/` 都不提交，也不自动下载权重。
7. **命名。** 新文件名只用小写 ASCII 和连字符；文档内容用中文。

## 4. 文档体系

| 位置 | 内容 | 谁写 | 写法 |
|---|---|---|---|
| `AGENTS.md`、`CLAUDE.md` | 协作规则 | 用户确认后由 A 修改 | 稳定，少改 |
| [WORKFLOW.md](WORKFLOW.md) | 写给用户的协作流程说明 | A，随规则同步更新 | 不重复规则，只讲用法 |
| [STATE.md](STATE.md) | 当前状态：进行中的任务、训练、待决事项、下一步 | 交接时由当前一端更新 | **覆盖写**，不超过 40 行 |
| [docs/protocol.md](docs/protocol.md) | 数据、训练、评价协议 | A | 改动需用户决定 |
| [docs/codemap.md](docs/codemap.md) | 代码导航、扩展约定、修改禁区 | A | 随代码更新 |
| [docs/env.md](docs/env.md) | 两端环境 | A 写 A 段，B 写 B 段 | 只写现行环境 |
| [docs/runbook.md](docs/runbook.md) | B 端标准操作 | B 维护，A 可提议修改 | 只写现行命令 |
| [docs/plotting.md](docs/plotting.md) | 离线绘图 | A | |
| `docs/methods/<方法>.md` | 方法结构、配置、工程检查 | A | 方法退役时删除 |
| [research/ideas/](research/ideas/index.md) | 创新点：动机、依据、设计、状态 | A 起草；状态由用户决定 | 每个 IDEA 一个文件 |
| [research/experiments/](research/experiments/index.md) | 实验结果与报告 | B：脚本生成结果，AI 写 `report.md` | 每个 EXP 一个目录 |
| [research/decisions.md](research/decisions.md) | 已确认的决定 | 用户确认后，任一端都可写 | 追加写，每条不超过 3 行 |
| `handoff/TASK-xxx.md` | A→B 任务单，B 在文末填回执 | A 写任务，B 写回执 | 关闭后移入 `handoff/archive/` |

## 5. 交接协议

编号统一为三位数递增：IDEA-001、TASK-001、EXP-001。链路是 **IDEA → TASK → EXP**，互相用链接引用。

**A → B**
1. 改代码、写配置和方法文档，CPU 测试通过。
2. 按 [handoff/TEMPLATE.md](handoff/TEMPLATE.md) 写任务单，必须写明：要检验什么、运行清单、通过标准、每种结果对应的决策。
3. 更新 STATE，提交（消息以 `[A]` 开头），推送。

**B 执行**
1. 拉取代码，然后按任务单执行：先检查，再训练。GPU 用 `--gpus` 指定，不要为了选卡修改配置文件。
2. 训练完成后用 `collect_brats.py` 收集结果，按模板写 `report.md`，在任务单文末填回执，更新 STATE。
3. 提交（消息以 `[B]` 开头），推送。

**B 端可以改的代码：** 只限运行问题，比如路径、依赖、CUDA 兼容、明显的 bug。改完单独提交，消息用 `[B] fix: ...`，并在回执里写明。涉及模型、损失、数据处理或协议的改动，B 不动手，写进回执交回 A。

**用户裁决：** 用户做出决定后写入 decisions.md，再由 A 开始下一轮。

## 6. 防臃肿

- **一个事实只写一处，其他地方链接过去。** 数字写在 summary.json，协议写在 protocol.md，现状写在 STATE.md。
- **"当前""目前""截至"这类现状描述只出现在 STATE.md。** 其他文档只写稳定的事实和用法。
- **只写结论和依据，不写过程。** 讨论经过、试错记录、验收流水账不进文档；验收结果写在任务单回执里，随任务单一起归档。
- **被取代的内容直接删除，** git 里有历史。不保留"旧版""历史"段落。
- **篇幅上限：** STATE 40 行，report 40 行，TASK 60 行，IDEA 60 行，docs 每篇 150 行。超出时先删过时内容，再写新内容。
- **分析脚本放进 `VM-UNet-main/analysis/` 并提交，** 文档里只放结论和产物路径。不允许"一次性脚本未入库"。

## 7. 常用检查

A 端（Windows CPU）运行：

```bash
cd VM-UNet-main
/d/Dev_Env/Anaconda/envs/vmunet-local/python.exe -B -m pytest tests -q -p no:cacheprovider --basetemp .test_tmp -k "not loss_is_finite_and_differentiable"
```

B 端的检查、训练、收集与提交见 [docs/runbook.md](docs/runbook.md)。
