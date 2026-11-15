# VM-UNet（BraTS2020 改进实验）源码

这里是源码目录，所有 Python 入口都在这里运行。协作规则、实验协议和研究记录都在仓库根目录：先读 [AGENTS.md](../AGENTS.md) 和 [STATE.md](../STATE.md)，代码导航见 [docs/codemap.md](../docs/codemap.md)。

| 入口 | 用途 |
|---|---|
| `run_brats.py` | 统一启动：train / resume / smoke / test / prepare（只能在 Linux 上运行；`--dry-run` 可在任意平台预览命令） |
| `collect_brats.py` | 把运行结果收集到 `../research/experiments/` |
| `train_brats.py` | 底层的训练、评价和预测入口，通常由 `run_brats.py` 调用 |
| `prepare_brats.py` | 生成清单、校验数据、建缓存 |
| `plot_brats.py` | 离线绘图（[说明](../docs/plotting.md)） |
| `check_local.py` | Windows 端 CPU 检查（[环境说明](../docs/env.md)） |

## 来源与许可

原论文、上游仓库和许可说明见 [docs/upstream.md](../docs/upstream.md)。数据使用须遵循各数据集自己的使用约定。
