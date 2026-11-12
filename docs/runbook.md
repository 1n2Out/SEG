# B 端操作手册（Linux 服务器）

B 端按任务单执行实验时，照本手册操作。全部命令都在源码目录下运行。

```bash
conda activate vm_unet
cd /home/hzh/hwh/Seg/VM-UNet-main      # 路径见 env.md
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

## 1. 开工

```bash
git status --short          # 代码和配置应无改动：运行问题的修复按 [B] fix 提交，其余丢弃；文档改动可以保留
ls logs/*.pid 2>/dev/null   # 查看有没有训练在跑：ps -p "$(cat logs/<名称>.pid)" -o pid,etime,args
git pull
nvidia-smi                  # 挑选空闲的卡
```

- 训练进行中可以 `git pull`：已启动的进程在启动时就加载了代码，不受影响。
- 续跑或测试某个运行时，启动器会把当前代码和该运行记录的提交做比较（`tests/`、`analysis/` 和 `*.md` 除外）。若有差异会拒绝执行，处理办法见 §6。

## 2. 检查（任务单列出的步骤）

```bash
python -m pytest -q tests/<任务指定的测试>.py
CUDA_VISIBLE_DEVICES=<卡> python -m pytest -q tests/<方法>_cuda.py              # 必须 passed，skipped 不算通过
BRATS_TEST_DISTRIBUTED=1 python -m pytest -q tests/<方法>_distributed.py        # 任务要求时才跑
python run_brats.py smoke --config configs/<配置>.json --gpus <卡>
```

## 3. 训练

```bash
python run_brats.py train --config configs/<配置>.json --gpus <卡> --dry-run       # 先预览命令和输出目录
python run_brats.py train --config configs/<配置>.json --gpus <卡> --background
tail -f logs/<运行名>.log
```

- **选卡。** 一个运行只用一张卡，不同运行放在不同卡上，否则 `train_seconds` 无法比较。多卡请写成 `--gpus 0,1`。不要为了选卡修改配置文件。
- **代码不能有未提交改动。** 如果启动器提示 "Uncommitted code changes" 而拒绝启动，先提交或丢弃这些代码改动。文档和报告的改动不影响启动。不要对正式实验使用 `--allow-dirty`。
- **登记。** 把运行目录名写进任务单的回执里。

## 4. 中断与续跑

```bash
kill -TERM "$(cat logs/<运行名>.pid)"      # 先用 ps 确认这是要停的那个进程
python run_brats.py resume --config configs/<配置>.json --run-dir results/<运行目录> --gpus <原卡数对应的卡> --background
```

续跑必须用原配置、原目录和原卡数。

## 5. 收集结果与写报告

```bash
python collect_brats.py --run-dir results/<运行目录> --exp EXP-<编号> --title "<短标题>"   # 热启动筛选加 --epoch 10
```

- **生成内容。** 命令会生成 `research/experiments/EXP-xxx/`，并重新生成 `index.md`。重新收集需要加 `--force`。
- **写报告。** 按 [EXP 报告模板](../research/experiments/TEMPLATE.md) 写 `report.md`，不超过 40 行。数字只引用生成的文件。
- **配对比较。** 比较两个实验时，用两者的 `val_per_patient.csv` 做逐患者配对差分析。分析脚本放进 `analysis/` 并提交。

## 6. 提交与故障处理

```bash
git add ../research ../handoff ../STATE.md
git commit -m "[B] EXP-xxx <标题>：<一句话结论>"
git push
```

| 情况 | 处理 |
|---|---|
| GPU 被占用 | 换一张卡，用 `--gpus` 指定 |
| 续跑或测试被拒，提示代码有变化 | 优先 `git checkout <运行的提交>` 后再执行，完成后切回分支。若变化与本次运行无关，可以加 `--allow-code-change`，但要在回执里写明理由 |
| 缺少缓存 | 运行 `python run_brats.py prepare`（需要原始数据） |
| 运行问题，比如路径、依赖、CUDA 兼容、明显 bug | 可以修，单独提交，消息用 `[B] fix: ...`，并在回执里写明 |
| 模型、损失、数据处理或协议本身有问题 | 不改，写进回执交回 A |

## 7. FINAL：测试集评价（仅在用户批准后）

```bash
python run_brats.py test --config configs/<配置>.json --run-dir results/<运行目录> --gpus <卡> --confirm-test
python collect_brats.py --run-dir results/<运行目录> --exp EXP-<编号> --title "<标题>" --include-test --force
```

测试结果只会写到 `EXP-xxx/final/`，不进入 index。每个方法只评价一次。
