# V-COCO 均值残差训练与最后四轮同步评估

本脚本用于 V-COCO 的24类人物—物体交互动作，使用 COCO RGB 图片及其同名 DVS 文件。不是普通 COCO 目标检测训练。

原项目 `main.py --epochs` 默认20轮，`launch_template.sh` 中的 V-COCO 命令没有覆盖这一设置。新脚本继续训练20轮，在第17、18、19、20轮结束后执行“保存该轮权重→完整test推理→AP评估→继续训练”。前16轮仍按原流程训练和保存checkpoint。DVS的时间均值、全局残差及交互分类结构保持不变。

```bash
# 先检查RGB/DVS文件配对、评估工具和标注；不启动训练
bash scripts/train_vcoco_dvs.sh --check-only

# 正式训练，默认4张GPU、每卡batch size=2
bash scripts/train_vcoco_dvs.sh

# 指定设备与输出目录
CUDA_VISIBLE_DEVICES=0,1 WORLD_SIZE=2 OUT_DIR=checkpoints/vcoco_mean_run1 \
  bash scripts/train_vcoco_dvs.sh
```

默认学习率 `1e-4`，AdamW，StepLR在第10轮结束后降低学习率；事件帧8个时间段、两个极性通道，Spikformer base。调整卡数或每卡batch会改变总batch，复现实验时应保持一致。

## 数据与评估依赖

- `DATA_ROOT` 默认项目内 `vcoco/`，里面有 `instances_vcoco_trainval.json`、`instances_vcoco_test.json`，以及 `mscoco2014/{train2014,val2014}` 或 `v_coco/images/{train2014,val2014}`。本机已有 `vcoco/v_coco` 数据链接。
- `DVS_ROOT` 默认 `~/Data/vcoco-dvs`，使用 `train/` 和 `test/` 中同名 `.npz`。`trainval` 对应事件目录 `train`。
- `PRETRAINED` 默认 `checkpoints/detr/detr-r50-vcoco.pth`。训练冻结DETR，只优化交互头及DVS分支。
- `VCOCO_EVAL_ROOT` 默认 `vcoco/v_coco`，需要 [s-gupta/v-coco](https://github.com/s-gupta/v-coco) 的Python3可运行评估器 `vsrl_eval.py`，以及 `data/vcoco/vcoco_test.json`、`data/instances_vcoco_all_2014.json`、`data/splits/vcoco_test.ids`。需要pycocotools；这些外部文件不会写入本仓库。

例如数据位于其他路径：

```bash
DVS_ROOT=/data/vcoco-dvs VCOCO_EVAL_ROOT=/data/v-coco \
  bash scripts/train_vcoco_dvs.sh --check-only
```

脚本沿用本机评估设置：IoU=0.5，排除 `point`，输出 Agent AP、Scenario 1 Role AP 和 Scenario 2 Role AP，单位均为百分数。此口径与包含 `point` 的全动作均值不同，比较历史结果时必须一致。若需要包含全部动作，可以在命令末尾追加 `--vcoco-exclude-actions`（不跟任何值）；日志与JSON会记录实际排除列表。

使用 `trainval → test` 是原模板的测试性能监测流程。选择新超参数或融合结构时改为 `--partitions train val`，避免依据test反复调参；不能用已参加训练的val作独立验证。

## 结果文件

默认使用带时间戳的新输出目录，或通过 `OUT_DIR` 指定：

```text
checkpoints/upt-dvs-r50-vcoco-mean_<时间戳>/
  train.log
  ckpt_<iteration>_17.pt
  ckpt_<iteration>_18.pt
  ckpt_<iteration>_19.pt
  ckpt_<iteration>_20.pt
  epoch_17_eval/{cache.pkl,inference.log,metrics.json}
  epoch_18_eval/{cache.pkl,inference.log,metrics.json}
  epoch_19_eval/{cache.pkl,inference.log,metrics.json}
  epoch_20_eval/{cache.pkl,inference.log,metrics.json}
  vcoco_metrics.csv
```

`vcoco_metrics.csv`在每轮评估完成后立即追加一行，保留epoch、checkpoint、评估split、排除动作、三项AP和耗时。`inference.log`包含逐动作AP。没有自动按test挑选或覆盖“最佳”权重，四轮结果与权重都保留。

只有rank 0推理，其他训练进程等待；推理使用未包装的模型，避免单卡调用DDP导致collective挂起。推理结束恢复训练模式和随机状态。任何rank 0评估异常都会通知其他rank并停止任务，不把失败评估当成完成。分布式等待超时设为180分钟，可以用 `--distributed-timeout-minutes`调整。

总轮数可通过 `EPOCHS` 或 `--epochs`调整，最后N轮自动按实际总轮数计算。例如 `EPOCHS=12`时评估第9–12轮；`EVAL_LAST_EPOCHS=0`禁用训练中评估。单独使用 `--resume`仍只加载模型权重；同时加上 `--resume-training`时会恢复AdamW状态、StepLR、梯度缩放器（若checkpoint包含）以及epoch和iteration计数，`--epochs`表示最终轮数。

## 从第12轮继续训练四轮

```bash
# 先检查续训checkpoint、数据和评估依赖
bash scripts/resume_vcoco_dvs_4epochs.sh --check-only

# 单卡、batch size=4，只执行第13–16轮，每轮结束后同步评估
bash scripts/resume_vcoco_dvs_4epochs.sh
```

脚本默认读取 `checkpoints/upt-dvs-vcoco-bs4-12epochs-20261007_163525/ckpt_14904_12.pt`，恢复后的学习率为 `1e-5`。命令中的 `--lr-head 1e-4`是原始调度器的基础学习率，加载checkpoint后会由保存的优化器状态覆盖。Pocket在每轮调度器step之前保存checkpoint，续训会补上该次step，避免第10轮等降学习率边界恢复错误。

输出使用新的 `checkpoints/upt-dvs-vcoco-bs4-epochs13-16_<时间戳>/`目录，保留原12轮实验。新权重编号为 `ckpt_16146_13.pt`到 `ckpt_19872_16.pt`，四轮AP汇总到新目录的 `vcoco_metrics.csv`，仍排除 `point`。可用环境变量 `RESUME`、`OUT_DIR`、`CUDA_VISIBLE_DEVICES`指定路径和设备；续训需保持原训练split、batch size及world size，以维持迭代计数一致。

旧checkpoint未保存Python/NumPy/Torch随机状态，所以恢复训练状态不会保证随机增强和dropout序列与不中断训练完全相同。

仅对一个已有模型推理和评估：

```bash
bash scripts/train_vcoco_dvs.sh --eval \
  --resume checkpoints/<运行目录>/ckpt_<iteration>_20.pt \
  --output-dir checkpoints/vcoco_epoch20_eval
```

只重新评估已生成的cache（不需要GPU）：

```bash
.venv/bin/python -m vcoco.evaluation \
  --vcoco-eval-root vcoco/v_coco --split test \
  --cache checkpoints/<运行目录>/epoch_20_eval/cache.pkl
```
