# V-COCO 时空 DVS query 训练与评估

当前模型已移除 DVS 时间均值及全图共享特征残差。Spikformer保留 `[B,T,C,H,W]` 的时空特征；每个候选人物–物体pair与每个动作–角色类别共同构造query，读取带时间和空间位置编码的事件token，再输出动作分数残差。V-COCO有24个动作–角色类别，最终logits仍是 `[pair数量,24]`，沿用多标签focal loss及检测先验。

## 训练

```bash
# 单卡、batch size=4、12轮，第9–12轮同步评估
CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 BATCH_SIZE=4 EPOCHS=12 \
  bash scripts/train_vcoco_dvs.sh

# 先检查数据和评估工具，不启动训练
CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 BATCH_SIZE=4 EPOCHS=12 \
  bash scripts/train_vcoco_dvs.sh --check-only
```

脚本环境变量默认仍为4卡、每卡batch=2、20轮，第17–20轮评估。默认AdamW学习率 `1e-4`，StepLR在第10轮结束后降至 `1e-5`。单卡设置如上。DETR冻结，交互头、事件编码器和query模块一起训练。

Query默认128维、4个注意力头，每个时间段保留最多4×6个空间token；8个时间段得到192个memory token。这里仅在每个时间段内部压缩空间，不做时间平均。按16个pair分块计算cross-attention，降低临时矩阵的峰值；训练仍需要保存各块的反向传播状态，显存节省主要来自空间token数量的限制。

## 相邻变化token与关系推理前残差实验

普通训练入口保持基线结构；`--dvs-adjacent-changes`和`--dvs-precomp-residual`分别启用两个改进。相邻变化追加T−1个区间的空间token，仍保留全部T段原始token；pair关系残差位于competitive layer之前，动作query仍位于之后。两分支共享一次构建的时序memory。

新的单卡实验入口默认从基线第13轮权重初始化，batch=4，学习率`1e-5`，新训练4轮，每轮同步评估并排除point：

```bash
# 两模块同时启用；先检查资源和checkpoint结构
bash scripts/train_vcoco_temporal_relation.sh --check-only
bash scripts/train_vcoco_temporal_relation.sh

# 相同预算的四组实验；分别写入新目录
ADJACENT_CHANGES=0 PRECOMP_RESIDUAL=0 bash scripts/train_vcoco_temporal_relation.sh
ADJACENT_CHANGES=1 PRECOMP_RESIDUAL=0 bash scripts/train_vcoco_temporal_relation.sh
ADJACENT_CHANGES=0 PRECOMP_RESIDUAL=1 bash scripts/train_vcoco_temporal_relation.sh
ADJACENT_CHANGES=1 PRECOMP_RESIDUAL=1 bash scripts/train_vcoco_temporal_relation.sh
```

`BASELINE`可指定其他基线query checkpoint，`EPOCHS`、`LR_HEAD`等覆盖实验预算。默认基线是`checkpoints/upt-dvs-vcoco-query-bs4-epochs13-16_20261008_163427/ckpt_16146_13.pt`，Scenario 2为64.9078、Scenario 1为59.4560。checkpoint不会上传GitHub，在其他设备上需自行提供。

上述四组均只加载模型权重，重新建立AdamW和StepLR，从新实验epoch 1开始；不是恢复到旧实验第14轮。启用新模块时脚本显式添加`--init-query-baseline`，只允许新增模块的参数缺失，原RGB层、Spikformer及原动作query权重完整加载。基线对照也重建优化器，保证四组训练预算可比。原学习率`1e-5`在默认4轮期间保持不变，仍沿用第10轮降学习率的调度。

只启用关系残差时，零初始化末层保持基线初始输出；启用相邻变化时已有动作query会读取额外token，初始预测可能改变，需要单独记录训练前性能及训练后的结果。

评估或恢复**新结构自身**的checkpoint时，必须传入训练时相同的两个分支开关，并去掉`--init-query-baseline`：

```bash
CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 BATCH_SIZE=4 \
  bash scripts/train_vcoco_dvs.sh --eval \
  --resume checkpoints/<新实验>/ckpt_<iteration>_04.pt \
  --dvs-adjacent-changes --dvs-precomp-residual

RESUME=checkpoints/<新实验>/ckpt_<iteration>_04.pt FINAL_EPOCH=8 \
  bash scripts/resume_vcoco_dvs_4epochs.sh \
  --dvs-adjacent-changes --dvs-precomp-residual
```

分支开关与保存权重不匹配会报错。旧基线直接评估及续训不添加这两个开关即可，历史脚本仍运行原基线。基线迁移到新结构不允许`--resume-training`，因为新增参数改变了优化器参数集合。

## 旧均值模型的权重

旧均值模型的结构、优化器参数集合和新query模型不同，不能使用 `--resume-training`继续旧实验，也不能用新代码直接评估旧模型。

可以显式加载旧模型的共享RGB交互层及Spikformer权重，作为一个新实验的初始化：

```bash
CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 BATCH_SIZE=4 EPOCHS=12 \
  bash scripts/train_vcoco_dvs.sh \
  --resume checkpoints/upt-dvs-vcoco-bs4-epochs13-16_20261007_234950/ckpt_19872_16.pt \
  --init-legacy-dvs
```

旧的 `dvs_norm`、`dvs_adapter`权重被丢弃，query模块重新初始化；优化器、调度器、轮数从新实验开始。共享层缺失或维度不匹配会报错，不能用任意 `strict=False`静默忽略。Query分数头零初始化，所以新模型初始预测等于所加载的RGB路径；它不等于旧RGB+DVS模型，因为旧的全局残差已移除。

若正在运行旧模型训练，已启动进程使用其已加载的模型定义；重启时需明确旧实验与新query实验的结构差异。旧结构可在GitHub提交 `0e89b80`中复现，原checkpoint及历史结果保留用于比较。

## Query模型断点续训

```bash
# 示例：已完成新query模型的12轮，继续到16轮
RESUME=checkpoints/<query运行目录>/ckpt_14904_12.pt FINAL_EPOCH=16 \
  bash scripts/resume_vcoco_dvs_4epochs.sh
```

该脚本单卡batch=4，恢复query模型的AdamW、StepLR、epoch和iteration，评估最终四轮。`FINAL_EPOCH`是最终累计轮数，保持训练split、batch和world size一致。Pocket在调度器step之前保存checkpoint，恢复时补上该次step。旧checkpoint未保存完整随机状态，续训不会严格复现不中断运行时的随机增强与dropout序列。

## 数据与评估

- `DATA_ROOT`默认项目的 `vcoco/`，RGB目录为 `mscoco2014/{train2014,val2014}`或 `v_coco/images/{train2014,val2014}`。
- `DVS_ROOT`默认 `~/Data/vcoco-dvs`，trainval使用 `train/`，test使用 `test/`中的同名npz。
- `PRETRAINED`默认 `checkpoints/detr/detr-r50-vcoco.pth`。
- `VCOCO_EVAL_ROOT`默认 `vcoco/v_coco`，需要s-gupta/v-coco的Python3评估器、官方动作标注、COCO实例标注和split ids，需要pycocotools。

IoU=0.5，默认从AP计算及均值分母中排除 `point/points`；输出Agent AP、Scenario 1和Scenario 2 Role AP，单位为百分数。原始全动作口径与本口径不同，比较时保持一致。

训练沿用 `trainval → test`。结构和超参数选择应另建未参加训练的验证协议，例如 `--partitions train val`；不能使用已经参加trainval训练的val来判断初始化模型的独立验证效果。

## 输出

新目录默认 `checkpoints/upt-dvs-r50-vcoco-query_<时间戳>/`，包含train.log、每轮checkpoint、最后四轮 `epoch_XX_eval/{cache.pkl,inference.log,metrics.json}`以及 `vcoco_metrics.csv`。

只有rank 0推理，其他rank等待。推理使用未包装的模型，结束后恢复训练模式与随机状态；评估失败会终止任务。默认分布式等待超时180分钟。

仅评估已有query模型：

```bash
bash scripts/train_vcoco_dvs.sh --eval \
  --resume checkpoints/<query运行目录>/ckpt_<iteration>_12.pt
```

已有预测cache可在CPU重新计算AP：

```bash
.venv/bin/python -m vcoco.evaluation \
  --vcoco-eval-root vcoco/v_coco --split test \
  --cache checkpoints/<运行目录>/epoch_12_eval/cache.pkl
```
