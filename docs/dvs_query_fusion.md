# Pair–动作query读取时空DVS证据

## 当前实现

Spikformer的四级spiking patch stem、六级attention/MLP及LIF时间状态仍然保留。编码器输出改为 `[B,T,256,H,W]`，不再在返回前平均空间token。每个时间段内部将空间网格压缩到最多4×6，然后按 `(时间,行,列)`展平；时间维度从输入到attention memory始终完整保留。

对图片b的第p个人物–物体pair，RGB交互头产生向量 `r_p`。第a个动作–角色类别使用可学习embedding `e_a`：

```
q[p,a] = LayerNorm(Linear(LayerNorm(r[p])) + e[a])
memory[t,y,x] = Linear(LayerNorm(DVS[t,y,x])) + Linear([time[t], y, x])
d[p,a] = LayerNorm(CrossAttention(q[p,a], memory, memory))
logits[p,a] = RGB_logits[p,a] + Linear(d[p,a])
```

时间坐标在各时间bin之间按0到1变化，空间坐标按-1到1变化。没有时间平均、全图共享DVS向量、动作softmax或由GT标签选择query；所有动作query同时参与候选分类。所有query共享事件编码器及attention参数，动作embedding按类别索引，pair条件来自各自的人物–物体特征。V-COCO的24个动作–角色query顺序与原分类器相同。

新分支位于RGB competitive layer之后，独立输出每个pair/动作的logit补充。分数头零初始化，其余query、投影和注意力参数正常初始化。第一个更新先训练分数头，输出权重变为非零后梯度即可传播至query与编码器，没有额外零初始化乘法门。

## 可选改进：相邻变化与关系推理前残差

基线默认保持不变，两个新增模块由独立参数启用，允许比较四种组合。

`--dvs-adjacent-changes`在每个空间位置构建相邻区间证据：

```
delta[t] = E[t+1] - E[t]
change_input[t] = concat(E[t], E[t+1], delta[t], abs(delta[t]))
change_token[t,y,x] = Linear(LayerNorm(change_input[t,y,x]))
                     + Position([(time[t]+time[t+1])/2, y, x])
                     + Interval([time[t], time[t+1]]) + Type[change]
memory = concat(original_tokens + Type[original], change_tokens)
```

差分在各时间bin分别做完空间池化后计算，保留有符号变化和变化幅值。所有原始token仍保留；T=1时仅有原始token。8段、4×6网格下，192个原始token加168个相邻变化token，共360个。类型embedding区分原始证据与变化证据；原始类型初值为零，变化类型正常初始化。时间为相对坐标，差分不是物理速度，也不能自动排除视频生成或相机运动引起的变化。

`--dvs-precomp-residual`用competitive layer之前的RGB pair构建pair-only query，读取上述同一份memory：

```
q_relation[p] = Linear(LayerNorm(r[p]))
event_relation[p] = CrossAttention(q_relation[p], memory, memory)
r_augmented[p] = r[p] + ZeroInitLinear(LayerNorm(event_relation[p]))
r_comp = CompetitiveLayer(r_augmented)
logits[p,a] = RGBClassifier(r_comp)[p,a] + ActionQuery(r_comp, memory)[p,a]
```

关系query每个pair读取一次，与每个pair/动作读取一次的动作query独立。仅残差末层零初始化，其余层正常初始化；不另加零初始化门。事件证据因此可以影响competitive layer中的pair竞争，而动作分支仍直接读取完整时序。memory每张图片准备一次，两个分支共享，均按pair分块读取。启用关系残差后RGB分类器的输入也受DVS影响，不能将其输出解释为独立RGB预测；真正的RGB对照需要绕过两个事件分支。

从已训练的基线query初始化时，动作分数头已经非零。仅启用关系残差，其零初值保持原预测；启用相邻token会改变动作attention读取的memory，初始预测可能随之变化。不能把这一变化当成训练增益。

## 连续性与边界

LIF按照时间顺序更新膜电位，保留历史依赖；显式时间位置让cross-attention能够区分事件发生的阶段。最终attention会按query聚合时空证据用于分类，这与在query读取前执行固定时间均值不同。保留时间维度与时间位置提供利用连续性的能力，不证明模型已学到真实动作轨迹，更不能保证AP提升。

测试验证了：memory包含所有时间bin；同一pair的不同动作及不同pair可以读取不同特征；反转已经编码的时间特征会改变query输出；去掉时间位置后cross-attention恢复时间块排列不变性；分块和不分块的结果及梯度一致；零初始化输出与RGB路径一致；空候选与旧checkpoint迁移得到正确处理。

DVS来源仍是静态图生成视频后得到的事件。时间坐标表示片段中的相对阶段，未额外传入真实持续时长。输入与编码器有能力表示顺序，但生成片段是否确实包含标签对应动作仍需核验。

当前query对整幅事件特征做软注意力。它没有按RGB框裁剪DVS，也未假定原图到生成视频的坐标只经过宽高缩放；人物–物体对应由RGB pair条件驱动。未来显式ROI需要核实生成几何映射及视频中主体运动，不能把软注意力解释为已经实现了精准轨迹绑定。

## 训练验证

继续使用24类多标签focal loss、检测先验、正样本数下限保护与排除point的官方AP评估。建议以相同训练预算的RGB模型和历史均值模型作对照，进行时间打乱、时间反转和时间坍缩实验，检查逐动作AP及多随机种子结果。

基线query已有训练结果：排除point后，第13轮Scenario 1为59.4560、Scenario 2为64.9078。新增两模块已完成CPU功能、接入位置、共享memory和梯度验证；尚未训练新增模块，也未测量其GPU显存与AP。基线在GitHub提交`f26853e`中保存。

建议从同一个基线checkpoint按相同优化器初始化、学习率、训练轮数和随机种子比较：基线续训、仅相邻变化、仅关系残差、两者同时启用。具体入口及迁移规则见[训练说明](vcoco_training.md)。
