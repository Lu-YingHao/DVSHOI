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

## 连续性与边界

LIF按照时间顺序更新膜电位，保留历史依赖；显式时间位置让cross-attention能够区分事件发生的阶段。最终attention会按query聚合时空证据用于分类，这与在query读取前执行固定时间均值不同。保留时间维度与时间位置提供利用连续性的能力，不证明模型已学到真实动作轨迹，更不能保证AP提升。

测试验证了：memory包含所有时间bin；同一pair的不同动作及不同pair可以读取不同特征；反转已经编码的时间特征会改变query输出；去掉时间位置后cross-attention恢复时间块排列不变性；分块和不分块的结果及梯度一致；零初始化输出与RGB路径一致；空候选与旧checkpoint迁移得到正确处理。

DVS来源仍是静态图生成视频后得到的事件。时间坐标表示片段中的相对阶段，未额外传入真实持续时长。输入与编码器有能力表示顺序，但生成片段是否确实包含标签对应动作仍需核验。

当前query对整幅事件特征做软注意力。它没有按RGB框裁剪DVS，也未假定原图到生成视频的坐标只经过宽高缩放；人物–物体对应由RGB pair条件驱动。未来显式ROI需要核实生成几何映射及视频中主体运动，不能把软注意力解释为已经实现了精准轨迹绑定。

## 训练验证

继续使用24类多标签focal loss、检测先验、正样本数下限保护与排除point的官方AP评估。建议以相同训练预算的RGB模型和历史均值模型作对照，进行时间打乱、时间反转和时间坍缩实验，检查逐动作AP及多随机种子结果。

目前已完成CPU功能和梯度验证；尚未实际训练或测量新query模型的GPU显存与AP。
