# 不依赖RGB–DVS框坐标映射的主体时序读取

## 问题与本次范围

旧关系分支由RGB pair query一次读取整幅图的所有时空事件token。第20轮模型的3张测试图诊断中，不同pair的残差变化幅度仅约为残差整体幅度的0.10%–0.24%，提示接近共享全图补充的退化。这个抽样不是全测试集的身份对应评估。

生成视频中的主体会移动，RGB原图与DVS尺寸、裁剪方式也可能不同。因此本次用主体内容查询替换全图pair关系读取，不假设两种模态的框可以直接缩放对齐。Spikformer、事件分帧、相邻变化token和动作query的定义保持不变。

## 已核验的论文依据

| 论文 | 可引用的机制与位置 | 本项目借鉴与边界 |
|---|---|---|
| Locatello et al., **Object-Centric Learning with Slot Attention**, NeurIPS 2020. [正文](https://arxiv.org/html/2006.15055) | §2与Algorithm 1：attention先沿slot维softmax，令输入证据在slot之间竞争，再沿输入位置归一化并通过GRU更新slot。 | 按独立检测实例建立slot，事件token在实例及背景slot间竞争。原文没有证明跨RGB–生成DVS的身份对应。 |
| Kipf et al., **Conditional Object-Centric Learning from Video**, ICLR 2022（SAVi）. [正文](https://arxiv.org/html/2111.12594) | §2的Corrector/Predictor：逐帧提取证据并传播slot状态；Eq. (1)明确给出slot竞争与位置归一化。原方法使用光流目标和初始对象位置提示。 | 借鉴时序corrector与条件初始化，改用RGB实例特征初始化身份query，不使用RGB位置提示。这里的跨模态条件初始化是本项目设计；本实现也没有完整复制SAVi的predictor或光流监督。 |
| Zeng et al., **MOTR: End-to-End Multiple-Object Tracking with Transformer**, ECCV 2022. [正文](https://arxiv.org/html/2105.03247) | §3.2及§3.5：track query逐帧传播、更新以描述实例；训练依赖tracklet-aware label assignment。 | 支持“身份保存在query状态中，位置可以变化”的设计思路。本模型不使用其轨迹标签分配，因此称为软内容关联，不称为已验证的检测跟踪器。 |
| Manasyan et al., **Temporally Consistent Object-Centric Learning by Contrasting Slots**, CVPR 2025（SlotContrast）. [官方页面](https://openaccess.thecvf.com/content/CVPR2025/html/Manasyan_Temporally_Consistent_Object-Centric_Learning_by_Contrasting_Slots_CVPR_2025_paper.html) · [正文](https://arxiv.org/html/2412.14295) | §3的slot-slot contrastive与batch-video contrastive目标，以及Appendix C/D：仅靠递归处理不足以保证长期一致性；仅同视频对比可能利用初始化捷径，需扩大负例集合，并以特征重建约束内容。 | 这篇近年论文支持身份一致性的重要性，也限制本次结论。本次未加入其对比/重建目标，不能宣称复现SlotContrast或仅凭GRU保证身份稳定。后续若出现slot交换或坍缩，应据此设计内容约束，而非强迫所有pair输出不同。 |

这些论文证明相关机制在各自任务和监督条件下有效，不直接证明本项目生成事件数据上的HOI AP会提升。

## 模型定义与实现对应

代码：[entity_query.py](../dvs_data/entity_query.py)。每张图中的每个RGB检测实例只建立一个slot；同一个人物出现在多个HOI pair中时共享同一条事件主体序列，避免把同一人物当成多个互斥槽。另设一个可学习背景slot容纳非候选实例证据。

检测实例特征来自RGB cooperative layer，记为u_i，角色r_i为人/非人；没有GT动作、GT框或预生成的GT轨迹输入：

```
a_i = EntityProjection(LayerNorm(u_i)) + RoleEmbedding(r_i)
s_i,0 = a_i
q_i,t = Normalize(Wq LayerNorm(a_i + s_i,t-1))
k_j,t = Normalize(Wk LayerNorm(m_j,t))
```

m_j,t来自每个原始时间bin的事件内容投影，不混入RGB坐标，也不混入DVS绝对位置编码。动作分支仍使用其原有带位置及相邻变化token的memory，两者复用同一次事件内容投影。

对每个head，在该时间bin内计算：

```
alpha[i,j,t] = softmax_over_entities(tau * dot(q_i,t, k_j,t))
beta[i,j,t] = alpha[i,j,t] / sum_over_locations(alpha[i,j,t])
e_i,t = sum_j beta[i,j,t] * Value(m_j,t)
s_i,t = GRU(e_i,t, s_i,t-1)
```

每个时间bin执行两次corrector更新，再用更新后的slot读出事件证据。tau按head学习并限制在[1,10]。查询状态保留RGB内容anchor以减少身份漂移；只有事件readout进入pair memory，不把RGB anchor或GRU状态直接拼入事件值来制造pair差异。

空间归一化在log域实现，即`softmax_over_locations(log(alpha))`，数学上等于`alpha / sum_locations(alpha)`；避免小责任值下用分母截断破坏空间权重和为1的性质。人/物角色embedding使用较小的初始尺度，让初始化身份主要由各实例特征提供。

在DVS自身的归一化网格中，用最终空间权重计算软中心和空间方差，将其投影后加入事件readout。这些位置只作为DVS内部证据值，既不作为RGB框定位条件，也不参与内容匹配。因此主体移动造成的软位置变化可保留到后续序列。它们不是经过监督验证的实体框或物理速度。

将人物h和物体o的事件readout组合成逐时间段pair状态：

```
z[p,t] = PairEventProjection(LayerNorm(concat(e[h,t], e[o,t],
                                             e[h,t]-e[o,t], e[h,t]*e[o,t])))
         + RelativeTimeEmbedding(t)
residual[p] = ZeroInitProjection(TemporalAttention(rgb_pair[p], z[p,:]))
```

保留所有T个bin，最终由pair query进行有时间位置的可学习读取，不进行固定时间均值。残差加入competitive layer之前；动作query继续读取原来的完整全图时序及相邻变化memory。

## 能论证的性质

1. **不需要跨模态坐标同一性。** 关联分数只依赖RGB实例query和DVS内容key，式中没有RGB框坐标到DVS像素的映射。DVS网格位置只用于描述读出证据在DVS内部的位置。
2. **实例排列等变性。** 实例顺序重排，且同步重排其角色及pair索引时，slot竞争、共享GRU和背景slot使实例输出按相同顺序重排；不依赖人工pair编号或固定槽编号。
3. **位置可随证据移动。** 对同一时间bin的内容token做置换，空间attention会相应置换，主体内容readout保持一致，而软位置随token所在网格变化。这个性质支持运动主体关联，但仅针对内容不变的受控置换，不证明真实遮挡、形变或跨模态外观变化下的准确性。
4. **共享主体不会因HOI pair重复而被排斥。** 竞争发生在唯一检测实例间；多个pair引用同一条人物序列。竞争机制不要求所有pair彼此正交，也不要求人物和接触物体占据完全不重叠的区域。
5. **数值边界。** 归一化q、k的余弦值在[-1,1]，tau在[1,10]，单head关联logit在[-10,10]。它防止单纯权重范数增长造成无界logit，不保证attention不会集中。

主体竞争可以在相同query或难区分事件下仍产生相同读出。背景slot也只是可学习容纳项，不保证正确识别不可见主体。这些限制应在论述中保留。

RGB中被裁剪掉的实例与原DVS全景中的额外实例可能由背景slot容纳，但这不是已验证的跨增强对齐。两个外观/事件特征无法区分的主体，也不能仅靠内容查询保证身份正确；遮挡、生成变形、相似人物以及缺失目标是必须单独检查的情况。

## 可用于文章的方法表述

> 为适应生成视频中主体位置变化及RGB与事件流的空间不对齐，我们构建由RGB实例特征条件初始化的事件主体槽。借鉴Slot Attention的实例竞争与视频主体槽的递归更新，每个槽逐时间段读取事件内容，形成共享身份索引下的事件序列。模型在事件自身坐标系中描述软位置变化，并据此组合人物–物体时序关系，向HOI关系推理提供事件残差。该设计不要求RGB检测框与事件像素直接对齐。

在完成验证前不写“精准跟踪”“保证动态解耦”“已解决身份对应”或“提升检测性能”。从RGB实例到DVS主体的真实对应仍需要实验支持。

## 验证要求

已有CPU测试覆盖受控主体移动、实例及pair重排、共享主体、q/k尺度放大、跨时间梯度、时序传播、RGB–DVS不同尺寸、空候选和零初始化。受控测试验证程序性质，不能代替真实定位和HOI评测。

正式实验应从头训练，固定事件编码器配置、相邻变化token和训练预算，对比global与entity-slots。除HOI AP外，抽查预测主体attention的语义正确性、跨时间slot交换、背景读取与共享人物一致性。注意：仅比较不同query是否输出不同向量，不足以证明对应正确。若使用训练轨迹验证/监督，需区分轨迹是否来自GT初始化，测试阶段不能用GT轨迹提供主体身份。

一张真实trainval图已通过CPU训练前向与反向：6个独立检测实例组成15个候选pair，保留`[8,24,128]`原始事件内容memory；存在匹配的正HOI，loss与新增残差梯度均有限。这个检查只证明模型训练链路可用，不是训练结果。原20轮global checkpoint在指定旧模式后仍可严格加载，新模式会明确拒绝该结构。

文章中需要分开报告三类证据：数学/受控测试证明结构性质；真实attention与主体对应检查验证语义；相同预算、多种子的HOI实验验证收益。后两类尚未完成，不能用前一类替代。

## 运行与旧模型兼容

`--dvs-precomp-residual --dvs-relation-mode entity-slots`启用本次方案；关系模式默认entity-slots。旧20轮模型要显式用`--dvs-relation-mode global`评估或续训。两种关系结构权重不兼容，错误模式会报错，不会静默忽略旧参数。从头训练不传`--resume`或任何初始化参数。

没有新增sh文件；训练命令直接输出给用户。
