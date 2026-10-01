# RGB 与生成 DVS 的融合：文献依据、定义与实验方案

核查日期：2026-10-01。主要参考 CVPR 2025、ICCV 2025、CVPR 2026 正式论文，已读取官方摘要和八篇论文全文。本文只提出研究与实验方案，没有修改模型。文献的实验结果、本文的数学推导、针对本项目的假设分别标明；不同任务的数值不作直接比较。

## 1. 建议采用的主线

保留目前在 V-COCO 上取得约 0.5 提升的残差融合模型，作为完整基线 B。新增一个可单独关闭的、针对人物—物体对的动态分类分支，先采用小型时序卷积和 logits 残差；验证动态贡献之后，再增加 RGB 条件的单向时间注意力。

这项建议旨在减少每一步的实验变量，使失败原因可以定位。它不意味着任何新增模块必然提高测试 AP。用户报告的“0.5”目前尚未核实具体 Scenario、计分单位和随机种子重复次数，暂按约 0.5 AP 百分点理解。

## 2. 近期论文提供了哪些证据

| 文献 | 已核对的方法与证据 | 对本项目的启发与适用边界 |
| --- | --- | --- |
| [Seeing Motion Through Polarity for Event-based Action Recognition，CVPR 2026，POKER](https://openaccess.thecvf.com/content/CVPR2026/html/Cao_Seeing_Motion_Through_Polarity_for_Event-based_Action_Recognition_CVPR_2026_paper.html) | §3.2、式(2)–(3)：分别计算正、负极性的相邻时间差分，以及同一时刻正负极性的空间差异，再融合这些运动线索。表5（PDF第7页）：THU-EACT-50-CHL 上 Concat 61.51%、Add 63.35%、Router 63.72%。 | 优先明确动态输入的构成，再选择融合。Router 比 Add 高0.37个百分点，不能据此推断复杂融合在任何任务上明显更优。论文还有文本运动推理与对齐训练，其完整收益不能归因于差分或路由一个模块。 |
| [SMV-EAR: Bring Spatiotemporal Multi-View Representation Learning into Efficient Event-Based Action Recognition，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Fan_SMV-EAR_Bring_Spatiotemporal_Multi-View_Representation_Learning_into_Efficient_Event-Based_Action_CVPR_2026_paper.html) | §3.3、式(5)–(7)：T-H、T-W 事件视图采用独立分支，融合分支 logits；动态权重从分类头之前的池化语义特征获得。表6（PDF第7页）：logit averaging 64.3±0.24%，class-wise weighting 65.2±0.17%，sample-wise weighting 66.7±0.29%。早期拼接55.9±0.56%，但该设置分支数和参数量也不同，不能视为严格等容量比较。 | 支持“先获得各分支可辨识的表示，再做后期融合”。其两路都是事件时空投影视图，既非静态 RGB 与 DVS，也非 HOI；迁移需要本项目消融。 |
| [Beyond Duality: A Hybrid Framework of Leveraging Shared and Private Features for RGB-Event Object Detection，CVPR 2026，SPFD](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Beyond_Duality_A_Hybrid_Framework_of_Leveraging_Shared_and_Private_CVPR_2026_paper.html) | §3：在频域按跨模态相干性分离共享与私有分支，通过 TriAdapt、TriInject 利用私有信息。表2：DSEC-Det 基线mAP47.3，共享/私有分离47.9，再加TriAdapt49.0，完整49.2。 | 支持保留事件特有表征，而非强制两种模态全部对齐。其空间频率私有特征不等于纯时间动态；真实同步传感器上的检测结果不能证明生成 DVS 的动作真实性。 |
| [InfoBridge: Balanced Multimodal Integration through Conditional Dependency Modeling，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Li_InfoBridge_Balanced_Multimodal_Integration_through_Conditional_Dependency_Modeling_ICCV_2025_paper.html) | §3、表2：以条件互信息目标进行上下文相关的跨模态学习，提出 protective margin 以缓解过度对齐。CREMA-D 上完整模型62.85，去掉 protective margin 为58.52。 | 支持避免无条件地把所有 DVS 特征拉向 RGB 表征。论文的互信息目标和保护项不是 HOI AP 不下降定理，也不保证完全剥离静态外观。 |
| [Adaptive Unimodal Regulation for Balanced Multimodal Information Acquisition，CVPR 2025，InfoReg](https://openaccess.thecvf.com/content/CVPR2025/html/Huang_Adaptive_Unimodal_Regulation_for_Balanced_Multimodal_Information_Acquisition_CVPR_2025_paper.html) | §3：研究早期学习阶段的信息充分模态压制另一模态。表5：普通联合训练下 Gated/SUM/FiLM/Concat 为65.32/64.38/66.67/66.61；加入InfoReg为69.18/70.12/70.23/71.90。 | 融合结构与优化过程需要分别诊断。支持监控动态分支的梯度、独立表现和收敛，而不是只看联合 AP。本文建议的冻结基线训练是工程选择，并非照搬 InfoReg。 |
| [Information-Theoretic Decomposition for Multimodal Interaction Learning，CVPR 2026，DMIL](https://openaccess.thecvf.com/content/CVPR2026/html/Yang_Information-Theoretic_Decomposition_for_Multimodal_Interaction_Learning_CVPR_2026_paper.html) | §3.1、式(1)：按冗余、各模态独有、协同信息分解多模态信息；§3.2讨论样本级交互构成。这里的“interaction”指信息交互，不特指人物—物体交互。 | 解释为什么重复 RGB 语义未必带来收益，事件独有线索与条件协同更值得测试。不能把事件私有、时间敏感和动作因果信息三个概念等同。 |
| [Exploiting Frequency Dynamics for Enhanced Multimodal Event-based Action Recognition，ICCV 2025，EMP](https://openaccess.thecvf.com/content/ICCV2025/html/Cao_Exploiting_Frequency_Dynamics_for_Enhanced_Multimodal_Event-based_Action_Recognition_ICCV_2025_paper.html) | §3.2：从同一个事件流获得堆叠帧和重建帧，以频域增强及高频引导token选择降低冗余干扰。 | 支持事件表示中有不同类型的线索。但空间高频轮廓不能直接定义为时间运动；这两种输入均来自事件流，也不是原图与生成视频的独立观测。 |
| [LLaFEA: Frame-Event Complementary Fusion for Fine-Grained Spatiotemporal Understanding in LMMs，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_LLaFEA_Frame-Event_Complementary_Fusion_for_Fine-Grained_Spatiotemporal_Understanding_in_LMMs_ICCV_2025_paper.html) | §3.1：分层跨注意力与自注意力进行帧/事件时空融合；§5.3讨论融合策略及训练流程。 | 为后续单向条件注意力提供结构参照。该任务是大模型时空理解，不能直接套用完整模型或其超参数到24类HOI预测。 |

## 3. 本项目需要采用的定义

**静态特征。** 对第 i 个人物—物体对，RGB 特征 s_i 包含人物/物体外观、位置关系以及场景上下文。其作用是提供原始静态 HOI 标注所对应的语义参照。

**动态候选特征。** 保留事件时间轴与空间区域对应关系后，对人物框、物体框、交互联合框分别获得时间序列 z^h_{i,t}、z^o_{i,t}、z^u_{i,t}，构造：

\[
m_{i,t}=\operatorname{concat}\left(z^h_{i,t}-z^h_{i,t-1},\ z^o_{i,t}-z^o_{i,t-1},\ z^u_{i,t}-z^u_{i,t-1}\right).
\]

这里采用现有 Spikformer 的冻结中间空间特征，先开放其空间输出、再做区域池化；避免立即替换整个编码器。按时间变化构造候选描述符，是受 POKER 时间差分启发的本项目改写，论文原式是在极性帧上差分，两者并不等同。

特征差分会包含编码器的状态响应。特别是 LIF 神经元即便面对重复输入，也可能产生启动阶段的时序变化。因此 m 只能称为动态候选描述符，必须通过时间坍缩对照验证，不能凭公式宣称已经剥离静态信息。

也不能直接对有符号差分取均值来代表动作：\(\frac{1}{T-1}\sum_{t=2}^{T}(z_t-z_{t-1})=(z_T-z_1)/(T-1)\)，中间过程会被抵消，循环动作可能接近零。后续采用带非线性的时间卷积，保留邻接变化；持续运动在某些表征下也可能表现为近恒定，因此原始事件基线仍有保留价值。

**动态互补收益。** 动态分支应在控制 RGB、分类头容量、训练预算和物体先验之后，带来稳定的增量收益；针对声称利用顺序的模块，还应显示有序输入与破坏顺序输入之间的差异。依赖时间顺序是可检验性质，但不等于真实动作的因果证明。

**共享、独有与协同信息。** 借用部分信息分解的记号，对选定表示 S、E 及标签 Y：

\[
I(Y;S,E)=R+U_S+U_E+\mathrm{Syn},\qquad
I(Y;E\mid S)=U_E+\mathrm{Syn}.
\]

这说明“RGB 已有表示之外的事件贡献”可能同时包含独有和协同信息，不能一律叫纯动态信息。上述分解遵循所选 PID 定义；这些量不是在本项目中已被精确估计的可观测数值。

## 4. 能证明什么，不能证明什么

**4.1 事件本身不是纯运动观测。** 事件触发近似满足：

\[
\log L(x,t)-\log L(x,t_{\mathrm{prev}})\approx p\,c_{\mathrm{th}}.
\]

在亮度恒定运动近似下：

\[
\partial_t\log L\approx-\mathbf u^\top\nabla\log L+\eta.
\]

其中 u 是图像运动，η 表示照明变化及近似误差；在生成视频中，还要考虑闪烁与生成伪影。事件响应受到空间轮廓梯度与运动共同影响，正负极性本身也不唯一决定运动方向。因而“事件特征与RGB正交”“高频特征”“正负极性分离”均不能单独证明纯动作动态已被提取。

**4.2 生成事件的独立信息边界。以下是本文推导，不是上述论文的 V-COCO 结论。** 设原图 I、标签 Y、生成随机量 ξ，且生成器只接收 I 和与 Y 条件独立的 ξ：

\[
V=G(I,\xi),\quad E=H(V),\quad Y\perp\xi\mid I.
\]

则 Y→I→E 构成马尔可夫链：

\[
p(E\mid I,Y)=p(E\mid I),\quad I(Y;E\mid I)=0.
\]

所以生成 DVS 没有增加相对于完整原图的独立标签观测。不过实际 RGB 模型只保留 S=f(I)，这是有损表示；此时可以有 I(Y;E|S)>0。生成事件可能把原图中模型难以利用的线索展开成更容易学习的表示，因此约0.5的收益与上述结论完全相容。

如果生成视频还使用了动作提示 P、真实动作标签或其他输入，上述“仅给原图”的条件就不成立。需明确 P 的来源，并分析 Y→(I,P)→E；来自动作标签的收益不能解释为新观测到的真实动作证据。

**4.3 残差融合与分类器之间的关系。** 如果残差确实直接加在一个线性分类器之前，且中间没有非线性层：

\[
\ell=W(s+A d)+b=Ws+b+WA d.
\]

因此这种特征残差已经等价于一种受参数化约束的 logits 补充。单纯改名为“双分类器”不构成新的动态机制。本仓库当前残差是在 competitive layer 之前注入，之后还有非线性变换，因此不能把整条现有模型严格化简为上述公式。

**4.4 新分支可以包含基线解。** 对冻结完整基线 B，设新分类器为：

\[
\ell_i=\ell_{B,i}+\delta_\theta(m_i).
\]

若 δ 的输出层权重和偏置可设为零，则函数类包含 B。对相同样本分布、相同损失定义，且正则项在δ=0时为零：

\[
\inf_\theta\mathbb E\left[\mathcal L(Y,\ell_B+\delta_\theta)+\lambda\|\delta_\theta\|^2\right]
\leq\mathbb E\mathcal L(Y,\ell_B).
\]

这是函数类包含关系；它不保证有限数据的训练能够找到更优解，也不保证 surrogate loss 的下降带来测试 AP 提升。验证时保留δ=0作为候选，可以保留验证集上的基线结果，同样不能保证未知测试集不下降。

## 5. 预先确定的分类器与训练方案

**第一阶段采用的结构：**

\[
d_i=\operatorname{Mean}_t\!\left[\operatorname{TCN}_{\mathrm{small}}(m_{i,1:T-1})\right],\qquad
\delta_i=W_D\operatorname{LN}(d_i)+b_D,\qquad
\ell_i=\ell_{B,i}+\delta_i.
\]

TCN 使用轻量一维时间卷积，保留时间邻接关系；首轮固定为 Conv1d(kernel size=3, hidden dimension=128)→GELU→时间池化，避免线性均值差分只剩端点变化。输出为24个动作logits，W_D、b_D零初始化。每类动作由其对应的输出行学习不同贡献，不先把24类动作硬划为RGB类或DVS类。

完整 B 指已经取得最好结果的 RGB+DVS 残差模型，不仅是 RGB 模型。冻结其参数、BatchNorm运行统计和Dropout行为；冻结已有事件编码器，从其中间空间输出读取特征，只训练新TCN及输出头。W_D为零时，编码器/TCN最初不会从该输出收到非零任务梯度，输出层开始更新后梯度即可传入；不能同时把额外乘法门和输出层都初始化为零，否则会阻断学习。

继续使用现有 UPT 的人/物体检测先验及交互 focal loss，保持多标签 sigmoid；不用24类 softmax。初始不增加“DVS 单独预测全部 HOI 标签”的辅助损失，避免迫使动态分支重新承担静态语义识别。可以对δ加入小幅平方惩罚；正则系数只在验证集选定，不作为理论保证。

**第二阶段的升级只增加单向时间注意力：**

\[
a_{i,t}=\operatorname{softmax}_t\left((W_qs_i)^\top W_km_{i,t}/\sqrt d+b_t\right),\qquad
d_i=\sum_ta_{i,t}W_vm_{i,t}.
\]

原始RGB交互特征作为冻结查询，动态候选特征作为key/value，b_t保留时间位置。保持相同24类残差输出头和完整基线路径。该结构允许动态证据的选择依赖静态交互语义；它是受近期条件融合/跨注意力研究启发的本项目方案，没有论文直接证明它在生成DVS的V-COCO上有效。

**区域与轨迹的要求。** 先使用人物框、物体框和联合框三种固定ROI。坐标必须经过“增强后RGB→原图→生成视频→DVS→特征图”映射，不能只按宽高比例猜测。已有 geom_matrix 只记录RGB增强，还需要生成视频的缩放、裁剪/偏移及其到DVS的变换。评测时区域来自RGB预测框；若后续使用视频轨迹，应由同一可部署的无标签检测/跟踪流程产生，不使用测试GT框或真实动作标签来选择区域。先不同时引入轨迹模块，避免首轮增加新的误差源。

## 6. 固定的实验顺序与通过标准

| 编号 | 实验 | 回答的问题 |
| --- | --- | --- |
| B0 | 原RGB模型复现 | 现有约0.5提升相对于什么基线？ |
| B1 | 现有最好残差模型复现 | 提升是否跨种子稳定？以后所有升级与B1比较。 |
| C0 | 冻结B1，同容量的新残差头输入原始全局事件特征 | 增益是否仅来自额外参数或额外训练？ |
| C1 | 冻结B1，同一新残差头输入全局时间差分特征 | 时间变化描述符是否有效？ |
| C2 | C1改为三种对齐ROI的差分序列 | 区域对应是否有效？全局/局部描述符统一输出维度控制容量。 |
| C3 | C2仅增加RGB条件的单向时间注意力 | 确定动态表征之后，条件融合是否进一步有效？ |

任何阶段未通过，都先诊断该阶段，不同时追加门控、频域分解、多层cross-attention、额外对比损失和编码器微调。冻结/解冻的收益本身也是变量，首轮通过后另做独立微调实验。

提前固定一个V-COCO主指标，Scenario 1/2同时报告；相同数据划分、检测器、proposal、先验、训练预算和checkpoint选择规则。至少3个配对随机种子，报告均值、标准差、逐动作AP和训练代价。pilot阶段使用train训练、val选择；如果B1已使用trainval训练，官方val已经参与训练，不能再把它当作独立选择集。此时需要重新建立未参与训练的源图划分，或先用train-only模型进行方案选择。最终方案固定后，再按正式协议训练并评测test。

建议采用的阶段通过标准：配对种子的主指标平均增量为正，且源图级配对bootstrap置信区间支持稳定收益；预先选定的动态动作集合确有贡献，整体收益不是少数静态类别偶然波动。置信区间跨零时把结论标为未确定，不直接宣称有效。bootstrap须对源图重采样后重算官方AP；不能把“单张图片AP”平均当作官方结果。同源生成视频的多个版本保持在同一数据划分。

## 7. 证明动态贡献需要的对照

1. **关闭新增分支。** 恢复B1，检查数值和评价结果，排除基线被意外改变。
2. **等容量原始事件特征对照。** C0与C1比较，区分增加模型容量和使用动态描述符。
3. **时间坍缩。** 把每段事件帧替换为其时间平均，保持每个像素总量的平均表示，再从头计算中间特征；对LIF启动响应尤其必要。另训练一个相同容量的时间坍缩对照，降低仅在测试时改输入造成分布偏移的混淆。
4. **时间打乱。** 保留各时间段的空间内容而破坏顺序，同时测试固定模型敏感性与重训对照。二者回答不同问题；性能下降不自动等于动作因果证据。
5. **时间反转。** 只在顺序确实改变动作含义的类别或人工核验样例上解释结果。很多循环动作、V-COCO静态标签或生成片段在反转后标签仍可成立，不能要求所有类别都下降。
6. **背景对照。** 同容量动态头只使用交互ROI之外事件；如果背景也能获得同样收益，不能将全部收益归因于人物交互。
7. **同类错配与生成版本。** 保持RGB与检测框不变，使用同类其他样本的事件或同一原图另一生成版本。错配是分布干预，需结合区域质量解释；不能单凭掉分证明严格解耦。核对生成视频是否真的呈现目标动作、主体是否一致、是否主要是镜头运动。

首轮保持原事件分帧设置不变，以减少变量。现有count_clip=1会丢失事件计数强度，后续可单独测试保留计数的log压缩；但不能与融合结构一起更改后把收益全归给新融合。

## 8. 当前最值得执行的决策

将“基线B1 + 小型局部时间差分分类分支”设为主方案，将“RGB条件单向时间注意力”设为通过前一阶段后的升级方案。用条件互补、零残差基线包含关系和明确消融定义来推进，不预先承诺纯动态分离或测试AP单调提升。

所有文献的数值来自其自身任务；本项目中尚未运行上述新方案。近期论文提供设计依据，本项目的配对实验才提供V-COCO上的有效性证据。
