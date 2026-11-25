# 硕士论文研究蓝图

**中文题目**　基于层级 RQ-VAE 的临床语义 ID 生成与跨疾病迁移研究——面向大语言模型驱动的出院用药推荐

**English title**　Hierarchical RQ-VAE-Based Clinical Semantic ID Generation for Cross-Disease Transfer in LLM-Driven Discharge Medication Recommendation

**关键词**　Residual Quantization（RQ-VAE）、Medication Semantic ID、Cross-Disease Cohort Transfer、Constrained Generation、Chinese EHR

**研究阶段**　Interim study design，2026-11-25。下文以「已报道」「拟采用」「假设」「待核验」标记证据状态。

## 1. Executive Summary

本研究预测一次住院记录中实际开具的**出院药物名称集合**。核心问题是当目标主诊断队列只提供少量有标签患者时，给药理或语义相近的候选药物共享输出前缀，是否能让从其他队列学到的信息更容易迁移。方案将药品元数据向量离散化为 RQ-VAE 药物 ID，由同一个中文 LLM 读取病历并在候选目录内生成药物集合。病历仍以文本作为输入，Semantic ID 的对象始终是药物。使用等长、同树形的置换 ID 对照，隔离前缀语义与解码结构本身的作用。

| 决策 | 理由与判据 |
| --- | --- |
| 核心贡献聚焦于药物输出编码和跨队列评测 | 残差量化、Semantic ID、医疗本体引导和跨领域生成推荐均有先例。将贡献定位为中文出院标签上的受控比较及适用条件。 |
| 先做数据审计与多标签基线 | 工作区未发现原始数据，且当前下载及研究授权尚未证实。651 个标签使多标签分类具备计算优势，也为生成式方法设置性能参照。 |
| 主模型采用小型开源权重 LLM、冻结的药物编码与单一迁移流程 | 两阶段设计允许只替换输出编码。模型规模、数据清洗、Adapter 与搜索预算都可对齐。 |
| 以患者隔离的目标队列低样本适配为主 | 对同院诊断队列迁移做有限且可检验的推断。队列规模或独立性不足时，执行第 13 节的范围调整。 |

## 2. Research Background and Motivation

住院与出院的用药集合具有不同的生成机制。病历中的既往用药、院内短期治疗、出院后继续用药和停药都可能出现在同一份文本中。CDrugRed 将其中的出院带药抽象成标准药名集合，为中文的历史处方复现提供了可比较的任务。历史处方同时反映既往决策、当地药品目录和诊疗实践。离线结果可用于研究模型如何利用临床文本及标签结构，临床合理性、药物安全和患者结局需要另行验证。 [D1] [D2]

普通分类头为每种药分配一个输出，学习不同药物间的联系主要依赖共享病历表示。生成式建模可再在输出层显式共享前缀。例如，模型学习某个共同前缀时，同组多个药物的样本都会更新相关参数，随后由后续 token 区分具体药物。只有当这些共享统计规律在目标队列仍有效时，压缩结构才可能带来迁移收益。若共同前缀主要反映名称词形或来源队列的频繁用药，前缀共享也可能导致错误聚类和负迁移。 [R3] [R4]

## 3. Problem Definition

一个样本为 $(p_i,v_i,x_i,Y_i,g_i)$。$p_i$ 是患者分组标识，$v_i$ 是就诊标识，$x_i$ 是于定义的预测截点可获取的脱敏病历字段，$g_i$ 是冻结规则得到的主诊断队列，$Y_i\subseteq\mathcal M$ 是历史出院带药集合。$\mathcal M$ 为组织方给定的固定候选目录，论文报告大小为 651。模型估计 $\hat Y_i=f_\theta(x_i)\subseteq\mathcal M$。同一患者的多次就诊仍是多个样本，划分及统计推断以患者为组。$p_i$、$v_i$ 只用于分组和导出，在所有方法的模型输入及提示中均排除。 [D1] [D2]

定义两个评测视角。

1. **同分布参照**。从有标签部分建立患者隔离的混合队列训练、验证和测试。它用于回答编码是否损害整体表现，也提供对传统分类的参照。官方 A/B 榜结果只在使用相应官方切分及评分时直接可比。
2. **跨疾病队列迁移**。目标为一个预先确定的主诊断队列。先在其余符合条件的队列训练，再以 $k$ 名目标患者适配，在独立的目标患者上测试。$k$ 计入用于适配和目标域调参的全部标注患者。核心现象是**同院病例构成改变时的样本效率**。主诊断队列允许共病，严格敏感性实验进一步排除其他可识别的目标疾病证据。

任务输出是药名集合及其研究评估所需的统计量。剂量、给药频率、疗程和真实临床决策不在评估范围内。将出院诊断用作输入时，推断时点设为相应诊断可取得的出院记录阶段。若无法重建真实录入时序，主要结论限定为基于经过字段审计的病历做回顾性预测。 [D2]

## 4. Literature Review and Research Gap

下表以任务组织者、原始论文、作者实现和正式会议记录为主。研究空白是对已核验近邻的归纳，论文写作时可用第 16 节的检索式更新。

| 研究线 | 代表研究与证据 | 对本课题的约束 |
| --- | --- | --- |
| 结构化 EHR 用药 | GAMENet 结合纵向记忆和 DDI graph。SafeDrug 用药物分子表示与 DDI 控制。COGNet 建模历史药物的复制。它们在结构化 EHR 上验证各自机制。 [M1] [M2] [M3] | 历史用药复制、频率先验及多标签判别均是合理参照。英文结构化数据、ATC 粒度与中文出院药名的分数不能直接横比。 |
| 中文出院带药 | CDrugRed v1 与 CHIP2025 组织方提供定义和分数。DeepDrug 用药理类别、条件先验及多模型组合，MediRec 用推理链蒸馏及强化学习。 [D1] [D2] [D3] [D4] | 药理信息已用于该任务。必须在一致的骨干、输入和训练数据上识别**输出前缀组织**的增益。 |
| Discrete latent representation | VQ-VAE 将连续向量映射成 codeword，RQ-VAE 逐级逼近残差，最早应用之一是图像生成。 [R1] [R2] | 残差深度描述近似过程。临床语义与前缀粒度必须独立验证。 |
| Semantic ID 和层级 | TIGER 用 item tuples 做生成式检索。HiD-VAE 预印本研究多级标签监督及碰撞。GenCDR 研究领域适配的 SID 及词典树约束。 [R3] [R4] [R5] | 层级输出、约束解码和跨域 SID 都有研究先例。需要保持词典树结构的置换对照，检验收益是否来自真实分组。 |
| 医疗概念量化 | UDC、MedRQ 和 SCQMed 将量化后的医学概念用于下游用药预测。UDC 在疾病频率分组中评估跨频率的表示共享。MedTok 融合文本与关系表示，SMI 用本体路径表示医学概念并评估药物预测。 [M4] [M5] [M7] [M8] [M9] | 药物推荐中的医疗概念量化已有多个先例。本研究干预输出药物的分组，保持病历输入表示和 LLM 一致。 |
| 医疗代码生成与疾病迁移 | SafeRx-Agent 生成 ATC 层级药物编码，CARing 从 EHR 生成量化后的诊断 ID。KnowRare 评估跨疾病迁移的 ICU 预测。 [M6] [M10] [M11] | 本研究在中文叙述病历上生成固定目录中的具体药名路径，并对照测试低样本目标主诊断队列的处方集合。 |

**拟研究空白。** 这些近邻提供了有效的编码和生成先例。对 CDrugRed，仍需测试一个具体命题：在患者隔离、目标队列标注受限、解码树和 LLM 相同的条件下，基于药物元数据构造的输出前缀，比保持同样树结构的随机分组更有用吗？其效果是否随目标药物在源域的出现频数或前缀支持而变化？若结果与置换一致，应将观测到的收益归于多 token 参数共享、约束或训练设置这一层面的候选解释，并用对应对照缩小范围。 [D2] [M4] [R3] [R5]

## 5. Dataset Analysis

### 5.1 已报道数据与本地证据

CDrugRed v1 报告大连医科大学附属第二医院 2013–2023 年内分泌科成人的脱敏住院数据。组织方按患者划分，任务侧提供标准候选名称。下表来自 v1 表 2 和评测总结，均为已发表的汇总数字。本地已有任务速读、Poster 模板和评分指南，未找到候选词表或病历记录，故无法复算分布、确认交付文件的实际格式或检查重叠。 [D1] [D2] [L1]

| 官方划分 | 患者数 | 就诊数 | 本划分中不同的药物 | 就诊平均带药数 | 就诊平均诊断数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 训练 | 1,910 | 3,602 | 586 | 6.08 | 6.82 |
| A 榜 / 开发 | 320 | 570 | 335 | 6.06 | 6.85 |
| B 榜 / 测试 | 960 | 1,722 | 459 | 6.13 | 6.79 |
| 全量 | 3,190 | 5,894 | 651 | 6.09 | 6.82 |

651 与 586 的差为 65，表示完整目录中 65 个条目在报道的官方训练标签中未出现。各划分标签集合会重叠，A、B 各自的未见标签数量和候选库外比例仍待核验。论文图 3 报告常见出院诊断包括 2 型糖尿病、高血压、脂肪肝及并发症，常见药物涉及他汀类、肠溶阿司匹林及部分降糖药。这些汇总不足以给出互斥主诊断队列的人数、每类与每药计数、源/目标标签重叠或共同前缀支持。v1 说明排除了部分复杂病史、严重合并症及有严重过敏史等病例，这限制了结论可覆盖的人群。 [D1]

### 5.2 逻辑字段与用途

| 字段组 | 已有信息与拟用方式 |
| --- | --- |
| 患者序号与就诊标识 | 患者序号用作分组主键，就诊标识用作样本主键和官方导出（第 10 节）。核验标识的唯一性、关联和缺失。 |
| 人口学信息、BMI 及时间 | 核验实际取值、单位、脱敏方式及缺失。可用已批准的年龄分箱、性别及 BMI 作为输入。日期只在可靠且获准时用于时间漂移审计。 |
| 主诉、现病史、既往史、入院情况 | 使用固定字段边界，保留否定、患者自述与既往用药语境。 |
| 诊疗过程 | 可能有检查检验、院内治疗以及转抄的带药内容。保留符合预测截点的字段或片段，审计直接披露处方的语句。 |
| 出院诊断 | v1 描述首列为主诊断。若实际发布保持此顺序，以首列定义队列，并在模型允许的时间点保留诊断字段。对未知和顺序异常做审计。 |
| 出院带药列表 | 已标准化药名的多标签集合。按候选表映射、去重和保留标签语义，任何未映射标签不得静默丢弃。 |

v1 描述为按患者唯一编号组织的字典式 JSON，模型样本与提交按就诊构造。评测提交为 UTF-8 JSON 数组，按就诊标识提供 `ID` 与 `prediction` 药名数组。A/B 标签按历史赛题为隐藏标签，独立跨队列测试需要从授权的有标签部分划出。表中的字段是逻辑字段，实际文件名、容器、键名、嵌套结构与编码需以获准发布包为准。 [D1] [D2] [L1]

### 5.3 许可闸门与审计输出

- **待核验的访问。** 目标文件给出的 CHIP 赛题页本次未能通过浏览工具打开，天池原入口跳转到竞赛列表，数据论文列出的仓库本次返回 404。v1 写明访问受 data usage agreements 约束，本地速读记录原赛事数据限评测用途。应以组织方提供的当前版本和条款为准，并确认允许毕业论文、存储环境、云计算、模型权重与派生统计的使用范围。v1 列有官方通信作者，获取前向其确认学术用途和版本。 [D1] [D2] [D5] [D6] [D7] [L1]
- **按患者审计。** 统计主诊断和所有诊断的患者数及就诊数，检查同人不同主诊断、次诊断交叉、近重复文书、无标签就诊和缺失字段。用去标识化审计输出保留患者交集为零的断言，展示各源/目标划分的退出原因及余量。
- **按目录审计。** 汇总标签频数与每次带药数量分位点。各队列报告有标签患者数、目录覆盖率、共享与仅在目标出现的药、源域仅见一次的药、ATC 映射的按条目与按标签事件覆盖率。人工复核组合药、中文通用名与剂型，保留一对多及未映射状态。核对目录外记录并向组织方确认。
- **按字段审计。** 检查诊疗过程与结尾的直接处方抄录。用仅诊断、可用字段及显式披露遮蔽三种配置，辅以药名字符串复制基线和固定抽查规则。只分享获准的汇总。

### 5.4 对研究需求的支撑

651 个目录条目足以构造可穷举检查的编码和有效输出集合，同时对训练深层 codebook 构成样本量限制。诊断种类丰富为队列设计提供候选；能否执行跨疾病实验取决于有标签患者数、共病重叠及每个目标集合对源域药物的可学习支持。先用授权开发部分锁定队列和最小样本条件，再运行目标实验。 [D1]

## 6. Research Questions and Proposed Contributions

| 研究问题 | 可证伪假设与比较 | 贡献及解读 |
| --- | --- | --- |
| **RQ1** 药物 RQ 前缀是否保留可检验的药物语义，同分布下能否支持药名集合预测？ | H1。共享前缀的药物在独立 ATC 标签上更一致，且同骨干 RQ-ID 生成优于置换 ID。比较 ATC 同类富集、代码利用率、覆盖率及同分布 Jaccard。和原子 ID、直接药名及分类器对照。 | 给出目录级诊断与预测指标之间的联系。若一致性提高但预测不变，编码可提供描述性结构，预测收益仍未得到支持。 |
| **RQ2** 源队列训练后，语义前缀能否提高少量目标标签下的表现？ | H2。固定 $k=32$ 个目标标注患者时，RQ-ID 的源→目标适配相对目标单独训练的增益，大于置换 ID 相应增益。同步报告 $k=0,8,16,64$ 学习曲线与源/目标支持。 | 一个患者隔离、可重复按诊断队列评估的迁移协议，以及可归因的前缀实验。负结果或在严格共病筛选下消失，同样是实质发现。 |

论文可形成两项可交付贡献。第一项是药物元数据到确定性 ID 的可复现构建流程，附独立语义检验和与等结构随机对照的结果。第二项是患者级跨队列迁移协议和按支持分层的分析，解释哪些病例或标签贡献了平均收益。计划中的期望是首级前缀提供部分跨队列信号，并可能受共病、标签频率和目录映射质量限制。方向和幅度均由实验决定。

## 7. Theoretical Foundations

### 7.1 残差量化与参数共享

给药物 $m$ 的目录文本 $a_m$ 生成固定向量 $u_m=\operatorname{norm}(\mathrm{BGE}(a_m))$。小型 Encoder $E$ 将其映射为 $z_m=E(u_m)$。第 $\ell$ 级有独立 codebook $C_\ell=\{e_{\ell,1},\ldots,e_{\ell,K_\ell}\}$。令 $r_0=z_m$，计算：

$$
c_\ell=\arg\min_{j\in\{1,\dots,K_\ell\}}\|r_{\ell-1}-e_{\ell,j}\|_2^2,\qquad
r_\ell=r_{\ell-1}-e_{\ell,c_\ell},\qquad
\tilde z=\sum_{\ell=1}^{L}e_{\ell,c_\ell}.
$$

第一级选一个初始近似，第二级逼近这次近似留下的残差，以后逐级重复。共享首级 codeword 即共享生成路径的第一步。同分布下的压缩误差、临床类别一致性和迁移预测表现是三种不同的指标。后续 codeword 跨各父前缀复用，因此只有**完整前缀** $(c_1,\ldots,c_\ell)$ 表示树节点，单个后级 codeword 不提供临床类别定义。 [R1] [R2] [R3]

### 7.2 RQ-VAE 学习目标

以轻量 Decoder $D$ 重建原始目录向量。令 $\operatorname{sg}$ 为 stop-gradient。重建使用 straight-through 路径 $z_{\mathrm{st}}=z_m+\operatorname{sg}(\tilde z-z_m)$，前向是量化和，重建反传更新 Encoder。训练时每一级的残差用 $z_m-\sum_{j<\ell}\operatorname{sg}(e_{j,c_j})$ 来实现，使更晚层的损失路径可控。每个目录条目的目标为：

$$
\mathcal L_{\mathrm{RQ}}=
\|u_m-D(z_{\mathrm{st}})\|_2^2
+\frac{1}{L}\sum_{\ell=1}^{L}
\left(
\|\operatorname{sg}(r_{\ell-1})-e_{\ell,c_\ell}\|_2^2
+\beta\|r_{\ell-1}-\operatorname{sg}(e_{\ell,c_\ell})\|_2^2
\right).
$$

第一项要求 ID 保存目录向量信息。第二项把选中的 codeword 拉向它代表的残差，第三项约束 Encoder 的连续输出靠近已选 codeword。这里的 RQ-VAE 采用向量量化自编码目标和离散潜变量，按 TIGER 的 item tokenizer 思路实现。拟用 $\beta=0.25$ 起步，codebook 按级从 Encoder 向量和后续残差做 K-means 初始化。拟试 $L=3,K=(16,16,16)$、潜变量维度 32，小 MLP 隐藏层 128。$16^3$ 是理论路径数，实际利用率、树深及碰撞分布由数据决定。与同输入的直接 hierarchical K-means 比较，核实 MLP 和残差重建是否提供额外价值。 [R1] [R2] [R3]

### 7.3 何时预期有迁移

若两个队列的药品目录相同，并且某组药物在相似的病历线索下出现，源域样本可同时强化共享前缀的概率。目标适配再细调各叶节点。若组内药物具有不同适用条件、剂型或使用习惯，共享也可能模糊细粒度差异。依据是统计上的参数共享。前缀相同并不代表药物可替换。ATC 以治疗用途、药理、化学等维度组织药物，路线、强度和复方会影响映射，只把它用于分类一致性诊断。 [R9] [T1]

## 8. Proposed Methodology

### 8.1 病历处理

1. 按获准的原始结构读取记录与目录。以显式字段模板做 Unicode 归一、空白和缺失处理。对诊断中的别名与否定保留可审计规则。
2. 以可预测截点为前提，移除标签字段、患者/就诊标识、显式带药或处方抄录片段。切除规则只读输入字段、标题、时间及语境，完全独立于逐病例金标。保留符合时点的历史用药和住院治疗。
3. 用分字段 token 预算保留诊断、关键病史和临床过程，拟从 1,536 或 2,048 输入 token 起做显存试跑。仅用对应训练/开发折拟定预算和规则，记录每字段截断与缺失比例。主实验与同骨干对照共享处理、分割和上下文。
4. 输出固定目录中的归一化药名。保持复方、剂型和相近名称在官方标签层级的区别。目录映射的置信度、来源及版本可复核。锁定模型输入前，对同人不同就诊及跨患者近重复做审计。

### 8.2 药物 Semantic ID 构建

目录条目输入默认为官方标准药名，可补充获准的通用名和成分字段，且每项有版本和来源。可以用药名与药名加成分做敏感性分析。**主方案不将 ATC 类别字段送入 Encoder 或 RQ 训练，也不用其诊断结果调 codebook**，使分类一致性检验具备独立性。抑制泄漏的边界是可用固定目录及先验元数据，禁止使用目标测试处方、其统计量或从测试金标重建候选目录。共享的是任务先验，不含测试记录。

用目录向量拟合小型 RQ 模型，按重建误差、codeword 利用率和预消歧碰撞率做目录级试跑。可将目录按成分组分出小型验证子集以检查过拟合，定完架构后在完整获准目录重拟合并冻结。下游所有同一协议的迁移折使用同一版本，发布候选映射前核对再分配规则与审核权限。若用由病例统计派生的表示，另作探索，且只允许在该实验的允许训练折内计算。

**碰撞处理。** 每条实际路径固定为 $(c_1,c_2,c_3,s_m)$。目录按固定顺序分配互异的规范编号，在相同三元组内以编号排序分配后缀 $s_m=0,1,\ldots$。即使无碰撞也带后缀 $0$，便于控制实际生成步数。使用按层分开的 token 名称，防止同一数字在不同层语义混合。共享后缀只是确定性消歧，具备研究意义的是前三层。例如，占位药 A/B 的 ID 为 $(2,7,4,0)$ 与 $(2,7,4,1)$，药 C 为 $(2,9,1,0)$。A/B 共享三个已学习层，C 与其共享一层，最终三个叶节点仍唯一。检查目录到完整 ID 的双射和 100% round trip。 [R3] [R8]

**关键对照。** 对整个药物到完整路径的映射做均匀随机置换，复用同一批路径、输出 token、长度、碰撞后缀分布和词典树。每种药仍只有一个叶节点。随机重命名 token 会保留原语义分组，置换的对象必须是**药物归属的路径**。另以只用源域频数分箱的置换做敏感性分析，观察头尾频率与前缀拓扑是否混淆结果。最强实用参照包括 651 个独立原子 ID 以及同骨干约束生成药名。 [R8]

**可选固定层级参照。** 在映射质量足够时，以冻结的 ATC 第 2 或第 3 层加唯一叶节点进行生成，对模糊与缺失药设明示分支，保留全候选。它回答廉价人工分类是否已足够，属于使用额外知识的实用参照。其 ATC 同类富集为设计属性，作为实现校验。若编码频繁坍缩，先减小每级 codebook、改初始化，降为两级后按同样协议重建置换对照。 [T1]

### 8.3 LLM 条件生成药物集合

拟以一个 Qwen2.5-1.5B-Instruct 骨干和一套 LoRA 配置读取中文病历。追加候选编码 token 和终止 token，训练其嵌入及输出权重，复用骨干并锁定可训练参数和初始化。用候选的规范编号及预定种子生成一个与编码和标签频数无关的稳定排名，把目标集合 $Y$ 序列化为药物编码序列 $y$，最后接 `<STOP>`。固定排序每条药物恰好序列化一次，输出仍按集合评估。若次要的排序敏感性分析采用另一种固定次序，所有生成臂共用新次序。

训练使用 teacher forcing，仅在目标序列及终止 token 上求交叉熵。主实现对词典树及顺序状态允许的 token 作归一化，使训练和解码空间一致。定义 $A_t$ 为当前允许集合，$a_j$ 为 logits，$T_i$ 是含终止符的目标 token 数，训练损失为

$$
\mathcal L_{\mathrm{gen}}
=-\frac{1}{N}\sum_{i=1}^{N}\frac{1}{T_i}
\sum_{t=1}^{T_i}\log\frac{\exp a_{y_{it}}}{\sum_{j\in A_t}\exp a_j}.
$$

如 $A_t$ 仅含一个 token，该步概率为 1，对交叉熵贡献为 0。数值实现用 log-sum-exp。无论输出编码是否为文字，边界 token 都使用独立的保留符号。合法序列由药物路径及末尾终止符构成。对于名称编码，按目录预先固定每个药名的 token ID 序列，用显式 `<MED_END>` 消歧名称间的前缀重叠。在药物边界按排序限制未来叶节点并允许终止，药物路径内部只能向允许的候选继续。各生成臂使用同一候选目录及药数上限，训练前断言所有金标路径合法。字典校验保证输出可解析且在目录内。实际用药适当性需另行评估。 [R3] [R5] [R9] [T2]

### 8.4 跨疾病迁移协议

| 步骤 | 预先固定的规则 |
| --- | --- |
| 队列定义 | 在获准开发部分核对是否以首列为主诊断。预先固定候选顺序为 2 型糖尿病、高血压、脂肪肝，并定义诊断别名、确诊/疑似、主诊断顺序异常及无法分类的处理。合并症保留作为记录属性。由此得到按就诊划分的队列和按患者隔离的折。 |
| 全局外层留出 | 对有标签患者做一次留出。用预定种子从目标队列留出 $\max(48,\lceil0.25n_{\mathrm{tgt}}\rceil)$ 位，其余队列患者按主诊断指征分层留出约 25%。组内相关记录始终一起分配。同分布与迁移可共用冻结的外层患者，二者独立训练、独立调参，直到实验锁定才读取外层处方标签。 |
| 源域与目标池 | 按主诊断定义，目标患者曾有任何一次目标主诊断时，其全部就诊退出源域。源域仅保留规定内的其余疾病队列。目标适配池来自未进入外层留出的目标患者。目标适配与测试只用这些患者的目标主诊断就诊。 |
| 严格敏感性 | 源域进一步排除患者任何合规记录中的目标次诊断及可确认的病史证据，避免把共病迁移解释成完全未见疾病。记录排除引起的患者数、药物支持及病情分布变化。可用规模匹配的主定义源域重抽样，区分源规模变化和共病筛选。 |
| 预算与选择 | $k$ 按患者计算，适配允许使用每位所抽患者的全部目标主诊断就诊，并同步报告就诊数。每个种子用嵌套样本，$k=8,16,32,64$ 逐级包含。模型选择、训练步数和停止规则在源域内用模拟目标队列及患者隔离开发折确定，各臂共用校准规则。未选中患者的目标文本或标签不参与拟合及校准。外层诊断参与分组，外层文本只用于最终预测。 |
| 迁移臂 | 对每种编码做源→目标和目标单独训练，均从同一基础 LLM 配置出发。目标适配时共享样本、训练步数和超参数。为 $k=0$ 报源域模型直接评估，不把未见药物当作可保证生成。可在同预算下记录源→目标再次更新导致的源验证退化。 |

重点区分三种支持条件：药在源域出现，药只在 $k$ 名患者中出现，药在所有允许训练患者中未出现。预注册支持分层使用训练集统计。目录内存在某个 ID 说明模型可合法输出该药，实际召回仍需观察。跨疾病迁移在单中心内部成立时，也仅支持相应诊断规则及病例分布。

## 9. Overall System Architecture

图 1 展示两条独立的输入通路。目录生成冻结的编码与词典树，病历进入同一个 LLM。规范药名集合是预测和对照的共同接口。

~~~mermaid
flowchart LR
    A["候选目录与药物元数据"] --> B["冻结文本嵌入"]
    B --> C["RQ-VAE 与碰撞消歧"]
    C --> D["药名、完整 ID、词典树"]
    D --> E["训练目标序列化"]
    G["获准的中文住院记录"] --> H["字段审计与患者级划分"]
    H --> I["处理后的病历文本"]
    H --> J["获准训练折的出院药集合"]
    J --> E
    I --> K["同一中文 LLM 与 LoRA"]
    E --> K
    D --> L["约束生成与 ID 回映"]
    K --> L
    L --> M["标准药名集合"]
    H --> N["冻结的评测金标"]
    M --> O["集合、队列、药物支持分析"]
    N --> O
~~~

图 2 展示数据流与可读边界。外层处方一直封存至评估。共用目录仅承载获准的候选元数据。对照只替换编码映射，复用同一患者与就诊清单。

~~~mermaid
flowchart TB
    A["原始数据只读区"] --> B["字段白名单、患者关联和诊断审计"]
    B --> C["冻结患者与就诊划分清单"]
    C --> D["允许开发的记录"]
    C --> E["外层留出患者"]
    D --> F["同分布训练与开发"]
    D --> G["源队列与目标适配池"]
    G --> H["源域开发与嵌套 k 位适配患者"]
    E --> I["留出输入文本"]
    E --> J["独立保存的留出金标"]
    K["固定候选目录"] --> L["目录映射、树和工件哈希"]
    F --> M["同分布训练"]
    H --> N["迁移训练"]
    L --> M
    L --> N
    M --> O["冻结模型预测"]
    N --> O
    I --> O
    O --> P["按患者聚类汇总"]
    J --> P
~~~

图 3 展示模型结构。RQ 残差的逐级符号化可脱离病例训练。目录反查只消费完整路径。

~~~mermaid
flowchart LR
    A["药名与可用成分文本"] --> B["BGE 与小型 Encoder"]
    B --> Q1["Q1 选择粗粒度 codeword"]
    Q1 -->|"残差 1"| Q2["Q2 逼近剩余信息"]
    Q2 -->|"残差 2"| Q3["Q3 再量化"]
    Q1 --> Z["量化向量求和"]
    Q2 --> Z
    Q3 --> Z
    Z --> R["小型 Decoder 重建目录向量"]
    Q1 --> S["离散码组合与消歧"]
    Q2 --> S
    Q3 --> S
    S --> T["药物词典树"]
    U["病历字段模板"] --> V["LoRA 微调的 Qwen"]
    V --> W["合法下一 token 的概率"]
    T --> W
    W --> X["完整药物 ID 序列"]
~~~

## 10. Model Training and Inference Pipeline

图 4 展示实验运行流程。训练和生成共用合法动作规则。训练阶段的输出监督取自相应训练折，推断所需的是输入文本、冻结模型与目录。

~~~mermaid
flowchart TB
    A["锁定划分与各臂 ID"] --> B["用源域内部验证冻结训练预算"]
    B --> C["训练源域 Adapter"]
    B --> D["复制基础模型作为目标单独训练起点"]
    C --> E["在相同 k 个目标患者上适配"]
    D --> E
    C -->|"k = 0"| F["锁定相应模型"]
    E --> F
    F --> H["测试时读取脱敏病历输入"]
    H --> I["批量前向与合法路径搜索"]
    I --> J["到达叶节点后收集药名"]
    J --> K{"已生成终止符？"}
    K -->|"是"| L["验证候选与集合格式"]
    K -->|"否"| I
    L --> M["用冻结金标评估"]
~~~

**训练协议。** 使用模板与冻结 ID 对构造样本，把合法 trie 状态与目标 token 对齐。先验证合成样本的 loss mask、teacher forcing 和目标序列合法性，再用少量授权训练病例检查能否过拟合、模型是否更新新增 token。之后运行已定训练预算。每个 k 从本条件的固定初始点开始，避免从前一个 k 的模型继续训练造成实际步数差异。保存元数据版本、映射和划分哈希、种子、软件环境、超参数、训练步数、GPU 时间和最终检查点。

**推断实现。** 使用按就诊条件化的词典树搜索。每个搜索状态持有模型缓存、当前前缀、上一完整药的固定排名以及已生成集合。候选叶必须排名递增。各臂使用同一 beam width、最大药数及评分规则。采用有效序列累计 log probability，必要时在开发集统一选择按药物数的长度校准，固定后评测。对涉及不同 token 长度的结果同时报告运行代价及药数分布。结束后按字典精确回映，保留生成空集合、提前停止、最大药数触发和内部规则拒绝等汇总。主迁移用完整目录大小作为训练和推断共享的逻辑上限。为输入预算、最长许可输出和终止符留足上下文。可加一项较小推断上限的效率敏感性实验，在源域开发折固定该值，训练保留所有合法金标。推断到达上限后必须在边界终止，评分保留超上限病例并报告其占比。

~~~text
paths = load_frozen_catalog_paths(encoding)
for note in evaluation_inputs:
    active, finished = initialize_from_note(note), []
    while active:
        proposals = []
        for path in active:
            allowed = valid_next_tokens(paths, path, maximum_medications)
            for extension in masked_expansion(model, path, allowed):
                if extension emitted STOP:
                    add extension to finished
                else:
                    commit any completed leaf and reset its trie prefix
                    add extension to proposals
        active = keep_best(proposals, beam_width)
        finished = keep_best(finished, beam_width)
    yield medication_set_from_best_finished_beam(finished)
~~~

评测导出时验证每个就诊只出现一次且全覆盖。只有经数据字典确认的无药处方才编码为空金标。原始字段缺失或标签尚未发布的就诊只生成预测，留待有金标时评分。若重新开放正式平台，按发布的真实键名和规则再次校验。

~~~json
[
  {"ID": "synthetic-visit-001", "prediction": ["药物A", "药物B"]}
]
~~~

## 11. Experimental Design and Evaluation

### 11.1 最小充分实验

| 实验 | 比较和控制 | 目的 |
| --- | --- | --- |
| E0 数据与复制审计 | 总体 top-$k$、按主诊断频次、历史药物候选匹配、诊断单独与消除处方抄录后的全字段 | 识别目录覆盖、类别先验和文本复制能带来多少收益。所有 $k$ 和阈值在允许的开发折决定。 |
| E1 同分布实际性能 | 中文字符 2–4 gram TF–IDF + L2 one-vs-rest logistic regression，冻结中文病历向量 + sigmoid 多标签头，同骨干原子 ID、药名生成、置换 ID、RQ-ID | 给出便宜且有竞争力的分类参照，测试生成与输出结构是否具备实用价值。预先过滤缺少正例的分类头并在零支持分层记录。 |
| E2 编码质量 | RQ-ID、置换、简单分层 K-means 的目录级诊断，可加 ATC 固定前缀的下游参照 | 检验可重建性、前缀语义、碰撞与同组错误。此实验可在合法目录可用时先开展。 |
| E3 跨队列 | RQ-ID 和树形置换 ID 各做目标单独训练与源→目标，主比较 $k=32$，并报告其他 $k$。原子 ID 复核 $k=0,32$ 的选择是否具备实用价值。 | 识别真语义结构对迁移的增量。用同种子、患者和病例输入做成对比较。 |
| E4 敏感性与误差 | 严格共病排除、标签训练支持分层、主要前缀内错药、输入泄漏与预测药数 | 说明增益来自哪些患者和药，对源域定义强度及误差是否稳健。 |

同骨干模型共用基座版本、LoRA 位置与秩、训练折、输入模板、优化预算、候选目录、beam 及上限。RQ/置换臂共用同层 token 集合和新增权重初始化，总参数数、有效路径长度及树节点相同。保持药物排序统一可避免序列化引入额外差异。全局树同构时，按已生成药物排序过滤后的**局部合法分支**仍会随真实药物到路径的分配而变化。额外用统一的去重但不限制排序规则或已定备选排序做一次敏感性检查，解释结果对该约束的依赖。直接药名利用预训练词义，原子 ID 具有短序列和较大的身份词表。它们是实用比较，归因以 RQ 对等树置换为准。 [R8] [R9]

### 11.2 主要指标与历史比较

对第 $i$ 次就诊，用规范药物标签计算 $J_i=|Y_i\cap\hat Y_i|/|Y_i\cup\hat Y_i|$ 和 $F_i=2|Y_i\cap\hat Y_i|/(|Y_i|+|\hat Y_i|)$。取各就诊平均 $\bar J,\bar F$，同时报告 CHIP 风格总分 $S=(\bar J+\bar F)/2$。本地 scorer 拟将双空集合记为 1，单空为 0，并报告双空病例数。正式平台对双空的处理须以提供的 scorer 为准。微平均 precision/recall/F1、药物宏平均 recall、预测药数误差为补充，逐项写清统计分母。 [D2]

| 公开结果 | Jaccard | F1 | 榜单总分 |
| --- | ---: | ---: | ---: |
| CHIP2025 B 榜组织方 GLM4-9B-Chat + LoRA | 0.4477 | 0.5648 | 0.5062 |
| CHIP2025 B 榜 DeepDrug 集成 | 0.5102 | 0.6267 | 0.5685 |
| MediRec v3 论文报告 | 0.4626 | 0.5813 | — |

分数来自不同设置，用于提供背景。最终结果在内部拆分时明确标注划分和临床字段版本，仅与同一拆分下的实现作直接比较。历史参考单独列明模型、标签与资源差异。 [D2] [D3] [D4]

### 11.3 估计量、统计区间与复现

对每个固定种子，在同一外层目标就诊上计算平均 Jaccard 的迁移增益。设 $J_{c,b}$ 为编码 $c$ 和训练方式 $b$ 的测试分数，预注册 RQ2 的主要量为

$$
\Delta_{\mathrm{transfer}} =
(J_{\mathrm{RQ},\,\mathrm{src}\to\mathrm{tgt}}-J_{\mathrm{RQ},\,\mathrm{tgt}})
-
(J_{\mathrm{perm},\,\mathrm{src}\to\mathrm{tgt}}-J_{\mathrm{perm},\,\mathrm{tgt}}).
$$

它衡量 RQ 带来的**额外源域训练收益**。同步报告四个原始分数与 $\Delta_{\mathrm{direct}}=J_{\mathrm{RQ},\,\mathrm{src}\to\mathrm{tgt}}-J_{\mathrm{perm},\,\mathrm{src}\to\mathrm{tgt}}$。正 $\Delta_{\mathrm{transfer}}$ 且负 $\Delta_{\mathrm{direct}}$ 意味着 RQ 从源域得到更多帮助但最终表现仍低于对照。$k=0$ 只报告源模型的直接差值。主推断集中为一个队列、$k=32$、Jaccard 的 $\Delta_{\mathrm{transfer}}$。其他队列、$k$、诊断和 F1 为描述性或探索性比较。

计划使用至少 3 个配对种子，固定目录的主 RQ 构建，保存置换与嵌套采样种子。按外层**患者**成对 bootstrap 2,000 次，每次重采患者并保留其全部符合队列的就诊，对同一重采样计算全部模型的分数差和跨种子均值，报告 95% percentile CI。该区间反映测试患者的抽样变异，另报各随机种子的结果及波动。少患者或极少支持的结果报告分母和不确定性，并考虑等权患者均值作为敏感性分析。内外划分、阈值、样本种子、软件及统计脚本均冻结。 [T3]

### 11.4 编码、长尾与误差诊断

| 诊断 | 定义与解读 |
| --- | --- |
| RQ 重建与占用 | 目录验证集 MSE 与 cosine；每级有效条目数及使用率；perplexity $\exp(-\sum_j p_j\log p_j)$；深度 $\ell$ 的占用前缀数和最大簇占比。 |
| 碰撞与可逆性 | 报告有同码邻居的条目占比、三元组的不同值数量及最大碰撞组。在显式消歧后要求 100% 编解码往返。评分始终匹配药物叶节点。 [R8] |
| 独立 ATC 一致性 | 在高置信映射子集计算各深度「共享前缀药对中 ATC 2/3 级一致的比例」，报告药对数、符合条件的药和标签事件覆盖率。与重复匹配树置换的目录零分布比较，同时报告组内 vs 全目录充足差异。避免把成分同名药的相似性当成唯一证据，可排除重复成分组合重算。 [T1] |
| 长尾与零支持 | 在允许训练折按药的标签事件频数预定 0、1–5、6–20、21+ 桶，另外按源/目标出现情况分层。按桶报告测试金标事件数、药种数、recall 及错药集中度。完全未见的候选药只作描述，解释可能的首级泛化和末级缺乏监督。 |
| 错误与输出数量 | 类别内替代、组合药与单成分混淆、漏报及多报、历史药误复制、共病病例、截断、空集、上限命中。抽样依据事先冻结，人工复核只在获准环境，临床解释如需专业人员须先确认。 |
| 运行效率 | 在同一设备测吞吐、单个就诊 p50/p95 推断耗时、生成 token 数、峰值显存、训练 GPU 小时和可训练参数。将 catalog tokenizer 的一次性成本单列。 |

如没有可信药物交互映射，核心实验无需 DDI 分数。若许可与映射覆盖允许，可另报有注释药对的冲突率与覆盖率，按已映射无序药对作分母，并按药数比较历史和预测集合。此指标不构成安全性验证。 [M1] [M2] [T1]

## 12. Technology Stack and Implementation Plan

| 层 | 建议起点 | 备选与取舍 |
| --- | --- | --- |
| 环境 | Python 3.11、PyTorch、Transformers、PEFT，使用 Linux 或 WSL2 的学校或本地获准环境。CPU 足够做审计和传统基线。 | 固定 Torch/CUDA/PEFT 兼容版本与 model revision，记录安装锁文件。先在选定环境完成最小样例，降低多平台维护。 [T2] [T4] |
| 病历与基线 | Python 标准库读 JSON，pandas 或 Polars 做汇总，scikit-learn 做字符 TF–IDF、OVR 和向量模型。 | 压缩长病历可先用分字段截断。专用临床编码器只有在许可、输入形式及验证都合适时才有比较价值。 |
| 药物 embedding | 冻结 `BAAI/bge-small-zh-v1.5`，对目录文本离线 batch 编码。先确认药名长度、向量维度与规格敏感性。 | 多语模型或稀疏字符向量可用于对照。元数据与模型版本一并冻结。规模小，优先选公开且易重现的模型。 [T5] |
| Quantizer | 用 PyTorch 实现较小的残差 MLP 和最近邻查找，借鉴 TIGER 的分层 codebook。参考作者实现，保留自定义消歧及诊断。 | hierarchical K-means 容易复现。深模型造成稀疏码、成本或调参压力时，较浅量化已足够测试主假设。 [R2] [R3] |
| 生成 | `Qwen/Qwen2.5-1.5B-Instruct` + BF16 LoRA。建议从 rank 8、alpha 16、attention 的 q/v 投影起步，梯度累积、checkpointing 与源域开发折选择训练步数。 | 优先用同一骨干覆盖对照。显存受限可改较短输入或 QLoRA，并统一全组配置。3B 以上只作有余力时的稳健性检查。 [T2] [T4] [T6] |
| 解码 | PyTorch/Transformers 内部自回归循环及稀疏 trie 掩码，单元级检验排序、路径、STOP 与新增 token。 | 原生 `prefix_allowed_tokens_fn` 适合先试可行性。若训练和推断需要同一个状态机，保留共享 `valid_next_tokens` 减少重复规则。 [T4] |
| 复现与评估 | 本地 Git 跟踪代码、YAML 配置和清洗规则。使用 JSONL/CSV 汇总指标、清单校验哈希与绘图脚本，训练日志不保存病历全文。 | 本地 MLflow 可选。原始 EHR、分组密钥、逐病例预测或含数据的 checkpoint 放在允许的受控存储。 [T3] |

新增 token 的索引、初始向量与字典一起固定。若基座的词嵌入与 LM head 绑定，使用已锁版本 PEFT 的 trainable tokens 保持权重绑定，并做一步梯度和重载检查。验证加入的 token 确为单 ID，检查新增输入嵌入、对应输出概率及 Adapter 保存后等价。若版本组合不支持该路径，可为编号生成单独使用小型 code embedding/output head，RQ 和置换臂同时切换。 [T2] [T4]

**资源预算是规划值。** 建议一张约 24 GB 显存的 GPU 作主训练起点，12–16 GB 可通过短上下文、小 microbatch、累积梯度及统一 QLoRA 试跑。1.54B 参数用 BF16 存放仅模型权重约 3.1 GB，激活、优化器、KV cache、logits 与运行时需要额外显存。BF16 可用性先看运行环境。原始数据许可允许的前提下，规划专用受控存储与约 32 GB 内存。 [T2] [T6]

先在固定环境实际跑 100 个更新步和一批完整解码，记录显存峰值、步时与不同药数的延迟。主要四臂在三个种子约为 6 次训练/种子，包括 2 个源域模型和 4 个 $k=32$ 目标训练，源 Adapter 随后复用。仅在此基础上追加其他预算、原子 ID、额外队列和严格筛选。额外预算由 `运行数 × 实测步数 × 单步时间` 与推断时间估算。

### 建议代码结构

~~~text
configs/                 # 数据版本、输入规则、每项实验及种子
src/
  data/                  # JSON 适配、标准化、诊断分组、泄漏审计
  splits/                # 患者级清单、外层留出、嵌套目标抽样
  catalog/               # 标准名、成分、ATC 映射及目录审计
  semantic_ids/          # 文本嵌入、RQ、随机置换、可逆词典
  models/                # 基线、Qwen/LoRA、目标序列训练
  decoding/             # trie 状态、约束动作、beam 与反查
  evaluation/           # 集合评分、bootstrap、诊断和正式导出
scripts/                 # audit、make_splits、fit_id、train、score
tests/                   # 合成例：患者隔离、往返、合法搜索、评分边界
reports/                 # 获准公开的汇总和图表
~~~

CLI 以冻结清单和工件版本为输入。每个运行写下游所需摘要及对应哈希。数据、模型和派生物的存放位置按数据使用条款确定。

## 13. Feasibility Analysis and Risk Assessment

| 启动条件或风险 | 观察与应对 |
| --- | --- |
| 论文使用授权和标签可得性 | 第一优先级是组织方数据协议、合规存储和带患者分组的授权标签。可并行准备合成结构、代码接口、文献和 Poster。若无法访问 CDrugRed，则用获准开放药品目录或模拟标签检验编码及协议，或者经导师同意调整评估所用的数据。明确写出证据支持的性能指标与跨疾病结论。 [D1] [D2] [D5] [D6] |
| 队列容量与污染 | 启动规划门槛为一个队列约 128 名可用目标患者、外层至少 48 名且适配池至少 64 名，初始要求至少 200 名源患者。数字用于保证流程与初步可读性，区间宽度仍需通过授权开发数据估计。按预先顺序选首个满足条件的队列。阈值根据开发病例和专家意见审定并冻结。若无法做 $k=64$，降为能支持的预算并先于评估冻结。若 $k=32$ 主比较及独立目标测试均无法支撑，报告混合队列的低标注学习和编码分析，并注明跨疾病迁移尚待验证。 |
| 处方或患者泄漏 | 不给模型患者及就诊标识。共用外层清单与一致的字段截点，近重复和同人场景做断言。用遮蔽敏感性、复制基线及抽查反映出院文书可能先有答案的问题。实际时序无法重建时，限定为病历回顾性任务。 [T3] [T7] |
| 目录不全、ATC 映射偏差 | 原始数据和候选表分别版本化。只有获准完整目录可复现 CHIP 范围。若只有训练可见目录，独立报告目录覆盖上界并修改任务描述。若高置信 ATC 覆盖不足，改做目录占用、成分抽查与同族错误，语义临床对齐结论随之收窄。 [T1] |
| 量化无收益或长尾受损 | 保留置换、原子和分类参照，优先解释类内近似错误、置信区间和成分词形影响。没有正效应仍可得出支持范围清楚的结果。 |
| 模型和计算超出计划 | 首先完成目录诊断、频率、分类基线及 $k=32$ 四臂。根据试跑确定是否补全其余曲线、严格过滤、额外类别、输入时间漂移与第二骨干。 |
| 对数据集和安全性外推 | 单中心、选择后人群、历史处方金标、标签只含药名且无结局。报告群体构成及漏报和多报。其他医院、年代或临床场景的验证单独设计。 [D1] [T3] |

替代数据路线需要先核对端点与许可。例如，PhysioNet 的 MIMIC Note Extract v1.0 有从英文出院摘要提取的 discharge medications 字段，属于受控且机器抽取的数据，需核验处方标签、文本时间点与目录对齐。MIMIC-IV 的院内处方表和 CDrugRed 出院带药属于不同端点。若采用替代数据，重新确定评测任务、比较基线和结论范围。 [T8]

## 14. Research Roadmap

| 阶段 | 周期估计 | 可见成果与退出条件 |
| --- | --- | --- |
| A 研究准备 | 第 1–2 周 | 文献证据表、主要估计量、数据访问/合规确认。Interim Poster 可展示方案与已核实数据摘要。 |
| B 数据与基线 | 获准后第 1–2 周 | 目录与病历字段审计、按患者统计诊断支持、冻结划分、完成频率/复制/多标签基线。用病例数量判断跨疾病主实验能否开展。 |
| C Tokenizer 与最小生成 | 获准后第 3–4 周 | 冻结 RQ 和置换码，验证全部双射、掩码和推断输出，完成显存与步时试跑和初始同分布比较。 |
| D 核心迁移实验 | 获准后第 5–7 周 | 固定三种子、$k=32$、冻结四臂评估及源模型零样本结果。预算允许时补充原子 ID、学习曲线和严格队列敏感性。 |
| E 诊断与论文 | 获准后第 8–10 周 | 盲评错误、按支持分层、统计区间和计算报告。整理可共享的复现物与局限，再更新 Poster 或论文结果。 |

最小完成条件是一个可复算的病历集合预测基线、可逆目录编码、同树置换比较，以及在数据支持的范围内对主要问题给出区间估计。权限延迟时优先完成只依赖公开证据或合成数据的成果，并明确标注任何未执行的临床实验。

## 15. Limitations and Future Work

- 主诊断列表可以含共病与编码噪声，严格筛选引入人群变化。需要同时报告筛选方式、实际支持和焦点样本。
- 651 个条目使码本拟合便宜，也增加过拟合和细粒度碰撞的风险。目录描述相似性、ATC 对齐和条件用药相关性各有局限。
- 生成把无序集合写成固定序列，停止规则和搜索会影响输出数量及错误结构。等树控制、统计诊断与排序敏感性是解释结果所必需的。
- 同院回顾性准确度对后来时间、其他医院、药品目录变化、处方必要性或用药安全的外推有限。后续可在获准资料上构建时间分割和异院外部评估，再研究有专家审阅的适当性指标。 [T3] [T7]

## 16. References

文献核验截至 2026-11-25。可沿 `CDrugRed / Chinese discharge medication`、`medication recommendation + residual quantization / semantic IDs`、`generative recommendation + cross-domain / hierarchy` 和 `diagnosis holdout + few-shot transfer` 更新，从原始出版页、作者稿和官方代码交叉确认。如引用 arXiv 2510.21084，固定版本标识与对应标题。

### 数据、任务与本地材料

- **D1** Li, J., Yuan, H., Luo, L., et al. (2025). *CDrugRed: A Chinese Drug Recommendation Dataset for Discharge Medications in Metabolic Diseases*. arXiv v1. [D1]
- **D2** Li, J., Yuan, H., Luo, L., et al. (2025/2026). *Overview of CHIP 2025 Shared Task 2: Discharge Medication Recommendation for Metabolic Diseases Based on Chinese Electronic Health Records*. CHIP 2025 proceedings，2026 年出版。 [D2] [D2P]
- **D3** Zhu, Z., Huang, H., & Wu, X. (2026). *Towards Discharge Medication Recommendation via Multi-scale Model Training and Multi-dimensional Feature Enhancement*（DeepDrug）。CHIP 2025 proceedings. [D3]
- **D4** Li, J., Yuan, H., Luo, L., Sun, Y., Wang, J., & Lin, H. (2026). *MediRec: Enhancing Chinese Medication Recommendation with Explainable Clinical Reasoning*. arXiv v3. [D4]
- **D5–D7** CHIP2025 原任务页、天池入口与 v1 所列数据仓库，当前访问状态见第 5.3 节。 [D5] [D6] [D7]
- **L1** 工作区提供的赛题速读，用于核对原始链接和提交说明。 [L1]
- **L2–L3** 工作区提供的 PSP408 Interim Poster 评分细则与既有竖版模板。 [L2] [L3]

### 医疗推荐与相关迁移

- **M1** Shang, J., Xiao, C., Ma, T., Li, H., & Sun, J. (2019). *GAMENet: Graph Augmented MEmory Networks for Recommending Medication Combination*. AAAI. [M1]
- **M2** Yang, C., Xiao, C., Ma, F., Glass, L. M., & Sun, J. (2021). *SafeDrug: Dual Molecular Graph Encoders for Recommending Effective and Safe Drug Combinations*. IJCAI. [M2]
- **M3** Wu, R., Qiu, Z., Jiang, J., Qi, G., & Wu, X. (2022). *Conditional Generation Net for Medication Recommendation*. The Web Conference. [M3]
- **M4** Lv, H., Zhang, K., Tan, Y., & Chen, X. (2026). *Towards Efficient and Interpretable Medical Concept Representation via Ontology-driven Residual Vector Quantization*. The Web Conference. [M4] [M4C]
- **M5** Su, X., et al. (2025). *Multimodal Medical Code Tokenizer*. ICML, PMLR 267, 57030–57045. [M5]
- **M6** Wang, X., et al. (2026). *SafeRx-Agent: A Knowledge-Grounded Multi-Agent Framework for Safe and Explainable Medication Recommendation*. EMNLP. [M6]
- **M7** Zhao, C., Tang, H., Zhang, J., & Li, X. (2025). *Unveiling Discrete Clues: Superior Healthcare Predictions for Rare Diseases*（UDC）。The Web Conference. [M7]
- **M8** Lv, H., et al. (2026). *Structure-enhanced Multimodal Medical Concept Quantization for Medication Recommendation*. ICME，作者稿。 [M8]
- **M9** Song, Z., et al. (2026). *SMI: Ontology-Grounded Semantic IDs for Hierarchy-Aware Medical Concept Representation*. arXiv，作者注明 NeurIPS 2026 Main Track 接收。 [M9]
- **M10** Zhang, K., Fang, H., Zhou, J., Lv, H., Pan, Y., & Tan, Y. (2026). *Coverage-Aware Reasoning with Medical Tokens for Diagnosis Prediction*. arXiv. [M10]
- **M11** Zhu, M., Liu, Y., Luo, Z., & Zhu, T. (2026). *Bridging data gaps of rare conditions in ICU: a multi-disease adaptation approach for clinical prediction*. npj Digital Medicine, 9, 7. [M11]

### 离散表示、层级与生成式推荐

- **R1** van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). *Neural Discrete Representation Learning*. NeurIPS. [R1]
- **R2** Lee, D., Kim, C., Kim, S., Cho, M., & Han, W.-S. (2022). *Autoregressive Image Generation Using Residual Quantization*. CVPR. [R2]
- **R3** Rajput, S., et al. (2023). *Recommender Systems with Generative Retrieval*. NeurIPS. [R3]
- **R4** Fang, D., et al. (2025). *HiD-VAE: Interpretable Generative Recommendation via Hierarchical and Disentangled Semantic IDs*. arXiv. [R4]
- **R5** Hu, P., Lu, W., & Wang, J. (2026). *From IDs to Semantics: A Generative Framework for Cross-Domain Recommendation with Adaptive Semantic Tokenization*（GenCDR）。AAAI, 40(17), 14874–14882. [R5]
- **R8** Zhang, Q., Szymanski, L., Zhang, H., & Deng, J. D. (2026). *Faithful Evaluation of Semantic-ID Tokenizers for Generative Recommendation*. CIKM. [R8]
- **R9** Hou, Y., et al. (2026). *Expressiveness Limits of Autoregressive Semantic ID Generation in Generative Recommendation*. arXiv. [R9]

### 分类标准、实现与报告指引

- **T1** WHO. *ATC Classification*. 分类原则、给药途径、剂型及复方的处理。 [T1]
- **T2** Qwen Team. *Qwen2.5-1.5B-Instruct* 模型卡。 [T2]
- **T3** Collins, G. S., et al. (2024). *TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods*. BMJ, 385, e078378. [T3]
- **T4** Hu, E. J., et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR。Hugging Face PEFT 及 Transformers 官方实现文档附后。 [T4] [T4P] [T4G]
- **T5** BAAI. *bge-small-zh-v1.5* 模型卡。 [T5]
- **T6** Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs*. NeurIPS. [T6]
- **T7** Davis, S. E., et al. (2024). *A framework for understanding label leakage in machine learning for health care*. JAMIA, 31(1), 274–280. [T7]
- **T8** Kuntzer, J., et al. (2026). *MIMIC-IV-Ext MIMIC Note Extract: A Large-Scale Structured Dataset Extracted from MIMIC-IV-Note Discharge Summaries*, v1.0. PhysioNet. [T8]

[D1]: https://arxiv.org/pdf/2510.21084v1
[D2]: https://arxiv.org/pdf/2511.06230
[D2P]: https://link.springer.com/chapter/10.1007/978-981-95-7299-1_29
[D3]: https://link.springer.com/chapter/10.1007/978-981-95-7299-1_30
[D4]: https://arxiv.org/abs/2510.21084v3
[D5]: http://www.cips-chip.org.cn/2025/eval2
[D6]: https://tianchi.aliyun.com/competition/entrance/532411
[D7]: https://github.com/DUTIR-BioNLP/CDrugRed
[L1]: <../references/chip2025/CHIP2025_出院用药推荐任务速读.md>
[L2]: <../references/psp408/PSP408 Interim Poster Presentation Marking Guidelines 26-27.pdf>
[L3]: <../references/psp408/InterimPoster_Template.ppt>
[M1]: https://cdn.aaai.org/ojs/3905/3905-13-6964-1-10-20190702.pdf
[M2]: https://www.ijcai.org/proceedings/2021/514
[M3]: https://doi.org/10.1145/3485447.3511936
[M4]: https://doi.org/10.1145/3774904.3792930
[M4C]: https://github.com/lvhangkenn/MedRQ
[M5]: https://proceedings.mlr.press/v267/su25b.html
[M6]: https://aclanthology.org/2026.emnlp-main.1003/
[M7]: https://doi.org/10.1145/3696410.3714831
[M8]: https://www.cs.emory.edu/~jyang71/files/scqmed.pdf
[M9]: https://arxiv.org/abs/2610.18672
[M10]: https://arxiv.org/abs/2610.10641
[M11]: https://www.nature.com/articles/s41746-025-02176-y.pdf
[R1]: https://papers.nips.cc/paper/7210-neural-discrete-representation-learning
[R2]: https://openaccess.thecvf.com/content/CVPR2022/html/Lee_Autoregressive_Image_Generation_Using_Residual_Quantization_CVPR_2022_paper.html
[R3]: https://papers.nips.cc/paper_files/paper/2023/hash/20dcab0f14046a5c6b02b61da9f13229-Abstract-Conference.html
[R4]: https://arxiv.org/abs/2508.04618
[R5]: https://doi.org/10.1609/aaai.v40i17.38508
[R8]: https://doi.org/10.1145/3799682.3841124
[R9]: https://arxiv.org/abs/2605.06331
[T1]: https://www.who.int/tools/atc-ddd-toolkit/atc-classification
[T2]: https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct
[T3]: https://www.bmj.com/content/bmj/385/bmj-2023-078378.full.pdf
[T4]: https://arxiv.org/abs/2106.09685
[T4P]: https://huggingface.co/docs/peft/en/package_reference/lora
[T4G]: https://huggingface.co/docs/transformers/en/internal/generation_utils
[T5]: https://huggingface.co/BAAI/bge-small-zh-v1.5
[T6]: https://arxiv.org/abs/2305.14314
[T7]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10746313/
[T8]: https://physionet.org/content/mimic-iv-ext-note-extract/1.0/
