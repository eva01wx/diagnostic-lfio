# 近期 forgetting 研究对当前论文的启示

检索日期：2026-10-16。重点核对 2024–2026 年的逻辑遗忘、必要／充分解释及知识编译研究。
以下将论文实际结论、当前实现的已有能力、尚未实施的借鉴建议分别表述。不是穷尽性综述。
配套书目为 `FORGETTING_LITERATURE_2026-10-16.bib`。

## 1. Xu Wang 的 DATA 2024 论文

**Xu Wang and Christopher Brewster. Forgetting in Knowledge Graph Based Recommender Systems.
DATA 2024, pp. 309–317. DOI: 10.5220/0012757300003756.**

[学校出版记录](https://cris.maastrichtuniversity.nl/en/publications/forgetting-in-knowledge-graph-based-recommender-systems/)
与[出版社正文](https://www.scitepress.org/publishedPapers/2024/127573/pdf/index.html)均已核对。

该文区分被动和主动遗忘，以最小模型变化及 WSC impact 评价删除三元组的影响；后者采用
图中心性、Personalized PageRank 等评分。还讨论规则体优化，并评估推荐与解释多样性指标。
适合借鉴的是“先评估任务影响，再选择遗忘对象”的流程。评分可用于候选排序，不能替代
当前的全补全证书；给定知识图谱上保持规则触发，也不同于对所有数值补全保持分类。

当前工作的自然延伸关系可写为：从知识图谱推荐中的影响评估，发展到数值规则代理中
可验证的双侧投影与部分输入判断。应明确新旧问题、语义及实验的差异。

## 2. 最值得对照的近期工作

| 工作与已核实发表状态 | 原文内容与对当前稿件的启示 |
| --- | --- |
| Doherty & Szałas, **Dual forgetting operators in the context of weakest sufficient and strongest necessary conditions**, Artificial Intelligence 326:104036, **2024**。[出版社](https://www.sciencedirect.com/science/article/pii/S0004370223001820)；[作者预印本全文](https://arxiv.org/html/2305.07233v1) | 直接研究标准／弱遗忘的对偶及必要／充分条件；第 5 节包括背景理论。当前双侧公式应直接对照并引用，不能作为新理论。可借鉴其背景理论处理数值合法性约束。 |
| Liberatore, **Common equivalence and size of forgetting from Horn formulae**, AMAI 92:1545–1584, **2024**。[正文](https://link.springer.com/content/pdf/10.1007/s10472-024-09955-5.pdf) | 研究遗忘后的规模增长、公共词汇上的等价及辅助变量压缩。可借鉴共享子表达式与表示成本分析；其 Horn 复杂性结果不能直接当作当前数值 DNF 的下界。 |
| Liberatore, **One Head is Better than Two: A Polynomial Restriction for Propositional Definite Horn Forgetting**, JoLLI 34:49–88, **2025**。[正文](https://link.springer.com/article/10.1007/s10849-025-09428-w) | 寻找可转化为特定 single-head Horn 形式的可处理结构。注意该文指“每个命题至多作为一条规则的头”，不同于本项目多条规则共享一个输出类别的 single-head DNF，不能移植多项式结论。 |
| Zeyen, Tit, Cordy & Perrouin, **DivKC: A Divide-and-Conquer Approach to Knowledge Compilation**, FASE **2026**, pp.126–146，2026-04-17 在线发表。[全文](https://link.springer.com/chapter/10.1007/978-3-032-22774-4_7) | 利用投影分解公式后进行知识编译。适合借鉴分解、缓存和共享表示，缓解当前补集的中间表示膨胀；其计数界不是本稿的预测证书，不能直接替换。 |
| Knorr, Saribatur & Gonçalves, **Beyond Uniform: To Boldly Abstract What Has Not Been Abstracted Before**, KR **2026**, pp.412–422。[论文](https://proceedings.kr.org/2026/39/kr2026-0039-knorr-et-al.pdf) | 在 ASP 中相对于允许添加的事实集合研究抽象，并联系简化和 forgetting。可借鉴“明确允许上下文”的问题设计；ASP 等价与当前实数域的约束补全属于不同语义。 |
| Calautti, Malizia & Molinaro, **On the Complexity of Global Necessary Reasons to Explain Classification**, KR **2025**, pp.206–217。[论文](https://proceedings.kr.org/2025/21/kr2025-0021-calautti-et-al.pdf) | 区分全局必要解释的有效性与最小性，并分析不同模型表示的计算复杂性。当前“指定保留词汇上的最强条件”不等于其受限解释语言上的最小性问题，应明确解释语言。 |
| Izza et al., **Most General Explanations of Tree Ensembles**, IJCAI **2025**, pp.5463–5471。[论文页面](https://www.ijcai.org/proceedings/2025/608) | 通过区间膨胀寻找覆盖更广的局部充分解释。适合借鉴实例解释的覆盖度与最小性对照；本稿当前按选择集挑选隐藏坐标，尚未实现该类优化。此文已在当前主稿引用。 |
| Beierle et al., **A General Framework of Epistemic Forgetting and its Instantiation by Ranking Functions**, **2025 预印本**，2025-08-29。[全文](https://arxiv.org/pdf/2508.21441) | 以性质和公理比较多种认知遗忘操作。可借鉴性质对照表的呈现方式；本文未核实其后续期刊发表状态，Spohn 排名不是推荐分数，不能直接映射到本项目。 |
| Jung, Koopmann & Knorr, **Interpolation in Knowledge Representation**, **2025 预印本**，2025-12-09。[全文](https://arxiv.org/html/2512.08833v1) | 综述插值、遗忘、解释及不同逻辑中的可表达性限制。帮助界定目标语言、语义保持与计算成本，不能把一般描述逻辑的不存在性结果套用到当前有限阈值表示。 |

另发现 **Symbolic Explanations for ML Models: A Formal Approach via FCA and Forgetting** 的
[塞维利亚大学 2026-04-14 研讨会摘要](https://casiopea.eii.us.es/index.php/eventos-pasados/1083-symbolic-explanations-for-ml-models-a-formal-approach-via-fca-and-forgetting/2026-04-14-11-00)。
摘要涉及用户选定词汇、互斥约束及二值神经网络解释，与本问题接近；此次未定位其可核验的
正式论文全文／发表记录，因此只列为待跟踪线索，不把摘要中的效果主张当作已验证结果。

## 3. 我建议优先借鉴的三个方向

### A. 影响度驱动、带成本约束的遗忘选择

目前 Certainty1/2 选择的是覆盖率最高的一个／两个已使用特征。下一步可把目标改为：
在满足最低认证覆盖率和表示预算时，尽量节省特征采集成本或隐藏更多指定类型的特征。

候选排序可以借鉴 DATA 2024 的任务影响思路，但最终以来源代理的证书确认接受。建议先比较
固定索引、规则使用频次、拟合／选择数据上的影响排序、现有覆盖率贪心及成本感知选择。
特征图及影响分数若从数据估计，只能用拟合／选择分区。没有知识图谱时不应机械照搬 PPR。

主要输出应是“隐藏成本／数量—认证覆盖率—表示规模”的权衡曲线，同时保留类别条件覆盖率、
精确完成率和运行成本。当前部分指标已经具备，新增指标应体现真实任务目标。仅增加一个
评分公式的贡献有限；有明确优化问题和可核验接受条件，才更有研究价值。

### B. 在背景约束下进行遗忘

当前对所有有限实数补全量化，不自动采用跨特征约束。设保留输入为 r、遗忘输入为 s，
来源代理头类别公式为 F，允许补全的背景理论为 C。可定义：

\[
V_C(r)=\exists s\,C(r,s),\qquad
U_C(r)=\exists s\,[C(r,s)\land F(r,s)],
\]
\[
L_C(r)=V_C(r)\land\neg\exists s\,[C(r,s)\land\neg F(r,s)].
\]

V_C 成立且 L_C 成立时确认头类别；V_C 成立且 U_C 不成立时确认默认类别。
若 V_C 不成立，应标记“无合法补全”，不能靠空真或空集合直接作出业务分类。
这使用既有背景理论下的量化思想；公式本身不构成新理论。

例如 F 为 x>0，隐藏 x、保留 y，且业务规则明确给定 x=y。当前无约束投影对所有 y
都无法确定类别；使用该合法约束后，y>0 与 y≤0 可分别确认两个类别。这个例子说明约束
能够消除离开合法域的反例，不说明任意数据相关性都可以视为硬约束。

应区分真实定义约束与从数据学习的经验约束。后者可能失效，其证书只在假定的 C 下成立。
建议先用已知约束的合成正／负对照验证：真实约束、缺失约束、错误约束和无合法补全。
随后再加入有明确数据字典约束的任务；不直接把观测最小／最大值当作真实合法域。

### C. 把全公式编译与实例查询分开评估

当前双侧显式公式有明显增长，且 752/14,834 个查询进入保守回退。可借鉴知识编译研究，
尝试共享子表达式、按特征依赖分解和缓存。BDD／多值决策图也是待比较的表示选择。
普通 d-DNNF 不提供任意低成本否定；投影后也不应默认保留所有编译性质。

另一条工程路径是固定 r 后分别检查 C∧F 和 C∧¬F 是否可满足：两者都 SAT 则确有歧义，
只有一侧 SAT 且另一侧 UNSAT 时可认证单一类别，两侧 UNSAT 则无合法补全；含 UNKNOWN
时只保留已经证明的结论，不能把 UNKNOWN 当 UNSAT。这能支持实例证书，无需先展开整个
全局 DNF，但也不自动提供可读的全局公式。

比较须同时报告编译时间、查询时间、总时间、峰值内存、共享节点数和展开后条件数。
共享节点数少不能直接当作同等数量的可读规则，也不能据此宣称人类理解度提高。

## 4. 对论文定位的直接影响

建议先补充 DATA 2024、AI 2024 双重遗忘及 Liberatore 2024/2025 的相关工作对照。
把研究主张集中为“面向任务与输入成本的、保持数值语义的可验证遗忘”，而不是遗忘算子的
首次提出。解释的逻辑最强性、描述的最短性和人的易理解性是三个不同目标。

可新增一张性质对照表：数值阈值一致性、整体特征删除、精确 U/L、限额时的保守有效性、
背景约束、解释最小性、部分输入接口、表示成本和实验可重放性。当前实现已有前述多项
性质与独立测试；不应把背景约束和最小解释标成已实现。

当前实验结论保持来源代理范围：80.8% 来自选择集选定的单特征策略，61.4% 来自具有至少
两个活动特征的合格模型；不是任意特征缺失的覆盖率，也不是教师／真实标签正确率。
上述 A/B/C 是本次调研提出的后续方向，不是新增实验结论。
