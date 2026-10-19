# KBS 投稿检查清单（当前版本：2026-10-16）

当前稿件为 **Compact Rule Distillation with Diagnostic Allocation and Certified Feature Forgetting**。
统一入口及三个发布包见 [CURRENT_RELEASE.md](CURRENT_RELEASE.md)，当前报告目录为
`contextual_forgetting_2026-10-16/`。同步前的九月清单保存在
`archive/before_version_sync_2026-10-16/docs/research/SUBMISSION_CHECKLIST.md`；
其统计、页数和命令属于历史版本。

## 已完成的技术工作

- [x] 作者依次为 Xu Wang、Zhiqiang Chen；Xu Wang 为通讯作者，
  两位作者同属现列三项单位。
- [x] 完成诊断/预算分配、显式泛化、全局特征遗忘及背景约束/成本/实例查询扩展。
- [x] 全工作区验证记录为 **167 passed**，保留 8 条既有上游警告。
  可复现包另在独立目录完成 **47** 项相关测试。
- [x] 当前扩展复用 **648** 个 Original 代理，保存 **19,440** 条策略记录；
  回放 **7,507** 个旧版 Original 投影及 **18,984** 个候选子集。
- [x] 完成来源/输入哈希、选择回放、独立逻辑检查和测试预测向量核验。
- [x] 数值差值约束采用精确有理数语义，区分歧义、UNKNOWN 和无合法补全。
- [x] 投稿入口为 `elsarticle-template-num.tex`，采用 `elsarticle` 的
  `review,12pt` 单栏格式及 `elsarticle-num` 数字编号引用；主文 **62 页**，
  两个兼容入口同为 **62 页**，历史开发补充材料 **16 页**。
- [x] BibTeX 共 **67** 条，主文引用 **63** 条；无缺失引用或重复键。
- [x] 五条 Highlights 的 TXT/DOCX、当前题目的投稿信草稿和 AI 辅助披露已准备。
- [x] 已按作者提供的信息加入浙江省自然科学基金青年科学基金资助声明，
  项目编号 **LQN26F020019**；论文、投稿信及声明工作表一致。
- [x] Zhiqiang Chen 的贡献已按作者提供的信息写入：论文撰写、审阅修改、文献收集。
- [x] Xu Wang 的贡献已按 2026-10-17 作者补充的信息写入：研究构思、方法设计、
  软件实现、研究实施、验证、论文初稿和审阅修改。
- [x] 当前三个发布包、扁平源码、代码/记录附件和成员哈希已准备；
  逐项验证见当前目录中的 `validation.json` 与 `delivery_manifest.json`。

## 投稿前仍需作者核实的事实

- [ ] 竞争利益，按事实填写。
- [ ] 最终人工核对作者、单位、联系方式、全文及 AI 披露。
- [ ] 核实原创性、发表历史、独家投稿和作者批准等声明。
- [ ] 如选择公开发布，确定实际仓库/tag 或 DOI；当前材料为本地交付。
- [ ] 按投稿时实际系统核对所需文件和版式要求；本清单不声称已提交。

参见 [声明工作表](AUTHOR_DECLARATIONS_DRAFT.md) 与
[投稿信草稿](KBS_COVER_LETTER_DRAFT.md)。资助与作者贡献信息来自作者；其余事实仍待作者核实。

## 科学结论的边界

- 同一批任务的后续扩展不增加独立数据集数量，重复查询不增加独立样本量。
- 背景约束在已知合成域验证；成本为透明控制设定，尚非真实采集成本。
- 确认证书针对来源代理及明确补全域，不证明教师正确性、最小解释或人的理解度。
- 固定 MDL 控制强于保守分配器；保留原比较与区间，不包装为普遍优化优势。
- 页数、测试数和哈希核验不构成录用保证。正式声明与期刊系统要求仍须逐项完成。

## 当前构建及核验入口

```bash
venv/bin/python build_paper.py --all
venv/bin/python audit_recent_manuscript.py --output docs/research/contextual_forgetting_2026-10-16/manuscript_audit_main.json --render-dir results/tex_build/contextual_main
venv/bin/python audit_recent_manuscript.py --paper LFIO_Negation_Paper_review.pdf --output docs/research/contextual_forgetting_2026-10-16/manuscript_audit_review.json --render-dir results/tex_build/contextual_review
venv/bin/python package_contextual_forgetting.py
```

PDF 审计脚本会重置人工视觉审查标记；查看实际渲染后再记录完成。
实验重放和新运行说明见 [REPRODUCE.md](REPRODUCE.md)。历史打包器应在相应历史副本中运行。
