# 当前论文与发布版本（2026-10-16）

当前版本标识：**contextual-forgetting-2026-10-16**。

论文题目：**Compact Rule Distillation with Diagnostic Allocation and Certified Feature Forgetting**。
作者依次为：Xu Wang、Zhiqiang Chen；Xu Wang 为通讯作者，两位作者同列三项单位。
投稿入口为 `elsarticle-template-num.tex`，采用
`\documentclass[review,12pt]{elsarticle}` 和 `elsarticle-num` 数字编号引用。
源码文件名 `LFIO_Negation_Paper.tex` 为兼容既有工具保留；
论文题目、PDF 属性、投稿信和当前说明均使用上面的题目。

| 内容 | 从仓库根目录起的路径 |
|---|---|
| 首选投稿稿件（62 页单栏、12pt） | `elsarticle-template-num.pdf` |
| 同文兼容入口（62 页） | `LFIO_Negation_Paper.pdf` |
| 同文审稿兼容入口（62 页） | `LFIO_Negation_Paper_review.pdf` |
| 历史开发补充材料（16 页） | `LFIO_Development_Supplement.pdf` |
| 当前发布包及审计 | `docs/research/contextual_forgetting_2026-10-16/` |
| 当前扩展实验记录 | `results/contextual_forgetting_20261016/` |
| 实现及结果说明 | `docs/research/CONTEXTUAL_FORGETTING_2026-10-16.md` |
| 复现说明 | `docs/research/REPRODUCE.md` |

当前三个发布包均位于上表的当前发布目录：

- `KBS_Contextual_Forgetting_LaTeX_sources_2026-10-16.zip`
- `KBS_Contextual_Forgetting_Reproducibility_Code_2026-10-16.zip`
- `KBS_Contextual_Forgetting_Experiment_Records_2026-10-16.zip`

源码包包含扁平化的投稿主文件、兼容入口、独立开发补充材料及科学输入。
其中 `elsarticle-template-num.tex` 是可独立编译的完整主文。
代码包与实验记录包应解压到同一目录。
`package_manifest.json` 记录包哈希，包内 `BUNDLE_CONTENTS_SHA256.json` 记录成员哈希；
`validation.json` 和 `delivery_manifest.json` 记录当前交付的验证及文件哈希。
这些是本地交付材料；此记录不表示已经公开上传或投稿。

历史研究阶段保持原来的目录和源文件锁：

| 阶段 | 历史发布目录 |
|---|---|
| 全局数值特征遗忘 | `docs/research/feature_forgetting_2026-10-16/` |
| 显式规则泛化 | `docs/research/rule_generalization_2026-10-16/` |
| 诊断与预算分配 | `docs/research/major_revision_2026-10-15/` |
| 九月开发与排版 | 相应的 `paper_2026-09-*` / `kbs_2026-09-*` 目录 |

此次版本同步只调整题目元数据、当前目录说明和发布附件，实验源文件及结果保持原样。
同步前的论文、说明和三个发布包保存在
`docs/research/archive/before_version_sync_2026-10-16/`，附 `snapshot_manifest.json`。

后续已按作者提供的信息加入浙江省自然科学基金青年项目资助声明，英文采用省基金委
统一标注名称 **Zhejiang Provincial Natural Science Foundation of China**，
Grant No. **LQN26F020019**。
Zhiqiang Chen 的贡献为论文撰写、审阅修改和文献收集。
2026-10-17 已补全 Xu Wang 的贡献：研究构思、方法设计、软件实现、研究实施、验证、
论文初稿和审阅修改。补全前的文件保存在
`docs/research/archive/before_credit_completion_2026-10-17/`。
当前 PDF 和附件的资助及作者更新核验见 `submission_metadata_audit.json`；
`version_sync_audit.json` 保留此前版本同步的核验记录。
资助更新前的文件保存在 `docs/research/archive/before_funding_update_2026-10-16/`。

2026-10-19 按作者指定，将主稿切换为 `elsarticle` 的 `review,12pt` 单栏审稿格式。
原 19 页双栏版和 34 页 CAS 单栏版归档于
`archive/before_elsarticle_review_2026-10-19/`。开发补充材料保持独立的历史 CAS 格式。
算法、实验结果和参考文献不变；格式迁移核验见当前发布目录中的
`format_migration_audit.json`。GitHub 上传材料同步到仓库根目录的“论文源代码和复现”。
